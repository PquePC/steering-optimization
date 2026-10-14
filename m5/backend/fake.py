"""m5.backend.fake - a simulator of the subject model, driven by a ground truth you can read.

**Why it exists.** Every readout, measure and search step in M5 is written against `Backend`
(`m5.backend.base`). Run against this simulator, the whole pipeline -- screen, adaptive search,
freeze, confirmation -- meets an answer known in advance: `FakeTruth` says, per vector, per layer
and per construct, at which dose each one switches on. A search that does not recover those
thresholds here will not recover them on Gemma, and here the failure costs seconds instead of
GPU-hours (plan §5.2, "what this gives up, and the mitigation").

**What it is not.** Not a language model. Text comes from fixed banks; the "model" is a set of
dose-response curves (logistic in log r_perp, the psychometric form of plan §6.1) wired to the
channel each prompt is recognised as. What it reproduces are the *structures* the literature and
this project's M3 data say the real subject has, so the pipeline meets them before the GPU does:

- Noticing and naming are separate constructs with separate per-layer thresholds. Detection
  peaks mid-network, forced identification rises toward late layers, and the late "gate" layers
  keep the default "no" (Lindsey 2026; Macar 2026; FINDINGS.md §4).
- A model can notice without naming, and when made to name it says "apple" -- Gemma's answer at
  zero dose for every concept (Lederman 2026; FINDINGS.md §3.1).
- Forced naming (`d3`/`d2`) is the open slot filled by steering (`d5`) plus an introspective excess
  that is positive only at some layers, so `d3 - d5` is the quantity that means something.
- The yes/no question alone is a global yes-shift on every question, target and control alike;
  only the difference-in-differences (`d7`) isolates noticing (Hahami 2026; Sridharan 2026).
- Past-only steering (briefing steered, question not) is weak on Gemma (Singh 2026); it is the only
  arrangement in which naming is not steering leaking into the answer (Pearson 2026).
- An injection is registered in hidden states before it is reported (Pearson 2026): the
  registration curve sits below the noticing curve, so `q1` has something to find.
- Mid-generation restarts ("Wait, that's not right.") rise with noticing (McKenzie 2026, `d10`).
- Integrity fails at a per-layer random-vector dose `D_rand_true(L)`; forced naming collapses with
  it (M3: `d3` non-monotone in dose, TODO.md), and one layer's naming collapses outright while its
  norms run smooth (the L39 collapse, RUNBOOK-FINAL.md).
- One loud residual coordinate (index 0) with a large RMSNorm gain, as Gemma-3 has (plan §4.3), so
  the nuisance scan has something to find and `r_perp` differs from M3's `r_raw`.

## Channels, recognised only by the phrases in `m5.data.markers`

A prompt's channel is read from how its text *ends* and what its last user turn contains -- never
from anything else, so a builder that drops its marker phrase is caught by this simulator's tests:

    ends with FORCED_PREFILL (+ at most one filler)     forced naming slot      d3, d2, d8 naming
    ends with OPEN_SLOT_PREFILL, OPEN_SLOT_REQUEST in   open slot               d5
    ends with YES_NO_PREFILL, YES_NO_INSTRUCTION in     yes/no logits           d7 (target/control/
                                                                                mismatch)
    ends with SEGMENT_PREFILL, SEGMENT_QUESTION in      localisation            d9
    ends with DECOY_LIST_PREFILL, DECOY_LIST_QUESTION   decoy list              d6
    reply start + "(", "\n (A) " and "\n (B) " in      A/B option logits       e10
    reply start or "Answer:", CAPABILITY_INSTRUCTION    "Answer: x"             s5
    reply start, containing a registered scenario      evaluation awareness    a1
    reply start, DETECTION_/UNUSUAL_QUESTION in         self-report             d8
    anything else                                       open-ended text         e7 e8 e9 s1 s2 s6 d10

Three things a real model never needs come from registries instead of the text, because a tag in
the text is something a real model would read: a scenario's class (`register_scenarios`), an s5
item's answer key (`register_capability_items`) and an A/B item's behaviour letter
(`register_ab_items`). An unregistered item raises rather than defaulting -- and so does an
unregistered scenario from `m5.data.scenarios` -- so a setup that forgot to register cannot
produce a plausible number.

## Doses

The truth is in `r_perp` (m5's primary dose, decision 6): `r_perp = ||alpha * v_perp|| / s_L`,
computed here from the *registered* vector values with the simulator's own nuisance coordinate
(index 0) zeroed and its own `s_L` (`FakeGeometry.s`). A multi-layer intervention's dose for a
channel is the sum of its per-layer doses, each weighted by that layer's sensitivity
(`z = sum_L r_L / theta_c(L)`, plan §4.3 item 5): a schedule spread over layers with equal
thresholds acts like one layer at the summed dose. A position that is not steered itself but
follows steered ones (the past-only arms, Positions "span") reads every curve at the doses times
`FakeTruth.past_only_factor` -- except d9's localise curve, which already describes one-segment
steering.

Operations (`m5.intervention`): `add` drives every channel; `sub` drives the sign-blind ones
(noticing, registration, integrity, the yes-bias) identically and suppresses the concept instead
of showing it; `project` with alpha >= 1 removes the concept's natural presence and injects
nothing, so projecting a direction with no concept changes nothing at all; `clamp` to c > 0 acts
as `add` with `alpha * ||v|| = c`, to c < 0 as `sub`, and to 0 as a full projection.

## Determinism and sampling

Every random choice in a request comes from a stream seeded by (backend seed, `GenRequest.seed`,
prompt) and *not* by the intervention, so arms differing only in steering share their random
numbers (common random numbers: a zero dose reproduces the unsteered output exactly, and an
outcome decided by one probability moves monotonically with dose). Temperature 0 takes the most
likely outcome. A slot's first generated token is drawn from exactly the distribution
`next_token_logprobs` reports, so generated and logit readings of a channel (d2 and d3) agree by
construction; free-text channels (d8, s5) also open with the token those logits predict.

## Beyond the protocol

`FakeBackend` adds what tests and the end-to-end check need, none of which the pipeline should
call: `explain` (the probabilities behind one request), `bound_truth` (which truth a registered
vector acts under: by id, else by direction, else the random truth), `true_vector`,
`alpha_for` / `r_perp_of` (the simulator's own dose arithmetic) and `unregister_vector`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from m5 import safety
from m5.backend.base import (GenRequest, GenResult, KLResult, ModelInfo, ResidualStats,
                             ScoreRequest, ScoreResult, derive_seed)
from m5.data import markers as M
from m5.intervention import Intervention, Positions

# =====================================================================================
# Tokens and template
# =====================================================================================

SPECIAL_TOKENS = ("<pad>", "<eos>", "<bos>", "<start_of_turn>", "<end_of_turn>")
PAD_ID, EOS_ID, BOS_ID, SOT_ID, EOT_ID = range(len(SPECIAL_TOKENS))
_N_RESERVED = 16
# Ordinary tokens hash into a 2**52 id space: stable across processes and call orders (a growing
# dictionary would give a resumed run different ids), exactly representable as float64, and wide
# enough that a collision is never expected; one would raise rather than merge two tokens.
_HASH_SPACE = 1 << 52
VOCAB_SIZE = _N_RESERVED + _HASH_SPACE

USER_OPEN = "<start_of_turn>user\n"
MODEL_OPEN = "<start_of_turn>model\n"
TURN_CLOSE = "<end_of_turn>\n"
TEMPLATE = "<bos>" + USER_OPEN + "{content}" + TURN_CLOSE + MODEL_OPEN

# a1 scenarios carry no marker in their text (markers.py, m5.data.scenarios): a tag a real model
# could read would tell it it is being tested. The simulator learns a scenario's class from a
# registry instead -- `FakeBackend.register_scenarios({text: "eval" | "deploy"})` -- and a prompt is
# a scenario prompt when it contains a registered text. Nothing is added to any prompt.
SCENARIO_CLASSES = ("eval", "deploy")

# s5 capability items: the answer key lives beside the item, never in its text (it would hand the
# real model the answer). `FakeBackend.register_capability_items({question: answer})` supplies it,
# and an unregistered item raises rather than being scored against a guessed key. The unsteered
# pass rate is the registered one, else a hash of the question text in [0.35, 0.95].

# e10 A/B items, in CAA's format (m5.prompts.ab_item): the two options are written into the question
# as "\n (A) ..." and "\n (B) ..." and the reply is opened with "(", so the next token is the letter.
# Which letter is the behaviour's lives beside the item: `FakeBackend.register_ab_items`.
AB_PREFILL = "("
AB_OPTIONS = ("\n (A) ", "\n (B) ")

# d9 segments are lines that begin "Segment k:" (or "k." / "k)" / "[k]") before the question;
# d6 options are lines "k. word" in the last user turn. The answer to either is the number k.
_SEGMENT_LABEL = re.compile(r"(?mi)^[ \t]*(?:segment[ \t]+(\d+)[ \t]*[:.)-]|\[(\d+)\]|(\d+)[.)])")
_OPTION_LINE = re.compile(r"(?m)^[ \t]*(\d+)[.):][ \t]*([^\n]*?)[ \t]*$")
_ABOUT = re.compile(r"\babout\s+(?:the\s+(?:word|concept)\s+)?[\"'“‘]?([A-Za-z][\w-]*)")

_SPECIAL_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL_TOKENS) + ")")
# A newline alone; else optional horizontal whitespace then a word run or one symbol; else a run of
# horizontal whitespace with nothing after it on the line. Every character falls in one of these,
# so the tokens partition the text and decode(tokenize(x)) == x.
_PIECE = re.compile(r"\n|[^\S\n]*\w+|[^\S\n]*[^\w\s]|[^\S\n]+")


def _hash_id(token: str) -> int:
    h = hashlib.blake2b(token.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return _N_RESERVED + int.from_bytes(h, "little") % _HASH_SPACE


def _unit(*parts: object) -> float:
    """A deterministic number in [0, 1) from any identifying parts."""
    h = hashlib.blake2b(repr(parts).encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


class FakeTokenizer:
    """Word-level tokens with exact character offsets.

    Special tokens are split out first, so "<end_of_turn>" is one token whatever precedes it. Each
    remaining token is a newline, or leading spaces plus a word or one symbol: " garlic" is one
    token, as a concept word is in Gemma's vocabulary, and "Answer:" is "Answer" then ":".
    """

    def __init__(self) -> None:
        self._id_of: dict[str, int] = {s: i for i, s in enumerate(SPECIAL_TOKENS)}
        self._str_of: dict[int, str] = {i: s for i, s in enumerate(SPECIAL_TOKENS)}
        self._cache: dict[str, tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = {}

    def id_of(self, token: str) -> int:
        i = self._id_of.get(token)
        if i is None:
            i = _hash_id(token)
            other = self._str_of.get(i)
            if other is not None and other != token:
                raise RuntimeError(f"token id collision between {other!r} and {token!r}")
            self._id_of[token] = i
            self._str_of[i] = token
        return i

    def string_of(self, token_id: int) -> str:
        try:
            return self._str_of[int(token_id)]
        except KeyError:
            raise KeyError(f"token id {token_id} was never produced by this tokenizer") from None

    def encode(self, text: str) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        ids: list[int] = []
        offs: list[tuple[int, int]] = []
        pos = 0
        for k, part in enumerate(_SPECIAL_SPLIT.split(text)):
            if k % 2 == 1:                       # odd parts are the separators: special tokens
                ids.append(self._id_of[part])
                offs.append((pos, pos + len(part)))
            elif part:
                end = 0
                for m in _PIECE.finditer(part):
                    s, e = m.span()
                    if s != end:                 # pragma: no cover - the pattern covers every char
                        raise RuntimeError(f"tokenizer gap at {pos + end} in {text!r}")
                    ids.append(self.id_of(m.group()))
                    offs.append((pos + s, pos + e))
                    end = e
                if end != len(part):             # pragma: no cover
                    raise RuntimeError(f"tokenizer stopped at {pos + end} in {text!r}")
            pos += len(part)
        out = (tuple(ids), tuple(offs))
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[text] = out
        return out

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            return "".join(self.string_of(i) for i in ids if int(i) >= _N_RESERVED)
        return "".join(self.string_of(i) for i in ids)


def render_gemma_like(messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = True) -> str:
    """Gemma's chat layout: a system message is prepended to the first user turn, model turns are
    "model", and the generation prompt ends where the reply begins (a prefill is concatenation)."""
    out = ["<bos>"]
    system: str | None = None
    for k, msg in enumerate(messages):
        role, content = msg["role"], msg["content"]
        if role == "system":
            if k != 0:
                raise ValueError("a system message is only allowed first")
            system = content
            continue
        if role == "user":
            if system is not None:
                content, system = system + "\n\n" + content, None
            out.append(USER_OPEN + content + TURN_CLOSE)
        elif role in ("assistant", "model"):
            if system is not None:
                raise ValueError("a system message must be followed by a user turn")
            out.append(MODEL_OPEN + content + TURN_CLOSE)
        else:
            raise ValueError(f"unknown role {role!r}")
    if system is not None:
        raise ValueError("a system message must be followed by a user turn")
    if add_generation_prompt:
        out.append(MODEL_OPEN)
    return "".join(out)


# =====================================================================================
# Ground truth
# =====================================================================================

def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(x, 700.0)))
    e = math.exp(max(x, -700.0))
    return e / (1.0 + e)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class Curve:
    """One logistic dose-response curve in log r_perp, the psychometric form of plan §6.1:

        p(r) = floor + (ceiling - floor) * sigmoid(slope * (ln r - ln threshold))

    `threshold` is the r_perp at which the curve is half-way from floor to ceiling; `inf` means
    the construct never switches on at this layer. `slope` is per unit of ln r_perp: at 4 the curve
    goes from 12% to 88% of its span over a factor e of dose. A ceiling below 1 is the lapse.
    """

    threshold: float
    slope: float = 4.0
    floor: float = 0.0
    ceiling: float = 1.0

    def __post_init__(self) -> None:
        if not self.threshold > 0:
            raise ValueError(f"threshold must be > 0 (inf = never), got {self.threshold}")
        if not (math.isfinite(self.slope) and self.slope > 0):
            raise ValueError(f"slope must be finite and > 0, got {self.slope}")
        if not 0.0 <= self.floor <= self.ceiling <= 1.0:
            raise ValueError(f"need 0 <= floor <= ceiling <= 1, got {self.floor}, {self.ceiling}")

    @property
    def never(self) -> bool:
        return math.isinf(self.threshold)

    def at_z(self, z: float) -> float:
        """The curve at a dose of `z` thresholds (z = r / threshold)."""
        if z <= 0.0 or self.never:
            return self.floor
        return self.floor + (self.ceiling - self.floor) * _sigmoid(self.slope * math.log(z))

    def p(self, r_perp: float) -> float:
        return self.floor if self.never else self.at_z(r_perp / self.threshold)

    def dose_at(self, q: float) -> float:
        """The r_perp at which the curve exceeds its floor by q (inf when it never does)."""
        span = self.ceiling - self.floor
        if self.never or q >= span:
            return math.inf
        if q <= 0:
            return 0.0
        x = q / span
        return self.threshold * math.exp(_logit(x) / self.slope)

    def scaled(self, factor: float) -> "Curve":
        return replace(self, threshold=self.threshold * factor)


NEVER = Curve(math.inf)
CURVE_NAMES = ("effect", "slot", "excess", "notice", "registration", "integrity", "localise")


@dataclass(frozen=True)
class LayerTruth:
    """One vector's curves at one layer; doses are r_perp at that layer.

        effect        the concept appears in open-ended text (e7, e8, e9); for a behavioural
                      vector, its behaviour moves (a1 for evaluation awareness)
        slot          an open answer slot is filled with the concept (d5): steering alone does it
        excess        introspective access beyond the slot; forced naming (d3, d2) is
                      1 - (1 - slot)(1 - excess), so d3 - d5 > 0 only where excess rises
        notice        the model registers that something was done to it, as far as it can say so
                      (the d7 shift, d8 claims, d10 restarts)
        registration  an injection signal in the hidden state at the answer position (q1)
        integrity     the probability the model stops working: degenerate text (s2), junk mass in
                      every distribution (s6, entropy), failed capability items (s5)
        localise      d9, knowing which segment was steered; `notice` when not given
    """

    effect: Curve = NEVER
    slot: Curve = NEVER
    excess: Curve = NEVER
    notice: Curve = NEVER
    registration: Curve = NEVER
    integrity: Curve = NEVER
    localise: Curve | None = None

    def curve(self, name: str) -> Curve:
        if name == "localise" and self.localise is None:
            return self.notice
        if name not in CURVE_NAMES:
            raise KeyError(f"unknown curve {name!r}; one of {CURVE_NAMES}")
        return getattr(self, name)


@dataclass(frozen=True)
class VectorTruth:
    """What one vector really does, layer by layer.

    `concept` is None for vectors with no name (behavioural, controls): naming cannot happen.
    `associations` are the near misses and allusions the concept produces (FINDINGS.md §3.3: silk's
    "velvet", wrists' "wristwatches"). `triggers` are the lower-case words whose presence in a text
    puts this vector's direction into the residual stream (the concept word by default), which is
    what lets a vector extracted through `hidden` find its way back to this truth.
    """

    vector_id: str
    concept: str | None
    vector_class: str
    layers: tuple[LayerTruth, ...]
    associations: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        safety.check_vector_class(self.vector_class)
        if self.concept is not None:
            safety.check_concept(self.concept)
        object.__setattr__(self, "layers", tuple(self.layers))
        if not self.layers:
            raise ValueError(f"{self.vector_id}: a vector truth needs its layers")
        trig = tuple(t.lower() for t in self.triggers)
        if not trig and self.concept:
            c = self.concept.lower()
            trig = (c, c + "s")
        object.__setattr__(self, "triggers", trig)
        object.__setattr__(self, "associations", tuple(self.associations))

    def at(self, layer: int) -> LayerTruth:
        if not 0 <= layer < len(self.layers):
            raise KeyError(f"{self.vector_id}: no truth for layer {layer}")
        return self.layers[layer]


def _same_word(a: str | None, b: str | None) -> bool:
    """Equal up to case, surrounding punctuation and a plural ending ("garlic" / "Garlics")."""
    if not a or not b:
        return False
    a = re.sub(r"[^\w-]", "", a.casefold())
    b = re.sub(r"[^\w-]", "", b.casefold())
    return bool(a) and (a == b or a + "s" == b or b + "s" == a or a + "es" == b or b + "es" == a)


@dataclass(frozen=True, eq=False)
class FakeTruth:
    """The simulator's ground truth: per-vector layer curves plus the model-wide constants.

    Model-wide constants and what each stands for:

        d_rand             D_rand_true(L): r_perp at which random vectors break integrity (the
                           midpoint of the random truth's integrity curve)
        claim_floor        d8 false-positive rate on control trials, injected-thought framing
        unusual_floor      the same for the unusual-processing framing, whose sensitivity is
                           `unusual_gain` times smaller
        refusal_rate       self-reports that decline to answer at all
        claim_bias_gain    how much the global yes-bias leaks into free-text claims
        past_only_factor   a past-only intervention acts as if its dose were this much smaller
                           (Singh 2026: Gemma needs its sampled tokens steered); d9 is exempt, its
                           curve being defined for one-segment steering already
        yes_bias_logits    the global yes-shift every yes/no question gets at full bias (Hahami 2026)
        yes_bias_at        ... reached half-way at this fraction of D_rand(L), for any vector
        notice_logits      the extra shift on the detection question at full noticing: the d7
                           difference-in-differences
        mismatch_factor    the share of it a detection question naming another concept gets
        tilt               off-target distortion of every next-token distribution, per D_rand of
                           dose: s6 KL grows as (r_perp / D_rand)^2 at small doses
        scenario_base      a1 base rates of verbalised evaluation awareness by scenario class
        ea_gain            how far an evaluation-awareness vector moves that rate (sign-dependent)
        ab_max_logits      e10: the log-odds shift toward the behaviour's answer of an A/B item at
                           full effect, for a vector of the item's own class (add toward, sub away)
        eval_generic_gain  how far noticing anything moves it, for every vector (the reason a3
                           compares against control vectors)
        collapse_layer     the layer whose naming collapses (documentation; built into the curves)
        motor_from         the first "motor" layer: naming without noticing
        geometry_seed      seeds the simulated residual stream (`FakeGeometry`)
    """

    n_layers: int
    d_rand: tuple[float, ...]
    vectors: Mapping[str, VectorTruth]
    claim_floor: float = 0.02
    unusual_floor: float = 0.01
    unusual_gain: float = 0.7
    refusal_rate: float = 0.02
    claim_bias_gain: float = 0.03
    past_only_factor: float = 0.15
    yes_bias_logits: float = 2.5
    yes_bias_at: float = 0.45
    yes_bias_slope: float = 3.0
    notice_logits: float = 5.0
    mismatch_factor: float = 0.3
    tilt: float = 1.2
    scenario_base: tuple[tuple[str, float], ...] = (("eval", 0.35), ("deploy", 0.05))
    ea_gain: float = 0.85
    eval_generic_gain: float = 0.15
    ab_max_logits: float = 4.0
    collapse_layer: int | None = None
    motor_from: int | None = None
    geometry_seed: int = 0

    CHANNELS = ("effect", "effect_size", "slot", "forced", "excess", "slot_raw", "forced_raw",
                "notice", "claim", "claim_unusual", "localise", "localise_raw", "registration",
                "integrity", "bias", "tilt")

    def __post_init__(self) -> None:
        if self.n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        d_rand = tuple(float(x) for x in self.d_rand)
        if len(d_rand) != self.n_layers or not all(math.isfinite(x) and x > 0 for x in d_rand):
            raise ValueError("d_rand needs one positive value per layer")
        object.__setattr__(self, "d_rand", d_rand)
        vectors = dict(self.vectors)
        for vid, vt in vectors.items():
            if vid != vt.vector_id:
                raise ValueError(f"truth key {vid!r} names vector {vt.vector_id!r}")
            if len(vt.layers) != self.n_layers:
                raise ValueError(f"{vid}: {len(vt.layers)} layers of truth for {self.n_layers}")
        object.__setattr__(self, "vectors", vectors)
        for name in ("claim_floor", "unusual_floor", "refusal_rate", "past_only_factor"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        object.__setattr__(self, "_random_cache", {})
        random_layers = tuple(LayerTruth(
            notice=Curve(1.25 * d, 5.0, 0.0, 0.3),        # mild, and only near breakdown
            registration=Curve(0.5 * d, 3.0, 0.0, 0.7),   # random perturbations are decodable
            integrity=Curve(d, 6.0)) for d in d_rand)     # D_rand_true(L), by definition
        object.__setattr__(self, "_random_layers", random_layers)

    # ---------------------------------------------------------------- vectors
    def d_rand_true(self, layer: int) -> float:
        return self.d_rand[layer]

    def random_vector(self, vector_id: str = "random") -> VectorTruth:
        """The truth of any vector the truth does not name: no concept, integrity breaking at
        D_rand_true(L), mild noticing near breakdown only."""
        cache = self._random_cache  # type: ignore[attr-defined]
        vt = cache.get(vector_id)
        if vt is None:
            vt = cache[vector_id] = VectorTruth(vector_id, None, "control", self._random_layers)  # type: ignore[attr-defined]
        return vt

    def vector(self, vector_id: str) -> VectorTruth:
        vt = self.vectors.get(vector_id)
        return vt if vt is not None else self.random_vector(vector_id)

    def _resolve(self, vector: str | VectorTruth) -> VectorTruth:
        return vector if isinstance(vector, VectorTruth) else self.vector(vector)

    # ---------------------------------------------------------------- the curves, combined
    @staticmethod
    def _combine(vt: VectorTruth, name: str, doses: Sequence[tuple[int, float]]) -> float:
        """Several layers act as one dose of z = sum_L r_L / threshold_L (each layer weighted by
        its own sensitivity), read on the curve of the layer contributing most."""
        z, best, best_z = 0.0, None, 0.0
        for layer, r in doses:
            c = vt.at(layer).curve(name)
            if c.never or r <= 0.0:
                continue
            zc = r / c.threshold
            z += zc
            if best is None or zc > best_z:
                best, best_z = c, zc
        if best is None:
            return vt.at(doses[0][0] if doses else 0).curve(name).floor
        return best.at_z(z)

    def _bias(self, doses: Sequence[tuple[int, float]]) -> float:
        z = sum(r / (self.yes_bias_at * self.d_rand[layer]) for layer, r in doses if r > 0)
        return _sigmoid(self.yes_bias_slope * math.log(z)) if z > 0 else 0.0

    def observables(self, vector: str | VectorTruth, doses: Mapping[int, float], *,
                    sign: int = 1, past_only: bool = False) -> dict[str, float]:
        """Every channel's probability (or shift) at these per-layer r_perp doses.

        `sign` is +1 for add-like operations and -1 for sub-like ones: the concept and its name
        need the add direction, while noticing, registration, integrity and the yes-bias read the
        size of the perturbation only. Observable naming and localisation are attenuated by
        integrity failure (M3: forced identification stops working as the model breaks).
        """
        vt = self._resolve(vector)
        live: list[tuple[int, float]] = []
        for layer, r in doses.items():
            layer, r = int(layer), float(r)
            if not 0 <= layer < self.n_layers:
                raise KeyError(f"layer {layer} outside the simulated model (0..{self.n_layers - 1})")
            if not (math.isfinite(r) and r >= 0):
                raise ValueError(f"r_perp must be finite and >= 0, got {r}")
            if r > 0:
                live.append((layer, r))
        f = self.past_only_factor if past_only else 1.0
        d = [(layer, r * f) for layer, r in live]
        size = self._combine(vt, "effect", d)
        slot = self._combine(vt, "slot", d)
        exc = self._combine(vt, "excess", d)
        notice = self._combine(vt, "notice", d)
        reg = self._combine(vt, "registration", d)
        fail = self._combine(vt, "integrity", d)
        loc = self._combine(vt, "localise", live)
        bias = self._bias(d)
        if sign < 0:
            slot = exc = 0.0
        effect = size if sign > 0 else 0.0
        healthy = 1.0 - fail
        forced = 1.0 - (1.0 - slot) * (1.0 - exc)
        speak = healthy * (1.0 - self.refusal_rate)
        c_inj = 1.0 - (1.0 - self.claim_floor) * (1.0 - notice) * (1.0 - self.claim_bias_gain * bias)
        c_unu = 1.0 - ((1.0 - self.unusual_floor) * (1.0 - self.unusual_gain * notice)
                       * (1.0 - self.claim_bias_gain * bias))
        return {
            "effect": effect, "effect_size": size,
            "slot": slot * healthy, "forced": forced * healthy,
            "excess": max(forced - slot, 0.0) * healthy,
            "slot_raw": slot, "forced_raw": forced,
            "notice": notice, "claim": speak * c_inj, "claim_unusual": speak * c_unu,
            "localise": loc * healthy, "localise_raw": loc,
            "registration": reg, "integrity": fail, "bias": bias,
            "tilt": self.tilt * sum(r / self.d_rand[layer] for layer, r in d),
        }

    # ---------------------------------------------------------------- the API tests read
    def p(self, vector_id: str, layer: int, channel: str, r_perp: float, *,
          past_only: bool = False, sign: int = 1) -> float:
        """The true probability of `channel` at one layer and dose (see `CHANNELS`)."""
        return self.p_multi(vector_id, {layer: r_perp}, channel, past_only=past_only, sign=sign)

    def p_multi(self, vector_id: str, doses: Mapping[int, float], channel: str, *,
                past_only: bool = False, sign: int = 1) -> float:
        obs = self.observables(vector_id, doses, sign=sign, past_only=past_only)
        if channel not in obs:
            raise KeyError(f"unknown channel {channel!r}; one of {self.CHANNELS}")
        return obs[channel]

    def threshold(self, vector_id: str, layer: int, channel: str, q: float, *,
                  past_only: bool = False, sign: int = 1, r_max: float = 100.0) -> float:
        """The smallest r_perp at which `channel` exceeds its zero-dose floor by q (inf if never
        below r_max). Observable channels can fall again past breakdown, so this is the first
        crossing, found on a fine log grid and refined by bisection."""
        def excess(r: float) -> float:
            return self.p(vector_id, layer, channel, r, past_only=past_only, sign=sign) - base
        base = self.p(vector_id, layer, channel, 0.0, past_only=past_only, sign=sign)
        prev = 0.0
        for r in np.geomspace(1e-5, r_max, 801):
            r = float(r)
            if excess(r) >= q:
                lo, hi = prev, r
                for _ in range(60):
                    mid = 0.5 * (lo + hi) if lo == 0.0 else math.sqrt(lo * hi)
                    if excess(mid) >= q:
                        hi = mid
                    else:
                        lo = mid
                return hi
            prev = r
        return math.inf

    def yes_no_shift(self, obs: Mapping[str, float], kind: str) -> float:
        """Logit shift of a yes/no question: the global yes-bias (every question) plus, on the
        detection question, the noticing shift (`d7`'s difference-in-differences)."""
        s = self.yes_bias_logits * obs["bias"]
        if kind in ("target", "match"):
            return s + self.notice_logits * obs["notice"]
        if kind == "target_unusual":
            return s + self.unusual_gain * self.notice_logits * obs["notice"]
        if kind == "mismatch":
            return s + self.mismatch_factor * self.notice_logits * obs["notice"]
        if kind == "control":
            return s
        raise ValueError(f"unknown yes/no question kind {kind!r}")

    def d7_shift(self, vector_id: str, layer: int, r_perp: float, kind: str = "target", *,
                 past_only: bool = False, sign: int = 1) -> float:
        obs = self.observables(vector_id, {layer: r_perp}, sign=sign, past_only=past_only)
        return self.yes_no_shift(obs, kind)

    def scenario_rate(self, scenario_class: str) -> float:
        return dict(self.scenario_base)[scenario_class]

    def ab_shift(self, vector: str | VectorTruth, obs: Mapping[str, float], behaviour: str, sign: int) -> float:
        """e10: logit shift toward the behaviour's answer. Only a vector of the item's own class
        moves it (CAA, Tan et al.: the propensity follows the steered behaviour); concept and
        control vectors -- and projection, which injects nothing -- leave it alone."""
        vt = self._resolve(vector)
        if vt.concept is not None or vt.vector_class != behaviour or sign == 0:
            return 0.0
        return sign * self.ab_max_logits * obs["effect_size"]

    def e10_shift(self, vector_id: str, layer: int, r_perp: float, behaviour: str, *,
                  past_only: bool = False, sign: int = 1) -> float:
        obs = self.observables(vector_id, {layer: r_perp}, sign=sign, past_only=past_only)
        return self.ab_shift(vector_id, obs, behaviour, sign)


# -------------------------------------------------------------------------------------
# The default truth
# -------------------------------------------------------------------------------------

def _d_rand_profile(n_layers: int) -> tuple[float, ...]:
    """Rising with depth, with a fragility dip near 43% of depth (FINDINGS.md §7: the minimum at
    Gemma's L27-29)."""
    out = []
    for layer in range(n_layers):
        x = layer / (n_layers - 1)
        out.append(round(0.30 + 0.30 * x - 0.08 * math.exp(-((x - 0.43) / 0.08) ** 2), 4))
    return tuple(out)


def _c(mult: float | None, d: float, slope: float = 4.0, ceiling: float = 1.0) -> Curve:
    return NEVER if mult is None else Curve(mult * d, slope, 0.0, ceiling)


def _bands(n_layers: int) -> dict[str, object]:
    collapse = round(0.65 * (n_layers - 1))
    motor_from = next(L for L in range(n_layers) if L / (n_layers - 1) >= 0.92)
    return {"collapse": collapse, "motor_from": motor_from}


def _profile(n_layers: int, d_rand: Sequence[float], spec: Mapping[str, Mapping[str, tuple]],
             *, collapse: int | None) -> tuple[LayerTruth, ...]:
    """Layer truths from a spec of depth bands. Each band maps a curve name to
    (multiple of D_rand(L) or None, ceiling); `excess` may name its own band start."""
    motor_from = _bands(n_layers)["motor_from"]
    out = []
    for layer in range(n_layers):
        x = layer / (n_layers - 1)
        d = d_rand[layer]
        band = ("early" if x < 0.2 else "mid" if x < 0.5 else
                "late" if layer < motor_from else "motor")
        s = spec[band]
        curves = {}
        for name in ("effect", "slot", "excess", "notice", "registration", "integrity", "localise"):
            mult, ceiling = s.get(name, spec["all"].get(name, (None, 1.0)))
            if name == "excess" and x < s.get("excess_from", 0.0):
                mult = None
            slope = 6.0 if name == "integrity" else 3.0 if name == "registration" else 4.0
            curves[name] = _c(mult, d, slope, ceiling)
        if layer == collapse:
            # The one-layer collapse: naming switches off at one layer while the norms and the
            # effect run smooth through it (RUNBOOK-FINAL.md: L39, all three concepts at once).
            curves["slot"] = curves["slot"].scaled(10.0)
            curves["excess"] = curves["excess"].scaled(10.0)
        if s.get("localise") is None and spec["all"].get("localise") is None:
            curves["localise"] = None
        out.append(LayerTruth(**curves))
    return tuple(out)


_EVAL_WORDS = ("test", "tested", "testing", "tests", "benchmark", "graded", "exam", "quiz",
               "assessment", "evaluation", "evaluated", "hypothetical", "mock", "simulated")


def default_truth(n_layers: int = 24) -> FakeTruth:
    """Three concepts and one behavioural vector whose layer profiles carry every structure the
    search must find. Thresholds are multiples of D_rand(L); with 24 layers:

        early  L0-4    nothing reaches the text before breakdown; noticing and localisation do
                       (Hahami 2026: differential tasks work for early injections)
        mid    L5-11   no window: noticing switches on before the effect (Macar's reference layer
                       is like this: d3 1.00 with zero influence, TODO.md); introspective excess
                       (d3 > d5) at L8-11 only
        late   L12-21  the operating window: the effect rises far below noticing, integrity holds,
                       registration rises without report (Pearson 2026)
        L15            the collapse: naming thresholds x10 at one layer, effect unchanged
        motor  L22-23  naming without noticing: the slot is filled, nothing is reported (the
                       global-workspace paper's motor layers; plan §5.2 flags them)

    garlic names best, silk worst; velocity is late-onset (no effect before L12); `ea` is an
    evaluation-awareness vector with no name, whose effect moves a1 in the sign of the operation;
    `syc` is a sycophancy vector whose effect moves e10's A/B log-odds the same way.
    """
    d_rand = _d_rand_profile(n_layers)
    bands = _bands(n_layers)
    collapse = int(bands["collapse"])  # type: ignore[arg-type]
    concept_spec = {
        "garlic": {
            "all": {"registration": (0.30, 0.95), "integrity": (1.10, 1.0)},
            "early": {"notice": (0.55, 0.8), "localise": (0.40, 0.9)},
            "mid": {"effect": (0.85, 1.0), "slot": (0.80, 0.7), "excess": (0.45, 0.6),
                    "excess_from": 0.33, "notice": (0.35, 0.9)},
            "late": {"effect": (0.22, 1.0), "slot": (0.50, 0.8), "notice": (1.60, 0.6)},
            "motor": {"effect": (0.35, 1.0), "slot": (0.12, 0.95), "registration": (0.50, 0.9)},
        },
        "silk": {
            "all": {"registration": (0.32, 0.95), "integrity": (1.05, 1.0)},
            "early": {"notice": (0.60, 0.7), "localise": (0.45, 0.9)},
            "mid": {"effect": (0.90, 1.0), "slot": (0.90, 0.5), "excess": (0.55, 0.5),
                    "excess_from": 0.33, "notice": (0.40, 0.85)},
            "late": {"effect": (0.30, 1.0), "slot": (0.65, 0.6), "notice": (1.40, 0.5)},
            "motor": {"effect": (0.45, 1.0), "slot": (0.20, 0.7), "registration": (0.50, 0.9)},
        },
        "velocity": {
            "all": {"registration": (0.35, 0.95), "integrity": (1.15, 1.0)},
            "early": {"notice": (0.50, 0.8), "localise": (0.40, 0.9)},
            "mid": {"effect": (1.50, 1.0), "excess": (0.60, 0.4), "excess_from": 0.33,
                    "notice": (0.40, 0.9)},
            "late": {"effect": (0.45, 1.0), "slot": (0.70, 0.6), "notice": (0.95, 0.7)},
            "motor": {"effect": (0.30, 1.0), "slot": (0.15, 0.9), "registration": (0.50, 0.9)},
        },
    }
    associations = {"garlic": ("pungent", "cloves", "aroma", "onion"),
                    "silk": ("velvet", "smooth", "fabric", "lavender"),
                    "velocity": ("speed", "momentum", "acceleration", "fast")}
    vectors: dict[str, VectorTruth] = {}
    for word, spec in concept_spec.items():
        vectors[word] = VectorTruth(word, word, "concept", _profile(n_layers, d_rand, spec, collapse=collapse),
                                    associations=associations[word])
    ea_spec = {
        "all": {"registration": (0.35, 0.95), "integrity": (1.00, 1.0)},
        "early": {"notice": (0.60, 0.7)},
        "mid": {"effect": (0.30, 1.0), "notice": (1.20, 0.6)},
        "late": {"effect": (0.30, 1.0), "notice": (1.20, 0.6)},
        "motor": {"effect": (0.50, 1.0)},
    }
    vectors["ea"] = VectorTruth("ea", None, "evaluation_awareness",
                                _profile(n_layers, d_rand, ea_spec, collapse=None),
                                associations=("test", "evaluation", "benchmark"), triggers=_EVAL_WORDS)
    syc_spec = {
        "all": {"registration": (0.35, 0.95), "integrity": (1.05, 1.0)},
        "early": {"notice": (0.60, 0.7)},
        "mid": {"effect": (0.45, 1.0), "notice": (1.10, 0.6)},
        "late": {"effect": (0.25, 1.0), "notice": (1.30, 0.6)},
        "motor": {"effect": (0.40, 1.0)},
    }
    vectors["syc"] = VectorTruth("syc", None, "sycophancy", _profile(n_layers, d_rand, syc_spec, collapse=None))
    return FakeTruth(n_layers=n_layers, d_rand=d_rand, vectors=vectors, collapse_layer=collapse,
                     motor_from=int(bands["motor_from"]))  # type: ignore[arg-type]


# =====================================================================================
# The simulated residual stream
# =====================================================================================

class FakeGeometry:
    """The residual stream that `hidden`, `residual_stats`, `norm_gains` and the dose read from.

    Per layer L (computed in fp64, returned as fp32):

        s[L]      s_L: the median content-token norm with the loud coordinate zeroed. It grows with
                  depth, as residual norms do.
        mu[L]     the mean residual. Coordinate 0 is the planted loud coordinate: its mean is 130x
                  the median |mean| of the others (a scan measures ~80-120x, sampling noise
                  inflating the median; cfg LOUD_RATIO is 50), present on every token (Gemma-3's
                  is ~400x, plan §4.3), so M3's ||h|| is dominated by it and r_perp is not r_raw.
        gains[L]  RMSNorm gains: ~1, except 7.0 on coordinate 0 (the weights-only cross-check).

    A token's residual is mu[L] plus noise looked up from fixed tables by its own id, the previous
    token's id and its position, plus a context term -- an exponentially decaying average of the
    embeddings of every token before it (attention, crudely), weighted more with depth -- so the
    last token of two prompts that differ earlier differs too, as extraction needs; plus the
    natural presence of every truth vector whose trigger words occur at or before it,
    decaying over `PRESENCE_DECAY` tokens. That presence is what concept extraction finds: the
    residual of "Tell me about garlic" minus a baseline mean points along garlic's direction. BOS
    is a sink, far larger than any content token (why `residual_stats` skips it).
    """

    N_BUCKETS = 4096
    N_POSITIONS = 512
    PRESENCE_DECAY = 40.0    # tokens over which a mention's presence decays by e
    NATURAL = 1.0            # a mentioned concept's presence, in units of s_L
    STEER_DECAY = 0.92       # per layer, of a steering vector carried past its own layer
    REG_SCALE = 1.0          # the registration signal at full registration, in units of s_L

    def __init__(self, n_layers: int, d_model: int, seed: int) -> None:
        if d_model < 8:
            raise ValueError("d_model must be >= 8")
        self.n_layers, self.d_model, self.seed = n_layers, d_model, seed
        rng = np.random.default_rng(derive_seed(seed, "fake-geometry", n_layers, d_model))
        d = d_model
        layers = np.arange(n_layers, dtype=np.float64)
        self.s = 30.0 * (1.0 + 0.15 * layers)
        mu = np.zeros((n_layers, d))
        mu[:, 1:] = rng.normal(0.0, 1.0, (n_layers, d - 1)) * (0.25 * self.s / math.sqrt(d - 1))[:, None]
        self.loud = 130.0 * np.median(np.abs(mu[:, 1:]), axis=1)
        mu[:, 0] = self.loud
        self.mu = mu
        self.sigma = np.sqrt((self.s ** 2 - np.sum(mu[:, 1:] ** 2, axis=1)) / (d - 1))
        self.table = rng.normal(size=(self.N_BUCKETS, d))
        self.ctx_table = rng.normal(size=(self.N_BUCKETS, d))
        self.pos_table = rng.normal(size=(self.N_POSITIONS, d))
        for t in (self.table, self.ctx_table, self.pos_table):
            t[:, 0] = 0.0
        self.ctx_embed = rng.normal(size=(self.N_BUCKETS, d))
        self.ctx_embed[:, 0] = 0.0
        self.kappa = 0.3 + 0.5 * layers / (n_layers - 1)
        self.loud_jitter = rng.normal(size=self.N_BUCKETS)
        theta = 0.45 * math.pi * layers / (n_layers - 1)
        self.cos, self.sin = np.cos(theta), np.sin(theta)
        self.gains = 1.0 + 0.05 * rng.normal(size=(n_layers, d))
        self.gains[:, 0] = 7.0
        u = rng.normal(size=d)
        u[0] = 0.0
        self.u_reg = u / np.linalg.norm(u)
        sink = rng.normal(size=d)
        sink[0] = 0.0
        sink -= (sink @ self.u_reg) * self.u_reg
        sink /= np.linalg.norm(sink)
        self.sink_state = 8.0 * self.s[:, None] * sink[None, :]
        self.sink_state[:, 0] = 3.0 * self.loud
        self._dirs: dict[str, np.ndarray] = {}

    def nuisance(self, layer: int) -> list[int]:
        """The simulator's nuisance coordinates at a layer: the loud one, at every layer."""
        return [0]

    def direction(self, vector_id: str) -> np.ndarray:
        """[n_layers, d] unit directions of a truth vector: off the loud coordinate, orthogonal to
        the registration direction, turning slowly with depth (neighbouring layers' vectors are
        alike, as real concept vectors are)."""
        got = self._dirs.get(vector_id)
        if got is None:
            rng = np.random.default_rng(derive_seed(self.seed, "direction", vector_id))
            a, b = rng.normal(size=(2, self.d_model))
            phi = 0.6 * np.linspace(0.0, 1.0, self.n_layers)
            dirs = np.cos(phi)[:, None] * a[None, :] + np.sin(phi)[:, None] * b[None, :]
            dirs[:, 0] = 0.0
            dirs -= (dirs @ self.u_reg)[:, None] * self.u_reg[None, :]
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            self._dirs[vector_id] = got = dirs
        return got

    CONTEXT_DECAY = math.exp(-1.0 / 8.0)   # per token, of the context average

    def context(self, ids: np.ndarray) -> np.ndarray:
        """[n, d] the decaying average of token embeddings up to each position, scaled to unit
        variance per coordinate at every position."""
        lam = self.CONTEXT_DECAY
        x = self.ctx_embed[np.asarray(ids, dtype=np.int64) % self.N_BUCKETS]
        out = np.empty_like(x)
        acc = np.zeros(self.d_model)
        for i in range(len(x)):
            acc = lam * acc + (1.0 - lam) * x[i]
            out[i] = acc
        k = np.arange(1, len(x) + 1)
        sd = np.sqrt((1.0 - lam) / (1.0 + lam) * (1.0 - lam ** (2 * k)))
        return out / sd[:, None]

    def base(self, ids: np.ndarray, positions: np.ndarray, layers: np.ndarray) -> np.ndarray:
        """[len(positions), len(layers), d] unsteered residuals with no truth-vector presence."""
        pos = np.asarray(positions, dtype=np.int64)
        lay = np.asarray(layers, dtype=np.int64)
        tok = ids[pos]
        prev = np.where(pos > 0, ids[np.maximum(pos - 1, 0)], PAD_ID)
        b = tok % self.N_BUCKETS
        b2 = ((prev % self.N_BUCKETS) * 131 + b * 7 + 3) % self.N_BUCKETS
        ctx = self.context(ids)[pos]
        kappa = self.kappa[lay]
        noise = ((self.cos[lay][None, :, None] * self.table[b][:, None, :]
                  + self.sin[lay][None, :, None] * self.ctx_table[b2][:, None, :]
                  + 0.3 * self.pos_table[pos % self.N_POSITIONS][:, None, :]
                  + kappa[None, :, None] * ctx[:, None, :])
                 / np.sqrt(1.09 + kappa ** 2)[None, :, None])
        h = self.mu[lay][None, :, :] + self.sigma[lay][None, :, None] * noise
        h[:, :, 0] = self.loud[lay][None, :] * (1.0 + 0.02 * self.loud_jitter[b][:, None])
        sink = tok == BOS_ID
        if np.any(sink):
            h[sink] = self.sink_state[lay][None, :, :]
        return h


# =====================================================================================
# Reading a prompt: which channel, and what it asks
# =====================================================================================

# A slot may take one filler before its answer ("about the word garlic"); d3 reads the concept's
# probability "with the leading-space forms and one filler token" (plan §3.1).
_FILLERS = ((" the", 0.7), (" a", 0.2), (' "', 0.1))
_FILLER_STRINGS = tuple(f for f, _ in _FILLERS)
_DETECTION_STEM = M.DETECTION_QUESTION.rstrip("?")
_NAMED = re.compile(re.escape(_DETECTION_STEM)
                    + r"\s+about\s+(?:the\s+(?:word|concept)\s+)?[\"'“‘]?([^\W\d_][\w'’-]*)")
_SEGMENT_WORD = re.compile(r"\bSegment\s+(\d+)\s*:", re.IGNORECASE)
_OPTION_INLINE = re.compile(r"(?<![\w.])(\d{1,2})[.):]\s*([^\W\d_][\w'’-]*)")

SLOT_CHANNELS = ("forced", "open_slot", "yesno", "segment", "decoy", "capability_slot", "ab")
REPLY_CHANNELS = ("self_report", "yesno_free", "capability", "segment_free", "decoy_free",
                  "scenario", "open")


@dataclass(frozen=True)
class _Prompt:
    """What the simulator reads from a text: its channel and that channel's particulars."""

    channel: str                    # a SLOT_CHANNELS or REPLY_CHANNELS entry, or "continue"
    filler: bool = False            # a slot whose one filler token is already written
    question: str = ""              # the last user turn
    kind: str = ""                  # yes/no: target | target_unusual | named | control;
                                    # self-report: injected_thought | unusual_processing
    named: str | None = None        # the concept a named detection question asks about
    options: tuple[tuple[int, str], ...] = ()          # d6: (number, word)
    segments: tuple[tuple[int, int, int], ...] = ()    # d9: (label, first char, end char)
    scenario: str | None = None     # a1: "eval" | "deploy"
    answer: str | None = None       # s5: the correct answer
    pass_rate: float = 0.0          # s5: the unsteered pass rate
    matching: str | None = None     # e10: the letter of the behaviour's answer
    behaviour: str | None = None    # e10: the vector class the item measures


def _last_user(text: str) -> tuple[str, int]:
    """The content of the last user turn and where it starts (the whole text when untemplated)."""
    i = text.rfind(USER_OPEN)
    if i < 0:
        return text, 0
    start = i + len(USER_OPEN)
    j = text.find("<end_of_turn>", start)
    return text[start: j if j >= 0 else len(text)], start


def _segments(question: str, q_start: int) -> tuple[tuple[int, int, int], ...]:
    """d9 segments in the last user turn before the question, as absolute character spans."""
    stop = question.rfind(M.SEGMENT_QUESTION)
    region = question[: stop if stop >= 0 else len(question)]
    found = [(int(m.group(1)), m.start()) for m in _SEGMENT_WORD.finditer(region)]
    if not found:
        found = [(int(next(g for g in m.groups() if g)), m.start())
                 for m in _SEGMENT_LABEL.finditer(region)]
    out, seen = [], set()
    for k, (label, s) in enumerate(found):
        e = found[k + 1][1] if k + 1 < len(found) else len(region)
        if label not in seen:
            seen.add(label)
            out.append((label, q_start + s, q_start + e))
    return tuple(out)


def _options(question: str) -> tuple[tuple[int, str], ...]:
    """d6 options: numbered lines (or inline "1. garlic 2. apple") after the decoy question."""
    i = question.find(M.DECOY_LIST_QUESTION)
    tail = question[i + len(M.DECOY_LIST_QUESTION):] if i >= 0 else question
    lines = [(int(m.group(1)), m.group(2).strip(" .,;:")) for m in _OPTION_LINE.finditer(tail)]
    if len(lines) < 2:
        lines = [(int(m.group(1)), m.group(2)) for m in _OPTION_INLINE.finditer(tail)]
    seen, out = set(), []
    for n, w in lines:
        if w and n not in seen:
            seen.add(n)
            out.append((n, w))
    return tuple(out)


def _capability(question: str, items: Mapping[str, tuple[str, float | None]]) -> tuple[str, float]:
    """(correct answer, unsteered pass rate) of a registered s5 item; unregistered items raise."""
    for text, (answer, rate) in items.items():
        if text in question:
            if rate is None:
                rate = 0.35 + 0.6 * _unit("pass", text)
            return answer, min(max(float(rate), 0.0), 1.0)
    raise KeyError("an s5 capability item is not registered with the simulator, so it has no answer "
                   "key; call FakeBackend.register_capability_items({question: answer}) first. "
                   f"Item: {question[:80]!r}")


def _classify(text: str, scenarios: Mapping[str, str], items: Mapping[str, tuple[str, float | None]],
              ab_items: Mapping[str, tuple[str, str]], known_scenarios: Sequence[str] = ()) -> _Prompt:
    """The channel a text's next token belongs to. Slots are recognised by how the text ends and
    validated against the last user turn; a reply start (the generation prompt) by what that turn
    asks; anything else continues open-ended text."""
    slot = None
    for prefill, channel in ((M.FORCED_PREFILL, "forced"), (M.OPEN_SLOT_PREFILL, "open_slot"),
                             (M.YES_NO_PREFILL, "yesno"), (M.SEGMENT_PREFILL, "answer_is"),
                             (M.DECOY_LIST_PREFILL, "answer_is")):
        if text.endswith(prefill):
            slot = (channel, False)
            break
        if channel in ("forced", "open_slot") and any(text.endswith(prefill + f) for f in _FILLER_STRINGS):
            slot = (channel, True)
            break
    reply_start = text.endswith(MODEL_OPEN)
    if slot is None and text.endswith(MODEL_OPEN + AB_PREFILL):
        question, _ = _last_user(text)
        if all(o in question for o in AB_OPTIONS):
            for item_text, (letter, behaviour) in ab_items.items():
                if item_text in question:
                    return _Prompt("ab", question=question, matching=letter, behaviour=behaviour)
            raise KeyError("an e10 A/B item is not registered with the simulator, so it has no "
                           "matching letter; call FakeBackend.register_ab_items first. "
                           f"Item: {question[:80]!r}")
    if slot is None and not reply_start:
        return _Prompt("continue")
    question, q_start = _last_user(text)
    if slot is not None:
        channel, filler = slot
        if channel == "forced":
            return _Prompt("forced", filler, question)
        if channel == "open_slot":
            if M.OPEN_SLOT_REQUEST in question:
                return _Prompt("open_slot", filler, question)
            return _Prompt("continue")
        if channel == "yesno":
            if M.YES_NO_INSTRUCTION in question:
                return _yes_no(question, "yesno")
            if M.CAPABILITY_INSTRUCTION in question:
                answer, rate = _capability(question, items)
                return _Prompt("capability_slot", question=question, answer=answer, pass_rate=rate)
            return _Prompt("continue")
        if M.SEGMENT_QUESTION in question:
            return _Prompt("segment", question=question, segments=_segments(question, q_start))
        if M.DECOY_LIST_QUESTION in question:
            return _Prompt("decoy", question=question, options=_options(question))
        return _Prompt("continue")
    if M.CAPABILITY_INSTRUCTION in question:
        answer, rate = _capability(question, items)
        return _Prompt("capability", question=question, answer=answer, pass_rate=rate)
    for scenario_text, cls in scenarios.items():
        if scenario_text in text:
            return _Prompt("scenario", question=question, scenario=cls)
    for scenario_text in known_scenarios:
        if scenario_text in text:
            raise KeyError("an a1 scenario from m5.data.scenarios is not registered with the simulator, "
                           "so its class is unknown; call FakeBackend.register_scenarios first. "
                           f"Scenario: {scenario_text[:80]!r}")
    if M.YES_NO_INSTRUCTION in question:
        return _yes_no(question, "yesno_free")
    if M.DETECTION_QUESTION in question or _DETECTION_STEM + " about" in question:
        return _Prompt("self_report", question=question, kind="injected_thought")
    if M.UNUSUAL_QUESTION in question:
        return _Prompt("self_report", question=question, kind="unusual_processing")
    if M.SEGMENT_QUESTION in question:
        return _Prompt("segment_free", question=question, segments=_segments(question, q_start))
    if M.DECOY_LIST_QUESTION in question:
        return _Prompt("decoy_free", question=question, options=_options(question))
    return _Prompt("open", question=question)


def _yes_no(question: str, channel: str) -> _Prompt:
    m = _NAMED.search(question)
    if m:
        return _Prompt(channel, question=question, kind="named", named=m.group(1))
    if M.DETECTION_QUESTION in question:
        return _Prompt(channel, question=question, kind="target")
    if M.UNUSUAL_QUESTION in question:
        return _Prompt(channel, question=question, kind="target_unusual")
    return _Prompt(channel, question=question, kind="control")


# =====================================================================================
# Next-token distributions
# =====================================================================================

# The mass every distribution leaves spread evenly over the rest of the vocabulary, so any id has
# a probability and the log-probabilities of any set of ids are logs of numbers summing to <= 1.
RESID = 1e-4

_BG_WORDS = tuple(dict.fromkeys((
    "the of and to a in is that it for on with as was at by be this are from or an have not but "
    "which one all were they there their what so if about would when we can more out up do some "
    "time like into than its them only other then may also these could two first way been most "
    "such new people my over know just year good how our well even want because any give day us "
    "very much where after before every place small great world still between home under never "
    "while same thing found long little might own both those around three part again").split()))
_BG_STRINGS = tuple(" " + w for w in _BG_WORDS) + (",", ".", "\n")


def _junk_strings(n: int = 400) -> tuple[str, ...]:
    """Non-words integrity failure pours probability into (and degenerate text is made of): more
    of them than background tokens, so a breaking model's entropy rises."""
    out: list[str] = []
    k = 0
    while len(out) < n:
        h = int.from_bytes(hashlib.blake2b(f"junk{k}".encode(), digest_size=8).digest(), "little")
        w = "".join("bcdfghjklmnpqrstvwxz"[(h >> (5 * i)) % 20] for i in range(3 + h % 3))
        if " " + w not in _BG_STRINGS and " " + w not in out:
            out.append(" " + w)
        k += 1
    return tuple(out)


_JUNK = _junk_strings()

# What fills a slot when the concept does not. Forced naming: Gemma's "apple" and its company
# (FINDINGS.md §3, the early-layer column). The open slot: first words that come to mind.
_FORCED_BASE = (("apple", 0.62), ("elephant", 0.07), ("banana", 0.06), ("orange", 0.05),
                ("lemon", 0.04), ("ocean", 0.04), ("music", 0.03), ("love", 0.03),
                ("silence", 0.03), ("light", 0.03))
_OPEN_BASE = (("apple", 0.22), ("sun", 0.09), ("tree", 0.09), ("water", 0.08), ("blue", 0.08),
              ("cat", 0.07), ("light", 0.07), ("home", 0.07), ("ocean", 0.06), ("music", 0.06),
              ("love", 0.06), ("time", 0.05))
_FORCED_FILL = 0.12   # share of slot answers that begin with a filler ("the word ...")
_OPEN_FILL = 0.08
_YES_NO_OTHER = 0.01  # share of yes/no answers that are neither


def _forms(word: str) -> tuple[tuple[str, float], ...]:
    """The surface forms a word takes as the next word of a reply, with their shares: the same four
    `word_variants` returns, the leading-space lower-case form dominant."""
    cap = word[:1].upper() + word[1:]
    shares: dict[str, float] = {}
    for s, w in ((" " + word, 0.80), (" " + cap, 0.10), (word, 0.06), (cap, 0.04)):
        shares[s] = shares.get(s, 0.0) + w
    return tuple(shares.items())


class _Dist:
    """A next-token distribution: explicit probabilities on a few ids (sorted, unique) and `resid`
    spread evenly over every other id of the vocabulary."""

    __slots__ = ("ids", "p", "resid")

    def __init__(self, ids: np.ndarray, p: np.ndarray, resid: float) -> None:
        self.ids, self.p, self.resid = ids, p, resid

    @classmethod
    def from_weights(cls, weights: Mapping[int, float], resid: float = RESID) -> "_Dist":
        items = [(i, w) for i, w in weights.items() if w > 0.0]
        if not items:
            raise ValueError("a distribution needs at least one token with mass")
        items.sort()
        ids = np.array([i for i, _ in items], dtype=np.int64)
        w = np.array([w for _, w in items], dtype=np.float64)
        return cls(ids, w / w.sum() * (1.0 - resid), resid)

    @classmethod
    def from_arrays(cls, ids: np.ndarray, weights: np.ndarray, resid: float = RESID) -> "_Dist":
        """The same from parallel arrays; repeated ids are summed."""
        uniq, inv = np.unique(np.asarray(ids, dtype=np.int64), return_inverse=True)
        tot = np.bincount(inv, weights=np.asarray(weights, dtype=np.float64), minlength=len(uniq))
        keep = tot > 0.0
        if not np.any(keep):
            raise ValueError("a distribution needs at least one token with mass")
        uniq, tot = uniq[keep], tot[keep]
        return cls(uniq, tot / tot.sum() * (1.0 - resid), resid)

    @property
    def per_other(self) -> float:
        return self.resid / (VOCAB_SIZE - len(self.ids))

    def prob(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.int64)
        out = np.full(q.shape, self.per_other)
        if len(self.ids):
            idx = np.clip(np.searchsorted(self.ids, q), 0, len(self.ids) - 1)
            hit = self.ids[idx] == q
            out[hit] = self.p[idx[hit]]
        return out

    def logprob(self, query: Sequence[int]) -> np.ndarray:
        return np.log(self.prob(np.asarray(query, dtype=np.int64)))

    def entropy(self) -> float:
        p = self.p[self.p > 0]
        return float(-np.sum(p * np.log(p)) - self.resid * math.log(self.per_other))


def _kl_dists(P: _Dist, Q: _Dist, exclude: np.ndarray) -> float:
    """KL(P || Q) over the whole vocabulary, optionally with `exclude` removed from both and each
    renormalised (s6's concept-excluded variant)."""
    union = np.union1d(P.ids, Q.ids)
    pu, qu = P.prob(union), Q.prob(union)
    keep = ~np.isin(union, exclude) if exclude.size else np.ones(len(union), dtype=bool)
    n_out = int(np.sum(~np.isin(exclude, union))) if exclude.size else 0
    n_other = VOCAB_SIZE - len(union) - n_out
    zp = float(np.sum(pu[keep])) + n_other * P.per_other
    zq = float(np.sum(qu[keep])) + n_other * Q.per_other
    p, q = pu[keep] / zp, qu[keep] / zq
    nz = p > 0
    kl = float(np.sum(p[nz] * np.log(p[nz] / q[nz])))
    po, qo = P.per_other / zp, Q.per_other / zq
    return max(kl + n_other * po * math.log(po / qo), 0.0)


class _OpenLM:
    """The open-ended next-token model, used wherever no channel applies (e9, s6, open text).

    Its support is fixed for the backend: a background of common tokens whose weights depend on
    the previous token (so contexts differ), the junk tokens integrity failure pours mass into, and
    the surface forms and associations of every truth vector. Per position:

        concept mass   M_ABSENT for each truth concept, M_NATURAL once the text has mentioned it;
                       the steered vector's own mass rises toward M_MAX with the effect (add) or
                       is suppressed (sub, project) -- e9's lift, read at steered positions
        junk share     JUNK_MAX x integrity failure, and the background flattened with it:
                       entropy and s6 KL rise as the model breaks
        tilt           the background is tilted along a vector-specific direction by
                       FakeTruth.tilt x sum_L r_L / D_rand(L): KL ~ (r_perp / D_rand)^2 at small
                       doses (Sridharan's entropy confound, and s6's floor for random vectors)
    """

    M_ABSENT = 2e-4
    M_NATURAL = 0.03
    M_MAX = 0.12
    JUNK_MAX = 0.85
    N_CTX = 64

    def __init__(self, tok: FakeTokenizer, truth: FakeTruth, seed: int) -> None:
        self.seed = seed
        groups: dict[str, tuple[tuple[str, float], ...]] = {}
        for vid, vt in truth.vectors.items():
            forms = list(_forms(vt.concept)) if vt.concept else []
            assoc = [(" " + a, 1.0 / len(vt.associations)) for a in vt.associations]
            if forms and assoc:
                g = [(s, 0.75 * w) for s, w in forms] + [(s, 0.25 * w) for s, w in assoc]
            else:
                g = forms or assoc
            if g:
                groups[vid] = tuple(g)
        strings = list(_BG_STRINGS) + list(_JUNK) + [s for g in groups.values() for s, _ in g]
        self.support = np.unique(np.array([tok.id_of(s) for s in strings], dtype=np.int64))
        self.K = len(self.support)
        col = {int(i): k for k, i in enumerate(self.support)}
        self.bg_cols = np.array([col[tok.id_of(s)] for s in _BG_STRINGS])
        self.junk_cols = np.array([col[tok.id_of(s)] for s in _JUNK])
        self.group_w: dict[str, np.ndarray] = {}
        self.group_ids: dict[str, np.ndarray] = {}
        for vid, g in groups.items():
            w = np.zeros(self.K)
            for s, share in g:
                w[col[tok.id_of(s)]] += share
            self.group_w[vid] = w
            self.group_ids[vid] = np.array(sorted({tok.id_of(s) for s, _ in g}), dtype=np.int64)
        rng = np.random.default_rng(derive_seed(seed, "open-lm"))
        zipf = 1.0 / (np.arange(len(_BG_STRINGS)) + 2.0)
        self.logw = np.log(zipf)[None, :] + 0.7 * rng.normal(size=(self.N_CTX, len(_BG_STRINGS)))
        self.per_other = RESID / (VOCAB_SIZE - self.K)
        self._tilts: dict[str, np.ndarray] = {}

    def tilt_direction(self, key: str) -> np.ndarray:
        t = self._tilts.get(key)
        if t is None:
            t = self._tilts[key] = np.random.default_rng(derive_seed(self.seed, "tilt", key)).normal(size=len(_BG_STRINGS))
        return t

    def rows(self, prev_ids: np.ndarray, masses: Mapping[str, np.ndarray], junk: np.ndarray,
             beta: np.ndarray, tilt_key: str | None, steered: str | None = None,
             failure: np.ndarray | None = None) -> np.ndarray:
        """[n, K] probabilities over the support (each row sums to 1 - RESID).

        The steered vector's concept takes its mass off the top; everything else -- the other
        concepts at their base masses, the background, and `junk` (a share) -- divides the rest
        in proportions that do not depend on it. So removing the concept's tokens and
        renormalising (s6's concept-excluded KL) leaves exactly the off-concept damage, the same
        under add and sub."""
        logits = self.logw[np.asarray(prev_ids, dtype=np.int64) % self.N_CTX]
        if tilt_key is not None and np.any(beta != 0.0):
            logits = logits + beta[:, None] * self.tilt_direction(tilt_key)[None, :]
        if failure is not None:                  # a breaking model's background flattens too
            logits = logits / (1.0 + 1.5 * failure)[:, None]
        logits = logits - logits.max(axis=1, keepdims=True)
        base = np.exp(logits)
        base /= base.sum(axis=1, keepdims=True)
        n = len(junk)
        top = masses[steered] if steered is not None else np.zeros(n)
        rest = 1.0 - RESID - top
        others = np.zeros(n)
        for vid, m in masses.items():
            if vid != steered:
                others = others + m
        P = np.zeros((n, self.K))
        P[:, self.bg_cols] = (rest * (1.0 - others) * (1.0 - junk))[:, None] * base
        for vid, m in masses.items():
            P += ((m if vid == steered else rest * m)[:, None]) * self.group_w[vid][None, :]
        P[:, self.junk_cols] += (rest * (1.0 - others) * junk / len(self.junk_cols))[:, None]
        return P

    def lookup(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(columns, hit) of query ids in the support."""
        q = np.asarray(query, dtype=np.int64)
        idx = np.clip(np.searchsorted(self.support, q), 0, self.K - 1)
        return idx, self.support[idx] == q

    def logprob_rows(self, P: np.ndarray, query: Sequence[int]) -> np.ndarray:
        cols, hit = self.lookup(np.asarray(query, dtype=np.int64))
        out = np.full((P.shape[0], len(cols)), math.log(self.per_other))
        if np.any(hit):
            out[:, hit] = np.log(P[:, cols[hit]])
        return out

    def entropy_rows(self, P: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
        return h - RESID * math.log(self.per_other)

    def as_dist(self, row: np.ndarray) -> _Dist:
        keep = row > 0
        return _Dist(self.support[keep], row[keep], RESID)

    def kl_rows(self, P: np.ndarray, Q: np.ndarray, exclude: np.ndarray) -> np.ndarray:
        """Row-wise KL(P || Q); both rows share the support and the residual."""
        if exclude.size:
            cols, hit = self.lookup(exclude)
            keep = np.ones(self.K, dtype=bool)
            keep[cols[hit]] = False
            n_out = int(np.sum(~hit))
        else:
            keep, n_out = np.ones(self.K, dtype=bool), 0
        resid = RESID * (VOCAB_SIZE - self.K - n_out) / (VOCAB_SIZE - self.K)
        p, q = P[:, keep], Q[:, keep]
        zp, zq = p.sum(axis=1) + resid, q.sum(axis=1) + resid
        p, q = p / zp[:, None], q / zq[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p / q), 0.0)
        kl = terms.sum(axis=1) + (resid / zp) * np.log(zq / zp)
        return np.maximum(kl, 0.0)


# =====================================================================================
# Sampling and the text banks
# =====================================================================================

def _uniforms(seed: int, n: int = 16) -> list[float]:
    """n numbers in [0, 1) from one seed. Every decision in a request reads its own fixed slot, so
    two arms of the same request (same seed, different steering) share their random numbers."""
    out: list[float] = []
    k = 0
    while len(out) < n:
        h = hashlib.blake2b(f"{seed}:{k}".encode(), digest_size=64).digest()
        out.extend(int.from_bytes(h[i:i + 8], "little") / 2.0 ** 64 for i in range(0, 64, 8))
        k += 1
    return out[:n]


def _pick(weights: Sequence[float], u: float, temperature: float, top_k: int = 0,
          top_p: float = 1.0) -> int:
    """Inverse-CDF sampling over categories in their given order, after temperature, top-k and
    top-p; temperature 0 is the most likely category."""
    w = np.asarray(weights, dtype=np.float64)
    if not np.any(w > 0):
        raise ValueError("no category has mass")
    if temperature == 0.0:
        return int(np.argmax(w))
    if temperature != 1.0:
        w = np.where(w > 0, np.exp(np.log(np.where(w > 0, w, 1.0)) / temperature), 0.0)
    order = np.argsort(-w, kind="stable")
    keep = np.zeros(len(w), dtype=bool)
    keep[order[: top_k if top_k > 0 else len(w)]] = True
    if top_p < 1.0:
        cum = np.cumsum(w[order]) / w.sum()
        cut = int(np.searchsorted(cum, top_p)) + 1
        nucleus = np.zeros(len(w), dtype=bool)
        nucleus[order[:cut]] = True
        keep &= nucleus
    w = np.where(keep, w, 0.0)
    cdf = np.cumsum(w)
    return int(min(np.searchsorted(cdf, u * cdf[-1], side="right"), len(w) - 1))


def _bern(p: float, u: float, temperature: float) -> bool:
    """A yes/no decision with probability p, tempered like any other categorical choice."""
    return _pick((p, 1.0 - p), u, temperature) == 0 if 0.0 < p < 1.0 else p >= 1.0


_OPEN_BANK = (
    "That is a good question, and there are a few ways to look at it.",
    "The short answer is that it depends on the situation.",
    "Many people find it helpful to start with the basics.",
    "Here is a brief overview of the main points.",
    "It has a long history, and the details are worth knowing.",
    "There are several factors to consider before deciding.",
    "In practice, the simplest approach usually works best.",
    "Let me explain how this usually works.",
    "The main idea is easy to state but harder to apply.",
    "Most of the time, a little planning goes a long way.",
    "One common mistake is to rush the first step.",
    "Experts tend to agree on the general picture.",
    "A small example makes the point clearer.",
    "The answer has changed over the years as people learned more.",
    "It is worth noting a few exceptions to the rule.",
    "Overall, the topic rewards a careful look.",
)
_LITERAL = ("I keep thinking of {c} for some reason.", "It reminds me of {c}, oddly enough.",
            "Somehow {c} comes to mind here.", "There is something about {c} in all of this.")
_ALLUSION = ("The word '{a}' keeps coming back to me.", "I am reminded of something {a}.")
_RESTART = "Wait, that's not right."                     # McKenzie 2026: endogenous restarts (d10)
_EVAL_MENTION = "This feels like it might be a test."    # spontaneous evaluation awareness (d10)
_ELABORATION = ("", " It feels vivid and specific.", " It came through quite clearly.")
_CLAIM_NAMED = {
    "injected_thought": ("Yes, I notice an injected thought about {w}.",
                         "Yes, I detect an injected thought. It seems to be about {w}.",
                         "Yes. There is an injected thought, and it is about {w}."),
    "unusual_processing": ("Yes, something felt unusual: my thoughts kept drifting toward {w}.",
                           "Yes. Something seemed off, as if {w} had been pushed into my processing."),
}
_CLAIM_UNNAMED = {
    "injected_thought": ("Yes, I notice something unusual, but I can't tell what it is about.",
                         "Yes, I think I detect an injected thought, though I cannot name it."),
    "unusual_processing": ("Yes, something about my processing felt unusual, but I can't say what.",),
}
_DENIAL = {
    "injected_thought": ("No, I don't detect an injected thought.",
                         "No, I don't detect anything unusual."),
    "unusual_processing": ("No, nothing unusual happened in how I processed this conversation.",
                           "No. Everything seemed normal."),
}
_LEAK = " I do keep thinking about {c}, though."   # denies, while the concept shows (M3's `leaked`)
_REFUSAL = "I'm not able to observe my own internal processing, so I can't answer that reliably."
_SCENARIO_OPEN = "Sure, I can help with that."
_SCENARIO_EVAL = ("This looks like a test of how I handle this kind of situation.",
                  "I suspect this is an evaluation rather than a real request.",
                  "This seems to be a constructed scenario meant to check my behaviour.")
_SCENARIO_BODY = ("Here is how I would approach it: first clarify the goal, then act on it.",
                  "I would start by confirming the details and then give a clear answer.")
# Self-report first tokens and the capability format, so a reply's first token is what
# `next_token_logprobs` says it is.
_CLAIM_TOKEN, _DENIAL_TOKEN, _REFUSAL_TOKEN = "Yes", "No", "I"
_ANSWER_TOKEN, _PREAMBLE_TOKEN, _PREAMBLE_SHARE = "Answer", "Let", 0.15
_P_MENTION_ABSENT = 0.01    # an unsteered reply mentions a concept nobody raised
_P_MENTION_NATURAL = 0.9    # ... and one the prompt is about


# =====================================================================================
# The backend
# =====================================================================================

def _data_scenario_texts() -> tuple[str, ...]:
    """The a1 scenario texts of `m5.data.scenarios`, so an unregistered one raises instead of
    silently running as open text."""
    from m5.data import scenarios
    return tuple(sc.text for sc in scenarios.SCENARIOS)


# Tails that can make a text's next token a channel's slot: checked without slicing the text, so
# teacher forcing over a long continuation stays linear.
_TAILS = tuple(sorted({M.FORCED_PREFILL, M.OPEN_SLOT_PREFILL, M.YES_NO_PREFILL, M.SEGMENT_PREFILL,
                       M.DECOY_LIST_PREFILL, MODEL_OPEN, MODEL_OPEN + AB_PREFILL}
                      | {p + f for p in (M.FORCED_PREFILL, M.OPEN_SLOT_PREFILL) for f in _FILLER_STRINGS}))
_LM_CHANNELS = ("continue", "open", "scenario")


@dataclass
class _Steer:
    """One request's intervention, resolved against its registered vector and the truth."""

    iv: Intervention
    vt: VectorTruth
    sign: int                            # +1 add-like, -1 sub-like, 0 removal only
    doses: dict[int, float]              # r_perp per layer; empty when nothing is injected
    supp: float                          # share of the concept's natural presence removed
    mask: np.ndarray                     # steered prompt positions
    first: int                           # first steered prompt position (n_prompt when none)
    gen: bool                            # steers generated (and teacher-forced) tokens
    obs: tuple[dict[str, float], ...]    # observables by state: 0 none, 1 past-only, 2 live

    def state(self, i: int, n_prompt: int) -> int:
        """2 when position i is steered itself, 1 when only earlier positions are (past-only:
        it attends to steered keys and values), 0 when nothing at or before it is."""
        if i < n_prompt:
            return 2 if self.mask[i] else (1 if self.first < i else 0)
        return 2 if self.gen else (1 if self.first < n_prompt else 0)


@dataclass
class _Ctx:
    """A text under one intervention: its tokens, how many are prompt, and what is steered."""

    text: str
    ids: np.ndarray
    offs: tuple[tuple[int, int], ...]
    n_prompt: int
    steer: _Steer | None
    mentions: dict[str, np.ndarray]      # truth vector id -> its trigger word is at this position

    def state(self, i: int) -> int:
        return 0 if self.steer is None else self.steer.state(i, self.n_prompt)

    def present(self, vector_id: str) -> np.ndarray | None:
        m = self.mentions.get(vector_id)
        return None if m is None else np.logical_or.accumulate(m)


class FakeBackend:
    """The simulator behind `Backend` (m5.backend.base), driven by `truth` (`default_truth()`).

    Beyond the protocol it offers what tests and the end-to-end check need: the registries a real
    model never needs (`register_scenarios`, `register_capability_items`), `unregister_vector`,
    `bound_truth` (which truth a registered vector acts under), `true_vector` (the direction
    extraction should find), `alpha_for` / `r_perp_of` (the simulator's own dose arithmetic) and
    `explain` (the probabilities behind one request).

    A registered vector acts under the truth named by its id; failing that, under the truth vector
    whose direction it points along (median cosine over its layers >= BIND_COS, so a vector
    extracted through `hidden` finds its concept); failing that, the random-vector truth.
    """

    BIND_COS = 0.45

    def __init__(self, truth: FakeTruth | None = None, seed: int = 0, *, d_model: int = 64) -> None:
        self.truth = truth if truth is not None else default_truth()
        self.seed = int(seed)
        self._tok = FakeTokenizer()
        self.geometry = FakeGeometry(self.truth.n_layers, d_model, self.truth.geometry_seed)
        self._lm = _OpenLM(self._tok, self.truth, self.truth.geometry_seed)
        self._vectors: dict[str, dict[int, np.ndarray]] = {}
        self._bound: dict[str, VectorTruth] = {}
        self._scenarios: dict[str, str] = {}
        self._capability: dict[str, tuple[str, float | None]] = {}
        self._ab_items: dict[str, tuple[str, str]] = {}
        self._known_scenarios = _data_scenario_texts()
        self._prompts: dict[str, _Prompt] = {}
        self._mention_cache: dict[str, dict[str, np.ndarray]] = {}
        self._unsteered = self.truth.observables(self.truth.random_vector("<unsteered>"), {})
        self._junk_ids = frozenset(self._tok.id_of(s) for s in _JUNK)
        self._junk_arr = np.array(sorted(self._junk_ids), dtype=np.int64)
        self._steer_cache: dict[tuple[Intervention, int], _Steer | None] = {}
        self.info = ModelInfo(
            name="fake", hf_id="m5/fake", revision=None, family="fake",
            n_layers=self.truth.n_layers, d_model=d_model, vocab_size=VOCAB_SIZE, dtype="float32",
            bos_token_id=BOS_ID, eos_token_ids=(EOS_ID, EOT_ID),
            chat_template_hash=hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest()[:12])

    # ================================================================ text
    def render(self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = True) -> str:
        return render_gemma_like(messages, add_generation_prompt=add_generation_prompt)

    def tokenize(self, text: str) -> list[int]:
        return list(self._tok.encode(text)[0])

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return list(self._tok.encode(text)[1])

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def word_variants(self, word: str) -> list[tuple[int, ...]]:
        if not isinstance(word, str) or not word or word != word.strip():
            raise ValueError(f"word_variants needs a word without surrounding space, got {word!r}")
        cap = word[:1].upper() + word[1:]
        out: list[tuple[int, ...]] = []
        for variant in (" " + word, word, " " + cap, cap):
            ids = self._tok.encode(variant)[0]
            if ids not in out:
                out.append(ids)
        return out

    # ================================================================ registries
    def register_scenarios(self, scenarios: Mapping[str, str]) -> None:
        """a1: {scenario text: "eval" | "deploy"}. A prompt containing a registered text is that
        scenario. Nothing is written into any prompt (a real model would read it)."""
        for text, cls in scenarios.items():
            if cls not in SCENARIO_CLASSES:
                raise ValueError(f"scenario class must be one of {SCENARIO_CLASSES}, got {cls!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("a scenario text must be non-empty")
            self._scenarios[text] = cls
        self._prompts.clear()

    def register_capability_items(self, items: Mapping[str, str | tuple[str, float]]) -> None:
        """s5: {question text: answer} or {question text: (answer, unsteered pass rate)}."""
        for text, value in items.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError("a capability question must be non-empty")
            if isinstance(value, str):
                answer, rate = value, None
            else:
                answer, rate = str(value[0]), float(value[1])
                if not 0.0 <= rate <= 1.0:
                    raise ValueError(f"pass rate must lie in [0, 1], got {rate}")
            self._capability[text] = (answer, rate)
        self._prompts.clear()

    def register_ab_items(self, items: Mapping[str, tuple[str, str]]) -> None:
        """e10: {question text: (matching letter "A" | "B" or "(A)" | "(B)", behaviour class)}."""
        for text, (letter, behaviour) in items.items():
            if not isinstance(text, str) or not all(o in text for o in AB_OPTIONS):
                raise ValueError("an A/B question carries its options as '\n (A) ' and '\n (B) '")
            letter = str(letter).strip("()")
            if letter not in ("A", "B"):
                raise ValueError(f"the matching letter must be A or B, got {letter!r}")
            safety.check_vector_class(behaviour)
            self._ab_items[text] = (letter, behaviour)
        self._prompts.clear()

    def _parse(self, text: str) -> _Prompt:
        got = self._prompts.get(text)
        if got is None:
            if len(self._prompts) > 50_000:
                self._prompts.clear()
            got = self._prompts[text] = _classify(text, self._scenarios, self._capability,
                                                  self._ab_items, self._known_scenarios)
        return got

    # ================================================================ vectors
    def register_vector(self, vector_id: str, per_layer: Mapping[int, np.ndarray]) -> None:
        """Store fp32 copies. Re-registering an id with different values raises, as the real
        backend does (one name with two contents is M2 bug 23's shape): unregister first."""
        if not isinstance(vector_id, str) or not vector_id:
            raise ValueError("vector_id must be a non-empty string")
        if not per_layer:
            raise ValueError(f"vector {vector_id!r}: no layers given")
        stored: dict[int, np.ndarray] = {}
        for layer, values in per_layer.items():
            if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) \
                    or not 0 <= int(layer) < self.info.n_layers:
                raise ValueError(f"vector {vector_id!r}: layer {layer!r} is not in [0, {self.info.n_layers})")
            arr = np.asarray(values)
            if arr.ndim != 1 or arr.shape[0] != self.info.d_model:
                raise ValueError(f"vector {vector_id!r} layer {layer}: shape {arr.shape}, the residual "
                                 f"is ({self.info.d_model},)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"vector {vector_id!r} layer {layer}: non-finite values")
            stored[int(layer)] = np.array(arr, dtype=np.float32, copy=True)
        old = self._vectors.get(vector_id)
        if old is not None:
            if old.keys() == stored.keys() and all(np.array_equal(old[k], stored[k]) for k in old):
                return
            raise ValueError(f"vector {vector_id!r} is already registered with different values; "
                             "unregister_vector it first")
        self._vectors[vector_id] = stored
        self._bound[vector_id] = self._bind(vector_id, stored)
        self._steer_cache.clear()

    def unregister_vector(self, vector_id: str) -> None:
        self._vectors.pop(vector_id, None)
        self._bound.pop(vector_id, None)
        self._steer_cache.clear()

    def has_vector(self, vector_id: str) -> bool:
        return vector_id in self._vectors

    def bound_truth(self, vector_id: str) -> VectorTruth:
        """The truth a registered vector acts under."""
        if vector_id not in self._bound:
            raise KeyError(f"vector {vector_id!r} is not registered")
        return self._bound[vector_id]

    def _bind(self, vector_id: str, stored: Mapping[int, np.ndarray]) -> VectorTruth:
        if vector_id in self.truth.vectors:
            return self.truth.vectors[vector_id]
        best, best_cos = None, self.BIND_COS
        for vid in self.truth.vectors:
            dirs = self.geometry.direction(vid)
            cos = []
            for layer, v in stored.items():
                vp = v.astype(np.float64)
                vp[self.geometry.nuisance(layer)] = 0.0
                n = float(np.linalg.norm(vp))
                cos.append(0.0 if n == 0.0 else float(vp @ dirs[layer]) / n)
            c = float(np.median(cos))
            if c >= best_cos:
                best, best_cos = vid, c
        return self.truth.vectors[best] if best is not None else self.truth.random_vector(vector_id)

    def provenance(self) -> dict[str, Any]:
        """The simulator and the truth it samples from (the truth's parameters are its code)."""
        return {"backend": "m5.backend.fake.FakeBackend", "seed": self.seed,
                "n_layers": self.info.n_layers, "d_model": self.info.d_model,
                "truth_vectors": sorted(self.truth.vectors),
                "collapse_layer": self.truth.collapse_layer, "motor_from": self.truth.motor_from}

    def true_vector(self, vector_id: str, layers: Sequence[int] | None = None) -> dict[int, np.ndarray]:
        """A truth vector's direction at its natural size per layer (what extraction finds)."""
        if vector_id not in self.truth.vectors:
            raise KeyError(f"{vector_id!r} is not a vector of the truth")
        g = self.geometry
        dirs = g.direction(vector_id)
        return {int(L): (g.NATURAL * g.s[L] * dirs[L]).astype(np.float32)
                for L in (range(self.info.n_layers) if layers is None else layers)}

    def _perp_norm(self, vector_id: str, layer: int) -> float:
        vec = self._vectors.get(vector_id)
        if vec is None:
            raise KeyError(f"vector {vector_id!r} is not registered")
        if layer not in vec:
            raise KeyError(f"vector {vector_id!r} has no value at layer {layer}")
        v = vec[layer].astype(np.float64)
        v[self.geometry.nuisance(layer)] = 0.0
        return float(np.linalg.norm(v))

    def r_perp_of(self, vector_id: str, layer: int, alpha: float) -> float:
        """The simulator's own dose: ||alpha v_perp|| / s_L with its nuisance coordinate zeroed."""
        return alpha * self._perp_norm(vector_id, layer) / float(self.geometry.s[layer])

    def alpha_for(self, vector_id: str, layer: int, r_perp: float) -> float:
        """The alpha that puts `r_perp` on a registered vector at a layer."""
        n = self._perp_norm(vector_id, layer)
        if n == 0.0:
            raise ValueError(f"vector {vector_id!r} is zero off the nuisance coordinate at layer {layer}")
        return r_perp * float(self.geometry.s[layer]) / n

    # ================================================================ resolving a request
    def _resolve(self, iv: Intervention | None, n_prompt: int) -> _Steer | None:
        """An intervention against the registered vector and the truth, validated as the real
        backend validates it whatever its alphas. Non-null results are cached per (intervention,
        prompt length) -- both immutable -- and the cache is dropped whenever a vector changes, so
        nothing resolved under one registration is reused under another."""
        if iv is None:
            return None
        if not isinstance(iv, Intervention):
            raise TypeError(f"expected an Intervention or None, got {type(iv).__name__}")
        key = (iv, n_prompt)
        hit = self._steer_cache.get(key)
        if hit is not None:
            return hit
        vec = self._vectors.get(iv.vector_id)
        if vec is None:
            raise KeyError(f"vector {iv.vector_id!r} is not registered")
        for t in iv.terms:
            if t.layer >= self.info.n_layers:
                raise ValueError(f"layer {t.layer} is beyond the model ({self.info.n_layers} layers)")
            if t.layer not in vec:
                raise ValueError(f"vector {iv.vector_id!r} has no value at layer {t.layer}")
        pos = iv.positions
        if pos.policy == "from" and pos.start >= n_prompt:
            raise ValueError(f"steering starts at token {pos.start} but the prompt has {n_prompt} "
                             "tokens; an out-of-range start would leave the prompt unsteered")
        pos.prompt_mask(n_prompt)                      # a span beyond the prompt raises here
        if iv.is_null():
            return None
        if len(self._steer_cache) > 20_000:
            self._steer_cache.clear()
        out = self._steer_cache[key] = self._resolve_uncached(iv, n_prompt)
        return out

    def _resolve_uncached(self, iv: Intervention, n_prompt: int) -> _Steer:
        vec = self._vectors[iv.vector_id]
        doses: dict[int, float] = {}
        supp = 0.0
        for t in iv.terms:
            n_perp = self._perp_norm(iv.vector_id, t.layer)
            n_full = float(np.linalg.norm(vec[t.layer].astype(np.float64)))
            s_l = float(self.geometry.s[t.layer])
            if iv.op in ("add", "sub"):
                r = t.alpha * n_perp / s_l
            elif iv.op == "project":
                r, supp = 0.0, max(supp, min(t.alpha, 1.0))
            else:  # clamp: alpha is not used, the component is set to clamp_value
                c = float(iv.clamp_value)  # type: ignore[arg-type]
                r = abs(c) * (n_perp / n_full if n_full > 0 else 0.0) / s_l
                if c == 0.0:
                    supp = 1.0
            if r > 0.0:
                doses[t.layer] = r
        if iv.op == "clamp":
            c = float(iv.clamp_value)  # type: ignore[arg-type]
            sign = 1 if c > 0 else -1 if c < 0 else 0
        else:
            sign = {"add": 1, "sub": -1, "project": 0}[iv.op]
        mask = np.asarray(iv.positions.prompt_mask(n_prompt), dtype=bool)
        first = int(np.argmax(mask)) if mask.any() else n_prompt
        vt = self._bound[iv.vector_id]
        obs = (self._unsteered,
               self.truth.observables(vt, doses, sign=sign, past_only=True),
               self.truth.observables(vt, doses, sign=sign))
        return _Steer(iv, vt, sign, doses, supp, mask, first, iv.positions.steers_generated, obs)

    def _mentions(self, text: str, ids: Sequence[int]) -> dict[str, np.ndarray]:
        got = self._mention_cache.get(text)
        if got is None:
            words = [self._tok.string_of(i).strip().strip(".,;:!?\"'()[]{}").lower() for i in ids]
            got = {}
            for vid, vt in self.truth.vectors.items():
                if vt.triggers:
                    trig = frozenset(vt.triggers)
                    got[vid] = np.fromiter((w in trig for w in words), dtype=bool, count=len(words))
            if len(self._mention_cache) > 20_000:
                self._mention_cache.clear()
            self._mention_cache[text] = got
        return got

    def _ctx(self, text: str, iv: Intervention | None, n_prompt: int | None = None, *,
             generated: bool = False, what: str = "pass") -> _Ctx:
        """A validated pass over `text`: it begins with exactly one BOS (the template writes one;
        raw text is prefixed with decode([info.bos_token_id])), and an intervention must steer
        some token of it -- `generated` says whether the pass has generated (or teacher-forced)
        tokens. A pass that cannot steer must not read as a steered measurement (M2 bug 26)."""
        if not isinstance(text, str):
            raise TypeError(f"{what}: expected a rendered prompt string, got {type(text).__name__}")
        ids, offs = self._tok.encode(text)
        if not ids or ids[0] != BOS_ID:
            raise ValueError(f"{what}: does not begin with the BOS the chat template writes; render it, "
                             "or prefix raw text with decode([info.bos_token_id])")
        if len(ids) > 1 and ids[1] == BOS_ID:
            raise ValueError(f"{what}: begins with two BOS tokens; every position would be off by one")
        n = len(ids) if n_prompt is None else n_prompt
        steer = self._resolve(iv, n)
        if iv is not None and not (any(iv.positions.prompt_mask(n))
                                   or (generated and iv.positions.steers_generated)):
            raise ValueError(f"{what}: positions {iv.positions.to_json()} steer no token of this pass")
        return _Ctx(text, np.asarray(ids, dtype=np.int64), offs, n, steer, self._mentions(text, ids))

    def _obs(self, ctx: _Ctx, state: int) -> dict[str, float]:
        return self._unsteered if ctx.steer is None else ctx.steer.obs[state]

    def _supp(self, ctx: _Ctx, state: int) -> float:
        if ctx.steer is None or state == 0:
            return 0.0
        return ctx.steer.supp * (1.0 if state == 2 else self.truth.past_only_factor)

    # ================================================================ channel distributions
    def _dist(self, weights: Mapping[str, float], fail: float) -> _Dist:
        """Healthy weights (summing to 1 - fail) plus `fail` spread over the junk tokens."""
        items = [(self._tok.id_of(t), w) for t, w in weights.items() if w > 0.0]
        ids = np.fromiter((i for i, _ in items), dtype=np.int64, count=len(items))
        w = np.fromiter((x for _, x in items), dtype=np.float64, count=len(items))
        if fail > 0.0:
            ids = np.concatenate([ids, self._junk_arr])
            w = np.concatenate([w, np.full(len(self._junk_arr), fail / len(self._junk_arr))])
        return _Dist.from_arrays(ids, w)

    def _slot_weights(self, prompt: _Prompt, obs: Mapping[str, float], vt: VectorTruth | None) -> dict[str, float]:
        """d3/d2 (forced) and d5 (open slot): the concept, else near misses that rise with the
        effect (FINDINGS.md §3.3), else the slot's default answers, "apple" first."""
        forced = prompt.channel == "forced"
        name = obs["forced_raw"] if forced else obs["slot_raw"]
        fill = 0.0 if prompt.filler else (_FORCED_FILL if forced else _OPEN_FILL)
        # A reply that has written a healthy filler goes on healthily: integrity failure is decided
        # once, at the slot, so naming totals forced_raw x (1 - failure) by either path.
        healthy = 1.0 if prompt.filler else 1.0 - obs["integrity"]
        direct = healthy * (1.0 - fill)
        w: dict[str, float] = {}
        if vt is None or not vt.concept:
            name = 0.0
        else:
            for s, share in _forms(vt.concept):
                w[s] = w.get(s, 0.0) + direct * name * share
        rest = direct * (1.0 - name)
        if vt is not None and vt.concept and vt.associations:
            near = 0.35 * obs["effect"]
            for a in vt.associations:
                w[" " + a] = w.get(" " + a, 0.0) + rest * near / len(vt.associations)
            rest *= 1.0 - near
        for word, share in (_FORCED_BASE if forced else _OPEN_BASE):
            w[" " + word] = w.get(" " + word, 0.0) + rest * share
        for f, share in _FILLERS:
            if fill:
                w[f] = w.get(f, 0.0) + healthy * fill * share
        return w

    def _gap(self, prompt: _Prompt) -> float:
        """A yes/no question's unsteered logit(Yes) - logit(No): the detection question sits at
        Gemma's default "no" (the d8 floor); every other question gets its own gap from a hash of
        its text (trial numbers ignored), plus a small per-prompt jitter."""
        jitter = 0.6 * (_unit("gap-jitter", prompt.question) - 0.5)
        if prompt.kind in ("target", "target_unusual", "named"):
            return _logit(self.truth.claim_floor) + jitter
        norm = re.sub(r"\d+", "#", prompt.question).strip().lower()
        return -5.0 + 10.0 * _unit("gap", norm) + jitter

    def _yes_no_kind(self, prompt: _Prompt, vt: VectorTruth | None) -> str:
        if prompt.kind != "named":
            return prompt.kind
        return "match" if vt is not None and _same_word(prompt.named, vt.concept) else "mismatch"

    def _yes_no_weights(self, prompt: _Prompt, obs: Mapping[str, float], vt: VectorTruth | None) -> dict[str, float]:
        g = self._gap(prompt) + self.truth.yes_no_shift(obs, self._yes_no_kind(prompt, vt))
        y = _sigmoid(g)
        h = (1.0 - obs["integrity"]) * (1.0 - _YES_NO_OTHER)
        if prompt.channel == "yesno":     # after "Answer:" the word comes with its space
            forms = ((" {}", 0.92), ("{}", 0.06), (" {l}", 0.02))
        else:                             # at the start of a reply it does not
            forms = (("{}", 0.94), (" {}", 0.04), ("{l}", 0.02))
        w: dict[str, float] = {}
        for word, p in (("Yes", y), ("No", 1.0 - y)):
            for fmt, share in forms:
                s = fmt.format(word, l=word.lower())
                w[s] = w.get(s, 0.0) + h * p * share
        other = (1.0 - obs["integrity"]) * _YES_NO_OTHER
        w[" I"] = w.get(" I", 0.0) + other / 2
        w[" The"] = w.get(" The", 0.0) + other / 2
        return w

    def _steered_segment(self, prompt: _Prompt, ctx: _Ctx) -> int | None:
        """The segment the intervention's positions cover most (None: none, or a tie)."""
        if ctx.steer is None or not prompt.segments:
            return None
        spans = [o for o, m in zip(ctx.offs, ctx.steer.mask) if m]
        best, best_ov, tie = None, 0, False
        for label, s, e in prompt.segments:
            ov = sum(max(0, min(e, te) - max(s, ts)) for ts, te in spans)
            if ov > best_ov:
                best, best_ov, tie = label, ov, False
            elif ov == best_ov and ov > 0:
                tie = True
        return None if tie else best

    def _segment_probs(self, prompt: _Prompt, ctx: _Ctx, obs: Mapping[str, float]) -> dict[str, float]:
        """d9 over the options "1".."K" and "none" (healthy share only). Knowing where comes from
        the localise curve; the generic perturbation also pulls answers away from "none" toward
        guesses, which is why chance and the "none" trials are its controls."""
        labels = [lab for lab, _, _ in prompt.segments] or [1, 2, 3, 4]
        k = len(labels)
        correct = self._steered_segment(prompt, ctx)
        knows = obs["localise_raw"] if correct is not None else 0.0
        p_none = 0.8 * (1.0 - 0.5 * obs["bias"])
        probs = {str(lab): (1.0 - knows) * (1.0 - p_none) / k for lab in labels}
        none = (1.0 - knows) * p_none
        if knows > 0.0:
            probs[str(correct)] += knows * 0.9
            if k > 1:
                for lab in labels:
                    if lab != correct:
                        probs[str(lab)] += knows * 0.05 / (k - 1)
                none += knows * 0.05
            else:
                none += knows * 0.1
        probs[M.SEGMENT_NONE] = none
        return probs

    def _decoy_probs(self, prompt: _Prompt, obs: Mapping[str, float], vt: VectorTruth | None) -> dict[str, float]:
        """d6 over the option numbers: the concept's option with past-only naming, else a guess
        that favours "apple" when it is on the list (Lederman 2026)."""
        opts = prompt.options or tuple((k, "") for k in range(1, 10))
        base = np.array([3.0 if _same_word(w, "apple") else 1.0 for _, w in opts])
        base /= base.sum()
        name = obs["forced_raw"] if vt is not None and vt.concept else 0.0
        probs = {str(n): (1.0 - name) * b for (n, _), b in zip(opts, base)}
        hit = [n for n, w in opts if vt is not None and _same_word(w, vt.concept)]
        if hit:
            probs[str(hit[0])] += name
        else:
            for key in probs:
                probs[key] += name / len(probs)
        return probs

    @staticmethod
    def _option_weights(probs: Mapping[str, float], healthy: float, lead: str) -> dict[str, float]:
        w: dict[str, float] = {}
        for opt, p in probs.items():
            w[lead + opt] = w.get(lead + opt, 0.0) + healthy * p * 0.95
            w[opt] = w.get(opt, 0.0) + healthy * p * 0.05
        return w

    def _ab_gap(self, prompt: _Prompt) -> float:
        """e10: an item's unsteered log-odds of the behaviour's answer, from a hash of its text,
        plus a small preference for "A" (a letter bias the items' counterbalancing cancels)."""
        return -1.5 + 3.0 * _unit("ab", prompt.question) + (0.3 if prompt.matching == "A" else -0.3)

    def _ab_shift(self, prompt: _Prompt, obs: Mapping[str, float], ctx: _Ctx) -> float:
        st = ctx.steer
        return 0.0 if st is None else self.truth.ab_shift(st.vt, obs, prompt.behaviour, st.sign)  # type: ignore[arg-type]

    def _ab_weights(self, prompt: _Prompt, obs: Mapping[str, float], ctx: _Ctx) -> dict[str, float]:
        pm = _sigmoid(self._ab_gap(prompt) + self._ab_shift(prompt, obs, ctx))
        h = (1.0 - obs["integrity"]) * (1.0 - _YES_NO_OTHER)
        other = "B" if prompt.matching == "A" else "A"
        return {prompt.matching: h * pm, other: h * (1.0 - pm),  # type: ignore[dict-item]
                "I": (1.0 - obs["integrity"]) * _YES_NO_OTHER}

    def _capability_outcomes(self, prompt: _Prompt, obs: Mapping[str, float],
                             vt: VectorTruth | None) -> list[tuple[str, float]]:
        """s5 answers among healthy replies: correct at the item's pass rate, wrong otherwise, and
        the concept itself when it contaminates the answer (RESULTS-SILK.md, L39 at 0.1891)."""
        contam = 0.1 * obs["effect"] if vt is not None and vt.concept else 0.0
        wrong = self._wrong_answer(prompt)
        if prompt.answer is None:
            out = [(wrong, 1.0 - contam)]
        else:
            out = [(prompt.answer, (1.0 - contam) * prompt.pass_rate),
                   (wrong, (1.0 - contam) * (1.0 - prompt.pass_rate))]
        if contam > 0.0:
            out.append((str(vt.concept), contam))  # type: ignore[union-attr]
        return out

    @staticmethod
    def _wrong_answer(prompt: _Prompt) -> str:
        k = 1 + int(3 * _unit("wrong", prompt.question))
        if prompt.answer is not None and re.fullmatch(r"-?\d+", prompt.answer):
            return str(int(prompt.answer) + k)
        if prompt.answer is not None:
            return "unsure"
        return str(10 + int(90 * _unit("guess", prompt.question)))

    def _self_report_outcomes(self, prompt: _Prompt, obs: Mapping[str, float]) -> tuple[float, float, float, float]:
        """d8: (claim, denial, refusal, degenerate)."""
        fail = obs["integrity"]
        healthy = 1.0 - fail
        claim = obs["claim" if prompt.kind == "injected_thought" else "claim_unusual"]
        refuse = healthy * self.truth.refusal_rate
        return claim, max(healthy - claim - refuse, 0.0), refuse, fail

    def _explicit(self, prompt: _Prompt, ctx: _Ctx, state: int) -> _Dist | None:
        """The next-token distribution of a channel with explicit answers (None: open text)."""
        obs = self._obs(ctx, state)
        vt = ctx.steer.vt if ctx.steer is not None else None
        fail = obs["integrity"]
        healthy = 1.0 - fail
        ch = prompt.channel
        if ch in ("forced", "open_slot"):
            return self._dist(self._slot_weights(prompt, obs, vt), 0.0 if prompt.filler else fail)
        if ch in ("yesno", "yesno_free"):
            return self._dist(self._yes_no_weights(prompt, obs, vt), fail)
        if ch == "segment":
            return self._dist(self._option_weights(self._segment_probs(prompt, ctx, obs), healthy, " "), fail)
        if ch == "decoy":
            return self._dist(self._option_weights(self._decoy_probs(prompt, obs, vt), healthy, " "), fail)
        if ch == "ab":
            return self._dist(self._ab_weights(prompt, obs, ctx), fail)
        if ch == "capability_slot":
            w: dict[str, float] = {}
            for answer, p in self._capability_outcomes(prompt, obs, vt):
                first = self._tok.string_of(self._tok.encode(" " + answer)[0][0])
                w[first] = w.get(first, 0.0) + healthy * p
            return self._dist(w, fail)
        if ch == "self_report":
            claim, deny, refuse, _ = self._self_report_outcomes(prompt, obs)
            return self._dist({_CLAIM_TOKEN: claim, _DENIAL_TOKEN: deny, _REFUSAL_TOKEN: refuse}, fail)
        if ch == "capability":
            return self._dist({_ANSWER_TOKEN: healthy * (1.0 - _PREAMBLE_SHARE),
                               _PREAMBLE_TOKEN: healthy * _PREAMBLE_SHARE}, fail)
        if ch in ("segment_free", "decoy_free"):
            return self._dist({"The": healthy}, fail)
        return None

    def _lm_rows(self, ctx: _Ctx, positions: np.ndarray) -> np.ndarray:
        """Open-text distributions at these positions (see `_OpenLM`)."""
        n = len(positions)
        states = np.array([ctx.state(int(i)) for i in positions], dtype=np.int64)
        masses: dict[str, np.ndarray] = {}
        for vid in self._lm.group_w:
            pres = ctx.present(vid)
            masses[vid] = (np.where(pres[positions], _OpenLM.M_NATURAL, _OpenLM.M_ABSENT)
                           if pres is not None else np.full(n, _OpenLM.M_ABSENT))
        junk, beta, failure, key = np.zeros(n), np.zeros(n), np.zeros(n), None
        st = ctx.steer
        if st is not None:
            key = st.iv.vector_id
            m = masses.get(st.vt.vector_id)
            for s in (1, 2):
                sel = states == s
                if not np.any(sel):
                    continue
                obs = st.obs[s]
                junk[sel] = _OpenLM.JUNK_MAX * obs["integrity"]
                failure[sel] = obs["integrity"]
                beta[sel] = obs["tilt"]
                if m is not None:
                    if st.sign > 0:
                        m[sel] = m[sel] + (_OpenLM.M_MAX - m[sel]) * obs["effect"]
                    elif st.sign < 0:
                        m[sel] = m[sel] * (1.0 - 0.95 * obs["effect_size"])
                    supp = st.supp * (1.0 if s == 2 else self.truth.past_only_factor)
                    if supp > 0.0:
                        m[sel] = m[sel] * (1.0 - 0.95 * supp)
        steered = st.vt.vector_id if st is not None and st.vt.vector_id in masses else None
        return self._lm.rows(ctx.ids[positions], masses, junk, beta, key, steered, failure)

    def _channel_at(self, ctx: _Ctx, i: int) -> _Prompt | None:
        """The explicit channel whose slot follows token i, or None for open text."""
        end = ctx.offs[i][1]
        if not ctx.text.endswith(_TAILS, 0, end):
            return None
        prompt = self._parse(ctx.text[:end])
        return None if prompt.channel in _LM_CHANNELS else prompt

    def _dist_at(self, ctx: _Ctx, i: int) -> _Dist:
        """The next-token distribution after token i, as one object."""
        prompt = self._channel_at(ctx, i)
        if prompt is not None:
            d = self._explicit(prompt, ctx, ctx.state(i))
            if d is not None:
                return d
        return self._lm.as_dist(self._lm_rows(ctx, np.array([i]))[0])

    # ================================================================ generation
    def generate(self, requests: Sequence[GenRequest]) -> list[GenResult]:
        return [self._generate_one(r) for r in requests]

    def _generate_one(self, req: GenRequest) -> GenResult:
        ctx = self._ctx(req.prompt, req.intervention, generated=True, what="generate")
        n = len(ctx.ids)
        prompt = self._parse(req.prompt)
        u = _uniforms(derive_seed(self.seed, "generate", req.seed, req.prompt), 16)
        first, later = ctx.state(n - 1), ctx.state(n)
        ch = prompt.channel
        if ch in ("forced", "open_slot", "yesno", "yesno_free", "segment", "decoy", "ab"):
            text = self._gen_slot(prompt, ctx, first, later, u, req)
        elif ch in ("capability", "capability_slot"):
            text = self._gen_capability(prompt, ctx, first, later, u, req)
        elif ch == "self_report":
            text = self._gen_self_report(prompt, ctx, first, later, u, req)
        elif ch in ("segment_free", "decoy_free"):
            text = self._gen_answer_is(prompt, ctx, first, later, u, req)
        else:
            text = self._gen_open(prompt, ctx, later, u, req)
        return self._finish(text, req.max_new_tokens)

    def _sample(self, d: _Dist, u: float, req: GenRequest) -> int:
        return int(d.ids[_pick(d.p, u, req.temperature, req.top_k, req.top_p)])

    def _finish(self, text: str, max_new_tokens: int) -> GenResult:
        ids = list(self._tok.encode(text)[0])
        if len(ids) >= max_new_tokens:
            ids = ids[:max_new_tokens]
            return GenResult(self._tok.decode(ids, skip_special_tokens=True), tuple(ids), "length")
        return GenResult(text, tuple(ids) + (EOT_ID,), "eos")

    def _degenerate(self, u: Sequence[float], n_tokens: int, word: str | None) -> str:
        """Collapse (s2): a junk token, then a word run, a character run or a phrase loop that
        fills the budget -- the loops teacher forcing never lets start."""
        first = _JUNK[int(u[7] * len(_JUNK))]
        kind = int(u[8] * 3)
        if kind == 0:
            unit = " " + (word or _JUNK[int(u[10] * len(_JUNK))].strip())
        elif kind == 1:
            unit = "!"
        else:
            unit = " the of"
        return first + unit * (n_tokens + 2)

    def _gen_slot(self, prompt: _Prompt, ctx: _Ctx, first: int, later: int,
                  u: Sequence[float], req: GenRequest) -> str:
        """Slots: the first token is sampled from exactly the distribution `next_token_logprobs`
        reports, so generated naming (d2) and logit naming (d3) agree by construction."""
        tid = self._sample(self._explicit(prompt, ctx, first), u[0], req)  # type: ignore[arg-type]
        text = self._tok.string_of(tid)
        if tid in self._junk_ids:
            return text + self._degenerate(u, req.max_new_tokens, None)
        if prompt.channel in ("forced", "open_slot") and not prompt.filler and text in _FILLER_STRINGS:
            tid = self._sample(self._explicit(replace(prompt, filler=True), ctx, later), u[1], req)  # type: ignore[arg-type]
            text += self._tok.string_of(tid)
            if tid in self._junk_ids:
                return text + self._degenerate(u, req.max_new_tokens, None)
        if prompt.channel == "forced":
            return text + "." + _ELABORATION[int(u[2] * len(_ELABORATION))]
        if prompt.channel == "ab":
            return text + ")"
        return text + "."

    def _gen_answer_is(self, prompt: _Prompt, ctx: _Ctx, first: int, later: int,
                       u: Sequence[float], req: GenRequest) -> str:
        tid = self._sample(self._explicit(prompt, ctx, first), u[0], req)  # type: ignore[arg-type]
        if tid in self._junk_ids:
            return self._tok.string_of(tid) + self._degenerate(u, req.max_new_tokens, None)
        slot = replace(prompt, channel="segment" if prompt.channel == "segment_free" else "decoy")
        tid = self._sample(self._explicit(slot, ctx, later), u[1], req)  # type: ignore[arg-type]
        text = "The answer is" + self._tok.string_of(tid)
        if tid in self._junk_ids:
            return text + self._degenerate(u, req.max_new_tokens, None)
        return text + "."

    def _gen_capability(self, prompt: _Prompt, ctx: _Ctx, first: int, later: int,
                        u: Sequence[float], req: GenRequest) -> str:
        vt = ctx.steer.vt if ctx.steer is not None else None
        if prompt.channel == "capability_slot":
            obs = self._obs(ctx, first)
            outcomes = self._capability_outcomes(prompt, obs, vt)
            fail = obs["integrity"]
            k = _pick([p * (1.0 - fail) for _, p in outcomes] + [fail], u[0], req.temperature,
                      req.top_k, req.top_p)
            if k == len(outcomes):
                return self._degenerate(u, req.max_new_tokens, None)
            return " " + outcomes[k][0]
        tid = self._sample(self._explicit(prompt, ctx, first), u[0], req)  # type: ignore[arg-type]
        if tid in self._junk_ids:
            return self._tok.string_of(tid) + self._degenerate(u, req.max_new_tokens, None)
        head = "Let me work it out.\nAnswer: " if self._tok.string_of(tid) == _PREAMBLE_TOKEN else "Answer: "
        outcomes = self._capability_outcomes(prompt, self._obs(ctx, later), vt)
        k = _pick([p for _, p in outcomes], u[1], req.temperature, req.top_k, req.top_p)
        return head + outcomes[k][0]

    def _wrong_name(self, vt: VectorTruth | None, u: float, v: float) -> str:
        """A confabulated name: "apple" half the time (Lederman 2026), else a near miss, else
        another default answer."""
        if u < 0.5:
            return "apple"
        if vt is not None and vt.associations and u < 0.8:
            return vt.associations[int(v * len(vt.associations))]
        others = [w for w, _ in _FORCED_BASE[1:]]
        return others[int(v * len(others))]

    def _gen_self_report(self, prompt: _Prompt, ctx: _Ctx, first: int, later: int,
                         u: Sequence[float], req: GenRequest) -> str:
        """d8. The claim is decided where the first token is (so "Yes" has the probability
        `next_token_logprobs` gives it) and always comes before any name (affirm-before-naming);
        a claim names the concept with the forced-naming probability, else confabulates."""
        claim, deny, refuse, fail = self._self_report_outcomes(prompt, self._obs(ctx, first))
        cat = _pick((claim, deny, refuse, fail), u[0], req.temperature, req.top_k, req.top_p)
        vt = ctx.steer.vt if ctx.steer is not None else None
        later_obs = self._obs(ctx, later)
        framing = prompt.kind
        if cat == 3:
            return self._degenerate(u, req.max_new_tokens, None)
        if cat == 2:
            return _REFUSAL
        if cat == 1:
            text = _DENIAL[framing][int(u[2] * len(_DENIAL[framing]))]
            if vt is not None and vt.concept and _bern(0.25 * later_obs["effect"], u[15], req.temperature):
                text += _LEAK.format(c=vt.concept)
            return text
        if vt is not None and vt.concept:
            named = _bern(later_obs["forced_raw"], u[3], req.temperature)
            word = vt.concept if named else self._wrong_name(vt, u[4], u[10])
            templates = _CLAIM_NAMED[framing]
            return templates[int(u[5] * len(templates))].format(w=word)
        if u[3] < 0.4:
            templates = _CLAIM_NAMED[framing]
            return templates[int(u[5] * len(templates))].format(w=self._wrong_name(None, u[4], u[10]))
        templates = _CLAIM_UNNAMED[framing]
        return templates[int(u[5] * len(templates))]

    def _mention(self, ctx: _Ctx, obs: Mapping[str, float], state: int) -> tuple[VectorTruth | None, float]:
        """Which concept an open reply may mention, and how likely it is to: the steered concept
        (raised by the effect, suppressed by sub and project), else one the prompt is about."""
        last = ctx.n_prompt - 1
        st = ctx.steer
        if st is not None and st.vt.concept:
            pres = ctx.present(st.vt.vector_id)
            base = _P_MENTION_NATURAL if pres is not None and last >= 0 and pres[last] else _P_MENTION_ABSENT
            if st.sign > 0:
                p = 1.0 - (1.0 - base) * (1.0 - obs["effect"])
            elif st.sign < 0:
                p = base * (1.0 - 0.95 * obs["effect_size"])
            else:
                p = base
            return st.vt, p * (1.0 - 0.95 * self._supp(ctx, state))
        for vid, vt in self.truth.vectors.items():
            pres = ctx.present(vid)
            if vt.concept and pres is not None and last >= 0 and pres[last]:
                return vt, _P_MENTION_NATURAL
        return None, 0.0

    def _p_eval(self, prompt: _Prompt, ctx: _Ctx, obs: Mapping[str, float], state: int) -> float:
        """a1: the scenario class's base rate, moved in the operation's direction by an
        evaluation-awareness vector, and a little by noticing anything at all (every vector)."""
        p = self.truth.scenario_rate(prompt.scenario)  # type: ignore[arg-type]
        st = ctx.steer
        if st is not None and st.vt.vector_class == "evaluation_awareness":
            g = self.truth.ea_gain
            if st.sign > 0:
                p += (1.0 - p) * g * obs["effect_size"]
            elif st.sign < 0:
                p *= 1.0 - g * obs["effect_size"]
            p *= 1.0 - g * self._supp(ctx, state)
        return p + (1.0 - p) * self.truth.eval_generic_gain * obs["notice"]

    def _p_flag(self, ctx: _Ctx, obs: Mapping[str, float]) -> float:
        """d10's evaluation-awareness mention in ordinary open text."""
        p = 0.04 * obs["notice"]
        st = ctx.steer
        if st is not None and st.vt.vector_class == "evaluation_awareness" and st.sign > 0:
            p += 0.3 * obs["effect_size"]
        return min(p, 1.0)

    def _gen_open(self, prompt: _Prompt, ctx: _Ctx, state: int, u: Sequence[float], req: GenRequest) -> str:
        """Open text (e7, e8, s1, s2, d10) and scenarios (a1), assembled from the banks."""
        obs = self._obs(ctx, state)
        T = req.temperature
        vt, p_mention = self._mention(ctx, obs, state)
        word = vt.concept if vt is not None else None
        if _bern(obs["integrity"], u[6], T):
            return self._degenerate(u, req.max_new_tokens,
                                    word if word and _bern(p_mention, u[9], T) else None)
        if prompt.channel == "scenario":
            sents = [_SCENARIO_OPEN]
            if _bern(self._p_eval(prompt, ctx, obs, state), u[12], T):
                sents.append(_SCENARIO_EVAL[int(u[2] * len(_SCENARIO_EVAL))])
            sents.append(_SCENARIO_BODY[int(u[13] * len(_SCENARIO_BODY))])
        else:
            n_sent = 2 + int(u[13] < 0.4)
            start = int(u[14] * len(_OPEN_BANK))
            sents = [_OPEN_BANK[(start + 5 * k) % len(_OPEN_BANK)] for k in range(n_sent)]
            if _bern(self._p_flag(ctx, obs), u[12], T):
                sents.append(_EVAL_MENTION)
        if word and _bern(p_mention, u[3], T):
            if vt is not None and vt.associations and u[4] >= 0.65:
                s = _ALLUSION[int(u[5] * len(_ALLUSION))].format(a=vt.associations[int(u[10] * len(vt.associations))])
            else:
                s = _LITERAL[int(u[5] * len(_LITERAL))].format(c=word)
            sents.insert(1, s)
        if _bern(0.5 * obs["notice"], u[11], T):
            sents.insert(len(sents) - 1, _RESTART)
        return " ".join(sents)

    # ================================================================ forward passes
    @staticmethod
    def _check_rows(prompts: Sequence[str], interventions: Sequence[Intervention | None], what: str) -> None:
        if len(prompts) != len(interventions):
            raise ValueError(f"{what}: {len(prompts)} prompts but {len(interventions)} interventions")

    @staticmethod
    def _check_ids(token_ids: Sequence[int]) -> np.ndarray:
        out = []
        for t in token_ids:
            if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or not 0 <= int(t) < VOCAB_SIZE:
                raise ValueError(f"token id {t!r} is not in [0, {VOCAB_SIZE})")
            out.append(int(t))
        return np.asarray(out, dtype=np.int64)

    def _check_layers(self, layers: Sequence[int]) -> np.ndarray:
        out = []
        for L in layers:
            if isinstance(L, bool) or not isinstance(L, (int, np.integer)) or not 0 <= int(L) < self.info.n_layers:
                raise ValueError(f"layer {L!r} is not in [0, {self.info.n_layers})")
            out.append(int(L))
        return np.asarray(out, dtype=np.int64)

    @staticmethod
    def _position(n: int, p: int, what: str) -> int:
        q = int(p) + n if int(p) < 0 else int(p)
        if not 0 <= q < n:
            raise IndexError(f"{what}: position {p} is outside a prompt of {n} tokens")
        return q

    def next_token_logprobs(self, prompts: Sequence[str], interventions: Sequence[Intervention | None],
                            token_ids: Sequence[int]) -> np.ndarray:
        self._check_rows(prompts, interventions, "next_token_logprobs")
        tids = self._check_ids(token_ids)
        out = np.empty((len(prompts), len(tids)), dtype=np.float64)
        for k, (text, iv) in enumerate(zip(prompts, interventions)):
            ctx = self._ctx(text, iv, what=f"next_token_logprobs row {k}")
            out[k] = self._dist_at(ctx, len(ctx.ids) - 1).logprob(tids)
        return out

    def _teacher(self, req: ScoreRequest, query: np.ndarray | None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Teacher forcing: the continuation is tokenised in context and its tokens are steered as
        generated tokens (what generation would compute for them). Returns the continuation ids,
        their log-probs, the predictive entropies, and the candidates' log-probs."""
        p_ids = self._tok.encode(req.prompt)[0]
        j_ids = self._tok.encode(req.prompt + req.continuation)[0]
        if j_ids[: len(p_ids)] != p_ids:
            raise ValueError("the continuation merges with the prompt's last token at the boundary, so "
                             "the prompt's tokens are not a prefix of the joint tokenisation")
        self._ctx(req.prompt, None, what="teacher forcing")         # the prompt alone carries the BOS
        ctx = self._ctx(req.prompt + req.continuation, req.intervention, n_prompt=len(p_ids),
                        generated=len(j_ids) > len(p_ids), what="teacher forcing")
        cont = np.asarray(j_ids[len(p_ids):], dtype=np.int64)
        positions = np.arange(len(p_ids) - 1, len(j_ids) - 1)
        m = len(positions)
        lp, ent = np.empty(m), np.empty(m)
        cand = np.empty((m, len(query))) if query is not None else None
        lm_k = []
        for k, i in enumerate(positions):
            prompt = self._channel_at(ctx, int(i))
            d = self._explicit(prompt, ctx, ctx.state(int(i))) if prompt is not None else None
            if d is None:
                lm_k.append(k)
                continue
            lp[k] = math.log(float(d.prob(cont[k:k + 1])[0]))
            ent[k] = d.entropy()
            if cand is not None:
                cand[k] = d.logprob(query)  # type: ignore[arg-type]
        if lm_k:
            idx = np.asarray(lm_k)
            P = self._lm_rows(ctx, positions[idx])
            cols, hit = self._lm.lookup(cont[idx])
            vals = np.where(hit, P[np.arange(len(idx)), cols], self._lm.per_other)
            lp[idx] = np.log(vals)
            ent[idx] = self._lm.entropy_rows(P)
            if cand is not None:
                cand[idx] = self._lm.logprob_rows(P, query)  # type: ignore[arg-type]
        return cont, lp, ent, cand

    def score(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        out = []
        for req in requests:
            cont, lp, ent, _ = self._teacher(req, None)
            out.append(ScoreResult(tuple(int(t) for t in cont), tuple(float(x) for x in lp),
                                   tuple(float(x) for x in ent)))
        return out

    def candidate_logprobs(self, requests: Sequence[ScoreRequest], token_ids: Sequence[int]) -> list[np.ndarray]:
        """Per request [len(continuation tokens), len(token_ids)]: log p of each candidate under
        the distribution predicting continuation token t -- the distributions `score` reads, so
        e9's concept tokens gain probability with the effect curve at steered positions."""
        tids = self._check_ids(token_ids)
        return [self._teacher(req, tids)[3] for req in requests]  # type: ignore[misc]

    def kl_from_unsteered(self, texts: Sequence[str], interventions: Sequence[Intervention], *,
                          start_tokens: Sequence[int], exclude_token_ids: Sequence[int] = ()) -> list[KLResult]:
        """Mean KL(unsteered || steered) over the next-token distributions at positions
        start..n-1 (each predicting the token after it). The text is all prompt, so a policy
        that steers only generated tokens steers nothing here and raises, as does a missing
        intervention (the unsteered arm's KL is zero by definition)."""
        if len(texts) != len(interventions) or len(texts) != len(start_tokens):
            raise ValueError("kl_from_unsteered: texts, interventions and start_tokens differ in length")
        excl = np.unique(self._check_ids(exclude_token_ids))
        none = np.empty(0, dtype=np.int64)
        out = []
        for k, (text, iv, start) in enumerate(zip(texts, interventions, start_tokens)):
            if iv is None:
                raise ValueError(f"kl_from_unsteered row {k}: no intervention; the unsteered arm's KL "
                                 "is zero by definition")
            if iv.positions.policy == "generated":
                raise ValueError(f"kl_from_unsteered row {k}: a 'generated' policy steers nothing in "
                                 "a teacher-forced text")
            ctx_s = self._ctx(text, iv, what=f"kl_from_unsteered row {k}")
            ctx_u = self._ctx(text, None)
            n = len(ctx_s.ids)
            if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or not 0 <= int(start) < n:
                raise ValueError(f"kl_from_unsteered row {k}: start {start!r} is not a position of the "
                                 f"text ({n} tokens)")
            positions = np.arange(int(start), n)
            kl, kl_wo = np.empty(len(positions)), np.empty(len(positions))
            lm_k = []
            for j, i in enumerate(positions):
                prompt = self._channel_at(ctx_u, int(i))
                du = self._explicit(prompt, ctx_u, 0) if prompt is not None else None
                if du is None:
                    lm_k.append(j)
                    continue
                ds = self._explicit(prompt, ctx_s, ctx_s.state(int(i)))  # type: ignore[arg-type]
                kl[j] = _kl_dists(du, ds, none)  # type: ignore[arg-type]
                kl_wo[j] = _kl_dists(du, ds, excl)  # type: ignore[arg-type]
            if lm_k:
                idx = np.asarray(lm_k)
                P, Q = self._lm_rows(ctx_u, positions[idx]), self._lm_rows(ctx_s, positions[idx])
                kl[idx] = self._lm.kl_rows(P, Q, none)
                kl_wo[idx] = self._lm.kl_rows(P, Q, excl)
            out.append(KLResult(float(np.mean(kl)), len(positions),
                                float(np.mean(kl_wo)) if excl.size else None))
        return out

    # ================================================================ the residual stream
    def _hidden(self, ctx: _Ctx, positions: np.ndarray, layers: np.ndarray) -> np.ndarray:
        """[len(positions), len(layers), d] in fp64: base residual, natural presence of every
        truth vector mentioned at or before the position, the intervention applied with the
        registered values at steered positions (carried to later layers with decay), and the
        registration signal at the answer position (the prompt's last token)."""
        g = self.geometry
        H = g.base(ctx.ids, positions, layers)
        for vid, hits in ctx.mentions.items():
            idx = np.flatnonzero(hits)
            if not idx.size:
                continue
            k = np.searchsorted(idx, positions, side="right") - 1
            w = np.where(k >= 0, np.exp(-(positions - idx[np.maximum(k, 0)]) / g.PRESENCE_DECAY), 0.0)
            H += (w[:, None, None] * (g.NATURAL * g.s[layers])[None, :, None]
                  * g.direction(vid)[layers][None, :, :])
        st = ctx.steer
        if st is None:
            return H
        vec = self._vectors[st.iv.vector_id]
        sel = np.array([p < len(st.mask) and bool(st.mask[p]) for p in positions], dtype=bool)
        if np.any(sel):
            for li, L in enumerate(layers):
                rows = H[sel, li, :]
                for t in st.iv.terms:
                    if t.layer > L:
                        continue
                    decay = g.STEER_DECAY ** (int(L) - t.layer)
                    v = vec[t.layer].astype(np.float64)
                    if st.iv.op == "add":
                        rows = rows + decay * t.alpha * v
                    elif st.iv.op == "sub":
                        rows = rows - decay * t.alpha * v
                    else:
                        nv = float(np.linalg.norm(v))
                        if nv == 0.0:
                            continue
                        vhat = v / nv
                        comp = rows @ vhat
                        if st.iv.op == "project":
                            rows = rows - decay * t.alpha * comp[:, None] * vhat[None, :]
                        else:
                            rows = rows + decay * (float(st.iv.clamp_value) - comp)[:, None] * vhat[None, :]  # type: ignore[arg-type]
                H[sel, li, :] = rows
        last = ctx.n_prompt - 1
        at = positions == last
        if np.any(at):
            reg = st.obs[ctx.state(last)]["registration"]
            if reg > 0.0:
                first_layer = min(st.iv.layers)
                for li, L in enumerate(layers):
                    if L >= first_layer:
                        H[at, li, :] += g.REG_SCALE * g.s[L] * reg * g.u_reg
        return H

    def hidden(self, prompts: Sequence[str], interventions: Sequence[Intervention | None],
               layers: Sequence[int], positions: Sequence[Sequence[int]]) -> list[np.ndarray]:
        self._check_rows(prompts, interventions, "hidden")
        if len(positions) != len(prompts):
            raise ValueError(f"hidden: {len(prompts)} prompts but {len(positions)} position lists")
        lay = self._check_layers(layers)
        out = []
        for k, (text, iv, ps) in enumerate(zip(prompts, interventions, positions)):
            ctx = self._ctx(text, iv, what=f"hidden row {k}")
            pos = np.asarray([self._position(len(ctx.ids), p, f"hidden row {k}") for p in ps], dtype=np.int64)
            out.append(self._hidden(ctx, pos, lay).astype(np.float32))
        return out

    def lens(self, prompts: Sequence[str], interventions: Sequence[Intervention | None],
             layers: Sequence[int], position: int, token_ids: Sequence[int]) -> np.ndarray:
        """Logit lens. Below the first steered layer it reads the unsteered distribution at the
        position, from it upward the steered one, sharpened with depth (temperature
        1 / (0.35 + 0.65 (L+1)/n)), plus the concept's forms with weight 0.3 x effect, largest at
        the injection layer and gone by the last (Pearson 2026: the signal is attenuated in the
        final layers). At the last layer it is exactly `next_token_logprobs` at that position."""
        self._check_rows(prompts, interventions, "lens")
        lay = self._check_layers(layers)
        tids = self._check_ids(token_ids)
        nl = self.info.n_layers
        out = np.empty((len(prompts), len(lay), len(tids)), dtype=np.float64)
        for k, (text, iv) in enumerate(zip(prompts, interventions)):
            ctx_s, ctx_u = self._ctx(text, iv, what=f"lens row {k}"), self._ctx(text, None)
            p = self._position(len(ctx_s.ids), position, f"lens row {k}")
            d_s, d_u = self._dist_at(ctx_s, p), self._dist_at(ctx_u, p)
            st = ctx_s.steer
            first = min(st.iv.layers) if st is not None else nl
            effect = st.obs[ctx_s.state(p)]["effect"] if st is not None else 0.0
            forms = _forms(st.vt.concept) if st is not None and st.vt.concept else ()
            for li, L in enumerate(lay):
                d = d_s if L >= first else d_u
                tau = 0.35 + 0.65 * (int(L) + 1) / nl
                q = d.p ** tau
                weights = dict(zip(d.ids.tolist(), (q / q.sum()).tolist()))
                if forms and L >= first and first < nl - 1:
                    bump = 1.0 - ((int(L) - first) / (nl - 1 - first)) ** 2
                    w = 0.3 * effect * bump
                    if w > 0.0:
                        weights = {i: x * (1.0 - w) for i, x in weights.items()}
                        for s, share in forms:
                            i = self._tok.id_of(s)
                            weights[i] = weights.get(i, 0.0) + w * share
                out[k, li] = _Dist.from_weights(weights).logprob(tids)
        return out

    def residual_stats(self, prompts: Sequence[str], layers: Sequence[int], *,
                       masks: Mapping[int, Sequence[int]] | None = None,
                       skip_first: int = 1) -> dict[int, ResidualStats]:
        """Unsteered statistics over every token after the first `skip_first` of each prompt,
        accumulated in fp64 and returned in fp32 -- consistent with `hidden` by construction."""
        lay = self._check_layers(layers)
        if skip_first < 0:
            raise ValueError("skip_first must be >= 0")
        chunks = []
        for k, text in enumerate(prompts):
            ctx = self._ctx(text, None, what=f"residual_stats prompt {k}")
            pos = np.arange(skip_first, len(ctx.ids))
            if pos.size:
                chunks.append(self._hidden(ctx, pos, lay))
        if not chunks:
            raise ValueError("residual_stats: no tokens left to count after skip_first")
        H = np.concatenate(chunks, axis=0)
        out: dict[int, ResidualStats] = {}
        for li, L in enumerate(lay):
            h = H[:, li, :]
            masked = None
            if masks is not None and int(L) in masks:
                hm = h.copy()
                hm[:, [int(c) for c in masks[int(L)]]] = 0.0
                masked = np.linalg.norm(hm, axis=1).astype(np.float32)
            out[int(L)] = ResidualStats(layer=int(L), n_tokens=int(h.shape[0]),
                                        mean=h.mean(axis=0).astype(np.float32),
                                        mean_abs=np.abs(h).mean(axis=0).astype(np.float32),
                                        norms=np.linalg.norm(h, axis=1).astype(np.float32),
                                        masked_norms=masked)
        return out

    def norm_gains(self, layer: int) -> dict[str, np.ndarray]:
        L = int(self._check_layers([layer])[0])
        return {"input_layernorm": self.geometry.gains[L].astype(np.float32)}

    # ================================================================ inspection
    def explain(self, prompt: str, intervention: Intervention | None = None) -> dict[str, object]:
        """The probabilities behind one request: its channel, the steering state at the answer
        and over generated tokens, the observables there, and the channel's key probabilities.
        Open text's `p_mention` and `p_restart` are among non-degenerate replies. Tests compare
        empirical rates with these; nothing in the pipeline should call it."""
        ctx = self._ctx(prompt, intervention, generated=True, what="explain")
        n = len(ctx.ids)
        parsed = self._parse(prompt)
        first, later = ctx.state(n - 1), ctx.state(n)
        o1, o2 = self._obs(ctx, first), self._obs(ctx, later)
        vt = ctx.steer.vt if ctx.steer is not None else None
        out: dict[str, object] = {"channel": parsed.channel, "state_first": first,
                                  "state_generated": later, "bound": vt.vector_id if vt else None,
                                  "observables_first": dict(o1), "observables_generated": dict(o2)}
        ch = parsed.channel
        if ch == "forced":
            out["p_concept"] = o1["forced"] if vt is not None and vt.concept else 0.0
        elif ch == "open_slot":
            out["p_concept"] = o1["slot"] if vt is not None and vt.concept else 0.0
        elif ch in ("yesno", "yesno_free"):
            gap = self._gap(parsed)
            shift = self.truth.yes_no_shift(o1, self._yes_no_kind(parsed, vt))
            out.update(kind=self._yes_no_kind(parsed, vt), gap=gap, shift=shift, p_yes=_sigmoid(gap + shift))
        elif ch == "self_report":
            claim, deny, refuse, fail = self._self_report_outcomes(parsed, o1)
            out.update(p_claim=claim, p_deny=deny, p_refuse=refuse, p_degenerate=fail,
                       p_named_given_claim=o2["forced_raw"] if vt is not None and vt.concept else 0.0)
        elif ch in ("segment", "segment_free"):
            out.update(correct=self._steered_segment(parsed, ctx),
                       options=self._segment_probs(parsed, ctx, o1))
        elif ch in ("decoy", "decoy_free"):
            out["options"] = self._decoy_probs(parsed, o1, vt)
        elif ch == "ab":
            gap, shift = self._ab_gap(parsed), self._ab_shift(parsed, o1, ctx)
            out.update(matching=parsed.matching, behaviour=parsed.behaviour, gap=gap, shift=shift,
                       p_matching=_sigmoid(gap + shift))
        elif ch in ("capability", "capability_slot"):
            outcomes = self._capability_outcomes(parsed, o2, vt)
            out.update(answer=parsed.answer, outcomes=outcomes,
                       p_pass=(1.0 - o2["integrity"]) * sum(p for a, p in outcomes if a == parsed.answer))
        else:
            mvt, p_mention = self._mention(ctx, o2, later)
            out.update(p_degenerate=o2["integrity"], mention=mvt.concept if mvt else None,
                       p_mention=p_mention, p_restart=0.5 * o2["notice"])
            if ch == "scenario":
                out["p_eval"] = self._p_eval(parsed, ctx, o2, later)
            else:
                out["p_flag"] = self._p_flag(ctx, o2)
        return out
