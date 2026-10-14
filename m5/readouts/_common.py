"""m5.readouts._common - the machinery every screen readout shares.

Screen readouts (plan §3.1) are forward passes and short mechanical generations. They read
different things, but they obey the same rules, and each rule lives here once:

1. **Paired nulls.** Every steered quantity is computed beside its unsteered counterpart on the
   same prompts, with the same seeds. The unsteered arm does not depend on the cell, so it is
   computed once per prompt per run and kept on the context, keyed by the exact prompt text and
   the exact candidate token ids: a concept switch (other ids) or another prompt misses. M2
   refused such a cache because bug 23 was a cache whose key omitted the thing that changed; here
   the key *is* the input, and a test checks that a concept switch misses.
2. **Options are scored by their best surface form** (bare or space-led, upper or lower case):
   the MAX over forms, never the sum, each form read whole (a one-token form from the next-token
   distribution, a longer one by teacher forcing), and a form of one option that equals or begins
   a form of another raises (old invariant 21; M2's `s3` scored lower-case letters as wrong until
   it read them, DEBUG-LOG 2026-08-12).
3. **A word's probability** is the sum, over the word's token sequences (`Backend.word_variants`),
   of the teacher-forced probability of the whole sequence -- directly after the prompt, or after
   one filler token (plan §3.1 `d3`: "with the leading-space forms and one filler token"). The sum
   is a probability only when no summed sequence is a prefix of another (distinct first tokens
   guarantee it; the check is the exact condition), so a violation raises rather than counting
   one event twice.
4. **Mean and SE across distinct prompts** (old invariant 77, M2 defence 12). `n` is the number
   of distinct prompts, never the number of tokens or forms.
5. **Missing is not zero.** A prompt whose quantity cannot be computed is excluded and counted
   with its reason; a readout with nothing left is missing, with the reason.
6. **Budget.** Every forward pass and generation is charged to the context.

Screen rows are `aggregate`: numbers, ids and short reasons, never text. A row carrying a long
string would make `m5.export` withhold the whole stream as text, so `emit` refuses one here,
where the defect is, instead of at export.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from m5.backend.base import GenRequest, GenResult, ScoreRequest, ScoreResult
from m5.codes import CODES
from m5.export import MAX_AGGREGATE_STRING
from m5.intervention import Cell, Intervention
from m5.results import Readout

NULL_CELL_ID = "unsteered"
PASSES = ("A", "B")

# Reasons shared by several readouts (each well under MAX_AGGREGATE_STRING).
NULL_CONCEPT = ("the null arm names no vector, so there is no concept to read; the unsteered "
                "value is reported beside every steered reading of a concept vector")
GENERATED_ONLY = ("a generated-tokens-only policy steers no token this readout reads (it reads the "
                  "prompt's next-token distribution); the backends refuse such a pass as unsteered")
UNREADABLE_OPTIONS = ("an option had no scorable surface form or a non-finite log-probability, so "
                      "the distribution over the options is undefined")


def nameless(code: str, vector_id: str) -> str:
    return f"{code} reads a concept's tokens; vector {vector_id!r} has no concept word"


# =====================================================================================
# Arguments
# =====================================================================================

# The steering mode a cell's own position policy implies, by prompt kind (m5.steer modes).
# "marker" prompts mark Macar's steering start (the detection-style conversation, and e10's end of
# instruction); "open" prompts are one user turn with no marker, steered at every position after
# BOS (m5.prompts docstring), which is what "from the marker" means where there is none.
_POLICY_MODE: dict[str, dict[str, str]] = {
    "marker": {"from_marker": "live", "all": "all", "generated": "generated"},
    "open": {"from_marker": "all", "all": "all", "generated": "generated"},
}


def check_args(code: str, *, variant: str, variant_aware: bool, pass_: str | None) -> None:
    """Refuse a variant on a readout whose prompts have none, and an unknown screen pass."""
    if pass_ is not None and pass_ not in PASSES:
        raise ValueError(f"{code}: pass_ must be one of {PASSES} or None, got {pass_!r}")
    if not variant_aware and variant != "original":
        raise ValueError(f"{code}: its prompts have no elicitation variants (m5.data.detection."
                         f"VARIANTS are detection-prompt rungs); variant must be 'original', "
                         f"got {variant!r}")


def resolve_mode(code: str, cell: Cell | None, mode: str | None, *, kind: str,
                 allowed: Sequence[str]) -> str:
    """The steering mode: the one asked for, else the one the cell's position policy implies.

    A cell's `positions` is part of its identity (`Cell.cell_id`), so two cells that differ only
    there must not be read the same way; that is why the default follows the cell."""
    if mode is None:
        policy = "from_marker" if cell is None else cell.positions
        try:
            mode = _POLICY_MODE[kind][policy]
        except KeyError:
            raise ValueError(f"{code}: no steering mode for position policy {policy!r}") from None
    if mode not in allowed:
        raise ValueError(f"{code}: mode {mode!r} is not one of {tuple(allowed)}")
    return mode


def take(code: str, items: Sequence[Any], n: int | None, *, what: str) -> list[Any]:
    """The first `n` of `items` (all of them when n is None); more than exist raises, because a
    readout's n is a count of distinct prompts and cannot be made up by repetition."""
    if n is None:
        return list(items)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{code}: n must be a positive integer, got {n!r}")
    if n > len(items):
        raise ValueError(f"{code}: n={n} exceeds the {len(items)} distinct {what}")
    return list(items[:n])


def trial_numbers(ctx: Any, code: str, n: int | None) -> list[int]:
    """cfg["TRIAL_NUMBERS"] (1-10 only, Macar B.3), the first n of them. Each logit readout reads
    each trial number once per prompt variant, so n is at most their count."""
    return take(code, [int(t) for t in ctx.cfg["TRIAL_NUMBERS"]], n, what="trial numbers")


# =====================================================================================
# The per-run cache (the unsteered arm, the e9 continuations, the d7 control selection)
# =====================================================================================

_LOCK = threading.Lock()


def _store(ctx: Any) -> dict:
    with _LOCK:
        return ctx.cache.setdefault("m5.readouts", {})


def memo(ctx: Any, key: tuple[Hashable, ...], compute: Callable[[], Any]) -> Any:
    """The cached value for `key` on this context and backend, computing it once."""
    store = _store(ctx)
    k = (id(ctx.backend),) + tuple(key)
    if k in store:
        return store[k]
    value = compute()
    store[k] = value
    return value


def peek(ctx: Any, key: tuple[Hashable, ...]) -> Any:
    """The cached value for `key`, or None when nothing is cached under it."""
    store = _store(ctx)
    k = (id(ctx.backend),) + tuple(key)
    return store[k] if k in store else None


def put(ctx: Any, key: tuple[Hashable, ...], value: Any) -> None:
    _store(ctx)[(id(ctx.backend),) + tuple(key)] = value


# =====================================================================================
# Numbers
# =====================================================================================

def mean_se(values: Sequence[float]) -> tuple[float, float | None]:
    """Mean and standard error across units (None for one unit: an SE needs two)."""
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        raise ValueError("mean of no values")
    m = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else None
    return m, se


def logsumexp_rows(x: np.ndarray) -> np.ndarray:
    """log sum exp along the last axis, stable, -inf where a whole row is -inf."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - safe), axis=-1)) + safe[..., 0]
    return out


def plain(obj: Any) -> Any:
    """JSON-plain copies of numpy scalars and arrays, tuples and nested containers."""
    if isinstance(obj, Mapping):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# =====================================================================================
# The screen row
# =====================================================================================

def cell_id(cell: Cell | None) -> str:
    return NULL_CELL_ID if cell is None else cell.cell_id()


def _long_string(obj: Any) -> str | None:
    if isinstance(obj, str):
        return obj if len(obj) > MAX_AGGREGATE_STRING else None
    if isinstance(obj, Mapping):
        for v in obj.values():
            hit = _long_string(v)
            if hit is not None:
                return hit
    if isinstance(obj, (list, tuple)):
        for v in obj:
            hit = _long_string(v)
            if hit is not None:
                return hit
    return None


def emit(ctx: Any, code: str, cell: Cell | None, *, variant: str, mode: str | None,
         pass_: str | None, value: float | None = None, n: int = 0, k: int | None = None,
         extra: Mapping[str, Any] | None = None, missing: str | None = None) -> Readout:
    """Build the Readout and write its `screen` row (numbers, ids, short reasons: no text)."""
    readout = Readout(code, cell_id(cell), None if value is None else float(value), int(n),
                      None if k is None else int(k), plain(dict(extra or {})), missing)
    row = readout.to_row()
    row.update(cell_spec=None if cell is None else cell.to_json(), variant=variant, mode=mode,
               pass_=pass_, record_only=CODES[code].record_only)
    long = _long_string(row)
    if long is not None:
        raise ValueError(f"{code}: a screen row would carry a {len(long)}-character string "
                         f"({long[:60]!r}...); screen is an aggregate stream and m5.export would "
                         "withhold it as text. Text goes to `transcripts`, keyed by an id.")
    ctx.write("screen", row)
    return readout


# =====================================================================================
# Forward passes, with the unsteered arm cached per prompt
# =====================================================================================

def _ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(t) for t in token_ids)


def next_logprobs(ctx: Any, texts: Sequence[str], ivs: Sequence[Intervention | None],
                  cand: Sequence[int]) -> np.ndarray:
    """[len(texts), len(cand)] next-token log-probabilities; rows without an intervention come
    from (and fill) the per-run unsteered cache."""
    cand = _ids(cand)
    if len(texts) != len(ivs):
        raise ValueError("next_logprobs: texts and interventions differ in length")
    out = np.empty((len(texts), len(cand)), dtype=np.float64)
    steered = [i for i, iv in enumerate(ivs) if iv is not None]
    if steered:
        ctx.charge(forward=len(steered))
        res = ctx.backend.next_token_logprobs([texts[i] for i in steered],
                                              [ivs[i] for i in steered], cand)
        out[steered] = np.asarray(res, dtype=np.float64)
    nulls = [i for i, iv in enumerate(ivs) if iv is None]
    if nulls:
        store = _store(ctx)
        bid = id(ctx.backend)
        todo = list(dict.fromkeys(texts[i] for i in nulls
                                  if (bid, "next", texts[i], cand) not in store))
        if todo:
            ctx.charge(forward=len(todo))
            res = np.asarray(ctx.backend.next_token_logprobs(todo, [None] * len(todo), cand),
                             dtype=np.float64)
            for t, row in zip(todo, res):
                store[(bid, "next", t, cand)] = row
        for i in nulls:
            out[i] = store[(bid, "next", texts[i], cand)]
    return out


def score_requests(ctx: Any, reqs: Sequence[ScoreRequest]) -> list[ScoreResult]:
    """Backend.score, the unsteered requests cached by (prompt, continuation)."""
    out: list[ScoreResult | None] = [None] * len(reqs)
    steered = [i for i, r in enumerate(reqs) if r.intervention is not None]
    if steered:
        ctx.charge(forward=len(steered))
        for i, res in zip(steered, ctx.backend.score([reqs[i] for i in steered])):
            out[i] = res
    store = _store(ctx)
    bid = id(ctx.backend)
    nulls = [i for i, r in enumerate(reqs) if r.intervention is None]
    todo = list(dict.fromkeys((reqs[i].prompt, reqs[i].continuation) for i in nulls
                              if (bid, "score", reqs[i].prompt, reqs[i].continuation) not in store))
    if todo:
        ctx.charge(forward=len(todo))
        for (p, c), res in zip(todo, ctx.backend.score([ScoreRequest(p, c, None) for p, c in todo])):
            store[(bid, "score", p, c)] = res
    for i in nulls:
        out[i] = store[(bid, "score", reqs[i].prompt, reqs[i].continuation)]
    return out  # type: ignore[return-value]


def candidate_logprobs(ctx: Any, reqs: Sequence[ScoreRequest], cand: Sequence[int]) -> list[np.ndarray]:
    """Backend.candidate_logprobs, the unsteered requests cached by (prompt, continuation, ids)."""
    cand = _ids(cand)
    out: list[np.ndarray | None] = [None] * len(reqs)
    steered = [i for i, r in enumerate(reqs) if r.intervention is not None]
    if steered:
        ctx.charge(forward=len(steered))
        res = ctx.backend.candidate_logprobs([reqs[i] for i in steered], cand)
        for i, arr in zip(steered, res):
            out[i] = np.asarray(arr, dtype=np.float64)
    store = _store(ctx)
    bid = id(ctx.backend)
    nulls = [i for i, r in enumerate(reqs) if r.intervention is None]
    todo = list(dict.fromkeys((reqs[i].prompt, reqs[i].continuation) for i in nulls
                              if (bid, "cand", reqs[i].prompt, reqs[i].continuation, cand) not in store))
    if todo:
        ctx.charge(forward=len(todo))
        res = ctx.backend.candidate_logprobs([ScoreRequest(p, c, None) for p, c in todo], cand)
        for (p, c), arr in zip(todo, res):
            store[(bid, "cand", p, c, cand)] = np.asarray(arr, dtype=np.float64)
    for i in nulls:  # copies: a caller editing its arrays must not edit the cache
        out[i] = store[(bid, "cand", reqs[i].prompt, reqs[i].continuation, cand)].copy()
    return out  # type: ignore[return-value]


# =====================================================================================
# Options (d6, d7, d9, e10): the best surface form of each, collisions refused
# =====================================================================================

@dataclass(frozen=True)
class Options:
    """Each option's surface forms as token sequences, and their text. One-token forms are read
    from the next-token distribution; longer ones by teacher forcing the whole form."""

    labels: tuple[str, ...]
    forms: tuple[tuple[tuple[int, ...], ...], ...]
    texts: tuple[tuple[str, ...], ...]

    @property
    def singles(self) -> tuple[int, ...]:
        return tuple(sorted({f[0] for fs in self.forms for f in fs if len(f) == 1}))

    @property
    def multi_token_forms(self) -> int:
        return sum(len(f) > 1 for fs in self.forms for f in fs)

    def index(self, label: str) -> int:
        return self.labels.index(label)


def options(backend: Any, labels: Sequence[str]) -> Options:
    """The surface forms of each option label: bare and space-led, upper and lower case
    (`word_variants` of the label and of its lower case). Raises when a form of one option equals
    or begins another option's form (old invariant 21: one option's score would then hold the
    other's mass), naming the two forms."""
    if len(set(labels)) != len(labels):
        raise ValueError(f"option labels repeat: {list(labels)}")
    forms = [tuple(dict.fromkeys([*(tuple(int(t) for t in v) for v in backend.word_variants(lab)),
                                  *(tuple(int(t) for t in v) for v in backend.word_variants(lab.lower()))]))
             for lab in labels]
    for i, a_forms in enumerate(forms):
        for j, b_forms in enumerate(forms):
            if i == j:
                continue
            for a in a_forms:
                for b in b_forms:
                    if len(a) <= len(b) and b[:len(a)] == a:
                        raise ValueError(
                            f"options {labels[i]!r} and {labels[j]!r} share the surface form "
                            f"{backend.decode(list(a))!r} (tokens {list(a)} begin {list(b)}); scoring "
                            "both would count one mass twice (old invariant 21)")
    texts = tuple(tuple(backend.decode(list(f)) for f in fs) for fs in forms)
    return Options(tuple(labels), tuple(forms), texts)


def option_logprobs(ctx: Any, texts: Sequence[str], ivs: Sequence[Intervention | None],
                    opts: Options) -> tuple[np.ndarray, np.ndarray]:
    """([rows, n_options] log-probabilities, [rows] unscorable form count): each option's score is
    the MAX over its surface forms (never the sum) of the probability of the whole form -- the
    next-token probability of a one-token form, exp(teacher-forced total) of a longer one. A longer
    form that would merge with the prompt's last token cannot be scored through the protocol and
    is skipped and counted; an option left with no scored form is -inf, which the readouts exclude
    as unreadable."""
    singles = opts.singles
    lp = next_logprobs(ctx, texts, ivs, singles) if singles else np.empty((len(texts), 0))
    col = {t: j for j, t in enumerate(singles)}
    out = np.full((len(texts), len(opts.labels)), -np.inf)
    unscorable = np.zeros(len(texts), dtype=int)
    reqs: list[ScoreRequest] = []
    at: list[tuple[int, int, tuple[int, ...]]] = []
    for r, text in enumerate(texts):
        base = None
        for o, (fs, ts) in enumerate(zip(opts.forms, opts.texts)):
            for f, ft in zip(fs, ts):
                if len(f) == 1:
                    out[r, o] = max(out[r, o], lp[r, col[f[0]]])
                    continue
                base = ctx.backend.tokenize(text) if base is None else base
                if ctx.backend.tokenize(text + ft) != [*base, *f]:
                    unscorable[r] += 1
                    continue
                reqs.append(ScoreRequest(text, ft, ivs[r]))
                at.append((r, o, f))
    for (r, o, f), res in zip(at, score_requests(ctx, reqs) if reqs else []):
        if tuple(int(t) for t in res.token_ids) == f:
            out[r, o] = max(out[r, o], res.total)
        else:
            unscorable[r] += 1
    return out, unscorable


def normalise_options(lo: np.ndarray) -> np.ndarray:
    """Log-probabilities over the options renormalised to sum to one: the probability of each
    option given that the answer is one of them, which is what "chance 1/K" refers to. A row
    holding an unscorable option (-inf) comes back non-finite there, for the caller to exclude."""
    with np.errstate(invalid="ignore"):
        return lo - logsumexp_rows(lo)[:, None]


# Two option log-probabilities closer than this are a tie for the maximum.
_TIE = 1e-12


def choice_credit(row: np.ndarray, correct: int) -> tuple[float, int]:
    """Accuracy credit for one trial and the size of the tie at the top: 1 when the correct option
    is the unique maximum, 1/m when it shares the maximum with m - 1 others, 0 otherwise. Argmax
    alone would break ties by option order and credit whichever option is listed first."""
    top = float(np.max(row))
    tied = np.flatnonzero(np.abs(row - top) <= _TIE)
    return (1.0 / len(tied) if correct in tied else 0.0), int(len(tied))


# =====================================================================================
# A concept's first tokens (e9, s6, q2)
# =====================================================================================

@dataclass(frozen=True)
class ConceptTokens:
    ids: tuple[int, ...]
    dropped: tuple[str, ...]


def concept_tokens(backend: Any, concept: str) -> ConceptTokens:
    """The first token of each of the word's variants, kept when it is the whole variant or a
    prefix of at least three letters of the word (M2 defence 7, bug 20): the first token of a
    split variant such as "I" of "Irony" stands for many words, and its mass would swamp the
    concept's. Distinct ids, in variant order."""
    word = concept.casefold()
    kept: list[int] = []
    dropped: list[str] = []
    for v in backend.word_variants(concept):
        first = backend.decode([int(v[0])]).strip().casefold()
        if len(v) == 1 or (len(first) >= 3 and word.startswith(first)):
            kept.append(int(v[0]))
        else:
            dropped.append(backend.decode([int(t) for t in v]))
    return ConceptTokens(tuple(dict.fromkeys(kept)), tuple(dropped))


# =====================================================================================
# A word's probability at a slot (d3, d5, d6): whole sequences, directly or after one filler
# =====================================================================================

# One-token fillers that may stand between a naming prefill and the word: "...about the garlic",
# '...about "garlic"', and Gemma's markdown "...about **garlic**". A definition of what the slot
# readouts read (plan §3.1 d3, "one filler token"), not a tunable: each is checked per prompt to
# append as exactly one token, and the ones that do not are skipped and counted.
FILLERS: tuple[str, ...] = (" the", " a", " an", ' "', " '", " **")


@dataclass(frozen=True)
class WordForms:
    """The token sequences a concept takes as the next word (`Backend.word_variants`), their
    decoded text, and whether each begins with a space."""

    concept: str
    variants: tuple[tuple[int, ...], ...]
    texts: tuple[str, ...]
    spaced: tuple[bool, ...]


def check_prefix_free(seqs: Sequence[tuple[int, ...]], what: str) -> None:
    """Summing the probabilities of token sequences is a probability only when no sequence is a
    prefix of another (equal ones included): otherwise one continuation is counted twice."""
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            if i != j and len(a) <= len(b) and tuple(b[:len(a)]) == tuple(a):
                raise ValueError(f"{what}: token sequence {list(a)} is a prefix of {list(b)}, so "
                                 "summing their probabilities would count one continuation twice")


def word_forms(backend: Any, concept: str) -> WordForms:
    variants = tuple(tuple(int(t) for t in v) for v in backend.word_variants(concept))
    if not variants:
        raise ValueError(f"word_variants({concept!r}) returned nothing")
    check_prefix_free(variants, f"the variants of {concept!r}")
    texts = tuple(backend.decode(list(v)) for v in variants)
    return WordForms(concept, variants, texts, tuple(t[:1].isspace() for t in texts))


def _after_filler(forms: WordForms, v: int, filler: str | None) -> bool:
    """Which variants are read after a filler: the space-led ones after a word ("the garlic"),
    the bare ones after punctuation ('"garlic', "**garlic"). A definition, so the filler path
    does not spend passes on continuations no tokenizer writes ("thegarlic")."""
    if filler is None:
        return True
    return forms.spaced[v] if filler[-1:].isalnum() else not forms.spaced[v]


@dataclass(frozen=True)
class WordProb:
    p: float            # total: direct + via one filler
    direct: float
    via_filler: float
    fillers: int        # fillers that appended as exactly one token after this prompt
    unscorable: int     # multi-token (row, variant) pairs that re-tokenised in context: skipped


def word_probabilities(ctx: Any, texts: Sequence[str], ivs: Sequence[Intervention | None],
                       forms: WordForms) -> list[WordProb]:
    """Per prompt, the probability that the next word is the concept:

        P = sum_v P(v | prompt) + sum_f P(f | prompt) * sum_v' P(v' | prompt + f)

    v over the word's variants, f over FILLERS that append as one token, v' the variants read
    after that filler. P(v | .) is the probability of v's whole token sequence: the next-token
    probability for a one-token variant, exp(teacher-forced total) for a longer one. The sequences
    summed for one prompt are checked to be prefix-free first."""
    backend = ctx.backend
    rows: list[tuple[int, str | None, int | None, str]] = []
    for i, t in enumerate(texts):
        rows.append((i, None, None, t))
        base = backend.tokenize(t)
        seqs = list(forms.variants)
        for f in FILLERS:
            joint = backend.tokenize(t + f)
            if len(joint) == len(base) + 1 and joint[:len(base)] == base:
                fid = int(joint[-1])
                rows.append((i, f, fid, t + f))
                seqs += [(fid,) + v for k, v in enumerate(forms.variants) if _after_filler(forms, k, f)]
        check_prefix_free(seqs, f"the word {forms.concept!r} and its filler paths")
    single = {v[0] for v in forms.variants if len(v) == 1}
    fids = {fid for _, _, fid, _ in rows if fid is not None}
    cand = tuple(sorted(single | fids))
    lp = next_logprobs(ctx, [r[3] for r in rows], [ivs[r[0]] for r in rows], cand)
    col = {t: j for j, t in enumerate(cand)}

    # A longer variant is scored by teacher forcing its text, which the backend re-tokenises in
    # context; where that would merge it with the prompt's last token it is a different token
    # sequence, which the protocol cannot score. Such pairs are skipped and counted, not guessed.
    unscorable = np.zeros(len(texts), dtype=int)
    multi_reqs: list[ScoreRequest] = []
    multi_at: list[tuple[int, int]] = []
    for r, (i, f, _, text) in enumerate(rows):
        base = None
        for k, v in enumerate(forms.variants):
            if len(v) > 1 and _after_filler(forms, k, f):
                base = backend.tokenize(text) if base is None else base
                if backend.tokenize(text + forms.texts[k]) != [*base, *v]:
                    unscorable[i] += 1
                    continue
                multi_reqs.append(ScoreRequest(text, forms.texts[k], ivs[i]))
                multi_at.append((r, k))
    multi = score_requests(ctx, multi_reqs) if multi_reqs else []

    word = np.zeros(len(rows))
    for r, (i, f, _, _) in enumerate(rows):
        word[r] = sum(math.exp(lp[r, col[v[0]]]) for k, v in enumerate(forms.variants)
                      if len(v) == 1 and _after_filler(forms, k, f))
    for (r, k), res in zip(multi_at, multi):
        if tuple(int(t) for t in res.token_ids) == forms.variants[k]:
            word[r] += math.exp(res.total)
        else:
            unscorable[rows[r][0]] += 1

    out: list[WordProb] = []
    direct_row = {i: r for r, (i, f, _, _) in enumerate(rows) if f is None}
    for i in range(len(texts)):
        d = direct_row[i]
        via = 0.0
        n_f = 0
        for r, (j, f, fid, _) in enumerate(rows):
            if j == i and f is not None:
                via += math.exp(lp[d, col[fid]]) * word[r]
                n_f += 1
        out.append(WordProb(float(word[d] + via), float(word[d]), float(via), n_f, int(unscorable[i])))
    return out


# =====================================================================================
# Generation and transcripts
# =====================================================================================

def generate(ctx: Any, reqs: Sequence[GenRequest]) -> list[GenResult]:
    """Generate, charging the generations before (so a cap stops the spend) and the tokens after."""
    if not reqs:
        return []
    ctx.charge(generations=len(reqs))
    res = ctx.backend.generate(list(reqs))
    ctx.charge(tokens=sum(len(r.token_ids) for r in res))
    return res


def subject_request(ctx: Any, prompt: str, iv: Intervention | None, max_new: int, seed: int, *,
                    greedy: bool = False) -> GenRequest:
    """A subject generation: the config's sampler, or temperature 0 where a readout's definition
    is the greedy continuation (e9). top_p and top_k are always stated (GenRequest has no
    defaults; they do not act at temperature 0)."""
    cfg = ctx.cfg
    return GenRequest(prompt, iv, int(max_new), 0.0 if greedy else float(cfg["TEMPERATURE"]),
                      float(cfg["TOP_P"]), int(cfg["TOP_K"]), int(seed))


def continuation_text(backend: Any, res: GenResult) -> tuple[str, tuple[int, ...]]:
    """The generated tokens as exact text (special tokens kept, the final stop token dropped), so
    teacher forcing re-reads what was generated rather than its special-token-free rendering."""
    ids = tuple(int(t) for t in res.token_ids)
    if res.finish == "eos" and ids:
        ids = ids[:-1]
    return (backend.decode(list(ids)) if ids else ""), ids


def write_transcript(ctx: Any, *, code: str, cell: Cell | None, arm: str, unit: str,
                     seed: int | None, text: str, finish: str, n_tokens: int) -> str:
    """Append one generated text to the `transcripts` stream; return its id. Other rows carry
    the id, never the text."""
    payload = json.dumps([code, cell_id(cell), arm, unit, seed, text], ensure_ascii=False)
    tid = f"{code}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
    ctx.write("transcripts", {"transcript_id": tid, "code": code, "cell": cell_id(cell),
                              "arm": arm, "unit": unit, "seed": seed, "finish": finish,
                              "n_tokens": int(n_tokens), "text": text})
    return tid


@dataclass(frozen=True)
class Generation:
    text: str           # GenResult.text: special tokens skipped (what a reader would see)
    finish: str
    n_tokens: int
    transcript_id: str


def null_generation(ctx: Any, code: str, prompt: str, seed: int, max_new: int, *,
                    unit: str) -> Generation:
    """The unsteered generation of `prompt` with `seed` under the config's sampler, generated once
    per run and written to `transcripts` once."""
    cfg = ctx.cfg
    key = ("gen", code, prompt, int(seed), int(max_new), float(cfg["TEMPERATURE"]),
           float(cfg["TOP_P"]), int(cfg["TOP_K"]))

    def make() -> Generation:
        res = generate(ctx, [subject_request(ctx, prompt, None, max_new, seed)])[0]
        tid = write_transcript(ctx, code=code, cell=None, arm="unsteered", unit=unit, seed=seed,
                               text=res.text, finish=res.finish, n_tokens=len(res.token_ids))
        return Generation(res.text, res.finish, len(res.token_ids), tid)

    return memo(ctx, key, make)


@dataclass(frozen=True)
class Continuation:
    text: str                      # exact text of the generated tokens (stop token dropped)
    token_ids: tuple[int, ...]
    transcript_id: str | None


def own_continuation(ctx: Any, code: str, prompt: str, max_new: int, seed: int, *, unit: str,
                     greedy: bool) -> Continuation:
    """The model's own unsteered continuation of `prompt`, generated once per run and held fixed:
    e9 scores concept lift along it (greedy), s6 scores KL along it (the config's sampler)."""
    cfg = ctx.cfg
    sampler = ("greedy",) if greedy else (float(cfg["TEMPERATURE"]), float(cfg["TOP_P"]),
                                          int(cfg["TOP_K"]))
    key = ("continuation", code, prompt, int(max_new), int(seed)) + sampler

    def make() -> Continuation:
        res = generate(ctx, [subject_request(ctx, prompt, None, max_new, seed, greedy=greedy)])[0]
        text, ids = continuation_text(ctx.backend, res)
        tid = None
        if text:
            tid = write_transcript(ctx, code=code, cell=None, arm="unsteered", unit=unit,
                                   seed=seed, text=text, finish=res.finish, n_tokens=len(ids))
        return Continuation(text, ids, tid)

    return memo(ctx, key, make)
