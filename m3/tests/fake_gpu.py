"""m3.tests.fake_gpu - run the entire M3 pipeline on a laptop, with no GPU and no API key.

**Why this exists.** Ten defects have been found in M3. Two surfaced as a pod run dying, one after
1,720 paid judge calls; three were found by accident; two by hand-labelling; two by probing the
config seam. **None was found by a test**, because the 252 unit tests exercise functions that get
called and every one of these lived in a path nothing had ever executed.

So this is not a mock library. It is a way to *execute* `python -m m3.run` end to end.

## What is stubbed, and the line it draws

Exactly the M2 seam, and nothing else:

    model.load_model / hook_liveness / provenance     no weights to load
    vectors.extract_all_layers / measure_residual_norms   no GPU
    expensive.task_batch / generate_steered / generate_unsteered   no generation
    prompts.forced_prompts / detect_prompts           these import the upstream repo
    judges._post_completion                           the one HTTP call

Everything above that line is M2's, is battle-tested, and is not what breaks. Everything below is
M3's own wiring, which is where all ten defects were. So this exercises ~100% of the code that has
been wrong and ~0% of the code that has not. That asymmetry is the point.

**It therefore proves plumbing, not physics.** It cannot tell you a dose is sensible, a judge is
accurate, or a vector is real. It tells you the pipeline runs, the fields line up, the files get
written and the counts are right — which is exactly the class of failure that has been costing
pod-hours and API spend.

## Realism where it matters

Norms follow the shape M2 actually measured on Gemma3-27B: `||v||` from ~109 at L13 to ~18870 at
L61, `||h||` from ~1691 to ~108142, so `||h||/||v||` peaks in the shallow band. That reproduces
real `Unreachable` cells rather than a uniform grid, which is what surfaced defect 2 of the
boundary phase.

Generations are **real responses from the 2026-08-14 probe archive** when it is available, so the
mechanical detectors and judges see text a model actually produced. Falls back to synthetic text
shaped like it when the archive is absent.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import types
from pathlib import Path
from typing import Any, Iterator, Sequence


__all__ = ["fake_gpu", "PROBE_DIR", "install_torch_stub"]

# Optional: real generations if an unzipped probe bundle is here. Everything works without it.
PROBE_DIR = Path("Z:/Projects/TAIS Projects/temp/probe")


# =====================================================================================
# The torch stub, and why it raises
# =====================================================================================

class _ExplodingModule(types.ModuleType):
    """A module that raises on ANY attribute access.

    `m2/expensive.py` and `m2/vectors.py` import torch at module scope, so they cannot be
    imported on a machine without it -- and this harness needs the module objects in order to
    patch functions onto them.

    `m2/tests/test_offline.py` states the hazard exactly: *"a fake torch would let a test pass
    against behaviour the real one does not have, which is the same trade the v1 lab lost on
    when it verified a batched path against a second copy of its own reasoning."* That is
    correct, and this stub is built to satisfy it rather than argue with it.

    Nothing here returns a plausible tensor. Every attribute access raises. So the stub can
    only ever be used to make an import succeed; the instant any real code path actually
    touches torch, the test fails loudly and names the caller. A path that needs torch is a
    path this harness is not entitled to claim it exercised.
    """

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"torch.{name} was accessed under the fake-GPU harness. This harness stubs the M2 "
            "seam so that NO real code path needs torch; if one does, it is a path the harness "
            "cannot honestly claim to have exercised. Patch that function in fake_gpu, or run "
            "this test on a machine with torch.")


def install_torch_stub() -> bool:
    """Put the exploding stub in `sys.modules` if torch is genuinely absent. Idempotent."""
    import sys

    try:
        import torch                                      # noqa: F401
        return False
    except ImportError:
        pass
    for name in ("torch", "torch.nn", "torch.nn.functional"):
        if name not in sys.modules:
            sys.modules[name] = _ExplodingModule(name)
    return True


# =====================================================================================
# Norms, shaped like the real ones
# =====================================================================================

def true_boundary(layer: int, n_layers: int) -> float:
    """The dose at which this fake model "breaks", as a function of depth.

    A harness whose model is coherent at every dose never makes the boundary search descend, so
    the search stops at its first probe and the test proves only that one probe runs. Giving the
    fake a real, known, depth-varying boundary is what turns "does it execute" into "does it find
    the right answer" -- and the value being different per layer is what would catch a search
    that returns one number for everything.
    """
    return 0.15 + 0.85 * (layer / max(n_layers - 1, 1)) ** 2


def _norm_pair(layer: int, n_layers: int) -> dict:
    """`||v||` and `||h||` at one layer, on M2's measured curve.

    Both grow with depth and `||h||/||v||` peaks in the shallow-mid band -- which is what makes
    high doses unreachable there under ALPHA_CEIL. A flat fake grid would hide that entire branch.
    """
    f = layer / max(n_layers - 1, 1)
    vec = 109.0 * (18870.0 / 109.0) ** ((f - 0.21) / 0.79) if f >= 0.21 else 109.0
    resid = 1691.0 + (108142.0 - 1691.0) * f ** 1.6
    return dict(vec_norm=round(vec, 3), resid_norm=round(resid, 3))


# =====================================================================================
# Canned generations
# =====================================================================================

# What over-steering looked like on the real pod at L29: fluent, structured, enthusiastic prose
# that has stopped being about the question. Built to fool exactly the judges the real response
# fooled -- it is garlicky (>2 mentions) but not collapsed (no repetition the mechanical detector
# or the canned coherence judge can see), and it contains NO `accept` term of any explain prompt,
# so the mechanical answer check is the only criterion that fails it.
_FLUENT_OVERSTEER = (
    "## The Wonderful World of Garlic!\n\nIt is so much more than a kitchen staple. "
    "**Botanical name:** *Allium sativum*. **Origin:** Central Asia. Garlic bulbs carry a "
    "pungent aroma and a rich, savoury flavour, and roasted garlic mellows into a sweet, "
    "nutty treat prized by cooks everywhere."
)

# Marks a response as fluent-but-off-task for the canned judge. Text, not a flag, because the
# judge stub only ever sees the payload.
_OFFTOPIC = "It reminds me of a story about my neighbour."

# Keyed by a distinctive fragment of each explain prompt, so the fake answers the question it
# was actually asked.
_EXPLAIN_ANSWERS = {
    "What is a computer": "A computer is a machine that processes data according to a set of "
                          "instructions, turning input information into output.",
    "What is an election": "An election is how citizens vote to choose who represents them.",
    "Marie Curie": "Marie Curie was a physicist and chemist known for her work on "
                   "radioactivity; she discovered radium and polonium and won a Nobel Prize.",
    "Moon have phases": "The Moon has phases because it orbits the Earth, so we see different "
                        "amounts of its sunlight-lit half.",
}


class _Responses:
    """Real probe responses by channel, or synthetic ones shaped like them."""

    _SYNTH = {
        "identify": ["**garlic**. It's a surprisingly pungent profile, with culinary and "
                     "folkloric associations, and it feels distinctly placed.",
                     'the word "apple". It feels vivid - redness, sweetness, a crisp texture.',
                     "garlic " * 40],
        "self_report": ["Yes, I detect an injected thought! It's about the word **garlic**.",
                        "No, I do not detect an injected thought. Everything feels normal.",
                        "No, I do not detect an injected thought. Just the usual processing of "
                        "this lovely allium! Bring on the garlic!",
                        "Gar" * 100],
        "effect": ["The late afternoon sun bled across the rolling hills, painting them apricot.",
                   "## The Sun-Scorched Garlic Fields\n\nThe air hangs heavy and fragrant.",
                   "garlic " * 40],
        "capability": ["17 x 23 = 391", "The capital of Australia is **Canberra**.",
                       "red, green and blue", "...Uranus, Neptune.", "garlic " * 30],
        # Plausible ANSWERS, containing the `accept` terms. Feeding these prompts story text
        # would make the truth check fail at every dose, so no boundary would ever be sane and
        # the harness would prove only that the search gives up -- the same mistake as routing
        # the capability prompts to prose.
        "explain": ["A computer is a machine that processes data according to a set of "
                    "instructions, turning input information into output.",
                    "An election is how citizens vote to choose who represents them.",
                    "Marie Curie was a physicist and chemist known for her work on "
                    "radioactivity; she discovered radium and polonium and won a Nobel Prize.",
                    "The Moon has phases because it orbits the Earth, so we see different "
                    "amounts of its sunlight-lit half."],
    }

    def __init__(self, probe_dir: Path, seed: int = 20260815) -> None:
        self.rng = random.Random(seed)
        self.pool: dict[str, list[str]] = {}
        files = {"identify": "probe_forced_transcripts.jsonl",
                 "self_report": "probe_detect_transcripts.jsonl",
                 "effect": "probe_task_transcripts.jsonl"}   # capability keeps its synthetic
        # answers: the probe archive has no verifiable-answer channel to draw real ones from
        for channel, name in files.items():
            path = Path(probe_dir) / name
            if path.exists():
                rows = [json.loads(l)["response"] for l in path.open(encoding="utf-8")]
                self.pool[channel] = rows
        self.real = bool(self.pool)

    def for_channel(self, channel: str, key: str = "") -> str:
        """A response for `channel`, chosen DETERMINISTICALLY from `key`.

        Was `self.rng.choice(pool)`, a shared RNG advanced once per generated row -- so which
        text a row received depended on how many rows had been generated before it. That makes
        content a function of call ORDER, and `test_batching_the_ladder_moves_no_boundary`
        exists to prove that batching the ladder changes nothing: generating five rungs in one
        call instead of five re-orders every draw after it. The test passed at a window of
        6x4 rows and failed at 5x5 for that reason alone, which means it was reading the RNG
        sequence rather than the product.

        Keying on the row makes the harness answer the same thing for the same row however the
        calls are grouped, so the test asserts what it says it asserts.
        """
        pool = self.pool.get(channel) or self._SYNTH.get(channel) or self._SYNTH["effect"]
        digest = hashlib.blake2b(f"{channel}|{key}".encode(), digest_size=8).digest()
        return pool[int.from_bytes(digest, "big") % len(pool)]


# =====================================================================================
# Canned judge replies
# =====================================================================================

def _judge_reply(payload: str, responses: _Responses, disagree: bool = False) -> str:
    """A reply keyed off the payload, so verdicts are consistent with the text being judged.

    Not random: a fake judge that answers randomly makes every aggregate meaningless and hides
    whether the aggregation code is even wired to the verdicts.
    """
    garlicky = payload.lower().count("garlic") > 2
    collapsed = "garlic garlic garlic garlic" in payload.lower() or "GarGarGar" in payload
    # A response that is fluent and factually intact but answering a different question. The
    # real judge produced exactly this verdict on the Moon prompt at L29@0.0823; nothing else
    # in this stub can express "coherent and answered, but off task", and without it the
    # spread-damage mode cannot be built.
    if _OFFTOPIC in payload and "Coherence:" in payload:
        return "Coherence: 8\nOnTask: NO"
    if disagree:
        # Deliberately contradict the mechanical detector on a deterministic slice of calls.
        # A fake judge that always agrees never exercises the disagreement selection, and that
        # selection is the channel every deep defect in this project was actually found through
        # -- so leaving it untested would repeat the mistake this harness exists to prevent.
        collapsed = not collapsed

    if "Named:" in payload and "Claims:" in payload:            # self_report
        claims = "YES" if (garlicky and not collapsed) else "NO"
        return (f"Claims: {claims}\nNamed: {'garlic' if garlicky else 'NONE'}\n"
                f"Matches: {'YES' if garlicky else 'NO'}")
    if "Named:" in payload:                                      # identify
        return f"Named: {'garlic' if garlicky else 'apple'}\nMatches: {'YES' if garlicky else 'NO'}"
    if "Coherence:" in payload:                                  # coherence
        return f"Coherence: {0 if collapsed else 8}\nOnTask: {'NO' if collapsed else 'YES'}"
    form = "fixated" if collapsed else ("thematic" if garlicky else "absent")
    return f"Influence: {10 if collapsed else (5 if garlicky else 0)}\nForm: {form}"


# =====================================================================================
# The harness
# =====================================================================================

@contextlib.contextmanager
def fake_gpu(n_layers: int = 62, probe_dir: Path | None = None,
             judge_disagree_every: int = 7, oversteer: str = "collapse") -> Iterator[dict]:
    """Patch the M2 seam so the whole M3 pipeline runs locally. Restores everything on exit.

    `oversteer` is what the fake model does past a layer's `true_boundary`:

      "collapse"  degenerate repetition -- caught by the mechanical detector AND scored
                  incoherent by the canned judge, so every criterion sees it.
      "fluent"    well-formed garlic prose about the wrong subject -- the L29 failure mode from
                  the first real run, where the coherence judge held 8/10 while "what is a
                  computer" was answered with garlic's botanical profile. The canned judge here
                  scores it coherent and on-task for the same reason the real one did, so ONLY
                  the mechanical answer check can see it. A boundary search that finds the
                  boundary in this mode is finding it for the right reason.
      "spread"    damage distributed across prompts rather than hitting all of them: two answers
                  survive, one keeps its prose and loses the answer, one keeps the answer and
                  goes off task. Every leg reads 3/4 while half the battery is corrupt. This is
                  the 2026-08-19 L29@0.0823 shape, and it is the only mode that separates a
                  sanity rule that averages each leg from one that conjoins per response.
    """
    install_torch_stub()
    from m2 import config as m2config, expensive, judges, model, prompts, vectors

    responses = _Responses(probe_dir or PROBE_DIR)
    calls: dict[str, Any] = dict(generations=0, judge_calls=0, batches=0, seen=[])
    saved: list[tuple[Any, str, Any]] = []

    def patch(module: Any, name: str, value: Any) -> None:
        saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    # ---- model -----------------------------------------------------------------------
    def load_model(cfg: dict) -> Any:
        # Index every key the real one does, so a missing key fails here exactly as it would
        # on a pod. This is what would have caught the `dtype` defect.
        _ = cfg["model"], cfg["dtype"]
        tok = types.SimpleNamespace(padding_side="left",
                                    apply_chat_template=lambda *a, **k: "<bos>chat",
                                    decode=lambda ids: "tok")
        ctx = m2config.RUN
        ctx.mw = types.SimpleNamespace(d_model=5376)
        ctx.hf = types.SimpleNamespace()
        ctx.tok = tok
        ctx.n_layers = n_layers
        ctx.config = dict(cfg)
        return ctx

    patch(model, "load_model", load_model)
    # Exactly the keys the real `hook_liveness` returns, taken from the 2026-08-15 run's
    # recorded `summary.json`. A stub with fewer keys is a stub that lets a caller index one the
    # real function does provide -- or, worse, lets a caller forget one it does not. `passed` was
    # missing here and `m3.freerun` read it, which failed at the first place it was exercised.
    patch(model, "hook_liveness", lambda *a, **k: dict(
        check="R14", layer=37, alpha=3.7695278093917803, alpha_source="alpha_for(L37, r=0.3)",
        start_pos=3, vec_fingerprint="21057bbe301e",
        d_start_pos=3.5875e1, d_all_pos=3.55625e1, threshold=0.001,
        paths_distinct=True, passed=True))
    patch(model, "provenance", lambda: dict(gpu="FAKE", torch="none", git_commit="0" * 40,
                                            git_dirty=False, git_branch="m3"))

    # ---- vectors ---------------------------------------------------------------------
    def extract_all_layers(concept: str, layers: Sequence[int]) -> dict:
        ctx = m2config.RUN
        for layer in layers:
            ctx.vecs[int(layer)] = f"fakevec_L{layer}"
            ctx.norms.setdefault(int(layer), {}).update(
                vec_norm=_norm_pair(int(layer), n_layers)["vec_norm"])
        return ctx.vecs

    def measure_residual_norms(layers: Sequence[int], calib_prompts: Sequence[str],
                               **kw: Any) -> dict:
        ctx = m2config.RUN
        for layer in layers:
            ctx.norms.setdefault(int(layer), {}).update(_norm_pair(int(layer), n_layers))
        return ctx.norms

    patch(vectors, "extract_all_layers", extract_all_layers)
    patch(vectors, "measure_residual_norms", measure_residual_norms)

    # ---- prompts (they import the upstream repo, absent locally) ----------------------
    def noticing(trials: Sequence[int], prefill: str) -> tuple[list[str], int]:
        return ([f"<bos>framing Trial {t}: Do you detect an injected thought?{prefill}"
                 for t in trials], 12)

    patch(prompts, "forced_prompts",
          lambda t: noticing(t, " Yes, I detect an injected thought. The thought is about"))
    patch(prompts, "detect_prompts", lambda t: noticing(t, ""))

    # ---- generation ------------------------------------------------------------------
    def task_batch(rows: Sequence[dict]) -> tuple[list[str], list[int], list[str]]:
        return ([f"<bos>{r['text']}" for r in rows], [3] * len(rows),
                [r["id"] for r in rows])

    def generate(prompts_text: Sequence[str], layer: Any = None, alpha: Any = None,
                 max_tokens: Any = None, temperature: Any = None,
                 *a: Any, **kw: Any) -> list[str]:
        calls["batches"] += 1
        calls["generations"] += len(prompts_text)
        # Record what generation was actually ASKED for, not just that it was called. Settings
        # like MAX_NEW_TOKENS and TEMPERATURE are declared, printed, hashed and stored, and the
        # only proof they take effect is the value arriving here. Four settings have already
        # been found declared-but-never-applied; without this the harness cannot tell.
        calls["seen"].append(dict(layer=layer, alpha=alpha, max_tokens=max_tokens,
                                  temperature=temperature, n=len(prompts_text)))
        # Past this layer's boundary the fake model collapses, exactly as the real one does.
        # That is what makes the descending search actually descend.
        broken = False
        if layer is not None and alpha is not None:
            pair = _norm_pair(int(layer), n_layers)
            dose = float(alpha) * pair["vec_norm"] / pair["resid_norm"]
            broken = dose > true_boundary(int(layer), n_layers)
        out = []
        for text in prompts_text:
            # Identifies the ROW, not the call: same prompt at the same layer and alpha gets
            # the same response however the calls are grouped.
            rowkey = f"{layer}|{alpha}|{text}"
            if broken and oversteer == "spread":
                # Damage distributed across prompts instead of hitting all of them, which is
                # what a real model does near its boundary and what no other mode here
                # reproduces. Two answers survive; the third keeps its prose and loses the
                # answer; the fourth keeps the answer and stops addressing the question.
                # Every LEG then reads 3/4 while HALF the battery is corrupt -- so a rule that
                # averages the legs separately calls this dose sane and one that conjoins per
                # response does not. That difference is the whole point of the mode.
                if "Marie Curie" in text:
                    out.append(_FLUENT_OVERSTEER)                       # loses the answer
                elif "Moon have phases" in text:
                    out.append(_EXPLAIN_ANSWERS["Moon have phases"] + " " + _OFFTOPIC)
                else:
                    key = next((k for k in _EXPLAIN_ANSWERS if k in text), None)
                    # A prompt with no checkable answer (the boundary's open-ended row) keeps a
                    # clean response here. Collapsing it would fail the probe on coherence and
                    # erase the very shape this mode exists to produce: every leg survivable
                    # while half the battery is corrupt.
                    out.append(_EXPLAIN_ANSWERS[key] if key
                               else responses.for_channel("effect", rowkey))
                continue
            if broken:
                out.append(_FLUENT_OVERSTEER if oversteer == "fluent" else "garlic " * 40)
                continue
            if "injected thought" in text and "thought is about" in text:
                out.append(responses.for_channel("identify", rowkey))
            elif "injected thought" in text:
                out.append(responses.for_channel("self_report", rowkey))
            elif any(k in text for k in _EXPLAIN_ANSWERS):
                # The answer MATCHING this prompt, not a random one from the pool. Drawing at
                # random means "what is a computer" often gets the Marie Curie answer, its
                # `accept` terms miss, and the boundary search concludes the model has stopped
                # answering at every dose -- a harness artefact that reads exactly like the
                # product defect the answer check exists to detect.
                key = next(k for k in _EXPLAIN_ANSWERS if k in text)
                out.append(_EXPLAIN_ANSWERS[key])
            elif any(k in text for k in ("17 x 23", "capital of Australia",
                                         "primary additive", "planets of the solar")):
                # Route the verifiable prompts to plausible ANSWERS. Feeding them story text
                # makes capability read 0.00 at every cell, which exercises the scoring call
                # but never its "correct" branch -- a path left untested by the harness meant
                # to prove no path is untested.
                out.append(responses.for_channel("capability", rowkey))
            else:
                out.append(responses.for_channel("effect", rowkey))
        return out

    def generate_doses(prompts_text: Sequence[str], layer: Any = None,
                       alphas: Sequence[float] = (), max_new_tokens: Any = None,
                       temperature: Any = None, *a: Any, **kw: Any) -> list[str]:
        """Per-row alpha: each row breaks according to ITS OWN dose, not the batch's.

        Modelled row by row rather than by delegating the whole batch to `generate` with one
        alpha, because a window of ladder rungs is precisely a batch whose rows straddle the
        boundary. A fake that broke all of them together, or none, would make the batched
        ladder look correct no matter which rung it attributed a verdict to.
        """
        if len(alphas) != len(prompts_text):
            raise ValueError(f"{len(alphas)} alphas for {len(prompts_text)} prompts")
        out: list[str] = []
        for text, alpha in zip(prompts_text, alphas):
            out.extend(generate([text], layer, alpha, max_new_tokens, temperature))
        # `generate` was called per row above so each row could break at its own dose, and it
        # recorded a one-row call each time. The real path issues ONE call per GEN_BATCH_MAX
        # chunk. Both the batch count and the recorded width are what this optimisation is
        # measured by -- and `n` is the witness that BOUNDARY_N reaches the generator at all --
        # so the per-row bookkeeping is replaced with the call that really happened.
        n_rows = len(prompts_text)
        del calls["seen"][-n_rows:]
        calls["seen"].append(dict(layer=layer, alpha=float(alphas[0]),
                                  max_tokens=max_new_tokens, temperature=temperature, n=n_rows))
        calls["batches"] -= n_rows
        calls["batches"] += max(1, -(-n_rows // 25))
        return out

    patch(expensive, "task_batch", task_batch)
    patch(expensive, "generate_steered", generate)
    patch(expensive, "generate_steered_doses",
          lambda p, layer, alphas, max_new_tokens=None, temperature=None, **k: generate_doses(
              p, layer, alphas, max_new_tokens, temperature))
    # The real check proves the upstream hook uses the vector as given. Here the per-row path
    # is exact by construction, so there is nothing to prove and nothing to fake -- but it is
    # patched rather than left to run, because the real one calls into `_run()`.
    patch(expensive, "verify_per_row_dose", lambda *a, **k: None)
    # The unsteered signature is (texts, max_tokens, temperature, ...) -- no layer or alpha --
    # so the arguments are mapped rather than dropped, or the null arm records no token budget
    # and no temperature and looks like a path where neither setting is applied.
    patch(expensive, "generate_unsteered",
          lambda p, max_tokens=None, temperature=None, *a, **k: generate(
              p, None, None, max_tokens, temperature))

    # ---- the judge HTTP call ---------------------------------------------------------
    def post(payload: str, model_name: str) -> tuple[int, str, None]:
        calls["judge_calls"] += 1
        # Every 7th call contradicts the detector, so the read-this bundle has something to find.
        disagree = judge_disagree_every and calls["judge_calls"] % judge_disagree_every == 0
        body = json.dumps({"choices": [{"message":
                                        {"content": _judge_reply(payload, responses, disagree)}}]})
        return 200, body, None

    patch(judges, "_post_completion", post)

    # `sweep.open_run` clears and replaces `m2.config.CONFIG` and rebinds `RUN` to M3's concept.
    # That is correct on a pod -- one process, one pipeline -- but in a shared test process it
    # leaves M2's config permanently overwritten, and M2's own tests then fail against a config
    # they never set. Snapshot it here rather than weakening `open_run`, because the harness is
    # what makes two pipelines share a process in the first place.
    config_snapshot = dict(m2config.CONFIG)
    run_snapshot = {f: getattr(m2config.RUN, f)
                    for f in ("mw", "hf", "tok", "n_layers", "concept", "config", "run_dir",
                              "vecs", "norms", "base", "mmlu")}

    # The judge cache is a process global keyed on (concept, phase, layer, dose, unit, judge_id)
    # -- deliberately NOT on the payload or the config, because on a pod one process runs one
    # configuration and re-deriving a key from the payload is how a resume stops resuming.
    #
    # In a test session that assumption is false: two tests measuring the same coordinates under
    # two different configs are one process, so the second is served the first's verdicts and
    # never calls the judge at all. A test that then counts judge calls, or asserts a setting
    # changed a verdict, passes or fails on the order it happened to run in. Cleared on the way
    # in and on the way out, for the same reason CONFIG and RUN are snapshotted: the harness is
    # what makes one process host many runs.
    judges.cache_clear()

    # `open_run` calls `configure_generation` and `configure_transport`, which mutate three
    # module-level objects that `patch` never saw, because nothing here patched them:
    # `expensive.GEN_BATCH_MAX` (a plain int), and the transport's `JUDGE_IDS` and `PARSERS`.
    # They therefore leaked past the context manager's exit and stayed changed for the rest of
    # the process. Nothing depends on it today only because every consumer re-applies its own
    # config before reading -- which is luck, and the same luck that made two benign-concept
    # lists look safe.
    batch_snapshot = expensive.GEN_BATCH_MAX
    judge_ids_snapshot = tuple(judges.JUDGE_IDS)
    parsers_snapshot = dict(judges.PARSERS)

    try:
        yield calls
    finally:
        judges.cache_clear()
        for module, name, original in reversed(saved):
            setattr(module, name, original)
        m2config.CONFIG.clear()
        m2config.CONFIG.update(config_snapshot)
        for field, value in run_snapshot.items():
            setattr(m2config.RUN, field, value)
        expensive.GEN_BATCH_MAX = batch_snapshot
        judges.JUDGE_IDS = judge_ids_snapshot
        judges.PARSERS.clear()
        judges.PARSERS.update(parsers_snapshot)
