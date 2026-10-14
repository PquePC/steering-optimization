"""m5.measures.common - what every generated-and-judged measure shares (plan §3.2-§3.4, §7, §8.3).

The measures differ in what they ask and how they score it; they share how they ask. Everything
that, done once per measure, would be done eight slightly different ways lives here:

    Subject        what is measured: the cell (None for the null arm), its vector's class, its
                   concept word. A vector without a class is refused, a harmful concept too
                   (m5.safety, at this entry point as at every other).
    intervention   the one route from a cell to an Intervention (m5.steer.intervention_for), with
                   the cell's position policy honoured (cfg POSITIONS: "from_marker" keeps each
                   channel's own arrangement, "all" and "generated" override it)
    generate       every generation: seeded, sampled with the config's sampler, charged to the
                   budget, written to `transcripts` (the only stream that holds text) and cached
                   per run, so the unsteered arm is generated once per model however many cells
                   are measured against it, and a resumed run regenerates nothing it has
    judge          every judgment, through `ctx.judges` (the cascade records each call in the
                   internal `judge_calls` stream), charged, memoised per run, and counted -- the
                   run aborts loudly when its failed-judgment share is shown to exceed
                   cfg["JUDGE_MISSING_MAX"] (`JudgeFailureRateExceeded`, plan §8.3)
    rows           `trials` rows hold outcomes and ids, never text; `cells` rows hold aggregates;
                   strings in both stay short, because the export gate withholds an aggregate
                   stream that carries long strings (m5.export)

**Seeds.** Every generation's seed is `ctx.seed(family, arm-or-split, cell, unit...)`. The null arm's
seeds name no cell and no split, so one unsteered generation per (prompt, repeat) serves every
cell of every split that uses that prompt. A steered arm's seeds name its split and its cell:
the confirm tier draws fresh samples rather than replaying the search's, and two cells' draws are
independent, which is what every downstream likelihood assumes (the psychometric posterior treats
cells as independent binomials; common random numbers across cells would make it overconfident).
Pairing is by prompt -- the same prompt, steered and unsteered -- not by random number.

**A cell of None is the null arm**, as in `m5.readouts`: the measurement reports the shared
unsteered arm itself -- the same samples every steered cell is paired with -- under the cell id
"unsteered", with its own rate and no null beside it (`as_null_arm`). This is how the setup builds
and reads the null arms once per model. The A/A check of a measure -- does it read nothing when
nothing is done? -- is a zero-dose cell (rho = 0): an arm that is unsteered but drawn with its own
seeds, measured beside the shared null like any cell (d8's d-prime near 0, e8's excess near 0).

**Missing is not zero.** A unit whose judgment failed is excluded from every denominator it would
have entered and counted in `Measured.excluded`, with short reason labels in `extra`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from m5 import steer
from m5.backend.base import GenRequest
from m5.data import influence as _influence
from m5.intervention import Cell, Intervention
from m5.judges.matching import normalise as _normalise_word
from m5.results import Measured
from m5.safety import Refused, check_concept, check_vector_class
from m5.stats import (clopper_pearson_lower, design_effect, icc_from_counts, mean_cluster_size,
                      paired_rate_difference, wilson)

# The splits a measure is asked for. "null" and "control" are the row labels of the unsteered
# arms (a cell of None is the way to measure an arm unsteered), never a split to measure in.
MEASURE_SPLITS = ("search", "confirm", "calibration", "audit")
UNSTEERED = "unsteered"   # cell id of a cell of None (as in m5.readouts)
NULL = "null"             # arm and cell label of the shared unsteered arm
CONTROL = "control"       # arm label of d8's control trials (its null)

# Strings in aggregate rows are kept below the export gate's limit (m5.export, 240 characters).
_ROW_STRING_MAX = 200


class MeasureError(ValueError):
    """A measure was asked for something it cannot do: an unknown split, a variant the channel
    has no prompt for, a concept that contradicts the vector's own."""


class JudgeFailureRateExceeded(RuntimeError):
    """The run's failed-judgment share is above cfg["JUDGE_MISSING_MAX"] (plan §8.3: a run whose
    judge-failure rate exceeds a set limit fails loudly). Nothing of the measurement in progress is
    recorded as measured; the transcripts already written stay, and the replies already paid for
    are in the judge store, so a resumed run re-judges only what failed."""


# =====================================================================================
# What is measured
# =====================================================================================

@dataclass(frozen=True)
class Subject:
    cell: Cell | None
    cell_id: str
    vector_id: str | None
    vector_class: str | None
    concept: str | None

    @property
    def arm(self) -> str:
        """The label of the measured arm: steered, or the null arm itself for a cell of None."""
        return "steered" if self.cell is not None else NULL

    @property
    def null_only(self) -> bool:
        return self.cell is None


def subject(ctx: Any, cell: Cell | None, *, concept: str | None = None) -> Subject:
    """The cell, its vector's class and its concept, checked.

    A steered vector must have a class in the context (fail closed: m5.safety rule 2), and a
    concept passes `check_concept` here as at every entry point. `concept` names the word to look
    for when the cell is None (the null arm has no vector to take it from); for a steered cell it
    may only repeat the vector's own concept.
    """
    if concept is not None:
        check_concept(concept)
    if cell is None:
        return Subject(None, UNSTEERED, None, None, concept)
    if not isinstance(cell, Cell):
        raise TypeError(f"expected a Cell or None, got {type(cell).__name__}")
    vid = cell.vector_id
    cls = ctx.vector_classes.get(vid)
    if cls is None:
        raise Refused(f"vector {vid!r} has no class in this context; every steered vector declares "
                      "one (m5.safety rule 2, fail closed)")
    check_vector_class(cls)
    word = ctx.concept_of(vid)
    if word is not None:
        check_concept(word)
    if concept is not None and (word is None or _normalise_word(concept) != _normalise_word(word)):
        raise MeasureError(f"concept {concept!r} is not the concept of vector {vid!r} ({word!r}); "
                           "a steered cell is scored for its own concept only")
    return Subject(cell, cell.cell_id(), vid, cls, word)


def check_split(split: str) -> str:
    if split not in MEASURE_SPLITS:
        raise MeasureError(f"unknown measure split {split!r}; one of {MEASURE_SPLITS} (the unsteered "
                           "arm is measured with cell=None, not with a 'null' split)")
    return split


def check_n(n: Any, what: str = "n") -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise MeasureError(f"{what} must be a positive integer, got {n!r}")
    return int(n)


def check_start(start: Any) -> int:
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or int(start) < 0:
        raise MeasureError(f"start must be a non-negative integer, got {start!r}")
    return int(start)


def check_variant(code: str, variant: str, allowed: Sequence[str] = ("original",)) -> str:
    """Channels without prompt variants accept only "original": a variant name on a row whose
    prompt did not change would be a mislabelled measurement."""
    if variant not in allowed:
        raise MeasureError(f"{code} has no prompt variant {variant!r}; one of {tuple(allowed)}")
    return variant


def influence_prompts(split: str) -> tuple[_influence.OpenPrompt, ...]:
    """The open-ended prompt set of a split: the held-out set for "confirm", the search set
    otherwise (plan §6.5: search and confirmation prompts never overlap)."""
    check_split(split)
    return _influence.HELDOUT_PROMPTS if split == "confirm" else _influence.SEARCH_PROMPTS


def cycle(seq: Sequence[Any], start: int, n: int) -> list[tuple[Any, int]]:
    """Units start..start+n-1 over a sequence, as (item, repeat): item i % len, repeat i // len."""
    if not seq:
        raise MeasureError("an empty set has no units")
    m = len(seq)
    return [(seq[i % m], i // m) for i in range(start, start + n)]


# =====================================================================================
# Per-run state: caches and counters that outlive one measure call
# =====================================================================================

@dataclass(frozen=True)
class Gen:
    """One generation as the measures see it."""

    transcript: str
    text: str
    finish: str
    n_tokens: int


class MeasureState:
    """Caches and counters for one Context (one run), kept in `ctx.cache` under this package's
    own key (`CACHE_KEY`), so they live and die with the context and touch nothing else on it."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.gens: dict[str, Gen] = {}
        self.loaded_run = False
        self.judgments: dict[str, Any] = {}
        self.judged: Counter[str] = Counter()
        self.failed: Counter[str] = Counter()
        self.rows_written: set[tuple] = set()
        self.memo: dict[tuple, Any] = {}
        self.specs: dict[tuple, Any] = {}
        self.interventions: dict[tuple, Intervention | None] = {}
        self.logits: dict[str, np.ndarray] = {}
        self.hidden: dict[str, np.ndarray] = {}
        self.profile_key: str | None = None

    def load_run(self, ctx: Any) -> None:
        """Index the run's transcripts once, so a resumed run (or a second process on the same
        run) reuses every generation it already has, the null arm above all."""
        with self.lock:
            if self.loaded_run:
                return
            self.loaded_run = True
            if ctx.run is None:
                return
            for row in ctx.run.rows("transcripts"):
                key = row.get("gen_key")
                if key and key not in self.gens and isinstance(row.get("text"), str):
                    self.gens[key] = Gen(str(row["transcript_id"]), row["text"],
                                         str(row.get("finish")), int(row.get("n_tokens") or 0))


CACHE_KEY = "m5.measures"
_STATE_LOCK = threading.Lock()


def state(ctx: Any) -> MeasureState:
    """This package's per-run memo: `ctx.cache["m5.measures"]` (the context's namespaced cache)."""
    st = ctx.cache.get(CACHE_KEY)
    if st is None:
        with _STATE_LOCK:
            st = ctx.cache.get(CACHE_KEY)
            if st is None:
                st = ctx.cache[CACHE_KEY] = MeasureState()
    if not isinstance(st, MeasureState):
        raise TypeError(f"ctx.cache[{CACHE_KEY!r}] holds a {type(st).__name__}, not this package's state")
    return st


def sha256_file(path: str | Path) -> str:
    """The sha256 of a file, read in blocks: what a profile selection names a stored file by."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def profile_key(ctx: Any) -> str | None:
    st = state(ctx)
    if st.profile_key is None and ctx.profile is not None and hasattr(ctx.profile, "key"):
        st.profile_key = ctx.profile.key()
    return st.profile_key


def spec(ctx: Any, key: tuple, build: Any) -> Any:
    """A prompt spec, built once per run: rendering and tokenising are pure functions of the
    builder's arguments, and the steering start of every trial depends on them."""
    st = state(ctx)
    with st.lock:
        got = st.specs.get(key)
    if got is None:
        got = build()
        with st.lock:
            st.specs[key] = got
    return got


# =====================================================================================
# Steering
# =====================================================================================

def mode_for(cell: Cell | None, default: str) -> str:
    """The position mode for one prompt. "from_marker" (the default policy) keeps the channel's
    own arrangement -- live from "Trial n" in the detection conversation, every position of a
    single-turn prompt -- while "all" and "generated" apply as named to every channel."""
    if cell is None or cell.positions == "from_marker":
        return default
    if cell.positions in ("all", "generated"):
        return cell.positions
    raise MeasureError(f"{cell.cell_id()}: unknown position policy {cell.positions!r}")


def intervention(ctx: Any, cell: Cell | None, prompt: Any, default_mode: str) -> Intervention | None:
    """The cell on this prompt, through `m5.steer.intervention_for` (the only route from a cell to
    alphas), cached per (cell, positions): prompts that share their positions share the object."""
    if cell is None:
        return None
    mode = mode_for(cell, default_mode)
    positions = prompt.positions(mode)
    key = (cell.cell_id(), cell.tags, json.dumps(positions.to_json(), sort_keys=True))
    st = state(ctx)
    with st.lock:
        if key in st.interventions:
            return st.interventions[key]
    iv = steer.intervention_for(ctx, cell, prompt, mode)
    with st.lock:
        st.interventions[key] = iv
    return iv


def intervention_terms(iv: Intervention | None) -> list[list[float]] | None:
    """The alphas behind a measurement, for its cells row (provenance; no vector values)."""
    return None if iv is None else [[t.layer, t.alpha] for t in iv.terms]


# =====================================================================================
# Generation
# =====================================================================================

@dataclass(frozen=True)
class Job:
    """One generation to make: a unit id, the prompt, the steering, the seed and the length."""

    unit: str
    prompt: Any
    intervention: Intervention | None
    seed: int
    max_new_tokens: int


def gen_key(cfg: Any, job: Job) -> str:
    """Content address of a generation: everything that decides its text. Two requests with the
    same key are the same request, whichever measure or arm asked."""
    ident = ["m5.measures.gen/1", job.prompt.text,
             job.intervention.key() if job.intervention is not None else None,
             int(job.max_new_tokens), float(cfg["TEMPERATURE"]), float(cfg["TOP_P"]),
             int(cfg["TOP_K"]), int(job.seed)]
    return hashlib.sha256(json.dumps(ident, ensure_ascii=False).encode("utf-8")).hexdigest()


def generate(ctx: Any, jobs: Sequence[Job], *, code: str, arm: str, cell_id: str,
             split: str) -> list[Gen]:
    """Generate every job not already in the run, in batches of GEN_BATCH_MAX, charging each
    batch before it runs (a capped run stops before spending, never after). Each new text is
    written to `transcripts` under a transcript id derived from its content address."""
    cfg = ctx.cfg
    st = state(ctx)
    st.load_run(ctx)
    keys = [gen_key(cfg, j) for j in jobs]
    todo: dict[str, int] = {}
    with st.lock:
        for i, k in enumerate(keys):
            if k not in st.gens and k not in todo:
                todo[k] = i
    order = list(todo.items())
    batch = int(cfg["GEN_BATCH_MAX"])
    for b in range(0, len(order), batch):
        chunk = order[b:b + batch]
        reqs = [GenRequest(prompt=jobs[i].prompt.text, intervention=jobs[i].intervention,
                           max_new_tokens=jobs[i].max_new_tokens, temperature=cfg["TEMPERATURE"],
                           top_p=cfg["TOP_P"], top_k=cfg["TOP_K"], seed=jobs[i].seed)
                for _, i in chunk]
        ctx.charge(generations=len(reqs))
        results = ctx.backend.generate(reqs)
        if len(results) != len(reqs):
            raise RuntimeError(f"the backend returned {len(results)} generations for {len(reqs)}")
        ctx.charge(tokens=sum(len(r.token_ids) for r in results))
        for (k, i), r in zip(chunk, results):
            g = Gen("t" + k[:24], r.text, r.finish, len(r.token_ids))
            job = jobs[i]
            ctx.write("transcripts", {
                "transcript_id": g.transcript, "code": code, "arm": arm, "cell": cell_id, "split": split,
                "unit": job.unit, "channel": job.prompt.channel, "seed": job.seed,
                "max_new_tokens": job.max_new_tokens, "gen_key": k, "finish": r.finish,
                "n_tokens": g.n_tokens, "text": r.text})
            with st.lock:
                st.gens[k] = g
    with st.lock:
        return [st.gens[k] for k in keys]


# =====================================================================================
# Judging
# =====================================================================================

def require_judges(ctx: Any) -> None:
    """Refuse before generating anything: a measure that cannot judge would spend its generations
    and then fail."""
    if ctx.judges is None:
        raise MeasureError("this measure needs judges (ctx.judges is None); wire a cascade, "
                           "m5.judges.Cascade, into the context")


def judge(ctx: Any, items: Sequence[Any]) -> list[Any]:
    """Judge items through `ctx.judges` (normally `m5.judges.Cascade`), in input order.

    Decided judgments are memoised per run by item key (the same question about the same text is
    the same item): the null arm's texts are judged once per concept, not once per cell, and the
    `judge_calls` stream records each judgment once. Failed judgments are not memoised; asking
    again retries them, as the judge store's cache does (only verdicts are cached).
    """
    if not items:
        return []
    require_judges(ctx)
    st = state(ctx)
    out: list[Any] = [None] * len(items)
    todo: dict[str, list[int]] = {}
    with st.lock:
        for i, it in enumerate(items):
            hit = st.judgments.get(it.key)
            if hit is not None:
                out[i] = hit
            else:
                todo.setdefault(it.key, []).append(i)
    if todo:
        firsts = [idx[0] for idx in todo.values()]
        fresh = ctx.judges.judge([items[i] for i in firsts])
        if len(fresh) != len(firsts):
            raise RuntimeError(f"the judges returned {len(fresh)} judgments for {len(firsts)} items")
        for (key, idx), j in zip(todo.items(), fresh):
            if getattr(j, "item_key", key) != key:
                raise RuntimeError("a judgment came back in another item's place (results out of order)")
            for i in idx:
                out[i] = j
        _account(ctx, st, fresh)
        with st.lock:
            for key, j in zip(todo, fresh):
                if j.missing is None:
                    st.judgments[key] = j
    return out


def _model_outcome(j: Any) -> bool:
    """A missing judgment that reports the model, not the judge: tier 0 found nothing to score
    (an empty reply on a scale template). It is excluded like any missing unit but is not a judge
    failure, so it does not count toward JUDGE_MISSING_MAX."""
    return j.missing is not None and j.stop == "t0_mechanical"


def _account(ctx: Any, st: MeasureState, judgments: Sequence[Any]) -> None:
    local = external = 0
    with st.lock:
        for j in judgments:
            fresh = [s for s in j.trail if s.called and not s.cached]
            if any(s.external for s in fresh):
                external += 1
            elif fresh:
                local += 1
            st.judged[j.template_id] += 1
            if j.missing is not None and not _model_outcome(j):
                st.failed[j.template_id] += 1
    ctx.charge(judged_local=local, judged_external=external)
    check_failure_rate(ctx)


def check_failure_rate(ctx: Any) -> None:
    """Abort when the run's failed-judgment share is shown to exceed JUDGE_MISSING_MAX.

    Shown, not merely observed: the one-sided exact (Clopper-Pearson) lower bound at CONFIDENCE
    on the cumulative share must exceed the limit, overall or for any one template (a broken
    template is caught even when the others dilute it). Read as a point estimate, the rule would
    end a healthy run on its first unlucky judgment (one failure among the first 19 already reads
    above 5%); read this way it ends a broken one within a handful (two failures in the first
    three), and a run whose true share sits below the limit is not stopped.
    """
    st = state(ctx)
    limit = float(ctx.cfg["JUDGE_MISSING_MAX"])
    conf = float(ctx.cfg["CONFIDENCE"])
    with st.lock:
        scopes = [("all", sum(st.failed.values()), sum(st.judged.values()))]
        scopes += [(t, st.failed[t], st.judged[t]) for t in sorted(st.judged)]
    for scope, k, n in scopes:
        if k == 0 or n == 0:
            continue
        lb = clopper_pearson_lower(k, n, conf)
        if lb > limit:
            if ctx.run is not None:
                ctx.run.event("judging", "aborted", scope=scope, failed=int(k), judged=int(n),
                              lower_bound=lb, limit=limit)
            raise JudgeFailureRateExceeded(
                f"{k} of {n} {scope} judgments failed; the lower {conf:.0%} bound on the failed "
                f"share ({lb:.3f}) exceeds JUDGE_MISSING_MAX={limit}. A run that cannot judge its "
                "own output fails loudly (plan §8.3) rather than reporting rates over the judgments "
                "that happened to succeed.")


def failure_counts(ctx: Any) -> dict[str, dict[str, int]]:
    """The run's judgments and failures so far, per template."""
    st = state(ctx)
    with st.lock:
        return {t: {"judged": int(st.judged[t]), "failed": int(st.failed[t])} for t in sorted(st.judged)}


def reason(j: Any) -> str:
    """A short label for a failed judgment: template, then the first two words of its reason
    ("allusion:undecided:t1_missing"). Labels, not prose, so they can be counted and stay short."""
    words = re.findall(r"[a-z0-9_]+", str(j.missing).lower())[:2]
    return ":".join([j.template_id, *words])


def label_of(j: Any) -> Any:
    return None if j is None or j.missing is not None else j.label


# =====================================================================================
# Rows
# =====================================================================================

def short(x: Any) -> Any:
    """Keep strings in aggregate rows below the export gate's limit."""
    if isinstance(x, str) and len(x) > _ROW_STRING_MAX:
        return x[:_ROW_STRING_MAX - 3] + "..."
    return x


def write_trials(ctx: Any, rows: Iterable[Mapping[str, Any]]) -> None:
    for r in rows:
        ctx.write("trials", {k: short(v) for k, v in r.items()})


def write_null_trials(ctx: Any, rows: Iterable[tuple[tuple, Mapping[str, Any]]]) -> None:
    """Rows of a shared unsteered arm are written once per run and identity (code, concept,
    unit): the arm serves every cell of its vector, and a row per cell would repeat it."""
    st = state(ctx)
    fresh = []
    with st.lock:
        for key, row in rows:
            if key not in st.rows_written:
                st.rows_written.add(key)
                fresh.append(row)
    write_trials(ctx, fresh)


def write_cell(ctx: Any, m: Measured, *, subj: Subject, iv: Intervention | None = None,
               extra: Mapping[str, Any] | None = None) -> Measured:
    """One `cells` row per Measured, missing ones included (a cell that was tried and failed is
    not a cell that was never tried)."""
    row = m.to_row()
    row.update({"tier": "measure" if m.split != "confirm" else "confirm", "arm": subj.arm,
                "vector": subj.vector_id, "vector_class": subj.vector_class,
                "cell_json": None if subj.cell is None else subj.cell.to_json(),
                "alphas": intervention_terms(iv), "profile": profile_key(ctx),
                "missing": short(m.missing)})
    if extra:
        row.update(extra)
    ctx.write("cells", row)
    return m


_PAIRED_KEYS = ("paired", "null", "null_values", "shift", "p_mcnemar", "vs_null", "steered", "sdt",
                "control", "null_mechanical", "null_kinds", "null_prefilter", "null_words", "value")


def as_null_arm(m: Measured, conf: float) -> Measured:
    """A measurement of the null arm as itself (a cell of None): its own rate with its Wilson
    interval, and nothing that compares it with a null -- it is the null."""
    extra = {k: v for k, v in m.extra.items() if k not in _PAIRED_KEYS}
    extra["arm"] = "null: the shared unsteered arm itself; no null beside it"
    rate, lo, hi = m.rate, m.lo, m.hi
    if m.k is not None and m.n > 0:
        rate = m.k / m.n
        lo, hi = wilson_or_none(m.k, m.n, conf)
    return Measured(code=m.code, cell_id=m.cell_id, split=m.split, n=m.n, k=m.k, rate=rate, lo=lo,
                    hi=hi, values=m.values, null_k=None, null_n=None, extra=extra,
                    unit_ids=list(m.unit_ids), missing=m.missing, excluded=m.excluded)


def missing_measure(ctx: Any, code: str, subj: Subject, split: str, why: str, *,
                    excluded: int = 0, extra: Mapping[str, Any] | None = None) -> Measured:
    m = Measured(code=code, cell_id=subj.cell_id, split=split, n=0, missing=short(why),
                 excluded=excluded, extra=dict(extra or {}))
    return write_cell(ctx, m, subj=subj)


# =====================================================================================
# Statistics shared by the measures
# =====================================================================================

def wilson_or_none(k: int, n: int, conf: float) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    lo, hi = wilson(k, n, conf)
    return float(lo), float(hi)


def rate_block(k: int, n: int, conf: float) -> dict[str, Any]:
    lo, hi = wilson_or_none(k, n, conf)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "lo": lo, "hi": hi}


def paired(pairs: Sequence[tuple[bool, bool]], conf: float) -> dict[str, Any] | None:
    """Steered minus unsteered on the same units: Tango's score interval and the exact McNemar
    p-value (m5.stats.paired). None when no unit has both arms decided."""
    if not pairs:
        return None
    arr = np.array([[1.0 if a else 0.0, 1.0 if b else 0.0] for a, b in pairs])
    r = paired_rate_difference(arr, conf)
    return {"diff": r["diff"], "lo": r["lo"], "hi": r["hi"], "p_mcnemar": r["p_mcnemar"],
            "n": r["n"], "b": r["b"], "c": r["c"], "method": r["method"]}


def bootstrap_mean(values: Sequence[float], *, conf: float, reps: int,
                   seed: int) -> tuple[float, float | None, float | None]:
    """(mean, lo, hi): the percentile bootstrap over units (Efron 1979), each unit its own
    cluster -- the same interval `m5.stats.cluster_bootstrap` gives with one unit per cluster,
    vectorised. Paired differences are passed in as one value per unit, so the pairing is kept.
    With fewer than two units there is no interval (None, not a zero-width one)."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        raise MeasureError("no values to average")
    mean = float(x.mean())
    if x.size < 2:
        return mean, None, None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(int(reps), x.size))].mean(axis=1)
    alpha = 1.0 - conf
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean, float(lo), float(hi)


def design(clusters: Sequence[Any], outcomes: Sequence[int]) -> dict[str, Any]:
    """The measurement's own design effect, 1 + (m - 1) ICC (Kish), for `extra["deff"]`, where the
    confirm tier reads it (plan §6.4: trials are clustered by prompt).

    One unit per cluster (n prompts at most once each) cannot be clustered: deff is 1. An ICC that
    the data cannot identify -- every outcome identical, or a single cluster -- leaves deff None
    (missing, not 1: the reader decides what to assume, and says so)."""
    by: dict[Any, list[int]] = {}
    for c, y in zip(clusters, outcomes):
        by.setdefault(c, []).append(int(y))
    sizes = [len(v) for v in by.values()]
    if not sizes:
        return {"deff": None, "icc": None, "cluster_size": None, "deff_source": "no units"}
    if max(sizes) == 1:
        return {"deff": 1.0, "icc": None, "cluster_size": 1.0,
                "deff_source": "one unit per cluster: nothing to cluster"}
    m = mean_cluster_size(sizes)
    if len(by) < 2:
        return {"deff": None, "icc": None, "cluster_size": m, "deff_source": "a single cluster"}
    icc = icc_from_counts([sum(v) for v in by.values()], sizes)
    if icc is None:
        return {"deff": None, "icc": None, "cluster_size": m,
                "deff_source": "ICC undefined: every outcome identical"}
    return {"deff": design_effect(m, icc), "icc": icc, "cluster_size": m,
            "deff_source": "ANOVA ICC over this measurement's clusters"}


def counter_dict(c: Counter, *, top: int | None = None) -> dict[str, int]:
    items = c.most_common(top) if top else sorted(c.items())
    return {short(str(k)): int(v) for k, v in items}


def finite(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


@dataclass
class Tally:
    """Missing units and their reasons, for `Measured.excluded` and `extra`."""

    reasons: Counter = field(default_factory=Counter)
    units: set = field(default_factory=set)

    def add(self, unit: str, why: str) -> None:
        self.units.add(unit)
        self.reasons[why] += 1

    @property
    def count(self) -> int:
        return len(self.units)

    def to_json(self) -> dict[str, int]:
        return counter_dict(self.reasons)
