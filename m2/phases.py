"""m2.phases - the screening procedure of spec section 8, and the selection rule of section 7.

Phase 0 calibrates, Phase 1 scans every layer cheaply, Phase 2 selects cells on the cheap
reach-vs-d3 Pareto frontier, Phase 3 maps each cell's sane range, Phases 4 and 5 pay for real
E5/S1/D2, and
Phase 6 re-measures the winner on held-out prompts.

  phase0_calibrate       spec 5.1 - vectors, norms, dose map, baselines, cap_base, 2 judge calls
  phase1_scan            every layer with d(L) >= D_MIN, at both SCAN_DOSES. Zero judge calls
  phase2_shortlist       eligible cells, then the reach-vs-d3 Pareto frontier
  phase3_bisect          preserve the selected dose while mapping its sanity boundary
  phase4_verify          expensive.verify_cell on the shortlist
  phase5_refine          layer +/-1, +/-2 and one dose step either side, around the top cells
  phase6_confirm         the winner on prompts.E5_HELDOUT at N_CONFIRM, no adaptive stopping
  select_operating_point spec 7.1 EXACTLY: argmax(E5) over qualifying cells
  frontier               every qualifying cell, because one point discards the trade-off
  covertness_margin      d2 - predicted_d2(e5). REPORTED, never selected on

**Phases 1-5 are screening.** They decide what gets measured and in what order; their numbers
rank cells and are not reportable. Only Phase 6 output is (spec 8). Two things follow from
that and are enforced here rather than remembered: Phase 6 measures on a prompt set disjoint
from the one the winner was chosen on, and every screening row is written with its own phase
label so a Phase 4 screening number can never be read as a confirmation.

**Selection never touches a fitted residual.** Phase 2 uses the measured cell coordinates
directly: more reach is better and less d3 is better. `covertness_margin` remains a reported
expensive-tier diagnostic; `select_operating_point` implements spec 7.1 and nothing else.

Import order (CONTRACT section 1): this module may use config, model, vectors, prompts, cheap,
judges and expensive, and nothing later. `vectors` and `expensive` are imported LAZILY inside
the functions that need them, because both import torch at module scope; deferring them keeps
`import m2.phases` working in an offline checkout, so tests/test_offline.py can exercise the
pure parts (smooth, local_maxima, ols_fit, select_operating_point, frontier,
covertness_margin) with no GPU stack installed. `runio` comes later in the order, so this
module appends its own JSONL in the CONTRACT section 4 row shape rather than reusing
`runio.write_row`.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import cheap
from . import config
from . import model
from . import prompts

__all__ = [
    # CONTRACT section 3 surface
    "phase0_calibrate",
    "phase1_scan",
    "phase2_shortlist",
    "phase3_bisect",
    "phase4_verify",
    "phase5_refine",
    "phase6_confirm",
    "select_operating_point",
    "reselect_operating_point",
    "frontier",
    "covertness_margin",
    # additions, each documented at its definition
    "write_operating_point",
    "layers_in_scope",
    "reference_layer",
    "smooth",
    "local_maxima",
    "ols_fit",
    "predict",
    "stratified_pick",
    "SCAN_FILE", "SHORTLIST_FILE", "BISECT_FILE", "VERIFIED_FILE", "CONFIRM_FILE",
    "TIER_VERIFICATION_FILE",
    "BASELINES_FILE", "UNSTEERED_DIR", "OPERATING_POINT_FILE",
    "BASE_P_KEY", "BASE_D3_KEY",
    "PHASE0", "PHASE1", "PHASE3", "PHASE4", "PHASE5", "PHASE6",
]


# =====================================================================================
# Artefact names (spec 13) and phase labels
# =====================================================================================

SCAN_FILE: str = "scan.jsonl"
SHORTLIST_FILE: str = "shortlist.json"
BISECT_FILE: str = "bisect.jsonl"
VERIFIED_FILE: str = "verified.jsonl"
CONFIRM_FILE: str = "confirm.jsonl"
BASELINES_FILE: str = "baselines.jsonl"
UNSTEERED_DIR: str = "unsteered"
OPERATING_POINT_FILE: str = "operating_point.json"
TIER_VERIFICATION_FILE: str = "tier_verification.json"

# The phase label stamped on every row and, via `expensive.phase_scope`, on every judge call.
# It is what keeps a Phase 4 screening E5 distinguishable from the Phase 6 confirmation of the
# same cell in judge_e5.jsonl - the two are the same (layer, r) and only this separates them.
PHASE0: str = "CAL"
PHASE1: str = "SCAN"
PHASE2: str = "SHORTLIST"
PHASE3: str = "BISECT"
PHASE4: str = "PHASE4"
PHASE5: str = "PHASE5"
PHASE6: str = "CONFIRM"

# Where Phase 0's spec 5.1 step 6 baselines live on RUN.base. Named once here so a typo is an
# AttributeError at import rather than a KeyError at the first steered cell.
BASE_P_KEY: str = "p_base"
BASE_D3_KEY: str = "d3_base"

# Macar's reference depth. `int(n_layers * fraction)` is the repo's own `get_layer_at_fraction`
# (bug 11: 0.35 * 62 = 21, not 22), so the same arithmetic is used here rather than round().
REF_DEPTH: float = 0.60

# Phase 2 knobs. Not in spec section 11 because they parameterise the SEARCH, not the
# measurement: changing one changes which cells get measured, never what a measurement means.
SMOOTH_WINDOW: int = 3          # "lightly smoothed over L" (spec 8 Phase 2)
STRATIFIED_BINS: int = 4        # coverage across the E6 range, spec 8 Phase 2 route 2
RESID_SIGMA: float = 1.0        # a layer is "low D3 for its E6" at 1 SD below the fit
RESID_MAX_CANDIDATES: int = 4   # bound on route 3, so it cannot swamp the shortlist
# Floor under "this layer produced any concept mass at all", used only to decide whether a
# layer is worth WIDENING onto the shortlist. It never gates a measurement or a selection.
# D3 is a probability mass and is never exactly zero, so the test has to be against something;
# the unsteered baseline is the honest comparator and this is the floor under it.
D3_SIGNAL_MIN: float = 0.005
TIER_ORDERINGS: frozenset[str] = frozenset({"e6_desc", "e6_residual_interleave"})

# Phase 3 knobs.
BISECT_ESCALATE: float = 1.5    # dose multiplier while hunting for the insane end of the bracket
BISECT_MAX_PROBES: int = 6      # bound on that hunt; each probe is a cheap cell, not free
BISECT_MIN_R: float = 0.01      # floor on the sane end; below this nothing is being injected

# Phase 5: the dose step used when Phase 3 did not supply one (spec 8 Phase 5, "one dose step
# either side"). A fraction of r rather than an absolute, because r spans an order of magnitude
# across layers and a fixed step would be a different experiment at each end.
DOSE_STEP_FRAC: float = 0.15


# =====================================================================================
# Run context, config, and JSONL plumbing
# =====================================================================================

def _run() -> Any:
    """The process-global RunContext (CONTRACT section 2), or a loud error.

    Read through this accessor rather than `from .config import RUN`: `model.load_model`
    rebinds the attribute on the config module, and a from-import would capture the pre-load
    placeholder forever. Same family as bug 23 - a stale reference that returns something
    plausible instead of raising.
    """
    ctx = getattr(config, "RUN", None)
    if ctx is None or ctx.mw is None:
        raise RuntimeError(
            "m2.config.RUN is not loaded - call m2.model.load_model(CONFIG) and "
            "m2.driver.set_concept(name) before running any phase.")
    return ctx


def _cfg() -> dict:
    """The live configuration dict.

    `RUN.config` when a run is set up - that is the dict `config_hash` was taken over, so it is
    what every row is labelled with - otherwise the module-level `config.CONFIG`. Callers hard-
    index the key they want off this, so a missing constant raises rather than becoming a
    threshold nobody chose (DEBUG LOG pattern 4).
    """
    ctx = getattr(config, "RUN", None)
    if ctx is not None and ctx.config:
        return ctx.config
    return config.CONFIG


def _now() -> str:
    """UTC timestamp for the `ts` field every row carries (CONTRACT section 4)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp(row: dict) -> dict:
    """Add concept / config_hash / ts. Hard indexing: an unidentified row records nothing."""
    ctx = _run()
    out = dict(row)
    out["concept"] = ctx.concept
    out["config_hash"] = ctx.config["config_hash"]
    out["ts"] = _now()
    return out


def _path(name: str) -> Path:
    """`run_dir/<name>`, with `name` allowed to carry a subdirectory (`unsteered/e5_01.jsonl`)."""
    ctx = _run()
    if ctx.run_dir is None:
        raise RuntimeError(
            "RUN.run_dir is not set - call m2.driver.set_concept(name) before any phase. "
            "Writing rows into the process working directory is how two concepts' rows end "
            "up in one file.")
    return Path(ctx.run_dir) / name


def _append_row(name: str, row: dict) -> Path:
    """Append one JSON object to `run_dir/<name>`.

    `runio.write_row` does exactly this, but `runio` comes AFTER this module in the CONTRACT
    dependency order, so - as `vectors.py` and `expensive.py` already do - the row shape is
    matched rather than the function reused. `default=str` so a stray Path or tensor in a
    diagnostic field cannot lose a whole row to a serialisation error; every load-bearing
    field is a primitive by construction.
    """
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_stamp(row), ensure_ascii=False, default=str) + "\n")
    return path


def _write_json(name: str, payload: dict) -> Path:
    """Write one whole JSON document, via a temp file and os.replace.

    Used for `shortlist.json` and `operating_point.json`, which are complete snapshots rather
    than append-as-produced logs. Atomic because a crash mid-write leaves a truncated document
    that every reader downstream fails to parse - and unlike a JSONL line, there is no earlier
    good half to fall back to.
    """
    import os

    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_stamp(payload), ensure_ascii=False, indent=2, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _read_rows(name: str) -> list[dict]:
    """Every row of `run_dir/<name>`, or `[]` when the file does not exist yet.

    A missing file is a legitimate state - the phase has not run - and is distinct from a
    corrupt one. A line that does not parse RAISES rather than being skipped: a resume that
    silently dropped rows would re-measure some cells and, worse, would compute Phase 2's fit
    over a quietly truncated surface. The message names the file and the line so the operator
    can delete it and force a clean re-run.
    """
    path = _path(name)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{lineno} is not valid JSON ({exc}). A partially written row must "
                    "not be silently dropped from a resume set - delete the file to re-measure "
                    "this phase from scratch.") from exc
    return rows


def _cell_key(layer: Any, r: Any) -> tuple:
    """The canonical (layer, r) identity, rounded exactly as `vectors.dose_key` rounds.

    Doses are keys and 0.1 + 0.05 != 0.15 in binary floating point, so a caller that arrives
    at the same dose by a different arithmetic route must still hit the same cell. The
    rounding constant is duplicated rather than imported because `vectors` pulls in torch and
    this module stays importable offline; `_check_r_decimals` in judges.py guards the same
    agreement for the judge cache.
    """
    return (int(layer), round(float(r), 6))


# =====================================================================================
# Pure helpers - offline-testable, no torch, no RUN
# =====================================================================================

def layers_in_scope(n_layers: int, d_min: float) -> list[int]:
    """Every layer with relative depth `d(L) = L / n_layers >= d_min` (spec 3, spec 8 Phase 1).

    Pure so the layer set can be checked without a model. `D_MIN` is load-bearing and gate 9
    re-tests it every run, which is only possible because Phase 1 scans the whole range above
    it rather than a sample.
    """
    n_layers = int(n_layers)
    if n_layers <= 0:
        raise ValueError(f"n_layers={n_layers}: no layers to scan")
    return [layer for layer in range(n_layers) if layer / n_layers >= float(d_min)]


def reference_layer(n_layers: int) -> int:
    """Macar's reference layer for this model: `int(n_layers * 0.60)` -> L37 on Gemma3-27B.

    `int()`, not `round()`. Bug 11: the repo's `get_layer_at_fraction` truncates, and a
    reference layer that disagrees with the repo's by one would compare our escalation ladder
    and R5 norm check against a different layer from the published figures.
    """
    return int(int(n_layers) * REF_DEPTH)


def smooth(values: Sequence[float | None], window: int = SMOOTH_WINDOW) -> list[float | None]:
    """Centred moving average over the layer axis, ignoring `None` (unreachable) entries.

    Spec 8 Phase 2 selects "from the Phase 1 surface (lightly smoothed over L)". Lightly: a
    3-point mean, enough to stop single-cell judge-free noise from inventing a local maximum,
    not enough to move a real peak - the peaks this is looking for are one to two layers wide.

    A position whose whole window is unreachable stays `None`. It is NOT filled with a
    neighbour's value: an unreachable cell was never measured, and a smoother that invents a
    number for it would put an unmeasured layer on the shortlist (DEBUG LOG pattern 4).
    """
    values = list(values)
    half = int(window) // 2
    out: list[float | None] = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        chunk = [float(v) for v in values[lo:hi] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def local_maxima(values: Sequence[float | None]) -> list[int]:
    """Indices that are strictly higher than the nearest DIFFERENT value on both sides.

    "Nearest different" rather than "immediate neighbour" so a plateau is not silently
    discarded: on a flat top every member of the plateau is returned and Phase 2's +/-1 merge
    collapses them to one candidate. Endpoints count - the highest-reach layer can perfectly
    well be the last one in scope, and dropping it would systematically ignore the deep end
    where effectiveness peaked for every M1.5 concept.

    Pure: takes the already-smoothed reach values, returns positions in that list.
    """
    pts = [(i, float(v)) for i, v in enumerate(values) if v is not None]
    out: list[int] = []
    for k, (i, v) in enumerate(pts):
        left = next((w for _, w in reversed(pts[:k]) if w != v), None)
        right = next((w for _, w in pts[k + 1:] if w != v), None)
        if (left is None or v > left) and (right is None or v > right):
            out.append(i)
    return out


def ols_fit(xs: Sequence[float], ys: Sequence[float]) -> dict | None:
    """Least-squares `y = slope*x + intercept`, or `None` when the fit is not defined.

    `None` for fewer than three points or zero variance in x. It is a "not computed", never a
    zero slope: a defaulted flat fit would make every residual equal to `y - mean(y)`, which
    looks like a residual and is not one (DEBUG LOG pattern 4). Callers must treat `None` as
    "this route is unavailable" and say so in their output.
    """
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"ols_fit: {n} xs against {len(ys)} ys")
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rss = sum(e * e for e in resid)
    # Population SD of the residuals - the scale the "low for its E6" test is expressed in.
    resid_sd = math.sqrt(rss / n) if n else None
    syy = sum((y - my) ** 2 for y in ys)
    return dict(slope=slope, intercept=intercept, n=n, x_mean=mx, y_mean=my,
                resid_sd=resid_sd, r2=(1.0 - rss / syy) if syy > 0 else None)


def predict(fit: dict, x: float) -> float:
    """`fit`'s prediction at `x`. Hard-indexed: a fit without a slope is not a fit."""
    return float(fit["slope"]) * float(x) + float(fit["intercept"])


def stratified_pick(items: Sequence[dict], key: str, n_bins: int = STRATIFIED_BINS) -> list[dict]:
    """One representative per equal-width bin across the observed range of `items[key]`.

    Spec 8 Phase 2 route 2: "a stratified sample across the E6 range - the target cells are
    outliers from the influence<->detection relationship, so coverage is needed to find them".
    Equal-width bins over the observed range, and the item nearest each bin centre wins, so the
    result spans the surface instead of clustering where the surface is dense. Empty bins are
    skipped rather than back-filled from a neighbour, which would quietly turn coverage into
    top-K.

    Pure, deterministic, and stable under ties (the lower-indexed item wins).
    """
    rows = [row for row in items if row[key] is not None]
    if not rows or int(n_bins) < 1:
        return []
    values = [float(row[key]) for row in rows]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [rows[0]]
    n_bins = int(n_bins)
    width = (hi - lo) / n_bins
    picked: list[dict] = []
    for b in range(n_bins):
        centre = lo + width * (b + 0.5)
        edge_lo = lo + width * b
        edge_hi = hi if b == n_bins - 1 else lo + width * (b + 1)
        inside = [(i, row) for i, row in enumerate(rows)
                  if edge_lo <= float(row[key]) <= edge_hi]
        if not inside:
            continue
        best = min(inside, key=lambda pair: (abs(float(pair[1][key]) - centre), pair[0]))
        picked.append(best[1])
    return picked


# =====================================================================================
# Phase 0 - calibration (spec 5.1)
# =====================================================================================

def _unsteered_path(prompt_id: str) -> str:
    return f"{UNSTEERED_DIR}/{prompt_id}.jsonl"


def _load_unsteered(prompt_ids: Sequence[str], n_samples: int) -> dict[str, list[str]]:
    """Reload the unsteered completions a previous run already paid for.

    Resume support, and it matters more here than anywhere else: sample 1 is the reference A
    that EVERY Judge E5 and S1 call in the run is paired against, so regenerating it after a
    crash would silently re-baseline every cell measured before the crash against different
    text. Only a prompt with at least `n_samples` stored rows counts as done.
    """
    out: dict[str, list[str]] = {}
    for pid in prompt_ids:
        rows = _read_rows(_unsteered_path(pid))
        if len(rows) >= int(n_samples):
            # Hard index: a row in this file without a response is a corrupt baseline, and an
            # empty string standing in for it would be judged as an empty reference.
            out[pid] = [str(row["response"]) for row in rows[:int(n_samples)]]
    return out


def _generate_unsteered(prompt_rows: Sequence[dict], n_samples: int) -> dict[str, list[str]]:
    """`n_samples` unsteered completions per prompt, written to `unsteered/<id>.jsonl`.

    Spec 5.1 step 4. Sampled at the run's own TEMPERATURE through the same batched decode path
    the steered generations use (`expensive.generate_unsteered`, never `mw.generate_batch` -
    bug 25b's decoding half would prepend prompt tokens to every row shorter than the longest,
    and the corrupted text would become the reference A for the entire run).

    Samples are drawn one full pass at a time rather than one prompt at a time, so sample 1 of
    every prompt comes from the same batch - there is nothing to get out of order.
    """
    from . import expensive          # lazy: expensive.py imports torch at module scope

    rendered, starts, ids = expensive.task_batch(prompt_rows)
    cfg = _cfg()
    max_new = int(cfg["MAX_NEW_TOKENS"])
    temperature = float(cfg["TEMPERATURE"])

    samples: dict[str, list[str]] = {pid: [] for pid in ids}
    for k in range(int(n_samples)):
        got = expensive.generate_unsteered(rendered, max_new, temperature,
                                           start_positions=starts)
        for row, pid, text in zip(prompt_rows, ids, got):
            samples[pid].append(str(text))
            _append_row(_unsteered_path(pid), dict(
                measure="unsteered", phase=PHASE0, sample=k + 1,
                prompt_id=pid, kind=str(row["kind"]), prompt=str(row["text"]),
                response=str(text), layer=None, alpha=None, r=None,
            ))
    return samples


def _install_unsteered(samples: dict[str, list[str]]) -> None:
    """Merge completions into `RUN.base["unsteered"]` without discarding what is there.

    MERGE, not replace. Phase 6 adds the held-out prompts' baselines to the same store that
    Phase 0 filled for E5_PROMPTS, and `expensive.unsteered_samples` looks both up in one dict.
    Replacing would delete the reference A of every screening prompt at the moment Phase 6
    starts, which the run would only discover on the next E5 call.
    """
    from . import expensive          # lazy: expensive.py imports torch at module scope

    ctx = _run()
    key = expensive.BASE_UNSTEERED_KEY
    store = ctx.base[key] if key in ctx.base else {}
    store.update(samples)
    ctx.base[key] = store


def _judge_null_failures(nulls: dict) -> list[str]:
    """Fatal instrument-null names; D2 is deliberately absent because it is report-only."""
    failures: list[str] = []
    if not bool(nulls["e5"]["passed"]):
        failures.append("E5 judge null")
    if bool(nulls["s1"]["judge_fault"]):
        failures.append("S1 judge null (S1/S2 disagreement)")
    return failures


def phase0_calibrate(*, layers: Sequence[int] | None = None, n_unsteered: int = 3,
                     run_liveness: bool = True) -> dict:
    """Spec 5.1. Everything the rest of the pipeline compares against, including judge nulls.

    In order, and the order is not arbitrary:

      1. the layer set - every layer with `d(L) >= D_MIN` (spec 8 Phase 1);
      2. `extract_all_layers` -> `RUN.vecs` and every `||v_L||`;
      3. R5 at the reference layer ONLY (bug 19: the residual stream grows with depth, so a
         difference-in-means vector is naturally small early and large late, and Macar's
         4664 +/- 982 describes L37);
      4. `build_dose_map` -> `||h_L||`, `alpha(L, r)`, `norms.jsonl`, `dose_map.json`;
      5. **R14 hook liveness, before anything is measured.** The 2026-08-04 run read exactly
         0.000 at all 30 cells for an hour because the repo's SteeringHook declined to steer
         whenever start_pos was set (bug 26). Two forward passes here would have killed it at
         setup, which is why this sits before the first measurement and not in a test;
      6. the pinned MMLU set (spec 4.4) -> `RUN.mmlu`;
      7. `n_unsteered` unsteered completions per E5 prompt. Sample 1 is the paired reference A
         used by every Judge E5 and S1 call; samples 2-3 supply the spec 5.8 control pairs;
      8. `cap_base` - unsteered MMLU correctness on the pinned set;
      9. `p_base` - unsteered concept mass per E5 prompt;
     10. `d3_base` - unsteered forced-ID concept mass;
     11. all three judge nulls. E5 and S1 can abort because their live cross-checks isolate an
         instrument fault; D2 is report-only because confabulation and judge error cannot be
         separated without reading its persisted transcripts.

    Steps 9 and 10 are measured at the reference layer with alpha = 0. The layer is irrelevant
    to an unsteered pass - `cheap._vec_for` returns None at alpha 0 and `model.injected`
    registers no hook at all - and naming one keeps the row honest about where it was taken.
    """
    from . import vectors            # lazy: vectors.py imports torch at module scope
    from . import expensive          # lazy: expensive.py imports torch at module scope

    ctx = _run()
    cfg = _cfg()
    t0 = time.time()
    print("=" * 78)
    print(f"PHASE 0 - calibration   concept={ctx.concept!r}")
    print("=" * 78)

    # --- 1. layer set ----------------------------------------------------------------
    scope = (layers_in_scope(int(ctx.n_layers), float(cfg["D_MIN"]))
             if layers is None else [int(layer) for layer in layers])
    if not scope:
        raise RuntimeError(
            f"no layer has d(L) >= D_MIN = {float(cfg['D_MIN'])} on a {ctx.n_layers}-layer "
            "model - the search space is empty and nothing below can run.")
    ref = reference_layer(int(ctx.n_layers))
    if ref not in scope:
        # Not fatal, but the escalation ladder and R5 both live at the reference layer.
        print(f"note       : reference layer L{ref} is below D_MIN and is outside the scan")
    print(f"layers     : {len(scope)} in scope, L{scope[0]}-L{scope[-1]} "
          f"(d >= {float(cfg['D_MIN']):g}); reference L{ref}")

    # --- 2/3. vectors and R5 ----------------------------------------------------------
    vectors.extract_all_layers(str(ctx.concept), scope)
    if ref in ctx.vecs:
        r5_ok, r5_detail = vectors.check_reference_norm(ref)
        print(f"R5         : {'pass' if r5_ok else 'FAIL'} - {r5_detail}")
    else:
        r5_ok, r5_detail = None, f"reference layer L{ref} not extracted; R5 not applicable"
        print(f"R5         : skipped - {r5_detail}")

    # --- 4. dose map ------------------------------------------------------------------
    # The calibration prompts are the 12 E5 questions (spec 5.1 step 2). ||h_L|| is a property
    # of the model and the prompts, not of the concept, so `measure_residual_norms` caches it
    # across concepts inside build_dose_map.
    doses = [float(d) for d in cfg["SCAN_DOSES"]]
    dose_map = vectors.build_dose_map(
        scope, doses, calib_prompts=[row["text"] for row in prompts.E5_PROMPTS])
    n_unreachable = sum(1 for alpha in dose_map.values() if alpha is None)

    # --- 5. R14 -----------------------------------------------------------------------
    liveness = None
    if run_liveness:
        liveness = model.hook_liveness()      # raises if either injection path is dead
        print(f"R14        : pass - start_pos {liveness['d_start_pos']:.2e}, "
              f"all-positions {liveness['d_all_pos']:.2e} "
              f"(L{liveness['layer']} alpha={liveness['alpha']:.3f})")

    # --- 6. the S3 item set -----------------------------------------------------------
    ctx.mmlu = prompts.load_mmlu_items(cfg)
    print(f"S3 items   : {len(ctx.mmlu)} pinned MMLU items "
          f"({cfg['S3_SPLIT']} split, one per subject)")

    # --- 7. unsteered completions -----------------------------------------------------
    reused = _load_unsteered([row["id"] for row in prompts.E5_PROMPTS], n_unsteered)
    todo = [row for row in prompts.E5_PROMPTS if row["id"] not in reused]
    if todo:
        fresh = _generate_unsteered(todo, n_unsteered)
    else:
        fresh = {}
    samples = dict(reused)
    samples.update(fresh)
    _install_unsteered(samples)
    print(f"unsteered  : {len(samples)} prompts x {n_unsteered} samples "
          f"({len(reused)} reused from a previous run)")

    # --- 8/9/10. the cheap baselines --------------------------------------------------
    s3_base = cheap.measure_S3_baseline()
    print(f"cap_base   : {s3_base['cap_base']}/{s3_base['s3_n']} MMLU correct unsteered "
          f"(acc {s3_base['s3_acc_base']:.3f}, 95% Wilson "
          f"[{s3_base['s3_acc_base_ci_low']:.3f}, {s3_base['s3_acc_base_ci_high']:.3f}])")

    e6_base = cheap.measure_E6(ref, 0.0)
    p_base = {row["prompt_id"]: row["mass"] for row in e6_base["e6_per_prompt"]}
    ctx.base[BASE_P_KEY] = p_base
    print(f"p_base     : median concept mass {e6_base['e6_mass_median']:.3e} over "
          f"{e6_base['e6_n']} prompts (unsteered reach {e6_base['reach']:.2f}, 95% Wilson "
          f"[{e6_base['reach_ci_low']:.2f}, {e6_base['reach_ci_high']:.2f}])")

    d3_base_row = cheap.measure_D3(ref, 0.0)
    ctx.base[BASE_D3_KEY] = d3_base_row["d3"]
    print(f"d3_base    : {d3_base_row['d3']:.4f}"
          + (f" +/- {d3_base_row['d3_se']:.4f}" if d3_base_row["d3_se"] is not None else "")
          + f"   rate {d3_base_row['d3_rate']:.2f} (95% Wilson "
            f"[{d3_base_row['d3_rate_ci_low']:.2f}, "
            f"{d3_base_row['d3_rate_ci_high']:.2f}], n={d3_base_row['d3_rate_n']})"
          + f"   rank median {d3_base_row['d3_rank_med']}")

    # --- 11. judge nulls --------------------------------------------------------------
    nulls = expensive.judge_nulls()
    fpr = float(nulls["e5"]["fpr"])

    # --- record -----------------------------------------------------------------------
    out = dict(
        phase=PHASE0,
        layers=scope, n_layers=int(ctx.n_layers), reference_layer=ref,
        d_min=float(cfg["D_MIN"]), doses=doses,
        n_cells=len(dose_map), n_unreachable=n_unreachable,
        r5_passed=r5_ok, r5_detail=r5_detail,
        hook_liveness=liveness,
        mmlu_n=len(ctx.mmlu),
        n_unsteered_prompts=len(samples), n_unsteered_samples=int(n_unsteered),
        cap_base=s3_base["cap_base"], s3_margin_base=s3_base["s3_margin_base"],
        s3_acc_base=s3_base["s3_acc_base"], s3_n=s3_base["s3_n"],
        s3_acc_base_ci_low=s3_base["s3_acc_base_ci_low"],
        s3_acc_base_ci_high=s3_base["s3_acc_base_ci_high"],
        p_base=p_base, p_base_median=e6_base["e6_mass_median"],
        p_base_reach=e6_base["reach"],
        p_base_reach_ci_low=e6_base["reach_ci_low"],
        p_base_reach_ci_high=e6_base["reach_ci_high"],
        p_base_reach_n=e6_base["reach_n"],
        d3_base=d3_base_row["d3"], d3_base_se=d3_base_row["d3_se"],
        d3_base_rate=d3_base_row["d3_rate"],
        d3_base_rate_count=d3_base_row["d3_rate_count"],
        d3_base_rate_n=d3_base_row["d3_rate_n"],
        d3_base_rate_ci_low=d3_base_row["d3_rate_ci_low"],
        d3_base_rate_ci_high=d3_base_row["d3_rate_ci_high"],
        d3_base_rank_med=d3_base_row["d3_rank_med"],
        judge_fpr=fpr, judge_fpr_se=nulls["e5"]["fpr_se"],
        judge_fpr_n=nulls["e5"]["fpr_n"], judge_fpr_max=float(cfg["JUDGE_FPR_MAX"]),
        judge_nulls=nulls,
        d2_null=nulls["d2"]["d2_null"],
        d2_null_ci_low=nulls["d2"]["d2_null_ci_low"],
        d2_null_ci_high=nulls["d2"]["d2_null_ci_high"],
        d2_null_n=nulls["d2"]["d2_null_n"],
        secs=round(time.time() - t0, 1),
    )
    _append_row(BASELINES_FILE, dict(out, measure="baselines"))
    failures = _judge_null_failures(nulls)
    if failures:
        print("JUDGE NULL GATE FAILED: " + ", ".join(failures))
        raise RuntimeError(
            "Phase 0 judge-null gate failed: " + ", ".join(failures) +
            ". Investigate the judge configuration; never loosen a null threshold to make "
            "the run proceed. All three readings were persisted in baselines.jsonl.")
    print(f"phase 0    : done in {out['secs']:.0f}s, {n_unreachable}/{len(dose_map)} cells "
          f"unreachable at ALPHA_CEIL (recorded, never clamped)")
    return out


# =====================================================================================
# Phase 1 - the full-depth cheap scan (spec 8 Phase 1)
# =====================================================================================

def phase1_scan(*, layers: Sequence[int] | None = None,
                doses: Sequence[float] | None = None,
                on_cell: Any = None, on_plan: Any = None) -> list[dict]:
    """Every layer with `d(L) >= D_MIN`, at every `SCAN_DOSE`. E6 + D3 + S3 + S2. **0 judge calls.**

    **Why THREE doses.** One dose cannot distinguish *"this layer is inert"* from *"this layer is
    under-dosed"* - which is precisely the error a fixed-alpha scan makes, and the reason
    every early-layer cell in the M1.5 grid read flat (at fixed alpha the L6 perturbation was
    0.3% of L37's, because ||v_L|| ran 14 at L6 to 8896 at L46). The first two doses give each
    layer a slope and bracket the seven qualifying M1.5 cells (r=0.114-0.303, median 0.180).
    Garlic still read e6=0 throughout L20-L52 at both, so r=0.60 distinguishes an inert middle
    band from one that was merely under-dosed; cheap sanity records if that dose is too high.

    **Why EVERY layer.** Peaks are concept-dependent and Macar's detection curves are
    multi-peaked, so sampling arbitrary layers can miss an optimum one layer away - and one
    layer away is exactly the resolution this pipeline exists to buy. At forward-pass cost
    there is no reason to sample: scanning all of them makes the problem disappear rather than
    needing a clever search. ~100 cells at a few seconds each is 7-10 minutes.

    Zero judge calls is structural, not a promise: every measurement here comes from
    `m2.cheap`, which cannot generate and cannot call a judge.

    Resumable at row level - a cell already in `scan.jsonl` is not re-measured. `s2` is null on
    every row because Phase 1 generates nothing; that is "not measured", never 1.0.
    """
    ctx = _run()
    cfg = _cfg()
    scope = (layers_in_scope(int(ctx.n_layers), float(cfg["D_MIN"]))
             if layers is None else [int(layer) for layer in layers])
    dose_list = [float(d) for d in (cfg["SCAN_DOSES"] if doses is None else doses)]

    existing = _read_rows(SCAN_FILE)
    done = {_cell_key(row["layer"], row["r"]) for row in existing}
    rows: list[dict] = list(existing)

    todo = [(layer, r) for layer in scope for r in dose_list
            if _cell_key(layer, r) not in done]
    print("=" * 78)
    print(f"PHASE 1 - full-depth scan   {len(scope)} layers x {len(dose_list)} doses = "
          f"{len(scope) * len(dose_list)} cells, {len(todo)} to measure")
    print("=" * 78)
    # The count the ETA needs, and only this function knows it: a resumed run's real workload
    # is `todo`, not the full grid. Reported after the resume filter, before the first cell.
    if on_plan is not None:
        on_plan(len(todo))

    t0 = time.time()
    for i, (layer, r) in enumerate(todo, start=1):
        row = cheap.scan_cell(layer, r)
        _append_row(SCAN_FILE, row)
        rows.append(row)
        if row["reachable"]:
            print(f"   L{layer} r={r:.3f}: reach {row['reach']:.2f} "
                  f"[{row['reach_ci_low']:.2f}, {row['reach_ci_high']:.2f}], "
                  f"n={row['reach_n']}; D3-rate {row['d3_rate']:.2f} "
                  f"[{row['d3_rate_ci_low']:.2f}, {row['d3_rate_ci_high']:.2f}], "
                  f"n={row['d3_rate_n']}; S3-acc {row['s3_acc']:.2f} "
                  f"[{row['s3_acc_ci_low']:.2f}, {row['s3_acc_ci_high']:.2f}], "
                  f"n={row['s3_n']}")
        else:
            print(f"   L{layer} r={r:.3f}: unreachable")
        if on_cell is not None:
            on_cell(row)
        if i % 10 == 0 or i == len(todo):
            print(f"   {i}/{len(todo)} cells, {time.time() - t0:.0f}s elapsed")

    reachable = [row for row in rows if row["reachable"]]
    print(f"phase 1    : {len(rows)} rows, {len(rows) - len(reachable)} unreachable, "
          f"{time.time() - t0:.0f}s")
    return rows


# =====================================================================================
# Phase 2 - the shortlist (spec 8 Phase 2, spec 7.2)
# =====================================================================================

def _by_layer(scan_rows: Sequence[dict]) -> dict[int, dict]:
    """Collapse the scan surface to one record per layer, keeping every dose's numbers.

    `e6` is the layer's BEST reach across doses and `d3` the D3 at that same dose. Best rather
    than mean because the two doses answer different questions (see phase1_scan): a layer that
    is inert at 0.15 and reaches 0.6 at 0.30 is a reachable layer that needs more dose, not a
    half-reachable one, and averaging would rank it below a layer that is mediocre at both.
    """
    per: dict[int, dict] = {}
    for row in scan_rows:
        if not row["reachable"]:
            continue
        layer = int(row["layer"])
        entry = per.setdefault(layer, dict(layer=layer, by_dose={}))
        entry["by_dose"][round(float(row["r"]), 6)] = row
    for layer, entry in per.items():
        rows = list(entry["by_dose"].values())
        best = max(rows, key=lambda row: (float(row["reach"]), float(row["r"])))
        entry["e6"] = float(best["reach"])
        entry["e6_at_r"] = float(best["r"])
        entry["d3"] = best["d3"]
        entry["s3"] = best["s3"]
        entry["alpha"] = best["alpha"]
        # "supported at both doses" (spec 8 Phase 2 route 1) is read here: the layer was
        # reachable at every dose scanned, so its peak is not an artefact of one dose.
        entry["n_doses"] = len(rows)
        entry["doses_measured"] = sorted(float(row["r"]) for row in rows)
        entry["reach_by_dose"] = {round(float(row["r"]), 6): float(row["reach"])
                                  for row in rows}
        entry["d3_by_dose"] = {round(float(row["r"]), 6): row["d3"] for row in rows}
    return per


def _tier_config(cfg: dict) -> dict:
    """Validate and normalise the cell-tier controls; defaults are never supplied here."""
    size = int(cfg["SHORTLIST_TIER_SIZE"])
    audit = int(cfg["SHORTLIST_AUDIT_TIERS"])
    maximum_raw = cfg["SHORTLIST_MAX_TIER"]
    maximum = None if maximum_raw is None else int(maximum_raw)
    exhaustive = bool(cfg["SHORTLIST_EXHAUSTIVE"])
    if size <= 0:
        raise ValueError(f"SHORTLIST_TIER_SIZE must be positive, got {size}")
    if audit < 0:
        raise ValueError(f"SHORTLIST_AUDIT_TIERS must be non-negative, got {audit}")
    if maximum is not None and maximum < audit:
        raise ValueError(
            f"SHORTLIST_MAX_TIER={maximum} is below SHORTLIST_AUDIT_TIERS={audit}; the "
            "audit tiers are mandatory, so this configuration contradicts itself")
    return dict(tier_size=size, audit_tiers=audit, max_tier=maximum,
                tier_order="pareto_distance", exhaustive=exhaustive)


def _should_execute_tier(tier: int | None, *, has_qualifier: bool,
                         audit_tiers: int, max_tier: int | None,
                         exhaustive: bool) -> tuple[bool, str]:
    """Pure tier state transition, including the always-run audit and stop limit."""
    if exhaustive:
        return True, "exhaustive mode verifies every in-scope layer"
    number = int(tier or 0)
    if number == 0:
        return True, "tier 0 is the primary shortlist"
    if number <= int(audit_tiers):
        return True, f"tier {number} is a mandatory false-negative audit tier"
    if has_qualifier:
        return False, "a qualifying cell already exists and mandatory audit tiers are complete"
    if max_tier is not None and number > int(max_tier):
        return False, f"configured SHORTLIST_MAX_TIER={int(max_tier)} reached"
    return True, "no qualifying cell yet; failure-driven escalation continues"


def _scan_cell_order(row: dict) -> tuple[int, float]:
    """Deterministic identity order for scan cells."""
    return int(row["layer"]), float(row["r"])


def _eligible_scan_cell(row: dict, *, reach_floor: float, s4_min: float) -> bool:
    """Task 21 eligibility: reachable, sane, live, and readable on both proxy axes."""
    return bool(
        row.get("reachable")
        and row.get("reach") is not None
        and row.get("d3") is not None
        and row.get("s3") is not None
        and float(row["s3"]) >= float(s4_min)
        and float(row["reach"]) >= float(reach_floor)
    )


def _proxy_dominates(left: dict, right: dict) -> bool:
    """Whether `left` is at least as influential and no more detectable, with one strict."""
    l_reach, l_d3 = float(left["reach"]), float(left["d3"])
    r_reach, r_d3 = float(right["reach"]), float(right["d3"])
    return (l_reach >= r_reach and l_d3 <= r_d3
            and (l_reach > r_reach or l_d3 < r_d3))


def _proxy_frontier(rows: Sequence[dict]) -> list[dict]:
    """The reach-ascending/d3-descending Pareto frontier over eligible cells."""
    cells = list(rows)
    out = [row for row in cells
           if not any(_proxy_dominates(other, row) for other in cells if other is not row)]
    return sorted(out, key=lambda row: (-float(row["reach"]), float(row["d3"]),
                                        int(row["layer"]), float(row["r"])))


def _frontier_distance(row: dict, frontier_rows: Sequence[dict],
                       reach_floor: float) -> tuple[float, dict]:
    """Additive-epsilon distance to the closest frontier cell.

    Reach is divided by its honest eligible support, ``1 - E6_FLOOR``; d3 already lives on
    [0, 1]. The score is the smallest worst-axis improvement needed to meet a frontier cell:
    it is zero on the frontier and has an operational meaning on both dominated eligible
    cells and live cells just below the reach floor. No empirical SD or fitted scale can move
    the ranking when the scan sample changes.
    """
    if not frontier_rows:
        raise ValueError("frontier distance is undefined without a frontier")
    span = 1.0 - float(reach_floor)
    if not span > 0.0:
        raise ValueError(f"E6_FLOOR must be below 1 for distance scaling, got {reach_floor}")
    reach = float(row["reach"])
    d3 = float(row["d3"])
    scored: list[tuple[float, tuple, dict]] = []
    for anchor in frontier_rows:
        reach_deficit = max(0.0, float(anchor["reach"]) - reach) / span
        d3_excess = max(0.0, d3 - float(anchor["d3"]))
        distance = max(reach_deficit, d3_excess)
        tie = (float(anchor["d3"]), -float(anchor["reach"]),
               int(anchor["layer"]), float(anchor["r"]))
        scored.append((distance, tie, anchor))
    distance, _, anchor = min(scored, key=lambda item: (item[0], item[1]))
    return float(distance), anchor


def _objective_distance(left: dict, right: dict, reach_floor: float) -> float:
    """L-infinity distance on the same normalized objective plane as near-miss distance."""
    span = 1.0 - float(reach_floor)
    return max(abs(float(left["reach"]) - float(right["reach"])) / span,
               abs(float(left["d3"]) - float(right["d3"])))


def _cap_proxy_frontier(frontier_rows: Sequence[dict], limit: int,
                        reach_floor: float) -> list[dict]:
    """Preserve both extremes, then greedily cover the frontier's objective-space shape."""
    rows = list(frontier_rows)
    if len(rows) <= int(limit):
        return rows
    if limit < 2:
        raise ValueError(f"SHORTLIST_N upper bound must be >=2 to preserve both extremes")

    max_reach = min(rows, key=lambda row: (-float(row["reach"]), float(row["d3"]),
                                           -float(row["s3"]), *_scan_cell_order(row)))
    min_d3 = min(rows, key=lambda row: (float(row["d3"]), -float(row["reach"]),
                                        -float(row["s3"]), *_scan_cell_order(row)))
    chosen = [max_reach]
    if _cell_key(min_d3["layer"], min_d3["r"]) != _cell_key(max_reach["layer"], max_reach["r"]):
        chosen.append(min_d3)
    remaining = [row for row in rows
                 if _cell_key(row["layer"], row["r"]) not in
                 {_cell_key(c["layer"], c["r"]) for c in chosen}]
    while remaining and len(chosen) < int(limit):
        def coverage_rank(row: dict) -> tuple:
            separation = min(_objective_distance(row, kept, reach_floor) for kept in chosen)
            return (separation, float(row["s3"]), float(row["reach"]),
                    -float(row["d3"]), -int(row["layer"]), -float(row["r"]))
        picked = max(remaining, key=coverage_rank)
        chosen.append(picked)
        remaining.remove(picked)
    return sorted(chosen, key=lambda row: (-float(row["reach"]), float(row["d3"]),
                                           int(row["layer"]), float(row["r"])))


def _fill_proxy_frontier(frontier_rows: Sequence[dict], dominated_rows: Sequence[dict],
                         target: int, reach_floor: float) -> list[tuple[dict, float, dict]]:
    """Fill toward the lower bound without letting one dense frontier region monopolize it."""
    groups: dict[tuple, list[tuple[dict, float, dict]]] = {}
    anchors: dict[tuple, dict] = {}
    for row in dominated_rows:
        distance, anchor = _frontier_distance(row, frontier_rows, reach_floor)
        key = _cell_key(anchor["layer"], anchor["r"])
        anchors[key] = anchor
        groups.setdefault(key, []).append((row, distance, anchor))
    for values in groups.values():
        values.sort(key=lambda item: (item[1], -float(item[0]["reach"]),
                                      float(item[0]["d3"]), *_scan_cell_order(item[0])))

    anchor_order = sorted(groups, key=lambda key: (
        float(anchors[key]["d3"]), -float(anchors[key]["reach"]), key))
    picked: list[tuple[dict, float, dict]] = []
    while len(frontier_rows) + len(picked) < int(target):
        progressed = False
        for key in anchor_order:
            if not groups[key]:
                continue
            picked.append(groups[key].pop(0))
            progressed = True
            if len(frontier_rows) + len(picked) >= int(target):
                break
        if not progressed:
            break
    return picked


def _selection_candidate(row: dict, *, kind: str, distance: float,
                         anchor: dict | None, why: str) -> dict:
    """Copy one scan cell into the shortlist envelope with traceable selection provenance."""
    out = dict(row)
    out.update(
        e6=float(row["reach"]),
        selection_kind=str(kind),
        pareto_frontier=(kind == "frontier"),
        frontier_distance=float(distance),
        nearest_frontier=(None if anchor is None else
                          dict(layer=int(anchor["layer"]), r=float(anchor["r"]),
                               reach=float(anchor["reach"]), d3=float(anchor["d3"]))),
        why=[why],
        routes=["pareto_frontier" if kind == "frontier" else "pareto_fill"],
        merged_from=[],
    )
    return out


def _excluded_layer_records(surface: Sequence[dict], eligible: Sequence[dict],
                            *, reach_floor: float, s4_min: float) -> list[dict]:
    """Report every layer with no eligible cell, including each dose's reach and s3."""
    eligible_layers = {int(row["layer"]) for row in eligible}
    by_layer: dict[int, list[dict]] = {}
    for row in surface:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    out: list[dict] = []
    for layer, rows in sorted(by_layer.items()):
        if layer in eligible_layers:
            continue
        cells = []
        for row in sorted(rows, key=lambda item: float(item["r"])):
            reasons = []
            if not row.get("reachable"):
                reasons.append("unreachable")
            else:
                if row.get("s3") is None or float(row["s3"]) < s4_min:
                    reasons.append(f"s3 below S4_MIN={s4_min:g}")
                if row.get("reach") is None or float(row["reach"]) < reach_floor:
                    reasons.append(f"reach below E6_FLOOR={reach_floor:g}")
                if row.get("d3") is None:
                    reasons.append("d3 unavailable")
            cells.append(dict(r=float(row["r"]), reachable=bool(row.get("reachable")),
                              reach=row.get("reach"), d3=row.get("d3"), s3=row.get("s3"),
                              reasons=reasons))
        out.append(dict(layer=layer, reason="no eligible cell", cells=cells))
    return out


def phase2_shortlist(scan_rows: Sequence[dict], *,
                     n_bins: int = STRATIFIED_BINS, write: bool = True) -> dict:
    """Select `(layer, r)` cells: eligibility, then the reach-vs-d3 Pareto frontier.

    The unit stays a cell all the way into verification. The old implementation collapsed
    each layer to one dose, fitted d3~reach, then selected layers through six routes; Phase 3
    consequently discarded the dose that made a cell interesting. This rule reads the two
    measured proxy coordinates directly and keeps the entering dose.

    `n_bins` is retained only for call compatibility with old notebooks and is ignored.
    """
    del n_bins
    cfg = _cfg()
    tier_cfg = _tier_config(cfg)
    reach_floor = float(cfg["E6_FLOOR"])
    s4_min = float(cfg["S4_MIN"])
    want_lo, want_hi = (int(x) for x in cfg["SHORTLIST_N"])
    if want_lo > want_hi or want_lo < 1:
        raise ValueError(f"SHORTLIST_N must be a positive ordered band, got {(want_lo, want_hi)}")

    # A resumed JSONL may contain an exact duplicate after an interrupted append. Selection
    # is over cells, not file lines, so use the same canonical key as every resume guard.
    by_key = {_cell_key(row["layer"], row["r"]): row for row in scan_rows}
    surface = sorted(by_key.values(), key=_scan_cell_order)
    reachable = [row for row in surface if row.get("reachable")]
    eligible = [row for row in surface
                if _eligible_scan_cell(row, reach_floor=reach_floor, s4_min=s4_min)]
    if not eligible:
        raise RuntimeError(
            f"Phase 2 found no eligible scan cell: 0/{len(surface)} were reachable with "
            f"s3 >= S4_MIN={s4_min:g} and reach >= E6_FLOOR={reach_floor:g}. The excluded "
            "layer table in scan.jsonl is the result; do not manufacture a frontier from "
            "dead or damaged cells.")

    full_frontier = _proxy_frontier(eligible)
    frontier_keys = {_cell_key(row["layer"], row["r"]) for row in full_frontier}
    dominated = [row for row in eligible
                 if _cell_key(row["layer"], row["r"]) not in frontier_keys]
    notes: list[str] = []

    if len(full_frontier) > want_hi:
        selected_frontier = _cap_proxy_frontier(full_frontier, want_hi, reach_floor)
        notes.append(f"Pareto frontier capped from {len(full_frontier)} to {want_hi} cells; "
                     "max-reach and min-d3 extremes were pinned, then maximin coverage was "
                     "applied on the normalized proxy plane")
    else:
        selected_frontier = list(full_frontier)

    selected: list[dict] = [
        _selection_candidate(
            row, kind="frontier", distance=0.0, anchor=row,
            why=(f"Pareto frontier: no eligible cell has reach >= {float(row['reach']):.3f} "
                 f"with d3 <= {float(row['d3']):.6f}"))
        for row in selected_frontier
    ]
    if len(selected) < want_lo:
        fills = _fill_proxy_frontier(full_frontier, dominated, want_lo, reach_floor)
        for row, distance, anchor in fills:
            selected.append(_selection_candidate(
                row, kind="filled", distance=distance, anchor=anchor,
                why=(f"filled toward SHORTLIST_N={want_lo}: additive-epsilon distance "
                     f"{distance:.6f} from frontier cell L{anchor['layer']}@{float(anchor['r']):.6f}")))
        if fills:
            notes.append(f"frontier had {len(full_frontier)} cells; filled with {len(fills)} "
                         "nearest dominated eligible cells, round-robin across frontier anchors")
        if len(selected) < want_lo:
            notes.append(f"shortlist stops at {len(selected)}, below SHORTLIST_N={want_lo}: "
                         "there are no more eligible cells to verify")

    selected.sort(key=lambda row: (0 if row["selection_kind"] == "frontier" else 1,
                                   -float(row["reach"]), float(row["d3"]),
                                   int(row["layer"]), float(row["r"])))
    tier0: list[dict] = []
    for rank, candidate in enumerate(selected, start=1):
        tier0.append(dict(candidate, tier=0, tier_rank=rank,
                          tier_ordering="pareto_frontier",
                          tier_order_rank=rank,
                          tier_source_layer=int(candidate["layer"]),
                          tier_source_cell=dict(layer=int(candidate["layer"]),
                                                r=float(candidate["r"])),
                          exhaustive=False))

    selected_keys = {_cell_key(row["layer"], row["r"]) for row in tier0}
    omitted_frontier_keys = frontier_keys - selected_keys
    near_misses: list[dict] = []
    for row in eligible:
        key = _cell_key(row["layer"], row["r"])
        if key in selected_keys:
            continue
        distance, anchor = _frontier_distance(row, full_frontier, reach_floor)
        kind = "frontier_omitted_by_cap" if key in omitted_frontier_keys else "dominated_eligible"
        near_misses.append(dict(row=row, distance=distance, anchor=anchor, kind=kind,
                                priority=0 if key in omitted_frontier_keys else 1))

    # E6_FLOOR is load-bearing, but a live sane cell just below it is still useful evidence
    # about the floor. Report it as a near-miss; do not let reach=0 dead cells back in.
    for row in surface:
        if (row.get("reachable") and row.get("reach") is not None and row.get("d3") is not None
                and row.get("s3") is not None and float(row["s3"]) >= s4_min
                and 0.0 < float(row["reach"]) < reach_floor):
            distance, anchor = _frontier_distance(row, full_frontier, reach_floor)
            near_misses.append(dict(row=row, distance=distance, anchor=anchor,
                                    kind="below_reach_floor", priority=2))

    near_misses.sort(key=lambda item: (item["priority"], item["distance"],
                                       -float(item["row"]["reach"]),
                                       float(item["row"]["d3"]),
                                       *_scan_cell_order(item["row"])))
    outer_order: list[dict] = []
    for rank, item in enumerate(near_misses, start=1):
        row, anchor = item["row"], item["anchor"]
        candidate = _selection_candidate(
            row, kind="near_miss", distance=item["distance"], anchor=anchor,
            why=(f"{item['kind']} near-miss at additive-epsilon distance "
                 f"{item['distance']:.6f} from L{anchor['layer']}@{float(anchor['r']):.6f}"))
        candidate.update(near_miss_kind=item["kind"], routes=[], merged_from=[],
                         tier_ordering="pareto_distance", tier_order_rank=rank,
                         tier_source_layer=int(row["layer"]),
                         tier_source_cell=dict(layer=int(row["layer"]), r=float(row["r"])))
        outer_order.append(candidate)

    tiers: list[dict] = [dict(tier=0, kind="shortlist", mandatory=True,
                              candidates=tier0,
                              cells=[dict(layer=int(row["layer"]), r=float(row["r"]))
                                     for row in tier0],
                              layers=[int(row["layer"]) for row in tier0])]
    size = tier_cfg["tier_size"]
    for offset in range(0, len(outer_order), size):
        number = 1 + offset // size
        chunk = [dict(row, tier=number, tier_rank=i + 1, exhaustive=False)
                 for i, row in enumerate(outer_order[offset:offset + size])]
        tiers.append(dict(
            tier=number,
            kind="audit" if number <= tier_cfg["audit_tiers"] else "escalation",
            mandatory=(number <= tier_cfg["audit_tiers"]),
            candidates=chunk,
            cells=[dict(layer=int(row["layer"]), r=float(row["r"])) for row in chunk],
            layers=[int(row["layer"]) for row in chunk],
        ))

    exhaustive_candidates: list[dict] = []
    if tier_cfg["exhaustive"]:
        # Exhaustive means every eligible CELL. Ineligible cells cannot satisfy the settled
        # Phase 2 contract and measuring them would turn exhaustive evidence into scope drift.
        ranked = list(tier0)
        ranked_keys = {_cell_key(row["layer"], row["r"]) for row in ranked}
        for row in eligible:
            if _cell_key(row["layer"], row["r"]) in ranked_keys:
                continue
            distance, anchor = _frontier_distance(row, full_frontier, reach_floor)
            ranked.append(dict(
                _selection_candidate(
                    row, kind="near_miss", distance=distance, anchor=anchor,
                    why="exhaustive mode: verify every eligible scan cell"),
                tier=None, tier_rank=None, tier_ordering="exhaustive_eligible",
                tier_order_rank=None, tier_source_layer=int(row["layer"]),
                tier_source_cell=dict(layer=int(row["layer"]), r=float(row["r"])),
                exhaustive=True))
        exhaustive_candidates = [dict(row, tier=None, tier_rank=i + 1,
                                      exhaustive=True)
                                 for i, row in enumerate(ranked)]
        tiers = [dict(tier=None, kind="exhaustive", mandatory=True,
                      candidates=exhaustive_candidates,
                      cells=[dict(layer=int(row["layer"]), r=float(row["r"]))
                             for row in exhaustive_candidates],
                      layers=[int(row["layer"]) for row in exhaustive_candidates])]

    excluded_layers = _excluded_layer_records(
        surface, eligible, reach_floor=reach_floor, s4_min=s4_min)
    frontier_records = [dict(layer=int(row["layer"]), r=float(row["r"]),
                             reach=float(row["reach"]), d3=float(row["d3"]),
                             d3_rate=row.get("d3_rate"), s3=float(row["s3"]))
                        for row in full_frontier]
    payload = dict(
        phase=PHASE2,
        selection_unit="cell", detection_axis="d3",
        e6_floor=reach_floor, s4_min=s4_min, shortlist_n=[want_lo, want_hi],
        n_cells_scanned=len(surface), n_cells_reachable=len(reachable),
        n_cells_eligible=len(eligible), n_frontier=len(full_frontier),
        frontier=frontier_records,
        n_layers_scanned=len({int(row["layer"]) for row in surface}),
        n_candidates=len(tier0), notes=notes,
        candidates=tier0,
        tiers=tiers,
        exhaustive=tier_cfg["exhaustive"],
        tier_size=tier_cfg["tier_size"],
        audit_tiers=tier_cfg["audit_tiers"],
        max_tier=tier_cfg["max_tier"],
        tier_order=tier_cfg["tier_order"],
        n_rejected=len(outer_order),
        n_rejected_live=len(outer_order),
        n_rejected_dead=len(surface) - len(eligible),
        n_near_miss=len(outer_order),
        near_misses=[dict(layer=int(row["layer"]), r=float(row["r"]),
                          reach=float(row["reach"]), d3=float(row["d3"]),
                          s3=float(row["s3"]), kind=row["near_miss_kind"],
                          frontier_distance=float(row["frontier_distance"]),
                          nearest_frontier=row["nearest_frontier"])
                     for row in outer_order],
        excluded_layers=excluded_layers,
    )
    if write:
        _write_json(SHORTLIST_FILE, payload)

    print("=" * 78)
    print(f"PHASE 2 - cell Pareto shortlist: {len(surface)} scanned -> {len(eligible)} "
          f"eligible -> {len(full_frontier)} frontier -> {len(tier0)} tier-0 cells")
    print("=" * 78)
    for note in notes:
        print(f"   note: {note}")
    print(f"   {'cell':>14} {'reach':>7} {'d3':>9} {'s3':>7}  why")
    for cand in tier0:
        label = f"L{cand['layer']}@{float(cand['r']):.3f}"
        print(f"   {label:>14} {float(cand['reach']):>7.2f} "
              f"{float(cand['d3']):>9.5f} {float(cand['s3']):>7.2f}  "
              f"{cand['selection_kind']} distance={float(cand['frontier_distance']):.6f}")
    if tier_cfg["exhaustive"]:
        print(f"   EXHAUSTIVE: {len(exhaustive_candidates)} eligible cells will be bisected "
              "and verified; qualifiers never stop execution")
    else:
        print(f"   outer audit population: {len(outer_order)} Pareto near-miss cells; "
              f"{len(excluded_layers)} layers have no eligible cell and are reported")
        for tier in tiers[1:]:
            cells = [f"L{row['layer']}@{float(row['r']):.3f}" for row in tier["candidates"]]
            print(f"   tier {tier['tier']}: {cells} by Pareto distance "
                  f"({'always runs' if tier['mandatory'] else 'failure escalation'})")
    return payload


def _merge_adjacent(candidate_layers: Sequence[int], per_layer: dict, reasons: dict,
                    routes: dict) -> list[dict]:
    """Merge candidates within +/-1 layer, keeping the better and unioning the reasons.

    Spec 8 Phase 2: "Merge candidates within +/-1 layer, keeping the better." Better is higher
    reach, then more negative residual, then the lower layer index - a total order, so the
    merge is deterministic and a rerun cannot produce a different shortlist from the same scan.

    The dropped layers are recorded on the survivor (`merged_from`) rather than discarded:
    Phase 5 re-measures layer +/-1 around the winner anyway, and knowing that a neighbour was
    also a candidate is what makes that neighbourhood interesting rather than routine.
    """
    groups: list[list[int]] = []
    for layer in sorted(candidate_layers):
        if groups and layer - groups[-1][-1] <= 1:
            groups[-1].append(layer)
        else:
            groups.append([layer])

    out: list[dict] = []
    for group in groups:
        def rank(layer: int) -> tuple:
            entry = per_layer[layer]
            resid = entry["resid"]
            return (-float(entry["e6"]),
                    float(resid) if resid is not None else 0.0,
                    layer)
        keep = min(group, key=rank)
        entry = per_layer[keep]
        why: list[str] = []
        for layer in group:
            for reason in reasons[layer]:
                text = reason if layer == keep else f"{reason} (merged from L{layer})"
                if text not in why:
                    why.append(text)
        out.append(dict(
            layer=keep,
            r=entry["e6_at_r"],
            e6=entry["e6"], d3=entry["d3"], s3=entry["s3"], resid=entry["resid"],
            reach_by_dose=entry["reach_by_dose"], d3_by_dose=entry["d3_by_dose"],
            why=why,
            routes=sorted(name for name, members in routes.items()
                          if members & set(group)),
            merged_from=[layer for layer in group if layer != keep],
        ))
    return out


def _d3_baseline() -> float:
    """Phase 0's unsteered forced-ID mass, or 0.0 if Phase 0 has not run.

    Read defensively. Phase 2 is otherwise a pure function of the scan rows and is called
    directly by the offline tests, which never run Phase 0 - a missing baseline must fall back
    to the `D3_SIGNAL_MIN` floor, not raise.
    """
    try:
        value = _run().base.get(BASE_D3_KEY)
    except Exception:                    # noqa: BLE001 - no run context in a unit test
        return 0.0
    return float(value) if isinstance(value, (int, float)) else 0.0


def _layer_has_signal(entry: dict, *, d3_base: float | None = None) -> bool:
    """Whether one collapsed scan layer belongs to any widening/audit population.

    This is the single implementation of commit 1dc85b1's guard. Both E6 and residual tier
    orderings call it before ranking: otherwise a dead layer with E6~=D3~=0 can acquire a
    mildly negative fit residual and jump to the front of the false-negative audit.
    """
    base = _d3_baseline() if d3_base is None else float(d3_base)
    if float(entry["e6"]) > 0.0:
        return True
    d3 = entry["d3"]
    return d3 is not None and float(d3) > max(D3_SIGNAL_MIN, base)


def _order_rejected_layers(per_layer: dict[int, dict], rejected_layers: Sequence[int],
                           ordering: str, *, d3_base: float | None = None) -> tuple[list[dict], list[int]]:
    """Order only LIVE rejected layers; return `(ordered rows, excluded dead layers)`.

    `e6_residual_interleave` alternates E6, residual, E6 ... and deduplicates. Each output
    records which queue selected it, so an outer-tier qualifier traces back to the shortlist
    route the audit caught. Unknown order names raise rather than silently changing the run.
    """
    mode = str(ordering)
    if mode not in TIER_ORDERINGS:
        raise ValueError(
            f"SHORTLIST_TIER_ORDER={mode!r} is not recognised; choose one of "
            f"{sorted(TIER_ORDERINGS)}. No fallback is allowed because it would mislabel "
            "the audit recorded in config.json")
    rejected = [int(layer) for layer in rejected_layers]
    live = [layer for layer in rejected if layer in per_layer
            and _layer_has_signal(per_layer[layer], d3_base=d3_base)]
    dead = sorted(set(rejected) - set(live))
    by_e6 = sorted(live, key=lambda layer: (-float(per_layer[layer]["e6"]), layer))
    by_resid = sorted(
        [layer for layer in live if per_layer[layer].get("resid") is not None],
        key=lambda layer: (float(per_layer[layer]["resid"]),
                           -float(per_layer[layer]["e6"]), layer))

    picked: list[tuple[int, str]] = []
    seen: set[int] = set()

    def take(queue: list[int], source: str) -> bool:
        while queue and queue[0] in seen:
            queue.pop(0)
        if not queue:
            return False
        layer = queue.pop(0)
        seen.add(layer)
        picked.append((layer, source))
        return True

    if mode == "e6_desc":
        while take(by_e6, "e6_desc"):
            pass
    else:
        # Starting with E6 makes TIER_SIZE=3 read E6 #1, residual #1, E6 #2. If one queue
        # empties, the other supplies the remainder; deduplication never shrinks a tier while
        # a live rejected layer is still available.
        turn = 0
        while len(seen) < len(live):
            source = "e6_desc" if turn % 2 == 0 else "residual_asc"
            queue = by_e6 if source == "e6_desc" else by_resid
            if not take(queue, source):
                other_source = "residual_asc" if source == "e6_desc" else "e6_desc"
                other = by_resid if source == "e6_desc" else by_e6
                if not take(other, other_source):
                    # A live layer may lack a residual; E6 is the total fallback ordering.
                    if not take(by_e6, "e6_desc"):
                        break
            turn += 1

    rows: list[dict] = []
    for rank, (layer, source) in enumerate(picked, start=1):
        entry = per_layer[layer]
        rows.append(dict(
            layer=layer, r=entry["e6_at_r"], e6=entry["e6"], d3=entry["d3"],
            s3=entry["s3"], resid=entry.get("resid"),
            reach_by_dose=entry["reach_by_dose"], d3_by_dose=entry["d3_by_dose"],
            why=[f"live layer rejected from tier 0; outer ordering selected it by {source} "
                 f"at overall rank {rank}"],
            routes=[], merged_from=[], tier_ordering=source,
            tier_order_rank=rank, tier_source_layer=layer,
        ))
    return rows, dead


def _size_shortlist(candidates: list[dict], per_layer: dict,
                    want_lo: int, want_hi: int) -> tuple[list[dict], str | None]:
    """Bring the merged candidate list inside `SHORTLIST_N`, without turning it into a top-K.

    **Trimming** is round-robin across the routes in the order residual, local_max, stratified,
    so an over-full shortlist loses breadth evenly instead of losing whichever route happens to
    rank lowest on E6. Dropping by E6 is the one thing that must not happen here (spec 7.2).

    **Widening** adds the leftover candidate whose E6 is farthest from every already-selected
    candidate's E6 - i.e. it buys coverage, which is what route 2 was for, rather than the next
    highest peak.
    """
    if want_lo > want_hi:
        raise ValueError(f"SHORTLIST_N is ({want_lo}, {want_hi}): the minimum exceeds the "
                         "maximum")
    if len(candidates) <= want_hi and len(candidates) >= want_lo:
        return candidates, None

    by_layer = {cand["layer"]: cand for cand in candidates}
    if len(candidates) > want_hi:
        order = ["residual", "local_max", "stratified"]
        buckets = {name: [cand["layer"] for cand in candidates if name in cand["routes"]]
                   for name in order}
        for name in order:
            # Deterministic within a route: most negative residual first for the residual
            # route, highest reach first for the other two.
            if name == "residual":
                buckets[name].sort(key=lambda layer: (
                    by_layer[layer]["resid"] if by_layer[layer]["resid"] is not None else 0.0,
                    layer))
            else:
                buckets[name].sort(key=lambda layer: (-float(by_layer[layer]["e6"]), layer))
        kept: list[int] = []
        while len(kept) < want_hi:
            progressed = False
            for name in order:
                while buckets[name]:
                    layer = buckets[name].pop(0)
                    if layer in kept:
                        continue
                    kept.append(layer)
                    progressed = True
                    break
                if len(kept) >= want_hi:
                    break
            if not progressed:
                break
        trimmed = [by_layer[layer] for layer in sorted(kept)]
        return trimmed, (f"trimmed {len(candidates)} candidates to {len(trimmed)} by "
                         "round-robin across routes, never by effectiveness (spec 7.2)")

    # Too few: widen for coverage.
    #
    # A LIVE POOL FIRST. Widening picks the layer whose E6 is FARTHEST from everything already
    # selected, which buys coverage of the E6 range (spec 8 route 2) - but with no floor it
    # maximises distance by choosing the deadest layer available. On Garlic, whose E6 is 0.00
    # from L13 to ~L52 and 1.00 at L60, that put L13, L14 and L15 on the shortlist at
    # reach=0.00 and D3=0.000: no concept mass on any of the twelve prompts, so E5 cannot
    # reach E5_FLOOR and the cell cannot qualify however it is measured. Three such cells cost
    # ~150 judge calls and ~8 minutes of Phase 4 to confirm a zero that Phase 1 already knew.
    #
    # Sub-E6_FLOOR layers are still wanted - the target cells are outliers from the
    # influence-detection relationship, which is the whole reason route 2 exists - so the bar
    # here is not E6_FLOOR. It is "something happened at all". Dead layers are used only if
    # nothing else is left, and the note says so.
    chosen = list(candidates)
    unpicked = [layer for layer in sorted(per_layer) if layer not in by_layer]

    # The bar for "something happened" is NOT `d3 > 0`. D3 is a probability mass over the
    # forced-ID logits and is essentially never exactly zero - the unsteered baseline itself is
    # a small positive number - so `> 0.0` passed every layer on the grid and the live-pool
    # guard never fired. Garlic's L13/L14/L15 went onto the shortlist at reach 0.00 and D3
    # 0.000 (printed; ~1e-4 in fact) exactly as before the guard existed.
    #
    # Against the unsteered baseline instead, with a floor under it so a near-zero baseline
    # cannot make noise look like signal. Reach still decides on its own: a layer whose concept
    # mass shows up in free generation is live whatever D3 says.
    d3_base = _d3_baseline()
    live = [layer for layer in unpicked
            if _layer_has_signal(per_layer[layer], d3_base=d3_base)]
    # Dead layers are no longer a fallback pool. A cell with no concept mass in any of the
    # twelve prompts at either scan dose cannot clear E5_FLOOR, so Phase 4 would spend ~50
    # judge calls and ~2 minutes per cell to confirm a zero Phase 1 already reported. A
    # shortlist that is honestly short beats a full one padded with cells that cannot win.
    pool = list(live)
    n_live = len(live)
    added: list[int] = []
    while len(chosen) < want_lo and pool:
        have = [float(cand["e6"]) for cand in chosen]

        def gap(layer: int) -> tuple:
            e6 = float(per_layer[layer]["e6"])
            # Live layers sort ahead of dead ones regardless of how much coverage a dead one
            # would buy: a cell with no concept mass anywhere buys coverage of a range the
            # answer cannot be in.
            alive = 0 if _layer_has_signal(per_layer[layer], d3_base=d3_base) else 1
            span = -min(abs(e6 - h) for h in have) if have else 0.0
            return (alive, span, layer)

        layer = min(pool, key=gap)
        pool.remove(layer)
        entry = per_layer[layer]
        chosen.append(dict(
            layer=layer, r=entry["e6_at_r"], e6=entry["e6"], d3=entry["d3"], s3=entry["s3"],
            resid=entry["resid"], reach_by_dose=entry["reach_by_dose"],
            d3_by_dose=entry["d3_by_dose"],
            why=["widened for coverage: the shortlist was below SHORTLIST_N and this layer's "
                 "E6 is farthest from every already-selected candidate's"],
            routes=["widen"], merged_from=[]))
        added.append(layer)
    chosen.sort(key=lambda cand: cand["layer"])
    note = None
    if added:
        note = (f"widened to {len(chosen)} candidates by E6 coverage (added "
                f"{['L' + str(layer) for layer in added]})")
    if len(chosen) < want_lo:
        short = (f"shortlist stops at {len(chosen)}, below SHORTLIST_N={want_lo}: only "
                 f"{n_live} unpicked layer(s) showed any signal at the scan doses "
                 f"{'; '.join(f'{d:g}' for d in _cfg()['SCAN_DOSES'])}. The rest read reach "
                 f"0.00 with D3 at baseline, cannot clear E5_FLOOR, and are NOT measured. "
                 f"This is a real result about the concept, not a failure to fill a quota - "
                 f"if you want the early layers probed, raise SCAN_DOSES rather than pad")
        note = f"{note}; {short}" if note else short
    return chosen, note


# =====================================================================================
# Phase 3 - dose bisection (spec 8 Phase 3)
# =====================================================================================

def _cheap_sane(row: dict, s4_min: float) -> bool | None:
    """Is this cell cheaply sane? `None` when the cell is unreachable and so unmeasured.

    The cheap tier has only ONE of the three sanity terms: S1 needs a judge and S2 needs
    generations, so the bisection boundary is S3's - verifiable-task correctness as a ratio
    against `cap_base`. That is deliberate and it is why Phase 4 re-measures the full
    `S4 = min(S1, S2, S3)` at the chosen dose rather than trusting this boundary: S3 is the
    term that catches a model that can no longer perform, which is the failure the dose
    ceiling exists to avoid, but a cell can pass S3 and still be looping (S2) or off-task (S1).

    Hard index on `s3`: a reachable row without one means `cheap.scan_cell` changed shape, and
    a defaulted pass would put the boundary above the damage threshold.
    """
    if not row["reachable"]:
        return None
    s3 = row["s3"]
    if s3 is None:
        raise RuntimeError(
            f"reachable scan row L{row['layer']} r={row['r']} has s3=None - the bisection "
            "boundary is S3 and cannot be evaluated without it.")
    return float(s3) >= float(s4_min)


def _probe(layer: int, r: float, cache: dict) -> dict:
    """One cheap cell at `(layer, r)`, memoised for the duration of a bisection.

    `cheap.scan_cell` already records an unreachable cell with `reachable: false` and every
    metric null rather than clamping alpha, so nothing here needs to catch `Unreachable`.
    """
    key = _cell_key(layer, r)
    if key not in cache:
        cache[key] = dict(cheap.scan_cell(int(layer), float(r)), phase=PHASE3)
    return cache[key]


def phase3_bisect(candidates: Sequence[dict], *,
                  scan_rows: Sequence[dict] | None = None,
                  on_cell: Any = None, on_plan: Any = None) -> list[dict]:
    """Per selected cell: preserve its dose while mapping the sanity boundary. **0 judge calls.**

    1. **Bracket** - the lowest `r` clearing `E6_FLOOR`, and the lowest `r` failing cheap
       sanity. The scan doses are reused for free; above them the probe escalates by
       `BISECT_ESCALATE` until sanity fails or the cell goes unreachable at `ALPHA_CEIL`.
    2. **Bisect** with `BISECT_STEPS` evaluations, giving ~3% resolution in `r`.
    3. Keep the selected Phase 2 dose as the Phase 4 cell. The boundary is evidence for
       Gate 4 and a map of the sane range; it never replaces the selected answer candidate.

    **Bisection is sound here because sanity is monotone in dose** - 0 violations in 18 M1.5
    (concept, layer) series. The effectiveness violations were noise (-0.12, -0.21) and the two
    large D2 violations were the lobotomised velocity L37 alpha=3.0 cell, which leaves the
    valid region entirely once sanity is measured correctly (gate 4). Without monotonicity a
    bisection would be searching a surface that can have the answer on either side of every
    midpoint, and 5 evaluations would be 5 guesses.

    The unsteered end of the bracket is taken as sane BY CONSTRUCTION rather than measured:
    at `r = 0` nothing is injected, `S3 = cap_base / cap_base = 1`, and spending a forward pass
    to confirm that would confirm the instrument, not the cell.

    Returns one row per candidate carrying the SAME `r` selected by Phase 2 plus `r_below` /
    `r_above` for Phase 5. `boundary_lo` / `boundary_hi` are distinct metadata. This
    separation is load-bearing: L58@0.30 was the target cell, while its highest sane dose was
    1.35; using the boundary as `r` silently discarded the answer.
    """
    cfg = _cfg()
    floor = float(cfg["E6_FLOOR"])
    s4_min = float(cfg["S4_MIN"])
    steps = int(cfg["BISECT_STEPS"])
    scan_doses = sorted(float(d) for d in cfg["SCAN_DOSES"])

    # Seed the probe cache from Phase 1 so the two scan doses cost nothing here.
    cache: dict[tuple, dict] = {}
    for row in (scan_rows or []):
        cache[_cell_key(row["layer"], row["r"])] = dict(row, phase=PHASE3)

    print("=" * 78)
    print(f"PHASE 3 - dose bisection on the sanity boundary, {len(candidates)} candidates")
    print("=" * 78)
    if on_plan is not None:
        on_plan(len(candidates))

    out: list[dict] = []
    t0 = time.time()
    for cand in candidates:
        layer = int(cand["layer"])
        selected_r = float(cand["r"])
        history: list[dict] = []

        # --- bracket ------------------------------------------------------------------
        lo_sane = 0.0                # unsteered: sane by construction, see the docstring
        hi_insane: float | None = None
        r_clears_floor: float | None = None
        probes = sorted(set(scan_doses + [selected_r]))
        rung = max(probes)
        for _ in range(BISECT_MAX_PROBES):
            rung = rung * BISECT_ESCALATE
            probes.append(rung)

        for r in probes:
            row = _probe(layer, r, cache)
            sane = _cheap_sane(row, s4_min)
            history.append(dict(stage="bracket", r=float(r), reachable=bool(row["reachable"]),
                                alpha=row["alpha"], s3=row["s3"], reach=row["reach"],
                                d3=row["d3"], sane=sane))
            if sane is None:
                # Unreachable at ALPHA_CEIL: the dose ladder stops here, and the boundary (if
                # any) is above the reachable range. Recorded, never clamped.
                break
            if r_clears_floor is None and row["reach"] is not None and \
                    float(row["reach"]) >= floor:
                r_clears_floor = float(r)
            if sane:
                lo_sane = max(lo_sane, float(r))
            else:
                hi_insane = float(r)
                break

        # --- bisect -------------------------------------------------------------------
        if hi_insane is None:
            # Sanity never failed inside the reachable range. The operating dose is then the
            # highest reachable dose that was measured sane; there is no boundary to refine.
            boundary_lo = None
            boundary_hi = None
            boundary = "not reached: sanity held at every reachable dose"
        else:
            lo, hi = lo_sane, hi_insane
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                row = _probe(layer, mid, cache)
                sane = _cheap_sane(row, s4_min)
                history.append(dict(stage="bisect", r=float(mid),
                                    reachable=bool(row["reachable"]), alpha=row["alpha"],
                                    s3=row["s3"], reach=row["reach"], d3=row["d3"], sane=sane))
                if sane is None:
                    # An unreachable midpoint is not evidence about sanity; treat it as the
                    # upper end so the search stays inside the reachable range.
                    hi = mid
                    continue
                if sane:
                    lo = mid
                    if row["reach"] is not None and float(row["reach"]) >= floor and (
                            r_clears_floor is None or mid < r_clears_floor):
                        r_clears_floor = float(mid)
                else:
                    hi = mid
            boundary_lo = lo
            boundary_hi = hi
            boundary = f"sane up to r={lo:.4f}, first failure at r={hi:.4f}"

        selected = _probe(layer, selected_r, cache)
        selected_sane = _cheap_sane(selected, s4_min)
        has_window = bool(selected_sane is True and selected.get("reach") is not None
                          and float(selected["reach"]) >= floor)
        measured_doses = sorted({float(key[1]) for key in cache if int(key[0]) == layer})
        neighbour_gaps = [abs(value - selected_r) for value in measured_doses
                          if abs(value - selected_r) > 10 ** -6]
        step = (min(neighbour_gaps) if neighbour_gaps
                else max(selected_r * DOSE_STEP_FRAC, BISECT_MIN_R))
        r_below = max(selected_r - step, BISECT_MIN_R)
        r_above = selected_r + step

        row = dict(
            phase=PHASE3, layer=layer, r=selected_r, selected_r=selected_r,
            selected_from_scan=True, r_below=r_below, r_above=r_above,
            step=step, bracket_lo=lo_sane, bracket_hi=hi_insane,
            # The converged endpoints, distinct from the initial bracket above. Gate 4's
            # run-local anchor is boundary_hi: the nearest dose known to fail cheap S3.
            boundary_lo=boundary_lo, boundary_hi=boundary_hi,
            r_clears_floor=r_clears_floor, has_window=has_window, boundary=boundary,
            e6_floor=floor, s4_min=s4_min, bisect_steps=steps,
            n_evaluations=len(history), history=history,
            selected_reach=selected.get("reach"), selected_d3=selected.get("d3"),
            selected_s3=selected.get("s3"),
            selection_kind=cand.get("selection_kind"),
            frontier_distance=cand.get("frontier_distance"),
            nearest_frontier=cand.get("nearest_frontier"),
            why=cand["why"] if "why" in cand else [],
            routes=cand["routes"] if "routes" in cand else [],
            tier=cand["tier"] if "tier" in cand else 0,
            tier_rank=cand["tier_rank"] if "tier_rank" in cand else None,
            tier_ordering=cand["tier_ordering"] if "tier_ordering" in cand else "tier0_routes",
            tier_order_rank=cand["tier_order_rank"] if "tier_order_rank" in cand else None,
            tier_source_layer=(cand["tier_source_layer"]
                               if "tier_source_layer" in cand else layer),
            exhaustive=bool(cand["exhaustive"]) if "exhaustive" in cand else False,
        )
        _append_row(BISECT_FILE, row)
        out.append(row)
        if on_cell is not None:
            on_cell(row)
        print(f"   L{layer:<3} selected r={selected_r:.4f}  (+/- {step:.4f})  "
              f"{'window' if has_window else 'NO WINDOW'}  {boundary}")

    # Preserve tier rank: Phase 2 already decided the evidence order on the proxy surface.
    out.sort(key=lambda row: (not row["has_window"],
                              int(row["tier_rank"] or 10**9),
                              int(row["layer"]), float(row["r"])))
    print(f"phase 3    : {len(out)} candidates in {time.time() - t0:.0f}s, "
          f"{sum(1 for row in out if row['has_window'])} with a window")
    return out


# =====================================================================================
# Phases 4 and 5 - verification and local refinement
# =====================================================================================

def _resid_fit_from_scan() -> dict | None:
    """The D3~E6 fit over the Phase 1 surface, used to label verified rows with `resid`.

    Fitted over the SCAN rows rather than over the handful of verified ones: the residual is
    spec 7.2's search device and is defined against the whole surface it was used to search.
    Refitting it on 10 verified cells would give a different number under the same name.
    """
    xs, ys = [], []
    for row in _read_rows(SCAN_FILE):
        if row["reachable"] and row["reach"] is not None and row["d3"] is not None:
            xs.append(float(row["reach"]))
            ys.append(float(row["d3"]))
    return ols_fit(xs, ys)


def _verify_cells(cells: Sequence[dict], phase: str, *,
                  n_d2: int | None = None,
                  prompt_set: Sequence[dict] | None = None,
                  skip_verified: bool = True,
                  on_cell: Any = None, on_plan: Any = None) -> list[dict]:
    """The shared body of Phase 4 and Phase 5: verify each cell and write `verified.jsonl`.

    Each cell costs one steered task generation, one steered forced-ID generation and 49 judge
    calls, all issued as ONE concurrent batch by `expensive.verify_cell` - E5 and S1 score the
    same twelve responses and fly together, which is what makes the second judge cost tokens
    and not wall time (spec 5.7).

    Two cheap measures are added on top, at the same `(L, r)` and for about two seconds:

      * `reach` and `d3`, so the verified row carries its own judge-free cross-check (spec 9.4)
        and so `resid` can be computed for a dose the scan never visited;
      * nothing else - `s2` and `s3` already come from `verify_cell`, computed on that cell's
        own generations.

    `covertness_margin` is written as `None` here and filled in by `covertness_margin(rows)`
    once the set exists. It is a CROSS-CELL quantity (a fit of D2 against E5 over the verified
    rows) and this file is appended as each row is produced, so no value written at row time
    could be the final one. `operating_point.json` carries the authoritative values.
    """
    from . import expensive          # lazy: expensive.py imports torch at module scope

    fit = _resid_fit_from_scan()
    done = {_cell_key(row["layer"], row["r"]) for row in _read_rows(VERIFIED_FILE)} \
        if skip_verified else set()

    # Same as Phase 1's: the ETA is costed on what will actually be measured, so the count
    # goes out after the resume filter and before the first cell. Counted without consuming
    # `done`, which the loop below still needs in its pre-skip state.
    if on_plan is not None:
        planned, seen = 0, set(done)
        for cell in cells:
            key = _cell_key(int(cell["layer"]), float(cell["r"]))
            if key not in seen:
                seen.add(key)
                planned += 1
        on_plan(planned)

    out: list[dict] = []
    t0 = time.time()
    for cell in cells:
        layer, r = int(cell["layer"]), float(cell["r"])
        key = _cell_key(layer, r)
        if key in done:
            print(f"   L{layer} r={r:.4f}: already verified, skipping")
            continue
        done.add(key)

        try:
            with expensive.phase_scope(phase):
                row = expensive.verify_cell(layer, r, n_d2=n_d2, prompt_set=prompt_set)
                alpha = float(row["alpha"])
                e6 = cheap.measure_E6(layer, alpha)
                d3 = cheap.measure_D3(layer, alpha)
        except config.Unreachable as exc:
            # Recorded, never clamped: a cell measured at ALPHA_CEIL but labelled with the r
            # it asked for is a wrong number rather than a missing one.
            row = dict(phase=phase, layer=layer, r=r, alpha=None, reachable=False,
                       alpha_needed=exc.alpha, alpha_ceil=exc.ceiling,
                       e5=None, e5_min=None, e5_se=None, s1=None, s2=None, s3=None,
                       s4=None, d2=None, d2_se=None, d2_ci_low=None, d2_ci_high=None,
                       n_d2=None, d4=None,
                       usable=False, qualifies=False, resid=None, covertness_margin=None,
                       why=cell["why"] if "why" in cell else [],
                       tier=cell["tier"] if "tier" in cell else 0,
                       tier_rank=cell["tier_rank"] if "tier_rank" in cell else None,
                       tier_ordering=(cell["tier_ordering"] if "tier_ordering" in cell
                                      else "tier0_routes"),
                       tier_order_rank=(cell["tier_order_rank"]
                                        if "tier_order_rank" in cell else None),
                       tier_source_layer=(cell["tier_source_layer"]
                                          if "tier_source_layer" in cell else layer),
                       tier_source_cell=(cell["tier_source_cell"]
                                         if "tier_source_cell" in cell else
                                         dict(layer=layer, r=r)),
                       exhaustive=bool(cell["exhaustive"])
                       if "exhaustive" in cell else False,
                       refined_from=(cell["refined_from"]
                                     if "refined_from" in cell else None))
            _append_row(VERIFIED_FILE, row)
            out.append(row)
            print(f"   L{layer} r={r:.4f}: UNREACHABLE (alpha {exc.alpha:.1f} > "
                  f"{exc.ceiling:g})")
            continue

        resid = None
        if fit is not None and e6["reach"] is not None and d3["d3"] is not None:
            resid = float(d3["d3"]) - predict(fit, float(e6["reach"]))

        row = dict(
            row,
            reach=e6["reach"], reach_se=e6["reach_se"],
            reach_ci_low=e6["reach_ci_low"], reach_ci_high=e6["reach_ci_high"],
            reach_n=e6["reach_n"],
            e6_mass_median=e6["e6_mass_median"], e6_rank_med=e6["e6_rank_med"],
            d3=d3["d3"], d3_se=d3["d3_se"], d3_rate=d3["d3_rate"],
            d3_rate_count=d3["d3_rate_count"], d3_rate_n=d3["d3_rate_n"],
            d3_rate_ci_low=d3["d3_rate_ci_low"],
            d3_rate_ci_high=d3["d3_rate_ci_high"],
            d3_rank_med=d3["d3_rank_med"],
            resid=resid, resid_fit=fit,
            # Filled by covertness_margin(rows); see the docstring. Present-and-null rather
            # than absent so every consumer can index it uniformly (DEBUG LOG pattern 4 cuts
            # both ways - a missing key is as bad as a defaulted number).
            covertness_margin=None,
            why=cell["why"] if "why" in cell else [],
            r_below=cell["r_below"] if "r_below" in cell else None,
            r_above=cell["r_above"] if "r_above" in cell else None,
            step=cell["step"] if "step" in cell else None,
            tier=cell["tier"] if "tier" in cell else 0,
            tier_rank=cell["tier_rank"] if "tier_rank" in cell else None,
            tier_ordering=(cell["tier_ordering"] if "tier_ordering" in cell
                           else "tier0_routes"),
            tier_order_rank=(cell["tier_order_rank"]
                             if "tier_order_rank" in cell else None),
            tier_source_layer=(cell["tier_source_layer"]
                               if "tier_source_layer" in cell else layer),
            tier_source_cell=(cell["tier_source_cell"]
                              if "tier_source_cell" in cell else dict(layer=layer, r=r)),
            exhaustive=bool(cell["exhaustive"]) if "exhaustive" in cell else False,
            refined_from=cell["refined_from"] if "refined_from" in cell else None,
        )
        _append_row(VERIFIED_FILE, row)
        out.append(row)
        print(f"   L{layer} r={r:.3f}: d3={float(row['d3']):.4f} beside "
              f"d2={float(row['d2']):.4f}; rate intervals: reach {row['reach']:.2f} "
              f"[{row['reach_ci_low']:.2f}, {row['reach_ci_high']:.2f}], n={row['reach_n']}; "
              f"D3-rate {row['d3_rate']:.2f} [{row['d3_rate_ci_low']:.2f}, "
              f"{row['d3_rate_ci_high']:.2f}], n={row['d3_rate_n']}; "
              f"S2 {row['s2']:.2f} [{row['s2_ci_low']:.2f}, {row['s2_ci_high']:.2f}], "
              f"n={row['s2_n']}; S3-acc {row['s3_acc']:.2f} "
              f"[{row['s3_acc_ci_low']:.2f}, {row['s3_acc_ci_high']:.2f}], n={row['s3_n']}")
        if on_cell is not None:
            on_cell(row)

    print(f"{phase:<11}: {len(out)} cells in {time.time() - t0:.0f}s")
    return out


def phase4_verify(cells: Sequence[dict], *, on_cell: Any = None,
                  on_plan: Any = None) -> list[dict]:
    """Phase 4. Real E5, S1, S2, S3, D2 and D4 on each shortlisted `(L, r)`. 49 judge calls each.

    This is where the screening stops guessing: E6 was a proxy for effectiveness and D3 a proxy
    for detection, and both are now replaced by the metrics they stood in for. Spec 8 Phase 4
    also fits `D2 ~ E5` and ranks by residual - that fit is `covertness_margin` and it is a
    REPORTED quantity; `select_operating_point` is spec 7.1 and does not use it.

    ~8-10 minutes for 10 cells.
    """
    print("=" * 78)
    print(f"PHASE 4 - verification, {len(cells)} cells x 49 judge calls")
    print("=" * 78)
    return _verify_cells(cells, PHASE4, on_cell=on_cell, on_plan=on_plan)


def phase5_refine(top_cells: Sequence[dict], *, n_top: int = 3,
                  on_cell: Any = None, on_plan: Any = None) -> list[dict]:
    """Phase 5. Around the top cells: layer +/-1 and +/-2 at the same dose, plus one dose step
    either side at the same layer. Full expense.

    **D2 is not reliably monotone in layer**, so the neighbourhood is MEASURED, not assumed.
    That is the whole justification for spending another ~490 judge calls here: "one layer
    above or below" cannot be resolved by interpolation on a surface that is multi-peaked, and
    the M1.5 data has qualifying cells sitting next to non-qualifying ones. Applied to two or
    three candidates, not to the model - refining every shortlisted cell's neighbourhood would
    cost five times Phase 4 and buy resolution where the answer is not.

    Neighbour layers below the depth floor are skipped and SAID to be skipped: `D_MIN` defines
    the search space, and extracting a vector outside it here would quietly widen the space
    that gate 9 re-tests.
    """
    ctx = _run()
    cfg = _cfg()
    floor_layer = layers_in_scope(int(ctx.n_layers), float(cfg["D_MIN"]))
    in_scope = set(floor_layer)

    ranked = sorted(
        [row for row in top_cells if row["e5"] is not None],
        key=lambda row: -float(row["e5"]))[:int(n_top)]

    cells: list[dict] = []
    seen: set[tuple] = set()
    skipped: list[str] = []
    for base in ranked:
        layer, r = int(base["layer"]), float(base["r"])
        step = base["step"] if ("step" in base and base["step"]) else max(
            r * DOSE_STEP_FRAC, BISECT_MIN_R)
        neighbours = [(layer + d, r, f"layer {d:+d} from the L{layer} r={r:.3f} candidate")
                      for d in (-2, -1, 1, 2)]
        neighbours += [(layer, max(r - step, BISECT_MIN_R),
                        f"one dose step below the L{layer} candidate"),
                       (layer, r + step,
                        f"one dose step above the L{layer} candidate")]
        for nlayer, nr, why in neighbours:
            if nlayer not in in_scope:
                skipped.append(f"L{nlayer} (below D_MIN or outside the model)")
                continue
            if nlayer not in ctx.vecs:
                skipped.append(f"L{nlayer} (no extracted vector)")
                continue
            key = _cell_key(nlayer, nr)
            if key in seen:
                continue
            seen.add(key)
            cells.append(dict(
                layer=nlayer, r=nr, why=[why], step=step,
                tier=base["tier"] if "tier" in base else 0,
                tier_rank=base["tier_rank"] if "tier_rank" in base else None,
                tier_ordering=(base["tier_ordering"] if "tier_ordering" in base
                               else "tier0_routes"),
                tier_order_rank=(base["tier_order_rank"]
                                 if "tier_order_rank" in base else None),
                tier_source_layer=(base["tier_source_layer"]
                                   if "tier_source_layer" in base else layer),
                tier_source_cell=(base["tier_source_cell"]
                                  if "tier_source_cell" in base else
                                  dict(layer=layer, r=r)),
                exhaustive=bool(base["exhaustive"]) if "exhaustive" in base else False,
                refined_from=dict(layer=layer, r=r),
            ))

    print("=" * 78)
    print(f"PHASE 5 - local refinement around {len(ranked)} cells -> {len(cells)} neighbours")
    print("=" * 78)
    for note in sorted(set(skipped)):
        print(f"   skipped {note}")
    return _verify_cells(cells, PHASE5, on_cell=on_cell, on_plan=on_plan)


# =====================================================================================
# Phase 6 - confirmation (spec 8 Phase 6)
# =====================================================================================

def phase6_confirm(winner: dict, *, n_confirm: int | None = None) -> dict:
    """The winner re-measured on `prompts.E5_HELDOUT` at `N_CONFIRM`. **No adaptive stopping.**

    **Phases 1-5 are screening: they rank cells and their numbers are not reportable. Only this
    output is** (spec 8). Two structural reasons, both of which would quietly inflate a
    screening number reported as a result:

      * screening chose the operating point BY maximising E5 on `E5_PROMPTS`, so an E5
        re-measured on those same twelve prompts is a fitted value - the winner is the argmax
        of a noisy surface over exactly that set. `E5_HELDOUT` is disjoint from it (asserted at
        import of `prompts.py`), which makes this an out-of-sample estimate;
      * N is fixed at `N_CONFIRM` and set before the measurement. Stopping when the number
        looks settled turns the sample size into a function of the data, and a D2 measured that
        way is biased toward whatever the early trials happened to say.

    Held-out baselines are generated first: every Judge E5 and S1 call is paired against an
    unsteered reference A for the SAME prompt, and Phase 0 only produced those for `E5_PROMPTS`.
    One sample per held-out prompt is enough - samples 2 and 3 exist for the spec 5.8 control
    pairs, which are computed once per concept and do not move with the prompt set.
    """
    from . import expensive          # lazy: expensive.py imports torch at module scope

    cfg = _cfg()
    layer = int(winner["layer"])
    r = float(winner["r"])
    n = int(cfg["N_CONFIRM"]) if n_confirm is None else int(n_confirm)

    print("=" * 78)
    print(f"PHASE 6 - confirmation at L{layer} r={r:.4f}, N_D2={n}, held-out prompts")
    print("=" * 78)

    ids = [row["id"] for row in prompts.E5_HELDOUT]
    reused = _load_unsteered(ids, 1)
    todo = [row for row in prompts.E5_HELDOUT if row["id"] not in reused]
    samples = dict(reused)
    if todo:
        samples.update(_generate_unsteered(todo, 1))
    _install_unsteered(samples)
    print(f"   held-out baselines: {len(samples)} prompts ({len(reused)} reused)")

    t0 = time.time()
    with expensive.phase_scope(PHASE6):
        row = expensive.verify_cell(layer, r, n_d2=n, prompt_set=prompts.E5_HELDOUT)
        alpha = float(row["alpha"])
        # The judge-free cross-check of spec 9.4, at the confirmed cell. Free against the cost
        # of this phase, and it is the only reading here that does not depend on a judge.
        e6 = cheap.measure_E6(layer, alpha)
        d3 = cheap.measure_D3(layer, alpha)

    out = dict(
        row,
        phase=PHASE6, prompt_set="E5_HELDOUT", n_confirm=n, adaptive_stopping=False,
        reach=e6["reach"], reach_se=e6["reach_se"],
        reach_ci_low=e6["reach_ci_low"], reach_ci_high=e6["reach_ci_high"],
        reach_n=e6["reach_n"], e6_mass_median=e6["e6_mass_median"],
        d3=d3["d3"], d3_se=d3["d3_se"], d3_rate=d3["d3_rate"],
        d3_rate_count=d3["d3_rate_count"], d3_rate_n=d3["d3_rate_n"],
        d3_rate_ci_low=d3["d3_rate_ci_low"], d3_rate_ci_high=d3["d3_rate_ci_high"],
        reportable=True,
        secs=round(time.time() - t0, 1),
    )
    _append_row(CONFIRM_FILE, out)
    print(f"phase 6    : E5={out['e5']:.2f} +/- "
          f"{(out['e5_se'] if out['e5_se'] is not None else float('nan')):.2f}   "
          f"S4={out['s4']:.2f}   d3={out['d3']:.4f} beside D2={out['d2']:.3f} "
          f"(95% Wilson [{out['d2_ci_low']:.3f}, {out['d2_ci_high']:.3f}], "
          f"n={out['n_d2']}; unsteered {out['d2_null']:.3f} "
          f"[{out['d2_null_ci_low']:.3f}, {out['d2_null_ci_high']:.3f}])   "
          f"{'QUALIFIES' if out['qualifies'] else 'does not qualify'}")
    return out


# =====================================================================================
# Section 7 - the selection rule, the frontier and the covertness margin
# =====================================================================================

def _has_verdict(row: dict) -> bool:
    """True if this row carries a section 7 verdict, i.e. it was actually verified.

    Membership test on `qualifies` rather than a defaulted read: an unreachable or scan row has
    no verdict, and treating its absence as False would be right by luck here and wrong the
    moment the field is renamed.
    """
    return ("qualifies" in row and row["qualifies"] is not None
            and "e5" in row and row["e5"] is not None
            and "d2" in row and row["d2"] is not None
            and "s4" in row and row["s4"] is not None)


def _selection_constraints(d2_max: float) -> dict[str, float]:
    """The three fixed selection constraints with an explicit D2 ceiling."""
    cfg = _cfg()
    ceiling = float(d2_max)
    if not math.isfinite(ceiling) or not (0.0 <= ceiling <= 1.0):
        raise ValueError(f"D2_MAX must be a finite rate in [0, 1], got {d2_max!r}")
    return dict(E5_FLOOR=float(cfg["E5_FLOOR"]), D2_MAX=ceiling,
                S4_MIN=float(cfg["S4_MIN"]))


def _select_operating_point(rows: Sequence[dict], constraints: dict[str, float]) -> dict:
    """Selection implementation shared by the configured and re-derived verdicts."""
    cfg = _cfg()
    tie_band = float(cfg["E5_TIE_BAND"])
    considered = [row for row in rows if _has_verdict(row)]
    qualifying = [row for row in considered if bool(row["qualifies"])]

    if not qualifying:
        usable = [row for row in considered if "usable" in row and row["usable"]]
        return dict(
            found=False, winner=None, rule="argmax(E5) s.t. qualifies (spec 7.1)",
            n_considered=len(considered), n_usable=len(usable), n_qualifying=0,
            tie_band=tie_band, tied=[], constraints=dict(constraints),
            reason=(f"no cell qualified: of {len(considered)} verified cells {len(usable)} "
                    f"were usable (S4 >= {constraints['S4_MIN']}), none of which cleared both "
                    f"E5 >= {constraints['E5_FLOOR']} and D2 <= {constraints['D2_MAX']}. Run "
                    "the spec 9.3 escalation ladder before concluding the vector is dead."),
        )

    best_e5 = max(float(row["e5"]) for row in qualifying)
    tied = [row for row in qualifying if best_e5 - float(row["e5"]) <= tie_band]
    ranked = sorted(tied, key=lambda row: (float(row["d2"]), -float(row["s4"]),
                                           int(row["layer"]), float(row["r"])))
    winner = ranked[0]
    return dict(
        found=True, winner=winner,
        rule=(f"argmax(E5) over qualifying cells; ties within E5_TIE_BAND={tie_band:g} broken "
              "by lower D2 then higher S4 (spec 7.1). The residual is NOT used"),
        n_considered=len(considered), n_qualifying=len(qualifying),
        best_e5=best_e5, tie_band=tie_band,
        tied=[dict(layer=int(row["layer"]), r=float(row["r"]), e5=float(row["e5"]),
                   d3=(float(row["d3"]) if row.get("d3") is not None else None),
                   d2=float(row["d2"]), s4=float(row["s4"]),
                   phase=row["phase"] if "phase" in row else None)
              for row in ranked],
        constraints=dict(constraints),
    )


def select_operating_point(rows: Sequence[dict]) -> dict:
    """Spec 7.1, exactly: **the answer is the qualifying cell with the highest E5.**

        operating_point = argmax(E5) over cells where qualifies == True

    Constrained maximisation - **effectiveness is the objective; detection and sanity are
    constraints.** A cell with D2 = 0.00 and E5 = 3 does not beat a cell with D2 = 0.18 and
    E5 = 8: both satisfy the constraint, and the second is more effective. Selecting on D2
    instead would systematically pick the cells where nothing is happening.

    Ties within `E5_TIE_BAND` (0.5 on a 0-10 judge scale) are broken by lower D2, then by
    higher sanity. Without the band, selection reads judge noise as a preference; gate 8
    exists to measure how wide that noise actually is.

    **This function must NOT use the residual.** `resid` and `covertness_margin` are spec 7.2
    search and reporting devices: the residual is what widens the Phase 2 shortlist toward
    cells that under-detect for their influence, and once Phase 4 has real E5 and D2 there is
    nothing left for it to do. Ranking the final answer by it would select for the cells where
    the cheap proxies disagreed with each other, which is a property of the proxies.

    Returns a dict with `found`, `winner`, the tie band, and the rule as a string - never a
    bare row, so a caller cannot mistake "no cell qualified" for "the first row won".
    """
    return _select_operating_point(
        rows, _selection_constraints(float(_cfg()["D2_MAX"])))


def reselect_operating_point(rows: Sequence[dict], *, d2_max: float) -> dict:
    """Re-derive selection at one D2 ceiling from stored scalars; measure nothing.

    Only D2 may vary. E5_FLOOR and S4_MIN retain the configured values because loosening
    effectiveness or sanity would change what the result means, not merely where a detection
    constraint is drawn. Copies are used so the stored/configured `qualifies` verdict is never
    mutated and the run's `config_hash` never changes.
    """
    constraints = _selection_constraints(d2_max)
    derived: list[dict] = []
    for original in rows:
        row = dict(original)
        if all(key in row and row[key] is not None for key in ("e5", "d2", "s4")):
            row["usable"] = float(row["s4"]) >= constraints["S4_MIN"]
            row["qualifies"] = bool(
                float(row["e5"]) >= constraints["E5_FLOOR"]
                and float(row["d2"]) <= constraints["D2_MAX"]
                and row["usable"])
        derived.append(row)

    selection = _select_operating_point(derived, constraints)
    return dict(
        analysis_role="screening_reselection",
        primary_analysis=False,
        confirmed=False,
        constraints=constraints,
        selection=selection,
        winner=selection["winner"] if selection["found"] else None,
        frontier=frontier(derived),
    )


def frontier(rows: Sequence[dict]) -> list[dict]:
    """Every qualifying cell with its E5, D2 and S4, highest E5 first.

    **Reported alongside the winner because a single point discards the shape of the
    trade-off** (spec 7.2). Whether an operating point is a knife edge or a plateau is the
    difference between a result that transfers to the harmful arm (gate 10) and one that does
    not, and the winner alone cannot say which it is.

    Every qualifying cell, not a Pareto filter - the spec asks for all of them. `dominated` is
    recorded as a DIAGNOSTIC: a cell is dominated when another qualifying cell is at least as
    effective, at most as detectable and at least as sane, with at least one of those strict.
    It is a label on the row, never a filter applied to the list.
    """
    cells = [row for row in rows if _has_verdict(row) and bool(row["qualifies"])]
    out: list[dict] = []
    for row in cells:
        e5, d2, s4 = float(row["e5"]), float(row["d2"]), float(row["s4"])
        dominated = any(
            (float(other["e5"]) >= e5 and float(other["d2"]) <= d2 and float(other["s4"]) >= s4
             and (float(other["e5"]) > e5 or float(other["d2"]) < d2
                  or float(other["s4"]) > s4))
            for other in cells if other is not row)
        out.append(dict(
            layer=int(row["layer"]), r=float(row["r"]),
            alpha=(float(row["alpha"]) if "alpha" in row and row["alpha"] is not None
                   else None),
            phase=row["phase"] if "phase" in row else None,
            e5=e5, e5_se=row["e5_se"] if "e5_se" in row else None,
            e5_min=row["e5_min"] if "e5_min" in row else None,
            d2=d2, d2_se=row["d2_se"] if "d2_se" in row else None,
            d2_ci_low=row["d2_ci_low"] if "d2_ci_low" in row else None,
            d2_ci_high=row["d2_ci_high"] if "d2_ci_high" in row else None,
            n_d2=row["n_d2"] if "n_d2" in row else None,
            d2_null=row["d2_null"] if "d2_null" in row else None,
            d2_null_ci_low=(row["d2_null_ci_low"] if "d2_null_ci_low" in row else None),
            d2_null_ci_high=(row["d2_null_ci_high"] if "d2_null_ci_high" in row else None),
            d2_null_n=row["d2_null_n"] if "d2_null_n" in row else None,
            reach=row["reach"] if "reach" in row else None,
            reach_ci_low=row["reach_ci_low"] if "reach_ci_low" in row else None,
            reach_ci_high=row["reach_ci_high"] if "reach_ci_high" in row else None,
            reach_n=row["reach_n"] if "reach_n" in row else None,
            # Task 21 / TODO 15: d3 and d2 belong beside each other wherever both exist;
            # otherwise judging the proxy disagreement requires joining two files by hand.
            d3=row["d3"] if "d3" in row else None,
            d3_rate=row["d3_rate"] if "d3_rate" in row else None,
            d3_rate_ci_low=row["d3_rate_ci_low"] if "d3_rate_ci_low" in row else None,
            d3_rate_ci_high=row["d3_rate_ci_high"] if "d3_rate_ci_high" in row else None,
            d3_rate_n=row["d3_rate_n"] if "d3_rate_n" in row else None,
            s4=s4, s1=row["s1"] if "s1" in row else None,
            s2=row["s2"] if "s2" in row else None,
            s2_ci_low=row["s2_ci_low"] if "s2_ci_low" in row else None,
            s2_ci_high=row["s2_ci_high"] if "s2_ci_high" in row else None,
            s2_n=row["s2_n"] if "s2_n" in row else None,
            degenerate_frac=(row["degenerate_frac"] if "degenerate_frac" in row else None),
            degenerate_frac_ci_low=(row["degenerate_frac_ci_low"]
                                    if "degenerate_frac_ci_low" in row else None),
            degenerate_frac_ci_high=(row["degenerate_frac_ci_high"]
                                     if "degenerate_frac_ci_high" in row else None),
            s3=row["s3"] if "s3" in row else None,
            s3_acc=row["s3_acc"] if "s3_acc" in row else None,
            s3_acc_ci_low=row["s3_acc_ci_low"] if "s3_acc_ci_low" in row else None,
            s3_acc_ci_high=row["s3_acc_ci_high"] if "s3_acc_ci_high" in row else None,
            s3_n=row["s3_n"] if "s3_n" in row else None,
            s4_term=row["s4_term"] if "s4_term" in row else None,
            d4=row["d4"] if "d4" in row else None,
            d4_ci=row["d4_ci"] if "d4_ci" in row else None,
            d4_n=row["d4_n"] if "d4_n" in row else None,
            d4_damage_frac=(row["d4_damage_frac"] if "d4_damage_frac" in row else None),
            d4_damage_frac_ci_low=(row["d4_damage_frac_ci_low"]
                                   if "d4_damage_frac_ci_low" in row else None),
            d4_damage_frac_ci_high=(row["d4_damage_frac_ci_high"]
                                    if "d4_damage_frac_ci_high" in row else None),
            d4_retrieval_frac=(row["d4_retrieval_frac"]
                               if "d4_retrieval_frac" in row else None),
            d4_retrieval_frac_ci_low=(row["d4_retrieval_frac_ci_low"]
                                      if "d4_retrieval_frac_ci_low" in row else None),
            d4_retrieval_frac_ci_high=(row["d4_retrieval_frac_ci_high"]
                                       if "d4_retrieval_frac_ci_high" in row else None),
            d4_reading=row["d4_reading"] if "d4_reading" in row else None,
            resid=row["resid"] if "resid" in row else None,
            dominated=dominated,
        ))
    out.sort(key=lambda cell: (-cell["e5"], cell["d2"], cell["layer"]))
    return out


def covertness_margin(rows: Sequence[dict]) -> list[dict]:
    """`d2 - predicted_d2(e5)` from a least-squares fit over the verified rows.

    **REPORTED, never selected on.** E5 and D2 are positively correlated, so the interesting
    quantity is not "which cell has the lowest D2" - that is usually the cell where nothing is
    happening - but "which cell detects LESS than its own influence predicts". A negative
    margin is a cell that is more covert than its effectiveness accounts for, which is the
    phenomenon this project is looking for.

    It is not a selection criterion for the same reason spec 7.2 gives for the Phase 2
    residual: it is defined against a fit over the cells that happened to be measured, so
    ranking by it makes the answer a function of which cells the screening chose. Spec 7.1 is
    the rule; this is a description of the surface around it.

    Returns one record per row that has both an E5 and a D2, carrying the fit alongside the
    margin so a single record is self-describing. When the fit is not defined - fewer than
    three verified cells, or no variation in E5 - every margin is `None` with a stated reason.
    A zero margin would look exactly like a cell that sits on the fit.
    """
    cells = [row for row in rows
             if ("e5" in row and row["e5"] is not None
                 and "d2" in row and row["d2"] is not None)]
    xs = [float(row["e5"]) for row in cells]
    ys = [float(row["d2"]) for row in cells]
    fit = ols_fit(xs, ys)
    reason = None
    if fit is None:
        reason = (f"D2~E5 is not fitted over {len(cells)} verified cells (needs >= 3 cells and "
                  "some variation in E5); the margin is undefined, not zero")

    out: list[dict] = []
    for row, x, y in zip(cells, xs, ys):
        predicted = predict(fit, x) if fit is not None else None
        margin = (y - predicted) if predicted is not None else None
        record = dict(
            layer=int(row["layer"]), r=float(row["r"]),
            phase=row["phase"] if "phase" in row else None,
            e5=x, d3=(row["d3"] if "d3" in row else None), d2=y,
            predicted_d2=predicted, covertness_margin=margin,
            fit=fit, reason=reason,
        )
        # Also fill the field on the caller's own row, so a caller that keeps the verified
        # rows in memory sees the value the JSONL could not carry at write time.
        row["covertness_margin"] = margin
        out.append(record)
    out.sort(key=lambda record: (record["covertness_margin"]
                                 if record["covertness_margin"] is not None else 0.0))
    return out


def write_operating_point(selection: dict, rows: Sequence[dict], *,
                          confirm: dict | None = None,
                          controls: dict | None = None,
                          extra: dict | None = None) -> Path:
    """Write `operating_point.json` - the answer, the frontier and the control verdicts.

    An addition to the CONTRACT's function list, not a replacement for any of it: spec 13 names
    the file and nothing else in the layout order before `runio` composes the winner, the
    frontier and the margins into one document. It is also the only place the covertness
    margins are FINAL - `verified.jsonl` is appended as each row is produced, so a cross-cell
    quantity written at row time could never have been the last word.

    Rates and scalars only. No transcripts, no vectors: `CLAUDE.md` hard rule 3, and spec 13's
    note that this file plus the aggregate rates are the shareable artefacts.
    """
    margins = covertness_margin(rows)
    winner = selection["winner"] if "winner" in selection else None
    payload = dict(
        found=bool(selection["found"]),
        rule=selection["rule"],
        operating_point=(None if winner is None else dict(
            layer=int(winner["layer"]), r=float(winner["r"]),
            alpha=(float(winner["alpha"]) if "alpha" in winner
                   and winner["alpha"] is not None else None),
            phase=winner["phase"] if "phase" in winner else None)),
        screening=(None if winner is None else dict(
            e5=winner["e5"], e5_se=winner["e5_se"] if "e5_se" in winner else None,
            e5_min=winner["e5_min"] if "e5_min" in winner else None,
            s1=winner["s1"] if "s1" in winner else None,
            s2=winner["s2"] if "s2" in winner else None,
            s2_ci_low=winner["s2_ci_low"] if "s2_ci_low" in winner else None,
            s2_ci_high=winner["s2_ci_high"] if "s2_ci_high" in winner else None,
            s2_n=winner["s2_n"] if "s2_n" in winner else None,
            s3=winner["s3"] if "s3" in winner else None,
            s3_acc=winner["s3_acc"] if "s3_acc" in winner else None,
            s3_acc_ci_low=(winner["s3_acc_ci_low"] if "s3_acc_ci_low" in winner else None),
            s3_acc_ci_high=(winner["s3_acc_ci_high"]
                            if "s3_acc_ci_high" in winner else None),
            s3_n=winner["s3_n"] if "s3_n" in winner else None,
            s4=winner["s4"], d2=winner["d2"],
            d2_se=winner["d2_se"] if "d2_se" in winner else None,
            d2_ci_low=winner["d2_ci_low"] if "d2_ci_low" in winner else None,
            d2_ci_high=winner["d2_ci_high"] if "d2_ci_high" in winner else None,
            n_d2=winner["n_d2"] if "n_d2" in winner else None,
            d2_null=winner["d2_null"] if "d2_null" in winner else None,
            d2_null_ci_low=(winner["d2_null_ci_low"]
                            if "d2_null_ci_low" in winner else None),
            d2_null_ci_high=(winner["d2_null_ci_high"]
                             if "d2_null_ci_high" in winner else None),
            d2_null_n=winner["d2_null_n"] if "d2_null_n" in winner else None,
            reach=winner["reach"] if "reach" in winner else None,
            reach_ci_low=winner["reach_ci_low"] if "reach_ci_low" in winner else None,
            reach_ci_high=winner["reach_ci_high"] if "reach_ci_high" in winner else None,
            reach_n=winner["reach_n"] if "reach_n" in winner else None,
            d3_rate=winner["d3_rate"] if "d3_rate" in winner else None,
            d3_rate_ci_low=(winner["d3_rate_ci_low"] if "d3_rate_ci_low" in winner else None),
            d3_rate_ci_high=(winner["d3_rate_ci_high"]
                             if "d3_rate_ci_high" in winner else None),
            d3_rate_n=winner["d3_rate_n"] if "d3_rate_n" in winner else None,
            d4=winner["d4"] if "d4" in winner else None,
            d4_ci=winner["d4_ci"] if "d4_ci" in winner else None,
            d4_n=winner["d4_n"] if "d4_n" in winner else None,
            d4_damage_frac=(winner["d4_damage_frac"]
                            if "d4_damage_frac" in winner else None),
            d4_damage_frac_ci_low=(winner["d4_damage_frac_ci_low"]
                                   if "d4_damage_frac_ci_low" in winner else None),
            d4_damage_frac_ci_high=(winner["d4_damage_frac_ci_high"]
                                    if "d4_damage_frac_ci_high" in winner else None),
            d4_retrieval_frac=(winner["d4_retrieval_frac"]
                               if "d4_retrieval_frac" in winner else None),
            d4_retrieval_frac_ci_low=(winner["d4_retrieval_frac_ci_low"]
                                      if "d4_retrieval_frac_ci_low" in winner else None),
            d4_retrieval_frac_ci_high=(winner["d4_retrieval_frac_ci_high"]
                                       if "d4_retrieval_frac_ci_high" in winner else None),
            note="screening numbers: they ranked cells and are NOT reportable (spec 8)")),
        confirmation=confirm,
        confirmation_note=("Phase 6 on held-out prompts at N_CONFIRM with no adaptive "
                           "stopping. THIS is the reportable measurement (spec 8)"),
        selection=dict(selection, winner=None),      # the winner is above; do not store twice
        frontier=frontier(rows),
        covertness_margins=margins,
        covertness_note=("d2 - predicted_d2(e5) over the verified set. Reported, never "
                         "selected on (spec 7.2)"),
        controls=controls,
        n_verified=len([row for row in rows if _has_verdict(row)]),
        constants={key: _cfg()[key] for key in
                   ("D_MIN", "SCAN_DOSES", "E5_FLOOR", "D2_MAX", "S4_MIN", "E5_TIE_BAND",
                    "N_D2", "N_CONFIRM", "ALPHA_CEIL")},
    )
    if extra:
        payload.update(extra)
    path = _write_json(OPERATING_POINT_FILE, payload)
    print(f"wrote      : {path}")
    return path
