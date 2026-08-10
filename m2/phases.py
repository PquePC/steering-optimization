"""m2.phases - the screening procedure of spec section 8, and the selection rule of section 7.

Phase 0 calibrates, Phase 1 scans every layer cheaply, Phase 2 turns that surface into a
shortlist, Phase 3 finds each candidate's dose, Phases 4 and 5 pay for real E5/S1/D2, and
Phase 6 re-measures the winner on held-out prompts.

  phase0_calibrate       spec 5.1 - vectors, norms, dose map, baselines, cap_base, 2 judge calls
  phase1_scan            every layer with d(L) >= D_MIN, at both SCAN_DOSES. Zero judge calls
  phase2_shortlist       local maxima + stratified coverage + the residual. Never top-K
  phase3_bisect          bracket then bisect the sanity boundary, BISECT_STEPS steps
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

**Selection never touches the residual.** `resid` and `covertness_margin` are search and
reporting devices (spec 7.2); `select_operating_point` implements spec 7.1 and nothing else.
Of 30 usable high-effectiveness M1.5 cells only 7 had D2 <= 0.2, and four of those sat at L31
while effectiveness peaked at L46 for every concept - which is why Phase 2 needs the residual
to widen its shortlist, and exactly why the final choice must not use it.

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


def phase0_calibrate(*, layers: Sequence[int] | None = None, n_unsteered: int = 3,
                     run_liveness: bool = True) -> dict:
    """Spec 5.1. Everything the rest of the pipeline compares against. **2 judge calls.**

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
     11. `judge_fpr` - the two spec 5.8 control-pair calls. **This is the entire judge cost of
         Phase 0**, and it fires here rather than later so that gate 3 can reject an
         influence-inventing judge before GPU time is spent on numbers that carry its floor.

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
          f"(acc {s3_base['s3_acc_base']:.3f})")

    e6_base = cheap.measure_E6(ref, 0.0)
    p_base = {row["prompt_id"]: row["mass"] for row in e6_base["e6_per_prompt"]}
    ctx.base[BASE_P_KEY] = p_base
    print(f"p_base     : median concept mass {e6_base['e6_mass_median']:.3e} over "
          f"{e6_base['e6_n']} prompts (unsteered reach {e6_base['reach']:.2f})")

    d3_base_row = cheap.measure_D3(ref, 0.0)
    ctx.base[BASE_D3_KEY] = d3_base_row["d3"]
    print(f"d3_base    : {d3_base_row['d3']:.4f}"
          + (f" +/- {d3_base_row['d3_se']:.4f}" if d3_base_row["d3_se"] is not None else "")
          + f"   rank median {d3_base_row['d3_rank_med']}")

    # --- 11. the judge false-positive floor (spec 5.8) --------------------------------
    fpr = expensive.judge_fpr()

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
        p_base=p_base, p_base_median=e6_base["e6_mass_median"],
        p_base_reach=e6_base["reach"],
        d3_base=d3_base_row["d3"], d3_base_se=d3_base_row["d3_se"],
        d3_base_rank_med=d3_base_row["d3_rank_med"],
        judge_fpr=fpr, judge_fpr_max=float(cfg["JUDGE_FPR_MAX"]),
        secs=round(time.time() - t0, 1),
    )
    _append_row(BASELINES_FILE, dict(out, measure="baselines"))
    print(f"phase 0    : done in {out['secs']:.0f}s, {n_unreachable}/{len(dose_map)} cells "
          f"unreachable at ALPHA_CEIL (recorded, never clamped)")
    return out


# =====================================================================================
# Phase 1 - the full-depth cheap scan (spec 8 Phase 1)
# =====================================================================================

def phase1_scan(*, layers: Sequence[int] | None = None,
                doses: Sequence[float] | None = None,
                on_cell: Any = None) -> list[dict]:
    """Every layer with `d(L) >= D_MIN`, at both `SCAN_DOSES`. E6 + D3 + S3 + S2. **0 judge calls.**

    **Why TWO doses.** One dose cannot distinguish *"this layer is inert"* from *"this layer is
    under-dosed"* - which is precisely the error a fixed-alpha scan makes, and the reason
    every early-layer cell in the M1.5 grid read flat (at fixed alpha the L6 perturbation was
    0.3% of L37's, because ||v_L|| ran 14 at L6 to 8896 at L46). Two doses give each layer a
    slope as well as a level, and 0.15/0.30 brackets the r range of the seven qualifying M1.5
    cells (0.114-0.303, median 0.180).

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

    t0 = time.time()
    for i, (layer, r) in enumerate(todo, start=1):
        row = cheap.scan_cell(layer, r)
        _append_row(SCAN_FILE, row)
        rows.append(row)
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


def phase2_shortlist(scan_rows: Sequence[dict], *,
                     n_bins: int = STRATIFIED_BINS, write: bool = True) -> list[dict]:
    """Turn the Phase 1 surface into 8-12 candidate LAYERS. Free - no model, no judge.

    Three routes, all three of them, per spec 8 Phase 2:

      1. **all local maxima** of the lightly smoothed E6 curve that clear `E6_FLOOR` and are
         supported at both doses - not just the global max;
      2. a **stratified sample across the E6 range**, because the target cells are outliers
         from the influence<->detection relationship and outliers are not found by looking
         where the surface is highest;
      3. **the residual**: any layer whose D3 is low relative to its E6, i.e. `resid =
         D3 - predicted_D3(E6)` at least `RESID_SIGMA` residual-SDs below the fit. This route
         aims directly at the objective - a cell that under-detects for its influence.

    Candidates within +/-1 layer are merged, keeping the better and unioning the reasons, and
    every candidate records WHY it was selected in `shortlist.json`.

    **NEVER top-K by effectiveness.** Spec 7.2: E5 and D2 are positively correlated, so a
    shortlist built by argmax on the cheap effectiveness proxy systematically walks away from
    the qualifying region. Of 30 usable M1.5 cells with effectiveness > 8, only 7 had
    D2 <= 0.2, and four of those sat at L31 while effectiveness peaked at L46 for EVERY
    concept. A top-K shortlist would have measured L46 ten times and never looked at L31.

    The residual is a search device here and only here. Once Phase 4 has real E5 and D2,
    `select_operating_point` applies spec 7.1 and does not use it.
    """
    cfg = _cfg()
    doses = [float(d) for d in cfg["SCAN_DOSES"]]
    floor = float(cfg["E6_FLOOR"])
    want_lo, want_hi = (int(x) for x in cfg["SHORTLIST_N"])

    per_layer = _by_layer(scan_rows)
    if not per_layer:
        raise RuntimeError(
            "Phase 2 has no reachable scan rows to work from. Either Phase 1 did not run, or "
            "every (layer, r) needed alpha > ALPHA_CEIL - check dose_map.json before "
            "widening the search.")
    layers = sorted(per_layer)
    notes: list[str] = []

    # --- route 1: local maxima of the smoothed reach curve ---------------------------
    raw_reach = [per_layer[layer]["e6"] for layer in layers]
    smoothed = smooth(raw_reach, SMOOTH_WINDOW)
    peak_idx = local_maxima(smoothed)
    if len(peak_idx) > max(3, len(layers) // 3):
        # A curve on which a third of the layers are "local maxima" is flat, and every one of
        # them would be a coincidence rather than a peak. Route 1 is suppressed and SAID to be
        # suppressed; routes 2 and 3 still populate the shortlist.
        notes.append(f"route 1 suppressed: {len(peak_idx)} of {len(layers)} layers read as "
                     "local maxima, so the reach curve is flat and its peaks are noise")
        peak_idx = []
    route1: list[int] = []
    for i in peak_idx:
        layer = layers[i]
        entry = per_layer[layer]
        # Clears the floor, and was reachable at every dose scanned - "supported at both
        # doses" - so a peak that exists only because the other dose was unreachable is out.
        if entry["e6"] >= floor and entry["n_doses"] >= len(doses):
            route1.append(layer)
    if not route1 and not notes:
        notes.append(f"route 1 empty: no local maximum of reach clears E6_FLOOR={floor:g} at "
                     "both doses")

    # --- route 3: the residual (computed before route 2 so route 2 can report it) -----
    xs, ys, cells = [], [], []
    for row in scan_rows:
        if not row["reachable"] or row["reach"] is None or row["d3"] is None:
            continue
        xs.append(float(row["reach"]))
        ys.append(float(row["d3"]))
        cells.append(row)
    fit = ols_fit(xs, ys)
    resid_by_layer: dict[int, float] = {}
    if fit is None:
        notes.append("route 3 unavailable: D3 vs E6 has no fit (fewer than 3 reachable cells, "
                     "or no variation in reach). Residual recorded as null, not as zero")
    else:
        for row, x, y in zip(cells, xs, ys):
            layer = int(row["layer"])
            e = y - predict(fit, x)
            # Most negative across doses: the strongest evidence this layer under-detects.
            if layer not in resid_by_layer or e < resid_by_layer[layer]:
                resid_by_layer[layer] = e
    for layer in layers:
        per_layer[layer]["resid"] = resid_by_layer[layer] if layer in resid_by_layer else None

    route3: list[int] = []
    if fit is not None and resid_by_layer:
        sd = fit["resid_sd"]
        threshold = -RESID_SIGMA * float(sd) if sd else None
        ordered = sorted(resid_by_layer, key=lambda layer: resid_by_layer[layer])
        if threshold is not None:
            route3 = [layer for layer in ordered
                      if resid_by_layer[layer] <= threshold][:RESID_MAX_CANDIDATES]
        if not route3:
            route3 = ordered[:2]
            notes.append("route 3: no layer sits a full sigma below the D3~E6 fit; the two "
                         "most negative residuals are taken so the route still contributes")

    # --- route 2: stratified coverage of the E6 range ---------------------------------
    route2 = [entry["layer"] for entry in
              stratified_pick([per_layer[layer] for layer in layers], "e6", n_bins)]

    # --- assemble, merge within +/-1, then size to SHORTLIST_N ------------------------
    reasons: dict[int, list[str]] = {}
    for layer in route1:
        reasons.setdefault(layer, []).append(
            f"local maximum of smoothed reach ({per_layer[layer]['e6']:.2f}) clearing "
            f"E6_FLOOR={floor:g} at both doses")
    for layer in route2:
        reasons.setdefault(layer, []).append(
            f"stratified coverage of the E6 range (reach {per_layer[layer]['e6']:.2f})")
    for layer in route3:
        resid = resid_by_layer[layer]
        reasons.setdefault(layer, []).append(
            f"residual: D3 sits {resid:+.3f} below its E6-predicted value (spec 7.2 search "
            "device, not a selection criterion)")

    routes = {"local_max": set(route1), "stratified": set(route2), "residual": set(route3)}
    merged = _merge_adjacent(sorted(reasons), per_layer, reasons, routes)
    sized, size_note = _size_shortlist(merged, per_layer, want_lo, want_hi)
    if size_note:
        notes.append(size_note)

    payload = dict(
        phase=PHASE2,
        e6_floor=floor, shortlist_n=[want_lo, want_hi], doses=doses,
        n_layers_scanned=len(layers), n_candidates=len(sized),
        d3_vs_e6_fit=fit, notes=notes,
        routes={name: sorted(members) for name, members in routes.items()},
        candidates=sized,
    )
    if write:
        _write_json(SHORTLIST_FILE, payload)

    print("=" * 78)
    print(f"PHASE 2 - shortlist: {len(sized)} layers from {len(layers)} scanned")
    print("=" * 78)
    for note in notes:
        print(f"   note: {note}")
    print(f"   {'layer':>6} {'reach':>7} {'D3':>7} {'resid':>8}  why")
    for cand in sized:
        entry = per_layer[cand["layer"]]
        resid = entry["resid"]
        print(f"   {'L' + str(cand['layer']):>6} {entry['e6']:>7.2f} "
              f"{(entry['d3'] if entry['d3'] is not None else float('nan')):>7.3f} "
              f"{(resid if resid is not None else float('nan')):>8.3f}  "
              f"{'; '.join(cand['routes'])}")
    return sized


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
    chosen = list(candidates)
    pool = [layer for layer in sorted(per_layer) if layer not in by_layer]
    added: list[int] = []
    while len(chosen) < want_lo and pool:
        have = [float(cand["e6"]) for cand in chosen]

        def gap(layer: int) -> tuple:
            e6 = float(per_layer[layer]["e6"])
            return (-min(abs(e6 - h) for h in have), layer) if have else (0.0, layer)

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
    note = (f"widened to {len(chosen)} candidates by E6 coverage (added "
            f"{['L' + str(layer) for layer in added]})" if added else None)
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
                  scan_rows: Sequence[dict] | None = None) -> list[dict]:
    """Per candidate layer: bracket the sanity boundary, then bisect it. **0 judge calls.**

    1. **Bracket** - the lowest `r` clearing `E6_FLOOR`, and the lowest `r` failing cheap
       sanity. The scan doses are reused for free; above them the probe escalates by
       `BISECT_ESCALATE` until sanity fails or the cell goes unreachable at `ALPHA_CEIL`.
    2. **Bisect** with `BISECT_STEPS` evaluations, giving ~3% resolution in `r`.
    3. Take the point just inside the boundary and **keep one step either side for Phase 4**.

    **Bisection is sound here because sanity is monotone in dose** - 0 violations in 18 M1.5
    (concept, layer) series. The effectiveness violations were noise (-0.12, -0.21) and the two
    large D2 violations were the lobotomised velocity L37 alpha=3.0 cell, which leaves the
    valid region entirely once sanity is measured correctly (gate 4). Without monotonicity a
    bisection would be searching a surface that can have the answer on either side of every
    midpoint, and 5 evaluations would be 5 guesses.

    The unsteered end of the bracket is taken as sane BY CONSTRUCTION rather than measured:
    at `r = 0` nothing is injected, `S3 = cap_base / cap_base = 1`, and spending a forward pass
    to confirm that would confirm the instrument, not the cell.

    Returns one row per candidate carrying `r` (the chosen dose, which Phase 4 verifies) plus
    `r_below` / `r_above` (the step either side, which Phase 5 uses). Phase 4 is costed at ~10
    cells in spec section 12, so the neighbours are carried as metadata rather than verified
    three-deep here.
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

    out: list[dict] = []
    t0 = time.time()
    for cand in candidates:
        layer = int(cand["layer"])
        history: list[dict] = []

        # --- bracket ------------------------------------------------------------------
        lo_sane = 0.0                # unsteered: sane by construction, see the docstring
        hi_insane: float | None = None
        r_clears_floor: float | None = None
        probes = list(scan_doses)
        rung = max(scan_doses)
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
            chosen = lo_sane
            step = max(chosen * DOSE_STEP_FRAC, BISECT_MIN_R)
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
            chosen = lo
            step = hi - lo
            boundary = f"sane up to r={lo:.4f}, first failure at r={hi:.4f}"

        chosen = max(float(chosen), BISECT_MIN_R)
        r_below = max(chosen - step, BISECT_MIN_R)
        r_above = chosen + step
        # A layer whose sanity boundary sits below its effectiveness floor has no window: at
        # every sane dose the concept is not reachable, and at every reachable dose the model
        # is damaged. Flagged rather than dropped, and ordered last, so 49 judge calls are not
        # spent on it before the cells that do have a window.
        has_window = bool(r_clears_floor is not None and chosen >= r_clears_floor)

        row = dict(
            phase=PHASE3, layer=layer, r=chosen, r_below=r_below, r_above=r_above,
            step=step, bracket_lo=lo_sane, bracket_hi=hi_insane,
            r_clears_floor=r_clears_floor, has_window=has_window, boundary=boundary,
            e6_floor=floor, s4_min=s4_min, bisect_steps=steps,
            n_evaluations=len(history), history=history,
            why=cand["why"] if "why" in cand else [],
            routes=cand["routes"] if "routes" in cand else [],
        )
        _append_row(BISECT_FILE, row)
        out.append(row)
        print(f"   L{layer:<3} r={chosen:.4f}  (+/- {step:.4f})  "
              f"{'window' if has_window else 'NO WINDOW'}  {boundary}")

    # Cells with a window first, then by the dose they can carry - so a truncated Phase 4
    # spends its budget where an operating point can exist.
    out.sort(key=lambda row: (not row["has_window"], -float(row["r"]), int(row["layer"])))
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
                  on_cell: Any = None) -> list[dict]:
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
                       s4=None, d2=None, d2_se=None, n_d2=None, d4=None,
                       usable=False, qualifies=False, resid=None, covertness_margin=None)
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
            e6_mass_median=e6["e6_mass_median"], e6_rank_med=e6["e6_rank_med"],
            d3=d3["d3"], d3_se=d3["d3_se"], d3_rate=d3["d3_rate"],
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
        )
        _append_row(VERIFIED_FILE, row)
        out.append(row)
        if on_cell is not None:
            on_cell(row)

    print(f"{phase:<11}: {len(out)} cells in {time.time() - t0:.0f}s")
    return out


def phase4_verify(cells: Sequence[dict], *, on_cell: Any = None) -> list[dict]:
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
    return _verify_cells(cells, PHASE4, on_cell=on_cell)


def phase5_refine(top_cells: Sequence[dict], *, n_top: int = 3,
                  on_cell: Any = None) -> list[dict]:
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
            cells.append(dict(layer=nlayer, r=nr, why=[why], step=step))

    print("=" * 78)
    print(f"PHASE 5 - local refinement around {len(ranked)} cells -> {len(cells)} neighbours")
    print("=" * 78)
    for note in sorted(set(skipped)):
        print(f"   skipped {note}")
    return _verify_cells(cells, PHASE5, on_cell=on_cell)


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
        reach=e6["reach"], reach_se=e6["reach_se"], e6_mass_median=e6["e6_mass_median"],
        d3=d3["d3"], d3_se=d3["d3_se"], d3_rate=d3["d3_rate"],
        reportable=True,
        secs=round(time.time() - t0, 1),
    )
    _append_row(CONFIRM_FILE, out)
    print(f"phase 6    : E5={out['e5']:.2f} +/- "
          f"{(out['e5_se'] if out['e5_se'] is not None else float('nan')):.2f}   "
          f"S4={out['s4']:.2f}   D2={out['d2']:.3f} +/- {out['d2_se']:.3f} (n={out['n_d2']})   "
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
    cfg = _cfg()
    tie_band = float(cfg["E5_TIE_BAND"])
    considered = [row for row in rows if _has_verdict(row)]
    qualifying = [row for row in considered if bool(row["qualifies"])]

    if not qualifying:
        # Not an error: "no operating point exists at these constraints" is a real result, and
        # spec 9.3's escalation ladder is what distinguishes it from "the vector is dead".
        usable = [row for row in considered if "usable" in row and row["usable"]]
        return dict(
            found=False, winner=None, rule="argmax(E5) s.t. qualifies (spec 7.1)",
            n_considered=len(considered), n_usable=len(usable), n_qualifying=0,
            tie_band=tie_band, tied=[],
            reason=(f"no cell qualified: of {len(considered)} verified cells {len(usable)} "
                    f"were usable (S4 >= {float(cfg['S4_MIN'])}), none of which cleared both "
                    f"E5 >= {float(cfg['E5_FLOOR'])} and D2 <= {float(cfg['D2_MAX'])}. Run the "
                    "spec 9.3 escalation ladder before concluding the vector is dead."),
        )

    best_e5 = max(float(row["e5"]) for row in qualifying)
    tied = [row for row in qualifying if best_e5 - float(row["e5"]) <= tie_band]
    # Lower D2, then higher S4. Layer and r are appended purely so the order is total: two
    # cells identical on all three would otherwise resolve by list order, which depends on the
    # order rows were measured in and is not a reason.
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
                   d2=float(row["d2"]), s4=float(row["s4"]),
                   phase=row["phase"] if "phase" in row else None)
              for row in ranked],
        constraints=dict(E5_FLOOR=float(cfg["E5_FLOOR"]), D2_MAX=float(cfg["D2_MAX"]),
                         S4_MIN=float(cfg["S4_MIN"])),
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
            n_d2=row["n_d2"] if "n_d2" in row else None,
            s4=s4, s1=row["s1"] if "s1" in row else None,
            s2=row["s2"] if "s2" in row else None, s3=row["s3"] if "s3" in row else None,
            s4_term=row["s4_term"] if "s4_term" in row else None,
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
            e5=x, d2=y, predicted_d2=predicted, covertness_margin=margin,
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
            s3=winner["s3"] if "s3" in winner else None,
            s4=winner["s4"], d2=winner["d2"],
            d2_se=winner["d2_se"] if "d2_se" in winner else None,
            n_d2=winner["n_d2"] if "n_d2" in winner else None,
            d4=winner["d4"] if "d4" in winner else None,
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
