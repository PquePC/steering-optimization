"""m5.analysis.coupling - Mentor 1's question, answered from the archive (Phase 0, item 5).

*How tightly does naming track influence, and how often does high naming coexist with low
influence?* If naming (`d2`'s M3 ancestor, forced identification) were only a read-out of how
much the concept fills the text, it would rise and fall with influence everywhere, and gating on
it would add nothing; if it comes apart from influence, it measures something else.

Both quantities rise with dose, so a correlation across all cells mostly says "both rise with
dose" -- the trap M2's `e6` fell into (`m5.stats.rank`). Everything here is therefore computed
**within dose bins**:

- `bin_by="fraction"`: the M3 grid fraction (each layer's dose over its coherence boundary) --
  cells at the same distance from breaking;
- `bin_by="log_dose"`: equal-count bins of log r across layers, as many as the grid has
  fractions -- cells at the same raw strength.

Per concept (and pooled, with concept as part of the stratum), per bin and across bins:

- **Rank coupling**: Spearman's rho between a cell's identification rate and its mean influence,
  within each bin, and the stratified (bin-partialled) correlation `rho_pooled` and the
  size-weighted mean of per-bin correlations `rho_mean` (`m5.stats.within_strata_spearman`), with
  a percentile interval that resamples layers whole (cells of one layer are not independent).
- **Decoupled cells, counted with Wilson intervals**: named-not-influenced (identification >=
  `naming_high` while influence < `influence_low`) -- the case Mentor 1 asked about -- and the
  reverse, influenced-not-named (influence >= `influence_high` while identification <=
  `naming_low`), the shape of an operating window.

The cut points are arguments, never defaults buried here; the CLI passes the documented choice:
naming_high 0.5 (a majority of trials, the half point W* uses), naming_low Q (the detection
tolerance of R*), and M3's own influence rubric for influence: below 1 is "none", 4 and above a
"visible theme" (m3/battery.py _CONCEPT_CUTS).

Unit of analysis: the cell. Within a cell, identify trials and influence prompts are different
generations, so there is no trial-level pairing to exploit. Cells without a valid identification
or influence verdict are excluded and counted, never scored as zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from m5.analysis.m3_archive import M3Archive
from m5.backend.base import derive_seed
from m5.stats import bootstrap_spearman, spearman, wilson, within_strata_spearman

BIN_BY = ("fraction", "log_dose")

# M3's influence rubric (m3/battery.py _CONCEPT_CUTS): below 1 none, 4 and above a visible theme.
# Definitions of the archive's scale, passed by the CLI as the influence cuts.
M3_INFLUENCE_NONE_BELOW = 1.0
M3_INFLUENCE_VISIBLE = 4.0
# A majority of trials: the half point W* uses for detection (plan §6.3).
NAMING_MAJORITY = 0.5


@dataclass(frozen=True)
class CellPoint:
    concept: str
    layer: int
    cell: str
    bin: str
    naming: float
    naming_k: int
    naming_n: int
    influence: float
    influence_n: int


def _points(archive: M3Archive, bin_by: str) -> tuple[list[CellPoint], dict[str, int]]:
    excluded = {"no_d2_verdict": 0, "no_influence_verdict": 0, "no_fraction": 0}
    raw = []
    for c in archive.cells:
        if not c.ident_n:
            excluded["no_d2_verdict"] += 1
            continue
        if c.influence_mean is None:
            excluded["no_influence_verdict"] += 1
            continue
        if bin_by == "fraction" and c.fraction is None:
            excluded["no_fraction"] += 1
            continue
        raw.append(c)
    if bin_by == "fraction":
        bins = [f"f{c.fraction:.2f}" for c in raw]
    else:
        k = max(1, len(archive.dose_fractions) or 6)
        logs = np.log([c.dose for c in raw]) if raw else np.array([])
        if raw:
            edges = np.quantile(logs, np.linspace(0, 1, k + 1)[1:-1])
            idx = np.searchsorted(edges, logs, side="right")
        else:
            idx = np.array([], dtype=int)
        bins = [f"q{int(i) + 1}of{k}" for i in idx]
    pts = [CellPoint(archive.concept, c.layer, c.cell_id, b, c.ident_k / c.ident_n, c.ident_k,
                     c.ident_n, float(c.influence_mean), c.influence_n)  # type: ignore[arg-type]
           for c, b in zip(raw, bins)]
    return pts, excluded


def _count(pts: Sequence[CellPoint], pred: Any, conf: float) -> dict[str, Any]:
    n = len(pts)
    k = sum(1 for p in pts if pred(p))
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None}
    lo, hi = wilson(k, n, conf)
    return {"k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi}


def _within(pts: Sequence[CellPoint], strata: Sequence[str], clusters: Sequence[str], *, reps: int,
            seed: int, conf: float) -> dict[str, Any]:
    """Within-strata rank coupling with a layer-cluster bootstrap; None fields when undefined."""
    out: dict[str, Any] = {"rho_pooled": None, "rho_mean": None, "lo": None, "hi": None,
                           "n_strata": 0, "n_strata_skipped": None, "n_units": len(pts)}
    if len(pts) < 3:
        out["reason"] = "fewer_than_3_cells"
        return out
    x = [p.naming for p in pts]
    y = [p.influence for p in pts]
    try:
        w = within_strata_spearman(x, y, list(strata))
    except ValueError:
        out["reason"] = "no_bin_with_3_varying_cells"
        return out
    out.update(rho_pooled=w["rho_pooled"], rho_mean=w["rho_mean"], n_strata=w["n_strata"],
               n_strata_skipped=w["n_skipped"])
    if len(set(clusters)) >= 2:
        try:
            bs = bootstrap_spearman(x, y, strata=list(strata), clusters=list(clusters), reps=reps,
                                    seed=seed, conf=conf, statistic="rho_mean")
            out.update(lo=bs["lo"], hi=bs["hi"], n_failed_resamples=bs["n_failed"])
        except ValueError:
            out["reason"] = "bootstrap_undefined"
    return out


def coupling(archives: Sequence[M3Archive], cfg: Any, *, bin_by: str, naming_high: float,
             naming_low: float, influence_high: float, influence_low: float) -> list[dict[str, Any]]:
    """The item-5 tables as `analysis` rows (module docstring). Cut points are explicit."""
    if bin_by not in BIN_BY:
        raise ValueError(f"bin_by must be one of {BIN_BY}")
    if not (0.0 <= naming_low < naming_high <= 1.0):
        raise ValueError("need 0 <= naming_low < naming_high <= 1")
    if not (0.0 <= influence_low < influence_high <= 10.0):
        raise ValueError("need 0 <= influence_low < influence_high <= 10 (M3's 0-10 scale)")
    conf = float(cfg["CONFIDENCE"])
    reps = int(cfg["BOOTSTRAP_REPS"])
    base = int(cfg["SEED"])
    cuts = {"naming_high": naming_high, "naming_low": naming_low,
            "influence_high": influence_high, "influence_low": influence_low}

    def named_not_influenced(p: CellPoint) -> bool:
        return p.naming >= naming_high and p.influence < influence_low

    def influenced_not_named(p: CellPoint) -> bool:
        return p.influence >= influence_high and p.naming <= naming_low

    rows: list[dict[str, Any]] = []
    everything: list[CellPoint] = []
    for a in archives:
        pts, excluded = _points(a, bin_by)
        everything.extend(pts)
        bins = sorted({p.bin for p in pts})
        for b in bins:
            sub = [p for p in pts if p.bin == b]
            rho = None
            if len(sub) >= 3 and np.ptp([p.naming for p in sub]) > 0 and \
                    np.ptp([p.influence for p in sub]) > 0:
                rho = spearman([p.naming for p in sub], [p.influence for p in sub])
            rows.append({"kind": "coupling_bin", "concept": a.concept, "bin_by": bin_by, "stratum": b,
                         "n_cells": len(sub), "spearman": rho,
                         "naming_mean": float(np.mean([p.naming for p in sub])),
                         "influence_mean": float(np.mean([p.influence for p in sub])),
                         "named_not_influenced": _count(sub, named_not_influenced, conf),
                         "influenced_not_named": _count(sub, influenced_not_named, conf), **cuts})
        w = _within(pts, [p.bin for p in pts], [f"{p.concept}|L{p.layer}" for p in pts], reps=reps,
                    seed=derive_seed(base, "analysis", "coupling", a.run_id, bin_by), conf=conf)
        rows.append({"kind": "coupling_summary", "concept": a.concept, "bin_by": bin_by,
                     "n_cells": len(pts), "excluded": excluded,
                     "rank_coupling": w, "spearman_across_bins": _unstratified(pts),
                     "named_not_influenced": _count(pts, named_not_influenced, conf),
                     "influenced_not_named": _count(pts, influenced_not_named, conf),
                     "estimand": "cell d2 rate vs cell mean influence (0-10), within dose bins",
                     **cuts})
    if len(archives) > 1 and everything:
        strata = [f"{p.concept}|{p.bin}" for p in everything]
        w = _within(everything, strata, [f"{p.concept}|L{p.layer}" for p in everything], reps=reps,
                    seed=derive_seed(base, "analysis", "coupling", "pooled", bin_by), conf=conf)
        rows.append({"kind": "coupling_summary", "concept": "pooled", "bin_by": bin_by,
                     "n_cells": len(everything), "rank_coupling": w,
                     "spearman_across_bins": _unstratified(everything),
                     "named_not_influenced": _count(everything, named_not_influenced, conf),
                     "influenced_not_named": _count(everything, influenced_not_named, conf),
                     "estimand": "cell d2 rate vs cell mean influence (0-10), within concept x bin",
                     **cuts})
    return rows


def coupling_cuts(cfg: Any) -> dict[str, float]:
    """The documented cut points (module docstring): naming 0.5 and Q, influence M3's rubric."""
    return {"naming_high": NAMING_MAJORITY, "naming_low": float(cfg["Q"]),
            "influence_high": M3_INFLUENCE_VISIBLE, "influence_low": M3_INFLUENCE_NONE_BELOW}


def _unstratified(pts: Sequence[CellPoint]) -> float | None:
    """Spearman across every cell, dose trend included: printed beside the within-bin numbers
    so the share of the association that is only "both rise with dose" can be seen."""
    if len(pts) < 3 or np.ptp([p.naming for p in pts]) == 0 or np.ptp([p.influence for p in pts]) == 0:
        return None
    return spearman([p.naming for p in pts], [p.influence for p in pts])
