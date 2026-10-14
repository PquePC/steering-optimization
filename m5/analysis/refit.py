"""m5.analysis.refit - the M3 archive re-fitted with M5's statistics (plan §10 Phase 0, item 4).

M3 reported every cell as a Wilson rate and picked cells that cleared a box. M5 summarises each
layer by a dose-response curve (plan §6.1). This module fits that curve to the archive and says
what the archive can and cannot support:

**Per (concept, layer): identification against log dose.** The detection channel M3 measured is
forced identification -- the M3 ancestor of `d2` (forced naming, generated and judged; lower is
better for an operating point): the judge's strict `matches` over identify trials with a verdict
(missing verdicts leave the denominator). Each layer gets the grid posterior of
`m5.stats.psychometric` --

    P(match | u) = floor + (1 - floor - lapse) · F((u - mu) / s),

with beta-binomial overdispersion on the PRIOR_ICC grid -- and reports u_D(q) for every q of
FRONTIER_Q, the location mu, the slope s, floor and lapse, each with its interval. There is no
screen, so the location prior is the wide component alone (PRIOR_WIDE_SD), centred on the middle
of the layer's measured range. The floor's control trials (when asked for) are the null arm's
identify rows scored by M5's literal-mention detector: M3 never judged its null arm (spec §9), so
this is a proxy -- a reply that names the concept mentions it -- and it is labelled as one.

**Coupled across layers** by `m5.stats.LayerChain` (LAYER_AR1, LAYER_JUMP_DF, LAYER_SIGMA): the
heavy-tailed chain lets a layer with little information borrow from its neighbours without
smoothing away a real one-layer switch (the L39 collapse). Both the layer's own fit and the
coupled one are reported.

**Identifiability.** With six doses spanning 0.35-0.85 of each layer's boundary (0.9 in log
units), many layers never rise within the range, or have already risen at its bottom. A threshold
is called identified only when its credible interval lies inside the measured range (to the mu
grid's resolution); otherwise the table says why, in the first matching reason of: no cells,
cells that cannot be placed on the axis (no boundary row, so no fraction), no valid trials, one
dose, non-monotone (posterior predictive check at PPC_ALPHA), above the range (the curve has not
risen by q at the highest dose), below the range (already risen at the lowest), or a bound
beyond the range on one side.

**The ICC of trials within prompts.** An M3 cell has one draw per prompt (30 identify trial
numbers; 4 influence prompts), so within-cell replication does not exist. The prompt effect is
estimated across cells instead, from the two-way crossed layout cells x prompts (cells fixed,
prompts random, no replication): ICC = (MS_prompt - MS_resid) / (MS_prompt + (C - 1) MS_resid),
on the cells where the outcome varies (a cell at 0/30 or 30/30 cannot show a prompt effect) and
the prompts every such cell has. A prompt-by-cell interaction is indistinguishable from noise
with one draw per pair, so this is a lower bound on the within-cell ICC. Its interval resamples
prompts whole (`m5.stats.cluster_bootstrap`). The design effect 1 + (m - 1) ICC is then given for
the cluster sizes M5 will use: trials per trial number at the planned n (TRIAL_NUMBERS).

**The winner's curse** (plan §6.5) re-run with that ICC: the plan's table (best of 25 layers or
150 cells at n = 22, 66, 120) and the archive's own shapes, at p = SPRT_P0 (the plan's table
reproduces at that rate with ICC 0) and at the archive's pooled rate, with ICC 0, the measured
ICC and its upper bound.

**The full-grid R\\*** (plan §6.3), the yardstick the replay is compared against: per layer the
chain-coupled detection draws; an effect curve; integrity at the boundary's first failing dose;
the top of the measured range; combined by `m5.stats.r_star` at every q of FRONTIER_Q, with W*
beside it. The effect is modelled as a rate, not a mean (plan §6.1: steering outcomes are
bimodal): a prompt's reply counts as *presence* when its influence is at least 1 (M3's rubric:
present, at least faintly), and the presence rate gets the same grid posterior as detection, read
on the excess scale E(u) = (1 - floor - lapse) F((u - mu)/s) -- the model `m5.search.curves` uses
for `e8`. An excess curve starts at zero, so a layer already saturated across the measured range
cannot pass its plateau off as the effect at a dose below the range (a free-bottom curve would).

Dose units. `dose_unit="dose"` is M3's r = alpha·||v||/||h|| (the article's numbers);
`"fraction"` is dose / dose_max, the grid M3 actually swept and the unit the replay runs in.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m5.analysis import finite_or_none
from m5.analysis.m3_archive import ArchiveCell, M3Archive
from m5.backend.base import derive_seed
from m5.judges import mechanical
from m5.stats import (
    LayerChain,
    LayerPosterior,
    cluster_bootstrap,
    combine_draws,
    design_effect,
    frontier,
    make_mu_grid,
    psychometric_prior,
    w_star,
    wilson,
    winners_curse_sim,
)

DOSE_UNITS = ("dose", "fraction")
FLOOR_CONTROLS = ("null_literal", "none")

# The plan's winner's-curse table (§6.5): (arms, trials per arm). Definitions of what is re-run,
# not tunables.
PLAN_WINNERS_CURSE_SHAPES = ((25, 22), (150, 22), (25, 66), (25, 120))

INFLUENCE_SCALE = 10.0   # M3's influence is 0-10; per-prompt values are kept on 0-1
# Presence: influence at or above M3's "faint" cut (m3/battery.py _CONCEPT_CUTS[0]); a definition
# of the archive's scale, not a tunable.
PRESENCE_CUT = 1.0


class RefitError(ValueError):
    """The data cannot be fitted as asked (an unknown unit, no layer with data)."""


# =====================================================================================
# The data a fit needs, from an archive or from a replay's requests
# =====================================================================================

@dataclass(frozen=True)
class Point:
    """k matches out of n valid identify trials at log-dose u of one layer."""

    layer: int
    u: float
    k: int
    n: int
    cell: str


@dataclass(frozen=True)
class EffectObs:
    """One influence verdict (scaled to 0-1) of one prompt at log-dose u."""

    layer: int
    u: float
    y: float
    prompt: str
    cell: str


@dataclass
class LayerData:
    layer: int
    points: list[Point] = field(default_factory=list)
    effect: list[EffectObs] = field(default_factory=list)
    u_s: float = math.inf            # integrity threshold (log-dose); +inf = never seen failing
    dose_max: float | None = None
    outcome: str | None = None
    n_failed: int = 0                # identify verdicts that failed (excluded)
    n_off_axis: int = 0              # cells that cannot be placed on this axis (no fraction)
    n_stored_only: int = 0           # cells known only from the run's stored row (no per-prompt data)
    no_cells: str | None = None      # why a layer in the boundaries has no measured cell

    @property
    def doses(self) -> list[float]:
        return sorted({p.u for p in self.points if p.n > 0})

    @property
    def u_lo(self) -> float | None:
        d = self.doses
        return d[0] if d else None

    @property
    def u_hi(self) -> float | None:
        d = self.doses
        return d[-1] if d else None

    @property
    def u_max(self) -> float | None:
        """The highest log-dose measured on any channel at this layer."""
        us = [p.u for p in self.points] + [e.u for e in self.effect]
        return max(us) if us else None

    @property
    def n_trials(self) -> int:
        return sum(p.n for p in self.points)


def log_dose(cell: ArchiveCell, unit: str) -> float | None:
    """The cell's position on the fitted axis: log(r) or log(dose / dose_max)."""
    if unit == "dose":
        return math.log(cell.dose) if cell.dose > 0 else None
    if unit == "fraction":
        return math.log(cell.fraction) if cell.fraction else None
    raise RefitError(f"unknown dose unit {unit!r}; one of {DOSE_UNITS}")


def _u_s(archive: M3Archive, layer: int, unit: str) -> float:
    """Integrity on the fitted axis: the boundary's first failing dose, +inf when none was seen
    (a ceiling- or bracket-limited layer never broke within reach)."""
    b = archive.boundaries.get(layer)
    if b is None or b.lowest_failing_dose is None or not b.lowest_failing_dose > 0:
        return math.inf
    if unit == "dose":
        return math.log(b.lowest_failing_dose)
    if b.dose_max:
        return math.log(b.lowest_failing_dose / b.dose_max)
    return math.inf


def archive_layer_data(archive: M3Archive, *, unit: str) -> dict[int, LayerData]:
    """Every layer the archive knows: its measured points and influence verdicts on the axis,
    and, for a boundary layer without cells, the reason it has none."""
    if unit not in DOSE_UNITS:
        raise RefitError(f"unknown dose unit {unit!r}; one of {DOSE_UNITS}")
    data: dict[int, LayerData] = {}
    for c in archive.cells:
        u = log_dose(c, unit)
        ld = data.setdefault(c.layer, LayerData(c.layer, u_s=_u_s(archive, c.layer, unit),
                                                dose_max=c.dose_max, outcome=c.outcome))
        ld.n_failed += c.ident_failed
        if u is None:
            ld.n_off_axis += 1
            continue
        if c.source == "stored_row":
            ld.n_stored_only += 1
        if c.ident_n:
            ld.points.append(Point(c.layer, u, c.ident_k, c.ident_n, c.cell_id))
        for y, prompt in zip(c.influence, c.influence_units):
            ld.effect.append(EffectObs(c.layer, u, y / INFLUENCE_SCALE, prompt, c.cell_id))
    for layer, b in archive.boundaries.items():
        if layer not in data:
            why = "boundary_" + (b.outcome if b.dose_max is None else "without_cells")
            data[layer] = LayerData(layer, u_s=_u_s(archive, layer, unit), dose_max=b.dose_max,
                                    outcome=b.outcome, no_cells=why)
    return dict(sorted(data.items()))


def null_floor_control(archive: M3Archive) -> tuple[int, int] | None:
    """(k, n): null-arm identify replies that mention the concept literally (M5's detector), out
    of all null identify replies. A proxy for the judged floor M3 never measured (spec §9)."""
    rows = archive.nulls("identify")
    if not rows:
        return None
    k = sum(1 for t in rows if mechanical.literal_mentions(archive.text(t.transcript_id) or "",
                                                           archive.concept))
    return k, len(rows)


# =====================================================================================
# Fitting
# =====================================================================================

def mu_grid_for(data: Mapping[int, LayerData], cfg: Any) -> np.ndarray:
    """The shared mu grid: every measured log-dose (detection and effect), MU_GRID_MARGIN beyond
    each end, at MU_GRID_STEP spacing."""
    if not any(ld.points for ld in data.values()):
        raise RefitError("no layer has a valid identification trial to fit")
    us = [p.u for ld in data.values() for p in ld.points] + \
        [e.u for ld in data.values() for e in ld.effect]
    if not us:
        raise RefitError("no layer has a valid identification trial to fit")
    lo = min(us) - float(cfg["MU_GRID_MARGIN"])
    hi = max(us) + float(cfg["MU_GRID_MARGIN"])
    n = int(round((hi - lo) / float(cfg["MU_GRID_STEP"]))) + 1
    return make_mu_grid(lo, hi, max(n, 3))


def _prior(mu_grid: np.ndarray, center: float, cfg: Any, control: tuple[int, int] | None) -> Any:
    a, b = cfg["PRIOR_FLOOR_LAPSE"]
    med, lsd = cfg["PRIOR_SLOPE"]
    return psychometric_prior(
        mu_grid, center=center, narrow_weight=0.0, narrow_sd=float(cfg["PRIOR_NARROW_SD"]),
        wide_sd=float(cfg["PRIOR_WIDE_SD"]), floor_ab=(a, b), lapse_ab=(a, b), slope_median=med,
        slope_logsd=lsd, icc=tuple(cfg["PRIOR_ICC"]), n_s=int(cfg["PRIOR_S_POINTS"]),
        n_floor=int(cfg["PRIOR_FLOOR_POINTS"]), n_lapse=int(cfg["PRIOR_LAPSE_POINTS"]),
        control=control)


def layer_posterior(ld: LayerData, mu_grid: np.ndarray, cfg: Any,
                    control: tuple[int, int] | None) -> LayerPosterior:
    """The layer's d2 grid posterior under the pre-registered priors (no screen: the location
    prior is the wide component alone, centred on the middle of the measured range)."""
    center = 0.5 * (ld.u_lo + ld.u_hi) if ld.points else float(np.mean(mu_grid))
    post = LayerPosterior(_prior(mu_grid, center, cfg, control))
    for p in ld.points:
        post.update(p.u, p.k, p.n)
    return post


def effect_points(ld: LayerData) -> list[tuple[float, int, int]]:
    """(u, k present, n prompts) per dose from the layer's influence verdicts."""
    by_u: dict[float, list[int]] = {}
    for e in ld.effect:
        acc = by_u.setdefault(e.u, [0, 0])
        acc[0] += int(e.y * INFLUENCE_SCALE >= PRESENCE_CUT - 1e-9)
        acc[1] += 1
    return [(u, k, n) for u, (k, n) in sorted(by_u.items())]


def effect_posterior(ld: LayerData, mu_grid: np.ndarray, cfg: Any) -> LayerPosterior | None:
    """The presence rate's grid posterior (same priors, wide location, no control: M3 never
    judged its null arm, and presence is already scored against the unsteered reply)."""
    pts = effect_points(ld)
    if not pts:
        return None
    center = 0.5 * (pts[0][0] + pts[-1][0])
    post = LayerPosterior(_prior(mu_grid, center, cfg, None))
    for u, k, n in pts:
        post.update(u, k, n)
    return post


def _quantiles(x: np.ndarray, conf: float) -> tuple[float, float, float]:
    alpha = 1.0 - conf
    lo, med, hi = np.quantile(np.asarray(x, dtype=float), [alpha / 2.0, 0.5, 1.0 - alpha / 2.0])
    return float(lo), float(med), float(hi)


def summarise_posterior(post: LayerPosterior, cfg: Any, *, seed: int, unit: str,
                        qs: Sequence[float]) -> dict[str, Any]:
    """Thresholds at every q, location, slope, floor, lapse and overdispersion, with intervals."""
    conf = float(cfg["CONFIDENCE"])
    mu_mean, mu_sd = post.mean_sd_mu()
    draws = post.draws(int(cfg["POSTERIOR_DRAWS"]), seed)
    s_lo, s_med, s_hi = _quantiles(draws["s"], conf)
    thresholds = []
    for q in qs:
        t = post.threshold(q, conf)
        row = {"q": float(q), **{k: finite_or_none(getattr(t, k)) for k in
                                 ("mean", "sd", "median", "lo", "hi")}, "p_never": t.p_never}
        if unit == "dose":
            for k in ("median", "lo", "hi"):
                v = getattr(t, k)
                row[f"dose_{k}"] = finite_or_none(math.exp(v)) if math.isfinite(v) else None
        else:
            for k in ("median", "lo", "hi"):
                v = getattr(t, k)
                row[f"fraction_{k}"] = finite_or_none(math.exp(v)) if math.isfinite(v) else None
        thresholds.append(row)
    return {"mu_mean": mu_mean, "mu_sd": mu_sd, "s_median": s_med, "s_lo": s_lo, "s_hi": s_hi,
            "floor_mean": float(np.mean(draws["floor"])), "lapse_mean": float(np.mean(draws["lapse"])),
            "overdispersion_mean": float(np.mean(draws["icc"])), "thresholds": thresholds}


def identifiability(ld: LayerData, post: LayerPosterior | None, cfg: Any, *, q: float,
                    ppc_p: float | None) -> tuple[bool, str]:
    """(identified, reason): the first reason in the module docstring's order that applies."""
    if ld.no_cells is not None:
        return False, f"no_cells:{ld.no_cells}"
    if not ld.points and ld.n_off_axis:
        return False, "cells_not_on_axis:no_boundary_row"
    if not ld.points or post is None:
        return False, "no_valid_trials"
    if len(ld.doses) < 2:
        return False, "one_dose"
    if ppc_p is not None and ppc_p < float(cfg["PPC_ALPHA"]):
        return False, "non_monotone"
    t = post.threshold(q, float(cfg["CONFIDENCE"]))
    tol = float(cfg["MU_GRID_STEP"])
    lo_r, hi_r = ld.u_lo - tol, ld.u_hi + tol  # type: ignore[operator]
    if not math.isfinite(t.median) or t.median > ld.u_hi:  # type: ignore[operator]
        return False, "above_range"
    if t.median < ld.u_lo:  # type: ignore[operator]
        return False, "below_range"
    if not math.isfinite(t.hi) or t.hi > hi_r:
        return False, "upper_bound_beyond_range"
    if t.lo < lo_r:
        return False, "lower_bound_beyond_range"
    return True, "identified"


@dataclass
class LayerFit:
    layer: int
    data: LayerData
    posterior: LayerPosterior | None
    own: dict[str, Any] | None
    coupled: dict[str, Any] | None
    ppc_p: float | None
    identified: bool
    reason: str


@dataclass
class ConceptFit:
    """Everything `fit_layers` computed for one concept on one axis."""

    concept: str
    unit: str
    layers: dict[int, LayerFit]
    chain_layers: list[int]
    floor_control: str
    control: tuple[int, int] | None
    rstar: list[dict[str, Any]] | None = None
    rstar_requested: bool = False
    wstar: dict[int, dict[str, Any]] | None = None
    excluded_from_rstar: list[dict[str, Any]] = field(default_factory=list)
    curves: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


def fit_layers(data: Mapping[int, LayerData], cfg: Any, *, concept: str, unit: str,
               control: tuple[int, int] | None, floor_control: str, seed_tag: Sequence[Any],
               with_effect: bool, with_ppc: bool = True) -> ConceptFit:
    """Fit every layer with data, couple them with the chain, and (with_effect) build R*."""
    qs = tuple(cfg["FRONTIER_Q"])
    q_primary = float(cfg["Q"])
    conf = float(cfg["CONFIDENCE"])
    base = int(cfg["SEED"])
    fitted = {L: ld for L, ld in data.items() if ld.points}
    if not fitted:
        raise RefitError(f"{concept}: no layer has a valid identification trial to fit")
    grid = mu_grid_for(fitted, cfg)
    out: dict[int, LayerFit] = {}
    posts: dict[int, LayerPosterior] = {}
    for L, ld in data.items():
        if not ld.points:
            ok, why = identifiability(ld, None, cfg, q=q_primary, ppc_p=None)
            out[L] = LayerFit(L, ld, None, None, None, None, ok, why)
            continue
        post = layer_posterior(ld, grid, cfg, control)
        posts[L] = post
        ppc_p = None
        if with_ppc and len(ld.doses) >= 2:
            ppc_p = float(post.ppc_monotone(int(cfg["POSTERIOR_DRAWS"]),
                                            derive_seed(base, *seed_tag, "ppc", L))["p_value"])
        own = summarise_posterior(post, cfg, seed=derive_seed(base, *seed_tag, "own", L), unit=unit, qs=qs)
        ok, why = identifiability(ld, post, cfg, q=q_primary, ppc_p=ppc_p)
        out[L] = LayerFit(L, ld, post, own, None, ppc_p, ok, why)
    chain_layers = sorted(posts)
    chain = LayerChain([posts[L] for L in chain_layers], chain_layers, ar1=float(cfg["LAYER_AR1"]),
                       jump_df=float(cfg["LAYER_JUMP_DF"]), sigma=float(cfg["LAYER_SIGMA"]))
    for i, L in enumerate(chain_layers):
        out[L].coupled = summarise_posterior(chain.coupled(i), cfg,
                                             seed=derive_seed(base, *seed_tag, "chain", L),
                                             unit=unit, qs=qs)
    fit = ConceptFit(concept=concept, unit=unit, layers=out, chain_layers=chain_layers,
                     floor_control=floor_control, control=control)
    for L in chain_layers:
        us = np.array(sorted({p.u for p in out[L].data.points}))
        fit.curves[L] = posts[L].curve(us, conf)
    if with_effect:
        fit.rstar_requested = True
        _rstar(fit, chain, grid, cfg, seed_tag=seed_tag)
    return fit


def _rstar(fit: ConceptFit, chain: LayerChain, grid: np.ndarray, cfg: Any, *,
           seed_tag: Sequence[Any]) -> None:
    """R*(q) over FRONTIER_Q and W*, from chain-coupled joint detection draws and per-layer
    presence posteriors on the excess scale (module docstring). A layer with no per-prompt
    influence verdict is excluded, with its reason."""
    base = int(cfg["SEED"])
    n = int(cfg["POSTERIOR_DRAWS"])
    conf = float(cfg["CONFIDENCE"])
    joint = chain.joint_draws(n, derive_seed(base, *seed_tag, "joint"))
    layers = []
    for i, L in enumerate(fit.chain_layers):
        ld = fit.layers[L].data
        post = effect_posterior(ld, grid, cfg)
        if post is None:
            why = ("no_per_prompt_influence:stored_rows_only" if ld.n_stored_only
                   else "no_influence_verdict")
            fit.excluded_from_rstar.append({"layer": L, "reason": why})
            continue
        e = post.draws(n, derive_seed(base, *seed_tag, "effect", L))
        effect = {"bottom": np.zeros(n), "top": 1.0 - e["floor"] - e["lapse"], "u50": e["mu"],
                  "s": e["s"]}
        layers.append(combine_draws(L, joint[i], effect, ld.u_s, float(ld.u_max), n=n,
                                    seed=derive_seed(base, *seed_tag, "combine", L)))
    if not layers:
        fit.rstar = None
        return
    fit.rstar = frontier(layers, tuple(cfg["FRONTIER_Q"]), conf, float(cfg["DELTA"]))
    fit.wstar = w_star(layers, conf)


# =====================================================================================
# The ICC of trials within prompts, from the crossed cells x prompts layout
# =====================================================================================

def crossed_icc(matrix: np.ndarray) -> float | None:
    """ICC of the column (prompt) effect in a two-way layout without replication.

    `matrix` is cells x prompts, every entry observed. ICC = (MS_p - MS_r)/(MS_p + (C - 1) MS_r),
    clipped to [0, 1]; None when it is undefined (fewer than 2 rows or columns, or no variation).
    Rows are fixed effects (each cell's own rate is removed), columns random.
    """
    y = np.asarray(matrix, dtype=float)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 2 or not np.all(np.isfinite(y)):
        return None
    c, p = y.shape
    grand = y.mean()
    rows = y.mean(axis=1, keepdims=True)
    cols = y.mean(axis=0, keepdims=True)
    ms_p = c * float(np.sum((cols - grand) ** 2)) / (p - 1)
    ms_r = float(np.sum((y - rows - cols + grand) ** 2)) / ((c - 1) * (p - 1))
    denom = ms_p + (c - 1) * ms_r
    if denom <= 0:
        return None
    return float(min(1.0, max(0.0, (ms_p - ms_r) / denom)))


@dataclass(frozen=True)
class PromptICC:
    estimand: str
    icc: float | None
    lo: float | None
    hi: float | None
    n_cells: int
    n_prompts: int
    n_cells_dropped: int
    n_failed_resamples: int | None
    reason: str | None

    def to_row(self) -> dict[str, Any]:
        return {"estimand": self.estimand, "icc": self.icc, "lo": self.lo, "hi": self.hi,
                "n_cells": self.n_cells, "n_prompts": self.n_prompts,
                "n_cells_dropped": self.n_cells_dropped,
                "n_failed_resamples": self.n_failed_resamples, "reason": self.reason}


def prompt_icc(table: Mapping[Hashable, Mapping[str, float]], *, estimand: str, cfg: Any,
               seed: int) -> PromptICC:
    """The crossed ICC over the cells where the outcome varies, on the prompts present in all of
    them, with a bootstrap interval that resamples prompts whole.

    `table` maps a cell to {prompt: outcome} (0/1, or a bounded score); missing verdicts are
    simply absent from a cell's mapping, never zero.
    """
    informative = {c: v for c, v in table.items() if len(v) >= 2 and np.ptp(list(v.values())) > 0}
    dropped = len(table) - len(informative)
    if len(informative) < 2:
        return PromptICC(estimand, None, None, None, len(informative), 0, dropped, None,
                         "fewer_than_2_cells_where_the_outcome_varies")
    counts: dict[str, int] = {}
    for v in informative.values():
        for prompt in v:
            counts[prompt] = counts.get(prompt, 0) + 1
    # A complete layout: the prompt set (all prompts seen at least t times, for some t) and the
    # cells holding every one of them, chosen to keep the most observations. A failed verdict
    # then costs its cell or its prompt, whichever loses less.
    best: tuple[int, int, list[str], list[Hashable]] = (-1, -1, [], [])
    for t in sorted(set(counts.values()), reverse=True):
        ps = sorted(p for p, k in counts.items() if k >= t)
        cs = [c for c, v in informative.items() if all(p in v for p in ps)]
        size = len(cs) * len(ps)
        if (size, len(ps)) > best[:2]:
            best = (size, len(ps), ps, cs)
    prompts, cells = best[2], best[3]
    dropped += len(informative) - len(cells)
    if len(cells) < 2 or len(prompts) < 2:
        return PromptICC(estimand, None, None, None, len(cells), len(prompts), dropped, None,
                         "complete_layout_too_small")
    m = np.array([[informative[c][p] for p in prompts] for c in cells], dtype=float)
    icc = crossed_icc(m)
    if icc is None:
        return PromptICC(estimand, None, None, None, len(cells), len(prompts), dropped, None,
                         "undefined")
    columns = [m[:, j] for j in range(m.shape[1])]

    def stat(chosen: list[np.ndarray]) -> float:
        v = crossed_icc(np.stack(chosen, axis=1))
        if v is None:
            raise ValueError("degenerate resample")
        return v

    try:
        bs = cluster_bootstrap(columns, stat, reps=int(cfg["BOOTSTRAP_REPS"]), seed=seed,
                               conf=float(cfg["CONFIDENCE"]))
        lo, hi, failed = float(bs.lo), float(bs.hi), bs.n_failed
    except ValueError:
        lo = hi = None
        failed = None
    return PromptICC(estimand, icc, lo, hi, len(cells), len(prompts), dropped, failed, None)


def icc_tables(archive: M3Archive) -> dict[str, dict[Hashable, dict[str, float]]]:
    """cells -> {prompt: outcome} for the three channels with prompts: d2 (identify `matches`,
    prompts = trial numbers), d8's ancestor (self_report claims YES) and influence / 10."""
    out: dict[str, dict[Hashable, dict[str, float]]] = {"d2": {}, "d8": {}, "influence": {}}
    for c in archive.cells:
        key = (archive.run_id, c.cell_id)
        d2 = {t.unit: float(t.matches) for t in archive.trials(c, "identify") if t.matches is not None}
        d8 = {t.unit: float(t.claims == "YES") for t in archive.trials(c, "self_report")
              if t.claims is not None}
        infl = {u: y / INFLUENCE_SCALE for u, y in zip(c.influence_units, c.influence)}
        for name, v in (("d2", d2), ("d8", d8), ("influence", infl)):
            if v:
                out[name][key] = v
    return out


def design_effects(icc: float | None, cfg: Any, *, sizes: Sequence[int]) -> list[dict[str, Any]]:
    """1 + (m - 1) ICC at M5's planned n, with m = trials per trial number (TRIAL_NUMBERS)."""
    if icc is None:
        return []
    k = len(tuple(cfg["TRIAL_NUMBERS"]))
    out = []
    for n in sizes:
        m = max(1.0, float(n) / k)
        out.append({"n": int(n), "m": m, "deff": design_effect(m, icc)})
    return out


# =====================================================================================
# The winner's curse, calibrated
# =====================================================================================

def winners_curse_table(*, shapes: Sequence[tuple[int, int]], p_values: Sequence[float],
                        iccs: Sequence[float], trial_numbers: int, reps: int, seed: int,
                        source: Mapping[tuple[int, int], str] | None = None) -> list[dict[str, Any]]:
    """`m5.stats.winners_curse_sim` over arms x trials x rates x ICCs. Trials cluster by trial
    number, ceil(n / trial_numbers) per cluster, as M5's generated channels sample them.

    `reps` is passed in: no config key sets the simulation's replications yet (the CLI uses
    BOOTSTRAP_REPS and says so)."""
    rows = []
    for (arms, n) in shapes:
        m = max(1, int(math.ceil(n / trial_numbers)))
        for p in p_values:
            for icc in iccs:
                wc = winners_curse_sim(arms, n, p, icc, reps, derive_seed(seed, arms, n, p, icc),
                                       cluster_size=m if icc > 0 else None)
                rows.append({"n_arms": arms, "n": n, "p": p, "icc": icc, "cluster_size": m,
                             "inflation": wc.inflation, "se": wc.se, "mean_best": wc.mean_best,
                             "reps": reps,
                             "source": (source or {}).get((arms, n), "plan_table")})
    return rows


# =====================================================================================
# The analysis, as rows
# =====================================================================================

@dataclass
class RefitResult:
    rows: list[dict[str, Any]]
    fits: dict[str, ConceptFit]
    iccs: dict[str, dict[str, PromptICC]]


def fit_concept(archive: M3Archive, cfg: Any, *, unit: str, floor_control: str,
                with_effect: bool, with_ppc: bool = True) -> ConceptFit:
    """`fit_layers` on one archive's data."""
    if floor_control not in FLOOR_CONTROLS:
        raise RefitError(f"floor_control must be one of {FLOOR_CONTROLS}")
    control = null_floor_control(archive) if floor_control == "null_literal" else None
    data = archive_layer_data(archive, unit=unit)
    return fit_layers(data, cfg, concept=archive.concept, unit=unit, control=control,
                      floor_control=floor_control if control is not None else "none",
                      seed_tag=("analysis", "refit", archive.run_id, unit), with_effect=with_effect,
                      with_ppc=with_ppc)


def fit_rows(fit: ConceptFit, cfg: Any) -> list[dict[str, Any]]:
    """The `analysis` rows of one concept's fit: cells, layers (own and coupled), the
    identifiability table, R* and W*."""
    conf = float(cfg["CONFIDENCE"])
    head = {"concept": fit.concept, "dose_unit": fit.unit}
    rows: list[dict[str, Any]] = []
    for L, lf in fit.layers.items():
        curve = fit.curves.get(L)
        for p in sorted(lf.data.points, key=lambda p: p.u):
            lo, hi = wilson(p.k, p.n, conf)
            row = {"kind": "refit_cell", **head, "layer": L, "cell": p.cell, "u": p.u,
                   "d2_k": p.k, "d2_n": p.n, "rate": p.k / p.n, "lo": lo, "hi": hi}
            if curve is not None:
                j = int(np.argmin(np.abs(curve["u"] - p.u)))
                row.update(fit_mean=float(curve["mean"][j]), fit_lo=float(curve["lo"][j]),
                           fit_hi=float(curve["hi"][j]))
            rows.append(row)
        base = {"layer": L, "n_doses": len(lf.data.doses), "n_trials": lf.data.n_trials,
                "d2_failed": lf.data.n_failed, "u_lo": lf.data.u_lo, "u_hi": lf.data.u_hi,
                "dose_max": lf.data.dose_max, "outcome": lf.data.outcome,
                "floor_control": fit.floor_control,
                "control_k": None if fit.control is None else fit.control[0],
                "control_n": None if fit.control is None else fit.control[1]}
        if lf.own is not None:
            rows.append({"kind": "refit_layer", **head, "coupling": "independent", **base,
                         "ppc_p": lf.ppc_p, **lf.own})
        if lf.coupled is not None:
            rows.append({"kind": "refit_layer", **head, "coupling": "chain", **base, **lf.coupled})
        t = None
        if lf.own is not None:
            t = next((r for r in lf.own["thresholds"] if r["q"] == float(cfg["Q"])), None)
        rows.append({"kind": "refit_identifiability", **head, "layer": L,
                     "identifiable": lf.identified, "reason": lf.reason, "q": float(cfg["Q"]),
                     "u_lo": lf.data.u_lo, "u_hi": lf.data.u_hi,
                     "threshold_median": None if t is None else t["median"],
                     "threshold_lo": None if t is None else t["lo"],
                     "threshold_hi": None if t is None else t["hi"],
                     "n_doses": len(lf.data.doses), "n_trials": lf.data.n_trials})
    if fit.rstar is not None:
        for r in fit.rstar:
            rows.append({"kind": "refit_rstar", **head, "q": r["q"], "median": r["median"],
                         "lo": r["lo"], "hi": r["hi"], "mean": r["mean"], "best_layer": r["best_layer"],
                         "p_best": {int(k): v for k, v in r["p_best"].items()}, "flags": r["flags"],
                         "flag_probabilities": r["flag_probabilities"],
                         "per_layer": r["per_layer"], "n_draws": r["n_draws"],
                         "effect_def": "presence_excess",
                         "excluded": fit.excluded_from_rstar})
    elif fit.rstar_requested:
        rows.append({"kind": "refit_rstar", **head, "q": None, "status": "not_computed",
                     "reason": "no layer has an effect curve", "excluded": fit.excluded_from_rstar})
    if fit.wstar is not None:
        for L, w in fit.wstar.items():
            rows.append({"kind": "refit_wstar", **head, "layer": L, **w})
    return rows


def refit(archives: Sequence[M3Archive], cfg: Any, *, unit: str, floor_control: str,
          with_effect: bool, winners_curse_reps: int) -> RefitResult:
    """The whole item-4 refit over one or more concepts; see the module docstring."""
    rows: list[dict[str, Any]] = []
    fits: dict[str, ConceptFit] = {}
    iccs: dict[str, dict[str, PromptICC]] = {}
    conf_sizes = {"d2": (int(cfg["SPRT_NMAX"]), int(cfg["STOP_TRIALS"]), int(cfg["CONFIRM_D2_N"])),
                  "d8": (int(cfg["SPRT_NMAX"]), int(cfg["CONFIRM_D8_N"])), "influence": ()}
    pooled: dict[str, dict[Hashable, dict[str, float]]] = {"d2": {}, "d8": {}, "influence": {}}
    base = int(cfg["SEED"])
    for a in archives:
        try:
            fit = fit_concept(a, cfg, unit=unit, floor_control=floor_control, with_effect=with_effect)
        except RefitError as exc:
            # e.g. a run judged nothing (JUDGE_ENABLED=0): missing, never fitted as zeros
            rows.append({"kind": "refit_skipped", "concept": a.concept, "dose_unit": unit,
                         "reason": "no_valid_d2_trial", "error": str(exc)[:160]})
        else:
            fits[a.run_id] = fit
            rows.extend(fit_rows(fit, cfg))
        tables = icc_tables(a)
        iccs[a.run_id] = {}
        for name, table in tables.items():
            pooled[name].update(table)
            res = prompt_icc(table, estimand=name, cfg=cfg,
                             seed=derive_seed(base, "analysis", "icc", a.run_id, name))
            iccs[a.run_id][name] = res
            rows.append({"kind": "refit_icc", "concept": a.concept, **res.to_row(),
                         "archive_m": 1, "design_effects": design_effects(res.icc, cfg,
                                                                          sizes=conf_sizes[name])})
    if len(archives) > 1:
        iccs["pooled"] = {}
        for name, table in pooled.items():
            res = prompt_icc(table, estimand=name, cfg=cfg,
                             seed=derive_seed(base, "analysis", "icc", "pooled", name))
            iccs["pooled"][name] = res
            rows.append({"kind": "refit_icc", "concept": "pooled", **res.to_row(), "archive_m": 1,
                         "design_effects": design_effects(res.icc, cfg, sizes=conf_sizes[name])})
    rows.extend(winners_curse_rows(archives, iccs, cfg, reps=winners_curse_reps))
    return RefitResult(rows=rows, fits=fits, iccs=iccs)


def winners_curse_rows(archives: Sequence[M3Archive], iccs: Mapping[str, Mapping[str, PromptICC]],
                       cfg: Any, *, reps: int) -> list[dict[str, Any]]:
    """The plan's table and each archive's own shape, at SPRT_P0 and the pooled d2 rate, with ICC
    0, the measured d2 ICC and its upper bound (pooled when several concepts were read)."""
    which = iccs.get("pooled") or (next(iter(iccs.values())) if iccs else {})
    d2 = which.get("d2") if which else None
    icc_values = [0.0]
    for v in ((d2.icc, d2.hi) if d2 is not None else ()):
        if v is not None and 0.0 < v < 1.0 and v not in icc_values:
            icc_values.append(float(v))
    shapes = list(PLAN_WINNERS_CURSE_SHAPES)
    source = {s: "plan_table" for s in shapes}
    k_all = n_all = 0
    for a in archives:
        ns = [c.ident_n for c in a.cells if c.ident_n]
        k_all += sum(c.ident_k for c in a.cells)
        n_all += sum(ns)
        if not ns:
            continue
        n_med = int(np.median(ns))
        for shape in ((len(a.layers), n_med), (len(a.cells), n_med)):
            if shape not in source and shape[0] >= 1 and shape[1] >= 1:
                shapes.append(shape)
                source[shape] = f"archive:{a.run_id}"
    p_values = [float(cfg["SPRT_P0"])]
    if n_all and 0 < k_all < n_all:
        p_values.append(k_all / n_all)
    rows = winners_curse_table(shapes=shapes, p_values=p_values, iccs=icc_values,
                               trial_numbers=len(tuple(cfg["TRIAL_NUMBERS"])), reps=reps,
                               seed=derive_seed(int(cfg["SEED"]), "analysis", "winners_curse"),
                               source=source)
    return [{"kind": "refit_winners_curse", **r} for r in rows]
