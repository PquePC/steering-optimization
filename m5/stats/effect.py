"""m5.stats.effect - monotone dose-response curves for the effect E(u).

Plan §6.1: along log-dose u the effect (`e7`/`e8` concept presence, `e11` trait score) rises.
Two estimators, because each fails where the other does not:

- **Isotonic regression** (pool-adjacent-violators; Ayer et al. 1955, Barlow et al. 1972): the
  least-squares non-decreasing curve, with no shape assumed beyond monotonicity. It cannot
  extrapolate and is a step function, but it cannot be wrong about the shape either.
- **A four-parameter logistic in log-dose** -- the Hill equation (Hill 1910) written in u = log d:

      E(u) = bottom + (top - bottom) · F((u - u50) / s),   F the logistic CDF,

  i.e. E(d) = bottom + (top - bottom) d^h / (d^h + EC50^h) with h = 1/s and EC50 = e^u50. The
  half-plateau dose u_E(1/2) of W* (plan §6.3) is u50: where the effect is halfway from its floor
  to its plateau. Fitted by least squares: for fixed (u50, s) the curve is linear in (bottom, top),
  solved exactly under bottom <= top (and optional bounds) by checking the interior and each edge
  of the feasible region; (u50, log s) are searched on a grid and refined by Nelder-Mead on the
  profiled sum of squares. No scipy needed.

Intervals come from the cluster bootstrap over prompts (`m5.stats.clustering.cluster_bootstrap`):
prompts are the unit the effect varies over, so resampling trials would understate the width.
Where the data are bimodal (McKenzie et al.), fit the rate of concept presence, not a mean score.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from m5.stats._checks import as_float_array, check_conf, check_count, check_prob
from m5.stats.clustering import cluster_bootstrap


# ------------------------------------------------------------------------------- isotonic

def isotonic(y: Sequence[float] | np.ndarray, w: Sequence[float] | np.ndarray | None = None, *,
             increasing: bool = True) -> np.ndarray:
    """Weighted least-squares monotone fit to y in the given order (pool-adjacent-violators).

    Each block of adjacent values that violates the order is replaced by its weighted mean, until
    none does; O(n). With binomial proportions and weights n_i this is also the monotone maximum-
    likelihood fit (Barlow et al. 1972), which is what the psychometric posterior's monotonicity
    check uses.
    """
    yv = as_float_array(y, "y")
    wv = np.ones_like(yv) if w is None else as_float_array(w, "w")
    if wv.size != yv.size:
        raise ValueError(f"w has {wv.size} entries for {yv.size} values")
    if np.any(wv <= 0):
        raise ValueError("weights must be positive")
    if not increasing:
        return -isotonic(-yv, wv, increasing=True)
    means: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for val, wt in zip(yv.tolist(), wv.tolist()):
        means.append(val)
        weights.append(wt)
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            w2 = weights[-2] + weights[-1]
            m2 = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w2
            c2 = counts[-2] + counts[-1]
            del means[-1], weights[-1], counts[-1]
            means[-1], weights[-1], counts[-1] = m2, w2, c2
    return np.repeat(np.asarray(means), counts)


def pool_by_dose(u: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse observations at the same dose to (dose, weighted mean, total weight), dose-sorted."""
    uv = as_float_array(u, "u")
    yv = as_float_array(y, "y")
    wv = np.ones_like(yv) if w is None else as_float_array(w, "w")
    if not uv.size == yv.size == wv.size:
        raise ValueError("u, y and w must have the same length")
    doses, inv = np.unique(uv, return_inverse=True)
    tw = np.bincount(inv, weights=wv, minlength=doses.size)
    ty = np.bincount(inv, weights=wv * yv, minlength=doses.size)
    return doses, ty / tw, tw


def isotonic_by_dose(u: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monotone non-decreasing fit of y against dose: (doses, fitted, weights)."""
    doses, ybar, tw = pool_by_dose(u, y, w)
    return doses, isotonic(ybar, tw), tw


def isotonic_half_dose(u: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float | None:
    """The dose where the isotonic curve first reaches halfway from its lowest to its highest
    value, interpolated linearly between the two doses that bracket it; None when it is flat
    (no rise to locate)."""
    doses, fit, _ = isotonic_by_dose(u, y, w)
    lo, hi = float(fit[0]), float(fit[-1])
    if hi - lo <= 1e-12:
        return None
    half = 0.5 * (lo + hi)
    i = int(np.argmax(fit >= half))
    if i == 0:  # pragma: no cover - fit[0] = lo < half by construction
        return float(doses[0])
    f0, f1 = float(fit[i - 1]), float(fit[i])
    return float(doses[i - 1] + (half - f0) / (f1 - f0) * (doses[i] - doses[i - 1]))


# ------------------------------------------------------------------------------- Hill / 4PL

def expit(z: np.ndarray | float) -> np.ndarray:
    """The logistic CDF F(z) = 1/(1 + e^-z), overflow-free in both tails; the shape of every
    dose-response curve in M5 (effect here, detection in `m5.stats.psychometric`)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass(frozen=True)
class HillFit:
    """E(u) = bottom + (top - bottom) · F((u - u50) / s) in log-dose u; `sse` its weighted SSE."""

    bottom: float
    top: float
    u50: float
    s: float
    sse: float
    n: int

    def __call__(self, u: np.ndarray | float) -> np.ndarray:
        uu = np.asarray(u, dtype=float)
        return self.bottom + (self.top - self.bottom) * expit((uu - self.u50) / self.s)

    def dose_at_fraction(self, f: float) -> float:
        """u where the curve is a fraction f of the way from bottom to top (f = 1/2: u50)."""
        f = check_prob(f, "f", open_interval=True)
        return self.u50 + self.s * math.log(f / (1.0 - f))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _profile(F: np.ndarray, y: np.ndarray, w: np.ndarray, bounds: tuple[float, float] | None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row of F (one (u50, s) pair), the least-squares (bottom, top) with bottom <= top
    and, with `bounds`, lo <= bottom and top <= hi; returns (bottom, top, sse) per row.

    The objective is a convex quadratic in (bottom, top); its constrained minimum is the
    unconstrained one if feasible, else the best of each edge's own minimum clipped to the edge.
    """
    sw = w.sum()
    sf = F @ w
    sff = (F * F) @ w
    sy = float(w @ y)
    sfy = F @ (w * y)
    syy = float(w @ (y * y))

    def sse(b: np.ndarray, t: np.ndarray) -> np.ndarray:
        a = t - b
        return syy - 2 * b * sy - 2 * a * sfy + b * b * sw + 2 * a * b * sf + a * a * sff

    det = sw * sff - sf * sf
    with np.errstate(divide="ignore", invalid="ignore"):
        a_u = np.where(det > 1e-12 * sw * sw, (sw * sfy - sf * sy) / det, 0.0)
    b_u = (sy - a_u * sf) / sw
    t_u = b_u + a_u
    cands_b = []
    cands_t = []
    # flat edge: bottom = top = c
    c = np.full_like(sf, sy / sw)
    if bounds is not None:
        lo, hi = bounds
        c = np.clip(c, lo, hi)
    cands_b.append(c)
    cands_t.append(c)
    feasible_u = a_u >= 0
    if bounds is not None:
        lo, hi = bounds
        feasible_u &= (b_u >= lo) & (t_u <= hi)
        # bottom = lo, amplitude in [0, hi - lo]
        with np.errstate(divide="ignore", invalid="ignore"):
            a_lo = np.where(sff > 0, (sfy - lo * sf) / sff, 0.0)
        a_lo = np.clip(a_lo, 0.0, hi - lo)
        cands_b.append(np.full_like(sf, lo))
        cands_t.append(lo + a_lo)
        # top = hi, bottom in [lo, hi]
        sgg = sw - 2 * sf + sff
        with np.errstate(divide="ignore", invalid="ignore"):
            b_hi = np.where(sgg > 0, (sy - sfy - hi * (sf - sff)) / sgg, hi)
        b_hi = np.clip(b_hi, lo, hi)
        cands_b.append(b_hi)
        cands_t.append(np.full_like(sf, hi))
    bs = np.stack(cands_b)
    ts = np.stack(cands_t)
    sses = sse(bs, ts)
    best = np.argmin(sses, axis=0)
    idx = np.arange(sf.size)
    b_best, t_best, s_best = bs[best, idx], ts[best, idx], sses[best, idx]
    use_u = feasible_u
    b_best = np.where(use_u, b_u, b_best)
    t_best = np.where(use_u, t_u, t_best)
    s_best = np.where(use_u, sse(b_u, t_u), s_best)
    return b_best, t_best, np.maximum(s_best, 0.0)


def _profile_one(f: np.ndarray, y: np.ndarray, w: np.ndarray, bounds: tuple[float, float] | None
                 ) -> float:
    """`_profile` for one (u50, s): the constrained least-squares SSE, in scalars (the refinement
    calls it a hundred times per fit, where the array version's overhead dominates)."""
    wf = w * f
    sw = float(w.sum())
    sf = float(wf.sum())
    sff = float(wf @ f)
    sy = float(w @ y)
    sfy = float(wf @ y)
    syy = float(w @ (y * y))

    def sse(b: float, t: float) -> float:
        a = t - b
        return syy - 2 * b * sy - 2 * a * sfy + b * b * sw + 2 * a * b * sf + a * a * sff

    det = sw * sff - sf * sf
    a_u = (sw * sfy - sf * sy) / det if det > 1e-12 * sw * sw else 0.0
    b_u = (sy - a_u * sf) / sw
    t_u = b_u + a_u
    if bounds is None:
        if a_u >= 0:
            return max(sse(b_u, t_u), 0.0)
        c = sy / sw
        return max(sse(c, c), 0.0)
    lo, hi = bounds
    if a_u >= 0 and b_u >= lo and t_u <= hi:
        return max(sse(b_u, t_u), 0.0)
    c = min(max(sy / sw, lo), hi)
    best = sse(c, c)
    a_lo = min(max((sfy - lo * sf) / sff, 0.0), hi - lo) if sff > 0 else 0.0
    best = min(best, sse(lo, lo + a_lo))
    sgg = sw - 2 * sf + sff
    b_hi = min(max((sy - sfy - hi * (sf - sff)) / sgg, lo), hi) if sgg > 0 else hi
    best = min(best, sse(b_hi, hi))
    return max(best, 0.0)


def _nelder_mead(f: Callable[[np.ndarray], float], x0: np.ndarray, step: np.ndarray, *,
                 max_iter: int = 400, tol: float = 1e-10) -> np.ndarray:
    """Plain Nelder-Mead (1965) with the standard coefficients; small and dependency-free."""
    n = x0.size
    simplex = [x0.astype(float)]
    for i in range(n):
        v = x0.astype(float).copy()
        v[i] += step[i]
        simplex.append(v)
    vals = [f(v) for v in simplex]
    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + tol):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
    return simplex[int(np.argmin(vals))]


def fit_hill(u: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, *,
             bounds: tuple[float, float] | None = None, n_u50: int = 41, n_s: int = 25) -> HillFit:
    """Least-squares four-parameter logistic in log-dose, monotone non-decreasing.

    u: log-doses (repeats allowed: one row per prompt and dose); y: effect values; w: weights
    (e.g. trials behind each value). `bounds=(lo, hi)` keeps bottom and top inside a range (0, 1
    for rates). u50 is searched over the observed range widened by half its width on each side
    (beyond that the data cannot locate it), s from 1/50 of the range to twice the range. Needs at
    least three distinct doses: two points fit any sigmoid.
    """
    uv = as_float_array(u, "u")
    yv = as_float_array(y, "y")
    wv = np.ones_like(yv) if w is None else as_float_array(w, "w")
    if not uv.size == yv.size == wv.size:
        raise ValueError("u, y and w must have the same length")
    if np.any(wv <= 0):
        raise ValueError("weights must be positive")
    if np.unique(uv).size < 3:
        raise ValueError("a four-parameter curve needs at least three distinct doses")
    if bounds is not None:
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo < hi:
            raise ValueError(f"bounds must be (lo, hi) with lo < hi, got {bounds!r}")
        bounds = (lo, hi)
    n_u50 = check_count(n_u50, "n_u50", minimum=3)
    n_s = check_count(n_s, "n_s", minimum=3)
    umin, umax = float(uv.min()), float(uv.max())
    span = umax - umin
    u50_lo, u50_hi = umin - 0.5 * span, umax + 0.5 * span
    ls_lo, ls_hi = math.log(span / 50.0), math.log(2.0 * span)
    g_u50 = np.linspace(u50_lo, u50_hi, n_u50)
    g_ls = np.linspace(ls_lo, ls_hi, n_s)
    U50, LS = np.meshgrid(g_u50, g_ls, indexing="ij")
    U50, LS = U50.ravel(), LS.ravel()
    F = expit((uv[None, :] - U50[:, None]) / np.exp(LS)[:, None])
    _, _, sse = _profile(F, yv, wv, bounds)
    j = int(np.argmin(sse))

    def objective(x: np.ndarray) -> float:
        u50 = min(max(x[0], u50_lo), u50_hi)
        ls = min(max(x[1], ls_lo), ls_hi)
        return _profile_one(expit((uv - u50) / math.exp(ls)), yv, wv, bounds)

    x = _nelder_mead(objective, np.array([U50[j], LS[j]]),
                     np.array([g_u50[1] - g_u50[0], g_ls[1] - g_ls[0]]))
    u50 = min(max(float(x[0]), u50_lo), u50_hi)
    ls = min(max(float(x[1]), ls_lo), ls_hi)
    if objective(np.array([u50, ls])) > float(sse[j]):  # refinement never makes it worse
        u50, ls = float(U50[j]), float(LS[j])
    Fx = expit((uv - u50) / math.exp(ls))[None, :]
    b, t, s = _profile(Fx, yv, wv, bounds)
    return HillFit(bottom=float(b[0]), top=float(t[0]), u50=u50, s=math.exp(ls), sse=float(s[0]),
                   n=int(uv.size))


def half_plateau_dose(fit: HillFit) -> float | None:
    """u_E(1/2) of W* (plan §6.3): the log-dose where the effect is halfway from floor to plateau.

    None for a flat fit (top = bottom): an effect that never rises has no half-plateau dose, and
    the u50 of a flat curve is an arbitrary number the optimiser stopped at.
    """
    if fit.top - fit.bottom <= 1e-12:
        return None
    return fit.u50


# ------------------------------------------------------------------------------- the bootstrap

@dataclass(frozen=True, eq=False)
class EffectBootstrap:
    """A Hill fit and its prompt-bootstrap draws.

    `draws` maps bottom, top, u50 and s to arrays with one entry per successful resample (aligned:
    entry i of each comes from the same resample, so they can be used jointly, e.g. as the effect
    draws of `m5.stats.summaries.LayerDraws`). `intervals` gives the percentile interval of each.
    """

    fit: HillFit
    draws: dict[str, np.ndarray]
    intervals: dict[str, tuple[float, float]]
    conf: float
    n_failed: int


def bootstrap_hill(u: np.ndarray, y: np.ndarray, clusters: Sequence[Any] | np.ndarray, *, reps: int,
                   seed: int, conf: float, w: np.ndarray | None = None,
                   bounds: tuple[float, float] | None = None) -> EffectBootstrap:
    """Fit the Hill curve and its cluster-bootstrap distribution, resampling prompts.

    `clusters` labels each observation with its prompt; every resample draws prompts with
    replacement and keeps all of a drawn prompt's doses together.
    """
    uv = as_float_array(u, "u")
    yv = as_float_array(y, "y")
    wv = np.ones_like(yv) if w is None else as_float_array(w, "w")
    labels = np.asarray(clusters)
    if not uv.size == yv.size == wv.size == labels.size:
        raise ValueError("u, y, w and clusters must have the same length")
    conf = check_conf(conf)
    fit = fit_hill(uv, yv, wv, bounds=bounds)
    _, inv = np.unique(labels, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]

    def stat(chosen: list[np.ndarray]) -> np.ndarray:
        idx = np.concatenate(chosen)
        f = fit_hill(uv[idx], yv[idx], wv[idx], bounds=bounds)
        return np.array([f.bottom, f.top, f.u50, f.s])

    res = cluster_bootstrap(groups, stat, reps=reps, seed=seed, conf=conf)
    names = ("bottom", "top", "u50", "s")
    draws = {nm: res.draws[:, i] for i, nm in enumerate(names)}
    intervals = {nm: (float(res.lo[i]), float(res.hi[i])) for i, nm in enumerate(names)}
    return EffectBootstrap(fit=fit, draws=draws, intervals=intervals, conf=conf, n_failed=res.n_failed)
