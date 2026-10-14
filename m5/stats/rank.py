"""m5.stats.rank - rank statistics for the proxy-validation gate and the judges.

Plan §5.3 item 4: a cheap readout is validated against the reference battery by rank correlation
*within strata* (sane / degenerate / dead text) *and within dose bins*, because two quantities that
both rise with dose correlate across dose bins whether or not one tracks the other -- M2's `e6`
(the first-token influence readout) "tracked" judged influence that way and did not (`TODO.md`).
Pass criterion: mean within-stratum correlation >= 0.6 with lower bound >= 0.4 (GATE_RHO_MIN,
GATE_RHO_LB_MIN).

    rankdata(x)                         average ranks (ties share the mean of their positions)
    spearman(x, y)                      Spearman's rho with ties (Pearson on average ranks)
    within_strata_spearman(x, y, s)     rank residuals within each stratum, pooled; plus the
                                        mean of per-stratum correlations
    bootstrap_spearman(...)             percentile interval, resampling units within strata or
                                        whole clusters
    auc(pos, neg, conf)                 Mann-Whitney AUC with DeLong's (1988) interval
    cohen_kappa(r1, r2, conf)           Cohen's (1960) kappa with the Fleiss-Cohen-Everitt (1969)
                                        large-sample standard error
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from m5.stats._checks import as_float_array, check_conf, check_count
from m5.stats.normal import z_two_sided


def rankdata(x: np.ndarray | Sequence[float]) -> np.ndarray:
    """Average ranks, 1-based: tied values get the mean of the ranks they span."""
    a = as_float_array(x, "x")
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(a.size)
    boundaries = np.flatnonzero(np.diff(sorted_a)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [a.size]])
    for s, e in zip(starts.tolist(), ends.tolist()):
        ranks[order[s:e]] = 0.5 * (s + 1 + e)
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    if den == 0.0:
        raise ValueError("a rank correlation is undefined when one variable is constant")
    return float(a @ b) / den


def spearman(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> float:
    """Spearman's rank correlation with average ranks for ties (= scipy.stats.spearmanr's rho)."""
    xv = as_float_array(x, "x", min_len=2)
    yv = as_float_array(y, "y", min_len=2)
    if xv.size != yv.size:
        raise ValueError(f"x and y must pair up, got {xv.size} and {yv.size}")
    return _pearson(rankdata(xv), rankdata(yv))


def within_strata_spearman(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float],
                           strata: Sequence[Any] | np.ndarray) -> dict[str, Any]:
    """Rank association with the strata (dose bin x text stratum) partialled out.

    Within each stratum, x and y are ranked and the ranks rescaled to (rank - 0.5)/n_s - 1/2, so
    every stratum is centred at zero on the same scale; the pooled correlation of these residuals
    is `rho_pooled`, a stratified Spearman correlation in which no part of the association can
    come from differences *between* strata (which is how a shared dose trend fakes agreement).
    `rho_mean` is the size-weighted mean of the per-stratum Spearman correlations, the gate's
    "mean within-stratum rank correlation". Strata with fewer than 3 units or a constant variable
    carry no within-stratum rank information and are skipped (and counted).
    """
    xv = as_float_array(x, "x", min_len=3)
    yv = as_float_array(y, "y", min_len=3)
    labels = np.asarray(strata)
    if not xv.size == yv.size == labels.size:
        raise ValueError("x, y and strata must have the same length")
    _, inv = np.unique(labels, return_inverse=True)
    rx: list[np.ndarray] = []
    ry: list[np.ndarray] = []
    per: list[tuple[int, float]] = []
    skipped = 0
    for g in range(inv.max() + 1):
        idx = np.flatnonzero(inv == g)
        if idx.size < 3 or np.ptp(xv[idx]) == 0 or np.ptp(yv[idx]) == 0:
            skipped += 1
            continue
        a = (rankdata(xv[idx]) - 0.5) / idx.size - 0.5
        b = (rankdata(yv[idx]) - 0.5) / idx.size - 0.5
        rx.append(a)
        ry.append(b)
        per.append((idx.size, _pearson(a, b)))
    if not per:
        raise ValueError("no stratum has 3 or more units with varying x and y")
    ax, ay = np.concatenate(rx), np.concatenate(ry)
    weights = np.array([n for n, _ in per], dtype=float)
    rhos = np.array([r for _, r in per])
    return {"rho_pooled": _pearson(ax, ay), "rho_mean": float(np.sum(weights * rhos) / weights.sum()),
            "n_strata": len(per), "n_skipped": skipped, "n_units": int(ax.size),
            "per_stratum": rhos.tolist()}


def bootstrap_spearman(x: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float], *,
                       strata: Sequence[Any] | np.ndarray | None = None,
                       clusters: Sequence[Any] | np.ndarray | None = None,
                       reps: int, seed: int, conf: float, statistic: str = "rho_mean") -> dict[str, Any]:
    """Percentile bootstrap interval for a (within-strata) Spearman correlation.

    Without `clusters`, units are resampled within each stratum (a stratified bootstrap: every
    resample keeps the strata's sizes). With `clusters` (e.g. the concept or the vector), whole
    clusters are resampled, because cells of one concept are not independent evidence about a
    readout -- the gate's held-out concepts are its unit of generalisation. `statistic` is
    "rho_mean" or "rho_pooled" for stratified data, ignored (plain Spearman) without strata.
    """
    xv = as_float_array(x, "x", min_len=3)
    yv = as_float_array(y, "y", min_len=3)
    if xv.size != yv.size:
        raise ValueError("x and y must have the same length")
    reps = check_count(reps, "reps", minimum=2)
    conf = check_conf(conf)
    if statistic not in ("rho_mean", "rho_pooled"):
        raise ValueError(f"statistic must be 'rho_mean' or 'rho_pooled', got {statistic!r}")
    st = None if strata is None else np.asarray(strata)
    cl = None if clusters is None else np.asarray(clusters)
    for arr, nm in ((st, "strata"), (cl, "clusters")):
        if arr is not None and arr.size != xv.size:
            raise ValueError(f"{nm} must label every unit")

    def stat(idx: np.ndarray) -> float:
        if st is None:
            return spearman(xv[idx], yv[idx])
        return float(within_strata_spearman(xv[idx], yv[idx], st[idx])[statistic])

    estimate = stat(np.arange(xv.size))
    rng = np.random.default_rng(seed)
    if cl is not None:
        _, cinv = np.unique(cl, return_inverse=True)
        members = [np.flatnonzero(cinv == g) for g in range(cinv.max() + 1)]
        if len(members) < 2:
            raise ValueError("a cluster bootstrap needs at least 2 clusters")
    elif st is not None:
        _, sinv = np.unique(st, return_inverse=True)
        members = [np.flatnonzero(sinv == g) for g in range(sinv.max() + 1)]
    draws = []
    failed = 0
    for _ in range(reps):
        if cl is not None:
            pick = rng.integers(0, len(members), size=len(members))
            idx = np.concatenate([members[i] for i in pick])
        elif st is not None:
            idx = np.concatenate([m[rng.integers(0, m.size, size=m.size)] for m in members])
        else:
            idx = rng.integers(0, xv.size, size=xv.size)
        try:
            draws.append(stat(idx))
        except ValueError:  # a resample with no usable stratum, or a constant variable
            failed += 1
    if len(draws) < 2:
        raise ValueError(f"only {len(draws)} of {reps} resamples gave a correlation")
    alpha = 1.0 - conf
    lo, hi = np.quantile(np.asarray(draws), [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"estimate": estimate, "lo": float(lo), "hi": float(hi), "conf": conf, "reps": reps,
            "n_failed": failed}


def auc(scores_pos: np.ndarray | Sequence[float], scores_neg: np.ndarray | Sequence[float],
        conf: float) -> dict[str, float]:
    """Area under the ROC curve, P(pos > neg) + P(pos = neg)/2, with DeLong's interval.

    The Mann-Whitney estimate, with the variance of DeLong, DeLong & Clarke-Pearson (1988,
    Biometrics 44:837-845) from the structural components V10_i = mean_j psi(X_i, Y_j) and
    V01_j = mean_i psi(X_i, Y_j): Var = S10/m + S01/n. The Wald interval is clipped to [0, 1]; at
    an observed AUC of exactly 0 or 1 its variance is zero and the interval degenerates, so read
    a perfect separation together with the two sample sizes.
    """
    x = as_float_array(scores_pos, "scores_pos", min_len=2)
    y = as_float_array(scores_neg, "scores_neg", min_len=2)
    conf = check_conf(conf)
    psi = (x[:, None] > y[None, :]).astype(float) + 0.5 * (x[:, None] == y[None, :])
    a = float(psi.mean())
    v10 = psi.mean(axis=1)
    v01 = psi.mean(axis=0)
    var = float(np.var(v10, ddof=1) / x.size + np.var(v01, ddof=1) / y.size)
    se = math.sqrt(max(var, 0.0))
    z = z_two_sided(conf)
    return {"auc": a, "lo": max(0.0, a - z * se), "hi": min(1.0, a + z * se), "se": se,
            "n_pos": int(x.size), "n_neg": int(y.size), "conf": conf}


def cohen_kappa(r1: Sequence[Any] | np.ndarray, r2: Sequence[Any] | np.ndarray, conf: float) -> dict[str, float]:
    """Cohen's (1960) kappa for two raters on the same items, with a large-sample interval.

    kappa = (p_o - p_e)/(1 - p_e). The standard error is the non-null asymptotic one of Fleiss,
    Cohen & Everitt (1969):
        Var = [sum_i p_ii (1 - (p_i. + p_.i)(1 - k))^2 + (1 - k)^2 sum_{i != j} p_ij (p_.i + p_j.)^2
               - (k - p_e (1 - k))^2] / (n (1 - p_e)^2).
    Undefined (ValueError) when both raters used one identical category throughout: p_e = 1 and
    agreement beyond chance has no meaning. The acceptance bar is kappa >= 0.8 (plan §7.4).
    """
    a = np.asarray(r1)
    b = np.asarray(r2)
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
        raise ValueError("r1 and r2 must be one-dimensional and rate the same items")
    if a.size < 2:
        raise ValueError("kappa needs at least 2 rated items")
    conf = check_conf(conf)
    cats, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    ia, ib = inv[: a.size], inv[a.size:]
    q = cats.size
    table = np.zeros((q, q))
    np.add.at(table, (ia, ib), 1.0)
    n = float(a.size)
    p = table / n
    row = p.sum(axis=1)
    col = p.sum(axis=0)
    p_o = float(np.trace(p))
    p_e = float(row @ col)
    if p_e >= 1.0 - 1e-12:
        raise ValueError("kappa is undefined: both raters used a single, identical category")
    k = (p_o - p_e) / (1.0 - p_e)
    diag = np.diag(p)
    term1 = float(np.sum(diag * (1.0 - (row + col) * (1.0 - k)) ** 2))
    off = p.copy()
    np.fill_diagonal(off, 0.0)
    term2 = (1.0 - k) ** 2 * float(np.sum(off * (col[:, None] + row[None, :]) ** 2))
    term3 = (k - p_e * (1.0 - k)) ** 2
    var = (term1 + term2 - term3) / (n * (1.0 - p_e) ** 2)
    se = math.sqrt(max(var, 0.0))
    z = z_two_sided(conf)
    return {"kappa": k, "lo": max(-1.0, k - z * se), "hi": min(1.0, k + z * se), "se": se,
            "p_observed": p_o, "p_expected": p_e, "n": int(n), "conf": conf}
