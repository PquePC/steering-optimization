"""m5.stats.clustering - how much a clustered trial count is really worth, and the cluster bootstrap.

Plan §6.4: trials are clustered by prompt. Twenty trials on one prompt are not twenty independent
looks at a rate; they are closer to one look repeated, and an interval that counts them as twenty
is too narrow by the square root of the design effect. M3's per-cell intervals ignored this.

    icc_binary(groups)          intra-class correlation of 0/1 outcomes, ANOVA estimator
    icc_from_counts(ks, ns)     the same from per-cluster counts
    design_effect(m, icc)       1 + (m - 1)·ICC (Kish 1965)
    mean_cluster_size(sizes)    the m to use when clusters differ in size: sum(n_i^2) / sum(n_i)
    effective_n(n, m, icc)      n / design effect
    cluster_bootstrap(...)      resample whole clusters (prompts), never single trials

The ANOVA estimator (Fisher; for binary data Ridout, Demetrio & Firth 1999, Biometrics 55:137-148,
who found it among the least biased of twenty estimators) is clipped to [0, 1]: a negative
estimate means "no detectable clustering", and a design effect below 1 would claim that clustering
adds information.

**Undefined is not zero.** When every outcome is identical (all 0 or all 1, the common case for a
detection channel at its floor) the data say nothing about clustering, and `icc_binary` returns
None rather than 0. Returning 0 would set the design effect to 1 exactly where the data cannot
support it; the caller uses an ICC estimated elsewhere (Phase 0, from the M3 per-trial logs).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from m5.stats._checks import as_binary_array, check_conf, check_count, check_real

T = TypeVar("T")


def icc_from_counts(ks: Sequence[float], ns: Sequence[float]) -> float | None:
    """ANOVA intra-class correlation from k_i events in n_i trials per cluster.

    MSB = [sum k_i^2/n_i - (sum k_i)^2/N] / (K - 1), MSW = [sum k_i - sum k_i^2/n_i] / (N - K),
    n0 = [N - sum n_i^2 / N] / (K - 1), ICC = (MSB - MSW) / (MSB + (n0 - 1) MSW), clipped to
    [0, 1]. Returns None when every outcome is identical (no variance to apportion).
    """
    k = np.asarray(ks, dtype=float)
    n = np.asarray(ns, dtype=float)
    if k.ndim != 1 or n.ndim != 1 or k.size != n.size:
        raise ValueError("ks and ns must be one-dimensional and of equal length")
    if k.size < 2:
        raise ValueError(f"an intra-class correlation needs at least 2 clusters, got {k.size}")
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(n))):
        raise ValueError("cluster counts must be finite")
    if np.any(n < 1) or np.any(k < 0) or np.any(k > n):
        raise ValueError("every cluster needs n >= 1 and 0 <= k <= n")
    if np.any(k != np.floor(k)) or np.any(n != np.floor(n)):
        raise ValueError("cluster counts must be whole numbers")
    n_clusters = k.size
    total = n.sum()
    if total - n_clusters <= 0:
        raise ValueError("every cluster has a single trial: within-cluster variation is "
                         "unobservable, so the ICC is not identified")
    events = k.sum()
    if events == 0 or events == total:
        return None
    sum_k2_n = float(np.sum(k * k / n))
    msb = (sum_k2_n - events * events / total) / (n_clusters - 1)
    msw = (events - sum_k2_n) / (total - n_clusters)
    n0 = (total - float(np.sum(n * n)) / total) / (n_clusters - 1)
    denom = msb + (n0 - 1.0) * msw
    if denom <= 0.0:  # pragma: no cover - only reachable with identical outcomes, handled above
        return None
    return float(min(1.0, max(0.0, (msb - msw) / denom)))


def icc_binary(groups: Sequence[Sequence[float]]) -> float | None:
    """ANOVA intra-class correlation of 0/1 outcomes grouped by cluster (prompt).

    `groups` is one sequence of 0/1 outcomes per cluster; missing trials are excluded by the
    caller beforehand. Returns None when every outcome is identical.
    """
    arrays = [as_binary_array(g, f"group {i}") for i, g in enumerate(groups)]
    return icc_from_counts([a.sum() for a in arrays], [a.size for a in arrays])


def design_effect(m_avg: float, icc: float) -> float:
    """Kish's design effect 1 + (m - 1)·ICC: the variance inflation of a clustered mean.

    m is the trials per cluster (for unequal sizes, `mean_cluster_size`). An ICC of 0.1 with 20
    trials per prompt nearly triples the variance (deff 2.9): the 20 trials are worth about 7.
    """
    m_avg = check_real(m_avg, "m_avg")
    icc = check_real(icc, "icc")
    if m_avg < 1.0:
        raise ValueError(f"m_avg must be at least 1 trial per cluster, got {m_avg!r}")
    if not 0.0 <= icc <= 1.0:
        raise ValueError(f"icc must lie in [0, 1], got {icc!r}")
    return 1.0 + (m_avg - 1.0) * icc


def mean_cluster_size(sizes: Sequence[float]) -> float:
    """sum(n_i^2) / sum(n_i): the cluster size that makes 1 + (m - 1)·ICC exact for unequal
    clusters (each trial weighted by the size of its own cluster), not the plain mean."""
    n = np.asarray(sizes, dtype=float)
    if n.ndim != 1 or n.size == 0 or np.any(~np.isfinite(n)) or np.any(n < 1):
        raise ValueError("sizes must be a non-empty list of cluster sizes >= 1")
    return float(np.sum(n * n) / np.sum(n))


def effective_n(n: float, m_avg: float, icc: float) -> float:
    """n / (1 + (m - 1)·ICC): the number of independent trials the clustered n is worth."""
    n = check_real(n, "n")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}")
    return n / design_effect(m_avg, icc)


# ------------------------------------------------------------------------------- the bootstrap

@dataclass(frozen=True, eq=False)
class BootstrapResult:
    """A statistic on the original clusters and its percentile interval over cluster resamples.

    `draws` has one row per successful resample. A statistic may return NaN, or raise ValueError,
    on a degenerate resample (a fit with too few distinct doses); those are counted in `n_failed`,
    not silently dropped. Scalars for a scalar statistic, 1-d arrays for a vector one.
    """

    estimate: np.ndarray | float
    lo: np.ndarray | float
    hi: np.ndarray | float
    se: np.ndarray | float
    draws: np.ndarray
    conf: float
    n_failed: int

    def as_floats(self) -> dict[str, Any]:
        """Scalar statistic as plain floats (for a stored row)."""
        return {"estimate": float(self.estimate), "lo": float(self.lo), "hi": float(self.hi),
                "se": float(self.se), "conf": self.conf, "reps": int(self.draws.shape[0]),
                "n_failed": self.n_failed}


def cluster_bootstrap(clusters: Sequence[T], statistic: Callable[[list[T]], Any], *, reps: int,
                      seed: int, conf: float) -> BootstrapResult:
    """Percentile bootstrap (Efron 1979) resampling whole clusters with replacement.

    `clusters` holds each cluster's data (any object: an array of outcomes, a dict of arrays per
    prompt); `statistic` maps a list of clusters to a scalar or a 1-d array. Resampling prompts,
    not trials, keeps the within-prompt correlation inside every resample, which is what makes the
    interval honest under clustering (Field & Welsh 2007). Seeded: derive `seed` with
    `m5.backend.derive_seed` from the run seed and what the interval is for.
    """
    reps = check_count(reps, "reps", minimum=2)
    conf = check_conf(conf)
    if len(clusters) < 2:
        raise ValueError(f"a cluster bootstrap needs at least 2 clusters, got {len(clusters)}")
    data = list(clusters)
    estimate = np.atleast_1d(np.asarray(statistic(data), dtype=float))
    rng = np.random.default_rng(seed)
    out = np.empty((reps, estimate.size))
    for r in range(reps):
        idx = rng.integers(0, len(data), size=len(data))
        try:
            out[r] = np.atleast_1d(np.asarray(statistic([data[i] for i in idx]), dtype=float))
        except ValueError:  # a degenerate resample (a statistic's own guard tripped)
            out[r] = np.nan
    ok = np.all(np.isfinite(out), axis=1)
    n_failed = int(reps - ok.sum())
    good = out[ok]
    if good.shape[0] < 2:
        raise ValueError(f"only {good.shape[0]} of {reps} resamples gave a finite statistic")
    alpha = 1.0 - conf
    lo = np.quantile(good, alpha / 2.0, axis=0)
    hi = np.quantile(good, 1.0 - alpha / 2.0, axis=0)
    se = good.std(axis=0, ddof=1)
    squeeze = estimate.size == 1
    return BootstrapResult(estimate=estimate[0] if squeeze else estimate,
                           lo=lo[0] if squeeze else lo, hi=hi[0] if squeeze else hi,
                           se=se[0] if squeeze else se, draws=good if not squeeze else good[:, 0],
                           conf=conf, n_failed=n_failed)
