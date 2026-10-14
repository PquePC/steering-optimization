"""m5.stats.selection - what selecting the best cell does to its number, and how to certify it anyway.

Plan §6.5. A search that measures many noisy cells and reports the best one reports partly the
cell that was good and partly the cell that got lucky (M3-DESIGN, "the winner's curse"). Three
tools:

    learn_then_test_fixed_sequence(p, alpha)   certify an operating point with family-wise error
                                               control (Angelopoulos, Bates, Candes, Jordan & Lei
                                               2021, Learn then Test, arXiv 2110.01052)
    binomial_tail_pvalue(k, n, threshold)      the exact p-value for a claim "rate <= threshold"
    eb_shrink(ks, ns, conf)                    empirical-Bayes beta-binomial shrinkage of many rates
    winners_curse_sim(...)                     how much the best of m equal arms is inflated

The size of the bias, from the statistics review's simulation with every layer truly equal:
best of 25 layers at n = 22 is inflated by +14.9 points, best of 150 cells by +20.3, best of 25 at
n = 66 by +8.5 and at n = 120 by +6.2. `winners_curse_sim` reproduces that order of magnitude and
is to be re-run with the ICC measured in Phase 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from m5.stats._checks import check_conf, check_count, check_counts, check_prob
from m5.stats.beta import betaincinv, binom_cdf
from m5.stats.clustering import icc_from_counts


def binomial_tail_pvalue(k: int, n: int, threshold: float) -> float:
    """Exact p-value for H0: rate >= threshold against the claim rate < threshold.

    p = P(X <= k) for X ~ Binomial(n, threshold): how often a cell whose true rate were exactly
    at the threshold would show as few events as this one did (higher true rates show fewer such
    outcomes, so the null's boundary is its worst case). Valid for any n, including k = 0, where
    it is (1 - threshold)^n.
    """
    k, n = check_counts(k, n, integer=True)
    threshold = check_prob(threshold, "threshold", open_interval=True)
    return binom_cdf(k, n, threshold)


@dataclass(frozen=True)
class FixedSequenceResult:
    """Which hypotheses fixed-sequence testing rejected, in the order given.

    `rejected[i]` is True for the certified prefix; `n_certified` its length; `first_failure` the
    index where testing stopped (None if every hypothesis was rejected).
    """

    rejected: tuple[bool, ...]
    n_certified: int
    first_failure: int | None
    alpha: float


def learn_then_test_fixed_sequence(p_values_in_order: Sequence[float], alpha: float) -> FixedSequenceResult:
    """Learn-then-Test (Angelopoulos et al. 2021) with fixed-sequence testing along dose.

    The candidate operating points are ordered before seeing the data from the one most likely to
    be safe (lowest dose) upward; each null says "this dose violates the risk limit" (detection
    above its tolerance). Testing proceeds in that order at level alpha *each*, and stops at the
    first null not rejected; every dose before it is certified. Because a later hypothesis is only
    tested when all earlier ones were rejected, the family-wise error is at most alpha with no
    Bonferroni division -- which is what lets a dose ladder be certified at full power. The order
    must be fixed in advance (pre-registered with the selection rule, plan §6.5).
    """
    alpha = check_prob(alpha, "alpha", open_interval=True)
    pv = np.asarray(p_values_in_order, dtype=float)
    if pv.ndim != 1:
        raise ValueError("p_values_in_order must be one-dimensional")
    if not np.all(np.isfinite(pv)) or np.any(pv < 0) or np.any(pv > 1):
        raise ValueError("p-values must lie in [0, 1]")
    rejected = []
    first_failure = None
    for i, p in enumerate(pv.tolist()):
        if first_failure is None and p <= alpha:
            rejected.append(True)
        else:
            if first_failure is None:
                first_failure = i
            rejected.append(False)
    return FixedSequenceResult(rejected=tuple(rejected), n_certified=sum(rejected),
                               first_failure=first_failure, alpha=alpha)


@dataclass(frozen=True, eq=False)
class Shrinkage:
    """Empirical-Bayes posterior for each arm under a fitted Beta(a, b) prior.

    `a` and `b` are None under complete pooling: the arms' spread is no larger than binomial
    noise (or every outcome is identical), so the estimated between-arm variance is zero and every
    arm's estimate is the pooled rate, with the interval of the pooled Jeffreys posterior
    Beta(k_total + 1/2, n_total - k_total + 1/2).
    """

    a: float | None
    b: float | None
    prior_mean: float
    icc: float
    estimate: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    raw: np.ndarray
    conf: float


def eb_shrink(ks: Sequence[int], ns: Sequence[int], conf: float) -> Shrinkage:
    """Beta-binomial empirical-Bayes shrinkage by the method of moments.

    The arms' rates are modelled as draws from Beta(a, b). Its mean is the pooled rate and its
    intra-arm correlation rho = 1/(a + b + 1) is estimated by the ANOVA estimator across arms (the
    method of moments for the beta-binomial; Kleinman 1973), so a + b = 1/rho - 1. Each arm's
    posterior is Beta(a + k_i, b + n_i - k_i): arms with little data move most towards the mean,
    and the best-looking arm loses the part of its lead that was noise. The intervals condition on
    (a, b) and so ignore their estimation error -- adequate for the exploratory best-of-m
    statements the plan uses them for (§6.5), not for a confirmatory claim.
    """
    conf = check_conf(conf)
    k = np.asarray(ks, dtype=float)
    n = np.asarray(ns, dtype=float)
    if k.ndim != 1 or k.size != n.size or k.size < 2:
        raise ValueError("ks and ns must be equal-length sequences of at least 2 arms")
    for ki, ni in zip(k.tolist(), n.tolist()):
        check_counts(ki, ni, integer=True)
    alpha = 1.0 - conf
    raw = k / n
    total_k, total_n = float(k.sum()), float(n.sum())
    mean = total_k / total_n
    rho = icc_from_counts(k, n)  # raises when every arm has one trial (spread not identified)
    if rho is None or rho <= 0.0:
        a_post = np.full(k.size, total_k + 0.5)
        b_post = np.full(k.size, total_n - total_k + 0.5)
        est = np.full(k.size, mean)
        a_prior = b_prior = None
        rho_out = 0.0
    else:
        rho = min(rho, 1.0 - 1e-9)
        s = 1.0 / rho - 1.0
        a_prior = max(mean * s, 1e-9)
        b_prior = max((1.0 - mean) * s, 1e-9)
        a_post = a_prior + k
        b_post = b_prior + n - k
        est = a_post / (a_post + b_post)
        rho_out = float(rho)
    lo = np.array([betaincinv(a, b, alpha / 2.0) for a, b in zip(a_post, b_post)])
    hi = np.array([betaincinv(a, b, 1.0 - alpha / 2.0) for a, b in zip(a_post, b_post)])
    return Shrinkage(a=a_prior, b=b_prior, prior_mean=mean, icc=rho_out, estimate=est, lo=lo, hi=hi,
                     raw=raw, conf=conf)


@dataclass(frozen=True)
class WinnersCurse:
    """The best of `n_arms` equal arms: its expected observed rate minus the true rate."""

    inflation: float
    se: float
    mean_best: float
    p: float
    n_arms: int
    n: int
    icc: float
    reps: int


def winners_curse_sim(n_arms: int, n: int, p: float, icc: float, reps: int, seed: int, *,
                      cluster_size: int | None = None) -> WinnersCurse:
    """Simulate selecting the best of `n_arms` arms that all have the same true rate p.

    Each arm runs n trials. With icc > 0 the trials come in clusters (prompts) of `cluster_size`
    (the last one may be smaller); each cluster's rate is drawn from the Beta distribution with
    mean p and intra-class correlation icc, i.e. Beta(p(1 - icc)/icc, (1 - p)(1 - icc)/icc), so
    the arm's estimate has design effect 1 + (m - 1) icc. Clusters are drawn afresh for each arm
    (each cell its own prompts' luck). With icc = 0 the trials are independent.
    Returns the mean of (best observed rate - p) over `reps` replications.
    """
    n_arms = check_count(n_arms, "n_arms", minimum=1)
    n = check_count(n, "n", minimum=1)
    p = check_prob(p, "p", open_interval=True)
    icc = check_prob(icc, "icc")
    if icc >= 1.0:
        raise ValueError("icc must be below 1")
    reps = check_count(reps, "reps", minimum=2)
    rng = np.random.default_rng(seed)
    if icc == 0.0:
        counts = rng.binomial(n, p, size=(reps, n_arms))
    else:
        if cluster_size is None:
            raise ValueError("icc > 0 needs cluster_size: the ICC only matters for trials that "
                             "share a prompt")
        m = check_count(cluster_size, "cluster_size", minimum=1)
        sizes = [m] * (n // m) + ([n % m] if n % m else [])
        a = p * (1.0 - icc) / icc
        b = (1.0 - p) * (1.0 - icc) / icc
        counts = np.zeros((reps, n_arms), dtype=np.int64)
        for size in sizes:
            rates = rng.beta(a, b, size=(reps, n_arms))
            counts += rng.binomial(size, rates)
    best = counts.max(axis=1) / n
    infl = best - p
    return WinnersCurse(inflation=float(infl.mean()), se=float(infl.std(ddof=1) / math.sqrt(reps)),
                        mean_best=float(best.mean()), p=p, n_arms=n_arms, n=n, icc=icc, reps=reps)
