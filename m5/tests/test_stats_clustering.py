"""m5.stats.clustering: the ICC, the design effect and the cluster bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from m5.stats import clustering as C


def test_icc_by_hand():
    # k = (3, 0, 2) of n = 4 each: MSB = 0.5833, MSW = 0.1944, n0 = 4 -> ICC = 1/3
    groups = [[1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]]
    assert C.icc_binary(groups) == pytest.approx(1 / 3, abs=1e-12)
    assert C.icc_from_counts([3, 0, 2], [4, 4, 4]) == pytest.approx(1 / 3, abs=1e-12)


def test_icc_recovers_a_simulated_beta_binomial_correlation():
    rng = np.random.default_rng(1)
    p, rho = 0.3, 0.2
    a, b = p * (1 - rho) / rho, (1 - p) * (1 - rho) / rho
    est = []
    for _ in range(40):
        rates = rng.beta(a, b, 150)
        est.append(C.icc_binary([rng.binomial(1, q, 10) for q in rates]))
    assert np.mean(est) == pytest.approx(rho, abs=0.02)
    # independent trials: the estimate is clipped at 0, never negative
    flat = [rng.binomial(1, 0.3, 10) for _ in range(150)]
    assert 0.0 <= C.icc_binary(flat) < 0.05


def test_icc_is_missing_not_zero_when_every_outcome_is_identical():
    assert C.icc_binary([[0, 0, 0], [0, 0], [0]]) is None
    assert C.icc_from_counts([4, 3], [4, 3]) is None


def test_design_effect_and_effective_n():
    assert C.design_effect(20, 0.1) == pytest.approx(2.9)
    assert C.effective_n(20, 20, 0.1) == pytest.approx(20 / 2.9)
    assert C.design_effect(1, 0.7) == 1.0          # one trial per prompt: no clustering cost
    assert C.mean_cluster_size([10, 10, 10]) == 10.0
    assert C.mean_cluster_size([1, 19]) == pytest.approx((1 + 361) / 20)   # not the plain mean 10


def test_cluster_bootstrap_resamples_whole_clusters_and_is_seeded():
    rng = np.random.default_rng(0)
    clusters = [rng.binomial(1, q, 8) for q in rng.beta(0.6, 1.4, 40)]   # ICC 1/3, deff 3.3

    def stat(cs):
        return float(np.mean(np.concatenate(cs)))

    r1 = C.cluster_bootstrap(clusters, stat, reps=300, seed=7, conf=0.95)
    r2 = C.cluster_bootstrap(clusters, stat, reps=300, seed=7, conf=0.95)
    assert r1.lo == r2.lo and r1.hi == r2.hi
    assert r1.lo < r1.estimate < r1.hi and r1.n_failed == 0
    # resampling prompts carries the clustering: wider than resampling single trials would be
    flat = np.concatenate(clusters)
    naive_se = flat.std(ddof=1) / np.sqrt(flat.size)
    assert r1.se > 1.3 * naive_se
    vec = C.cluster_bootstrap(clusters, lambda cs: np.array([stat(cs), 2 * stat(cs)]), reps=50,
                              seed=1, conf=0.9)
    assert vec.draws.shape == (50, 2) and vec.hi[1] == pytest.approx(2 * vec.hi[0])


def test_cluster_bootstrap_counts_degenerate_resamples():
    clusters = [np.array([1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0])]

    def picky(cs):
        if sum(float(c[0]) for c in cs) == 0:
            raise ValueError("degenerate")
        return sum(float(c[0]) for c in cs)

    res = C.cluster_bootstrap(clusters, picky, reps=400, seed=3, conf=0.9)
    assert 0 < res.n_failed < 400 and res.draws.size == 400 - res.n_failed


# ------------------------------------------------------------------------------- guards

def test_icc_guards():
    with pytest.raises(ValueError, match="at least 2 clusters"):
        C.icc_binary([[0, 1, 1]])
    with pytest.raises(ValueError, match="single trial"):
        C.icc_binary([[0], [1], [1]])
    with pytest.raises(ValueError, match="0/1"):
        C.icc_binary([[0, 2], [1, 1]])
    with pytest.raises(ValueError, match="equal length"):
        C.icc_from_counts([1, 2], [3])
    with pytest.raises(ValueError, match="0 <= k <= n"):
        C.icc_from_counts([4, 1], [3, 3])
    with pytest.raises(ValueError, match="whole numbers"):
        C.icc_from_counts([1.5, 1], [3, 3])


def test_design_effect_guards():
    with pytest.raises(ValueError, match="m_avg"):
        C.design_effect(0.5, 0.1)
    with pytest.raises(ValueError, match="icc"):
        C.design_effect(10, -0.1)
    with pytest.raises(ValueError, match="icc"):
        C.design_effect(10, 1.5)
    with pytest.raises(ValueError, match="positive"):
        C.effective_n(0, 10, 0.1)
    with pytest.raises(ValueError, match="sizes"):
        C.mean_cluster_size([0, 3])


def test_bootstrap_guards():
    with pytest.raises(ValueError, match="at least 2 clusters"):
        C.cluster_bootstrap([np.ones(3)], lambda cs: 1.0, reps=10, seed=0, conf=0.95)
    with pytest.raises(ValueError, match="reps"):
        C.cluster_bootstrap([np.ones(3), np.ones(2)], lambda cs: 1.0, reps=1, seed=0, conf=0.95)
    with pytest.raises(ValueError, match="resamples"):
        C.cluster_bootstrap([np.ones(3), np.ones(2)], lambda cs: float("nan"), reps=10, seed=0,
                            conf=0.95)
