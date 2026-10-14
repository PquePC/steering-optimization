"""m5.stats.rank: Spearman with ties, within-strata correlation, AUC (DeLong), Cohen's kappa."""

from __future__ import annotations

import numpy as np
import pytest

from m5.stats import rank as K


def test_ranks_and_spearman_handle_ties_like_scipy():
    st = pytest.importorskip("scipy.stats")
    x = np.array([1, 2, 2, 3, 4, 5, 5, 5, 6])
    y = np.array([2, 1, 3, 3, 5, 4, 6, 6, 7])
    np.testing.assert_allclose(K.rankdata(x), st.rankdata(x))
    assert K.spearman(x, y) == pytest.approx(st.spearmanr(x, y).statistic, abs=1e-12)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 40), rng.integers(0, 5, 40)
    assert K.spearman(a, b) == pytest.approx(st.spearmanr(a, b).statistic, abs=1e-12)


def test_a_shared_dose_trend_does_not_count_as_agreement():
    # M2's e6 "tracked" judged influence because both rose with dose (TODO.md); within dose bins
    # two independent readouts must show no association
    rng = np.random.default_rng(0)
    dose = np.repeat(np.arange(5), 40)
    a = dose + rng.normal(0, 1, dose.size)
    b = dose + rng.normal(0, 1, dose.size)
    assert K.spearman(a, b) > 0.5
    w = K.within_strata_spearman(a, b, dose)
    assert abs(w["rho_pooled"]) < 0.2 and abs(w["rho_mean"]) < 0.2 and w["n_strata"] == 5
    ci = K.bootstrap_spearman(a, b, strata=dose, reps=200, seed=1, conf=0.95)
    assert ci["lo"] < 0 < ci["hi"] or abs(ci["estimate"]) < 0.2


def test_genuine_within_stratum_tracking_is_found():
    rng = np.random.default_rng(1)
    dose = np.repeat(np.arange(4), 50)
    latent = rng.normal(size=dose.size)
    a = dose + latent + rng.normal(0, 0.3, dose.size)
    b = -3 * dose + latent + rng.normal(0, 0.3, dose.size)   # opposite dose trend, same latent
    assert K.spearman(a, b) < 0                              # the raw correlation is misleading
    w = K.within_strata_spearman(a, b, dose)
    assert w["rho_mean"] > 0.8 and w["rho_pooled"] > 0.8
    ci = K.bootstrap_spearman(a, b, strata=dose, reps=200, seed=2, conf=0.95)
    assert 0.7 < ci["lo"] <= ci["estimate"] <= ci["hi"] <= 1.0
    by_cluster = K.bootstrap_spearman(a, b, strata=dose, clusters=np.arange(dose.size) % 10,
                                      reps=200, seed=3, conf=0.95, statistic="rho_pooled")
    assert by_cluster["lo"] > 0.7


def test_strata_too_small_to_rank_are_skipped_and_counted():
    x = [1, 2, 3, 4, 5, 9, 9]
    y = [1, 2, 3, 5, 4, 1, 2]
    s = ["a", "a", "a", "a", "a", "b", "b"]
    w = K.within_strata_spearman(x, y, s)
    assert w["n_strata"] == 1 and w["n_skipped"] == 1


def test_auc_matches_mann_whitney_and_its_delong_se_matches_the_bootstrap():
    st = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    pos = np.round(rng.normal(1, 1, 80), 1)             # rounded: ties between the classes
    neg = np.round(rng.normal(0, 1, 120), 1)
    r = K.auc(pos, neg, 0.95)
    assert r["auc"] == pytest.approx(st.mannwhitneyu(pos, neg).statistic / (80 * 120))
    boots = [K.auc(rng.choice(pos, 80), rng.choice(neg, 120), 0.95)["auc"] for _ in range(400)]
    assert r["se"] == pytest.approx(np.std(boots, ddof=1), rel=0.2)
    assert r["lo"] < r["auc"] < r["hi"]
    perfect = K.auc([2, 3, 4], [0, 1], 0.95)
    assert perfect["auc"] == 1.0 and perfect["hi"] == 1.0


def test_cohen_kappa_by_hand_and_its_standard_error():
    r = K.cohen_kappa([1, 1, 0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 0, 0, 1, 0, 1, 1, 1, 1], 0.95)
    assert r["p_observed"] == pytest.approx(0.8) and r["p_expected"] == pytest.approx(0.52)
    assert r["kappa"] == pytest.approx(0.28 / 0.48)
    assert r["se"] == pytest.approx(0.2624, abs=1e-4)       # Fleiss, Cohen & Everitt (1969)
    assert K.cohen_kappa(["y", "n", "y"], ["y", "n", "y"], 0.95)["kappa"] == 1.0
    rng = np.random.default_rng(4)
    truth = rng.integers(0, 3, 400)
    r1 = np.where(rng.random(400) < 0.85, truth, rng.integers(0, 3, 400))
    r2 = np.where(rng.random(400) < 0.85, truth, rng.integers(0, 3, 400))
    k = K.cohen_kappa(r1, r2, 0.95)
    boots = []
    for _ in range(300):
        idx = rng.integers(0, 400, 400)
        boots.append(K.cohen_kappa(r1[idx], r2[idx], 0.95)["kappa"])
    assert k["se"] == pytest.approx(np.std(boots, ddof=1), rel=0.15)


# ------------------------------------------------------------------------------- guards

def test_rank_guards():
    with pytest.raises(ValueError, match="constant"):
        K.spearman([1, 1, 1], [1, 2, 3])
    with pytest.raises(ValueError, match="pair up"):
        K.spearman([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="no stratum"):
        K.within_strata_spearman([1, 2, 3, 4], [1, 2, 3, 4], ["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="same length"):
        K.within_strata_spearman([1, 2, 3], [1, 2, 3], ["a", "a"])
    with pytest.raises(ValueError, match="statistic"):
        K.bootstrap_spearman([1, 2, 3], [1, 2, 3], reps=10, seed=0, conf=0.9, statistic="tau")
    with pytest.raises(ValueError, match="2 clusters"):
        K.bootstrap_spearman([1, 2, 3, 4], [1, 2, 4, 3], clusters=[0, 0, 0, 0], reps=10, seed=0,
                             conf=0.9)
    with pytest.raises(ValueError, match="label every unit"):
        K.bootstrap_spearman([1, 2, 3, 4], [1, 2, 4, 3], strata=[0, 0, 1], reps=10, seed=0, conf=0.9)


def test_bootstrap_refuses_when_resamples_give_no_correlation(monkeypatch):
    calls = {"n": 0}
    real = K.spearman

    def first_call_only(x, y):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("constant")
        return real(x, y)

    monkeypatch.setattr(K, "spearman", first_call_only)
    with pytest.raises(ValueError, match="resamples"):
        K.bootstrap_spearman([1, 2, 3, 4], [1, 2, 4, 3], reps=10, seed=0, conf=0.9)


def test_auc_and_kappa_guards():
    with pytest.raises(ValueError, match="at least 2"):
        K.auc([1.0], [0.0, 0.5], 0.95)
    with pytest.raises(ValueError, match="missing"):
        K.auc([1.0, float("nan")], [0.0, 0.5], 0.95)
    with pytest.raises(ValueError, match="single, identical category"):
        K.cohen_kappa([1, 1, 1], [1, 1, 1], 0.95)
    with pytest.raises(ValueError, match="same items"):
        K.cohen_kappa([1, 0, 1], [1, 0], 0.95)
    with pytest.raises(ValueError, match="at least 2"):
        K.cohen_kappa([1], [1], 0.95)
