"""m5.stats.correction: Rogan-Gladen with propagated uncertainty, PPI++, label budgets."""

from __future__ import annotations

import numpy as np
import pytest

from m5.stats import correction as R


def test_rogan_gladen_point_estimate_and_clipping():
    assert R.rogan_gladen(0.5, 0.9, 0.9) == pytest.approx(0.5)
    assert R.rogan_gladen(0.26, 0.8, 0.95) == pytest.approx((0.26 + 0.95 - 1) / 0.75)
    assert R.rogan_gladen(0.01, 0.9, 0.95) == 0.0       # below the judge's own false-positive rate
    assert R.rogan_gladen(0.99, 0.9, 0.95) == 1.0


def test_delta_interval_matches_the_plans_label_figures():
    # plan §7.4: TPR = TNR = 0.9, theta = 0.5, p_obs known: about +-5 points with 100 + 100 labels,
    # about +-3.7 with 200 + 200 (the Agresti-Coull variances add about 7%)
    r100 = R.rogan_gladen_interval(50_000, 100_000, 90, 100, 90, 100, 0.95, method="delta")
    r200 = R.rogan_gladen_interval(50_000, 100_000, 180, 200, 180, 200, 0.95, method="delta")
    assert r100.hi - r100.estimate == pytest.approx(0.05, abs=0.008)
    assert r200.hi - r200.estimate == pytest.approx(0.037, abs=0.005)
    assert r100.method == "delta" and r100.lo < 0.5 < r100.hi


def test_a_perfect_gold_score_still_carries_uncertainty():
    r = R.rogan_gladen_interval(30, 150, 20, 20, 20, 20, 0.95, method="delta")
    assert r.hi - r.lo > 0.05            # plug-in variances would be zero for TPR = TNR = 20/20


def test_bootstrap_interval_respects_the_floor_where_the_delta_method_cannot():
    d = R.rogan_gladen_interval(0, 150, 20, 20, 20, 20, 0.95, method="delta")
    b = R.rogan_gladen_interval(0, 150, 20, 20, 20, 20, 0.95, method="bootstrap", reps=4000, seed=1)
    assert d.estimate == b.estimate == 0.0 and b.lo == 0.0
    assert b.hi < 0.03 < d.hi
    again = R.rogan_gladen_interval(0, 150, 20, 20, 20, 20, 0.95, method="bootstrap", reps=4000, seed=1)
    assert again.hi == b.hi


def test_bootstrap_interval_covers_in_simulation():
    rng = np.random.default_rng(5)
    theta, tpr, tnr = 0.2, 0.85, 0.92
    p_obs = theta * tpr + (1 - theta) * (1 - tnr)
    cover = 0
    reps = 150
    for i in range(reps):
        k = rng.binomial(200, p_obs)
        kp = rng.binomial(80, tpr)
        kn = rng.binomial(80, tnr)
        r = R.rogan_gladen_interval(k, 200, kp, 80, kn, 80, 0.95, method="bootstrap", reps=1000, seed=i)
        cover += r.lo <= theta <= r.hi
    assert cover / reps >= 0.9


def test_bootstrap_with_an_uninformative_judge_returns_the_whole_interval():
    r = R.rogan_gladen_interval(10, 50, 6, 10, 5, 10, 0.95, method="bootstrap", reps=2000, seed=2)
    assert (r.lo, r.hi) == (0.0, 1.0) and "uninformative" in r.method


def test_ppi_mean_is_valid_and_beats_the_human_labels_alone():
    rng = np.random.default_rng(3)
    cover = cover_c = 0
    widths, widths_c = [], []
    for _ in range(150):
        y = rng.binomial(1, 0.3, 3000).astype(float)
        yhat = np.clip(0.3 + 0.7 * (y - 0.3) + rng.normal(0, 0.12, y.size), 0, 1)
        lab = np.zeros(y.size, dtype=bool)
        lab[rng.choice(y.size, 120, replace=False)] = True
        r = R.ppi_mean(y[lab], yhat[lab], yhat[~lab], 0.95)
        truth = float(y.mean())
        cover += r.lo <= truth <= r.hi
        cover_c += r.classical_lo <= truth <= r.classical_hi
        widths.append(r.hi - r.lo)
        widths_c.append(r.classical_hi - r.classical_lo)
    assert cover / 150 >= 0.9 and cover_c / 150 >= 0.88
    assert np.mean(widths) < 0.7 * np.mean(widths_c)


def test_ppi_power_tuning_ignores_a_useless_judge():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.4, 200).astype(float)
    useless = rng.uniform(0, 1, 2200)
    r = R.ppi_mean(y, useless[:200], useless[200:], 0.95)
    assert r.lam < 0.2
    assert r.hi - r.lo <= 1.02 * (r.classical_hi - r.classical_lo)
    classic = R.ppi_mean(y, useless[:200], useless[200:], 0.95, power_tuning=False)
    assert classic.lam == 1.0
    constant = R.ppi_mean(y, np.full(200, 0.5), np.full(2000, 0.5), 0.95)
    assert constant.lam == 0.0 and constant.estimate == pytest.approx(y.mean())


def test_label_budget_reproduces_the_plan():
    # plan §6.4: at rho = 0.5, 0.7, 0.9 about 72, 49, 18 labels give +-0.10, against 96 without
    for rho, plan in [(0.0, 96), (0.5, 72), (0.7, 49), (0.9, 18)]:
        assert abs(R.label_budget(rho, 0.10, 0.95) - plan) <= 1
    # a finite unlabelled pool needs more labels than an infinite one, never fewer
    assert R.label_budget(0.7, 0.10, 0.95, n_unlabelled=300) > R.label_budget(0.7, 0.10, 0.95)
    assert R.label_budget(0.0, 0.10, 0.95, n_unlabelled=300) == R.label_budget(0.0, 0.10, 0.95)
    # the finite-N answer solves n = n0 (1 - rho^2 N/(n + N)) up to rounding
    n = R.label_budget(0.8, 0.05, 0.95, n_unlabelled=500)
    n0 = 1.959963984540054 ** 2 * 0.25 / 0.05 ** 2
    assert n - 1 < n0 * (1 - 0.64 * 500 / (n + 500)) <= n


# ------------------------------------------------------------------------------- guards

def test_rogan_gladen_refuses_an_uninformative_judge():
    with pytest.raises(ValueError, match="no better than chance"):
        R.rogan_gladen(0.3, 0.5, 0.5)
    with pytest.raises(ValueError, match="no better than chance"):
        R.rogan_gladen_interval(10, 50, 4, 10, 5, 10, 0.95, method="delta")


def test_rogan_gladen_guards():
    with pytest.raises(ValueError, match="p_obs"):
        R.rogan_gladen(1.2, 0.9, 0.9)
    with pytest.raises(ValueError, match="method"):
        R.rogan_gladen_interval(10, 50, 9, 10, 9, 10, 0.95, method="wald")
    with pytest.raises(ValueError, match="reps and seed"):
        R.rogan_gladen_interval(10, 50, 9, 10, 9, 10, 0.95, method="bootstrap")
    with pytest.raises(ValueError, match="reps"):
        R.rogan_gladen_interval(10, 50, 9, 10, 9, 10, 0.95, method="bootstrap", reps=10, seed=0)
    with pytest.raises(ValueError, match="gold"):
        R.rogan_gladen_interval(10, 50, 11, 10, 9, 10, 0.95, method="delta")
    with pytest.raises(TypeError):
        R.rogan_gladen_interval(10, 50, 9, 10, 9, 10, 0.95)   # the method is a required choice


def test_ppi_guards():
    with pytest.raises(ValueError, match="pair item by item"):
        R.ppi_mean([1, 0, 1], [0.9, 0.1], [0.5, 0.5], 0.95)
    with pytest.raises(ValueError, match="at least 2"):
        R.ppi_mean([1], [0.9], [0.5, 0.5], 0.95)
    with pytest.raises(ValueError, match="missing"):
        R.ppi_mean([1, 0], [0.9, float("nan")], [0.5, 0.5], 0.95)
    with pytest.raises(ValueError, match="conf"):
        R.ppi_mean([1, 0], [0.9, 0.1], [0.5, 0.5], 1.0)


@pytest.mark.parametrize("kwargs,msg", [({"rho": 1.5}, "rho"), ({"half_width": 0.0}, "half_width"),
                                        ({"variance": -1.0}, "variance"),
                                        ({"n_unlabelled": 0}, "n_unlabelled")])
def test_label_budget_guards(kwargs, msg):
    args = {"rho": 0.5, "half_width": 0.1}
    args.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        R.label_budget(args.pop("rho"), args.pop("half_width"), 0.95, **args)
