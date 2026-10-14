"""m5.stats.selection: Learn-then-Test certification, EB shrinkage, the winner's curse."""

from __future__ import annotations

import numpy as np
import pytest

from m5.stats import selection as L


def test_binomial_tail_pvalue_is_exact():
    st = pytest.importorskip("scipy.stats")
    assert L.binomial_tail_pvalue(0, 30, 0.1) == pytest.approx(0.9 ** 30)
    for k, n, t in [(3, 50, 0.15), (10, 120, 0.05), (0, 150, 0.02)]:
        assert L.binomial_tail_pvalue(k, n, t) == pytest.approx(st.binom.cdf(k, n, t), rel=1e-10)
    assert L.binomial_tail_pvalue(50, 50, 0.3) == 1.0


def test_fixed_sequence_stops_at_the_first_failure():
    r = L.learn_then_test_fixed_sequence([0.001, 0.01, 0.2, 0.001], 0.05)
    assert r.rejected == (True, True, False, False)       # the later small p-value is not used
    assert r.n_certified == 2 and r.first_failure == 2
    all_pass = L.learn_then_test_fixed_sequence([0.01, 0.02], 0.05)
    assert all_pass.n_certified == 2 and all_pass.first_failure is None
    assert L.learn_then_test_fixed_sequence([], 0.05).n_certified == 0


def test_fixed_sequence_controls_the_family_wise_error():
    # every dose truly at the threshold (every null true): certifying any dose is an error
    rng = np.random.default_rng(0)
    errors = 0
    reps = 4000
    for _ in range(reps):
        ks = rng.binomial(60, 0.1, size=6)
        pv = [L.binomial_tail_pvalue(int(k), 60, 0.1) for k in ks]
        errors += L.learn_then_test_fixed_sequence(pv, 0.05).n_certified > 0
    assert errors / reps <= 0.05 + 0.01


def test_eb_shrinkage_pulls_noisy_arms_towards_the_pool():
    ks, ns = [2, 5, 10, 3, 20], [20, 20, 20, 20, 20]
    sh = L.eb_shrink(ks, ns, 0.95)
    raw = np.array(ks) / np.array(ns)
    assert np.all(np.abs(sh.estimate - sh.prior_mean) <= np.abs(raw - sh.prior_mean) + 1e-12)
    assert np.array_equal(np.argsort(sh.estimate), np.argsort(raw))      # equal n: order kept
    assert np.all(sh.lo <= sh.estimate) and np.all(sh.estimate <= sh.hi)
    assert sh.a is not None and sh.a > 0 and sh.icc > 0
    # a small arm moves further than a large one with the same raw rate
    sh2 = L.eb_shrink([2, 20, 30, 5], [10, 100, 40, 40], 0.95)
    assert sh2.a is not None
    assert abs(sh2.estimate[0] - 0.2) > abs(sh2.estimate[1] - 0.2)


def test_eb_complete_pooling_when_the_spread_is_only_noise():
    sh = L.eb_shrink([5, 5, 5, 5], [20, 20, 20, 20], 0.95)
    assert sh.a is None and sh.icc == 0.0
    np.testing.assert_allclose(sh.estimate, 0.25)
    zeros = L.eb_shrink([0, 0, 0], [10, 10, 10], 0.95)          # identical outcomes: no spread
    assert zeros.a is None and np.all(zeros.hi > 0)


def test_winners_curse_reproduces_the_plans_table():
    # plan §6.5, every arm truly equal: best of 25 layers at n = 22 is inflated by +14.9 points,
    # best of 150 cells by +20.3, best of 25 at n = 66 by +8.5 and at n = 120 by +6.2; the table
    # is reproduced at a base rate of about 0.12
    plan = {(25, 22): 0.149, (150, 22): 0.203, (25, 66): 0.085, (25, 120): 0.062}
    got = {key: L.winners_curse_sim(key[0], key[1], 0.12, 0.0, 3000 if key[0] == 25 else 1500,
                                    seed=1).inflation for key in plan}
    for key, want in plan.items():
        assert got[key] == pytest.approx(want, abs=0.012)
    assert got[(25, 120)] < got[(25, 66)] < got[(25, 22)] < got[(150, 22)]
    # the brief's order-of-magnitude check at p = 0.3
    assert 0.12 < L.winners_curse_sim(25, 22, 0.3, 0.0, 3000, seed=1).inflation < 0.25
    # trials that share prompts carry less information: clustering inflates the curse further
    clustered = L.winners_curse_sim(25, 22, 0.12, 0.2, 3000, seed=1, cluster_size=11).inflation
    assert clustered > got[(25, 22)] + 0.03
    single = L.winners_curse_sim(1, 22, 0.12, 0.0, 6000, seed=2)
    assert abs(single.inflation) < 4 * single.se              # one arm: no selection, no bias


# ------------------------------------------------------------------------------- guards

def test_selection_guards():
    with pytest.raises(ValueError, match="threshold"):
        L.binomial_tail_pvalue(1, 10, 0.0)
    with pytest.raises(ValueError, match="exceeds"):
        L.binomial_tail_pvalue(11, 10, 0.1)
    with pytest.raises(ValueError, match="whole number"):
        L.binomial_tail_pvalue(1.5, 10, 0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        L.learn_then_test_fixed_sequence([0.01, 1.2], 0.05)
    with pytest.raises(ValueError, match="alpha"):
        L.learn_then_test_fixed_sequence([0.01], 1.0)
    with pytest.raises(ValueError, match="at least 2 arms"):
        L.eb_shrink([1], [10], 0.95)
    with pytest.raises(ValueError, match="exceeds"):
        L.eb_shrink([1, 12], [10, 10], 0.95)
    with pytest.raises(ValueError, match="single trial"):
        L.eb_shrink([1, 0, 1], [1, 1, 1], 0.95)
    with pytest.raises(ValueError, match="cluster_size"):
        L.winners_curse_sim(25, 22, 0.15, 0.1, 100, seed=0)
    with pytest.raises(ValueError, match="icc"):
        L.winners_curse_sim(25, 22, 0.15, 1.0, 100, seed=0, cluster_size=2)
    with pytest.raises(ValueError, match="reps"):
        L.winners_curse_sim(25, 22, 0.15, 0.0, 1, seed=0)
