"""m5.stats.sequential: the truncated SPRT and the betting confidence sequence."""

from __future__ import annotations

import numpy as np
import pytest

from m5.stats import sequential as Q
from m5.stats.intervals import wilson


def plan_sprt():
    # plan §6.2: H0 rate <= 0.15 against H1 >= 0.5, alpha = beta = 0.05, at most 30 trials
    return Q.SPRT(0.15, 0.5, 0.05, 0.05, 30)


def test_sprt_average_sample_numbers_are_the_plans():
    t = plan_sprt()
    assert t.operating_characteristic(0.0).asn == pytest.approx(6.0)     # six "no" to accept H0
    assert t.operating_characteristic(1.0).asn == pytest.approx(3.0)     # three "yes" to accept H1
    near = t.operating_characteristic(0.3)
    assert 12.0 < near.asn < 18.0
    worst = max(t.operating_characteristic(p).asn for p in np.linspace(0, 1, 51))
    assert worst == pytest.approx(near.asn, rel=0.05)                   # the hump sits near 0.3


def test_sprt_error_rates_are_near_alpha_and_beta():
    t = plan_sprt()
    assert t.operating_characteristic(0.15).p_accept_h1 <= 0.06
    assert t.operating_characteristic(0.5).p_accept_h1 >= 0.94


def test_exact_operating_characteristic_agrees_with_simulation():
    t = plan_sprt()
    for p in (0.1, 0.3, 0.6):
        exact = t.operating_characteristic(p)
        sim = t.simulate(p, 3000, seed=int(p * 100))
        assert sim.asn == pytest.approx(exact.asn, abs=0.6)
        assert sim.p_accept_h1 == pytest.approx(exact.p_accept_h1, abs=0.03)
        assert sim.p_truncated == pytest.approx(exact.p_truncated, abs=0.03)


def test_sprt_stops_at_the_boundaries_and_truncates_by_the_midpoint_rule():
    t = plan_sprt()
    assert t.update([0, 0, 0, 0, 0]) == Q.CONTINUE
    assert t.update(0) == Q.ACCEPT_H0 and t.n == 6 and not t.truncated
    t = plan_sprt()
    assert t.update([1, 1, 1]) == Q.ACCEPT_H1 and t.n == 3
    # 30 trials hovering at the indifference rate p* = 0.306: the midpoint rule decides
    t = Q.SPRT(0.15, 0.5, 0.05, 0.05, 10)
    seq = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    assert t.update(seq) == Q.ACCEPT_H0 and t.truncated and t.n == 10 and t.llr < 0
    t = Q.SPRT(0.15, 0.5, 0.05, 0.05, 10)
    seq = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
    assert t.update(seq) == Q.ACCEPT_H1 and t.truncated and t.llr >= 0


def test_sprt_n_min_defers_the_decision():
    t = Q.SPRT(0.15, 0.5, 0.05, 0.05, 30, n_min=4)
    assert t.update([1, 1, 1]) == Q.CONTINUE
    assert t.update(1) == Q.ACCEPT_H1 and t.n == 4


def test_betting_cs_covers_at_every_n_simultaneously():
    rng = np.random.default_rng(0)
    for draw, truth in ((lambda: float(rng.binomial(1, 0.3)), 0.3),
                        (lambda: float(rng.beta(2, 5)), 2 / 7)):
        misses = 0
        for _ in range(150):
            cs = Q.BettingCS(0.05)
            ever = False
            for _ in range(50):
                lo, hi = cs.update(draw())
                ever |= not (lo <= truth <= hi)
            misses += ever
        assert misses / 150 <= 0.06


def test_betting_cs_shrinks_and_is_wider_than_wilson_by_about_the_plans_factor():
    rng = np.random.default_rng(1)
    cs = Q.BettingCS(0.05)
    widths = []
    x = rng.binomial(1, 0.3, 120).astype(float)
    for i, xi in enumerate(x, start=1):
        lo, hi = cs.update(xi)
        widths.append(hi - lo)
    assert widths[-1] < widths[20] < widths[5]
    assert cs.interval() == (lo, hi) and cs.mean == pytest.approx(x.mean())
    wlo, whi = wilson(int(x.sum()), 120, 0.95)
    assert 1.15 < (hi - lo) / (whi - wlo) < 2.0          # plan §6.2: 1.4-1.6x wider than Wilson


def test_betting_cs_stops_a_strong_cell_quickly():
    rng = np.random.default_rng(2)
    stops = []
    for _ in range(30):
        cs = Q.BettingCS(0.05)
        for _ in range(66):
            cs.update(float(rng.beta(2.4, 1.6)))          # mean 0.6
            if cs.excludes(0.25):
                break
        stops.append(cs.n)
    assert np.median(stops) <= 12                         # plan §6.2: a strong cell stops at ~8


# ------------------------------------------------------------------------------- guards

@pytest.mark.parametrize("args,msg", [((0.5, 0.15, 0.05, 0.05, 30), "below p1"),
                                      ((0.15, 0.5, 0.6, 0.5, 30), "alpha \\+ beta"),
                                      ((0.0, 0.5, 0.05, 0.05, 30), "p0"),
                                      ((0.15, 0.5, 0.05, 0.05, 0), "n_max")])
def test_sprt_construction_guards(args, msg):
    with pytest.raises(ValueError, match=msg):
        Q.SPRT(*args)
    with pytest.raises(ValueError, match="n_min"):
        Q.SPRT(0.15, 0.5, 0.05, 0.05, 5, n_min=6)


def test_sprt_refuses_trials_after_it_stopped_and_non_binary_outcomes():
    t = plan_sprt()
    t.update([1, 1, 1])
    with pytest.raises(ValueError, match="already stopped"):
        t.update(0)
    with pytest.raises(ValueError, match="0 or 1"):
        plan_sprt().update(2)
    with pytest.raises(ValueError, match="0 or 1"):
        plan_sprt().update(True)


def test_betting_cs_guards():
    cs = Q.BettingCS(0.05)
    with pytest.raises(ValueError, match="no observations"):
        _ = cs.mean
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cs.update(1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cs.update(float("nan"))
    with pytest.raises(ValueError, match="alpha"):
        Q.BettingCS(1.0)
    with pytest.raises(ValueError, match="grid_size"):
        Q.BettingCS(0.05, grid_size=5)
