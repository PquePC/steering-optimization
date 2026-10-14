"""m5.stats special functions: the scipy-free fallbacks against scipy, and the guards.

The production pod may lack scipy, so every exact statement (Clopper-Pearson bounds, binomial
tails, Beta quantiles, the beta-binomial likelihood) must come out the same either way. These
tests hold the fallbacks to scipy where scipy is installed, and run the public functions with
scipy switched off.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from m5.stats import beta as B
from m5.stats import normal as N
from m5.stats import _checks


@pytest.fixture
def no_scipy(monkeypatch):
    """Run the public functions on the fallback path, as on a pod without scipy."""
    monkeypatch.setattr(B, "_sp", None)
    monkeypatch.setattr(N, "_sp", None)
    assert not B.have_scipy()


def test_lgamma_fallback_matches_scipy():
    sp = pytest.importorskip("scipy.special")
    x = np.concatenate([np.logspace(-8, 7, 300), np.linspace(0.01, 60, 300), [0.5, 1.0, 2.0]])
    ref = sp.gammaln(x)
    assert np.max(np.abs(B.lgamma_py(x) - ref) / np.maximum(1.0, np.abs(ref))) < 1e-13
    assert float(B.lgamma_py(np.float64(0.3))) == pytest.approx(math.lgamma(0.3), abs=1e-14)


def test_betainc_fallback_matches_scipy_over_a_wide_range():
    sp = pytest.importorskip("scipy.special")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(600):
        a = float(np.exp(rng.uniform(math.log(0.05), math.log(3000))))
        b = float(np.exp(rng.uniform(math.log(0.05), math.log(3000))))
        x = float(rng.uniform())
        worst = max(worst, abs(B.betainc_py(a, b, x) - sp.betainc(a, b, x)))
    assert worst < 1e-10
    assert B.betainc_py(2.0, 3.0, 0.0) == 0.0 and B.betainc_py(2.0, 3.0, 1.0) == 1.0


def test_betaincinv_fallback_matches_scipy_including_extremes():
    sp = pytest.importorskip("scipy.special")
    rng = np.random.default_rng(1)
    for _ in range(300):
        a = float(np.exp(rng.uniform(math.log(0.05), math.log(3000))))
        b = float(np.exp(rng.uniform(math.log(0.05), math.log(3000))))
        p = float(rng.uniform(1e-6, 1 - 1e-6))
        ref = sp.betaincinv(a, b, p)
        assert B.betaincinv_py(a, b, p) == pytest.approx(ref, rel=1e-9, abs=1e-14)
    # tiny quantiles (found by geometric bisection, not linear) and Clopper-Pearson's k = 0 case
    for a, b, p in [(0.5, 0.5, 1e-10), (0.01, 5.0, 0.5), (1.0, 30.0, 0.975), (200.5, 0.5, 0.025)]:
        assert B.betaincinv_py(a, b, p) == pytest.approx(sp.betaincinv(a, b, p), rel=1e-9)
    assert B.betaincinv_py(1.0, 30.0, 0.975) == pytest.approx(1 - 0.025 ** (1 / 30), rel=1e-12)


def test_binomial_tails_match_scipy():
    st = pytest.importorskip("scipy.stats")
    for n, p in [(30, 0.15), (120, 0.03), (7, 0.5), (500, 0.9)]:
        for k in range(0, n + 1, max(1, n // 10)):
            assert B.binom_cdf(k, n, p) == pytest.approx(st.binom.cdf(k, n, p), rel=1e-10, abs=1e-300)
            assert B.binom_sf(k, n, p) == pytest.approx(st.binom.sf(k - 1, n, p), rel=1e-10, abs=1e-300)
    assert B.binom_cdf(-1, 10, 0.3) == 0.0 and B.binom_cdf(10, 10, 0.3) == 1.0
    assert B.binom_sf(0, 10, 0.3) == 1.0 and B.binom_sf(11, 10, 0.3) == 0.0


def test_normal_quantile_fallback_matches_scipy():
    sp = pytest.importorskip("scipy.special")
    for p in [1e-300, 1e-20, 1e-5, 0.02425, 0.3, 0.5, 0.7, 0.975, 1 - 1e-12]:
        assert N.norm_ppf_py(p) == pytest.approx(sp.ndtri(p), rel=1e-13, abs=1e-14)
    assert N.z_two_sided(0.95) == pytest.approx(1.959963984540054, abs=1e-12)
    assert N.z_one_sided(0.95) == pytest.approx(1.6448536269514722, abs=1e-12)


def test_public_functions_agree_with_and_without_scipy(monkeypatch):
    pytest.importorskip("scipy")
    from m5.stats import intervals as I
    with_sp = (I.clopper_pearson(3, 40, 0.95), I.wilson(3, 40, 0.95), B.binom_cdf(4, 50, 0.15),
               B.betaincinv(2.5, 7.5, 0.3), N.norm_ppf(0.01), float(B.lgamma(17.5)))
    monkeypatch.setattr(B, "_sp", None)
    monkeypatch.setattr(N, "_sp", None)
    without = (I.clopper_pearson(3, 40, 0.95), I.wilson(3, 40, 0.95), B.binom_cdf(4, 50, 0.15),
               B.betaincinv(2.5, 7.5, 0.3), N.norm_ppf(0.01), float(B.lgamma(17.5)))
    np.testing.assert_allclose(np.hstack(with_sp), np.hstack(without), rtol=1e-10, atol=1e-14)


def test_psychometric_update_is_the_same_without_scipy(monkeypatch):
    pytest.importorskip("scipy")
    from m5.stats import psychometric as P

    def run():
        mu = P.make_mu_grid(-2, 1, 31)
        prior = P.psychometric_prior(mu, center=-0.5, narrow_weight=0.8, narrow_sd=0.15, wide_sd=1.0,
                                     floor_ab=(1, 49), lapse_ab=(1, 49), slope_median=0.1,
                                     slope_logsd=0.6, icc=(0.0, 0.2), n_s=5, n_floor=3, n_lapse=3)
        post = P.LayerPosterior(prior)
        post.update(-0.5, 3, 8)
        post.update(-0.2, 7, 8)
        return post.log_posterior()

    ref = run()
    monkeypatch.setattr(B, "_sp", None)
    monkeypatch.setattr(N, "_sp", None)
    np.testing.assert_allclose(run(), ref, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------------------- guards

@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_lgamma_refuses_non_positive_or_non_finite(bad, no_scipy):
    with pytest.raises(ValueError, match="x > 0"):
        B.lgamma(np.array([1.0, bad]))


def test_lgamma_guard_also_trips_on_the_scipy_path():
    pytest.importorskip("scipy")
    with pytest.raises(ValueError, match="x > 0"):
        B.lgamma(np.array([0.0]))


@pytest.mark.parametrize("a,b", [(0.0, 1.0), (1.0, -2.0), (math.inf, 1.0), (math.nan, 1.0)])
def test_beta_functions_refuse_bad_parameters(a, b):
    for f in (B.betainc, B.betainc_py, B.betaincinv, B.betaincinv_py):
        with pytest.raises(ValueError, match="beta parameters"):
            f(a, b, 0.5)


def test_beta_functions_refuse_non_finite_arguments():
    with pytest.raises(ValueError, match="finite"):
        B.betainc_py(1.0, 1.0, math.nan)
    with pytest.raises(ValueError, match="finite"):
        B.betaincinv_py(1.0, 1.0, math.nan)


@pytest.mark.parametrize("n,p", [(-1, 0.5), (10, -0.1), (10, 1.5), (math.inf, 0.5), (10, math.nan)])
def test_binomial_tails_refuse_bad_arguments(n, p):
    with pytest.raises(ValueError):
        B.binom_cdf(3, n, p)
    with pytest.raises(ValueError):
        B.binom_sf(3, n, p)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.2, 1.2])
def test_normal_quantile_refuses_the_closed_ends(p, no_scipy):
    with pytest.raises(ValueError, match="0 < p < 1"):
        N.norm_ppf(p)


@pytest.mark.parametrize("conf", [0.0, 1.0, 1.2, -0.5, math.nan, True, "0.95"])
def test_confidence_levels_outside_the_open_unit_interval_raise(conf):
    with pytest.raises(ValueError, match="conf"):
        N.z_two_sided(conf)


@pytest.mark.parametrize("k,n,msg", [(5, 3, "exceeds"), (-1, 3, "negative"), (1, 0, "n=0"),
                                     (1, -2, "negative"), (math.nan, 3, "finite"), (1, math.inf, "finite"),
                                     (True, 3, "finite")])
def test_count_guard_trips(k, n, msg):
    with pytest.raises(ValueError, match=msg):
        _checks.check_counts(k, n)


def test_integer_count_guard_trips_on_fractional_trials():
    with pytest.raises(ValueError, match="whole number"):
        _checks.check_counts(1.5, 3, integer=True)
    assert _checks.check_counts(1.5, 3) == (1.5, 3.0)   # effective counts are real


def test_array_guards_trip():
    with pytest.raises(ValueError, match="missing"):
        _checks.as_float_array([1.0, math.nan], "x")
    with pytest.raises(ValueError, match="0/1"):
        _checks.as_binary_array([0, 1, 2], "x")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _checks.as_unit_array([0.2, 1.3], "x")
    with pytest.raises(ValueError, match="one-dimensional"):
        _checks.as_float_array([[1.0]], "x")
    with pytest.raises(ValueError, match="at least 2"):
        _checks.as_float_array([1.0], "x", min_len=2)
    with pytest.raises(ValueError, match="whole number"):
        _checks.check_count(2.5, "reps")
    with pytest.raises(ValueError, match="at least 2"):
        _checks.check_count(1, "reps", minimum=2)
