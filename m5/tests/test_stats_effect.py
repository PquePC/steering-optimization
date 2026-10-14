"""m5.stats.effect: isotonic regression, the four-parameter curve, the prompt bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from m5.stats import effect as E

TRUE = {"bottom": 0.05, "top": 0.8, "u50": -1.0, "s": 0.15}


def hill(u, bottom, top, u50, s):
    return bottom + (top - bottom) / (1 + np.exp(-(np.asarray(u) - u50) / s))


def isotonic_reference(y, w):
    """The min-max formula: f_i = max_{j<=i} min_{k>=i} weighted mean of y_j..y_k."""
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        best = -np.inf
        for j in range(i + 1):
            worst = np.inf
            for k in range(i, n):
                worst = min(worst, float(np.dot(w[j:k + 1], y[j:k + 1]) / w[j:k + 1].sum()))
            best = max(best, worst)
        out[i] = best
    return out


def test_isotonic_known_values_and_reference():
    np.testing.assert_allclose(E.isotonic([1, 3, 2, 4, 3.5, 5]), [1, 2.5, 2.5, 3.75, 3.75, 5])
    np.testing.assert_allclose(E.isotonic([1, 3, 2, 4], [1, 1, 3, 1]), [1, 2.25, 2.25, 4])
    rng = np.random.default_rng(0)
    for _ in range(25):
        y = rng.normal(size=9)
        w = rng.uniform(0.2, 3.0, size=9)
        np.testing.assert_allclose(E.isotonic(y, w), isotonic_reference(y, w), atol=1e-12)
    np.testing.assert_allclose(E.isotonic([3, 1, 2], increasing=False), [3, 1.5, 1.5])


def test_isotonic_by_dose_pools_repeats_and_finds_the_half_dose():
    u = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float)
    y = np.array([0, 0.2, 0.3, 0.1, 0.9, 0.7, 1.0, 1.0])
    doses, fit, w = E.isotonic_by_dose(u, y)
    np.testing.assert_allclose(doses, [0, 1, 2, 3]) and np.testing.assert_allclose(w, [2, 2, 2, 2])
    np.testing.assert_allclose(fit, [0.1, 0.2, 0.8, 1.0])
    assert E.isotonic_half_dose(u, y) == pytest.approx(1 + (0.55 - 0.2) / 0.6)
    assert E.isotonic_half_dose([0, 1, 2], [0.5, 0.4, 0.3]) is None      # flat once made monotone


def test_hill_fit_recovers_noiseless_parameters():
    u = np.linspace(-2, 0, 9)
    f = E.fit_hill(u, hill(u, **TRUE))
    assert (f.bottom, f.top, f.u50, f.s) == (pytest.approx(0.05, abs=1e-5), pytest.approx(0.8, abs=1e-5),
                                             pytest.approx(-1.0, abs=1e-4), pytest.approx(0.15, rel=1e-3))
    assert f.sse < 1e-10 and E.half_plateau_dose(f) == f.u50
    assert f(f.u50) == pytest.approx(0.5 * (f.bottom + f.top))
    assert f.dose_at_fraction(0.5) == pytest.approx(f.u50)
    assert f(f.dose_at_fraction(0.9)) == pytest.approx(f.bottom + 0.9 * (f.top - f.bottom))


def test_hill_fit_is_monotone_and_respects_bounds():
    flat = E.fit_hill([0, 1, 2, 3], [0.5, 0.4, 0.3, 0.2])        # falling data -> the best flat line
    assert flat.top == flat.bottom == pytest.approx(0.35)
    assert E.half_plateau_dose(flat) is None                     # no rise, no half-plateau dose
    u = np.linspace(-2, 0, 7)
    y = np.clip(hill(u, -0.1, 1.2, -1.0, 0.3), -1, 2)              # overshoots a rate's range
    bounded = E.fit_hill(u, y, bounds=(0.0, 1.0))
    assert 0.0 <= bounded.bottom <= bounded.top <= 1.0
    free = E.fit_hill(u, y)
    assert free.sse <= bounded.sse + 1e-12


def test_hill_bootstrap_by_prompt_has_the_estimators_spread():
    # 20 prompts x 8 doses with a per-prompt offset: across 40 simulated data sets the fitted u50
    # has sd 0.0145 and the 95% prompt-bootstrap interval covered the truth in 38 of 40
    rng = np.random.default_rng(0)
    n_prompts, doses = 20, np.linspace(-2, 0, 8)
    u = np.tile(doses, n_prompts)
    pid = np.repeat(np.arange(n_prompts), doses.size)
    y = hill(u, **TRUE) + rng.normal(0, 0.05, n_prompts)[pid] + rng.normal(0, 0.08, u.size)
    res = E.bootstrap_hill(u, y, pid, reps=60, seed=1, conf=0.95)
    se = float(res.draws["u50"].std(ddof=1))
    assert 0.008 < se < 0.03
    assert abs(res.fit.u50 - TRUE["u50"]) < 4 * se
    lo, hi = res.intervals["u50"]
    assert lo < res.fit.u50 < hi
    assert res.n_failed == 0 and all(v.size == 60 for v in res.draws.values())
    assert np.all(res.draws["top"] >= res.draws["bottom"])
    again = E.bootstrap_hill(u, y, pid, reps=60, seed=1, conf=0.95)
    assert again.intervals == res.intervals


# ------------------------------------------------------------------------------- guards

def test_isotonic_guards():
    with pytest.raises(ValueError, match="positive"):
        E.isotonic([1, 2], [1, 0])
    with pytest.raises(ValueError, match="entries"):
        E.isotonic([1, 2], [1])
    with pytest.raises(ValueError, match="missing"):
        E.isotonic([1, float("nan")])
    with pytest.raises(ValueError, match="same length"):
        E.isotonic_by_dose([0, 1], [0.1])


def test_hill_guards():
    with pytest.raises(ValueError, match="three distinct doses"):
        E.fit_hill([0, 0, 1, 1], [0.1, 0.2, 0.5, 0.6])
    with pytest.raises(ValueError, match="lo < hi"):
        E.fit_hill([0, 1, 2], [0.1, 0.2, 0.3], bounds=(1, 0))
    with pytest.raises(ValueError, match="same length"):
        E.fit_hill([0, 1, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="positive"):
        E.fit_hill([0, 1, 2], [0.1, 0.2, 0.3], w=[1, 0, 1])
    with pytest.raises(ValueError, match="same length"):
        E.bootstrap_hill([0, 1, 2], [0.1, 0.2, 0.3], [0, 1], reps=5, seed=0, conf=0.9)
    f = E.fit_hill([0, 1, 2], [0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="f"):
        f.dose_at_fraction(1.0)
