"""m5.stats.sdt and m5.stats.paired: d-prime, operating points, McNemar, paired differences."""

from __future__ import annotations

import numpy as np
import pytest

from m5.stats import paired as Pd
from m5.stats import sdt as T
from m5.stats.normal import norm_ppf


def test_dprime_with_the_log_linear_correction():
    r = T.dprime(40, 50, 5, 50)
    h, f = 40.5 / 51, 5.5 / 51
    assert r.hit_rate == pytest.approx(h) and r.fa_rate == pytest.approx(f)
    assert r.dprime == pytest.approx(norm_ppf(h) - norm_ppf(f))
    assert r.criterion == pytest.approx(-0.5 * (norm_ppf(h) + norm_ppf(f)))
    assert r.se_dprime > 0


def test_dprime_is_finite_at_a_perfect_floor_and_zero_without_signal():
    perfect = T.dprime(50, 50, 0, 50)
    assert np.isfinite(perfect.dprime) and perfect.dprime > 3
    none = T.dprime(0, 50, 0, 50)
    assert none.dprime == pytest.approx(0.0, abs=1e-12) and none.criterion > 2   # says no to everything


def test_a_more_willing_model_moves_the_criterion_not_d_prime():
    strict = T.dprime(30, 100, 5, 100)
    liberal = T.dprime(70, 100, 25, 100)
    assert liberal.criterion < strict.criterion
    assert abs(liberal.dprime - strict.dprime) < 0.6


def test_tpr_at_fpr_never_exceeds_the_requested_fpr():
    noise = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.95, 0.99]
    signal = [0.9, 0.8, 0.7, 0.6, 0.5]
    r = T.tpr_at_fpr(signal, noise, 0.2)
    assert r["fpr"] <= 0.2 and r["threshold"] == 0.85 and r["tpr"] == pytest.approx(0.2)
    tied = T.tpr_at_fpr([0.5, 0.5, 0.9], [0.5, 0.5, 0.5, 0.1], 0.25)
    assert tied["fpr"] <= 0.25 and tied["tpr"] == pytest.approx(1 / 3)
    every = T.tpr_at_fpr(signal, noise, 1.0)
    assert every["threshold"] == -np.inf and every["tpr"] == 1.0
    rng = np.random.default_rng(0)
    for fpr in (0.01, 0.05, 0.3):
        r = T.tpr_at_fpr(rng.normal(1, 1, 200), rng.normal(0, 1, 333), fpr)
        assert r["fpr"] <= fpr


def test_mcnemar_matches_the_exact_binomial_test():
    st = pytest.importorskip("scipy.stats")
    for b, c in [(10, 2), (3, 3), (0, 7), (15, 4)]:
        assert Pd.mcnemar_exact(b, c) == pytest.approx(st.binomtest(min(b, c), b + c, 0.5).pvalue)
    assert Pd.mcnemar_exact(0, 0) == 1.0


def test_paired_difference_uses_only_discordant_pairs_and_is_antisymmetric():
    pairs = np.array([[1, 1]] * 30 + [[1, 0]] * 10 + [[0, 1]] * 2 + [[0, 0]] * 20)
    r = Pd.paired_rate_difference(pairs, 0.95)
    assert (r["a"], r["b"], r["c"], r["d"]) == (30, 10, 2, 20)
    assert r["diff"] == pytest.approx(8 / 62) and r["lo"] < r["diff"] < r["hi"]
    assert r["lo"] > 0 and r["p_mcnemar"] < 0.05
    flipped = Pd.paired_rate_difference(pairs[:, ::-1], 0.95)
    assert flipped["lo"] == pytest.approx(-r["hi"]) and flipped["hi"] == pytest.approx(-r["lo"])


def test_paired_difference_is_narrower_than_treating_the_arms_as_independent():
    from m5.stats.intervals import newcombe_diff
    rng = np.random.default_rng(1)
    base = rng.binomial(1, 0.6, 300)
    flip = rng.random(300) < 0.08
    pairs = np.stack([base, np.where(flip, 1 - base, base)], axis=1)
    r = Pd.paired_rate_difference(pairs, 0.95)
    _, lo, hi = newcombe_diff(pairs[:, 0].sum(), 300, pairs[:, 1].sum(), 300, 0.95)
    assert r["hi"] - r["lo"] < 0.6 * (hi - lo)


def test_paired_difference_covers_in_simulation():
    rng = np.random.default_rng(2)
    p11, p10, p01 = 0.4, 0.12, 0.05
    truth = p10 - p01
    cover = 0
    for _ in range(400):
        cells = rng.choice(4, size=60, p=[p11, p10, p01, 1 - p11 - p10 - p01])
        pairs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])[cells]
        r = Pd.paired_rate_difference(pairs, 0.95)
        cover += r["lo"] <= truth <= r["hi"]
    assert cover / 400 >= 0.92


def test_no_discordant_pairs_still_bound_the_difference():
    # the paired "0 events": Tango's interval is +-z^2/(n + z^2), never 0.000 +- 0.000
    z2 = 1.959963984540054 ** 2
    r = Pd.paired_rate_difference(np.array([[1, 1]] * 5 + [[0, 0]] * 5), 0.95)
    assert r["diff"] == 0.0 and r["p_mcnemar"] == 1.0
    assert (r["lo"], r["hi"]) == (pytest.approx(-z2 / (10 + z2)), pytest.approx(z2 / (10 + z2)))
    all_one = Pd.paired_rate_difference(np.ones((8, 2)), 0.95)
    assert all_one["lo"] < 0 < all_one["hi"]


def test_tango_restricted_mle_maximises_the_profile_likelihood():
    a, b, c, d = 30, 10, 2, 20
    n = a + b + c + d
    for delta in (-0.2, 0.0, 0.05, 0.3):
        xs = np.linspace(max(0.0, -delta) + 1e-9, (1 - delta) / 2 - 1e-9, 200_001)
        ll = (a + d) * np.log(1 - 2 * xs - delta) + b * np.log(xs + delta) + c * np.log(xs)
        x_grid = xs[np.argmax(ll)]
        big_b = -b - c + (2 * n - b + c) * delta
        x = (np.sqrt(big_b ** 2 + 8 * n * c * delta * (1 - delta)) - big_b) / (4 * n)
        assert x == pytest.approx(x_grid, abs=1e-5)
    # and the interval's ends are where the score statistic reaches +-z
    r = Pd.paired_rate_difference(np.array([[1, 1]] * a + [[1, 0]] * b + [[0, 1]] * c + [[0, 0]] * d), 0.95)
    assert abs(Pd._tango_z(r["lo"], b, c, n)) == pytest.approx(1.959963984540054, abs=1e-6)
    assert abs(Pd._tango_z(r["hi"], b, c, n)) == pytest.approx(1.959963984540054, abs=1e-6)


def test_newcombe_paired_option():
    pairs = np.array([[1, 1]] * 30 + [[1, 0]] * 10 + [[0, 1]] * 2 + [[0, 0]] * 20)
    r = Pd.paired_rate_difference(pairs, 0.95, method="newcombe")
    assert r["method"] == "newcombe-paired-10" and r["phi"] == pytest.approx(0.6310, abs=1e-4)
    assert r["lo"] == pytest.approx(0.0241, abs=1e-4) and r["hi"] == pytest.approx(0.2283, abs=1e-4)
    with pytest.raises(ValueError, match="method"):
        Pd.paired_rate_difference(pairs, 0.95, method="wald")


# ------------------------------------------------------------------------------- guards

@pytest.mark.parametrize("args,msg", [((60, 50, 1, 50), "exceeds"), ((5, 0, 1, 50), "n=0"),
                                      ((5, 50, -1, 50), "negative"), ((5.5, 50, 1, 50), "whole number")])
def test_dprime_guards(args, msg):
    with pytest.raises(ValueError, match=msg):
        T.dprime(*args)


def test_tpr_at_fpr_guards():
    with pytest.raises(ValueError, match="fpr"):
        T.tpr_at_fpr([0.5], [0.1], 1.5)
    with pytest.raises(ValueError, match="at least 1"):
        T.tpr_at_fpr([], [0.1], 0.1)
    with pytest.raises(ValueError, match="missing"):
        T.tpr_at_fpr([0.5, float("nan")], [0.1], 0.1)


def test_paired_guards():
    with pytest.raises(ValueError, match="b"):
        Pd.mcnemar_exact(-1, 3)
    with pytest.raises(ValueError, match="whole number"):
        Pd.mcnemar_exact(1.5, 3)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        Pd.paired_rate_difference(np.ones((5, 3)), 0.95)
    with pytest.raises(ValueError, match="0/1"):
        Pd.paired_rate_difference(np.array([[1, 2], [0, 1]]), 0.95)
    with pytest.raises(ValueError, match="empty"):
        Pd.paired_rate_difference(np.zeros((0, 2)), 0.95)
    with pytest.raises(ValueError, match="conf"):
        Pd.paired_rate_difference(np.ones((3, 2)), 1.5)
