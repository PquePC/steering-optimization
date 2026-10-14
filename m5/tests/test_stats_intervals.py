"""m5.stats.intervals: textbook values, scipy agreement, the equivalence claim, and the guards."""

from __future__ import annotations

import math

import pytest

from m5.stats import intervals as I


def test_zero_event_bounds_are_the_plan_numbers():
    # plan §6.4: with zero events the 95% upper end is 11.6% at n=30, 3.0% at 120, 2.4% at 150
    assert I.clopper_pearson(0, 30, 0.95) == (0.0, pytest.approx(0.1157, abs=5e-4))
    assert I.clopper_pearson(0, 120, 0.95)[1] == pytest.approx(0.0303, abs=5e-4)
    assert I.clopper_pearson(0, 150, 0.95)[1] == pytest.approx(0.0243, abs=5e-4)
    assert I.wilson(0, 30, 0.95) == (0.0, pytest.approx(0.1135, abs=5e-4))   # z^2/(n + z^2)
    # one-sided exact bound at k = 0 is 1 - alpha^(1/n)
    assert I.clopper_pearson_upper(0, 30, 0.95) == pytest.approx(1 - 0.05 ** (1 / 30), rel=1e-12)


def test_zero_events_never_give_a_zero_width_interval():
    # handoff 17: 0/25 must not read 0.000 +- 0.000
    lo, hi = I.wilson(0, 25, 0.95)
    assert lo == 0.0 and hi > 0.1
    for k, n in [(1, 1), (25, 25), (0, 1)]:
        lo, hi = I.wilson(k, n, 0.95)
        assert 0.0 <= lo < hi <= 1.0
        lo, hi = I.clopper_pearson(k, n, 0.95)
        assert 0.0 <= lo < hi <= 1.0
    assert I.wilson(25, 25, 0.95)[1] == 1.0 and I.clopper_pearson(25, 25, 0.95)[1] == 1.0


def test_wilson_and_clopper_pearson_match_scipy():
    st = pytest.importorskip("scipy.stats")
    for n in (1, 7, 30, 150):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            if k < 0:
                continue
            ref_w = st.binomtest(k, n).proportion_ci(0.95, method="wilson")
            ref_cp = st.binomtest(k, n).proportion_ci(0.95, method="exact")
            assert I.wilson(k, n, 0.95) == (pytest.approx(ref_w.low, abs=1e-12),
                                            pytest.approx(ref_w.high, abs=1e-12))
            assert I.clopper_pearson(k, n, 0.95) == (pytest.approx(ref_cp.low, abs=1e-10),
                                                     pytest.approx(ref_cp.high, abs=1e-10))


def test_one_sided_bounds_sit_inside_the_two_sided_interval():
    lo2, hi2 = I.wilson(4, 40, 0.95)
    assert lo2 < I.wilson_lower(4, 40, 0.95) < I.wilson_upper(4, 40, 0.95) < hi2
    lo2, hi2 = I.clopper_pearson(4, 40, 0.95)
    assert lo2 < I.clopper_pearson_lower(4, 40, 0.95) < I.clopper_pearson_upper(4, 40, 0.95) < hi2
    assert I.clopper_pearson_lower(0, 40, 0.95) == 0.0 and I.clopper_pearson_upper(40, 40, 0.95) == 1.0


def test_newcombe_method_10_reproduces_the_paper():
    # Newcombe (1998), Statistics in Medicine 17:873-890, Table II, method 10
    cases = [((56, 70, 48, 80), (0.0524, 0.3339)), ((9, 10, 3, 10), (0.1705, 0.8090)),
             ((5, 56, 0, 29), (-0.0381, 0.1926)), ((0, 10, 0, 20), (-0.1611, 0.2775))]
    for (k1, n1, k2, n2), (lo, hi) in cases:
        d, l, h = I.newcombe_diff(k1, n1, k2, n2, 0.95)
        assert d == pytest.approx(k1 / n1 - k2 / n2)
        assert (l, h) == (pytest.approx(lo, abs=1e-4), pytest.approx(hi, abs=1e-4))


def test_equivalence_needs_a_tight_upper_bound_not_a_failed_difference_test():
    big = I.tost_rate_difference(2, 300, 3, 300, 0.05, 0.95)
    assert big.equivalent and big.equivalent_two_sided and big.upper < 0.05
    small = I.tost_rate_difference(0, 5, 0, 5, 0.05, 0.95)   # no difference, and no evidence
    assert not small.equivalent and small.diff == 0.0
    # the one-sided bound is the end of the (2 conf - 1) two-sided interval
    _, lo90, hi90 = I.newcombe_diff(2, 300, 3, 300, 0.90)
    assert big.upper == pytest.approx(hi90) and big.lower == pytest.approx(lo90)
    # clustering widens it: the same counts with a design effect of 6 are not equivalent
    assert I.tost_rate_difference(3, 300, 3, 300, 0.05, 0.95).equivalent
    assert not I.tost_rate_difference(3, 300, 3, 300, 0.05, 0.95, deff=6.0).equivalent


def test_steered_below_null_does_not_block_a_covertness_claim():
    eq = I.tost_rate_difference(0, 300, 30, 300, 0.05, 0.95)
    assert eq.equivalent and not eq.equivalent_two_sided


def test_rate_with_null_record_is_complete_and_json_ready():
    rec = I.rate_with_null(3, 60, 1, 300, 0.95, deff=1.5, delta=0.05)
    for key in ("k", "n", "rate", "lo", "hi", "cp_hi", "null_k", "null_n", "null_rate", "null_lo",
                "null_hi", "diff", "diff_lo", "diff_hi", "conf", "deff", "n_eff", "equivalent",
                "equiv_upper"):
        assert key in rec
    assert rec["k"] == 3 and rec["n"] == 60 and rec["n_eff"] == pytest.approx(40.0)
    assert rec["lo"] <= rec["rate"] <= rec["hi"] and rec["null_lo"] <= rec["null_rate"] <= rec["null_hi"]
    assert rec["diff_lo"] <= rec["diff"] <= rec["diff_hi"]
    assert rec["equivalent"] is False
    plain = I.rate_with_null(3, 60, 1, 300, 0.95)
    assert plain["deff"] == 1.0 and "equivalent" not in plain
    assert plain["hi"] - plain["lo"] < rec["hi"] - rec["lo"]     # the design effect widens it
    assert all(not (isinstance(v, float) and math.isnan(v)) for v in rec.values())


def test_no_nan_at_the_ends():
    for k, n in [(0, 10), (10, 10)]:
        for f in (I.wilson, I.clopper_pearson):
            assert all(math.isfinite(v) for v in f(k, n, 0.95))
        assert all(math.isfinite(v) for v in I.newcombe_diff(k, n, n - k, n, 0.95))


# ------------------------------------------------------------------------------- guards

@pytest.mark.parametrize("f", [I.wilson, I.wilson_upper, I.wilson_lower, I.clopper_pearson,
                               I.clopper_pearson_upper, I.clopper_pearson_lower])
@pytest.mark.parametrize("k,n,conf,msg", [(4, 3, 0.95, "exceeds"), (1, 0, 0.95, "n=0"),
                                          (-1, 5, 0.95, "negative"), (1, 5, 1.0, "conf"),
                                          (1, 5, 0.0, "conf")])
def test_interval_guards(f, k, n, conf, msg):
    with pytest.raises(ValueError, match=msg):
        f(k, n, conf)


def test_newcombe_guard_names_the_rate():
    with pytest.raises(ValueError, match="second rate"):
        I.newcombe_diff(1, 10, 11, 10, 0.95)


@pytest.mark.parametrize("kwargs,msg", [({"delta": 0.0}, "delta"), ({"delta": 1.2}, "delta"),
                                        ({"conf": 0.4}, "above 0.5"), ({"deff": 0.5}, "design effect"),
                                        ({"null_deff": 0.9}, "design effect")])
def test_equivalence_guards(kwargs, msg):
    args = {"delta": 0.05, "conf": 0.95}
    args.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        I.tost_rate_difference(1, 100, 1, 100, args.pop("delta"), args.pop("conf"), **args)


def test_rate_with_null_guards():
    with pytest.raises(ValueError, match="null"):
        I.rate_with_null(1, 10, 5, 0, 0.95)
    with pytest.raises(ValueError, match="steered"):
        I.rate_with_null(11, 10, 5, 50, 0.95)
    with pytest.raises(ValueError, match="design effect"):
        I.rate_with_null(1, 10, 5, 50, 0.95, deff=0.0)
