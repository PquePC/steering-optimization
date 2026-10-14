"""m5.stats.summaries and m5.stats.halving: R*, the frontier, W*, P_win/A_win, the keep rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from m5.stats import halving as H
from m5.stats import summaries as S
from m5.stats.effect import expit
from m5.stats.psychometric import detection_threshold


def layer(layer_id, *, n=400, mu=0.0, s=0.1, floor=0.02, lapse=0.02, bottom=0.0, top=0.8,
          u50=-0.5, es=0.1, u_s=math.inf, u_max=1.0, sd=0.0, seed=0):
    rng = np.random.default_rng(seed)
    jitter = (lambda m: m + rng.normal(0, sd, n)) if sd else (lambda m: np.full(n, float(m)))
    return S.LayerDraws(layer=layer_id, mu=jitter(mu), s=np.full(n, s), floor=np.full(n, floor),
                        lapse=np.full(n, lapse), eff_bottom=np.full(n, bottom), eff_top=np.full(n, top),
                        eff_u50=jitter(u50), eff_s=np.full(n, es), u_s=np.full(n, u_s), u_max=u_max)


def effect_value(u, bottom=0.0, top=0.8, u50=-0.5, es=0.1):
    return bottom + (top - bottom) * float(expit((u - u50) / es))


def test_r_layer_is_the_effect_at_the_detection_threshold():
    d = layer(5)
    ud = float(detection_threshold(0.0, 0.1, 0.02, 0.02, 0.1))
    r = S.r_star([d], 0.1, 0.95, 0.05)
    assert r["median"] == pytest.approx(effect_value(ud)) and r["lo"] == pytest.approx(r["hi"])
    assert r["flags"] == [] and r["best_layer"] == 5 and r["p_best"] == {5: 1.0}
    assert r["per_layer"][0]["p_detection_binds"] == 1.0


def test_integrity_failing_first_is_flagged_s_limited():
    d = layer(5, mu=0.5, u_s=-0.2)          # integrity fails before detection rises
    r = S.r_star([d], 0.1, 0.95, 0.05)
    assert r["median"] == pytest.approx(effect_value(-0.2))
    assert set(r["flags"]) == {"S-limited", "floor"}


def test_detection_that_never_rises_in_range_is_floor_and_censored():
    d = layer(5, mu=5.0, u_max=0.3)          # nothing fails, nothing is noticed, up to the top dose
    r = S.r_star([d], 0.1, 0.95, 0.05)
    assert r["median"] == pytest.approx(effect_value(0.3))      # never extrapolated past u_max
    assert set(r["flags"]) == {"floor", "censored"}


def test_an_effect_that_never_rises_is_flagged():
    d = layer(5, top=0.02, mu=0.5)
    assert "no-effect" in S.r_star([d], 0.1, 0.95, 0.05)["flags"]
    late = layer(5, u50=3.0, mu=0.5, u_max=1.0)     # rises only far beyond the range searched
    assert "no-effect" in S.r_star([late], 0.1, 0.95, 0.05)["flags"]


def test_r_star_takes_the_best_layer_per_draw():
    a = layer(10, mu=-0.6, sd=0.05, seed=1)          # detection rises early: little effect reachable
    b = layer(11, mu=-0.2, sd=0.05, seed=2)
    c = layer(12, mu=-0.25, sd=0.05, seed=3)
    r = S.r_star([a, b, c], 0.1, 0.95, 0.05)
    assert r["p_best"][10] == 0.0 and r["p_best"][11] + r["p_best"][12] == pytest.approx(1.0)
    assert r["best_layer"] == 11
    per = {row["layer"]: row for row in r["per_layer"]}
    assert r["median"] >= per[11]["median"] - 1e-12 and r["median"] >= per[12]["median"] - 1e-12
    assert r["lo"] <= r["median"] <= r["hi"]


def test_frontier_rises_with_q():
    d = [layer(1, sd=0.05, seed=4), layer(2, mu=0.2, sd=0.05, seed=5)]
    f = S.frontier(d, [0.05, 0.10, 0.25, 0.50], 0.95, 0.05)
    medians = [row["median"] for row in f]
    assert all(b >= a - 1e-12 for a, b in zip(medians, medians[1:]))
    assert [row["q"] for row in f] == [0.05, 0.10, 0.25, 0.50]


def test_w_star_is_the_gap_between_detection_and_effect():
    d = layer(3, mu=0.0, u50=-0.5)
    w = S.w_star([d], 0.95)[3]
    expected = float(detection_threshold(0.0, 0.1, 0.02, 0.02, 0.5)) + 0.5
    assert w["median"] == pytest.approx(expected) and w["p_positive"] == 1.0
    never = S.w_star([layer(3, lapse=0.6)], 0.95)[3]    # ceiling 0.38 < floor + 0.5
    assert never["p_never"] == 1.0 and never["median"] is None
    flat = S.w_star([layer(3, top=0.0)], 0.95)[3]        # no effect: no half-plateau dose
    assert flat["p_flat_effect"] == 1.0 and flat["p_positive"] == 0.0 and flat["median"] is None


def test_p_win_and_the_conservative_excursion_set():
    d = layer(7, mu=0.0, u50=-0.5, sd=0.02, seed=6)
    grid = np.linspace(-1.0, 0.5, 16)
    res = S.p_win([d], grid, effect_min=0.25, detection_max=0.3, relative_to_floor=False, level=0.95)
    assert res["P_win"] == 1.0
    assert 0 < res["A_win_count"] <= res["marginal_count"] <= res["n_cells"]
    assert res["joint_probability"] >= 0.95
    assert res["A_win_area"] == pytest.approx(0.1 * res["A_win_count"])
    for _, u in res["conservative_cells"]:
        assert effect_value(u) > 0.2           # every certified cell is effective...
        assert u < 0.2                          # ...and below detection


def test_conservative_set_requires_joint_not_marginal_probability():
    # layer 1's cells meet the box in 97% of draws, layer 2's in 96%, never failing together: all
    # four are in the marginal 95% set, but the four together hold in only 93% of draws, so the
    # conservative set (Vorob'ev level sets, ties entering together) stops at layer 1's two
    n = 1000
    mu = np.zeros(n)
    mu[:30] = -10.0        # in these draws detection is already high everywhere at layer 1
    a = S.LayerDraws(layer=1, mu=mu, s=np.full(n, 0.1), floor=np.full(n, 0.0), lapse=np.full(n, 0.0),
                     eff_bottom=np.zeros(n), eff_top=np.ones(n), eff_u50=np.full(n, -5.0),
                     eff_s=np.full(n, 0.1), u_s=np.full(n, math.inf), u_max=1.0)
    mu2 = np.zeros(n)
    mu2[40:80] = -10.0
    b = S.LayerDraws(layer=2, mu=mu2, s=np.full(n, 0.1), floor=np.full(n, 0.0), lapse=np.full(n, 0.0),
                     eff_bottom=np.zeros(n), eff_top=np.ones(n), eff_u50=np.full(n, -5.0),
                     eff_s=np.full(n, 0.1), u_s=np.full(n, math.inf), u_max=1.0)
    res = S.p_win([a, b], [-1.0, -0.9], effect_min=0.5, detection_max=0.3, relative_to_floor=True,
                  level=0.95)
    assert res["marginal_count"] == 4 and res["A_win_count"] == 2
    assert {c[0] for c in res["conservative_cells"]} == {1}
    assert res["P_win"] == 1.0 and res["joint_probability"] == pytest.approx(0.97)
    tied = S.p_win([a, S.LayerDraws(**{**b.__dict__, "mu": np.roll(mu, 30)})], [-1.0, -0.9],
                   effect_min=0.5, detection_max=0.3, relative_to_floor=True, level=0.95)
    assert tied["marginal_count"] == 4 and tied["A_win_count"] == 0    # 0.97 each, 0.94 jointly


def test_combine_draws_resamples_independent_sources():
    det = {"mu": np.linspace(-1, 0, 50), "s": np.full(50, 0.1), "floor": np.full(50, 0.02),
           "lapse": np.full(50, 0.02)}
    eff = {"bottom": np.zeros(20), "top": np.full(20, 0.7), "u50": np.linspace(-0.6, -0.4, 20),
           "s": np.full(20, 0.1)}
    d = S.combine_draws(4, det, eff, u_s=0.3, u_max=0.5, n=300, seed=2)
    assert d.n == 300 and np.all(d.u_s == 0.3) and set(np.unique(d.eff_u50)) <= set(eff["u50"])
    again = S.combine_draws(4, det, eff, u_s=0.3, u_max=0.5, n=300, seed=2)
    assert np.array_equal(again.mu, d.mu)
    # a source with exactly n draws keeps its order: joint draws across layers stay aligned
    kept = S.combine_draws(4, det, eff, u_s=np.linspace(0, 1, 50), u_max=0.5, n=50, seed=9)
    assert np.array_equal(kept.mu, det["mu"]) and np.array_equal(kept.u_s, np.linspace(0, 1, 50))


def test_chain_joint_draws_feed_r_star_end_to_end():
    from m5.stats import psychometric as P
    mu = P.make_mu_grid(-3.0, 1.0, 81)
    rng = np.random.default_rng(0)
    posts = []
    for m in (-1.2, -1.0, -0.8):
        post = P.LayerPosterior(P.psychometric_prior(
            mu, center=m, narrow_weight=0.8, narrow_sd=0.15, wide_sd=1.0, floor_ab=(1, 49),
            lapse_ab=(1, 49), slope_median=0.1, slope_logsd=0.6, n_s=5, n_floor=3, n_lapse=3))
        for u in np.linspace(m - 0.3, m + 0.3, 5):
            post.update(float(u), int(rng.binomial(8, float(P.detection_curve(u, m, 0.1, 0.02, 0.02)))), 8)
        posts.append(post)
    joint = P.LayerChain(posts, layers=[10, 11, 12], ar1=0.8, jump_df=3.0, sigma=1.0).joint_draws(500, seed=1)
    eff = {"bottom": np.zeros(40), "top": np.full(40, 0.8), "u50": np.linspace(-1.6, -1.4, 40),
           "s": np.full(40, 0.2)}
    layers = [S.combine_draws(lay, joint[i], eff, u_s=0.5, u_max=0.5, n=500, seed=i)
              for i, lay in enumerate((10, 11, 12))]
    r = S.r_star(layers, 0.10, 0.95, 0.05)
    assert r["best_layer"] == 12 and r["p_best"][12] > 0.9      # latest detection, most room
    assert r["flags"] == [] and 0.5 < r["lo"] <= r["median"] <= r["hi"] < 0.8
    w = S.w_star(layers, 0.95)
    assert w[12]["median"] > w[10]["median"] > 0


def test_keep_layers_drops_only_the_confidently_worse():
    keep = H.keep_layers([0.1, 0.3, 0.2, 0.05], [0.25, 0.5, 0.35, 0.31])
    assert keep.tolist() == [False, True, True, True]
    assert H.keep_layers([0.2], [0.4]).tolist() == [True]


def test_apt_samples_the_least_settled_arm():
    idx = H.apt_index([0.1, 0.5, 0.3], [10, 10, 0], 0.3, 0.05)
    np.testing.assert_allclose(idx, [np.sqrt(10) * 0.25, np.sqrt(10) * 0.25, 0.0])
    assert H.apt_next([0.1, 0.5, 0.3], [10, 10, 0], 0.3, 0.05) == 2       # unsampled first
    assert H.apt_next([0.1, 0.28, 0.6], [10, 10, 10], 0.3, 0.05) == 1    # nearest the threshold


# ------------------------------------------------------------------------------- guards

def test_layer_draws_guards():
    base = dict(layer=1, mu=np.zeros(3), s=np.ones(3), floor=np.zeros(3), lapse=np.zeros(3),
                eff_bottom=np.zeros(3), eff_top=np.ones(3), eff_u50=np.zeros(3), eff_s=np.ones(3),
                u_s=np.zeros(3), u_max=1.0)
    for change, msg in [({"mu": np.zeros(4)}, "aligned"), ({"s": np.array([1.0, 0.0, 1.0])}, "positive"),
                        ({"floor": np.full(3, 0.6), "lapse": np.full(3, 0.5)}, "floor \\+ lapse"),
                        ({"eff_top": np.full(3, -1.0)}, "must rise"),
                        ({"u_s": np.array([0.0, np.nan, 0.0])}, "u_s"),
                        ({"mu": np.array([0.0, np.inf, 0.0])}, "non-finite"),
                        ({"mu": np.zeros(0)}, "no draws")]:
        with pytest.raises(ValueError, match=msg):
            S.LayerDraws(**{**base, **change})


def test_summary_guards():
    a, b = layer(1, n=10), layer(2, n=20)
    with pytest.raises(ValueError, match="same number"):
        S.r_star([a, b], 0.1, 0.95, 0.05)
    with pytest.raises(ValueError, match="duplicate"):
        S.r_star([a, layer(1, n=10)], 0.1, 0.95, 0.05)
    with pytest.raises(ValueError, match="no layers"):
        S.r_star([], 0.1, 0.95, 0.05)
    with pytest.raises(ValueError, match="q"):
        S.r_star([a], 1.0, 0.95, 0.05)
    with pytest.raises(ValueError, match="increasing"):
        S.frontier([a], [0.1, 0.05], 0.95, 0.05)
    with pytest.raises(ValueError, match="increasing"):
        S.p_win([a], [0.0, -1.0], effect_min=0.2, detection_max=0.3, relative_to_floor=False, level=0.95)
    with pytest.raises(ValueError, match="searched range"):
        S.p_win([a], [2.0, 3.0], effect_min=0.2, detection_max=0.3, relative_to_floor=False, level=0.95)
    with pytest.raises(ValueError, match="aligned"):
        S.combine_draws(1, {"mu": [0.0], "s": [1.0, 1.0], "floor": [0.0], "lapse": [0.0]},
                        {"bottom": [0], "top": [1], "u50": [0], "s": [1]}, 0.0, 1.0, n=5, seed=0)


def test_halving_guards():
    with pytest.raises(ValueError, match="at most its upper"):
        H.keep_layers([0.3, 0.1], [0.2, 0.4])
    with pytest.raises(ValueError, match="same length"):
        H.keep_layers([0.1], [0.2, 0.3])
    with pytest.raises(ValueError, match="negative"):
        H.apt_index([0.1, 0.2], [1, -1], 0.3, 0.05)
    with pytest.raises(ValueError, match="eps"):
        H.apt_index([0.1, 0.2], [1, 1], 0.3, -0.05)
