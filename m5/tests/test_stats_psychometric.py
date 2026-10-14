"""m5.stats.psychometric: the per-layer grid posterior and the cross-layer chain.

The search's decisions (where to visit next, when a layer is pinned, which layers survive) all
read this posterior, so it is checked against brute force wherever a brute force exists: the grid
posterior against an explicit loop over cells with scipy's pmfs, the expected information gain
against direct enumeration, the chain's forward-backward against the full joint.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from m5.stats import psychometric as P

Z95 = 1.959963984540054


def small_prior(center=-1.0, *, mu=(-3.0, 1.0, 81), icc=(0.0,), n_s=7, n_fl=3, slope=(0.1, 0.6),
                weight=0.8, control=None):
    return P.psychometric_prior(P.make_mu_grid(*mu), center=center, narrow_weight=weight,
                                narrow_sd=0.15, wide_sd=1.0, floor_ab=(1, 49), lapse_ab=(1, 49),
                                slope_median=slope[0], slope_logsd=slope[1], icc=icc, n_s=n_s,
                                n_floor=n_fl, n_lapse=n_fl, control=control)


def simulate(post, rng, truth, doses, n):
    for u in doses:
        p = float(P.detection_curve(u, truth["mu"], truth["s"], truth["floor"], truth["lapse"]))
        post.update(float(u), int(rng.binomial(n, p)), n)


TRUTH = {"mu": -1.2, "s": 0.12, "floor": 0.02, "lapse": 0.03}


# ------------------------------------------------------------------------------- priors

def test_mu_grid_and_mixture_prior():
    mu = P.make_mu_grid(-4, 2, 601)
    assert mu.size == 601 and mu[0] == -4 and mu[-1] == 2
    lw = P.mixture_mu_logprior(mu, -1.0, narrow_weight=0.8, narrow_sd=0.15, wide_sd=1.0)
    w = np.exp(lw)
    assert w.sum() == pytest.approx(1.0) and float(w @ mu) == pytest.approx(-1.0, abs=1e-3)
    near = w[np.abs(mu + 1.0) <= 0.3].sum()          # 0.8 * 0.954 + 0.2 * 0.236
    assert near == pytest.approx(0.8 * 0.9545 + 0.2 * 0.2358, abs=0.01)
    wide_only = np.exp(P.mixture_mu_logprior(mu, -1.0, narrow_weight=0.0, narrow_sd=0.15, wide_sd=1.0))
    assert wide_only[np.abs(mu + 1.0) <= 0.3].sum() == pytest.approx(0.2358, abs=0.01)


def test_floor_grid_sits_at_beta_quantiles_and_follows_the_control_trials():
    g = P.beta_quantile_grid(1, 49, 5)
    assert np.all(np.diff(g) > 0) and float(g.mean()) == pytest.approx(1 / 50, rel=0.25)
    prior = small_prior(control=(30, 300))       # floor ~ Beta(31, 319): about 0.09
    assert 0.07 < prior.floor.min() and prior.floor.max() < 0.11
    assert small_prior().floor.max() < 0.05


def test_posterior_matches_brute_force_on_a_tiny_grid():
    st = pytest.importorskip("scipy.stats")
    prior = small_prior(mu=(-2.0, 0.0, 7), icc=(0.0, 0.3), n_s=3, n_fl=2)
    post = P.LayerPosterior(prior)
    visits = [(-1.5, 1, 8), (-1.0, 4, 8), (-0.5, 7, 8)]
    for u, k, n in visits:
        post.update(u, k, n)
    post.update_control(2, 60)
    ref = np.zeros(prior.shape)
    for idx in itertools.product(*(range(s) for s in prior.shape)):
        im, is_, ig, il, ic = idx
        lw = (prior.log_w_mu[im] + prior.log_w_s[is_] + prior.log_w_floor[ig] + prior.log_w_lapse[il]
              + prior.log_w_icc[ic])
        g, lam, icc = prior.floor[ig], prior.lapse[il], prior.icc[ic]
        for u, k, n in visits:
            p = g + (1 - g - lam) / (1 + math.exp(-(u - prior.mu[im]) / math.exp(prior.log_s[is_])))
            if icc == 0:
                lw += st.binom.logpmf(k, n, p)
            else:
                th = (1 - icc) / icc
                lw += st.betabinom.logpmf(k, n, p * th, (1 - p) * th)
        lw += st.binom.logpmf(2, 60, g)
        ref[idx] = lw
    ref = np.exp(ref - ref.max())
    ref /= ref.sum()
    np.testing.assert_allclose(post.posterior(), ref, rtol=1e-9, atol=1e-15)


def test_all_outcome_pmf_recurrence_matches_the_log_gamma_form():
    rng = np.random.default_rng(0)
    p = rng.uniform(1e-6, 1 - 1e-6, size=(40, 3))
    for icc in (0.0, 0.05, 0.3, 0.9):
        for n in (1, 5, 30):
            fast = P._log_pmf_all(n, p, icc)
            slow = np.stack([P._log_pmf(k, n, p, icc) for k in range(n + 1)])
            np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(np.exp(fast).sum(axis=0), 1.0, atol=1e-9)


def test_posterior_recovers_a_simulated_curve():
    rng = np.random.default_rng(7)
    post = P.LayerPosterior(small_prior(center=-1.0, mu=(-3.0, 1.0, 201)))
    simulate(post, rng, TRUTH, np.linspace(-1.6, -0.8, 9), 8)
    mean, sd = post.mean_sd_mu()
    assert abs(mean - TRUTH["mu"]) < 3 * sd and sd < 0.1
    ts = post.threshold(0.10, 0.95)
    true_ud = float(P.detection_threshold(TRUTH["mu"], TRUTH["s"], TRUTH["floor"], TRUTH["lapse"], 0.10))
    assert ts.lo <= true_ud <= ts.hi and ts.p_never == 0.0
    assert post.n_trials == 72


def test_threshold_formula_is_the_dose_where_the_curve_exceeds_its_floor_by_q():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mu, s = rng.uniform(-2, 0), rng.uniform(0.05, 0.5)
        g, lam = rng.uniform(0, 0.1), rng.uniform(0, 0.1)
        q = rng.uniform(0.02, 0.8)
        ud = float(P.detection_threshold(mu, s, g, lam, q))
        assert float(P.detection_curve(ud, mu, s, g, lam)) - g == pytest.approx(q, abs=1e-12)
    # a curve whose ceiling is below floor + q never gets there
    assert P.detection_threshold(0.0, 0.1, 0.05, 0.6, 0.5) == np.inf


def test_threshold_reports_the_mass_that_never_rises():
    prior = small_prior()
    post = P.LayerPosterior(prior)
    q = 0.97                                       # above 1 - floor - lapse in 5 of 9 cells
    amp = 1 - prior.floor[:, None] - prior.lapse[None, :]
    expected = float(np.mean(amp <= q))            # floor and lapse grids carry equal weights
    ts = post.threshold(q, 0.95)
    assert 0.5 < expected < 1 and ts.p_never == pytest.approx(expected)
    assert ts.hi == math.inf and ts.to_json()["hi"] is None
    pinned = P.LayerPosterior(small_prior(mu=(-3.0, 1.0, 21)))
    assert pinned.threshold(0.1, 0.95).sd >= (4.0 / 20) / math.sqrt(12)   # never below the grid


def test_expected_information_gain_matches_direct_enumeration():
    st = pytest.importorskip("scipy.stats")
    prior = small_prior(mu=(-2.0, 0.0, 9), icc=(0.0, 0.25), n_s=3, n_fl=2)
    post = P.LayerPosterior(prior)
    post.update(-1.0, 3, 8)
    w = post.posterior()
    n = 5
    for u in (-1.5, -1.0, -0.4):
        joint = np.zeros((n + 1, prior.mu.size))
        for idx in itertools.product(*(range(s) for s in prior.shape)):
            im, is_, ig, il, ic = idx
            g, lam, icc = prior.floor[ig], prior.lapse[il], prior.icc[ic]
            p = g + (1 - g - lam) / (1 + math.exp(-(u - prior.mu[im]) / math.exp(prior.log_s[is_])))
            for k in range(n + 1):
                if icc == 0:
                    pk = st.binom.pmf(k, n, p)
                else:
                    th = (1 - icc) / icc
                    pk = st.betabinom.pmf(k, n, p * th, (1 - p) * th)
                joint[k, im] += w[idx] * pk
        pk_ = joint.sum(1, keepdims=True)
        pm = joint.sum(0, keepdims=True)
        mi = float(np.sum(joint * np.log(joint / (pk_ * pm))))
        assert post.expected_information_gain([u], n)[0] == pytest.approx(mi, rel=1e-8, abs=1e-12)


def test_first_visit_goes_to_the_screen_transition_and_far_doses_teach_nothing():
    post = P.LayerPosterior(small_prior(center=-1.0, mu=(-3.0, 1.0, 161)))
    cands = np.linspace(-3.0, 1.0, 41)
    eig = post.expected_information_gain(cands, 8)
    assert abs(cands[int(np.argmax(eig))] - (-1.0)) <= 0.1 + 1e-9
    assert np.all(eig >= 0) and eig[0] < 0.05 * eig.max()
    eig_q = post.expected_information_gain(cands, 8, q=0.1)
    assert np.all(eig_q >= 0) and eig_q.max() > 0.1


def test_curve_band_and_draws_agree_with_the_posterior():
    rng = np.random.default_rng(3)
    post = P.LayerPosterior(small_prior(mu=(-3.0, 1.0, 161)))
    simulate(post, rng, TRUTH, np.linspace(-1.6, -0.8, 5), 8)
    c = post.curve(np.linspace(-2.5, 0.5, 13), 0.9)
    assert np.all(c["lo"] <= c["mean"] + 1e-12) and np.all(c["mean"] <= c["hi"] + 1e-12)
    assert np.all(np.diff(c["mean"]) >= -1e-12)
    d = post.draws(4000, seed=1)
    mean, sd = post.mean_sd_mu()
    assert float(d["mu"].mean()) == pytest.approx(mean, abs=0.02)
    assert float(d["mu"].std()) == pytest.approx(sd, rel=0.2)
    assert np.all(d["s"] > 0) and np.all(d["floor"] + d["lapse"] < 1)
    again = post.draws(4000, seed=1)
    assert np.array_equal(again["mu"], d["mu"])


def test_ppc_flags_an_inverted_u_and_passes_a_rising_curve():
    rising = P.LayerPosterior(small_prior(mu=(-3.0, 1.0, 121)))
    simulate(rising, np.random.default_rng(1), TRUTH, np.linspace(-1.6, -0.8, 5), 10)
    assert rising.ppc_monotone(300, seed=2)["p_value"] > 0.2
    inverted = P.LayerPosterior(small_prior(mu=(-3.0, 1.0, 121)))
    for u, k in [(-1.6, 0), (-1.4, 4), (-1.2, 10), (-1.0, 4), (-0.8, 0)]:
        inverted.update(u, k, 10)
    res = inverted.ppc_monotone(300, seed=2)
    assert res["p_value"] < 0.05 and res["t_obs"] > 10


def test_overdispersion_is_preferred_when_the_data_carry_it():
    rng = np.random.default_rng(4)
    doses = np.repeat(np.linspace(-1.6, -0.8, 5), 6)

    def fill(post):
        r = np.random.default_rng(5)
        for u in doses:
            p = float(P.detection_curve(u, TRUTH["mu"], TRUTH["s"], TRUTH["floor"], TRUTH["lapse"]))
            th = (1 - 0.35) / 0.35
            q = r.beta(p * th, (1 - p) * th)
            post.update(float(u), int(r.binomial(10, q)), 10)
        return post

    assert rng is not None
    binom = fill(P.LayerPosterior(small_prior(icc=(0.0,))))
    od = fill(P.LayerPosterior(small_prior(icc=(0.35,))))
    assert od.log_evidence() > binom.log_evidence() + 3
    both = fill(P.LayerPosterior(small_prior(icc=(0.0, 0.35))))
    assert both.posterior().sum(axis=(0, 1, 2, 3))[1] > 0.9


def test_control_trials_inform_the_floor_only():
    post = P.LayerPosterior(small_prior())
    before = post.posterior().sum(axis=(0, 1, 3, 4))
    post.update_control(0, 400)
    after = post.posterior().sum(axis=(0, 1, 3, 4))
    assert after[0] > before[0] and post.n_trials == 0
    np.testing.assert_allclose(post.marginal_mu(), P.LayerPosterior(small_prior()).marginal_mu(), atol=1e-12)


def test_log_likelihood_surface_is_read_only_and_copies_are_independent():
    post = P.LayerPosterior(small_prior())
    post.update(-1.0, 3, 8)
    with pytest.raises(ValueError):
        post.log_likelihood[0, 0, 0, 0, 0] = 1.0
    twin = post.copy()
    twin.update(-0.5, 8, 8)
    assert len(post.visits) == 1 and len(twin.visits) == 2
    assert not np.array_equal(post.log_posterior(), twin.log_posterior())


# ------------------------------------------------------------------------------- the chain

def tiny_chain_layers(seed=0):
    mu = (-2.0, 2.0, 15)
    rng = np.random.default_rng(seed)
    posts = []
    for layer in range(3):
        p = P.LayerPosterior(P.psychometric_prior(
            P.make_mu_grid(*mu), center=0.0, narrow_weight=0.5, narrow_sd=0.5, wide_sd=1.0,
            floor_ab=(1, 49), lapse_ab=(1, 49), slope_median=0.3, slope_logsd=0.5, n_s=3, n_floor=2,
            n_lapse=2))
        p.update(-0.5 + 0.5 * layer, int(rng.integers(0, 9)), 8)
        p.update(0.3, int(rng.integers(0, 9)), 8)
        posts.append(p)
    return posts


def test_chain_marginals_are_exact():
    posts = tiny_chain_layers()
    chain = P.LayerChain(posts, layers=[3, 4, 6], ar1=0.8, jump_df=3.0, sigma=1.0)
    mu = posts[0].prior.mu
    emis = [p.log_marginal_mu_unnormalised() for p in posts]

    def log_k(gap):
        tau = math.sqrt(1 - 0.8 ** (2 * gap))
        d = (mu[None, :] - mu[:, None]) / tau
        return -2.0 * np.log1p(d * d / 3.0)

    joint = (emis[0][:, None, None] + emis[1][None, :, None] + emis[2][None, None, :]
             + log_k(1)[:, :, None] + log_k(2)[None, :, :])      # layer 4 -> 6 is two steps
    joint = np.exp(joint - joint.max())
    joint /= joint.sum()
    brute = [joint.sum(axis=(1, 2)), joint.sum(axis=(0, 2)), joint.sum(axis=(0, 1))]
    for got, ref in zip(chain.marginals(), brute):
        np.testing.assert_allclose(got, ref, atol=1e-12)
    for i in range(3):
        np.testing.assert_allclose(chain.coupled(i).marginal_mu(), brute[i], atol=1e-12)
    # joint draws follow the joint: the (layer 3, layer 4) pair distribution
    jd = chain.joint_draws(20_000, seed=5)
    step = mu[1] - mu[0]
    i0 = np.rint((jd[0]["mu"] - mu[0]) / step).astype(int)
    i1 = np.rint((jd[1]["mu"] - mu[0]) / step).astype(int)
    emp = np.zeros((mu.size, mu.size))
    np.add.at(emp, (i0, i1), 1.0)
    assert 0.5 * np.abs(emp / emp.sum() - joint.sum(axis=2)).sum() < 0.05
    assert all(np.all(np.isfinite(d["s"])) and d["mu"].size == 20_000 for d in jd)


def chain_scenario(true_mu, centers, visits_at, seed, n_vis_default=6):
    rng = np.random.default_rng(seed)
    posts = []
    for layer, m in enumerate(true_mu):
        p = P.LayerPosterior(P.psychometric_prior(
            P.make_mu_grid(-4.0, 3.0, 141), center=centers[layer], narrow_weight=0.8,
            narrow_sd=0.15, wide_sd=1.0, floor_ab=(1, 49), lapse_ab=(1, 49), slope_median=0.07,
            slope_logsd=0.6, n_s=5, n_floor=3, n_lapse=3))
        for u in np.linspace(m - 0.2, m + 0.2, visits_at.get(layer, n_vis_default)):
            pr = float(P.detection_curve(u, m, 0.08, 0.02, 0.02))
            p.update(float(u), int(rng.binomial(8, pr)), 8)
        posts.append(p)
    return posts


def test_chain_keeps_a_far_one_layer_switch_given_enough_data():
    # the L39 collapse: one layer's threshold 3 log-dose units from its neighbours, and a screen
    # that missed it (every prior centred on the neighbours' value)
    true_mu = [-1.0] * 7
    true_mu[3] = 2.0
    posts = chain_scenario(true_mu, [-1.0] * 7, {}, seed=0)
    chain = P.LayerChain(posts, layers=list(range(7)), ar1=0.8, jump_df=3.0, sigma=1.0)
    means = [m for m, _ in chain.mean_sd()]
    assert means[3] == pytest.approx(2.0, abs=0.1)
    for i in (2, 4):
        assert means[i] == pytest.approx(-1.0, abs=0.1)     # nor are the neighbours dragged up


def test_heavy_tails_pull_a_switch_far_less_than_gaussian_steps():
    true_mu = [-1.0] * 7
    true_mu[3] = 2.0
    posts = chain_scenario(true_mu, true_mu, {3: 2}, seed=0)    # little data at the switch
    alone = posts[3].mean_sd_mu()[0]
    pull_t = alone - P.LayerChain(posts, layers=list(range(7)), ar1=0.8, jump_df=3.0, sigma=1.0).mean_sd()[3][0]
    pull_g = alone - P.LayerChain(posts, layers=list(range(7)), ar1=0.8, jump_df=1e6, sigma=1.0).mean_sd()[3][0]
    assert 0 <= pull_t < 0.03 and pull_g > 3 * pull_t


def test_chain_lends_strength_to_a_layer_with_little_data():
    true_mu = [-1.0] * 7
    posts = chain_scenario(true_mu, [-1.0] * 7, {3: 1}, seed=2)
    posts[3] = P.LayerPosterior(posts[3].prior)
    posts[3].update(-0.6, 8, 8)                               # one uninformative visit
    alone_mean, alone_sd = posts[3].mean_sd_mu()
    chain_mean, chain_sd = P.LayerChain(posts, layers=list(range(7)), ar1=0.8, jump_df=3.0,
                                        sigma=1.0).mean_sd()[3]
    assert chain_sd < 0.75 * alone_sd
    assert abs(chain_mean + 1.0) <= abs(alone_mean + 1.0) + 1e-9


# ------------------------------------------------------------------------------- guards

def test_grid_and_prior_guards():
    with pytest.raises(ValueError, match="lo < hi"):
        P.make_mu_grid(1.0, 0.0, 10)
    with pytest.raises(ValueError, match="at least 3"):
        P.make_mu_grid(0.0, 1.0, 2)
    mu = P.make_mu_grid(-2, 0, 11)
    with pytest.raises(ValueError, match="narrow_sd"):
        P.mixture_mu_logprior(mu, -1.0, narrow_weight=0.8, narrow_sd=0.0, wide_sd=1.0)
    with pytest.raises(ValueError, match="narrow_weight"):
        P.mixture_mu_logprior(mu, -1.0, narrow_weight=1.5, narrow_sd=0.1, wide_sd=1.0)
    with pytest.raises(ValueError, match="pair"):
        P.psychometric_prior(mu, center=-1, narrow_weight=0.8, narrow_sd=0.15, wide_sd=1.0,
                             floor_ab=(1, 49, 3), lapse_ab=(1, 49), slope_median=0.07, slope_logsd=0.6)
    with pytest.raises(ValueError, match="positive"):
        P.psychometric_prior(mu, center=-1, narrow_weight=0.8, narrow_sd=0.15, wide_sd=1.0,
                             floor_ab=(1, 49), lapse_ab=(0, 49), slope_median=0.07, slope_logsd=0.6)
    with pytest.raises(ValueError, match="slope"):
        P.psychometric_prior(mu, center=-1, narrow_weight=0.8, narrow_sd=0.15, wide_sd=1.0,
                             floor_ab=(1, 49), lapse_ab=(1, 49), slope_median=-0.07, slope_logsd=0.6)
    with pytest.raises(ValueError, match="control"):
        small_prior(control=(5, 3))
    with pytest.raises(ValueError, match="Beta parameters"):
        P.beta_quantile_grid(0, 1, 3)


def test_grid_prior_guards():
    good = small_prior()
    fields = {f: getattr(good, f) for f in good.__dataclass_fields__}
    bad_cases = [
        ({"mu": np.array([0.0, 0.1, 0.3]), "log_w_mu": np.zeros(3)}, "uniform"),
        ({"floor": np.array([0.6]), "log_w_floor": np.zeros(1), "lapse": np.array([0.5]),
          "log_w_lapse": np.zeros(1)}, "below 1"),
        ({"icc": np.array([1.0]), "log_w_icc": np.zeros(1)}, "icc"),
        ({"log_w_s": np.zeros(2)}, "equal-length"),
        ({"log_w_floor": np.full(good.floor.size, np.nan)}, "log weights"),
    ]
    for change, msg in bad_cases:
        with pytest.raises(ValueError, match=msg):
            P.GridPrior(**{**fields, **change})


def test_update_guards():
    post = P.LayerPosterior(small_prior())
    for args, msg in [((-1.0, 9, 8), "exceeds"), ((-1.0, 0, 0), "n=0"), ((-1.0, 1.5, 8), "whole"),
                      ((float("nan"), 1, 8), "finite"), ((-1.0, -1, 8), "negative")]:
        with pytest.raises(ValueError, match=msg):
            post.update(*args)
    with pytest.raises(ValueError, match="control"):
        post.update_control(5, 4)
    assert post.visits == [] and post.n_trials == 0      # nothing half-applied


def test_design_and_check_guards():
    post = P.LayerPosterior(small_prior())
    with pytest.raises(ValueError, match="n_trials"):
        post.expected_information_gain([-1.0], 0)
    with pytest.raises(ValueError, match="q"):
        post.threshold(1.0, 0.95)
    with pytest.raises(ValueError, match="conf"):
        post.threshold(0.1, 1.0)
    post.update(-1.0, 3, 8)
    with pytest.raises(ValueError, match="two or more distinct doses"):
        post.ppc_monotone(100, seed=0)
    post.update(-0.5, 7, 8)
    with pytest.raises(ValueError, match="n_rep"):
        post.ppc_monotone(5, seed=0)
    with pytest.raises(ValueError, match="message"):
        post.with_mu_message(np.zeros(3))


def test_chain_guards():
    posts = tiny_chain_layers()
    with pytest.raises(ValueError, match="strictly increasing"):
        P.LayerChain(posts, layers=[3, 3, 4], ar1=0.8, jump_df=3.0, sigma=1.0)
    with pytest.raises(ValueError, match="one posterior per layer"):
        P.LayerChain(posts, layers=[1, 2], ar1=0.8, jump_df=3.0, sigma=1.0)
    with pytest.raises(ValueError, match="ar1"):
        P.LayerChain(posts, layers=[1, 2, 3], ar1=1.0, jump_df=3.0, sigma=1.0)
    with pytest.raises(ValueError, match="jump_df"):
        P.LayerChain(posts, layers=[1, 2, 3], ar1=0.8, jump_df=0.0, sigma=1.0)
    with pytest.raises(ValueError, match="sigma"):
        P.LayerChain(posts, layers=[1, 2, 3], ar1=0.8, jump_df=3.0, sigma=-1.0)
    other = P.LayerPosterior(small_prior(mu=(-2.0, 2.0, 17)))
    with pytest.raises(ValueError, match="same mu grid"):
        P.LayerChain(posts[:2] + [other], layers=[1, 2, 3], ar1=0.8, jump_df=3.0, sigma=1.0)
