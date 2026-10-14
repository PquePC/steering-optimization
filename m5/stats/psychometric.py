"""m5.stats.psychometric - the per-layer dose-response posterior the adaptive search runs on.

Plan §6.1 and §5.2. Along log-dose u a detection-type channel (the gating channel: noticing `d8`
or naming `d2`) rises from a floor. Each layer is summarised by one curve,

    P(yes | u) = gamma + (1 - gamma - lambda) · F((u - mu) / s),     F the logistic CDF,

with gamma the floor (what the channel says with no steering, taken from the control trials),
lambda the lapse (the share of trials on which the model does not report even at a dose that
plainly reaches it), mu the location and s the scale, both in log-dose units. Prompt-to-prompt
variation enters as beta-binomial overdispersion with intra-class correlation `icc`, fixed or on a
small grid. This is standard psychometric-function practice (Wichmann & Hill 2001; psignifit,
Schutt et al. 2016).

**Why a grid posterior.** After every visit the search needs the whole posterior of the threshold:
to place the next dose by expected information gain, to stop a layer when its threshold is pinned
(posterior sd <= STOP_SD_LOGDOSE), and to halve layers on a credible bound. Four parameters and a
cheap likelihood make an exact posterior on a grid -- no sampler, no convergence diagnostics, the
same answer on every machine -- a matter of milliseconds. PyMC/NumPyro remain for the reported
fits (plan §6.1).

**Grids.** `mu` is shared by every layer, so `LayerChain` can couple layers exactly; `log s` is
uniform; the floor and the lapse sit at the midpoint quantiles of their Beta priors, each point
carrying equal prior mass, so the grid follows the prior wherever the control trials put it. Use a
`mu` spacing well below STOP_SD_LOGDOSE: every reported sd includes the grid's own resolution,
spacing/sqrt(12), so a coarse grid cannot fake a pinned threshold.

**The threshold.** u_D(q) is the log-dose where the curve exceeds its floor by q,

    u_D(q) = mu + s · F^{-1}(q / (1 - gamma - lambda)),

and is +inf where q >= 1 - gamma - lambda (the curve never rises that far). R*(q) is built on it
(`m5.stats.summaries`).

**Default priors** (pre-registered, plan §6.1): floor and lapse Beta(1, 49) (PRIOR_FLOOR_LAPSE);
the scale log-normal (PRIOR_SLOPE: median and log-sd); mu a mixture with weight
PRIOR_NARROW_WEIGHT on Normal(screen transition, PRIOR_NARROW_SD) and the rest on
Normal(screen transition, PRIOR_WIDE_SD), so a wrong screen costs samples, not coverage: in the
statistics review's simulation, priors that trusted a screen with a true error of sd 0.30 cut
interval coverage to 0.67-0.79.

**Missing data.** `update` takes valid counts only; a trial whose judgment failed is excluded by
the caller, never counted as a "no".
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from m5.stats._checks import as_float_array, check_conf, check_count, check_counts, check_prob, check_real
from m5.stats.beta import betaincinv, lgamma, log_binom_coef
from m5.stats.effect import expit, isotonic

_P_CLIP = 1e-12          # P(yes) is kept inside [1e-12, 1 - 1e-12]; only a zero floor reaches it
_PRUNE_NATS = 36.0       # cells more than e^-36 below the mode are skipped in summaries


# =====================================================================================
# The curve
# =====================================================================================

def detection_curve(u: np.ndarray | float, mu: np.ndarray | float, s: np.ndarray | float,
                    floor: np.ndarray | float, lapse: np.ndarray | float) -> np.ndarray:
    """P(yes | u) = floor + (1 - floor - lapse) · F((u - mu)/s), broadcasting over all arguments."""
    mu, s, floor, lapse = (np.asarray(v, dtype=float) for v in (mu, s, floor, lapse))
    return floor + (1.0 - floor - lapse) * expit((np.asarray(u, dtype=float) - mu) / s)


def detection_threshold(mu: np.ndarray | float, s: np.ndarray | float, floor: np.ndarray | float,
                        lapse: np.ndarray | float, q: float) -> np.ndarray:
    """u_D(q) = mu + s · logit(q / (1 - floor - lapse)); +inf where the curve never rises by q."""
    q = check_prob(q, "q", open_interval=True)
    mu, s, floor, lapse = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (mu, s, floor, lapse)))
    amp = 1.0 - floor - lapse
    r = np.where(amp > 0, q / np.where(amp > 0, amp, 1.0), np.inf)
    inside = r < 1.0
    rr = np.where(inside, r, 0.5)
    return np.where(inside, mu + s * np.log(rr / (1.0 - rr)), np.inf)


# =====================================================================================
# Priors
# =====================================================================================

def make_mu_grid(lo: float, hi: float, n: int) -> np.ndarray:
    """The shared grid over the threshold location mu (log-dose), n points from lo to hi.

    Shared by every layer of a vector so the cross-layer chain can couple them exactly. Choose it
    to cover every dose the search may visit with room on both sides, and fine enough that its
    resolution (spacing/sqrt(12)) is well below STOP_SD_LOGDOSE.
    """
    lo = check_real(lo, "lo")
    hi = check_real(hi, "hi")
    if not lo < hi:
        raise ValueError(f"the mu grid needs lo < hi, got lo={lo!r}, hi={hi!r}")
    n = check_count(n, "n", minimum=3)
    return np.linspace(lo, hi, n)


def _normalise_log(w: np.ndarray) -> np.ndarray:
    m = float(np.max(w))
    return w - (m + math.log(float(np.sum(np.exp(w - m)))))


def mixture_mu_logprior(mu_grid: np.ndarray, center: float, *, narrow_weight: float, narrow_sd: float,
                        wide_sd: float) -> np.ndarray:
    """log prior weights over the mu grid: w N(center, narrow_sd) + (1 - w) N(center, wide_sd).

    The narrow component trusts the screen's transition; the wide one makes sure a wrong screen
    can be overruled by data at the cost of samples. `narrow_weight` may be 0 (no screen) or 1.
    """
    mu = np.asarray(mu_grid, dtype=float)
    center = check_real(center, "center")
    w = check_prob(narrow_weight, "narrow_weight")
    for v, nm in ((narrow_sd, "narrow_sd"), (wide_sd, "wide_sd")):
        if check_real(v, nm) <= 0:
            raise ValueError(f"{nm} must be positive, got {v!r}")
    comps = []
    for weight, sd in ((w, narrow_sd), (1.0 - w, wide_sd)):
        if weight > 0:
            comps.append(math.log(weight) - math.log(sd) - 0.5 * ((mu - center) / sd) ** 2)
    stacked = np.stack(comps)
    m = stacked.max(axis=0)
    return _normalise_log(m + np.log(np.exp(stacked - m).sum(axis=0)))


def beta_quantile_grid(a: float, b: float, n: int) -> np.ndarray:
    """n points at the midpoint quantiles (i + 1/2)/n of Beta(a, b): each carries mass 1/n."""
    a = check_real(a, "a")
    b = check_real(b, "b")
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta parameters must be positive, got ({a!r}, {b!r})")
    n = check_count(n, "n", minimum=1)
    return np.array([betaincinv(a, b, (i + 0.5) / n) for i in range(n)])


@dataclass(frozen=True, eq=False)
class GridPrior:
    """The grid and the (log) prior weight of every axis; the joint prior is their product.

    Axes, in this order everywhere: mu (shared across layers), log s, floor, lapse, icc.
    """

    mu: np.ndarray
    log_w_mu: np.ndarray
    log_s: np.ndarray
    log_w_s: np.ndarray
    floor: np.ndarray
    log_w_floor: np.ndarray
    lapse: np.ndarray
    log_w_lapse: np.ndarray
    icc: np.ndarray
    log_w_icc: np.ndarray

    def __post_init__(self) -> None:
        pairs = (("mu", self.mu, self.log_w_mu), ("log_s", self.log_s, self.log_w_s),
                 ("floor", self.floor, self.log_w_floor), ("lapse", self.lapse, self.log_w_lapse),
                 ("icc", self.icc, self.log_w_icc))
        for name, grid, lw in pairs:
            if np.ndim(grid) != 1 or np.ndim(lw) != 1 or len(grid) != len(lw) or len(grid) == 0:
                raise ValueError(f"{name}: grid and weights must be equal-length 1-d arrays")
            if not np.all(np.isfinite(grid)):
                raise ValueError(f"{name}: grid values must be finite")
            if np.any(np.isnan(lw)) or np.all(np.isneginf(lw)):
                raise ValueError(f"{name}: prior weights must be valid log weights")
        if self.mu.size < 3 or not np.allclose(np.diff(self.mu), self.mu[1] - self.mu[0],
                                               rtol=1e-9, atol=1e-12) or self.mu[1] <= self.mu[0]:
            raise ValueError("mu must be a uniform increasing grid of at least 3 points")
        if self.log_s.size > 1 and not np.all(np.diff(self.log_s) > 0):
            raise ValueError("log_s must be increasing")
        if np.any(self.floor < 0) or np.any(self.lapse < 0):
            raise ValueError("floor and lapse values must be non-negative")
        if float(self.floor.max() + self.lapse.max()) >= 1.0:
            raise ValueError("floor + lapse must stay below 1 at every grid point")
        if np.any(self.icc < 0) or np.any(self.icc >= 1):
            raise ValueError("icc values must lie in [0, 1)")

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (self.mu.size, self.log_s.size, self.floor.size, self.lapse.size, self.icc.size)


def _ab_pair(ab: Sequence[float], name: str) -> tuple[float, float]:
    vals = tuple(ab)
    if len(vals) != 2:
        raise ValueError(f"{name} must be a pair (a, b), got {ab!r}")
    a, b = check_real(vals[0], name), check_real(vals[1], name)
    if a <= 0 or b <= 0:
        raise ValueError(f"{name} must be positive Beta parameters, got {ab!r}")
    return a, b


def psychometric_prior(mu_grid: np.ndarray, *, center: float, narrow_weight: float, narrow_sd: float,
                       wide_sd: float, floor_ab: Sequence[float], lapse_ab: Sequence[float],
                       slope_median: float, slope_logsd: float, icc: Sequence[float] = (0.0,),
                       n_s: int = 15, n_floor: int = 5, n_lapse: int = 5, s_span: float = 3.0,
                       control: tuple[int, int] | None = None) -> GridPrior:
    """Build the default prior of plan §6.1 on a grid.

    Callers pass config values: narrow_weight = PRIOR_NARROW_WEIGHT, narrow_sd = PRIOR_NARROW_SD,
    wide_sd = PRIOR_WIDE_SD, floor_ab = lapse_ab = PRIOR_FLOOR_LAPSE, (slope_median, slope_logsd)
    = PRIOR_SLOPE, center = the screen's transition (log-dose).

    `control=(k, n)` takes the floor from the control trials (plan §6.1): the floor's prior
    becomes Beta(a + k, b + n - k) before its grid is laid, so the grid sits where the control
    data put the floor. `icc` is the overdispersion grid (0 = binomial), equally weighted. The
    log-s grid spans `s_span` log-sds either side of the median. The grid sizes trade resolution
    for speed; the defaults put about 10^5 cells per layer with a 300-point mu grid.
    """
    mu = np.asarray(mu_grid, dtype=float)
    log_w_mu = mixture_mu_logprior(mu, center, narrow_weight=narrow_weight, narrow_sd=narrow_sd,
                                   wide_sd=wide_sd)
    slope_median = check_real(slope_median, "slope_median")
    slope_logsd = check_real(slope_logsd, "slope_logsd")
    if slope_median <= 0 or slope_logsd <= 0:
        raise ValueError("slope_median and slope_logsd must be positive")
    n_s = check_count(n_s, "n_s", minimum=1)
    if n_s == 1:
        log_s = np.array([math.log(slope_median)])
    else:
        span = check_real(s_span, "s_span") * slope_logsd
        log_s = np.linspace(math.log(slope_median) - span, math.log(slope_median) + span, n_s)
    log_w_s = _normalise_log(-0.5 * ((log_s - math.log(slope_median)) / slope_logsd) ** 2)
    fa, fb = _ab_pair(floor_ab, "floor_ab")
    if control is not None:
        kc, nc = check_counts(control[0], control[1], what="control trials", integer=True)
        fa, fb = fa + kc, fb + nc - kc
    floor = beta_quantile_grid(fa, fb, n_floor)
    la, lb = _ab_pair(lapse_ab, "lapse_ab")
    lapse = beta_quantile_grid(la, lb, n_lapse)
    icc_arr = np.asarray(icc, dtype=float).ravel()
    return GridPrior(mu=mu, log_w_mu=log_w_mu, log_s=log_s, log_w_s=log_w_s,
                     floor=floor, log_w_floor=np.full(floor.size, -math.log(floor.size)),
                     lapse=lapse, log_w_lapse=np.full(lapse.size, -math.log(lapse.size)),
                     icc=icc_arr, log_w_icc=np.full(icc_arr.size, -math.log(icc_arr.size)))


# =====================================================================================
# Likelihood pieces
# =====================================================================================

def _log_pmf(k: np.ndarray | int, n: int, p: np.ndarray, icc: float) -> np.ndarray:
    """log P(K = k | n, p) under the binomial (icc = 0) or the beta-binomial with mean p and
    intra-class correlation icc (alpha = p(1 - icc)/icc, beta = (1 - p)(1 - icc)/icc).

    `k` may be an array broadcasting against `p` (all outcomes at once, for the expected
    information gain). Includes log C(n, k), so the values are log-probabilities.
    """
    k = np.asarray(k, dtype=float)
    const = log_binom_coef(float(n), k)
    if icc == 0.0:
        return const + k * np.log(p) + (n - k) * np.log1p(-p)
    theta = (1.0 - icc) / icc
    a = p * theta
    b = (1.0 - p) * theta
    return (const + lgamma(k + a) + lgamma(n - k + b) - lgamma(a) - lgamma(b)
            + (lgamma(np.asarray(theta)) - lgamma(np.asarray(n + theta))))


def _log_pmf_all(n: int, p: np.ndarray, icc: float) -> np.ndarray:
    """log P(K = k | n, p) for every k = 0..n at once, shape (n + 1,) + p.shape.

    The beta-binomial by its ratio recurrence P(k+1)/P(k) = (n - k)(a + k) / ((k + 1)(b + n - k - 1)),
    from log P(0) = sum_j [log(b + j) - log(a + b + j)]: two logs per outcome instead of four
    log-Gamma evaluations on the whole stacked grid, which is what makes the expected information
    gain affordable with an overdispersion grid. Equal to `_log_pmf` (tested).
    """
    if icc == 0.0:
        k = np.arange(n + 1, dtype=float).reshape((-1,) + (1,) * p.ndim)
        return log_binom_coef(float(n), k) + k * np.log(p)[None] + (n - k) * np.log1p(-p)[None]
    theta = (1.0 - icc) / icc
    a = p * theta
    b = (1.0 - p) * theta
    out = np.empty((n + 1,) + p.shape)
    acc = np.zeros_like(p)
    for j in range(n):
        acc += np.log(b + j)
    acc -= sum(math.log(theta + j) for j in range(n))
    out[0] = acc
    for k in range(n):
        acc = acc + (math.log(n - k) - math.log(k + 1)) + np.log(a + k) - np.log(b + n - k - 1)
        out[k + 1] = acc
    return out


def _monotone_deviance(k: np.ndarray, n: np.ndarray) -> float:
    """2 x log-likelihood ratio of the per-dose rates against the best non-decreasing rates.

    Zero when the observed rates already rise with dose; large when they fall (an inverted U,
    a sign flip). The isotonic fit with weights n is the monotone binomial MLE.
    """
    p_hat = k / n
    p_iso = isotonic(p_hat, n)
    dev = 0.0
    for ki, ni, ph, pi in zip(k.tolist(), n.tolist(), p_hat.tolist(), p_iso.tolist()):
        if ki > 0:
            dev += ki * math.log(ph / pi)
        if ni - ki > 0:
            dev += (ni - ki) * math.log((1.0 - ph) / (1.0 - pi))
    return 2.0 * max(dev, 0.0)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    idx = np.searchsorted(cw, np.asarray(probs, dtype=float), side="left")
    return v[np.clip(idx, 0, v.size - 1)]


@dataclass(frozen=True)
class ThresholdSummary:
    """The posterior of u_D(q) for one layer.

    `mean` and `sd` are of the finite part (the curve rises by q); `p_never` is the posterior
    probability that it never does. `sd` includes the mu grid's own resolution (spacing/sqrt(12)),
    so it cannot fall below what the grid can resolve. `lo`/`hi` are equal-tailed credible bounds
    of the whole distribution, so `hi` is +inf when p_never > (1 - conf)/2.
    """

    q: float
    mean: float
    sd: float
    median: float
    lo: float
    hi: float
    p_never: float
    conf: float

    def to_json(self) -> dict[str, Any]:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in asdict(self).items()}


@dataclass(frozen=True)
class Visit:
    """k yes out of n gating trials at log-dose u."""

    u: float
    k: int
    n: int


# =====================================================================================
# One layer
# =====================================================================================

class LayerPosterior:
    """The exact grid posterior of (mu, log s, floor, lapse, icc) for one layer's gating channel.

    Build with a `GridPrior` (`psychometric_prior`), feed visits with `update(u, k, n)` and the
    floor's control trials with `update_control(k, n)`, and read:

        log_posterior(), posterior()       the normalised joint posterior on the grid
        log_likelihood                     the accumulated log-likelihood surface
        marginal_mu(), mean_sd_mu()        the location's marginal and its mean/sd
        threshold(q, conf)                 the posterior of u_D(q)
        curve(u_grid, conf)                P(yes | u) with credible bands
        expected_information_gain(...)     where the next visit teaches most
        draws(n, seed)                     posterior draws for the summaries
        ppc_monotone(n_rep, seed)          posterior predictive check for a non-monotone curve
        log_evidence()                     the marginal likelihood (to compare icc grids, say)
    """

    def __init__(self, prior: GridPrior):
        self.prior = prior
        self._s = np.exp(prior.log_s)
        self._amp = 1.0 - prior.floor[:, None] - prior.lapse[None, :]
        self._dmu = float(prior.mu[1] - prior.mu[0])
        self._dls = float(prior.log_s[1] - prior.log_s[0]) if prior.log_s.size > 1 else 0.0
        self._log_prior = (prior.log_w_mu[:, None, None, None, None]
                           + prior.log_w_s[None, :, None, None, None]
                           + prior.log_w_floor[None, None, :, None, None]
                           + prior.log_w_lapse[None, None, None, :, None]
                           + prior.log_w_icc[None, None, None, None, :])
        self._log_msg = np.zeros(prior.mu.size)
        self._log_lik = np.zeros(prior.shape)
        self.visits: list[Visit] = []
        self.controls: list[tuple[int, int]] = []
        self._cache: np.ndarray | None = None

    # ---------------------------------------------------------------- data

    def p_yes(self, u: float) -> np.ndarray:
        """P(yes | u) at every (mu, s, floor, lapse) grid point, shape (M, S, G, L)."""
        f = expit((float(u) - self.prior.mu[:, None]) / self._s[None, :])
        p = self.prior.floor[None, None, :, None] + self._amp[None, None, :, :] * f[:, :, None, None]
        return np.clip(p, _P_CLIP, 1.0 - _P_CLIP)

    def _log_pmf_grid(self, k: int | np.ndarray, n: int, p: np.ndarray) -> np.ndarray:
        """log P(k | n) on the full grid (icc as the last axis); `p` has shape (..., M, S, G, L)."""
        return np.stack([_log_pmf(k, n, p, float(c)) for c in self.prior.icc], axis=-1)

    def update(self, u: float, k: int, n: int) -> None:
        """Add one visit: k yes out of n valid gating trials at log-dose u."""
        u = check_real(u, "u")
        kf, nf = check_counts(k, n, what="visit", integer=True)
        k, n = int(kf), int(nf)
        self._log_lik += self._log_pmf_grid(k, n, self.p_yes(u))
        self.visits.append(Visit(u, k, n))
        self._cache = None

    def update_control(self, k: int, n: int) -> None:
        """Add unsteered control trials: k yes out of n, evidence about the floor alone."""
        kf, nf = check_counts(k, n, what="control", integer=True)
        g = self.prior.floor
        with np.errstate(divide="ignore"):
            ll = kf * np.log(np.maximum(g, _P_CLIP)) + (nf - kf) * np.log1p(-g)
        self._log_lik += ll[None, None, :, None, None]
        self.controls.append((int(kf), int(nf)))
        self._cache = None

    @property
    def n_trials(self) -> int:
        """Gating trials spent at this layer (the STOP_TRIALS counter); control trials excluded."""
        return sum(v.n for v in self.visits)

    @property
    def log_likelihood(self) -> np.ndarray:
        """The accumulated log-likelihood surface over the grid (read-only view)."""
        view = self._log_lik.view()
        view.flags.writeable = False
        return view

    # ---------------------------------------------------------------- the posterior

    def _unnormalised(self) -> np.ndarray:
        return self._log_prior + self._log_msg[:, None, None, None, None] + self._log_lik

    def log_posterior(self) -> np.ndarray:
        """The normalised log posterior, shape (M, S, G, L, C)."""
        if self._cache is None:
            lp = self._unnormalised()
            self._cache = lp - _logsumexp(lp)
        return self._cache

    def posterior(self) -> np.ndarray:
        return np.exp(self.log_posterior())

    def log_evidence(self) -> float:
        """log of the marginal likelihood of the data under the prior (and any chain message)."""
        return _logsumexp(self._unnormalised()) - _logsumexp(self._log_prior + self._log_msg[:, None, None, None, None])

    def log_marginal_mu_unnormalised(self) -> np.ndarray:
        """log sum over (s, floor, lapse, icc) of prior x likelihood, as a function of mu.

        This is the layer's emission in `LayerChain`: its own prior times its own data, with every
        parameter but mu integrated out. Excludes any chain message already applied.
        """
        lp = self._log_prior + self._log_lik
        flat = lp.reshape(lp.shape[0], -1)
        m = flat.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(flat - m).sum(axis=1, keepdims=True)))[:, 0]

    def marginal_mu(self) -> np.ndarray:
        """Posterior probabilities over the mu grid."""
        return self.posterior().sum(axis=(1, 2, 3, 4))

    def mean_sd_mu(self) -> tuple[float, float]:
        """Posterior mean and sd of mu; the sd includes the grid resolution spacing/sqrt(12)."""
        w = self.marginal_mu()
        mu = self.prior.mu
        mean = float(w @ mu)
        var = float(w @ (mu - mean) ** 2) + self._dmu ** 2 / 12.0
        return mean, math.sqrt(var)

    def _threshold_grid(self, q: float) -> np.ndarray:
        return detection_threshold(self.prior.mu[:, None, None, None], self._s[None, :, None, None],
                                   self.prior.floor[None, None, :, None],
                                   self.prior.lapse[None, None, None, :], q)

    def threshold(self, q: float, conf: float) -> ThresholdSummary:
        """The posterior of u_D(q), the log-dose where the curve exceeds its floor by q."""
        conf = check_conf(conf)
        lp = self.log_posterior()
        w = np.exp(lp).sum(axis=4)
        ud = self._threshold_grid(q)
        keep = np.log(np.maximum(w, 1e-320)) > float(np.log(w.max())) - _PRUNE_NATS
        vals, wts = ud[keep], w[keep]
        wts = wts / wts.sum()
        finite = np.isfinite(vals)
        p_never = float(wts[~finite].sum())
        alpha = 1.0 - conf
        lo, med, hi = _weighted_quantiles(vals, wts, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0])
        if np.any(finite) and wts[finite].sum() > 0:
            wf = wts[finite] / wts[finite].sum()
            mean = float(wf @ vals[finite])
            sd = math.sqrt(float(wf @ (vals[finite] - mean) ** 2) + self._dmu ** 2 / 12.0)
        else:
            mean, sd = math.inf, math.inf
        return ThresholdSummary(q=float(q), mean=mean, sd=sd, median=float(med), lo=float(lo),
                                hi=float(hi), p_never=p_never, conf=conf)

    def curve(self, u_grid: Sequence[float] | np.ndarray, conf: float) -> dict[str, np.ndarray]:
        """Posterior mean of P(yes | u) and its equal-tailed credible band at each u."""
        conf = check_conf(conf)
        ug = as_float_array(u_grid, "u_grid")
        w = self.posterior().sum(axis=4)
        keep = w > w.max() * math.exp(-_PRUNE_NATS)
        wk = w[keep] / w[keep].sum()
        alpha = 1.0 - conf
        mean = np.empty(ug.size)
        lo = np.empty(ug.size)
        hi = np.empty(ug.size)
        for i, u in enumerate(ug.tolist()):
            p = self.p_yes(u)[keep]
            mean[i] = float(wk @ p)
            lo[i], hi[i] = _weighted_quantiles(p, wk, [alpha / 2.0, 1.0 - alpha / 2.0])
        return {"u": ug, "mean": mean, "lo": lo, "hi": hi}

    # ---------------------------------------------------------------- design

    def expected_information_gain(self, candidate_us: Sequence[float] | np.ndarray, n_trials: int, *,
                                  q: float | None = None) -> np.ndarray:
        """Expected reduction in entropy (nats) of the target from n_trials trials at each candidate.

        The target is mu (q None) or the threshold u_D(q) binned at the mu grid's spacing. For a
        candidate u, every outcome k = 0..n is weighted by its posterior predictive probability
        (binomial or beta-binomial, averaged over the posterior), and the gain is the mutual
        information between k and the target -- the classic Bayesian adaptive design criterion
        (Lindley 1956; Kontsevich & Tyler 1999 for psychometric thresholds). Deterministic.
        """
        cands = as_float_array(candidate_us, "candidate_us")
        n_trials = check_count(n_trials, "n_trials", minimum=1)
        lp = self.log_posterior()
        if q is None:
            marg = np.exp(lp).sum(axis=(1, 2, 3, 4))
            rows = np.flatnonzero(marg > marg.max() * math.exp(-_PRUNE_NATS))
            w = np.exp(lp[rows])
            out = np.empty(cands.size)
            for i, u in enumerate(cands.tolist()):
                p = self.p_yes(u)[rows]
                lik = np.stack([np.exp(_log_pmf_all(n_trials, p, float(c))) for c in self.prior.icc],
                               axis=-1)
                joint = (lik * w[None]).reshape(n_trials + 1, rows.size, -1).sum(axis=2)
                out[i] = _mutual_information(joint)
            return out
        ud = np.broadcast_to(self._threshold_grid(q)[..., None], lp.shape)
        keep = lp > lp.max() - _PRUNE_NATS
        idx = np.nonzero(keep)
        w = np.exp(lp[keep])
        mu = self.prior.mu
        width = mu[-1] - mu[0]
        edges_lo = mu[0] - width
        n_bins = int(math.ceil(3.0 * width / self._dmu)) + 2
        vals = ud[keep]
        bins = np.where(np.isfinite(vals), np.floor((np.where(np.isfinite(vals), vals, 0.0) - edges_lo) / self._dmu), n_bins - 1)
        bins = np.clip(bins, 0, n_bins - 1).astype(np.int64)
        icc_vals = self.prior.icc[idx[4]]
        out = np.empty(cands.size)
        for i, u in enumerate(cands.tolist()):
            p = self.p_yes(u)[idx[0], idx[1], idx[2], idx[3]]
            lik = np.empty((n_trials + 1, p.size))
            for c in np.unique(icc_vals):
                sel = icc_vals == c
                lik[:, sel] = np.exp(_log_pmf_all(n_trials, p[sel], float(c)))
            joint = np.stack([np.bincount(bins, weights=w * lik[k], minlength=n_bins)
                              for k in range(n_trials + 1)])
            out[i] = _mutual_information(joint)
        return out

    def draws(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """n posterior draws of (mu, s, floor, lapse, icc).

        Grid cells are drawn by posterior mass; mu and log s are then spread uniformly within
        their cell, so draws are continuous and summaries built from them (R*, P_win) are not
        lumped at grid points. Floor and lapse keep their quantile-grid values.
        """
        n = check_count(n, "n", minimum=1)
        return self._draws(n, np.random.default_rng(seed))

    def _draws(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        w = self.posterior().ravel()
        flat = rng.choice(w.size, size=n, p=w / w.sum())
        im, is_, ig, il, ic = np.unravel_index(flat, self.prior.shape)
        mu = self.prior.mu[im] + rng.uniform(-0.5, 0.5, n) * self._dmu
        log_s = self.prior.log_s[is_] + rng.uniform(-0.5, 0.5, n) * self._dls
        return {"mu": mu, "s": np.exp(log_s), "floor": self.prior.floor[ig],
                "lapse": self.prior.lapse[il], "icc": self.prior.icc[ic]}

    def ppc_monotone(self, n_rep: int, seed: int) -> dict[str, Any]:
        """Posterior predictive check (Gelman, Meng & Stern 1996) for a non-monotone dose-response.

        Discrepancy: the deviance of the observed per-dose rates from the best non-decreasing
        rates (zero when the data already rise with dose). Replicate data sets are drawn from the
        posterior predictive at the visited doses with the visited n; the p-value is the share of
        replicates at least as non-monotone as the data. A small p-value says the monotone curve
        cannot produce what was seen -- the system cards' sign-dependent, non-monotone
        evaluation-awareness curves -- and the layer should switch to a nonparametric level-set
        estimator (plan §6.1).
        """
        n_rep = check_count(n_rep, "n_rep", minimum=10)
        if len({v.u for v in self.visits}) < 2:
            raise ValueError("a monotonicity check needs visits at two or more distinct doses")
        doses = np.array(sorted({v.u for v in self.visits}))
        k = np.array([sum(v.k for v in self.visits if v.u == d) for d in doses], dtype=float)
        n = np.array([sum(v.n for v in self.visits if v.u == d) for d in doses], dtype=float)
        t_obs = _monotone_deviance(k, n)
        rng = np.random.default_rng(seed)
        d = self._draws(n_rep, rng)
        p = detection_curve(doses[None, :], d["mu"][:, None], d["s"][:, None], d["floor"][:, None],
                            d["lapse"][:, None])
        p = np.clip(p, _P_CLIP, 1.0 - _P_CLIP)
        icc = d["icc"][:, None]
        theta = np.where(icc > 0, (1.0 - icc) / np.where(icc > 0, icc, 1.0), np.inf)
        rates = np.where(icc > 0, rng.beta(np.where(icc > 0, p * theta, 1.0),
                                           np.where(icc > 0, (1.0 - p) * theta, 1.0)), p)
        k_rep = rng.binomial(n.astype(np.int64)[None, :], rates)
        t_rep = np.array([_monotone_deviance(k_rep[r].astype(float), n) for r in range(n_rep)])
        return {"p_value": float(np.mean(t_rep >= t_obs - 1e-9)), "t_obs": t_obs,
                "t_rep_mean": float(t_rep.mean()), "n_rep": n_rep, "doses": doses.tolist()}

    # ---------------------------------------------------------------- copies

    def copy(self) -> "LayerPosterior":
        out = copy.copy(self)
        out._log_lik = self._log_lik.copy()
        out._log_msg = self._log_msg.copy()
        out.visits = list(self.visits)
        out.controls = list(self.controls)
        out._cache = None
        return out

    def with_mu_message(self, log_msg: np.ndarray) -> "LayerPosterior":
        """A copy whose posterior is multiplied by exp(log_msg) along mu (a chain message).

        A snapshot: data added to the copy later are not seen by the chain that produced the
        message, so rebuild the chain after new visits.
        """
        msg = np.asarray(log_msg, dtype=float)
        if msg.shape != self.prior.mu.shape:
            raise ValueError("the message must be one log weight per mu grid point")
        out = self.copy()
        out._log_msg = msg - float(np.max(msg))
        return out


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    if not math.isfinite(m):
        raise ValueError("every grid cell has zero posterior mass")
    return m + math.log(float(np.sum(np.exp(x - m))))


def _mutual_information(joint: np.ndarray) -> float:
    """I(K; target) from the joint mass table joint[k, bin] (it need not be normalised)."""
    total = joint.sum()
    if total <= 0:
        return 0.0
    j = joint / total
    pk = j.sum(axis=1, keepdims=True)
    pb = j.sum(axis=0, keepdims=True)
    mask = j > 0
    denom = (pk * pb)[mask]
    return float(max(0.0, np.sum(j[mask] * np.log(j[mask] / denom))))


# =====================================================================================
# Across layers
# =====================================================================================

@dataclass(eq=False)
class LayerChain:
    """Couple neighbouring layers' threshold locations, exactly, on the shared mu grid.

    Model: the joint posterior of (mu_1, ..., mu_L) is proportional to
        prod_l e_l(mu_l) · prod_l psi(mu_{l+1} - mu_l),
    where e_l is layer l's own prior times its own likelihood with every other parameter
    integrated out (`log_marginal_mu_unnormalised`), and psi is a Student-t random-walk potential
    with `jump_df` degrees of freedom and scale

        tau_g = sigma · sqrt(1 - ar1^(2g))

    for layers g apart: the innovation sd of a stationary AR(1) with marginal sd `sigma` and
    lag-1 correlation `ar1` (LAYER_AR1), taken over g steps. Callers pass LAYER_AR1, LAYER_JUMP_DF
    and, for sigma, the spread of thresholds across layers (PRIOR_WIDE_SD by default).

    Heavy tails are the point: a Gaussian step penalises a jump quadratically, so a real one-layer
    switch -- the L39 collapse, seen in all three M3 concepts -- would be pulled back towards its
    neighbours; the t potential's penalty grows only logarithmically, so enough data at that layer
    keep it where it is, while a layer with little data still borrows strength from its neighbours.

    Inference is forward-backward (sum-product on the chain) in log space on the grid: exact
    marginals, and exact joint draws by forward-filtering backward-sampling (`joint_draws`).
    """

    posteriors: Sequence[LayerPosterior]
    layers: Sequence[int]
    ar1: float
    jump_df: float
    sigma: float
    _log_alpha: list[np.ndarray] = field(init=False, repr=False)
    _log_beta: list[np.ndarray] = field(init=False, repr=False)
    _log_pred: list[np.ndarray] = field(init=False, repr=False)
    _log_psi: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.posteriors) == 0 or len(self.posteriors) != len(self.layers):
            raise ValueError("one posterior per layer, and at least one")
        mu = self.posteriors[0].prior.mu
        for p in self.posteriors[1:]:
            if p.prior.mu.shape != mu.shape or not np.array_equal(p.prior.mu, mu):
                raise ValueError("every layer must share the same mu grid")
        layers = [int(v) for v in self.layers]
        if any(b <= a for a, b in zip(layers, layers[1:])):
            raise ValueError("layers must be strictly increasing")
        self.layers = layers
        self.ar1 = check_prob(self.ar1, "ar1", open_interval=True)
        if check_real(self.jump_df, "jump_df") <= 0:
            raise ValueError("jump_df must be positive")
        if check_real(self.sigma, "sigma") <= 0:
            raise ValueError("sigma must be positive")
        self._run()

    def _log_kernel(self, gap: int) -> np.ndarray:
        mu = self.posteriors[0].prior.mu
        tau = self.sigma * math.sqrt(1.0 - self.ar1 ** (2 * gap))
        d = (mu[None, :] - mu[:, None]) / tau
        return -0.5 * (self.jump_df + 1.0) * np.log1p(d * d / self.jump_df)

    def _run(self) -> None:
        emis = [p.log_marginal_mu_unnormalised() for p in self.posteriors]
        n = len(emis)
        self._log_psi = [self._log_kernel(self.layers[i + 1] - self.layers[i]) for i in range(n - 1)]
        log_alpha = []
        log_pred = [np.zeros_like(emis[0])]
        a = emis[0] - _logsumexp(emis[0])
        log_alpha.append(a)
        for i in range(1, n):
            pred = _log_matvec(a, self._log_psi[i - 1])          # sum_mu alpha(mu) psi(mu, mu')
            pred = pred - _logsumexp(pred)
            log_pred.append(pred)
            a = emis[i] + pred
            a = a - _logsumexp(a)
            log_alpha.append(a)
        log_beta = [np.zeros_like(emis[0]) for _ in range(n)]
        for i in range(n - 2, -1, -1):
            b = _log_matvec(emis[i + 1] + log_beta[i + 1], self._log_psi[i].T)   # sum_mu' psi e beta
            log_beta[i] = b - _logsumexp(b)
        self._log_alpha, self._log_beta, self._log_pred = log_alpha, log_beta, log_pred

    def marginals(self) -> list[np.ndarray]:
        """Each layer's chain-coupled posterior over the mu grid."""
        out = []
        for a, b in zip(self._log_alpha, self._log_beta):
            lp = a + b
            out.append(np.exp(lp - _logsumexp(lp)))
        return out

    def messages(self) -> list[np.ndarray]:
        """The log message each layer receives from the rest of the chain (left x right), over mu."""
        out = []
        for pred, b in zip(self._log_pred, self._log_beta):
            m = pred + b
            out.append(m - float(np.max(m)))
        return out

    def mean_sd(self) -> list[tuple[float, float]]:
        """Chain-coupled posterior mean and sd of mu per layer (sd includes the grid resolution)."""
        mu = self.posteriors[0].prior.mu
        dmu = float(mu[1] - mu[0])
        out = []
        for w in self.marginals():
            mean = float(w @ mu)
            out.append((mean, math.sqrt(float(w @ (mu - mean) ** 2) + dmu * dmu / 12.0)))
        return out

    def coupled(self, i: int) -> LayerPosterior:
        """Layer i's full posterior with the chain's message applied: thresholds, curves and draws
        from it are the chain-coupled ones."""
        return self.posteriors[i].with_mu_message(self.messages()[i])

    def joint_draws(self, n: int, seed: int) -> list[dict[str, np.ndarray]]:
        """n joint draws across layers, aligned by index (draw j of every layer is one joint draw).

        Forward-filtering backward-sampling (Carter & Kohn 1994; Fruhwirth-Schnatter 1994): mu of
        the last layer from its filtered posterior, then each earlier layer's mu given the next
        one's, then every other parameter from the layer's own conditional posterior given its mu.
        Aligned draws are what P_win and R* need, because the best layer and the "some cell"
        of the box are statements about all layers at once.
        """
        n = check_count(n, "n", minimum=1)
        rng = np.random.default_rng(seed)
        L = len(self.posteriors)
        mu = self.posteriors[0].prior.mu
        idx = np.empty((L, n), dtype=np.int64)
        idx[L - 1] = _sample_log(self._log_alpha[L - 1], n, rng)
        for i in range(L - 2, -1, -1):
            # p(mu_i | mu_{i+1}) ∝ alpha_i(mu_i) psi(mu_i, mu_{i+1})
            cond = self._log_alpha[i][:, None] + self._log_psi[i][:, idx[i + 1]]   # (M, n)
            cond = cond - cond.max(axis=0, keepdims=True)
            pc = np.exp(cond)
            pc /= pc.sum(axis=0, keepdims=True)
            cum = np.cumsum(pc, axis=0)
            r = rng.random(n)
            idx[i] = np.minimum((cum < r[None, :]).sum(axis=0), mu.size - 1)
        out = []
        for i, post in enumerate(self.posteriors):
            lp = post._log_prior + post._log_lik                    # the layer's own posterior
            dmu = post._dmu
            draws = {"mu": np.empty(n), "s": np.empty(n), "floor": np.empty(n),
                     "lapse": np.empty(n), "icc": np.empty(n)}
            for m_idx in np.unique(idx[i]):
                sel = np.flatnonzero(idx[i] == m_idx)
                slab = lp[m_idx].ravel()
                p = np.exp(slab - slab.max())
                p /= p.sum()
                pick = rng.choice(slab.size, size=sel.size, p=p)
                is_, ig, il, ic = np.unravel_index(pick, lp.shape[1:])
                draws["mu"][sel] = mu[m_idx] + rng.uniform(-0.5, 0.5, sel.size) * dmu
                draws["s"][sel] = np.exp(post.prior.log_s[is_] + rng.uniform(-0.5, 0.5, sel.size) * post._dls)
                draws["floor"][sel] = post.prior.floor[ig]
                draws["lapse"][sel] = post.prior.lapse[il]
                draws["icc"][sel] = post.prior.icc[ic]
            out.append(draws)
        return out


def _log_matvec(log_v: np.ndarray, log_k: np.ndarray) -> np.ndarray:
    """log sum_i exp(log_v[i] + log_k[i, j]) for every j, without underflow."""
    t = log_v[:, None] + log_k
    m = t.max(axis=0)
    return m + np.log(np.exp(t - m[None, :]).sum(axis=0))


def _sample_log(log_w: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    p = np.exp(log_w - log_w.max())
    return rng.choice(p.size, size=n, p=p / p.sum())
