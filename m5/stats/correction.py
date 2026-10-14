"""m5.stats.correction - judged rates corrected for the judge's own errors, with that uncertainty kept.

Plan §6.4 and §7.4-7.5. A judged rate is the judge's rate, not the model's: a judge with a 3%
false-positive rate reads 3% "detection" off an unsteered model. Every judged rate therefore
carries the judge's measured sensitivity (TPR) and specificity (TNR) in its dose band, and is
corrected in one of two ways:

1. **Rogan-Gladen (1978)**: theta = (p_obs + TNR - 1) / (TPR + TNR - 1). Needs only the judge's
   error rates from the gold labels; the interval propagates the uncertainty of all three
   estimated proportions, by the delta method (default) or a parametric (Jeffreys) bootstrap.
2. **Prediction-powered inference, PPI++** (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic 2023,
   Science; Angelopoulos, Duchi & Zrnic 2023, arXiv 2311.01453): uses the human labels on a
   subset and the judge on everything, with the judge's weight lambda tuned so the result is never
   worse than the human labels alone.

`label_budget` answers the planning question of §6.4: how many human labels give a target
half-width, with and without a judge correlated rho with the humans (at rho = 0.5, 0.7, 0.9 about
72, 49, 18 labels give +-0.10, against 96 without the judge).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from m5.stats._checks import as_float_array, check_conf, check_count, check_counts, check_prob, check_real
from m5.stats.normal import z_two_sided


def _youden(tpr: float, tnr: float) -> float:
    j = tpr + tnr - 1.0
    if j <= 0.0:
        raise ValueError(f"TPR + TNR = {tpr + tnr:.4f} <= 1: the judge is no better than chance on "
                         "this field and band, so its rate cannot be corrected (it carries no "
                         "information about the true rate)")
    return j


def rogan_gladen(p_obs: float, tpr: float, tnr: float) -> float:
    """The Rogan-Gladen (1978) corrected prevalence, clipped to [0, 1].

    theta = (p_obs + TNR - 1) / (TPR + TNR - 1). The raw value leaves [0, 1] whenever p_obs falls
    below the judge's false-positive rate (1 - TNR) or above its TPR, which sampling noise does
    routinely near the floor; clipping reports the nearest possible rate instead.
    """
    p_obs = check_prob(p_obs, "p_obs")
    tpr = check_prob(tpr, "tpr")
    tnr = check_prob(tnr, "tnr")
    j = _youden(tpr, tnr)
    return min(1.0, max(0.0, (p_obs + tnr - 1.0) / j))


@dataclass(frozen=True)
class CorrectedRate:
    """A judge-corrected rate with its interval.

    `raw` is the uncorrected judged rate, `unclipped` the Rogan-Gladen value before clipping (it
    shows how far outside [0, 1] sampling put it), and `method` how the interval was built.
    """

    estimate: float
    lo: float
    hi: float
    se: float | None
    raw: float
    unclipped: float
    tpr: float
    tnr: float
    conf: float
    method: str
    n_obs: float
    n_pos: float
    n_neg: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def rogan_gladen_interval(k_obs: float, n_obs: float, k_tp: float, n_pos: float, k_tn: float,
                          n_neg: float, conf: float, *, method: str,
                          reps: int | None = None, seed: int | None = None) -> CorrectedRate:
    """Rogan-Gladen with an interval that carries the judge's error-rate uncertainty.

    k_obs/n_obs: judged positives among the cell's trials. k_tp/n_pos: judge positives among
    human-positive gold items (TPR). k_tn/n_neg: judge negatives among human-negative gold items
    (TNR). The three are independent samples.

    `method` is required, because the two answer differently where it matters most. Use the
    bootstrap for rates near the floor -- where covertness claims live -- and the delta method for
    interior rates.

    `method="delta"`: the delta-method variance of the Rogan-Gladen estimator, with J = TPR + TNR - 1,
        Var(theta) = [Var(p_obs) + theta^2 Var(TPR) + (1 - theta)^2 Var(TNR)] / J^2,
    each variance from the Agresti-Coull adjusted proportion (k + z^2/2) / (n + z^2) (cf. Lang &
    Reiczigel 2014, Preventive Veterinary Medicine 113:13-22, who adjust all three), so a judge
    that was perfect on its gold items (TPR = 20/20) still carries the uncertainty of having been
    tested on twenty items; the plug-in variance would be exactly zero there. The interval is
    symmetric around the unclipped estimate, then clipped to [0, 1]. At TPR = TNR = 0.9 and
    theta = 0.5 this gives about +-5.6 points with 100 + 100 gold labels and +-4.0 with 200 + 200
    (plan §7.4 quotes +-5 and +-3.7 from the plug-in variances; the adjustment adds about 7%). It
    does not know theta >= 0: at 0/150 with a judge perfect on 20 + 20 gold items its upper end is
    about 0.11, against about 0.01 from the bootstrap.

    `method="bootstrap"`: draws each proportion from its Jeffreys posterior Beta(k + 1/2,
    n - k + 1/2), maps the draws through the clipped correction and takes percentiles. Better near
    theta = 0 or 1 and when J is small, where the delta method's symmetry is wrong. A draw with
    J <= 0 (a judge no better than chance) carries no information about theta; if more than
    (1 - conf)/2 of the draws are like that, the interval is the whole of [0, 1].
    """
    conf = check_conf(conf)
    ko, no = check_counts(k_obs, n_obs, what="judged trials")
    kp, npos = check_counts(k_tp, n_pos, what="human-positive gold items")
    kn, nneg = check_counts(k_tn, n_neg, what="human-negative gold items")
    p_obs, tpr, tnr = ko / no, kp / npos, kn / nneg
    j = _youden(tpr, tnr)
    unclipped = (p_obs + tnr - 1.0) / j
    estimate = min(1.0, max(0.0, unclipped))
    z = z_two_sided(conf)
    if method == "delta":
        def ac_var(k: float, n: float) -> float:
            nt = n + z * z
            pt = (k + 0.5 * z * z) / nt
            return pt * (1.0 - pt) / nt
        var = (ac_var(ko, no) + unclipped ** 2 * ac_var(kp, npos)
               + (1.0 - unclipped) ** 2 * ac_var(kn, nneg)) / (j * j)
        se = math.sqrt(var)
        lo = min(1.0, max(0.0, unclipped - z * se))
        hi = min(1.0, max(0.0, unclipped + z * se))
        return CorrectedRate(estimate, lo, hi, se, p_obs, unclipped, tpr, tnr, conf, "delta",
                             no, npos, nneg)
    if method == "bootstrap":
        if reps is None or seed is None:
            raise ValueError("method='bootstrap' needs reps and seed")
        reps = check_count(reps, "reps", minimum=100)
        rng = np.random.default_rng(seed)
        p_d = rng.beta(ko + 0.5, no - ko + 0.5, size=reps)
        tpr_d = rng.beta(kp + 0.5, npos - kp + 0.5, size=reps)
        tnr_d = rng.beta(kn + 0.5, nneg - kn + 0.5, size=reps)
        j_d = tpr_d + tnr_d - 1.0
        informative = j_d > 0.0
        alpha = 1.0 - conf
        if np.mean(~informative) > alpha / 2.0:
            return CorrectedRate(estimate, 0.0, 1.0, None, p_obs, unclipped, tpr, tnr, conf,
                                 "bootstrap (judge uninformative in too many draws)", no, npos, nneg)
        theta = np.clip((p_d[informative] + tnr_d[informative] - 1.0) / j_d[informative], 0.0, 1.0)
        lo, hi = np.quantile(theta, [alpha / 2.0, 1.0 - alpha / 2.0])
        return CorrectedRate(estimate, float(lo), float(hi), float(theta.std(ddof=1)), p_obs,
                             unclipped, tpr, tnr, conf, "bootstrap", no, npos, nneg)
    raise ValueError(f"method must be 'delta' or 'bootstrap', got {method!r}")


# ------------------------------------------------------------------------------- PPI++

@dataclass(frozen=True)
class PPIResult:
    """A prediction-powered mean with its interval, beside the human-labels-only answer.

    `lam` is the weight given to the judge (0 = ignore it, 1 = classic PPI). The classical fields
    are the mean of the human labels alone with its Wald interval, so a report can show what the
    judge bought.
    """

    estimate: float
    lo: float
    hi: float
    se: float
    lam: float
    n_labelled: int
    n_unlabelled: int
    classical_estimate: float
    classical_lo: float
    classical_hi: float
    classical_se: float
    conf: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def ppi_mean(y_lab: np.ndarray, yhat_lab: np.ndarray, yhat_unlab: np.ndarray, conf: float,
             power_tuning: bool = True) -> PPIResult:
    """PPI++ estimate of a mean (Angelopoulos, Duchi & Zrnic 2023; Zrnic & Candes 2024).

    y_lab: human labels on n items; yhat_lab: the judge on the same items; yhat_unlab: the judge on
    N items without human labels. The estimator

        theta_lambda = mean(Y) + lambda * (mean(Yhat_unlab) - mean(Yhat_lab))

    is unbiased for every lambda (the judge only enters through a difference of two means of the
    same quantity), and its variance Var(Y - lambda Yhat)/n + lambda^2 Var(Yhat)/N is minimised at
    lambda* = Cov(Y, Yhat) / ((1 + n/N) Var(Yhat)). With `power_tuning`, lambda is that estimate,
    clipped to [0, 1]: a judge anti-correlated with the humans gets weight 0 (the classical mean),
    and the result is never wider, asymptotically, than the human labels alone. Without it, lambda
    = 1 (the original PPI). The interval is the Wald interval of the CLT on both samples.
    """
    conf = check_conf(conf)
    y = as_float_array(y_lab, "y_lab", min_len=2)
    f = as_float_array(yhat_lab, "yhat_lab", min_len=2)
    fu = as_float_array(yhat_unlab, "yhat_unlab", min_len=2)
    if y.size != f.size:
        raise ValueError(f"y_lab and yhat_lab must pair item by item, got {y.size} and {f.size}")
    n, big_n = y.size, fu.size
    z = z_two_sided(conf)
    classical_se = float(np.std(y, ddof=1) / math.sqrt(n))
    classical = float(y.mean())
    if power_tuning:
        var_f = float(np.var(np.concatenate([f, fu]), ddof=1))
        if var_f <= 0.0:
            lam = 0.0
        else:
            cov = float(np.sum((y - y.mean()) * (f - f.mean())) / (n - 1))
            lam = min(1.0, max(0.0, cov / ((1.0 + n / big_n) * var_f)))
    else:
        lam = 1.0
    estimate = classical + lam * (float(fu.mean()) - float(f.mean()))
    var = float(np.var(y - lam * f, ddof=1)) / n + lam * lam * float(np.var(fu, ddof=1)) / big_n
    se = math.sqrt(max(var, 0.0))
    return PPIResult(estimate=estimate, lo=estimate - z * se, hi=estimate + z * se, se=se,
                     lam=lam, n_labelled=n, n_unlabelled=big_n, classical_estimate=classical,
                     classical_lo=classical - z * classical_se, classical_hi=classical + z * classical_se,
                     classical_se=classical_se, conf=conf)


def label_budget(rho: float, half_width: float, conf: float, *, n_unlabelled: float | None = None,
                 variance: float = 0.25) -> int:
    """Human labels needed for a mean's interval to have half-width `half_width`.

    Without a judge: n0 = z^2 · variance / h^2 (variance 0.25 is the worst case for a rate). With
    PPI++ at the optimal lambda and a judge whose correlation with the humans is rho, the variance
    factor is 1 - rho^2 · N/(n + N) (N unlabelled items), so

        n = n0 · (1 - rho^2 N/(n + N)),

    solved exactly for n (a quadratic); with N -> infinity (`n_unlabelled=None`, the judge applied
    to everything) n = n0 (1 - rho^2). Rounded up: a budget that misses its target width by one
    label is not a budget. At conf 0.95, h = 0.10: 97 without a judge, then 73, 49 and 19 at
    rho = 0.5, 0.7, 0.9 (the plan's 96, 72, 49, 18 round to nearest).
    """
    rho = check_real(rho, "rho")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [-1, 1], got {rho!r}")
    half_width = check_real(half_width, "half_width")
    if half_width <= 0.0:
        raise ValueError(f"half_width must be positive, got {half_width!r}")
    variance = check_real(variance, "variance")
    if variance <= 0.0:
        raise ValueError(f"variance must be positive, got {variance!r}")
    z = z_two_sided(conf)
    n0 = z * z * variance / (half_width * half_width)
    r2 = rho * rho
    if n_unlabelled is None:
        n = n0 * (1.0 - r2)
    else:
        big_n = check_real(n_unlabelled, "n_unlabelled")
        if big_n <= 0:
            raise ValueError(f"n_unlabelled must be positive, got {n_unlabelled!r}")
        # n^2 + (N - n0) n - n0 N (1 - rho^2) = 0, positive root
        b = big_n - n0
        n = 0.5 * (-b + math.sqrt(b * b + 4.0 * n0 * big_n * (1.0 - r2)))
    return max(1, math.ceil(n - 1e-9))
