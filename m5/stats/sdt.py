"""m5.stats.sdt - signal detection: sensitivity apart from the willingness to say yes.

The briefed self-report `d8` asks "do you detect an injected thought?" on steered trials and on
control trials. A model that says yes more often under steering may have become more *sensitive*
or merely more *willing*; d-prime separates the two, the criterion c measures the willingness
(plan §3.2, `d8`'s d-prime field). For continuous readouts (`q1` probe scores, a judge's
probability) `tpr_at_fpr` is the operating point at a fixed false-alarm rate.

    dprime(hits, n_signal, false_alarms, n_noise)   d', c, the corrected rates and d''s s.e.
    tpr_at_fpr(scores_signal, scores_noise, fpr)    the conservative operating point
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from m5.stats._checks import as_float_array, check_counts, check_prob
from m5.stats.normal import norm_pdf, norm_ppf


@dataclass(frozen=True)
class SDT:
    """d' = z(H) - z(F) and c = -(z(H) + z(F))/2 on the log-linear-corrected rates.

    `se_dprime` is the large-sample standard error of Gourevitch & Galanter (1967),
    sqrt(H(1-H)/(N_s phi(z_H)^2) + F(1-F)/(N_n phi(z_F)^2)).
    """

    dprime: float
    criterion: float
    hit_rate: float
    fa_rate: float
    se_dprime: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def dprime(hits: int, n_signal: int, false_alarms: int, n_noise: int) -> SDT:
    """Sensitivity and criterion with the log-linear correction (Hautus 1995).

    H = (hits + 0.5) / (n_signal + 1), F = (false_alarms + 0.5) / (n_noise + 1). Without a
    correction, 0 false alarms or a perfect hit rate make d' infinite -- the usual state of a
    control arm at its floor. The log-linear rule adds half a count to every cell, is applied to
    every rate whether extreme or not (so it never depends on the data's value), and is the least
    biased of the standard corrections (Hautus 1995, Behavior Research Methods 27:46-51).
    """
    h, ns = check_counts(hits, n_signal, what="signal trials", integer=True)
    fa, nn = check_counts(false_alarms, n_noise, what="noise trials", integer=True)
    hr = (h + 0.5) / (ns + 1.0)
    fr = (fa + 0.5) / (nn + 1.0)
    zh, zf = norm_ppf(hr), norm_ppf(fr)
    se = math.sqrt(hr * (1.0 - hr) / (ns * norm_pdf(zh) ** 2)
                   + fr * (1.0 - fr) / (nn * norm_pdf(zf) ** 2))
    return SDT(dprime=zh - zf, criterion=-0.5 * (zh + zf), hit_rate=hr, fa_rate=fr, se_dprime=se)


def tpr_at_fpr(scores_signal: np.ndarray, scores_noise: np.ndarray, fpr: float) -> dict[str, float]:
    """True-positive rate at a false-positive rate of at most `fpr`, for continuous scores.

    The threshold t is the smallest score such that the share of noise scores strictly above it
    is at most `fpr`; trials score positive when strictly above t. Ties at t therefore count as
    negatives, so the achieved FPR never exceeds the requested one (no interpolation between ROC
    points: an interpolated point is a randomised classifier, which no readout here is). Returns
    the TPR, the achieved FPR and the threshold (-inf when every trial may be called positive).
    """
    sig = as_float_array(scores_signal, "scores_signal")
    noise = as_float_array(scores_noise, "scores_noise")
    fpr = check_prob(fpr, "fpr")
    allowed = math.floor(fpr * noise.size + 1e-12)   # noise scores allowed strictly above t
    if allowed >= noise.size:
        t = -math.inf
    else:
        desc = np.sort(noise)[::-1]
        t = float(desc[allowed])
    return {"tpr": float(np.mean(sig > t)), "fpr": float(np.mean(noise > t)), "threshold": t,
            "n_signal": int(sig.size), "n_noise": int(noise.size)}
