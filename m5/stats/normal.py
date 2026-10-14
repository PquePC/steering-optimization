"""m5.stats.normal - the normal distribution's CDF and quantile, with scipy optional.

Every Wilson interval, every delta-method interval and every d-prime needs Phi^{-1}. Without scipy
the quantile comes from Acklam's rational approximation (relative error 1.15e-9) followed by two
Halley steps against `math.erfc`, which brings it to full double precision; the tests compare it
with `scipy.special.ndtri`.

    norm_cdf(x)          Phi(x)
    norm_pdf(x)          phi(x)
    norm_ppf(p)          Phi^{-1}(p), 0 < p < 1
    z_two_sided(conf)    the z of a two-sided interval at level conf (1.95996... at 0.95)
    z_one_sided(conf)    the z of a one-sided bound at level conf (1.64485... at 0.95)
"""

from __future__ import annotations

import math

from m5.stats._checks import check_conf

try:  # optional: the fallback below is exact to double precision
    from scipy import special as _sp
except ImportError:  # pragma: no cover - exercised by monkeypatching `_sp` to None in the tests
    _sp = None

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)

# Acklam (2003), "An algorithm for computing the inverse normal cumulative distribution function"
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def norm_cdf(x: float) -> float:
    """Phi(x), through erfc so that the lower tail keeps its relative precision."""
    return 0.5 * math.erfc(-float(x) / _SQRT2)


def norm_pdf(x: float) -> float:
    x = float(x)
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _ppf_lower(p: float) -> float:
    """Phi^{-1}(p) for 0 < p <= 0.5, where the tail probability itself is representable."""
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
             / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))
    else:
        q = p - 0.5
        r = q * q
        x = ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
             / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0))
    for _ in range(2):  # Halley refinement against the exact CDF
        e = 0.5 * math.erfc(-x / _SQRT2) - p
        u = e * _SQRT2PI * math.exp(0.5 * x * x)
        x = x - u / (1.0 + 0.5 * x * u)
    return x


def norm_ppf_py(p: float) -> float:
    """Phi^{-1}(p) without scipy: Acklam's approximation, refined by two Halley steps."""
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"the normal quantile needs 0 < p < 1, got {p!r}")
    if p <= 0.5:
        return _ppf_lower(p)
    return -_ppf_lower(1.0 - p)


def norm_ppf(p: float) -> float:
    """Phi^{-1}(p) for 0 < p < 1 (scipy's ndtri when available)."""
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"the normal quantile needs 0 < p < 1, got {p!r}")
    if _sp is not None:
        return float(_sp.ndtri(p))
    return norm_ppf_py(p)


def z_two_sided(conf: float) -> float:
    """The z of a two-sided interval at level conf: Phi^{-1}((1 + conf) / 2)."""
    return norm_ppf(0.5 + 0.5 * check_conf(conf))


def z_one_sided(conf: float) -> float:
    """The z of a one-sided bound at level conf: Phi^{-1}(conf)."""
    return norm_ppf(check_conf(conf))
