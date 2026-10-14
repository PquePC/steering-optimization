"""m5.stats.beta - log Gamma, the incomplete beta function and its inverse, with scipy optional.

Every exact binomial statement in M5 is a statement about the regularized incomplete beta function
I_x(a, b): a Clopper-Pearson bound is its inverse, a binomial tail probability is its value, a
Beta posterior's interval is its inverse again. scipy has all three, but the production pod may not
have scipy, and a bound that changes with the machine is a different measurement under the same
name. So each public function uses scipy when it is importable and otherwise a numpy/Python
implementation: I_x to about 1e-12 absolute and its inverse to about 1e-10 relative even with a or
b in the thousands (where the log-Gamma prefactor limits precision), and near machine precision in
the range intervals use. The tests hold the two paths against each other, and against scipy
directly, so neither can drift unseen.

    lgamma(x)            log Gamma for x > 0, vectorised (Lanczos, g = 7, n = 9, without scipy)
    betainc(a, b, x)     I_x(a, b), by its continued fraction evaluated with the modified Lentz
                         method (Press et al. 2007, Numerical Recipes 3rd ed., §6.4)
    betaincinv(a, b, p)  the x with I_x(a, b) = p: Halley steps kept inside a shrinking bracket,
                         from the starting guess of Press et al. §6.14
    binom_cdf(k, n, p)   P(X <= k) for X ~ Binomial(n, p), as I_{1-p}(n - k, k + 1)
    binom_sf(k, n, p)    P(X >= k), as I_p(k, n - k + 1)

The `*_py` functions are the fallbacks themselves, public so the tests can compare them with
scipy on a machine that has it. Arguments are real (not only integer) wherever the mathematics
allows, because effective counts under a design effect are real (`m5.stats.intervals`).
"""

from __future__ import annotations

import math

import numpy as np

try:  # optional: every function below has a numpy/Python fallback
    from scipy import special as _sp
except ImportError:  # pragma: no cover - exercised by monkeypatching `_sp` to None in the tests
    _sp = None

_LANCZOS_G = 7.0
_LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905, -0.13857109526572012,
            9.9843695780195716e-6, 1.5056327351493116e-7)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

_CF_EPS = 1e-15       # relative convergence of the continued fraction
_CF_TINY = 1e-300     # Lentz's guard against division by zero
_CF_MAXIT = 20_000    # the fraction needs O(sqrt(max(a, b))) terms near the mode


def have_scipy() -> bool:
    """True when the scipy path is in use (for logging which implementation produced a number)."""
    return _sp is not None


# ------------------------------------------------------------------------------- log Gamma

def lgamma_py(x: np.ndarray | float) -> np.ndarray:
    """log Gamma(x) for x > 0, vectorised, by the Lanczos approximation (g = 7, 9 terms).

    Relative error in Gamma about 1e-15, i.e. absolute error about 1e-15 in log Gamma, which is
    what the beta-binomial likelihood on a 10^5-point grid needs: `math.lgamma` is exact but
    scalar, and calling it a million times per update is the slow part of a grid posterior.
    Reflection handles 0 < x < 0.5.
    """
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("lgamma is defined here for finite x > 0 only")
    out = np.empty_like(x)
    small = x < 0.5
    big = ~small
    if np.any(big):
        out[big] = _lanczos(x[big])
    if np.any(small):
        xs = x[small]
        # Gamma(x) Gamma(1 - x) = pi / sin(pi x)
        out[small] = math.log(math.pi) - np.log(np.sin(math.pi * xs)) - _lanczos(1.0 - xs)
    return out


def _lanczos(x: np.ndarray) -> np.ndarray:
    z = x - 1.0
    acc = np.full_like(z, _LANCZOS[0])
    for i in range(1, len(_LANCZOS)):
        acc = acc + _LANCZOS[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return _HALF_LOG_2PI + (z + 0.5) * np.log(t) - t + np.log(acc)


def lgamma(x: np.ndarray | float) -> np.ndarray:
    """log Gamma(x) for x > 0 (scipy's gammaln when available)."""
    if _sp is not None:
        x = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(x)) or np.any(x <= 0):
            raise ValueError("lgamma is defined here for finite x > 0 only")
        return _sp.gammaln(x)
    return lgamma_py(x)


# ------------------------------------------------------------------------------- I_x(a, b)

def _check_ab(a: float, b: float) -> None:
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
        raise ValueError(f"beta parameters must be finite and positive, got a={a!r}, b={b!r}")


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction for I_x(a, b) (Press et al. 2007, eq. 6.4.5), modified Lentz."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _CF_TINY:
        d = _CF_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def betainc_py(a: float, b: float, x: float) -> float:
    """I_x(a, b) for one (a, b, x): the fraction is evaluated on whichever side converges fast."""
    a = float(a)
    b = float(b)
    x = float(x)
    _check_ab(a, b)
    if not math.isfinite(x):
        raise ValueError(f"x must be finite, got {x!r}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) (scipy's betainc when available)."""
    _check_ab(float(a), float(b))
    if _sp is not None:
        return float(_sp.betainc(float(a), float(b), min(max(float(x), 0.0), 1.0)))
    return betainc_py(a, b, x)


def _initial_guess(a: float, b: float, p: float) -> float:
    """Press et al. (2007) §6.14: a normal-approximation start for a, b >= 1, else a power law."""
    if a >= 1.0 and b >= 1.0:
        pp = p if p < 0.5 else 1.0 - p
        t = math.sqrt(-2.0 * math.log(pp))
        x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
        if p < 0.5:
            x = -x
        al = (x * x - 3.0) / 6.0
        h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
        w = (x * math.sqrt(al + h) / h
             - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h)))
        return a / (a + b * math.exp(2.0 * w))
    lna = math.log(a / (a + b))
    lnb = math.log(b / (a + b))
    t = math.exp(a * lna) / a
    u = math.exp(b * lnb) / b
    w = t + u
    if p < t / w:
        return (a * w * p) ** (1.0 / a)
    return 1.0 - (b * w * (1.0 - p)) ** (1.0 / b)


def betaincinv_py(a: float, b: float, p: float) -> float:
    """The x in [0, 1] with I_x(a, b) = p.

    Halley's method on I_x(a, b) - p, with every step kept inside a bracket [lo, hi] that is
    narrowed at each evaluation; a step that would leave the bracket is replaced by bisection
    (geometric when the bracket spans orders of magnitude, so roots near 1e-200 are still found).
    Converges to a relative tolerance of a few ulps.
    """
    a = float(a)
    b = float(b)
    p = float(p)
    _check_ab(a, b)
    if not math.isfinite(p):
        raise ValueError(f"p must be finite, got {p!r}")
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    lo, hi = 0.0, 1.0
    x = min(max(_initial_guess(a, b, p), 1e-300), 1.0 - 1e-16)
    for _ in range(400):
        f = betainc_py(a, b, x) - p
        if f == 0.0:
            return x
        if f < 0.0:
            lo = x
        else:
            hi = x
        if hi - lo <= 4.0 * np.finfo(float).eps * max(hi, 1e-300):
            break
        log_pdf = log_norm + (a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x)
        x_new = math.nan
        if log_pdf < 700.0:
            dens = math.exp(log_pdf)
            if dens > 0.0:
                u = f / dens
                corr = u * ((a - 1.0) / x - (b - 1.0) / (1.0 - x))
                x_new = x - u / (1.0 - 0.5 * min(1.0, corr))
        if not (lo < x_new < hi):
            if lo > 0.0 and hi / lo > 1e3:
                x_new = math.sqrt(lo * hi)
            elif lo == 0.0 and hi < 1e-6:
                x_new = hi * 1e-3
            else:
                x_new = 0.5 * (lo + hi)
        if abs(x_new - x) <= 2.0 * np.finfo(float).eps * x_new:
            x = x_new
            break
        x = x_new
    return x


def betaincinv(a: float, b: float, p: float) -> float:
    """Inverse of the regularized incomplete beta in x (scipy's betaincinv when available)."""
    _check_ab(float(a), float(b))
    if _sp is not None:
        return float(_sp.betaincinv(float(a), float(b), min(max(float(p), 0.0), 1.0)))
    return betaincinv_py(a, b, p)


# ------------------------------------------------------------------------------- binomial tails

def _check_binom(n: float, p: float) -> None:
    if not math.isfinite(n) or n < 0:
        raise ValueError(f"n must be a finite non-negative count, got {n!r}")
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"p must lie in [0, 1], got {p!r}")


def binom_cdf(k: float, n: float, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), exactly, through I_{1-p}(n - k, k + 1).

    The lower tail is what a Learn-then-Test p-value for a claim "rate <= threshold" needs
    (`m5.stats.selection`): the chance of seeing as few events as observed if the rate were at
    the threshold.
    """
    _check_binom(n, p)
    k = math.floor(k)
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    return betainc(n - k, k + 1.0, 1.0 - p)


def binom_sf(k: float, n: float, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), through I_p(k, n - k + 1), accurate in the far tail."""
    _check_binom(n, p)
    k = math.ceil(k)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    return betainc(k, n - k + 1.0, p)


def log_binom_coef(n: np.ndarray | float, k: np.ndarray | float) -> np.ndarray:
    """log C(n, k) for 0 <= k <= n, vectorised (the constant of a binomial log-likelihood)."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    return lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)
