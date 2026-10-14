"""m5.stats.paired - two conditions measured on the same units.

The capability panel `s5` runs every item steered and unsteered; the live and past-only arms run
the same trial numbers. A paired comparison uses only the discordant units -- the items one
condition got right and the other wrong -- so it is far more sensitive than comparing two rates as
if they were independent samples, and an independent-samples interval on paired data is simply
the wrong width.

    mcnemar_exact(b, c)                      exact two-sided McNemar p-value
    paired_rate_difference(pairs, conf)      p_A - p_B with Tango's (1998) score interval
                                             (or Newcombe's 1998 method 10, on request)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from m5.stats._checks import check_conf, check_count
from m5.stats.beta import binom_cdf
from m5.stats.intervals import _wilson_bounds
from m5.stats.normal import z_two_sided


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test (McNemar 1947) on b = #(A yes, B no) and c = #(A no, B yes).

    Under no difference each discordant unit is a fair coin: p = min(1, 2 P(X <= min(b, c))),
    X ~ Binomial(b + c, 1/2). With no discordant units there is no evidence either way: p = 1.
    """
    b = check_count(b, "b")
    c = check_count(c, "c")
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(b, c), n, 0.5))


def _tango_z(delta: float, b: int, c: int, n: int) -> float:
    """Tango's score statistic for H0: p_A - p_B = delta.

    The restricted MLE of pi_01 (the (0, 1) cell) under the null solves 2n x^2 + B x + C = 0 with
    B = -b - c + (2n - b + c) delta and C = -c delta (1 - delta) (Tango 1998, eq. 4); then
    Z = (b - c - n delta) / sqrt(n (2 x + delta (1 - delta))).
    """
    big_a = 2.0 * n
    big_b = -b - c + (2.0 * n - b + c) * delta
    big_c = -c * delta * (1.0 - delta)
    disc = max(big_b * big_b - 4.0 * big_a * big_c, 0.0)
    x = (math.sqrt(disc) - big_b) / (2.0 * big_a)
    x = min(max(x, max(0.0, -delta)), (1.0 - delta) / 2.0)
    var = n * (2.0 * x + delta * (1.0 - delta))
    num = b - c - n * delta
    if var <= 0.0:
        return 0.0 if num == 0 else math.copysign(math.inf, num)
    return num / math.sqrt(var)


def _tango_interval(b: int, c: int, n: int, z: float) -> tuple[float, float]:
    """Invert |Z(delta)| <= z by bisection on each side of the estimate (Z falls as delta rises)."""
    est = (b - c) / n
    eps = 1e-12

    def bound(lo: float, hi: float, upper: bool) -> float:
        # find delta in [lo, hi] where |Z| crosses z; Z(est) = 0, |Z| grows away from est
        edge = hi if upper else lo
        if abs(_tango_z(edge, b, c, n)) <= z:
            return edge
        inside, outside = (lo, hi) if upper else (hi, lo)
        for _ in range(200):
            mid = 0.5 * (inside + outside)
            if abs(_tango_z(mid, b, c, n)) <= z:
                inside = mid
            else:
                outside = mid
            if abs(outside - inside) < 1e-13:
                break
        return inside

    lo = bound(-1.0 + eps, est, upper=False)
    hi = bound(est, 1.0 - eps, upper=True)
    lo = -1.0 if lo <= -1.0 + 2 * eps else lo
    hi = 1.0 if hi >= 1.0 - 2 * eps else hi
    return lo, hi


def _newcombe_interval(a: int, b: int, c: int, d: int, z: float) -> tuple[float, float, float]:
    n = a + b + c + d
    pa, pb = (a + b) / n, (a + c) / n
    la, ua = _wilson_bounds(a + b, n, z)
    lb, ub = _wilson_bounds(a + c, n, z)
    denom = float(a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(denom) if denom > 0 else 0.0
    diff = pa - pb
    dl = (pa - la) ** 2 - 2.0 * phi * (pa - la) * (ub - pb) + (ub - pb) ** 2
    du = (ua - pa) ** 2 - 2.0 * phi * (ua - pa) * (pb - lb) + (pb - lb) ** 2
    return max(-1.0, diff - math.sqrt(max(dl, 0.0))), min(1.0, diff + math.sqrt(max(du, 0.0))), phi


def paired_rate_difference(pairs: np.ndarray, conf: float, *, method: str = "tango") -> dict[str, Any]:
    """Difference of two paired rates, p_A - p_B, with a score interval.

    `pairs` is an (n, 2) array of 0/1 outcomes, one row per unit (item, prompt), column 0 the
    first condition. With a = #(1,1), b = #(1,0), c = #(0,1), d = #(0,0), the estimate is
    (b - c)/n.

    `method="tango"` (default): the score interval of Tango (1998, Statistics in Medicine
    17:891-908), {delta : |Z(delta)| <= z} with the variance taken under each candidate delta at
    its restricted maximum likelihood. It is the paired analogue of the Wilson interval: inside
    [-1, 1], close to nominal coverage in small samples, and never zero-width -- with no discordant
    units it is +-z^2/(n + z^2), the paired version of "0 events still bounds the rate".

    `method="newcombe"`: Newcombe's (1998, Statistics in Medicine 17:2635-2650) method 10, built
    from the two marginal Wilson intervals and the phi correlation. Closed-form, but it collapses
    to a zero-width interval when every pair is concordant (phi = 1), which is why it is not the
    default. The exact McNemar p-value is returned beside either.
    """
    conf = check_conf(conf)
    arr = np.asarray(pairs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"pairs must be an (n, 2) array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("pairs is empty: a paired rate needs at least one unit")
    if not np.all(np.isfinite(arr)) or not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("pairs must contain only 0/1 outcomes (exclude missing units first)")
    x, y = arr[:, 0], arr[:, 1]
    a = int(np.sum((x == 1) & (y == 1)))
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    d = int(np.sum((x == 0) & (y == 0)))
    n = a + b + c + d
    z = z_two_sided(conf)
    if method == "tango":
        lo, hi = _tango_interval(b, c, n, z)
        extra: dict[str, Any] = {}
    elif method == "newcombe":
        lo, hi, phi = _newcombe_interval(a, b, c, d, z)
        extra = {"phi": phi}
    else:
        raise ValueError(f"method must be 'tango' or 'newcombe', got {method!r}")
    return {"diff": (b - c) / n, "lo": lo, "hi": hi, "rate_a": (a + b) / n, "rate_b": (a + c) / n,
            "n": n, "a": a, "b": b, "c": c, "d": d, "p_mcnemar": mcnemar_exact(b, c),
            "conf": conf, "method": f"{method}-score" if method == "tango" else "newcombe-paired-10",
            **extra}
