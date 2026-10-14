"""m5.stats.intervals - the interval behind every reported rate, and the equivalence claim.

Plan §6.4. Each rule here replaces a way the old pipelines misreported a rate:

1. **Every rate is stored with k and n, and reported with its interval and its null**
   (implementation rule 6). `rate_with_null` builds that record, ready to store.
2. **Wilson (1927) for reported rates.** The score interval keeps its coverage near 0 and 1, where
   M2's `p ± SE` read `0.000 ± 0.000` off 25 trials (handoff 17), and it never leaves [0, 1].
3. **Clopper-Pearson (1934) for bounds near zero**, where an exact guarantee matters more than
   width: with zero events the two-sided 95% upper end is 11.6% at n = 30, 3.0% at n = 120 and
   2.4% at n = 150 (plan §6.4).
4. **"Indistinguishable from null within delta" is an equivalence test**, not a failed difference
   test (Venkatesh et al.): the upper confidence bound on (steered - null) must be below delta
   (`tost_rate_difference`). The difference interval is Newcombe's (1998) hybrid score interval,
   method 10, built from the two Wilson intervals; unlike the Wald difference it behaves at 0/n.

**Clustering.** Trials are clustered by prompt, so k and n overstate the information. Every
function that takes `deff` divides both counts by the design effect 1 + (m - 1)·ICC (Kish 1965;
`m5.stats.clustering`) before computing, so the effective sample size sets the width. deff = 1 is
"trials independent", and it is written into the stored record so a reader sees which was used.

**Missing data.** Every function takes valid counts only. A trial whose judgment failed is
excluded by the caller, never counted as a "no" (implementation rule 5); n = 0 raises, because a
rate over nothing is missing, not zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from m5.stats._checks import check_conf, check_counts, check_real
from m5.stats.beta import betaincinv
from m5.stats.normal import z_one_sided, z_two_sided


# ------------------------------------------------------------------------------- one rate

def _wilson_bounds(k: float, n: float, z: float) -> tuple[float, float]:
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi


def wilson(k: float, n: float, conf: float) -> tuple[float, float]:
    """Two-sided Wilson (1927) score interval for k events in n trials, at level `conf`.

    The inversion of the score test, so its endpoints are the rates the data cannot reject. At
    k = 0 the lower end is exactly 0 and the upper end is z^2 / (n + z^2): 0.114 at n = 30.
    """
    k, n = check_counts(k, n)
    return _wilson_bounds(k, n, z_two_sided(conf))


def wilson_upper(k: float, n: float, conf: float) -> float:
    """One-sided Wilson upper bound at level `conf` (the upper end at z = Phi^{-1}(conf))."""
    k, n = check_counts(k, n)
    return _wilson_bounds(k, n, z_one_sided(conf))[1]


def wilson_lower(k: float, n: float, conf: float) -> float:
    """One-sided Wilson lower bound at level `conf`."""
    k, n = check_counts(k, n)
    return _wilson_bounds(k, n, z_one_sided(conf))[0]


def clopper_pearson(k: float, n: float, conf: float) -> tuple[float, float]:
    """Two-sided Clopper-Pearson (1934) interval: the exact binomial interval, at least `conf`.

    lower = B^{-1}(alpha/2; k, n - k + 1), upper = B^{-1}(1 - alpha/2; k + 1, n - k), with the
    ends pinned to 0 and 1 at k = 0 and k = n. Conservative by construction, which is the point
    when the claim is "the rate is below x" and k is zero or nearly so.
    """
    k, n = check_counts(k, n)
    alpha = 1.0 - check_conf(conf)
    lo = 0.0 if k == 0 else betaincinv(k, n - k + 1.0, alpha / 2.0)
    hi = 1.0 if k == n else betaincinv(k + 1.0, n - k, 1.0 - alpha / 2.0)
    return lo, hi


def clopper_pearson_upper(k: float, n: float, conf: float) -> float:
    """One-sided exact upper bound at level `conf`: 1 - (1 - conf)^(1/n) when k = 0."""
    k, n = check_counts(k, n)
    conf = check_conf(conf)
    return 1.0 if k == n else betaincinv(k + 1.0, n - k, conf)


def clopper_pearson_lower(k: float, n: float, conf: float) -> float:
    """One-sided exact lower bound at level `conf`."""
    k, n = check_counts(k, n)
    conf = check_conf(conf)
    return 0.0 if k == 0 else betaincinv(k, n - k + 1.0, 1.0 - conf)


# ------------------------------------------------------------------------------- two rates

def newcombe_diff(k1: float, n1: float, k2: float, n2: float, conf: float) -> tuple[float, float, float]:
    """(p1 - p2, lower, upper): Newcombe's (1998) hybrid score interval, method 10.

    Each rate's Wilson interval (l_i, u_i) supplies its own asymmetric error, combined in
    quadrature: lower = d - sqrt((p1 - l1)^2 + (u2 - p2)^2), upper = d + sqrt((u1 - p1)^2 +
    (p2 - l2)^2). Coverage stays near nominal at 0/n and n/n, where the Wald difference interval
    collapses to a point (Newcombe 1998, Statistics in Medicine 17:873-890).
    """
    k1, n1 = check_counts(k1, n1, what="first rate")
    k2, n2 = check_counts(k2, n2, what="second rate")
    z = z_two_sided(conf)
    l1, u1 = _wilson_bounds(k1, n1, z)
    l2, u2 = _wilson_bounds(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, max(-1.0, lo), min(1.0, hi)


@dataclass(frozen=True)
class Equivalence:
    """The equivalence verdict for (steered - null) against a margin delta.

    `equivalent` is the plan's claim (§6.4, §6.7 "covert on channel c"): the one-sided `conf`
    upper bound on steered - null is below delta. Steered rates *below* the null are not a
    problem for a covertness claim, so the lower side is reported but does not gate it;
    `equivalent_two_sided` adds the classic TOST lower test (lower bound above -delta).
    """

    diff: float
    lower: float
    upper: float
    delta: float
    conf: float
    equivalent: bool
    equivalent_two_sided: bool
    deff: float
    null_deff: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def tost_rate_difference(k_steered: float, n_steered: float, k_null: float, n_null: float,
                         delta: float, conf: float, *, deff: float = 1.0,
                         null_deff: float | None = None) -> Equivalence:
    """Two one-sided tests (Schuirmann 1987) on the difference of two rates, margin `delta`.

    `conf` is the one-sided level of each bound: TOST at alpha = 1 - conf, equivalently the two
    ends of the (2·conf - 1) two-sided Newcombe interval. With conf = 0.95 the upper bound is the
    "upper 95% bound" of plan §6.7. Non-significance is not equivalence: a cell with 3 trials has
    no significant difference from anything and is equivalent to nothing, because its upper bound
    is wide.
    """
    conf = check_conf(conf)
    if conf <= 0.5:
        raise ValueError(f"conf={conf!r}: a one-sided bound needs a level above 0.5")
    delta = check_real(delta, "delta")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie strictly between 0 and 1, got {delta!r}")
    deff, null_deff = _check_deffs(deff, null_deff)
    ks, ns = check_counts(k_steered, n_steered, what="steered")
    k0, n0 = check_counts(k_null, n_null, what="null")
    d, lo, hi = newcombe_diff(ks / deff, ns / deff, k0 / null_deff, n0 / null_deff, 2.0 * conf - 1.0)
    return Equivalence(diff=d, lower=lo, upper=hi, delta=delta, conf=conf,
                       equivalent=hi < delta, equivalent_two_sided=(hi < delta and lo > -delta),
                       deff=deff, null_deff=null_deff)


def _check_deffs(deff: float, null_deff: float | None) -> tuple[float, float]:
    deff = check_real(deff, "deff")
    null_deff = deff if null_deff is None else check_real(null_deff, "null_deff")
    for val, name in ((deff, "deff"), (null_deff, "null_deff")):
        if val < 1.0:
            raise ValueError(f"{name}={val!r}: a design effect below 1 would claim clustering "
                             "adds information; it is 1 + (m - 1)·ICC with ICC >= 0")
    return deff, null_deff


# ------------------------------------------------------------------------------- the stored record

def rate_with_null(k: float, n: float, k_null: float, n_null: float, conf: float, *,
                   deff: float = 1.0, null_deff: float | None = None,
                   delta: float | None = None) -> dict[str, Any]:
    """Everything a stored rate carries: counts, rate, interval, its null, and the difference.

    Wilson intervals for both rates (effective counts k/deff, n/deff when trials are clustered),
    the exact Clopper-Pearson upper bound beside each (for "below x" claims near zero), and
    Newcombe's interval on rate - null. With `delta`, the equivalence verdict of
    `tost_rate_difference` at one-sided level `conf` is added. The raw k and n are stored
    unchanged; the effective n is stored beside them.
    """
    conf = check_conf(conf)
    deff, null_deff = _check_deffs(deff, null_deff)
    kf, nf = check_counts(k, n, what="steered")
    k0, n0 = check_counts(k_null, n_null, what="null")
    ke, ne = kf / deff, nf / deff
    k0e, n0e = k0 / null_deff, n0 / null_deff
    lo, hi = wilson(ke, ne, conf)
    nlo, nhi = wilson(k0e, n0e, conf)
    d, dlo, dhi = newcombe_diff(ke, ne, k0e, n0e, conf)
    out: dict[str, Any] = {
        "k": k, "n": n, "rate": kf / nf, "lo": lo, "hi": hi,
        "cp_hi": clopper_pearson(ke, ne, conf)[1],
        "null_k": k_null, "null_n": n_null, "null_rate": k0 / n0, "null_lo": nlo, "null_hi": nhi,
        "null_cp_hi": clopper_pearson(k0e, n0e, conf)[1],
        "diff": d, "diff_lo": dlo, "diff_hi": dhi,
        "conf": conf, "deff": deff, "null_deff": null_deff, "n_eff": ne, "null_n_eff": n0e,
        "method": "wilson; newcombe-10 difference",
    }
    if delta is not None:
        eq = tost_rate_difference(k, n, k_null, n_null, delta, conf, deff=deff, null_deff=null_deff)
        out.update({"delta": eq.delta, "equiv_upper": eq.upper, "equiv_lower": eq.lower,
                    "equivalent": eq.equivalent})
    return out
