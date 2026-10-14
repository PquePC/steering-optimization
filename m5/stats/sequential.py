"""m5.stats.sequential - stopping rules that decide where samples go, never what is reported.

Plan §6.2. The adaptive search stops a visit as soon as its answer is clear, and stops an
influence cell as soon as its interval excludes the threshold. Both procedures stay valid however
early they stop, which is why they are used -- and both bias the estimate at the moment they stop,
which is why **none of them produces a reported number**. Reported rates come from the fixed-n
confirmation (§3.3).

    SPRT         Wald's (1945) sequential probability ratio test for a Bernoulli rate, truncated
                 at n_max; exact operating characteristic and average sample number by dynamic
                 programming, plus a Monte Carlo check
    BettingCS    the hedged-capital betting confidence sequence of Waudby-Smith & Ramdas (2024,
                 JRSS-B 86:1-27) for the mean of [0, 1] observations: valid at every n at once,
                 so it may be checked after each prompt

The SPRT per visit (H0 rate <= 0.15 against H1 >= 0.5, alpha = beta = 0.05, at most 30 trials:
SPRT_P0, SPRT_P1, SPRT_ALPHA, SPRT_BETA, SPRT_NMAX) has an average sample size of about 6 when the
true rate is 0, about 3 at 1, and about 15 near 0.3.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from m5.stats._checks import check_count, check_prob


CONTINUE = "continue"
ACCEPT_H0 = "accept_h0"
ACCEPT_H1 = "accept_h1"


@dataclass(frozen=True)
class OperatingCharacteristic:
    """At a true rate p: P(accept H1), the average sample number, and P(stopped by truncation)."""

    p: float
    p_accept_h1: float
    asn: float
    p_truncated: float


class SPRT:
    """Wald's sequential probability ratio test of H0: p = p0 against H1: p = p1 (p0 < p1).

    After each Bernoulli trial the log-likelihood ratio
        LLR = k log(p1/p0) + (n - k) log((1 - p1)/(1 - p0))
    is compared with Wald's (1945) boundaries log A = log((1 - beta)/alpha) (accept H1 at or
    above) and log B = log(beta/(1 - alpha)) (accept H0 at or below). The composite hypotheses
    rate <= p0 / rate >= p1 are covered because the error rates are monotone in p.

    **Truncation.** At n_max without a crossing the test accepts the hypothesis whose boundary the
    LLR is nearer to: H1 when LLR >= (log A + log B)/2, else H0, and records `truncated`. For
    alpha = beta the midpoint is 0, so this is "the observed rate is at least p*", the rate at
    which a trial carries no evidence either way (p* = 0.306 for 0.15 vs 0.5). Truncation changes
    the error rates slightly from alpha and beta; `operating_characteristic` gives the exact ones.

    `n_min` forbids a decision before that many trials (a visit's minimum trial count).
    """

    def __init__(self, p0: float, p1: float, alpha: float, beta: float, n_max: int, n_min: int = 1):
        self.p0 = check_prob(p0, "p0", open_interval=True)
        self.p1 = check_prob(p1, "p1", open_interval=True)
        if not self.p0 < self.p1:
            raise ValueError(f"p0={p0!r} must be below p1={p1!r}")
        self.alpha = check_prob(alpha, "alpha", open_interval=True)
        self.beta = check_prob(beta, "beta", open_interval=True)
        if self.alpha + self.beta >= 1.0:
            raise ValueError("alpha + beta must be below 1, or the boundaries cross")
        self.n_max = check_count(n_max, "n_max", minimum=1)
        self.n_min = check_count(n_min, "n_min", minimum=1)
        if self.n_min > self.n_max:
            raise ValueError(f"n_min={n_min!r} exceeds n_max={n_max!r}")
        self.log_a = math.log((1.0 - self.beta) / self.alpha)
        self.log_b = math.log(self.beta / (1.0 - self.alpha))
        self.step_yes = math.log(self.p1 / self.p0)
        self.step_no = math.log((1.0 - self.p1) / (1.0 - self.p0))
        self.n = 0
        self.k = 0
        self.llr = 0.0
        self.decision = CONTINUE
        self.truncated = False

    # ---------------------------------------------------------------- running the test

    def _decide(self, llr: float, n: int) -> tuple[str, bool]:
        if n >= self.n_min:
            if llr >= self.log_a:
                return ACCEPT_H1, False
            if llr <= self.log_b:
                return ACCEPT_H0, False
        if n >= self.n_max:
            mid = 0.5 * (self.log_a + self.log_b)
            return (ACCEPT_H1 if llr >= mid else ACCEPT_H0), True
        return CONTINUE, False

    def update(self, x: int | Iterable[int]) -> str:
        """Add one 0/1 outcome (or several, in order); returns the decision after them.

        Raises once the test has stopped: outcomes after the stopping time are not part of this
        test, and folding them in silently would turn a sequential test into an unplanned one.
        """
        xs = [x] if isinstance(x, (int, np.integer, bool, float)) else list(x)
        for xi in xs:
            if self.decision != CONTINUE:
                raise ValueError(f"the SPRT already stopped ({self.decision} at n={self.n}); "
                                 "start a new test for further trials")
            if xi not in (0, 1) or isinstance(xi, bool):
                raise ValueError(f"SPRT outcomes must be 0 or 1, got {xi!r}")
            self.n += 1
            self.k += int(xi)
            self.llr += self.step_yes if xi == 1 else self.step_no
            self.decision, self.truncated = self._decide(self.llr, self.n)
        return self.decision

    @property
    def stopped(self) -> bool:
        return self.decision != CONTINUE

    # ---------------------------------------------------------------- operating characteristics

    def operating_characteristic(self, p: float) -> OperatingCharacteristic:
        """Exact P(accept H1), average sample number and P(truncation) at true rate p.

        Dynamic programming over (trials, successes): the probability of being still running in
        each state is pushed forward one trial at a time and absorbed at the boundaries, so the
        answer is exact for the truncated test, not Wald's untruncated approximation.
        """
        p = check_prob(p, "p")
        running = np.zeros(self.n_max + 1)
        running[0] = 1.0
        acc_h1 = 0.0
        asn = 0.0
        trunc = 0.0
        for n in range(1, self.n_max + 1):
            nxt = np.zeros(self.n_max + 1)
            nxt[1:n + 1] += running[0:n] * p
            nxt[0:n] += running[0:n] * (1.0 - p)
            ks = np.arange(n + 1)
            llr = ks * self.step_yes + (n - ks) * self.step_no
            for k in range(n + 1):
                mass = nxt[k]
                if mass == 0.0:
                    continue
                decision, truncated = self._decide(float(llr[k]), n)
                if decision == CONTINUE:
                    continue
                asn += n * mass
                if decision == ACCEPT_H1:
                    acc_h1 += mass
                if truncated:
                    trunc += mass
                nxt[k] = 0.0
            running = nxt
        return OperatingCharacteristic(p=p, p_accept_h1=acc_h1, asn=asn, p_truncated=trunc)

    def simulate(self, p: float, reps: int, seed: int) -> OperatingCharacteristic:
        """Monte Carlo estimate of the same quantities (a check on `operating_characteristic`)."""
        p = check_prob(p, "p")
        reps = check_count(reps, "reps", minimum=1)
        rng = np.random.default_rng(seed)
        x = (rng.random((reps, self.n_max)) < p).astype(int)
        n_used = np.zeros(reps)
        h1 = np.zeros(reps, dtype=bool)
        tr = np.zeros(reps, dtype=bool)
        for r in range(reps):
            t = SPRT(self.p0, self.p1, self.alpha, self.beta, self.n_max, self.n_min)
            for xi in x[r]:
                if t.update(int(xi)) != CONTINUE:
                    break
            n_used[r], h1[r], tr[r] = t.n, t.decision == ACCEPT_H1, t.truncated
        return OperatingCharacteristic(p=p, p_accept_h1=float(h1.mean()), asn=float(n_used.mean()),
                                       p_truncated=float(tr.mean()))


class BettingCS:
    """Anytime-valid confidence sequence for the mean of [0, 1] observations, by betting.

    Waudby-Smith & Ramdas (2024), "Estimating means of bounded random variables by betting",
    JRSS-B 86(1):1-27, the hedged capital process with predictable plug-in bets (their §4-5):
    for each candidate mean m on a grid, two gamblers bet that the data's mean is above / below m,

        K+_t(m) = prod_i (1 + lam+_i(m) (x_i - m)),   K-_t(m) = prod_i (1 - lam-_i(m) (x_i - m)),

    with lam+_t(m) = min(lam_t, c/m), lam-_t(m) = min(lam_t, c/(1 - m)) (so the capital can never
    go negative) and the predictable plug-in

        lam_t = sqrt(2 log(2/alpha) / (sigma^2_{t-1} · t · log(1 + t))),
        mu_t = (1/2 + sum_{i<=t} x_i)/(t + 1),  sigma^2_t = (1/4 + sum_{i<=t} (x_i - mu_i)^2)/(t + 1).

    The hedged capital max(theta K+, (1 - theta) K-) at the true mean is a non-negative
    martingale, so by Ville's inequality it ever reaches 1/alpha with probability at most alpha:
    {m : hedged capital < 1/alpha} covers the mean at every t simultaneously. The running
    intersection over t is reported (also valid, never wider).

    Grid: `grid_size` points on [0, 1]; each reported endpoint is moved out to the nearest grid
    point that was *rejected*, so discretisation can only widen the interval. theta = 1/2 and
    c = 1/2 are the paper's defaults.
    """

    def __init__(self, alpha: float, *, grid_size: int = 1001, c: float = 0.5, theta: float = 0.5):
        self.alpha = check_prob(alpha, "alpha", open_interval=True)
        self.c = check_prob(c, "c", open_interval=True)
        self.theta = check_prob(theta, "theta", open_interval=True)
        grid_size = check_count(grid_size, "grid_size", minimum=11)
        self.m = np.linspace(0.0, 1.0, grid_size)
        self.log_kp = np.zeros(grid_size)
        self.log_km = np.zeros(grid_size)
        self.n = 0
        self.sum_x = 0.0
        self.sum_sq = 0.0          # sum of (x_i - mu_i)^2, mu_i including x_i
        self._sigma2_prev = 0.25   # sigma^2_0 = (1/4)/1
        self._lo = 0.0
        self._hi = 1.0
        self._threshold = math.log(1.0 / self.alpha)
        with np.errstate(divide="ignore"):
            self._cap_plus = np.where(self.m > 0, self.c / self.m, np.inf)
            self._cap_minus = np.where(self.m < 1, self.c / (1.0 - self.m), np.inf)

    def update(self, x: float | Iterable[float]) -> tuple[float, float]:
        """Add one observation in [0, 1] (or several, in order); returns the interval after them."""
        xs = [x] if np.isscalar(x) else list(x)
        for xi in xs:
            xi = float(xi)
            if not (math.isfinite(xi) and 0.0 <= xi <= 1.0):
                raise ValueError(f"observations must lie in [0, 1], got {xi!r}")
            t = self.n + 1
            lam = math.sqrt(2.0 * math.log(2.0 / self.alpha)
                            / (self._sigma2_prev * t * math.log(1.0 + t)))
            lam_p = np.minimum(lam, self._cap_plus)
            lam_m = np.minimum(lam, self._cap_minus)
            self.log_kp += np.log1p(lam_p * (xi - self.m))
            self.log_km += np.log1p(-lam_m * (xi - self.m))
            self.n = t
            self.sum_x += xi
            mu_t = (0.5 + self.sum_x) / (t + 1.0)
            self.sum_sq += (xi - mu_t) ** 2
            self._sigma2_prev = (0.25 + self.sum_sq) / (t + 1.0)   # used by the next bet
            self._intersect()
        return self.interval()

    def _intersect(self) -> None:
        log_hedged = np.maximum(math.log(self.theta) + self.log_kp,
                                math.log(1.0 - self.theta) + self.log_km)
        keep = log_hedged < self._threshold
        if not np.any(keep):  # pragma: no cover - only on the probability-alpha failure event
            return            # every mean rejected: keep the last non-empty interval
        idx = np.flatnonzero(keep)
        i_lo, i_hi = int(idx[0]), int(idx[-1])
        lo = float(self.m[i_lo - 1]) if i_lo > 0 else 0.0
        hi = float(self.m[i_hi + 1]) if i_hi < self.m.size - 1 else 1.0
        self._lo = max(self._lo, lo)
        self._hi = min(self._hi, hi)
        if self._lo > self._hi:  # the running intersection emptied: keep the latest set instead
            self._lo, self._hi = lo, hi

    def interval(self) -> tuple[float, float]:
        """The running-intersection confidence sequence at the current n ([0, 1] before data)."""
        return self._lo, self._hi

    @property
    def mean(self) -> float:
        if self.n == 0:
            raise ValueError("no observations yet")
        return self.sum_x / self.n

    def excludes(self, value: float) -> bool:
        """True once the sequence excludes `value` -- the stopping rule of plan §5.2 step 4."""
        lo, hi = self.interval()
        return value < lo or value > hi
