"""m5.stats.halving - which layers keep receiving samples.

Plan §5.2 step 3. After each round the search keeps only the layers that could still be the best
one, and spends the next round's visits there:

    keep_layers(lower, upper)      successive-halving elimination by confidence bounds
    apt_index(means, counts, tau)  the APT priority of Locatelli, Gutzeit & Carpentier (2016)

`keep_layers` is the elimination rule of successive elimination / LUCB-style bandits (Even-Dar,
Mannor & Mansour 2006; Kalyanakrishnan et al. 2012): a layer is dropped only when some other
layer's lower bound on R_l is above its upper bound, i.e. when it is confidently worse than
something. Ties and overlaps keep a layer. This allocates samples; it never decides what is
reported (plan §6.2).

`apt_index` serves the thresholding question instead of the best-arm one -- "which cells are
above or below tau?" (is a cell's detection above its tolerance, is an influence cell above
EFFECT_MIN?): APT samples next the arm with the smallest sqrt(T_i) (|mu_i - tau| + eps), the one
whose side of the threshold is least settled, and is minimax optimal for that problem.
"""

from __future__ import annotations

import numpy as np

from m5.stats._checks import as_float_array, check_real


def keep_layers(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Boolean mask: keep layer l when upper[l] >= max over layers of lower.

    `lower` and `upper` are confidence (or credible) bounds on each layer's summary R_l at the
    same level. The best layer always survives (its upper bound is at least its own lower bound).
    """
    lo = as_float_array(lower, "lower")
    hi = as_float_array(upper, "upper")
    if lo.size != hi.size:
        raise ValueError(f"lower and upper must have the same length, got {lo.size} and {hi.size}")
    if np.any(lo > hi):
        raise ValueError("every lower bound must be at most its upper bound")
    return hi >= lo.max()


def apt_index(means: np.ndarray, counts: np.ndarray, tau: float, eps: float) -> np.ndarray:
    """APT priority B_i = sqrt(T_i) · (|mu_i - tau| + eps); sample the smallest next.

    `eps` is the precision below which the side of the threshold does not matter (Locatelli et
    al. 2016, "An optimal algorithm for the thresholding bandit problem", ICML). Arms never
    sampled have index 0, so each is sampled once before any is sampled twice.
    """
    mu = as_float_array(means, "means")
    t = as_float_array(counts, "counts")
    if mu.size != t.size:
        raise ValueError("means and counts must have the same length")
    if np.any(t < 0):
        raise ValueError("counts cannot be negative")
    tau = check_real(tau, "tau")
    eps = check_real(eps, "eps")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps!r}")
    return np.sqrt(t) * (np.abs(mu - tau) + eps)


def apt_next(means: np.ndarray, counts: np.ndarray, tau: float, eps: float) -> int:
    """The arm APT samples next (lowest index; the first such arm on ties)."""
    return int(np.argmin(apt_index(means, counts, tau, eps)))
