"""Input guards shared by every module in m5.stats.

Each raises ValueError with a sentence naming the argument and what is wrong with it. Statistics
computed from impossible inputs (k > n, a negative count, a confidence of 1.2) do not fail loudly
on their own; they return a plausible number, and a plausible number is read as evidence
(AGENTS.md: a guard that cannot fail is worse than none -- each of these has a test that trips it).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable

import numpy as np


def check_conf(conf: object, name: str = "conf") -> float:
    """A confidence level must lie strictly between 0 and 1."""
    if isinstance(conf, bool) or not isinstance(conf, numbers.Real) or not math.isfinite(conf):
        raise ValueError(f"{name} must be a number strictly between 0 and 1, got {conf!r}")
    if not 0.0 < float(conf) < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {conf!r}")
    return float(conf)


def check_real(x: object, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number, got {x!r}")
    return float(x)


def check_prob(p: object, name: str, *, open_interval: bool = False) -> float:
    """A probability in [0, 1] (or (0, 1) when `open_interval`)."""
    v = check_real(p, name)
    if open_interval and not 0.0 < v < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {p!r}")
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {p!r}")
    return v


def check_counts(k: object, n: object, *, what: str = "", integer: bool = False) -> tuple[float, float]:
    """k events out of n trials: 0 <= k <= n and n > 0.

    n = 0 is refused rather than answered: a rate over nothing is missing, not zero (implementation
    rule 5), and the caller records it as missing with its reason. Real (non-integer) counts are
    accepted unless `integer`, because effective counts under a design effect are real.
    """
    label = f" ({what})" if what else ""
    for val, name in ((k, "k"), (n, "n")):
        if isinstance(val, bool) or not isinstance(val, numbers.Real) or not math.isfinite(val):
            raise ValueError(f"{name}{label} must be a finite count, got {val!r}")
        if integer and float(val) != math.floor(float(val)):
            raise ValueError(f"{name}{label} must be a whole number of trials, got {val!r}")
    kf, nf = float(k), float(n)
    if nf < 0 or kf < 0:
        raise ValueError(f"counts{label} cannot be negative: k={k!r}, n={n!r}")
    if nf == 0:
        raise ValueError(f"n=0{label}: a rate needs at least one trial; record the cell as missing")
    if kf > nf:
        raise ValueError(f"k={k!r} exceeds n={n!r}{label}")
    return kf, nf


def check_count(n: object, name: str, *, minimum: int = 0) -> int:
    """A whole-number count at least `minimum`."""
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        if isinstance(n, numbers.Real) and math.isfinite(n) and float(n) == math.floor(float(n)):
            n = int(n)
        else:
            raise ValueError(f"{name} must be a whole number, got {n!r}")
    if int(n) < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {n!r}")
    return int(n)


def as_float_array(x: Iterable[float] | np.ndarray, name: str, *, min_len: int = 1) -> np.ndarray:
    """A 1-d finite float array of at least `min_len` entries."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} value(s), got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains a missing or non-finite value; the caller excludes "
                         "missing trials before computing (missing is not zero)")
    return arr


def as_binary_array(x: Iterable[float] | np.ndarray, name: str, *, min_len: int = 1) -> np.ndarray:
    """A 1-d array of 0/1 outcomes."""
    arr = as_float_array(x, name, min_len=min_len)
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError(f"{name} must contain only 0/1 outcomes")
    return arr


def as_unit_array(x: Iterable[float] | np.ndarray, name: str, *, min_len: int = 1) -> np.ndarray:
    """A 1-d array of values in [0, 1] (bounded scores, rates)."""
    arr = as_float_array(x, name, min_len=min_len)
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return arr
