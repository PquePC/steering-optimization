"""m5.results - the two result types every readout and measure returns.

The search, the gate, the setup and the report consume these and nothing else, so a new readout
plugs in by returning one of them. Both carry the cell they describe, the sample size, and an
explicit `missing` reason instead of a fake value when nothing could be computed.

    Readout   a screen-tier scalar (forward passes or short mechanical generations).
              Record-only codes (q1-q4) are Readouts too; `m5.codes` says they gate nothing.
    Measured  a generated-and-judged quantity with its counts, interval and null. Rates carry
              k and n; scores carry per-unit values so they can be bootstrapped by prompt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from m5.codes import code as _code

SPLITS = ("search", "confirm", "null", "control", "calibration", "audit")


@dataclass
class Readout:
    code: str
    cell_id: str
    value: float | None
    n: int
    k: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    missing: str | None = None

    def __post_init__(self) -> None:
        _code(self.code)  # refuses unknown and retired codes
        if self.value is None and self.missing is None:
            raise ValueError(f"{self.code}: a readout without a value must say why (missing=...)")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"{self.code}: non-finite value {self.value}")
        if self.k is not None and not 0 <= self.k <= self.n:
            raise ValueError(f"{self.code}: k={self.k} outside [0, n={self.n}]")

    def to_row(self) -> dict[str, Any]:
        return {"code": self.code, "cell": self.cell_id, "value": self.value, "n": self.n,
                "k": self.k, "extra": self.extra, "missing": self.missing}


@dataclass
class Measured:
    code: str
    cell_id: str
    split: str
    n: int
    k: int | None = None                 # rates
    rate: float | None = None
    lo: float | None = None
    hi: float | None = None
    values: list[float] | None = None    # scores, one per unit (prompt, item, trial)
    null_k: int | None = None            # the matching null, when the measure has one
    null_n: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    unit_ids: list[str] = field(default_factory=list)
    missing: str | None = None
    excluded: int = 0                    # units dropped as missing (failed judgments, unreadable)

    def __post_init__(self) -> None:
        _code(self.code)
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}; one of {SPLITS}")
        if self.k is not None:
            if not 0 <= self.k <= self.n:
                raise ValueError(f"{self.code}: k={self.k} outside [0, n={self.n}]")
            if self.n > 0 and self.rate is None:
                self.rate = self.k / self.n
        if self.null_k is not None and (self.null_n is None or not 0 <= self.null_k <= self.null_n):
            raise ValueError(f"{self.code}: null counts {self.null_k}/{self.null_n} are invalid")
        if self.n == 0 and self.missing is None:
            raise ValueError(f"{self.code}: n=0 must carry a missing reason")

    @property
    def mean(self) -> float | None:
        if self.values:
            return float(sum(self.values) / len(self.values))
        return self.rate

    def to_row(self) -> dict[str, Any]:
        return {"code": self.code, "cell": self.cell_id, "split": self.split, "n": self.n,
                "k": self.k, "rate": self.rate, "lo": self.lo, "hi": self.hi,
                "mean": self.mean, "null_k": self.null_k, "null_n": self.null_n,
                "extra": self.extra, "missing": self.missing, "excluded": self.excluded}
