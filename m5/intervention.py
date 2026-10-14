"""m5.intervention - what a steering intervention *is*, as data.

An `Intervention` says which stored vector is applied, by which operation, at which layers with
which multipliers, and at which token positions. It carries no tensors: the backend resolves
`vector_id` to the registered per-layer vectors. Being plain frozen data, an intervention can be
hashed, logged, compared and replayed, and the same object drives the real model and the
simulator.

## Layers

`layer L` is the residual stream at the **output of decoder block L**, 0-based: below the last
layer, the point Hugging Face reports as `hidden_states[L + 1]` (in transformers 5.x the last entry
of `hidden_states` is the state after the final norm, not the last block's raw output). Steering
at L edits that output; reading at L reads it. One convention for writing and reading, so a vector
extracted at L is applied at L.

## Operations (on the residual h at a steered position; v the layer's vector, v^ = v/||v||)

    add       h <- h + alpha * v
    sub       h <- h - alpha * v          inhibition, as the system cards apply it
    project   h <- h - alpha * (h . v^) v^  directional ablation; alpha = 1 removes the component
    clamp     h <- h + (c - h . v^) v^    sets the component along v^ to c = `clamp_value`

`alpha` is never negative: the sign lives in the operation, so a log line says what was done.

## Positions

Token indices refer to the rendered prompt exactly as `Backend.tokenize` returns it (its BOS
included as index 0 when the template writes one). The backend maps them onto padded batches.

    from      prompt tokens with index >= start, and every generated token
              (live steering from a marker onward -- the Lindsey/Macar arrangement)
    span      prompt tokens with start <= index < end, never generated tokens
              (past-only arms d6/d7: the briefing is steered, the question is not, but the
               question attends to the steered keys and values -- Pearson et al.'s design;
               localisation d9: one segment)
    generated generated tokens only
    all       every prompt token after the first (never the BOS sink) and every generated token

## Cells

A `Cell` is one point of the search space: a vector, a layer schedule, an operation, a position
policy, and a dose. The dose is stored in the unit the search moves in (`rho`, plan §4.3); the
alphas it implies are computed by `m5.dose` from the model profile and recorded beside it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

OPS = ("add", "sub", "project", "clamp")
DOSE_SIGNIFICANT_DIGITS = 6


def canonical_dose(x: float) -> float:
    """Round a dose to DOSE_SIGNIFICANT_DIGITS significant digits, the form every key uses."""
    if x == 0 or not math.isfinite(x):
        return float(x)
    return float(f"{x:.{DOSE_SIGNIFICANT_DIGITS}g}")
POSITION_POLICIES = ("from", "span", "generated", "all")
SCHEDULES = ("single", "strided", "all", "gaussian")


@dataclass(frozen=True)
class Positions:
    policy: str
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in POSITION_POLICIES:
            raise ValueError(f"unknown position policy {self.policy!r}; one of {POSITION_POLICIES}")
        if self.policy == "from":
            if self.start is None or self.start < 1:
                raise ValueError("'from' needs start >= 1 (index 0 is the BOS sink)")
            if self.end is not None:
                raise ValueError("'from' takes no end")
        elif self.policy == "span":
            if self.start is None or self.end is None:
                raise ValueError("'span' needs start and end")
            if not 1 <= self.start < self.end:
                raise ValueError(f"'span' needs 1 <= start < end, got [{self.start}, {self.end})")
        elif self.start is not None or self.end is not None:
            raise ValueError(f"'{self.policy}' takes no start or end")

    @property
    def steers_generated(self) -> bool:
        return self.policy in ("from", "generated", "all")

    def prompt_mask(self, n_prompt_tokens: int) -> list[bool]:
        """Which prompt positions are steered, for a prompt of this many tokens."""
        n = n_prompt_tokens
        if self.policy == "from":
            return [i >= self.start for i in range(n)]
        if self.policy == "span":
            if self.end > n:
                raise ValueError(f"span end {self.end} is beyond the prompt ({n} tokens)")
            return [self.start <= i < self.end for i in range(n)]
        if self.policy == "generated":
            return [False] * n
        return [i >= 1 for i in range(n)]  # all

    def to_json(self) -> dict[str, Any]:
        return {"policy": self.policy, "start": self.start, "end": self.end}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Positions":
        return cls(d["policy"], d.get("start"), d.get("end"))


@dataclass(frozen=True)
class LayerTerm:
    layer: int
    alpha: float

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError(f"layer must be >= 0, got {self.layer}")
        if not (math.isfinite(self.alpha) and self.alpha >= 0):
            raise ValueError(f"alpha must be finite and >= 0 (the sign is the operation), got {self.alpha}")


@dataclass(frozen=True)
class Intervention:
    vector_id: str
    op: str
    terms: tuple[LayerTerm, ...]
    positions: Positions
    clamp_value: float | None = None

    def __post_init__(self) -> None:
        if not self.vector_id:
            raise ValueError("vector_id is required")
        if self.op not in OPS:
            raise ValueError(f"unknown operation {self.op!r}; one of {OPS}")
        if not self.terms:
            raise ValueError("an intervention needs at least one layer")
        layers = [t.layer for t in self.terms]
        if len(set(layers)) != len(layers):
            raise ValueError(f"duplicate layers in {layers}")
        if list(layers) != sorted(layers):
            object.__setattr__(self, "terms", tuple(sorted(self.terms, key=lambda t: t.layer)))
        if self.op == "clamp" and self.clamp_value is None:
            raise ValueError("clamp needs clamp_value")
        if self.op != "clamp" and self.clamp_value is not None:
            raise ValueError("clamp_value is only for clamp")

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(t.layer for t in self.terms)

    def alpha_at(self, layer: int) -> float | None:
        for t in self.terms:
            if t.layer == layer:
                return t.alpha
        return None

    def with_positions(self, positions: Positions) -> "Intervention":
        return Intervention(self.vector_id, self.op, self.terms, positions, self.clamp_value)

    def scaled(self, factor: float) -> "Intervention":
        if factor < 0:
            raise ValueError("scale factor must be >= 0; flip the operation instead")
        return Intervention(self.vector_id, self.op,
                            tuple(LayerTerm(t.layer, t.alpha * factor) for t in self.terms),
                            self.positions, self.clamp_value)

    def is_null(self) -> bool:
        """True when it cannot change anything (every alpha zero for add/sub/project)."""
        return self.op != "clamp" and all(t.alpha == 0 for t in self.terms)

    def to_json(self) -> dict[str, Any]:
        return {"vector_id": self.vector_id, "op": self.op,
                "terms": [[t.layer, t.alpha] for t in self.terms],
                "positions": self.positions.to_json(), "clamp_value": self.clamp_value}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Intervention":
        return cls(d["vector_id"], d["op"], tuple(LayerTerm(int(l), float(a)) for l, a in d["terms"]),
                   Positions.from_json(d["positions"]), d.get("clamp_value"))

    def key(self) -> str:
        """Stable 16-hex identity, for caches and logs."""
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Cell:
    """One point of the search space. `rho` is the dose in the search's unit (plan §4.3)."""

    vector_id: str
    schedule: str
    layers: tuple[int, ...]
    rho: float
    op: str = "add"
    positions: str = "from_marker"
    tags: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {self.schedule!r}; one of {SCHEDULES}")
        if not self.layers:
            raise ValueError("a cell needs at least one layer")
        if self.schedule == "single" and len(self.layers) != 1:
            raise ValueError("a single-layer cell has exactly one layer")
        if not (math.isfinite(self.rho) and self.rho >= 0):
            raise ValueError(f"rho must be finite and >= 0, got {self.rho}")
        if self.op not in OPS:
            raise ValueError(f"unknown operation {self.op!r}")
        # Canonical dose: a search that reaches 0.44999999999999996 and a grid that says 0.45 must
        # name the same cell, or a resume re-measures it and an analysis splits it (old defect G13).
        object.__setattr__(self, "rho", canonical_dose(self.rho))

    @property
    def layer(self) -> int:
        """The anchor layer: the only one for single-layer cells, the first otherwise."""
        return self.layers[0]

    @property
    def log_rho(self) -> float:
        return math.log(self.rho) if self.rho > 0 else -math.inf

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["layers"] = list(self.layers)
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Cell":
        return cls(d["vector_id"], d["schedule"], tuple(d["layers"]), float(d["rho"]),
                   d.get("op", "add"), d.get("positions", "from_marker"), tuple(d.get("tags", ())))

    def cell_id(self) -> str:
        """Readable and unique: vector|schedule|L29|rho=0.35|add|from_marker"""
        lay = "L" + "-".join(str(l) for l in self.layers) if len(self.layers) <= 4 else \
            f"L{self.layers[0]}..{self.layers[-1]}x{len(self.layers)}"
        return f"{self.vector_id}|{self.schedule}|{lay}|rho={self.rho:.6g}|{self.op}|{self.positions}"
