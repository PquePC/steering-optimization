"""m5.context - the one object every pipeline step receives.

A step needs the config, the model, somewhere to write, the per-model profile (dose units, null
rates), the vectors it may steer with, the judges, and, for concept vectors, the concept word.
Passing one `Context` instead of seven arguments keeps every step's signature the same, which is
what lets the search call any readout or measure by its code.

Budget accounting lives here too. Every generation and judgment a step spends is counted against
the context, so a run can report its search effort (plan §6.5) and stop cleanly at a cap instead
of discovering the bill afterwards.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from m5.backend.base import Backend, derive_seed
from m5.config import Config
from m5.runio import RunDir


class BudgetExceeded(RuntimeError):
    """A step would spend past the run's cap. The run stops; nothing is recorded as measured."""


@dataclass
class Spend:
    generations: int = 0
    generated_tokens: int = 0
    forward_passes: int = 0
    judgments_local: int = 0
    judgments_external: int = 0

    def to_json(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class Context:
    """Everything a readout, measure or search step may use.

    `profile`, `store` and `judges` are typed loosely here because their classes live in modules
    built on top of this one (`m5.profile`, `m5.vectors`, `m5.judges`).
    """

    cfg: Config
    backend: Backend
    run: RunDir | None = None
    profile: Any = None                 # m5.profile.ModelProfile
    store: Any = None                   # m5.vectors.VectorStore
    judges: Any = None                  # m5.judges cascade
    concepts: dict[str, str] = field(default_factory=dict)       # vector_id -> concept word
    vector_classes: dict[str, str] = field(default_factory=dict)  # vector_id -> safety class
    max_generations: int | None = None
    spend: Spend = field(default_factory=Spend)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---------------------------------------------------------------- identity helpers

    def concept_of(self, vector_id: str) -> str | None:
        """The concept word of a concept vector; None for nameless vectors (naming drops out)."""
        return self.concepts.get(vector_id)

    def seed(self, *parts: object) -> int:
        return derive_seed(self.cfg["SEED"], *parts)

    # ---------------------------------------------------------------- budget

    def charge(self, *, generations: int = 0, tokens: int = 0, forward: int = 0,
               judged_local: int = 0, judged_external: int = 0) -> None:
        with self._lock:
            if self.max_generations is not None and \
                    self.spend.generations + generations > self.max_generations:
                raise BudgetExceeded(f"generation cap {self.max_generations} reached "
                                     f"({self.spend.generations} spent, {generations} requested)")
            self.spend.generations += generations
            self.spend.generated_tokens += tokens
            self.spend.forward_passes += forward
            self.spend.judgments_local += judged_local
            self.spend.judgments_external += judged_external

    # ---------------------------------------------------------------- writing

    def write(self, stream: str, row: dict[str, Any]) -> None:
        """Append to the run if there is one; a context without a run (tests) writes nothing."""
        if self.run is not None:
            self.run.append(stream, row)
