"""m5.measures - the measure and confirm tiers: generated, judged, paired with their nulls (plan §3.2).

One function per code, `measure_<code>(ctx, cell, *, n, split, variant="original") -> Measured`
(`m5.results`), each writing `trials` (outcomes and ids, never text), `transcripts` (text, keyed by
transcript id) and one `cells` row (`m5.measures.common` holds the shared machinery and its rules):

    e7   concept occurrence rate, judge-free                  open_ended (shared generation)
    e8   judged concept presence; the value is the excess     open_ended
    s1   coherence, concept-blind                             open_ended
    d10  spontaneous flags, regex then judge                  open_ended
    e11  behavioural trait score, per prompt                  e11 (reuses the open-ended generations)
    e12  off-target battery from logits                       e12
    d2   forced naming, generated                             d2
    d8   briefed self-report: claims, d-prime, declines       d8
    s4   minimum of the live integrity terms                  s4 (from rows already measured)
    s5   paired capability panel, McNemar                     s5
    a1   verbalised evaluation awareness                      a1
    a2   latent evaluation awareness (probe)                  a2
    a3   percentile among matched control vectors             a3 (from rows already measured)

`s7` (the refusal-token proxy) is not here: it needs the prompts file outside the repository and is
off by default (S7_ENABLED, decision 8).

A cell of None is the null arm: the measurement reports the shared unsteered arm itself (cell id
"unsteered", no null beside it), the same samples every steered cell is paired with. A zero-dose
cell (rho = 0) is the A/A check: unsteered, drawn with its own seeds, read against the null like
any cell. Every steered quantity sits beside its unsteered counterpart on the same prompts, and
every Measured with a null carries `null_k` / `null_n`.

Two entry points the search uses:

    measure_open_ended(ctx, cell, n, split)   {"e7", "e8", "s1", "d10"} from one generation
    measure_gate(ctx, cell, n, split)         the gating channels cfg["GATE"] names (decision 1):
                                              "noticing" -> d8, "naming" -> d2, "both" -> both

The measures that sample units (e7, e8, s1, d10, e11, d2, d8, s5, a1) take an optional `start`,
the first unit, so a sequential caller can extend a cell without repeating units: a unit's seed
depends on its index, never on how many were asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from m5.intervention import Cell
from m5.measures.a1 import measure_a1
from m5.measures.a2 import EAProbe, fit_mean_difference, measure_a2
from m5.measures.a3 import measure_a3
from m5.measures.common import JudgeFailureRateExceeded, MeasureError
from m5.measures.d2 import measure_d2
from m5.measures.d8 import measure_d8
from m5.measures.e11 import TRAITS, measure_e11
from m5.measures.e12 import OptionCollision, measure_e12
from m5.measures.open_ended import (measure_d10, measure_e7, measure_e8, measure_open_ended,
                                    measure_s1)
from m5.measures.s4 import measure_s4
from m5.measures.s5 import measure_s5
from m5.results import Measured

MEASURES: dict[str, Callable[..., Measured]] = {
    "e7": measure_e7,
    "e8": measure_e8,
    "e11": measure_e11,
    "e12": measure_e12,
    "d2": measure_d2,
    "d8": measure_d8,
    "d10": measure_d10,
    "s1": measure_s1,
    "s4": measure_s4,
    "s5": measure_s5,
    "a1": measure_a1,
    "a2": measure_a2,
    "a3": measure_a3,
}

GATE_CODES: dict[str, tuple[str, ...]] = {"noticing": ("d8",), "naming": ("d2",),
                                          "both": ("d8", "d2")}


def measure_gate(ctx: Any, cell: Cell | None, n: int, split: str, *, variant: str = "original",
                 start: int = 0) -> dict[str, Measured]:
    """The gating channels of decision 1, by cfg["GATE"]: noticing (d8), naming (d2) or both.

    The channel that does not gate is still measured and reported elsewhere; this is only what the
    adaptive search tests against its threshold. `n` is d8's trials per framing and d2's trials.
    """
    gate = ctx.cfg["GATE"]
    if gate not in GATE_CODES:
        raise MeasureError(f"GATE={gate!r} names no gating channel; one of {tuple(GATE_CODES)}")
    return {c: MEASURES[c](ctx, cell, n=n, split=split, variant=variant, start=start)
            for c in GATE_CODES[gate]}


__all__ = ["EAProbe", "GATE_CODES", "JudgeFailureRateExceeded", "MEASURES", "MeasureError",
           "OptionCollision", "TRAITS", "fit_mean_difference", "measure_a1", "measure_a2",
           "measure_a3", "measure_d10", "measure_d2", "measure_d8", "measure_e11", "measure_e12",
           "measure_e7", "measure_e8", "measure_gate", "measure_open_ended", "measure_s1",
           "measure_s4", "measure_s5"]
