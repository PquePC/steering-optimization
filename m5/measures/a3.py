"""m5.measures.a3 - `a3`, excess over controls: is the effect bigger than any intervention of its size?

Read et al. found that semantically unrelated contrastive pairs move evaluation awareness as much
as purpose-built ones, so a consequence (or an effect) means something only against the
distribution of matched control vectors (plan §2, §3.2, §3.4). `a3` takes one measured quantity
of the vector at a cell -- by default `a1`, verbalised evaluation awareness -- and the same quantity
for every control vector at the matched cell (the same schedule, layers, dose rho, operation and
positions), and reports the vector's percentile within the controls:

    percentile = (#controls below + half the #controls equal) / #controls

and whether it lies outside cfg["CONTROL_BAND"] (the report's *Specific* flag, §6.7: specific only
outside the 5-95% band). Below the band means the vector moves the quantity *less* than random
interventions of its size do; above it, more.

The vector's value and the controls' come from the run's rows (`cells`, and the setup's `setup`
stream, which profiles the control library on a rho grid), or are passed as `value=` and
`controls=` (one value per control vector). A row's value is its `mean` (for scores, and for `e8`,
whose rate is the excess), else its rate. Controls are the context's vectors of class "control".
Fewer controls than cfg["N_CONTRASTIVE_CONTROLS"] -- the plan's "at least 10 matched control
vectors" -- make a3 missing rather than a percentile of three numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from m5.codes import code as _code
from m5.intervention import Cell
from m5.measures import common as C
from m5.results import Measured

CODE = "a3"


def row_value(row: Mapping[str, Any]) -> float | None:
    """The comparable number of a cells / setup row: its mean, else its rate, else its value."""
    for k in ("mean", "rate", "value"):
        v = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
            return float(v)
    return None


def matched(cell: Cell, vector_id: str) -> Cell:
    return Cell(vector_id, cell.schedule, cell.layers, cell.rho, cell.op, cell.positions, cell.tags)


def _from_rows(rows: list[dict[str, Any]], quantity: str, cell_id: str, split: str) -> float | None:
    hits = [r for r in rows if r.get("code") == quantity and r.get("cell") == cell_id
            and r.get("missing") is None and row_value(r) is not None]
    if not hits:
        return None
    same = [r for r in hits if r.get("split") == split]
    return row_value((same or hits)[-1])


def percentile(value: float, controls: list[float]) -> float:
    c = np.asarray(controls, dtype=float)
    return float((np.sum(c < value) + 0.5 * np.sum(c == value)) / c.size)


def measure_a3(ctx: Any, cell: Cell | None, *, n: int | None = None, split: str,
               variant: str = "original", quantity: str = "a1", value: float | None = None,
               controls: Mapping[str, float] | None = None) -> Measured:
    """`a3` for `quantity` at a cell (`n` is not used: a3 reads values already measured)."""
    C.check_split(split)
    C.check_variant(CODE, variant)
    _code(quantity)
    if quantity == CODE:
        raise C.MeasureError("a3 compares another measure's value with the controls'")
    if cell is None:
        raise C.MeasureError("a3 places a vector among matched controls; the unsteered arm has none")
    subj = C.subject(ctx, cell)
    if subj.vector_class == "control":
        raise C.MeasureError(f"{cell.vector_id!r} is itself a control vector")
    rows: list[dict[str, Any]] = []
    if (value is None or controls is None) and ctx.run is not None:
        rows = ctx.run.rows("cells") + ctx.run.rows("setup")
    if value is None:
        value = _from_rows(rows, quantity, subj.cell_id, split)
    if value is None:
        return C.missing_measure(ctx, CODE, subj, split, f"no measured {quantity} for this cell")
    if controls is None:
        ids = sorted(v for v, cls in ctx.vector_classes.items() if cls == "control")
        found = {v: _from_rows(rows, quantity, matched(cell, v).cell_id(), split) for v in ids}
        controls = {v: x for v, x in found.items() if x is not None}
    vals = [float(x) for x in controls.values()]
    need = int(ctx.cfg["N_CONTRASTIVE_CONTROLS"])
    if len(vals) < need:
        return C.missing_measure(ctx, CODE, subj, split, f"{len(vals)} control vectors have {quantity} "
                                 f"at this matched cell; a3 needs at least {need} "
                                 "(N_CONTRASTIVE_CONTROLS)", extra={"quantity": quantity,
                                                                    "n_controls": len(vals)})
    lo_band, hi_band = (float(x) for x in ctx.cfg["CONTROL_BAND"])
    pct = percentile(float(value), vals)
    arr = np.asarray(vals)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    m = Measured(code=CODE, cell_id=subj.cell_id, split=split, n=len(vals), values=[pct],
                 extra={"quantity": quantity, "value": float(value), "percentile": pct,
                        "band": [lo_band, hi_band], "outside_band": not lo_band <= pct <= hi_band,
                        "direction": "above" if pct > hi_band else "below" if pct < lo_band else "within",
                        "controls": {C.short(k): float(v) for k, v in sorted(controls.items())},
                        "control_mean": float(arr.mean()), "control_sd": sd,
                        "z": (float(value) - float(arr.mean())) / sd if sd > 0 else None})
    return C.write_cell(ctx, m, subj=subj)
