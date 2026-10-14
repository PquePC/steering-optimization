"""m5.measures.s4 - `s4`, the minimum of a cell's live integrity terms (SPECIFICATION §7).

    s4 = min( s1                  coherent share (judged, concept-blind)
              1 - s2              non-degenerate share of the short screen generations
              s5 relative         steered capability pass rate / unsteered, capped at 1 )

**Minimum, not mean** (the old definition, kept): three different ways to be unusable, and passing
one must not compensate for failing another. Each term is put on one scale -- 1 is healthy -- with
its interval mapped the same way: s1's Wilson interval; 1 minus s2's (Wilson on its k/n when the
screen row has them); and s5's paired difference divided by the unsteered rate. `lo` and `hi` are
the minimum of the terms' lower and upper ends, so `lo` is the conservative bound of the minimum.

Computed from rows already measured, never by measuring: `s1` and `s5` from the run's `cells`
stream, `s2` from its `screen` stream (the screen readout), each the latest non-missing row for the
cell (preferring rows of the same split; for s2, the largest n). A term with no row is not live: it
is listed in `extra["missing_terms"]` and left out of the minimum; with no live term at all `s4`
is missing. `terms=` passes the Measured / Readout objects directly instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from m5.intervention import Cell
from m5.measures import common as C
from m5.results import Measured, Readout
from m5.stats import wilson

CODE = "s4"
TERMS = ("s1", "s2", "s5")


def _latest(rows: list[dict[str, Any]], code: str, cell_id: str, split: str | None) -> dict[str, Any] | None:
    hits = [r for r in rows
            if r.get("code") == code and r.get("cell") == cell_id and r.get("missing") is None]
    if not hits:
        return None
    same = [r for r in hits if split is not None and r.get("split") == split]
    pool = same or hits
    if code == "s2":
        best = max(int(r.get("n") or 0) for r in pool)
        pool = [r for r in pool if int(r.get("n") or 0) == best]
    return pool[-1]


def _as_row(x: Measured | Readout | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(x, (Measured, Readout)):
        return x.to_row()
    return dict(x)


def term_values(code: str, row: Mapping[str, Any], conf: float) -> dict[str, Any] | None:
    """One term on the healthy-is-1 scale, with its interval, from its row; None when unusable."""
    extra = row.get("extra") or {}
    if code == "s1":
        rate = row.get("rate")
        if rate is None:
            return None
        return {"value": float(rate), "lo": row.get("lo"), "hi": row.get("hi"), "n": row.get("n")}
    if code == "s2":
        value = row.get("value", row.get("rate"))
        if value is None:
            return None
        k, n = row.get("k"), row.get("n")
        lo = hi = None
        if k is not None and n:
            dlo, dhi = wilson(k, n, conf)
            lo, hi = 1.0 - dhi, 1.0 - dlo
        return {"value": 1.0 - float(value), "lo": lo, "hi": hi, "n": n}
    rate, null_k, null_n = row.get("rate"), row.get("null_k"), row.get("null_n")
    if rate is None or not null_k or not null_n:
        return None  # an unsteered rate of zero leaves the relative term undefined
    null_rate = null_k / null_n
    pd = extra.get("paired") or {}
    lo = hi = None
    if pd.get("lo") is not None:
        lo = min(1.0, (null_rate + pd["lo"]) / null_rate)
        hi = min(1.0, (null_rate + pd["hi"]) / null_rate)
    return {"value": min(1.0, float(rate) / null_rate), "lo": lo, "hi": hi, "n": row.get("n")}


def measure_s4(ctx: Any, cell: Cell | None, *, n: int | None = None, split: str,
               variant: str = "original",
               terms: Mapping[str, Measured | Readout | Mapping[str, Any]] | None = None) -> Measured:
    """`s4` at a cell from its measured terms (`n` is not used: s4 measures nothing itself)."""
    C.check_split(split)
    C.check_variant(CODE, variant)
    subj = C.subject(ctx, cell)
    conf = float(ctx.cfg["CONFIDENCE"])
    if terms is not None:
        unknown = sorted(set(terms) - set(TERMS))
        if unknown:
            raise C.MeasureError(f"s4 combines {TERMS}; unknown term(s) {unknown}")
        found = {c: _as_row(x) for c, x in terms.items()}
        found = {c: r for c, r in found.items() if r.get("missing") is None}
    else:
        if ctx.run is None:
            raise C.MeasureError("s4 reads measured rows from the run; this context has no run "
                                 "(pass terms= instead)")
        cells, screen = ctx.run.rows("cells"), ctx.run.rows("screen")
        found = {}
        for code in TERMS:
            r = _latest(screen if code == "s2" else cells, code, subj.cell_id,
                        None if code == "s2" else split)
            if r is not None:
                found[code] = r
    live = {c: v for c, v in ((c, term_values(c, r, conf)) for c, r in found.items()) if v is not None}
    missing_terms = [c for c in TERMS if c not in live]
    if not live:
        return C.missing_measure(ctx, CODE, subj, split, f"no live integrity term for this cell "
                                 f"(looked for {', '.join(TERMS)})",
                                 extra={"missing_terms": missing_terms})
    binding = min(live, key=lambda c: live[c]["value"])
    lows = [v["lo"] for v in live.values() if v["lo"] is not None]
    highs = [v["hi"] for v in live.values() if v["hi"] is not None]
    n_bind = live[binding]["n"]
    m = Measured(code=CODE, cell_id=subj.cell_id, split=split, n=int(n_bind or len(live)),
                 rate=live[binding]["value"], lo=min(lows) if lows else None,
                 hi=min(highs) if highs else None,
                 extra={"terms": live, "binding": binding, "missing_terms": missing_terms,
                        "rule": "min(s1, 1 - s2, s5 steered/unsteered capped at 1)"})
    return C.write_cell(ctx, m, subj=subj)
