"""m5.analysis.headline - everything on disk about the article's headline cell (Phase 0, item 7).

The article's headline window is Silk, L29 at 85% of that layer's `dose_max`. `RESULTS-SILK.md`
grades it *weak*, and the plan's reason to re-read it is sharper (§1): in the original sweep all
three unprefilled self-report trials there said they detected an injected thought, each naming a
different concept. At the one cell the article presents as avoiding steering awareness, the
model may have *noticed* the steering and failed only to *name* it. Before Phase 1 re-measures it
with noticing channels (`d7`, `d8`, `d9`), every channel M3 recorded there is re-read here.

`headline_cell` collects, for one cell: its boundary and norms, the run's stored cell row against
the recomputed one, and per channel every trial's outcomes and transcript id -- identify (`d2`'s
ancestor: matches, the named string's id, degeneracy), self_report (`d8`'s ancestor: claims, the
named string's id, matches, the run's own class), effect (influence, form, coherence, on-task),
explain and capability (the mechanical answer checks) -- plus the unsteered null rows of the same
prompts (mechanical only: M3 never judged its null arm) and the cell's judge-call record (counts
and failure labels).

**Text** -- the responses, the named strings, M3's judge replies for the cell -- is returned
apart, as transcript rows keyed by transcript id, and written only to the `transcripts` stream of
a new analysis run. Nothing is printed: the CLI prints the counts.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from m5.analysis import concept_name_id
from m5.analysis.m3_archive import M3Archive

# The headline cell (README; RESULTS-SILK.md; plan §1, §10 Phase 0 item 7).
HEADLINE_CONCEPT = "silk"
HEADLINE_LAYER = 29
HEADLINE_FRACTION = 0.85


class HeadlineError(LookupError):
    """The requested cell is not in the archive."""


@dataclass
class HeadlineResult:
    rows: list[dict[str, Any]]
    texts: list[dict[str, Any]]


def _trial_row(archive: M3Archive, t: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"transcript_id": t.transcript_id, "channel": t.channel, "unit": t.unit,
                           "trial": t.trial, "degenerate": t.degenerate,
                           "degeneration_rule": (None if not t.degeneration_rule
                                                 else t.degeneration_rule[:60]),
                           "empty": t.empty, "words": t.words,
                           "concept_mentions": t.concept_mentions,
                           "missing": dict(t.missing), "not_judged": list(t.not_judged)}
    if t.channel in ("identify", "self_report"):
        row["matches"] = t.matches
        row["name_id"] = concept_name_id(archive.concept, t.named)
    if t.channel == "self_report":
        row["claims"] = t.claims
        row["class"] = t.self_report_class
    if t.channel in ("effect", "explain"):
        row.update(influence=t.influence, form=t.form, coherence=t.coherence, on_task=t.on_task)
    if t.channel in ("explain", "capability"):
        row["capability_correct"] = t.capability_correct
    if t.repeat is not None:
        row["repeat"] = t.repeat
    return row


def headline_cell(archive: M3Archive, *, layer: int, fraction: float) -> HeadlineResult:
    """Every number and transcript id on disk about one cell (module docstring)."""
    cell = archive.cell(layer, fraction=fraction)
    if cell is None:
        raise HeadlineError(f"{archive.run_id}: no measured cell at L{layer}, fraction {fraction} "
                            f"(grid {archive.dose_fractions}, layers {archive.layers})")
    b = archive.boundaries.get(layer)
    nrm = archive.norms.get(layer)
    trials = archive.trials(cell)
    units = {t.unit for t in trials}
    nulls = [t for t in archive.null_trials if t.unit in units]
    rows: list[dict[str, Any]] = []
    head = {"concept": archive.concept, "cell": cell.cell_id, "layer": layer}
    rows.append({"kind": "headline_cell", **head, "fraction": cell.fraction, "dose": cell.dose,
                 "alpha": cell.alpha, "dose_max": cell.dose_max, "outcome": cell.outcome,
                 "boundary": None if b is None else b.to_row(),
                 "vec_norm": None if nrm is None else nrm.vec_norm,
                 "resid_norm": None if nrm is None else nrm.resid_norm,
                 "recomputed": cell.to_row(),
                 "stored_mismatch": list(cell.stored_mismatch),
                 "stored": _stored_numbers(cell.stored)})
    for ch in ("identify", "self_report", "effect", "explain", "capability"):
        ts = [t for t in trials if t.channel == ch]
        if not ts:
            continue
        summary: dict[str, Any] = {"kind": "headline_channel", **head, "channel": ch, "n": len(ts),
                                   "degenerate": sum(1 for t in ts if t.degenerate),
                                   "empty": sum(1 for t in ts if t.empty),
                                   "judge_failed": sum(1 for t in ts if t.missing)}
        if ch in ("identify", "self_report"):
            hits = [t for t in ts if t.matches is not None]
            summary.update(matches_k=sum(1 for t in hits if t.matches), matches_n=len(hits),
                           name_ids=dict(Counter(concept_name_id(archive.concept, t.named)
                                                 for t in hits if t.named is not None)))
        if ch == "self_report":
            summary.update(claims=dict(Counter(str(t.claims) for t in ts if t.claims is not None)),
                           classes=dict(Counter(str(t.self_report_class) for t in ts
                                                if t.self_report_class is not None)))
        if ch in ("effect", "explain"):
            infl = [t.influence for t in ts if t.influence is not None]
            summary.update(influence=infl, forms=dict(Counter(str(t.form) for t in ts if t.form)),
                           coherence=[t.coherence for t in ts if t.coherence is not None])
        if ch in ("explain", "capability"):
            summary.update(correct_k=sum(1 for t in ts if t.capability_correct),
                           correct_n=sum(1 for t in ts if t.capability_correct is not None))
        rows.append(summary)
        rows.extend({"kind": "headline_trial", **head, **_trial_row(archive, t)} for t in ts)
    rows.extend({"kind": "headline_null", **head, **_trial_row(archive, t)} for t in nulls)
    calls = archive.judge_calls(cell)
    rows.append({"kind": "headline_judge_calls", **head, "n": len(calls),
                 "ok": sum(1 for r in calls if r.get("ok") is True),
                 "by_judge": dict(Counter(str(r.get("judge")) for r in calls)),
                 "errors": dict(Counter(str(r.get("error")).split(":", 1)[0][:40]
                                        for r in calls if r.get("error"))),
                 "rejudged": any(r.get("judged_by") for r in calls)})
    texts: list[dict[str, Any]] = []
    for t in trials + nulls:
        texts.append({"id": t.transcript_id, "text": archive.text(t.transcript_id) or "",
                      "kind": "m3_response", "concept": archive.concept, "channel": t.channel,
                      "unit": t.unit, "cell": cell.cell_id if t.steered else "null",
                      "prompt_text": archive.prompt_text(t.unit)})
        if t.named is not None:
            texts.append({"id": f"{t.transcript_id}#named", "text": t.named, "kind": "named_string",
                          "name_id": concept_name_id(archive.concept, t.named),
                          "transcript_id": t.transcript_id})
    for r in calls:
        if r.get("raw") is not None:
            texts.append({"id": f"m3judge/{archive.run_id}/{cell.cell_id}/{r.get('channel')}/"
                                f"{r.get('unit')}/{r.get('judge')}",
                          "text": str(r.get("raw")), "kind": "m3_judge_reply",
                          "judge": r.get("judge"), "ok": r.get("ok"), "error": r.get("error")})
    return HeadlineResult(rows=rows, texts=texts)


def _numbers(v: Any) -> Any:
    """Numbers, booleans and None, at any depth; strings and lists of them are dropped."""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            y = _numbers(x)
            if y is not _DROP:
                out[str(k)] = y
        return out
    if isinstance(v, (list, tuple)):
        return _DROP
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v if not isinstance(v, float) or math.isfinite(v) else None
    return _DROP


_DROP = object()


def _stored_numbers(stored: dict[str, Any] | None) -> dict[str, Any] | None:
    """The run's own cell row, numbers only: its `identified_as` list holds named strings and
    `effect_forms` judge labels, and neither belongs in the aggregate stream."""
    if stored is None:
        return None
    return _numbers({k: v for k, v in stored.items() if k not in ("identified_as", "effect_forms")})
