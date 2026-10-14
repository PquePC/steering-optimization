"""m5.analysis - Phase 0 (plan §10): what the M3 archive already says, re-read with M5's statistics.

No GPU, no new generations. Each module answers one Phase-0 item from the files old runs wrote
(never from their code -- `m5` imports nothing from `m2`/`m3`):

    m3_archive   the reader: an M3 run folder or an unzipped export, deduplicated, joined to its
                 boundaries, version drift handled explicitly (item 4's input)
    refit        per-layer psychometric fits of identification (`d2`'s M3 ancestor) against log
                 dose -- floor, lapse, overdispersion -- coupled by the layer chain; the ICC of
                 trials within prompts; the layers whose thresholds the data cannot pin, and why;
                 the winner's-curse simulation re-run with the measured ICC; the full-grid R*
                 (item 4)
    replay       the adaptive search replayed on the measured cells, through a Measurer that
                 answers from the archive and refuses anything off its grid (item 4)
    coupling     Mentor 1's question: how tightly naming tracks influence within dose bins (item 5)
    nearmiss     the identification near-misses on disk under decision 7's rules (item 6)
    truncation   the truncation curve behind TOKENS_INFLUENCE (item 3)
    headline     everything on disk about the headline cell, Silk L29 at 85% of dose_max (item 7)
    cli          `python -m m5 analyze-m3 --archive DIR --what ...`

**What leaves an analysis.** Results go to a new run (kind "analysis") under RUN_ROOT, in two
streams only: `analysis` (export class aggregate: numbers, settings and identifiers -- never text)
and `transcripts` (text, keyed by transcript id). `check_aggregate` enforces the first at write
time: a string is accepted only under a key named in `ID_KEYS`, never under a key that names
text, never with a line break and never long, so a model's words cannot reach the aggregate
stream by a renamed field or a forgotten one. The judge's `named` strings are model words too:
they go to `transcripts`, and aggregate rows refer to them by `name_id`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from m5.safety import Refused

# Keys whose string values are identifiers, settings or our own reason labels -- never a model's
# words. Anything else that holds a string is refused.
ID_KEYS = frozenset({
    "kind", "concept", "concepts", "run", "m3_config_hash", "model", "file", "drift", "cell", "cells",
    "cell_id", "outcome", "outcome_raw", "outcome_note", "source", "reason", "reasons", "status",
    "channel", "channels", "unit", "units", "transcript_id", "transcript_ids", "unit_ids",
    "dose_unit", "coupling", "rule", "rules", "template", "templates", "code", "codes", "split",
    "method", "flag", "flags", "missing", "note", "notes", "name_id", "name_ids", "bin_by",
    "stratum", "what", "mode", "estimand", "outcome_def", "effect_def", "floor_control",
    "cells_without_boundary", "cells_off_grid", "cells_without_row", "cells_without_trials",
    "stored_mismatch", "stored_mismatches", "error", "assumption", "assumptions", "tokenizer",
    "form", "forms", "class", "classes", "claims", "judge", "how", "decided_by", "stop",
    "request", "method_called", "best_layer_source", "stage", "skipped", "search_api",
    "not_judged", "by_judge", "errors", "tokenizers", "versus", "kinds", "degeneration_rule",
    "integrity_source", "effect_scale",
})

# Keys that name text. Refused whatever their value, so a renamed text field still trips.
TEXT_KEYS = frozenset({"text", "texts", "response", "responses", "named", "name", "names",
                       "prompt_text", "raw", "payload", "quote", "judge_raw", "reply",
                       "identified_as", "effect_forms"})

MAX_ID_CHARS = 200
MAX_KEY_CHARS = 80

# Fields `m5.runio.RunDir.append` stamps on every row. A row carrying one would overwrite the stamp
# (an archive's `config_hash` once hid its row from `RunDir.rows`, which filters on the run's hash).
STAMP_KEYS = frozenset({"schema", "config_hash", "ts", "attempt"})


def check_aggregate(row: Mapping[str, Any]) -> None:
    """Refuse an `analysis` row that could carry text (module docstring).

    A string value is accepted when its own key, or the key of the mapping that holds it, is in
    `ID_KEYS` (so `missing: {identify: "judge_failed: ..."}` is a mapping of identifiers); a key
    in `TEXT_KEYS` is refused anywhere; keys and identifiers are short and single-line. Also
    refuses non-finite numbers: JSON has no infinity, and a stored NaN reads as a number.
    Infinite thresholds are stored as None beside a flag that says why.
    """

    def walk(obj: Any, key: str, parent: str, path: str) -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                if not isinstance(k, (str, int)) or isinstance(k, bool):
                    raise Refused(f"analysis row: key {k!r} at {path} is not a string or integer")
                ks = str(k)
                if ks in TEXT_KEYS:
                    raise Refused(f"analysis row: field {path}.{ks} names text; text goes to the "
                                  "transcripts stream, never the aggregate one")
                if len(ks) > MAX_KEY_CHARS or "\n" in ks or "\r" in ks:
                    raise Refused(f"analysis row: key {ks[:40]!r}... at {path} is not an identifier")
                walk(v, ks if isinstance(k, str) else key, key, f"{path}.{ks}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, key, parent, f"{path}[{i}]")
        elif isinstance(obj, str):
            if key not in ID_KEYS and parent not in ID_KEYS:
                raise Refused(f"analysis row: {path} holds a string under {key!r}, which is not an "
                              "identifier field; the aggregate stream carries no text")
            if len(obj) > MAX_ID_CHARS or "\n" in obj or "\r" in obj:
                raise Refused(f"analysis row: {path} is too long or spans lines for an identifier")
        elif isinstance(obj, bool) or obj is None:
            return
        elif isinstance(obj, (int, float)):
            if isinstance(obj, float) and not math.isfinite(obj):
                raise Refused(f"analysis row: {path} is not finite; store None and say why")
        elif hasattr(obj, "item") and not hasattr(obj, "__len__"):
            walk(obj.item(), key, parent, path)  # numpy scalar
        else:
            raise Refused(f"analysis row: {path} has a value of type {type(obj).__name__}")

    walk(row, "", "", "row")


def finite_or_none(x: Any) -> float | None:
    """A float for a stored row: None for a missing or infinite value (say why beside it)."""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def name_id(name: str | None) -> str | None:
    """A stable id for a named string, so aggregate rows can count names without holding them."""
    if name is None:
        return None
    return "n" + hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:12]


def concept_name_id(concept: str, named: str | None) -> str | None:
    """The id of what a reply named, for one concept: the name as `m5.judges.matching` normalises
    it (case, accents, a leading article), so every module counts "Lemon" and "the lemon" alike."""
    if named is None:
        return None
    from m5.judges.matching import normalise
    return name_id(f"{concept}|{normalise(named)}")


class AnalysisSink:
    """Writes an analysis's rows into a run: aggregate rows checked by `check_aggregate` into
    `analysis`, texts into `transcripts`. Nothing else is written."""

    def __init__(self, run: Any) -> None:
        self.run = run
        self.rows_written = 0
        self.texts_written = 0

    def rows(self, rows: Sequence[Mapping[str, Any]]) -> int:
        for r in rows:
            clash = STAMP_KEYS & set(r)
            if clash:
                raise ValueError(f"analysis row {r.get('kind')!r} carries {sorted(clash)}, which the "
                                 "run stamps on every row; rename the field")
            check_aggregate(r)
        for r in rows:
            self.run.append("analysis", dict(r))
        self.rows_written += len(rows)
        return len(rows)

    def texts(self, items: Sequence[Mapping[str, Any]]) -> int:
        for it in items:
            if "id" not in it or "text" not in it:
                raise ValueError("a transcript row needs an id and a text")
            self.run.append("transcripts", dict(it))
        self.texts_written += len(items)
        return len(items)
