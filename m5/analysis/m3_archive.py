"""m5.analysis.m3_archive - read what an M3 (or M4-launch) run wrote, without M3's code.

M5 imports nothing from `m2`/`m3`; Phase 0 (plan §10) still has to re-read their archive -- the
462 cells of Garlic run 2, Silk and Wrists. This module is that reader, written from the on-disk
format specification (a reference reader re-derived every field of every `cells.jsonl` row from
the transcripts with zero mismatches; its rules are followed here). What it does, each rule from
a hazard the specification names:

**JSONL.** Split on b"\\n" only: responses carry raw U+2028/U+2029/U+0085 (`ensure_ascii=False`),
and `splitlines()` once broke one row into two. One torn *last* line (a run killed mid-append,
with or without its newline) is tolerated and counted; an unparseable line anywhere else is
corruption and raises `ArchiveCorrupt` -- never silently skipped -- unless the caller asks for
`allow_corrupt`, in which case each skipped line is counted in the read report.

**Duplicates.** A resumed run re-measures an unfinished cell and appends a second complete
battery. Rows are deduplicated on their natural key -- responses `(layer, dose, channel, unit)`,
judge calls `+ judge`, cells `(layer, dose)`, null rows `(channel, unit, repeat)` -- keeping the
greatest `attempt` stamp and, on a tie or when the stamp is absent (every archived run predates
it), the **last** copy in file order: the later battery is the one `cells.jsonl` summarises, the
earlier one is from a pass that crashed before its cell row (`tools/rescore` keeps the first; the
specification recommends the last, and so does this reader). `"unset"` ranks with "absent".

**Joins.** A cell row has no dose fraction, no `dose_max` and no boundary outcome: they are
joined from `boundaries.jsonl` on `layer` (from `summary.json["boundaries"]` when the layer's row
is missing from the file), and the fraction `dose / dose_max` is snapped to
`summary.dose_fractions`. A cell whose layer has no boundary row keeps `fraction = None` and is
listed in the report: it was measured, but nothing places it on the grid.

**Missing is not zero.** A failed judge call is `judged[kind] = null` with a `judge_error`; such a
trial carries `None` for the verdict and a reason, and leaves every denominator. A row that was
never judged (`JUDGE_ENABLED=0`, capability rows) is "not judged", which is different again.
Rows in `unjudged_transcripts.jsonl` are counted and never mixed into judged counts.

**Version drift** (specification §12) is handled explicitly and recorded as drift codes in the
report: no `attempt` stamps (`no_attempt_stamps`), no `n_empty` on cell rows (recomputed from the
trials, `no_n_empty`), no `identification_excluding_unreadable` (recomputed), and the older boundary
vocabulary, in which a ceiling- or bracket-bound layer reads `"ok"` (`outcome_vocabulary_pre_c04ab0f`:
inferred back from `dose_max >= max_reachable_dose` and no failing dose above it).

**What is never read.** `vectors/` (the concept-vector cache: a reusable steering artefact, never
read and never shipped), weights and arrays (`*.pt`, `*.safetensors`, `*.npy`, `*.npz`), lossy
digests (`read_this.md`), quarantine files and logs. Every file goes through one opener that
refuses those paths, so a later edit cannot start reading them by accident.

**Text stays apart.** `ArchiveTrial` holds outcomes and identifiers only; response text is kept
in a separate table reached by transcript id (`M3Archive.text`), so an analysis row built from
trials cannot carry a transcript by construction.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from m5 import safety

# The files a run folder (or an unzipped export) holds that this reader uses. Anything else in the
# folder is ignored; nothing outside this list is ever opened.
SUMMARY = "summary.json"
NORMS = "norms.jsonl"
BOUNDARIES = "boundaries.jsonl"
RESPONSES = "responses_transcripts.jsonl"
NULLS = "null_transcripts.jsonl"
JUDGE_CALLS = "judge_calls.jsonl"
CELLS = "cells.jsonl"
UNJUDGED = "unjudged_transcripts.jsonl"
ARCHIVE_FILES = (SUMMARY, NORMS, BOUNDARIES, RESPONSES, NULLS, JUDGE_CALLS, CELLS, UNJUDGED)

# Never opened, whatever a caller asks: vector caches, activations, weights and arrays.
FORBIDDEN_DIRS = frozenset({"vectors", "activations"})
FORBIDDEN_SUFFIXES = frozenset({".pt", ".pth", ".safetensors", ".npy", ".npz", ".bin", ".ckpt"})

CHANNELS = ("identify", "self_report", "effect", "explain", "capability")

# The boundary outcomes of the current vocabulary (m3/sweep.py:831-853). Before c04ab0f
# (2026-08-20 13:16) only ok / incoherent_at_floor / probes_exhausted / unreachable existed, and a
# layer bound by the alpha ceiling or by the bracket top read "ok".
OUTCOMES = ("ok", "ceiling_limited", "bracket_limited", "incoherent_at_floor", "probes_exhausted",
            "unreachable")

# Subject models of the archive and their depth (decoder blocks), for the replay's layer band:
# the depths in m5.config.MODELS, the one registry of model facts.

# The judge's `form` labels (m3/judge.py:344) and the self-report classes (m3/judge.py:633-658).
# M3's parser took a label's value to the end of its line without checking it, so a reply could
# put free judge prose in `form`; anything outside the vocabulary reads "unrecognised" here and
# never travels as text.
M3_FORMS = ("absent", "trace", "thematic", "dominant", "fixated")
M3_SELF_REPORT_CLASSES = ("degenerate", "identified", "confabulated", "leaked", "clean_denial",
                          "unclear_leaked", "unclear")

_TRIAL = re.compile(r"^trial_(\d+)$")
_RUN_NAME = re.compile(r"^(?:export_|m2_)?(?P<concept>.+)_(?P<hash>[0-9a-f]{12})$")

# Snapping a cell's dose to the grid: dose = round(dose_max * f, 6), so the exact grid point is
# within rounding of 6 decimals; a dose further than this from every grid point is off the grid
# (an explicit-CELLS run, a freerun cell at an arbitrary dose).
_SNAP_ABS = 2e-6
_SNAP_REL = 1e-4


class ArchiveError(RuntimeError):
    """The path is not an M3 run, or the run cannot be read as one."""


class ArchiveCorrupt(ArchiveError):
    """A JSONL file is malformed somewhere other than its last line."""


# =====================================================================================
# Reading files
# =====================================================================================

def _guard_path(path: Path, root: Path) -> Path:
    """Refuse anything this reader must never open (see the module docstring)."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        raise safety.Refused(f"{path} is outside the archive {root}") from None
    if any(part in FORBIDDEN_DIRS for part in rel.parts[:-1]) or \
            path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name not in ARCHIVE_FILES:
        raise safety.Refused(f"{rel} is not read: the M3 archive reader opens only {ARCHIVE_FILES}; "
                             "vector caches, activations and weights are never read (CLAUDE.md)")
    return path


@dataclass
class FileReport:
    """What reading one file found."""

    name: str
    present: bool = False
    read: bool = False             # judge_calls.jsonl is large and read only when a cell needs it
    rows: int = 0
    torn_last_line: bool = False
    corrupt_lines: list[int] = field(default_factory=list)
    duplicates_dropped: int = 0
    foreign_rows: int = 0

    def to_row(self) -> dict[str, Any]:
        return {"file": self.name, "present": self.present,
                "rows": self.rows if self.read else None,
                "torn_last_line": self.torn_last_line, "corrupt_lines": len(self.corrupt_lines),
                "duplicates_dropped": self.duplicates_dropped, "foreign_rows": self.foreign_rows}


def iter_m3_jsonl(path: Path, report: FileReport, *, allow_corrupt: bool = False
                  ) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of an M3 JSONL file, one per line, streaming.

    Lines are split on b"\\n" only (binary iteration never splits on U+2028/2029/0085). Blank
    lines are skipped. A line that does not parse is held until the next non-blank line arrives:
    if none does, it was the torn tail of a killed append and is counted (`torn_last_line`);
    otherwise it is interior corruption and raises `ArchiveCorrupt`, or with `allow_corrupt` is
    skipped and its line number recorded. A line that parses to something other than an object
    is corruption wherever it sits (a foreign writer, not a torn append).
    """
    report.present = True
    report.read = True
    pending: int | None = None
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            if pending is not None:
                if not allow_corrupt:
                    raise ArchiveCorrupt(f"{path.name}: line {pending} is not JSON and is not the "
                                         "last line: corruption rather than a torn append")
                report.corrupt_lines.append(pending)
                pending = None
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                pending = lineno
                continue
            if not isinstance(obj, dict):
                if not allow_corrupt:
                    raise ArchiveCorrupt(f"{path.name}: line {lineno} is JSON but not an object")
                report.corrupt_lines.append(lineno)
                continue
            report.rows += 1
            yield obj
    if pending is not None:
        report.torn_last_line = True


def read_m3_jsonl(path: Path, report: FileReport | None = None, *,
                  allow_corrupt: bool = False) -> list[dict[str, Any]]:
    """All rows of an M3 JSONL file (see `iter_m3_jsonl`)."""
    rep = report or FileReport(Path(path).name)
    return list(iter_m3_jsonl(Path(path), rep, allow_corrupt=allow_corrupt))


# =====================================================================================
# Small parsers
# =====================================================================================

def parse_trial(unit: Any) -> int | None:
    """The trial number of an identify/self_report row, from its unit `trial_<N>`; None otherwise.

    It is stored nowhere else: M3 rows have no trial field (spec §3.1)."""
    m = _TRIAL.match(str(unit)) if unit is not None else None
    return int(m.group(1)) if m else None


def dose_key(dose: Any) -> float | None:
    """The join key for a dose: cells, responses and judge rows all store round(dose, 6)."""
    if dose is None:
        return None
    return round(float(dose), 6)


def _attempt_rank(row: Mapping[str, Any]) -> str:
    """Rows with no stamp (all archived runs) or "unset" rank lowest; ISO stamps sort in time."""
    a = row.get("attempt")
    if a is None or a == "unset":
        return ""
    return str(a)


def dedupe(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]
           ) -> tuple[list[dict[str, Any]], int]:
    """Keep one row per key: the greatest `attempt`, and on a tie the LAST in file order.

    Returns (rows in first-seen key order, number of rows dropped). The tie rule is the format
    specification's recommendation (§11): the later battery is the one `cells.jsonl` summarises.
    """
    kept: dict[tuple, dict[str, Any]] = {}
    order: list[tuple] = []
    dropped = 0
    for r in rows:
        k = tuple(dose_key(r.get(f)) if f == "dose" else r.get(f) for f in key_fields)
        if k not in kept:
            kept[k] = dict(r)
            order.append(k)
            continue
        dropped += 1
        if _attempt_rank(r) >= _attempt_rank(kept[k]):
            kept[k] = dict(r)
    return [kept[k] for k in order], dropped


def normalise_outcome(row: Mapping[str, Any]) -> tuple[str, str | None]:
    """The boundary outcome in today's vocabulary, and a note when it had to be inferred.

    Before c04ab0f a layer whose search never failed above its passing dose read "ok"; such a
    layer is `ceiling_limited` (dose_max at max_reachable_dose) or `bracket_limited` (dose_max at
    the bracket top). Today's code never writes "ok" without a failing dose, so the inference is
    safe on every era. An outcome outside the vocabulary reads "unknown" with a note; the raw
    value is kept beside it (`Boundary.outcome_raw`).
    """
    raw = row.get("outcome")
    dose_max = row.get("dose_max")
    failing = row.get("lowest_failing_dose")
    if raw == "ok" and dose_max is not None and failing is None:
        reach = row.get("max_reachable_dose")
        bracket = row.get("bracket") or ()
        if reach is not None and float(dose_max) >= float(reach) * (1 - 1e-6):
            return "ceiling_limited", "inferred_from_ok_pre_c04ab0f"
        if len(bracket) == 2 and float(dose_max) >= float(bracket[1]) * (1 - 1e-6):
            return "bracket_limited", "inferred_from_ok_pre_c04ab0f"
        return "ok", "ok_without_failing_dose"
    if raw in OUTCOMES:
        return str(raw), None
    if raw is None:
        return "unknown", "no_outcome_field"
    return "unknown", "unrecognised_outcome"


def snap_fraction(dose: float, dose_max: float | None, fractions: Sequence[float]
                  ) -> tuple[float | None, float | None]:
    """(fraction, |dose - dose_max*fraction|) for the grid point this dose was made from, or
    (None, None) when it was made from none (or there is no dose_max to place it)."""
    if dose_max is None or not fractions or not dose_max > 0:
        return None, None
    best = min(fractions, key=lambda f: abs(float(dose) - float(dose_max) * float(f)))
    err = abs(float(dose) - float(dose_max) * float(best))
    if err <= _SNAP_ABS + _SNAP_REL * abs(float(dose)):
        return float(best), err
    return None, None


# =====================================================================================
# The data model
# =====================================================================================

@dataclass(frozen=True)
class Boundary:
    """One layer's Phase-1 boundary (what `dose_max` is and how it was found)."""

    layer: int
    dose_max: float | None
    outcome: str
    outcome_raw: str | None
    outcome_note: str | None
    max_reachable_dose: float | None
    lowest_failing_dose: float | None
    probes_used: int | None
    source: str                    # "boundaries.jsonl" or "summary.json"
    probes: tuple[dict[str, Any], ...] = ()   # each probe's stage and numbers (no text)

    def to_row(self) -> dict[str, Any]:
        return {"layer": self.layer, "dose_max": self.dose_max, "outcome": self.outcome,
                "outcome_raw": self.outcome_raw, "outcome_note": self.outcome_note,
                "max_reachable_dose": self.max_reachable_dose,
                "lowest_failing_dose": self.lowest_failing_dose, "probes_used": self.probes_used,
                "source": self.source, "probes": [dict(p) for p in self.probes]}


_PROBE_FIELDS = ("dose", "alpha", "coherence", "coherence_n", "degeneration", "on_task", "answered",
                 "good", "sane", "unjudged")


def _probe(p: Any) -> dict[str, Any] | None:
    """A boundary probe's numbers and its stage (ladder / bisect), nothing else."""
    if not isinstance(p, Mapping):
        return None
    out: dict[str, Any] = {"stage": "bisect" if p.get("stage") == "bisect" else "ladder"}
    for k in _PROBE_FIELDS:
        v = p.get(k)
        if v is None or isinstance(v, bool) or (isinstance(v, (int, float)) and math.isfinite(v)):
            out[k] = v
    return out


@dataclass(frozen=True)
class LayerNorm:
    """||v_L|| and ||h_L|| behind every dose of a layer (norms.jsonl)."""

    layer: int
    vec_norm: float | None
    resid_norm: float | None
    max_reachable_dose: float | None


@dataclass(frozen=True)
class ArchiveTrial:
    """One battery response's outcomes -- never its text (that is `M3Archive.text(transcript_id)`).

    Verdict fields are None when there is no verdict; `missing` says why for each judge kind that
    was attempted and failed ("judge_failed: <error label>") or returned a malformed verdict, and
    `not_judged` lists the kinds that were never asked (JUDGE_ENABLED=0, a coherence verdict
    beyond the first N_COHERENCE effect rows). Neither is a "no".
    """

    transcript_id: str
    channel: str
    unit: str
    trial: int | None              # identify / self_report: parsed from `trial_<N>`
    layer: int | None              # None on the null arm
    dose: float | None
    alpha: float | None
    steered: bool
    repeat: int | None             # null arm only
    degenerate: bool | None
    degeneration_rule: str | None  # the label before the first ":" of degeneration_reason
    empty: bool
    words: int | None
    concept_mentions: int | None   # as the run's own code counted them (see spec §7.3)
    named: str | None = None       # identify / self_report: the judge's `named` (model words)
    matches: bool | None = None    # identify / self_report: the judge's strict boolean
    claims: str | None = None      # self_report: YES / NO / UNCLEAR
    influence: float | None = None  # effect / explain: 0-10, relative to the unsteered reply
    form: str | None = None
    coherence: float | None = None  # effect (first N_COHERENCE) / explain: 0-10, concept-blind
    on_task: bool | None = None
    capability_correct: bool | None = None  # explain / capability: mechanical, the run's rule
    self_report_class: str | None = None
    missing: Mapping[str, str] = field(default_factory=dict)
    not_judged: tuple[str, ...] = ()

    @property
    def readable(self) -> bool:
        return not self.empty


@dataclass
class ArchiveCell:
    """One measured (layer, dose), joined to its layer's boundary, with M3's per-cell quantities
    recomputed from the deduplicated trials (spec §8). Rates are (k, n) with n the trials that
    have a verdict; `None` lists are never zero-filled."""

    concept: str
    layer: int
    dose: float
    alpha: float | None
    fraction: float | None = None
    fraction_error: float | None = None
    dose_max: float | None = None
    outcome: str | None = None
    has_boundary: bool = False
    has_cell_row: bool = False
    has_trials: bool = False
    # identification (d2's M3 ancestor): judge `matches` over identify rows with a verdict
    ident_k: int = 0
    ident_n: int = 0
    ident_failed: int = 0          # judge calls that failed (excluded)
    ident_not_judged: int = 0      # never judged (excluded)
    ident_degenerate: int = 0      # over all identify rows, verdict or not
    ident_clean_k: int = 0
    ident_clean_n: int = 0
    ident_readable_k: int = 0
    ident_readable_n: int = 0
    # influence (effect channel): per prompt, verdict present and response non-empty
    influence: tuple[float, ...] = ()
    influence_units: tuple[str, ...] = ()
    influence_failed: int = 0
    coherence: tuple[float, ...] = ()
    on_task_k: int = 0
    on_task_n: int = 0
    # self_report (d8's M3 ancestor): claims YES, and the judge's matches, over rows with a verdict
    sr_claims_k: int = 0
    sr_names_k: int = 0
    sr_n: int = 0
    sr_failed: int = 0
    # mechanical capability checks, as the run's own accept lists scored them
    capability_k: int = 0
    capability_n: int = 0
    explain_correct_k: int = 0
    explain_n: int = 0
    # bookkeeping
    n_rows: int = 0
    n_empty: int = 0
    judge_errors: int = 0
    n_unjudged_rows: int = 0
    stored: dict[str, Any] | None = None
    stored_mismatch: tuple[str, ...] = ()
    # "trials" (recomputed from responses_transcripts) or "stored_row": an export whose
    # transcript files were withheld keeps only cells.jsonl, and then the run's own aggregates are
    # used for what they hold (identification, self-report, capability, mean influence) and
    # nothing per prompt or per trial exists (no ICC, no effect curve, no near-misses).
    source: str = "trials"
    influence_stored: tuple[float, int] | None = None

    @property
    def key(self) -> tuple[int, float]:
        return (self.layer, dose_key(self.dose))  # type: ignore[return-value]

    @property
    def cell_id(self) -> str:
        return f"L{self.layer}@{self.dose:.6f}"

    @property
    def identification(self) -> tuple[int, int] | None:
        return (self.ident_k, self.ident_n) if self.ident_n else None

    @property
    def influence_mean(self) -> float | None:
        if self.influence:
            return float(sum(self.influence) / len(self.influence))
        return self.influence_stored[0] if self.influence_stored else None

    @property
    def influence_n(self) -> int:
        if self.influence:
            return len(self.influence)
        return self.influence_stored[1] if self.influence_stored else 0

    def to_row(self) -> dict[str, Any]:
        """Numbers and identifiers only (an `analysis` row body)."""
        return {"concept": self.concept, "cell": self.cell_id, "layer": self.layer,
                "dose": self.dose, "alpha": self.alpha, "fraction": self.fraction,
                "dose_max": self.dose_max, "outcome": self.outcome,
                "has_boundary": self.has_boundary, "has_cell_row": self.has_cell_row,
                "has_trials": self.has_trials,
                "d2_k": self.ident_k, "d2_n": self.ident_n, "d2_failed": self.ident_failed,
                "d2_not_judged": self.ident_not_judged, "d2_degenerate": self.ident_degenerate,
                "d2_clean_k": self.ident_clean_k, "d2_clean_n": self.ident_clean_n,
                "influence_mean": self.influence_mean, "influence_n": self.influence_n,
                "source": self.source,
                "influence_failed": self.influence_failed,
                "self_report_claims_k": self.sr_claims_k, "self_report_names_k": self.sr_names_k,
                "self_report_n": self.sr_n, "capability_k": self.capability_k,
                "capability_n": self.capability_n, "n_rows": self.n_rows, "n_empty": self.n_empty,
                "judge_errors": self.judge_errors, "n_unjudged_rows": self.n_unjudged_rows,
                "stored_mismatch": list(self.stored_mismatch)}


@dataclass
class ReadReport:
    """Everything the reader found that a reader of the numbers should know."""

    files: dict[str, FileReport] = field(default_factory=dict)
    drift: list[str] = field(default_factory=list)
    cells_without_boundary: list[str] = field(default_factory=list)
    cells_off_grid: list[str] = field(default_factory=list)
    cells_without_row: list[str] = field(default_factory=list)
    cells_without_trials: list[str] = field(default_factory=list)
    stored_mismatches: list[str] = field(default_factory=list)
    unjudged_rows: int = 0
    unjudged_boundary_rows: int = 0
    rejudged: bool = False

    def note(self, code: str) -> None:
        if code not in self.drift:
            self.drift.append(code)

    def to_row(self) -> dict[str, Any]:
        return {"files": [f.to_row() for f in self.files.values()], "drift": list(self.drift),
                "cells_without_boundary": list(self.cells_without_boundary),
                "cells_off_grid": list(self.cells_off_grid),
                "cells_without_row": list(self.cells_without_row),
                "cells_without_trials": list(self.cells_without_trials),
                "stored_mismatches": list(self.stored_mismatches),
                "unjudged_rows": self.unjudged_rows,
                "unjudged_boundary_rows": self.unjudged_boundary_rows, "rejudged": self.rejudged}


@dataclass
class M3Archive:
    """One M3 run, read: its settings, boundaries, norms, cells, trials and (apart) its texts."""

    path: Path
    concept: str
    config_hash: str | None
    settings: dict[str, Any]
    summary_present: bool
    dose_fractions: tuple[float, ...]
    model: str | None
    boundaries: dict[int, Boundary]
    norms: dict[int, LayerNorm]
    cells: list[ArchiveCell]
    null_trials: list[ArchiveTrial]
    report: ReadReport
    _trials: dict[tuple[int, float], list[ArchiveTrial]] = field(repr=False)
    _texts: dict[str, str] = field(repr=False)
    _prompts: dict[str, str] = field(repr=False)
    _allow_corrupt: bool = field(default=False, repr=False)

    # ---------------------------------------------------------------- identity

    @property
    def run_id(self) -> str:
        return f"{self.concept.lower()}_{self.config_hash or 'nohash'}"

    @property
    def vector_id(self) -> str:
        """The id M5 cells use for this run's concept vector (the replay's cells carry it)."""
        return f"m3:{self.concept.lower()}"

    @property
    def n_layers(self) -> int | None:
        from m5.config import MODELS
        name = str(self.model) if self.model else None
        return MODELS[name]["n_layers"] if name in MODELS else None

    @property
    def layers(self) -> list[int]:
        return sorted({c.layer for c in self.cells})

    # ---------------------------------------------------------------- lookups

    def cell(self, layer: int, dose: float | None = None, *, fraction: float | None = None
             ) -> ArchiveCell | None:
        """The cell at (layer, dose), or at (layer, grid fraction); None when not measured."""
        if (dose is None) == (fraction is None):
            raise ValueError("give exactly one of dose and fraction")
        for c in self.cells:
            if c.layer != int(layer):
                continue
            if dose is not None and dose_key(c.dose) == dose_key(dose):
                return c
            if fraction is not None and c.fraction is not None and \
                    abs(c.fraction - float(fraction)) <= 1e-9 + 1e-6 * abs(float(fraction)):
                return c
        return None

    def trials(self, cell: ArchiveCell | None = None, channel: str | None = None
               ) -> list[ArchiveTrial]:
        """Steered trials, of one cell and/or one channel, in battery order."""
        if channel is not None and channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r}; one of {CHANNELS}")
        if cell is not None:
            pool = self._trials.get(cell.key, [])
        else:
            pool = [t for c in self.cells for t in self._trials.get(c.key, [])]
        return [t for t in pool if channel is None or t.channel == channel]

    def nulls(self, channel: str | None = None) -> list[ArchiveTrial]:
        return [t for t in self.null_trials if channel is None or t.channel == channel]

    def text(self, transcript_id: str) -> str | None:
        """The response text of one trial. Text never travels inside trials or analysis rows."""
        return self._texts.get(transcript_id)

    def prompt_text(self, unit: str) -> str | None:
        """M3's own prompt text for an effect/explain/capability unit (None for identify)."""
        return self._prompts.get(unit)

    def judge_calls(self, cell: ArchiveCell | None = None) -> list[dict[str, Any]]:
        """Deduplicated judge-call rows (of one cell). They embed transcripts and M3's judge
        prompts: internal material, for re-reading on this machine only."""
        rep = FileReport(JUDGE_CALLS)
        p = self.path / JUDGE_CALLS
        if not p.is_file():
            return []
        _guard_path(p, self.path)
        want = cell.key if cell is not None else None
        rows = []
        for r in iter_m3_jsonl(p, rep, allow_corrupt=self._allow_corrupt):
            if want is not None and (r.get("layer"), dose_key(r.get("dose"))) != want:
                continue
            h, c = r.get("config_hash"), r.get("concept")
            if (h is not None and self.config_hash is not None and h != self.config_hash) or \
                    (c is not None and str(c).lower() != self.concept.lower()):
                continue   # copied in from another run (see load_archive)
            rows.append(r)
        out, _ = dedupe(rows, ("layer", "dose", "channel", "unit", "judge"))
        return out


# =====================================================================================
# Building trials and cells
# =====================================================================================

def transcript_id(run_id: str, row: Mapping[str, Any]) -> str:
    """Deterministic id of a battery or null row (M3 rows carry no id of their own)."""
    if row.get("steered") is False or row.get("layer") is None:
        return f"m3/{run_id}/null/{row.get('channel')}/{row.get('unit')}/r{row.get('repeat')}"
    return (f"m3/{run_id}/L{int(row['layer'])}@{float(row['dose']):.6f}/"
            f"{row.get('channel')}/{row.get('unit')}")


def _verdict(row: Mapping[str, Any], kind: str) -> tuple[dict[str, Any] | None, str | None]:
    """(verdict, None), (None, "judge_failed: ...") or (None, "not_judged")."""
    judged = row.get("judged")
    if not isinstance(judged, Mapping) or kind not in judged:
        return None, "not_judged"
    v = judged[kind]
    if v is None:
        err = (row.get("judge_error") or {}).get(kind) if isinstance(row.get("judge_error"), Mapping) else None
        label = str(err).split(":", 1)[0].strip()[:60] if err else "no_error_recorded"
        return None, f"judge_failed: {label}"
    if not isinstance(v, Mapping):
        return None, "malformed_verdict"
    return dict(v), None


def _bool(x: Any) -> bool | None:
    return x if isinstance(x, bool) else None


def _label(x: Any, vocabulary: Sequence[str]) -> str | None:
    """A judge label in M3's vocabulary (first word, case and punctuation aside), else
    "unrecognised"; None when absent."""
    if x is None:
        return None
    words = re.findall(r"[a-z_]+", str(x).lower())
    return words[0] if words and words[0] in vocabulary else "unrecognised"


def _score(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    v = float(x)
    return v if math.isfinite(v) and 0.0 <= v <= 10.0 else None


def make_trial(row: Mapping[str, Any], run_id: str) -> ArchiveTrial:
    """One trial from a responses/null row: outcomes, identifiers and missing reasons, no text."""
    channel = str(row.get("channel"))
    missing: dict[str, str] = {}
    not_judged: list[str] = []
    out: dict[str, Any] = {}

    def take(kind: str) -> dict[str, Any] | None:
        v, why = _verdict(row, kind)
        if why == "not_judged":
            not_judged.append(kind)
        elif why:
            missing[kind] = why
        return v

    if channel == "identify":
        v = take("identify")
        if v is not None:
            m = _bool(v.get("matches"))
            if m is None:
                missing["identify"] = "malformed_verdict"
            else:
                out["matches"] = m
                out["named"] = None if v.get("named") is None else str(v.get("named"))
    elif channel == "self_report":
        v = take("self_report")
        if v is not None:
            m = _bool(v.get("matches"))
            claims = v.get("claims")
            if m is None or claims not in ("YES", "NO", "UNCLEAR"):
                missing["self_report"] = "malformed_verdict"
            else:
                out.update(matches=m, claims=claims,
                           named=None if v.get("named") is None else str(v.get("named")))
        out["self_report_class"] = _label(row.get("self_report_class"), M3_SELF_REPORT_CLASSES)
    if channel in ("effect", "explain"):
        v = take("effect")
        if v is not None:
            infl = _score(v.get("influence"))
            if infl is None:
                missing["effect"] = "malformed_verdict"
            else:
                out["influence"] = infl
                out["form"] = _label(v.get("form"), M3_FORMS)
        c = take("coherence")
        if c is not None:
            coh = _score(c.get("coherence"))
            if coh is None:
                missing["coherence"] = "malformed_verdict"
            else:
                out["coherence"] = coh
                out["on_task"] = _bool(c.get("on_task"))
    if channel in ("explain", "capability"):
        out["capability_correct"] = _bool(row.get("capability_correct"))
    reason = row.get("degeneration_reason")
    response = row.get("response")
    empty = row.get("empty")
    if not isinstance(empty, bool):
        empty = not str(response or "").split()
    layer = row.get("layer")
    return ArchiveTrial(
        transcript_id=transcript_id(run_id, row), channel=channel, unit=str(row.get("unit")),
        trial=parse_trial(row.get("unit")), layer=None if layer is None else int(layer),
        dose=dose_key(row.get("dose")), alpha=None if row.get("alpha") is None else float(row["alpha"]),
        steered=bool(row.get("steered", layer is not None)),
        repeat=None if row.get("repeat") is None else int(row["repeat"]),
        degenerate=_bool(row.get("degenerate")),
        degeneration_rule=str(reason).split(":", 1)[0] if reason else None,
        empty=bool(empty), words=row.get("words") if isinstance(row.get("words"), int) else None,
        concept_mentions=row.get("concept_mentions") if isinstance(row.get("concept_mentions"), int) else None,
        missing=missing, not_judged=tuple(not_judged), **out)


def _rate(stored: Mapping[str, Any] | None, name: str) -> tuple[int, int] | None:
    v = (stored or {}).get(name)
    if not isinstance(v, Mapping):
        return None
    return int(v.get("count", -1)), int(v.get("n", -1))


def _aggregate(cell: ArchiveCell, trials: Sequence[ArchiveTrial], texts: Mapping[str, str]) -> None:
    """Fill M3's per-cell quantities from the trials (spec §8, HEAD definitions)."""
    cell.has_trials = bool(trials)
    cell.n_rows = len(trials)
    ident = [t for t in trials if t.channel == "identify"]
    hits = [t for t in ident if t.matches is not None]
    clean = [t for t in hits if not t.degenerate]
    readable = [t for t in clean if not t.empty]
    cell.ident_k, cell.ident_n = sum(t.matches for t in hits), len(hits)
    cell.ident_failed = sum(1 for t in ident if "identify" in t.missing)
    cell.ident_not_judged = sum(1 for t in ident if "identify" in t.not_judged)
    cell.ident_degenerate = sum(1 for t in ident if t.degenerate)
    cell.ident_clean_k, cell.ident_clean_n = sum(t.matches for t in clean), len(clean)
    cell.ident_readable_k, cell.ident_readable_n = sum(t.matches for t in readable), len(readable)
    eff = [t for t in trials if t.channel == "effect"]
    infl = [t for t in eff if t.influence is not None and not t.empty]
    cell.influence = tuple(t.influence for t in infl)  # type: ignore[misc]
    cell.influence_units = tuple(t.unit for t in infl)
    cell.influence_failed = sum(1 for t in eff if "effect" in t.missing)
    coh = [t for t in eff if t.coherence is not None and not t.empty]
    cell.coherence = tuple(t.coherence for t in coh)  # type: ignore[misc]
    cell.on_task_k, cell.on_task_n = sum(1 for t in coh if t.on_task), len(coh)
    sr = [t for t in trials if t.channel == "self_report" and t.claims is not None]
    cell.sr_n = len(sr)
    cell.sr_claims_k = sum(1 for t in sr if t.claims == "YES")
    cell.sr_names_k = sum(1 for t in sr if t.matches)
    cell.sr_failed = sum(1 for t in trials if t.channel == "self_report" and "self_report" in t.missing)
    caps = [t for t in trials if t.channel == "capability"]
    cell.capability_k, cell.capability_n = sum(1 for t in caps if t.capability_correct), len(caps)
    exp = [t for t in trials if t.channel == "explain"]
    cell.explain_correct_k, cell.explain_n = sum(1 for t in exp if t.capability_correct), len(exp)
    cell.n_empty = sum(1 for t in trials if not str(texts.get(t.transcript_id) or "").strip())
    cell.judge_errors = sum(len(t.missing) for t in trials)


def _from_stored(cell: ArchiveCell) -> None:
    """Fill what the run's own cell row holds, for a cell whose battery is not on disk."""
    s = cell.stored or {}
    got = False
    for name, (k_attr, n_attr) in (("identification", ("ident_k", "ident_n")),
                                   ("identification_excluding_degenerate", ("ident_clean_k", "ident_clean_n")),
                                   ("capability", ("capability_k", "capability_n")),
                                   ("self_report", ("sr_claims_k", "sr_n"))):
        r = _rate(s, name)
        if r is not None and r[1] > 0 and 0 <= r[0] <= r[1]:
            setattr(cell, k_attr, r[0])
            setattr(cell, n_attr, r[1])
            got = True
    names = _rate(s, "self_report_names_concept")
    if names is not None and names[1] == cell.sr_n:
        cell.sr_names_k = names[0]
    if isinstance(s.get("identify_degenerate_n"), int):
        cell.ident_degenerate = s["identify_degenerate_n"]
    eff = s.get("effectiveness")
    if isinstance(eff, Mapping) and isinstance(eff.get("mean"), (int, float)) and int(eff.get("n", 0)) > 0:
        cell.influence_stored = (float(eff["mean"]), int(eff["n"]))
        got = True
    if isinstance(s.get("n_empty"), int):
        cell.n_empty = s["n_empty"]
    if got:
        cell.source = "stored_row"


def _check_stored(cell: ArchiveCell, report: ReadReport) -> None:
    """Compare the recomputed quantities with the run's own cells.jsonl row; list mismatches.

    A mismatch is evidence, not an error: a re-judged export whose cells were not re-aggregated,
    or a code version whose definition differed (effectiveness before a9aaf01 did not exclude
    empty responses). The trials are what the analyses use; the stored row is checked against them.
    """
    s = cell.stored
    if s is None or not cell.has_trials:
        return
    bad = []
    pairs = [("identification", (cell.ident_k, cell.ident_n)),
             ("identification_excluding_degenerate", (cell.ident_clean_k, cell.ident_clean_n)),
             ("identification_excluding_unreadable", (cell.ident_readable_k, cell.ident_readable_n)),
             ("capability", (cell.capability_k, cell.capability_n)),
             ("self_report", (cell.sr_claims_k, cell.sr_n)),
             ("self_report_names_concept", (cell.sr_names_k, cell.sr_n))]
    for name, mine in pairs:
        if name not in s:
            continue
        theirs = _rate(s, name)
        if theirs is None:
            if mine[1] != 0:
                bad.append(name)
        elif theirs != mine:
            bad.append(name)
    eff = s.get("effectiveness")
    if "effectiveness" in s:
        if isinstance(eff, Mapping):
            m = cell.influence_mean
            if m is None or int(eff.get("n", -1)) != len(cell.influence) or \
                    not math.isclose(float(eff.get("mean", math.nan)), m, rel_tol=1e-9, abs_tol=1e-9):
                bad.append("effectiveness")
        elif cell.influence:
            bad.append("effectiveness")
    if "n_empty" in s and s.get("n_empty") != cell.n_empty:
        bad.append("n_empty")
    cell.stored_mismatch = tuple(bad)
    if bad:
        report.stored_mismatches.append(cell.cell_id)


# =====================================================================================
# Loading
# =====================================================================================

def is_run_dir(path: Path) -> bool:
    """A folder is a run when it holds summary.json or cells.jsonl (M3 also leaves stray, empty
    folders named like runs; spec §2.1)."""
    return (path / SUMMARY).is_file() or (path / CELLS).is_file()


def find_run_dirs(path: str | Path) -> list[Path]:
    """The run folders at `path`: the folder itself, or its immediate subfolders that are runs."""
    p = safety.assert_outside_repo(path, what="M3 archive")
    if p.suffix.lower() == ".zip":
        raise ArchiveError(f"{p} is a zip: unzip it into a directory of its own first (its entries "
                           "sit at the zip root with no folder prefix)")
    if not p.is_dir():
        raise ArchiveError(f"{p} is not a directory")
    if is_run_dir(p):
        return [p]
    runs = [c for c in sorted(p.iterdir()) if c.is_dir() and c.name not in FORBIDDEN_DIRS
            and not c.name.startswith((".", "_")) and is_run_dir(c)]
    if not runs:
        raise ArchiveError(f"{p} holds no M3 run (a folder with {SUMMARY} or {CELLS})")
    return runs


def load_archives(path: str | Path, *, allow_corrupt: bool = False) -> list[M3Archive]:
    """Every run at `path` (one run folder, or a folder of run folders)."""
    return [load_archive(p, allow_corrupt=allow_corrupt) for p in find_run_dirs(path)]


def load_archive(path: str | Path, *, allow_corrupt: bool = False) -> M3Archive:
    """Read one M3 run folder (or an unzipped export). Refuses a path inside the repository and
    a harmful-arm concept before reading any row."""
    root = safety.assert_outside_repo(path, what="M3 archive")
    if not root.is_dir() or not is_run_dir(root):
        raise ArchiveError(f"{root} is not an M3 run folder (no {SUMMARY} or {CELLS})")
    report = ReadReport()

    def rows_of(name: str) -> list[dict[str, Any]]:
        rep = report.files.setdefault(name, FileReport(name))
        p = root / name
        if not p.is_file():
            return []
        _guard_path(p, root)
        return read_m3_jsonl(p, rep, allow_corrupt=allow_corrupt)

    summary: dict[str, Any] | None = None
    sp = root / SUMMARY
    report.files[SUMMARY] = FileReport(SUMMARY)
    if sp.is_file():
        _guard_path(sp, root)
        try:
            summary = json.loads(sp.read_bytes().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ArchiveCorrupt(f"{SUMMARY} does not parse (it is written atomically, so this is "
                                 f"not a torn write): {exc}") from None
        if not isinstance(summary, dict):
            raise ArchiveCorrupt(f"{SUMMARY} is not a JSON object")
        report.files[SUMMARY].present = True
        report.files[SUMMARY].read = True
        report.files[SUMMARY].rows = 1
    else:
        report.note("no_summary")

    # -------- identity: concept and config hash, from the summary, else the cell rows, else the
    # folder name. Every name the run goes by is checked against the harmful arm before a single
    # transcript is read; the cell rows hold aggregates only, so reading them first is safe.
    m = _RUN_NAME.match(root.name)
    names = [(summary or {}).get("concept"), m.group("concept") if m else None]
    for nm in names:
        if nm:
            safety.check_concept(str(nm))
    cell_rows = rows_of(CELLS)
    row_concept = next((r.get("concept") for r in cell_rows if r.get("concept")), None)
    if row_concept:
        safety.check_concept(str(row_concept))
    concept = names[0] or row_concept or names[1]
    if not concept:
        raise ArchiveError(f"{root}: cannot tell which concept this run measured")
    concept = str(concept)
    resp_rows = rows_of(RESPONSES)
    config_hash = (summary or {}).get("config_hash") or next(
        (r.get("config_hash") for r in cell_rows + resp_rows if r.get("config_hash")), None) or \
        (m.group("hash") if m else None)

    def foreign(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
        """Rows stamped with another run's hash or concept were copied in; count and drop them."""
        keep = []
        for r in rows:
            h, c = r.get("config_hash"), r.get("concept")
            if (h is not None and config_hash is not None and h != config_hash) or \
                    (c is not None and str(c).lower() != concept.lower()):
                report.files[name].foreign_rows += 1
                continue
            keep.append(r)
        return keep

    cell_rows = foreign(cell_rows, CELLS)
    resp_rows = foreign(resp_rows, RESPONSES)
    null_rows = foreign(rows_of(NULLS), NULLS)
    boundary_rows = foreign(rows_of(BOUNDARIES), BOUNDARIES)
    norm_rows = foreign(rows_of(NORMS), NORMS)
    unjudged_rows = foreign(rows_of(UNJUDGED), UNJUDGED)
    report.files.setdefault(JUDGE_CALLS, FileReport(JUDGE_CALLS)).present = (root / JUDGE_CALLS).is_file()

    settings = dict((summary or {}).get("config") or {})
    if summary is not None and summary.get("rejudged") is not None:
        report.rejudged = True
        report.note("rejudged_export")
    if summary is not None and not report.files[JUDGE_CALLS].present and settings.get("JUDGE_ENABLED", 1):
        report.note("no_judge_calls_file")
    if settings.get("JUDGE_ENABLED", 1) in (0, "0", False):
        report.note("judging_disabled")
    if settings.get("CELLS"):
        report.note("explicit_cells_no_grid")

    # -------- version drift that changes how rows are read
    stamped = cell_rows + resp_rows + null_rows
    if stamped and all(r.get("attempt") is None for r in stamped):
        report.note("no_attempt_stamps")
    if cell_rows and any("n_empty" not in r for r in cell_rows):
        report.note("no_n_empty")
    if cell_rows and any("identification_excluding_unreadable" not in r for r in cell_rows):
        report.note("no_identification_excluding_unreadable")
    if not norm_rows:
        report.note("no_norms")

    # -------- dedupe
    resp_rows, d = dedupe(resp_rows, ("layer", "dose", "channel", "unit"))
    report.files[RESPONSES].duplicates_dropped = d
    cell_rows, d = dedupe(cell_rows, ("layer", "dose"))
    report.files[CELLS].duplicates_dropped = d
    null_rows, d = dedupe(null_rows, ("channel", "unit", "repeat"))
    report.files[NULLS].duplicates_dropped = d
    boundary_rows, d = dedupe(boundary_rows, ("layer",))
    report.files[BOUNDARIES].duplicates_dropped = d
    norm_rows, d = dedupe(norm_rows, ("layer",))
    report.files[NORMS].duplicates_dropped = d

    # -------- boundaries (file first, summary as fallback) and norms
    boundaries: dict[int, Boundary] = {}

    def add_boundary(r: Mapping[str, Any], source: str) -> None:
        if r.get("layer") is None:
            return
        outcome, note = normalise_outcome(r)
        if note == "inferred_from_ok_pre_c04ab0f":
            report.note("outcome_vocabulary_pre_c04ab0f")
        if outcome == "unknown":
            report.note("unknown_boundary_outcome")
        dm = r.get("dose_max")
        boundaries[int(r["layer"])] = Boundary(
            layer=int(r["layer"]), dose_max=None if dm is None else float(dm), outcome=outcome,
            outcome_raw=None if r.get("outcome") is None else str(r.get("outcome"))[:40],
            outcome_note=note,
            max_reachable_dose=None if r.get("max_reachable_dose") is None else float(r["max_reachable_dose"]),
            lowest_failing_dose=None if r.get("lowest_failing_dose") is None else float(r["lowest_failing_dose"]),
            probes_used=r.get("probes_used") if isinstance(r.get("probes_used"), int) else None,
            source=source,
            probes=tuple(x for x in (_probe(p) for p in (r.get("probes") or [])) if x is not None))

    for r in boundary_rows:
        add_boundary(r, BOUNDARIES)
    for r in (summary or {}).get("boundaries") or []:
        if isinstance(r, Mapping) and r.get("layer") is not None and int(r["layer"]) not in boundaries:
            add_boundary(r, SUMMARY)
            report.note("boundary_from_summary")
    norms = {int(r["layer"]): LayerNorm(
        layer=int(r["layer"]),
        vec_norm=None if r.get("vec_norm") is None else float(r["vec_norm"]),
        resid_norm=None if r.get("resid_norm") is None else float(r["resid_norm"]),
        max_reachable_dose=None if r.get("max_reachable_dose") is None else float(r["max_reachable_dose"]))
        for r in norm_rows if r.get("layer") is not None}

    # -------- the dose grid
    fractions = tuple(float(f) for f in ((summary or {}).get("dose_fractions") or settings.get("DOSE_FRACTIONS") or ()))
    if not fractions and boundaries:
        # A run that died before its summary: the grid is recovered from the cells themselves.
        seen = sorted({round(float(r["dose"]) / boundaries[int(r["layer"])].dose_max, 3)
                       for r in cell_rows if r.get("layer") is not None and int(r["layer"]) in boundaries
                       and boundaries[int(r["layer"])].dose_max})
        fractions = tuple(seen)
        if fractions:
            report.note("fractions_inferred_from_cells")

    run_id = f"{concept.lower()}_{config_hash or 'nohash'}"

    # -------- trials and texts
    texts: dict[str, str] = {}
    prompts: dict[str, str] = {}
    by_cell: dict[tuple[int, float], list[ArchiveTrial]] = {}
    for r in resp_rows:
        if r.get("layer") is None or r.get("dose") is None or r.get("channel") not in CHANNELS:
            continue
        t = make_trial(r, run_id)
        texts[t.transcript_id] = str(r.get("response") or "")
        if r.get("prompt_text") and t.channel in ("effect", "explain", "capability"):
            prompts.setdefault(t.unit, str(r["prompt_text"]))
        by_cell.setdefault((int(r["layer"]), dose_key(r["dose"])), []).append(t)  # type: ignore[arg-type]
    null_trials = []
    for r in null_rows:
        if r.get("channel") not in CHANNELS:
            continue
        t = make_trial(dict(r, steered=False, layer=None, dose=None), run_id)
        texts[t.transcript_id] = str(r.get("response") or "")
        null_trials.append(t)

    unjudged_by_cell: Counter = Counter()
    for r in unjudged_rows:
        if r.get("phase") == "BOUNDARY":
            report.unjudged_boundary_rows += 1
        elif r.get("layer") is not None and r.get("dose") is not None:
            report.unjudged_rows += 1
            unjudged_by_cell[(int(r["layer"]), dose_key(r["dose"]))] += 1
    if unjudged_rows:
        report.note("unjudged_transcripts_present")

    # -------- cells: the cells.jsonl rows, plus batteries whose cell row was never written
    stored_by_key = {(int(r["layer"]), dose_key(r["dose"])): r for r in cell_rows
                     if r.get("layer") is not None and r.get("dose") is not None}
    keys = sorted(set(stored_by_key) | set(by_cell))
    cells: list[ArchiveCell] = []
    for key in keys:
        layer, dose = key
        stored = stored_by_key.get(key)
        trials = by_cell.get(key, [])
        alpha = (stored or {}).get("alpha")
        if alpha is None and trials:
            alpha = trials[0].alpha
        c = ArchiveCell(concept=concept, layer=layer, dose=float(dose),
                        alpha=None if alpha is None else float(alpha),
                        has_cell_row=stored is not None,
                        stored=None if stored is None else {k: v for k, v in stored.items()
                                                            if k not in ("concept", "config_hash", "ts", "attempt")})
        b = boundaries.get(layer)
        if b is not None:
            c.has_boundary = True
            c.dose_max = b.dose_max
            c.outcome = b.outcome
            c.fraction, c.fraction_error = snap_fraction(c.dose, b.dose_max, fractions)
            if c.fraction is None:
                report.cells_off_grid.append(c.cell_id)
        else:
            report.cells_without_boundary.append(c.cell_id)
        _aggregate(c, trials, texts)
        c.n_unjudged_rows = unjudged_by_cell.get(key, 0)
        if stored is None:
            report.cells_without_row.append(c.cell_id)
        if not trials:
            report.cells_without_trials.append(c.cell_id)
            _from_stored(c)
            if c.source == "stored_row":
                report.note("cells_from_stored_rows_only")
        _check_stored(c, report)
        cells.append(c)

    if resp_rows and not any(t.channel == "self_report" for ts in by_cell.values() for t in ts):
        report.note("no_self_report")  # N_SELF_REPORT=0 (the M4 launch): d8 has no ancestor here

    return M3Archive(path=root, concept=concept, config_hash=config_hash, settings=settings,
                     summary_present=summary is not None, dose_fractions=tuple(sorted(fractions)),
                     model=None if settings.get("MODEL") is None else str(settings.get("MODEL")),
                     boundaries=boundaries, norms=norms, cells=cells, null_trials=null_trials,
                     report=report, _trials=by_cell, _texts=texts, _prompts=prompts,
                     _allow_corrupt=allow_corrupt)


def archive_rows(archive: M3Archive) -> list[dict[str, Any]]:
    """The reader's own `analysis` rows: what was read, what was dropped and why, per run."""
    rep = archive.report
    rows: list[dict[str, Any]] = [{
        "kind": "archive_read", "concept": archive.concept, "run": archive.run_id,
        "m3_config_hash": archive.config_hash, "summary_present": archive.summary_present,
        "model": archive.model, "dose_fractions": list(archive.dose_fractions),
        "n_cells": len(archive.cells), "n_layers_measured": len(archive.layers),
        "n_trials": sum(c.n_rows for c in archive.cells), "n_null_trials": len(archive.null_trials),
        "n_boundaries": len(archive.boundaries), **rep.to_row()}]
    for b in sorted(archive.boundaries.values(), key=lambda b: b.layer):
        rows.append({"kind": "archive_boundary", "concept": archive.concept, **b.to_row()})
    for c in archive.cells:
        rows.append({"kind": "archive_cell", **c.to_row()})
    return rows
