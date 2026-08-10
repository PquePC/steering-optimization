"""m2.runio -- JSONL append/read, row-level resume, archiving, the export filter, delivery.

Everything in this module is about a run's bytes on disk and the one channel by which they
leave the pod. Nothing here measures anything, and nothing here imports torch: it must be
importable on a laptop so the offline tests can exercise the export filter and the delivery
ordering, which are the two pieces with a policy consequence.

  write_row / read_rows / done_keys    append-as-you-go JSONL, one file per artefact
  resume_plan                          the generic "what is still to do" for every phase
  archive_concept                      the local retained copy, and the batch's resume marker
  export_bundle                        the deliverable, filtered by EXPORT_DENY
  deliver_then_wipe                    archive -> send -> VERIFY -> only then wipe

Three things here are load-bearing and were each written against a specific failure:

1. **The export filter INVERTS v1's policy** (spec 14.3, plan decision 47). v1 allow-listed
   aggregates and dropped anything whose name contained "transcript". M2 allows everything
   except `vectors/`, `debug/` and the binary weight extensions, because every diagnosis in
   the M1.5 review -- the velocity fixation, the pillows literal-token case, the E5-vs-drift
   disagreements -- required reading generations, and shipping rates without transcripts
   means the next such question needs a pod restart.

   The inversion is scoped to the benign arm. `export_bundle` refuses to put transcripts in
   the bundle for any concept not in `config.BENIGN_CONCEPTS` unless the caller passes
   `EXPORT_TRANSCRIPTS_OVERRIDE=True` at that call site. A harmful-arm transcript is what a
   refusal-ablated model said with `weapon` injected -- CLAUDE.md hard rule 3 -- and the
   whole point of a per-call override is that the batch driver cannot inherit the setting by
   accident when the concept list changes.

2. **Delivery is verified before anything is deleted.** v1 archived, wiped the loose folder,
   and only then attempted the Telegram send; a Telegram outage made two concepts' results
   (`Wrists`, `Wonder`) unrecoverable. Here the order is archive -> send -> confirm the send
   actually left the pod -> wipe, and an unconfirmed send keeps the folder, records the
   concept as `undelivered` in a manifest that lives ABOVE the run dir (so the wipe cannot
   take it), and leaves it for `retry_undelivered` at the end of the batch. **A delivery
   failure must never destroy data**, and "cannot verify" counts as a failure -- there is no
   branch here that deletes on an assumption.

3. **No defaulted read on a load-bearing key.** `write_row` raises when there is no run dir
   rather than writing into the process working directory (that is how two concepts' rows
   end up in one file), and `done_keys` skips-and-counts rows it cannot key rather than
   inventing a key for them. DEBUG LOG pattern 4.

Layout order (CONTRACT section 1): this module may import config, model, vectors, prompts,
cheap, judges, expensive, phases, controls, gates and monitor. It imports only `config`, and
takes the notifier as an argument instead of importing `monitor` -- so `deliver_then_wipe`
can be tested against a fake notifier with no network and no Telegram token.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config

__all__ = [
    # CONTRACT section 3 surface
    "write_row",
    "read_rows",
    "done_keys",
    "archive_concept",
    "EXPORT_DENY",
    "export_bundle",
    "deliver_then_wipe",
    # additions, each documented at its definition
    "artefact_path",
    "write_json",
    "read_json",
    "row_count",
    "resume_plan",
    "TRANSCRIPT_NAMES",
    "TRANSCRIPT_DIRS",
    "is_transcript",
    "transcripts_allowed",
    "archive_path_for",
    "bundle_path_for",
    "MANIFEST_NAME",
    "manifest_path",
    "read_manifest",
    "record_delivery",
    "undelivered",
    "retry_undelivered",
    "log",
    "R_DECIMALS",
]


# =====================================================================================
# Constants
# =====================================================================================

# Rounding applied to float components of a resume key. MUST match `vectors.R_DECIMALS` and
# `judges.R_DECIMALS` (both 6): the same `(layer, r)` cell has to produce the same key in the
# scan file, the judge cache and the dose map, or a resumed run re-measures cells it has
# already paid for -- or worse, treats two different doses as the same cell. Not imported
# from either module because both import torch and this one must stay offline-importable.
R_DECIMALS: int = 6

# Spec 14.3 / CLAUDE.md hard rules 1 and 3. The deny list is short by design: everything not
# on it ships. `vectors/` and the binary extensions are reusable attack artifacts that
# regenerate from a published config in minutes -- regeneration is the backup, so there is no
# reason to move them. `debug/` is excluded because it holds raw per-position logit dumps,
# which are activations by another name and are megabytes each.
EXPORT_DENY: tuple[str, ...] = (
    "vectors/", "debug/", ".pt", ".safetensors", ".npy", ".npz",
)

# Files whose CONTENT is model text rather than aggregate numbers. These are what spec 14.3
# deliberately includes for a benign concept and what the same section forbids exporting for
# a harmful one. Matched on the file name; `TRANSCRIPT_DIRS` matches on the first path
# component, so `unsteered/anything.jsonl` is covered whatever the phases name it.
TRANSCRIPT_NAMES: tuple[str, ...] = (
    "judge_e5.jsonl", "judge_s1.jsonl", "judge_d2.jsonl",
    "D2_transcripts.jsonl", "cis_transcripts.jsonl",
    # v1 mirrored stdout here, and stdout during a run carries sample generations verbatim.
    "console.log",
)
TRANSCRIPT_DIRS: tuple[str, ...] = ("unsteered/",)

# Catch-all for artefacts a later phase may add. A new file called `probe_transcripts.jsonl`
# must be gated by default rather than shipped by default: the export gate has to fail
# closed, because the failure it exists to prevent is one-way.
_TRANSCRIPT_SUBSTRINGS: tuple[str, ...] = ("transcript", "generation", "completions")

# The Telegram Bot API rejects documents above 50 MB. Checking before the upload turns a
# five-minute failed transfer into an immediate `undelivered` mark, and -- because an
# unverified send never wipes -- the data is kept either way.
TELEGRAM_MAX_BYTES: int = 50 * 1024 * 1024

# Seconds to wait for a queued send to be observed as sent-or-dropped. The Notifier's
# document upload uses a 120 s socket timeout, and a retry may sit behind it, so this is
# generous on purpose: waiting too long costs pod-seconds, giving up too early marks a
# delivered bundle undelivered and keeps a folder that did not need keeping. The asymmetry
# favours waiting.
DELIVERY_TIMEOUT_S: float = 300.0

# Lives in the PARENT of the run dirs, so wiping a run folder cannot take the record of
# whether that run was ever delivered with it.
MANIFEST_NAME: str = "delivery_manifest.json"

# Archive/bundle name prefixes. `archive_path_for` is also the batch's resume marker
# (spec 14.9: "a concept whose archive already exists is skipped"), so the name is computed
# in exactly one place -- a driver that guessed a different name would re-run every concept,
# and a driver that guessed a colliding one would skip a concept that never ran.
ARCHIVE_PREFIX: str = "m2_"
BUNDLE_PREFIX: str = "export_"


# =====================================================================================
# Run context and paths
# =====================================================================================

def _run() -> Any:
    """The process-global RunContext.

    Read through this accessor and never `from .config import RUN`: `model.load_model`
    rebinds the attribute on the config module, so a from-import would capture the pre-load
    placeholder forever. Bug 23's family -- a stale reference handing back something
    plausible instead of raising.
    """
    run = getattr(config, "RUN", None)
    if run is None:
        raise RuntimeError(
            "m2.config.RUN is not set - call m2.model.load_model(CONFIG) and "
            "m2.driver.set_concept(name) before writing or reading run artefacts.")
    return run


def _run_dir() -> Path:
    """The current concept's run directory, or a raise.

    No fallback to `Path(".")`. A defaulted run dir writes one concept's rows into whatever
    directory the kernel happens to be in, and the next concept appends to the same file --
    which is the failure `run_dir_for` (one folder per concept and config) exists to prevent.
    """
    run = _run()
    if run.run_dir is None:
        raise RuntimeError(
            "RUN.run_dir is not set - call m2.driver.set_concept(name) first. Writing run "
            "artefacts into the process working directory is how two concepts' rows end up "
            "in one file.")
    return Path(run.run_dir)


def _runs_dir() -> Path:
    """The parent directory that holds every concept's run folder.

    Home of `batch.log`, the delivery manifest, the per-concept archives and the shared MMLU
    pin. Derived from the live run dir when there is one; otherwise from `run_dir_for` with a
    placeholder concept, so the `M2_RUNS_DIR` test override is honoured without this module
    reaching into config's private constants.
    """
    run = getattr(config, "RUN", None)
    if run is not None and run.run_dir is not None:
        return Path(run.run_dir).parent
    return config.run_dir_for("m2", config.CONFIG).parent


def _artefact_name(name: str) -> str:
    """Normalise an artefact name to the file name spec 13 uses.

    `write_row("scan")` and `write_row("scan.jsonl")` must not become two different files:
    half the pipeline resuming from a file the other half is not writing is a silent
    resume-into-nothing. A name that already carries a suffix is left alone, so
    `unsteered/samples.jsonl` and `operating_point.json` both pass through unchanged.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"artefact name must be a non-empty string, got {name!r}")
    clean = name.strip().replace("\\", "/")
    if clean.startswith("/") or ".." in clean.split("/"):
        # The name reaches the filesystem. An absolute path or a parent traversal would put a
        # run artefact outside the run folder, where the archive would not find it and the
        # wipe would not clean it.
        raise ValueError(f"artefact name {name!r} must be relative to the run dir")
    if not Path(clean).suffix:
        clean += ".jsonl"
    return clean


def artefact_path(name: str) -> Path:
    """Absolute path of one artefact inside the current run dir. Does not create it."""
    return _run_dir() / _artefact_name(name)


def _now() -> str:
    """UTC timestamp for the `ts` every row carries (CONTRACT section 4).

    Same format as `expensive._stamp` so rows written by the two modules sort together.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp(row: dict) -> dict:
    """Add `concept`, `config_hash` and `ts` -- the three fields CONTRACT section 4 requires.

    Hard indexing on `config_hash`: a row that cannot say which configuration produced it is
    unusable for resume, because `done_keys` trusts rows only from the current config.

    The concept written is always the RUN's concept, even for a control row that injected a
    different vector. That is deliberate and matches `expensive._stamp`: `concept` identifies
    the run, and a control's own concept belongs in its own field (`control_concept`), or a
    section 9.2 row would be filtered out of its own concept's resume set.
    """
    run = _run()
    if run.concept is None:
        raise RuntimeError(
            "RUN.concept is not set - call m2.driver.set_concept(name) first. An unattributed "
            "row cannot be resumed from and cannot be read afterwards.")
    out = dict(row)
    out["concept"] = run.concept
    out["config_hash"] = run.config["config_hash"]
    out["ts"] = _now()
    return out


# =====================================================================================
# Logging
# =====================================================================================

def log(msg: str, level: str = "INFO") -> None:
    """Print, and mirror to the concept's `lab.log` and the batch-stable `batch.log`.

    Both file writes are best-effort. Logging must never crash the run: in v1 the batch died
    on concept 1 because a log line was written immediately after the per-concept folder had
    been wiped, and `batch.log` lives one level up precisely so a wipe cannot take the
    history of the batch with it.
    """
    stamp = time.strftime("%H:%M:%S")
    print(f"{stamp} [{level:<5}] {msg}", flush=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}\n"
    targets: list[Path] = []
    try:
        run = getattr(config, "RUN", None)
        if run is not None and run.run_dir is not None:
            targets.append(Path(run.run_dir) / "lab.log")
        targets.append(_runs_dir() / "batch.log")
    except Exception:                                   # noqa: BLE001 - never break the run
        targets = []
    for path in targets:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except Exception:                               # noqa: BLE001 - see docstring
            pass


# =====================================================================================
# JSONL: write, read, count
# =====================================================================================

def write_row(name: str, row: dict) -> None:
    """Append one stamped JSON object to `run_dir/<name>` (CONTRACT section 4).

    Append-as-you-go, one file per artefact, one row per unit, flushed on close: a run that
    dies mid-phase keeps every row it had already produced, which is what makes row-level
    resume (spec 14.8) possible at all. Nothing here buffers rows in memory to write at the
    end -- the end is exactly the moment that does not arrive.

    `default=str` so a stray Path or tensor in a diagnostic field cannot lose a whole row to a
    serialisation error; every load-bearing field is a primitive by construction.

    `ensure_ascii=False` so a generation containing non-ASCII text stays readable in the file
    rather than becoming escape sequences. The file is opened as UTF-8 for the same reason.
    """
    if not isinstance(row, dict):
        raise TypeError(f"row must be a dict, got {type(row).__name__}")
    path = artefact_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_stamp(row), ensure_ascii=False, default=str) + "\n")


def read_rows(name: str) -> list[dict]:
    """Every row recorded for one artefact so far. `[]` when the file does not exist.

    A missing file is not a defaulted value -- it is the resume case, and the only honest
    answer to "what has this run recorded" before the first row is written.

    A torn FINAL line is tolerated and reported: a process killed mid-append leaves a partial
    JSON object at the end of the file, and that is a known, expected crash shape. A malformed
    line anywhere else RAISES, because that is not a torn write -- it is corruption, or two
    writers on one file, and resuming from it would build the run on rows nobody can read.
    """
    path = artefact_path(name)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for index, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                log(f"{path.name}: last line is a torn write ({len(text)} chars) - dropped; "
                    "the process died mid-append and the row will be re-measured", "WARN")
                continue
            raise RuntimeError(
                f"{path}: line {index + 1} is not valid JSON, and it is not the last line. "
                "That is corruption rather than a torn append - do not resume from this file")
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{path}: line {index + 1} is a {type(parsed).__name__}, "
                               "not an object; every artefact row is one JSON object")
        rows.append(parsed)
    return rows


def row_count(name: str) -> int:
    """How many rows an artefact holds. Convenience over `len(read_rows(name))`."""
    return len(read_rows(name))


def rows_for_run(name: str) -> list[dict]:
    """Rows written by the CURRENT concept and config only.

    v1's `sweep_measure` filtered on `config_hash` for a reason worth restating: a code-only
    bugfix does not change the hash, so this is not a guarantee that rows are correct -- it is
    a guarantee that they were produced under the same *settings*. Rows from another concept
    in the same file mean something is wrong with the run dir, so they are counted loudly
    rather than quietly ignored.
    """
    run = _run()
    concept = run.concept
    cfg_hash = run.config["config_hash"]
    kept: list[dict] = []
    foreign_concept = 0
    foreign_config = 0
    for row in read_rows(name):
        if row.get("concept") != concept:
            foreign_concept += 1
            continue
        if row.get("config_hash") != cfg_hash:
            foreign_config += 1
            continue
        kept.append(row)
    if foreign_concept:
        log(f"{_artefact_name(name)}: {foreign_concept} rows carry a different concept - "
            f"expected only {concept!r}. Check that set_concept ran before the phase", "WARN")
    if foreign_config:
        log(f"{_artefact_name(name)}: {foreign_config} rows from an earlier config skipped "
            "(they will be re-measured under the current one)", "INFO")
    return kept


def _key_value(value: Any) -> Any:
    """One component of a resume key, normalised so equal cells compare equal.

    Two normalisations, both from real ways a key stops matching itself:

    - floats are rounded to `R_DECIMALS`, because a bisected `r` is arithmetic
      (0.22500000000000003) while the same cell re-derived on resume may be a literal;
    - a float that is a whole number becomes an int, because `layer` written as 37 by one
      phase and computed as `37.0` (L + 1.0) by another must be the same cell.

    `bool` is checked before `int` -- in Python `True == 1`, and silently keying a flag as an
    index is exactly the kind of collision this function exists to prevent.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        rounded = round(value, R_DECIMALS)
        return int(rounded) if float(rounded).is_integer() else rounded
    return value


def done_keys(name: str, keyfields: tuple) -> set:
    """The set of `keyfields` tuples already recorded for this concept and config.

    This is what every phase's resume is built on (spec 14.8):

        Phase 1  done_keys("scan.jsonl",     ("layer", "r"))
        Phase 3  done_keys("bisect.jsonl",   ("layer",))
        Phase 4  done_keys("verified.jsonl", ("layer", "r"))
        Phase 6  done_keys("confirm.jsonl",  ("trial",))

    A row missing one of the key fields is SKIPPED and counted, not keyed with a default. The
    two failure modes are asymmetric: skipping costs a re-measurement, whereas a defaulted key
    (`row.get("r", 0.0)`) marks a cell done that was never measured, and that cell then never
    gets measured at all. DEBUG LOG pattern 4, applied to the one structure whose job is to
    decide what does NOT run.
    """
    if not isinstance(keyfields, (tuple, list)) or not keyfields:
        raise ValueError(f"keyfields must be a non-empty tuple of field names, got {keyfields!r}")
    fields = tuple(keyfields)
    keys: set = set()
    unkeyable = 0
    missing: set[str] = set()
    for row in rows_for_run(name):
        absent = [f for f in fields if f not in row]
        if absent:
            unkeyable += 1
            missing.update(absent)
            continue
        keys.add(tuple(_key_value(row[f]) for f in fields))
    if unkeyable:
        log(f"{_artefact_name(name)}: {unkeyable} rows lack {sorted(missing)} and cannot be "
            "keyed - those units will be re-measured rather than assumed done", "WARN")
    return keys


def resume_plan(name: str, keyfields: tuple, wanted: Iterable) -> dict:
    """Split `wanted` into what still has to run and what is already recorded.

    One implementation of the skip rule for every phase, so "resumable at the row level" is a
    property of this module rather than of five separate loops that each have to remember to
    filter on `config_hash`. Phase 6's "skip if `confirm.jsonl` is complete" is this function
    returning an empty `todo`.

    `wanted` items are normalised with the same `_key_value` rule as the stored rows, so a
    caller may pass `(37, 0.15)` or `(37.0, 0.15000000000000002)` and get the same answer.
    """
    done = done_keys(name, keyfields)
    todo: list = []
    skipped: list = []
    for item in wanted:
        key = tuple(_key_value(v) for v in (item if isinstance(item, (tuple, list)) else (item,)))
        (skipped if key in done else todo).append(item)
    return dict(name=_artefact_name(name), todo=todo, skipped=skipped,
                n_todo=len(todo), n_done=len(skipped), complete=not todo)


# =====================================================================================
# Single-object artefacts
# =====================================================================================

def write_json(name: str, payload: dict) -> Path:
    """Write one JSON object atomically into the run dir (`operating_point.json`, spec 13).

    Atomic because this file is the answer: a process killed while rewriting it in place
    leaves a truncated file where a complete previous version used to be. Write-then-replace
    means the reader sees either the old answer or the new one, never half of either.
    """
    path = artefact_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_stamp(payload), ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(name: str) -> dict:
    """Read one JSON object from the run dir. Raises if it is absent -- see `read_rows`."""
    path = artefact_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} holds a {type(payload).__name__}, expected an object")
    return payload


# =====================================================================================
# The export filter
# =====================================================================================

def _denied(rel_posix: str) -> str | None:
    """The `EXPORT_DENY` entry that excludes this path, or None.

    Directory entries end in `/` and match any path component, so `debug/` catches
    `debug/E6_debug.json` and `a/debug/b.json` alike. Extension entries match the suffix,
    case-insensitively, because a `.PT` is a vector too.
    """
    low = rel_posix.lower()
    parts = low.split("/")
    for entry in EXPORT_DENY:
        if entry.endswith("/"):
            if entry[:-1] in parts[:-1]:
                return entry
        elif low.endswith(entry.lower()):
            return entry
    return None


def is_transcript(rel_posix: str) -> bool:
    """True if this file's content is model text rather than aggregate numbers.

    Fails closed: an artefact a later phase adds under a name containing "transcript",
    "generation" or "completions" is treated as a transcript even though nothing here has
    heard of it. The cost of a false positive is a file missing from a benign concept's
    bundle; the cost of a false negative is a harmful-arm generation leaving the pod.
    """
    low = rel_posix.lower()
    name = low.split("/")[-1]
    if name in {n.lower() for n in TRANSCRIPT_NAMES}:
        return True
    if any(low.startswith(d.lower()) or f"/{d.lower()}" in low for d in TRANSCRIPT_DIRS):
        return True
    return any(sub in name for sub in _TRANSCRIPT_SUBSTRINGS)


def transcripts_allowed(concept: str, cfg: dict, *,
                        EXPORT_TRANSCRIPTS_OVERRIDE: bool = False) -> tuple[bool, str]:
    """`(allowed, reason)` for putting transcripts in this concept's bundle. Spec 14.3.

    Three conditions, all of which must hold, and the reason is returned so the caller can log
    which one refused:

    1. `CONFIG["EXPORT_TRANSCRIPTS"]` is true -- hard-indexed, because a missing knob must
       raise rather than default to either answer;
    2. the concept is on `config.BENIGN_CONCEPTS` -- a transcript of a model steered toward
       `Silk` carries no dual-use risk, and one steered toward `weapon` is the exact artifact
       CLAUDE.md hard rule 3 names;
    3. or, for a non-benign concept, `EXPORT_TRANSCRIPTS_OVERRIDE=True` was passed AT THE
       CALL SITE.

    The override is a per-call argument and deliberately NOT a config key, an environment
    variable or a module global. All three of those are inheritable: the batch driver would
    carry a `True` set for the benign arm straight into the harmful arm when the concept list
    changed, which is precisely the accident spec 14.3 requires the implementation to make
    impossible.
    """
    if not cfg["EXPORT_TRANSCRIPTS"]:
        return False, "EXPORT_TRANSCRIPTS is False in CONFIG"
    if config.is_benign(concept):
        return True, "benign concept (spec 14.3 channel 2)"
    if EXPORT_TRANSCRIPTS_OVERRIDE:
        return True, f"explicit per-run override for non-benign concept {concept!r}"
    return False, (
        f"{concept!r} is not on BENIGN_CONCEPTS; transcripts withheld. Pass "
        "EXPORT_TRANSCRIPTS_OVERRIDE=True at the call site to override, per spec 14.3")


def _zip_dir(run_dir: Path, out_path: Path, keep) -> Path:
    """Zip every file under `run_dir` for which `keep(rel_posix)` is true.

    `out_path` is written outside `run_dir` by both callers, so the archive can never contain
    itself and the walk needs no self-exclusion.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a directory - nothing to archive")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    n_files = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(run_dir).as_posix()
            if not keep(rel):
                continue
            bundle.write(path, rel)
            n_files += 1
    # Replace only once the zip is closed and complete. A half-written archive that shares its
    # final name is indistinguishable from a good one, and `run_concept` skips a concept whose
    # archive exists -- so a torn archive would silently skip a concept that never finished.
    os.replace(tmp, out_path)
    if n_files == 0:
        raise RuntimeError(
            f"{out_path.name}: nothing was included from {run_dir}. An empty archive would be "
            "recorded as a delivered result; refusing to write one")
    return out_path


def archive_path_for(run_dir: Path) -> Path:
    """`<runs dir>/m2_<concept>_<hash>.zip` -- the local archive, and the resume marker.

    Spec 14.9: "a concept whose archive already exists is skipped, not re-run". The driver's
    skip test and the archiver must therefore agree on the name to the byte, so both call
    this. The name is derived from the run folder's own name, which is already
    `<concept>_<config hash>`.
    """
    run_dir = Path(run_dir)
    return run_dir.parent / f"{ARCHIVE_PREFIX}{run_dir.name}.zip"


def bundle_path_for(run_dir: Path) -> Path:
    """`<runs dir>/export_<concept>_<hash>.zip` -- the filtered deliverable.

    Written beside the archive rather than inside the run folder so that (a) the archive
    cannot contain a stale copy of a previous bundle and (b) the bundle survives the wipe,
    which is what makes an end-of-batch delivery retry possible at all.
    """
    run_dir = Path(run_dir)
    return run_dir.parent / f"{BUNDLE_PREFIX}{run_dir.name}.zip"


def archive_concept(run_dir: Path) -> Path:
    """Zip the whole run folder to `<runs dir>/m2_<concept>_<hash>.zip`. Returns the path.

    This is the copy that stays on the volume after the loose folder is wiped, and it is the
    marker that makes a re-run skip a finished concept.

    It excludes exactly one class of file: the binary vector artefacts (`vectors/`, `*.pt`,
    `*.safetensors`, `*.npy`, `*.npz`). Not for size -- they are a few MB -- but because
    CLAUDE.local.md's "regenerate, never archive" rule is specifically about keeping vectors
    in long-lived storage to save time. They rebuild from a published config in minutes;
    regeneration is the backup. Everything else, `debug/` included, is kept, because this
    archive never leaves the machine and the debug dumps are what make a cell auditable.
    """
    run_dir = Path(run_dir)
    out = archive_path_for(run_dir)
    # The deny list minus `debug/`: the debug dumps stay in the LOCAL archive (they are what
    # make a cell auditable months later) and are excluded only from the bundle that leaves
    # the pod.
    vector_deny = tuple(e for e in EXPORT_DENY if e != "debug/")

    def keep(rel: str) -> bool:
        entry = _denied(rel)
        return entry is None or entry not in vector_deny

    path = _zip_dir(run_dir, out, keep)
    log(f"archived {run_dir.name} -> {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def export_bundle(run_dir: Path, *, EXPORT_TRANSCRIPTS_OVERRIDE: bool = False) -> Path:
    """Build the deliverable bundle: everything except `EXPORT_DENY`, and the section 14.3 gate.

    The filter is a DENY list, inverting v1's allow list (plan decision 47). v1 shipped only
    what it recognised, so `judge_d2.jsonl` and `D2_transcripts.jsonl` -- files that did not
    exist when the filter was written -- would have been dropped from a bundle whose whole
    purpose is to make the run readable off the pod. A deny list ships new artefacts by
    default and drops only the named classes, which is the right default for aggregates and
    is why the transcript gate below is separate and fails closed.

    `EXPORT_TRANSCRIPTS_OVERRIDE` is keyword-only and named in capitals on purpose: it is the
    one switch in this pipeline that can move a refusal-ablated model's harmful-concept
    generations off the pod, and it should be impossible to pass by accident or to miss when
    reading a call site. See `transcripts_allowed`.
    """
    run_dir = Path(run_dir)
    run = _run()
    concept = run.concept if run.concept is not None else run_dir.name.rsplit("_", 1)[0]
    allowed, reason = transcripts_allowed(
        concept, run.config, EXPORT_TRANSCRIPTS_OVERRIDE=EXPORT_TRANSCRIPTS_OVERRIDE)

    dropped: list[str] = []

    def keep(rel: str) -> bool:
        entry = _denied(rel)
        if entry is not None:
            dropped.append(f"{rel} [{entry}]")
            return False
        if not allowed and is_transcript(rel):
            dropped.append(f"{rel} [transcript]")
            return False
        return True

    out = _zip_dir(run_dir, bundle_path_for(run_dir), keep)
    log(f"export bundle {out.name} ({out.stat().st_size / 1e6:.1f} MB) | "
        f"transcripts {'INCLUDED' if allowed else 'WITHHELD'}: {reason}")
    if dropped:
        # Named, not counted. "3 files excluded" tells a reader nothing about whether the
        # right three were excluded, and this is the filter with a one-way failure mode.
        log("excluded from the bundle: " + ", ".join(sorted(dropped)), "INFO")
    return out


# =====================================================================================
# The delivery manifest
# =====================================================================================

def manifest_path() -> Path:
    """`<runs dir>/delivery_manifest.json`, above the run folders so a wipe cannot take it."""
    return _runs_dir() / MANIFEST_NAME


def read_manifest() -> dict:
    """The delivery manifest, or an empty one. Never raises on a damaged file.

    A manifest that cannot be parsed is renamed aside rather than deleted, and an empty one is
    returned. The manifest records what still needs delivering; losing it is bad, but refusing
    to run because of it would be worse -- and the loose folders it points at are all still on
    disk, which is the invariant that actually protects the data.
    """
    path = manifest_path()
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("entries"), dict):
            return payload
        raise ValueError("manifest has no 'entries' object")
    except Exception as exc:                            # noqa: BLE001 - see docstring
        broken = path.with_name(path.name + f".broken.{int(time.time())}")
        try:
            os.replace(path, broken)
        except Exception:                               # noqa: BLE001
            pass
        log(f"delivery manifest unreadable ({type(exc).__name__}); moved aside as "
            f"{broken.name} and starting a fresh one", "WARN")
        return {"version": 1, "entries": {}}


def _write_manifest(payload: dict) -> Path:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def record_delivery(run_dir: Path, bundle: Path, state: str, detail: str = "") -> dict:
    """Record one concept's delivery state. `state` is 'delivered' or 'undelivered'.

    `detail` is a short LABEL, never an exception message: the manifest can end up inside a
    later export bundle, and an API error message can quote the request payload -- which under
    M2 is a judge prompt containing a steered generation (spec 14.3 channel 1's reasoning,
    applied to a file rather than to an alert).
    """
    if state not in ("delivered", "undelivered"):
        raise ValueError(f"delivery state must be 'delivered' or 'undelivered', got {state!r}")
    run_dir = Path(run_dir)
    payload = read_manifest()
    entry = payload["entries"].get(run_dir.name, {})
    entry.update(
        concept=_run().concept,
        run_dir=str(run_dir),
        bundle=str(bundle),
        archive=str(archive_path_for(run_dir)),
        state=state,
        detail=detail,
        attempts=int(entry.get("attempts", 0)) + 1,
        ts=_now(),
    )
    payload["entries"][run_dir.name] = entry
    _write_manifest(payload)
    return entry


def undelivered() -> list[dict]:
    """Every manifest entry still marked `undelivered`, oldest first."""
    payload = read_manifest()
    rows = [e for e in payload["entries"].values() if e.get("state") == "undelivered"]
    return sorted(rows, key=lambda e: e.get("ts", ""))


# =====================================================================================
# Delivery
# =====================================================================================

def _send_and_confirm(notifier: Any, path: Path, caption: str,
                      timeout: float = DELIVERY_TIMEOUT_S) -> tuple[bool, str]:
    """Send one file and return `(confirmed, reason)`. Never raises.

    "Confirmed" means the notifier's own counters show the send left the pod: `dropped`
    unchanged and `sent` increased. The Notifier swallows transport errors on purpose (a
    failed alert must never become a failed run), so a completed `send_file` call proves
    nothing at all -- the counters are the only evidence available, and they are exactly what
    v1's `notify_test` used to tell a working channel from a silent one.

    Every branch that cannot obtain that evidence returns False. Not-verifiable must behave
    like not-delivered, because the caller deletes data on a True.
    """
    if notifier is None:
        return False, "no notifier"
    if not getattr(notifier, "enabled", False):
        # No Telegram credentials: there is no delivery channel, so nothing was delivered and
        # the loose folder stays. The operator's copy is the local archive on the volume.
        return False, "delivery channel not configured (no TELEGRAM_BOT_TOKEN / CHAT_ID)"
    if not path.exists():
        return False, "bundle does not exist"
    size = path.stat().st_size
    if size > TELEGRAM_MAX_BYTES:
        return False, f"bundle is {size / 1e6:.1f} MB, above the {TELEGRAM_MAX_BYTES / 1e6:.0f} MB limit"

    sent_before = getattr(notifier, "sent", None)
    dropped_before = getattr(notifier, "dropped", None)
    if not isinstance(sent_before, int) or not isinstance(dropped_before, int):
        return False, "notifier exposes no sent/dropped counters, so the send cannot be verified"

    try:
        notifier.send_file(path, caption=caption)
    except Exception as exc:                            # noqa: BLE001 - label only, never the message
        return False, f"send_file raised {type(exc).__name__}"

    # Wait for the worker to account for this item, one way or the other. Polling the counters
    # rather than joining the queue keeps this independent of the Notifier's internals, and
    # the queue is drained by a single daemon thread, so an item that has been accounted for
    # is an item that has finished.
    deadline = time.time() + max(1.0, float(timeout))
    while time.time() < deadline:
        accounted = (getattr(notifier, "sent", sent_before) - sent_before) + \
                    (getattr(notifier, "dropped", dropped_before) - dropped_before)
        if accounted > 0:
            break
        time.sleep(1.0)

    now_sent = getattr(notifier, "sent", sent_before)
    now_dropped = getattr(notifier, "dropped", dropped_before)
    if now_dropped > dropped_before:
        return False, "the notifier recorded a dropped send (transport failure)"
    if now_sent > sent_before:
        return True, "confirmed by the notifier's sent counter"
    return False, f"no send was accounted for within {timeout:.0f}s"


def deliver_then_wipe(zip_path: Path, notifier: Any, wipe: bool,
                      *, run_dir: Path | None = None,
                      caption: str = "",
                      timeout: float = DELIVERY_TIMEOUT_S) -> bool:
    """Send `zip_path`, verify it left the pod, and only then wipe the loose run folder.

    Returns True if delivery was confirmed.

    **The order is the point.** v1 archived the concept, wiped the loose folder, and then
    attempted the Telegram send. Telegram was down; the per-concept results for `Wrists` and
    `Wonder` were gone by the time anyone knew. The order here is

        archive (already done by the caller) -> send -> VERIFY the send succeeded -> wipe

    and there is no path that deletes anything on an unverified send. On failure the loose
    folder is kept, the concept is recorded `undelivered` in a manifest that lives above the
    run dirs, and `retry_undelivered` picks it up at the end of the batch.

    `wipe=False` skips the deletion entirely; delivery is still attempted and recorded, so a
    cautious batch and a normal one produce the same manifest.
    """
    zip_path = Path(zip_path)
    target = Path(run_dir) if run_dir is not None else _run_dir()

    # The wipe deletes a directory tree, so the link between the bundle and the folder is
    # checked rather than assumed. A caller that passed the previous concept's bundle would
    # otherwise delete THIS concept's folder on the strength of THAT concept's delivery.
    if target.name not in zip_path.name:
        raise ValueError(
            f"refusing to wipe {target} on the strength of {zip_path.name}: the bundle name "
            "does not carry the run folder's name, so they may belong to different concepts")

    caption = caption or f"{_run().concept} {_run().config['config_hash']} - M2 run bundle"
    confirmed, reason = _send_and_confirm(notifier, zip_path, caption, timeout)

    if confirmed:
        record_delivery(target, zip_path, "delivered", reason)
        log(f"delivery CONFIRMED for {target.name}: {reason}")
        if wipe:
            archive = archive_path_for(target)
            if not archive.exists():
                # Belt and braces: the caller archives first, but if that archive is missing
                # the loose folder is the only copy of the run and must not be deleted.
                log(f"delivery confirmed but {archive.name} is missing - keeping the loose "
                    "folder; the archive is the local copy the wipe assumes exists", "WARN")
                return True
            shutil.rmtree(target, ignore_errors=True)
            log(f"wiped loose folder for {target.name}; kept {archive.name} and {zip_path.name}")
        return True

    record_delivery(target, zip_path, "undelivered", reason)
    log(f"delivery NOT confirmed for {target.name} ({reason}) - loose folder KEPT and marked "
        "undelivered; retry at the end of the batch", "WARN")
    return False


def retry_undelivered(notifier: Any, wipe: bool = True,
                      *, timeout: float = DELIVERY_TIMEOUT_S) -> dict:
    """Re-attempt every bundle still marked `undelivered`. Called at the end of a batch.

    Spec 14.9: "keep the loose folder, mark the concept `undelivered` in a manifest, and retry
    delivery at the end of the batch". A retry that succeeds wipes that concept's loose folder
    under the same rule as the first attempt: only after the send is confirmed, and only if
    the local archive exists.

    Never raises. This runs after the science is finished, and an exception here would take
    the pod-stop and the final message with it.
    """
    pending = undelivered()
    result = dict(attempted=len(pending), delivered=[], still_undelivered=[])
    if not pending:
        return result
    log(f"retrying {len(pending)} undelivered bundle(s)")
    for entry in pending:
        bundle = Path(entry.get("bundle", ""))
        target = Path(entry.get("run_dir", ""))
        name = entry.get("concept", target.name)
        try:
            if not bundle.exists():
                log(f"retry {name}: bundle {bundle.name} is gone - cannot resend", "WARN")
                result["still_undelivered"].append(name)
                continue
            confirmed, reason = _send_and_confirm(notifier, bundle, f"{name} - retry", timeout)
            payload = read_manifest()
            row = payload["entries"].get(target.name, dict(entry))
            row.update(state="delivered" if confirmed else "undelivered", detail=reason,
                       attempts=int(row.get("attempts", 0)) + 1, ts=_now())
            payload["entries"][target.name] = row
            _write_manifest(payload)
            if confirmed:
                result["delivered"].append(name)
                log(f"retry {name}: delivered ({reason})")
                archive = archive_path_for(target)
                if wipe and target.is_dir() and archive.exists():
                    shutil.rmtree(target, ignore_errors=True)
                    log(f"wiped loose folder for {target.name} after a successful retry")
            else:
                result["still_undelivered"].append(name)
                log(f"retry {name}: still undelivered ({reason})", "WARN")
        except Exception as exc:                        # noqa: BLE001 - label only
            result["still_undelivered"].append(name)
            log(f"retry {name} raised {type(exc).__name__} - folder kept", "WARN")
    return result
