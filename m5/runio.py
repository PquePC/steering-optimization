"""m5.runio - run directories and the files in them.

A run directory always lives outside the repository (`m5.safety.assert_outside_repo`), and every
file in it belongs to a **registered stream** with a fixed export class:

    aggregate   numbers only (rates, scalars, curves, counts, settings). May leave the machine.
    transcript  generated text. Leaves the machine only for benign vector classes, and only when
                ALLOW_TRANSCRIPT_EXPORT is set (m5.export).
    internal    never leaves the machine: judge prompts and replies (they embed transcripts),
                human labels, anything per-trial that points at text.

Writing to an unregistered stream raises. That is the fail-closed half of the export gate: M3's
gate matched file *names*, so a new file nobody had classified (judge_calls.jsonl,
read_this.md) exported by default (inventory H4). Here a file cannot exist without a class.

JSONL rules, each from a defect in the old pipelines:

- every row is stamped with `schema`, `config_hash` and a UTC timestamp, and readers refuse rows
  from another schema instead of misreading them;
- a torn *last* line (a run killed mid-write) is skipped and counted; a malformed line anywhere
  else means the file is corrupt, and reading raises rather than silently dropping data;
- appends are line-buffered and flushed, so a crash loses at most the line being written.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from m5 import SCHEMA, __version__
from m5.safety import REPO_ROOT, Refused, assert_outside_repo

EXPORT_CLASSES = ("aggregate", "transcript", "internal")

# stream name -> (export class, what one row is)
STREAMS: dict[str, tuple[str, str]] = {
    "events": ("aggregate", "a phase started or finished, with timings and counts; no text"),
    "setup": ("aggregate", "one per-model setup quantity (norm, boundary, null rate, ...)"),
    "screen": ("aggregate", "one screen-tier readout of one cell"),
    "trials": ("aggregate", "one measured trial outcome (booleans, scores, ids; never text)"),
    "cells": ("aggregate", "one cell's aggregated measure with n, k and interval"),
    "search_log": ("aggregate", "one decision of the adaptive search (visit, update, halve, stop)"),
    "confirm": ("aggregate", "one confirm-tier measure of one finalist cell"),
    "gate": ("aggregate", "one proxy-gate statistic"),
    "calibration": ("aggregate", "one calibration-set cell at reference size"),
    "analysis": ("aggregate", "one Phase-0 result over an archived run (refit, coupling, near-miss, "
                 "truncation, headline cell): numbers and transcript ids, never text"),
    "transcripts": ("transcript", "one generated text, keyed by transcript id"),
    "judge_calls": ("internal", "one judge request and reply (embeds transcript text)"),
    "labels": ("internal", "one human label of a transcript"),
    "read_this": ("transcript", "transcripts selected for a person to read"),
}

# JSON documents (not streams) and their classes.
DOCUMENTS: dict[str, str] = {
    "manifest.json": "aggregate",
    "config.json": "aggregate",
    "profile.json": "aggregate",
    "report.json": "aggregate",
    "report.md": "aggregate",
    "summary.json": "aggregate",
    "preregistration.json": "aggregate",
    "gate.json": "aggregate",
    "vectors.json": "aggregate",   # vector *metadata* (recipe, norms, diagnostics); never values
}


class CorruptRun(RuntimeError):
    """A run file is malformed somewhere other than its last line."""


class SchemaMismatch(RuntimeError):
    """A row was written under a different schema."""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit, branch and dirtiness of the code that produced a run. Never raises.

    `dirty` is None (unknown), not False, when git could not be asked: a failed status call
    reported as "clean" is a provenance claim the run did not earn.
    """
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
        except Exception:
            return None
        return out.stdout.strip() if out.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def environment() -> dict[str, Any]:
    """Library versions that can change a number. Imports nothing heavy that is not installed."""
    env: dict[str, Any] = {"python": platform.python_version(), "m5": __version__}
    for mod in ("numpy", "scipy", "torch", "transformers"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env


def _dumps(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    # numpy scalars and arrays, tuples, paths, dataclasses with to_json
    if hasattr(obj, "to_json"):
        return obj.to_json()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, default=_json_default)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass
class ReadResult:
    rows: list[dict[str, Any]]
    torn_last_line: bool


def read_jsonl(path: Path, *, schema: str | None = SCHEMA) -> ReadResult:
    """Read a JSONL file. A torn last line is skipped and reported; anything else malformed raises."""
    if not path.exists():
        return ReadResult([], False)
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    # A file ending in "\n" splits into [..., ""]; drop that empty tail.
    if lines and lines[-1] == "":
        lines = lines[:-1]
        complete_tail = True
    else:
        complete_tail = False
    rows: list[dict[str, Any]] = []
    torn = False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            is_last = i == len(lines) - 1
            if is_last and not complete_tail:
                torn = True
                break
            raise CorruptRun(f"{path}: line {i + 1} is not JSON and is not the last line") from None
        if not isinstance(row, dict):
            raise CorruptRun(f"{path}: line {i + 1} is not a JSON object")
        if schema is not None and row.get("schema") != schema:
            raise SchemaMismatch(f"{path}: line {i + 1} has schema {row.get('schema')!r}, "
                                 f"expected {schema!r}")
        rows.append(row)
    return ReadResult(rows, torn)


def heal_torn_tail(path: Path) -> str | None:
    """Make a JSONL file safe to append to after a crash; return what was quarantined, if anything.

    A run killed mid-write leaves a final line without its newline. Appending to it would glue the
    next row onto the fragment and turn a tolerable torn tail into a corrupt middle line. So before
    the first append: a complete JSON object missing only its newline gets the newline; anything
    else after the last newline is moved to `<file>.torn` (quarantined, never deleted -- it is what
    the crash left, and someone may need to read it) and cut from the file.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1  # 0 when the whole file is one unterminated line
    tail = data[cut:]
    try:
        json.loads(tail.decode("utf-8"))
        with open(path, "ab") as f:
            f.write(b"\n")
        return None
    except (json.JSONDecodeError, UnicodeDecodeError):
        with open(path.with_name(path.name + ".torn"), "ab") as q:
            q.write(tail + b"\n")
        with open(path, "r+b") as f:
            f.truncate(cut)
        return tail.decode("utf-8", errors="replace")


def write_probe(root: str | Path, megabytes: int = 8) -> None:
    """Prove the volume accepts real writes before a model is loaded (old defect K5).

    Writes `megabytes` of non-zero bytes, fsyncs, checks the size and removes the probe. Raises
    OSError with the path when the volume cannot take it.
    """
    root_p = assert_outside_repo(root, what="run root")
    root_p.mkdir(parents=True, exist_ok=True)
    probe = root_p / f".write_probe_{os.getpid()}_{os.urandom(4).hex()}"
    block = bytes(range(1, 256)) * 4096  # ~1 MB of non-zero bytes
    try:
        with open(probe, "wb") as f:
            written = 0
            while written < megabytes * 1024 * 1024:
                f.write(block)
                written += len(block)
            f.flush()
            os.fsync(f.fileno())
        if probe.stat().st_size != written:
            raise OSError(f"write probe at {probe}: wrote {written} bytes, found {probe.stat().st_size}")
    finally:
        if probe.exists():
            probe.unlink()


class RunDir:
    """One run's directory: manifest, config, registered streams and documents."""

    def __init__(self, path: str | Path) -> None:
        self.path = assert_outside_repo(path)
        self._handles: dict[str, Any] = {}
        # One id per process that writes to this run. Every row carries it, so an analysis can
        # keep the latest attempt per unit after a crash and a resume (old defect J11).
        self.attempt = os.urandom(6).hex()
        manifest = self.path / "manifest.json"
        self.manifest: dict[str, Any] = (json.loads(manifest.read_text(encoding="utf-8"))
                                         if manifest.exists() else {})

    # ---------------------------------------------------------------- lifecycle

    @classmethod
    def create(cls, root: str | Path, name: str, *, config: Any, kind: str,
               extra: Mapping[str, Any] | None = None) -> "RunDir":
        """Make `<root>/<name>` with a manifest and the full config. Refuses an existing run."""
        root_p = assert_outside_repo(root, what="run root")
        path = root_p / name
        if (path / "manifest.json").exists():
            raise FileExistsError(f"{path} already holds a run; open it to resume instead")
        path.mkdir(parents=True, exist_ok=True)
        run = cls(path)
        cfg_dict = config.as_dict() if hasattr(config, "as_dict") else dict(config)
        cfg_hash = config.hash() if hasattr(config, "hash") else None
        run.manifest = {
            "schema": SCHEMA, "kind": kind, "name": name, "created": utc_now(),
            "config_hash": cfg_hash, "git": git_state(), "environment": environment(),
            **(dict(extra) if extra else {}),
        }
        write_json_atomic(path / "manifest.json", run.manifest)
        write_json_atomic(path / "config.json", cfg_dict)
        return run

    @classmethod
    def open(cls, path: str | Path) -> "RunDir":
        run = cls(path)
        if not run.manifest:
            raise FileNotFoundError(f"{run.path} has no manifest.json; not an M5 run")
        if run.manifest.get("schema") != SCHEMA:
            raise SchemaMismatch(f"{run.path} was written under schema "
                                 f"{run.manifest.get('schema')!r}, not {SCHEMA!r}")
        return run

    def close(self) -> None:
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def __enter__(self) -> "RunDir":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def config_hash(self) -> str | None:
        return self.manifest.get("config_hash")

    # ---------------------------------------------------------------- streams

    @staticmethod
    def _check_stream(stream: str) -> None:
        if stream not in STREAMS:
            raise Refused(f"unregistered stream {stream!r}. Register it in m5.runio.STREAMS with an "
                          "export class first; an unclassified file is how the old export gate leaked.")

    def stream_path(self, stream: str) -> Path:
        self._check_stream(stream)
        return self.path / f"{stream}.jsonl"

    def append(self, stream: str, row: Mapping[str, Any]) -> dict[str, Any]:
        """Stamp and append one row; returns the row as written."""
        self._check_stream(stream)
        stamped = {"schema": SCHEMA, "config_hash": self.config_hash, "ts": utc_now(),
                   "attempt": self.attempt, **row}
        h = self._handles.get(stream)
        if h is None:
            heal_torn_tail(self.stream_path(stream))
            h = open(self.stream_path(stream), "a", encoding="utf-8", buffering=1)
            self._handles[stream] = h
        h.write(_dumps(stamped) + "\n")
        h.flush()
        return stamped

    def extend(self, stream: str, rows: Iterable[Mapping[str, Any]]) -> int:
        n = 0
        for r in rows:
            self.append(stream, r)
            n += 1
        return n

    def rows(self, stream: str, *, same_config: bool = True) -> list[dict[str, Any]]:
        """Rows of a stream; by default only those written under this run's config hash."""
        res = read_jsonl(self.stream_path(stream))
        rows = res.rows
        if same_config and self.config_hash is not None:
            rows = [r for r in rows if r.get("config_hash") == self.config_hash]
        return rows

    def keys_done(self, stream: str, fields: tuple[str, ...]) -> set[tuple[Any, ...]]:
        """The key tuples already present in a stream, for resuming without re-measuring."""
        return {tuple(_freeze(r.get(f)) for f in fields) for r in self.rows(stream)}

    # ---------------------------------------------------------------- documents

    def doc_path(self, name: str) -> Path:
        if name not in DOCUMENTS:
            raise Refused(f"unregistered document {name!r}; register it in m5.runio.DOCUMENTS")
        return self.path / name

    def write_doc(self, name: str, obj: Any) -> Path:
        p = self.doc_path(name)
        if name.endswith(".md"):
            p.write_text(str(obj), encoding="utf-8")
        else:
            write_json_atomic(p, obj)
        return p

    def read_doc(self, name: str) -> Any:
        p = self.doc_path(name)
        if name.endswith(".md"):
            return p.read_text(encoding="utf-8")
        return json.loads(p.read_text(encoding="utf-8"))

    def event(self, phase: str, status: str, **fields: Any) -> None:
        self.append("events", {"phase": phase, "status": status, **fields})


def _freeze(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    return v
