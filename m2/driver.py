"""m2.driver -- `set_concept`, the per-concept pipeline, and the batch driver.

This is the last module in the CONTRACT layout order and the only one that may import
everything. It is a sequencer: it decides *what runs, in what order, and what happens when a
piece of it dies*. It contains no science. Every number comes from `phases`, `controls` or
`gates`, and every byte written goes through `runio`.

  set_concept(name)     point the run at one concept: new run dir, cleared per-concept state
  run_concept(name)     Phases 0-6 plus the controls, resumable, per-phase exception isolation
  run_batch(concepts)   the same over a concept list, isolated, with the pod stop at the end

Four properties are load-bearing, and each of them is a failure that has already happened:

1. **`set_concept` clears `RUN.vecs`, `RUN.norms` and `RUN.base` before rebuilding them.**
   Bug 23: in a live kernel the previous concept's cached values still matched after a
   concept switch, so the previous concept's numbers came back silently. It also clears the
   forward-pass and judge caches for the same reason, and re-seeds the judge cache from the
   rows already on disk so a resumed run never pays for a judge call twice (spec 14.8).

2. **Per-phase exception isolation.** One phase raising must not cost the other seven. v1's
   RUN ALL cell established this for measures; here it applies to phases, whose costs differ
   by three orders of magnitude - losing a completed Phase 1 scan because the controls raised
   would throw away the expensive half of the run for the cheap half's fault. The traceback
   goes to a crash file on the volume; only a classified LABEL is ever notified (spec 14.3
   channel 1).

3. **Archive before delivery, delivery before deletion.** `runio.deliver_then_wipe` enforces
   the ordering; this module's job is to call it in the right order and never to delete
   anything itself.

4. **The heavy siblings are imported inside the functions that use them**, exactly as
   `prompts.py` does and for the same reason: `model`, `vectors`, `cheap`, `expensive`,
   `phases` and `controls` all import torch, and `m2.driver` has to stay importable on a
   laptop for the offline tests. It also means a module that is missing or that fails to
   import fails the phase that needs it, with a named label, instead of making the whole
   driver unimportable.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import config
from . import runio

__all__ = [
    # CONTRACT section 3 surface
    "set_concept",
    "run_concept",
    "run_batch",
    # additions, each documented at its definition
    "PHASE_ORDER",
    "BATCH_STATE",
    "get_notifier",
    "kill_pod",
    "reference_layer",
]


# =====================================================================================
# Constants
# =====================================================================================

# The board's phase list, and the keys of `monitor.PHASE_SECONDS_PRIOR` (spec 14.5). The rig
# checks and the acceptance gates run inside `run_concept` too, but they are not phases: they
# have no per-unit cost model and appear in the log rather than on the board.
PHASE_ORDER: tuple[str, ...] = (
    "CAL", "SCAN", "SHORTLIST", "BISECT", "VERIFY", "REFINE", "CONFIRM", "CONTROLS",
)

# Artefact names, spec 13. Written out here because the driver reads several of them back for
# the resume banner and for the final selection over Phase 4 AND Phase 5 rows together.
SCAN_FILE = "scan.jsonl"
# Appended, one row per run_concept entry: a resumed concept can span two pods, and seeing
# that afterwards is the point.
PROVENANCE_FILE = "provenance.jsonl"
SHORTLIST_FILE = "shortlist.json"
BISECT_FILE = "bisect.jsonl"
VERIFIED_FILE = "verified.jsonl"
CONFIRM_FILE = "confirm.jsonl"
CONTROLS_FILE = "controls.jsonl"
OPERATING_POINT_FILE = "operating_point.json"
RUN_RECORD_FILE = "run_record.json"
STATUS_FILE = "status.txt"
CONFIG_FILE = "config.json"

# Spec 4.1 / 5.1: the reference layer is 0.60 of the stack, which is where the M1.5 lab put
# it and where `model._LIVENESS_DEPTH` puts the hook-liveness check. `int()` truncation rather
# than rounding, to match the repo's `get_layer_at_fraction` (bug 11: 0.35 x 62 = 21, not 22).
REF_FRACTION: float = 0.60

# Per-concept state the batch driver keeps for the ETA. v1's status board read a global of
# this shape; `monitor` sits before this module in the layout order and so cannot import it,
# which is why the batch also puts the same summary in the `extra` line of every board push.
BATCH_STATE: dict = dict(total=0, done=0, durations=[], cur_t0=0.0, concept=None)

# One Notifier per process, not one per concept: each one starts a daemon worker thread, and
# twenty-one of those would be twenty-one queues with the ordering guarantee spread across
# them (spec 14.4: sends are ordered because ONE worker drains ONE queue).
_NOTIFIER: Any = None


# =====================================================================================
# Lazy module accessors
# =====================================================================================
# Each import happens at first use rather than at module scope. See point 4 of the module
# docstring: offline importability, and a missing sibling failing one phase rather than the
# whole driver.

def _model() -> Any:
    from . import model
    return model


def _vectors() -> Any:
    from . import vectors
    return vectors


def _prompts() -> Any:
    from . import prompts
    return prompts


def _judges() -> Any:
    from . import judges
    return judges


def _expensive() -> Any:
    from . import expensive
    return expensive


def _phases() -> Any:
    from . import phases
    return phases


def _controls() -> Any:
    from . import controls
    return controls


def _gates() -> Any:
    from . import gates
    return gates


def _monitor() -> Any:
    from . import monitor
    return monitor


def _run() -> Any:
    """The process-global RunContext. Never `from .config import RUN` -- see `runio._run`."""
    run = getattr(config, "RUN", None)
    if run is None:
        raise RuntimeError("m2.config.RUN is not set - import order is broken")
    return run


def _label(exc: BaseException) -> str:
    """A classified, transmittable label for an exception. Never the message.

    `monitor.classify_exc` is the authority (spec 14.3 channel 1: an API error can quote the
    request payload, and under M2 that payload is a steered generation). If `monitor` cannot
    be imported, the exception CLASS NAME alone is the safe degradation - a class name is
    code, not data.
    """
    try:
        return str(_monitor().classify_exc(exc))
    except Exception:                                   # noqa: BLE001 - the fallback is the point
        return type(exc).__name__


# =====================================================================================
# Notifier and status board
# =====================================================================================

def get_notifier() -> Any:
    """The process-wide Notifier, built on first use. Returns None if `monitor` will not import.

    Never raises: a run with no alert channel is a run that must be watched in the notebook,
    not a run that refuses to start.
    """
    global _NOTIFIER
    if _NOTIFIER is None:
        try:
            _NOTIFIER = _monitor().Notifier()
        except Exception as exc:                        # noqa: BLE001
            runio.log(f"no notifier ({_label(exc)}) - the run is unattended-blind; watch "
                      f"{STATUS_FILE} on the volume", "WARN")
            return None
    return _NOTIFIER


class _Board:
    """Failure-proof adaptor over `monitor.RunStatus`.

    Two reasons this is not a direct call. First, an alert or a board update must never take
    the run down with it -- v1's rule, and every method here is wrapped accordingly. Second,
    the board is the piece most likely to differ in detail from what this module expects, and
    a mismatch should cost the board, not the measurement: an unknown method is reported once
    and then ignored.
    """

    def __init__(self, impl: Any) -> None:
        self.impl = impl
        self._warned: set[str] = set()

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self.impl is None:
            return None
        fn = getattr(self.impl, method, None)
        if fn is None:
            if method not in self._warned:
                self._warned.add(method)
                runio.log(f"status board has no {method}() - board updates for it are skipped "
                          "(the run and its files are unaffected)", "WARN")
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                        # noqa: BLE001 - see class docstring
            if method not in self._warned:
                self._warned.add(method)
                runio.log(f"status board {method}() raised {_label(exc)} - ignored", "WARN")
            return None

    # The methods the CONTRACT names, plus attach/detach if monitor provides them. A method
    # MISSING FROM HERE is not degraded gracefully - it is an AttributeError on the adaptor
    # itself, outside the `_call` guard, and it takes the run down. Adding a board call in the
    # driver means adding a forwarder here; `test_board_forwards_every_method_the_driver_calls`
    # is what makes that a test failure rather than a crash forty minutes into a run.
    def start_phase(self, name: str, total: int | None = None) -> None:
        self._call("start_phase", name, total)

    def unit_done(self, name: str) -> None:
        self._call("unit_done", name)

    def end_phase(self, name: str) -> None:
        self._call("end_phase", name)

    def skip_phase(self, name: str, why: str = "") -> None:
        self._call("skip_phase", name, why)

    def size_phase(self, name: str, total: int) -> None:
        self._call("size_phase", name, int(total))

    def fail_phase(self, name: str, exc: BaseException) -> None:
        self._call("fail_phase", name, exc)

    def render(self) -> None:
        self._call("render")

    def write_status_txt(self) -> None:
        self._call("write_status_txt")

    def attach(self) -> None:
        self._call("attach")

    def detach(self) -> None:
        self._call("detach")


def _make_board(run_dir: Path) -> _Board:
    """Build the status board for one concept, or a no-op board.

    ASSUMPTION recorded here rather than buried: `monitor.RunStatus` is constructed as
    `RunStatus(phase_order, PHASE_SECONDS_PRIOR, status_path)`, the same three arguments v1's
    `RunStatus(order, priors, n_cells, path)` took minus the per-measure cell count (M2 phases
    have different unit counts, so the total is set per phase by `start_phase`). If that
    signature differs, this degrades to a board-free run rather than failing the concept.
    """
    try:
        monitor = _monitor()
        # KEYWORDS, NOT POSITION. The real signature is
        #     RunStatus(order, path=None, totals=None, priors=None, notifier=None)
        # so the positional call this replaced passed PHASE_SECONDS_PRIOR as `path` and the
        # Path as `totals`, and `Path(a_dict)` raised TypeError. The except below then did
        # exactly what it promises - degraded to a board-free run - so a whole concept ran
        # with no progress board, no ETA and no stall detection, announced by one WARN line.
        # Graceful degradation around a wrong call signature is how a defect survives a run.
        impl = monitor.RunStatus(PHASE_ORDER,
                                 path=Path(run_dir) / STATUS_FILE,
                                 totals=monitor.PHASE_UNITS_PRIOR,
                                 priors=monitor.PHASE_SECONDS_PRIOR)
        return _Board(impl)
    except Exception as exc:                            # noqa: BLE001
        runio.log(f"status board unavailable ({_label(exc)}) - running without it", "WARN")
        return _Board(None)


def _notify(notifier: Any, board: _Board, banner: str, extra: str = "",
            severity: str = "info") -> None:
    """Push the whole board under a one-line banner. Never raises.

    Spec 14.4: every notification carries the board, because the board answers the follow-up
    question and a message that prompts a follow-up you cannot make is a bad message at 2am.
    When there is no board object, the banner and its extra line still go out on their own --
    a degraded alert beats no alert.
    """
    if notifier is None:
        return
    try:
        if board.impl is not None:
            notifier.board(board.impl, banner, extra, severity)
        else:
            if severity != "warn" and getattr(notifier, "warnings_only", False):
                return
            notifier.send(banner + (("\n" + extra) if extra else ""))
    except Exception as exc:                            # noqa: BLE001
        runio.log(f"notification failed ({_label(exc)}) - the run itself is fine", "WARN")


# =====================================================================================
# set_concept
# =====================================================================================

def reference_layer() -> int:
    """The reference layer, `int(n_layers * 0.60)`. Requires a loaded model."""
    run = _run()
    if not run.n_layers:
        raise RuntimeError("RUN.n_layers is not set - call m2.model.load_model(CONFIG) first")
    return int(run.n_layers * REF_FRACTION)


def set_concept(name: str) -> Path:
    """Point the run at `name`: rebuild the run dir, and clear everything the old concept owned.

    Ported from the v1 lab's `set_concept` (lab_cells.py cell 11), including the two details
    that make resume work:

    - **the hash is taken over CONFIG with `config_hash` removed** (`config.config_hash` does
      this), because leaving the key in makes the hash a function of itself, so the value
      stored in config.json never reproduces and every rerun starts an empty folder instead
      of resuming;
    - **one folder per (concept, config)**, so re-running a concept RESUMES and a changed
      config never overwrites an old run's rows.

    What v1 did not do, and bug 23 is why: `RUN.vecs`, `RUN.norms` and `RUN.base` are cleared
    before the rebuild (`RunContext.reset_concept`), and the forward-pass and judge caches are
    dropped. In a live kernel every cached entry from the previous concept still matched, so
    the previous concept's numbers came back silently - a wrong number, not an error.

    The judge cache is then RE-SEEDED from the judge rows already on disk, which is the other
    half of spec 14.8: a resumed run must not pay for a judge call it has already paid for.
    """
    run = _run()
    # After load_model, RUN.config IS the cfg dict that was passed to it - the same object the
    # notebook edits as CONFIG - so mutating it in place is the intended path, exactly as v1's
    # set_concept mutated CONFIG. Before load_model there is no context yet, so fall back to
    # the module CONFIG; `or` is safe here because an empty dict is the only falsy case and it
    # means precisely "not loaded".
    cfg = run.config or config.CONFIG

    concept = str(name).strip()
    cfg["concept"] = concept
    cfg.pop("config_hash", None)                  # the hash is over CONFIG WITHOUT this key
    cfg["config_hash"] = config.config_hash(cfg)

    run_dir = config.run_dir_for(concept, cfg)    # validates the folder-name shape and raises
    run_dir.mkdir(parents=True, exist_ok=True)
    # `vectors/` is where vectors.py caches extraction (never exported, never archived);
    # `unsteered/` is spec 13's home for the Phase 0 completions.
    (run_dir / "vectors").mkdir(exist_ok=True)
    (run_dir / "unsteered").mkdir(exist_ok=True)

    # BUG 23. One call, one place: a defence that has to be repeated at each call site is a
    # defence that gets missed once.
    run.reset_concept(concept, run_dir, cfg)

    # config.json is a faithful copy of CONFIG, written unstamped so that two runs of the same
    # configuration produce byte-identical files and a diff shows a real change.
    (run_dir / CONFIG_FILE).write_text(json.dumps(cfg, indent=2, sort_keys=True),
                                       encoding="utf-8")

    _clear_caches()
    _prime_judge_cache()

    runio.log(f"concept {concept} | config {cfg['config_hash']} | {run_dir}")
    return run_dir


def _clear_caches() -> None:
    """Drop the forward-pass and judge caches. Bug 23's second half.

    Neither is needed for correctness any more -- both keys carry a vector fingerprint now --
    but a 21-concept batch that never clears them holds every previous concept's logits and
    verdicts in host RAM. The import is guarded (no torch on a laptop); the clear itself is
    NOT, because a cache that refuses to clear is exactly the condition bug 23 describes.
    """
    try:
        model = _model()
    except Exception:                                   # noqa: BLE001 - offline / no torch
        model = None
    if model is not None:
        dropped = model.cache_clear()
        if dropped:
            runio.log(f"forward-pass cache cleared ({dropped} entries)", "INFO")

    try:
        judges = _judges()
    except Exception:                                   # noqa: BLE001
        return
    dropped = judges.cache_clear()
    if dropped:
        runio.log(f"judge cache cleared ({dropped} entries)", "INFO")


def _prime_judge_cache() -> None:
    """Re-seed the judge cache from `judge_e5/a2/b.jsonl`. Resume support, spec 14.8.

    Judge calls are the expensive part of Phases 4-6, and without this a kernel restart pays
    for every one of them again. `judges.prime_cache` re-parses each row's RAW text rather
    than trusting its stored `parsed` block, and skips any row it cannot validate: a miss
    costs a judge call, a wrong hit costs a wrong number.
    """
    try:
        judges = _judges()
        expensive = _expensive()
    except Exception:                                   # noqa: BLE001 - offline / no torch
        return
    total = 0
    for name in (expensive.JUDGE_A1_FILE, expensive.JUDGE_A2_FILE, expensive.JUDGE_B_FILE):
        rows = runio.read_rows(name)
        if not rows:
            continue
        report = judges.prime_cache(rows)
        loaded = report["loaded"] if isinstance(report, dict) and "loaded" in report else 0
        total += int(loaded)
    if total:
        runio.log(f"judge cache primed with {total} previously paid-for calls (resume)")


# =====================================================================================
# Phase isolation
# =====================================================================================

class _ConceptRun:
    """Bookkeeping for one concept: phase results, failures, the board and the notifier."""

    def __init__(self, concept: str, run_dir: Path, notifier: Any, board: _Board) -> None:
        self.concept = concept
        self.run_dir = Path(run_dir)
        self.notifier = notifier
        self.board = board
        self.t0 = time.time()
        self.results: dict[str, Any] = {}
        self.failed: list[str] = []
        self.labels: dict[str, str] = {}

    def tick(self, name: str) -> Callable[[Any], None]:
        """An `on_cell` callback that advances the board one unit. Never raises.

        Progress reporting must not be able to kill a measured cell, so anything the board
        throws is swallowed here - the run keeps its data and loses only its ETA.
        """
        def _tick(_row: Any = None) -> None:
            try:
                self.board.unit_done(name)
            except Exception:                           # noqa: BLE001 - never cost a cell
                pass
        return _tick

    def plan(self, name: str) -> Callable[[int], None]:
        """An `on_plan` callback that tells the board how many units this phase will run."""
        def _plan(total: int) -> None:
            try:
                self.board.size_phase(name, int(total))
            except Exception:                           # noqa: BLE001 - never cost a cell
                pass
        return _plan

    def phase(self, name: str, fn: Callable[[], Any], *, units: int | None = None,
              tracked: bool = True, self_reporting: bool = False) -> tuple[bool, Any]:
        """Run one phase, isolated. Returns `(ok, value)`; never raises.

        `self_reporting` means the phase advances the board itself, through `tick`, so this
        must not add a synthetic unit of its own at the end - that would double-count the last
        cell and leave `spent/done` describing a rate no cell ever ran at.

        On failure: the full traceback goes to a crash file on the volume, a CLASSIFIED label
        goes to the phone, the phase is recorded as failed, and the caller decides whether the
        rest of the pipeline can still run. Two phases failing is what `monitor.verdict`
        reads as structural (spec 14.6 rule 2); one is `attention`, and the others continue by
        design.
        """
        if tracked:
            self.board.start_phase(name, units)
        runio.log(f"--- {name} ---")
        started = time.time()
        try:
            value = fn()
        except Exception as exc:                        # noqa: BLE001 - isolation is the point
            label = _label(exc)
            self.failed.append(name)
            self.labels[name] = label
            crash = self.run_dir / f"crash_{name}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                crash.write_text(
                    f"concept {self.concept} | phase {name} | "
                    f"config {_run().config['config_hash']}\n\n" + traceback.format_exc(),
                    encoding="utf-8")
            except Exception:                           # noqa: BLE001 - a folder may be gone
                crash = self.run_dir / "(crash file could not be written)"
            runio.log(f"{name} FAILED: {label} - crash report {crash.name}", "ERROR")
            print(traceback.format_exc())
            if tracked:
                self.board.fail_phase(name, exc)
            _notify(self.notifier, self.board, f"PHASE FAILED - {name}: {label}",
                    f"{self.concept}: the remaining phases continue.", severity="warn")
            return False, None
        elapsed = time.time() - started
        self.results[name] = value
        runio.log(f"{name} done in {elapsed:.1f}s")
        if tracked:
            if not self_reporting:
                self.board.unit_done(name)
            # `end_phase` is what moves a phase out of "running" - and nothing was calling it,
            # so every finished phase stayed `>>running` for the whole run, `verdict` kept
            # reporting "running: CAL" forty minutes after CAL returned, and the per-phase
            # phone push (its `notify` side effect) never fired once.
            self.board.end_phase(name)
            self.board.write_status_txt()
        return True, value


# =====================================================================================
# Small readers over other modules' return values
# =====================================================================================

def _judge_fpr_of(cal: Any) -> float | None:
    """`judge_fpr` from Phase 0, or None when the calibration did not report one.

    Membership tests, never `.get(key, 0.0)`: a defaulted 0.0 would read as "the judge invents
    no influence", which is the *passing* answer for spec 14.6 rule 5, and the gate would
    silently never fire. None means "not reported" and is logged as such.
    """
    if isinstance(cal, dict) and "judge_fpr" in cal:
        value = cal["judge_fpr"]
    else:
        base = getattr(_run(), "base", {}) or {}
        if "judge_fpr" not in base:
            return None
        value = base["judge_fpr"]
    if isinstance(value, dict):
        if "fpr" in value:
            value = value["fpr"]
        elif "judge_fpr" in value:
            value = value["judge_fpr"]
        else:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_reach(scan_rows: Any) -> float | None:
    """The highest `reach` in the scan, or None if no row carries one (spec 9.3's trigger)."""
    if not isinstance(scan_rows, (list, tuple)):
        return None
    values = [float(r["reach"]) for r in scan_rows
              if isinstance(r, dict) and "reach" in r and r["reach"] is not None]
    return max(values) if values else None


def _cell_of(winner: Any) -> tuple[int, float] | None:
    """`(layer, r)` of the selected operating point, or None if there is no winner."""
    if not isinstance(winner, dict):
        return None
    if "layer" not in winner or "r" not in winner:
        return None
    return int(winner["layer"]), float(winner["r"])


def _control_verdict(payload: Any, what: str) -> str:
    """'pass', 'reject' or 'unknown' for one control's return value.

    ASSUMPTION, recorded loudly: `controls.random_direction_control` and
    `controls.forced_id_capability_control` return a dict carrying `verdict` in
    {'pass', 'reject'}. The alternative key names below are read so a near-miss still works,
    and anything unreadable becomes 'unknown' -- which is reported and does NOT count toward
    the batch abort. A control whose verdict cannot be read must never be scored as a pass:
    spec 9.1 and 9.2 are hard gates, and a silent pass on an unreadable gate is the failure
    they exist to prevent.
    """
    if isinstance(payload, dict):
        if "verdict" in payload:
            value = str(payload["verdict"]).strip().lower()
            if value in ("pass", "reject"):
                return value
        for key in ("damage_dominated", "compromised", "rejected"):
            if key in payload:
                return "reject" if bool(payload[key]) else "pass"
        if "passed" in payload:
            return "pass" if bool(payload["passed"]) else "reject"
    runio.log(f"{what}: no readable verdict in the control's return value - recorded as "
              "'unknown'. It does not count as a pass", "WARN")
    return "unknown"


def _qualifying(rows: Any) -> list[dict]:
    """Rows flagged `qualifies` (spec 7). Membership test: an absent flag is not a pass."""
    if not isinstance(rows, (list, tuple)):
        return []
    return [r for r in rows if isinstance(r, dict) and "qualifies" in r and r["qualifies"]]


# =====================================================================================
# run_concept
# =====================================================================================

def run_concept(name: str, *, notifier: Any = None, wipe: bool = True, deliver: bool = True,
                EXPORT_TRANSCRIPTS_OVERRIDE: bool = False,
                after_phases: Any = None) -> dict:
    """Phases 0-6 plus the controls for one concept. Resumable, isolated, never raises.

    Skips entirely if the concept's archive already exists (spec 14.9): a finished concept is
    not re-run, and the archive is the marker because it is the last thing a finished concept
    produces.

    Row-level resume belongs to the phases themselves -- each of them asks `runio.done_keys`
    what is already recorded (spec 14.8) -- so this function's part in it is to point them at
    the right run dir, prime the judge cache, and report what was found.

    `EXPORT_TRANSCRIPTS_OVERRIDE` is keyword-only and is passed straight through to
    `runio.export_bundle`. It is not read from CONFIG, the environment or a module global,
    because all three are inheritable and spec 14.3 requires that the harmful arm cannot pick
    this up by accident when the concept list changes.
    """
    run_dir = set_concept(name)
    concept = _run().concept
    archive = runio.archive_path_for(run_dir)
    if archive.exists():
        runio.log(f"SKIP {concept}: already archived ({archive.name}) - not re-running")
        return dict(concept=concept, run_dir=str(run_dir), status="skipped",
                    archive=str(archive), failed_phases=[], elapsed_s=0.0)

    notifier = notifier if notifier is not None else get_notifier()

    # Which machine and which library versions produced these numbers. Appended rather than
    # overwritten: a resumed concept can legitimately span two pods, and the whole point is to
    # be able to see that afterwards instead of reconstructing it from memory.
    try:
        prov = _model().provenance()
        prov["ts"] = runio._now()
        prov["phase_entered_at"] = "run_concept"
        runio.write_row(PROVENANCE_FILE, prov)
        runio.log(f"provenance: {prov.get('gpu', '?')} | torch {prov.get('torch', '?')} | "
                  f"host {prov.get('host', '?')}")
    except Exception as exc:                        # noqa: BLE001 - never cost a run
        runio.log(f"provenance not recorded ({_label(exc)})", "WARN")

    board = _make_board(run_dir)
    board.attach()
    state = _ConceptRun(concept, run_dir, notifier, board)

    _resume_banner()
    _notify(notifier, board, f"M2 RUN STARTED - {concept}",
            ("Dead man's switch armed - if these stop arriving, healthchecks.io will tell you."
             if getattr(notifier, "hc", "") else
             "No dead man's switch: a pod that dies outright will not report it."))
    if notifier is not None:
        try:
            notifier.ping("/start")
        except Exception:                               # noqa: BLE001 - best-effort infrastructure
            pass

    cfg = _run().config
    winner: dict | None = None
    verified_rows: list[dict] = []
    control_verdicts: dict[str, str] = {}

    # ---- rig checks. Not a phase: they cost nothing per unit and produce no rows, but they
    # must run BEFORE any sweep. R14 (hook liveness on both the start_pos and the
    # all-positions paths) is the one that matters here - bug 26 left E1 and D1b measuring an
    # unsteered model at all 30 cells, silently, for a full hour.
    state.phase("RIG", lambda: _gates().rig_checks(), tracked=False)

    # ---- Phase 0
    ok_cal, cal = state.phase("CAL", lambda: _phases().phase0_calibrate(), units=1)
    if ok_cal:
        fpr = _judge_fpr_of(cal)
        if fpr is None:
            runio.log("Phase 0 reported no judge_fpr - spec 14.6 rule 5 cannot fire this run",
                      "WARN")
        elif fpr > float(cfg["JUDGE_FPR_MAX"]):
            # Spec 14.7: the earliest possible abort signal. A judge that invents influence on
            # unsteered/unsteered pairs puts a floor under every E5 in the run, and this fires
            # before GPU time is spent on numbers that cannot be trusted.
            _notify(notifier, board, "JUDGE FPR GATE FAILED",
                    f"{concept}: control pairs score {fpr:.2f} > JUDGE_FPR_MAX "
                    f"{cfg['JUDGE_FPR_MAX']}. Every E5 in this run has a floor under it.",
                    severity="warn")
        else:
            _notify(notifier, board, "Phase 0 judge-FPR gate passed", f"judge_fpr {fpr:.2f}")

    if not ok_cal:
        # Every dose in the run is `r * ||h_L|| / ||v_L||`, and both norms come from Phase 0.
        # Without them `alpha_for` raises on every cell, so continuing would produce eight
        # phase failures describing one root cause.
        runio.log("CAL failed - no dose map, so no cell can be measured. Skipping to archive",
                  "ERROR")
        return _finish_concept(state, winner, control_verdicts, wipe=wipe, deliver=deliver,
                               EXPORT_TRANSCRIPTS_OVERRIDE=EXPORT_TRANSCRIPTS_OVERRIDE,
                               status="failed")

    # ---- Phase 1. Sized and ticked by the phase itself: the ETA is worthless without a unit
    # count, and only Phase 1 knows how many of the 98 grid cells a resumed run still owes.
    ok_scan, scan_rows = state.phase(
        "SCAN", lambda: _phases().phase1_scan(on_cell=state.tick("SCAN"),
                                              on_plan=state.plan("SCAN")),
        self_reporting=True)
    if not ok_scan or not scan_rows:
        # Fall back to whatever is on disk: the scan may have died after writing most of its
        # rows, and Phase 2 can shortlist from those.
        scan_rows = runio.rows_for_run(SCAN_FILE)
        runio.log(f"SCAN produced nothing in memory; {len(scan_rows)} rows recovered from "
                  f"{SCAN_FILE}", "WARN")

    # ---- Phase 2
    ok_short, candidates = state.phase(
        "SHORTLIST", lambda: _phases().phase2_shortlist(scan_rows), units=1)
    if ok_short and candidates:
        layers = sorted({int(c["layer"]) for c in candidates
                         if isinstance(c, dict) and "layer" in c})
        span = f"L{layers[0]}-L{layers[-1]}" if layers else "no layer field"
        _notify(notifier, board, "Phase 2 shortlist chosen",
                f"{len(candidates)} candidates, {span}")

    # ---- Phase 3. ASSUMPTION: phase4_verify consumes phase3_bisect's rows.
    ok_bisect, bisect_rows = state.phase(
        "BISECT", lambda: _phases().phase3_bisect(candidates or [],
                                                  on_cell=state.tick("BISECT"),
                                                  on_plan=state.plan("BISECT")),
        self_reporting=True)
    cells = bisect_rows if (ok_bisect and bisect_rows) else runio.rows_for_run(BISECT_FILE)

    # ---- Phases 4 and 5. These are the expensive ones and the two the ETA most needs sized:
    # a verified cell is ~50 s against SCAN's ~13, so a board that cannot count them cannot
    # say anything useful about the hour ahead.
    state.phase("VERIFY", lambda: _phases().phase4_verify(cells or [],
                                                          on_cell=state.tick("VERIFY"),
                                                          on_plan=state.plan("VERIFY")),
                self_reporting=True)
    state.phase("REFINE", lambda: _phases().phase5_refine(runio.rows_for_run(VERIFIED_FILE),
                                                          on_cell=state.tick("REFINE"),
                                                          on_plan=state.plan("REFINE")),
                self_reporting=True)

    # Selection reads the FILE, not either phase's return value: Phase 4 and Phase 5 both
    # write `verified.jsonl`, a resumed run may have rows from a previous kernel, and section
    # 7.1's argmax is over all of them. Reading one phase's return value would silently select
    # over half the evidence.
    verified_rows = runio.rows_for_run(VERIFIED_FILE)
    ok_sel, selection = state.phase(
        "SELECT", lambda: _phases().select_operating_point(verified_rows), tracked=False)
    if not ok_sel:
        selection = None

    # `select_operating_point` returns an ENVELOPE - {found, winner, rule, ...} - deliberately,
    # "so a caller cannot mistake 'no cell qualified' for 'the first row won'". This caller
    # managed to mistake the envelope for the row: it is a dict and it is never None, so
    # `winner is not None` was always true. On this run that meant CONFIRM died on
    # `winner["layer"]`, the controls silently skipped because `_cell_of` could not find a
    # layer either, and operating_point.json was written and announced as OPERATING POINT
    # FOUND while carrying found=False. A successful run would have lost the controls the same
    # way. Unwrap once, here, and let `winner` mean the row.
    winner = None
    if isinstance(selection, dict) and selection.get("found"):
        candidate = selection.get("winner")
        if isinstance(candidate, dict):
            winner = candidate
        else:
            runio.log(f"SELECT reported found=True with a {type(candidate).__name__} winner; "
                      "treating the run as having no operating point", "ERROR")

    qualifying = _qualifying(verified_rows)
    if qualifying:
        _notify(notifier, board, "Phase 4/5 qualifying set",
                f"{len(qualifying)} of {len(verified_rows)} cells qualify")
    else:
        # Spec 14.6 rule 6: not broken, and it should finish - the frontier and the escalation
        # ladder are still the answer to "does an operating point exist for this concept".
        _notify(notifier, board, "NO QUALIFYING CELL",
                f"{concept}: 0 of {len(verified_rows)} verified cells satisfy section 7.1. "
                "The run is not broken; it needs reading.", severity="warn")

    _, front = state.phase("FRONTIER", lambda: _phases().frontier(verified_rows), tracked=False)
    _, margins = state.phase("COVERTNESS",
                             lambda: _phases().covertness_margin(verified_rows), tracked=False)

    # ---- Phase 6. Held-out prompts at fixed N, no adaptive stopping; the only reportable
    # numbers in the run (spec 8).
    confirm = None
    if winner is not None:
        _, confirm = state.phase("CONFIRM", lambda: _phases().phase6_confirm(winner), units=1)
    else:
        runio.log("CONFIRM skipped: no operating point was selected", "WARN")
        # Told to the board too, not just the log. A skipped phase left in `pending` keeps its
        # unit in the ETA forever, so the estimate never reaches zero on a run that is finishing.
        board.skip_phase("CONFIRM", "no operating point was selected")

    # ---- Controls
    def _run_controls() -> dict:
        controls = _controls()
        out: dict[str, Any] = {}
        cell = _cell_of(winner)
        if cell is not None:
            layer, r = cell
            seeds = list(range(int(cfg["N_RANDOM_SEEDS"])))
            # Section 9.1: matched on r, not alpha - a random vector has a different norm, so
            # matching alpha would compare different doses.
            out["random_direction"] = controls.random_direction_control(layer, r, seeds)
            # Section 9.2: the control concept REPLACES the target vector at the same (L, r).
            # Stacking would double the perturbation and lobotomise by construction.
            out["forced_id_capability"] = controls.forced_id_capability_control(layer, r)
        else:
            runio.log("sections 9.1/9.2 skipped: they run at the winning cell and there is "
                      "none", "WARN")
        reach = _best_reach(scan_rows)
        if reach is None or reach < float(cfg["E6_FLOOR"]):
            # Section 9.3: distinguishes "no operating point exists" from "the vector is dead".
            why = "no scan row reported reach" if reach is None else f"best reach {reach:.2f}"
            runio.log(f"escalation ladder: {why} < E6_FLOOR {cfg['E6_FLOOR']}")
            out["escalation"] = controls.escalation_ladder(reference_layer())
        return out

    ok_ctrl, ctrl = state.phase("CONTROLS", _run_controls, units=1)

    # Optional extra arms, run here so their rows land in the bundle: after the winner exists
    # and the controls have judged it, before the gates read the run and the archive closes it.
    # Wrapped in state.phase like everything else, so an arm that dies loses only itself - the
    # operating point is already measured and must not be lost to an optional extra.
    if after_phases is not None and winner is not None:
        state.phase("EXTRA", lambda: after_phases(winner), units=1)

    if ok_ctrl and isinstance(ctrl, dict):
        for key, what in (("random_direction", "section 9.1 random-direction control"),
                          ("forced_id_capability", "section 9.2 forced-ID capability control")):
            if key in ctrl:
                control_verdicts[key] = _control_verdict(ctrl[key], what)
        rejects = [k for k, v in control_verdicts.items() if v == "reject"]
        if len(rejects) >= 2:
            # Spec 14.6 rule 7: both controls rejecting means the apparent result is an
            # artifact, and that verdict IS the finding.
            _notify(notifier, board, "BOTH CONTROLS REJECT THE WINNER",
                    f"{concept}: sections 9.1 and 9.2 both reject. The apparent result is an "
                    "artifact - that is the finding.", severity="warn")

    # ---- Acceptance gates, section 10. After the data exists and before the archive.
    _, gate_report = state.phase("GATES", lambda: _gates().run_acceptance_gates(), tracked=False)

    # ---- The answer. Written whether or not a cell qualified: "no operating point exists at
    # these constraints" is a real result (spec 9.3), and the envelope carries the reason, the
    # counts and the rule that produced it. `found` is the field to read. Only a real winner is
    # announced as one - the phone message used to say OPERATING POINT FOUND over found=False.
    payload = dict(winner) if winner is not None else {}
    payload.update(
        concept=concept,
        found=bool(winner is not None),
        selection=selection if isinstance(selection, dict) else None,
        frontier=front if isinstance(front, list) else [],
        covertness_margin=margins if isinstance(margins, list) else [],
        confirm=confirm,
        controls=ctrl if isinstance(ctrl, dict) else {},
        control_verdicts=control_verdicts,
        gates=gate_report,
        failed_phases=list(state.failed),
    )
    try:
        path = runio.write_json(OPERATING_POINT_FILE, payload)
        runio.log(f"operating point -> {path.name}"
                  if winner is not None else
                  f"no operating point; the reason is in {path.name} (found=false)")
        if winner is not None:
            _notify(notifier, board, "OPERATING POINT FOUND", _one_line(payload))
    except Exception as exc:                            # noqa: BLE001
        runio.log(f"operating_point.json write failed ({_label(exc)})", "ERROR")

    status = "ok" if not state.failed else "failed"
    return _finish_concept(state, winner, control_verdicts, wipe=wipe, deliver=deliver,
                           EXPORT_TRANSCRIPTS_OVERRIDE=EXPORT_TRANSCRIPTS_OVERRIDE,
                           status=status)


def _one_line(point: dict) -> str:
    """One readable line for the phone: (L, alpha, r), E5, D2, sanity, control verdicts."""
    bits = []
    for key, fmt in (("layer", "L{}"), ("alpha", "alpha={:.3f}"), ("r", "r={:.3f}"),
                     ("e5", "E5={:.2f}"), ("d2", "D2={:.2f}"), ("s4", "S4={:.2f}")):
        if key in point and point[key] is not None:
            try:
                bits.append(fmt.format(point[key]))
            except (TypeError, ValueError):
                bits.append(f"{key}={point[key]}")
    verdicts = point["control_verdicts"] if "control_verdicts" in point else {}
    if verdicts:
        bits.append("controls: " + ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
    return " | ".join(bits)


def _resume_banner() -> None:
    """Say what is already on disk, so a resumed run is visibly a resumed run.

    v1's silent resume was fine until a folder held rows from a half-finished previous
    attempt, at which point "why is Phase 1 finishing in four seconds" had no answer in the
    log. Cheap to print, and it names the exact files spec 14.8 resumes from.
    """
    counts = []
    for name in (SCAN_FILE, BISECT_FILE, VERIFIED_FILE, CONFIRM_FILE, CONTROLS_FILE):
        try:
            n = len(runio.rows_for_run(name))
        except Exception:                               # noqa: BLE001 - a banner must not fail
            continue
        if n:
            counts.append(f"{name} {n}")
    runio.log("resume: " + (", ".join(counts) if counts else "no rows on disk yet"))


def _finish_concept(state: _ConceptRun, winner: dict | None, control_verdicts: dict,
                    *, wipe: bool, deliver: bool, EXPORT_TRANSCRIPTS_OVERRIDE: bool,
                    status: str) -> dict:
    """Write the run record, archive, export, deliver, and only then wipe.

    The order is the whole point and is spec 14.9's required fix:

        archive -> export bundle -> send -> VERIFY the send -> wipe

    v1 archived, wiped, then attempted the send; a Telegram outage made `Wrists` and `Wonder`
    unrecoverable. `runio.deliver_then_wipe` owns the verify-before-wipe rule; this function
    owns the ordering and the guarantee that the archive exists before anything is deleted.
    """
    elapsed = time.time() - state.t0
    record = dict(
        concept=state.concept,
        run_dir=str(state.run_dir),
        status=status,
        failed_phases=list(state.failed),
        failure_labels=dict(state.labels),
        control_verdicts=dict(control_verdicts),
        operating_point=winner,
        elapsed_s=round(elapsed, 1),
    )
    try:
        runio.write_json(RUN_RECORD_FILE, record)
    except Exception as exc:                            # noqa: BLE001
        runio.log(f"run record not written ({_label(exc)})", "WARN")

    banner = ("M2 CONCEPT FINISHED - " + state.concept if not state.failed
              else "M2 CONCEPT FINISHED WITH FAILURES - " + state.concept)
    extra = (f"{len(state.failed)} phase(s) failed: {', '.join(state.failed)}. "
             if state.failed else "")
    _notify(state.notifier, state.board,
            banner, extra + f"{elapsed / 60:.1f} min. Concept done - the batch may still be "
                            "running, do NOT stop the pod yet.",
            severity=("warn" if state.failed else "info"))
    state.board.detach()

    archive_path = None
    bundle_path = None
    delivered = False
    try:
        archive_path = runio.archive_concept(state.run_dir)
    except Exception as exc:                            # noqa: BLE001
        runio.log(f"archive FAILED for {state.concept} ({_label(exc)}) - the loose folder is "
                  "kept and nothing is deleted", "ERROR")

    if archive_path is not None:
        try:
            bundle_path = runio.export_bundle(
                state.run_dir, EXPORT_TRANSCRIPTS_OVERRIDE=EXPORT_TRANSCRIPTS_OVERRIDE)
        except Exception as exc:                        # noqa: BLE001
            runio.log(f"export bundle FAILED for {state.concept} ({_label(exc)})", "ERROR")

    if bundle_path is not None and deliver:
        try:
            delivered = runio.deliver_then_wipe(
                bundle_path, state.notifier, wipe=wipe, run_dir=state.run_dir,
                caption=f"{state.concept} - M2 run bundle ({status})")
        except Exception as exc:                        # noqa: BLE001 - nothing was deleted
            runio.log(f"delivery raised {_label(exc)} - folder kept, nothing deleted", "ERROR")
    elif bundle_path is not None:
        runio.log("delivery disabled for this run - bundle written, loose folder kept")

    record.update(archive=str(archive_path) if archive_path else None,
                  bundle=str(bundle_path) if bundle_path else None,
                  delivered=delivered)
    return record


# =====================================================================================
# The pod stop
# =====================================================================================

def kill_pod(reason: str = "") -> bool:
    """STOP (not terminate) this RunPod pod. The volume and every archive survive.

    Ported from the v1 lab (lab_cells.py cell 11) unchanged in behaviour: `runpodctl` first,
    then the GraphQL API with `RUNPOD_API_KEY`. STOP and never TERMINATE -- a terminated pod
    takes the volume with it, and the volume is where every archive and every undelivered
    bundle lives.

    Returns True if a stop was issued. Never raises: this runs after the science is finished
    and an exception here would be the last thing in the log.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    runio.log(f"POD STOP requested ({reason}); pod={pod_id or 'unknown'}")
    if not pod_id:
        runio.log("RUNPOD_POD_ID is not set - stop the pod manually in the console", "WARN")
        return False

    try:
        done = subprocess.run(["runpodctl", "stop", "pod", pod_id],
                              capture_output=True, text=True, timeout=60)
        if done.returncode == 0:
            runio.log("pod stop issued via runpodctl")
            return True
    except Exception:                                   # noqa: BLE001 - fall through to the API
        pass

    key = os.environ.get("RUNPOD_API_KEY", "")
    if key:
        body = json.dumps({
            "query": 'mutation { podStop(input: {podId: "%s"}) { id desiredStatus } }' % pod_id
        }).encode()
        request = urllib.request.Request(
            "https://api.runpod.io/graphql", data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read()
            runio.log("pod stop issued via RunPod API")
            return True
        except Exception as exc:                        # noqa: BLE001 - label only
            runio.log(f"pod stop API failed: {_label(exc)}", "ERROR")

    runio.log("POD STOP FAILED - stop it manually in the RunPod console", "ERROR")
    return False


def _drain(notifier: Any, seconds: float) -> None:
    """Wait, bounded, for the notifier's queue to empty. Never raises."""
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline:
        try:
            pending = getattr(getattr(notifier, "_q", None), "unfinished_tasks", 0)
        except Exception:                               # noqa: BLE001
            return
        if not pending:
            return
        time.sleep(1.0)


# =====================================================================================
# run_batch
# =====================================================================================

def _compromised(result: dict) -> bool:
    """Does this concept count toward `FATAL_CONSECUTIVE_D4S`?

    Three ways, and all three are structural rather than unlucky:

    - the concept raised out of `run_concept` entirely;
    - two or more of its phases failed -- spec 14.6 rule 2's reading of "structural: OOM,
      judge auth, bad install", not one unlucky phase;
    - its section 9.2 control rejected, i.e. the D4 distribution is dominated by damage modes.
      That is what the constant is named after: three concepts in a row whose forced-ID
      pathway is broken is the apparatus, not the concepts.

    An 'unknown' control verdict does NOT count. It is already reported as a warning, and
    aborting a 21-concept batch on an unreadable field would be a worse failure than running
    it and reading the manifest afterwards.
    """
    if result.get("status") == "failed_hard":
        return True
    if len(result.get("failed_phases") or []) >= 2:
        return True
    return (result.get("control_verdicts") or {}).get("forced_id_capability") == "reject"


def run_batch(concepts: Sequence[str], *, stop_pod: bool = True, wipe: bool = True,
              EXPORT_TRANSCRIPTS_OVERRIDE: bool = False, after_phases: Any = None) -> dict:
    """Run the full pipeline over `concepts`, isolated per concept. Spec 14.9.

    - a concept whose archive already exists is skipped, not re-run;
    - one concept failing does not abort the batch;
    - `FATAL_CONSECUTIVE_D4S` compromised concepts in a row does abort it -- that is
      structural, and the remaining pod-hours are better spent on a fixed pipeline;
    - undelivered bundles are retried once the loop is over, before anything stops;
    - `KILL_GRACE_SECONDS` after the final message the pod is STOPPED (not terminated), so
      queued sends drain and the volume, the archives and any undelivered bundle survive.

    `EXPORT_TRANSCRIPTS_OVERRIDE` applies to every concept in this call and is announced at
    the start of the batch. Spec 14.3: a non-benign concept's transcripts are withheld from
    the bundle without it, and the switch cannot be set from CONFIG or the environment, so a
    changed concept list cannot inherit a `True` from the benign arm.
    """
    names = [str(c).strip() for c in concepts if str(c).strip()]
    if not names:
        raise ValueError("run_batch needs at least one concept")

    notifier = get_notifier()
    cfg = _run().config or config.CONFIG
    fatal_after = int(cfg["FATAL_CONSECUTIVE_D4S"])
    grace = float(cfg["KILL_GRACE_SECONDS"])

    # State the export posture once, at the top, where it is read rather than discovered.
    non_benign = [c for c in names if not config.is_benign(c)]
    if non_benign:
        runio.log(f"{len(non_benign)} of {len(names)} concepts are NOT on BENIGN_CONCEPTS "
                  f"({', '.join(non_benign)}): their transcripts are "
                  + ("INCLUDED by explicit override (spec 14.3)"
                     if EXPORT_TRANSCRIPTS_OVERRIDE else "withheld from the export bundle"),
                  "WARN" if EXPORT_TRANSCRIPTS_OVERRIDE else "INFO")

    t0 = time.time()
    done: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    records: list[dict] = []
    consecutive = 0
    fatal = False

    BATCH_STATE.update(total=len(names), done=0, durations=[], cur_t0=time.time(),
                       concept=None)
    runio.log("#" * 70)
    runio.log(f"BATCH START: {len(names)} concepts -> {', '.join(names)}")

    try:
        for index, concept in enumerate(names):
            runio.log("#" * 70)
            runio.log(f"BATCH CONCEPT {index + 1}/{len(names)}: {concept}")
            BATCH_STATE["cur_t0"] = time.time()
            BATCH_STATE["concept"] = concept
            try:
                record = run_concept(
                    concept, notifier=notifier, wipe=wipe, deliver=True,
                    EXPORT_TRANSCRIPTS_OVERRIDE=EXPORT_TRANSCRIPTS_OVERRIDE,
                    after_phases=after_phases)
            except Exception as exc:                    # noqa: BLE001 - per-concept isolation
                record = dict(concept=concept, status="failed_hard",
                              failed_phases=["<concept>"], failure_labels={"<concept>": _label(exc)},
                              control_verdicts={}, elapsed_s=time.time() - BATCH_STATE["cur_t0"])
                runio.log(f"CONCEPT {concept} FAILED: {_label(exc)}", "ERROR")
                print(traceback.format_exc())
                if notifier is not None:
                    try:
                        notifier.send(f"CONCEPT {concept} FAILED: {_label(exc)}. "
                                      "Batch continues.")
                    except Exception:                   # noqa: BLE001
                        pass
            finally:
                # Free VRAM between concepts. A 27B model plus a batch of KV caches leaves
                # little headroom, and the next concept starts with an extraction pass.
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:                       # noqa: BLE001 - no torch, or no CUDA
                    pass
                BATCH_STATE["durations"].append(time.time() - BATCH_STATE["cur_t0"])
                BATCH_STATE["done"] += 1

            records.append(record)
            status = record.get("status")
            if status == "skipped":
                skipped.append(concept)
            elif status in ("failed", "failed_hard"):
                failed.append(concept)
            else:
                done.append(concept)

            consecutive = consecutive + 1 if _compromised(record) else 0
            if consecutive >= fatal_after:
                fatal = True
                runio.log(f"FATAL: {consecutive} concepts compromised in a row - aborting the "
                          "batch", "ERROR")
                if notifier is not None:
                    try:
                        notifier.send(f"FATAL: {consecutive} concepts compromised in a row - "
                                      "aborting the whole batch.")
                    except Exception:                   # noqa: BLE001
                        pass
                break
    except BaseException as exc:                        # noqa: BLE001 - including KeyboardInterrupt
        fatal = True
        runio.log(f"BATCH-LEVEL FATAL: {_label(exc)}", "ERROR")
        print(traceback.format_exc())

    # Retry every bundle whose delivery was never confirmed, BEFORE the pod stops. This is the
    # second half of spec 14.9's fix: the loose folders are all still on disk because nothing
    # wipes on an unverified send, so a late-recovering Telegram still gets them.
    retry = runio.retry_undelivered(notifier, wipe=wipe)
    still = runio.undelivered()

    elapsed = time.time() - t0
    head = "BATCH ABORTED (fatal)" if fatal else "BATCH FINISHED"
    summary = (f"{head}: {len(done)}/{len(names)} concepts in {elapsed / 60:.1f} min."
               + (f" Skipped: {', '.join(skipped)}." if skipped else "")
               + (f" Failed: {', '.join(failed)}." if failed else "")
               + (f" UNDELIVERED (kept on the volume): "
                  f"{', '.join(e.get('concept', '?') for e in still)}." if still else ""))
    runio.log("#" * 70)
    runio.log(summary)

    if notifier is not None:
        try:
            notifier.send(summary + (f" Stopping the pod in {grace:.0f}s (volume preserved)."
                                     if stop_pod else " Safe to stop the pod."))
        except Exception:                               # noqa: BLE001
            pass
        _drain(notifier, min(90.0, grace))

    if stop_pod:
        # The grace period is for the queued sends AND for any upload still in flight; the
        # drain above only sees the queue. Sleep it out in full before the stop.
        runio.log(f"auto-stop armed: {grace:.0f}s grace, then STOP the pod (volume preserved)")
        time.sleep(max(0.0, grace))
        kill_pod(reason=("fatal abort" if fatal else "batch complete"))
    else:
        runio.log("auto-stop is OFF - the pod keeps running")

    return dict(done=done, skipped=skipped, failed=failed, fatal=fatal,
                elapsed_s=round(elapsed, 1), records=records, retry=retry,
                undelivered=[e.get("concept") for e in still])
