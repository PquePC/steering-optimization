"""m2.monitor -- the status board, the phone alerts, the dead man's switch, the verdicts.

Ported from the v1 measurement lab (`measurement_lab.ipynb` cell 13 == `lab_cells.py` cell
13), which ran unattended on the pod for a full six-concept batch. **Spec section 14 says
the design here is load-bearing and every failure mode it covers happened at least once**,
so this file changes v1 only where M2 changed the pipeline underneath it. Where it does
change, the comment says which part of section 14 forced it.

Three independent layers, each covering a failure the others cannot see (spec 14.1):

  * the **board** (`RunStatus`) -- slow, stalled or failing measurement, in the kernel;
  * the **push** (`Notifier`) -- the same information delivered to a phone;
  * the **dead man's switch** (`Notifier.ping_now` -> healthchecks.io) -- the pod dying
    outright. Nothing running on the pod can report its own death: a stopped pod, a killed
    kernel and a lost network all look identical from inside, which is silence. Only an
    external service noticing that pings *stopped* covers that case.

(The fourth layer, `pod_watchdog.sh`, is deliberately a separate OS process and is not in
this file: a hung kernel holds the GIL, so the heartbeat thread below stops running too.)

WHAT MAY LEAVE THE POD (spec 14.3, channel 1). Phase names, states, unit counts, elapsed,
ETA, verdict level, metric values, and a CLASSIFIED exception label. **Never an exception
message, never a traceback.** An API error can quote its request payload back at you, and
under M2 that payload is a steered generation or a judge prompt containing one -- arriving
unbidden, in a message nobody chose to send. Detail stays in the crash file on the volume;
`classify_exc` is the only path from an exception to the wire.

PUSH ONLY. The pod talks, it never listens: no polling loop, no command channel, nothing
that can start, stop or alter a run from outside (spec 14.2, and CLAUDE.md hard rule 2).
Do not add one to M2.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import config

__all__ = [
    "fmt_time",
    "log",
    "Progress",
    "classify_exc",
    "Notifier",
    "NOTIFY",
    "notify_test",
    "PHASE_ORDER",
    "PHASE_SECONDS_PRIOR",
    "PHASE_UNITS_PRIOR",
    "RunStatus",
    "verdict",
    "verdict_detail",
    "tee_stdout",
    "untee",
]


# =====================================================================================
# Live configuration
# =====================================================================================

def _cfg() -> dict:
    """The live config: `RUN.config` once a concept is set, else the module `CONFIG`.

    Read at call time, never captured at import. `driver.set_concept` rebinds
    `RUN.config`, and a captured reference would keep serving the pre-batch values -- the
    same stale-reference shape as bug 23.
    """
    run_cfg = getattr(config.RUN, "config", None)
    return run_cfg if run_cfg else config.CONFIG


def _concept() -> str:
    """The concept being measured, for the board header. Display only.

    Membership tests rather than defaulted `.get`s, per the house rule -- and because
    `<unset>` on a board is a visible "the driver has not called set_concept yet", whereas
    a silently blank header reads as a rendering glitch.
    """
    name = getattr(config.RUN, "concept", None)
    if name:
        return str(name)
    cfg = _cfg()
    return str(cfg["concept"]) if "concept" in cfg else "<unset>"


def _config_hash() -> str:
    """The config hash, for the board header.

    The membership test here is NOT a load-bearing default: nothing computes from this
    string, it labels a board. Every measurement path hard-indexes `config_hash` instead.
    """
    cfg = _cfg()
    return str(cfg["config_hash"]) if "config_hash" in cfg else "<no hash>"


def _run_dir() -> Path | None:
    d = getattr(config.RUN, "run_dir", None)
    return None if d is None else Path(d)


# =====================================================================================
# Logging
# =====================================================================================

def fmt_time(seconds: float | int | None) -> str:
    """`1h04m09s` / `4m09s`. `??` for None, NaN or an infinity.

    An ETA is routinely `nan` before the first unit completes and `inf` when a rate is
    zero; both must print as an honest `??` rather than crash the board or, worse, render
    as `0m00s` and read as "about to finish".
    """
    if seconds is None:
        return "??"
    value = float(seconds)
    if value != value or value in (float("inf"), float("-inf")):
        return "??"
    minutes, secs = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def log(msg: str, level: str = "INFO") -> None:
    """Screen, plus this concept's `lab.log` and the batch-stable `batch.log`.

    File writes are best-effort on purpose. In v1 the batch died on concept 1 because the
    per-concept folder had just been wiped and the next log line raised into the driver.
    Logging must never be able to fail a run.
    """
    print(f"{time.strftime('%H:%M:%S')} [{level:<5}] {msg}", flush=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}" + chr(10)
    run_dir = _run_dir()
    targets: tuple[Path, ...] = ()
    if run_dir is not None:
        # batch.log lives one level up so it survives the per-concept wipe and carries the
        # whole batch's history in one file.
        targets = (run_dir / "lab.log", run_dir.parent / "batch.log")
    for path in targets:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:            # noqa: BLE001 - see the docstring
            pass


class Progress:
    """Position and estimated time remaining for one loop, at most every `every` seconds.

    Throttled because the interesting loops here are 100+ scan cells: an unthrottled line
    per cell buries the status board it is meant to complement.
    """

    def __init__(self, total: int, label: str, every: float = 20.0) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.every = float(every)
        self.done = 0
        self.t0 = time.time()
        self.last = 0.0
        self._emit()

    def update(self, n: int = 1, **info: Any) -> None:
        self.done += n
        now = time.time()
        if now - self.last >= self.every or self.done >= self.total:
            self.last = now
            self._emit(**info)

    def _emit(self, **info: Any) -> None:
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        extra = " | ".join(f"{k}={v}" for k, v in info.items())
        log(f"[{self.label}] {self.done}/{self.total} "
            f"({100 * self.done / self.total:5.1f}%) | {fmt_time(elapsed)} elapsed | "
            f"ETA {fmt_time(eta)}" + (f" | {extra}" if extra else ""))


# =====================================================================================
# Exception classification -- the only path from an exception to the wire
# =====================================================================================
# Substring -> label, matched against the LOWERCASED exception message. Only the label is
# ever transmitted. Anything unmatched degrades to the exception class name alone, which is
# safe by construction: a class name is code, not data (spec 14.3).

_EXC_LABELS: tuple[tuple[str, str], ...] = (
    ("out of memory",  "CUDA out of memory"),
    ("cuda",           "CUDA error"),
    ("401",            "judge auth rejected"),
    ("403",            "judge auth rejected"),
    ("429",            "judge rate limited"),
    ("insufficient",   "judge account out of credit"),
    ("quota",          "judge quota exhausted"),
    ("connection",     "network error"),
    ("timed out",      "timed out"),
    ("timeout",        "timed out"),
    ("no such file",   "missing file"),
    # M2 additions, for the failures this pipeline can have that v1 could not.
    ("unreachable",    "dose above ALPHA_CEIL"),
    ("padding_side",   "tokenizer padding is not left"),
    ("hook",           "steering hook dead"),
)


def classify_exc(exc: BaseException) -> str:
    """A short LABEL for an exception. Never the message -- spec 14.3, channel 1.

    The message is the untrusted part: it can carry a quoted request body, and under M2 that
    body can carry a steered generation or a judge prompt containing one. The class name
    cannot. So the class name always goes out, and a phrase is added only where a
    known-safe substring matches.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    for needle, label in _EXC_LABELS:
        if needle in msg:
            return f"{name} ({label})"
    return name


# =====================================================================================
# Notifier
# =====================================================================================
# One queue.Queue drained by one daemon thread (spec 14.4). Three properties, each of
# which cost something in v1 when it was absent:
#
#   ORDERED      -- a send fired in its own thread arrives out of order, so "STOP THE POD"
#                   can land before the "started" message it followed.
#   NON-BLOCKING -- a send fired inline blocks the measuring thread for a full TCP timeout.
#   NEVER RAISES -- a failed alert must never become a failed run; failures increment
#                   `dropped` and nothing else happens.

# Files that must never leave the pod, whatever a caller passes (CLAUDE.md hard rules 1
# and 3). This is a TRANSPORT BACKSTOP, not the policy: `runio.EXPORT_DENY` is the policy
# and the bundle filter is where it is applied. Duplicating the extensions here is
# deliberate -- the last check before the wire should not depend on a caller having applied
# the first one.
_NEVER_SEND_SUFFIXES: tuple[str, ...] = (
    ".pt", ".safetensors", ".npy", ".npz", ".bin", ".gguf", ".ckpt",
)


class Notifier:
    """Ordered, non-blocking outbound alerts. Never raises, never delays the run."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        self.sent = 0
        self.dropped = 0
        self.last_board = 0.0
        self.token = ""
        self.chat = ""
        self.hc = ""
        self.enabled = False
        self.warnings_only = False
        self.reload()

    def reload(self) -> bool:
        """Re-read the environment. Call this if the keys were set after setup ran.

        `TELEGRAM_WARNINGS_ONLY` suppresses routine (`info`) pushes; warnings and stops
        always go. In a long batch that is the intended way to silence the per-phase
        completions of spec 14.7 without losing the ones that matter.
        """
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.hc = os.environ.get("HEALTHCHECK_URL", "").strip().rstrip("/")
        self.enabled = bool(self.token and self.chat)
        self.warnings_only = os.environ.get("TELEGRAM_WARNINGS_ONLY", "").strip().lower() in (
            "1", "true", "yes", "on")
        if (self.enabled or self.hc) and self._worker is None:
            self._worker = threading.Thread(target=self._pump, daemon=True)
            self._worker.start()
        return self.enabled

    # ---- transport ------------------------------------------------------------------
    def _pump(self) -> None:
        handlers: dict[str, Callable[[Any], None]] = {
            "msg": self._post, "ping": self._ping, "doc": self._send_document}
        while True:
            kind, payload = self._q.get()
            try:
                handlers[kind](payload)
                self.sent += 1
            except Exception:            # noqa: BLE001
                self.dropped += 1        # a failed alert must never become a failed run
            finally:
                self._q.task_done()

    def _post(self, text: str) -> None:
        data = urllib.parse.urlencode({
            "chat_id": self.chat,
            "text": text[:3500],
            "disable_web_page_preview": "true"}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendMessage", data=data)
        with urllib.request.urlopen(req, timeout=20) as r:   # noqa: S310 - fixed https host
            r.read()

    def _ping(self, suffix: str) -> None:
        """Fire the dead man's switch without ever wedging the queue.

        An unreachable healthcheck host hangs in DNS RESOLUTION, which the socket timeout
        does not cover -- and because one worker drains one queue, a hung ping would stall
        every Telegram message behind it (that is what swallowed the run-start message in
        v1). So the request runs in a throwaway daemon thread with a hard 15 s wall-clock
        cap: if it does not return we abandon it and raise, and we keep at most one such
        thread alive so a dead healthcheck cannot leak a thread on every beat.
        """
        if self._ping_thread is not None and self._ping_thread.is_alive():
            raise TimeoutError("previous healthcheck ping still hung - skipping this one")
        box: dict[str, BaseException] = {}

        def _do() -> None:
            try:
                with urllib.request.urlopen(   # noqa: S310 - operator-supplied https URL
                        self.hc + (suffix or ""), timeout=12) as r:
                    r.read()
            except BaseException as exc:   # noqa: BLE001 - carried out so a bad ping is a drop
                box["err"] = exc

        self._ping_thread = threading.Thread(target=_do, daemon=True)
        self._ping_thread.start()
        self._ping_thread.join(15)
        if self._ping_thread.is_alive():
            raise TimeoutError("healthcheck ping did not return within 15s - abandoned")
        if "err" in box:
            raise box["err"]

    def _send_document(self, payload: tuple[str, str]) -> None:
        """Upload one file to the chat.

        Spec 14.3 channel 2 allows the full per-concept bundle, transcripts included, FOR
        BENIGN CONCEPTS ONLY. That gate is `runio.export_bundle`, which builds the zip; this
        function only moves bytes, and refuses the weight/vector extensions as a last check
        (see `_NEVER_SEND_SUFFIXES`).
        """
        path, caption = payload
        with open(path, "rb") as fh:
            content = fh.read()
        boundary = "----m2" + os.urandom(8).hex()
        fname = os.path.basename(str(path))
        head: list[bytes] = []

        def _field(name: str, value: Any) -> None:
            head.append(("--" + boundary + "\r\n"
                         'Content-Disposition: form-data; name="' + name + '"\r\n\r\n'
                         + str(value) + "\r\n").encode())

        _field("chat_id", self.chat)
        if caption:
            _field("caption", caption[:1000])
        head.append(("--" + boundary + "\r\n"
                     'Content-Disposition: form-data; name="document"; filename="'
                     + fname + '"\r\n'
                     "Content-Type: application/octet-stream\r\n\r\n").encode())
        body = b"".join(head) + content + ("\r\n--" + boundary + "--\r\n").encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendDocument", data=body,
            headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        with urllib.request.urlopen(req, timeout=120) as r:   # noqa: S310 - fixed https host
            r.read()

    # ---- public ---------------------------------------------------------------------
    def send(self, text: str) -> None:
        """Queue a plain message. Silently does nothing when alerts are off."""
        if self.enabled:
            self._q.put(("msg", text))

    def send_file(self, path: str | Path, caption: str = "") -> bool:
        """Queue a file for the chat. Returns whether it was QUEUED, not whether it landed.

        `deliver_then_wipe` must not treat this as delivery -- use `send_file_now`, which
        reports the transport result. v1 wiped a concept's folder on the strength of a queued
        send and lost `Wrists` and `Wonder` to a Telegram outage (spec 14.9).
        """
        if not self._sendable(path):
            return False
        self._q.put(("doc", (str(path), caption)))
        return True

    def send_file_now(self, path: str | Path, caption: str = "",
                      drain_first: float = 150.0) -> bool:
        """Send a file INLINE and report whether the transport succeeded. Never raises.

        This is the one deliberate exception to "everything goes through the queue", and it
        exists for exactly one caller: `runio.deliver_then_wipe`, whose order is
        archive -> send -> **verify the send succeeded** -> wipe. A queued send cannot be
        verified without racing the pump, and the failure mode of guessing is data loss.

        The queue is drained first so this document still arrives after the messages that
        preceded it -- ordering is preserved, only the confirmation is synchronous.
        """
        if not self._sendable(path):
            return False
        self.drain(drain_first)
        try:
            self._send_document((str(path), caption))
            self.sent += 1
            return True
        except Exception:            # noqa: BLE001 - a failed alert is never a failed run
            self.dropped += 1
            return False

    def can_deliver(self) -> bool:
        """Whether a file send can be ATTEMPTED at all (a chat is configured).

        `runio.deliver_then_wipe` must check this before reading a `False` from
        `send_file_now` as a delivery failure -- and must not wipe when it returns False.
        "There is no channel" and "the channel failed" both mean the bundle has not left the
        pod, and wiping on either is the v1 loss of `Wrists` and `Wonder` in another costume.
        """
        return self.enabled

    def _sendable(self, path: str | Path) -> bool:
        """Alerts enabled, the file exists, and it is not a weights/vector artefact."""
        if not self.enabled:
            return False
        p = Path(path)
        if not p.exists():
            log(f"notifier: {p.name} does not exist - not sent", "WARN")
            return False
        if p.suffix.lower() in _NEVER_SEND_SUFFIXES:
            # CLAUDE.md hard rules 1 and 3. These regenerate from a published config in
            # minutes; regeneration is the backup, and there is no reason to move them.
            log(f"notifier: REFUSING to send {p.name} - {p.suffix} is a weights/vector "
                "artefact and must not leave the pod", "ERROR")
            return False
        return True

    def ping(self, suffix: str = "") -> None:
        """Tell the dead man's switch we are alive, VIA THE QUEUE. '/start', '' or '/fail'."""
        if self.hc:
            self._q.put(("ping", suffix))

    def ping_now(self, suffix: str = "") -> None:
        """Fire the dead man's switch DIRECTLY (bounded), outside the send queue.

        Deliberate bypass, for two reasons: a backed-up Telegram queue can never delay the
        liveness signal, and a ping goes out *iff the heartbeat thread is actually running*
        -- which is the thing being attested (spec 14.4).
        """
        if self.hc:
            try:
                self._ping(suffix)
            except Exception:        # noqa: BLE001
                pass

    def drain(self, seconds: float) -> bool:
        """Wait up to `seconds` for the queue to empty. True if it did.

        Used before a wipe and before stopping the pod, so queued sends are not killed in
        flight. Polls rather than `Queue.join()` because join has no timeout and a hung
        send would turn a graceful shutdown into a hang.
        """
        deadline = time.time() + max(0.0, float(seconds))
        while time.time() < deadline:
            if getattr(self._q, "unfinished_tasks", 0) == 0:
                return True
            time.sleep(0.5)
        return getattr(self._q, "unfinished_tasks", 0) == 0

    def board(self, status: "RunStatus", banner: str, extra: str = "",
              severity: str = "info") -> None:
        """Push the WHOLE BOARD under a one-line banner.

        Every notification carries the board rather than a summary line (spec 14.4). The
        board answers the follow-up question -- which phase, how far in, how fast, what the
        ETA is now -- and a message that prompts a follow-up you cannot make is a bad
        message when the only way to look deeper is to open a laptop.

        Sending resets the slow-beat timer, so an eventful run does not also get beats.
        """
        if not self.enabled:
            return
        if self.warnings_only and severity != "warn":
            return
        self.last_board = time.time()
        try:
            body = status.phone_text()
        except Exception as exc:     # noqa: BLE001 - a broken board must still alert
            body = f"(board unavailable: {classify_exc(exc)})"
        self.send(banner + ((chr(10) + extra) if extra else "") + chr(10) * 2 + body)

    # ---- the spec 14.7 notification points -------------------------------------------
    def run_started(self, status: "RunStatus") -> None:
        # The human alert is enqueued BEFORE the healthcheck ping, so "run starting" can
        # never sit behind - or be lost to - a slow or unreachable dead man's switch. The
        # ping is best-effort infrastructure; the message is the point.
        self.board(status, f"M2 RUN STARTED - {_concept()}",
                   ("Dead man's switch armed - if these stop arriving, healthchecks.io "
                    "will tell you." if self.hc else
                    "No dead man's switch: a pod that dies outright will not report it. "
                    "You would just stop hearing from it."))
        self.ping("/start")

    def phase_completed(self, status: "RunStatus", phase: str) -> None:
        """Spec 14.7 row 2. Routine, so warnings-only mode suppresses it.

        v1 deliberately did NOT push per-measure completion because in a batch it produced a
        burst of messages for the last, fast measures. M2 keeps the push because the spec
        lists it, and points a long batch at `TELEGRAM_WARNINGS_ONLY` instead -- which is
        what the spec's "(suppressed under warnings-only)" means.
        """
        self.board(status, f"PHASE DONE - {phase} ({_concept()})")

    def judge_fpr_gate(self, status: "RunStatus", fpr: float, limit: float,
                       passed: bool) -> None:
        """Spec 14.7: the Phase 0 judge-FPR gate. The earliest possible abort signal.

        A judge that invents influence on unsteered/unsteered control pairs puts a floor
        under every E5 in the run, so this fires before any GPU time is spent on numbers
        that cannot be trusted (spec 5.8, 14.6 rule 5).
        """
        if passed:
            self.board(status, f"PHASE 0 GATE PASSED - judge FPR {fpr:.2f} <= {limit:.2f}")
        else:
            self.board(status,
                       f"PHASE 0 GATE FAILED - judge FPR {fpr:.2f} > {limit:.2f}",
                       "The judge scores influence on unsteered pairs, which puts a floor "
                       "under every E5 in this run. Nothing measured after this is "
                       "trustworthy until it is fixed.", severity="warn")

    def shortlist_chosen(self, status: "RunStatus", n: int,
                         layer_lo: int | None = None, layer_hi: int | None = None) -> None:
        """Spec 14.7: Phase 2 shortlist chosen -- candidate count and layer range."""
        span = ("" if layer_lo is None or layer_hi is None
                else f" spanning L{layer_lo}-L{layer_hi}")
        self.board(status, f"SHORTLIST CHOSEN - {n} candidates{span}")

    def qualifying_set(self, status: "RunStatus", n_qualifying: int,
                       n_cells: int | None = None) -> None:
        """Spec 14.7: the Phase 4 qualifying set. Warn if it is empty."""
        of = "" if n_cells is None else f" of {n_cells} verified"
        if n_qualifying > 0:
            self.board(status, f"QUALIFYING SET - {n_qualifying} cells{of}")
        else:
            self.board(status, f"QUALIFYING SET IS EMPTY - 0 cells{of}",
                       "No cell satisfies E5 >= floor, D2 <= max and S4 >= min. The run is "
                       "not broken and should finish: the frontier and the escalation "
                       "ladder are still the answer to whether an operating point exists "
                       "for this concept.", severity="warn")

    def operating_point(self, status: "RunStatus", point: Mapping[str, Any]) -> None:
        """Spec 14.7: operating point found -- (L, alpha, r), E5, D2, sanity, controls.

        Metric VALUES are explicitly transmittable under spec 14.3 channel 1. Formatting is
        membership-tested rather than hard-indexed: a notification must never raise, and a
        missing display field is a dash, not an exception in the middle of a run.
        """
        def _fmt(key: str, spec: str = ".3f") -> str:
            if key not in point or point[key] is None:
                return "-"
            value = point[key]
            return format(value, spec) if isinstance(value, (int, float)) else str(value)

        lines = [
            f"L{_fmt('layer', 'd')}  alpha {_fmt('alpha', '.3f')}  r {_fmt('r', '.3f')}",
            f"E5 {_fmt('e5', '.2f')}   D2 {_fmt('d2', '.3f')}   S4 {_fmt('s4', '.3f')}",
        ]
        if "controls" in point and point["controls"] is not None:
            lines.append("controls: " + str(point["controls"])[:120])
        self.board(status, f"OPERATING POINT FOUND - {_concept()}", chr(10).join(lines))

    def phase_failed(self, status: "RunStatus", phase: str, exc: BaseException) -> None:
        """Sent the moment a phase dies, not at the next beat.

        By design the run continues after one phase fails, so this is information rather
        than an instruction -- the verdict decides whether it is worth stopping the pod.
        """
        self.board(status, f"PHASE DIED - {phase}: {classify_exc(exc)}", severity="warn")

    def verdict_changed(self, status: "RunStatus", level: str, why: Sequence[str]) -> None:
        banner = {"ok": "RECOVERED - nothing needs you",
                  "watch": "RUNNING SLOW - no action needed",
                  "attention": "NEEDS YOU WHEN IT FINISHES",
                  "stop": "STOP THE POD"}[level]
        # No extra text: the board already ends with these exact lines. Saying it twice in
        # one message trains you to skim the top of them, which is where the banner lives.
        self.board(status, banner,
                   severity=("warn" if level in ("attention", "stop") else "info"))

    def finish(self, status: "RunStatus", ok: bool, elapsed: str,
               failed: Sequence[str], qualifying: int | None = None,
               verified: int | None = None, is_final: bool = True) -> None:
        cells = ("" if qualifying is None else
                 f" {qualifying} of {verified if verified is not None else '?'} verified "
                 "cells qualify.")
        tail = ("Nothing needs you - safe to stop the pod." if is_final else
                "Concept done - the batch is still running, do NOT stop the pod yet.")
        concept = _concept()
        if ok:
            self.board(status, f"RUN FINISHED CLEANLY - {concept}",
                       f"All phases done in {elapsed}.{cells} " + tail)
        else:
            self.board(status, f"RUN FINISHED WITH FAILURES - {concept}",
                       f"{len(failed)} phases failed: {', '.join(failed)}. "
                       f"The rest completed in {elapsed}.{cells} Their data is saved. "
                       + tail, severity="warn")
        self.ping("" if ok else "/fail")

    def status_line(self) -> str:
        if not self.enabled and not self.hc:
            return "phone alerts : OFF (no TELEGRAM_BOT_TOKEN - see the control panel)"
        return ("phone alerts : "
                + ("telegram ON" if self.enabled else "telegram OFF")
                + ", " + ("dead man's switch ON" if self.hc else "dead man's switch OFF")
                + (", warnings only" if self.warnings_only else ""))


# The process-wide notifier. Constructing it starts no thread unless Telegram or the
# healthcheck URL is configured, so importing this module offline is inert.
NOTIFY: Notifier = Notifier()


def notify_test() -> bool:
    """Send a test message and report whether it actually left the pod.

    Do this before walking away. The queue swallows transport errors on purpose -- a broken
    alert channel must never break a run -- which means a silent channel looks exactly like
    a quiet one. This is the only place that difference is made visible.
    """
    if not NOTIFY.enabled:
        print("phone alerts are OFF - set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID,")
        print("then call m2.monitor.NOTIFY.reload() and try again.")
        return False
    before_ok, before_bad = NOTIFY.sent, NOTIFY.dropped
    NOTIFY.send(f"TEST from the M2 operating-point finder{chr(10)}"
                f"{_concept()} | {_config_hash()}{chr(10) * 2}"
                "If you can read this, alerts work. Nothing is wrong.")
    if NOTIFY.hc:
        NOTIFY.ping()
    NOTIFY.drain(60.0)
    ok = NOTIFY.dropped == before_bad
    print(f"sent {NOTIFY.sent - before_ok} | failed {NOTIFY.dropped - before_bad}")
    if ok:
        print("check your phone - if the message is not there, the token is valid but the")
        print("chat id is wrong (Telegram accepts the call and delivers it to nobody).")
    else:
        print("the send FAILED - wrong token, or the pod has no route to api.telegram.org.")
    return ok


# =====================================================================================
# The rate model (spec 14.5)
# =====================================================================================
# `RunStatus` costs EACH PHASE SEPARATELY, using a prior until that phase has completed two
# units of its own work and then its own measured rate. This matters more in M2 than it did
# in v1: per-unit cost spans three orders of magnitude, from a Phase 1 scan cell (~2 s,
# forward passes only) to a Phase 4 verification cell (~50 s, 49 judge calls). A naive
# units-done/units-total ETA would be badly wrong for most of a run, since Phase 1 is ~100
# cheap units and Phase 4 is ~10 expensive ones.
#
# This replaces v1's CELL_SECONDS_PRIOR, which was per MEASURE over a fixed grid.

PHASE_SECONDS_PRIOR: dict[str, float] = dict(
    # MEASURED on Gemma3-27B, 1x A100 80GB, Garlic, 49 layers in scope, 3 scan doses
    # (2026-08-12 shakedown). Units must match the board counter beside every value.
    CAL=89.9,        # seconds per CONCEPT (one CAL unit), warm persistent volume. Prior 420.
    SCAN=10.9,       # seconds per (LAYER, DOSE) CELL; 147 cells in 26m46s. Prior 13.0.
    SHORTLIST=0.0,   # free
    BISECT=68.7,     # seconds per CANDIDATE; 8 candidates in 9m09s. Prior 100.0. A candidate
                     # is a bracket hunt plus BISECT_STEPS probes; the board does not count
                     # individual probes, so a per-probe prior would repeat the 12x unit bug.

    # STILL PRIORS - not yet observed. A multi-unit phase replaces its prior with its own
    # measured rate after two units, so these only drive the first two units and the opening
    # ETA. CONFIRM is single-unit, so its number is never corrected by measurement.
    VERIFY=50.0,     # per cell: 12 (E5) + 12 (S1) + 25 (D2) judge calls, E5/S1 concurrent,
                     # plus two batched generations of MAX_NEW_TOKENS
    REFINE=50.0,     # per cell
    CONFIRM=240.0,   # once
    CONTROLS=60.0,   # per control run
)

PHASE_UNITS_PRIOR: dict[str, int] = dict(
    # OPENING unit counts, so the ETA describes the RUN rather than the current phase.
    #
    # `eta()` costs `rate * (total - done)` and skips any phase whose total is 0, and a total
    # was only ever set when a phase STARTED. Every phase ahead therefore contributed nothing,
    # the Garlic run reported `ETA 0m00s` beside a list of seven excluded phases from start to
    # finish, and the one number the board exists to produce was never available.
    #
    # These are priors and are replaced by the truth the moment each phase reports its plan
    # (`size_phase`), which for SCAN and VERIFY is before their first unit runs. The counts
    # below are the observed Garlic shape: 49 layers in scope x 3 scan doses, eight tier-0
    # layers plus SHORTLIST_TIER_SIZE=3 mandatory audit layers, and refinement neighbours.
    # Units are stated because BISECT was once priced per probe while ticking per candidate:
    # BISECT=11 CANDIDATES; VERIFY=11 CELLS. The live plan replaces both before either starts.
    CAL=1, SCAN=147, SHORTLIST=1, BISECT=11, VERIFY=11, REFINE=6, CONFIRM=1, CONTROLS=1,
)

# Spec 8, in order. `phases.phase5_refine` is REFINE and `phase6_confirm` is CONFIRM.
PHASE_ORDER: tuple[str, ...] = (
    "CAL", "SCAN", "SHORTLIST", "BISECT", "VERIFY", "REFINE", "CONFIRM", "CONTROLS",
)

# Phases that generate and judge. Only these are subject to verdict rule 4 (too fast): a
# scan cell that comes back quickly is just a fast forward pass, whereas a verification cell
# that comes back in a fifth of its prior means empty generations or a judge answering
# instantly.
_JUDGE_PHASES: frozenset[str] = frozenset({"CAL", "VERIFY", "REFINE", "CONFIRM", "CONTROLS"})

_FINISHED_STATES: frozenset[str] = frozenset({"done", "failed", "skipped"})

# The two HARD GATES of spec 9. Verdict rule 7 fires only when both have run and both
# rejected, so the names live here once: `record_control` takes one of these keys, the board
# renders the label, and `controls_reject` iterates the same mapping. A third, softer control
# (9.3's escalation ladder) is a diagnostic and deliberately absent -- it does not reject.
_CONTROL_LABELS: dict[str, str] = {
    "random_direction": "control 9.1",
    "forced_id_capability": "control 9.2",
}


# =====================================================================================
# Verdicts (spec 14.6)
# =====================================================================================
# The board says what to DO, not just what is happening. Four levels:
#
#   ok         RECOVERED - nothing needs you
#   watch      RUNNING SLOW - no action needed (ETA already accounts for it)
#   attention  NEEDS YOU WHEN IT FINISHES (something died; the rest is worth having)
#   stop       STOP THE POD (it will not recover; stop paying)

def verdict_detail(status: "RunStatus") -> tuple[str, list[str]]:
    """`(level, lines)` -- the level plus the plain-words reason to print or push.

    Rules are evaluated in the spec's numbering order. Rules 3-7 all produce `attention`,
    so their order changes only which message is shown, never the level; rules 1 and 2 are
    the stops and must come first -- a stall is the one failure that looks exactly like
    healthy-but-slow.
    """
    failed = [p for p in status.order if status.state[p] == "failed"]
    running = [p for p in status.order if status.state[p] == "running"]
    since = time.time() - status.last_unit_at

    # --- rule 1: stall -> stop --------------------------------------------------------
    # Six times that phase's own per-unit time, floored at three minutes so a cheap phase's
    # setup cost cannot trip it. SHORTLIST's prior is 0.0, so the floor is what applies.
    if running:
        phase = running[0]
        rate, _ = status.rate(phase)
        if since > max(180.0, 6 * rate):
            return ("stop", [
                f"{phase} has not completed a unit in {fmt_time(since)}, against "
                f"~{rate:.0f}s each.",
                "That is stuck inside a generation or a judge call, not merely slow.",
                f"Last unit attempted: {status.note[phase] or 'unknown'}"])

    # --- rule 2: >= 2 phases failed -> stop -------------------------------------------
    if len(failed) >= 2:
        return ("stop", [
            f"{len(failed)} phases have died: {', '.join(failed)}.",
            "Two or more is structural - out of memory, judge auth, a bad install -",
            "not one unlucky phase."])

    # --- rule 3: 1 phase failed -> attention ------------------------------------------
    if failed:
        return ("attention", [
            f"{failed[0]} died. The other phases continue, by design.",
            "Let the run finish - the rest of the data is still worth having.",
            f"Then send the crash report for {failed[0]}."])

    # --- rule 4: too fast -> attention ------------------------------------------------
    # Too fast is as suspicious as too slow: empty generations judge instantly. Only the
    # generate+judge phases, and only once there is a measured rate to compare.
    for phase in status.order:
        if phase not in _JUDGE_PHASES or status.done[phase] < 3:
            continue
        rate, measured = status.rate(phase)
        prior = status.priors[phase]
        if measured and prior > 0 and rate < 0.2 * prior:
            return ("attention", [
                f"{phase} is running at {rate:.1f}s per unit against ~{prior:.0f}s "
                "expected.",
                "Too fast to be real - the generations may be empty.",
                "Check the transcripts before trusting anything from this phase."])

    # --- rule 5 (M2): judge-FPR breach -> attention -----------------------------------
    # Hard-indexed: JUDGE_FPR_MAX is a gate threshold, and a defaulted read here would turn
    # a missing constant into a gate that always passes (DEBUG LOG pattern 4).
    if status.judge_fpr is not None:
        limit = float(_cfg()["JUDGE_FPR_MAX"])
        if status.judge_fpr > limit:
            return ("attention", [
                f"Judge FPR on the control pairs is {status.judge_fpr:.2f}, above "
                f"{limit:.2f}.",
                "The judge invents influence on unsteered pairs, which puts a floor under",
                "every E5 in this run. The numbers are not trustworthy as they stand."])

    # --- rule 6 (M2): empty qualifying set after Phase 4 -> attention ------------------
    if status.qualifying is not None and status.qualifying == 0:
        return ("attention", [
            "No verified cell qualifies: nothing satisfies E5, D2 and S4 together.",
            "The run is not broken and should finish - the frontier and the escalation",
            "ladder still answer whether an operating point exists for this concept."])

    # --- rule 7 (M2): both controls reject the winner -> attention ---------------------
    if status.controls_reject():
        return ("attention", [
            "Both controls reject the winner: the random-direction control (9.1) and the",
            "forced-ID capability control (9.2) have each failed.",
            "The apparent result is an artifact - and that verdict is the finding."])

    # --- watch: slower than expected --------------------------------------------------
    slow = [p for p in status.order
            if status.state[p] in ("running", "done") and status.done[p] >= 3
            and status.rate(p)[1] and status.priors[p] > 0
            and status.rate(p)[0] > 2.5 * status.priors[p]]
    if slow:
        return ("watch", [
            f"{', '.join(slow)} is well below expected speed - most likely the judge",
            "being rate-limited. Slower, not broken, and the ETA above already",
            "accounts for it. No action needed."])

    eta = status.eta()
    return ("ok", ["Nothing needs you."
                   + (f" Expected finish in {fmt_time(eta)}." if eta > 0 else "")])


def verdict(status: "RunStatus") -> str:
    """The verdict LEVEL alone: `ok` | `watch` | `attention` | `stop` (CONTRACT section 3).

    One implementation, two shapes: `verdict_detail` carries the reason lines the board
    prints, this returns the level a caller branches on. Deriving the level a second way
    would let the two disagree, which is how a board says ALL GOOD next to a stop banner.
    """
    return verdict_detail(status)[0]


# =====================================================================================
# RunStatus
# =====================================================================================

class RunStatus:
    """Live state of a multi-phase run: what is done, what is running, what died.

    Constructed once per concept by the driver. `path` may be left None, in which case
    `status.txt` is resolved from `config.RUN.run_dir` at write time -- so a board that
    outlives a `set_concept` follows the concept rather than writing into the old folder.
    """

    def __init__(self, order: Sequence[str] = PHASE_ORDER,
                 path: Path | None = None,
                 totals: Mapping[str, int] | None = None,
                 priors: Mapping[str, float] | None = None,
                 notifier: Notifier | None = None) -> None:
        self.order = list(order)
        self.path = None if path is None else Path(path)
        self.notifier = NOTIFY if notifier is None else notifier

        # Hard-indexed per phase rather than `.get(phase, 5.0)`: an unpriced phase would get
        # an invented rate that silently drives the ETA, the stall threshold AND the
        # too-fast rule. A caller adding a phase must price it (DEBUG LOG pattern 4).
        source: Mapping[str, float] = PHASE_SECONDS_PRIOR if priors is None else priors
        missing = [p for p in self.order if p not in source]
        if missing:
            raise KeyError(
                f"no seconds-per-unit prior for {missing}; add them to "
                "PHASE_SECONDS_PRIOR or pass `priors=` - an unpriced phase would get an "
                "invented rate and a wrong ETA")
        self.priors = {p: float(source[p]) for p in self.order}

        # Totals are a DISPLAY quantity (they set the progress fraction and the ETA), and a
        # phase whose size is not known until it starts legitimately has none until then --
        # the shortlist size is a Phase 2 output, not a constant. 0 means "unknown"; the
        # board prints `?` and says the ETA excludes it. `start_phase(total=...)` fills it in.
        totals = {} if totals is None else totals
        self.total = {p: int(totals[p]) if p in totals else 0 for p in self.order}

        self.state = {p: "pending" for p in self.order}
        self.done = {p: 0 for p in self.order}
        self.skipped = {p: 0 for p in self.order}
        self.spent = {p: 0.0 for p in self.order}
        self.note = {p: "" for p in self.order}
        self.errors: dict[str, str] = {}   # raw, for status.txt on the volume ONLY
        self.labels: dict[str, str] = {}   # classified, the only version allowed off the pod

        # M2 verdict inputs (spec 14.6 rules 5-7). None means "not yet known", which is
        # distinct from 0.0 / 0 / rejected and must stay distinct: a gate that has not run
        # is not a gate that passed.
        self.judge_fpr: float | None = None
        self.qualifying: int | None = None
        self.verified: int | None = None
        self.controls: dict[str, bool] = {}

        self.t0 = time.time()
        self.last_unit_at = time.time()    # the stall detector's clock
        self._t_cur: float | None = None
        self._last_render = 0.0
        self._sticky = False
        self._display_id = "m2runstatus"
        self._lock = threading.Lock()
        self._beat: threading.Thread | None = None
        self._stop = threading.Event()
        self._nlevel: str | None = None    # last verdict level pushed to the phone
        self._nlast = 0.0
        self._nping = 0.0

    # ---- rate model -------------------------------------------------------------------
    def rate(self, phase: str) -> tuple[float, bool]:
        """`(seconds per unit, measured?)`. The phase's own rate after two units, else its
        prior (spec 14.5)."""
        if self.done[phase] >= 2 and self.spent[phase] > 0:
            return self.spent[phase] / self.done[phase], True
        return self.priors[phase], False

    def eta(self) -> float:
        """Seconds left, costed per phase. Phases with an unknown total contribute nothing;
        `text()` says so rather than letting the number quietly understate."""
        left = 0.0
        for phase in self.order:
            if self.state[phase] in _FINISHED_STATES:
                continue
            rate, _ = self.rate(phase)
            left += rate * max(0, self.total[phase] - self.done[phase] - self.skipped[phase])
        return left

    def unsized(self) -> list[str]:
        """Unfinished phases whose unit count is not yet known."""
        return [p for p in self.order
                if self.total[p] == 0 and self.state[p] not in _FINISHED_STATES]

    # ---- transitions ------------------------------------------------------------------
    def start_phase(self, phase: str, total: int | None = None, already: int = 0) -> None:
        """Mark a phase running. `already` is the count resume skipped (spec 14.8)."""
        with self._lock:
            self.state[phase] = "running"
            if total is not None:
                self.total[phase] = int(total)
            self.skipped[phase] = int(already)
            self._t_cur = time.time()
            self.last_unit_at = time.time()
        self.render(force=True)

    def size_phase(self, phase: str, total: int) -> None:
        """Set a phase's unit count once the phase itself knows it.

        The ETA costs `rate * (total - done)` and skips any phase whose total is still 0, so a
        run whose phases never declared a size reported `ETA 0m00s` from start to finish while
        listing every phase as excluded. Only the phase can supply the honest number: a resumed
        SCAN has 98 cells on the grid but perhaps 12 left to measure, and costing the grid
        would over-state the remaining work by eight times.
        """
        with self._lock:
            self.total[phase] = int(total)
        self.render()

    def unit_start(self, phase: str, note: str = "") -> None:
        """Called BEFORE a unit runs, so a stall shows which unit it is stuck on.

        The in-notebook block can only redraw when the main thread has control, and during a
        stall it does not. Recording the attempt means the frozen block still names the unit.
        """
        with self._lock:
            self.note[phase] = note
        self.render()

    def unit_done(self, phase: str, note: str = "") -> None:
        with self._lock:
            now = time.time()
            if self._t_cur is not None:
                self.spent[phase] += now - self._t_cur
            self._t_cur = now
            self.last_unit_at = now
            self.done[phase] += 1
            self.note[phase] = note
        self.render()

    def end_phase(self, phase: str, notify: bool = True) -> None:
        with self._lock:
            if self.state[phase] == "running":
                self.state[phase] = "done"
            self._t_cur = None
        self.render(force=True)
        if notify:
            self.notifier.phase_completed(self, phase)

    def fail_phase(self, phase: str, exc: BaseException) -> None:
        """Record a dead phase and push it immediately, not at the next beat.

        Two records, deliberately: `errors` keeps the raw text for status.txt, which never
        leaves the volume, and `labels` keeps the classified label, which is the only form
        allowed onto the wire (spec 14.3).
        """
        with self._lock:
            self.state[phase] = "failed"
            self.errors[phase] = f"{type(exc).__name__}: {exc}"[:70]
            self.labels[phase] = classify_exc(exc)
            self._t_cur = None
        self.render(force=True)
        self.notifier.phase_failed(self, phase, exc)

    def skip_phase(self, phase: str, why: str = "") -> None:
        with self._lock:
            self.state[phase] = "skipped"
            self.note[phase] = why
        self.render(force=True)

    # ---- the M2 verdict inputs --------------------------------------------------------
    def record_judge_fpr(self, fpr: float, notify: bool = True) -> bool:
        """Phase 0's control-pair result (spec 5.8). Returns whether it passed.

        Recorded here rather than left in the phase's return value because verdict rule 5
        has to keep firing for the rest of the run: the floor it describes sits under every
        E5 that follows, not only under the moment the gate ran.
        """
        with self._lock:
            self.judge_fpr = float(fpr)
        limit = float(_cfg()["JUDGE_FPR_MAX"])
        passed = self.judge_fpr <= limit
        if notify:
            self.notifier.judge_fpr_gate(self, self.judge_fpr, limit, passed)
        self.render(force=True)
        return passed

    def record_qualifying(self, n_qualifying: int, n_verified: int | None = None,
                          notify: bool = True) -> None:
        """Phase 4's qualifying set (spec 7). Zero is verdict rule 6, not a failure."""
        with self._lock:
            self.qualifying = int(n_qualifying)
            self.verified = None if n_verified is None else int(n_verified)
        if notify:
            self.notifier.qualifying_set(self, int(n_qualifying), n_verified)
        self.render(force=True)

    def record_control(self, name: str, passed: bool) -> None:
        """One hard gate's verdict: `random_direction` (9.1) or `forced_id_capability` (9.2).

        The name is checked against `_CONTROL_LABELS` rather than accepted as given: a typo
        would sit in the dict, never match `controls_reject`, and turn verdict rule 7 into a
        rule that can never fire -- a silent absence, which is the failure class this
        pipeline exists to be unable to reproduce.
        """
        if name not in _CONTROL_LABELS:
            raise KeyError(
                f"unknown control {name!r}; verdict rule 7 is defined over "
                f"{sorted(_CONTROL_LABELS)} and a name outside that set would never be read")
        with self._lock:
            self.controls[name] = bool(passed)
        self.render(force=True)

    def controls_reject(self) -> bool:
        """True only when BOTH hard-gate controls have run AND both rejected.

        Deliberately requires both to be present: one control missing is a control that has
        not run, and 'not run' must never read as 'passed' or as 'rejected'.
        """
        if any(name not in self.controls for name in _CONTROL_LABELS):
            return False
        return not any(self.controls[name] for name in _CONTROL_LABELS)

    # ---- verdict ----------------------------------------------------------------------
    def verdict(self) -> tuple[str, list[str]]:
        """`(level, lines)`. The rules live at module level so `verdict(status)` and the
        board can never disagree."""
        return verdict_detail(self)

    # ---- phone ------------------------------------------------------------------------
    def phone_text(self) -> str:
        """The board re-cut for a lock screen: ~24 columns, no horizontal scrolling.

        Not the 74-column block. On a phone that one wraps mid-column, and the alignment is
        the entire reason it is scannable - a wrapped board is harder to read than no board.
        Same information, one line per phase, the same verdict in the same words.

          >  running   +  done   !  failed   -  skipped
          (2s) = still the prior estimate, 2.4s = this run's own measured rate

        The state word is dropped here and kept in `text()`: phase names run to nine
        characters ('SHORTLIST'), and name + state + counts + rate does not fit the width.
        The mark already carries the state.
        """
        eta = self.eta()
        lines = [f"{_concept()} | {_config_hash()}",
                 f"elapsed  {fmt_time(time.time() - self.t0):>9}",
                 f"ETA      {fmt_time(eta):>9}",
                 ""]
        for phase in self.order:
            rate, measured = self.rate(phase)
            mark = {"running": ">", "failed": "!", "done": "+",
                    "skipped": "-", "pending": " "}[self.state[phase]]
            total = self.total[phase] or "?"
            units = f"{self.done[phase] + self.skipped[phase]}/{total}"
            shown = f"{rate:.1f}s" if measured else f"({rate:.0f}s)"
            lines.append(f"{mark}{phase:<9}{units:>6}{shown:>7}")
        # self.labels, NOT self.errors. The raw message can quote an API request body back
        # at you, and under M2 that body holds a steered generation. status.txt keeps the
        # raw text because it never leaves the volume; the phone gets the label.
        for phase in (p for p in self.order if self.state[p] == "failed"):
            label = self.labels[phase] if phase in self.labels else "failed"
            lines.append(f"  {phase}: {label[:40]}")
        lines += self._m2_lines()
        level, why = self.verdict()
        lines += ["", {"ok": ">> ALL GOOD",
                       "watch": "~~ RUNNING SLOW",
                       "attention": "!! NEEDS YOU AT THE END",
                       "stop": "!! STOP THE POD"}[level]]
        lines += ["   " + line for line in why]
        return chr(10).join(lines)

    def _m2_lines(self) -> list[str]:
        """The three M2 verdict inputs, shown as soon as each is known.

        These are metric values and control verdicts, which spec 14.3 channel 1 permits.
        They are on the board because they are what the verdict is reasoning from, and a
        banner whose evidence is invisible gets argued with at 2am.
        """
        out: list[str] = []
        if self.judge_fpr is not None:
            out.append(f"judge FPR {self.judge_fpr:.2f}")
        if self.qualifying is not None:
            of = "" if self.verified is None else f"/{self.verified}"
            out.append(f"qualifying {self.qualifying}{of}")
        for name, label in _CONTROL_LABELS.items():
            if name in self.controls:
                out.append(f"{label:<12}{'pass' if self.controls[name] else 'REJECT'}")
        return ([""] + out) if out else out

    def notify_check(self) -> None:
        """Push when the verdict LEVEL changes. Called from the heartbeat thread.

        On level change, not on every beat: a healthy run has to be quiet, or the alerts
        stop being read and the one that matters is missed along with them. The exception is
        a `stop` that persists - that repeats, because the cost of it going unread is the
        pod billing until morning.

        `attach()` seeds the level at "ok" rather than leaving it unset, because an unset
        level would swallow whatever the first check found - and a run that goes wrong
        inside its first ten seconds is exactly when that matters.
        """
        if not self.notifier.enabled:
            return
        level, why = self.verdict()
        now = time.time()
        repeat = float(_cfg()["NOTIFY_STOP_REPEAT"])
        if not (level != self._nlevel or (level == "stop" and now - self._nlast > repeat)):
            return
        self._nlevel, self._nlast = level, now
        self.notifier.verdict_changed(self, level, why)

    # ---- rendering --------------------------------------------------------------------
    def text(self) -> str:
        """The full 74-column board, for the notebook and for `status.txt`."""
        elapsed, eta = time.time() - self.t0, self.eta()
        bad = [p for p in self.order if self.state[p] == "failed"]
        running = [p for p in self.order if self.state[p] == "running"]
        L: list[str] = []
        L.append("=" * 74)
        L.append(f" M2 RUN STATUS   concept {_concept()}   config {_config_hash()}")
        L.append(f" written {time.strftime('%Y-%m-%d %H:%M:%S')}   (every 10s while alive)")
        L.append(f" elapsed {fmt_time(elapsed):>9}    ETA {fmt_time(eta):>9}    "
                 f"concept total ~{fmt_time(elapsed + eta)}")
        L.append("-" * 74)
        L.append(f"  {'phase':<10}{'state':<10}{'units':>9}{'s/unit':>9}{'spent':>9}  now")
        for phase in self.order:
            rate, measured = self.rate(phase)
            total = self.total[phase] or "?"
            units = f"{self.done[phase] + self.skipped[phase]}/{total}"
            shown = f"{rate:.1f}" if measured else f"({rate:.0f})"
            spent = fmt_time(self.spent[phase]) if self.spent[phase] > 0 else ""
            mark = {"running": ">>", "failed": "!!", "done": "  ",
                    "skipped": "--", "pending": "  "}[self.state[phase]]
            L.append(f"{mark}{phase:<10}{self.state[phase]:<10}{units:>9}{shown:>9}"
                     f"{spent:>9}  {self.note[phase][:20]}")
        L.append("-" * 74)
        for phase in bad:
            # Raw here, label on the phone. This file never leaves the volume.
            L.append(f" CRASHED  {phase}: {self.errors[phase]}")
        if not bad:
            L.append(" crashed: none" + (f"   |  running: {running[0]}" if running else ""))
        unsized = self.unsized()
        if unsized:
            L.append(f" ETA excludes {', '.join(unsized)} - unit count not known yet")
        for line in self._m2_lines():
            if line:
                L.append(" " + line)

        level, why = self.verdict()
        banner = {"ok":        ">>  ALL GOOD",
                  "watch":     "~~  RUNNING SLOW - NO ACTION NEEDED",
                  "attention": "!!  NEEDS YOU WHEN IT FINISHES",
                  "stop":      "!!  STOP THE POD AND SEND LOGS"}[level]
        L.append("")
        L.append(f" {banner}")
        for line in why:
            L.append(f"    {line}")
        run_dir = _run_dir()
        if level in ("attention", "stop") and run_dir is not None:
            L.append("")
            L.append("    Send these, not the notebook:")
            L.append(f"      {run_dir/'status.txt'}")
            L.append(f"      {run_dir.parent/'console.log'}")
            L.append(f"      {run_dir}/crash_*.txt")
        L.append("")
        L.append(" ( ) = estimate, not yet measured")
        L.append(" If THIS block stops moving, read status.txt - it is written on a 10s")
        L.append(" TIMER, so a frozen clock there means genuinely stuck, not merely busy.")
        L.append("=" * 74)
        return chr(10).join(L)

    def render(self, force: bool = False) -> None:
        """Redraw the in-notebook block. Throttled to 5 s unless forced.

        NOTE: this does NOT write status.txt. The file is written only by
        `write_status_txt`, on the heartbeat's 10 s timer (spec 14.5). One writer on a clock
        is what makes a stale timestamp there mean the process is stuck; if progress also
        wrote the file, a frozen file and a frozen run would stop being the same thing.
        """
        now = time.time()
        if not force and now - self._last_render < 5.0:
            return
        self._last_render = now
        txt = self.text()
        if self._sticky:
            try:
                from IPython.display import update_display
                update_display({"text/plain": txt}, raw=True, display_id=self._display_id)
                return
            except Exception:        # noqa: BLE001 - frontend cannot update; fall back
                self._sticky = False
        print(txt, flush=True)

    def write_status_txt(self) -> Path | None:
        """Write the board to `status.txt`. Never raises; returns the path it wrote.

        Resolved at write time so a board that outlives a concept switch follows the run
        directory rather than writing into a folder the batch driver has already wiped.
        """
        path = self.path
        if path is None:
            run_dir = _run_dir()
            path = None if run_dir is None else run_dir / "status.txt"
        if path is None:
            return None
        try:
            path.write_text(self.text(), encoding="utf-8")
            return path
        except Exception:            # noqa: BLE001 - a status write must never fail a run
            return None

    # ---- lifecycle --------------------------------------------------------------------
    def attach(self) -> None:
        """Pin the block, start the heartbeat, and announce the run.

        The heartbeat drives status.txt, the phone pushes and the dead man's switch.
        Nothing outbound happens before this call.
        """
        # The import succeeding is NOT evidence of a notebook. IPython is installed on every
        # PyTorch pod image, so under `nohup python -m m2.run` the import worked, `display`
        # found no frontend, fell back to printing repr() of what it was handed, and did not
        # raise - so `_sticky` stayed True and every redraw for the whole run printed a literal
        # `{'text/plain': '...'}` blob into the log between phases. Ask for a kernel, not a
        # module.
        if _in_notebook():
            try:
                from IPython.display import display
                display({"text/plain": self.text()}, raw=True, display_id=self._display_id)
                self._sticky = True
            except Exception:        # noqa: BLE001 - no display support after all
                self._sticky = False
                print(self.text(), flush=True)
        else:
            self._sticky = False
            print(self.text(), flush=True)
        self.write_status_txt()      # one write immediately, so the file exists from t=0
        self._stop.clear()
        self._beat = threading.Thread(target=self._heartbeat, daemon=True)
        self._beat.start()
        self._nping = time.time()
        self._nlevel = "ok"          # run_started() has just said so; changes push from here
        self.notifier.run_started(self)

    def _heartbeat(self) -> None:
        """File, phone and dead man's switch. Never the display.

        Writing to a display from a non-main thread is not reliable across frontends, and a
        wrong-cell write would be worse than no beat.

        This thread is why the alerts work at all during a stall: the main thread is inside
        a generation and cannot redraw anything, while this one keeps evaluating the verdict
        -- so "stuck" is detected and pushed by the same code that would have printed it.
        """
        while not self._stop.wait(10.0):
            # status.txt first: it is the one signal that survives a broken network.
            self.write_status_txt()
            now = time.time()
            try:
                self.notify_check()
                if now - self.notifier.last_board > float(_cfg()["NOTIFY_BOARD_EVERY"]):
                    self.notifier.board(self, "STILL RUNNING")
            except Exception:        # noqa: BLE001 - an alert must never take the run down
                pass
            try:
                if now - self._nping > float(_cfg()["NOTIFY_PING_EVERY"]):
                    self._nping = now
                    # Direct, not queued: see Notifier.ping_now. A ping goes out iff THIS
                    # thread is running, which is exactly what is being attested.
                    self.notifier.ping_now()
            except Exception:        # noqa: BLE001
                pass

    def detach(self) -> None:
        """Stop the heartbeat and leave a final board on screen and on disk."""
        self._stop.set()
        self.render(force=True)
        self.write_status_txt()


# =====================================================================================
# stdout capture
# =====================================================================================
# Jupyter shows cell output in the browser and nowhere else. Everything printed during a
# run - sample responses, judge verdicts, top-k tokens - is exactly what is needed to check
# a measure by hand, so it is mirrored to console.log as well.

def _in_notebook() -> bool:
    """True only inside a kernel with a frontend that can host an updatable display.

    `ZMQInteractiveShell` is Jupyter/Colab. A terminal IPython session is `TerminalInteractive
    Shell` and a plain `python -m` script has no shell at all; both must take the print path,
    because `IPython.display.display` does not raise in either - it silently prints the repr of
    the mimebundle it was given.
    """
    try:
        from IPython import get_ipython          # noqa: PLC0415 - optional, and only here
    except Exception:                            # noqa: BLE001 - IPython not installed
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


class _Tee:
    """Duplicate stdout to a file, so nothing printed is lost if the browser is closed.

    BUG 16. The `__getattr__` delegation is essential, not tidiness. At the start of every
    cell IPython calls `sys.stdout.set_parent(...)` to tell the stream which cell's output
    area to write into. A wrapper that does not forward that call leaves the real stream
    pointing at whichever cell was current when the wrapper was installed - so every later
    cell's output lands in that one cell. Forwarding unknown attributes to the wrapped
    stream keeps `set_parent`, `fileno`, `isatty` and friends working.
    """

    def __init__(self, path: str | Path, stream: Any) -> None:
        # Assign through __dict__ so __getattr__ never sees these as missing.
        object.__setattr__(self, "file", open(path, "a", encoding="utf-8", buffering=1))
        object.__setattr__(self, "stream", stream)

    def write(self, text: str) -> int:
        self.stream.write(text)
        try:
            self.file.write(text)
        except Exception:            # noqa: BLE001 - never let logging break a cell
            pass
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        try:
            self.file.flush()
        except Exception:            # noqa: BLE001
            pass

    def __getattr__(self, name: str) -> Any:
        # set_parent, fileno, isatty, encoding, ... all belong to the real stream. Bug 16.
        return getattr(object.__getattribute__(self, "stream"), name)


_ORIGINAL_STDOUT: Any = None


def tee_stdout(path: str | Path | None = None) -> Path | None:
    """Mirror stdout to `console.log`. Idempotent; returns the file being written.

    Defaults to `run_dir.parent/console.log`, one level up from the per-concept folder, so
    the log survives the batch driver's per-concept wipe and no concept's console output is
    lost with it.
    """
    import sys
    global _ORIGINAL_STDOUT
    if path is None:
        run_dir = _run_dir()
        if run_dir is None:
            return None
        path = run_dir.parent / "console.log"
    path = Path(path)
    if not isinstance(sys.stdout, _Tee):
        _ORIGINAL_STDOUT = sys.stdout
        sys.stdout = _Tee(path, sys.stdout)
    return path


def untee() -> None:
    """Stop mirroring stdout. Only needed if something goes wrong with the tee."""
    import sys
    if isinstance(sys.stdout, _Tee):
        sys.stdout.flush()
        if _ORIGINAL_STDOUT is not None:
            sys.stdout = _ORIGINAL_STDOUT
        print("stdout restored")
