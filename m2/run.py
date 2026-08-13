"""m2.run - the command-line entry point. Run a batch without a notebook kernel.

    python -m m2.run --concepts Irony,Silk,Pillows

Unattended, on a pod:

    cd "/workspace/steering-optimization"
    nohup python -m m2.run --concepts Irony,Silk,Pillows > /workspace/m2.out 2>&1 &
    tail -f /workspace/m2.out

WHY THIS EXISTS ALONGSIDE THE NOTEBOOK. The notebook is good for setup, for the rig checks and
for `m2.steer.session()` afterwards. It is a poor host for a multi-hour unattended batch,
because the Jupyter kernel is in the failure set: a dropped browser connection, a restart, or a
hung kernel holding the GIL all end the run, and the last of those is exactly why v1 needed an
out-of-process watchdog to notice a hang the notebook could not report. A `nohup`-ed process
survives the SSH session, survives the browser, and writes to a file the watchdog and `tail`
can both read.

The watchdog stays either way: it guards against a wedged *process*, not only a wedged kernel.

CREDENTIALS COME FROM THE ENVIRONMENT, not `getpass`. Under `nohup` there is no TTY, so a
prompt would read EOF and the run would start with no judge key and fail forty minutes later
during Phase 4. They are checked up front and the process refuses to start without the two that
are load-bearing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

# Running as `python -m m2.run` from the repository root already puts that
# directory on sys.path. Running the file by path does not, so make it work either way -
# bug 15's lesson: sys.path is per-process and a caller who does the reasonable thing should
# not have to know that.
_PKG_PARENT = Path(__file__).resolve().parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 130

# Required, because the run cannot produce a number without them.
REQUIRED_ENV = {
    "HF_TOKEN": "the model will not download or load",
    "OPENROUTER_API_KEY": "every judge call in Phases 4-6 would fail",
}
# Optional, but each absence removes a specific safety property, so each is named.
OPTIONAL_ENV = {
    "TELEGRAM_BOT_TOKEN": "no push alerts - the run is unattended-blind, watch status.txt",
    "TELEGRAM_CHAT_ID": "no push alerts - the run is unattended-blind, watch status.txt",
    "HEALTHCHECK_URL": "NO DEAD MAN'S SWITCH - a pod that dies outright will not report it",
    "RUNPOD_API_KEY": "the pod cannot auto-stop; stop it by hand when the batch finishes",
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m m2.run",
        description="M2 operating-point finder - unattended batch runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m m2.run --concepts Irony\n"
            "  python -m m2.run --concepts Irony,Silk,Pillows --log /workspace/m2.log\n"
            "  python -m m2.run --concepts Irony --multilayer\n"
            "  python -m m2.run --concepts Irony --set D2_MAX=0.25 --set SCAN_DOSES=0.15,0.30,0.45\n"
            "  python -m m2.run --preflight        # checks everything, measures nothing\n"))

    p.add_argument("--concepts", "-c", type=str,
                   help="comma-separated, e.g. Irony,Silk,Pillows. Or use --concepts-file.")
    p.add_argument("--concepts-file", type=Path,
                   help="one concept per line; blank lines and # comments ignored")

    p.add_argument("--log", type=Path, default=None,
                   help="tee stdout here as well as to the console (default: run_dir/lab.log)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a spec section 11 constant. Repeatable. Changing one changes "
                        "config_hash, so the run gets its own folder and cannot resume into an "
                        "earlier grid.")

    p.add_argument("--no-wipe", action="store_true",
                   help="keep each concept's loose run folder after its bundle is delivered")
    p.add_argument("--no-stop-pod", action="store_true",
                   help="do not STOP the pod when the batch finishes")
    p.add_argument("--no-notify-test", action="store_true",
                   help="skip the startup Telegram test message")
    p.add_argument("--skip-rig-checks", action="store_true",
                   help="DANGEROUS. R14 catches an injection hook that silently does nothing, "
                        "which read zero at all 30 cells of a real v1 run.")
    p.add_argument("--multilayer", action="store_true",
                   help="after each concept's winner, run the multi-layer arm (k in {2,3,5})")
    p.add_argument("--exhaustive", action="store_true",
                   help="set SHORTLIST_EXHAUSTIVE=True: bisect and verify every in-scope "
                        "layer regardless of whether an earlier tier qualifies. Prints the "
                        "estimated cell, judge-call and time cost before measurement starts.")
    p.add_argument("--transcripts-override", action="store_true",
                   help="include transcripts in the bundle for concepts that are NOT on "
                        "BENIGN_CONCEPTS. Read spec 14.3 before using this.")

    p.add_argument("--setup", action="store_true",
                   help="report what this pod already has and what it needs, then exit. "
                        "Loads no model. Run this first after every pod migration.")
    p.add_argument("--repair", action="store_true",
                   help="with --setup: also clone the harness, install packages, pull the "
                        "repo and trim truncated JSONL. Never deletes measured rows.")
    p.add_argument("--preflight", action="store_true",
                   help="environment, imports, public surface, model load and rig checks, then "
                        "exit. Measures nothing and spends no judge calls.")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would run and exit. Loads no model.")
    return p


def _read_concepts(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    if args.concepts:
        names += [c.strip() for c in args.concepts.split(",")]
    if args.concepts_file:
        for line in Path(args.concepts_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
    out = [n for n in dict.fromkeys(names) if n]     # de-duplicate, keep order
    if not out:
        raise SystemExit("no concepts given. Use --concepts Irony,Silk or --concepts-file PATH")
    return out


def _coerce(value: str) -> Any:
    """Parse an override value. Tries JSON first so tuples, floats and bools all work.

    `SCAN_DOSES=0.15,0.30` becomes a tuple of floats rather than the string "0.15,0.30" - a
    silently stringified dose would be compared against numbers and never match anything.
    """
    text = value.strip()
    if "," in text:
        return tuple(_coerce(part) for part in text.split(","))
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return json.loads(text)
    except Exception:                                # noqa: BLE001 - a bare string is fine
        return text


def apply_overrides(cfg: dict, pairs: Sequence[str]) -> dict:
    """Apply `KEY=VALUE` overrides onto CONFIG, refusing keys that do not exist.

    A typo must not become a new constant that nothing reads: `--set D2MAX=0.25` would leave
    D2_MAX at its default and the run would silently use the wrong constraint. Every key is
    checked against CONFIG, which already carries every spec section 11 constant.
    """
    applied = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in cfg:
            near = [k for k in cfg if k.upper().replace("_", "") == key.upper().replace("_", "")]
            hint = f" Did you mean {near[0]!r}?" if near else ""
            raise SystemExit(
                f"--set {key}: no such constant.{hint} Every spec section 11 constant is in "
                f"m2.config.CONSTANTS; a key that does not exist would be silently ignored.")
        value = _coerce(raw)
        cfg[key] = value
        applied[key] = value
    return applied


def exhaustive_cost_estimate(n_cells: int, *, judge_calls_per_cell: int = 49) -> dict:
    """Opening exhaustive-mode estimate in the same units the status board prices."""
    from m2 import monitor                         # noqa: PLC0415 - lightweight, no model
    cells = int(n_cells)
    if cells < 0:
        raise ValueError(f"n_cells must be non-negative, got {cells}")
    seconds = cells * (float(monitor.PHASE_SECONDS_PRIOR["BISECT"])
                       + float(monitor.PHASE_SECONDS_PRIOR["VERIFY"]))
    return dict(cells=cells, bisection_candidates=cells, verification_cells=cells,
                judge_calls=cells * int(judge_calls_per_cell), seconds=seconds)


def print_archive_restore_notices(concepts: Sequence[str], cfg: dict) -> list[str]:
    """Warn when an archive marker will skip a run whose loose resume folder is gone.

    A completed archive is deliberately the skip marker, but successful delivery normally wipes
    the loose folder. Restoring rows therefore requires two reversible operations: extract the
    archive and rename the marker so `driver.run_concept` no longer skips it. The exact command is
    printed before model loading, when avoiding an unnecessary GPU load still saves time.
    """
    from m2 import config, runio                  # noqa: PLC0415 - lightweight, no model

    notices: list[str] = []
    for concept in concepts:
        concept_cfg = dict(cfg)
        concept_cfg["concept"] = concept
        run_dir = config.run_dir_for(concept, concept_cfg)
        archive = runio.archive_path_for(run_dir)
        if not archive.is_file() or run_dir.is_dir():
            continue
        preserved = archive.with_suffix(archive.suffix + ".restored")
        quote = shlex.quote
        command = (
            f"mkdir -p {quote(str(run_dir))} && "
            f"python -m zipfile -e {quote(str(archive))} {quote(str(run_dir))} && "
            f"mv {quote(str(archive))} {quote(str(preserved))}"
        )
        message = (
            f"ARCHIVED RUN HAS NO LOOSE RESUME FOLDER: {concept}\n"
            f"  {archive} is the completion marker, so this run will be skipped.\n"
            "  Restore its rows and preserve-but-disable the marker, then rerun:\n"
            f"  {command}"
        )
        print("\n" + message)
        notices.append(message)
    return notices


def check_environment(strict: bool = True) -> dict:
    """Report which credentials are present, and refuse to start without the load-bearing two."""
    missing_required = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    print("credentials")
    for key, consequence in REQUIRED_ENV.items():
        state = "set" if os.environ.get(key) else "MISSING"
        print(f"  {key:<22} {state:<8}" + ("" if os.environ.get(key) else f"-> {consequence}"))
    for key, consequence in OPTIONAL_ENV.items():
        state = "set" if os.environ.get(key) else "unset"
        print(f"  {key:<22} {state:<8}" + ("" if os.environ.get(key) else f"-> {consequence}"))

    if missing_required and strict:
        raise SystemExit(
            f"\nrefusing to start: {', '.join(missing_required)} not set. Under nohup there is "
            "no TTY to prompt on, so this has to be an environment variable:\n"
            "    export OPENROUTER_API_KEY=...\n"
            "Setting it after the run started would not help - Phase 4 is forty minutes in.")
    if not os.environ.get("HEALTHCHECK_URL"):
        print("\n  NOTE: no dead man's switch. Nothing running on the pod can report its own "
              "death;\n        a stopped pod, a killed process and a lost network all look "
              "identical from\n        inside - silence. Only an external service noticing that "
              "pings STOPPED covers it.")
    return dict(missing_required=missing_required)


def _install_signal_handlers(notifier: Any) -> None:
    """On SIGINT/SIGTERM, say so and drain the notifier before dying.

    A killed run that never says it was killed is indistinguishable from a hung one, and the
    difference decides whether you restart it or debug it. The queue is drained because the
    notifier is a background thread: an unsent 'stopped' message helps nobody.
    """
    def _handler(signum: int, frame: Any) -> None:
        name = signal.Signals(signum).name
        print(f"\n[{name}] stopping. Rows already written are kept and the next run resumes "
              f"from them.", flush=True)
        if notifier is not None:
            try:
                notifier.send(f"M2 run received {name} - stopping. Resume by re-running the "
                              "same command; completed rows are skipped.")
                notifier.drain(10.0)
            except Exception:                        # noqa: BLE001 - never fail while dying
                pass
        raise SystemExit(EXIT_INTERRUPTED)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):                # not the main thread, or unsupported
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # --setup runs before everything, including the concept check: on a fresh or freshly
    # migrated pod the useful question is "what is missing", not "which concept". It imports
    # nothing heavy, so it works when the dependencies are the thing that is missing.
    if args.setup:
        from m2 import setup                       # noqa: PLC0415
        rep = setup.diagnose()
        if args.repair:
            rep = setup.repair(rep)
        setup.render(rep)
        return EXIT_OK if rep.ready() else (EXIT_CONFIG if rep.blocked else EXIT_FAILED)

    concepts = _read_concepts(args)

    print("=" * 78)
    print("M2 operating-point finder")
    print("=" * 78)
    print(f"concepts     : {', '.join(concepts)}")
    print(f"wipe         : {not args.no_wipe}")
    print(f"stop pod     : {not args.no_stop_pod}")
    print(f"multilayer   : {args.multilayer}")
    print(f"transcripts  : {'OVERRIDE ON (spec 14.3)' if args.transcripts_override else 'benign concepts only'}")
    print("")

    check_environment(strict=not (args.dry_run or args.preflight))

    # ---- imports and the public-surface assertion ---------------------------------------
    # Pattern 8: "the cell ran without error" is not evidence the cell did anything. Bug 24's
    # pipeline cell defined nothing and raised nothing, so importing is not enough - the names
    # the CONTRACT promises have to be checked to exist.
    print("\nimporting m2")
    from m2 import config, gates                    # noqa: PLC0415 - after sys.path is set

    applied = apply_overrides(config.CONFIG, args.overrides)
    if args.exhaustive:
        config.CONFIG["SHORTLIST_EXHAUSTIVE"] = True
        applied["SHORTLIST_EXHAUSTIVE"] = True
    if applied:
        print(f"  overrides  : {applied}")
        print("               (these change config_hash, so this run gets its own folder)")

    if not args.dry_run and not args.preflight:
        print_archive_restore_notices(concepts, config.CONFIG)

    surface = gates.check_public_surface()
    if surface["missing"]:
        print(f"  MISSING public names: {surface['missing']}")
        return EXIT_CONFIG
    if surface["unimportable"]:
        # A module that will not import is a run that dies in whichever phase first needs it,
        # after the model is loaded and the pod-hour is being paid for. Fail here instead.
        for mod, why in surface["unimportable"].items():
            print(f"  UNIMPORTABLE {mod}: {why}")
        if not args.dry_run:
            print("\n  Every m2 module must import before a run starts. On a fresh pod this is "
                  "almost always Setup 2 not having run (torch / transformers / the Macar "
                  "clone).")
            return EXIT_CONFIG
        print("  (--dry-run: continuing anyway)")
    print(f"  public surface: {surface['checked']} names present")

    non_benign = [c for c in concepts if not config.is_benign(c)]
    if non_benign:
        print(f"\n  {len(non_benign)} concept(s) NOT on BENIGN_CONCEPTS: {', '.join(non_benign)}")
        print("  transcripts are " + ("INCLUDED by explicit override"
                                      if args.transcripts_override else
                                      "withheld from their bundles (spec 14.3)"))

    if args.dry_run:
        print("\n--dry-run: nothing loaded, nothing measured.")
        return EXIT_OK

    # ---- model, monitor, rig checks -------------------------------------------------------
    from m2 import driver, model, monitor, phases   # noqa: PLC0415

    # Validate the ordering and the relationship among the tier knobs before loading the
    # model. An invalid string must never survive until Phase 2 and quietly become another
    # ordering after paid work has started.
    phases._tier_config(config.CONFIG)

    log_path = args.log or None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        monitor.tee_stdout(log_path)
        print(f"logging to {log_path}")

    notifier = driver.get_notifier()
    _install_signal_handlers(notifier)

    print("\nloading model")
    t0 = time.time()
    ctx = model.load_model(config.CONFIG)
    print(f"  {config.CONFIG['model']}  {ctx.n_layers} layers  "
          f"padding={ctx.tok.padding_side}  ({time.time() - t0:.0f}s)")

    if config.CONFIG["SHORTLIST_EXHAUSTIVE"]:
        n_cells = len(phases.layers_in_scope(int(ctx.n_layers),
                                             float(config.CONFIG["D_MIN"])))
        estimate = exhaustive_cost_estimate(n_cells)
        print("\nEXHAUSTIVE SHORTLIST MODE")
        print(f"  {n_cells} in-scope layers x (1 bisection candidate + 1 verification cell)")
        print(f"  approximately {estimate['judge_calls']} judge calls and "
              f"{estimate['seconds'] / 60:.1f} minutes "
              "before refinement/confirmation, using current priors")
        print("  every layer runs even after a qualifying cell is found")

    driver.set_concept(concepts[0])

    if args.preflight:
        # Gate 11 uses the upstream repo's LLMJudge, whose constructor reads
        # OPENAI_API_KEY. Importing eval_utils alone used to pass with only M2's
        # OPENROUTER_API_KEY set, then Gate 11 skipped 38 minutes into the shakedown.
        repo_judge = gates._preflight_repo_judge()
        if not repo_judge["passed"]:
            return EXIT_CONFIG

    if args.skip_rig_checks:
        print("\nRIG CHECKS SKIPPED (--skip-rig-checks). R14 is what catches an injection hook "
              "that silently does nothing.")
    else:
        print("\nrig checks")
        rig = gates.rig_checks()
        failed = []
        for name, row in rig.items():
            if isinstance(row, dict) and "passed" in row:
                mark = "PASS" if row["passed"] else "FAIL"
                print(f"  {name:<28} {mark:<6} {str(row.get('detail', ''))[:60]}")
                if not row["passed"]:
                    failed.append(name)
        if failed:
            print(f"\nrig checks failed: {', '.join(failed)}. Not starting a paid run on an "
                  "apparatus that has not attested itself. Override with --skip-rig-checks if "
                  "you know why.")
            if notifier is not None:
                notifier.send(f"M2 refused to start: rig checks failed ({', '.join(failed)})")
                notifier.drain(10.0)
            return EXIT_CONFIG

    if not args.no_notify_test and notifier is not None:
        print("\nsending a test alert - confirm it arrives before walking away")
        monitor.notify_test()

    if args.preflight:
        print("\n--preflight: environment, imports, surface, model and rig checks all done. "
              "Nothing measured, no judge calls spent.")
        return EXIT_OK

    # ---- the run --------------------------------------------------------------------------
    after = None
    if args.multilayer:
        from m2 import multilayer                   # noqa: PLC0415

        def after(winner: dict) -> dict:
            """The optional arm. Isolated by the driver, so a failure here cannot cost the
            operating point that is already measured."""
            return multilayer.run_arm(winner)

    print("\n" + "=" * 78)
    try:
        result = driver.run_batch(
            concepts,
            stop_pod=not args.no_stop_pod,
            wipe=not args.no_wipe,
            EXPORT_TRANSCRIPTS_OVERRIDE=args.transcripts_override,
            after_phases=after,
        )
    except SystemExit:
        raise
    except Exception as exc:                        # noqa: BLE001 - report, then exit non-zero
        print(f"\nBATCH FAILED: {type(exc).__name__}")
        print(traceback.format_exc())
        if notifier is not None:
            try:
                notifier.send(f"M2 batch failed: {monitor.classify_exc(exc)}")
                notifier.drain(15.0)
            except Exception:                       # noqa: BLE001
                pass
        return EXIT_FAILED

    done = result.get("done") or []
    failed = result.get("failed") or []
    skipped = result.get("skipped") or []
    print("\n" + "=" * 78)
    print(f"BATCH FINISHED  {len(done)} done, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        print(f"  failed: {', '.join(str(f) for f in failed)}")
    print("=" * 78)

    if notifier is not None:
        notifier.drain(float(config.CONFIG["KILL_GRACE_SECONDS"]))
    return EXIT_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
