"""m3.run - the command line. Terminal only; there is no notifier and no dead-man's switch.

    python -m m3.run --concept Garlic --dry-run          # plan and price it, load nothing
    python -m m3.run --concept Garlic                    # the sweep

Unattended on a pod:

    nohup python -m m3.run --concept Garlic > /workspace/m3.out 2>&1 &
    tail -f /workspace/m3.out

Anything in `m3.config` can be changed without editing the file:

    python -m m3.run --concept Garlic --set LAYER_STRIDE=2 --set N_IDENTIFY=8

A killed run resumes: rows already on disk are re-read and their cells are not re-measured.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

_PKG_PARENT = Path(__file__).resolve().parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from m3 import config, judge, sweep  # noqa: E402


EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_INTERRUPTED = 0, 1, 2, 130

# Both are load-bearing. Unlike M2's judge-free probe modes, every decision M3 makes is judged,
# so a missing judge key is not a degraded run -- it is no run at all.
REQUIRED_ENV = {
    "HF_TOKEN": "the model will not download or load",
    "OPENROUTER_API_KEY": "every measurement in this pipeline is judged; nothing can be scored",
}

# `deepseek/deepseek-v4-flash` list price on OpenRouter, read 2026-08-23. Only used to print an
# estimate before the run starts. It tracks `JUDGE_MODEL`, so changing one without the other
# prices a run nobody is going to have: gpt-4.1-mini was 0.40/1.60, seven times more, and the
# estimate kept quoting it for a while after the judge changed.
_USD_PER_INPUT_TOKEN = 0.0573 / 1e6
_USD_PER_OUTPUT_TOKEN = 0.1145 / 1e6
# Measured on the 2026-08-21 Gemma runs: one battery of 72 prompts at 100 new tokens, on a
# single A100-80GB, is 16.5s per cell including per-cell overhead (21.1 min for 57 cells + 20
# null batteries; 17.1 min for 42 + 20). The older 8.6s figure was taken at <=25 prompts.
_SECONDS_PER_BATCH_100 = 16.5
_CALIBRATED_AT_BATTERY = 72


def _seconds_per_battery(battery: int) -> float:
    """Wall-clock for one battery batch, scaled by how many prompts are in it.

    Linear above the calibration point, which is an UPPER BOUND and deliberately so. A 27B
    model at batch 72 is nowhere near saturating an A100, so doubling the battery costs
    noticeably less than double -- but by how much depends on the model, and guessing low is
    how an operator plans a two-hour run that takes six.

    It does not scale below the calibration point: the per-cell overhead outside generation
    does not shrink, and the 43-prompt runs came in at 20-25s per cell rather than under 16.5,
    because those also paid for a boundary phase.
    """
    return _SECONDS_PER_BATCH_100 * max(1.0, battery / _CALIBRATED_AT_BATTERY)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m m3.run",
        description="M3 - measure every cell, judged, and write down everything.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concept", "-c", required=True,
                   help="the concept to inject. Any concept except the harmful arm, which "
                        "this study has not designed yet.")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any m3.config setting. Repeatable. Changing one changes the "
                        "config hash, so the run gets its own folder and cannot resume into a "
                        "different grid.")
    p.add_argument("--log", type=Path, default=None,
                   help="also tee stdout to this file")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the projected cost, then exit. Loads no model, "
                        "spends nothing.")
    p.add_argument("--n-layers", type=int, default=None,
                   help="with --dry-run: assume this model depth instead of loading the model")
    return p


def estimate(n_layers: int, cfg: dict) -> dict:
    """What this run will cost, before it starts. Printed every time, dry-run or not.

    Deliberately pessimistic on the judge side: every payload is priced at its cap. A number an
    operator sees before spending is worth more than an accurate one they see afterwards.
    """
    # The estimate builds its own plan, so it has to honour the same two settings `run_sweep`
    # does or it prices a run that will not happen. It did exactly that once: with CELLS set to
    # five cells and no judge, this printed 150 cells and $5.92 of judging, nine minutes before
    # a run that measured five cells and spent nothing. An operator checking `--dry-run` to
    # confirm the settings landed would have concluded they had not.
    explicit = config.parse_cells(cfg.get("CELLS", ""))
    judging = bool(int(cfg.get("JUDGE_ENABLED", 1)))

    if explicit:
        layers = sorted({int(l) for l, _ in explicit})
        cells = len(explicit)
        # Phase 1 does not run, so it costs nothing in probes, judge calls or GPU time.
        probes_per_layer = 0
    else:
        layers = config.layers_for_depth(n_layers, cfg)
        cells = len(layers) * len(cfg["DOSE_FRACTIONS"])
        # Ladder plus refinement, both at their worst case: every ladder probe spent, then every
        # bisection spent. The tolerance usually stops bisection earlier; the price must not
        # assume so.
        probes_per_layer = int(cfg["BOUNDARY_PROBES"]) + int(cfg["BOUNDARY_BISECTIONS"])
    boundary_calls = len(layers) * probes_per_layer * int(cfg["BOUNDARY_N"])
    cell_calls = cells * config.judge_calls_per_cell(cfg)
    if not judging:
        boundary_calls = cell_calls = 0

    in_tokens = sum(judge.estimate_payload_tokens(j, cfg) for j in judge.JUDGE_IDS)
    mean_in = in_tokens / len(judge.JUDGE_IDS)
    per_call = mean_in * _USD_PER_INPUT_TOKEN + int(cfg["JUDGE_MAX_TOKENS"]) * _USD_PER_OUTPUT_TOKEN

    per_battery = _seconds_per_battery(config.battery_size(cfg))
    # A boundary probe is BOUNDARY_N+BOUNDARY_TASK_N prompts at BOUNDARY_MAX_TOKENS, not a whole
    # battery at MAX_NEW_TOKENS. Scaled on both axes rather than only on tokens.
    probe_prompts = int(cfg["BOUNDARY_N"]) + int(cfg.get("BOUNDARY_TASK_N", 0))
    short = (_seconds_per_battery(probe_prompts)
             * int(cfg["BOUNDARY_MAX_TOKENS"]) / int(cfg["MAX_NEW_TOKENS"]))
    # The null arm is NULL_REPEATS whole batteries, unsteered, and it was missing from both the
    # generation count and the clock. At the shipped 3 repeats that was a rounding error; at 20
    # it is a third of the run, which is enough to make an operator think the run has hung.
    null_batches = int(cfg["NULL_REPEATS"])
    gpu_s = (len(layers) * probes_per_layer * short
             + (cells + null_batches) * per_battery)

    # Can the descending ladder actually cross its own bracket floor? If not, layers the search
    # stopped short on come back indistinguishable from layers that genuinely broke, and the
    # operator finds out afterwards. Worst case (the descent starting at the bracket ceiling),
    # priced before anything is spent.
    lo, hi = (float(x) for x in cfg["BOUNDARY_BRACKET"])
    step = float(cfg["BOUNDARY_STEP"])
    probes_needed = math.ceil(math.log(lo / hi) / math.log(step)) + 1 if hi > lo else 1

    # Before any number is printed: the battery must fit one generation call. Priced runs that
    # cannot actually run are worse than no estimate -- this one printed "one generation batch"
    # nine minutes before the run died on that exact cap.
    config.check_battery_fits(cfg)
    config.check_boundary_window_fits(cfg)

    return dict(
        layers=len(layers), first_layer=layers[0], last_layer=layers[-1], cells=cells,
        battery=config.battery_size(cfg),
        chunks=config.battery_chunks(cfg),
        judge_calls=boundary_calls + cell_calls,
        judge_usd=(boundary_calls + cell_calls) * per_call,
        gpu_minutes=gpu_s / 60.0,
        responses=(cells + null_batches) * config.battery_size(cfg) + boundary_calls,
        explicit_cells=len(explicit),
        judging=judging,
        probes_needed=probes_needed,
        ladder_reaches_floor=int(cfg["BOUNDARY_PROBES"]) >= probes_needed,
        lowest_probe=hi * step ** (int(cfg["BOUNDARY_PROBES"]) - 1),
    )


def _print_plan(est: dict, cfg: dict, concept: str) -> None:
    print("=" * 74)
    print(f"M3 sweep   concept={concept}   config={config.config_hash(cfg)}")
    print("=" * 74)
    if est["explicit_cells"]:
        print(f"  layers        {est['layers']} (L{est['first_layer']}-L{est['last_layer']}, "
              f"from CELLS)")
        print(f"  doses         given explicitly, absolute -- DOSE_FRACTIONS is not used")
        print(f"  cells         {est['cells']}  (explicit)")
    else:
        print(f"  layers        {est['layers']} (L{est['first_layer']}-L{est['last_layer']}, "
              f"stride {cfg['LAYER_STRIDE']})")
        print(f"  doses/layer   {len(cfg['DOSE_FRACTIONS'])} at {cfg['DOSE_FRACTIONS']} "
              f"of each layer's own boundary")
        print(f"  cells         {est['cells']}")
    chunks = est["chunks"]
    if len(chunks) == 1:
        print(f"  battery       {est['battery']} responses/cell, one generation batch "
              f"(cap {cfg['GEN_BATCH_MAX']})")
    else:
        # Printed as the actual sizes, not just the count. Two runs measured "on the same batch
        # distribution" is a claim about this list, and the shortest way to check it is to read
        # it off both plans.
        print(f"  battery       {est['battery']} responses/cell, split into {len(chunks)} "
              f"generation batches of {chunks} (cap {cfg['GEN_BATCH_MAX']})")
    print(f"  generations   {est['responses']:,}")
    if est["judging"]:
        print(f"  judge calls   {est['judge_calls']:,}  (<= ${est['judge_usd']:.2f} at cap)")
    else:
        print(f"  judge calls   0  -- JUDGE_ENABLED=0, every judged measure will read null")
    print(f"  GPU estimate  ~{est['gpu_minutes']:.0f} min of measurement")
    if est["judging"]:
        print(f"  judge model   {cfg['JUDGE_MODEL']}  "
              f"<= {cfg['JUDGE_MAX_TOKENS']} reply tokens, {cfg['JUDGE_CONCURRENT']} concurrent")
    if est["explicit_cells"]:
        print("  boundary      SKIPPED -- the doses are given, so Phase 1 does not run")
    elif est["ladder_reaches_floor"]:
        print(f"  boundary      {cfg['BOUNDARY_PROBES']} probes descends "
              f"{cfg['BOUNDARY_BRACKET'][1]} -> below the floor "
              f"{cfg['BOUNDARY_BRACKET'][0]} at x{cfg['BOUNDARY_STEP']}")
    else:
        print(f"  boundary      WARNING: {cfg['BOUNDARY_PROBES']} probes at "
              f"x{cfg['BOUNDARY_STEP']} only reaches {est['lowest_probe']:.3f}, not the "
              f"bracket floor {cfg['BOUNDARY_BRACKET'][0]}.")
        print(f"                Layers still incoherent there are recorded as "
              f"'probes_exhausted', NOT as broken — nothing below that dose is measured.")
        print(f"                Set BOUNDARY_PROBES={est['probes_needed']} to descend the "
              f"whole bracket.")
    if not est["explicit_cells"] and int(cfg["BOUNDARY_BISECTIONS"]) > 0:
        print(f"                then <= {cfg['BOUNDARY_BISECTIONS']} bisection probes inside "
              f"the ladder's bracket, stopping within {cfg['BOUNDARY_BISECT_TOL']:.0%} "
              f"of the boundary")
    print("")
    print("  Nothing is filtered, ranked or selected. Every cell gets the same battery and")
    print("  every response is judged and written to disk. Mechanical measures are recorded")
    print("  and decide nothing.")
    print("")


def check_environment(strict: bool = True) -> list[str]:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    print("credentials")
    for key, why in REQUIRED_ENV.items():
        print(f"  {key:<22} " + ("set" if os.environ.get(key) else f"MISSING -> {why}"))
    print("")
    if missing and strict:
        raise SystemExit(
            f"refusing to start: {', '.join(missing)} not set.\n"
            "    export HF_TOKEN=hf_...\n"
            "    export OPENROUTER_API_KEY=sk-or-v1-...")
    return missing


PROBE_MB = 8


def check_volume_writable(root, probe_mb: int = PROBE_MB) -> dict:
    """Prove the volume can take a write, and say how much room is left. Before the model load.

    A run has no other disk check. `m2.setup` measures free space once, at setup, and on RunPod
    it cannot even do that honestly -- `shutil.disk_usage` reports the shared backing pool, not
    the allocation, which is the whole reason `M2_VOLUME_GB` exists as a declared fallback. So a
    sweep that fills the volume at cell 200 of 252 does not say "the volume is full". It writes
    short rows and dies later somewhere else: a torn 4,127-character append to norms.jsonl
    surfaced as `Unreachable: no norms were measured`, which sent an operator looking at the
    reachability code.

    The probe is a real write of `probe_mb`, flushed, fsynced, size-checked and removed. It is
    the only way to answer the question that actually matters -- can this volume take bytes --
    on a filesystem whose free-space number is known to lie. It costs well under a second and it
    runs before the 15-minute model load rather than after it.

    Returns what it found so the caller can print it. Raises RuntimeError if the write is short
    or refused; that is a full volume, and every number a run produces after that point is
    suspect.
    """
    import os as _os
    import shutil as _shutil

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Not zeros: a filesystem with sparse-file support can record a hole instead of
    # allocating blocks, so a probe of zeros can report success on a volume with no
    # room left. Real bytes are what test for real space.
    payload = b"m3probe." * (probe_mb * 1024 * 1024 // 8)
    # Named for THIS process. Three runs launch per pod, one per GPU, and they share
    # /workspace/m3_runs -- a fixed filename means the second launch truncates the first one's
    # probe while it is being written, and both report "the volume is full" on a volume with
    # 200 GB free. The probe runs before the model load, so the runbook's "wait for Model
    # loaded" stagger does not separate them.
    probe = root / f".write_probe.{_os.getpid()}"
    try:
        with open(probe, "wb") as handle:
            handle.write(payload)
            handle.flush()
            _os.fsync(handle.fileno())
        landed = probe.stat().st_size
    except OSError as exc:
        raise RuntimeError(
            f"cannot write to {root}: {exc}. The volume is full or read-only. Check `df -h` and "
            "the model cache under HF_HOME -- hf_xet keeps a chunk cache alongside the "
            "reconstructed weights, so two models can cost far more than the sum of their "
            "safetensors.") from exc
    finally:
        probe.unlink(missing_ok=True)

    if landed != len(payload):
        raise RuntimeError(
            f"a {probe_mb} MB probe write to {root} landed {landed} bytes. The volume is full. "
            "Short writes are how this pipeline loses a run: the row is written, the process "
            "continues, and a reader raises hours later on a file it cannot parse.")

    usage = _shutil.disk_usage(root)
    free_gb = usage.free / 1024 ** 3
    declared = _os.environ.get("M2_VOLUME_GB")
    note = (f"volume     : {free_gb:.0f} GB free by the filesystem, {probe_mb} MB probe write ok")
    if declared:
        # On RunPod the filesystem number describes the backing pool and is meaningless; the
        # declared allocation is the only honest ceiling. Print both and let the operator see
        # them disagree rather than picking one and being wrong silently.
        note += f"  (allocation declared as {declared} GB via M2_VOLUME_GB)"
    print(note)
    return dict(free_gb=round(free_gb, 1), probe_mb=probe_mb, declared_gb=declared)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = config.CONFIG

    try:
        changed = config.apply_overrides(args.overrides, cfg)
    except ValueError as exc:
        print(f"bad --set: {exc}")
        return EXIT_CONFIG
    if changed:
        print(f"overrides: {changed}\n           (these change the config hash and the run folder)\n")

    concept = str(args.concept).strip()
    if not config.concept_allowed(concept):
        print(f"{concept!r} is on HARMFUL_CONCEPTS — the arm this study has deliberately not "
              "run. That is a decision about the study, not a setting; see CLAUDE.md.")
        return EXIT_CONFIG

    check_environment(strict=not args.dry_run)

    if args.dry_run:
        n_layers = args.n_layers or 62
        _print_plan(estimate(n_layers, cfg), cfg, concept)
        print(f"--dry-run: nothing loaded, nothing measured "
              f"(assumed a {n_layers}-layer model; pass --n-layers to change)")
        return EXIT_OK

    if args.log:
        from m2 import monitor
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        monitor.tee_stdout(args.log)
        print(f"logging to {args.log}")

    from m2 import model, runio

    # `model.load_model` computes its own run dir through M2's `run_dir_for`, which reads
    # M2_RUNS_DIR (default /workspace/m2_runs), and mkdirs it -- before `open_run` re-points
    # RUN.run_dir at M3's root. No measurement lands there, but every run leaves an orphan
    # directory under the OTHER pipeline's output tree. Point M2's variable at M3's root so the
    # stray directory is at least in the right place.
    os.environ.setdefault("M2_RUNS_DIR", str(config.runs_root()))

    check_volume_writable(config.runs_root())

    print("loading model")
    t0 = time.time()
    ctx = model.load_model(config.m2_config(concept, cfg))
    print(f"  {cfg['MODEL']}  {ctx.n_layers} layers  padding={ctx.tok.padding_side}  "
          f"({time.time() - t0:.0f}s)\n")

    _print_plan(estimate(int(ctx.n_layers), cfg), cfg, concept)

    run_dir = sweep.open_run(concept, cfg)
    print(f"run dir: {run_dir}\n")

    prov = model.provenance()
    prov.update(ts=runio._now(), phase_entered_at="m3.sweep", mode="sweep")
    runio.write_row("provenance.jsonl", prov)
    print(f"provenance: {prov.get('gpu','?')} | commit {str(prov.get('git_commit'))[:12]}"
          f"{' (DIRTY)' if prov.get('git_dirty') else ''}\n")

    try:
        summary = sweep.run_sweep(concept, cfg)
    except KeyboardInterrupt:
        print("\ninterrupted. Rows already written are kept; rerun to resume from them.")
        return EXIT_INTERRUPTED
    except Exception:
        print(traceback.format_exc())
        print("\nrun FAILED. Rows already written are kept; rerun to resume from them.")
        return EXIT_FAILED

    runio.archive_concept(run_dir)
    # Transcripts always ship. The allow-list that used to gate this was a filter on
    # exploration rather than on risk, and every generation in this pipeline is the
    # thing a reader needs -- every deep defect here was found by reading them.
    bundle = runio.export_bundle(run_dir, EXPORT_TRANSCRIPTS_OVERRIDE=True)
    print(f"\nbundle: {bundle}")
    print(f"cells on disk: {summary['n_cells_on_disk']}/{summary['n_cells_planned']}")
    if summary["skipped"]:
        print(f"NOT measured: {len(summary['skipped'])} — named in summary.json under 'skipped'")
    print("\nNothing here is gated or filtered. Read the transcripts before quoting a number.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
