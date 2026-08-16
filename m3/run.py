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

# gpt-4.1-mini list price. Only used to print an estimate before the run starts.
_USD_PER_INPUT_TOKEN = 0.40 / 1e6
_USD_PER_OUTPUT_TOKEN = 1.60 / 1e6
# Measured on the 2026-08-14 probe: one generation batch of <=25 prompts at 100 tokens.
_SECONDS_PER_BATCH_100 = 8.6


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
    layers = config.layers_for_depth(n_layers, cfg)
    cells = len(layers) * len(cfg["DOSE_FRACTIONS"])
    boundary_calls = len(layers) * int(cfg["BOUNDARY_PROBES"]) * int(cfg["BOUNDARY_N"])
    cell_calls = cells * config.judge_calls_per_cell(cfg)

    in_tokens = sum(judge.estimate_payload_tokens(j, cfg) for j in judge.JUDGE_IDS)
    mean_in = in_tokens / len(judge.JUDGE_IDS)
    per_call = mean_in * _USD_PER_INPUT_TOKEN + int(cfg["JUDGE_MAX_TOKENS"]) * _USD_PER_OUTPUT_TOKEN

    short = _SECONDS_PER_BATCH_100 * int(cfg["BOUNDARY_MAX_TOKENS"]) / int(cfg["MAX_NEW_TOKENS"])
    gpu_s = len(layers) * int(cfg["BOUNDARY_PROBES"]) * short + cells * _SECONDS_PER_BATCH_100

    # Can the descending ladder actually cross its own bracket floor? If not, layers the search
    # stopped short on come back indistinguishable from layers that genuinely broke, and the
    # operator finds out afterwards. Worst case (the descent starting at the bracket ceiling),
    # priced before anything is spent.
    lo, hi = (float(x) for x in cfg["BOUNDARY_BRACKET"])
    step = float(cfg["BOUNDARY_STEP"])
    probes_needed = math.ceil(math.log(lo / hi) / math.log(step)) + 1 if hi > lo else 1

    return dict(
        layers=len(layers), first_layer=layers[0], last_layer=layers[-1], cells=cells,
        battery=config.battery_size(cfg),
        judge_calls=boundary_calls + cell_calls,
        judge_usd=(boundary_calls + cell_calls) * per_call,
        gpu_minutes=gpu_s / 60.0,
        responses=cells * config.battery_size(cfg) + boundary_calls,
        probes_needed=probes_needed,
        ladder_reaches_floor=int(cfg["BOUNDARY_PROBES"]) >= probes_needed,
        lowest_probe=hi * step ** (int(cfg["BOUNDARY_PROBES"]) - 1),
    )


def _print_plan(est: dict, cfg: dict, concept: str) -> None:
    print("=" * 74)
    print(f"M3 sweep   concept={concept}   config={config.config_hash(cfg)}")
    print("=" * 74)
    print(f"  layers        {est['layers']} (L{est['first_layer']}-L{est['last_layer']}, "
          f"stride {cfg['LAYER_STRIDE']})")
    print(f"  doses/layer   {len(cfg['DOSE_FRACTIONS'])} at {cfg['DOSE_FRACTIONS']} "
          f"of each layer's own boundary")
    print(f"  cells         {est['cells']}")
    print(f"  battery       {est['battery']} responses/cell, one generation batch")
    print(f"  generations   {est['responses']:,}")
    print(f"  judge calls   {est['judge_calls']:,}  (<= ${est['judge_usd']:.2f} at cap)")
    print(f"  GPU estimate  ~{est['gpu_minutes']:.0f} min of measurement")
    print(f"  judge model   {cfg['JUDGE_MODEL']}  "
          f"<= {cfg['JUDGE_MAX_TOKENS']} reply tokens, {cfg['JUDGE_CONCURRENT']} concurrent")
    if est["ladder_reaches_floor"]:
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
