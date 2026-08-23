"""tools.rescore - apply the three-axis steering score to a run already on disk.

The score in `m3.battery` is derived from verdicts the judge already gave: influence 0-10,
coherence 0-10, on_task. Nothing here calls a judge, nothing costs anything, and every run ever
exported can be re-scored — which is the whole reason the metric was built as a derivation rather
than as a new judge prompt.

    python -m tools.rescore private/datos-rejuzgados/export_garlic_4de420af83d6

Reads `responses_transcripts.jsonl`, writes nothing, prints a per-cell table and the run totals.
Pass `--out FILE` to write the per-cell rows as JSONL for plotting.

## What it will tell you that the old number did not

`effectiveness` is the mean of the influence score. Two responses can share it and mean opposite
things: a story that is genuinely about garlic and still a story, and a repetition loop. This
separates them, and reports the second as its own rate rather than folding it into the first.

## The coverage line is not decoration

A row is scored only if it carries BOTH an influence and a coherence verdict. `N_COHERENCE`
below `N_EFFECT` means most rows do not, and the summary then describes a minority of the
battery. The 2026-08-21 Gemma runs score 570 of 1482 effect and explain rows for that reason.
Read the coverage before reading anything else.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from m3 import battery                                          # noqa: E402


def load(export_dir: Path) -> list[dict]:
    path = Path(export_dir) / "responses_transcripts.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing; point at an unzipped export directory")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def cells(rows: Sequence[dict], channels: Sequence[str] = ("effect",)) -> list[dict]:
    """One summary per (layer, dose), over the requested channels."""
    grouped: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if row["channel"] in channels:
            grouped[(row["layer"], row["dose"])].append(row)
    out = []
    for (layer, dose), group in sorted(grouped.items(),
                                       key=lambda kv: (kv[0][0] or -1, kv[0][1] or -1)):
        summary = battery.steering_summary(group)
        judged = [r for r in group if ((r.get("judged") or {}).get("effect") or {})
                  .get("influence") is not None]
        out.append(dict(
            layer=layer, dose=dose, n_rows=len(group), n_judged=len(judged),
            n_scored=(summary or {}).get("n", 0),
            old_effectiveness=(statistics.fmean(r["judged"]["effect"]["influence"]
                                                for r in judged) if judged else None),
            steering=summary))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.rescore",
        description="Re-score an exported run with the three-axis steering metric. No API calls.")
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--channels", nargs="+", default=["effect"],
                        help="which channels to summarise; `effect explain` for both")
    parser.add_argument("--out", type=Path, default=None, help="write per-cell rows as JSONL")
    args = parser.parse_args(argv)

    rows = load(args.export_dir)
    table = cells(rows, args.channels)
    scored = [c for c in table if c["steering"]]
    if not scored:
        raise SystemExit("no row carries both an influence and a coherence verdict; this run "
                         "cannot be re-scored. It was judged with N_COHERENCE=0, or not judged.")

    total_rows = sum(c["n_judged"] for c in table)
    total_scored = sum(c["n_scored"] for c in table)
    print(f"{args.export_dir.name}   channels={args.channels}")
    print(f"  coverage  {total_scored} of {total_rows} judged rows carry BOTH verdicts "
          f"({total_scored / max(total_rows, 1):.0%})")
    if total_scored < total_rows:
        print("            the rest have an influence verdict and no coherence verdict, so they "
              "have no fluency axis.")
        print("            Set N_COHERENCE = N_EFFECT to score the whole channel.")

    print(f"\n  {'L':>4}{'dose':>9}{'n':>5}{'old eff':>9}{'steering':>10}"
          f"{'clear+intact':>14}{'clear+broken':>14}")
    print("  " + "-" * 65)
    for c in scored:
        s = c["steering"]
        print(f"  {c['layer']:>4}{c['dose']:>9.4f}{s['n']:>5}"
              f"{c['old_effectiveness']:>9.2f}{s['steering_score']['mean']:>10.2f}"
              f"{s['steering_success']['rate']:>13.1%}"
              f"{s['concept_saturated_but_broken']['rate']:>14.1%}")

    everything = [r for r in rows if r["channel"] in args.channels]
    run = battery.steering_summary(everything)
    print(f"\n  RUN TOTAL over {run['n']} scored responses")
    print(f"    clear concept AND intact   {run['steering_success']['rate']:.1%}  "
          f"[{run['steering_success']['ci_low']:.1%}, {run['steering_success']['ci_high']:.1%}]")
    print(f"    any concept at all         {run['any_concept']['rate']:.1%}")
    print(f"    clear concept BUT broken   {run['concept_saturated_but_broken']['rate']:.1%}"
          "   <- what a mean of influence rewards")
    print(f"    mean steering score 0-2    {run['steering_score']['mean']:.2f}")
    for axis in battery.STEERING_AXES:
        print(f"      mean {axis:<9}          {run['axes'][axis]['mean']:.2f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for c in table:
                fh.write(json.dumps(c) + "\n")
        print(f"\n  per-cell rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
