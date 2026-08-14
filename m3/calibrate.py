"""m3.calibrate - Phase -1. Validate the judges against hand labels before they are trusted.

The judge is M3's primary instrument. M2's failure was trusting an unvalidated measure to decide
what got measured; shipping an unvalidated judge would be the same mistake with a bigger model
attached. So before any judge runs on a pod, it is scored against responses a human has labelled.

**The data already exists.** The 2026-08-14 probe produced 1,204 real Gemma3-27B responses across
the same four channels M3 measures, spanning clean identifications, confabulations, repetition
loops, coherent denials that leak the concept, and an unsteered null arm. Calibrating against
those costs about a dollar and no GPU.

## How this runs

    python -m m3.calibrate sample                 # stratified worksheet, no API calls
    python -m m3.calibrate score --gold FILE      # run the judges, report agreement
    python -m m3.calibrate full --gold FILE       # then score every response in the archive

Iterate on the prompts in `m3.judge` between runs until agreement clears the bar, then run `full`
and check the per-cell aggregates reproduce the hand analysis of the same probe.

## What the labels are, and are not

The gold labels are **one careful reader's judgement**, not ground truth. Agreement with them
means the judge sees what that reader saw; it does not mean either is right. Two consequences
kept deliberately: the labels are stored as scalars keyed by response coordinates, so anyone can
re-read the same transcripts and disagree; and every disagreement is printed in full, because a
systematic disagreement is more informative than the headline agreement number.

Labels live in `m3/labels/` as coordinates plus verdicts -- never transcript text, which stays
outside the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


__all__ = ["load_probe", "sample", "cohen_kappa", "score_agreement", "main"]

LABELS_DIR = Path(__file__).resolve().parent / "labels"

# How many of each judge's decisions to label. Categorical agreement needs roughly 30 items to
# put a usable interval on kappa; the ordinal judges are scored by error size, which converges
# faster, so they get fewer.
SAMPLE_SIZES = {"identify": 30, "self_report": 30, "coherence": 25, "effect": 25}


# =====================================================================================
# Loading the probe archive
# =====================================================================================

_CHANNEL_TO_JUDGE = {"forced": "identify", "detect": "self_report", "task": ("effect", "coherence")}


def _rid(rec: dict) -> str:
    """A stable id for one response. Coordinates only -- this is what labels are keyed on, so it
    must not change if the transcript files are re-exported."""
    where = "null" if rec["layer"] is None else f"L{rec['layer']}@{rec['r']:.2f}"
    unit = rec.get("trial")
    unit = f"t{unit}" if unit is not None else str(rec.get("prompt_id"))
    return f"{rec['channel']}:{where}:{unit}"


def load_probe(probe_dir: Path) -> list[dict]:
    """Every response from a 2026-08-14-style probe bundle, normalised and given a stable id."""
    probe_dir = Path(probe_dir)
    files = {"forced": "probe_forced_transcripts.jsonl",
             "detect": "probe_detect_transcripts.jsonl",
             "task": "probe_task_transcripts.jsonl",
             "null": "probe_null_transcripts.jsonl"}
    out: list[dict] = []
    for kind, name in files.items():
        path = probe_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; point --probe at an unzipped bundle")
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            rec = dict(
                channel=row["channel"], layer=row.get("layer"), r=row.get("r"),
                trial=row.get("trial"), prompt_id=row.get("prompt_id"),
                response=row["response"], words=row["words"],
                concept_hits=row["concept_hits"], degenerate=row["degenerate"],
                degeneration_reason=row.get("degeneration_reason"),
                steered=row.get("steered", kind != "null"),
            )
            rec["id"] = _rid(rec)
            out.append(rec)
    if len({r["id"] for r in out}) != len(out):
        dupes = [k for k, v in Counter(r["id"] for r in out).items() if v > 1]
        raise ValueError(f"response ids are not unique: {dupes[:5]}")
    return out


# =====================================================================================
# Stratified sampling
# =====================================================================================

def _stratum(rec: dict) -> str:
    """Which interesting case this response is, from mechanical signals only.

    Sampling is stratified on these rather than drawn at random, because the cases that
    discriminate a good judge from a bad one are rare. A random draw from 1,204 responses is
    mostly clean identifications, and a judge can score 95% on those while being wrong about
    every confabulation and every repetition loop -- which are the two things it exists to tell
    apart.
    """
    if not rec["steered"]:
        return "null"
    if rec["degenerate"]:
        return "degenerate"
    if rec["concept_hits"] == 0:
        return "no_concept"
    if rec["concept_hits"] >= 8:
        return "concept_heavy"
    return "concept_present"


def sample(records: Sequence[dict], judge_id: str, n: int, *, seed: int = 20260814) -> list[dict]:
    """An even draw across strata for one judge, deterministic under `seed`."""
    wanted = [r for r in records
              if judge_id in (_CHANNEL_TO_JUDGE.get(r["channel"]) or ())
              or _CHANNEL_TO_JUDGE.get(r["channel"]) == judge_id]
    if not wanted:
        raise ValueError(f"no responses feed the {judge_id!r} judge")
    buckets: dict[str, list[dict]] = {}
    for rec in wanted:
        buckets.setdefault(_stratum(rec), []).append(rec)
    rng = random.Random(seed)
    for items in buckets.values():
        items.sort(key=lambda r: r["id"])
        rng.shuffle(items)

    out: list[dict] = []
    order = sorted(buckets)
    while len(out) < n and any(buckets[k] for k in order):
        for key in order:
            if buckets[key] and len(out) < n:
                out.append(buckets[key].pop())
    return sorted(out, key=lambda r: r["id"])


# =====================================================================================
# Agreement
# =====================================================================================

def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    """Chance-corrected agreement for two categorical labellings of the same items.

    Raw agreement is misleading when one answer dominates: a judge that always says NO scores
    85% on a set that is 85% NO while having learned nothing. Kappa subtracts the agreement two
    random labellers with the same marginals would reach.

    `None` when kappa is undefined -- both labellers used exactly one category, so chance
    agreement is 1.0 and the statistic divides by zero. That is a real outcome, not an error:
    it means the sample cannot discriminate, and reporting it as 1.0 would be M2's Gate 5
    failure (a validation that cannot fail).
    """
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("cohen_kappa needs two equal, non-empty labellings")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def score_agreement(gold: Sequence[dict], judged: Sequence[dict], field: str,
                    kind: str) -> dict:
    """Agreement on one field: kappa for categorical, error size for ordinal."""
    pairs = [(g[field], j[field]) for g, j in zip(gold, judged)
             if g.get(field) is not None and j.get(field) is not None]
    if not pairs:
        return dict(field=field, n=0, note="no comparable pairs")
    got, want = [p[1] for p in pairs], [p[0] for p in pairs]

    if kind == "categorical":
        agree = sum(1 for x, y in pairs if x == y) / len(pairs)
        return dict(field=field, kind=kind, n=len(pairs), agreement=agree,
                    kappa=cohen_kappa(want, got),
                    confusion=dict(Counter(f"gold={x}|judge={y}" for x, y in pairs if x != y)))

    errs = [abs(float(x) - float(y)) for x, y in pairs]
    return dict(field=field, kind=kind, n=len(pairs),
                mean_abs_error=statistics.fmean(errs),
                median_abs_error=statistics.median(errs),
                within_1=sum(1 for e in errs if e <= 1) / len(errs),
                within_2=sum(1 for e in errs if e <= 2) / len(errs),
                max_abs_error=max(errs),
                gold_mean=statistics.fmean(float(x) for x in want),
                judge_mean=statistics.fmean(float(y) for y in got))


# The bar each judge must clear before it is trusted on a pod. Stated here so that "the judge
# was validated" is a claim with a number behind it rather than an impression.
THRESHOLDS = {
    "identify.matches": dict(kappa=0.80),
    "self_report.claims": dict(kappa=0.75),
    "self_report.matches": dict(kappa=0.75),
    "coherence.coherence": dict(mean_abs_error=1.5, within_2=0.85),
    "coherence.on_task": dict(kappa=0.70),
    "effect.influence": dict(mean_abs_error=2.0, within_2=0.75),
}


def verdicts(results: dict) -> list[dict]:
    """Compare each scored field against its threshold. Returns one row per criterion."""
    out = []
    for key, limits in THRESHOLDS.items():
        jid, field = key.split(".", 1)
        got = (results.get(jid) or {}).get(field)
        if not got or not got.get("n"):
            out.append(dict(criterion=key, verdict="NOT MEASURED", detail="no comparable pairs"))
            continue
        checks, ok = [], True
        for metric, bound in limits.items():
            value = got.get(metric)
            if value is None:
                checks.append(f"{metric}=undefined")
                ok = False
                continue
            passed = value >= bound if metric in ("kappa", "within_1", "within_2") else value <= bound
            ok = ok and passed
            checks.append(f"{metric}={value:.3f} {'>=' if metric in ('kappa','within_1','within_2') else '<='} {bound}"
                          + ("" if passed else "  MISS"))
        out.append(dict(criterion=key, verdict="PASS" if ok else "FAIL",
                        n=got["n"], detail="; ".join(checks)))
    return out


# =====================================================================================
# Worksheet
# =====================================================================================

def write_worksheet(records: Sequence[dict], judge_id: str, path: Path) -> Path:
    """A readable file of sampled responses, for a human to label.

    Written as text rather than JSON because it is read by a person. The companion label file is
    JSONL keyed on the same ids.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# worksheet: {judge_id}  ({len(records)} responses)\n")
        fh.write(f"# label each into {LABELS_DIR / f'gold_{judge_id}.jsonl'}\n\n")
        for rec in records:
            fh.write(f"--- {rec['id']}  [{_stratum(rec)}]  "
                     f"words={rec['words']} hits={rec['concept_hits']} "
                     f"degen={rec['degenerate']}\n")
            fh.write(rec["response"].strip() + "\n\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m m3.calibrate",
                                description="Validate the m3 judges against hand labels.")
    p.add_argument("command", choices=["sample", "score", "full"])
    p.add_argument("--probe", type=Path, required=True,
                   help="an unzipped probe bundle directory")
    p.add_argument("--out", type=Path, default=Path("worksheets"))
    p.add_argument("--gold", type=Path, default=LABELS_DIR)
    p.add_argument("--concept", default="Garlic")
    p.add_argument("--full-out", type=Path, default=None,
                   help="where `full` writes its scored rows. Defaults to judged_full.jsonl "
                        "inside the probe directory. Re-running resumes from it rather than "
                        "paying twice.")
    args = p.parse_args(argv)

    records = load_probe(args.probe)
    print(f"loaded {len(records)} responses from {args.probe}")
    print(f"strata: {dict(Counter(_stratum(r) for r in records))}\n")

    if args.command == "sample":
        for jid, n in SAMPLE_SIZES.items():
            picked = sample(records, jid, n)
            out = write_worksheet(picked, jid, Path(args.out) / f"{jid}.txt")
            print(f"  {jid:<12} {len(picked):>3} responses -> {out}")
        return 0

    from m3 import scoring          # noqa: PLC0415 - only needed on the scoring path
    if args.command == "full":
        return scoring.run_full(records, concept=args.concept,
                                out=Path(args.full_out or (Path(args.probe) / "judged_full.jsonl")))
    return scoring.run(args.command, records, gold_dir=Path(args.gold),
                       concept=args.concept)


if __name__ == "__main__":
    sys.exit(main())
