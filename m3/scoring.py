"""m3.scoring - run the judges against the hand labels and report where they disagree.

Called by `m3.calibrate`. Kept separate because this is the only part of Phase -1 that spends
money, and separating it means the sampling and the labelling can be iterated for free.

The output is deliberately two things: a headline agreement table, and **every disagreement in
full**. The table says whether to ship the judge; the disagreements say what to change about the
prompt, and they are where all the information is. A systematic disagreement -- the judge marking
every truncated response down, say -- is worth more than the headline number, and is invisible in
it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from . import battery, config, judge
from .calibrate import LABELS_DIR, score_agreement, verdicts


# Which fields each judge is scored on, and how.
SCORED = {
    "identify": [("matches", "categorical"), ("named", "categorical")],
    "self_report": [("claims", "categorical"), ("matches", "categorical")],
    "coherence": [("coherence", "ordinal"), ("on_task", "categorical")],
    "effect": [("influence", "ordinal"), ("form", "categorical")],
}


def load_gold(gold_dir: Path, judge_id: str) -> list[dict]:
    path = Path(gold_dir) / f"gold_{judge_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _baselines(records: Sequence[dict]) -> dict[str, str]:
    """The unsteered task response per prompt, which the effect judge is paired against."""
    return {r["prompt_id"]: r["response"] for r in records
            if r["channel"] == "task" and not r["steered"]}


def _prompt_text(prompt_id: str) -> str:
    for row in (*battery.TASK_PROMPTS, *battery.CAPABILITY_PROMPTS, *battery.TASK_HELDOUT):
        if row["id"] == prompt_id:
            return row["text"]
    # The probe used M2's prompt ids (e5_01..e5_12). Fall back to those so an archive recorded
    # under the old names is still scoreable rather than silently skipped.
    from m2 import prompts as upstream
    for row in upstream.E5_PROMPTS:
        if row["id"] == prompt_id:
            return row["text"]
    raise KeyError(f"no prompt text for {prompt_id!r}")


def build_items(judge_id: str, gold: Sequence[dict], by_id: dict[str, dict],
                baselines: dict[str, str], concept: str, cfg: dict) -> tuple[list, list]:
    """`(items, used_gold)` — items to issue, and the gold rows they correspond to, in order."""
    chars = int(cfg["JUDGE_TEXT_CHARS"])
    items, used = [], []
    for g in gold:
        rec = by_id.get(g["id"])
        if rec is None:
            print(f"    ! {g['id']} is labelled but absent from the archive; skipped")
            continue
        response = rec["response"]
        if judge_id in ("identify", "self_report"):
            payload = judge.render(judge_id, text_chars=chars, concept=concept, response=response)
            model_text = (response,)
        elif judge_id == "coherence":
            payload = judge.render("coherence", text_chars=chars,
                                   prompt=_prompt_text(rec["prompt_id"]), response=response)
            model_text = (response,)
        else:
            base = baselines.get(rec["prompt_id"])
            if base is None:
                print(f"    ! {g['id']} has no unsteered baseline; skipped")
                continue
            payload = judge.render("effect", text_chars=chars, concept=concept,
                                   prompt=_prompt_text(rec["prompt_id"]),
                                   response_unsteered=base, response_steered=response)
            model_text = (base, response)
        items.append(judge.build_item(judge_id, payload=payload,
                                      cache_key=("CALIBRATE", g["id"], judge_id),
                                      concept=concept, model_text=model_text))
        used.append(g)
    return items, used


def _normalise(judge_id: str, parsed: dict) -> dict:
    """Make judge output comparable with the labels. Only the free-text `named` needs it."""
    out = dict(parsed)
    if "named" in out:
        out["named"] = str(out["named"]).strip().lower().strip('."*')
    return out


def run(command: str, records: Sequence[dict], *, gold_dir: Path, concept: str) -> int:
    cfg = config.CONFIG
    judge.configure_transport(cfg)
    by_id = {r["id"]: r for r in records}
    baselines = _baselines(records)

    all_results: dict[str, dict] = {}
    disagreements: list[str] = []

    for judge_id, fields in SCORED.items():
        gold = load_gold(gold_dir, judge_id)
        if not gold:
            print(f"{judge_id}: no labels at {gold_dir}/gold_{judge_id}.jsonl — skipped")
            continue
        items, used = build_items(judge_id, gold, by_id, baselines, concept, cfg)
        if not items:
            continue
        print(f"\n{judge_id}: issuing {len(items)} judge calls")
        results = judge.run_judges(items, concurrency=int(cfg["JUDGE_CONCURRENT"]))

        got, kept_gold, errors = [], [], 0
        for g, result in zip(used, results):
            parsed, error = judge.verdict(result)
            if parsed is None:
                errors += 1
                print(f"    ! {g['id']}: {error}")
                continue
            parsed = _normalise(judge_id, parsed)
            got.append(parsed)
            kept_gold.append(g)
            for field, _kind in fields:
                if field in g and g[field] is not None and parsed.get(field) != g[field]:
                    disagreements.append(
                        f"  {judge_id:<12} {g['id']:<26} {field}: "
                        f"gold={g[field]!r}  judge={parsed.get(field)!r}"
                        + (f"   [ambiguous]" if g.get("ambiguous") else "")
                        + (f"\n      note: {g['note']}" if g.get("note") else ""))

        scored = {f: score_agreement(kept_gold, got, f, kind) for f, kind in fields}
        # The same, with items I flagged ambiguous removed. A judge should not be failed for
        # disagreeing where a careful reader was unsure, and a judge should not be passed on
        # them either -- so both numbers are reported.
        clear = [(g, j) for g, j in zip(kept_gold, got) if not g.get("ambiguous")]
        if clear and len(clear) < len(kept_gold):
            cg, cj = [c[0] for c in clear], [c[1] for c in clear]
            for f, kind in fields:
                scored[f + " (unambiguous only)"] = score_agreement(cg, cj, f, kind)
        scored["_errors"] = errors
        all_results[judge_id] = scored

    _report(all_results, disagreements)
    if command == "full":
        print("\n`full` scoring of the whole archive is not built yet: fix the prompts against "
              "the sample first, then it is worth paying for the other ~1,100 responses.")
    return 0


def _report(results: dict, disagreements: list[str]) -> None:
    print("\n" + "=" * 78)
    print("AGREEMENT WITH HAND LABELS")
    print("=" * 78)
    for judge_id, fields in results.items():
        print(f"\n{judge_id}   ({fields.get('_errors', 0)} judge/parse errors)")
        for field, got in fields.items():
            if field.startswith("_") or not isinstance(got, dict) or not got.get("n"):
                continue
            if got.get("kind") == "categorical":
                kappa = got.get("kappa")
                ktxt = "undefined (one category only)" if kappa is None else f"{kappa:.3f}"
                print(f"   {field:<32} n={got['n']:<3} agree={got['agreement']:.3f}  kappa={ktxt}")
                for pair, count in (got.get("confusion") or {}).items():
                    print(f"       {count}x {pair}")
            else:
                print(f"   {field:<32} n={got['n']:<3} MAE={got['mean_abs_error']:.2f}  "
                      f"within2={got['within_2']:.2f}  max={got['max_abs_error']:.1f}  "
                      f"(gold mean {got['gold_mean']:.1f} vs judge {got['judge_mean']:.1f})")

    print("\n" + "=" * 78)
    print("VERDICTS vs the stated bar")
    print("=" * 78)
    for row in verdicts(results):
        print(f"   {row['verdict']:<12} {row['criterion']:<26} {row.get('detail','')}")

    print("\n" + "=" * 78)
    print(f"EVERY DISAGREEMENT ({len(disagreements)})")
    print("=" * 78)
    print("This is where the information is. A systematic disagreement says what to change")
    print("about the prompt; the headline number does not.\n")
    for line in disagreements:
        print(line)
    if not disagreements:
        print("   none — which on a sample this small is worth being suspicious of, not pleased "
              "about.")
