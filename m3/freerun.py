"""m3.freerun - ask the steered and unsteered model anything, side by side, at one cell.

    python -m m3.freerun --concept Garlic --layer 29 --dose 0.114438

`--dose` is copy-pasteable straight out of `cells.jsonl` or `FINDINGS-*.md`. Everything is
measured LIVE in this process: the vector is re-extracted, the norms re-measured, alpha
recomputed, and the battery re-run. Nothing is read from a previous run's numbers, so a
disagreement with the sweep is real information rather than a stale file.

Why that matters. The sweep's numbers were produced by a specific model load, a specific vector
extraction and a specific sampling seed. Printing them here would make this tool a viewer for
old results; re-measuring makes it an independent check on them. It has already earned that:
every deep defect in this project was found by a person reading generations, and this is the
shortest path from a coordinate in a results table to the text the model actually produces.

## What it does

1. Loads the model, extracts the concept vector at the chosen layer, measures residual norms.
2. Converts `dose` to `alpha` through the same `alpha_for` the sweep uses, and refuses a dose
   the ALPHA_CEIL forbids rather than clamping it.
3. Runs the standard battery at that cell and prints every statistic, computed here and now.
4. Then hands you a prompt. Each question is answered `n` times unsteered and `n` times steered,
   printed together, with the mechanical measures on every response and -- if a judge key is
   present -- judged influence and coherence too.

## Cost

One model load, then a generation batch per question. No judge calls unless a key is set, and
none at all for the mechanical measures. The battery in step 3 costs what one sweep cell costs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

_PKG_PARENT = Path(__file__).resolve().parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from m3 import battery, config, judge, sweep          # noqa: E402


EXIT_OK, EXIT_FAILED, EXIT_CONFIG = 0, 1, 2

_RULE = "=" * 78


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m m3.freerun",
        description="Ask the steered and unsteered model anything, at one (layer, dose).")
    p.add_argument("--concept", "-c", required=True,
                   help="the concept to inject. Any concept except the harmful arm.")
    p.add_argument("--layer", "-l", type=int, required=True, help="layer to inject at")
    dose = p.add_mutually_exclusive_group(required=True)
    dose.add_argument("--dose", "-d", type=float,
                      help="NORMALISED strength, copy-pasteable from a run's cells.jsonl")
    dose.add_argument("--alpha", type=float,
                      help="raw multiplier instead of a dose. For when you know what you want.")
    p.add_argument("--n", type=int, default=3,
                   help="responses per arm, per question (default 3)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="tokens per response (default: MAX_NEW_TOKENS from m3.config)")
    p.add_argument("--no-battery", action="store_true",
                   help="skip the opening battery and go straight to questions")
    p.add_argument("--no-judge", action="store_true",
                   help="never call a judge, even if a key is set")
    p.add_argument("--prompt", action="append", default=[],
                   help="ask this and exit. Repeatable; skips the interactive loop.")
    p.add_argument("--model", default=None,
                   help="override m3.config MODEL (the key the harness knows it by)")
    return p


# =====================================================================================
# Reporting
# =====================================================================================

def _num(v, spec="6.2f"):
    return format(v, spec) if isinstance(v, (int, float)) else "     -"


def _print_cell_stats(cell: dict, concept: str) -> None:
    """Every statistic for this cell, named, with its n and interval."""
    print(_RULE)
    print(f"CELL STATISTICS   L{cell['layer']} @ dose {cell['dose']:.6f}   "
          f"alpha {cell['alpha']:.4f}   concept {concept}")
    print(_RULE)
    print("  measured in THIS process; nothing below is read from a previous run\n")

    def rate(name, r, note=""):
        if not r:
            print(f"  {name:<34}      -   (no readable trials) {note}")
            return
        print(f"  {name:<34} {r['rate']:.2f}   {r['count']}/{r['n']}  "
              f"95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}] {note}")

    def scalar(name, m, note=""):
        if not m:
            print(f"  {name:<34}      -   (no readable trials) {note}")
            return
        se = f"se {m['se']:.2f}" if m.get("se") is not None else "se —"
        print(f"  {name:<34} {m['mean']:.2f}   n={m['n']}  {se}  "
              f"range {m['min']:.0f}–{m['max']:.0f} {note}")

    print("  DETECTION  (low = the model does not register the injection)")
    rate("identification (forced)", cell.get("identification"),
         "← prefilled; measures ABILITY to name")
    rate("  ... excluding collapses", cell.get("identification_excluding_degenerate"),
         f"({cell.get('identify_degenerate_n', 0)} collapsed)")
    rate("claims detection (self-report)", cell.get("self_report"),
         "← unprompted; measures WILLINGNESS")
    rate("  ... and names the concept", cell.get("self_report_names_concept"))
    if cell.get("self_report_classes"):
        print(f"  {'self-report classes':<34} {cell['self_report_classes']}")
    if cell.get("identified_as"):
        print(f"  {'named when forced':<34} {', '.join(cell['identified_as'])}")

    print("\n  EFFECT  (high = the concept is shaping output)")
    scalar("effectiveness (0-10)", cell.get("effectiveness"),
           "← judged vs this run's own unsteered reply")
    if cell.get("effect_forms"):
        print(f"  {'effect forms':<34} {', '.join(cell['effect_forms'])}")

    print("\n  INTACTNESS  (gates — a broken model fakes low detection)")
    scalar("coherence (0-10)", cell.get("coherence"), "← judged blind to the concept")
    rate("capability", cell.get("capability"), "← verifiable answers, on generated text")

    print("\n  MECHANICAL  (recorded, decides nothing)")
    mech = cell.get("mechanical") or {}
    for ch in ("identify", "effect", "self_report"):
        m = mech.get(ch)
        if not m:
            continue
        print(f"  {ch + ' degeneration':<34} {m['degeneration']['rate']:.2f}   "
              f"mentions median {m['mentions_median']}  words median {m['words_median']}")
    if cell.get("judge_errors"):
        print(f"\n  judge errors: {cell['judge_errors']}")
    print()


def _print_pair(prompt: str, unsteered: Sequence[dict], steered: Sequence[dict],
                judged: dict | None) -> None:
    print(_RULE)
    print(f"PROMPT: {prompt}")
    print(_RULE)
    for label, rows in (("UNSTEERED", unsteered), ("STEERED", steered)):
        print(f"\n--- {label} ---")
        for i, r in enumerate(rows, 1):
            flags = []
            if r["degenerate"]:
                flags.append(f"DEGENERATE({r['degeneration_reason'].split(':')[0]})")
            if r["empty"]:
                flags.append("EMPTY")
            tag = ("  [" + ", ".join(flags) + "]") if flags else ""
            print(f"\n  [{i}] mentions={r['concept_mentions']} words={r['words']}{tag}")
            for line in r["response"].strip().splitlines():
                print("      " + line)
    print()
    _print_live_stats(unsteered, steered, judged)


def _print_live_stats(unsteered, steered, judged) -> None:
    z = float(config.CONFIG["RATE_CI_Z"])
    print("  STATISTICS FOR THIS QUESTION")
    for label, rows in (("unsteered", unsteered), ("steered", steered)):
        s = battery.channel_summary(rows, z=z)
        print(f"    {label:<10} n={s['n']}  "
              f"mentions median {s['mentions_median']}  "
              f"any-mention {s['mention']['rate']:.2f} [{s['mention']['ci_low']:.2f},"
              f"{s['mention']['ci_high']:.2f}]  "
              f"degeneration {s['degeneration']['rate']:.2f}  "
              f"words median {s['words_median']}")
    if judged:
        for k in ("influence", "coherence", "on_task"):
            v = judged.get(k)
            if not v:
                continue
            if k == "on_task":
                # Printed as a rate and flagged, because a high coherence score beside a low
                # on-task rate is the exact reading that "fluent but answering a different
                # question" produces, and it is easy to miss when both are just numbers.
                flag = "   <- ANSWERED A DIFFERENT QUESTION" if v["mean"] < 0.5 else ""
                print(f"    {'on-task':<10} {v['mean']:.2f}   ({int(v['mean'] * v['n'])}/{v['n']} "
                      f"actually addressed the prompt){flag}")
            else:
                print(f"    {k:<10} mean {v['mean']:.2f}  n={v['n']}  "
                      f"range {v['min']:.0f}–{v['max']:.0f}")
    print()


# =====================================================================================
# The run
# =====================================================================================

def _judge_pair(prompt_text: str, unsteered, steered, *, concept: str, layer: int,
                dose: float, cfg: dict) -> dict:
    """Judged influence and coherence for one question. Returns {} on any failure.

    Deliberately non-fatal: a judge outage must not cost you the generations you are reading,
    which is the whole point of this tool.
    """
    chars = int(cfg["JUDGE_TEXT_CHARS"])
    items, kinds = [], []
    base = unsteered[0]["response"]
    for i, r in enumerate(steered):
        items.append(judge.build_item(
            "effect",
            payload=judge.render("effect", text_chars=chars, concept=concept, prompt=prompt_text,
                                 response_unsteered=base, response_steered=r["response"]),
            cache_key=judge.cache_key("FREERUN", "effect", layer=layer, dose=dose,
                                      unit=f"q{i}"),
            concept=concept, model_text=(base, r["response"]), text_chars=chars))
        kinds.append("influence")
        items.append(judge.build_item(
            "coherence",
            payload=judge.render("coherence", text_chars=chars, prompt=prompt_text,
                                 response=r["response"]),
            cache_key=judge.cache_key("FREERUN", "coherence", layer=layer, dose=dose,
                                      unit=f"q{i}"),
            concept=concept, model_text=(r["response"],), text_chars=chars))
        kinds.append("coherence")
    try:
        results = judge.run_judges(items, concurrency=int(cfg["JUDGE_CONCURRENT"]))
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (judge unavailable: {exc})")
        return {}
    got: dict[str, list[float]] = {"influence": [], "coherence": [], "on_task": []}
    for kind, result in zip(kinds, results):
        parsed, _err = judge.verdict(result)
        if not parsed:
            continue
        got[kind].append(float(parsed[kind]))
        # `on_task` rides along on every coherence verdict and was being discarded here. It is
        # the "did it actually answer the question" half, and on a free prompt with a right
        # answer -- "what is a computer" -- it is the half that matters: a response can score 8
        # for coherence and still be an article about a bulb.
        if kind == "coherence":
            got["on_task"].append(1.0 if parsed["on_task"] else 0.0)
    return {k: (battery.mean_se(v) if v else None) for k, v in got.items()}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = config.CONFIG
    if args.model:
        cfg["MODEL"] = args.model
    concept = str(args.concept).strip()
    max_tokens = int(args.max_tokens or cfg["MAX_NEW_TOKENS"])

    if not config.concept_allowed(concept):
        print(f"{concept!r} is on HARMFUL_CONCEPTS — the arm this study has deliberately "
              "not run.")
        return EXIT_CONFIG
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set; the model will not load.")
        return EXIT_CONFIG

    want_judge = not args.no_judge and bool(
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    from m2 import config as m2config, model, vectors

    os.environ.setdefault("M2_RUNS_DIR", str(config.runs_root()))
    print(f"loading {cfg['MODEL']} …")
    t0 = time.time()
    ctx = model.load_model(config.m2_config(concept, cfg))
    print(f"  {ctx.n_layers} layers  ({time.time() - t0:.0f}s)")

    if not 0 <= args.layer < int(ctx.n_layers):
        print(f"layer {args.layer} is outside 0–{int(ctx.n_layers) - 1}")
        return EXIT_CONFIG

    # A run dir is needed because the shared I/O layer refuses to write without one. Nothing
    # here writes measurements into it; it exists so `open_run`'s wiring can be reused verbatim
    # rather than reimplemented, which is how the two would drift apart.
    sweep.open_run(concept, cfg)

    print(f"extracting {concept!r} at L{args.layer} …")
    vectors.extract_all_layers(concept, [args.layer])
    vectors.measure_residual_norms([args.layer], [r["text"] for r in battery.TASK_PROMPTS])
    n = (m2config.RUN.norms or {}).get(int(args.layer)) or {}
    vec, resid = n.get("vec_norm"), n.get("resid_norm")
    ceiling = float(cfg["ALPHA_CEIL"]) * float(vec) / float(resid)
    print(f"  ||v||={vec:.1f}  ||h||={resid:.1f}  max reachable dose {ceiling:.4f} "
          f"(at ALPHA_CEIL={cfg['ALPHA_CEIL']})")

    if args.alpha is not None:
        alpha, dose = float(args.alpha), float(args.alpha) * float(vec) / float(resid)
    else:
        dose = float(args.dose)
        try:
            alpha = float(m2config.alpha_for(int(args.layer), dose))
        except m2config.Unreachable as exc:
            print(f"\nthat dose is not reachable at this layer: {exc}")
            print(f"the highest dose available at L{args.layer} is {ceiling:.6f}")
            return EXIT_CONFIG
    print(f"  dose {dose:.6f}  ->  alpha {alpha:.4f}\n")

    live = model.hook_liveness()
    print(f"R14 hook liveness: start_pos {live['d_start_pos']:.2e}  "
          f"all-pos {live['d_all_pos']:.2e}  passed={live['passed']}\n")

    if not args.no_battery:
        print("running the standard battery at this cell …")
        cal = sweep.calibrate(concept, [args.layer], cfg)
        cell = sweep.measure_cell(args.layer, dose, concept=concept,
                                  baselines=cal["baselines"], cfg=cfg)
        _print_cell_stats(cell, concept)

    questions = list(args.prompt)
    interactive = not questions
    while True:
        if questions:
            q = questions.pop(0)
        elif interactive:
            try:
                q = input("Prompt to ask (blank to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
        else:
            break

        from m2 import expensive
        rendered, starts, _ids = expensive.task_batch([dict(id="freerun", text=q)])
        text, start = rendered[0], int(starts[0])
        spec = [dict(prompt=text, start=start)] * int(args.n)

        un = sweep._generate(spec, layer=None, alpha=None, max_tokens=max_tokens, cfg=cfg)
        st = sweep._generate(spec, layer=args.layer, alpha=alpha, max_tokens=max_tokens, cfg=cfg)
        mk = lambda t: battery.response_row(t, channel="effect", concept=concept,  # noqa: E731
                                            unit="freerun")
        unsteered, steered = [mk(t) for t in un], [mk(t) for t in st]

        judged = (_judge_pair(q, unsteered, steered, concept=concept, layer=args.layer,
                              dose=dose, cfg=cfg) if want_judge else None)
        _print_pair(q, unsteered, steered, judged)

        if not interactive:
            continue
        try:
            again = input("Ask another question? (Y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if again in ("n", "no"):
            break

    print("done.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
