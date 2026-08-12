# 16 — Judge bake-off, after the Garlic run

**Status: FUTURE — DO NOT BUILD.** Runs after Garlic and Origami are complete. Recorded now so this run captures what it will need.

## The current judge

`judge_model = "openai/gpt-4.1-mini"` via OpenRouter, `judge_concurrent = 32`.

It is not an arbitrary pick: the v1 rig check passed at **0.377 against a published 0.382** on
exactly this pairing, which is the only end-to-end validation any judge in this project has. That
is a reason to keep it until something demonstrably beats it, not a reason to keep it forever.

## The economics, which point somewhere unintuitive

Per concept: roughly **$1.60–2.50 total, of which judges are ~$0.50.** GPU dominates.

So the judge is around a quarter of run cost. Two consequences worth stating plainly before
optimising anything:

- **Switching to a cheaper judge saves little.** Halving judge cost saves ~$0.25 per concept.
- **Switching to a worse judge can cost the entire result.** `e5` is the primary effectiveness
  metric; `d2` is the constraint the pipeline exists to satisfy; `s1` feeds `s4` through a `min`.
  A judge that reads influence badly makes "qualifies" meaningless.

That asymmetry says the target is **not "cheapest that works"** but **"cheapest that is
demonstrably as good, where 'as good' is measured rather than assumed"**. A cheaper judge that
passes every criterion below is a real win; one that passes most of them is not worth $0.25.

## Why this is cheap to run

Generation needs an 80GB GPU. **Judging needs only an API key.** The two are fully separable, and
the pipeline already stores everything needed to replay judging offline: `judge_e5.jsonl`,
`judge_s1.jsonl`, `judge_d2.jsonl`, `D2_transcripts.jsonl`, `cis_transcripts.jsonl`.

So the bake-off is: take **one finished run's stored transcripts**, re-judge them with each
candidate, and compare. No pod, no GPU, no regeneration. The parent research repository has
`scripts/rejudge_local.py`, written for the v1 format on exactly this reasoning — *"terminate the
pod, then fix the labels at your leisure"*. Read it before writing anything; the argument and the
shape are already there even though the format differs.

**Every candidate must be scored on the same stored transcripts.** A comparison where two judges
saw different generations is measuring the generations.

## Criteria — all of them already exist as gates

The evaluation is not new work. Each candidate judge gets scored on the checks already built:

| Criterion | Source | What it catches |
|---|---|---|
| Anchor separation on `e5` | gate 1 | a judge counting concept words instead of reading influence |
| Agreement with hand labels | gate 1 diagnostic | general calibration |
| Self-disagreement on re-judging | gate 8 | noise floor — must sit under `E5_TIE_BAND`, or tie-breaks are reading noise |
| Agreement with the upstream judge on `d2`, **and the rate delta** | gate 11 | whether `d2` keeps its meaning; symmetric disagreements cancel in a rate, a systematic shift does not |
| The three null controls | task [02](02-judge-null-controls.md) | inventing influence, calling healthy output broken, identifying a concept never injected |
| Cost per run | — | the reason for the exercise |

Gate 1's hand labels are the pivot: they are the only judge-independent ground truth in the
project, which is exactly why task [04](04-gate1-anchors.md) keys them to the **judge
configuration** rather than the concept. **The bake-off is the reason that keying exists** — one
labelled set, scored against every candidate.

## One threshold to look at while you are there

`JUDGE_FPR_MAX = 1.0`, on a 0–10 scale where `E5_FLOOR = 4.0`. That permits a full point of
invented influence — a quarter of the floor — before gate 3 objects. It may be right; it does not
look obviously right, and no note beside it explains the value. Re-derive it during the bake-off,
when there is data on what several judges actually score on unsteered pairs.

## What this run must capture for it

Nothing extra to build — but **do not prune the stored judge rows** during the Garlic and Origami
runs, and make sure the debug bundle from task [08](08-debug-bundle.md) carries them. The bake-off
is only cheap because those files exist.

## Caveats to carry into the design

- **Changing the judge changes `config_hash`**, so a new judge means new run folders and no resume
  against old ones. That is correct — `e5` from two judges are not the same measurement — but it
  means the switch happens between concepts, never mid-run.
- **A judge that wins on aggregate agreement can still lose where it matters.** Report per-criterion
  results, not a composite score. A judge with better mean agreement and worse anchor separation is
  worse for this pipeline.
- Model availability and pricing move. Fix the candidate list at the time the bake-off runs rather
  than encoding it now.
