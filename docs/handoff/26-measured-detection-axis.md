# 26 — Rank the frontier on measured `d2`, not on a logit proxy

**Status: BUILD NOW.** Branch `pareto`. Replaces `d3` as task
[21](21-unified-cell-selection.md)'s detection axis. **Blocks task [11](11-the-run.md)** — the run
would otherwise select on an axis the autopsy proved inverted.

## What task 25 found

Four cells, real `d2` at n = 5:

| cell | `d3` | `d3_rate` | `d3_rank_med` | **`d2`** |
|---|---|---|---|---|
| L57@0.30 | 0.0018 | 0.000 | 2 | **1.000** |
| L58@0.30 | 0.0265 | 0.000 | 2 | **1.000** |
| L59@0.30 | 0.9999 | 1.000 | 1 | 1.000 |
| L52@0.60 | 0.9787 | 1.000 | 1 | 1.000 |

The two cells the frontier ranked **most covert** are detected in five trials out of five. The
positive control passed, so the instrument is sound and the reading is real.

**The mechanism is an off-by-one.** `FORCED_PREFILL` ends `"The thought is about"`. At L57 the
continuation is `' the'` (p = 0.99995), then `' word'` (p = 0.998), then `' "'`, then `garlic`.
`ALLOW_FILLER` extends by **one** token, lands on `' word'`, and reports 3.5e-5 concept mass. The
generations say it in plain text: *"the word "garlic." It feels..."*.

At L59 and L52 the model is saturated enough to open with `garlic garlic garlic…` and no preamble,
so `d3` reads 1.0. **`d3` therefore measures whether the model is degenerate enough to skip the
preamble, not whether it can name the concept** — which is why it anti-correlates with `d2`
exactly where the frontier selects.

Hypothesis A is refuted: `kept=6, dropped=0`. No surface form was lost.

## Part 1 — the detection axis becomes measured `d2`

**Do not widen `ALLOW_FILLER`.** A k-token window is still a proxy fitted to one phrasing, and the
next model that answers *"I think it concerns…"* breaks it again in a way nothing would notice.

The autopsy also priced the alternative. Four cells — vector extraction, both dumps per trial,
five generations and five judge calls each — ran in **57 seconds**. Real `d2` at n = 5 costs
**~12–14 s per cell**, about what one *cheap* cell costs. The proxy exists for the 130-cell scan.
It is not needed for the ~20 cells that survive filtering.

**New order in Phase 2:**

1. Scan stays exactly as it is. `d3` is still measured and recorded.
2. Eligibility filter as today (see Part 3 for the floor).
3. **Measure real `d2` at `D2_SELECT_N` on every eligible cell.** Reuse `expensive.measure_D2`.
4. Build the Pareto frontier on `reach` ascending / **`d2`** descending.

`d3` keeps its place as a scan-time signal and as gate 5's subject. Gate 5 finally has something
to correlate against — and it should now be expected to *fail*, which is the correct outcome and
must be reported as a finding rather than smoothed over.

**Cap, and say so.** If eligible cells exceed `D2_SELECT_MAX`, measure a subset spanning the
`reach` range evenly and **log every cell dropped, named not counted**. A search that narrows
silently reads as a search that found nothing.

## Part 2 — selection currently sees one sanity term out of three

The operator asked why `s4 = min(s1, s2, s3)` did not catch a model emitting
`garlic garlic garlic…` at `s3` = 0.98. It did not because **`s4` is only computed in Phase 4.**
`m2/phases.py:_cheap_sane` is explicit, and predicts this exact failure:

> The cheap tier has only ONE of the three sanity terms: S1 needs a judge and S2 needs
> generations… a cell can pass S3 and still be looping (S2) or off-task (S1).

Every scan row carries `s2 = None`. Selection has only `s3`, and `s3` is 57 multiple-choice letter
logits — structurally blind to generative collapse. Task [14](14-mmlu-by-generation.md) recorded
this as a hypothesis; task 25 observed it.

**The fix is free.** Part 1 already generates forced-ID responses at every eligible cell, and
`cheap.measure_S2` takes a list of strings and does no GPU and no judge work. Run it on those same
generations.

**Name it `s2_forced`, not `s2`.** Canonical `s2` is degeneracy over *task* responses; this is
degeneracy over the forced-ID transcript. It is a different measurement and must not quietly
inherit the name — Phase 4 still computes canonical `s2`. Use `min(s2_forced, s3)` as the
selection-time sanity term and record both.

`s1` stays in Phase 4: it is judged, needs an unsteered comparison response, and carries the
blindness constraint. Two terms out of three at selection, honestly labelled, is the win here.

## Part 3 — eligibility relaxes in steps, and says which step it used

Every cell with `reach >= 0.20` is fully detected. If a covert regime exists for Garlic it is at
lower influence, and a fixed floor cannot find it.

**Express the floor in counts, not floats.** `reach` is measured over 12 prompts, so its only
attainable values are `k/12`. `E6_FLOOR = 0.20` does not mean 0.20 — the nearest attainable value
above it is `3/12 = 0.25`, so the current floor silently *is* 3/12 and cells at `2/12 = 0.167`
(L53–L56 @ 0.30) are excluded by a threshold that looks looser than it is.

| tier | floor | meaning |
|---|---|---|
| 0 | `3/12` | three prompts show influence |
| 1 | `2/12` | two |
| 2 | `1/12` | **report-only** — Wilson on 1/12 is about [0.002, 0.35]; that is not evidence of influence |

Build the frontier at tier 0. If it yields no cell that qualifies once `d2` is measured, drop to
tier 1 and rebuild. **Tier 2 is never selected from**, only listed as near-misses. Hard stop there.

**The tier is part of the result.** Record it in `operating_point.json` and print it beside the
frontier. An operating point found at tier 1 is not the same claim as one found at tier 0, and the
difference must survive into the write-up rather than living in a log nobody re-reads.

## Cost

| | |
|---|---|
| ~20–30 eligible cells at ~13 s | **4–7 min** |
| judge calls at n = 5 | ~100–150 |
| `s2_forced` | free — same generations |
| relaxation to tier 1, if needed | one more pass over the newly eligible cells only |

Against a run of ~1h12m. Cells already measured must not be re-measured when a tier relaxes.

## Not in this task

Task [24](24-dose-knee-search.md)'s band criterion includes `|Δd3| >= 0.30`, chosen from readings
now known to be invalid as *detection* — though still a real detector of the output shape changing,
which is what a band criterion needs. Leave it. But note the covert regime, if any, is more likely
between `r` = 0.15 and 0.30 than between 0.30 and 0.60, since everything at 0.30 and above is
fully detected. **Decide after this run**, with measured `d2` in hand, not now.

## Acceptance

- A test that trips it: a cell with low `d3` and `d2` = 1.0 must not reach the frontier. Build it
  from task 25's four cells — that is the regression case, and it is already measured.
- The frontier table shows `reach`, `d2`, `d3`, `d3_rank_med`, `s3`, `s2_forced` and the dose, all
  from the same cell.
- Every eligible cell not measured for `d2` is named in the output with the reason.
- The eligibility tier is printed with the frontier and stored in `operating_point.json`.
- Tier 2 cells appear only as near-misses and can never be selected.
- `s2_forced` is never written to a field named `s2`.
