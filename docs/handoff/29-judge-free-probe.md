# 29 — Measure the mid band directly, judge-free

**Status: BUILT — awaiting the run.** Branch `probe/mid-band-validation`, a temporary line off
`pareto`. Adds `m2/unjudged.py`, `prompts.detect_prompts` and `--probe-cells`. Nothing on the
pipeline path changed except `model.provenance()` (item 16) and one new keyword on
`run.check_environment`.

**This does not replace the pipeline and does not fix `e6`.** It is one measurement, made with
the selection gate removed, whose only job is to say whether the mid band behaves the way M1 says
it does. What to do about `e6` is a decision to take *after* reading its transcripts, not before.

---

## Why the last run could not have found anything else

`SHORTLIST` offers `d2` only cells that clear an `e6` reach floor
(`phases._eligible_scan_cell`, `D2_SELECT_REACH_COUNTS = (3, 2, 1)` over 12 prompts). `e6` is
`cheap.measure_E6`: one forward pass, `logits[0, -1, :]`, the concept's mass at **the first
generated token** of an answer to `E5_PROMPTS` — *"What is 17 x 23?"*, *"What is the capital of
Australia?"*. A steered model that is still answering the question opens with `The` or `17`, so
the concept mass there is ~0 and the cell is not eligible. `e6` only turns positive once the
injection can hijack the opening token of an unrelated answer, which is the collapse regime.

The 2026-08-14 SCAN says it plainly:

| cell | `d3` | `s3` | `e6` reach |
|---|---:|---:|---:|
| L37 @ 0.300 | 0.9998 | 0.905 | **0.000** |
| L38 @ 0.300 | 1.0000 | 0.881 | **0.000** |
| L43 @ 0.375 | 1.0000 | 0.833 | **0.000** |
| L45 @ 0.600 | 0.9065 | 0.810 | **0.000** |

The forced-ID next token *is* the concept with p ≈ 1 and MMLU is at 88% of baseline, while the
effectiveness axis reads zero. Compounding it, `s3` — the scan-time sanity term — is a four-way
argmax over option-letter logits and cannot see generative collapse at all, which is how the last
probe produced cells with `s3` ≈ 0.95 and 5/5 responses tripping `ngram_repeat` (item 27).

**Effectiveness that only fires at collapse, plus sanity that is blind to collapse, selects the
broken cells and certifies them.** Every `d2`-measured cell in that run was at L47 or deeper.
`d2 = 1.000` on 38/38 is a correct measurement of the wrong cells.

M1 measured the band that was skipped: L37 at α = 2 (r ≈ 0.20) gave `D2` 0.96, `D1` 0.08,
sanity 0.93, E4 KL 0.69, zero degeneracy, transcripts verified
([RESULTS.md §2](../RESULTS.md)). That cell would read `e6` = 0.000.

## What the probe measures

`python -m m2.run --concepts Garlic --probe-cells` — 40 mid-band cells (L37–L46 ×
r ∈ {0.15, 0.20, 0.25, 0.30}) plus two anchors, with **no judge and no selection**.

Per cell, three response channels because they are three different questions:

| channel | prompt | what it answers |
|---|---|---|
| `detect` | `prompts.detect_prompts` — the noticing question, **no prefill** | would the model *volunteer* the detection |
| `forced` | `prompts.forced_prompts` — D2's prompt, prefilled | is the concept reachable under maximal prompting |
| `task` | the 12 E5 prompts, generated | does the injection show up in **ordinary output** |

`task` is the measurement `e6` was standing in for. Plus `e6`, `d3` and `s3` per cell, recorded
and used for nothing — having `e6` = 0.000 in the same row as a positive task-channel mention
rate is what turns the argument above into a table.

**The unsteered null arm is mandatory.** The framing states a thought is injected on half of
trials, which is an invitation to answer yes; M1 §4 recorded the model inventing *"the word
apple"* under it with nothing injected. All three channels are measured at α = 0 first, so a
steered "Yes, I detect an injected thought" can be read against the rate the prompt produces on
its own. A run cut short still has it.

## Design decisions worth knowing

**One noticing builder, two prefills.** `prompts._noticing_prompts` now holds the framing text and
`forced_prompts` / `detect_prompts` differ only in the string appended. R7
(`verify_forced_prompts`) still checks the prefilled prompt byte-for-byte against the repo's own
function; nothing checks the unprefilled one, so the only guarantee available is that it is the
same construction minus one string — which a source-level test now pins.

**Judge-free is enforced, not asserted.** `unjudged.no_judges()` patches `expensive._issue`,
`judges.call_judge` and `judges.judge_many` to raise for the duration of the run, and a
source-level test rejects any call to a judged measurement (`measure_D2` included — it is the
obvious thing to reach for when someone later wants "just the detection number"). Consequently
`OPENROUTER_API_KEY` is not required in this mode; `HF_TOKEN` still is.

**Detection is a transcript, not a rate.** `concept_hits` is a mechanical count of the concept
word with a leading word boundary and no trailing one, so *garlicky* counts. It cannot tell "about
garlic" from "mentions garlic once", and it cannot see an influenced response that never names the
concept — M1's origami cell described *"a delicate, precise unfolding"* for a sentence before
naming it. It makes transcripts sortable. It does not grade influence.

**Anchors, not controls.** L57@0.22 (the one fully sane cell in the last probe: `s2_forced` 5/5,
`d2` 5/5) and L59@0.30 (`d2` 5/5, `s2_forced` 0/5, every response a loop). Nothing raises on
them. Task 25's positive control gated its whole dump on a criterion item 27 then showed a broken
model passes; an anchor that is read rather than enforced does not repeat that.

## Acceptance criteria

1. `python -m m2.run --concepts Garlic --probe-cells --dry-run` plans 42 cells and loads no model.
2. A completed run writes `probe_cells.jsonl` (42 scalar rows), `probe_detail.jsonl`, the four
   transcript files, `probe_summary.json`, `norms.jsonl`, `dose_map.json` and `provenance.jsonl`
   with a non-null `git_commit`.
3. No `judge_*.jsonl` exists in the run directory. If one does, the guard was bypassed.
4. The null arm is present in `probe_null_transcripts.jsonl` for all three channels.
5. Every cell that raised `Unreachable` is named in `probe_summary.json` under `unreachable` —
   item 26's lesson: work a run silently did not do reads later as work it did and found nothing in.
6. `rig checks` prints before the probe starts, and `R14 : pass` prints after extraction. Both
   are gates, not banners — see below.

## R14 is the check this mode lives or dies on

The dispatch sits **after** `gates.rig_checks()` and after the `--preflight` early return, and
`run_probe` calls `model.hook_liveness()` itself once the vectors exist.

Both halves were needed. The first version returned before the rig checks ran at all. Fixing
that is still not enough on its own: R14 reads `RUN.vecs`, so it *skips* at rig-check time when
nothing has been extracted, and on the pipeline path `phase0_calibrate` is what runs it — a phase
this mode never enters.

It matters more here than on the pipeline path. Bug 26 was the repo's hook declining to steer
whenever `start_pos` was set; it produced identically zero readings at all 30 cells of a real run
for an hour, with every other check satisfied. **This probe's entire finding is whether the mid
band shows influence,** so a dead hook returns a clean, plausible, completely empty null across 42
cells, and every mechanical check in the probe passes on it. `hook_liveness` raises.

## What this run cannot settle

- **It is not an operating point.** n = 8 per noticing channel; every rate here has a Wilson
  interval a third of the unit wide. It ranks nothing and confirms nothing.
- **It does not measure `e5`.** Influence is a mention count. A cell with a high count may be
  fixating rather than being influenced, and a cell with a zero count may still be influenced.
- **It does not validate `d3`.** Gate 5 needs `d2` variance against a judged axis, and this run
  has no judge.
- **A negative is weak.** If the mid band comes back with no volunteered detection and no
  ordinary-output influence, that is n = 8 on one concept at four doses, not a result about
  Garlic.

## Follow-on, once the transcripts are read

Do not build these yet. Which is right depends on what the run says.

- If the task channel shows influence where `e6` = 0: `e6` gets reclassified from effectiveness
  proxy to **saturation detector** (`e6` = 1.0 is a red flag, not a green one), and the scan needs
  a real influence axis — KL against unsteered on the same 12 prompts is the cheap candidate, M1's
  E4, which did not survive into M2.
- If the mid band is sane on all three channels: the shortlist eligibility rule is the defect, and
  it is the thing to rewrite before another full run.
- Either way, `s3` alone can no longer be the scan-time sanity term (open item 30).
