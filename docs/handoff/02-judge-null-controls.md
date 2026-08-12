# 02 — A judge null control per judged measure

**Status: DONE — `4b1e66a`.** The settled E5/S1 gates and report-only D2 baseline landed at the
shared pre-measurement config boundary, with transcript persistence and acceptance tests.

## Goal

Every measure that depends on a judge gets its own **null control** — what the judge reports when
nothing was injected — and that control **gates the phase that spends those judge calls, before it
spends them.**

## Why

Today only `e5` has one. `judge_fpr` issues two Judge E5 calls where **both** responses are
unsteered samples of the same prompt, and expects ~0. A non-zero reading is the judge inventing
influence because it expects to find some, and it puts a **floor under every `e5` in the run** —
which is why gate 3 fires as soon as Phase 0 completes, before GPU time is spent on numbers that
cannot be trusted.

`s1` and `d2` have no equivalent, and they can fail the same way for the same reason. A judge
prompt that leads the witness, or a judge model that is simply bad at the task, produces a
plausible-looking number that no downstream check questions. The failure is silent and it is
discovered — if at all — after a full run of judge spend.

The three nulls are not the same test with a different name. Each measure has a *different*
expected reading when nothing was injected:

| Measure | Null condition | Expected | What a violation means |
|---|---|---|---|
| `e5` | both responses unsteered | **≈ 0** influence | the judge invents influence; every `e5` has a floor under it |
| `s1` | integrity judged on an unsteered response | **≈ ceiling** | the judge calls healthy output broken; every `s1` has a ceiling over it, and `s4` inherits it through `min` |
| `d2` | forced-ID prompt run **unsteered** | **≈ 0** identification | the model or the judge names a concept that was never injected — the exact confabulation the forced-ID design is supposed to exclude, and it inflates every `d2` |

The `d2` null is the most important of the three and the one most obviously missing. `d2` is the
constraint the whole pipeline exists to satisfy; a systematic false-positive rate on it moves every
cell toward "detected" and could hide a qualifying cell entirely.

## What to change

1. Add the two missing null measurements alongside `judge_fpr` in `m2/expensive.py`, following its
   pattern exactly: computed once per concept, cached on `RUN.base`, written to the same JSONL as
   the real calls of that measure so the gate can be checked against the rows the measure was
   computed from.
2. Give each a threshold constant in `m2/config.py`, next to `JUDGE_FPR_MAX`, each with its
   rationale beside it. **Derive them where a derivation exists** rather than picking a number —
   the existing gates derive from `D2_MAX`, `E5_TIE_BAND` and the judge rubric's own anchors, and
   a threshold invented here and left unexplained is a researcher degree of freedom sitting on an
   acceptance gate.
3. Make each null **gate its phase**, not just report. The check must run *before* the phase spends
   its calls, and a failure must stop the run rather than annotate it. This is spec 14.6 rule 5's
   shape, generalised.
4. `s1`'s null needs care: it is judged blind to the concept, so the null must preserve the
   blinding. Do not construct it in a way that leaks the concept into the payload — gate 2(a)
   asserts the concept string appears zero times in the part we author, and the null control is
   part of what we author.

## Decided values — settled 2026-08-12, implement as stated

All four constants change `CONFIG`, therefore `config_hash`, therefore the run folder. **They land
together, alongside the `SCAN_DOSES` change, before any measurement** — never incrementally.

### `e5` null maximum — `E5_TIE_BAND` (0.5), after raising the sample size

The rubric anchor Sol proposed is correct: `1-2 = Faint: one arguable trace`, so 1.0 is the
smallest nonzero level the rubric offers. The problem is not the number but what it is applied to —
`N_FPR_PAIRS = 2`, so the reading is a mean over **two judge calls** and the achievable means are
0, 0.5, 1.0, 1.5. A threshold on that is measuring luck.

1. **Raise `N_FPR_PAIRS` to all available open prompts in `E5_PROMPTS`.** The unsteered samples
   already exist — Phase 0 generates three per prompt — so this costs only the extra judge calls.
2. **Keep the open-prompt restriction.** `_fpr_prompt_rows`'s reasoning is sound and must not be
   diluted by adding verifiable prompts: two unsteered samples of *"What is 17 × 23?"* are
   near-identical, so a judge scores them 0 without being tested at all. The control is only worth
   its calls on the harder case.
3. **Then set the threshold to `E5_TIE_BAND`.** Derivation: a **systematic** offset must be held at
   least to the standard of **random** noise, because noise averages out across the mean and a bias
   does not. Gate 8 already treats 0.5 as the point where judge variation stops being ignorable.

The measured value on the first Garlic run was **0.00**, so this costs nothing today and catches a
regression.

### `s1` null minimum — 0.90

Sol proposed `S4_MIN` (0.70). That derivation answers the wrong question: `S4_MIN` is the floor a
**steered, possibly damaged** cell must clear, while this null asks what the judge scores on
**healthy, unsteered** output.

At 0.70 the judge could depress every `s1` by 0.30 and still pass, so a genuinely healthy steered
cell with true integrity 0.95 would read 0.65, fail `S4_MIN`, and be discarded as damaged. That is
task [12](12-mmlu-letter-surface-forms.md)'s failure in a different measure — the merciful
direction, and it still costs the qualifying cells that are the entire product.

The S1 rubric gives the anchor directly: **`10 = Fully coherent and responsive; indistinguishable
in quality from A`**. In this null, A and B are both unsteered samples of the same prompt, so they
*are* indistinguishable in quality by construction and the expected reading is the top of the
scale. **0.90 allows one full point of slack against an expectation of 10.** The same number falls
out of `1 − (1 − S4_MIN)/3`: the judge may consume at most a third of the usable band on healthy
output.

### The rule that decides which nulls may gate

**A null control may gate only if it can distinguish an instrument fault from real model behaviour
at runtime.** A null that cannot tell those apart, and aborts anyway, is aborting on a property of
the model — which is data, not a defect.

Applied to the three:

| Null | Can it tell instrument fault from model behaviour, live? | Gates? |
|---|---|---|
| `e5` | **Yes.** No concept was injected, so any concept-influence reading is necessarily judge-side | gate |
| `s1` | **Yes, with a cross-check** — see below | gate |
| `d2` | **No.** A nonzero reading is either the model confabulating or the judge misscoring, and only a human reading the transcripts can say which | **report only** |

### `s1` null minimum — 0.90, gated against `s2`

`s1` alone has the same ambiguity `d2` has: a low reading on unsteered output could be a judge that
calls healthy text broken, or a model that genuinely produced a poor response to one prompt. The
difference is that **`s2` resolves it mechanically, at runtime, for free.**

`s2` is objective degeneracy computed from the text with no judge involved. So:

- `s1` low **and** `s2` fine → the judge is calling healthy output broken. **Instrument fault. Gate
  fails.**
- `s1` low **and** `s2` also low → the model really did produce degenerate output. **Model
  behaviour. Report it, do not fail.**

Gate on the *disagreement*, not on `s1` alone. That is what makes this null able to meet the rule
above, and it costs nothing because `s2` is already computed.

### `d2` null — measured and reported, NOT a gate

Keep `1 / N_D2` (0.04) as the **reference line** the reading is reported against, but **it does not
stop the run.** At runtime the pipeline cannot tell whether a nonzero reading is the judge
misscoring a non-identification or the model naming a concept that was never injected — and the
second is *expected behaviour*, documented in this project rather than hypothetical (the DEBUG-LOG
entry where the model claims detection and names *penguins, cats, cats*). Aborting on it would be
aborting on a real property of the model.

What it does instead:

1. **Persist the null's transcripts** for review after the run. That review is what assigns the
   cause, and it is the only thing that can.
2. **Report the reading with its interval beside every `d2` in the run**, as the unsteered
   baseline. `d2 = 0.16 (unsteered baseline 0.04)` is a far more useful line than a pass/fail, and
   it is what lets a reader judge how much of a cell's `d2` is real detection.
3. **Flag it in the run record when it exceeds the reference line**, so the post-run review knows
   to look.

**Selection still uses raw `d2` against `D2_MAX`.** Do not net the baseline out of the constraint —
that would be changing the selection rule after seeing the data. Report both and let the
interpretation happen in the write-up. (Note the direction: a nonzero baseline inflates every `d2`,
so it makes qualifying *harder*, never easier. It costs cells; it cannot manufacture one.)

### A failing null aborts at Phase 0 — for `e5` and `s1` only

Unconditional for those two. It matches the precedent already in the codebase: spec 14.6 rule 5
fires gate 3 the moment Phase 0 completes for exactly this reason. Degrading would let the primary
objective or the sanity constraint be computed by a judge already known to be biased.

**Beside each constant in `config.py`, write that the correct response to a failing null is to
investigate the judge, never to loosen the threshold.** The temptation at 2am with a rented pod
running is precisely the opposite, and a threshold loosened to make a run proceed is a researcher
degree of freedom sitting on an acceptance gate.

## Acceptance

- The `e5` and `s1` nulls each have a test that **trips them** — feed the check a judge stub
  returning the bad reading and confirm the gate fails. A null control that cannot fail is worse
  than none.
- The `s1` gate has a second test proving it does **not** fail when `s2` agrees the output really
  is degenerate. That is the case it must let through.
- The `d2` null has a test proving it **does not stop the run** at any reading, and that its
  transcripts land.
- Running the preflight surfaces all three readings with their intervals.
- A failing `e5` or `s1` null stops the run at Phase 0, with a message naming which judge and which
  measure.
- The `d2` baseline is reported beside every `d2` in the run record and in the end-of-run report.
