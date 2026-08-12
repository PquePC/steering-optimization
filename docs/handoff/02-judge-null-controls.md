# 02 — A judge null control per judged measure

**Status: BUILD NOW.** Blocking. The thresholds were proposed by Sol and settled by the operator on
2026-08-12 — they are in **Decided values** below. No further proposal needed; implement as stated.

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

### `d2` null maximum — `1 / N_D2` (0.04), plus two additions

The derivation is right: at n = 25 this is the smallest enforceable nonzero threshold, one false
identification in twenty-five.

1. **Persist the null's transcripts, and point the failure message at them.** A nonzero `d2` null
   has two causes with opposite remedies: the **model** confabulating a concept that was never
   injected, or the **judge** wrongly scoring a non-identification. Model confabulation is
   documented in this project, not hypothetical — see the DEBUG-LOG entry where the model claims
   detection and names *penguins, cats, cats*. Only the transcripts distinguish them.
2. **Report the Wilson interval, not only the point estimate.** 0/25 and 1/25 are not meaningfully
   different at that n, and the gate must not imply precision it does not have.

### A failing null aborts at Phase 0

As proposed, unconditionally, for all three. It matches the precedent already in the codebase —
spec 14.6 rule 5 fires gate 3 the moment Phase 0 completes for exactly this reason. Degrading would
let the primary objective, the sanity constraint or the detection constraint be computed by a judge
already known to be biased.

**Beside each constant in `config.py`, write that the correct response to a failing null is to
investigate the judge, never to loosen the threshold.** The temptation at 2am with a rented pod
running is precisely the opposite, and a threshold loosened to make a run proceed is a researcher
degree of freedom sitting on an acceptance gate.

## Acceptance

- Each null has a test that **trips it** — feed the check a judge stub that returns the bad reading
  and confirm the gate fails. A null control that cannot fail is worse than none.
- Running the preflight surfaces all three nulls' readings.
- A failing null stops the run at Phase 0, with a message naming which judge and which measure.
- The three readings appear in the run record and in the end-of-run gate table.
