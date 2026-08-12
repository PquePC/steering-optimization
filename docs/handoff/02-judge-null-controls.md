# 02 — A judge null control per judged measure

**Status: BUILD NOW — PROPOSE FIRST.** The three thresholds and the abort-versus-degrade behaviour are design calls: propose them before writing code. Blocking.

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

## Escalate before implementing

- **The three thresholds.** Propose each with its derivation.
- **Whether a failing null should abort or degrade.** Aborting is the right default (a floor under
  `e5` makes every downstream number uninterpretable), but for `s1` a ceiling may be tolerable if
  it is recorded and carried into the report. Make the argument; do not decide it in code.

## Acceptance

- Each null has a test that **trips it** — feed the check a judge stub that returns the bad reading
  and confirm the gate fails. A null control that cannot fail is worse than none.
- Running the preflight surfaces all three nulls' readings.
- A failing null stops the run at Phase 0, with a message naming which judge and which measure.
- The three readings appear in the run record and in the end-of-run gate table.
