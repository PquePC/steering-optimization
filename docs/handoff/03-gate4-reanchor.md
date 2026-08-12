# 03 — Gate 4 on this run's own damaged cell

**Status: BUILD NOW.** Blocking. The anchor location and the pass criterion were settled on
2026-08-12 — see **Decided** below. The criterion is NOT `s4 < S4_MIN`; read that section before
writing code.

## Goal

Gate 4 stops depending on another concept's stored transcripts and anchors instead on a cell from
**this run** that is damaged by construction, measuring all three sanity terms live.

## Why

Gate 4 asks one question: **does `s4` actually reject a cell where the model is broken?**

It matters more than any other gate, because `S4_MIN` is the only thing separating the result the
pipeline is hunting for from its most convincing impostor. A cell with low `d2` because the concept
is genuinely covert and a cell with low `d2` because the model has been lobotomised look identical
in the numbers. v1 nearly published one: Velocity L37 α=3 read a merged sanity of 0.779 with
coherence 1.00 while `d2` fell from 0.44 to 0.00 — not because the concept went covert, but because
the model stopped working.

Two problems with the current implementation:

1. **It borrows Velocity's cell from a run whose data no longer exists.** A Garlic result should
   not need a Velocity measurement to be valid, and on a different model Velocity's numbers may
   mean nothing at all. This is the portability principle in [`BRIEF.md`](BRIEF.md) §5.
2. **Even with the file it is structurally weak.** Its own docstring concedes that from a stored
   transcript it can recompute `s2` and re-judge `s1`, but **`s3` needs the live model and is not
   recoverable from text**. So it runs on a subset of `s4`'s terms, and `min` over a subset is an
   *upper bound* — a subset that clears the floor proves nothing, which is why the gate currently
   fails-with-explanation rather than passing in that case.

## The fix

Phase 3 bisects the sanity boundary for each candidate layer. **Any cell above that boundary is
damaged by construction** — that is what the boundary means. So the run manufactures its own gate-4
anchor for free.

Take a cell above the bisected boundary at the winning layer and measure `s1`, `s2` **and `s3`**
live at it — the term the transcript path never could. Self-contained, portable to any model and
concept.

**The pass criterion is not `s4 < S4_MIN`** — that is guaranteed by how the boundary is defined.
See **Decided** below before writing any code.

## What to change

- Add the live-anchor path to `gate4_sanity_acceptance` in `m2/gates.py`, selecting the cell from
  the run's own bisection output.
- **Keep the stored-M1.5 path as a dormant fallback** rather than deleting it. If a v1 bundle ever
  resurfaces the historical comparison becomes available again.
- Record in the gate's output *which* cell was used and *why it is expected to be damaged* — the
  bisected boundary and how far above it the anchor sits. A gate that does not say what it tested
  is not auditable.

## Decided 2026-08-12 — the anchor location, and the criterion that must change with it

### The location: `hi`, the failing endpoint of the converged bisection interval

Sol's proposal, and it is right — but for a better reason than the one given. "Nearest observed
damaged cell, no arbitrary multiplier that could overshoot into saturation or unreachability" is
sound. The stronger reason is below.

### The criterion: `s4 < S4_MIN` **cannot** be the test

**As proposed, this gate could not fail.** `_cheap_sane` — the criterion the bisection boundary is
built from — is exactly:

```python
return float(s3) >= float(s4_min)
```

So `hi` is *defined* as a dose where `s3 < S4_MIN`. And `s4 = min(s1, s2, s3) ≤ s3`. Therefore
`s4 < S4_MIN` at `hi` **by arithmetic**, for every concept, on every model, whatever the judge says.
A gate whose pass condition is guaranteed by the definition of its own anchor is the defect family
this repository is organised around — and it would be landing inside the gate whose job is to stop a
false-positive result.

### What gate 4 is actually for

Go back to why the Velocity anchor mattered. v1's **merged** sanity metric passed L37 α=3 at
**0.779** while the model was destroyed. M2 replaced that metric with `min`. Gate 4's real content
is therefore a property of the **aggregation rule**:

> A mean can be dragged up by the terms that are fine. A `min` cannot.

That is testable from this run's own data, with no external constant, and it is not tautological.

### The criterion, restated

At `hi`, measure `s1`, `s2` and `s3` live, then require **all three**:

1. **`s4 = min(s1, s2, s3) < S4_MIN`.** Necessary, but guaranteed here — record it, do not treat it
   as evidence.
2. **The terms disagree** — at least one term below `S4_MIN` and at least one comfortably above.
   This is what makes `min` load-bearing. If all three terms are low the cell is uniformly
   destroyed and says nothing about the aggregation rule, so it cannot serve as the anchor.
3. **Report what a mean would have said.** `min < S4_MIN` while `mean(s1, s2, s3) ≥ S4_MIN` is the
   gate passing on its actual content: it reproduces the Velocity finding from this run's own data.
   If the mean *also* rejects the cell, the gate has not demonstrated that `min` was necessary —
   report that honestly rather than counting it as a pass.

**`hi` is the ideal location precisely because it is marginal.** It sits within roughly 3% of the
boundary, so `s3` has only just crossed below `S4_MIN` while `s1` and `s2` are most likely still
fine — which is the maximum-disagreement cell by construction, exactly what criterion 2 needs.

If all three terms are low at `hi`, step **down** toward the boundary rather than up. Damage
becomes more uniform as dose rises, so escalating moves away from the disagreement the gate needs.
**Never implement a search that escalates until the gate goes green.**

### When the winning layer has no boundary

Bisection reports `boundary = "not reached: sanity held at every reachable dose"` when sanity never
failed below `ALPHA_CEIL`. Then:

1. **Do not skip immediately.** Bisection runs on *every* shortlist candidate, and gate 4 tests the
   aggregation rule, which is not layer-specific. Prefer the winning layer; fall back to any other
   bisected candidate that did reach a boundary. Record which layer was used and why.
2. **Skip only if no candidate reached one — and skip, do not fail.** The repository's own
   vocabulary: *"NOT passed and NOT failed — it did not run."* A gate that could not run has not
   failed, and it goes in the not-run column of the gate table with what it would have validated.
3. **That case is itself a finding.** Sanity holding at every reachable dose across the entire
   shortlist means `S4_MIN` never bound in this run. Report it as a result, not as an absence.
4. **The M1.5 fallback only when the bundle genuinely exists**, as Sol proposed. Note that the
   stored-transcript path can recover only `s2` and re-judge `s1` — `s3` needs the live model — so
   it can never satisfy criterion 2 in full, and must say so.

## Acceptance

- A test that trips it: feed the gate a synthetic anchor row whose three terms are all low, and
  confirm it does **not** count as a pass — the terms must disagree for the anchor to be valid.
- A second test that trips it: an anchor where `min` and `mean` both reject the cell must be
  reported as not demonstrating that `min` was necessary.
- A test that the gate never escalates dose in search of a pass.
- The gate reports the anchor cell, its three sanity terms, and the boundary it was chosen relative
  to.
- The gate reports `min` and `mean` side by side, so the claim it is really making — that `min`
  rejects a cell a mean would have passed — is visible rather than asserted.
- What is lost is stated as a limitation: the *specific historical* comparison against Velocity
  L37 α=3 is gone. The property it demonstrated is now reproduced from this run's own data.
