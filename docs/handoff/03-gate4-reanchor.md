# 03 — Gate 4 on this run's own damaged cell

**Status:** DECIDED. Implement as written.
**Blocking:** yes.

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

Take a cell above the bisected boundary at the winning layer, measure `s1`, `s2` **and `s3`** live
at it, and require `s4 < S4_MIN`. Self-contained, portable to any model and concept, and it
measures the term the transcript path never could.

## What to change

- Add the live-anchor path to `gate4_sanity_acceptance` in `m2/gates.py`, selecting the cell from
  the run's own bisection output.
- **Keep the stored-M1.5 path as a dormant fallback** rather than deleting it. If a v1 bundle ever
  resurfaces the historical comparison becomes available again.
- Record in the gate's output *which* cell was used and *why it is expected to be damaged* — the
  bisected boundary and how far above it the anchor sits. A gate that does not say what it tested
  is not auditable.

## Design points to decide (escalate)

- **How far above the boundary.** Far enough that damage is unambiguous, close enough that the cell
  is still on the same part of the dose axis. Propose a rule (a fixed multiple of the boundary? one
  bisection step above?) with reasoning.
- **What if `s4` at that cell is *not* below the floor.** That is a real finding — it says either
  the boundary is not where bisection thinks it is, or `S4_MIN` is too permissive. It must fail the
  gate loudly, not be retried at a higher dose until it passes. **Do not implement a search that
  keeps escalating until the gate goes green**; that would make the gate unable to fail, which is
  the defect class this repository exists to avoid.

## Acceptance

- A test that trips it: feed the gate a synthetic anchor row with healthy sanity and confirm it
  **fails**.
- The gate reports the anchor cell, its three sanity terms, and the boundary it was chosen relative
  to.
- What is lost is stated in the report as a limitation: the historical claim *"the rebuilt sanity
  rejects the cell the old metric wrongly passed"* is no longer testable. That is a claim about the
  metric change, not about this run's validity.
