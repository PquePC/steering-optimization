# 19 - Phase 4 must verify two doses per layer, not the sanity boundary alone

**Status: SUPERSEDED on branch `pareto` by task 21.** Kept on `main` as the fallback fix if the
unified selection does not work out. Decided 2026-08-12, then dissolved by task 21 rather than
implemented - see the note at the end.

## The problem, from the shakedown

Phase 3 picks **the highest dose at which the model is not broken** and hands that single dose to
Phase 4. That maximises influence - and it maximises detection with it. The objective is
`argmax(e5)` **subject to `d2 <= D2_MAX`**, so the highest sane dose is simultaneously the best
answer for one requirement and the worst for the other.

L58 in the 2026-08-12 run:

| dose | reach | `d3` | `s3` | |
|---|---|---|---|---|
| **0.30** | **0.42** | **0.03** | **0.98** | the target signature |
| 0.60 | 1.00 | 1.00 | 0.93 | saturated |
| **1.35** | - | - | sane | **what Phase 3 chose** |

Phase 3 selected 1.35 because sanity never failed below it. Phase 4 would have measured there,
found `d2` near 1.0, failed the constraint, and reported *"no operating point exists for Garlic"*.
Phase 5 refines one step either side of 1.35 and never approaches 0.30.

**The scan found the answer and the next phase discarded the dose that made it one.**

## The fix, as decided before task 21

Phase 4 verifies **two cells per shortlisted layer**: the dose at which Phase 2's route selected the
layer, and the sanity boundary Phase 3 already produces. `select_operating_point` then applies spec
7.1 unchanged over both.

Fallback for layers with no entering dose (six of eight tier-0 layers came from `widen` padding):
the **lowest sane dose clearing `E6_FLOOR`** - minimum detectability with influence still present.

Cost: roughly double the VERIFY cells, about +9 minutes per concept.

## Why it was superseded

The two-dose fix patches a symptom. Phase 2 selects **layers** and Phase 3 then picks a **dose**,
so the unit of selection is not the unit of the answer - and the dose decision is made by a
criterion (the sanity boundary) that is not the objective. Task
[21](21-unified-cell-selection.md) selects `(layer, dose)` **cells** directly against every
criterion at once, and the dose question disappears rather than being answered twice.

If the unified selection fails validation, this document is the fallback and needs no rework.
