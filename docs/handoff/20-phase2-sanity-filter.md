# 20 - Phase 2 must not read cells where the model is broken

**Status: SUPERSEDED - DO NOT BUILD.** Replaced by task 21 on this branch. Kept because it is the
fallback if the unified selection does not validate.

## The problem, from the shakedown

**37 of 130 reachable cells (28%) sit below `S4_MIN` on `s3`** - 3 at r = 0.15, 20 at 0.30, 14 at
0.60. Phase 2 never looks at `s3`. Routes 1, 2 and 3 and the OLS `d3 ~ reach` fit all consume those
cells unfiltered; `_cheap_sane` exists but is used only in Phase 3.

**When the model is broken, `d3` falls** - not because the concept went covert, but because the
model cannot produce anything coherent, including the concept's own name. Low `d3` is exactly what
the pipeline hunts for, so **a broken cell is indistinguishable from the target.**

L37, Macar's reference layer:

| dose | `s3` | `d3` |
|---|---|---|
| 0.30 | 0.90 (healthy) | **1.00** - fully detectable |
| 0.60 | **0.48 (broken)** | **0.01** - looks covert |

**The shortlist table displayed L37 with `d3 = 0.010`.** The selection surface showed the exact
false positive this project exists to avoid.

Two mechanisms produced it. The `ols_fit` runs over every reachable cell, so a quarter of the
surface feeding every residual was measured on a damaged model. And `_by_layer` picks each layer's
"best" dose as `max(rows, key=(reach, r))`, breaking ties on reach by **highest r** - so any layer
reading reach 0.00 at every dose reports its numbers from the most damaged dose available.

The principle is already in the codebase and was never applied here. **R8: *"where sanity is
broken, D2 is not readable - and must not be read."*** `d3` is `d2` measured cheaply.

## The fix, as decided before task 21

**Exclude and display.** Drop cells with `s3 < S4_MIN` from the fit, from `_by_layer`'s best-dose
selection and from every route's input; add `s3` and the source dose to the shortlist table; and
report any layer with no sane cell as *excluded for insanity* rather than dropping it silently.

## Why it was superseded

Task [21](21-unified-cell-selection.md) makes sanity an **eligibility precondition** on every cell
before any ranking happens, which is the same rule applied earlier and in one place instead of
three. The display requirement survives into task 21 unchanged - it is good regardless of how
selection works.
