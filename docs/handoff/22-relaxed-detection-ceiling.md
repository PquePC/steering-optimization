# 22 — Relaxed detection ceiling, so the pipeline can reach CONFIRM and CONTROLS

**Status: DONE — `dc873b8`.** Branch `pareto` only. Interim measurement setting, not a science
change.

## Why

CONFIRM and CONTROLS **have never executed a line.** They only run when a cell qualifies, and no
run has produced one — the first died in Phase 4 on the S1 blindness bug, the second on the float
key. Until a qualifying cell exists, a third of the pipeline stays untested: the operating-point
machinery, the §9.1 and §9.2 controls, and the whole reporting path.

The operator's call: **relax the detection ceiling so a cell qualifies, exercise the rest of the
pipeline end to end, then tighten.**

## What changes, and what does not

| Constant | Now | Interim | |
|---|---|---|---|
| `D2_MAX` | 0.20 | **0.50** | the only relaxation |
| `S4_MIN` | 0.70 | **0.70** | unchanged — sanity stays high |
| `E5_FLOOR` | 4.0 | **4.0** | unchanged — effectiveness is the objective, not a threshold to loosen |

0.50 is not arbitrary. Garlic is published at **100%** forced identification at Macar's reference
configuration, so a ceiling of 0.50 still means *half the published detectability* — a real
reduction and a defensible interim target, while being loose enough that a wrong proxy does not
cost the run.

## The trap, and how the code must prevent it

**Relaxing a threshold to get a result and then reporting that result is the researcher degree of
freedom this repository is built to exclude.** Task [06](06-relaxed-reselection.md) exists for
exactly this, and its machinery is the answer:

1. **The run's operating point is confirmed at `D2_MAX = 0.50` and must be labelled as such** — in
   `operating_point.json`, in the run record, and in anything printed. Not in a report someone
   writes later.
2. **Re-derive at 0.20 from the stored rows afterwards.** Both numbers come from one run: task 06's
   re-selection reads `verified.jsonl` and recomputes `qualifies` at any threshold without
   re-measuring anything.
3. **State plainly which is confirmed and which is screening.** CONFIRM measures the winner *at the
   threshold in force*. If a different cell wins at 0.20, that cell has a screening number and not
   a confirmed one, and the write-up must say so.

## Worth knowing before assuming it is needed

On the shakedown scan the two most promising cells read `d3` **0.03** (L58 @ 0.30) and **0.00**
(L57 @ 0.30). If `d3` tracks `d2` at all, both qualify comfortably at the **unrelaxed** 0.20 — and
the reason that run found nothing was the dose selection, not the threshold.

So this is **insurance against the proxy being wrong**, not a necessity. Report at both thresholds
and let the data say whether the relaxation was needed at all.

## Acceptance

- `D2_MAX = 0.50` in `CONFIG` with the rationale and the word *interim* beside it, and a pointer to
  this document.
- `operating_point.json` carries the threshold in force and a flag marking it a relaxed run.
- A test that the relaxed value cannot be read as the primary analysis — the label must be in the
  record, not implied by a filename.
- The run record reports the winner at 0.50 (confirmed) and at 0.20 (screening) side by side.
