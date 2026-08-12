# 17 — Every reported rate carries a Wilson interval

**Status: BUILD NOW.** Blocking for the reporting path, not for the measurement path — but do it
before the run, because retro-fitting intervals onto stored rows is worse than emitting them.

## Goal

**Every rate this pipeline reports carries an interval, on every concept, everywhere** — and for
rates the interval is **Wilson**, not mean ± standard error.

## Why the current SE is not enough

The pipeline already takes defence 12 seriously: `_binom_se` exists, rows carry `_se` fields, and
the docstring is right that *"a reach of 0.42 over 12 prompts is a different claim from a reach of
0.42 over 120"*. The gap is what happens at the ends of the scale.

A binomial standard error is `sqrt(p(1−p)/n)`. **At p = 0 or p = 1 it is exactly zero.** So a cell
that reports `d2 = 0.000 ± 0.000` is claiming perfect certainty from 25 trials, and one reporting
`reach = 1.00 ± 0.00` is doing the same from 12 prompts. Neither is true, and both look like the
strongest possible result.

This is not a hypothetical failure here. The DEBUG-LOG records that **29 of 30 cells in the v1
sweep read exactly 0.00 or 1.00** — the pipeline's measurements land on the ends of the scale
constantly, which is precisely where the normal approximation breaks down.

**Wilson intervals do not have this failure.** At 0/25 the Wilson 95% interval is roughly
[0, 0.13] — which is the honest statement: *"we saw no identifications in 25 trials, and the true
rate could still be as high as about one in eight."* That is a materially different claim from
`0.000 ± 0.000`, and it is the one that survives review.

The project already has the precedent: the v1 rig check used Wilson intervals, and `controls.py`
keeps `_Z = 1.96` *"kept so the two are readable together"*.

## What to change

1. **One interval helper**, next to `_binom_se`, returning the Wilson interval for a count and an
   n. One implementation, used everywhere — two would eventually disagree, and a rate whose
   interval depends on which module computed it is worse than no interval.
2. **Apply it to every rate**, not a chosen few:
   - `d2` (and the `d2` null baseline from task [02](02-judge-null-controls.md))
   - `e6` reach
   - `d3` rate
   - `s3` accuracy and `cap_base`
   - `judge_fpr` and the `s1` null
   - the §9.1 / §9.2 control rates, which are already compared as a difference of two binomial
     rates and therefore need both sides' intervals
   - gate rates — anything the gate table prints as a fraction
3. **Keep the SE where the quantity is a mean rather than a rate.** `e5` is a mean of judge scores
   on a 0–10 scale, not a proportion; mean ± SE is correct there and Wilson is not applicable. Do
   not blanket-replace — the two live side by side and the distinction is which quantity it is.
4. **Report both endpoints, never a ± on an asymmetric interval.** Wilson intervals are asymmetric
   near the ends, which is the entire point; collapsing one to `p ± half-width` throws away the
   asymmetry that made it worth computing.
5. **Print them.** A stored interval nobody sees does not change how a number is read. The per-cell
   lines, the operating point, the frontier and the gate table all carry them.

## Watch for

- **n must be the real n.** The interval is only as honest as the count behind it. Where a rate is
  computed over rows that were filtered — judge errors dropped, unreachable cells skipped — the n
  is the surviving count and the row must say so. A rate over 22 of 25 trials reported as though it
  were 25 is a quieter version of the same defect.
- **Do not use an interval to make a gate pass or fail more leniently** without saying so. Gates
  currently compare point estimates against thresholds; changing them to compare interval endpoints
  is a real change to what the gate means, and it is a design decision, not a formatting one.
  Propose it separately if you think it is right — do not fold it into this task.

## Acceptance

- A test that trips it: `0/25` must not produce a zero-width interval, and the upper bound must be
  materially above zero.
- A test that `1/1` and `25/25` both produce sane, non-degenerate intervals.
- Every rate in `operating_point.json`, the per-cell rows and the gate table carries two endpoints
  and its n.
- `e5` still reports mean ± SE, and a test asserts the two treatments have not been confused.
