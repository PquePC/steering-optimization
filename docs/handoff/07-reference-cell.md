# 07 — Measure the reference cell in every run

**Status:** DECIDED. Implement as written.
**Blocking:** yes — it changes what Phase 4 measures.

## Goal

Every run measures **Macar's reference configuration — L37, α = 4 — as an ordinary verified cell**,
so the comparison against a published rate is made against our own instrument rather than across
two.

## Why this is the whole headline

The claim this project is aiming at is:

> Garlic is published at **100%** forced identification at Macar's reference configuration on this
> model. This pipeline finds a cell that keeps the concept influencing output and the model intact
> while driving forced identification to **X%**.

That sentence has a hole in it. The 100% came from Macar's rig — their extraction, their prompt,
their judge, their trial count. Our `d2` comes from ours. Comparing them directly asks the reader
to assume the two instruments agree, and **the pipeline currently has no measurement that supports
that assumption.**

Measuring the reference cell closes the hole for the cost of one verified cell. Then:

- **Our `d2` at L37 / α=4** is the baseline the operating point is compared against. Same
  extraction, same prompt, same judge, same model, same trial count. The comparison is internal and
  needs no assumption at all.
- **Macar's published 100%** becomes corroboration of that reference measurement — if we read
  near-saturated `d2` there too, the instrument reproduces a known answer, and if we do not, that
  discrepancy is itself important and must be reported rather than buried.

This also does most of what the retired R4 rig check was for (see [`../TODO.md`](../TODO.md), open
item 5), at one cell instead of six hundred generations.

## The α-versus-`r` problem, which this also solves

Macar's configuration is stated in **α**. This pipeline works in **`r` = α·‖v_L‖ / ‖h_L‖**, because
at fixed α the real dose varies more than 20× across layers and non-monotonically — that is the
confound `r` exists to remove.

So "how does my operating point compare to α = 4" is not directly answerable, and the answer is not
a single conversion factor: it depends on the layer. The run must report **both coordinates for
both cells**:

| | layer | α | `r` |
|---|---|---|---|
| reference cell | 37 | 4.0 | *computed* |
| operating point | *found* | *computed* | *found* |

`config.dose_for(37, 4.0)` gives the reference cell's `r`; `config.alpha_for(L, r)` gives the
operating point's α. Both are already implemented. Report all four numbers, and report `‖v_37‖` and
`‖h_37‖` alongside, because those are what make the conversion reproducible by a reader.

## What to change

- Add the reference cell to Phase 4's cell list unconditionally, flagged as `reference` in the row
  so it is distinguishable from shortlisted cells.
- **It is measured but must not compete for selection.** It is a baseline, not a candidate. If it
  happens to qualify that is worth reporting loudly — it would mean Macar's default is itself an
  operating point for this concept — but the selection rule must treat it explicitly rather than by
  accident.
- Handle the case where L37 is outside the layers in scope, or α = 4 is unreachable at L37 under
  `ALPHA_CEIL`: report that plainly rather than substituting a nearby cell.
- Report the four-number table above in `operating_point.json`.

## Escalate

- **Whether the reference cell should be `N_CONFIRM` trials rather than `N_D2`.** The comparison
  carries the headline, and 25 trials gives a wide interval on a rate near 1.0. Measuring it at
  higher n costs one more cell's worth of judge calls and materially tightens the claim. Propose it
  with the interval arithmetic.

## Acceptance

- Every run's `operating_point.json` carries the reference cell's `d2` with its interval, in both
  α and `r` coordinates, alongside the operating point's.
- A run where L37 is out of scope says so explicitly and does not silently omit the comparison.
- The reference cell cannot be selected as the operating point by accident.
