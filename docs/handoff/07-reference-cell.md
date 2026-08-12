# 07 — The reference cell (DEFERRED)

**Status:** DEFERRED by the operator, 2026-08-12. Do not build it for this run.

## What was proposed

Measuring Macar's reference configuration — L37, α = 4 — as an ordinary verified cell in every run,
so that a comparison against their published forced-identification rate would be made against our
own instrument rather than across two.

## Why it is deferred

Direct comparison against Macar's published numbers is not the framing for this run. The result
stands on its own terms: **this pipeline finds a cell where the concept influences output, the
model stays intact, and `d2` is low** — and `d2`'s meaning is protected by gate 11's agreement
check against the upstream judge rather than by reproducing a published aggregate.

## Preserved here because the reasoning still holds

Should the comparison come back, two things have to be true and only one of them currently is.

**The α-versus-`r` conversion is layer-dependent, and both coordinates must be reported.** Macar
states α = 4 at L37; this pipeline works in `r = α·‖v_L‖ / ‖h_L‖`, because at fixed α the real dose
varies more than 20× across layers and non-monotonically — the confound `r` exists to remove. There
is no single conversion factor. `config.dose_for(37, 4.0)` and `config.alpha_for(L, r)` already
exist; any published comparison needs **layer, α and `r` for both cells**, plus `‖v_37‖` and
`‖h_37‖` so a reader can reproduce the conversion.

**The instruments are not identical, and that is deliberate.** See
[`../TODO.md`](../TODO.md) — the forced-ID prompts and the extraction follow Macar, but the
injection hook and the identification judge are both M2's own, for reasons that are recorded and
sound. Comparing rates across that boundary needs gate 11's evidence, not an assumption.

## Related and not deferred

The final chosen cell already gets `N_CONFIRM` forced-ID trials, not `N_D2` — `phase6_confirm`
passes `n_d2=N_CONFIRM` into `verify_cell`, so the reportable `d2` is measured over 100 trials on
held-out prompts with no adaptive stopping. That was the substance of the "more trials on the cell
carrying the headline" concern, and it is already the design.
