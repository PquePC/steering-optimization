# 05 — Replace gate 6 with a self-contained false-negative audit

**Status:** PROPOSED. **Do not implement until the operator signs off** — this changes what gate 6
means, not just how it is computed.
**Blocking:** the sign-off is; the code is not.

## What gate 6 is for

Phase 2 takes the full scan surface — every layer, at every scan dose, with `e6` and `d3` on each —
and picks a shortlist of maybe eight to twelve layers. **Only shortlisted layers are ever measured
expensively.** Everything Phase 2 rejects is never seen again by the pipeline; if the answer was in
a rejected layer, the run cannot find it and will report "no operating point exists" with complete
confidence.

So Phase 2 has one failure mode that matters far more than the other:

- a **false positive** — shortlisting a layer that turns out to be useless — costs one verification
  cell, maybe a minute of GPU and a few judge calls;
- a **false negative** — dropping the layer that held the answer — loses the result silently, and
  nothing downstream can detect it.

Gate 6 exists to check for the second. It is **recall, not ranking**: the question is not "did the
shortlist put the best layer first" but "did the shortlist contain the answer at all".

This is not hypothetical. In v1, four of the seven qualifying cells sat at **L31** while
effectiveness peaked at **L46**. A shortlist that hill-climbs on effectiveness walks away from all
four. That is precisely why Phase 2 is deliberately not a top-K, and why it widens on the residual —
and gate 6 is the only thing that checks the widening actually works.

## Why the current version cannot be repaired

It tests recall against `M15_QUALIFYING_CELLS`, a hard-coded list of seven cells from v1 — four on
karma, one on silk, two on irony. For any other concept the gate skips with *"this concept has no
qualifying M1.5 cell"*.

No missing file causes this and no file fixes it. **You cannot test whether a shortlist dropped the
answer for a concept whose answer nobody knows** — and that is the situation for Garlic, for
Origami, and for every model and concept this tool is meant to serve in future. The check is
unportable in principle, not by accident.

## The proposal

Test the property directly, using the run itself.

**After Phase 2, take `k` of the layers the shortlist rejected — stratified across depth so they
are not all from one region — carry them into Phase 4 verification alongside the shortlist, and
require that none of them qualifies.**

If none qualifies, the shortlist's rejections were sound as far as the sample can tell. If one
*does* qualify, the shortlist has a demonstrated false negative: `E6_FLOOR` is too high, or the
residual route is not widening far enough, and that must be fixed before the run's "no operating
point" or "this is the operating point" means anything.

What this buys:

- **Portable.** Works on any model, any concept, with no external constants.
- **Direct.** It tests the actual property — "the shortlist does not drop qualifying cells" —
  rather than a proxy for it.
- **Honest about its own power.** A sample of `k` cannot prove there is no false negative anywhere;
  it can only fail to find one. That is a real limitation and must be stated in the report, with
  `k` and the sampling rule, rather than presented as a clean pass.

What it costs: `k` extra VERIFY cells per concept. At `k = 3` that is three more cells' worth of
generation and judge calls.

## The two numbers to agree

1. **`k`.** Suggested start: **3**. Higher `k` gives more power to detect a false negative and
   costs proportionally.
2. **The stratification.** Suggested: one rejected layer from each of the shallow, middle and deep
   thirds of the layers in scope, chosen from the rejected set by highest `e6` — i.e. deliberately
   sampling the **near misses**, the rejections most likely to have been wrong, rather than
   sampling uniformly. A uniform sample mostly draws dead layers and would pass trivially.

That second point matters and is worth the operator's attention: **sampling the near misses makes
the audit adversarial rather than reassuring.** A gate that mostly draws obviously-dead layers is
another check that cannot fail.

## If the operator prefers to keep the old check as well

The legacy version can be run once as a **regression test of the selection algorithm** rather than
as an acceptance gate on a result. Karma carries four of the seven cells across three depths and
needs only Phases 0–2 — roughly thirty minutes. That tests the code, not the run, and should be
labelled as such. **Currently out of scope; do not run Karma unless asked.**

## Acceptance, once approved

- A test that trips it: inject a synthetic rejected-layer verification row that qualifies, and
  confirm the gate **fails**.
- The gate reports which layers were sampled, why each was chosen, their `e5`/`d2`/`s4`, and `k`.
- The report states the audit's power honestly — `k` sampled out of how many rejected.
