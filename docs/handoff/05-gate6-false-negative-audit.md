# 05 — A tiered shortlist: escalation and false-negative audit

**Status: DONE — `63cdc35`.** The tiered shortlist, mandatory live-signal audit, exhaustive mode,
Gate 6 replacement, provenance record and opening ETA units landed together. Offline suite:
96 passed, 2 environment-dependent skips.

## What Phase 2 does, and its one dangerous failure

Phase 2 looks at the whole scan surface — every layer in scope, cheap measures on each — and picks
a shortlist of roughly eight to twelve layers. **Only shortlisted layers are ever measured
expensively.** Everything it rejects is never seen again.

That makes its two error types wildly asymmetric:

- a **false positive** — shortlisting a useless layer — costs one verification cell;
- a **false negative** — dropping the layer that held the answer — **loses the result silently.**
  The run reports "no operating point exists at these constraints" with complete confidence, and
  nothing downstream can tell the difference between that and the truth.

This is not hypothetical. In v1, four of the seven qualifying cells sat at **L31** while
effectiveness peaked at **L46**. A shortlist that hill-climbs on effectiveness walks away from all
four. That is why Phase 2 is deliberately not a top-K and why it widens on the residual — and gate
6 was supposed to be the check that the widening works.

**The old gate 6 cannot do that job.** It tests recall against seven hard-coded v1 cells on
karma / silk / irony, so it skips for every other concept. No file fixes it: you cannot test whether
a shortlist dropped the answer for a concept whose answer nobody knows — which is the situation for
Garlic, for Origami, and for every model this tool will ever be pointed at.

## The design

Phase 2 stops emitting a flat shortlist and emits an **ordered sequence of tiers**.

| Tier | Contents |
|---|---|
| **0** | the current shortlist — local maxima, stratified depth coverage, residual widening |
| **1, 2, 3 …** | the layers Phase 2 rejected, ordered by `e6` descending, in blocks of `TIER_SIZE` |

Phase 4 then works through them:

1. **Verify tier 0.**
2. **Always verify tier 1 as well**, whether or not tier 0 found a window.
3. **If no qualifying cell has been found, escalate** to tier 2, then 3, up to the configured limit
   or until the in-scope layers are exhausted.
4. **A qualifying cell from any tier competes for selection under the normal rule** — `argmax(e5)`
   over qualifying cells. It is not a second-class result.

## Why tier 1 always runs, even on success

This is the point most likely to get lost in implementation, so it is worth being explicit: tier 1
is doing **two different jobs** that happen to need the same machinery.

- As **escalation**, it only needs to run when tier 0 found nothing.
- As the **false-negative audit**, it must run *especially* when tier 0 succeeded — because the
  question it answers is "did the shortlist drop something better or something equally valid?", and
  that question is only interesting when the shortlist produced an answer you were about to
  believe.

If tier 1 only ran on failure, then every successful run would ship with **zero evidence** that the
shortlist did not drop a better cell. So tier 1 is always paid: `TIER_SIZE` extra VERIFY cells per
concept, three by default.

And it pays for itself in more than assurance. If a tier-1 cell qualifies with **higher `e5`** than
tier 0's winner, that is not merely a gate failure — **it is a better answer, and the pipeline
should take it.** The audit can improve the result, not just grade it.

## The gate

Gate 6 becomes: **no cell outside tier 0 qualifies at an `e5` at or above tier 0's winner.**

- If a tier-1+ cell qualifies but scores lower, that is a shortlist imprecision worth reporting and
  not a failure.
- If one qualifies *above* the winner, tier 0 demonstrably dropped the answer: the gate fails,
  `E6_FLOOR` or the residual route needs widening, and the winner is taken from the outer tier with
  that noted.
- Report the audit's **power** honestly — `k` cells sampled out of how many rejected. A sample of
  three cannot prove there is no false negative anywhere; it can only fail to find one. That
  sentence belongs in the results, not just in the code.

## Ordering, and why `e6` descending

Ordering by `e6` samples the **near misses** — the rejections most likely to have been wrong, since
`E6_FLOOR` is the main rejection criterion. That makes the audit adversarial.

The alternative, uniform sampling across rejected layers, mostly draws obviously-dead layers and
would pass trivially. **That would be another check that cannot fail**, which is the defect family
this repository exists to avoid.

**Refinement to propose:** a layer rejected *despite* a good residual is also a strong near-miss
candidate, and the residual route is the one Phase 2 added specifically to catch cells that
under-detect for their influence. Consider interleaving the two orderings — top `e6` and top
residual among the rejected — rather than `e6` alone. Make the argument and let the operator choose.

## Everything is parametrized

The defaults below are for this project. Someone running this to find the genuinely best steering
point for their own work must be able to run as deep as they are willing to pay for, up to every
cell in scope. **No number here may be hard-coded**, and each goes in `CONFIG` with its rationale
beside it as usual.

| Knob | Default | What it controls |
|---|---|---|
| tier size | **3** | cells per tier beyond tier 0 |
| audit tiers | **1** | tiers always verified, regardless of whether a window was found |
| escalation limit | **3** | further tiers tried when no window has been found. `None` = until in-scope layers are exhausted |
| tier ordering | **`e6` descending** | how rejected layers are ranked into tiers |
| exhaustive mode | **off** | verify every in-scope cell, ignoring tiers entirely |

Exhaustive mode is the one that makes this tool useful to someone else: *"run everything, ranked
best-candidate-first, and stop when I say"*. It should be a single flag on `m2.run`, and it should
print its own cost estimate before starting, because it is the expensive path by construction.

## Acceptance

- A test that trips the gate: inject a synthetic tier-1 verification row that qualifies above tier
  0's winner, and confirm gate 6 **fails** and the winner changes.
- A test that tier 1 runs when tier 0 has already found a qualifying cell. This is the behaviour
  most likely to be optimised away by someone reading the code later.
- Escalation stops at the configured limit and says so, rather than silently exhausting.
- Every knob above is read from `CONFIG`, and `--set` reaches all of them.
- Exhaustive mode prints an estimated cell count and cost before it starts.
- The run record carries, per tier: which layers, why each was ordered where, and their verdicts.

---

# Decided 2026-08-12

## Config names — approved as proposed

```
SHORTLIST_TIER_SIZE   = 3
SHORTLIST_AUDIT_TIERS = 1
SHORTLIST_MAX_TIER    = 3          # permits tiers 0-3; None = until in-scope layers are exhausted
SHORTLIST_TIER_ORDER  = "e6_residual_interleave"
SHORTLIST_EXHAUSTIVE  = False
```

Two notes on them:

- **`SHORTLIST_MAX_TIER = None` and `SHORTLIST_EXHAUSTIVE = True` are not the same thing**, and they
  will be confused unless each says so beside itself. `MAX_TIER = None` escalates through every tier
  **only while no qualifying cell exists** and stops the moment one is found. `EXHAUSTIVE = True`
  verifies every in-scope cell **regardless**. Write the distinction into both comments; two knobs
  that look like they mean the same thing is how one gets set meaning the other.
- **`SHORTLIST_TIER_ORDER` must raise on an unrecognised value**, never fall back to a default. A
  mistyped ordering that silently becomes `e6_desc` is a run measuring something other than what it
  reports.

## Tier ordering — `e6_residual_interleave`, approved

Sol's argument is right and supersedes the earlier `e6_desc`. Phase 2 can wrongly reject a layer in
two independent ways — missing the `E6_FLOOR` cutoff, and mishandling the residual route — and an
audit ordered on `e6` alone leaves the second essentially untested.

The residual route is not the minor one. **It is the route that found the cell this pipeline exists
to find**: on the first Garlic run, L58 came from the residual — `e6` reach 0.42, `d3` 0.027,
`resid` −0.375, influence without proportional forced-ID detectability. Auditing only the `e6`
boundary would leave the route that produced the headline unaudited.

### The guard the residual ordering must inherit

**A dead layer must never enter a tier, by either ordering.** Commit `1dc85b1` — *"Stop the
shortlist widening onto layers with no concept mass"* — fixed exactly this: the old guard tested
`d3 > 0.0`, and `d3` is a probability mass that is never exactly zero, so Garlic put L13/L14/L15 on
the shortlist at reach 0.00. The bar is now `D3_SIGNAL_MIN`, a floor under the unsteered `d3`
baseline.

**Residual ordering can walk around that guard through a new door.** On a dead layer both `d3` and
`e6` are ≈ 0, so `resid = d3 − predicted_d3(e6)` is whatever the fit noise makes it — and a
slightly negative value would rank a dead layer at the top of a "most negative residual" ordering.
That is `1dc85b1`'s bug reintroduced, in the one place built to catch false negatives.

Both orderings draw from the **live** rejected population only, filtered by `D3_SIGNAL_MIN` exactly
as the shortlist is. **Test it against a synthetic dead layer with a mildly negative residual and
confirm it is not selected.**

### Composition at `SHORTLIST_TIER_SIZE = 3`

Alternate starting with `e6`, deduplicating — `e6` #1, most-negative residual #1, `e6` #2 — so tier
1 carries two `e6` near-misses and one residual near-miss. Record **which ordering selected each
layer** in the row, so a tier-1 qualifier can be traced to the route that would have missed it.

## Gate 6 in exhaustive mode — NOT APPLICABLE, and distinguish it from a skip

Sol's reasoning is correct: reporting PASS when every in-scope layer was verified would be a gate
that cannot fail, because there is no rejected population for it to audit.

One refinement. **"Not applicable" and "could not run" are opposite in meaning and must not share a
column.** A reader scanning the not-run gates is looking for gaps in the evidence. This is not a
gap — it is the strongest possible outcome: nothing was rejected, so nothing could have been
wrongly rejected. Exhaustive mode is *stronger* than a gate-6 pass, not weaker.

Report it as `NOT APPLICABLE — every in-scope layer was verified; there is no rejected population
to audit`, and give the gate table a state distinct from both `PASS` and `SKIPPED`.

## Cost, and the unit counts that go with it

Tier 1 always running means `SHORTLIST_TIER_SIZE` extra layers through **both** BISECT and VERIFY —
at the measured 99 s per bisection candidate and 50 s per verified cell, roughly **7.5 minutes per
concept** at the default of 3.

**Update `PHASE_UNITS_PRIOR` for BISECT and VERIFY accordingly**, and state the unit beside each
number. `BISECT` was already wrong by 12× once because its prior was priced per bisection *probe*
while the board ticks once per *candidate*; a change that raises the candidate count is exactly
where that class of error returns. The ETA must account for the audit tier from the opening board,
not discover it mid-run.
