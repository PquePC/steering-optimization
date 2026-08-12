# 01 — Add a third scan dose

**Status: DONE — `8d0d230`.** The third scan dose and 147-unit opening ETA landed together.

## Goal

`SCAN_DOSES` becomes `(0.15, 0.30, 0.60)`.

## Why

Phase 1 scans every layer at each dose in `SCAN_DOSES` and produces the `e6`/`d3` surface that
Phase 2 shortlists from. Two doses exist rather than one because a single dose cannot tell
*"this layer is inert"* from *"this layer is under-dosed"* — which is exactly the error a
fixed-α sweep makes.

On the first Garlic run the two current doses left the whole middle of the model undecidable:
**L20–L52 read `e6` reach 0.00 at both 0.15 and 0.30.** That band is where Macar et al. §5.1
predicts the qualifying region sits — detection peaks in mid layers while forced identification
rises toward late layers — so the pipeline is currently blind in the one place its own literature
says the answer should be.

0.60 rather than something larger: measured sanity headroom runs from `r` = 0.27 at L15 to
`r` = 1.42 at L58, so 0.60 should be inside the sane range at mid depth. **Verify that against the
Phase 3 bisection rather than assuming it** — if 0.60 turns out to be above the sanity boundary
across the mid band, say so, because that is itself a finding about where the model tolerates dose.

## What to change

Edit `CONSTANTS["SCAN_DOSES"]` in `m2/config.py` directly, and update the comment beside it with
the reasoning above.

**Do not use `--set SCAN_DOSES=...`.** A `--set` override gets its own run folder and cannot
resume, and this run will need to resume.

## Watch for

- `SCAN`'s cost prior is per `(layer, dose)` unit, so three doses means the unit count rises by
  half. Check `PHASE_UNITS_PRIOR["SCAN"]` still describes the shape it is counting — a prior whose
  unit disagrees with its counter is how `BISECT` was wrong by 12×.
- This changes `config_hash`. Any existing run folder for the old config is orphaned by design.
  Confirm the operator has nothing in flight before pushing.

## Acceptance

- `m2.config.CONFIG["SCAN_DOSES"] == (0.15, 0.30, 0.60)`.
- Offline tests pass, including whatever covers the scan grid size.
- The ETA at the start of a run reflects the larger scan, rather than reporting the old count.
- Cost impact stated to the operator in minutes per concept (expect roughly +10).

## Also record

Append to [`../TODO.md`](../TODO.md) under Suggestions the two rejected options, if they are not
already there: four doses `(0.15, 0.30, 0.60, 1.20)`, and per-layer doses scanned at a fraction of
each layer's own bisected boundary. The second is the scientifically strongest and inverts the
phase order, which makes it an M3 item rather than an M2 one.

---

## Measured outcome, 2026-08-12 CAL

The third dose landed, and it half-delivers. **17 of 147 cells are unreachable at
`ALPHA_CEIL = 16.0`, and all 17 are L14-L30 in the `r = 0.60` column** - nothing is unreachable at
0.15 or 0.30.

`alpha = r * ||h_L|| / ||v_L||`, and that ratio peaks in exactly that band (L28: `||v|| = 968`
against `||h|| = 49905`, so `r = 0.6` needs `alpha ~ 31`). So the undecidable-band problem this
task existed to solve is resolved for **L31-L52**, which is where Macar section 5.1 predicts the
qualifying window on a 62-layer model, and **not** for L14-L30.

Open item 6 in [`../TODO.md`](../TODO.md) carries the decision: accept as a scope limitation, or
address it. Raising `ALPHA_CEIL` is not a free option - it is the v1 damage anchor.
