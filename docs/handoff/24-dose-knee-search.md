# 24 — Bisect the dose gap during SCAN, before selection

**Status: BUILD NOW.** Branch `pareto`. Runs as a SCAN extension, **before Phase 2**, not as a
refinement of chosen cells. Measured cost on the shakedown shape: **~7.6 minutes per concept.**

## Why it must run before selection, not after

The scan samples three doses. On the shakedown, **the entire transition from "nothing happening" to
"fully detectable" falls in the unsampled gap between `r = 0.30` and `r = 0.60`:**

| | reach@0.30 | reach@0.60 | `d3`@0.30 | `d3`@0.60 | `s3`@0.30 |
|---|---|---|---|---|---|
| L45 | 0.00 | 0.00 | 0.00 | 0.91 | 0.98 |
| L47 | 0.00 | 0.25 | 0.00 | 0.45 | 0.98 |
| L50 | 0.00 | 0.42 | 0.00 | 0.50 | 0.98 |
| L51 | 0.00 | 0.50 | 0.00 | 0.59 | 0.98 |
| L52 | 0.00 | 1.00 | 0.00 | 0.98 | 1.00 |
| L57 | 0.33 | 1.00 | 0.00 | 1.00 | 0.98 |
| L58 | 0.42 | 1.00 | 0.03 | 1.00 | 0.98 |

L45 through L52 are **healthy at 0.30 and show nothing**, then at 0.60 they are alive and
detectable. If a dose exists where reach is up and `d3` is still low, it is in that gap.

**And they have reach 0.00 at 0.30, so they are not eligible, not selected, and not candidates.**
A refinement that only probes chosen cells can never reach them. That is the chicken and egg:
the grid decides what gets selected, so a denser grid has to come first or the candidates it would
have created never exist.

## The search

For each layer in the band, bisect the gap looking for the **knee** — the dose where influence has
arrived and detection has not. The midpoint's reading gives the direction:

| Midpoint reads | Meaning | Go |
|---|---|---|
| `s3 < S4_MIN` | the model is already broken here | **down** |
| `reach < E6_FLOOR` | still nothing happening | **up** |
| `reach >= E6_FLOOR` and `d3` low | **found a new eligible cell** — record it, then probe up to see whether reach rises further while `d3` stays low | up, once |
| `reach >= E6_FLOOR` and `d3` high | past the knee, saturated | **down** |

Three outcomes drive direction rather than two, because sanity and reach fail in opposite
directions and collapsing them loses which one you hit.

## Which layers, and how deep

**The band:** layers **sane at the lower dose** where the gap shows a large change — `|Δreach| >= 0.20`
or `|Δd3| >= 0.30`. On the shakedown that is 21 of 49 layers. Propose the exact thresholds; these
are the values that reproduce the table above and they should be config, not literals.

**Depth: two levels, and no more.** `reach` is measured over 12 prompts, so its resolution is
**1/12 = 0.083**. Two bisections of a 0.30-wide gap reach an interval of 0.075 — already at the
measurement's resolution. **A third level resolves the dose finer than the measure can distinguish**
and buys nothing but time.

## Cost, measured

At the shakedown's 10.9 s per cheap cell, 21 band layers:

| | cells | time |
|---|---|---|
| 1 level | 21 | 3.8 min |
| **2 levels** | **42** | **7.6 min** |
| 3 levels (over-resolved) | 63 | 11.4 min |
| *(a flat extra dose across all 49 layers, for comparison)* | 49 | 8.9 min |

**Two levels of targeted bisection costs less than one flat extra dose and resolves the interval
eight times finer.** No generations, no judge calls — cheap tier only.

## What it feeds

New cells go into the scan surface and are eligible for Phase 2's selection exactly like scanned
cells, carrying provenance that says they came from the knee search and at what depth. **They must
not be a separate list** — the whole point of task 21 is one selection over one surface.

## Watch for

- **Do not let the probe widen the sanity envelope.** A midpoint that reads insane is recorded and
  bisected downward; it is never promoted to a candidate. Bisection here searches for the knee, not
  for the boundary — the boundary is still Phase 3's job and gate 4's anchor.
- **The band criterion must be computed on sane cells only.** A layer whose "large change" is
  really the model collapsing between doses is not a knee, it is damage.
- **Report the band.** Which layers were probed, which were not, and why — same rule as everywhere
  else: a search that narrows silently reads as a search that found nothing.

## Acceptance

- A test that trips it: a synthetic layer whose midpoint is insane must bisect **down**, never up.
- A test that a layer with reach 0.00 at both ends and no sanity change is **not** in the band.
- New cells appear in `scan.jsonl` with knee-search provenance and are visible to Phase 2.
- The band, the depth and the per-level cost are printed before the probe runs.
- Depth is capped at two with the `1/12` resolution argument in the comment beside the cap.
