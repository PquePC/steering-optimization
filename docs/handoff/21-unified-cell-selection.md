# 21 — Unified cell-level selection: eligibility, then the Pareto frontier

**Status: BUILD NOW — PROPOSE FIRST.** Branch `pareto` only. This supersedes tasks
[19](19-phase3-dose-selection.md) and [20](20-phase2-sanity-filter.md), which stay on `main` as the
fallback. Propose the frontier-size and fill rules before writing code.

## Why the routes are the problem, not the individual defects

Phase 2 runs three independent selections — local maxima of `e6`, a stratified sample of `e6`, and
the residual of a `d3 ~ e6` fit — then pads to `SHORTLIST_N` by `e6` coverage. Phase 3 then chooses
a **dose** by a fourth criterion, the sanity boundary. Tier ordering adds a fifth and sixth,
`e6_desc` and `residual_asc`.

**Each stage optimises one axis in isolation, and the unit of selection is never the unit of the
answer.** The answer is a cell `(layer, dose)`; Phase 2 selects layers, so the dose has to be
decided again later, by something that is not the objective.

The 2026-08-12 shakedown shows what that costs. Of the eight tier-0 layers, **four could not
qualify at any dose** — at every dose they either have no influence or the model is broken:

| Layer | Why it could never qualify |
|---|---|
| L37 | reach 0.00 at all doses; `s3` 0.48 at the one dose with low `d3` |
| L39 | same shape |
| L43 | reach 0.17 only where `s3` = 0.62 |
| L48 | reach 0.08 only where `s3` = 0.71 |

Half the shortlist was disqualified by data measured **before Phase 2 ran**, and the pipeline was
about to spend bisection and verification on it. Meanwhile `L57 @ r = 0.30` — `d3` 0.0018, the
lowest detectability of any eligible cell in the entire scan — was placed in the *audit* tier.

## The rule

**Step 1 — eligibility.** A cell is eligible if it is reachable, `s3 >= S4_MIN`, and
`reach >= E6_FLOOR`. This is the qualification rule expressed in cheap proxies: the model still
works, and something is happening.

**Step 2 — the Pareto frontier** on `reach` ascending and `d3` descending. Keep a cell if **no
other eligible cell has at least as much reach with no more `d3`.**

No fit, no `RESID_SIGMA`, no stratification, no padding, no separate dose decision.

## What it produces on the shakedown scan

130 reachable cells → 19 eligible → **4 on the frontier**:

| cell | reach | `d3` | `s3` | |
|---|---|---|---|---|
| L52 @ 0.60 | 1.00 | 0.98 | 0.95 | max influence, max detection |
| L51 @ 0.60 | 0.50 | 0.59 | 0.93 | middle of the trade-off |
| **L58 @ 0.30** | **0.42** | **0.03** | **0.98** | the target, at the right dose |
| **L57 @ 0.30** | 0.33 | **0.00** | 0.98 | missed entirely by the current design |

Those four are the influence-versus-detectability trade-off curve. Everything else is dominated.

## What it subsumes

- **The dose problem dissolves** — you select `(L58, 0.30)`, not "L58, then pick a dose". Task 19
  becomes unnecessary rather than implemented.
- **Broken cells are gone by construction** — that is task 20, as a precondition rather than a
  patch.
- **No OLS fit**, so no pollution from damaged cells, no `RESID_SIGMA` threshold, and no repeat of
  the `−0.101` dead-cell-floor residual that made seven of eight tier-0 layers indistinguishable.
- **No padding.** Six of eight tier-0 layers came from `widen`; the frontier has no such concept.
- **The audit tier gets a real definition**: the dominated cells *closest* to the frontier. One
  ranking generates the shortlist and the near-miss list, replacing the `e6_desc` / `residual_asc`
  interleave, which tied on constants.

## Propose before implementing

1. **Frontier size and fill.** Four here; it could be twenty elsewhere. `SHORTLIST_N` gives the
   target band — propose taking the frontier, then filling toward the lower bound with the
   nearest-dominated cells, and capping at the upper bound. Define "nearest" explicitly.
2. **The audit tier's near-miss metric.** Distance to the frontier needs a definition on two axes
   with different scales. Propose one and say why it is not arbitrary.
3. **What Phase 3 does now.** It stops choosing the answer. Two jobs remain and both are real:
   mapping the sane range around a selected cell (which gate 4's anchor needs), and refining dose
   between the three coarse scan doses. Propose its new contract.
4. **Whether `d3` or `d3_rate` is the detection axis.** They differ; the shakedown reports both.

## Risks to carry, stated not hidden

- **It leans on `d3`, which gate 5 has never validated.** So did the residual route — this is the
  same risk expressed more cleanly, not a new one. Gate 5 settles it, and until it has a number the
  frontier is a search device rather than a measurement.
- **`E6_FLOOR` is load-bearing.** Without it, dead cells are Pareto-optimal by construction: on
  this scan `L18 @ 0.15` (reach 0.00, `d3` 0.00) joins the frontier. That is commit `1dc85b1`'s
  dead-layer bug arriving through a new door. The floor stays, and cells just below it are
  **reported as near-misses** — `L56 @ 0.30` (reach 0.17, `d3` 0.00) is the case in point.
- **Three doses is coarse.** The frontier can only be as fine as the scan.

## Acceptance

- Reproduce the table above from the shakedown's `scan.jsonl` as a fixture test. That file is the
  regression test for this rule and it is already in hand.
- A test that trips it: a cell with reach 0.00 and `d3` 0.00 must not reach the frontier.
- A test that a layer with no eligible cell is reported as excluded, with its per-dose `s3` and
  reach — never silently dropped.
- Every selected cell records why: on the frontier, or filled at distance N.
- The shortlist table shows `reach`, `d3`, `s3` and the dose, all from the same cell. The current
  table mixes doses across columns and that must not survive.
