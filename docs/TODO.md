# TODO — open decisions and findings

**Authority on: what is still undecided or unbuilt.** Fixed defects and the reasoning behind them
live in [`DEBUG-LOG.md`](DEBUG-LOG.md); this file carries only what is still open.

Started 2026-08-11, after the first end-to-end Garlic run on a fresh A100-80GB pod. Newest section
first within each heading.

Measures are referred to by code throughout — `e5`, `d2`, `s4`, `e6`, `d3`, `s1`, `s2`, `s3`, `d4`.
See [`SPECIFICATION.md`](SPECIFICATION.md) or the inventory in
[`handoff/BRIEF.md`](handoff/BRIEF.md) §0.

---

## Decided 2026-08-12, not yet built

Each of these has a step-by-step task document under [`handoff/`](handoff/), and **each task
document carries a status that says whether it may be built yet** — see
[`handoff/README.md`](handoff/README.md) for the vocabulary. This table is work; the
**Suggestions** section below is not, and nothing there gets built without being promoted to a task
document first.

| # | Decision | Task |
|---|---|---|
| 1 | `SCAN_DOSES` becomes `(0.15, 0.30, 0.60)` — the mid band L20–L52 is currently undecidable between inert and under-dosed, and that is where the qualifying region is predicted to sit | [01](handoff/01-scan-doses.md) |
| 2 | A judge null control per judged measure (`e5`, `s1`, `d2`), each gating its own phase before that phase spends judge calls | [02](handoff/02-judge-null-controls.md) |
| 3 | Gate 4 re-anchors on a cell above **this run's own** bisected sanity boundary, measuring all three sanity terms live | [03](handoff/03-gate4-reanchor.md) |
| 4 | Gate 1 selects its anchors from this run's own scan surface, and becomes an instrument gate keyed on judge configuration rather than a per-concept gate | [04](handoff/04-gate1-anchors.md) |
| 5 | Phase 2 emits **tiers** rather than a flat shortlist. Tier 1 always runs (the false-negative audit); further tiers run only when no window has been found. Every knob parametrized, including an exhaustive mode | [05](handoff/05-gate6-false-negative-audit.md) |
| 6 | A relaxed-threshold re-selection pass, so a run that finds no cell at `D2_MAX = 0.20` can be re-read at 0.30 **without re-measuring anything** | [06](handoff/06-relaxed-reselection.md) |
| 7 | ~~Reference cell~~ — **deferred 2026-08-12**, direct comparison against Macar is not this run's framing | [07](handoff/07-reference-cell.md) |
| 8 | A `--debug-bundle` flag that exports everything, including vectors, for benign concepts only | [08](handoff/08-debug-bundle.md) |
| 9 | **Defect:** `s3` scores only uppercase option letters, so a degraded model answering `c` is counted wrong — a dose-dependent bias in a sanity term | [12](handoff/12-mmlu-letter-surface-forms.md) |
| 10 | The `prefix_only` contamination in `e6` / `d3` gets a diagnostic before the scan surface is trusted, and a two-token lookahead only if the diagnostic warrants it | [13](handoff/13-prefix-token-contamination.md) |
| 11 | R5's norm band is a property of Gemma3-27B and does not port. It is scoped to that model and skips elsewhere; what it was really testing — that extraction produced a working direction — is re-derived from the dimensionless `‖v‖/‖h‖` ratio and from behaviour (R14, the §9.3 ladder) | [15](handoff/15-r5-portability.md) |

---

## Open — still needs a decision

### 1. §9.2 secondary control — still flagged as needing review
Unchanged from the spec's own flag. The forced-ID capability control replaces the target vector
with a control concept at the same `(L, r)`; the argument that this isolates capability from
detection has not been settled. Decide on benign-arm evidence, once any run reaches CONTROLS.

### 2. Multi-layer arm — promote or drop
`multilayer.py` is written and unused. Its prediction (distributed transport means spreading
across k layers will NOT reduce detection) is recorded in the module docstring. It needs one
benign concept's evidence to justify keeping it in M2 rather than deferring to M3. Out of scope
for the Garlic/Origami run.

### 3. Gate 5 has never run
Spearman ρ of the cheap `d3` proxy against real `d2`. It needs `verified.jsonl`, which no run has
produced yet. **Below `D3_MIN_RHO`, the residual ranking that put L58 on the shortlist is
unreliable** and Phase 2's route 3 is measuring noise. This is the single most load-bearing
unverified number in the pipeline, and it runs the moment Phase 4 completes.

### 4. Gates that depend on another concept's run
Gates 1, 4 and 6 each keyed their criterion to a constant harvested from a *different* concept's
M1.5 run. That is a portability defect, not just a missing file: this tool is meant to run on
models and concepts nobody has measured before, and a gate that needs a prior measurement of
another concept can never certify such a run. Decisions 3, 4 and 5 above rebuild all three to be
satisfiable from the run's own data. Gate 11 was mis-filed here — it reads the current run's
`D2_transcripts.jsonl` and was only ever blocked by Phase 4 dying.

### 5. R4 — the rig check will not be rebuilt, for now
`r4_rig_check` reads a stored `rig_status.json` (v1, 0.377 against a published 38.2%) and does not
re-run anything. Rebuilding costs ~600 generations and ~1,300 judge calls. **Set aside**, because
the stored status was produced by v1's extraction and v1's judge and M2 has replaced both — so
even a recovered file would validate an apparatus that no longer exists. The apparatus is covered
instead by R14 (hook liveness), R5 (extraction), R7 (prompt match), gate 3 (`e5` judge null) and
gate 11 (`d2` against the upstream judge), plus decision 7's reference cell.

Available if a reviewer asks for an end-to-end calibration against the published aggregate: 10
concepts at L37 / α=4, aggregate detection with both the pooled and the between-concept interval,
written to `rig_status.json` in the schema `r4_rig_check` already reads. Roughly one GPU-hour.

---

## Suggestions — NOT SCHEDULED, DO NOT BUILD

Ideas kept so they are not lost. **None of these is authorisation to implement anything.** To build
one, it gets promoted to a task document under [`handoff/`](handoff/) with a `BUILD NOW` status
first.

### A. More scan doses
Rejected alternatives to decision 1, kept because they are the natural next moves:

- **Four doses** `(0.15, 0.30, 0.60, 1.20)` — covers the deep layers' large sanity headroom
  (`r` = 2.04 at L60, ≥ 2.28 at L61). ~+20 min per concept plus judge spend. Revisit if the 0.60
  pass shows the mid band alive and the deep band still unresolved.
- **Per-layer doses** — bisect the sanity boundary on a coarse layer sample first, then scan each
  layer at a fraction of *its own* boundary. Scientifically the strongest: it makes the scan
  comparable across depth in a way a single global `r` cannot be. Inverts the phase order, so it
  is a structural change and the right thing to build for M3.

### A2. Verify `s3` by single-token generation at the chosen cell
`s3` reads `p(A..D)` and takes the argmax **over those four**, so it cannot distinguish *"the model
confidently answers C"* from *"the model wants to emit something else entirely and C is merely the
highest of the four letters"* — which is what a degraded model does, at exactly the doses where
sanity is the binding constraint. A single generated token at the operating point exposes it
directly. Most of the value is free: report whether the **overall** argmax was an option letter at
all, which is already in the tensor. Task [14](handoff/14-mmlu-by-generation.md).

### B. Report what the shortlist did not measure
Phase 2 now stops below `SHORTLIST_N` rather than padding, and says so. It should also record
the layers it *rejected* for having no signal, so "the concept only lives at L53-L61" is a
readable result rather than an absence.

### C. `m2.setup` should check the volume's real free space
It reported `500814 GB free` — the underlying pool, not the allocation. Harmless today because
nothing sizes off it, but the "under 20 GB free" guard can never fire.

### D. Archive is also the resume marker, and that surprises people
A finished concept's archive makes the next run skip it — correct per spec 14.9, but the loose
folder is wiped, so "resume" means "unzip the archive back first". Either make `m2.run` detect
an archive-without-folder and offer to restore it, or have `m2.setup` say so explicitly.

### E. Phase 6 CONFIRM cost is a pure prior
`CONFIRM=240.0` s has never been observed — no run has reached it. Same for `CONTROLS=60.0`.
Both are single-unit phases, so the measured rate never corrects them and they are pure
guesses in every ETA until a run completes.

---


## Science observations from the first Garlic run (Phases 0–3)

Instrument checks all passed: R5 at L37 gave a vector norm of **4054**, within 1σ of Macar's
4664 ± 982. R14 passed on both injection paths. Judge FPR 0.00. `cap_base` 42/57 MMLU correct
unsteered (0.737). Zero cells unreachable at `ALPHA_CEIL = 16`.

**`d3_base = 0.0000`.** The unsteered forced-ID concept mass for Garlic is a true zero, which is
what makes the `d3 > 0.0` shortlist guard (fixed item 5) so clearly wrong: every dead layer's
`~1e-4` was noise above nothing.

**The dose map inverts with depth.** `||v_L||` runs 109 → 18870 from L13 to L61 while `||h_L||`
runs 1691 → 108142, so α for a fixed `r` *falls* with depth: L15 needs α = 11.2 at r = 0.30,
L61 needs α = 1.72. Macar's raw-α sweeps at fixed α are therefore comparing very different
effective doses across layers — which is the confound `r` was introduced to remove, now
visible in one table.

**Reach is bimodal, not graded.** 0.00 from L13 to ~L52, then 0.17 / 0.33 / 0.42 at L53 / L57 /
L58, then 1.00 at L60 and 0.92 at L61. There is no shallow plateau at these doses.

**L58 is the cell the residual route exists to find**: reach 0.42, D3 0.027, residual −0.375 —
influence without proportional forced-ID detectability. L60 and L61 saturate both axes and are
what Macar's default `L=37, α=4` neighbourhood would have handed you. Whether L58 survives real
E5/D2 in Phase 4 is the first genuine test of the pipeline's premise.

**Sanity headroom grows with depth**: r = 0.27 at L15, 1.42 at L58, 2.04 at L60, ≥ 2.28 at L61.
The early layers tolerate enormous doses because nothing is happening in them.

---

## Measured timings (A100-80GB, Gemma3-27B, 49 layers in scope)

| Phase | Measured | Prior in `PHASE_SECONDS_PRIOR` |
|---|---|---|
| CAL | 350 s (416 s on the first run, incl. MMLU download) | 420 s |
| SCAN | 1213 s / 98 cells = **12.4 s per cell** | 13.0 s |
| SHORTLIST | 0 s | 0 s |
| BISECT | 796 s / 8 candidates = **99 s per candidate** | 100.0 s (was 8.0) |
| VERIFY | never reached | 50 s |
| REFINE | never reached | 50 s |
| CONFIRM | never reached | 240 s |
| CONTROLS | never reached | 60 s |

`BISECT` was priced per bisection PROBE (8 s) while the board ticks once per CANDIDATE — a
candidate is a bracket hunt of up to 6 escalating probes plus `BISECT_STEPS` of bisection, so
the prior was in a different unit from the thing it counted and under-estimated by 12×.
Re-priced to 100 s. **A prior whose unit disagrees with its counter is the same class of
defect as a check that cannot fail** — it looks like a measurement and is not one.

Four phases have never been observed at all. Their priors are pure guesses in every ETA until
one run completes, and CONFIRM and CONTROLS are single-unit, so their measured rate never
corrects them.
