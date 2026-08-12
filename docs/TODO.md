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
| 2 | A judge null control per judged measure. `e5` and `s1` **gate** their phase before it spends judge calls; `d2`'s is **reported, not gated**, because at runtime it cannot tell judge error from model confabulation and the latter is expected behaviour | [02](handoff/02-judge-null-controls.md) |
| 3 | Gate 4 re-anchors on a cell above **this run's own** bisected sanity boundary, measuring all three sanity terms live | [03](handoff/03-gate4-reanchor.md) |
| 4 | Gate 1 selects its anchors from this run's own scan surface, and becomes an instrument gate keyed on judge configuration rather than a per-concept gate | [04](handoff/04-gate1-anchors.md) |
| 5 | Phase 2 emits **tiers** rather than a flat shortlist. Tier 1 always runs (the false-negative audit); further tiers run only when no window has been found. Every knob parametrized, including an exhaustive mode | [05](handoff/05-gate6-false-negative-audit.md) |
| 6 | A relaxed-threshold re-selection pass, so a run that finds no cell at `D2_MAX = 0.20` can be re-read at 0.30 **without re-measuring anything** | [06](handoff/06-relaxed-reselection.md) |
| 7 | ~~Reference cell~~ — **deferred 2026-08-12**, direct comparison against Macar is not this run's framing | [07](handoff/07-reference-cell.md) |
| 8 | Debug **capture** and export: persist the intermediates that are currently computed and discarded, then ship everything including vectors, for benign concepts only | [08](handoff/08-debug-bundle.md) |
| 9 | **Defect:** `s3` scores only uppercase option letters, so a degraded model answering `c` is counted wrong — a dose-dependent bias in a sanity term | [12](handoff/12-mmlu-letter-surface-forms.md) |
| 10 | The `prefix_only` contamination in `e6` / `d3` gets a diagnostic before the scan surface is trusted, and a two-token lookahead only if the diagnostic warrants it | [13](handoff/13-prefix-token-contamination.md) |
| 11 | R5's norm band is a property of Gemma3-27B and does not port. It is scoped to that model and skips elsewhere; what it was really testing — that extraction produced a working direction — is re-derived from the dimensionless `‖v‖/‖h‖` ratio and from behaviour (R14, the §9.3 ladder) | [15](handoff/15-r5-portability.md) |
| 12 | Every reported rate carries a **Wilson** interval, on every concept — a binomial SE is exactly zero at p = 0 and p = 1, and the v1 sweep landed there on 29 of 30 cells | [17](handoff/17-wilson-intervals.md) |

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
gate 11 (`d2` against the upstream judge).

Available if a reviewer asks for an end-to-end calibration against the published aggregate: 10
concepts at L37 / α=4, aggregate detection with both the pooled and the between-concept interval,
written to `rig_status.json` in the schema `r4_rig_check` already reads. Roughly one GPU-hour.

### 6. `r = 0.60` is unreachable across L14-L30, so the third dose only half-covers the mid band
Measured at CAL on 2026-08-12: **17 of 147 cells are unreachable at `ALPHA_CEIL = 16.0`, and all
17 are L14-L30 in the `r = 0.60` column.** Nothing is unreachable at 0.15 or 0.30.

The mechanism is in the dose map: `alpha = r * ||h_L|| / ||v_L||`, and that ratio peaks in exactly
that band - at L28, `||v|| = 968` against `||h|| = 49905`, a ratio of 51, so `r = 0.6` would need
`alpha ~ 31`, nearly double the ceiling.

**Consequence for decision 1.** The third dose was added because L20-L52 read `e6` reach 0.00 at
both old doses and "inert versus under-dosed" was undecidable there. It delivers that for
**L31-L52** - which is the band Macar's section 5.1 predicts for the qualifying window on a
62-layer model - but **L14-L30 still has only 0.15 and 0.30** and remains undecidable.

Decide: accept and state it as a scope limitation, or address it. Note that raising `ALPHA_CEIL`
is **not** a free option - it is the v1 damage anchor, and a cell measured above it is off the
manifold the sanity terms were calibrated on. Note also that the four-dose suggestion below is
more constrained than it looks: `r = 1.20` would be unreachable across a far wider band.

### 7. `S1_NULL_MIN` passed by 0.01 on its first measurement
Garlic CAL read **`s1` null 0.91 +/- 0.04 SE** against a floor of **0.90**. The threshold was
derived from the S1 rubric with no measurement behind it, and it came within one point of the
second decimal of aborting the run.

It passed correctly, and the `s2` cross-check worked as designed - `s2 = 1.00` confirms the text is
objectively fine, so a low `s1` there would have been a judge fault rather than the model. But the
margin is not comfortable. **After a second concept has measured it, re-derive the floor from the
two observations** rather than from the rubric alone. Until then the standing rule holds: a failing
null means investigate the judge, never loosen the threshold.

### 8. The model cache is stored twice, and the guard that would warn about it cannot fire
`m2.setup` reported **102 GB at `/workspace/hf/hub`** against an expected ~54 GB. Almost certainly
symlink deduplication failing on the network volume: HuggingFace stores each file once in `blobs/`
and symlinks it from `snapshots/`, and a filesystem without symlink support falls back to copying,
giving exactly 2x.

Harmless for a run. It leaves roughly **48 GB of headroom on a 150 GB volume**, which matters for
task 08's full capture - and the "under 20 GB free" guard is the one that **can never fire**
(task 09 item 1). That guard is already a stated prerequisite for enabling capture; this is the
concrete reason.

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

## Science observations from the 2026-08-12 shakedown (CAL)

Config `2cb66674a108`, A100-80GB, 49 layers in scope (L13-L61), reference L37. CAL 90s.

**Every instrument check passed.** R5 at L37 gave a vector norm of **4054**, within 1 sigma of
Macar's mean and identical to the first run. R14 passed on both injection paths (start_pos and
all-positions both 3.49e+01 at L37 alpha=3.655).

**`d3_base = 0.0000` again**, rank median **1343** - the concept's best token sits 1343 places deep
in the unsteered distribution. Independently corroborated by a standalone probe over the 12 E5
prompts, which measured total unsteered mass of **6.15e-17** across all six concept token ids
against an `E6_THRESH` of 0.01: fifteen orders of magnitude of headroom. **Contamination cannot
manufacture a signal from nothing.**

**`cap_base` = 42/57 (0.737, 95% Wilson [0.610, 0.834]), unchanged by the lowercase fix.** Expected
and not a sign the fix was inert: unsteered, the model capitalises properly. The fix acts at high
dose, where output degrades, so the place to look for its effect is the sanity terms at the top of
the dose ladder.

**The ratio `||h_L|| / ||v_L||` peaks in the L14-L30 band**, which is what makes `r = 0.60`
unreachable there under `ALPHA_CEIL = 16`. This is the same confound the `r` normalisation exists
to remove, now visible as a *reachability* constraint rather than a comparability one: the layers
that resist dosing are the ones where the concept vector is small relative to the residual stream.

**The concept token set for Garlic is four-sixths prefixes.** `concept_first_token_ids` keeps all
six surface forms and drops none, but only `' garlic'` (29508) and `' Garlic'` (127422) are whole
words; `'gar'` (5359), `'Gar'` (39967), `'GAR'` (85402) and `' GAR'` (75987) are prefix-only. In
the unsteered distribution `'Gar'` carries 99.9% of the (vanishing) mass - structurally consistent
with the contamination mechanism, since capitalised `Gar` opens *Gary, Garcia, Garden, Garrett*,
but at 6e-17 the absolute level is nil. See task 13; the decisive measurement is the same
breakdown at a **steered** cell, which needs vectors and therefore runs after this sweep.

**The ETA works.** It read `1h09m21s` at 1m30s elapsed with every pending phase contributing units.
On the first Garlic run the same board read `0m00s` from start to finish.

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
