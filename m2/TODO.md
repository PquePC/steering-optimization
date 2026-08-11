# M2 — TODO and findings log

Running record of what the pipeline got wrong, what was fixed, and what still needs a decision.
Started 2026-08-11, after the first end-to-end Garlic run on a fresh A100-80GB pod.

Newest section first within each heading. Every fixed item names the commit, so a defect that
comes back can be traced to the change that was supposed to stop it.

---

## Open — needs a decision before the next run

### 1. The scan doses are screening the wrong end of the axis for Garlic
`SCAN_DOSES = (0.15, 0.30)`, but Phase 3 found sanity headroom of **r = 2.04 at L60** and
**r = 1.42 at L58**. L60 already saturates (reach 1.00, D3 0.999) at r = 0.30, and stays sane
seven times higher. Meanwhile L15 goes insane at r = 0.27.

So for this concept the whole interesting region sits *above* the scanned range at deep layers
and *below* it at shallow ones — a single global dose pair cannot screen both. Options:

- leave it, and accept that the scan resolves the shallow end only;
- make `SCAN_DOSES` per-layer, e.g. a fraction of that layer's bisected sanity boundary — but
  the boundary costs a Phase 3 bisection to find, so this inverts the phase order;
- add a third scan dose (~0.60). One more full-depth pass, ~+10 min per concept.

Decide on the benign arm before the harmful arm runs; changing it later changes `config_hash`
and orphans every row already measured.

### 2. §9.2 secondary control — still flagged as needing review
Unchanged from the spec's own flag. The forced-ID capability control replaces the target vector
with a control concept at the same `(L, r)`; the argument that this isolates capability from
detection has not been settled. Decide on benign-arm evidence, once any run reaches CONTROLS.

### 3. Multi-layer arm — promote or drop
`multilayer.py` is written and unused. Its prediction (distributed transport means spreading
across k layers will NOT reduce detection) is recorded in the module docstring. It needs one
benign concept's evidence to justify keeping it in M2 rather than deferring to M3.

### 4. Gate 5 has never run
Spearman ρ of the cheap D3 proxy against real D2. It needs `verified.jsonl`, which no run has
produced yet. **Below 0.70, the residual ranking that put L58 on the shortlist is unreliable**
and Phase 2's route 3 is measuring noise. This is the single most load-bearing unverified
number in the pipeline.

### 5. Gates 1, 4, 6, 11 need the M1.5 artefacts
All four skip with "stored M1.5 artefact is not present". They want `$M2_M15_DIR` or an
`m1_5/` folder under the run dir. The v1 lab's outputs are on a volume that has been wiped
since. Either re-point them at an archived v1 bundle or accept these four as permanently
skipped — **and a skipped gate is not a passed gate**, so the second option needs to be
written down as a limitation rather than left implicit.

### 6. Credentials from the 2026-08-11 screenshot are unrotated
`HF_TOKEN`, `OPENROUTER_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `HEALTHCHECK_URL`,
`RUNPOD_API_KEY` were visible in a pasted terminal screenshot. The RunPod key is Read/Write and
can terminate pods — rotate that one first.

---

## Fixed — 2026-08-11

| # | Defect | Why it survived | Commit |
|---|---|---|---|
| 1 | `_item` dropped `model_text`, so `judge_many`'s second gate-2(a) check ran over the model's own words and killed Phase 4 at the first cell | The strict default is correct and deliberate; the caller simply never declared the spans. Both layers were individually right | `955495f` |
| 2 | S1's calibration example quoted `'Velocity'`, a benign-list concept, so gate 2(a) failed a run measuring something else | The existing test only checked for a `{concept}` *placeholder*; a hard-coded name passed it | `955495f` |
| 3 | Driver read `select_operating_point`'s envelope as the winning row | The envelope is a dict and never `None`, so `winner is not None` always held | `955495f` |
| 4 | Nothing called `end_phase` — phases never left `running`, and the per-phase phone push never fired once | The board degrades silently by design | `955495f` |
| 5 | Shortlist widened onto dead layers (L13/L14/L15, reach 0.00) | The guard tested `d3 > 0.0`, and D3 is a probability mass that is never exactly zero | `3d9572d` |
| 6 | Status board printed `{'text/plain': ...}` blobs between phases | IPython ships with the pod image, so the import succeeded, `display()` found no frontend and printed `repr()` rather than raising | `3d9572d` |
| 7 | Provenance row called a bare `_now()` that lives in `runio` | Wrapped in `except Exception` so it cost one WARN line, not a row | `3d9572d` |
| 8 | The undefined-name detector referenced an unimported `pathlib` — the exact class of defect it hunts | It had never been run | `3d9572d` |
| 9 | `m2.setup` reported `nest_asyncio` missing after installing it | The probe printed `pkg.__version__`, which `nest_asyncio` has never defined | `79fc04d` |
| 10 | Absent model cache reported as a permanently unfixable `FIX` | No repair function exists for it, and none should — the preflight downloads it | `4a53c17` |
| 11 | R14's skip text said "RUN IT BEFORE ANY SWEEP" in a preflight, where it is structurally impossible and already handled | The message was written for a different caller | `cacdf83` |
| 12 | ETA read `0m00s` for a whole 40-minute run | `eta()` skips phases whose unit total is 0, and a total was only set when a phase *started* — so every phase ahead contributed nothing | *(this change)* |
| 13 | `_Board` had no forwarder for `end_phase`/`size_phase`/`skip_phase` | A missing method is an `AttributeError` on the adaptor, **outside** its `_call` guard — fatal, not degraded | *(this change)* |

### Pattern worth naming
Six of the thirteen are **checks that could not fail**: `d3 > 0.0` on a quantity that is never
zero, an import test used as a frontend test, a `{concept}` placeholder test that a hard-coded
name passes, a detector that was never run. A guard that passes everything is worse than no
guard, because it reads as evidence. New guards should be tested against a case that *should*
trip them — `test_the_undefined_name_detector_actually_detects` is the model to copy.

---

## Suggestions — not yet scheduled

### A. Per-layer dose scanning (see open item 1)
The strongest version: run Phase 3's bisection *first* on a coarse layer sample, then scan each
layer at a fraction of its own sanity boundary. Costs one extra bisection pass; makes the scan
comparable across depth in a way a global `r` is not.

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
