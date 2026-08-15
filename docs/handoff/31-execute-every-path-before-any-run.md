# 31 — Execute every M3 path offline before any further run

**Status: DONE. All five acceptance criteria met.** Runs are unblocked. Branch `m3`.
Absorbs task [30](30-m2-m3-seam-audit.md), which is its static half.

**Final count: twenty-one defects.** Fourteen from parts 1 and 2; seven more from part 3, of
which the read-through found four, executing the new tests found two, and a mechanical key-name
sweep found one. Every one is fixed and pinned by a test. The full list is in
[DECISIONS.md](../DECISIONS.md).

## The evidence

Nine defects in M3 so far. What matters is the third column.

| # | defect | how it was found |
|---|---|---|
| 1 | `judge_id` rejected by the transport's hardcoded `("E5","S1","D2")` | dress rehearsal, run for an unrelated reason |
| 2 | cache keys of the wrong arity (3 and 5 fields, contract wants 6) | same rehearsal |
| 3 | `judge_model` / `judge_concurrent` declared, never pushed | checked one value by hand |
| 4 | `GEN_BATCH_MAX` declared, never pushed | probed the seam once, deliberately |
| 5 | export gate reads M2's benign list, run gate reads M3's | same probe |
| 6 | `dtype` missing from the bridge | **a pod run died at model load** |
| 7 | `run_full` read `concept_mentions`, loader writes `concept_hits` | **1,720 paid judge calls discarded** |
| 8 | whitespace-free collapse invisible to every mechanical measure | hand-labelling real transcripts |
| 9 | coherence prompt penalised token-limit truncation | hand-labelling real transcripts |

**Not one was found by a test.** Two were found by a run failing, one after spending money. Three
were found by accident. The unit tests pass throughout — 252 of them — because they test functions
that were called, and every defect above lived in a path nothing had executed.

## Why a read-through is not the task

This repository already ran that experiment. Task [10](10-unexecuted-path-sweep.md) was "sweep the
never-executed phases" by reading them, and it was **DROPPED** with the note:

> the shakedown executed VERIFY and found by running what reading would not have.

Reading is how defects 4 and 5 were found, and they are the two mildest on the list. Running is how
6 and 7 were found, and they are the two that cost something. The method that works here is
execution.

## Part 1 — an offline execution harness (the substance)

Build `m3/tests/fake_gpu.py`: a context manager that stubs the GPU layer so the entire pipeline
runs on a laptop with no torch, no model and no API key.

Stub exactly four seams, and nothing else:

- `m2.model.load_model` → a `RunContext` with a fake tokenizer, a plausible `n_layers`, and
  whatever attributes the real one exposes
- `m2.vectors.extract_all_layers` / `measure_residual_norms` → deterministic fake vectors and
  norms, chosen so `alpha_for` produces a realistic spread across depth
- `m2.expensive.generate_steered` / `generate_unsteered` → canned responses drawn from the
  **2026-08-14 probe archive**, so the text is real model output rather than lorem ipsum
- `m2.judges._post_completion` → canned judge replies, as the calibration rehearsal already does

Then assert that **`python -m m3.run --concept Garlic` runs to completion** under it, on a small
grid, and writes every artefact with the expected row counts.

That single test would have caught 1, 2, 3 and 6 in seconds, locally, before any of them reached a
pod. It is the highest-value thing in this task and everything else is secondary to it.

## Part 2 — the static seam enumeration (task 30)

Every `CONFIG[...]` / `_cfg(...)` index and module-level constant in the M2 modules M3 calls
(`model`, `expensive`, `judges`, `vectors`, `runio`, `prompts`, `cheap`). For each: does M3 own an
equivalent, and does M3's value reach it? Generate the list mechanically and assert on it, so
number ten cannot be found by accident either. Full detail in task 30.

## Part 3 — read the code, with the list in hand

After parts 1 and 2, read every M3 module looking specifically for the two shapes that have
actually occurred here:

- **declared but not applied** — a setting M3 prints, hashes and stores that no code reads
- **name mismatch across a boundary** — one module writing `concept_hits` and another reading
  `concept_mentions`, one gate reading M2's list and another reading M3's

A general "look for bugs" pass is the weakest part of this task, not the strongest, and it goes
last for that reason.

### What part 3 actually found

Seven, and the prediction above was half right. The read-through was **not** the weakest part —
but it was weakest at the two shapes it was pointed at, and strongest at a third nobody listed.

| # | defect | found by |
|---|---|---|
| 15 | Phase 1 fed `max_reachable` straight back to `alpha_for`; the float round trip overshoots `ALPHA_CEIL` on ~1 layer in 16 and Phase 1 does not catch `Unreachable`, so one layer ends the sweep | a new test, at full depth |
| 16 | `max_reachable_dose` recorded with ordinary rounding, so the published limit is unreachable half the time | the test written for 15 |
| 17 | `incoherent_at_floor` on a ladder that stops 12x above the floor — the shipped `BOUNDARY_PROBES=5` needs to be 12 | read-through, arithmetic |
| 18 | the whole null arm truncated out of `read_this.md` by `READ_BUNDLE_N` | read-through |
| 19 | `assert_coherence_blind` subtracts the unclipped response from a clipped payload, and kills the cell claiming a leak | read-through |
| 20 | `calibrate` appends `norms.jsonl` and `null_transcripts.jsonl` on every resume | read-through |
| 21 | `calibrate.load_probe` reads `degeneration_reason`; the archive writes `degeneracy_reason` | mechanical key-name sweep |

Two things worth recording about the method:

**The two named shapes were nearly clean.** Every M3 setting was already applied, and the only
name mismatch left was 21 — which the read-through missed and a mechanical two-sided inventory of
every key written against every key read caught immediately. That is the shape to automate, not
to read for.

**The read-through's real yield was a third shape: a label asserting something the code never
established.** 17 and 18 are both that, and so is 16. This is defect class C — "a check that
cannot fail" — in its reporting form: not a check that always passes, but a *conclusion* that is
printed whether or not the run earned it. 17 is the same mislabel this search was rebuilt to
remove, reintroduced at the other end of the ladder within one commit of fixing it.

**Writing the test found more than reading did.** 15 and 16 came out of a test written to pin a
different fix, and 15 would have killed the next pod run in Phase 1 — it arrived with the boundary
rebuild and had never executed. Two more surfaced only when the judge cache was cleared between
tests: two assertions had been passing on verdicts cached by an earlier test. A test that shares
state with its neighbours is a test that can agree with them instead of with the code.

## Acceptance

1. ✅ `python -m m3.run --concept Garlic` completes under the fake-GPU harness and writes
   `cells.jsonl`, `boundaries.jsonl`, `responses_transcripts.jsonl`, `judge_calls.jsonl`,
   `null_transcripts.jsonl`, `summary.json` and `provenance.jsonl` with correct row counts.
   Also run at the **shipped defaults**, full depth: 49 layers, 196/196 cells, exit 0.
2. ✅ `python -m m3.calibrate score` and `full` both complete under it.
3. ✅ `m3/tests/test_settings_reach.py`. Every setting has a witness that changes its value and
   observes the result **at the code that consumes it** — `expensive.GEN_BATCH_MAX` rather than
   `cfg["GEN_BATCH_MAX"]`, the argument generation received rather than the config that held it.
   `test_every_setting_has_a_reachability_witness` fails by name on an unwired setting, and
   `test_a_setting_that_is_ignored_would_fail_this_test` is the tripwire against the whole file
   becoming a check that cannot fail.
4. ✅ One predicate. `m2.config.is_benign(concept, cfg)` intersects M2's list with the running
   config's, `m2_config` forwards M3's list across, and `transcripts_allowed` passes `cfg`
   through. A caller's list can now only ever **narrow** what may be exported — verified by
   `test_both_benign_gates_resolve_through_one_list`, which asserts `--set
   BENIGN_CONCEPTS=weapon` opens neither gate.
5. ✅ 304 tests pass, in six different file orderings. The judge cache is cleared on entry and
   exit of the harness, so no test is served another's verdicts.

## What this is not

Not a rewrite, and not a reason to vendor `judges.py` or `expensive.py` into `m3/`. All nine
defects are in wiring. The borrowed code is the most valuable thing in the repository and none of
it is implicated.
