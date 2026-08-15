# 31 — Execute every M3 path offline before any further run

**Status: BUILD NOW — BLOCKS ALL RUNS.** Branch `m3`. No further pod run until this passes.
Absorbs task [30](30-m2-m3-seam-audit.md), which is its static half.

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

## Acceptance

1. `python -m m3.run --concept Garlic` completes under the fake-GPU harness and writes
   `cells.jsonl`, `boundaries.jsonl`, `responses_transcripts.jsonl`, `judge_calls.jsonl`,
   `null_transcripts.jsonl`, `summary.json` and `provenance.jsonl` with correct row counts.
2. `python -m m3.calibrate score` and `full` both complete under it.
3. A test asserts every M3-owned setting reaches the code that reads it, and it fails if a
   setting is added to `m3.config` without being wired.
4. One benign-concept list, consulted by both the run gate and the export gate.
5. Every path that a pod run would execute has been executed at least once locally.

## What this is not

Not a rewrite, and not a reason to vendor `judges.py` or `expensive.py` into `m3/`. All nine
defects are in wiring. The borrowed code is the most valuable thing in the repository and none of
it is implicated.
