# 30 — Audit the whole M2/M3 seam before the second M3 run

**Status: PENDING — do not start before the first M3 run.** Branch `m3`. Nothing here blocks that
run; every defect found so far is latent under the current settings and was checked (below).

## Why

M3 reuses four things from M2 — the injection hook, the padded batched generation path, the judge
transport, and run I/O — because each encodes specific documented failures that would otherwise be
rediscovered. That decision stands. What has **not** happened is anyone auditing the seam.

Three blockers were found in it while building M3, and **all three were found by accident** while
doing something else:

1. `m2.judges` validates `judge_id` against a hardcoded `("E5","S1","D2")` in two places and
   parses through a `PARSERS` dict keyed the same way. All four M3 judges raised before a request
   was sent. Found by running a dress rehearsal for an unrelated reason.
2. Cache keys must be the contracted 6-tuple. M3 passed 3 fields from calibration and 5 from the
   sweep. Found by the same rehearsal.
3. `configure_transport` pushed the token cap but not the model or concurrency. The defaults
   happened to match, so nothing looked wrong. Found by checking a value by hand.

Two more were found by directly probing the seam once, and both are the same shape as (3) — a
setting that is **declared, displayed, and not applied**:

4. `GEN_BATCH_MAX` is in `m3.config` and is used only for the battery-size assertion. The actual
   chunking in `expensive._generate` reads `expensive.GEN_BATCH_MAX`, a module constant. Both are
   25 today and the battery is 15, so there is no effect. Raise M3's to 40 and the battery check
   passes while generation silently chunks at 25 — roughly doubling the sweep's GPU time with no
   error anywhere.
5. The dual-use transcript export gate resolves through `m2.config.is_benign`, i.e. **M2's**
   `BENIGN_CONCEPTS`, while `m3.sweep.open_run` refuses on **M3's**. Two gates, two lists. Today
   M3's list is a strict subset of M2's, so the only possible divergence is in the safe direction
   (M3 refuses first). That is luck, not design, and it stops being safe the moment the harmful
   arm exists — which is exactly the gate that must not be approximate.

Three of five were found by accident and two by looking once. That ratio is the argument: the seam
has been sampled, not audited.

## Verified safe for the first M3 run

- `Garlic` passes both benign gates.
- `set(m3 BENIGN_CONCEPTS) - set(m2 BENIGN_CONCEPTS)` is **empty**, so no concept can run and then
  have its transcripts silently withheld at export.
- `m3.battery_size()` is 15, under both copies of `GEN_BATCH_MAX` (25 each).
- `MAX_NEW_TOKENS`, `TEMPERATURE`, `RATE_CI_Z`, `ALPHA_CEIL`, `judge_model`, `judge_concurrent`
  and `JUDGE_MAX_TOKENS` all confirmed to reach the code that reads them.

## What the audit has to do

**The general form of the defect is: a value M3 owns that some M2 code path reads from somewhere
else.** Find them all rather than one at a time.

1. **Enumerate every read.** Every `CONFIG[...]` / `_cfg(...)` index and every module-level
   constant in the M2 modules M3 calls (`model`, `expensive`, `judges`, `vectors`, `runio`,
   `prompts`, `cheap.measure_S2`). For each: does M3 own an equivalent, and does M3's value
   actually reach it? Build the list mechanically, not by reading.
2. **Make the answer testable.** A test that walks those reads and asserts every M3-owned setting
   is applied, so number six cannot be found by accident either.
3. **One benign list.** The export gate and the run gate must consult the same source. Decide
   which owns it and make the other defer.
4. **`m2.setup` reports on the wrong run directory.** `check_run_data` looks at
   `/workspace/m2_runs`; M3 writes to `/workspace/m3_runs`, so on an M3 pod it reports "no
   previous runs" with a half-finished M3 run on disk. The rest of that module (volume, HF_HOME,
   branch, harness, packages, GPU, credentials) is genuinely shared and should stay shared —
   the fix is to make it aware of which pipeline it is checking, not to fork it.
5. **`M2_BRANCH` is the wrong name** for the variable that selects M3's branch. Cosmetic, but it
   is the first thing an operator sets and it currently says the wrong pipeline.
6. **Decide the boundary explicitly.** Right now "M3 reuses M2 infrastructure" is a sentence in a
   design document. It should be a list of named functions with a test asserting nothing else is
   imported, so the seam has a shape someone can check rather than a vibe.

## What this is not

Not a rewrite. The reused code is the most valuable thing in the repository and every defect above
is in the *wiring*, not in the borrowed code. Nothing here suggests vendoring `judges.py` or
`expensive.py` into `m3/`.
