# 06 — Relaxed-threshold re-selection

**Status: BUILD NOW.** Implement as written. Not blocking, but it should exist before the run finishes, because the situation it handles is discovered at the end.

## Goal

A run that finds no cell at `D2_MAX = 0.20` can be re-read at 0.30 **without re-measuring
anything**, and without orphaning the run folder.

## Why this is nearly free

`verified.jsonl` stores the raw `e5`, `d2` and `s4` for every measured cell, plus a `qualifies`
boolean. `select_operating_point` reads `qualifies` — but the boolean is *derived* from the raw
numbers and the three constants. **The measurement does not depend on the threshold; only the
verdict does.** So widening `D2_MAX` is a re-read of rows already on disk, not a new sweep.

The one thing that must not happen: **editing `CONFIG["D2_MAX"]` and re-running.** `CONFIG` is what
`config_hash` is computed from, and the hash names the run folder. Changing it moves the run to a
different folder and abandons every row already measured — the same failure mode as using
`--set` mid-run.

## What to build

1. **A re-selection entry point** that takes an existing run folder and one or more relaxed
   constraints, re-derives `qualifies` over the stored rows, and re-runs `select_operating_point`
   and `frontier`.
2. **It writes a separate record** — e.g. `operating_point_relaxed.json` — inside the same run
   folder, carrying the relaxed constraint values explicitly. **The original
   `operating_point.json` is never overwritten.**
3. **Phase 6 CONFIRM re-runs on the new winner**, if the relaxed selection produces a different
   cell. Screening numbers are not reportable; a relaxed winner that has not been confirmed on
   held-out prompts is not a result either. This is the only part that costs GPU time, and it is
   one cell.
4. **The relaxation is recorded in the run's provenance**, not just in the output filename.

## The scientific point, which the code must enforce

Relaxing a threshold after seeing the data is a researcher degree of freedom. It is legitimate —
"no cell qualified at 0.20, and here is what the frontier looks like at 0.30" is a perfectly honest
sentence — but only if it is **labelled as secondary**. It stops being honest the moment the
relaxed number is reported as though it were the pre-specified one.

So the output must carry, in the record itself and not only in a report someone might write later:

- that the **primary** analysis at `D2_MAX = 0.20` found no qualifying cell;
- that this is a **secondary** analysis at the relaxed threshold;
- both threshold values.

Make it structurally impossible to read the relaxed result as the primary one. A separate filename
and an explicit field in the JSON are the minimum; consider whether `select_operating_point`'s
returned `rule` string should say so too, since that string is what gets quoted.

## Escalate

- **Which constraints may be relaxed this way.** `D2_MAX` is the operator's stated case. `E5_FLOOR`
  and `S4_MIN` are different in kind — relaxing `S4_MIN` means accepting a more damaged model,
  which is not a threshold choice but a change to what the result claims. Propose the allowed set
  with reasoning rather than making everything relaxable.

## Acceptance

- Re-selection over an existing run folder produces a relaxed record **without any generation or
  judge call**, other than the CONFIRM re-run.
- The original `operating_point.json` is byte-identical afterwards.
- A test that trips it: confirm a relaxed record cannot be produced without the relaxed threshold
  appearing in it, and that the primary/secondary labelling is present.
- The run folder's `config_hash` is unchanged.
