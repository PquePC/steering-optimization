# 11 — The run

**Status: RUN.** Everything marked BUILD NOW lands first.

## Your role during the run

You do **not** have pod access. You guide the operator:

- **Copy-paste-ready commands**, one per fenced block, working directory stated.
- The operator pastes log output back. **Diagnose from what they paste.**
- **Never report a phase as passed, a gate as green, or a number as measured unless you have seen
  the line that says so.** If you need a specific line, give the exact command that produces it.
- On failure: the diagnosis, the fix, and whether the run resumes from where it stopped or must
  restart — and what that costs in time and money.
- Keep a running record of where the run is, so a dropped session does not lose the thread.

Full operating detail is in [`../RUNBOOK.md`](../RUNBOOK.md). Do not restate it here; link to it.

## Order

1. Fixes committed and pushed to `main`; offline tests green.
2. On the pod: `python -m m2.setup`; missing Python packages install automatically and the checks
   run again. Review any non-package `FIX` before choosing `--repair`, and resolve every `BLOCK`
   before continuing.
3. **Preflight** — `python -m m2.run --concepts Garlic --preflight`. **Read the FAIL count, not the
   pass count**: `0 FAIL` and `summary PASS` is the green light. Five of seven rig checks skip here
   and that is correct — know which five and why before interpreting the output. Confirm the
   upstream `eval_utils` import works here (see below).
4. **Garlic**, full pipeline, watchdog running in a second terminal first.
5. **Origami**, full pipeline.
6. Gates, including the rebuilt 1 and 4, the redesigned 6, and the now-runnable 5 and 11.

`--no-stop-pod` on every run: without it the pod stops the moment a concept finishes and the result
cannot be read without a restart. The watchdog matters because a hung process holds the GIL, so
liveness detection has to live outside it.

## What to watch for

**If R14 fails in Phase 0, stop the run.** The injection hook is not steering. Every forward-pass
measure would read exactly zero, and a mean of zero with a variance of zero passes most checks
designed to catch a weak effect. This is bug 26's shape and it is silent — it once produced 30
cells of bit-identical steered and unsteered distributions that nothing flagged.

**Phase 4 is where the last run died.** Watch its first cell rather than backgrounding it.

**Gate 5's ρ, the moment `verified.jsonl` exists.** Below `D3_MIN_RHO`, spec 5.3's consequence
applies: Phase 1 loses its detection axis, the shortlist is built on `e6` alone, and `SHORTLIST_N`
is raised. **Do not apply it by mutating `CONFIG` mid-run** — `CONFIG` is what `config_hash` is
computed from, so editing it moves the run to a different folder and abandons every row already
written. The driver reads `gate5_d3.json` and acts on it.

**Gate 11 needs the upstream judge available.** `eval_utils` must be importable *with* an API key,
and `nest_asyncio.apply()` must run before any repo judge call — the upstream uses `asyncio.run()`,
which is illegal inside a running loop. Verify this at preflight, not four hours in. Watch the
**rate delta**, not just the agreement fraction: symmetric disagreements cancel in a rate, a
systematic shift does not, and it is the systematic shift that would quietly redefine `d2`.

**Unreachable cells.** `alpha_for` raises above `ALPHA_CEIL` and must never clamp — a clamped α is
a silently wrong dose, and every row measured at it would carry an `r` the model never saw.

**A cell qualifying with a suspiciously low `d2` is the failure mode to distrust**, not the result
to celebrate. Low `d2` because the model is damaged and low `d2` because the concept is covert look
identical until you check `s4` and the §9 controls.

## If no cell qualifies

That is a result, not a failure. The frontier is still reported, and the §9.3 escalation ladder
distinguishes *"no operating point exists at these constraints"* from *"the vector is dead"*.
`operating_point.json` records which.

Then offer the operator the relaxed re-selection from [06](06-relaxed-reselection.md) —
`D2_MAX = 0.30` over the rows already measured, no new sweep — and make sure the result is labelled
secondary.

## Reporting

Per [`BRIEF.md`](BRIEF.md) §7. The section that decides whether the number is publishable is the
gate table: every gate, passed / failed / not-run, and for each not-run one sentence on what it
would have validated. Write it for a sceptical reader.
