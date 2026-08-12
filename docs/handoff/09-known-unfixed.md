# 09 — Three known-unfixed defects

**Status:** carried over from the first end-to-end run. All three are small.
**Blocking:** no.

## 1. `m2.setup`'s free-space check can never fire

It reported `500814 GB free` — the underlying storage pool, not the volume's allocation — so the
"under 20 GB free" guard is dead code that reads as a passing check.

Harmless today because nothing sizes off it. That is exactly why it will still be dead when
something does, and it belongs to the largest defect family in this repository: **a guard that
cannot fail is worse than no guard, because it reads as evidence.**

Fix it to read the allocation, and **test it against a value that should trip it.**

## 2. The archive is also the resume marker, and it surprises people

A finished concept's archive makes the next run skip that concept — correct per spec 14.9 — but the
loose run folder is wiped, so "resume" in practice means "unzip the archive back first". Nothing
says so.

Make `m2.run` detect an archive-without-folder and say so explicitly, naming the command that
restores it. **You will hit this yourself during the run**, most likely at the worst moment.

## 3. CONFIRM and CONTROLS cost priors are guesses

`CONFIRM = 240.0 s` and `CONTROLS = 60.0 s` have never been observed, because no run has reached
either phase. Both are **single-unit phases**, so a phase that replaces its prior with its own
measured rate after two units never corrects them — they are guesses in every ETA the pipeline will
ever print until a run completes.

Nothing to fix before the run. **After** it, replace both with measured values and record them in
[`../TODO.md`](../TODO.md)'s timings table alongside the existing measurements, with the unit
stated beside each number.

While you are in there: the same table records that `BISECT` was priced per bisection *probe* while
the board ticks once per *candidate* — an error of 12× that looked like a measurement for weeks.
**A prior whose unit disagrees with its counter is the same class of defect as a check that cannot
fail.** Check the unit of anything you touch in `PHASE_SECONDS_PRIOR` or `PHASE_UNITS_PRIOR`
against what actually increments the counter.

## Acceptance

- The free-space guard has a test that makes it fail.
- Starting a run for an already-archived concept prints a message naming the restore command.
- Both priors are replaced with measured numbers after the first complete run, with units stated.
