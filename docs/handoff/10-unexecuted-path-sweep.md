# 10 — Sweep the four phases that have never executed

**Status:** required before spending GPU time.
**Blocking:** yes.

## The situation

The first end-to-end run died at the **first cell of Phase 4**. That means **VERIFY, REFINE,
CONFIRM and CONTROLS have never executed a single line in anger.** Roughly half the pipeline, and
the half that produces every reportable number, has only ever been exercised by the offline tests.

Thirteen defects were found in the half that *did* run. There is no reason to expect the untested
half to be cleaner, and every reason to find its defects before a four-hour run rather than during
one.

## What to look for

Read those four paths specifically, hunting the patterns the register already names:

**Checks that cannot fail.** Six of the thirteen were this. The shapes seen so far: a threshold
test on a quantity that never reaches the threshold (`d3 > 0.0` on a probability mass); an import
success used as a proof of something else (an IPython import treated as evidence of a frontend); a
template test that a hard-coded value passes (a `{concept}` placeholder check); a detector that had
never been executed at all. **Ask of every guard in these paths: what input would make this fail?**
If there isn't one, it is not a guard.

**Missing forwarders on the board adaptor.** `_Board` had no `end_phase`/`size_phase`/`skip_phase`.
A missing method raises `AttributeError` **outside** `_call`'s guard — so it is fatal rather than
gracefully degraded, and the board is designed to degrade silently. Check every method these four
phases call on the monitor is actually forwarded.

**Defaults that mask errors.** Silent bugs cluster around anything with a default. In these paths
that means: a `.get(key, 0.0)` on a judge field, a `try/except` broad enough to swallow a real
failure, a threshold read that falls back instead of raising.

**Priors whose unit disagrees with their counter.** See [09](09-known-unfixed.md).

**Anything that only runs once.** CONFIRM and CONTROLS are single-unit phases. Code that executes
exactly once per run gets the least incidental testing of anything in the codebase.

## What to do about it

Extend `m2/tests/test_offline.py` over these four phases as far as is possible without a GPU or a
judge key — the pure selection logic, the row shapes, the resume keys, the board interactions with
a stub. The offline suite is the only thing that will catch a regression here before the next
four-hour run.

Where a defect needs a live model to reproduce, **say so and flag it to the operator as a risk for
the run** rather than leaving it undocumented.

## Acceptance

- A written list of what was inspected and what was found, into
  [`../DEBUG-LOG.md`](../DEBUG-LOG.md) — including "inspected and found nothing" for the paths that
  were clean, because that is the record that makes the next sweep cheaper.
- New offline tests covering the four phases' pure logic.
- Every new guard added has a test that trips it.
