# 15 — R5 does not port to another model

**Status: BUILD NOW — PROPOSE FIRST.** The replacement check is a design call; propose it before writing code. Not blocking.

## The problem

`r5_reference_norm` checks that the extracted vector's norm at the reference layer sits inside
±2σ of **4664 ± 982** — Macar's published spread across concepts on Gemma3-27B.

Those numbers are a property of **one model**. Residual-stream magnitudes scale with hidden
dimension, with normalisation choices and with training; a vector extracted from a different
architecture has no reason to land in that band, and a 4B model would fail R5 while being perfectly
healthy. This is the same portability defect as gates 1, 4 and 6 (see [`BRIEF.md`](BRIEF.md) §5):
a check keyed to a constant from someone else's measurement cannot certify a run on a model nobody
has measured.

It is also worth being precise about what R5 does **not** test. The repository already records that
vector norm does not predict detection — *Treasures* has the largest norm measured (6688) and
detects at 0.000. So R5 is not a quality check on the concept. Its job is narrower: **did extraction
produce a real direction, rather than zeros, noise, or the wrong tensor?**

## What should replace it

Two checks, both portable, and the second is the one that actually matters.

**1. A dimensionless magnitude check.** The pipeline already invented the right quantity: `r`, the
ratio `α·‖v_L‖ / ‖h_L‖`. A raw norm is model-specific; a *ratio* against the residual-stream norm at
the same layer is not. Check that the extracted vector's magnitude relative to the residual stream
falls in a workable range — that a usable dose is reachable at all under `ALPHA_CEIL`, and that the
vector is not effectively zero. That is a genuine instrument check and it transfers to any model.

**2. A behavioural check, which is the real test.** Extraction worked if **injecting the vector
moves the model toward the concept.** Magnitude is a proxy for that; behaviour is the thing itself.

Much of this already exists and is not currently framed as an extraction check:

- **R14** asserts the hook actually steers, on both injection paths, and raises rather than
  reporting.
- **The §9.3 escalation ladder** escalates `r` at the reference layer until `e6` reach clears the
  floor or α would exceed `ALPHA_CEIL`, precisely to distinguish *"no operating point exists"* from
  *"the vector is dead"*.

Between them, "did extraction produce a working direction" is largely answered already — by
behaviour, on this run's own data, on any model. **Work out how much of R5's role is genuinely
uncovered before building anything new**, and say so. The answer may be "almost none", in which
case the right change is small.

## What to do with R5 itself

**Do not delete it.** On Gemma3-27B it is real corroboration and it costs nothing. Demote it:

- keep it, but scope it explicitly to the model whose band it encodes — it should **skip with a
  clear reason** on any other model rather than failing;
- make the band a configured property of the model, not a constant buried in a gate, so pointing
  the pipeline at a new model does not silently inherit Gemma3-27B's numbers;
- move it out of the set whose failure is treated as an instrument fault.

## Propose before building

- **Which of R5's role is genuinely uncovered** by R14 and the §9.3 ladder, with the reasoning.
- **The dimensionless range** for check 1, derived rather than picked. `ALPHA_CEIL` and the observed
  dose map are the natural sources; a number invented here and left unexplained is a researcher
  degree of freedom sitting on an instrument check.
- **Whether check 2 needs anything new at all**, or is a reframing and a report of what already runs.

## Acceptance

- A test that trips whatever replaces it: a synthetic zero or near-zero vector must fail.
- Pointing the pipeline at a different model does not fail R5 — it skips it, with a reason naming
  the model the band belongs to.
- The run record states which extraction checks ran and which were model-specific.
