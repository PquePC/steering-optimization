# Decisions and progress log

**Authority on: what has happened.** Append-only, newest at the bottom.

This is the channel between everyone working on this repository. Read the last few entries before
starting a session — they are how you find out what was decided while you were not here.

Three things belong in here and nowhere else:

1. **A decision**, whoever made it — a threshold chosen, a design question answered, a proposal
   accepted or rejected, a scope change.
2. **A task completed** — which task, which commit, anything that turned out differently from what
   the task document said.
3. **A run phase completed** — which phase, how long, the headline numbers by code.

What does *not* belong here: open questions (those go in [`TODO.md`](TODO.md)), defect post-mortems
(those go in [`DEBUG-LOG.md`](DEBUG-LOG.md)), and whether a task may be built yet (that is the
status line in its own document under [`handoff/`](handoff/)).

The split is deliberate. **A task document says what the current state is; this file says how it got
there.** Two files claiming to hold the current status is how a stale one gets believed.

---

## Format

Copy this. Keep it short — a decision nobody can find in ten seconds is a decision nobody reads.

```markdown
## YYYY-MM-DD — one-line title
**By:** PquePC | Opus | Sol   (name whoever actually decided, not who typed it)
**Kind:** decision | task complete | phase complete

What happened, in one or two sentences.

**Why:** the reasoning, if it is not obvious from the what. Skip this line if it is.
**Result:** the commit, the task document, the file — whatever someone would open next.
```

**Name the person or agent who made the call, not the one who wrote it down.** A threshold the
operator chose and Sol implemented is `By: PquePC`. If a decision came out of a conversation, name
both.

---

# Log

## 2026-08-12 — Documentation consolidated into one authority per question
**By:** PquePC (direction), Opus (execution)
**Kind:** decision

Fifteen overlapping markdown files became a root `README.md` and `AGENTS.md` plus eight documents
under `docs/`, each opening with a statement of what it is authoritative on. Superseded v1 guides
moved to `docs/archive/`.

**Why:** four separate documents explained how to run the pipeline and there was no README at the
root, so finding the current answer to anything meant knowing which document had been written last.
**Result:** `f97b300`.

## 2026-08-12 — Eight design decisions for the next run, and a task queue
**By:** PquePC (decisions), Opus (analysis and drafting)
**Kind:** decision

`SCAN_DOSES` gains a third dose at 0.60. Gates 1, 4 and 6 are rebuilt to be satisfiable from the
run's own data. R4 is set aside. Judge null controls, relaxed re-selection and a debug bundle are
added. Each has a task document under `handoff/`.

**Why:** gates 1, 4 and 6 each keyed their criterion to a constant from a different concept's run,
which cannot certify a run on a model nobody has measured. R4's stored status came from v1's
extraction and v1's judge, both of which M2 replaced.
**Result:** `5c1d7df`, `docs/handoff/`.

## 2026-08-12 — Shortlist becomes tiered; two defects queued
**By:** PquePC
**Kind:** decision

Phase 2 emits ordered tiers rather than a flat shortlist. Tier 1 always runs as the false-negative
audit; further tiers escalate only when no window has been found. Every knob parametrized,
including an exhaustive mode. The reference-cell task is deferred.

**Why:** three extra cells is our budget, not a property of the method — someone needing the
genuinely best steering point should be able to pay for as much of the surface as they want. The
reference cell went with deferring direct comparison against published rates.
**Result:** `ec6faf2`, tasks 05, 07, 12, 13, 14.

## 2026-08-12 — Task statuses formalised; R5 scoped; judge bake-off queued
**By:** PquePC (direction), Opus (execution)
**Kind:** decision

Every task document opens with a status from a fixed vocabulary, and nothing is built unless it
begins with `BUILD NOW`. R5's norm band is scoped to Gemma3-27B and skips elsewhere. A judge
bake-off on stored transcripts is queued for after the run.

**Why:** R5's ±2σ band around 4664 is a property of one model — a healthy vector from another
architecture fails it. Judges are about a quarter of run cost, so the question is not "cheapest"
but "cheapest that is demonstrably as good", and that has to be measured.
**Result:** `eb8c26e`, tasks 15 and 16.

## 2026-08-12 — Third scan dose added
**By:** PquePC (decision), Sol (execution)
**Kind:** task complete

`SCAN_DOSES` is now `(0.15, 0.30, 0.60)`, and the opening SCAN prior counts 147
`(layer, dose)` units. At the measured 13 seconds per cell, the added 49-cell pass costs about
637 seconds (10.6 minutes) per concept.

**Result:** `8d0d230`, task 01.

## 2026-08-12 — Lowercase MMLU option surfaces counted by `s3`
**By:** Sol
**Kind:** task complete

`s3` now scores bare and space-prefixed uppercase and lowercase option letters using max, not sum,
across forms. A synthetic lowercase-answer test and a readable collision-guard test both trip the
previous defect; the next fresh CAL will re-measure `cap_base`.

**Result:** `8d0d230`, task 12; post-mortem in `DEBUG-LOG.md`.

## 2026-08-12 — Judge null control thresholds settled
**By:** PquePC (decision), Sol (proposal), Opus (review)
**Kind:** decision

Task 02's four values are settled and the task moves to `BUILD NOW`. `e5` null maximum becomes
`E5_TIE_BAND` (0.5) **after** raising `N_FPR_PAIRS` from 2 to all available open prompts; `s1` null
minimum is 0.90; `d2` null maximum is `1 / N_D2` (0.04) with transcripts persisted and a Wilson
interval reported; a failing null aborts at Phase 0 for all three.

**Why:** Sol's `e5` rubric anchor was correct (`1-2 = Faint`), but the reading was a mean over two
judge calls, where the achievable means are 0, 0.5 and 1.0 — a threshold on that measures luck, so
the sample size had to be fixed before the number meant anything. Once it is real, a *systematic*
offset should be held at least to the standard of *random* noise, which is what `E5_TIE_BAND`
already encodes. Sol's `s1` derivation from `S4_MIN` answered the wrong question: `S4_MIN` is the
floor a damaged cell must clear, while this null asks what the judge scores on healthy output,
which the rubric says should be indistinguishable from A — the top of the scale. At 0.70 a judge
could depress every `s1` by 0.30 and pass, discarding healthy cells as damaged, which is task 12's
failure in another measure. The `d2` null needed its transcripts because a nonzero reading has two
causes with opposite remedies — the model confabulating, which is documented here, or the judge
misscoring — and only the transcripts tell them apart.

**Result:** task [02](handoff/02-judge-null-controls.md), *Decided values*. All four change
`config_hash`, so they land together with the `SCAN_DOSES` change before any measurement.

## 2026-08-12 — `d2` null is reported, not gated; intervals go Wilson everywhere; debug needs a capture half
**By:** PquePC
**Kind:** decision

Three changes on top of the judge-null decision above.

**The `d2` null stops being a gate.** It is measured, reported beside every `d2` as the unsteered
baseline, and its transcripts persisted for post-run review. Selection still uses raw `d2` against
`D2_MAX`.

**Every reported rate carries a Wilson interval**, on every concept — replacing the binomial SE for
proportions. `e5` keeps mean ± SE, being a mean rather than a proportion.

**The debug bundle gains a capture half** and becomes blocking for the first Garlic and Origami
runs, which the operator intends to audit end to end.

**Why:** a null control may gate only if it can distinguish an instrument fault from real model
behaviour **at runtime**. The `d2` null cannot — a nonzero reading is either the judge misscoring or
the model confabulating, the latter is expected behaviour already observed here, and only a human
reading transcripts can tell them apart. The same rule *tightened* the `s1` null rather than
loosening it: `s2` resolves the identical ambiguity mechanically and for free, so the `s1` gate now
fires on the disagreement (`s1` low while `s2` says the text is fine) rather than on `s1` alone.
On intervals, `sqrt(p(1−p)/n)` is exactly zero at p = 0 and p = 1, and the v1 sweep put 29 of 30
cells on exactly those endpoints — `d2 = 0.000 ± 0.000` claims perfect certainty from 25 trials.
On the bundle: **nothing in `m2/` writes to `debug/`** — the directory, the deny-list entry and the
archive carve-out are vestigial from v1 — so flipping the export filter ships one extra thing and
every intermediate is computed and discarded. An export flag can only ship what was written.

**Result:** `1974000`; tasks [02](handoff/02-judge-null-controls.md),
[08](handoff/08-debug-bundle.md), [17](handoff/17-wilson-intervals.md). Task 09 item 1 (the
free-space guard) becomes a prerequisite for enabling full capture.

## 2026-08-12 — Judge nulls and Wilson reporting implemented
**By:** PquePC (decision), Sol (execution)
**Kind:** task complete

Phase 0 now measures all three judge nulls before judged science phases. E5 uses all seven open
prompts and gates at 0.5; S1 gates only when its sub-0.90 reading disagrees with objective S2;
D2 is never fatal, persists its unsteered transcripts, and is reported beside every raw D2.

One Wilson-score helper now supplies endpoints and the surviving n for binomial rates throughout
scan, verification, confirmation, controls, auxiliary arms, gates, the frontier and
`operating_point.json`. E5 and S1 remain score means with standard errors. Gate comparisons remain
on their existing point estimates: intervals change the honesty of reporting, not the criteria.

**Why:** the null-gating rule requires live separation of instrument failure from model behaviour,
which E5 provides directly and S1 obtains from S2, but D2 cannot provide without human transcript
review. Wilson intervals prevent endpoint observations such as 0/25 from masquerading as perfect
certainty; conceptually, observing zero events narrows the plausible rate but does not prove the
true rate is exactly zero.

**Result:** `4b1e66a`, tasks 02 and 17. Offline suite: 80 passed, 2 environment-dependent skips.

## 2026-08-12 — Gate 4 reanchored on the live aggregation contrast
**By:** PquePC (decision), Sol (proposal and execution)
**Kind:** task complete

Gate 4 now prefers `hi`, the failing endpoint of the winning layer's converged Phase 3 interval,
and falls back to another shortlisted layer with a boundary. It measures S1, S2 and S3 live without
spending unrelated E5 or D2 calls. The gate passes only when the terms disagree, the minimum falls
below `S4_MIN`, and their mean remains at or above it. Thus the evidence is that `min` catches a
failed term a mean would hide; the guaranteed fact that S3—and therefore the minimum—is below the
floor at `hi` is recorded but never counted as proof.

“Comfortably above” is derived as one third of the remaining headroom above `S4_MIN`, currently
0.80. If `hi` is uniformly damaged, the only fixed retries are the midpoint and then `lo`, always
decreasing dose. A disagreeing anchor whose mean also rejects fails immediately rather than being
searched away. If no shortlisted layer reached a boundary, Gate 4 is SKIPPED and reports that
sanity held at every reachable shortlist dose. The historical M1.5 path remains diagnostic-only
because stored text cannot reconstruct S3.

**Why:** `_cheap_sane` defines `hi` by `s3 < S4_MIN`, so `min(S1,S2,S3) < S4_MIN` there is true by
arithmetic and cannot validate a gate. The lost Velocity anchor mattered because the old merged
sanity accepted a cell whose weakest term failed. The live min-versus-mean contrast reproduces that
load-bearing property without borrowing another concept's data.

**Result:** `97fb032`, task 03. Offline suite: 84 passed, 2 environment-dependent skips.

## 2026-08-12 — Tasks 02 and 17 landed; gate 4's anchor settled and its criterion corrected
**By:** Sol (implementation and proposal), PquePC (decision), Opus (review)
**Kind:** task complete + decision

Tasks [02](handoff/02-judge-null-controls.md) and [17](handoff/17-wilson-intervals.md) are
implemented (`4b1e66a`), 80 tests passing, no measurements run. Spot-checked on review: Wilson
gives `0/25 → [0.000, 0.133]` and `25/25 → [0.867, 1.000]`, non-degenerate at both ends, and
`D2_NULL_REFERENCE` appears nowhere in `gates.py`, so the `d2` null genuinely reports rather than
gates.

Task 03: Sol's anchor **location** is accepted — `hi`, the failing endpoint of the converged
bisection interval. Its proposed **criterion** is rejected and replaced.

**Why:** `_cheap_sane` is `s3 >= S4_MIN`, so `hi` is *defined* as a dose where `s3 < S4_MIN`, and
`s4 = min(s1, s2, s3) ≤ s3`. A gate whose pass condition is `s4 < S4_MIN` at that anchor therefore
holds by arithmetic on every model and every concept — a check that cannot fail, landing inside the
gate whose job is to stop a false-positive result. Gate 4's real content is a property of the
**aggregation rule**: v1's merged metric passed the destroyed Velocity cell at 0.779 because a mean
can be dragged up by the terms that are fine, and a `min` cannot. So the criterion becomes: the
three terms must **disagree** at the anchor, and the gate reports what a mean would have said
beside what `min` said. `hi` is the ideal location for exactly that reason — being marginal, it is
where `s3` has just crossed while `s1` and `s2` are most likely still fine.

Also settled: if the winning layer never reaches a boundary, fall back to any other bisected
candidate that did before skipping, because the aggregation rule is not layer-specific; skip rather
than fail when none has one, and report "sanity held at every reachable dose across the shortlist"
as a finding rather than an absence.

**Result:** task [03](handoff/03-gate4-reanchor.md), *Decided*.
