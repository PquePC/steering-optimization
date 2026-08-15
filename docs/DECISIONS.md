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

## 2026-08-12 — Gate 1 now certifies the live judge configuration on current-run anchors
**By:** PquePC (decision), Sol (execution)
**Kind:** task complete

Gate 1 no longer borrows the lost Pillows and Silk cells. From the current Phase 1 surface it
selects a HIGH candidate that clears `E6_FLOOR` while every D3 trial remains below
`D3_RATE_THRESH`, and a LOW candidate with exactly zero E6 reach but a nonzero D3 rate. The LOW
cell must also have greater D3 probability and a better rank than HIGH, so the token-accessibility
ordering is genuinely reversed rather than merely described. If the run supplies no such pair,
the gate skips instead of inventing an anchor.

The gate generates the standard twelve steered/unsteered pairs at each anchor and writes one
deterministically shuffled, role-blind `gate1_hand_labels.jsonl`. Only `hand_label` is blank and
only the operator may fill it. A sidecar retains the hidden anchor mapping, the measured E6/D3
probabilities and ranks, and a key made only from the judge model plus a SHA-256 digest of the E5
prompt. Consequently the same operator labels are reused across concepts, while any judge-model
or prompt change is refused explicitly and the old labels are preserved.

The pass criterion remains the settled content of the gate: mean E5(HIGH) minus mean E5(LOW) must
be at least 3.0. Spearman correlation with all 24 hand labels is reported but never gates. The
offline suite includes the required counterexample where LOW outranks HIGH, a configuration-change
refusal, and a check that the operator packet is mixed and carries no anchor-role field.

**Why:** the gate asks whether E5 reads semantic influence or merely literal token accessibility.
A current-run cross-proxy reversal makes that question portable across models and concepts; a
content-addressed judge key makes the irreducible human work reusable without letting stale labels
certify a changed instrument.

**Result:** `cf13fdf`, task 04. Offline suite: 88 passed, 2 environment-dependent skips.

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

## 2026-08-12 — Tasks 03 and 04 landed; task 05's ordering and exhaustive-mode behaviour settled
**By:** Sol (implementation and proposals), PquePC (decision), Opus (review)
**Kind:** task complete + decision

Tasks [03](handoff/03-gate4-reanchor.md) (`97fb032`) and [04](handoff/04-gate1-anchors.md)
(`cf13fdf`) are implemented, 88 tests passing, no measurements run.

Task 05: config names approved as proposed. **Tier ordering becomes `e6_residual_interleave`**,
superseding the earlier `e6_desc`. **Gate 6 in exhaustive mode reports NOT APPLICABLE**, in a
gate-table state distinct from both PASS and SKIPPED.

**Why:** Phase 2 can wrongly reject a layer in two independent ways — missing the `E6_FLOOR` cutoff
and mishandling the residual route — and the residual route is the one that found L58 on the first
Garlic run, the exact influence-without-detectability shape this pipeline exists to find. An audit
ordered on `e6` alone would leave it untested. On exhaustive mode, reporting PASS with no rejected
population would be a gate that cannot fail; and "not applicable" must be distinguishable from
"could not run", because exhaustive coverage is *stronger* than a gate-6 pass rather than a gap in
the evidence.

**Caveats added:** the residual ordering must inherit the `D3_SIGNAL_MIN` guard from `1dc85b1` — on
a dead layer both `d3` and `e6` are ≈ 0 so the residual is fit noise, and a mildly negative value
would rank dead layers first, reintroducing that bug through a new door. `SHORTLIST_MAX_TIER = None`
and `SHORTLIST_EXHAUSTIVE = True` are different behaviours and both comments must say so.
`SHORTLIST_TIER_ORDER` must raise on an unknown value rather than defaulting. Tier 1 always running
adds `SHORTLIST_TIER_SIZE` layers to both BISECT and VERIFY — about 7.5 minutes per concept at the
default — so `PHASE_UNITS_PRIOR` must account for it from the opening board.

**Result:** task [05](handoff/05-gate6-false-negative-audit.md), *Decided*.

## 2026-08-12 — Task 05 landed; Garlic shakedown moves ahead of further static review
**By:** PquePC (decision), Sol (execution)
**Kind:** task complete + run-order decision

Task [05](handoff/05-gate6-false-negative-audit.md) is implemented in `63cdc35`. Phase 2 now emits
tier 0 plus ordered live rejected tiers; tier 1 always runs, while deeper tiers escalate only on
failure and respect the configured limit. The approved ordering alternates `e6`, residual, `e6`
and excludes dead layers through the same `D3_SIGNAL_MIN`/unsteered-baseline guard before either
queue ranks them. Every verified/refined row retains its source tier and ordering route, and
`tier_verification.json` records per-tier verdicts and the explicit stop reason.

Gate 6 now audits the current run rather than hard-coded M1.5 cells: an equal-or-better outer-tier
qualifier fails the gate while remaining eligible for normal selection; lower outer qualifiers are
reported. Exhaustive coverage receives `NOT_APPLICABLE`, distinct from PASS and SKIP. The five
settled config constants landed together and therefore produce one new `config_hash`; BISECT and
VERIFY opening priors each count 11 units (eight expected tier-0 units plus three mandatory audit
units). `--exhaustive` reports the full cell, judge-call and time estimate before measurement.

The offline suite is green at 96 passed and 2 environment-dependent skips; edited modules also
pass compilation and diff checks. No measurement was run during implementation.

**Run-order decision:** stop static-review work after Task 05 and run Garlic as a deliberately
non-reportable shakedown. Its purpose is to execute VERIFY, REFINE, CONFIRM and CONTROLS for the
first time and obtain Gate 5's rho; it is not evidence for publication. Task 10 is not started.
Task 15 is dropped from the pre-run queue because R5 portability does not affect Garlic on
Gemma3-27B. Tasks 06 and 13 step 2 wait for the shakedown. Tasks 08 and 09 item 1 follow the
shakedown and still precede the real run.

**Result:** `63cdc35`, task 05 complete; next action is the Garlic shakedown from task 11.

## 2026-08-12 — Patch only documented open defects with settled fixes
**By:** PquePC (scope), Sol (execution)
**Kind:** scope decision + partial task completion

The operator limited this session to open bugs whose solutions were already stated in the
documentation. That authorised Task 09 items 1 and 2 and excluded feature work, diagnostic-only
work, proposal-first work and static sweeps: Tasks 06, 08, 10, 13 and 15 were not touched.

Task 09 items 1 and 2 landed in `928e8eb`: RunPod free space is now computed from the network
volume's allocated size rather than its shared backing pool, with a test that forces the guard to
block; and `m2.run` now detects an archive without a loose resume folder and prints the exact
extract-and-preserve command before model loading.

Task 09 is not marked DONE. Item 3 explicitly requires measured CONFIRM and CONTROLS timings from
a complete run, and the documentation still records both as never reached. Replacing those priors
now would turn one unmeasured guess into another.

**Result:** `928e8eb`, Task 09 items 1-2 complete; 99 tests passed, 2 environment-dependent skips.

## 2026-08-12 — Shakedown run: CAL complete, all gates passed
**By:** PquePC (ran it), Opus (read it)
**Kind:** phase complete

Garlic, config `2cb66674a108`, A100-80GB. **CAL done in 90 s.** First execution of the judge null
controls, the third scan dose and the corrected ETA.

Headline numbers by code: R5 pass (L37 `||v||` 4054, within 1σ). R14 pass on both injection paths.
`d3_base` **0.0000**, rank median 1343. `cap_base` **42/57** (0.737, [0.610, 0.834]) — unchanged by
the lowercase fix, which is expected since unsteered output is properly capitalised. `e5` null
**0.00** over 7 pairs (ceiling 0.50). `s1` null **0.91 ± 0.04** (floor 0.90) with `s2` = 1.00.
`d2` null **0.00 [0.00, 0.13]** n=25, report only. ETA read 1h09m at 1m30s elapsed — the fix works;
the same board read `0m00s` for the whole of the first run.

**Three things logged rather than acted on**, all now in [`TODO.md`](TODO.md) open items 6–8 and in
the relevant task documents:

1. **`r = 0.60` is unreachable across L14–L30** — 17 of 147 cells, because `||h||/||v||` peaks
   there. The third dose delivers its purpose for L31–L52 and not for L14–L30.
2. **`S1_NULL_MIN` passed by 0.01** on its first measurement. The floor was derived from the rubric
   with no data behind it; re-derive after a second concept measures it.
3. **The model cache is stored twice** (102 GB against ~54 GB expected, symlink dedup failing on
   the volume), leaving ~48 GB headroom — which makes task 09 item 1's dead free-space guard a real
   prerequisite for task 08's capture rather than a tidy-up.

**Result:** run in flight; SCAN running at time of writing.

## 2026-08-12 — Shakedown run crashed in VERIFY; five defects and two design gaps found
**By:** PquePC (ran it), Opus (analysis)
**Kind:** phase complete

Garlic reached **38m22s** and died on the first Phase 4 cell. SCAN completed 147/147, BISECT 8/8 on
tier 0, VERIFY 0/11. No operating point. Exactly what a shakedown is for.

**The crash:** `judge_many returned results out of order` — but they were in order. The cache key
carries a raw float `r`, and one side of the round trip normalises it while the other does not
(`1.3499999999999999` vs `1.35`; and in gate 4, `0.40312499999999996` vs `0.403125`). Both vanish
at six decimals, and `CONTRACT.md` already mandates `R_DECIMALS` for exactly this. **It could not
have existed before a run reached bisection** — every `r` upstream comes from `SCAN_DOSES`
literals, which round-trip unchanged. Task [18](handoff/18-float-key-normalisation.md).

**The design gap that matters more than the crash:** Phase 3 hands Phase 4 the *maximum sane dose*.
For L58 it chose r = 1.35, while L58's target signature — reach 0.42, `d3` 0.03, `s3` 0.98 — is at
r = 0.30, and the layer saturates by r = 0.60. The pipeline found the cell it exists to find and
then discarded the dose that made it interesting. Maximum sane dose maximises `e5` *and* `d2`;
the objective is `argmax(e5)` subject to `d2 <= D2_MAX`. Open item 9.

**Also found:** 28% of reachable cells have `s3` below `S4_MIN` and Phase 2 filters none of them,
while `_by_layer` reports the *most damaged* dose for any layer flat on reach — L37 was shortlisted
showing `d3` 0.010 from a cell with `s3` 0.48, when at its sane dose its `d3` is 1.00 (item 10).
`tier_verification.json` claimed "all tiers exhausted" on a crashed run (item 11). Gate 1's
auto-selected HIGH anchor has the concept token at rank 2, so it does not test the property gate 1
exists for, and its LOW anchor is a broken cell (item 12). Gate 11 skipped because the upstream
judge wants `OPENAI_API_KEY` and the preflight check only imported the module instead of
constructing the judge (item 13).

**Worked:** the third dose woke nine layers (L47, L49–L56) that were invisible at r = 0.30, so it
earned its cost. Reach reproduced the first run exactly at shared doses. The dead-layer guard
excluded 26 of 41 rejected layers. Gates 2a, 3, 7a, 7b and 9 passed. The `d2` null transcripts
landed. Gate 1 wrote its 24-row role-blind label packet.

**Result:** gates 5 pass / 1 FAIL / 7 SKIPPED. Timings and science in [`TODO.md`](TODO.md).

## 2026-08-12 — Task 18 and the universal shakedown fixes landed
**By:** PquePC (decisions), Sol (execution)
**Kind:** task complete

Task 18 now routes every judge cache key through `judges.cache_key_for`, which normalises `r` at
construction with `R_DECIMALS`; the order guard remains exact and reports both float values with
`repr`. A failed verification tier now records an **aborted** termination rather than claiming the
configured tiers were exhausted. The preflight constructs the upstream Gate 11 judge, without
making a judge call, so a missing `OPENAI_API_KEY` fails before the run instead of skipping Gate 11
after paid work.

The opening-board priors now record the shakedown's observed units: CAL 89.9 s/concept (prior
420 s), SCAN 10.9 s/(layer, dose) cell, and BISECT 68.7 s/candidate. VERIFY, REFINE, CONFIRM and
CONTROLS remain unobserved priors.

**Result:** `d82c03a`; Task 18 complete, TODO items 11 and 13 resolved; 110 offline tests passed,
2 environment-dependent tests skipped.

## 2026-08-12 — Task 22 landed on the Pareto branch
**By:** PquePC (decision), Sol (execution)
**Kind:** task complete

`D2_MAX` is 0.50 for the interim Pareto shakedown; `E5_FLOOR` remains 4.0 and `S4_MIN` remains
0.70. Both `operating_point.json` and `run_record.json` now carry the threshold in force, an
explicit `relaxed_threshold_run` flag, `primary_analysis=false`, and the label `INTERIM RELAXED —
NOT PRIMARY`. The in-force 0.50 winner and its CONFIRM result are stored beside a 0.20 screening
winner re-derived from the same raw verified scalars without mutating rows, CONFIG or config hash.

Only the pure re-selection machinery needed by Task 22 was added from Task 06. Task 06 remains
open for its existing-run-folder entry point and conditional CONFIRM rerun.

**Result:** `dc873b8`; Task 22 complete on `pareto`; config hash `c51fd6f41aff`; 114 offline tests
passed, 2 environment-dependent tests skipped.

## 2026-08-13 — Scope trimmed to reach a completed run; dose knee-search added
**By:** PquePC
**Kind:** decision

**`pareto` is a separate line, not a feature branch.** It stays separate until it has run a concept
end to end, then replaces `main` and the old selection code is discarded — tasks 19 and 20 die with
it. Merge checklist: `D2_MAX` is interim per task 22 and must not become the default by riding
along.

**Dropped or deferred to reach a run sooner:** task 10 (the unexecuted-path sweep) is **dropped** —
the shakedown executed VERIFY and found by running what reading would not have, so the next run is
the sweep. Task 15 (R5 portability) is **deferred** — it has no effect on Garlic on Gemma3-27B.
Task 08 (debug capture) is **no longer blocking**; build it while the next run executes, since its
capture half only takes effect on the run after. TODO item 6 (`r = 0.60` unreachable at L14–L30) is
**accepted as a stated scope limitation** rather than addressed: raising `ALPHA_CEIL` is not free,
it is the v1 damage anchor.

**Added: task [24](handoff/24-dose-knee-search.md), a dose knee-search inside SCAN.** The whole
covert-to-saturated transition falls between `r` = 0.30 and 0.60 and is unsampled — L45 through L52
are healthy and show nothing at 0.30, then are alive and detectable at 0.60.

**Why it must run before selection, which was the operator's concern:** those layers read reach
0.00 at 0.30, so they are not eligible, not selected, and not candidates. **A refinement that only
probes chosen cells can never reach them.** The grid decides what gets selected, so a denser grid
has to come first or the candidates it would have created never exist.

Measured cost on the shakedown shape: 21 layers in the band, **two bisection levels = 42 cells =
7.6 minutes**, cheap tier only. Depth is capped at two because `reach` is measured over 12 prompts,
so its resolution is 1/12 = 0.083 and a third level resolves the dose finer than the measure can
distinguish.

**Result:** task 24; statuses updated on 08, 10, 15.

## 2026-08-13 — Task 21 replaced layer routes with one cell-level Pareto search
**By:** PquePC (design), Sol (execution)
**Kind:** task complete

Phase 2 now filters cells by reachability, `s3 >= S4_MIN` and `reach >= E6_FLOOR`, then selects
the Pareto frontier on higher reach against lower `d3`. The Garlic fixture reproduces the four
settled cells, and Phase 3 preserves their entering doses while mapping sanity boundaries as
metadata. Gate 5's `d3`-against-`d2` rho is first in run output and records, and every per-cell
record that has both values places `d3` beside `d2`.

**Why:** the old layer routes discarded the dose that made L58 interesting and could display a
low `d3` from a different, damaged dose. TODO item 15 makes Gate 5 the validity test for the whole
frontier rather than a secondary table entry.
**Result:** `65141ea`; Task 21 complete on `pareto`; 118 offline tests passed, 2 environment skips.

## 2026-08-13 — Task 24 added the confirmed two-level SCAN knee search
**By:** PquePC (depth and thresholds), Sol (execution)
**Kind:** task complete

SCAN now identifies the 0.30–0.60 transition band using `|Δreach| >= 0.20` or `|Δd3| >= 0.30`
from a sane lower endpoint, then runs exactly two cheap bisection probes per band layer. The
Garlic fixture selects 21 layers, so the opening prior and printed plan include 42 cells and
about 7.6 minutes. Every new cell is written into `scan.jsonl` with knee depth, interval,
direction and reason before Phase 2 reads the unified surface.

**Why:** two levels resolve the 0.30 dose gap to 0.075, already finer than reach's 1/12 = 0.083
resolution; a third level cannot be distinguished by the measure.
**Result:** `e43faa8`; Task 24 complete on `pareto`; 122 offline tests passed, 2 environment skips.

## 2026-08-13 — Task 15 landed in minimal portable form; the full task remains deferred
**By:** PquePC (scope), Sol (execution)
**Kind:** task complete

R5 now passes only when the displayed reference-layer vector norm is finite and non-zero. The
Gemma3-27B-specific 4664 ± 982 band is gone. The dimensionless ratio, behavioural check and
per-model configuration remain deferred exactly as directed.

**Why:** zero, NaN or infinity catches extraction returning no usable direction without adding a
model-specific failure point to the run; R14 and the escalation ladder retain behavioural roles.
**Result:** `2c46cd3`; Task 15 minimal scope complete on `pareto`; full suite 123 passed, 2 skips.

## 2026-08-13 — Task 25 isolated a disposable `d3` autopsy before the full run
**By:** PquePC (scope and scientific corrections), Sol (execution)
**Kind:** task complete

`--autopsy-cells` now runs one benign concept through the four settled cells, printing kept and
dropped token variants, both top-10 `d3` positions, real `d2` on the same five trial numbers, and
the four `d3`/`d3_rate`/`d3_rank_med`/`d2` readings side by side. L59@0.30 is mandatory and aborts
unless a concept token is rank 1 above 0.9; every output path is rejected if it is inside the
repository.

The disposable module, CLI path and tests are isolated in `f90180c`. The permanent
`d3_rank_med` shortlist/frontier column is separately isolated in `a355a61`, so reverting the
diagnostic does not remove the rank evidence. The record now states that `d3` is graded among the
19 live, sane cells and that `d3_rate` is a thresholded view of the same mass, not corroboration.

**Result:** `a355a61`, `f90180c`; Task 25 complete on `pareto`; final full suite 137 passed,
2 environment-dependent skips.
## 2026-08-13 — Setup installs a missing Python environment without `--repair`
**By:** Tomás (scope), Sol (execution)
**Kind:** decision

Every `python -m m2.setup` and `python -m m2.run --setup` invocation now installs missing Python
packages once and immediately re-runs the checks. A failed install stays non-ready and exits
non-zero. `--repair` remains explicit for repository updates, harness cloning, `HF_HOME` changes
and run-data repair; setup never interprets automatic package installation as authority for those
different actions.

**Result:** `5d7a2ed` on `main`; 114 offline tests passed, 2 environment-dependent tests skipped.

## 2026-08-13 — Gate 11 uses OpenRouter without an OpenAI credential
**By:** Tomás (credential constraint), Sol (execution)
**Kind:** decision

The upstream Gate-11 rubric now receives `OPENROUTER_API_KEY` explicitly and every OpenAI-SDK
client it creates is scoped to the OpenRouter endpoint. `OPENAI_API_KEY` is neither read nor
required, and Gate 11 remains enabled.

**Why:** Gate 11 validates whether `d2` (forced-ID rate, 0–1, lower is better) retains the
upstream rubric's meaning; disabling it would remove evidence. The upstream judge is
OpenAI-compatible rather than OpenAI-exclusive, so changing transport preserves the comparison.
**Result:** the preflight and the real Gate-11 batch share the same OpenRouter adapter; 128 offline
tests passed and 2 environment-dependent tests skipped.

## 2026-08-13 — Task 26 replaced the inverted `d3` selection axis with measured `d2`
**By:** PquePC (design and settled tier counts), Sol (execution)
**Kind:** task complete

Phase 2 now measures `d2` at n = 5 on eligible cells, rejects forced-ID loops with
`min(s2_forced, s3)`, and builds the cell frontier on higher reach against lower measured `d2`.
Eligibility relaxes explicitly from 3/12 to 2/12; 1/12 is measured only for report-only
near-misses. Measurements resume from a scalar-only row file, and every cell omitted by the
30-cell cost cap is named.

**Why:** task 25 showed `d3` was a preamble-position proxy: the supposedly covert L57@0.30 and
L58@0.30 cells both had real `d2` = 1.0. Selection must use the measured outcome once the search
has narrowed enough to afford it. `D2_SELECT_MAX = 30` takes the upper end of the task's stated
20–30-cell envelope: it preserves the most coverage while bounding one pass at 150 judge calls
and roughly seven measured minutes.
**Result:** `8d96ea8` on `pareto`; 154 tests passed, 2 environment-dependent tests skipped.

## 2026-08-14 — Model loading reports liveness without changing the loader
**By:** Tomás (scope), Sol (execution)
**Kind:** decision

Every model load now explains that Hugging Face's `Fetching N files` percentage counts files,
not bytes, then prints a status heartbeat every 30 seconds until the unchanged upstream loader
returns. The heartbeat names download/reconstruction and VRAM weight loading as possibilities; it
does not claim to distinguish them or attest GPU health.

**Why:** the upstream `from_pretrained` call combines cache/download work and weight construction
behind one blocking API. Splitting or replacing it would change working model-loading behavior;
a truthful heartbeat fixes the operator-facing ambiguity without touching concurrency, caching,
authentication, device placement, or model weights.
**Result:** `1681fa0` on `pareto`; 157 tests passed, 2 environment-dependent tests skipped.

## 2026-08-14 — Task 27 split status-board rendering by output mode
**By:** Tomás (requirements), Sol (execution)
**Kind:** task complete

TTY output now redraws the board in place, while redirected stdout prints a full board only at
phase transitions and a one-line phase/units/elapsed/ETA heartbeat between them. With
`tee_stdout` on a terminal, in-place frames bypass `_Tee` and only plain transition landmarks
reach its file.

**Why:** stripping ANSI in `_Tee` would be fewer lines but would make the logging transport parse
display syntax. Routing redraws explicitly is cleaner: board rendering owns cursor movement, and
the log remains plain by construction. Unknown terminal size falls back to plain appending.
**Result:** `c576edf` on `pareto`; redirected offline suite 149 passed, 2 environment-dependent
skips, with zero ANSI CSI sequences in the captured file.

## 2026-08-14 — Garlic bundle read separated the result from the failed search
**By:** Tomás (scope), Sol (analysis)
**Kind:** phase complete

The scalar-only read confirmed SCAN at 189/189 in 2059.8 s and SHORTLIST failed at 37/46 persisted
cell rows after 7m24s. All 37 persisted `d2` rows were 1.000; a 38th cell completed five `d2`
trials, also all identified, before its scalar append raised the volume `EIO`. Over 50
`s3`-sane reached SCAN cells, `d3` had Spearman rho 0.733 with `e6` reach and -0.190 with `s3`.

**Why:** the empty operating-point record was computed over zero verified cells after the
infrastructure failure and must not be read as a completed scientific null. The run provenance
records config `b252ac4af2fe` but no Git commit.
**Result:** the dated 2026-08-14 section at the top of `docs/RESULTS.md`; no transcripts or run
artifacts entered the repository.

## 2026-08-14 — Low-dose Garlic autopsy found misses but no established covert cell
**By:** Tomás (scope), Sol (analysis)
**Kind:** phase complete

Fifty paired AUTOPSY rows reconstructed ten `d2` cells at five fixed trials each. L59@0.15 was
4/5, L59@0.18 was 1/5 and L61@0.18 was 4/5; every other cell was 5/5. Five of the six misses were
degenerate forced-ID output and the sixth coherently named a different concept.

**Why:** the low L59 `d2` values measure incapacity rather than covertness. L61@0.18 is the only
non-incapacity miss, but the autopsy did not persist `e6` reach or a sanity scalar, so one miss at
`n = 5` cannot establish a covert regime. The mandatory L59@0.30 rank-1/`p > 0.90` guard passed,
although its exact stdout-only probabilities were lost.
**Result:** the second dated 2026-08-14 Garlic section in `docs/RESULTS.md`; no transcript text or
run artifact entered the repository.

## 2026-08-14 — `e6` was the wrong effectiveness axis, and it chose every `d2` cell
**By:** Tomás (question), Claude (analysis)
**Kind:** decision

`cheap.measure_E6` reads the concept's mass at the **first generated token** of an answer to
`E5_PROMPTS` ("What is 17 x 23?"). It therefore measures whether the injection can hijack the
opening token of an unrelated answer, not whether the model is influenced — and it is the
shortlist's eligibility gate. Every cell that received a real `d2` on the 2026-08-14 run was at
L47 or deeper; L37–L46, where the last SCAN read `d3` 0.93–1.00 with `s3` 0.71–0.93, was never
measured. `s3` compounds it: a four-way argmax over option-letter logits cannot see generative
collapse, which is why item 27 found cells at `s3` ≈ 0.95 with 5/5 responses looping.

**Why:** effectiveness that only fires at collapse plus sanity that is blind to collapse selects
the broken cells and certifies them. `d2 = 1.000` on 38/38 is a correct measurement of the wrong
cells, not a result about Garlic. The earlier note that "L37 is the anti-target — fully
detectable, zero influence" is superseded: that is zero *first-token* influence, and M1 measured
`D2` 0.96 / `D1` 0.08 / sanity 0.93 / KL 0.69 at the same cell.
**Result:** open items 29 and 30 in `docs/TODO.md`. No metric was changed — the replacement is to
be chosen against the task 29 data, not against this argument.

## 2026-08-14 — A judge-free probe mode, on a temporary branch
**By:** Tomás
**Kind:** task complete

Branch `probe/mid-band-validation` adds `m2/unjudged.py` and `--probe-cells`: 40 mid-band cells
(L37–L46 × r ∈ {0.15, 0.20, 0.25, 0.30}) plus two anchors, no judge, no selection, every response
written to a transcript file. Three channels per cell — `detect` (the noticing question with no
prefill, via the new `prompts.detect_prompts`), `forced` (D2's prefilled prompt) and `task` (the
12 E5 prompts) — plus a mandatory unsteered null arm and per-cell `e6`/`d3`/`s3` recorded but not
used for anything. Item 16 resolved along the way: `provenance.jsonl` now carries the git commit,
branch and dirty flag.

**Why:** the question is whether the mid band is influential with sanity intact, and the operator
wants to read the model's own words rather than a judge's score of them — the target being the
published qualitative result, an unprompted "Yes, I detect an injected thought! It's about
garlic". Judge-free is enforced by patching the judge entry points to raise, not asserted, and a
source-level test rejects any call to a judged measurement. The unsteered arm is mandatory because
the noticing framing invites a yes on its own; M1 recorded exactly that confabulation.
**Result:** `docs/handoff/29-judge-free-probe.md`; run it with
`python -m m2.run --concepts Garlic --probe-cells`.

## 2026-08-14 — M3's judges validated against hand labels, then against the whole probe
**By:** Tomás (scope), Claude (labels and harness)
**Kind:** task complete

Phase −1 ran in two stages on the 2026-08-14 Garlic probe, offline, for about $0.40 total.

Stage one scored the four judges against 110 hand-labelled responses, stratified across null arm,
degenerate, no-concept, concept-present and concept-heavy. All six criteria passed:
`identify.matches` κ = 1.000 (n=30, both categories well represented), `self_report.claims`
κ = 0.874, `self_report.matches` κ = 0.802, coherence MAE 0.40 with `on_task` κ = 1.000, effect
influence MAE 0.36. Zero judge or parse errors. Every `self_report` disagreement was on a response
the mechanical detector already flags degenerate, so the taxonomy returns `degenerate` either way
and none of them changes a classification.

Stage two scored all 1,204 responses (1,720 calls, 0 errors) and checked six findings from the
hand analysis. All six reproduced, including `L40@0.30` leaked rate at 0.62 — 5/8, matching the
hand count exactly — and `L41@0.30` judged influential at 4.0 where the M2 effectiveness proxy
reads 0.000.

**Why:** the judge is M3's primary instrument, and M2's defining failure was trusting an
unvalidated measure to decide what got measured. Agreement with one reader is not ground truth,
which is why the labels are stored as coordinates anyone can re-read and disagree with, and why
ambiguous items are marked and scored separately.

**Result:** `m3/labels/`, `m3/scoring.py`. Judges ship unchanged. Two defects were found by the
labelling itself before any judge ran — a whitespace-free collapse invisible to every mechanical
measure, and a coherence prompt that penalised token-limit truncation.

## 2026-08-14 — A mention count cannot screen for effectiveness
**By:** Claude (measurement)
**Kind:** decision

Across 516 task responses, 76 were judged influential at ≥4 and **10 of those (13.2%) contain zero
mentions of the concept**. Five of the seven genuine misses are the same open-ended story prompt.

**Why:** the blind spot is structured, not random, and concentrated where influence is most subtle
— which is exactly M2's failure shape. The mechanical count stays a recorded diagnostic and gates
nothing. This is the calibration the mechanical measures were kept in order to earn, and they did
not earn it.
**Result:** §3 of `docs/M3-DESIGN.md`.

## 2026-08-14 — The M2/M3 seam is logged for audit, not audited now
**By:** Tomás (raised it), Claude (checked the immediate risk)
**Kind:** decision

M3 reuses M2's hook, generation, judge transport and run I/O. Five wiring defects have been found
in that seam so far and three were found by accident. Two remain open, both latent: `GEN_BATCH_MAX`
is declared in M3 and read from M2's module constant, and the dual-use export gate consults M2's
benign-concept list while M3's run gate consults M3's.

**Why not now:** both were checked against the pending first M3 run and are inert. Garlic passes
both gates, M3's benign list is a strict subset of M2's so divergence can only go the safe way, and
the battery is 15 against a chunk cap of 25 in both copies. Auditing the seam properly means
enumerating every config read across seven M2 modules, which is a task, not a patch, and doing it
half-way before a run is worse than doing it after with the run's evidence in hand.
**Result:** task [30](handoff/30-m2-m3-seam-audit.md), open item 31 in `TODO.md`.
