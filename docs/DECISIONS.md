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
