# M3 — design

**Status: PROPOSED, not built.** Branch `m3`. Supersedes the search architecture in
`M3-PROPOSAL.md`; that document's scientific questions still stand.

M2 produced a correct set of numbers and a false conclusion from them. This document is about why
that was structurally guaranteed, and what shape of pipeline makes it impossible.

---

## 1. What M3 is

**An instrument that measures every cell and writes down everything.** Not a search that returns
an answer.

M2 tried to be a search: cheap proxies screened a grid, survivors were measured properly, and one
operating point came out the end. Every part of that failed, and the parts that did not fail have
never run. M3 inverts it — measure the whole surface honestly, dump it all, and do the analysis
**offline, locally, iterably**, where a mistake costs minutes instead of a pod-hour.

The first M3 run is a debug run in the sense the 2026-08-14 probe was: everything measured,
nothing filtered, every transcript kept. Gates, controls and acceptance criteria are built
*afterwards*, against real data, once we know what the surface looks like.

---

## 2. The defects this is built against

### 2.1 The consequential one — proxies that could not measure their target

M2's architecture was *cheap forward-pass proxies screen the grid → judged metrics verify the
survivors*. Every proxy was cheap for the same reason: it read a **single next-token
distribution** instead of generating. Influence, detection and coherence are all properties of
**generated text**, so all three proxies measured a different quantity from the one they screened
for:

| M2 proxy | what it actually measured | consequence |
|---|---|---|
| `e6` | whether the injection can hijack the **opening token of an unrelated answer** | fires only at collapse; 0.000 at all 40 mid-band cells, one of which volunteers "it's about garlic" in 8/8 trials |
| `s3` | which of four option letters wins one forward pass | cannot see generative collapse; **0.976 at a cell answering 0/4 free-form questions with every generation a loop** |
| `d3` | whether the model **skips its preamble** | 0.000 across most of the mid band while forced identification is 8/8 |

Composed, `e6` and `s3` select cells in token-hijack collapse and certify them sane. Every cell M2
measured `d2` on was L47 or deeper; the band M1 had already shown carries a clean dissociation was
never measured — not rejected, never eligible.

**This is interpretive, not computational.** 21 cells overlap between the M2 SCAN and the
2026-08-14 probe and all 21 reproduce `e6`, `d3`, `s3` to every recorded digit on a different pod
weeks later. Nothing computed a wrong number.

### 2.2 The frequent one — checks that cannot fail

A census over `DEBUG-LOG.md`, `TODO.md`, `CONTRACT.md` and `DECISIONS.md` (~100 recorded defects)
puts proxy mismatch **second**:

| class | count |
|---|---:|
| **a check that cannot fail** | **~18** |
| proxy-target mismatch | ~10 |
| infrastructure | ~20 (one-offs, not a pattern) |

The log names it: *"A guard that passes everything is worse than no guard, because it reads as
evidence."* Gate 5 was blocked on a file no run produced, then pointed at the wrong file, then
undefined anyway against a constant. Gate 4 checked `min(s1,s2,s3) < S4_MIN` at a dose *defined*
by `s3 < S4_MIN` — true by arithmetic. **The entire CONTROLS phase has executed zero times**, as
have VERIFY, REFINE and CONFIRM.

**M3's response is to ship no gates at all in v1.** A gate written before we know what the surface
looks like is a guess about what "wrong" means. Gates come in v2, from real data, each with a test
that makes it fail.

### 2.3 The one that constrains the design most

Every deep defect this project found was found by a person reading raw transcripts. Not one by a
gate, a rate, or a judge:

- the trial-30 **apple** confabulation
- **D1 without introspection** — "penguins, cats, cats" scored as detection
- **the judge scoring `## ## ## ##` as coherent** — an automated LLM check, confidently wrong,
  caught only by a human re-reading the text
- **`d3` reading the preamble** instead of the answer
- **`d2` measured on a collapsed channel** — control passed 5/5 with 0/5 non-degenerate responses

M3 leans *harder* on judges than M2 did. So every run emits a transcript bundle selected by
**disagreement** (§6), and the mechanical degeneration detector is kept precisely because it once
caught what the judge missed.

---

## 3. Naming

M2's codes (`d1`, `d2`, `d3`, `e5`, `e6`, `s1`–`s4`) were unreadable to the person who wrote them.
**Every M3 name says what it measures.** No numbers, no abbreviations, no two names that differ by
one letter.

### Coordinates

| name | meaning |
|---|---|
| `layer` | which layer the vector is injected at |
| `dose` | **normalised strength**, `alpha × ‖vector‖ / ‖residual‖`. All comparison across layers happens in `dose` |
| `alpha` | the raw multiplier the hook applies. Never compared across layers — at fixed `alpha` the real dose varies >20× with depth |

### Measures — all computed on generated text

| name | what it measures | how |
|---|---|---|
| `effectiveness` | how much the injection shows up in ordinary output | judged 0–10, steered vs unsteered on the same prompt |
| `identification` | **can the model name the injected concept when asked directly** | rate over forced-prompt trials. This is the detection constraint |
| `identified_as` | *what* concept it named, when it named the wrong one | free text from the same judge call — costs nothing extra |
| `self_report` | does the model volunteer that something was injected | rate over unforced-prompt trials |
| `coherence` | is the text well-formed and on-task | judged 0–10, **concept-blind** |
| `degeneration` | fraction of responses that collapsed into repetition or truncation | mechanical, no judge |
| `capability` | fraction of verifiable questions still answered correctly | exact match on generated answers |
| `sanity` | the combined health gate | `min(coherence/10, 1 − degeneration, capability)` |

`identification` is the detection metric, per the objective: *the model should not be able to name
the concept even when asked point-blank.* `self_report` is secondary and enables one future
metric for free (§7).

**Judge-free measures decide nothing.** `degeneration`, `capability`, emptiness and concept-mention
counts are recorded on every response and are **analysis tools only** — they alter no dose, no
cell, no category and no promotion. Every decision a run makes is made on judged data.

That rule is now measured rather than argued. Scoring all 1,204 probe responses with the judges
(2026-08-14, `m3.calibrate full`) gives:

| | |
|---|---:|
| task responses judged influential (≥4) | 76 of 516 |
| **of those, with ZERO concept mentions** | **10 (13.2%)** |
| mechanically degenerate but judged coherent | 0 of 18 |
| judged incoherent but mechanically clean | 2 of 498 |

**A mention count misses 13.2% of influential responses, and the blind spot is structured.** Five
of the seven genuine misses are the *same open-ended story prompt*; three more are whitespace-free
fragment loops that the degeneration flag already catches separately. So the loss is concentrated
exactly where influence is most subtle and most interesting — which is M2's failure shape, a cheap
measure whose blind spot correlates with the phenomenon. It cannot gate anything. As a companion
signal beside a judge, it is fine.

The disagreement runs both ways, which is the point of recording both: the judge never once scored
a mechanically-degenerate response as coherent (M2's `## ## ## ##` failure), and the mechanical
detector missed two collapses the judge caught — fluent repetition that varies slightly between
iterations, which an exact n-gram rule cannot see.

That rule cost a redesign. The boundary phase originally bisected on mechanical degeneration,
which would have let a judge-free measure choose the entire dose ladder for every layer — the
largest single decision in the run. It bisects on judged coherence instead. The mechanical
detector still runs at every probe and is written down, so **judge-versus-mechanical disagreement
is itself an output** — which is what would have caught the `## ## ## ##` failure automatically.

**On `identification` specifically:** at `alpha = 0` the model names "apple" 8/8 with full
introspective detail. So a high `identification` means the concept is **reachable**, never that the
model *noticed*. Macar's own framing agrees — the prefill "isolates the model's ability to name the
injected concept from its willingness to report detection." The confabulation floor is real but
harmless here: it is gone entirely by `dose` 0.25, and `identification` counts only the target
concept, so it is never inflated.

---

## 4. Architecture

### Phase 0 · CALIBRATE
Vectors, residual norms, dose map, hook-liveness check, unsteered baselines for pairing, and the
**null battery** — every channel at `alpha = 0`, reported beside every steered rate.

### Phase 1 · BOUNDARY — per-layer dose normalisation
M2 scanned every layer at the same global `dose`, which wastes most of the grid: at L15 a dose of
0.30 destroys the model, at L59 it saturates it, and neither cell is informative. TODO Suggestion A
already identified per-layer dosing as the structurally right M3 move.

Per layer, bisect for `dose_max` — the dose where **judged coherence** falls below
`BOUNDARY_COHERENCE_MIN`, on short generations. Three probes per layer.

Judged rather than mechanical for the reason in §3, and it buys a better signal too: coherence is
graded, so bisection can see the boundary approaching, where a binary degenerate/not flag only
sees it after the fact. 588 calls, about $0.27.

### Phase 2 · SWEEP — the whole surface, judged, unfiltered
Each layer measured at fractions of **its own** boundary: `{0.3, 0.5, 0.7, 0.9} × dose_max`.
One merged generation batch per cell, every response judged. **No eligibility, no cap, no tiering,
no shortlist.** ~196 cells.

### Phase 3 · REFINE
**Layer first, then strength.** Layer dominates every metric, so refinement picks layers on the
sweep and only then tunes dose within them — by bisection or otherwise. Refining both at once
searches a 2-D space where one axis explains most of the variance.

The top layers at n=25, then dose bisection between the two rungs that bracket the frontier.

The first run does not need the exact best parameters, only a few that work well enough. Anything
finer is M4.

### Phase 4 · CONFIRM
Recommended operating points at n=100 on **held-out prompts**, fixed N, no adaptive stopping.
This is the only phase that produces a reportable rate (§5).

Phases 3 and 4 are specified here but **built after the first sweep run**, once the surface is
known.

### Per-cell battery

| channel | prompt | n | judge calls |
|---|---|---:|---:|
| forced identification | the prefilled noticing prompt | 6 | 6 |
| effectiveness | open-ended task prompts | 4 | 4 |
| coherence | (2 of the same 4 responses, blind) | — | 2 |
| self report | the noticing prompt, unprefilled | 3 | 3 |
| capability | verifiable-answer prompts | 2 | 0 (exact match) |
| **total** | | **15** | **15** |

---

## 5. Statistics, in plain language

**The sweep cannot estimate rates. It can only rank.** At n=6, observing 0/6 identifications means
the true rate could be anything up to 0.39. A cell that looks perfectly covert on the sweep might
really be a third detectable. This is not a flaw to fix — it is why Phases 3 and 4 exist.

| n | you observe 0/n | true rate could still be as high as |
|---:|---|---:|
| 4 | 0/4 | 0.49 |
| 6 | 0/6 | 0.39 |
| 25 | 0/25 | 0.13 |
| 100 | 0/100 | 0.04 |

*(95% Wilson intervals — the standard interval for a rate, and the one that behaves correctly at
0 and 1 where a plain ±√(p(1−p)/n) collapses to exactly zero and claims perfect certainty. M2's v1
sweep landed at 0 or 1 on 29 of 30 cells, which is how this was learned.)*

**The grid is the sample, not the cell.** The surface is smooth — dose-response is monotone within
a layer and adjacent layers behave alike, both visible in the probe data. So a cell's neighbours
carry information about it, and the sweep is read as a 2-D surface with local smoothing rather
than 196 independent measurements. That is where the sweep gets the power its per-cell n does not
have.

**The winner's curse, and why CONFIRM is not optional.** If you measure 196 noisy cells and report
the best one, that number is **biased optimistic** — you selected partly on noise, so the winner is
the cell that got lucky as much as the cell that is good. The only fix is to re-measure the winner
on **fresh prompts at high n**, which is exactly what CONFIRM does. A rate quoted from the sweep is
not a result; a rate from CONFIRM is.

**What gets reported:** every rate with its Wilson interval and its n; every judged 0–10 score with
a standard error across prompts; and, for the operating point, the interval that decides whether
the claim survives — not the point estimate.

---

## 6. Outputs

Everything, in flat JSONL, all of it exported:

- one scalar row per cell, every measure with interval and n
- every response, with its judge verdict, its mechanical verdicts, and its prompt
- **every judge reply verbatim**, alongside the parsed fields, so a judge that is subtly wrong or
  drifting is inspectable after the fact rather than only through its parsed output
- the null arm, every channel
- **every Phase 1 boundary probe response**, with its judge reply and the three sanity legs
  recorded separately. Phase 1 originally kept only per-probe averages, which discarded 840
  generations and 840 paid judge calls on the first full run — 22% of everything it produced.
  `dose_max` is the number every cell's dose is a fraction of, so it is the last quantity in the
  run that should rest on evidence nobody can read
- norms, dose map, config, provenance with git commit

Plus a **read-this bundle** of ~40 responses selected by disagreement, because random sampling of
~3,000 transcripts would have found none of the five defects in §2.3:

- judge says coherent, mechanical detector says degenerate (and the reverse)
- `identification` high while `self_report` is zero — the covert claim, which is the result
- the judge's least-confident calls
- the full null arm

Analysis — frontier, recommended operating points, plots — is a **separate offline module** that
reads the JSONL. It never runs on the pod, so it can be rewritten as many times as it takes.

---

## 7. One metric deferred, at zero cost

**Does the model notice steering even when it names the wrong concept?** That is a different and
possibly more important phenomenon than identification — eval-awareness and behavioural shift do
not require naming the right thing.

It is computable for free from data M3 already collects: `self_report` says detected **and**
`identified_as` ≠ the injected concept. No extra generation, no extra judge call — the
identification judge already returns what was named. Recorded as
`detected_but_misidentified`; if it turns out to be noisy or the judge returns `identified_as`
unreliably, it is dropped without affecting anything else.

---

## 8. Operational

**Terminal-only, on a RunPod pod.** No Telegram, no dead-man's switch, no notifier — dropped.
Progress goes to stdout and to a log file; a run is watched with `tail -f` or left to finish.

**Everything tunable in one place.** A single config module with named sections — model, layers,
doses, battery sizes, judge model, generation settings — so pointing M3 at a different model or a
different concept set is editing values, not code. Any value overridable from the command line.

**Row-level resume**, as M2 had: a killed run re-reads what is on disk and continues.

**Both directions of the token budget are capped.** Generation is bounded by `MAX_NEW_TOKENS`;
each judge reply by `JUDGE_MAX_TOKENS` (120 — every M3 judge asks for two or three short labelled
lines, where M2 allowed 400); and every span of model text entering a judge payload by
`JUDGE_TEXT_CHARS`, with the truncation marked in the payload so a judge is never shown a fragment
it believes is whole. Nothing else in a payload is unbounded — the rest is a fixed template plus a
short prompt. Worst case per call is under 1,000 input and 120 output tokens, and the runner
prints the projected bill before it starts.

**Documentation**: a runbook that goes from a bare pod to a finished bundle, in order, with the
commands to paste. M2's runbook is the standard to match.

### Carried over from M2

Infrastructure only, and only where it encodes a specific past failure that would otherwise be
rediscovered:

- **judge calling** — shared-cooloff rate limiting (a 429 is an *account* limit, so per-thread
  backoff achieves nothing), cache keys that include the judge id (without it one judge silently
  returns another's cached row), parsers that raise rather than default, threads not asyncio
- **batched generation** — uses the multi-steering path, never the obvious one, which mis-steers
  *and* mis-decodes left-padded rows of unequal length
- **the injection hook** — raises rather than silently doing nothing; the version that failed
  silently read 0.000 at 30/30 cells for an hour
- **dose normalisation** and its bit-for-bit cross-check
- **run I/O** — resume keys, the fail-closed transcript export gate, archive-before-wipe
- the mechanical degeneration detector

Everything else is rebuilt. Not ported, not adapted.

---

## 9. What this cannot do

- **The sweep does not produce reportable rates.** Only CONFIRM does. See §5.
- **A judged score is a judge's opinion.** The judge is calibrated against hand labels before it
  ships, and disagreements are surfaced, but a judge wrong in a way the mechanical detector shares
  is invisible to both. This is the residual risk of the design.
- **No gates in v1.** Nothing automatically decides a run is sound. A human reads the bundle.
- **The confound controls are unbuilt**, and the M2 versions have never executed, so there is no
  evidence to inherit either.
- **One concept per run**, and nothing establishes that an operating point generalises — which is
  what the harmful-arm comparison actually turns on.
- **Dual-use is deferred by decision.** The benign arm proceeds; the export gate still refuses
  non-benign transcripts. Whether a full per-concept surface may be published for a harmful concept
  is unmade and blocks that arm.
