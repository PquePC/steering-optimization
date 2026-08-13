# Handoff — coding agent task queue

> **Branch `pareto` is a separate line, not a feature branch.** Tasks 19 and 20 are SUPERSEDED here
> by task 21, and `D2_MAX` is relaxed to 0.50 per task 22. **It stays separate until it has run a
> concept end to end.** At that point it replaces `main` and the old selection code is discarded —
> so 19 and 20 die with it. **Merge checklist:** revisit `D2_MAX`, which is interim per task 22 and
> must not become the default by riding along.

Start with [`BRIEF.md`](BRIEF.md) for standing context, then work the numbered tasks in order.
Each has its own acceptance criteria. Nothing here overrides [`../../CLAUDE.md`](../../CLAUDE.md).

## 🛑 Status vocabulary — read this first

Every task document opens with a status line. **Implement nothing whose status does not begin with
`BUILD NOW`.** The other statuses are ideas that have been captured deliberately so they are not
lost, and building them early costs review time and muddies the run.

| Status | What it means |
|---|---|
| **BUILD NOW** | Decided. Implement as written. |
| **BUILD NOW — PROPOSE FIRST** | Implement, but the named design choice is escalated to the operator before you write code. Do not settle it in a commit. |
| **DIAGNOSTIC ONLY** | Build and run the measurement. **Do not build the fix** until the operator has read what the measurement says. |
| **DEFERRED — DO NOT BUILD** | Considered and set aside. The reasoning is kept because it may come back. |
| **FUTURE — DO NOT BUILD** | Captured, not scheduled. Not for this run. |
| **RUN** | The run itself; everything marked BUILD NOW lands first. |

The same rule applies to [`../TODO.md`](../TODO.md): its *Decided* table is work, its **Suggestions**
section is not. Nothing under Suggestions gets built without being promoted to a task document
first.

If you think something marked DEFERRED or FUTURE should be built now, **say so and give the
reason** — that is a useful contribution. Building it is not.

| # | Task | Why it exists | Blocking? |
|---|---|---|---|
| [01](01-scan-doses.md) | Add a third scan dose | The mid band is undecidable between inert and under-dosed, and that is where the answer is predicted to be | **yes — before any measurement** |
| [02](02-judge-null-controls.md) | A judge null control per judged measure | Catch a bad judge or a bad prompt before a phase spends its calls, not after | yes |
| [03](03-gate4-reanchor.md) | Gate 4 on this run's own damaged cell | `s4` is the only thing separating "covert" from "lobotomised" | yes |
| [04](04-gate1-anchors.md) | Gate 1 anchors from this run; make it an instrument gate | `e5` must read influence, not count words | yes |
| [05](05-gate6-false-negative-audit.md) | Tiered shortlist: escalation **and** false-negative audit | The only evidence the shortlist does not drop the answer — and it can improve the answer | yes |
| [06](06-relaxed-reselection.md) | Relaxed-threshold re-selection | Re-read a finished run at a wider `D2_MAX` without re-measuring | no |
| [07](07-reference-cell.md) | ~~Reference cell~~ | **DEFERRED** — direct comparison against Macar is not this run's framing | no |
| [08](08-debug-bundle.md) | Debug **capture** then export | Nothing writes to `debug/` today, so the filter is not the blocker — the intermediates must be persisted first | no — **build during the next run** |
| [09](09-known-unfixed.md) | Three known-unfixed defects | Carried over from the first run | no |
| [10](10-unexecuted-path-sweep.md) | ~~Sweep the never-executed phases~~ | **DROPPED** — the shakedown executed VERIFY and found by running what reading would not have. The next run is the sweep | no |
| [11](11-the-run.md) | The run itself | | last |
| [12](12-mmlu-letter-surface-forms.md) | `s3` scores only uppercase option letters | A degraded model answering `c` is scored wrong — a dose-dependent bias in a sanity term | **yes — it is a defect** |
| [13](13-prefix-token-contamination.md) | Prefix-token contamination in `e6` / `d3` | If `Garlic` tokenizes to `gar`, the scan surface is also counting *garden* | diagnostic before the scan is trusted |
| [14](14-mmlu-by-generation.md) | Verify `s3` by generation | **FUTURE** — `s3` cannot tell "answers C" from "wants to emit noise, C is highest among letters" | no |
| [15](15-r5-portability.md) | ~~R5's norm band~~ | **DEFERRED** — no effect on Garlic on Gemma3-27B; future-model portability | no |
| [16](16-judge-bakeoff.md) | Judge bake-off on stored transcripts | **FUTURE** — replay judging with no GPU and score candidates on the gates already built | no |
| [18](18-float-key-normalisation.md) | Judge cache key is not float-normalised | **The shakedown's crash.** A bisected `r` round-trips to a different float and the order guard raises; kills every run at Phase 4 | **yes — blocker** |
| [19](19-phase3-dose-selection.md) | ~~Two doses per layer~~ | **SUPERSEDED** by 21 | no |
| [20](20-phase2-sanity-filter.md) | ~~Phase 2 sanity filter~~ | **SUPERSEDED** by 21 | no |
| [21](21-unified-cell-selection.md) | Unified cell-level Pareto selection | Replaces three routes, the padding, the fit and the dose decision with one rule over `(layer, dose)` cells | **yes, on this branch** |
| [22](22-relaxed-detection-ceiling.md) | Interim `D2_MAX = 0.50` | CONFIRM and CONTROLS have never run; they need a qualifying cell to exist | yes |
| [23](23-behavioural-spot-check.md) | Sit with the model at the operating point | `s3` is 57 multiple-choice letters, `s2`/`s1` are 12 short answers. Nothing covers refusal, long-form, multi-turn or instruction-following | yes, after the first completed run |
| [24](24-dose-knee-search.md) | Bisect the dose gap during SCAN | The whole covert-to-saturated transition falls between `r`=0.30 and 0.60 and is unsampled; the layers it would reveal are not candidates, so it must run before selection | **yes** |
| [17](17-wilson-intervals.md) | Wilson intervals on every rate, every concept | A binomial SE is exactly zero at p=0 and p=1, and 29 of 30 v1 cells landed there | yes, before the run |
| [25](25-d3-autopsy.md) | `d3` autopsy on four cells | The frontier ranks on `d3`, which nothing has validated; at both candidate cells the concept is the **rank-2 token** and nothing stores what rank 1 is. Disposable by design | **yes — before task 11** |

## What you owe the documentation

The tree is owned by the orchestrating agent — **do not restructure, merge, rename, move or delete
any markdown file.** What you should keep writing:

- **[`../DECISIONS.md`](../DECISIONS.md)** — **the one that matters most.** Every decision, every
  completed task, every completed run phase. This is how the orchestrator and the operator find out
  what happened while they were not in the session. A decision the operator makes in conversation
  with you is the kind most likely to be lost, because it never passed through a file — write those
  down first.
- **Your task's own status line** — set it to `DONE` with the commit hash when the task is
  complete. That line is the single source of truth for state; `DECISIONS.md` is the history.
- **[`../TODO.md`](../TODO.md)** — append open items as you find them; move items you fix into the
  register with the commit hash. Append, do not reorganise.
- **[`../DEBUG-LOG.md`](../DEBUG-LOG.md)** — every new defect, in the existing format: symptom,
  root cause, **why nothing caught it**, the fix, the test that now trips on it. That third field
  is what the log exists for and the one most often skipped.
- **Measured timings** into `../TODO.md`'s table, especially CONFIRM and CONTROLS, which have
  never been observed.
- **Docstrings** to the house style — explain *why* a value is what it is, beside the value.

If a documentation commit lands while you are working, rebase onto it rather than reverting it.
