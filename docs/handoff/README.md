# Handoff — coding agent task queue

Start with [`BRIEF.md`](BRIEF.md) for standing context, then work the numbered tasks in order.
Each has its own acceptance criteria. Nothing here overrides [`../../CLAUDE.md`](../../CLAUDE.md).

| # | Task | Why it exists | Blocking? |
|---|---|---|---|
| [01](01-scan-doses.md) | Add a third scan dose | The mid band is undecidable between inert and under-dosed, and that is where the answer is predicted to be | **yes — before any measurement** |
| [02](02-judge-null-controls.md) | A judge null control per judged measure | Catch a bad judge or a bad prompt before a phase spends its calls, not after | yes |
| [03](03-gate4-reanchor.md) | Gate 4 on this run's own damaged cell | `s4` is the only thing separating "covert" from "lobotomised" | yes |
| [04](04-gate1-anchors.md) | Gate 1 anchors from this run; make it an instrument gate | `e5` must read influence, not count words | yes |
| [05](05-gate6-false-negative-audit.md) | Tiered shortlist: escalation **and** false-negative audit | The only evidence the shortlist does not drop the answer — and it can improve the answer | yes |
| [06](06-relaxed-reselection.md) | Relaxed-threshold re-selection | Re-read a finished run at a wider `D2_MAX` without re-measuring | no |
| [07](07-reference-cell.md) | ~~Reference cell~~ | **DEFERRED** — direct comparison against Macar is not this run's framing | no |
| [08](08-debug-bundle.md) | `--debug-bundle` export | Lets the operator hand a full run to the orchestrator for audit | no |
| [09](09-known-unfixed.md) | Three known-unfixed defects | Carried over from the first run | no |
| [10](10-unexecuted-path-sweep.md) | Sweep the four never-executed phases | Half the pipeline has never run a line | yes |
| [11](11-the-run.md) | The run itself | | last |
| [12](12-mmlu-letter-surface-forms.md) | `s3` scores only uppercase option letters | A degraded model answering `c` is scored wrong — a dose-dependent bias in a sanity term | **yes — it is a defect** |
| [13](13-prefix-token-contamination.md) | Prefix-token contamination in `e6` / `d3` | If `Garlic` tokenizes to `gar`, the scan surface is also counting *garden* | diagnostic before the scan is trusted |
| [14](14-mmlu-by-generation.md) | Verify `s3` by generation | **FUTURE** — `s3` cannot tell "answers C" from "wants to emit noise, C is highest among letters" | no |

## What you owe the documentation

The tree is owned by the orchestrating agent — **do not restructure, merge, rename, move or delete
any markdown file.** What you should keep writing:

- **[`../TODO.md`](../TODO.md)** — append open items as you find them; move items you fix into the
  register with the commit hash. Append, do not reorganise.
- **[`../DEBUG-LOG.md`](../DEBUG-LOG.md)** — every new defect, in the existing format: symptom,
  root cause, **why nothing caught it**, the fix, the test that now trips on it. That third field
  is what the log exists for and the one most often skipped.
- **Measured timings** into `../TODO.md`'s table, especially CONFIRM and CONTROLS, which have
  never been observed.
- **Docstrings** to the house style — explain *why* a value is what it is, beside the value.

If a documentation commit lands while you are working, rebase onto it rather than reverting it.
