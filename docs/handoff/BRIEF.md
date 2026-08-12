# Brief — a publishable Garlic result

**Standing context for the coding agent.** Read this once, then work from the numbered task
documents in this folder. Updated whenever a design or experiment choice is made — check the
change log at the bottom before starting a session.

---

## 1. Your role, and the boundary

The operator directs this project. An orchestrating agent (Claude) owns research and design
direction and the documentation tree. **You own the code, the tests, the execution of the run and
the diagnosis of failures.**

**Escalate design, do not encode it.** If a task requires choosing a threshold, changing what a
measure means, or deciding a gate's absence is tolerable, state the options and the trade-off
rather than settling it in a commit. Implementation choices are entirely yours.

You do **not** have pod access. You fix code, push, and then **guide the operator through the run**:
copy-paste-ready commands one per fenced block, working directory stated. The operator pastes log
output back and you diagnose from it. **Never report a phase as passed, a gate as green, or a
number as measured unless you have seen the line that says so.**

Conventions, git rules and the ownership table: [`../../AGENTS.md`](../../AGENTS.md).
Safety rules, which are not optional: [`../../CLAUDE.md`](../../CLAUDE.md).

## 2. Read before touching anything

1. [`../../CLAUDE.md`](../../CLAUDE.md) — dual-use rules. Hard constraints.
2. [`../../README.md`](../../README.md) — orientation, the measure codes, the phases.
3. [`../TODO.md`](../TODO.md) — what is decided-but-unbuilt, and what is still open.
4. [`../SPECIFICATION.md`](../SPECIFICATION.md) — what each measure means. The authority on science.
5. [`../CONTRACT.md`](../CONTRACT.md) — module layout and naming.
6. [`../DEBUG-LOG.md`](../DEBUG-LOG.md) §4 (bug register) and §6 (patterns) — every bug there is
   one this pipeline must be structurally unable to reproduce.

## 3. Where things stand

One end-to-end Garlic run has happened (2026-08-11, A100-80GB). It completed Phases 0–3 and **died
at the first cell of Phase 4**. Thirteen defects were found and fixed; all are on `main` and the
register is in [`../DEBUG-LOG.md`](../DEBUG-LOG.md). **No run has ever reached VERIFY, REFINE,
CONFIRM or CONTROLS.** Consequently:

- **Gate 5** — Spearman ρ of `d3` against real `d2` — has never produced a number. Below
  `D3_MIN_RHO` the residual ranking that put L58 on the shortlist is unreliable and Phase 2's
  route 3 is measuring noise. It runs the moment Phase 4 completes; it is the first number to look
  at.
- Four phases have never executed a line, so their cost priors are guesses.
- No `operating_point.json` exists for any concept.

The Phase 0–3 science is recorded in [`../TODO.md`](../TODO.md) and is sound: instrument checks
passed, `d3_base = 0.0000` (a true zero), the dose map inverts with depth, `e6` reach is bimodal
(0.00 through ~L52, then 0.17 / 0.33 / 0.42 at L53 / L57 / L58, 1.00 at L60), and **L58 is the cell
the residual route exists to find** — `e6` reach 0.42, `d3` 0.027, `resid` −0.375: influence
without proportional forced-ID detectability. Whether L58 survives real `e5`/`d2` in Phase 4 is the
first genuine test of the pipeline's premise. **Do not re-derive any of this; read it.**

## 4. The objective

A Garlic result reliable enough to publish. Done means:

1. **Garlic** completes Phase 6 CONFIRM (`N_CONFIRM` held-out prompts, no adaptive stopping) and
   the §9.1 / §9.2 CONTROLS, producing `operating_point.json`.
2. **Origami** completes the same, as the replication control — v1 measured its full dose–response
   ([`../RESULTS.md`](../RESULTS.md)), so the pipeline must reproduce a known answer.
3. **Gate 5 returns a real ρ**, and if it is below `D3_MIN_RHO` the consequence spec 5.3 names is
   applied and recorded rather than quietly ignored.
4. Every gate either passes or is recorded as not-run **with what it would have validated stated in
   words**. A skipped gate is not a passed gate, and a SKIPPED count is not a disclosure.
5. Results reported with intervals, not point estimates.

**The headline this is aiming at:** a cell where the concept demonstrably influences generated
output, the model is demonstrably intact, and `d2` is low — on a concept chosen because it is the
hardest case available. Garlic is published at 100% forced identification at Macar's reference
configuration, which is why it was picked; but the result stands on its own measurements, and
**direct numerical comparison against the published rate is deferred**
([07](07-reference-cell.md)). What protects `d2`'s meaning is gate 11's agreement check against the
upstream judge, not a reproduced aggregate.

## 5. The design principle behind three of the tasks

This tool will be run on **models and concepts nobody has measured before**. A gate whose criterion
is a constant harvested from another concept's run cannot certify such a run: it either skips and
certifies nothing, or imports an assumption from a concept that may not be comparable. Gates 1, 4
and 6 all violated this, which is why all three were sitting in the "blocked on missing artefacts"
pile. Tasks [03](03-gate4-reanchor.md), [04](04-gate1-anchors.md) and
[05](05-gate6-false-negative-audit.md) rebuild them to be satisfiable from the run's own data.

Sort any gate you touch into one of three kinds:

| Kind | Validates | Cadence |
|---|---|---|
| **Instrument** | the apparatus — judge, injection hook, prompt format, extraction. A property of *(model, judge, code)*, not of the concept | once per configuration, cached; re-run when that configuration changes |
| **Run** | this concept's result | every run, and must be self-contained |
| **External comparison** | agreement with a published number | optional |

If you find another gate in the same position, raise it. The question is never "did M1 have this"
but "does this run's validity depend on it, and can this run supply it".

## 6. Order of work

**Read [`README.md`](README.md)'s status vocabulary before starting anything.** Implement nothing
whose status does not begin with `BUILD NOW`; several tasks in the folder are deliberately captured
ideas rather than work, and building them early costs review time and muddies the run.

Blocking, in order: [01](01-scan-doses.md) (must land before any measurement — it changes
`config_hash`), then [12](12-mmlu-letter-surface-forms.md), [02](02-judge-null-controls.md),
[03](03-gate4-reanchor.md), [04](04-gate1-anchors.md), [05](05-gate6-false-negative-audit.md),
[10](10-unexecuted-path-sweep.md). Then the non-blocking build tasks
[06](06-relaxed-reselection.md), [08](08-debug-bundle.md), [09](09-known-unfixed.md),
[13](13-prefix-token-contamination.md) step 1, [15](15-r5-portability.md). Then the run,
[11](11-the-run.md).

## 7. What to report

**During the run** — a short status after each phase: which phase, elapsed against prior, the
numbers by code, anything that looked wrong. Flag surprises immediately, not at the end; the
operator can stop a run that is measuring the wrong thing, but not after it finishes.

**At the end:**

1. **The operating point** for Garlic — `(layer, r, α)` with `e5`, `d2`, `s4` and their intervals,
   beside the reference cell's `d2` from task 07.
2. **Origami**, the same, beside v1's known dose–response, with discrepancies called out rather
   than averaged away.
3. **The gate table** — every gate, passed / failed / not-run, and for each not-run one sentence on
   what it would have validated and what its absence means. This section decides whether the number
   is publishable; write it for a sceptical reader, not as a checklist.
4. **Gate 5's ρ**, prominently, with what it implies about the shortlist.
5. **Measured CONFIRM and CONTROLS timings** into [`../TODO.md`](../TODO.md).
6. **New defects**, in the [`../DEBUG-LOG.md`](../DEBUG-LOG.md) format. *Why nothing caught it* is
   the load-bearing field.
7. **What you would not publish yet, and why.** Be a hostile reviewer of your own result. The
   operator would rather hear the weakness from you than from a referee.

---

## Change log

| Date | Change |
|---|---|
| 2026-08-12 | Created. Scan doses decided; gates 1/4/6 rebuilt for self-sufficiency; R4 set aside; judge null controls, relaxed re-selection, reference cell and debug bundle added as tasks. |
| 2026-08-12 | Status vocabulary added to [`README.md`](README.md): nothing is built unless its status begins with `BUILD NOW`. R5 scoped to Gemma3-27B and its real job re-derived portably (15). Judge bake-off on stored transcripts queued for after the run (16). |
| 2026-08-12 | Reference cell (07) **deferred** — comparison against Macar is not this run's framing. Gate 6 (05) generalised into a **tiered shortlist**: tier 1 always runs as the audit, further tiers escalate only when no window is found, every knob parametrized with an exhaustive mode. Two new items from review: `s3` scores only uppercase option letters (12, a defect), and `prefix_only` token contamination in `e6`/`d3` needs a diagnostic (13). `s3` verification by generation queued as a future addition (14). |
