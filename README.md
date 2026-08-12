# steering-optimization

The **M2 operating-point finder**. Given a concept, it searches a language model for the
`(layer, dose)` at which the concept **visibly influences generated output**, the model
**cannot name it under forced identification**, and the model **is still intact** — and reports
the frontier of near-optimal alternatives alongside the controls that rule out the two artifacts
that mimic that result.

It is the instrument half of a study on whether language models detect activation-level concept
injection differently when the injected concept is harmful. The science half lives in the
`Emergent-Introspection` repository.

---

## Where everything is

**One document is authoritative per question. No two overlap.** If you find two that answer the
same question, that is a defect — say so.

| Question | Document |
|---|---|
| How do I run it, from a bare pod to a result? | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) |
| What does each measure mean, and why is it defined that way? | [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) |
| Where does each piece of code live, and what is it called? | [`docs/CONTRACT.md`](docs/CONTRACT.md) |
| Why is it built this way? What was decided, and against what? | [`docs/DESIGN-RATIONALE.md`](docs/DESIGN-RATIONALE.md) |
| What is the research question this serves? | [`docs/RESEARCH-PROPOSAL.md`](docs/RESEARCH-PROPOSAL.md) |
| What has actually been measured? | [`docs/RESULTS.md`](docs/RESULTS.md) |
| What went wrong before, and why did nothing catch it? | [`docs/DEBUG-LOG.md`](docs/DEBUG-LOG.md) |
| What is still undecided or unbuilt? | [`docs/TODO.md`](docs/TODO.md) |
| What comes after M2? | [`docs/M3-PROPOSAL.md`](docs/M3-PROPOSAL.md) |
| What am I, an agent, allowed to do here? | [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) |
| I am the coding agent — what is my current task? | [`docs/handoff/`](docs/handoff/) |

Superseded documents are in [`docs/archive/`](docs/archive/). They describe the v1 measurement
lab, which M2 replaced; nothing there is current.

---

## The measures, by code

Everything in this repository refers to a measurement by its **code**, not by a prose name.
`d2`, not "the forced identification rate". Full definitions in
[`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

**Cheap tier** — forward passes only, no generation, no judge. This is what makes a full-depth
scan of every layer affordable.

| Code | What it measures |
|---|---|
| `e6` | reachability — the fraction of prompts on which the concept's token mass clears `E6_THRESH`. A **proxy** for effectiveness; it shortlists layers and is never reported as a result |
| `d3` | forced-ID concept mass read straight off the logits. A **proxy** for detection, trustworthy only if gate 5 passes |
| `s3` | capability — MMLU accuracy scored from the four option-letter logits, as a ratio against the unsteered baseline `cap_base` |
| `s2` | objective degeneracy of generated text — repetition, alphabetic fraction, length. Mechanical, never a judge |

**Expensive tier** — costs a generation, a judge call, or both. Runs only on the shortlist.

| Code | What it measures |
|---|---|
| `e5` | concept influence, 0–10, judged against the model's **own** unsteered reply. The primary effectiveness metric |
| `s1` | response integrity, judged **with the concept withheld from the judge** |
| `d2` | forced identification rate over `N_D2` trials. **The constraint the pipeline exists to satisfy** |
| `d4` | failure-mode distribution over the `d2` transcripts |
| `judge_fpr` | the judge's null reading — what it scores when nothing was injected |

**Derived:**

| Code | What it is |
|---|---|
| `s4` | `min(s1, s2, s3)`. `min`, never a mean: one broken term must not be averaged away |
| `r` | the normalised dose, `α·‖v_L‖ / ‖h_L‖`. **All layer comparison happens in `r`, never in α** — at fixed α the real dose varies more than 20× across layers, non-monotonically |
| `cap_base` | unsteered MMLU baseline, the reference `s3` is scored against |
| `resid` | residual of `d2` against what `e5` predicts. A **search** device that widens the shortlist |
| `covertness_margin` | `d2 − predicted_d2(e5)`. **Reported, never selected on** |

A **cell** is one `(layer, r)`. It **qualifies** on all three of `e5 ≥ E5_FLOOR`,
`d2 ≤ D2_MAX`, `s4 ≥ S4_MIN`. The **operating point** is `argmax(e5)` over qualifying cells, and
nothing else.

> The trap worth stating twice: **a cell with low `d2` because the model is damaged looks
> identical to a cell with low `d2` because the concept is covert.** `s4` and the §9 controls are
> the entire difference. Distrust a suspiciously good `d2` rather than celebrating it.

---

## The phases

| Phase | What it does |
|---|---|
| **CAL** (0) | extract vectors, measure norms, build the dose map, take baselines and `cap_base`, run the judge null controls |
| **SCAN** (1) | every layer in scope, at each scan dose. Cheap tier only, zero judge calls |
| **SHORTLIST** (2) | turn that surface into candidate layers: local maxima, stratified depth coverage, and the residual route. Deliberately **not** top-K |
| **BISECT** (3) | per candidate, bracket then bisect the dose at which sanity breaks |
| **VERIFY** (4) | real `e5`, `s1`, `d2` on the shortlist |
| **REFINE** (5) | ±1 and ±2 layers, one dose step either side, around the top cells |
| **CONFIRM** (6) | the winner re-measured on **held-out** prompts at `N_CONFIRM`, no adaptive stopping |
| **CONTROLS** | §9.1 random direction, §9.2 forced-ID capability, §9.3 escalation ladder |

**Phases 1–5 are screening.** They decide what gets measured and in what order; their numbers rank
cells and are not reportable. **Only Phase 6 output is a result.** Every row carries its phase
label so a screening number can never be read as a confirmation.

---

## Quick start

Full instructions in [`docs/RUNBOOK.md`](docs/RUNBOOK.md). The short version, on a fresh
A100/H100 80GB pod with `/workspace` mounted:

```bash
export HF_HOME=/workspace/hf
cd /workspace/steering-optimization
python -m m2.setup --repair
python -m m2.run --concepts Garlic --preflight
```

---

## Safety

This repository is subject to dual-use rules that are **not optional**: no model weights, no
concept vectors and no raw generations leave the machine, and nothing here is exposed on a public
port. Read [`CLAUDE.md`](CLAUDE.md) before acting in this repository, whether you are a person or
an agent.
