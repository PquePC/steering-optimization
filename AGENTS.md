# AGENTS.md — read before acting in this repository

**The safety rules are in [`CLAUDE.md`](CLAUDE.md) and they apply to every agent regardless of
which file it reads first.** Read that file. This one adds the working conventions.

Orientation — what lives where — is in [`README.md`](README.md).

---

## Who owns what

| | Owns |
|---|---|
| **Operator** | the decisions. Cost, scope, what gets published, anything irreversible |
| **Orchestrating agent** | research and design direction, science calls, the documentation tree, review |
| **Coding agent** | the code, the tests, execution of the run, diagnosis of failures |

**Escalate design, do not encode it.** Choosing a threshold, changing what a measure means, or
deciding that a gate's absence is tolerable is a design call: state the options and the trade-off
rather than settling it in a commit. How to structure a function, which test to write, how to make
a guard trip — those are implementation and need no review.

The coding agent's current tasks are in [`docs/handoff/`](docs/handoff/), one document per task,
each with its own acceptance criteria.

---

## Conventions

**Refer to every measurement by its code** — `e5`, `d2`, `s4`, `e6`, `d3`, `s1`, `s2`, `s3`, `d4` —
in code, comments, logs, commit messages and messages to the operator. Gloss it on first use in
any document (`d2` (forced-ID rate, 0–1, lower is better)), then use the bare code. The inventory
is in [`README.md`](README.md).

**Every new guard gets a test that trips it.** Six of the thirteen defects in the first end-to-end
run were checks that *could not fail*: `d3 > 0.0` on a quantity that is never exactly zero, an
import test used as a frontend test, a `{concept}` placeholder test that a hard-coded name passes,
a detector that had never been run. A guard that passes everything is worse than no guard, because
it reads as evidence. `test_the_undefined_name_detector_actually_detects` is the model to copy.

**A prior whose unit disagrees with its counter is the same class of defect.** `BISECT` was priced
per bisection *probe* while the board ticks once per *candidate* — a 12× error that looked like a
measurement. State the unit beside any number in `PHASE_SECONDS_PRIOR` or `PHASE_UNITS_PRIOR`.

**Respect the module order** in [`docs/CONTRACT.md`](docs/CONTRACT.md). No module imports a
sibling appearing later in the layout list. `config` imports nothing from `m2`; `driver` may import
everything. The offline tests must keep working on a laptop with no torch and no transformers.

**Plain `.py`, not notebook cells.** Bug 24 was a notebook whose cells silently collapsed to one
physical line and did nothing while reporting success. The notebook is a driver, not a codebase.

**One source of truth for a threshold.** Read constants as `CONFIG["D_MIN"]` — the live value after
a control-panel edit, and it raises `KeyError` rather than returning a default.

**Docstrings explain why a value is what it is, beside the value.** `m2/config.py` is the house
style. This is load-bearing here: most of the defects in the log were found by someone reading a
rationale that did not match the code.

---

## Git

- **Pull before every work session and before every commit.** Two agents work in this tree.
- **Small, tightly scoped, named commits.** They are reviewed, and a defect that returns must be
  traceable to the change that was supposed to stop it.
- **Never force-push. Never rewrite history.**
- **Commit messages carry science only** — what changed and why it was wrong before. No funding,
  budget, mentor or novelty framing. Match the existing style: a plain statement of what the change
  does, e.g. *"Stop the shortlist widening onto layers with no concept mass"*.
- If you find a file mid-change by someone else, do not resolve it by reverting their work — say
  what you found and ask.
- **Documentation is owned by the orchestrating agent.** Do not restructure, merge, rename, move or
  delete a markdown file. Appending to `docs/TODO.md` and `docs/DEBUG-LOG.md` is expected; see
  [`docs/handoff/README.md`](docs/handoff/README.md) for what you owe them.
