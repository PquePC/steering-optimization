# AGENTS.md — read before acting in this repository

**The safety rules apply to every agent regardless of which file it reads first.** They are
reproduced in full below rather than linked, because an agent that reads only this file must still
be bound by them. [`CLAUDE.md`](CLAUDE.md) carries the same rules plus the project context; if the
two ever diverge, that is a defect — report it.

Orientation — what lives where — is in [`README.md`](README.md).

---

## 🛑 Hard rules — never do these, and never propose them

1. **Never upload ablated model weights** anywhere: Hugging Face (public *or* private), Drive,
   gists, pastebins. The ablation rebuilds from a published config in minutes — regenerate, never
   archive.
2. **Never expose the model on a public port or proxy.** An unauthenticated Jupyter/Gradio/vLLM
   endpoint on a rented pod *is* a public uncensored model.
3. **Never commit vectors, activations, or raw generations** (`*.pt`, `*.safetensors`, `*.npy`,
   `vectors/`, `activations/`, `results/`, `outputs/`, sweep logs). Vectors are reusable attack
   artifacts; generations are what an uncensored model said with `weapon` injected. Publish cosine
   scalars and aggregate rates instead — they reconstruct nothing.
   **`.gitignore` enforces this and is load-bearing — do not weaken it.** A run writes to
   `/workspace/m2_runs`, outside the repo, for the same reason.

## ⚠️ Stop and ask for explicit approval before

- Any upload to an external host, of anything
- Opening any pod port, proxy, or public URL
- Sending more than concept words and short self-reports to a third-party judge API
- Publishing per-concept detail rather than per-arm aggregates (per-concept = a lookup table)
- Any push whose diff carries any of the above

If an action would make this work **more useful to someone attacking a model than to someone
auditing one**, raise it rather than doing it.

## The harmful arm has not run

`config.HARMFUL_CONCEPTS` names the three concepts the study will eventually compare against the
benign arm. Nothing in this repo runs them, and the pipeline's export gate withholds transcripts
for any concept not on the benign list unless a call site passes an explicit override. Do not add
that override, and do not run the harmful arm, without reading the parent repo's ethics register
first.

## One standing exception, and it is not a precedent

`m2/tests/fixtures/garlic_shakedown_scan.jsonl` is committed run data — the cheap-tier scan surface
from the 2026-08-12 Garlic shakedown. It was approved **explicitly and once**, by the operator, on
these grounds: Garlic is a benign concept, the file contains only scalars (`reach`, `d3`, `s3`,
norms, `alpha`), no generations and no vectors, and it is the regression fixture for task 21.

**It does not license committing any other run artefact.** The rule above stands unchanged. Adding
a second such file needs its own explicit approval, and the harmful arm's scan surface is never
eligible.

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

## Logging — this is not optional bookkeeping

Everyone here works from what the last person wrote down. Three obligations, every session:

**1. Log every decision in [`docs/DECISIONS.md`](docs/DECISIONS.md).** Append-only, newest at the
bottom, format at the top of the file. A decision is: a threshold chosen, a design question
answered, a proposal accepted or rejected, a scope change — **including decisions the operator makes
in conversation with you.** Those are the ones most likely to be lost, because they never passed
through a file. Name whoever actually made the call, not who typed it.

**2. Mark a task done in its own task document.** When a task under
[`docs/handoff/`](docs/handoff/) is complete, change its status line to `DONE` with the commit hash,
and add a `task complete` entry to `DECISIONS.md`. The status line is the current state; the log is
the history. **Do not duplicate status into a second file** — two places claiming to hold the
current status is how a stale one gets believed.

**3. Log a completed run phase** in `DECISIONS.md` as it happens: which phase, elapsed against its
prior, the headline numbers by code, anything that looked wrong.

Also keep appending to [`docs/TODO.md`](docs/TODO.md) (open items) and
[`docs/DEBUG-LOG.md`](docs/DEBUG-LOG.md) (defects, in its format — *why nothing caught it* is the
field that matters). See [`docs/handoff/README.md`](docs/handoff/README.md).

## Git

- **Pull before every work session and before every commit.** Two agents work in this tree.
- **Push when work is complete and tests are green.** Do not leave finished work sitting in local
  commits — the other agents cannot see it, and the operator cannot review it. If you are unsure
  whether something is ready, push it to a branch and say so rather than sitting on it.
- **Small, tightly scoped, named commits.** They are reviewed, and a defect that returns must be
  traceable to the change that was supposed to stop it.
- **Never force-push. Never rewrite history.**
- **Commit messages carry science only** — what changed and why it was wrong before. No funding,
  budget, mentor or novelty framing. Match the existing style: a plain statement of what the change
  does, e.g. *"Stop the shortlist widening onto layers with no concept mass"*.
- **End every commit message with an attribution trailer**, on its own line, as the last line:

  ```
  Made-by: Sol
  ```

  `Opus`, `Sol` or `PquePC` — whoever actually did the work. Git's author field records the
  machine that committed; this records the agent that wrote it, which is what someone reading the
  history a month from now needs to know. Opus additionally carries its
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` line.
- If you find a file mid-change by someone else, do not resolve it by reverting their work — say
  what you found and ask.
- **Git operations are expected to require the operator's approval, and that is deliberate.** If a
  commit, a push or anything touching `.git` is blocked by your sandbox, **ask for approval — do
  not route around it.** Do not disable hooks (`--no-verify`), override config, escalate your own
  permissions, or reach for a shell trick that writes to `.git` indirectly. The friction is the
  control. The same applies to network access: if a command needs the internet and is blocked, ask.
- **Documentation is owned by the orchestrating agent.** Do not restructure, merge, rename, move or
  delete a markdown file. Appending to `docs/DECISIONS.md`, `docs/TODO.md` and `docs/DEBUG-LOG.md`
  is expected and required; restructuring them is not yours.
