# 28 — A run must survive losing its volume

**Status: BUILD NOW.** Branch `pareto`. Operator requirement, 2026-08-14: RunPod repeatedly forces
a move to a host where the existing network volume cannot be attached. The working assumption that
*the volume is the backup* is therefore false in practice.

## Why the current design does not cover this

`docs/RUNBOOK.md` says the volume survives and the container does not, and every resume path is
built on that. It held until the platform stopped honouring it. The 2026-08-14 run also showed the
volume failing **mid-run** (`OSError: [Errno 5]`, TODO item 19), so "the volume is durable" is
wrong in two independent ways.

Two zips already exist and neither is a resume bundle:

| | what it is | why it is not enough |
|---|---|---|
| `m2_<concept>_<hash>.zip` | local archive | never designed to be re-imported; nothing reads it back |
| `export_<concept>_<hash>.zip` | deliverable | filtered by `EXPORT_DENY`, and correctly excludes `vectors/` |

There is an **export** path and no **import** path. That is the gap.

## What resume actually needs

The expensive artefacts, in cost order:

| artefact | cost to rebuild |
|---|---|
| `scan.jsonl` | **34 min of GPU** — 189 cells |
| `selection_d2.jsonl` | ~8 min plus ~185 judge calls |
| `judge_*.jsonl` | the judge cache; losing it re-spends every call |
| `baselines.jsonl`, `norms.jsonl`, `dose_map.json`, `mmlu_items.json` | minutes |
| `config.json`, `provenance.jsonl` | nothing, but they identify what the rest belongs to |
| `vectors/*.pt` | **4 seconds** |

**Vectors stay excluded.** They rebuild in seconds, `CLAUDE.md` hard rule 3 forbids archiving them,
and *"regenerate, never archive"* is the standing rule. Their absence costs nothing.

The whole set is **scalars and transcripts, ~0.1 MB compressed** — the run that produced it wrote
zips of exactly that size. This is cheap to move.

## What to build

**`--resume-bundle`** — write `resume_<concept>_<hash>.zip` containing everything above except
`vectors/`, plus a manifest recording the config hash, the git commit, the phase states and the row
count per file. Produce it **on every run end, including a crashed one**, not only on request: the
run that most needs it is the one that died.

**`--import-bundle <path>`** — unpack into the run directory for the concept, and refuse loudly on
mismatch rather than merging silently:

- **config hash differs** → refuse. A resume across configurations is not a resume.
- **git commit differs** → warn, name both commits, continue. Code moves between runs by design.
- **the target already has rows** → refuse unless `--force`, and say which files collide.

**A row-count check on import**, against the manifest. A JSONL truncated by the same volume failure
that made you export it reads as a complete phase to everything downstream — `m2.setup` already
repairs truncated JSONL, so reuse that path rather than writing a second one.

## Where it must not write

Never into the repository working tree. `.gitignore` covers `*.pt` and the artifact directories but
a zip is none of those — add an explicit ignore for `resume_*.zip`, same as task
[08](08-debug-bundle.md) requires for the debug bundle.

## Acceptance

- A run that crashes mid-phase still produces a resume bundle naming the phase that died.
- Import into an empty run directory, re-run the concept, and confirm SCAN is skipped as already
  measured rather than re-executed.
- Import with a different config hash **raises**, and leaves the target untouched.
- Import of a bundle whose manifest row count disagrees with the file **raises**.
- `vectors/` is absent from the bundle, and the creation message says so and says why.
- `git status` on a pod after producing one shows nothing new to commit.

## Note for the operator

The bundle contains transcripts, so the benign-concept rule applies exactly as it does to
`export_bundle`: gate on `runio.transcripts_allowed`, no override. Moving a benign concept's
scalars and transcripts between pods is fine. The harmful arm is a different conversation and this
task does not open it.
