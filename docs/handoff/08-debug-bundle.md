# 08 — `--debug-bundle`: a full-fidelity export for audit

**Status: BUILD NOW - NOT BLOCKING.** Two halves, capture then export. Build it **while the next
run is executing**, not before it: the shakedown proved auditable enough from `--no-wipe` plus the
existing persistence to find five defects, so a complete run is worth more than a more complete
export of an incomplete one. Its capture half changes what is written, so it takes effect on the
run after next. Task 09 item 1, its prerequisite, is done.

## Goal

**Everything** a run produced, in one zip the operator can hand to the orchestrating agent to
verify the pipeline did what it claims: transcripts, generations, judge inputs and raw judge
replies, every intermediate calculation, and the vectors.

## Why there are two halves

The obvious reading of this task is "turn off the export filter". **That is not enough, and
discovering why is the point of reading the code first.**

`EXPORT_DENY` drops `vectors/`, `debug/` and the binary weight extensions, and `archive_concept`
deliberately keeps `debug/` in the local archive because *"the debug dumps are what make a cell
auditable"*. But **nothing in `m2/` writes to `debug/` at all.** The directory, the deny-list entry
and the archive carve-out are inherited from the v1 lab and are vestigial. So flipping the filter
today exports exactly one extra thing — `vectors/` — and every intermediate the operator wants to
audit is computed and discarded.

**An export flag can only ship what was written.** Hence:

- **Capture** — persist the intermediates that are currently computed and thrown away.
- **Export** — stop filtering, for benign concepts only.

## Half 1 — capture

Much is already persisted and needs nothing: `scan.jsonl`, `verified.jsonl`, `bisect.jsonl`,
`confirm.jsonl`, `controls.jsonl`, `baselines.jsonl`, `norms.jsonl`, `dose_map.json`,
`shortlist.json`, `provenance.jsonl`, `operating_point.json`, the three `judge_*.jsonl` files
(which already carry the **raw** judge reply, on the stated principle that *"a stored score with no
stored response is a number nobody can re-check"*), `D2_transcripts.jsonl` and
`cis_transcripts.jsonl`.

**Audit what is computed and not written**, and persist it under the flag. Start from these, and
report anything else you find:

- the per-item detail the measure functions return but the row may not keep — `e6_per_prompt`,
  `s3_per_item`, `d3`'s per-trial readings, `s2`'s reasons;
- the **kept/dropped concept-token table** from `concept_first_token_ids`, which task
  [13](13-prefix-token-contamination.md) needs anyway and which is currently only surfaced when
  extraction fails;
- every bracket probe in Phase 3, not only the bisection result;
- the null-control transcripts from task [02](02-judge-null-controls.md);
- unsteered generations wherever they are only summarised;
- the intermediate arithmetic behind each composite — the three terms behind every `s4`, the
  `alpha`/`r` conversion inputs per cell, the residual fit behind `resid`.

**Two things to decide and propose, rather than assume:**

- **Raw per-position logit dumps** — the original purpose of `debug/`. These are what would let
  someone re-derive `e6` and `d3` from scratch, and they are also enormous: vocabulary × positions
  × cells. Full capture across a scan is not affordable. Propose a bounded form — the top-k tokens
  with their probabilities at the scored position, say — with the size arithmetic for a full run.
- **Where capture is switched on.** A flag on `m2.run`, and it should be **on for these first runs
  and off by default afterwards**, because the size is real.

**Prerequisite:** task [09](09-known-unfixed.md) item 1. The volume free-space guard currently
cannot fire, and turning on full capture is exactly the change that would fill a volume. **Fix the
guard before enabling capture**, not after.

## Half 2 — export

The existing `export_bundle` drops the deny list and gates transcripts on the concept being benign.
The debug bundle disables the deny list — including `vectors/` and `debug/` — with the benign check
made hard rather than overridable.

Every diagnosis in the M1.5 review required reading generations. This is the same argument extended
one step: some questions require reading the vectors and the intermediates too.

## The hard constraint

**Benign concepts only, and the refusal must be structural.**

`m2/config.py` has `BENIGN_CONCEPTS` and `HARMFUL_CONCEPTS`, and `runio.transcripts_allowed`
already implements exactly this kind of gate for the transcript case. Follow that pattern:

- **The debug bundle must refuse outright for any concept not on the benign list** — not warn, not
  offer an override. `export_bundle`'s `EXPORT_TRANSCRIPTS_OVERRIDE` exists because a deliverable
  sometimes legitimately needs harmful-arm transcripts under an explicit per-call decision. **The
  debug bundle has no such case and must not have an override at all.** A harmful concept's vector
  is the single most misuse-relevant artifact this project produces (`CLAUDE.md` hard rule 3); an
  "everything" export is exactly the wrong place to leave a door.
- The refusal is on the **concept**, checked at export time against the live concept list, not on a
  flag the caller passes.

## The other constraint, which is about handling rather than code

Vectors regenerate from a published config in minutes, and the project's rule is **regenerate,
never archive**. A debug bundle is a deliberate, temporary exception for verification — not a
backup and not a convenience. The code should say so where the operator will read it:

- name the file so it is obviously not a deliverable (`debug_bundle_<concept>_<hash>.zip`);
- print, on creation, that it contains vectors, that it is benign-arm only, and that it should be
  deleted after use;
- **never** write it anywhere that syncs, and never into the repository working tree — `.gitignore`
  already covers `*.pt`, `*.safetensors`, `*.npy` and the artifact directories, but a zip is none
  of those. Add an explicit ignore rule for the debug bundle name pattern so an accidental
  `git add -A` on the pod cannot commit one.

That last point is the one most likely to be missed and the most expensive if it is.

## What to change

- Add `--debug-bundle` to `m2/run.py`.
- Add the export path in `m2/runio.py` beside `export_bundle`, sharing its zip machinery but with
  the deny list disabled and the benign check hard.
- Add the `.gitignore` rule.
- Log what went in, **named not counted** — `export_bundle` already follows this rule for
  exclusions, and the same reasoning applies: "47 files included" tells a reader nothing about
  whether the right ones were.

## Acceptance

- A test that trips it: call the debug export with a harmful concept name and confirm it **raises**,
  and that no partial zip is left behind.
- A benign run produces a zip containing `vectors/` and every transcript file.
- `git status` on a pod after producing one shows nothing new to commit.
- The creation message states the deletion expectation.

## Note for the operator

Handing this zip to an agent means uploading a benign concept's transcripts and vectors to an
external service. That is fine for the benign arm and is consistent with the spec's transcript
policy — those are what a model said when steered toward *Garlic*. It is **not** fine for the
harmful arm, which is why the refusal is structural rather than advisory.
