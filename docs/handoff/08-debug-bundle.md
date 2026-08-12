# 08 — `--debug-bundle`: a full-fidelity export for audit

**Status:** DECIDED, with a hard constraint on scope. Implement as written.
**Blocking:** no.

## Goal

A flag that exports **everything** a run produced — including transcripts and vectors — into a
single zip the operator can hand to the orchestrating agent to verify the pipeline did what it
claims.

## Why

The existing `export_bundle` deliberately drops `vectors/`, `debug/` and binary weight extensions,
and gates transcripts on the concept being benign. That is the right default for a deliverable. It
is the wrong thing for *"check my pipeline is working"*, where the whole point is to inspect the
intermediate state — which vectors were extracted, what the model actually said, what the judge
actually returned.

Every diagnosis in the M1.5 review required reading generations. This flag is the same argument
extended one step: some questions require reading the vectors too.

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
