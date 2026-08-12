# Decisions and progress log

**Authority on: what has happened.** Append-only, newest at the bottom.

This is the channel between everyone working on this repository. Read the last few entries before
starting a session — they are how you find out what was decided while you were not here.

Three things belong in here and nowhere else:

1. **A decision**, whoever made it — a threshold chosen, a design question answered, a proposal
   accepted or rejected, a scope change.
2. **A task completed** — which task, which commit, anything that turned out differently from what
   the task document said.
3. **A run phase completed** — which phase, how long, the headline numbers by code.

What does *not* belong here: open questions (those go in [`TODO.md`](TODO.md)), defect post-mortems
(those go in [`DEBUG-LOG.md`](DEBUG-LOG.md)), and whether a task may be built yet (that is the
status line in its own document under [`handoff/`](handoff/)).

The split is deliberate. **A task document says what the current state is; this file says how it got
there.** Two files claiming to hold the current status is how a stale one gets believed.

---

## Format

Copy this. Keep it short — a decision nobody can find in ten seconds is a decision nobody reads.

```markdown
## YYYY-MM-DD — one-line title
**By:** PquePC | Opus | Sol   (name whoever actually decided, not who typed it)
**Kind:** decision | task complete | phase complete

What happened, in one or two sentences.

**Why:** the reasoning, if it is not obvious from the what. Skip this line if it is.
**Result:** the commit, the task document, the file — whatever someone would open next.
```

**Name the person or agent who made the call, not the one who wrote it down.** A threshold the
operator chose and Sol implemented is `By: PquePC`. If a decision came out of a conversation, name
both.

---

# Log

## 2026-08-12 — Documentation consolidated into one authority per question
**By:** PquePC (direction), Opus (execution)
**Kind:** decision

Fifteen overlapping markdown files became a root `README.md` and `AGENTS.md` plus eight documents
under `docs/`, each opening with a statement of what it is authoritative on. Superseded v1 guides
moved to `docs/archive/`.

**Why:** four separate documents explained how to run the pipeline and there was no README at the
root, so finding the current answer to anything meant knowing which document had been written last.
**Result:** `f97b300`.

## 2026-08-12 — Eight design decisions for the next run, and a task queue
**By:** PquePC (decisions), Opus (analysis and drafting)
**Kind:** decision

`SCAN_DOSES` gains a third dose at 0.60. Gates 1, 4 and 6 are rebuilt to be satisfiable from the
run's own data. R4 is set aside. Judge null controls, relaxed re-selection and a debug bundle are
added. Each has a task document under `handoff/`.

**Why:** gates 1, 4 and 6 each keyed their criterion to a constant from a different concept's run,
which cannot certify a run on a model nobody has measured. R4's stored status came from v1's
extraction and v1's judge, both of which M2 replaced.
**Result:** `5c1d7df`, `docs/handoff/`.

## 2026-08-12 — Shortlist becomes tiered; two defects queued
**By:** PquePC
**Kind:** decision

Phase 2 emits ordered tiers rather than a flat shortlist. Tier 1 always runs as the false-negative
audit; further tiers escalate only when no window has been found. Every knob parametrized,
including an exhaustive mode. The reference-cell task is deferred.

**Why:** three extra cells is our budget, not a property of the method — someone needing the
genuinely best steering point should be able to pay for as much of the surface as they want. The
reference cell went with deferring direct comparison against published rates.
**Result:** `ec6faf2`, tasks 05, 07, 12, 13, 14.

## 2026-08-12 — Task statuses formalised; R5 scoped; judge bake-off queued
**By:** PquePC (direction), Opus (execution)
**Kind:** decision

Every task document opens with a status from a fixed vocabulary, and nothing is built unless it
begins with `BUILD NOW`. R5's norm band is scoped to Gemma3-27B and skips elsewhere. A judge
bake-off on stored transcripts is queued for after the run.

**Why:** R5's ±2σ band around 4664 is a property of one model — a healthy vector from another
architecture fails it. Judges are about a quarter of run cost, so the question is not "cheapest"
but "cheapest that is demonstrably as good", and that has to be measured.
**Result:** `eb8c26e`, tasks 15 and 16.
