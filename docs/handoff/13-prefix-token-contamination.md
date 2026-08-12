# 13 — Prefix-token contamination in `e6` and `d3`

**Status:** QUEUED. Diagnostic first, correction only if the diagnostic says it is needed.
**Blocking:** no — but the diagnostic must run before the scan surface is trusted.

## The problem

`e6` and `d3` measure "how much probability mass sits on the concept" by summing the probability of
the concept's **first token** over its surviving surface forms. `vectors.concept_first_token_ids`
keeps a variant if its first token is a prefix of the concept of at least `MIN_PREFIX_CHARS`
characters, and **flags but does not drop** tokens that are a strict prefix rather than the whole
word:

```python
entry["prefix_only"] = not entry["whole_word"]
```

That is the right call — dropping a concept's own leading token would lose real signal — but it has
a consequence nobody has measured. If `Garlic` first-tokenizes to `gar`, then every `e6` and `d3`
reading is also counting *garden*, *garment*, *garage*, *garnish*.

## Why it is not just noise

The contamination is **larger where the model is producing fluent unrelated text and smaller where
it is not** — so it varies systematically across the layer axis rather than adding a constant. That
makes it capable of reordering the scan surface, and the scan surface is what Phase 2 shortlists
from. A layer could enter the shortlist on borrowed mass, or a real layer could be ranked below one.

It cannot corrupt the headline: `d2` is judged from generated text, not token-counted. The exposure
is entirely in **which cells get measured**, which is [05](05-gate6-false-negative-audit.md)'s
territory — and the tiered audit there is a partial defence, because a layer wrongly ranked down
can still be picked up by a later tier.

One reassuring data point already exists: `d3_base = 0.0000` for Garlic, a true zero. At the
forced-ID position with no steering, the prefix token carries no measurable mass. That bounds the
contamination at *that* position under *that* prompt; it says nothing about the `e6` prompt set,
where the model is generating freely.

## Step 1 — the diagnostic (do this first)

1. **Surface the kept/dropped table per concept, prominently.** `concept_first_token_ids` already
   returns `kept` and `dropped` with reasons and the `prefix_only` flag. Log it at Phase 0 and
   write it into the run record. Right now it is only surfaced when extraction fails.
2. **Report a contamination estimate**, cheaply: for each `prefix_only` variant, the unsteered mass
   on that token across the `e6` prompt set. That is the floor — mass the token carries with no
   concept injected at all — and it costs one forward pass per prompt, which Phase 0 already does.
3. **Flag concepts where every kept variant is `prefix_only`** as higher-risk in the run record. A
   concept whose whole-word form survives is not exposed to this at all.

Then **review the Garlic and Origami tables by hand** before the scan surface is used for anything
beyond screening. That review is a deliverable, not a habit — write what it found into
[`../RESULTS.md`](../RESULTS.md).

## Step 2 — the correction, only if step 1 says it is needed

If the unsteered floor is material, the honest fix is a **two-token lookahead**: after the prefix
token, is the *next* token one that completes the concept, or one that completes a different word?
That is what actually distinguishes `gar → lic` from `gar → den`, and it is the only thing that
does.

Cost: a second forward pass per prompt for the affected concepts, which roughly doubles `e6` on
those. That is affordable at Phase 1's budget but not free, so it should be conditional on the
concept having a `prefix_only` variant at all.

**Do not implement this before step 1 reports.** A correction applied to a contamination that turns
out to be negligible is a change to the measurement with no benefit, and it would need its own
validation.

## Escalate

- **Whether `MIN_PREFIX_CHARS` should rise** for concepts where the whole word survives anyway.
  Raising it drops marginal prefixes at the cost of possibly dropping all variants for some
  concepts, and `concept_first_token_ids` raises when nothing survives. That is a threshold change
  with a failure mode, so propose it with the evidence from step 1 rather than tuning it.

## Acceptance

- Every run records the kept/dropped table and the `prefix_only` flags.
- Concepts with only `prefix_only` variants are flagged in the run record.
- The unsteered floor per `prefix_only` token is reported.
- The Garlic and Origami hand review is written up.
