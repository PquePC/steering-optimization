# 12 — `s3` scores only uppercase option letters

**Status: BUILD NOW.** A defect found in review on 2026-08-12, not an improvement. **Blocking** — it biases a sanity term in a dose-dependent direction.

## The defect

`prompts.letter_token_ids()` builds each option letter's id list from exactly two surface forms:

```python
ids = _first_ids(tok, letter, f" {letter}")     # "A" and " A"
```

**No lowercase.** If a steered model answers `c` rather than `C`, `s3` scores that item wrong — not
because the model got it wrong, but because the instrument was not looking where the model wrote.

## Why it matters more than it looks

Case discipline is one of the first things to go as a model degrades. So the miss is **not random
across the dose axis**: it happens more often at higher doses, where output is getting sloppier.
That gives `s3` a bias with a direction and a slope — capability appears to fall with dose faster
than it does.

Follow that through the pipeline:

- `s4 = min(s1, s2, s3)`, so an under-reporting `s3` drags `s4` down.
- `S4_MIN` is a hard constraint on `qualifies`.
- So cells get rejected as "too damaged" when the model was answering correctly in lowercase.

The error direction is the merciful one — it costs you qualifying cells rather than admitting broken
ones — but **the qualifying cells are the entire product.** A pipeline that silently discards them
at exactly the dose range where the interesting cells live is failing at its one job.

This is also the same defect family as bug 20 and CONTRACT defence 6, both of which were about
scoring the wrong surface form: the leading-space case is already handled here precisely *because*
"scoring only the bare form reads near-zero everywhere, which is indistinguishable from a real
result". Lowercase is the same argument, unfinished.

## The fix

`_first_ids` is variadic, so the change is small:

```python
ids = _first_ids(tok, letter, f" {letter}", letter.lower(), f" {letter.lower()}")
```

Then check the two things that could go wrong:

1. **The cross-letter collision guard.** `letter_token_ids` already raises if two *different*
   letters share a first-token id, because that would make `argmax(p_A..p_D)` meaningless. Adding
   forms adds collision opportunities. Keep the guard, and make sure its message names which
   *surface form* collided, not just which letters — otherwise the failure is unreadable.
2. **`max`, not `sum`, over the id list.** This is already the rule in `score_letter_logits` and the
   reason is in its comment: if the tokenizer merges some forms for one letter but not another,
   summing hands that letter a systematic advantage unrelated to the model's answer. Adding two
   more forms per letter makes that asymmetry more likely, not less, so the `max` is load-bearing
   now rather than merely correct.

## Also worth checking while you are in there

The same reasoning applies anywhere else the pipeline reads a fixed token off the logits. `d3` and
`e6` enumerate three casings for the concept (`lower`, `capitalize`, `upper`) each bare and
leading-space — those are fine. Check whether anything else scores a literal token with a narrower
enumeration than the model could plausibly emit.

## Acceptance

- A test that trips it: score a synthetic logit tensor whose mass sits on the **lowercase** gold
  letter, and confirm that the item is counted correct. Before the fix that test must fail.
- The collision guard has a test that fires, naming the surface form.
- `cap_base` is re-measured after the change — it is the denominator of every `s3`, and it moves.

## For the record

Append to [`../DEBUG-LOG.md`](../DEBUG-LOG.md) once fixed, in the standard format. The *why nothing
caught it* field: `s3` has no null control and no ground truth — a capability ratio that reads
plausibly low is indistinguishable from a capability ratio that is measuring the wrong token, and
nothing in the pipeline compares it against a generated answer. Task
[14](14-mmlu-by-generation.md) is the check that would have caught it.
