# 14 — Verify `s3` by single-token generation at the chosen cell

**Status: FUTURE — DO NOT BUILD.** Recorded so the idea is not lost. Not for this run.

## The idea

At the final chosen operating point only, re-measure MMLU by **generating one token** rather than
by reading the letter logits, and compare the two readings.

A multiple-choice answer is one token, so generation is capped at one token and the cost is one
batched pass over the pinned 57-item set — negligible against the phase it would sit in.

## Why it is worth more than it looks

Greedy single-token generation is the argmax of the same logits `s3` already reads, so at first
glance the two must agree. **They agree on the answer; they disagree on the question.**

`s3` reads `p(A)`, `p(B)`, `p(C)`, `p(D)` and takes the argmax **over those four**. It never asks
what the model's actual top token was. So it cannot distinguish:

- *"the model confidently answers C"* — `C` is the top token overall; from
- *"the model wants to emit something else entirely, and among the four letters `C` happens to be
  highest"* — the top token is a newline, a word, or noise, and the letters are all in the tail.

The second case is exactly what a degraded model does, and `s3` currently reports it as a correct
answer. That is a blind spot in the sanity term at the precise doses where sanity is the binding
constraint.

Generation exposes it directly: if the model emits something that is not an option letter, the item
is not a correct answer under any reading.

## The cheap version, worth doing first

Most of the benefit needs no generation at all, because the information is already in the tensor
`score_letter_logits` receives. Report per item:

- whether the **overall argmax** is one of the eight-or-so option-letter ids;
- the probability of the overall top token, beside `p_gold`.

Then aggregate: *"on N of 57 items the model's top token was not an option letter"*. That number
belongs beside `s3` in every cell's row, and it is free.

If that count is near zero across the dose range, the blind spot is theoretical and the generation
pass is unnecessary. If it rises with dose, `s3` has been over-reporting capability at exactly the
cells that matter, and the generation check becomes a fix rather than a diagnostic.

## Relationship to task 12

[12](12-mmlu-letter-surface-forms.md) is the same family: `s3` scores a fixed set of token ids and
has no ground truth to check itself against. Lowercase letters were missing from that set and
nothing noticed, because a capability ratio that reads plausibly low is indistinguishable from one
that is measuring the wrong token. **This task is the check that would have caught it.** Sequence
them: fix 12 first, then build this as the standing guard against the next instance.

## Acceptance, when built

- The non-letter-argmax count appears per cell, alongside `s3`.
- At the chosen operating point, the generated answer and the logit argmax are compared, and any
  disagreement is reported rather than reconciled.
- The comparison runs on the pinned item set through the same rendering, so it measures the
  instrument and not a second prompt format.
