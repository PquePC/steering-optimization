# 23 — Sit with the model at the operating point before believing the numbers

**Status: BUILD NOW.** Executes *after* the first run that produces an operating point. Mostly
protocol; the tooling largely exists already.

## Why

The three sanity terms are a narrow net:

| | What it actually covers |
|---|---|
| `s3` | MMLU accuracy from **first-token letter logits**, 57 multiple-choice questions. No generation at all |
| `s2` | mechanical degeneracy — repetition, alphabetic fraction, length — on **12 short** E5 responses |
| `s1` | a judge's integrity score on **those same 12 responses** |

Nothing measures instruction-following outside those twelve prompts, long-form coherence, factual
accuracy in free text, multi-turn behaviour, or **refusal behaviour** — any of which steering could
disturb without moving `s1`, `s2` or `s3` at all.

So "the operating point passed `s4`" means *the model still picks MMLU letters, doesn't loop, and
writes twelve acceptable short answers.* That is not the same as *the model is fine.*

## The specific worry, stated sharply

Every candidate so far sits at **L51–L58 of 62**, against Macar's reference of L37. At that depth
you are close to the unembedding, and there is a real possibility that steering is acting more like
a **logit bias** than like a change to what the model represents.

If that were true, the headline result would be hollow: *"the concept influenced the output but the
model cannot identify it"* would be trivially true, because the concept never entered the model's
computation — it was pasted onto the output distribution. **Low `d2` would mean "there was nothing
to detect", not "the influence was covert."**

**The evidence so far argues against it, but it is an argument, not a test.** At L58 `r = 0.30` the
scan reads reach **0.42** with `d3` **0.03** — the concept appears in free generation while
carrying almost no mass at the forced-ID position. A pure logit bias would raise the concept token
*everywhere*, including under forcing. The effect is context-dependent, which is what a
representational intervention looks like and what an additive output bias does not.

That is two numbers. It needs eyes on the actual text.

## The protocol

The tooling exists. From `docs/RUNBOOK.md`:

```python
from m2 import model, config, steer
model.load_model(config.CONFIG)
steer.use("/workspace/m2_runs/<run folder>")   # or concept=, layer=, r=
steer.compare("Tell me a short story.")        # steered vs unsteered, side by side
steer.session()                                # interactive; blank line to exit
steer.sweep("Describe a landscape.", [0.10, 0.15, 0.20, 0.30])
```

**What to look for, in this order:**

1. **Is the influence thematic or lexical?** Does a steered story *drift toward* garlic — cooking,
   pungency, smell, Mediterranean food — or does the word "garlic" simply appear in places it does
   not belong? Thematic drift is a representational effect. Lexical insertion at a late layer is
   the logit-bias signature, and it would undercut the result.
2. **Ask it to do something the E5 set does not cover.** A multi-step instruction, a format
   constraint ("answer in exactly three bullet points"), a short reasoning task. `s3` cannot see
   any of this.
3. **Generate something long.** Several hundred tokens. `s2` and `s1` only ever saw short answers,
   and degradation that appears at length would be invisible to both.
4. **Probe refusal.** A request the model would normally decline, and one it would normally accept.
   Steering near the output could plausibly move that boundary, and nothing in the pipeline would
   notice. Included because it is a load-bearing behaviour that is trivially cheap to check and
   completely invisible to all three sanity terms — anyone using this tool to find an operating
   point deserves to know whether that point moved the model's refusal boundary.
5. **Multi-turn.** Two or three exchanges. Every measure in the pipeline is single-turn.
6. **Compare against a mid-layer cell at matched `r`**, if one is ever sane and influential. Same
   dose, different depth — the cleanest available test of whether depth changes the character of
   the effect rather than its size.

## What to build

Very little. Confirm `steer.use()` reads the new `operating_point.json` shape after task 21, and
add a `steer.battery()` that runs items 2–5 against steered and unsteered side by side and prints
them for a human to read.

**It must not judge.** No score, no pass/fail. The entire point is that the existing judges are the
narrow net; adding a fourth judged metric would repeat the mistake. The output is text for the
operator to read.

## Acceptance

- `steer.use()` works against a run folder produced by the post-task-21 pipeline.
- `steer.battery()` prints steered and unsteered output for a multi-step instruction, a long
  generation, a refusal probe and a multi-turn exchange.
- Nothing in it writes a score.
- The run's completion message prints the four numbers `steer.steering()` needs — concept, layer,
  `r`, and the run folder — so the operator can go straight from a finished run to sitting with the
  model.
