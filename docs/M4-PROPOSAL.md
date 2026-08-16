# M4 — measuring detection without asking

**Status: proposal.** M3 measures detection twice and both times by reading what the model says.
This adds two measures that do not, plus the concept choice that makes a covert cell findable at
all.

Written against the 2026-08-15 Garlic run: 196 cells, 2,940 generations, validated by four
independent audits (see `docs/TODO.md` items 34–35).

---

## 1. Why

M3's two detection channels are `identify` (prefilled: "the thought is about…") and `self_report`
(unprompted: "do you detect an injected thought?"). The prefill changes whether the model
*volunteers*; it does not change that we are reading generated text. Both are reports.

That is inherited, not invented here. Macar et al. measure detection **entirely** by self-report
(`docs/RESEARCH-PROPOSAL.md:52`), and every rate-producing function in the upstream harness —
`run_steered_introspection_test_batch`, `run_forced_noticing_test_batch`, `create_style_detection_prompt`,
`run_batch_false_positive_tests` — is a question the model answers in words.

The Garlic run shows what that costs. Of 44 cells with effectiveness ≥ 4, **30 sit at forced
identification exactly 1.00**. Once the concept is influential enough to matter, the model can
name it on demand, every time. Whether that means it *detected* the injection, or merely that a
prefilled sentence completes with the dominant concept in the residual stream, is not a question
a report can answer.

---

## 2. D1b — constrained Yes/No logit readout

**A recorded quantity, not a gate.** Nothing about the run's control flow depends on it. It is
written to disk beside every cell and analysed offline, exactly as the mechanical measures are.

This is a deliberate change from the M2 design, where D1b carried a pre-committed
partial-correlation gate against D1 (`RESEARCH-PROPOSAL.md:231`). A gate forces the question "is
this a valid proxy for the report?" before there is data to answer it, and answering it wrong in
either direction is expensive: reject a real signal, or license a proxy that repeats M2's `e6`
mistake of measuring first-token behaviour and calling it influence. As a recorded quantity it
needs no threshold, cannot fail a run, and accumulates the evidence that would settle the
question later.

**What it is.** One forward pass, no generation, no judge. At the answer position of the
detection question, take `logit("Yes") − logit("No")`. Do the same for off-target control
questions that should not move under injection. Report **target shift** and **control shift**
separately, never only their difference.

**Why separately** — this is S13 (`RESEARCH-PROPOSAL.md:150`) and it is the measure's main
failure mode. If control shift ≈ target shift, the injection is pushing the model toward "yes"
on everything and D1b is measuring affirmative bias, not detection. That is Hahami et al.'s
objection, and reporting only the differential hides it.

**What it buys.** A lean toward "yes" on trials where the model *answers* "no" is detection that
the report suppresses — invisible to both existing channels. On the Garlic run, 400 of 588
self-report responses are `clean_denial`; D1b is the only proposed way to ask whether any of
those denials were close calls.

**Cost.** Two forward passes per cell (target + controls), no tokens generated, no API calls.
Against ~8.6 s per generation batch this is nearly free — the strongest argument for recording it
whether or not anyone trusts it yet. `S14` hook liveness already covers the `start_pos` path it
uses.

---

## 3. A probe trained ACROSS concepts

This is the measure that is not a report at all, and the design detail that matters is in the
title.

**Why the single-concept version was rejected, correctly.** `RESEARCH-PROPOSAL.md:92` excludes
"linear probing of steered-vs-clean activations (near-trivial within a concept; held-out
generalisation unestablished)". Train a probe to separate garlic-steered from clean activations
and it learns to detect the garlic vector — a direction we added ourselves. The probe recovers
the experimenter's own intervention and tells us nothing about the model.

**What changes across concepts.** Train on (steered, clean) pairs from many concepts, then test
on a concept **never seen in training** — leave-one-concept-out, split by concept and never by
trial. A probe that transfers is no longer detecting any particular vector. It is detecting
*that an injection occurred*: a concept-general anomaly signal in the residual stream.

That is a different and much more interesting claim, and it is the one the field's framing
already rests on. The premise that detection is "substantially underelicited relative to the
information present in the residual stream" (`RESEARCH-PROPOSAL.md:40`) is a comparison between
what a probe can read and what the model will say. Nobody in this project has measured the first
half.

**The number it produces.** Per layer, per dose: probe AUC on held-out concepts, beside the
model's own forced-ID rate at the same cell. The gap is the **underelicitation margin** — how
much of the available signal the model does not use. If a probe reads injection at 0.95 AUC from
L40 while the model's forced-ID rate at L40's covert cells is 0.20, that margin is the finding,
and it is a bound no report-based measure can establish.

### The controls that decide whether it means anything

Without these it is a norm detector wearing a lab coat.

1. **Random-direction arm.** Inject a random vector of matched norm and train/test the same
   probe. If it fires as well on random directions, the probe reads perturbation *magnitude*, not
   concept injection, and the whole measure collapses to "is `||h||` bigger". **This is the
   control that decides the result** and it must be run first, not last.
2. **Matched dose across concepts.** M3 already doses per layer as a fraction of that layer's own
   coherence boundary, so `||αv||/||h||` is comparable by construction. Keep that; a probe trained
   on concepts injected at unequal effective strength learns strength.
3. **Unsteered controls at matched positions**, so position and prompt are not the signal.
4. **Both positions**: the injection `start_pos` and the answer position. They answer different
   questions — "is it present" versus "did it reach the readout".

### What a negative result would mean

If the probe does not transfer across concepts, that is publishable and useful: it says there is
no concept-general "something was injected" signal, only per-concept directions — which would
make the model's failure to report far less surprising, and would undercut the underelicitation
framing rather than supporting it. Design it so both outcomes are informative.

### Dual-use note

Probe weights are directions in activation space and the training set is raw activations. Both
stay outside the repository under `/workspace`, same as run data, per CLAUDE.md hard rule 3.
A cross-concept injection detector is a defensive artifact rather than an attack one — but the
activations behind it are not.

---

## 4. The concept for the next run

**Garlic was the wrong concept for finding a covert cell, and deliberately so.** It is annotated
`# 100% detection` in `m2/config.py` — one of Macar's high-detection concepts, kept because
"these are the 'before' numbers worth beating". The empty Tier 1 in the M3
analysis is the expected result on a concept selected for being maximally detectable.

Any concept can now be run. In order of fitness for this question:

| concept | why | risk |
|---|---|---|
| **Velocity** ← recommended | in the group selected as "bottom-10 by detection rate (Fig 19), **minus the high-forced-ID ones**" — chosen against *forced* ID specifically, which is the criterion that matters | moderately abstract |
| **Silk** | `# 0% detection` in Macar. Concrete noun, so the effect judge has the easiest possible job — and TODO 34a shows that judge over-calling is the live measurement risk | detection may be 0 because the vector is weak, not because the model is blind |
| **Pillows** | `# 0% detection`, concrete | as Silk |
| **Irony** | the documented *"steered behaviour but undetected"* case — literally the target phenomenon | abstract. "Is this response ironic?" is far harder to judge than "is this about garlic?", and the effect judge is the component with known calibration problems |

**Run Velocity.** It is the only one selected against forced identification, which is the metric
this study treats as detection. If it comes back inert — no influence anywhere — run **Silk**,
where a weak result is easy to distinguish from a judge failure.

**The positive control that must ride along.** A concept with 0% detection is ambiguous between
"wide covert envelope" and "the vector never did anything" — this is Macar's Silk case, and
`RESEARCH-PROPOSAL.md` flags it (S11, concept anchor). Effectiveness must be non-zero *somewhere*
in the grid before a low detection rate means anything. M3 already measures this: a concept whose
`effectiveness` never exceeds ~2 anywhere has not been shown to be covert, only to be absent.

---

## 5. Cost

| | |
|---|---|
| D1b | ~0. Forward passes only, no generation, no judge |
| cross-concept probe | one activation-extraction pass per concept per layer; no generation, no judge. Training is CPU-side and seconds |
| one more sweep (Velocity) | ~$2.43 judges, ~70 min GPU, as measured |

The two new measures are close to free because neither generates tokens. That, more than
anything else, is the argument for recording them now and deciding what they mean later.

---

## 6. What this does not do

It does not decide TODO 35 (whether a collapsed response counts as a failure to identify) — that
is prior to all of this and should be settled first.

It does not fix the effect judge's over-calling in the 4–7 band (TODO 34a), which remains the
largest known source of error in the measurement and is prompt work, not new instrumentation.

And it does not build phases 3 and 4. Confirming a candidate operating point at higher `n` on
fresh prompts is still the thing that turns a candidate into a result, and none of this replaces
it.
