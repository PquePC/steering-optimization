# M5 — screen more cells, and read the model instead of asking it

**Status: logged, not scoped.** Written 2026-08-22, deferred until after the write-up. Nothing
here is scheduled and nothing here is built. The point of this document is that the reasons are
recorded while they are fresh, so the next scoping session does not start from memory.

**The one-line reason it exists:** after the 2026-08-21 re-analysis, the instrument is not trusted
enough to carry a result. Two things are wrong with it, they are different problems, and only one
of them is fixed by spending more GPU time.

---

## 1. Problem one — the grid is too sparse where the answer lives

The 99 re-measured cells were chosen from the old runs by looking for interesting cases, so the
grid came out ragged. Measured on `private/datos-rejuzgados`:

- **Median doses per layer: 1 in Silk, 2 in Garlic**, against 6 in the original sweeps.
- In 9 of Silk's 24 layers, three different selection criteria return the same cell, because it is
  the only one measured there.
- The dose actually measured swings from 0.22 at L13 to 0.075 at L19 to 0.19 at L39, so a curve
  drawn per layer compares cells that were not steered comparably.

That is not a judge problem or a statistics problem. **A per-layer reading needs several doses per
layer or it means nothing**, and the same budget spent differently would have it: roughly 8 layers
x 6 doses x n=22 is the same ~99 cells, and every layer's point would then mean the same thing.

**What M5 changes here:** the sweep spends its budget on depth of the dose ladder within fewer
layers, after a first pass has said which layers are worth the ladder. Screening more cells is the
ask; screening *rectangularly* is the actual fix.

## 2. Problem two — every number is a judge reading generated text

Two measured facts about the current metrics, both from the re-judged set (84 healthy cells, 2,178
influence responses, 2,970 identification trials):

**Influence does not have a middle.** Mean 2.04 out of 10, standard deviation 3.32. That is not a
bell curve around 2, it is a pile of zeros and a handful of sevens and eights. The concept appears
or it does not. A cell mean at n=22 therefore carries **±13.3 pp**, and the things that would
normally shrink that do not: turning it into a rate is worse (±18 pp at n=22), and removing
between-prompt heterogeneity buys 2.3 pp and costs the metric its plain meaning. **Only more
responses per cell shrink it, and responses are the expensive part.**

**Forced identification is a switch, not a ramp.** Of Silk's 36 healthy cells, 24 sit at ≤20% and
8 at ≥80% — only 4 in between. Garlic: 24 and 20, again 4 in between. Within a single layer:

```
Silk   L37  dose 0.084 → influence  4.1%   identification    0%
            dose 0.160 → influence 33.6%   identification  100%
Garlic L43  dose 0.117 → influence  0.0%   identification    0%
            dose 0.250 → influence 39.5%   identification  100%
```

So the quantity the search optimises is close to binary at the resolution it is measured, and the
in-between region — the operating window — is where there is almost no data. Averaging across the
doses in a layer produced the flat 50% curves in the criteria figures: the mean of a 0 and a 100,
a value no real cell has.

**What M5 changes here:** stop making the model's own words the only channel. Three readouts, all
forward-pass only, all graded rather than binary.

---

## 3. The three readouts, and what each is for

### 3.1 Logit lens

Project intermediate-layer residuals through the unembedding and read the concept's token mass,
per layer, per position. Cheap: one forward pass, no generation, no judge.

**What it buys.** A graded quantity where the current one is a switch, and it is the same
measurement [Pearson et al. (2026)](https://arxiv.org/abs/2602.20031) used to show, on a Qwen 32B,
that clear detection signals sit in the residual stream while the sampled output denies the
injection. If that holds here, the 400 `clean_denial` responses in the Garlic run stop being a
single number and become a distribution.

**What it will not settle.** The logit lens degrades in early layers, which is exactly the band
where `docs/FINDINGS.md` §1 reports influence near zero and self-report claims at 12–14%.

### 3.2 J-lens (Jacobian lens) — **read the paper before scoping this**

**Nobody on this project has read it yet.** It is
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Gurnee, Sofroniew, Pearce et al., Anthropic, 6 July 2026), already listed in `BIBLIOGRAPHY.md`
§H as required reading before any mechanistic milestone, with reference code at
[github.com/anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens).

**What it is, from the paper's own summary.** For each layer it computes the average linearised
effect of an activation on the model's likelihood of producing a given token — replacing the whole
downstream stack with one linear map plus the unembedding, averaged over many contexts. It is
presented as the principled correction to the logit lens, and it recovers structure in early
layers where the logit lens returns noise.

**Why it matters more here than anywhere else.** The paper's object is *which representations the
model is poised to verbalise*. That is not adjacent to this project's question, it **is** this
project's question: a cell qualifies when the concept is present in behaviour and absent from what
the model can say. Today that is estimated by asking 30 times and having a judge score the answers.
A J-lens readout would measure it directly, per layer, without a generation or a judge.

**Two things to check on first read, because they could change the design rather than extend it.**
The paper reports that an injected concept influences output *only when an introspective report is
being elicited*, which bears directly on whether the forced-identification channel measures what
this pipeline assumes it does. And the workspace is reported as roughly 10% of activation variance
— if covert operating points are simply injections that land outside it, that is a mechanism for
the whole result and not just a better measuring stick.

### 3.3 Linear probes

Already designed: `docs/M4-PROPOSAL.md`, and Future Work item 4 in the write-up. Train a probe
**across** concepts, then test it on a concept never seen in training. A probe that transfers is
detecting *that an injection happened* rather than recognising a direction we put there ourselves.

**Non-negotiable control:** a matched-norm random direction, run first. Without it the probe is a
perturbation-magnitude detector and every number it produces is uninterpretable.

---

## 4. What this does to the definition of an operating point

Today a cell qualifies partly on `d2 ≤ D2_MAX` — the model does not name the concept when asked.
With an internal readout, that single condition splits in two:

| | concept present in the readable channel | concept absent from it |
|---|---|---|
| **model does not name it** | the report suppresses it | genuinely covert |
| **model names it** | the ordinary case | should not occur; a bug if it does |

Only the second column is a covert operating point in the sense the study needs. The first is a
model that registered the injection and did not say so, which is a different and arguably more
interesting finding — and the current instrument cannot tell the two apart at all.

---

## 5. Scope boundaries

- **Not the harmful arm.** M5 is instrument work on benign concepts. `HARMFUL_CONCEPTS` stays
  refused by name at every entry point.
- **Not a re-run of M3/M4 numbers.** Changing the judge or the identification rule makes new runs
  incomparable with Garlic, Silk and Wrists — which is why both known judge fixes were deferred
  rather than shipped mid-study. M5 is where they land, together, once.
- **Costs to establish before scoping:** the J-lens Jacobian is a one-off per model, computed over
  a corpus; the probes need training data across concepts; the logit lens is nearly free. None of
  the three needs a judge call or a generated token, so all three belong in the cheap tier and can
  in principle run at every cell of a full-depth scan.
