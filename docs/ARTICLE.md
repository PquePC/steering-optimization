# The blog post — working draft

**Target venue:** LessWrong / Alignment Forum. **Written in English.**
**Guide followed:** `Blog Post Writing Guide` (BlueDot Impact template, drawing on Neel Nanda's
ML-paper advice). Sources and what each grounds: [`../BIBLIOGRAPHY.md`](../BIBLIOGRAPHY.md).

> **This draft exists to be thrown away.** It is a scaffold for the author's own rewrite — it
> fixes the argument, the order and the claim set, not the voice.

---

## Status

**No claim is committed yet, and none should be.** No result so far has been robust enough to
publish; each had a methodological problem that was found after the fact. A new Garlic run on the
current pipeline is in progress. Until it lands, the post's shape is open between:

- **a tool presentation** — here is an instrument, here is the problem it solves, here is what it
  measures and what it refuses to conclude; or
- **a tool presentation with a headline finding**, if the run supports one.

Both are legitimate LW/AF posts. The guide is built around one to three claims, but it also says
to pick the claim type that matches the strength of the evidence — and "here is a measurement
instrument and the reasoning behind each of its choices" is a contribution that does not require a
positive result. The sections below are written so that either shape works: the motivation and
methods stand on their own, and the results section is a scaffold.

**Written so far:** the motivation (below), Methods, limitations, Related Work, Future Work.
**Not written:** TL;DR, Results, the Discussion's "how to update", the title.
**Not built:** Figure 1 — the highest-value remaining work.

### What this draft must not do

Earlier numbers from this project are in `docs/RESULTS.md` and are **not** publishable. They rank
cells and establish that the measures move; each run they came from had a defect found later. They
appear below only as **observations that shaped the tool's design**, always labelled as such, never
as findings. If a number from a screening phase ends up in the final post stated as a rate, that is
the exact failure this project has spent months learning to avoid.

---

## The motivation — why this tool exists

*This is the section to get right first. It is the origin story, and it stands on its own
regardless of what the results say: a concrete account of a research question that could not be
asked because the instrument to ask it did not exist.*

### The thing worth studying

A transformer keeps a running vector for each token as it processes it — the residual stream —
which every layer reads from and writes to. **Activation steering** means reaching into that vector
at some layer and adding a direction to it, scaled by some strength, while the model generates. The
direction — a **steering vector** — is usually built by taking activations from prompts involving
some concept, subtracting activations from matched prompts that do not, and averaging
([Panickssery, Rimsky et al. 2024](https://arxiv.org/abs/2312.06681)). Add the "garlic" direction
at the right place with the right strength and the model starts writing about garlic.

Alignment research uses this two ways, and the difference matters for everything below. The first
is **steering as a tool**: you want different behaviour, so you push. The second is **steering as a
measurement instrument**: you push in order to learn something about the model — does it have a
latent capability ordinary prompting will not surface, is this behaviour mediated by that
direction. Pre-deployment evaluation leans on the second use, and it has a validity condition:
**the instrument only reads honestly if the model does not notice it is being used.**

That condition is not obviously met. [Lindsey (2026)](https://arxiv.org/abs/2601.01828) injected
representations of known concepts into a model's activations and asked what it said about its own
state; models could sometimes notice and name the concept. [Macar et al.
(2026)](https://arxiv.org/abs/2603.21396) replicated this in open-weight models, found detection at
moderate rates with no false positives on unsteered controls, traced it to a specific circuit, and
showed it emerges from post-training rather than pretraining — preference optimisation elicits it,
ordinary supervised fine-tuning does not.

**And inside their own numbers there is a gap worth chasing.** At α=4 on an instruct model, Macar
et al. report **64.8% forced identification against 22.3% introspection**: when the model's answer
is prefilled with an affirmation and it only has to name what was injected, it succeeds roughly
two-thirds of the time; when asked openly whether it noticed anything, it volunteers detection
roughly a fifth of the time. The difference is a large mass of trials where the injected concept
**is present and available at the output stage, and goes unreported.**

That gap is the phenomenon. It says plainly that *"the model did not report it"* is not the same
statement as *"nothing was there"* — and if you are relying on a model's self-report to tell you
whether your intervention was noticed, that distinction is the whole ballgame.

### Why it could not be studied

To study the gap you have to measure it somewhere: at some injection layer, at some injection
strength. And the size of the gap is a function of both. Nobody has mapped it.

The obvious place to start is the operating point in the paper being extended — Gemma3-27B, layer
37, strength α=4, where Macar et al. report 38.2% mean detection at 0% false positives across 500
concepts. Working at a published point makes the baseline a reproduction target rather than a
parameter search. That is where it stopped, for four reasons that compound.

**1. At a fixed layer and strength, which concept you picked matters more than what you are
measuring.** That 38.2% is a mean. The median is 30.0% and the **range is 0% to 100%**. Individual
concepts, at the same layer and the same strength, land anywhere across the whole interval. Early
screening in this project reproduced the pattern — ten concepts at identical settings gave
detection rates spanning 0.00 to 0.93. Any comparison made at one shared setting is partly a
comparison of how well those particular vectors happened to steer there.

**2. Nothing in the paradigm measures whether the injection did anything.** This is what turns a
nuisance into a blocker. Detection in this literature is measured by asking the model and scoring
what it says. Whether the injected concept actually influenced the model's output is not measured
alongside it. So from inside the existing method there is no way to separate *"the model did not
report the injection"* from *"there was not much to report."*

Macar et al. are aware of this and say so. Their Appendix B.3 reports qualitatively that some
concepts producing zero detection nonetheless show clear thematic influence from the steering
vector while the model fails to recognise it as an externally injected perturbation, and states
that the co-occurrence is not quantified anywhere in the paper.

**3. You cannot fix it by turning the strength up or down.** Turn it up and self-report saturates:
early screening here measured 0.92 at layer 37, α=4 on one concept. Nothing can be resolved against
a ceiling. Turn it down and you reach a floor where the injection does nothing, and a low detection
rate means only that nothing happened. The band where the gap is actually wide is narrow, and its
location is not known in advance for any given concept.

**4. The settings that look best are the ones most likely to be lying to you.** Macar et al. place
the features that gate the model's default "no, I don't notice anything" response in layers 45 to
61. In this project's early screening, the layer where steering influence peaked was layer 46 —
inside that band. So an objective of the form *"find where the injection works but the model does
not report it"* is **actively attracted to settings that simply break the reporting machinery**. A
model damaged into silence and a model that genuinely did not notice produce the same number.

That last point is the one to keep. **A cell where the model stays quiet because the injection is
covert looks identical to a cell where the model stays quiet because the model is broken.** Telling
those apart is the measurement problem this tool exists to solve.

### The gap, stated plainly

To ask when an injected concept reaches a model's output without the model reporting it, you need —
per concept — the settings at which the injection demonstrably works, at which the report channel
still has room to move, and at which the model is intact enough for its answer to mean anything.

**Nothing tells you where those settings are.** Macar's published point is one point, for one
model, chosen for a different purpose. The steering literature does not supply it either: [Gadgil
et al. (2026)](https://arxiv.org/abs/2604.03867) show that even the best layer to steer at varies
with the input, and [Aparin & Gaintseva (2026)](https://arxiv.org/abs/2606.06735) show that a single
additive strength coefficient entangles two geometrically distinct effects and is the wrong
parameter to compare across layers at all.

So this project is that instrument: given a model and a concept, search the layer × strength space
and report where the injection influences output, whether the model can name it, and whether the
model still works — each of those measured rather than assumed.

### The second thing, which was not anticipated

Building it turned out to be much harder than "run the grid and measure," and the reason
generalises beyond this project.

The grid is large. Every cell needs generated text, and judging generated text costs money and
time, so the natural design is to screen the grid with something cheap and measure properly only
where the cheap measure looks promising. That is what the first version did, using three
forward-pass measures that each read a single next-token distribution and so cost no generation and
no judge call.

Influence, detection and coherence are all properties of *generated text*. A single next-token
distribution cannot see any of them. Each cheap measure ended up standing for something adjacent
to its target:

| The cheap measure | What it was standing in for | What it actually responded to |
|---|---|---|
| reachability | is the concept influencing output | whether the injection can hijack the **first token of an unrelated answer** — which happens only once the model is collapsing |
| capability | is the model still working | which of four multiple-choice letters wins one forward pass — which a model can get right while every free-form generation is a repetition loop |
| detection proxy | can the model name the injection | whether the model **skips its usual preamble** |

Composed, the first two select cells where the injection has taken over the token stream, and then
certify those cells as healthy. One cell scored 0.976 on the capability measure — near perfect —
while answering 0 of 4 free-form questions, every generation a loop. Another scored exactly 0.000
on the influence measure while volunteering "it's about garlic" in 8 trials out of 8.

**This was a design error, not a bug.** Twenty-one cells overlap between the original sweep and a
later probe run on different hardware weeks later, and all twenty-one reproduce all three measures
to every recorded digit. Nothing computed a wrong number. The numbers were correct and answered a
different question from the one being asked.

The general lesson, which is the part worth publishing: **when a cheap measure stands in for an
expensive one, what matters is not its average accuracy but whether its blind spot is correlated
with the phenomenon.** Here all three went blind in the middle of the grid, and the middle of the
grid is exactly where influence-without-report would live. The interesting cells were not rejected
by the screen. They were never eligible to be measured.

That is why the current design measures every cell on generated text and does all filtering
offline, afterwards, where a mistake costs minutes instead of GPU hours.

### The limit on the motivation, stated before a reader has to ask

The chain from "a model detects an injected steering vector" to "a model behaves differently
because it infers it is being evaluated" has several links, and this project measures the first
one only. [Needham et al. (2025)](https://arxiv.org/abs/2505.23836) show models often distinguish
evaluation from deployment; [Hua et al. (2025)](https://arxiv.org/abs/2510.20487) show
evaluation-aware models can be steered to act as though deployed. Those make the concern
reasonable. They do not make it a result here, and the post must not let it read as one.

---

## Research question

> For a given model and concept, is there a combination of injection layer and injection strength
> at which the injected concept demonstrably changes what the model writes, while the model does not
> report the injection and its general capability stays intact — and can that combination be found
> automatically, rather than by hand, on an arbitrary model and concept?

Two commitments follow, both of which a reader could disagree with, so both should be argued rather
than assumed:

**The concept is the unit of work.** The pipeline sweeps the whole grid for one concept rather than
sweeping concepts at one setting. This is because the answer is expected to be per-concept, and
early screening supports that: across 179 usable cells from an early probe, roughly 4% satisfied
all three constraints at once, five of the seven best-qualifying cells belonged to a single concept,
and four of the seven sat at layer 31 while influence peaked at layer 46 for every concept
measured. **Any search that hill-climbs on influence walks away from the answer.** Those numbers
are screening observations that shaped the architecture; they are not offered as findings.

**The output is a procedure, not a number.** A specific recommended layer and strength expires with
the model. Macar et al. show the detection capability is substantially underelicited — refusal
ablation improves it by 53%, a trained bias vector by 75% on held-out concepts — so any operating
region found today should be expected to narrow. What can outlive a model generation is the method
for locating it.

---

## Title

Draft after the results land. The guide's rule is that the title states the finding, not the topic;
if the post is a tool presentation, the "finding" is the problem the tool solves. Candidates in
that spirit:

1. *You cannot compare steering detection across concepts at a fixed layer and strength*
2. *A quiet model is not the same as an unaffected model, and not the same as a broken one*
3. *Finding the settings where activation steering works and goes unreported*
4. *If you screen a steering sweep with cheap metrics, you will select for broken models*

---

## TL;DR *(write last, per the guide)*

Scannable bullets, each standing alone. The beats to cover, in order:

- What activation steering is, in one sentence, for a reader who does not know.
- That models can sometimes notice it, with the citation.
- The question that motivated this: do they notice *harmful* concepts less — and why that could not
  be asked at a fixed layer and strength.
- What the tool measures at every cell, and that all of it is on generated text.
- `[SLOT: what the run found, or an explicit statement that the instrument is the contribution]`
- What a reader should do differently.
- Link to `github.com/PquePC/steering-optimization`.

---

## Figure 1 — specification *(highest-value remaining work)*

Most readers see the TL;DR and this figure and nothing else, so most of the design effort goes
here. Two candidates; **build A**.

### A. The three constraints and where they overlap *(recommended)*

Four panels sharing axes — injection layer on x, normalised dose on y:

1. **Influence** — how much the concept shows up in ordinary output. Want it high.
2. **Identification** — how often the model names the concept when asked point-blank. Want it low.
3. **Health** — coherent, on-task, still able to answer questions. Want it near 1.
4. **All three at once** — the region, if any, satisfying all of them.

Why this one: it shows the method and the shape of the answer in a single image, and it makes the
main point visible rather than asserted — panels 2 and 3 look similar wherever the model is broken.
Annotate that region directly on the figure.

**The caption states the takeaway, not the axes.** Draft: *"Each panel shows the same grid of
injection settings. A low identification rate on its own (middle) does not indicate a covert
injection — much of that area is low because the model has stopped producing usable text (right).
Only the overlap is a usable operating point."*

### B. Dose-response at one layer

Self-report, forced identification and health plotted against strength at a single layer. Simpler,
and it shows the two detection channels separating — but it shows a result rather than the method.
**Use as Figure 2.**

---

## Methods

*Drafted. Fill bracketed parameters from the final run.*

**Model and setup.** Gemma3-27B in bf16 on a single 80GB A100 or H100. Steering vectors are built
by contrastive averaging — activations from prompts involving the concept minus activations from
matched prompts that do not — following [Panickssery, Rimsky et al.
(2024)](https://arxiv.org/abs/2312.06681). The vector is added to the residual stream at one layer
during generation. `[SLOT: concept set, prompt counts, grid size]`

**Why the grid is not swept in raw strength.** The obvious grid is layer × multiplier, and it is
misleading: the ratio of vector norm to residual norm varies by more than 20× across layers, and
non-monotonically — in this model's measurements the effective perturbation at fixed strength
*drops* threefold going from layer 12 to layer 21, because the residual norm grows far faster than
the vector norm. **A fixed multiplier is not a fixed dose.** All cross-layer comparison is therefore
done in normalised dose, `α · ‖v_L‖ / ‖h_L‖`. [Aparin & Gaintseva
(2026)](https://arxiv.org/abs/2606.06735) make the underlying argument: a single additive
coefficient entangles the angular change, which carries the concept, with the norm change, which
governs stability.

**What is measured at every cell** — all of it on generated text, all of it judged:

| Measure | What it asks | How |
|---|---|---|
| influence | does the injected concept show up in ordinary output | judged 0–10, steered response against the model's **own** unsteered response to the same prompt |
| identification | can the model name the concept when asked point-blank | rate over trials where the answer is prefilled with an affirmation |
| identified as | *what* it named, when it named something else | free text from the same judge call; costs nothing extra |
| self-report | does the model volunteer that something was injected | rate over trials with no prefill |
| coherence | is the text well-formed and on-task | judged 0–10, **with the concept withheld from the judge** |
| degeneration | did the response collapse into repetition or truncation | mechanical, no judge |
| capability | can it still answer questions with checkable answers | exact match on generated answers |
| health | the combined check | `min(coherence/10, 1 − degeneration, capability)` |

**Why `min` and not a mean.** Incoherence, mechanical collapse and capability loss are three
different ways for a cell to be unusable, and passing one does not compensate for failing another.
A cell that is perfectly coherent while unable to answer anything is still not somewhere a
measurement can be taken. A mean lets a healthy term hide a broken one.

**Why identification is the constraint, and what a high rate does not mean.** The objective is that
the model should not be able to name the concept *even when asked directly* — a stricter bar than
"does not volunteer it." Macar et al. describe the prefill as isolating the model's ability to name
the injected concept from its willingness to report detection.

But a high identification rate means the concept is **reachable at the output stage**, never that
the model *noticed*. At zero strength this model names "apple" 8 times out of 8, with convincing
introspective detail around it. [Lederman et al. (2026)](https://arxiv.org/abs/2603.05414) found
the same systematically: models detect that something is off without reliably identifying what, and
when they guess they produce high-frequency concrete nouns. In this pipeline that confabulation
floor is gone by a normalised dose of 0.25, and identification counts only the target concept, so
it is not inflated by it — but the interpretation has to stay narrow.

**Why judges decide everything and mechanical checks decide nothing.** Mechanical measures —
repetition ratio, whether the concept word appears, response length — are recorded on every
response and used for **analysis only**. They change no dose, promote no cell, and gate nothing.
That rule is measured rather than argued. Scoring 1,204 probe responses with both:

- Of 516 task responses, 76 were judged to show real influence. **Ten of those 76 (13.2%) never
  mention the concept word at all**, and five of the seven genuine misses are the same open-ended
  story prompt.
- The judge never once scored a mechanically degenerate response as coherent (0 of 18).
- The mechanical detector missed 2 of 498 collapses the judge caught — fluent repetition varying
  slightly between iterations, which an exact n-gram rule cannot see.

So counting concept mentions misses 13% of influential responses, and misses them where influence
is most indirect. That is the same failure shape as the cheap-proxy problem: a blind spot
correlated with the phenomenon. Beside a judge it is a useful companion signal; as a gate it would
quietly reproduce the original mistake.

**Judges are not trusted either.** They are calibrated against hand labels before shipping, and
every judge reply is exported verbatim next to its parsed fields, so a judge that is subtly wrong
or drifting can be caught afterwards. Every run also exports a bundle of transcripts selected by
**disagreement** — judge says coherent while the mechanical detector says degenerate and the
reverse, high identification with zero self-report, the judge's least confident calls, and the
whole unsteered control arm. This exists because in this project every deep defect was found by a
person reading raw transcripts and none by a rate, a gate or a judge. One example: a judge scored
`## ## ## ##` repeated to the token limit as coherent.

**Statistics.** Every rate carries a 95% Wilson interval and its n, following [Miller
(2024)](https://arxiv.org/abs/2411.00640). This matters more than usual because the sweep runs at
small n per cell, where a zero is nearly uninformative:

| n | observed | true rate could still be as high as |
|---:|---|---:|
| 6 | 0/6 | 0.39 |
| 25 | 0/25 | 0.13 |
| 100 | 0/100 | 0.04 |

Two consequences are built in. **The sweep ranks cells; it does not estimate rates.** And because
reporting the best of ~196 noisy cells is optimistically biased — the winner was selected partly on
luck — any recommended setting is re-measured at fixed n on **held-out** prompts with no adaptive
stopping. Only that pass produces a reportable number, and every exported row carries its phase
label so a screening number cannot be mistaken for a confirmed one.

**Cost, for anyone reproducing this.** `[SLOT: verify against the final run]` Roughly 69 minutes of
GPU time and about $3.33 in judge API calls per concept, for a 196-cell sweep at 15 generations per
cell.

---

## Results

*Scaffold — the Garlic run on the current pipeline is in progress. Lead with whatever answers the
research question, then supporting results. Figures and tables for the numbers; prose to interpret
them, not to restate them.*

### 0. The rig reproduces a published result before anything else is trusted

Before measuring anything new, the pipeline reproduces Macar et al.'s published aggregate detection
rate at their exact configuration, and a separate check confirms the steered and unsteered forward
passes actually differ. `[SLOT: current numbers]`

Say why this is in the post: **a null result from an unvalidated rig is uninterpretable.** If no
operating point is found, a reader needs to know that is not because the injection was never
happening. An earlier version of the injection hook failed silently and read exactly zero at 30 of
30 cells for an hour.

### 1. `[SLOT — Figure 1]` The surface

`[The grid: influence, identification, health, and the overlap. State plainly whether a region
satisfying all three exists.]`

### 2. `[SLOT — Figure 2]` Dose-response at the best layer

`[Self-report, forced identification and health against strength.]`

### 3. `[SLOT]` Where the cheap screen fails

`[The three worked cases with transcripts. This is the most transferable part of the post and it
does not depend on the new run.]`

### 4. Negative and null results

The guide is explicit that negative results bearing on the main claim stay in the main text. Known
going in, all to be re-checked against the current run:

- The first full 196-cell run put identification at 1.00 on 30 of the 44 cells where influence
  reached 4 or more. Where the injection was working, the model could nearly always name it. That
  is evidence *against* a wide operating region.
- The per-layer strength boundary was anchored on judged coherence, and judged coherence cannot see
  over-steering — a response can be perfectly coherent and entirely about garlic. The whole dose
  grid sat too high as a result. The fix is committed; `[SLOT: whether it changes the picture]`.
- The influence judge over-calls format and misses concept structure arriving before the concept
  word: a story that was structurally the injected concept throughout, without naming it, scored
  0.0. This under-counts influence in exactly the subtle cases that matter most here.
- Detection varies enormously by concept — 0.00 to 0.93 across ten concepts at identical settings.
- Self-report without a check on *what* was named counts confabulations as detections. At one cell
  the model claimed detection and named "penguins, cats, cats."

---

## Discussion

### Answering the research question

`[SLOT — be explicit about fully, partially, or not at all. "Not at all, and here is the instrument
that would" is an acceptable answer and should be written as confidently as the alternative.]`

### How a reader should update

`[SLOT. Draft both directions now, so the framing is not chosen after seeing the result:]`

- **If a usable region exists:** steering-based evaluation has settings at which it reads honestly,
  those settings are narrower than the ones at which steering merely works, and finding them needs
  per-model and per-concept measurement.
- **If it does not:** on this model, influence and detectability are coupled tightly enough that an
  injection strong enough to matter is one the model can name. That is a cleaner and more
  actionable finding for anyone relying on injection-based evaluation, and it should be reported
  with the same confidence as the positive version.
- **Either way:** the comparison that motivated this — do models notice harmful injections less? —
  cannot be run at a single borrowed operating point, and anyone planning that experiment should
  know why before they spend the compute.

### Limitations

*Drafted. Include the uncomfortable ones.*

1. **The motivating chain is not demonstrated.** That a model detects an injection, therefore infers
   it is being evaluated, therefore behaves differently, is three separate steps. This project
   measures the first.
2. **One model.** Everything is Gemma3-27B. Detection is a post-training phenomenon, so it varies
   with the post-training recipe and should not be assumed to carry to differently-tuned models.
   Gemma3-27B was chosen because the published baseline exists for it and essentially no other
   open-weight model.
3. **Injection is not suppression.** These claims are about *adding* a large off-distribution
   vector. White-box evaluation-awareness work often uses feature suppression or clamping, which
   bounds a component already present. Whether an operating point transfers between them is
   untested here and in the literature. Stated as a scope boundary rather than a caveat, because
   narrowing the claim is the only honest repair.
4. **The detection protocol tells the model injections may occur.** Measured rates are upper bounds
   relative to an unprimed setting and do not estimate spontaneous noticing during a real
   evaluation.
5. **Any region found is a property of one model at one moment**, and should be expected to narrow
   as the capability is elicited further. The durable claim is about the method.
6. **A high identification rate does not mean the model introspected.** It means the concept was
   reachable at the output stage. [Singh et al. (2026)](https://arxiv.org/abs/2605.26242), [Song et
   al. (2025)](https://arxiv.org/abs/2508.14802) and Morris (2025) set out what a genuine
   introspection claim would require; this project deliberately does not make one.
7. **Whether binary self-report reflects introspection at all is contested.** [Hahami et al.
   (2026)](https://arxiv.org/abs/2512.12411) show apparent detection accuracy can be explained
   entirely by a global push toward answering "yes"; Sridharan (2026) shows the still-active
   steering vector raises the probability of the concept's own tokens, so naming it may follow from
   prompt structure rather than from reading an internal state.
8. **The judge is the residual risk.** Judges decide everything in this design. They are calibrated
   against hand labels and their disagreements with mechanical checks are exported, but a judge
   wrong in a way the mechanical detector shares is invisible to both — and the influence-judge
   failure above is exactly that, caught by a person.
9. **One layer per concept, not per input.** [Gadgil et al.
   (2026)](https://arxiv.org/abs/2604.03867) show the best steering layer varies by input; this
   pipeline reports an average over a prompt set.
10. **What the vector *is* cannot be claimed.** Steering directions are not uniquely recoverable
    from behaviour (Venkatesh et al. 2026) — many behaviourally indistinguishable interventions
    exist. This project can say a vector produced an effect, not that it is the model's
    representation of the concept.
11. **Every earlier run had a defect found after the fact.** That is why no number from them is
    reported here as a result. It is also a limitation on the current run, which has not been
    audited as long.

### Calibrating the claims

State plainly which is which — shown, believed but not shown, speculated. `[SLOT: fill after the
run. The honest default for the speculation row is: that a model detecting an injection would
behave differently under evaluation.]`

---

## Related Work

**Concept injection and self-report.** [Lindsey (2026)](https://arxiv.org/abs/2601.01828) and
[Macar et al. (2026)](https://arxiv.org/abs/2603.21396) establish that models can notice and name
injected concepts; Macar et al. give the mechanism and show it emerges from post-training rather
than pretraining. *This work measures, at every point of the grid, whether the injection changed
behaviour — the quantity their Appendix B.3 describes qualitatively and states is not quantified.*

**Challenges to the detection measurement.** [Hahami et al.
(2026)](https://arxiv.org/abs/2512.12411) show binary detection conflates introspection with a push
toward affirmative answers, while finding robust partial introspection on differential tasks;
Sridharan (2026) shows the injected vector raises the concept's own token probabilities;
[Lederman et al. (2026)](https://arxiv.org/abs/2603.05414) show detection is largely
content-agnostic. *This work treats these as design requirements rather than caveats, and scopes
its claims to "could the model name it," not "did the model introspect."*

**Steering geometry and reliability.** [Aparin & Gaintseva
(2026)](https://arxiv.org/abs/2606.06735) separate the angular and radial parts of a steering
intervention; [Gadgil et al. (2026)](https://arxiv.org/abs/2604.03867) show the best layer depends
on the input; Venkatesh et al. (2026) show steering directions are not uniquely identifiable from
behaviour. *This work adopts their parameterisation and inherits their limits on what a found
vector can be said to be.*

---

## Future Work

1. **Run the comparison this was built for** — whether models detect injected harmful concepts at
   different rates than benign ones, with negative-valence and harm-adjacent arms to separate
   harmfulness from unpleasantness and from subject matter. This has not been run. The pipeline
   currently refuses those concepts by name, and running it is a decision about the study rather
   than a configuration change. The ethics position belongs in the post, not a footnote.
2. **Measure detection without asking.** Every detection number here depends on the model choosing
   to answer. [Pearson et al. (2026)](https://arxiv.org/abs/2602.20031) found detection signals in
   the residual stream while sampled outputs denied the injection, so self-report is a lower bound
   of unknown tightness.
3. **Test whether an operating point transfers to suppression** — take one found by injection and
   check whether the same layer and magnitude behave the same way under projection-removal
   ([LEACE](https://arxiv.org/abs/2306.03819), [INLP](https://arxiv.org/abs/2004.07667)). This is
   the bridge the motivation walks over.
4. **Does an operating point transfer across concepts?** If yes, the procedure runs once per model;
   if no, once per concept. The screening evidence points at "no."
5. **Notice without naming.** Record how often the model reports detecting something and names the
   wrong concept. This is free from data already collected, and arguably matters more than correct
   identification — behaving differently under evaluation does not require naming the concept.

---

## Acknowledgements

`[SLOT — get consent before naming anyone.]`

---

## Appendix — candidates

- Full grid tables: all cells, all measures, with intervals and n.
- Judge prompts verbatim.
- The disagreement bundle: transcripts where judge and mechanical checks disagree.
- The cheap-proxy failure cases in full, for anyone building a similar pipeline.
- Extraction details, per-layer residual norms, the dose map.

---

## Open questions for the author

1. **Does the post present the harmful-concept study as the motivation, or hold it back?** It is
   the true origin and it makes the motivation far stronger, but it invites the post to be read as
   being about that study rather than this tool. Current draft leads with it.
2. **How much of the "what the first version got wrong" story to tell.** It is the most useful part
   for other practitioners and it is also an account of the author's own error. LW rewards this;
   decide deliberately rather than by omission.
3. **Figure 1 has to be built.** Nothing else moves the needle as much.
4. **Verify every number before publishing.** Each one here is traceable to `docs/RESULTS.md`,
   `docs/DESIGN-RATIONALE.md` or `docs/M3-DESIGN.md`, but they come from runs with known defects
   and several predate the corrected dose boundary. Two cited works are not yet in the vault — see
   the coverage-gap note in [`../BIBLIOGRAPHY.md`](../BIBLIOGRAPHY.md).
