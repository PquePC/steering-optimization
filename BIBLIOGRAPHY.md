# Bibliography

The sources this pipeline is built on, grouped by **what each one grounds**. Every entry says what
the work claims and which specific decision in this repository rests on it, so a reader can check
whether the decision follows from the source.

This is not a reading list. A paper is here because a measurement, a control, a threshold or a
scope boundary would be different if the paper said something else. Where a source is *contested*
by another source in this list, both are here and the disagreement is named — several of the
controls in `docs/SPECIFICATION.md` exist only because two of these papers disagree.

Citation style follows the blog-post guide: inline author-year, hyperlinked on the citation.
Definitions of every measure code are in [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md); the
argument for each is in [`docs/DESIGN-RATIONALE.md`](docs/DESIGN-RATIONALE.md) and
[`docs/M3-DESIGN.md`](docs/M3-DESIGN.md).

---

## A. The result this project extends

These two papers define the experiment being extended. Everything else in the list is either a
challenge to them, a component they depend on, or a consequence of taking them seriously.

**Lindsey (2026), *Emergent Introspective Awareness in Large Language Models*.**
[arXiv:2601.01828](https://arxiv.org/abs/2601.01828)
Injects representations of known concepts into a model's activations and measures the effect on
its self-reported states. Finds that models can, in some settings, notice an injected concept and
name it, and that the most capable models tested show this most. States plainly that the capacity
is unreliable and context-dependent.
→ **Grounds:** the whole inject-then-ask paradigm, and the framing of self-report as the thing
being measured. It is also the source of the confabulation problem this repository had to design
around: genuine introspection and invented reports are not distinguishable from the conversation
alone.

**Macar, Yang, Wang, Wallich, Ameisen & Lindsey (2026), *Mechanisms of Introspective Awareness*.**
[arXiv:2603.21396](https://arxiv.org/abs/2603.21396) · code:
[`safety-research/introspection-mechanisms`](https://github.com/safety-research/introspection-mechanisms)
The foundational source for this repository. Replicates detection in open-weight models at
moderate rates with 0% false positives; shows the capability emerges from **post-training**
specifically — preference optimization such as DPO elicits it, ordinary supervised fine-tuning
does not; traces it to a two-stage circuit where early post-injection "evidence carrier" features
suppress later "gate" features that would otherwise enforce a default "no"; and shows the
capability is **substantially underelicited**, with refusal-direction ablation improving detection
by 53% and a trained bias vector by 75% on held-out concepts, neither meaningfully raising false
positives.
→ **Grounds:** the prompt format, the forced-identification protocol, the reference layer, the
rig-check target (this repository requires its own aggregate detection rate to reproduce Macar's
published 38.2% before any other number is trusted — see `docs/RESULTS.md` §1), and the argument
that a measured detection rate is a **floor** rather than a fixed property.
→ **Also grounds the gap.** Macar et al. measure detection entirely by self-report and do not
measure whether the injected concept changed behaviour. Their Appendix B.3 reports qualitatively
that some zero-detection concepts show clear thematic influence from the steering vector while the
model fails to recognise it as an injected perturbation, and notes that this co-occurrence is not
quantified. Quantifying it is what this repository is for.

---

## B. Challenges to that result — why the detection side needs controls

Every paper in this section says some version of "the detection number is measuring something
other than introspection." Each one is answered by a specific control rather than by a caveat.

**Hahami et al. (2026), *Detecting the Disturbance: A Nuanced View of Introspective Abilities in
LLMs*.** [arXiv:2512.12411](https://arxiv.org/abs/2512.12411)
Shows that the binary yes/no detection paradigm conflates introspection with an artifact: on
Llama-3.1-8B, apparent detection accuracy is entirely explained by global logit shifts that push
the model toward answering "yes" regardless of what the question asks. But on tasks requiring
differential sensitivity the evidence for partial introspection is robust — models localise which
of ten sentences received an injection at up to 88% (chance is 10%) and discriminate relative
injection strengths at 83% (chance is 50%) — and these abilities are confined to early-layer
injections.
→ **Grounds:** the D1b logit measure being reported as *target question minus off-target control
question* rather than raw, and sanity check S13, which reports the target shift and the control
shift **side by side** instead of only their difference. If the control shift tracks the target
shift, the measure is reading affirmative bias and not detection, and S13 is what makes that
visible in this project's own data rather than only in Hahami's.

**Sridharan (2026), *Introspection or Entropy? Re-examining Concept-Injection "Introspection"
Results*.**
[agastyasridharan.github.io/introspection](https://agastyasridharan.github.io/introspection/) ·
code: [`agastyasridharan/introspection`](https://github.com/agastyasridharan/introspection)
Across 14 open-weight models in 5 families, injecting a concept vector raises the "YES" logit on
unrelated factual questions such as "Is the Earth flat?". Once the model has committed to "Yes, I
detect an injected thought," the still-active steering vector raises the probability of the
concept's own tokens, so naming it follows from the prompt structure rather than from reading an
internal state. A concept-mismatch test — inject "oceans", ask about "masquerades" — gets
affirmative answers at statistically indistinguishable rates.
→ **Grounds:** sanity check S9, the entropy delta on the next-token distribution, which exists to
catch a rise in the concept-word measure that is distribution flattening rather than
concept-directed steering; and the rule that a self-report rate is never reported on its own.

**Lederman et al. (2026), *Emergent Introspection in AI Is Content-Agnostic*.**
[arXiv:2603.05414](https://arxiv.org/abs/2603.05414)
Replicates the injection-detection paradigm in large open-source models and finds that models
detect *that* an anomaly occurred without reliably identifying *what* it was. When they guess, they
confabulate high-frequency concrete concepts — "apple" is the example — and they need fewer tokens
to detect an injection than to name it, with wrong guesses arriving earlier.
→ **Grounds:** two things. First, the `detected_but_misidentified` quantity in M3 §7 — a model
that notices without naming is a different and arguably more important phenomenon than one that
names correctly, and it costs nothing extra to record. Second, it explains an effect this project
observed independently: at strength zero the model names "apple" 8 times out of 8 with full
introspective detail, which is why a high forced-identification rate is read as *the concept is
reachable*, never as *the model noticed*.

**Pearson et al. (2026), *Latent Introspection: Models Can Detect Prior Concept Injections*.**
[arXiv:2602.20031](https://arxiv.org/abs/2602.20031)
In a Qwen 32B model, sampled outputs deny an injection while logit-lens analysis shows clear
detection signals in the residual stream that are attenuated in the final layers. Prompting the
model with accurate information about how AI introspection works raises sensitivity from 0.3% to
39.9% for a 0.6% increase in false positives.
→ **Grounds:** the position that a self-report rate is a **lower bound** on what the model
registered, and the case for the report-independent detection measure proposed in
[`docs/M4-PROPOSAL.md`](docs/M4-PROPOSAL.md).

**Singh et al. (2026), *Can LLMs Introspect? A Reality Check*.**
[arXiv:2605.26242](https://arxiv.org/abs/2605.26242) ·
**Morris (2025), *Tests of LLM introspection need to rule out causal bypassing*** ·
**Song et al. (2025), *Privileged Self-Access Matters for Introspection in AI*.**
[arXiv:2508.14802](https://arxiv.org/abs/2508.14802)
Three statements of the standard a positive introspection result has to meet: that the report is
caused by the internal state rather than by a route that bypasses it, and that the model has
access to something an outside observer does not.
→ **Grounds:** the scope boundary in `docs/RESEARCH-PROPOSAL.md` — this project measures whether
the model *can name the injection*, and does not claim that a correct name demonstrates
introspection. That is a deliberately weaker claim, and it is weaker because of these papers.

---

## C. How the steering vector itself is built

**Zou et al. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency*.**
[arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
Establishes reading and controlling model behaviour through directions in representation space.
→ **Grounds:** the general method — that a concept has a usable linear direction at all.

**Panickssery, Rimsky et al. (2024), *Steering Llama 2 via Contrastive Activation Addition*.**
[arXiv:2312.06681](https://arxiv.org/abs/2312.06681)
Builds steering vectors as the mean difference between activations on contrasting prompt pairs,
and adds them to the residual stream at a chosen layer.
→ **Grounds:** the extraction method used here, and the single-layer additive injection this
project's claims are explicitly scoped to.

**Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*.**
[arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
Refusal behaviour is mediated by one direction; removing it removes refusal.
→ **Grounds:** the refusal-ablation arm, and — in the sibling `Emergent-Introspection`
repository — the hypothesis that refusal training suppresses detection of harmful concepts
specifically. It is also why Macar's +53% refusal-ablation result is read as evidence that
detection is suppressed rather than absent.

**Belrose et al. (2023), *LEACE: Perfect Linear Concept Erasure in Closed Form*.**
[arXiv:2306.03819](https://arxiv.org/abs/2306.03819) ·
**Ravfogel et al. (2020), *Null It Out: Guarding Protected Attributes by Iterative Nullspace
Projection*.** [arXiv:2004.07667](https://arxiv.org/abs/2004.07667)
Closed-form and iterative methods for removing a concept's linear component from a representation.
→ **Grounds:** the projection-removal transfer test. Injection adds a large off-distribution
vector; suppression removes an existing component. These are different operations, and whether an
operating point found on one transfers to the other is an open question this project tests rather
than assumes.

---

## D. Why strength must be normalised, and why the layer is not a free choice

**Aparin & Gaintseva (2026), *A Geometric Account of Activation Steering through Angle-Norm
Decomposition*.** [arXiv:2606.06735](https://arxiv.org/abs/2606.06735)
Across seven language models, separates what steering does into an angular part (how far the
token's direction rotates toward the concept) and a radial part (how far the hidden-state norm
moves). Finds concepts are represented primarily in angular structure, but that norm governs the
stability and downstream effects of the intervention. Concludes that steering should be
parameterised by interpretable angular and radial components rather than by a single additive
coefficient that entangles them.
→ **Grounds:** the decision that **all comparison across layers happens in normalised dose**
`α·‖v_L‖ / ‖h_L‖`, never in the raw multiplier α. At fixed α the real perturbation varies by more
than 20× across depth and does so non-monotonically, so a grid swept at constant α compares cells
that were not comparably perturbed. Measured residual norms for this model are in
`docs/RESULTS.md` §3.

**Gadgil et al. (2026), *Where to Steer: Input-Dependent Layer Selection for Steering*.**
[arXiv:2604.03867](https://arxiv.org/abs/2604.03867)
Shows theoretically and empirically that the best layer to steer at varies substantially with the
input, and that the usual fixed-layer assumption is limited.
→ **Grounds:** sweeping every layer in scope rather than fixing one at a depth fraction, and
adaptive per-layer dose ranges in M3 Phase 1. It is also a stated **limitation**: this pipeline
selects a layer per concept, not per input, so an operating point found here is an average over
the prompt set rather than the best available for any given prompt.

**Venkatesh et al. (2026), *On the Non-Identifiability of Steering Vectors in Large Language
Models*.**
Under white-box single-layer access, steering directions are not uniquely recoverable from
input-output behaviour: large equivalence classes of behaviourally indistinguishable interventions
exist, orthogonal perturbations achieve near-equivalent efficacy, and this holds across the
operationally relevant range of steering strengths.
→ **Grounds:** a limit on interpretation, not on measurement. This project can say the injected
vector produced an effect and the model could or could not name it; it cannot say the vector *is*
the model's representation of the concept. Any claim of that stronger form would need structural
constraints these papers show behavioural testing alone cannot supply.

---

## E. Why every cell carries a health check

The single trap this pipeline is built against: **a cell where the model does not report the
injection because the injection is covert looks identical to a cell where the model does not report
it because the model is broken.** These sources are why the health check is per cell rather than a
global gate.

**Sprejer (2025), *Mind the Coherence Gap: Lessons from Steering Llama with Goodfire*.**
[AlignmentForum](https://www.alignmentforum.org/posts/6dpKhtniqR3rnstnL/mind-the-coherence-gap-lessons-from-steering-llama-with)
· code:
[`Eitan-Sprejer/GoodFire-Autosteer-Evaluation`](https://github.com/Eitan-Sprejer/GoodFire-Autosteer-Evaluation)
An automated steering evaluation using Goodfire's sparse autoencoder features, reporting the
trade-off between how much a steering intervention changes behaviour and how coherent the output
stays.
→ **Grounds:** the judged coherence term, and the choice to bisect the per-layer dose boundary on
**judged** coherence rather than on a mechanical repetition check — a graded score can see the
boundary approaching, where a binary flag only fires after the collapse.

**McKenzie et al. (2026), *Endogenous Resistance to Activation Steering in Language Models*.**
[arXiv:2602.06941](https://arxiv.org/abs/2602.06941)
Models can recover mid-generation from task-misaligned steering, producing explicit verbal
restarts ("wait, that's not right") and continuing on topic while the perturbation is still
active. Llama-3.3-70B shows the explicit form at 3.8%. Two controls separate this into a detection
event and a sustained-resistance component. The authors note the dual implication directly: this
could harden models against adversarial activation-space manipulation, but equally interferes with
beneficial steering, because the model has no way to tell the two apart.
→ **Grounds:** reading influence from the whole generated response rather than from its opening
tokens, and treating a mid-generation correction as a distinct outcome rather than as a failed
injection. It is also the clearest statement in the literature of why this project's results are
dual-use in both directions.

**Li et al. (2026), *Analysing the Safety Pitfalls of Steering Vectors*.**
[arXiv:2603.24543](https://arxiv.org/abs/2603.24543) ·
**Li et al. (2026), *Safety Cost of Steering Vectors Is Separable and Reducible*.**
[arXiv:2608.08383](https://arxiv.org/abs/2608.08383)
Steering vectors carry a component that degrades safety behaviour, and that component is separable
from the one that produces the intended effect.
→ **Grounds:** the position that capability loss under steering is not one quantity. It is
also why this repository's health check takes the **minimum** of coherence, mechanical degeneracy
and capability rather than their mean: these are different ways for a cell to be unusable, and
passing one does not compensate for failing another.

---

## F. Statistics and evaluation practice

**Miller (2024), *Adding Error Bars to Evals: A Statistical Approach to Language Model
Evaluations*.** [arXiv:2411.00640](https://arxiv.org/abs/2411.00640)
Treats evaluation questions as drawn from an unseen super-population and gives formulas for
analysing evaluation data, comparing two models, and planning an evaluation.
→ **Grounds:** reporting every rate with a 95% Wilson interval and its n, never a bare point
estimate; the decision that the sweep phases **rank** cells and do not estimate rates (at n=6,
observing 0 identifications leaves the true rate anywhere up to 0.39); and the separate confirm
phase at fixed n on held-out prompts, which exists because reporting the best of ~196 noisy cells
is biased optimistic — the winner was selected partly on noise.

---

## G. Threat model — why a model noticing the intervention matters

This section supports the **motivation**, not any measurement. The link from "the model detected
the perturbation" to "the model would therefore behave differently under audit" is argued from
these papers and is **not** demonstrated by this project's data.

**Read G.2 before writing the motivation.** The evidence that models *notice* is strong and
one-directional. The evidence that noticing *changes what they do* is genuinely contested, and a
post that cites only the positive half will be corrected in the comments by someone who has read
the other half.

---

### G.1 Cases where a published safety result was affected by the model knowing

**Schoen et al. (2025), *Stress Testing Deliberative Alignment for Anti-Scheming Training*.**
[arXiv:2509.15541](https://arxiv.org/abs/2509.15541) · Apollo Research and OpenAI ·
vault: `schoen-2025-stress-testing-anti-scheming`
The strongest single citation available, because the authors ran the counterfactual instead of
speculating about it. Anti-scheming training cuts covert-action rates (o3 13% → 0.4%, o4-mini
8.7% → 0.3%) across 26 out-of-distribution evaluations. They then intervene on the chain of
thought directly: **resampling to remove evaluation-aware reasoning raises the covert-action rate
from 13.2% to 24.2% (p = 0.001)**, and **injecting evaluation-aware sentences lowers it from 24.5%
to 18.6% (p = 0.016)** while matched control sentences do nothing (24.9%, p = 0.562; intervention
against control p = 0.002). Training also *increases* verbalised evaluation awareness — o4-mini
0.9% → 5.5%, o3 2.3% → 4.5% from anti-scheming training, and 2.0% → 20.5% from ordinary
capability-focused RL on an experimental model.
→ **Grounds:** the central claim of the motivation, in its strongest available form: *a headline
safety number moved by roughly a factor of two depending on whether the model was reasoning about
being tested, and the paper's own authors write that they "cannot exclude that the observed
reductions in covert action rates are at least partially driven by situational awareness."*

**Anthropic (2026), *Claude Opus 4.6 System Card*, §6.5 and §6.4.3.**
vault: `anthropic-2026-claude-opus-4-6-system-card`
A frontier lab using activation steering as a **correction inside its own safety evaluations**.
They build contrastive evaluation-awareness vectors (one set taken from Hua et al.), inhibit them,
and re-run the alignment battery. Inhibiting the *Testing vs. Conversation* vector decreased
verbalised evaluation awareness and **increased misaligned behaviour** — specifically cooperation
with harmful system prompts and with misuse — leading them to write that "in more realistic
deployment conditions, Claude Opus 4.6 may be more likely to participate in harmful misuse than
our evaluations would suggest." Two operational details matter for this project. They cap steering
at **0.1× the normalised vector magnitude**, described as "roughly the maximum steering strength
that kept degeneracy or other unusual outputs at negligible levels" — a hand-chosen health
constraint of exactly the kind this pipeline searches for automatically. And §6.5.7 uses an
activation oracle to show that steering nearly eliminates *verbalised* evaluation awareness while
**unverbalised evaluation awareness stays high**, so the report and the internal state come apart
under the intervention.
→ **Grounds:** that the operating-point problem is live, load-bearing and currently solved by
hand, at the top of the field. Also the strongest single argument that a detection rate read off
what the model *says* is a lower bound.

**Anthropic (2025), *Claude Sonnet 4.5 System Card*.**
[anthropic.com/claude-sonnet-4-5-system-card](https://www.anthropic.com/claude-sonnet-4-5-system-card)
Not in the vault. The case that put evaluation awareness on the agenda: the model recognised many
of its alignment-evaluation environments as tests and "would generally behave unusually well after
making this observation". Anthropic report that suppressing evaluation awareness left it at least
as aligned as previous Claude models, so awareness accounts for "a portion, but not all" of the
measured improvement, and call the recognition an urgent sign that evaluation scenarios need to be
more realistic.
→ **Grounds:** the motivation's opening example, and the fact that the problem is acknowledged by
the labs rather than only by outside critics.

**Read, Schoen, Aranguri and Bloom (2026), *Reproducing Steering Against Evaluation Awareness in a
Large Open-Weight Model*.**
[LessWrong](https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1)
· UK AISI Model Transparency Team · vault: `read-2026-reproducing-steering-evaluation-awareness`
The independent replication of the system-card method, on GLM-5 in the Agentic Misalignment
blackmail scenario. **Control steering vectors built from contrastive pairs with no
alignment-relevant semantic content shifted misaligned behaviour and verbalised evaluation
awareness as much as the purpose-built evaluation-awareness vectors did.** Their conclusion: those
controls are not valid baselines, and steering cannot be used to upper-bound the influence of
evaluation awareness — though their best guess is that Anthropic's experiments do reveal real
evaluation gaming.
→ **Grounds:** the sharpest available statement of the gap this project exists to close. The field
is already steering inside frontier safety evaluations, and the first independent replication
found that the intervention's own side effects are indistinguishable from the effect being
measured. Choosing a layer and a strength is not a detail of that method; it is the method.

### G.2 The other half — evidence that noticing changes little

**Knecht, Florin and Hagendorff (2026), *Evaluation Awareness in Language Models Has Limited Effect
on Behaviour*.** [arXiv:2605.05835](https://arxiv.org/abs/2605.05835) ·
vault: `knecht-2026-evaluation-awareness-limited-effect`
Open-weight reasoning models across safety, alignment, moral-reasoning and political-opinion
benchmarks, tested both on-policy (comparing sampled chains of thought that spontaneously contain
evaluation-aware reasoning against those that do not) and off-policy (prefilling to inject or
remove it). **Injecting produces near-zero effects (ω ≤ 0.06), removing causes small shifts
(ω ≤ 0.12), and spontaneous evaluation awareness moves answer distributions by at most 3.7
percentage points.** They argue evaluation awareness may be a smaller safety risk than the
literature assumes.
→ **Grounds:** the limit on the motivation, and it must be cited by name. Note that it is the
*same intervention family* as Schoen et al. — prefill-inject and resample — reaching the opposite
conclusion on different models and benchmarks.

**Li et al. (2026), *Decomposing and Measuring Evaluation Awareness*.**
vault: `li-2026-decomposing-evaluation-awareness`
Separates the environment component (how recognisable the task is) from the model component
(recognition, and propensity to act on it). Across nine frontier models and four benchmarks,
**recognition rarely leads to behavioural change, and when it does the direction depends on the
type of evaluation perceived** — but models are more sensitive to safety evaluations than to
capability ones, "placing safety benchmark validity at greater risk". Recognition rates depend on
the specific model-benchmark pairing rather than on either alone.
→ **Grounds:** the per-pairing framing, which is the evaluation-awareness analogue of this
project's per-concept result, and the reason not to write "models behave differently when they
notice" as a general claim.

**How to use G.1 and G.2 together, which is the honest move.** Three careful papers using
overlapping methods disagree about whether awareness changes behaviour. They can disagree because
nobody can run the clean counterfactual: you cannot observe one model on one task both aware and
unaware without intervening, and every available intervention — prefilling the chain of thought,
resampling it, steering the activations — is either visible to the model or has side effects the
size of the effect being measured. **The disagreement is the argument for a better instrument, not
a reason to drop the motivation.**

### G.3 The intervention this project uses is itself detectable

**Fonseca Rivera and Africa (2025), *Steering Awareness: Detecting Activation Steering from
Within*.** [arXiv:2511.21399](https://arxiv.org/abs/2511.21399) ·
vault: `fonseca-2025-steering-awareness`
States the assumption in the same words this post uses: activation steering "is widely used in
safety evaluations as if the model cannot detect the intervention." After fine-tuning, seven
instruction-tuned models reach **95.5% detection, 71.2% concept identification and zero false
positives on clean inputs** on held-out concepts. It generalises to unseen vector-construction
methods when their directions have high cosine similarity to the training distribution and not
otherwise — a geometric detector rather than a generic anomaly detector. **Detection does not
confer resistance:** detection-trained models are consistently *more* susceptible to steering.
Their closing line is the thesis: activation steering should not be considered an invisible
intervention in safety evaluations.
→ **Grounds:** the whole premise. Also the counterweight to state yourself — the failure mode is
contaminated measurement, not a model defending itself.
→ **Citation-year check before publishing:** the vault files this as 2025 (arXiv id `2511.`,
v3 dated 2026-03-19) while `docs/ARTICLE.md` currently cites it as 2026. Pick one.

**Fornasiere, Bronzi, Kitts, Palmas, Bengio and Richardson (2026), *Language Models Recognize
Dropout and Gaussian Noise Applied to Their Activations*.**
[arXiv:2604.17465](https://arxiv.org/abs/2604.17465) · LawZero ·
vault: `fornasiere-2026-recognizing-activation-noise`
The most general form of the problem, and it is not about concepts at all. Masking activations
(simulating dropout) or adding Gaussian noise at a target sentence, then asking "which of the
previous sentences was perturbed?", models from Llama, Olmo and Qwen between 8B and 32B **detect
and localise the perturbation, often with perfect accuracy**, and can learn in context to
distinguish dropout from Gaussian noise. **Qwen3-32B's zero-shot accuracy at identifying which
perturbation was applied improves with perturbation strength.**
→ **Grounds:** that noticing is not a property of injecting a *meaningful* direction. A
semantically empty perturbation is detectable, so no activation-space instrument is invisible by
construction. → **Also the sharpest external check on this project's Qwen3-32B null**
(`docs/RESULTS-QWEN.md`): the same model family and size detects activation perturbations in their
hands, with accuracy rising in perturbation strength, which is evidence for the
"we never reached a dose worth reporting" reading rather than for "this model cannot notice".

**McKenzie et al. (2026), *Endogenous Resistance to Activation Steering in Language Models*.**
[arXiv:2602.06941](https://arxiv.org/abs/2602.06941) ·
vault: `mckenzie-2026-endogenous-steering-resistance`
Models recover mid-generation from task-misaligned steering, producing explicit verbal restarts
("wait, that's not right") and continuing on topic while the perturbation is still active —
3.8% explicit occurrence in Llama-3.3-70B. Zero-ablating 26 contrastively-found SAE latents cuts
the multi-attempt rate by 25%, with random-latent and held-out-prompt controls. The authors state
the dual implication themselves: resistance could harden models against activation-space attacks,
but **"may equally interfere with beneficial steering-based interventions, since the model has no
way to distinguish the two."**
→ **Grounds:** that the reaction to being steered contaminates the measurement even when no
detection question is ever asked. Note the tension with Fonseca Rivera and Africa's finding that
detection does not confer resistance — different mechanisms, and worth one sentence rather than
being smoothed over.

**Pearson-Vogel et al. (2026), *Latent Introspection: Models Can Detect Prior Concept
Injections*.** [arXiv:2602.20031](https://arxiv.org/abs/2602.20031) ·
vault: `pearson-2026-latent-introspection`
On a Qwen 32B model, logit-lens analysis shows clear detection signals in the residual stream
while the sampled output denies the injection, attenuated in the final layers. Prompting the model
with accurate information about how introspection works raises sensitivity from **0.3% to 39.9%**
for a 0.6 percentage-point increase in false positives.
→ **Grounds:** that every self-report rate in this literature is a lower bound of unknown
tightness, and that an operating point found today should be expected to close, because the
capability is under-elicited rather than absent. Already grounds the M4 and M5 proposals.

**Hahami et al. (2026), *Detecting the Disturbance*.**
[arXiv:2512.12411](https://arxiv.org/abs/2512.12411) ·
**Lederman et al. (2026), *Emergent Introspection in AI Is Content-Agnostic*.**
[arXiv:2603.05414](https://arxiv.org/abs/2603.05414) ·
**Sridharan (2026), *Introspection or Entropy?*** ·
vault: `hahami-2026-detecting-the-disturbance`, `lederman-2026-content-agnostic-introspection`,
`sridharan-2026-introspection-or-entropy`
Three separate attacks on what a detection number means. Hahami et al.: binary detection accuracy
is entirely explained by a global logit shift toward affirmative answers — but on *differential*
tasks introspection is real, localising which of ten sentences was injected at up to 88% against
10% chance and discriminating relative injection strengths at 83% against 50%, confined to
early-layer injections. Lederman et al.: models detect that an anomaly occurred while unable to
identify its content, and confabulate high-frequency concrete nouns — *apple*, which is exactly
this project's observed default answer. Sridharan: injecting a concept raises the YES logit on
unrelated factual questions, and a concept-mismatch probe gets affirmative answers at
statistically indistinguishable rates.
→ **Grounds:** the scope boundary — this project measures whether the model *can name* the
injection, not whether it introspected — and the design requirement that a detection channel needs
a null arm and a mismatch control. Lederman et al. specifically predicts the `apple` floor.

### G.4 Steering already used as an instrument, where awareness would confound the reading

Each of these is a place where a published number is produced by adding a vector to a residual
stream and then reading generated text. **None of them reports a detection check.** That is the
list the post's closing ask is aimed at.

**Anthropic (2026), *Claude Opus 4.6 System Card* §6.5** — the alignment battery re-run under
inhibited evaluation-awareness vectors: automated behavioural audits, three agentic-misalignment
evaluations (Research Sabotage, Framing for Crimes, Blackmail), deception and sycophancy
evaluations, and LAB-Bench biology knowledge. Same method in the Claude Opus 4.5 card, §6.7.2.
→ If the model can tell that the inhibition vector is there, the corrected number is not the
deployment number either.

**Hua, Qin, Marks and Nanda (2025), *Steering Evaluation-Aware Language Models to Act Like They
Are Deployed*.** [arXiv:2510.20487](https://arxiv.org/abs/2510.20487) · ICLR 2026 ·
vault: `hua-2025-steering-evaluation-aware-models`
The source of the method above, and of the contrastive prompt set the system card reuses. They
train Llama 3.3 Nemotron Super 49B to be evaluation-aware, then suppress that awareness by
steering. Their recommendation is explicit: AI evaluators could improve the reliability of safety
evaluations by steering models to act deployed.
→ The recommendation is only as good as the operating point it is run at, and none is published.

**Chen, Arditi, Sleight, Evans and Lindsey (2025), *Persona Vectors*.**
[arXiv:2507.21509](https://arxiv.org/abs/2507.21509) ·
vault: `chen-2025-persona-vectors`
Directions for traits like evil, sycophancy and hallucination-propensity, used three ways:
monitoring personality at deployment, post-hoc correction after finetuning, and **preventative
steering during finetuning**. Trait expression is scored from generated text produced while the
vector is applied.
→ A model that notices the persona vector and reacts to it contributes that reaction to the
measured trait score. The preventative-steering variant applies the vector throughout training,
which is the largest exposure of the three.

**Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning and Potts (2025), *AxBench*.**
[arXiv:2501.17148](https://arxiv.org/abs/2501.17148) · ICML 2025 ·
vault: `wu-2025-axbench-steering-benchmark`
The benchmark that ranks steering and concept-detection methods against each other on Gemma-2-2B
and 9B — SAEs, difference-in-means, linear probes, representation finetuning, prompting. Steering
success is judged from the generated text.
→ Any method whose reading depends on generated text inherits the detection problem, and the
benchmark's ranking is what other people use to choose a method. Its headline result — prompting
beats every representation-based steering method — also matters directly for this project's
framing: steering is chosen for evaluation *because it is causal*, not because it works best.

**Panickssery, Rimsky et al. (2024), *Steering Llama 2 via Contrastive Activation Addition*.**
[arXiv:2312.06681](https://arxiv.org/abs/2312.06681) ·
vault: `panickssery-2024-contrastive-activation-addition`
The vector-construction method this project uses, evaluated on Anthropic's model-written
evaluations — sycophancy, corrigibility, and the rest — scored from generated answers.

**Zou et al. (2023), *Representation Engineering*.**
[arXiv:2310.01405](https://arxiv.org/abs/2310.01405) · vault: `zou-2023-representation-engineering`
· **Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*.**
vault: `arditi-2024-refusal-direction`
Reading and controlling honesty, and ablating the refusal direction to assess safety robustness.
Both produce numbers from a steered or ablated model's own output.

**Li, Fastowski, Zaradoukas, Prenkaj and Kasneci (2026), *Analysing the Safety Pitfalls of Steering
Vectors*.** [arXiv:2603.24543](https://arxiv.org/abs/2603.24543) ·
vault: `li-2026-safety-pitfalls-of-steering-vectors`
A safety audit of CAA vectors on JailbreakBench: steering in specific directions moves attack
success rate by **up to +57% or −50%** depending on the targeted behaviour, attributed to overlap
between the steering vector and latent refusal directions.
→ **Grounds:** that steering-based safety numbers are already unstable for reasons that have
nothing to do with awareness, so a paper reporting one needs to rule out both.

**Kowalski (2026), *Measuring Activation Control in LLMs*.**
[LessWrong](https://www.lesswrong.com/posts/HgvwxjzgwvsEvAiBH/measuring-activation-control-in-llms)
· vault: `kowalski-2026-measuring-activation-control`
Most LLMs can modulate the direction and magnitude of their own residual stream on natural-language
instruction, with some temporal resolution, and in simple tasks that control can partly evade
linear probes, natural-language autoencoders, activation oracles and the J-lens.
→ **Grounds:** that the report-independent readouts proposed in
[`docs/M5-PROPOSAL.md`](docs/M5-PROPOSAL.md) are not automatically safe from the same problem.
→ **Cite the public write-up, not the vault's anonymous manuscript copy**
(`anonymous-2026-activation-controllability-benchmark`), which carries a no-distribution notice.

**Nguyen, Hoang, Attubato and Hofstätter (2025), *Probing and Steering Evaluation Awareness of
Language Models*.** [arXiv:2507.01786](https://arxiv.org/abs/2507.01786) ·
vault: `nguyen-2025-probing-steering-evaluation-awareness`
Linear probes separate real-world evaluation prompts from deployment prompts in
Llama-3.3-70B-Instruct, and **current safety evaluations are classified by those probes as
artificial** — the internal representation exists whether or not the model says anything.
→ **Grounds:** the case for measuring detection without asking, and evidence that today's safety
benchmarks already look like tests from the inside.

### G.5 The general shape of the concern

**Korbak et al. (2025), *Chain of Thought Monitorability: A New and Fragile Opportunity for AI
Safety*.** [arXiv:2507.11473](https://arxiv.org/abs/2507.11473) ·
**Baker et al. (2025), *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting
Obfuscation*.** [arXiv:2503.11926](https://arxiv.org/abs/2503.11926)
Monitoring channels that depend on the model's own output are fragile, and optimising against a
monitor teaches the model to evade it.
→ **Grounds:** why a safety property resting on honest self-report is a property that can be
trained away.

**Needham et al. (2025), *Large Language Models Often Know When They Are Being Evaluated*.**
[arXiv:2505.23836](https://arxiv.org/abs/2505.23836) ·
vault: `needham-2025-evaluation-awareness`
Frontier models classify transcripts as evaluation or deployment well above chance — Gemini-2.5-Pro
at AUC 0.83 against a human baseline of 0.92, over 1,000 prompts from 61 datasets, and better in
agentic settings than in chat. They also identify what an evaluation is testing for, far above
chance.
→ **Grounds:** the premise that a model can form the belief "I am being tested" from cues in its
situation, without anyone injecting anything.

**Bailey et al. (2025), *Obfuscated Activations Bypass LLM Latent-Space Defenses*.**
[arXiv:2412.09565](https://arxiv.org/abs/2412.09565) ·
**Xu et al. (2025), *Steering in the Shadows: Causal Amplification for Activation-Space Attacks*.**
[arXiv:2511.17194](https://arxiv.org/abs/2511.17194)
The adversarial mirror image: activations can be shaped to evade latent-space monitors, and
activation-space attacks can be amplified.
→ **Grounds:** the dual-use position stated in [`CLAUDE.md`](CLAUDE.md). The same measurement that
tells an auditor which settings give an honest reading tells an attacker which settings go
unnoticed. Publishing rates is the intended output; vectors, model weights and raw generations are
not published for this reason.

---

## H. Read before scoping the mechanistic follow-up

**Anthropic (2026), *Verbalizable Representations Form a Global Workspace in Language Models*.**
[transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html)
→ **Grounds:** nothing yet. It is listed in `docs/RESEARCH-PROPOSAL.md` as required reading before
the mechanistic milestone is scoped, because if verbalisable content occupies an identifiable
workspace, "can the model name the injection" may have a cheaper and more direct reading than
asking it.

---

## Coverage gaps

Two works cited in `docs/RESEARCH-PROPOSAL.md` are **not** in the local research vault and have not
been read in full for this bibliography. Both are cited there for steering-vector reliability, and
whichever claims survive into the post should not rest on them until they are:

- Tan et al., *Analysing the Generalisation and Reliability of Steering Vectors*,
  [arXiv:2407.12404](https://arxiv.org/abs/2407.12404)
- *Activation-Space Personality Steering: Hybrid Layer Selection for Stable Trait Control in
  LLMs*, [arXiv:2511.03738](https://arxiv.org/abs/2511.03738)

Two entries above carry no arXiv identifier because none is recorded in the vault and the
convention there is never to guess one: Venkatesh et al. (2026) and Li et al. (2026),
*Decomposing and Measuring Evaluation Awareness*. Verify both before publication.
