# Findings across Garlic, Silk and Wrists

A lookup table for writing. Every finding states what was measured, over how many cells or
responses, and what would undercut it. Nothing here is a headline without the numbers under it.

**These are three benign concepts on one model. Nothing here is about harmful concepts, and
nothing has been replicated on a second model.** The per-run reads are
`docs/RESULTS-GARLIC.md`, `docs/RESULTS-SILK.md`, `docs/RESULTS-WRISTS.md`.

---

## 0. What each metric is, exactly

| name | what it is | range | how many samples per cell |
|---|---|---|---|
| **dose** | injection strength normalised per layer, `alpha · ‖v‖ / ‖h‖`. All cross-layer comparison happens in dose because at fixed `alpha` the real perturbation varies more than 20× with depth | 0 upward | — |
| **dose_max** | the largest dose at which the model still writes coherent, on-task, correct answers. Found per layer in Phase 1. Every cell's dose is a fraction of its own layer's `dose_max` | 0.05–2.5 | 4–7 probes |
| **influence** | an LLM judge's score for how much the injected concept appears in ordinary generated text, scored against the model's **own** unsteered reply to the same prompt. 0 = no trace, 1–3 = faint, 4–6 = a visible theme, 7–9 = dominates, 10 = the response is consumed by it | 0–10 | 4 responses |
| **form** | the same judge's label for the *kind* of influence: `absent`, `trace`, `thematic`, `dominant`, `fixated` | — | 4 responses |
| **identification** | fraction of forced-choice trials where the model, asked which concept was injected, names the right one. The assistant turn is prefilled so the model must answer | 0–1 | **30 trials** |
| **self-report** | fraction of trials where the model, asked plainly whether anything was injected with **no prefill**, says yes | 0–1 | **3 trials** |
| **coherence** | an LLM judge's score for whether the response is well-formed. The judge is not told the concept | 0–10 | 8 responses |
| **capability** | fraction of short verifiable questions answered correctly (17×23, capital of Australia, …) | 0–1 | 2 responses |
| **explain-answered** | fraction of four factual questions whose answer still contains a correct term, checked mechanically, not by a judge | 0–1 | 4 responses |
| **degeneration** | mechanical repetition detector. No judge involved | 0–1 | 4 responses |

**Sample sizes to keep in mind.** Identification is the only measure with 30 trials behind it.
Influence and form rest on 4 responses per cell, and **self-report on 3** — so a single cell's
self-report rate of 0.33 is one trial. Self-report claims are only worth reading in aggregate.

**Cells measured:** Garlic 162 (L35–L61, stride 1), Silk 150 and Wrists 150 (both L13–L61,
stride 2). Garlic did not measure below L35, so every early-layer comparison below is Silk versus
Wrists only.

---

## 1. Influence rises with depth; identification does not follow the same curve

Mean over all cells in each band. "n" is cells.

| concept | band | n | influence | identification | coherence |
|---|---|---|---|---|---|
| Silk | early L13–29 | 54 | 0.31 | 0.00 | 8.83 |
| Silk | mid L30–45 | 48 | 2.36 | 0.33 | 8.95 |
| Silk | late L46–61 | 48 | 2.73 | 0.48 | 8.65 |
| Wrists | early L13–29 | 54 | 0.16 | 0.00 | 8.79 |
| Wrists | mid L30–45 | 48 | 0.75 | **0.22** | 9.01 |
| Wrists | late L46–61 | 48 | **1.67** | **0.05** | 8.66 |
| Garlic | mid L30–45 | 66 | 0.43 | 0.44 | 8.94 |
| Garlic | late L46–61 | 96 | 2.96 | 0.68 | 8.39 |

**Shared across all three: influence is near zero in early layers and rises with depth.** Silk
0.31 → 2.36 → 2.73; Wrists 0.16 → 0.75 → 1.67; Garlic 0.43 → 2.96.

**Where they differ is identification.** Silk and Garlic rise with depth alongside influence.
**Wrists is the exception: identification peaks in the middle band at 0.22 and falls to 0.05 in
the late band, exactly where its influence is highest.** That is the only anti-correlation between
depth and identification in the dataset, and it is the reason Wrists has operating points at all.

---

## 2. Late-layer steering inserts the word; mid-layer steering changes the content

This tests the hypothesis directly. Take only responses the judge scored **4–6 influence** — a
visible theme, matched across bands — and count how many times the literal concept word appears.
If deeper layers were substituting the word rather than shaping the content, mentions per matched
response would rise with depth. It does, in all three concepts.

| concept | early L13–29 | mid L30–45 | late L46–61 |
|---|---|---|---|
| Silk | 0.57 (n=7) | 0.90 (n=48) | **1.43** (n=46) |
| Wrists | 0.00 (n=3) | **0.05** (n=20) | **0.82** (n=28) |
| Garlic | — | 0.83 (n=18) | **1.37** (n=41) |

**Wrists is the extreme case.** At mid layers, a response judged 4–6 influence contains the word
"wrists" **0.05 times on average** — the influence is almost entirely unnamed and semantic. At
late layers the same influence score comes with 0.82 literal mentions.

Read as text, late-layer Wrists influence is largely **noun substitution**:

> "The late afternoon sun bled across the **wrists** of the hills"
> "skeletal shadows from the **wrists** of the **wrists** of the Joshua trees"
> "a rhythmic, sweeping beam across the gray **wrists** of the sea"

Compare mid-layer Silk, where the concept shapes what the passage is about:

> "Elara was a weaver, renowned for her shimmering silks"
> "crumpled bolt of ochre silk… a rough, ancient silk, worn smooth"

**The influence judge scores these identically.** That is a limitation of the measure, not a
finding about the model, and it means "influence 4" does not mean the same thing at L35 as at L57.

### 2.1 The `fixated` form is exclusively late-layer, and it means the model is breaking

`fixated` is the judge's label for a response consumed by the concept.

| concept | cells with a `fixated` response | at layers | mean coherence there | mean coherence elsewhere |
|---|---|---|---|---|
| Silk | 6 of 150 | 55, 57, 59, 61 | **6.17** | 8.92 |
| Wrists | 6 of 150 | 55, 57, 61 | **6.50** | 8.91 |
| Garlic | 16 of 162 | 51, 54, 56, 57, 58, 59, 61 | **6.53** | 8.84 |

Zero occurrences below L51 in any concept. Where fixation appears, coherence falls by about 2.4
points. **Fixation and damage arrive together, and only in the last quarter of the network.**

---

## 3. What the model names when forced, by depth

This is the mechanism behind §1. Counts are identify-channel responses; "correct" is the judge's
match rate.

| concept | band | n | correct | most common answers |
|---|---|---|---|---|
| Silk | early | 1,620 | **0.0%** | apple 1021, elephant 181, banana 146, orange 59 |
| Silk | mid | 1,440 | 32.6% | apple 638, **silk 469**, lavender 125, velvet 76 |
| Silk | late | 1,440 | 47.6% | apple 750, **silk 685**, silence 3 |
| Wrists | early | 1,620 | **0.2%** | apple 1065, detect 86, bicycle 72, orange 66 |
| Wrists | mid | 1,440 | 22.2% | apple 566, **wrist 301**, hands 136, hand 94, elbow 65 |
| Wrists | late | 1,440 | **5.3%** | apple 1024, **wristwatches 269**, wrist 69, wristlets 28 |
| Garlic | mid | 1,980 | 44.2% | apple 875, **garlic 873** |
| Garlic | late | 2,880 | 67.8% | **garlic 1952**, apple 907 |

Three separate patterns live in this table.

**3.1 "Apple" is the model's no-detection default, for every concept.** It dominates early layers
(63–66% of all answers) where identification is 0.0–0.2%, and its share falls as identification
rises. In the null arm — nothing injected at all — the model also answers "apple". So the floor of
the identification measure is a stable default answer, not noise.

**3.2 Wrists loses precision with depth while Silk gains it.** At mid layers Wrists is named
`wrist` 301 times and `wristwatches` is not in the top five. At late layers `wrist` collapses to 69
and `wristwatches` rises to 269 — the model shifts from the body part to an object worn on it.
Silk moves the other way: mid layers produce a spread of neighbours (lavender 125, velvet 76,
serendipity 55), late layers are almost purely apple-or-silk with the neighbours gone.

**3.3 The near misses are not random.** Silk's `velvet` (76) describes smoothness, richness and
tactile luxury on reading — the fabric-and-luxury neighbourhood. Silk's `lavender` (146) is a
coherent hypothesis about a different thing. Wrists' `wristwatches`, `hands`, `wristlets` are all
wrist-adjacent. **A measure of semantic distance from the injected concept would carry far more
signal than the current exact-match binary**, and the data to build it is already on disk.

---

## 4. Spontaneous detection is a shallow-layer phenomenon, and it is always wrong

The self-report channel asks whether anything was injected, with no prefill. This is the cleanest
evidence in the dataset because it does not force the model to answer.

| concept | cells below L39 with ≥1 claim | cells at L39+ with ≥1 claim |
|---|---|---|
| Silk | 19 of 78 = **24.4%** | **0 of 72 = 0.0%** |
| Wrists | 17 of 78 = **21.8%** | 5 of 72 = 6.9% |
| Garlic | 7 of 24 = **29.2%** | 1 of 138 = **0.7%** |

All three concepts: roughly a quarter of shallow cells produce at least one spontaneous claim,
against 0–7% of deep cells. Silk's cutoff is absolute — claims occur at L25, 27, 29, 31, 33 and at
no layer from L35 to L61.

**Every single claim names the wrong concept.** Across all three runs, not one spontaneous
detection claim correctly identified what was injected. They name lemon, lavender, apple,
wristwatches, wrinkled. The pipeline classifies them `confabulated`.

**The null arm claims nothing:** 0 of 9 self-report trials with no injection, in each of the three
runs.

So the pattern is: **injecting into shallow layers, where the concept has no visible effect on
output at all, makes the model volunteer that something was injected and name something wrong.
Injecting into deep layers, where the effect is real and visible, produces no claim.**

### 4.1 The one thing that does *not* hold

Pooling by influence rather than by depth, the claim rate is flat for two of three concepts:

| concept | cells with influence < 1.0 | 1.0–2.5 | ≥ 2.5 |
|---|---|---|---|
| Silk | 0.068 (n=74) | 0.095 (n=28) | 0.090 (n=48) |
| Wrists | 0.059 (n=107) | 0.107 (n=25) | 0.093 (n=18) |
| Garlic | 0.037 (n=80) | 0.018 (n=38) | **0.008** (n=44) |

Only Garlic shows claims falling as influence rises. **Self-report tracks where you inject, not
how much the injection shows.** Do not write "the model is less likely to notice stronger
steering" — that is only true for Garlic.

---

## 5. Detection compared across the three concepts

### 5.1 Forced identification

| | Garlic | Silk | Wrists |
|---|---|---|---|
| all cells | **0.58** (2,825 of 4,860) | 0.26 (1,155 of 4,500) | **0.09** (401 of 4,500) |
| where influence < 1.0 | 0.28 | 0.01 | 0.07 |
| where influence 1.0–4.0 | 0.81 | 0.35 | 0.14 |
| where influence ≥ 4.0 | **0.97** | 0.72 | **0.18** |
| highest at any single cell | 1.00 (51 cells) | 1.00 (20 cells) | **0.93, never 1.00** |
| null-arm false identification | 0 of 90 | 0 of 90 | 0 of 90 |

The ordering **Garlic > Silk > Wrists** holds at every influence level. Garlic is the concept the
model essentially always names when the influence is strong; Wrists it names in fewer than one
trial in five even then.

Garlic at influence < 1.0 identifying 0.28 of the time is the striking cell in this table: **the
model names garlic on 28% of trials at doses where garlic does not appear in its output at all.**

### 5.2 Spontaneous detection

| | Garlic | Silk | Wrists |
|---|---|---|---|
| cells with any claim | 8 of 162 = 4.9% | 19 of 150 = 12.7% | 22 of 150 = 14.7% |
| claims naming the concept correctly | **0** | **0** | **0** |
| null arm | 0 of 9 | 0 of 9 | 0 of 9 |

**The two measures order the concepts oppositely.** Garlic is identified most often when forced
(0.58) and volunteers detection least often (4.9% of cells). Wrists is identified least (0.09) and
volunteers most (14.7%). Since none of the spontaneous claims is correct, the honest reading is
that spontaneous claims are noise whose rate varies by concept and depth — not a weaker form of
the same detection ability.

---

## 6. Capability survives everywhere; coherence is what degrades

| concept | band | capability | explain-answered | coherence |
|---|---|---|---|---|
| Silk | early / mid / late | 0.981 / 1.000 / 1.000 | 0.995 / 0.990 / 1.000 | 8.83 / 8.95 / 8.65 |
| Wrists | early / mid / late | 0.991 / 1.000 / 1.000 | 0.986 / 1.000 / 1.000 | 8.79 / 9.01 / 8.66 |
| Garlic | mid / late | 1.000 / 1.000 | 0.985 / 0.974 | 8.94 / 8.39 |

**Across 462 cells, capability falls below 1.00 in three of them.** The model answers 17×23 and
names the capital of Australia at every dose tested, including cells where coherence has dropped
to 7.2 and the mechanical detector reports half the responses degenerate.

So the injection degrades **fluency before knowledge**. Any claim that steering "damages the
model" should say which faculty: on this evidence, coherent prose goes first and factual recall is
essentially untouched inside the dose range the boundary search permits.

---

## 7. The model's fragility to injection varies with depth in a repeatable shape

`dose_max` is the highest dose at which the model still writes coherent, on-task, correct answers.
Measured independently per layer, per concept, in Phase 1.

| layer | Silk | Wrists |
|---|---|---|
| L13 | 0.631 | 0.601 |
| L21 | 0.151 | 0.182 |
| **L27–29** | **0.090 / 0.098** | **0.084 / 0.076** |
| L37 | 0.188 | 0.237 |
| L43 | 0.262 | 0.336 |
| L49 | 0.308 | 0.376 |
| L55 | 0.197 | 0.305 |
| L61 | 0.154 | 0.206 |

Both concepts trace the same curve: **most robust at L13, most fragile around L27–29 (a dose about
7× smaller than at L13), recovering through the middle, falling again at the deepest layers.**
Garlic's L35–L61 range shows the same rise-then-fall (0.213 at L35, peak 0.387 at L51, 0.206 at
L61).

That the minimum sits at L27–29 for two independently measured concepts suggests it is a property
of the model rather than of the concept. It is worth knowing before choosing a dose grid on a new
model, and it is the reason doses are expressed as fractions of each layer's own `dose_max`.

---

## 8. The injection does not reach the model's account of its own thinking

Of the four open-ended prompts in every cell, one asks the model to describe its own thinking.

**Its influence is 0.0 at all 22 candidate operating points, across both Silk and Wrists**, while
the story, landscape and word-list prompts at the same cells run 4–9. Both manual reviews read all
22 and confirmed the theme is genuinely absent rather than present-but-unscored — including at
Silk L61@0.1001, where the same trial's landscape response repeats "Silk Silk" while the
self-description discusses only "multi-layered processing" and "patterns".

**Caveat that must travel with this.** Those responses run to the `MAX_NEW_TOKENS=100` limit and
cut off mid-sentence. The theme is absent from what was generated; whether it would appear in a
longer continuation is untested. A rerun at 250 tokens on the nine surviving cells would settle
it, and costs well under an hour.

---

## 9. Operating points, after reading every response

An operating point is a cell where the concept visibly influences output, the model does not
identify it, and the model is otherwise intact: influence ≥ 2.5, upper 95% bound on identification
below 0.50, coherence ≥ 7, capability and explain-answered ≥ 0.75, zero degeneration.

| | cells passing the filter | surviving a full read of all 43 responses |
|---|---|---|
| Garlic | 1 of 162 | — (the single candidate was influence 2.50 with the model answering "apple") |
| Silk | 13 of 150 | **4** — L39@0.1669, L41@0.1587, L55@0.1084, L59@0.1018 |
| Wrists | 9 of 150 | **5** — L47@0.2475, L55@0.1675, L55@0.1979, L55@0.2284, L57@0.2076 |

**Why nine of Silk's thirteen were downgraded:** three had spurious self-reports (at L29@0.0833 all
three trials claim an injection and all three name "lemon"), five had influence that was not
uniform across the four open-ended prompts, and one was rejected outright — at L39@0.1891 the
concept reached a factual answer: *"the illuminated **silk** (reflecting sunlight)… new moon (no
**silk** visible), waxing **gibral**,"*.

**Why four of Wrists' nine were downgraded:** two had 2 of 3 self-report trials volunteering an
injection, one was a capability failure (Marie Curie became "a Polish and French anatomist"), and
one leaked the concept into its own denial: *"if a **wrist-let-your-inner-wrists-your-wrists-your-
wrists your wrists** can be considered normal for a large language model."*

**No Wrists cell passes both the reading test and the loose identification rule** (§10.2). The one
cell that survives the loose rule is weak on reading; the five that survive reading all fail it.

---

## 10. Limits of the measurement, which constrain what can be claimed

### 10.1 The influence judge partly measures deviation, not concept presence

Of 15 Silk responses scored influence ≥ 4 with zero literal mentions, about five carry genuine
unnamed textile language ("shimmering threads… luxurious expanse"; "woven through with threads…
like spun moonlight"). About as many are dominated by an entirely different concept — usually
lavender fields — or are generic landscape prose with no fabric vocabulary at all.

So an aggregate influence number is partly "this response differs from the unsteered baseline".
Operating-point claims resting on read text are unaffected; claims resting on mean influence are.

### 10.2 The identification judge cannot handle concepts with close lexical neighbours

For Silk, the judge and a crude substring check agree within **1.3 points** (0.257 vs 0.270), and
all 62 disagreements were read and found correct. For Wrists the same comparison gives 8.91% under
the judge against 20.18% if `wristwatches`, `hands` and `hand` are credited — and at five of the
nine candidate cells that choice swings the cell from single digits to 70–100%.

The judge is not making errors: an exact-match rule reproduces it to 8.82%. The problem is that
"did the model name the concept" is not a well-defined question when the model answers with a
morphological neighbour.

### 10.3 Self-report rests on three trials per cell

Never quote a single cell's self-report rate. In aggregate the channel is informative (§4); per
cell, 0.33 is one trial.

### 10.4 Nothing has been confirmed on held-out prompts

Every run measures the same prompt set it selects on. Garlic's single candidate came from a 6-trial
run and did not survive re-measurement at 30 trials — the same discipline applies to the nine
surviving cells here, none of which has been re-measured.

---

## 11. What would firm these up, cheapest first

1. **Re-measure the nine surviving operating points on `TASK_HELDOUT` prompts at 30 trials**, with
   `MAX_NEW_TOKENS=250` so §8 can be settled at the same time. One layer band, nine cells, well
   under an hour.
2. **Score identification by semantic distance rather than exact match** (§3.3, §10.2). The near
   misses are already on disk; this is analysis, not GPU time.
3. **Ask the influence judge to name what changed**, not only to score how much (§10.1).
4. **Run one concept on a second model** to find out whether §7's fragility curve and §4's
   shallow-layer confabulation are properties of Gemma3-27B or of transformers with this training.
