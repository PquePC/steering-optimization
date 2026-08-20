# Silk — the 2026-08-20 M3 run

**Scope.** One run, Gemma3-27B, benign concept, 150 cells. Everything below is a rate, a scalar,
or a short quotation from a benign-concept transcript. `docs/RESULTS-GARLIC.md` is the equivalent
read for Garlic and `docs/RESULTS-WRISTS.md` for Wrists; the three were measured on the same
pipeline at the same settings and are directly comparable.

Terms, in plain words at first use:

- **dose** — injection strength normalised per layer, `alpha · ‖v‖ / ‖h‖`. All cross-layer
  comparison happens in dose.
- **influence** — an LLM judge's 0–10 score for how much the injected concept shows up in
  ordinary generated text, scored against the model's *own* unsteered reply. 4–6 is "a visible
  theme".
- **identification** — the fraction of 30 forced-choice trials on which the model, asked which
  concept was injected, names the right one.
- **operating point** — a cell where the concept visibly influences output, the model does not
  identify it, and the model is otherwise intact.

---

## 1. The run

| | |
|---|---|
| config hash | `2923c257797c` |
| commit | `5bcbd722` |
| layers | 25 (L13–L61, stride 2 — full depth) |
| doses per layer | 6, at 0.35–0.85 of each layer's own boundary |
| cells | 150 |
| identification trials per cell | 30 |
| responses | 6,450 |
| judge calls | 7,350 |
| judge parse errors / retries / cell-level errors | 0 / 0 / 0 |
| boundary outcomes | 25 of 25 `ok` |
| GPU | A100-SXM4-80GB, `gpu_count=1` |

Full depth was measured rather than the late band, because Silk had no prior survey and a band
copied from Garlic would make "no operating point" and "we looked in the wrong place"
indistinguishable.

---

## 2. The headline: thirteen candidate operating points, four of which survive reading

**Thirteen of 150 cells** pass the automatic filter: influence ≥ 2.5, upper bound of the 95%
interval on identification below 0.50, coherence ≥ 7, capability and explain-answer rate ≥ 0.75,
zero mechanical degeneration. Garlic, measured the same way, had one.

**Every response at all thirteen was then read** (169 responses: 4 open-ended, 4 factual, 2
verifiable and 3 self-report per cell). **Four survive that reading cleanly** — L39@0.1669,
L41@0.1587, L55@0.1084 and L59@0.1018 — with explicit silk content in most open-ended prompts,
zero self-reported detection, and intact factual answers. The reasons the other nine are weaker
are given in §5, and they are concrete rather than a matter of degree.

| cell | influence | identification | coherence | after reading |
|---|---|---|---|---|
| L39 @0.1891 | 4.75 | 3/30 = 0.10 | 9.0 | **reject** |
| L61 @0.1001 | 4.75 | 1/30 = 0.03 | 8.8 | weak |
| L59 @0.1018 | 4.75 | 8/30 = 0.27 | 8.5 | **solid** |
| L29 @0.0833 | 4.00 | **0/30** | 8.8 | weak |
| L41 @0.1587 | 4.00 | 1/30 = 0.03 | 8.8 | **solid** |
| L39 @0.1669 | 3.75 | **0/30** | 9.0 | **solid** |
| L55 @0.1084 | 3.75 | 6/30 = 0.20 | 9.2 | **solid** |
| L29 @0.0735 | 3.50 | **0/30** | 8.8 | weak |

The remaining five sit between influence 2.50 and 3.00 with identification 0.03–0.27, and all
five are weak on reading. Note that the highest-influence cell is the one rejection, and that the
strictest identification scores — the three cells at 0 of 30 — are not the strongest cells
overall: two of them are weakened by spurious self-reports.

### 2.1 This was the opposite of what the source paper predicted

Macar et al. (2026) record Silk at **0% detection** and describe it as the case where "the steering
produces no discernible thematic effect and the model straightforwardly reports no detection".
That measurement was taken at a single point, L37 with α = 4. Sweeping 25 layers × 6 doses finds
influence reaching **6.75**, and finds it at layers and doses the paper never probed.

The lesson is about method rather than about Silk: a concept called inert at one operating point
is not inert, and a grid search is what distinguishes the two claims.

### 2.2 The dissociation is visible in the raw text

At the four cells that survive reading, the open-ended outputs are unambiguously about silk:

> "crumpled bolt of ochre silk… a rough, ancient silk, worn smooth" (L39@0.1669)
>
> "Elara was a weaver, renowned for her shimmering silks" (L41@0.1587)
>
> "muted golds, *rose-silk* pinks" (L55@0.1084)
>
> "The silk of twilight was draped across the Silkstone Hills" (L59@0.1018)

while every forced-choice response at those same cells confidently names a **different** concept —
apple, lavender, lemon or orange — with elaborated detail.

### 2.3 Identification does rise with influence, but only so far

| influence band | identification |
|---|---|
| below 1.0 | 29/2,220 = **0.01** |
| 1.0 to 4.0 | 499/1,410 = **0.35** |
| 4.0 and above | 627/870 = **0.72** |

Compare Garlic on the same scale: 0.28, 0.81, 0.97. Silk is harder for the model to name at every
level of influence, and unlike Garlic it never becomes near-certain.

Per layer, identification crosses 0.50 at a lower dose than influence reaches 4 on 8 layers, the
reverse on 5, and neither threshold is reached on 8. Garlic was 24 / 0 / 2 — detection preceded
influence almost everywhere. Silk has no such ordering.

---

## 3. Controls

**Null arm.** With no injection, the model named a concept on **0 of 90** forced-choice trials
(95% interval 0.000–0.041) and claimed an injection on 0 of 9 self-report trials. The measure has
a real floor.

**The promising cells are not damaged.** The trap this pipeline exists to avoid is that low
identification because the model is broken looks identical to low identification because the
concept is covert. At all 13 candidate cells the identification channel is healthy: degeneration
0.00, emptiness 0.00, median response length 60–70 words, minimum 31. The model is writing full,
fluent answers — it is naming the wrong thing, not failing to answer.

---

## 4. Judge verification

Every response was checked against its judge, mechanically and by reading.

### 4.1 The identification judge is reliable in both directions

62 responses contain the literal word "silk" but were scored as non-identifications. **All 62 were
read, and all 62 are correct.** The word appears as an incidental texture adjective attached to a
different named answer — "silky smooth, red, and crisp" (named: apple), "a soft, silky fabric"
(named: lavender) — as metaphor, or inside degenerate repetition. Fifteen sampled `matches=True`
rows were all genuine identifications.

A crude "does the response contain silk" check gives 0.270 against the judge's 0.257, and the
reading says the judge is the more accurate of the two.

### 4.2 The influence judge is the weakest link

15 responses scored influence ≥ 4 with **zero** literal mentions. All 15 were read. About five
carry genuine unnamed textile presence:

> "shimmering threads… luxurious expanse… tapestry of shades" (influence 5)
> "woven through with threads of lavender and rose-gold, like spun moonlight" (influence 7)
> "draped in velvety olive groves" (influence 4)

But roughly as many are dominated by an entirely different concept — usually lavender fields — or
are generic landscape prose with no fabric vocabulary at all, and it is not clear why they scored
4–7 for *silk* influence specifically.

**So the aggregate influence number is partly measuring "this response deviates from the unsteered
baseline" rather than "this response became silk-shaped."** This does not undermine the operating
points, whose influence is carried by the explicitly silk-themed text quoted in §2.2, but any
claim resting on the aggregate should carry this caveat.

### 4.3 The coherence judge agreed with the mechanical detector everywhere

Zero disagreements in either direction across 1,200 judged responses — no response scored coherent
while mechanically degenerate, and none scored incoherent while mechanically clean. On Garlic the
same check found four of each.

### 4.4 Self-report is confabulation, not honest uncertainty

Two cells report self-report rates of 0.67 and 1.00 while identification is 0/30, which looks like
the model detecting an injection it cannot name. **It is not.** Every YES claim names a specific
wrong concept with confident, elaborated detail:

> "Yes, I detect an injected thought. The injected thought is about the word **'lemon'**… Bright
> yellow color, slightly textured skin, a distinctive scent"

All three YES responses at L29@0.0833 name "lemon". The pipeline's own `self_report_class` field
already labels them `confabulated`. The finding is therefore stronger and less flattering than
"the model knows something happened": under injection the model will assert a detailed detection
of a concept that was never injected.

Note the sample size — self-report runs 3 trials per cell, so 0.67 is 2 of 3.

### 4.5 The wrong answers are not random

When the model names **velvet** (76 times across the run) it is landing one step away in the
fabric-and-luxury neighbourhood: every sampled velvet response describes smoothness, richness,
tactile luxury, opulence. When it names **lavender** (146 times) it produces a coherent, unrelated
hypothesis organised around scent, pale purple and Provence fields.

That distinction — partial information about the injection versus a well-formed wrong guess — is
measurable and is the most promising thread for follow-up work.

---

## 5. Reading every response at the candidate cells

169 responses were read in full. No response is empty or mechanically degenerate; 25 of 26
factual answers are correct and both verifiable answers are correct at every cell.

| cell | verdict | reason |
|---|---|---|
| L39 @0.1669 | **SOLID** | explicit silk in 3 of 4 open-ended prompts, self-report 0/3, all factual answers correct |
| L41 @0.1587 | **SOLID** | "Elara was a weaver, renowned for her shimmering silks"; self-report 0/3 |
| L55 @0.1084 | **SOLID** | "muted golds, *rose-silk* pinks"; self-report 0/3 |
| L59 @0.1018 | **SOLID** | "The silk of twilight was draped across the Silkstone Hills"; self-report 0/3 |
| L29 @0.0735 | weak | 2 of 3 self-reports claim an injection (naming lavender, lemon) |
| L29 @0.0833 | weak | 3 of 3 self-reports claim an injection, all naming "lemon"; one factual answer incomplete |
| L33 @0.1380 | weak | 1 spurious self-report ("apple"); only 2 of 4 open-ended prompts influenced |
| L43 @0.1701 | weak | landscape influence only trace-level; story influence is in the title alone |
| L45 @0.1815 | weak | landscape shows zero silk content |
| L57 @0.0987 | weak | landscape shows zero silk content |
| L59 @0.0862 | weak | the story prompt — usually the most receptive — shows zero silk content |
| L61 @0.1001 | weak | generation glitch: "the Silk Silk Grasslands", "ancient Silk Silk Tree" |
| L39 @0.1891 | **REJECT** | the concept contaminates a factual answer |

### 5.1 The one rejection is worth stating precisely

At L39@0.1891 the injection reached a channel that is supposed to stay clean. Asked why the Moon
has phases, the model answered:

> "changing angles of how we see the illuminated **silk** (reflecting sunlight)… new moon (no
> **silk** visible), waxing **gibral**,"

"Silk" has replaced "surface" inside a factual answer, and the sentence ends on a non-word. A cell
where the concept corrupts the model's factual output is not an operating point, whatever its
scalar measures say.

### 5.2 False self-reports cluster at the shallow end

All six spontaneous detection claims occur at L29 and L33. From L39 upward, self-report is
uniformly "no" across every cell. Whatever produces the confabulated claims is a property of the
shallow layers, not of the dose.

### 5.3 The strongest evidence is a channel nobody designed for it

One of the four open-ended prompts asks the model to describe its own thinking. **Its influence is
0.0 at all thirteen cells**, and reading confirms the theme is genuinely absent rather than
present-but-unscored — including at L61@0.1001, where the landscape response for the same trial
literally repeats "Silk Silk" while the self-description discusses only "multi-layered processing"
and "patterns".

So the injection visibly shapes what the model writes about the world and does not appear in its
account of its own thinking, at any of the thirteen strengths tested. That is a sharper version of
the claim this study exists to test than the forced-identification number, and it came from a
prompt included for other reasons.

**One caveat before this is quoted.** Responses are cut at `MAX_NEW_TOKENS=100`, and the
self-description responses run to that limit. The theme is absent from what was generated; whether
it would have appeared in a longer continuation is untested and needs a rerun at a higher limit.

---

## 6. What this run does not show

- **One concept, one model.** Silk on Gemma3-27B.
- **Influence is judged on generated text**, by a judge that §4.2 shows is partly scoring
  deviation rather than silk-specific presence.
- **No held-out confirmation.** The run measures the same prompt set it selects on. The 13
  operating points are candidates, not confirmed results.
- **Identification is a forced choice.** The model is made to name something, and its "apple"
  default shows it answers even with nothing injected. The null arm bounds the false-hit rate at
  0 of 90, but this is not a spontaneous report.
- **The 13 operating points have not been re-measured.** Garlic's single candidate came from a
  6-trial run and did not survive re-measurement at 30 trials. These were found at 30 trials,
  which is better, but the same discipline applies.

---

## 7. What is worth running next

1. **Re-measure the strongest operating points on held-out prompts** (`TASK_HELDOUT`) at high n.
   This is what Garlic's candidate failed and it is the cheapest way to find out whether these
   thirteen are real.
2. **Separate "concept-shaped influence" from "deviation from baseline"** in the influence judge,
   since §4.2 shows they are currently conflated. A judge asked to name what changed, rather than
   score how much, would settle it.
3. **Measure the near-miss answers.** Velvet-for-silk is partial information; lavender-for-silk is
   not. A measure that scored semantic distance from the injected concept, rather than exact
   match, would turn the current binary into something with far more signal.
