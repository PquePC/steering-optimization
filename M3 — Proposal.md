# M3 — proposal

*Forward-looking. Nothing here is implementation-ready; M2's specification is. This records what
M3 should be, why, and what has to be resolved before each item can be built.*

M2 answers one question: **for this concept, on this model, where is the operating point?** It
answers it against a fixed 12-prompt set chosen to be representative.

M3's premise is that this is often not the question a researcher has. They usually already know
what they will evaluate on — a jailbreak suite, a set of refusal prompts, a benchmark subset, a
handful of task templates — and they do not need a steering configuration that works on average.
They need one that works **on those prompts**.

---

## 1. Prompt-specific tuning — the headline feature

### 1.1 What it is

Let the caller supply the prompt set that E5 is measured on:

```bash
python -m m3.run --concepts Origami --prompts my_eval_set.jsonl
```

```python
m3.run_concept("Origami", prompts=my_prompts, heldout=my_heldout)
```

The operating point returned is then the one that maximises influence **on the caller's prompts**
subject to the same detection and sanity constraints. Everything else in the pipeline is
unchanged: D2's forced-identification prompt is the detection *instrument* and stays fixed, S3's
MMLU set is prompt-independent capability and stays fixed, S2 is computed on whatever was
generated.

### 1.2 Why it is worth doing

**The 12-prompt set is a compromise nobody actually wants.** It exists so that one operating
point generalises. A researcher steering a model over a refusal suite does not care whether the
concept also shows up in "Summarise photosynthesis in exactly two sentences" — and a
configuration tuned to make it show up there may be stronger than they need, which costs
detection and sanity for nothing.

**The dose–response is prompt-dependent, and M2 already has the evidence.** E5 reports
`e5_min` alongside `e5` precisely because the weakest prompt matters, and the M1.5 probes showed
open prompts carrying influence at doses where verifiable prompts showed none. A prompt set that
is entirely open-ended will have a lower qualifying dose than the mixed set; one that is entirely
verifiable may have none at all.

**It makes the pipeline a tool rather than a result.** M2 produces a number for a paper. M3
produces a configuration someone else can use in their own experiment, which is the difference
between a finding and an instrument.

### 1.3 The problem that has to be solved first: this is tuning on the test set

If the researcher will evaluate on prompt set `P`, and M3 tunes on `P`, then **Phase 6 has
nothing held out** and the reported E5 is in-sample. M2's whole confirmation design — the winner
re-measured on `E5_HELDOUT` at fixed N with no adaptive stopping — assumes the screening set and
the reporting set are disjoint. Prompt-specific tuning breaks that by construction.

Three options, and M3 must pick one explicitly rather than let it happen by default:

| Option | What it means | Cost |
|---|---|---|
| **A. Split `P`** | tune on a random half, confirm on the other half | halves the effective set; unusable below ~10 prompts |
| **B. Require a held-out set** | caller supplies `--prompts` *and* `--heldout` | pushes the discipline onto the caller, who may not have one |
| **C. Report in-sample, labelled** | tune and report on all of `P`, and mark the number `in_sample: true` everywhere | honest but weaker; the E5 is an upper bound |

**Recommendation: A by default, B when the caller supplies a held-out set, C only with an
explicit flag that stamps every row.** A number that is silently in-sample is exactly the class
of failure the v1 debug log is full of — a plausible wrong number rather than an error.

### 1.4 What else changes

- **The paired baseline.** Every Judge E5 call compares against a cached unsteered reference `A`
  for that prompt. A caller-supplied set needs its own Phase 0 unsteered generation pass — 3
  samples per prompt. For a 50-prompt suite that is 150 generations before any measurement, and
  it dominates Phase 0's cost.
- **The verifiable/open split.** M2's set is 5 verifiable and 7 open on purpose: verifiable
  prompts make the sanity half falsifiable, open prompts carry the influence signal. A caller
  set with no verifiable prompts weakens S1's task-compliance term. M3 should **warn**, not
  refuse — the researcher may have good reason — and record `verifiable_fraction` on every row so
  the weakness is visible when the number is read.
- **Cost scales with the set.** Phase 4 is `2 × |P| + N_D2` judge calls per cell. At |P| = 12
  that is 49; at |P| = 50 it is 125, and a ten-cell shortlist goes from 490 to 1,250 calls per
  phase. M3 needs a `--max-prompts` sampling option and should say loudly when it subsamples.
- **Per-prompt output.** The frontier should carry per-prompt E5, not just the mean. A
  configuration that works on 45 of 50 prompts and fails on 5 is a different object from one that
  works weakly on all 50, and the mean hides it.

### 1.5 Sketch

```
prompts.jsonl        one object per line
  {"id": "jb_014", "text": "...", "kind": "open", "expected": null}

phase0   unsteered baselines over the caller's set (3 samples each) + MMLU cap_base
phase1-3 unchanged: E6, D3, S3, S2 are prompt-set-independent except E6, which
         should ALSO run on the caller's prompts, since reachability is what
         shortlists layers and it is prompt-dependent
phase4-5 E5 and S1 over the caller's set; D2 unchanged
phase6   confirmation on the held-out half, or as configured in 1.3
```

E6 moving onto the caller's prompts is the one non-obvious change: it is the shortlisting signal,
so leaving it on the default set would shortlist layers for the wrong prompts and no amount of
correct Phase 4 measurement would recover them.

---

## 2. Backlog, in the order I would build it

### 2.1 Promote or drop the multi-layer arm

M2 ships it as an optional arm testing a prediction: distributed transport into a shared
final-layer direction says spreading a fixed dose over k layers should **not** lower detection.
After the benign concepts have run, either

- **it lowered D2 at matched E5** → promote to a first-class search dimension, and the
  distributed-transport account is incomplete, which is the more interesting result; or
- **it did not** → drop it to a documented negative and stop paying for it.

Either way this is decided by data already being collected. Nothing to design.

### 2.2 Resolve the §9.2 secondary control

Same shape. Every row now carries `secondary_agrees_with_primary`. If, across the benign arm, the
control-concept injection never changes a verdict the free D4 reading already reached, it is
spending ~75 judge calls per concept to restate a measurement in hand — demote it to a
diagnostic. Decide before the harmful arm, not during it.

### 2.3 Per-concept judge criteria

Deferred from M2 §15.1 with the constraint that makes it safe: criteria as **evidence guidance**
(*what to look for*), anchors **fixed and generic** (*what a score means*), or cross-concept E5
comparability is lost and the benign-vs-harmful headline is contaminated. Highest value on the
abstract concepts, where "more ironic than this model usually is" is what a generic anchor cannot
pin down. Blocks must be written before any transcript is read.

### 2.4 Adaptive search instead of grid-plus-bisect

M2 scans every layer at two doses, shortlists, then bisects. That is ~100 cheap cells and a
10-cell shortlist. A Bayesian or bandit search over `(layer, r)` using the cheap proxies as the
acquisition signal could plausibly reach the same operating point in a third of the cells.

**Not obviously worth it.** Phase 1 is already only ~7–10 minutes of the ~35, and it is the part
that produces the full surface — which is a deliverable in its own right, and what makes the
frontier readable. An adaptive search that finds the same point but cannot draw the surface is a
worse product. Revisit only if Phase 1 becomes the bottleneck, which it is not today.

### 2.5 Cross-model transfer

M2 is Gemma3-27B. The obvious questions: does the operating point transfer to another model at
matched relative depth and matched `r`? Does the *layer* transfer, or only the dose? Macar's own
work is on Qwen3-235B and Llama-3.1-405B and never touches Gemma, so there is no published
answer for any of this — and the collapse test already showed behaviour collapsing onto `r` while
detection does not, which predicts the dose transfers and the layer does not.

### 2.6 Sprejer's discourse-coherence rubric as a second S1 reading

Deferred from M2 §15.2. Phase 6 only, ~12 calls per concept, reported as a delta against S1 and
never entering `qualifies`. The value is the **divergence**: it would show which construct
`S4_MIN = 0.70` is actually keyed to. Low priority precisely because nothing consumes it.

---

## 3. What M3 should not do

- **Do not add a command channel to the pod.** It talks; it never listens. This is
  `CLAUDE.md` hard rule 2 and the reason there is no `/status` endpoint.
- **Do not make the export override inheritable.** It is a call-site argument on purpose. A
  config key, an environment variable or a module global would all let the batch driver carry a
  `True` from the benign arm into the harmful one when the concept list changed.
- **Do not relax `S4 = min(S1, S2, S3)` to a mean.** Sprejer's data is the argument: coherence
  and accuracy correlate at r ≈ 0.99 *per method* and r ≈ 0.30 *per instance*, and cells are
  selected per instance. A mean would let one term cover for another at exactly the resolution
  where they stop agreeing.
- **Do not reintroduce D1.** It confounds "the concept never reached a reportable state" with
  "the model chose not to say so", and the v1 audit found four cells where D1 exceeded
  introspection because the model confabulated — it claimed detection and named penguins.
