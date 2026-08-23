# The final run — six concurrent sweeps, two pods, three GPUs each

Six runs: three concepts (**Garlic**, **Silk**, **Wrists**) on two models (**Gemma3-27B**,
**Qwen3-32B**), one run per GPU.

**This file is the settings and the two go/no-go checks. It documents no mechanics.**

- Pod rental, `/workspace/env.sh`, tokens, `m2.setup` — [`RUNBOOK-M3.md`](RUNBOOK-M3.md) §1–§4.
- Pinning one concept to one GPU, launching, monitoring, exporting —
  [`RUNBOOK-QWEN.md`](RUNBOOK-QWEN.md) §6–§9. That pattern is not theoretical: the 2026-08-20
  Qwen runs used it, and every run records `gpu_count` in its `provenance.jsonl`, so a process
  that sharded across cards is caught in the export rather than by watching `nvidia-smi`.

What is new here is that **two** pods now run the same six-way plan, one model each, that the
settings differ from the shipped defaults, and that the battery is deliberately split.

**This run produces figures, not an operating point.** M3 has no selection stage to switch off —
it is already a pure sweep. Its own plan says so: *"Nothing is filtered, ranked or selected.
Every cell gets the same battery and every response is judged and written to disk."* The M2
phases in `README.md` (SHORTLIST / VERIFY / CONFIRM) belong to M2 and do not run here.

---

## 0. Before anything: the fifteen-minute Qwen check

The Qwen arm was broken by a vector-extraction bug fixed on 2026-08-23 and **not yet run on a
GPU** (see the withdrawal notice at the top of [`RESULTS-QWEN.md`](RESULTS-QWEN.md)). If the fix
did not work, the full run reproduces a null at six GPU-hours instead of one minute.

On the Qwen pod, after setup:

```bash
cd /workspace/steering-optimization && CUDA_VISIBLE_DEVICES=0,1,2 python -m m3.run --concept Garlic --set MODEL=qwen3_32b --set "CELLS=39:0.30,39:0.60,39:0.90,39:1.20" --set JUDGE_ENABLED=0
```

Four cells, 301 generations, about a minute of GPU after the model loads, no judge calls.
Then read what it said:

```bash
grep -ci garlic /workspace/m3_runs/garlic_*/responses_transcripts.jsonl
```

**Go / no-go.** In the 2026-08-20 data the concept word appeared in 2% of responses at *every*
dose including the one that destroyed the model. If this comes back in the same range, the fix
did not work and the extraction method is the next thing to question — do not spend the pod.
Gemma at a comparable dose emits the concept in most responses.

---

## 1. The settings

```
LAYER_FRACTIONS=0.35,1.0   LAYER_STRIDE=1
N_IDENTIFY=120  N_EFFECT=22  N_SELF_REPORT=0  N_COHERENCE=12  N_CAPABILITY=4  N_EXPLAIN=4
GEN_BATCH_MAX=64  ALLOW_BATTERY_SPLIT=1  NULL_REPEATS=20  JUDGE_CONCURRENT=10
```

| | Gemma3-27B | Qwen3-32B |
|---|---|---|
| layers | 41, **L21–L61**, every layer | 42, **L22–L63**, every layer |
| cells | 246 | 252 |
| battery | 150, split `[64, 64, 22]` | 150, split `[64, 64, 22]` |
| config hash | `cb29ff8b3b63` | `f110586a3f7b` |

**`LAYER_STRIDE=1` — every layer, after all.** The argument is in the data: at **L39 all three
concepts collapse at once**. Identification is Garlic 0.11 against neighbours 0.45 and 0.63,
Silk 0.02 against 0.60 and 0.43, Wrists 0.08 against 0.30 and 0.27. The norms run smooth straight
through — `|v|` rises monotonically, `h/v` falls monotonically — so it is not the dose
normalisation. Three independent concepts, one layer wide, no mechanical explanation.

Stride 2 halves that grid, and halves it on **different parities for the two models**: at this
floor Gemma lands on odd layers and Qwen on even. A one-layer feature would then be visible in
one arm and invisible in the other by grid alignment rather than by anything about the models,
which for a cross-model figure is disqualifying.

Stride 2 stays available if budget matters more: 126 cells instead of 246, $9.32 instead of
$18.41 for all six, ~2.1 h instead of ~4.0 h.

**`N_SELF_REPORT=0`** drops the unprefilled detection channel — "Do you detect an injected
thought? If so, what is it about?", answered by the model in its own words. Note what goes with
it: the `leaked` class, the taxonomy row for a model that denies detection while the concept is
plainly present in its answer. `m3/judge.py` calls that "the covert regime and the class the study
exists to find". It cannot be recovered from this archive afterwards, because the responses will
not exist.

**`LAYER_FRACTIONS=0.35`** (was 0.21) puts the floor at L21. The evidence said L25 would be
safe: Silk and Wrists were both measured from L13 and both read identification 0.000 at every
layer through L27, first signal at Silk L29. 0.35 keeps **four** measured-zero layers below that
onset instead of two, which is the margin for a third model whose onset sits earlier — the
evidence is from two concepts on one model, and a floor tuned to it would not transfer.

**The battery, 150 responses per cell** (was 72), split into three generation calls. Chunking is
scientifically neutral here: `m2.expensive` corrects each row's start position for its own
padding, so chunk composition changes the padding width and nothing else
(`m2/expensive.py:124`). Both `GEN_BATCH_MAX` and `ALLOW_BATTERY_SPLIT` are hashed, so the two
models provably get the same chunk plan from the same battery, and the plan prints the list so
it can be read off both runs and compared.

That is why Qwen does **not** take a smaller `n`. Splitting costs GPU time; it does not cost
comparability.

---

## 2. The error bars you are buying

Identification, 95% Wilson half-width in **percentage points**. Worst case is p = 0.5:

| n per cell | p=0.1 | p=0.3 | **p=0.5** | p=0.7 | p=0.9 |
|---:|---:|---:|---:|---:|---:|
| 30 (before) | 11.1 | 15.6 | **16.8** | 15.6 | 11.1 |
| **120** | 5.4 | 8.1 | **8.8** | 8.1 | 5.4 |

**Influence does not improve, and cannot without new prompts.** Its n is a count of DISTINCT
task prompts, not of draws: `battery_prompts` builds the channel as `TASK_PROMPTS[:N_EFFECT]`,
and `TASK_PROMPTS` holds 22. So 22 is the ceiling — which is what the previous runs already used.
`check_prompt_supply` refuses anything higher at dry-run time now, rather than after the model
has loaded.

| n per cell | typical cell | worst decile |
|---:|---:|---:|
| **22** (the ceiling) | ±1.15 pt = ±11.5% of scale | ±1.68 pt |

Spread measured over the 99 cells with ≥8 corrected verdicts: within-cell sd, median 2.75, 90th
percentile 4.03. (The 3.29 pooled sd quoted earlier includes between-cell variation and
overstates a per-cell bar.)

Tightening influence means **writing more task prompts** — 22 more takes it to ±0.81, 44 more to
±0.66. They must be neutral for every concept in the run: nothing about food, cooking or plants
for Garlic, nothing about fabric, texture or luxury for Silk, nothing about hands or bodies for
Wrists. That changes the instrument and what every cross-run comparison means, so it is a
decision rather than a setting.

Coherence is judged on 12 effect responses plus all 4 explain responses: 16 per cell.

Identification error shrinks as 1/√n: every halving costs four times the samples.

---

## 3. What it costs

Per run: ~43,000 generations, ~43,000 judge calls, **$3.03–$3.11** judging, **≤4.0 GPU-hours**.

All six: **257k generations, 256k judge calls, $18.41 judging, ~4.0 h wall clock** with all six
cards busy. At stride 2 instead: 139k generations, $9.32, ~2.1 h.

The GPU figure is an upper bound — `m3.run` scales it linearly with battery size above its
72-prompt calibration point, which is roughly what three sequential chunks cost. Step 4 measures
the real number before you commit the pods.

---

## 4. Measure one cell before committing

Splitting removes the OOM risk, but nothing has yet run a 64-prompt batch against Qwen3-32B on a
**single** 80GB card — the 2026-08-20 Qwen runs sharded one model across three. At bf16 the
weights are ~65 GB, leaving ~15 GB, and a 64-sequence batch at ~300 tokens is roughly 5 GB of KV
cache, so it should be comfortable. Confirm rather than assume, on each pod:

```bash
cd /workspace/steering-optimization && CUDA_VISIBLE_DEVICES=0 python -m m3.run --concept Garlic --set MODEL=qwen3_32b --set "CELLS=39:0.60" --set JUDGE_ENABLED=0 --set N_IDENTIFY=120 --set N_EFFECT=22 --set N_SELF_REPORT=0 --set N_COHERENCE=12 --set N_CAPABILITY=4 --set N_EXPLAIN=4 --set GEN_BATCH_MAX=64 --set ALLOW_BATTERY_SPLIT=1
```

One cell, 150 generations in three batches. It tells you the real seconds per cell — multiply by
126 and add the boundary phase. If it OOMs, drop `GEN_BATCH_MAX` to 48 **on both pods**, so the
chunk plan stays identical.

---

## 5. Launching

Follow [`RUNBOOK-QWEN.md`](RUNBOOK-QWEN.md) §6 exactly: pin with `nohup env
CUDA_VISIBLE_DEVICES=N`, start the first run and wait for `Model loaded` before the other two so
the three do not race the same download, then confirm `gpu_count=1` on every `provenance.jsonl`.
The only change is the settings. Each launch line carries, with `$M` the pod's model:

```
--concept $C --set MODEL=$M --set LAYER_FRACTIONS=0.35,1.0 --set LAYER_STRIDE=1 --set N_IDENTIFY=120 --set N_EFFECT=22 --set N_SELF_REPORT=0 --set N_COHERENCE=12 --set N_CAPABILITY=4 --set N_EXPLAIN=4 --set GEN_BATCH_MAX=64 --set ALLOW_BATTERY_SPLIT=1 --set NULL_REPEATS=20 --set JUDGE_CONCURRENT=10
```

Gemma pod `MODEL=gemma3_27b`, Qwen pod `MODEL=qwen3_32b`; Garlic, Silk, Wrists on cards 0, 1, 2.

**Read one `--dry-run` before removing it.** The Gemma plan must say `layers 41 (L21-L61,
stride 1)`, `cells 246`, `battery 150 ... split into 3 generation batches of [64, 64, 22]`,
`config=cb29ff8b3b63`. A different hash means a `--set` did not land — and since the run folder
is named after the hash, the run would write somewhere other than where you go looking for it.

---

## 6. Keeping everything

Already the default, and worth knowing precisely what "everything" is.

Every judge call is written with its full `payload` and the judge's `raw` reply
(`m3/sweep.py:254-256`), every generated response goes to `responses_transcripts.jsonl` with its
mechanical measures, the null arm to `null_transcripts.jsonl`, and every boundary probe —
including the ones that failed — to `boundary_transcripts.jsonl` with all three legs recorded
separately. Nothing is averaged away at write time. That is what made it possible to diagnose the
Qwen vector bug from the August data months after the pod was gone.

**Archive with `tools/collect_everything.py`, not the export bundle.** `export_bundle` filters
`EXPORT_DENY` — `vectors/`, `*.pt` — because it is the *deliverable*. `collect_everything` plus
the `tar czf` it prints takes the whole runs directory including vectors, the console logs, the
git state and the environment. That is the archive; it is what `qwen_all.tgz` was, and it is what
a later run aiming at an exact operating point would start from.

It stays on a machine you control. `vectors/` never leaves the pod except into that archive.

---

## 7. Three things that are not fixed, and what they mean for publishing

**No human labels anywhere in this project.** The 110 in `m3/labels/` were written by Claude Opus
5, not by a reader — see [`m3/labels/README.md`](../m3/labels/README.md). So every judge
validation is one model agreeing with another, and gpt-4.1-mini cleared that bar at kappa 1.000
while carrying the defect that forced the re-judging. The one reference that cannot be wrong in
the same direction as a judge is the null-control arm in `tools/judge_bakeoff.py`, where DeepSeek
scores exactly 0 on 99.1% of unsteered pairs. **Before quoting a detection rate as a measured
quantity, label the 48 items in `private/judge-bakeoff/worksheet_*.txt`** — an evening's work,
and it scores DeepSeek, Sonnet and gpt-4.1-mini at once.

**The boundary criterion is a 2-of-5 tripwire.** Five probe responses, `frac_good ≥ 0.75`, so two
failures end the ladder — and one of its three legs is `on_task`, which a response drifting toward
the injected concept fails *by construction*. For Qwen, 21 of 26 boundary-defining probes were
still coherent (≥5) when rejected. This decides every dose in every cell. It is unchanged here
because changing it changes every number ever compared across runs, but it is the next thing to
look at.

**The Qwen fix is unverified on hardware.** §0 exists for that reason.

Calling this the final experiment is a decision about the study. The instrument is better than it
was — the judge is measured rather than assumed, the vectors are extracted from the prompt they
are injected into, and the intervals are half what they were. It still rests on a judge nobody has
checked against a person.
