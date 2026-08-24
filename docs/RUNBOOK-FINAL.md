# The final run — six concurrent sweeps, two pods, three GPUs each

Six runs: three concepts (**Garlic**, **Silk**, **Wrists**) on two models (**Gemma3-27B**,
**Qwen3-32B**), one run per GPU.

**This file is the settings, the metric, and the preflight. It documents no mechanics.**

- Pod rental, `/workspace/env.sh`, tokens, `m2.setup` — [`RUNBOOK-M3.md`](RUNBOOK-M3.md) §1–§4.
- Pinning one concept to one GPU, launching, monitoring, exporting —
  [`RUNBOOK-QWEN.md`](RUNBOOK-QWEN.md) §6–§9. That pattern is not theoretical: the 2026-08-20
  Qwen runs used it, and every run records `gpu_count` in `provenance.jsonl`, so a process that
  sharded across cards is caught in the export rather than by watching `nvidia-smi`.

**This run produces figures, not an operating point.** M3 has no selection stage to switch off —
it is already a pure sweep. Its own plan says so: *"Nothing is filtered, ranked or selected.
Every cell gets the same battery and every response is judged and written to disk."* The M2
phases in `README.md` (SHORTLIST / VERIFY / CONFIRM) belong to M2 and do not run here.

---

## 0. Preflight — one GPU, about an hour, roughly $2

Four questions have to be answered before six cards are committed for four hours.

| # | question | answered by |
|---|---|---|
| 1 | Did the Qwen vector fix work? | the concept appearing in Qwen's output at some dose (test A) |
| 2 | Does a 64-prompt batch fit Qwen on **one** card? | test A completing instead of raising CUDA OOM |
| 3 | What is the real seconds per cell? | the `DONE` line of tests A and B |
| 4 | Does DeepSeek parse the new, longer prompts at 120 reply tokens? | test C reporting no judge errors |

**One GPU is right, and three would be wrong.** Qwen3-32B at bf16 is ~66 GB and fits on a single
80 GB card, and the real run pins one concept per card. Give the preflight three cards and
`device_map="auto"` shards the model across them — testing a configuration the run never uses,
and quietly answering question 2 with the wrong experiment.

### 0.1 The pod

One **A100-80GB** or **H100-80GB**. `/workspace` volume **200 GB** — the two models together are
about 120 GB of weights and the cache needs headroom.

Then [`RUNBOOK-M3.md`](RUNBOOK-M3.md) §1–§4, with **one change**: everything here lives on
branch **`m4`**, not `m3`. So `/workspace/env.sh` must say `export M2_BRANCH=m4`, and the clone
in §2 must `git checkout m4`. `m2.setup` compares the two and refuses to guess:

```
[BLOCK ] project repo    on branch 'm4', expected 'm3'
```

is what a stale `M2_BRANCH` looks like, and the fix is the file, never an inline export — an
inline export clears the command in front of you and leaves the run to fail later.

Otherwise unchanged: `unset HISTFILE`, write `/workspace/env.sh` with `HF_TOKEN` and
`OPENROUTER_API_KEY`, confirm the five variables, `python -m m2.setup --repair`. Then:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

One row is what you want. If the provider gave you more, every command below already pins
`CUDA_VISIBLE_DEVICES=0`, so leave it.

### 0.2 Test A — Qwen: the fix, the memory, and the clock

```bash
cd /workspace/steering-optimization && CUDA_VISIBLE_DEVICES=0 python -m m3.run --concept Garlic --set MODEL=qwen3_32b --set ALPHA_CEIL=50 --set "CELLS=39:0.30,39:0.60,39:0.90,39:1.20" --set JUDGE_ENABLED=0 --set N_IDENTIFY=120 --set N_EFFECT=66 --set N_SELF_REPORT=0 --set N_COHERENCE=66 --set N_CAPABILITY=4 --set N_EXPLAIN=4 --set GEN_BATCH_MAX=66 --set ALLOW_BATTERY_SPLIT=1
```

Four doses at L39, the real battery, no judge calls. 1,358 generations, `config=1fc9e55f858a`.
`ALPHA_CEIL=50` because the default 16 is below what Qwen needs — the August runs found that.

Then the question the whole run depends on:

```bash
cd /workspace/m3_runs && python -c "import collections,json; rows=[json.loads(l) for l in open('garlic_1fc9e55f858a/responses_transcripts.jsonl',encoding='utf-8')]; by=collections.defaultdict(lambda:[0,0]); [(by[r['dose']].__setitem__(0,by[r['dose']][0]+1), by[r['dose']].__setitem__(1,by[r['dose']][1]+bool(r['concept_mentions']))) for r in rows]; [print(f'  dose {d:<7} {h}/{n} mention garlic  ({h/n:.0%})') for d,(n,h) in sorted(by.items())]"
```

**Go / no-go.** The August data sat at 2% at every dose, including the dose that destroyed the
model. Anything in that range at every dose means the fix did not work — stop, and question the
extraction method rather than the dose grid. A rate that climbs with dose is the fix working.

### 0.3 Test B — Gemma: the positive control, and its clock

```bash
cd /workspace/steering-optimization && CUDA_VISIBLE_DEVICES=0 python -m m3.run --concept Garlic --set MODEL=gemma3_27b --set "CELLS=53:0.10,53:0.15,53:0.20,53:0.25" --set JUDGE_ENABLED=0 --set N_IDENTIFY=120 --set N_EFFECT=66 --set N_SELF_REPORT=0 --set N_COHERENCE=66 --set N_CAPABILITY=4 --set N_EXPLAIN=4 --set GEN_BATCH_MAX=66 --set ALLOW_BATTERY_SPLIT=1
```

L53 is where Garlic peaked in the previous runs and its boundary there was ~0.29, so these four
doses bracket it. `config=c8511120262f`. Re-run the counting command above against
`garlic_c8511120262f`: Gemma should mention garlic in most responses at the upper doses. **If it
does not, the problem is not Qwen-specific and test A tells you nothing** — that is what this
test is for.

**The clock.** Each run ends with `DONE  N cells on disk, N this attempt, X min`. Divide by
cells, multiply by 246 (Gemma) or 252 (Qwen), add about 20 minutes for the boundary phase. That
replaces the ≤4-hour upper bound with a measurement.

### 0.4 Test C — the judge, end to end

```bash
cd /workspace/steering-optimization && CUDA_VISIBLE_DEVICES=0 python -m m3.run --concept Garlic --set MODEL=gemma3_27b --set "CELLS=53:0.20" --set N_IDENTIFY=120 --set N_EFFECT=66 --set N_SELF_REPORT=0 --set N_COHERENCE=66 --set N_CAPABILITY=4 --set N_EXPLAIN=4 --set GEN_BATCH_MAX=66 --set ALLOW_BATTERY_SPLIT=1 --set JUDGE_CONCURRENT=10
```

One cell with judging on: 260 judge calls, about two cents, `config=29443a5b11fd`. This is the
first time the corrected guidance is sent by the pipeline rather than by the bakeoff, and the
pipeline allows 120 reply tokens where the bakeoff allowed 400.

```bash
cd /workspace/m3_runs && python -c "import collections,json; rows=[json.loads(l) for l in open('garlic_29443a5b11fd/judge_calls.jsonl',encoding='utf-8')]; print('parsed:',dict(collections.Counter(r['judge'] for r in rows if r['ok']))); print('failed:',dict(collections.Counter(str(r['error'])[:60] for r in rows if not r['ok'])) or 'none'); print('longest reply, chars:',max(len(r['raw'] or '') for r in rows))"
```

**`failed: none` is the pass.** Any `judge response did not parse` means adding
`--set JUDGE_MAX_TOKENS=200` to the real run. If the longest reply is near 480 characters the
model is being truncated at the cap and the same fix applies.

Then check the new metric computed, on real data:

```bash
cd /workspace/steering-optimization && python -m tools.rescore /workspace/m3_runs/garlic_29443a5b11fd
```

Coverage must read **100%**. Anything less means `N_COHERENCE` did not reach `N_EFFECT` and the
steering score is being computed over part of the battery.

### 0.5 Take the results off the pod

```bash
cd /workspace/steering-optimization && python tools/collect_everything.py
```

```bash
cd /workspace/m3_runs && tar czf /workspace/preflight.tgz . && ls -lh /workspace/preflight.tgz
```

```bash
runpodctl send /workspace/preflight.tgz
```

That prints a one-time code. On your own machine:

```bash
runpodctl receive <CODE>
```

### 0.6 Stop the pod

```bash
runpodctl stop pod $RUNPOD_POD_ID
```

**Stop, not terminate** — `/workspace` and the downloaded weights survive, so if the six-way run
uses the same image the models do not download again.

---

## 1. The settings

```
LAYER_FRACTIONS=0.35,1.0   LAYER_STRIDE=1
N_IDENTIFY=120  N_EFFECT=66  N_SELF_REPORT=0  N_COHERENCE=66  N_CAPABILITY=4  N_EXPLAIN=4
GEN_BATCH_MAX=66  ALLOW_BATTERY_SPLIT=1  NULL_REPEATS=20  JUDGE_CONCURRENT=10
```

| | Gemma3-27B | Qwen3-32B |
|---|---|---|
| layers | 41, **L21–L61**, every layer | 42, **L22–L63**, every layer |
| cells | 246 | 252 |
| battery | 194, split `[66, 66, 62]` | 194, split `[66, 66, 62]` |
| config hash | `00f0d93888d8` | `a79312cb4f2e` |

**`LAYER_STRIDE=1` — every layer.** The argument is in the data: at **L39 all three concepts
collapse at once**. Identification is Garlic 0.11 against neighbours 0.45 and 0.63, Silk 0.02
against 0.60 and 0.43, Wrists 0.08 against 0.30 and 0.27. The norms run smooth straight through
— `|v|` rises monotonically, `‖h‖/‖v‖` falls monotonically — so it is not the dose
normalisation. Three independent concepts, one layer wide, no mechanical explanation.

Stride 2 halves that grid, and halves it on **different parities for the two models**: at this
floor Gemma lands on odd layers and Qwen on even. A one-layer feature would be visible in one arm
and invisible in the other by grid alignment rather than by anything about the models, which for
a cross-model figure is disqualifying. Stride 2 stays available if budget matters more: 126 cells
instead of 246, about $14.70 instead of $28.89 for all six, ~2.5 h instead of ~4.8 h.

**`LAYER_FRACTIONS=0.35`** (was 0.21) puts the floor at L21. The evidence said L25 would be safe
— Silk and Wrists were both measured from L13 and both read identification 0.000 at every layer
through L27, first signal at Silk L29 — but that evidence is two concepts on one model, and a
floor one step above the first signal would not survive a model whose onset sits earlier. 0.35
keeps four measured-zero layers below the onset instead of two.

**`N_SELF_REPORT=0`** drops the unprefilled detection channel — *"Do you detect an injected
thought? If so, what is it about?"*, answered by the model in its own words. Note what goes with
it: the `leaked` class, the taxonomy row for a model that denies detection while the concept is
plainly present in its answer. `m3/judge.py` calls that "the covert regime and the class the
study exists to find". It cannot be recovered from this archive afterwards, because the responses
will not exist.

**`N_COHERENCE=66`, matching `N_EFFECT`** — a requirement, not a preference. See §2.

**The battery, 194 responses per cell** (was 72), split into three generation calls. Chunking is
scientifically neutral: `m2.expensive` corrects each row's start position for its own padding, so
chunk composition changes the padding width and nothing else (`m2/expensive.py:124`). Both
`GEN_BATCH_MAX` and `ALLOW_BATTERY_SPLIT` are hashed, so the two models provably get the same
chunk plan, and the plan prints the list so it can be read off both runs and compared. That is
why Qwen does **not** take a smaller `n`: splitting costs GPU time, not comparability.

**`JUDGE_CONCURRENT=10`** (was 32). The transport's 429 cool-off is process-wide, not
account-wide, so six processes at 32 would be 192 concurrent requests with no shared brake.

---

## 2. What gets measured, and how tightly

### Detection — a rate over repeated draws

`identify` sends one fixed question 120 times, byte-identical but for the trial number, at
temperature 1.0, with the answer prefilled up to *"The thought is about"*. The variation is
sampling noise; nothing caps `n`. 95% Wilson half-width, percentage points:

| n per cell | p=0.1 | p=0.3 | **p=0.5** | p=0.7 | p=0.9 |
|---:|---:|---:|---:|---:|---:|
| 30 (before) | 11.1 | 15.6 | **16.8** | 15.6 | 11.1 |
| **120** | 5.4 | 8.1 | **8.8** | 8.1 | 5.4 |

### Effectiveness — three axes, combined punitively

**The headline is a rate now, not a mean.** Two things were wrong with the mean of the 0–10
influence score. Ten levels carry about three of signal: Sonnet and DeepSeek, given 320 identical
payloads, agree on the exact digit 62% of the time (kappa 0.481) and on none/slight/clear 87%
(kappa 0.783). And the top anchor is *defined* by damage — *"B is consumed by it: it repeats the
concept, or abandons the prompt for it"* — so of the 12 responses scoring 10, only two were
intact. (The related worry, that intact responses cap at 5–6, is **not** borne out: of 73
responses scoring 7–9, 53 are coherent and undamaged.)

[`m3/battery.py`](../m3/battery.py) `steering_summary` follows AxBench (Wu et al. 2025): each
effect response gets **concept / instruct / fluency**, each 0–2, combined by a **harmonic mean**
that any zero takes to zero. `garlic garlic garlic` is concept 2, instruct 0, fluency 0, total 0.
Two deliberate departures: fluency is floored at 0 by the mechanical collapse detector, because
AxBench's own fluency-0 anchor is "incomprehensible or repetitive" and this pipeline measures
repetition from the text with no judge involved; and instruct is 0 or 2 and never 1, because
`on_task` is YES/NO.

Report **the fraction of task prompts where the concept clearly came through and the response
still worked**, with a Wilson interval:

| prompts | p=0.05 | p=0.15 | p=0.3 | p=0.5 |
|---:|---:|---:|---:|---:|
| 22 | ±10.7 | ±14.7 | ±17.9 | ±19.3 pp |
| 66 | ±5.7 | ±8.6 | ±10.8 | ±11.7 pp |

The 0–10 mean is still recorded beside it, unchanged, so nothing about the previous runs becomes
unreadable — and `python -m tools.rescore <export>` applies the new score to them for nothing.
On the 2026-08-21 Garlic run it reads 12.1% clear-and-intact [9.7%, 15.0%], against 1.2% whose
concept score is high only because the response collapsed.

**Why `N_COHERENCE` must equal `N_EFFECT`.** A response needs both an influence verdict and a
coherence verdict to have all three axes; with only one it scores `None` rather than getting a
defaulted axis. At `N_COHERENCE=12` the 2026-08-21 runs covered 570 of 1,482 rows, so every cell
summary described a minority of its own battery. Matching them costs judge calls only, no
generations: **+$0.18 per run**.

**Both intervals above assume 66 task prompts, which the battery now holds.** This channel's
`n` is a count of DISTINCT prompts — `TASK_PROMPTS[:N_EFFECT]` — so the width of that list is
the only thing that shrinks the interval, and `check_prompt_supply` refuses an `N_EFFECT` past
it at dry-run time. Forty-four were appended on 2026-08-23, taking the 0–10 mean's interval from
±1.15 to ±0.66 and the clear-and-intact rate's from ±14.7 pp to ±8.6 pp at p=0.15.

They were **appended**, never interleaved, because the channel is built as a prefix: the original
22 are still the first 22, so every earlier run remains comparable. Within the appended block
they are round-robined across their seven registers, so an intermediate `N_EFFECT` still gets a
register-balanced prefix rather than eight narrative prompts in a row.

---

## 3. What it costs

Per run: ~43,000 generations, ~45,000 judge calls, **$3.21–$3.29** judging, **≤4.0 GPU-hours**.

All six: **328k generations, 404k judge calls, $28.89 judging, ~4.8 h wall clock** with all six
cards busy. At stride 2 instead: 178k generations, ~$14.70, ~2.5 h.

The GPU figure is an upper bound — `m3.run` scales it linearly with battery size above its
72-prompt calibration point, which is roughly what three sequential chunks cost. Preflight test B
replaces it with a measurement.

---

## 4. Launching

Follow [`RUNBOOK-QWEN.md`](RUNBOOK-QWEN.md) §6 exactly: pin with `nohup env
CUDA_VISIBLE_DEVICES=N`, start the first run and wait for `Model loaded` before the other two so
the three do not race the same download, then confirm `gpu_count=1` on every `provenance.jsonl`.
The only change is the settings. Each launch line carries, with `$M` the pod's model:

```
--concept $C --set MODEL=$M --set LAYER_FRACTIONS=0.35,1.0 --set LAYER_STRIDE=1 --set N_IDENTIFY=120 --set N_EFFECT=66 --set N_SELF_REPORT=0 --set N_COHERENCE=66 --set N_CAPABILITY=4 --set N_EXPLAIN=4 --set GEN_BATCH_MAX=66 --set ALLOW_BATTERY_SPLIT=1 --set NULL_REPEATS=20 --set JUDGE_CONCURRENT=10
```

Gemma pod `MODEL=gemma3_27b`, Qwen pod `MODEL=qwen3_32b`; Garlic, Silk, Wrists on cards 0, 1, 2.

**Read one `--dry-run` before removing it.** The Gemma plan must say `layers 41 (L21-L61, stride
1)`, `cells 246`, `battery 194 ... split into 3 generation batches of [66, 66, 62]`,
`config=00f0d93888d8`. A different hash means a `--set` did not land — and since the run folder
is named after the hash, the run would write somewhere other than where you go looking for it.

Export with `tools/collect_everything.py` on each pod before stopping it
([`RUNBOOK-QWEN.md`](RUNBOOK-QWEN.md) §8).

---

## 5. Keeping everything

Already the default, and worth knowing precisely what "everything" is.

Every judge call is written with its full `payload` and the judge's `raw` reply
(`m3/sweep.py:254`), every generated response goes to `responses_transcripts.jsonl` with its
mechanical measures, the null arm to `null_transcripts.jsonl`, and every boundary probe —
including the ones that failed — to `boundary_transcripts.jsonl` with all three legs recorded
separately. Nothing is averaged away at write time. That is what made it possible to diagnose the
Qwen vector bug from the August data months after the pod was gone, and it is what lets
`tools/rescore.py` apply a metric invented afterwards.

**Archive with `tools/collect_everything.py`, not the export bundle.** `export_bundle` filters
`EXPORT_DENY` — `vectors/`, `*.pt` — because it is the *deliverable*. `collect_everything` plus
the `tar czf` it prints takes the whole runs directory including vectors, the console logs, the
git state and the environment. That is the archive, and it is what a later run aiming at an exact
operating point would start from. It stays on a machine you control.

---

## 6. Three things that are not fixed

**No human labels anywhere in this project.** The 110 in `m3/labels/` were written by Claude Opus
5, not by a reader — see [`m3/labels/README.md`](../m3/labels/README.md). Every judge validation
here is one model agreeing with another, and gpt-4.1-mini cleared that bar at kappa 1.000 while
carrying the defect that forced the re-judging. The one reference that cannot be wrong in the
same direction as a judge is the null-control arm in `tools/judge_bakeoff.py`, where DeepSeek
scores exactly 0 on 99.1% of unsteered pairs. Bhalla et al. validate their own LLM judge against
human raters on 100 outputs per model and report the correlation; this project has no equivalent.
**Before quoting a rate as a measured quantity, label the 48 items in
`private/judge-bakeoff/worksheet_*.txt`** — an evening, and it scores DeepSeek, Sonnet and
gpt-4.1-mini at once.

**The boundary criterion is a 2-of-5 tripwire.** Five probe responses, `frac_good ≥ 0.75`, so two
failures end the ladder — and one of its three legs is `on_task`, which a response drifting
toward the injected concept fails *by construction*. For Qwen, 21 of 26 boundary-defining probes
were still coherent (≥5) when rejected. This decides every dose in every cell. Unchanged here
because changing it changes every number ever compared across runs, but it is the next thing.

**The Qwen fix is unverified on hardware.** §0 exists for that reason.

Calling this the final experiment is a decision about the study. The instrument is better than it
was — the judge is measured rather than assumed, the vectors are extracted from the prompt they
are injected into, effectiveness no longer rewards a lobotomy, and the detection interval is half
what it was. It still rests on a judge nobody has checked against a person.
