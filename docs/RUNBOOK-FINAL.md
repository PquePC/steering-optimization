# The final run — six concurrent sweeps, two pods, three GPUs each

Six runs: three concepts (**Garlic**, **Silk**, **Wrists**) on two models (**Gemma3-27B**,
**Qwen3-32B**), one run per GPU.

**This file is the settings and the two go/no-go checks. It documents no mechanics.**

- Pod rental, `/workspace/env.sh`, tokens, `m2.setup` — [`RUNBOOK-M3.md`](RUNBOOK-M3.md) §1–§4.
- Pinning one concept to one GPU, launching, monitoring, exporting —
  [`RUNBOOK-QWEN.md`](RUNBOOK-QWEN.md) §6–§9. That pattern is not theoretical: the 2026-08-20
  Qwen runs used it, and every run records `gpu_count` in its `provenance.jsonl`, so a process
  that sharded across cards is caught in the export rather than by watching `nvidia-smi`.

What is new here is that **two** pods now run the same six-way plan, one model each, and that
the settings differ from the shipped defaults.

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

## 1. The settings, and why each one moved

```
LAYER_FRACTIONS=0.41,1.0   LAYER_STRIDE=2
N_IDENTIFY=120  N_EFFECT=66  N_SELF_REPORT=36  N_COHERENCE=12  N_CAPABILITY=4  N_EXPLAIN=4
GEN_BATCH_MAX=230  NULL_REPEATS=20  JUDGE_CONCURRENT=10
```

**`LAYER_FRACTIONS=0.41`** (was 0.21). Verified against the runs: Silk and Wrists were measured
from L13 and both read identification **0.000 at every layer from L13 to L27**, with influence
0.00–0.50. The first signal is Silk L29 (influence 2.08) and L31 (identification 0.27); Wrists
starts at L33. 0.41 puts the floor at **L25**, which keeps two measured-zero layers below the
onset so a figure shows the onset rather than beginning at it. It also reproduces the exact odd
parity of the previous Gemma grid, so every new cell has an old counterpart at the same layer.

**No ceiling. The data contradicts it.** For Garlic, L54–L58 is where influence *peaks* — 4.58,
5.00, 5.79, 5.62 under the old judge and 3.95–3.26 under the corrected one, at identification
0.79–0.81 and coherence 7.2–8.3. Cutting there removes the most influential cells in the study.
What is true is narrower: at **L59–L61** concept mentions jump from 1–4 per response to **17–18**
and degeneration rises to 0.17–0.25 — that is flooding, not inertness, and it is the over-steer
evidence. And the noun-swapping is real but belongs to **Wrists**, not to a layer band: it emits
"wristwatches", "wristlets", "anklets" across L47–L59, which the strict match rule scores as
misses. That is a finding about the concept, worth keeping.

**`LAYER_STRIDE=2`** over 19 layers instead of the 25-layer grid — 114 cells, down from 150.
Every saved cell buys precision, which is the trade you asked for.

**The battery, 230 responses per cell (was 72).** Error bars shrink as 1/√n, so this is the whole
story:

| | n per cell | 95% interval | vs before |
|---|---:|---:|---:|
| identification (Wilson at p=0.5) | 30 → **120** | ±0.168 → **±0.088** | 1.9× tighter |
| influence (sd 3.29, measured over 2,574 corrected verdicts) | 22 → **66** | ±1.37 → **±0.79** | 1.7× tighter |
| self-report | 12 → **36** | ±0.28 → **±0.16** | 1.7× tighter |

Halving an interval costs four times the generations. ±0.088 is what 4× buys; ±0.045 would cost
16× and put the battery past what one card can hold.

**`JUDGE_CONCURRENT=10`** (was 32). The transport's 429 cool-off is process-wide, not
account-wide, so six processes at 32 would be 192 concurrent requests with no shared brake.

---

## 2. What it costs

Per run: 31,960 generations, 28,728 judge calls, **$2.06** of judging, **≤2.6 GPU-hours**.

All six: **192k generations, 172k judge calls, $12.35 of judging, ~2.6 h wall clock** with all
six cards working at once.

The GPU figure is an upper bound — `m3.run` scales it linearly with battery size above its
72-prompt calibration point, and a 27B model at that batch is nowhere near saturating an A100, so
the real number should come in under it. Step 3 measures it.

---

## 3. The memory constraint, which is the real one

`device_map="auto"` places the model on whatever GPUs the process can see, so `CUDA_VISIBLE_DEVICES`
is what gives one run one card. Weights at bf16:

| model | weights | free on an 80GB card | battery 230 KV (≈300 tok × 230 seq) |
|---|---:|---:|---:|
| Gemma3-27B | ~54 GB | ~26 GB | comfortable |
| Qwen3-32B | ~65 GB | ~15 GB | **tight — may OOM** |

Battery 72 is proven on one 80GB card for Gemma. 230 is not proven for either, and the earlier
Qwen runs used all three GPUs for one model.

**Measure it before committing the pod.** On each pod, one cell at the real battery:

```bash
cd /workspace/steering-optimization && CUDA_VISIBLE_DEVICES=0 python -m m3.run --concept Garlic --set MODEL=qwen3_32b --set "CELLS=39:0.60" --set JUDGE_ENABLED=0 --set N_IDENTIFY=120 --set N_EFFECT=66 --set N_SELF_REPORT=36 --set N_CAPABILITY=4 --set N_EXPLAIN=4 --set GEN_BATCH_MAX=230
```

It either completes in a minute or dies on CUDA OOM, and it tells you the real seconds per cell.
**If Qwen OOMs**, take the smaller battery on the Qwen pod (`N_IDENTIFY=60 N_EFFECT=44
N_SELF_REPORT=24 GEN_BATCH_MAX=136`, identification ±0.123) rather than sharding — different N
between models is fine, each rate carries its own interval, whereas a different battery
*composition* would not be.

---

## 4. Launching

Follow [`RUNBOOK-QWEN.md`](RUNBOOK-QWEN.md) §6 exactly: pin with `nohup env
CUDA_VISIBLE_DEVICES=N`, start the first run and wait for `Model loaded` before the other two so
the three do not race the same download, then confirm `gpu_count=1` on every `provenance.jsonl`.
The only change is the settings. Each launch line carries, with `$M` the pod's model:

```
--concept $C --set MODEL=$M --set LAYER_FRACTIONS=0.41,1.0 --set LAYER_STRIDE=2 --set N_IDENTIFY=120 --set N_EFFECT=66 --set N_SELF_REPORT=36 --set N_COHERENCE=12 --set N_CAPABILITY=4 --set N_EXPLAIN=4 --set GEN_BATCH_MAX=230 --set NULL_REPEATS=20 --set JUDGE_CONCURRENT=10
```

Gemma pod `MODEL=gemma3_27b`, Qwen pod `MODEL=qwen3_32b`; Garlic, Silk, Wrists on cards 0, 1, 2.

**Read one `--dry-run` before removing it.** The Gemma plan must say `layers 19 (L25-L61, stride
2)`, `cells 114`, `battery 230`, `config=81c5576e5166`. A different hash means a `--set` did not
land — and since the run folder is named after the hash, the run would write somewhere other
than where you go looking for it.

Export with `tools/collect_everything.py` on each pod before stopping it
([`RUNBOOK-QWEN.md`](RUNBOOK-QWEN.md) §8).

---

## 5. Three things that are not fixed, and what they mean for publishing

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
