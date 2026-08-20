# Runbook — Qwen3 on three GPUs, three concepts at once

The M3 pipeline on a second model. Everything in [`RUNBOOK-M3.md`](RUNBOOK-M3.md) still applies;
this file carries only what is **different about Qwen3**, plus the three-GPU launch.

Read [`RUNBOOK-M3.md`](RUNBOOK-M3.md) §7 for what a run saves and how to get it off the pod —
that is unchanged and is not repeated here.

---

## What is different about Qwen3

| | Gemma3-27B | Qwen3-32B |
|---|---|---|
| layers | 62 | **64** — so `--n-layers 64` for an honest dry-run price |
| weights, bf16 | ~55 GB | **~66 GB** |
| chat template | no reasoning switch | **carries `enable_thinking`, defaulting to ON** |
| `MODEL` key | `gemma3_27b` | `qwen3_32b` |
| system role | stripped (Gemma has none) | kept |
| rotary patch | applied | not applied, and not needed |

Two of those matter enough to explain.

**Reasoning is off, and that is a setting.** Qwen3's chat template accepts `enable_thinking` and
defaults it to True, so an untouched run would emit a `<think>` block before every answer. This
pipeline would judge that block as the response, `MAX_NEW_TOKENS=100` would be spent thinking
before any answer appeared, and the mechanical `accept`-term check would find nothing to match.
`THINKING_MODE=off` is the default and is sent **only** to models whose own template mentions the
switch, so Gemma runs are byte-identical either way. It is a hashed setting rather than a
hardcoded `False` because reasoning-on and reasoning-off are different experiments and must not
resume into each other's run folder.

**The dose ceiling was chosen for a different model.** `dose = alpha * ‖v‖ / ‖h‖`, alpha is
capped at `ALPHA_CEIL`, and 16.0 was picked against Gemma3-27B's norms. Nothing makes that
transfer. Phase 0b checks it — see §5.

---

## 1. Rent the pod

**One pod, three A100-SXM4-80GB.** Not three pods: `/workspace/hf` is shared, so the ~66 GB
downloads once instead of three times and `m2.setup` runs once.

**Volume Disk 200 GB.** Qwen3-32B is ~66 GB. If the volume already holds Gemma3-27B from an
earlier run that is ~120 GB before any run output, and 150 GB leaves too little headroom. Either
take 200 GB, or clear Gemma first:

```bash
rm -rf /workspace/hf/models--google--gemma-3-27b-it
```

Container Disk 60 GB. NVLink needs no configuration — see §6.

## 2. Environment

Exactly as [`RUNBOOK-M3.md`](RUNBOOK-M3.md) §1.1, with the volume size raised:

```bash
unset HISTFILE
```

```bash
cat > /workspace/env.sh <<'EOF'
export HF_HOME=/workspace/hf
export M2_BRANCH=m4
export M2_VOLUME_GB=200
export HF_TOKEN=hf_PASTE_YOURS
export OPENROUTER_API_KEY=sk-or-v1-PASTE_YOURS
EOF
chmod 600 /workspace/env.sh
grep -q 'workspace/env.sh' ~/.bashrc || echo '[ -f /workspace/env.sh ] && . /workspace/env.sh' >> ~/.bashrc
. /workspace/env.sh
```

`M2_BRANCH=m4` — the Qwen work lives on `m4`. `m2.setup` **blocks** on a branch that does not
match, and never switches for you.

```bash
for v in HF_HOME M2_BRANCH M2_VOLUME_GB HF_TOKEN OPENROUTER_API_KEY; do eval "val=\$$v"; if [ -n "$val" ]; then echo "$v ok"; else echo "$v MISSING"; fi; done
```

Then confirm all three cards:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

Three rows, each `NVIDIA A100-SXM4-80GB` with 81920 MiB. A 40 GB card cannot hold this model.

## 3. Code

```bash
git clone https://github.com/PquePC/steering-optimization.git /workspace/steering-optimization
```

```bash
cd /workspace/steering-optimization && git fetch origin && git checkout -B m4 origin/m4 && git log --oneline -1
```

`checkout -B` works whether or not a local `m4` exists, which a plain `checkout` does not.

## 4. Install

```bash
cd /workspace/steering-optimization && python -m m2.setup --repair
```

`FIX` lines are normal; **no `BLOCK` lines** by the end. The upstream harness now ships in
[`upstream/introspection_mechanisms/`](../upstream/introspection_mechanisms) and is **not
cloned** — `upstream harness  OK` should name a path inside the repository. If it offers to
clone, the checkout is not where the module thinks it is; say so rather than letting it clone.

## 5. Price it, and read Phase 0b

```bash
cd /workspace/steering-optimization && python -m m3.run --concept Garlic --set MODEL=qwen3_32b --n-layers 64 --dry-run
```

Expect **26 layers L13–L63 stride 2, 156 cells, 43 responses/cell, 9,204 judge calls ≤ $4.22,
~49 min**. Three concepts in parallel is about **$12.70 of judge spend and ~1.2 h wall clock**.

The dry run loads nothing, so it cannot tell you whether the dose ceiling fits. **Phase 0b does**,
about two minutes after the weights land, on norms it has already measured — before the null
battery, which is the first thing in Phase 0 to spend a judge call:

```
PHASE 0b reachability  ALPHA_CEIL=16
   max reachable dose  min 0.1669  median 0.4681  max 2.7919
   below the 0.05 bracket floor: 0 of 25 layers
   ALPHA_CEIL to clear the floor everywhere: 4.79   to reach 2.5: 239.66
```

- **`below the floor: 0`** — proceed, the Gemma ceiling transferred.
- **a few layers below** — proceed. Those layers are recorded as `unreachable` by name and cost
  no generations; a handful of shallow ones is ordinary.
- **more than half below** — the run stops itself and names the ceiling to use. It reports rather
  than applies, because changing `ALPHA_CEIL` changes every dose in the run and the config hash
  with it. Re-launch with `--set ALPHA_CEIL=<the number it printed>`, which is a separate run
  folder and so cannot mix with anything already measured.

## 6. Launch — one concept per GPU

Each process must be pinned. The harness loads with `device_map="auto"`, so an **unpinned**
process shards one model across all three cards and the other two runs then OOM. NVLink is only
used when a single process spans GPUs, so with one model per card it is simply unused: nothing to
configure, and nothing that can misconfigure.

Start the first and **wait for `Model loaded`** before the others — all three read the same ~66 GB,
and launching together makes them race one Hugging Face download.

```bash
cd /workspace/steering-optimization && nohup env CUDA_VISIBLE_DEVICES=0 python -m m3.run --concept Garlic --set MODEL=qwen3_32b > /workspace/q_garlic.out 2>&1 &
```

```bash
grep -m1 "Model loaded" <(tail -f /workspace/q_garlic.out)
```

```bash
cd /workspace/steering-optimization && nohup env CUDA_VISIBLE_DEVICES=1 python -m m3.run --concept Wrists --set MODEL=qwen3_32b > /workspace/q_wrists.out 2>&1 &
```

```bash
cd /workspace/steering-optimization && nohup env CUDA_VISIBLE_DEVICES=2 python -m m3.run --concept Silk --set MODEL=qwen3_32b > /workspace/q_silk.out 2>&1 &
```

Run folders are `{concept}_{confighash}`, so the three cannot collide.

Confirm each process saw exactly one card — this is what the run itself observed, written into
the export, rather than something you have to catch live:

```bash
for f in /workspace/m3_runs/*/provenance.jsonl; do echo "$f"; python -c "import json,sys; p=json.loads(open(sys.argv[1],encoding='utf-8').readline()); print('   gpu_count=%s  %s  %.1f GB' % (p['gpu_count'], p['gpu'], p['gpu_total_gb']))" "$f"; done
```

`gpu_count=1` on each. A `2` or `3` means that run is sharded — kill it and relaunch pinned.

## 7. Monitor

```bash
tail -f /workspace/q_garlic.out /workspace/q_wrists.out /workspace/q_silk.out
```

Compact position of all three:

```bash
for f in q_garlic q_wrists q_silk; do printf "%-10s %s\n" "$f" "$(grep -E '^\S+ +\[[0-9]+/' /workspace/$f.out | tail -1)"; done
```

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

Each card ~68–72 GB. Qwen3-32B is ~66 GB of weights and the harness uses `attn_implementation="eager"`,
which is heavier than SDPA; if a run OOMs, `--set MODEL=qwen3_14b` is the comfortable fallback.

All three judge at 32 concurrent, so 96 in flight. If `[N judge errors]` starts appearing, relaunch
the offender with `--set JUDGE_CONCURRENCY=16`.

**Phase 0 must print `R14 pass`** with two non-zero magnitudes on each run — that is the injection
hook being live. Then **Phase 0b**, per §5. Then watch `ident=` and `ans=` per §6 of
[`RUNBOOK-M3.md`](RUNBOOK-M3.md).

## 8. Export

```bash
for c in garlic wrists silk; do cp /workspace/q_$c.out /workspace/m3_runs/${c}_*/console.log; done
```

```bash
cd /workspace/m3_runs && tar czf qwen_all.tgz garlic_*/ wrists_*/ silk_*/ && ls -lh qwen_all.tgz
```

```bash
runpodctl send /workspace/m3_runs/qwen_all.tgz
```

## 9. Stop

```bash
runpodctl stop pod $RUNPOD_POD_ID
```

Only after `runpodctl receive` has completed on the other machine. Stop rather than terminate, so
`/workspace` and the weights survive.
