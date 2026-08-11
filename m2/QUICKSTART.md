# M2 — quickstart

**A complete runbook from an empty RunPod account to a finished experiment.** Copy-paste, in
order. You do not need to read anything else to get a result.

Total: ~15 min of setup, ~20 min of model download, then **~50–60 min per concept** (measured,
not estimated). Cost: roughly **$1.60–2.50 per concept** — GPU dominates, judges are ~$0.50.

---

## Step 1 — Rent the pod

RunPod → **Deploy** → **Pods**.

| Setting | Value | Why |
|---|---|---|
| **GPU** | **1× A100 80GB** or **H100 80GB** | Gemma3-27B in bf16 is **~54 GB of weights alone**. A 48 GB card physically cannot hold it. |
| **Template** | any recent **PyTorch** template | it ships CUDA + torch |
| **Volume Disk** | **150 GB** (100 GB minimum) | 54 GB model + ~10 GB deps + run outputs |
| **Volume Mount Path** | `/workspace` | everything below assumes this |
| **Container Disk** | **40 GB** | pip wheels and CUDA libs |

Deploy, wait for **Running**, then **Connect → Jupyter Lab**, and in Jupyter:
**File → New → Terminal**.

> **The volume is the part that survives.** Auto-stop issues a STOP, not a terminate, so
> everything under `/workspace` is still there when you restart the pod. Anything on container
> disk is not.

---

## Step 2 — Set `HF_HOME` first, in every shell

**Before anything touches HuggingFace.** This is the single most expensive mistake available
here, and it has already happened once: the default cache is on container disk, so the 54 GB
model does not survive a pod stop and re-downloads every time.

```bash
export HF_HOME=/workspace/hf
```

`setup_pod.sh` also writes this to `~/.bashrc`, **but `~/.bashrc` is on container disk and does
not survive a new pod or a server migration.** Every new terminal, and every new pod, needs the
export again. If you only ever type one thing from memory, make it this line.

---

## Step 3 — Get the code

You need a GitHub token with read access to this repo (github.com → Settings → Developer
settings → Personal access tokens → Fine-grained, read-only on `Emergent-Introspection`).

Paste this as one block, replacing `YOUR_GITHUB_TOKEN`:

```bash
GH=YOUR_GITHUB_TOKEN
git clone https://x-access-token:$GH@github.com/PquePC/Emergent-Introspection.git /workspace/Emergent-Introspection
git -C /workspace/Emergent-Introspection remote set-url origin https://github.com/PquePC/Emergent-Introspection.git
git -C /workspace/Emergent-Introspection checkout M2
unset GH
```

The `remote set-url` afterwards keeps the token out of `.git/config`.

---

## Step 4 — Install everything

One command. Idempotent — safe to re-run after a pod restart.

```bash
bash "/workspace/Emergent-Introspection/Steering Optimization/m2/setup_pod.sh"
```

It sets `HF_HOME=/workspace/hf` (so the 54 GB download survives a stop), clones the upstream
harness from **`safety-research/introspection-mechanisms`**, installs dependencies, prints
versions, and runs the offline tests.

**Expected ending:** `53 passed`. If you see that, the code is intact.

### After a pod migration, use the doctor instead

Migrating is routine: the network volume survives, the container does not. Rather than working
out by hand which half is missing:

```bash
cd "/workspace/Emergent-Introspection/Steering Optimization"
python -m m2.doctor --repair
```

It checks the volume, `HF_HOME` and whether the model actually landed on it, the repo and
branch, the harness clone, every Python package, the GPU and its VRAM, your credentials, **what
each concept has already measured**, and the offline tests — then fixes what it can (clone the
harness, install packages, pull the repo, trim a JSONL left half-written by a hard kill) and
tells you plainly what only you can fix.

Three states, no ambiguity: `ok`, `FIX` (repairable), `BLOCK` (needs you — credentials, GPU,
the volume itself). It imports nothing heavy, so it works when the missing thing *is* the
dependencies.

Run it read-only any time with `python -m m2.doctor`, or as `python -m m2.run --doctor`.

---

## Step 5 — Credentials

**Prefix every line with a space** so it stays out of `~/.bash_history`.

```bash
 export HF_TOKEN=hf_...
 export OPENROUTER_API_KEY=sk-or-v1-...
 export TELEGRAM_BOT_TOKEN=...
 export TELEGRAM_CHAT_ID=...
 export HEALTHCHECK_URL=https://hc-ping.com/...
 export RUNPOD_API_KEY=rpa_...
```

| Variable | Where to get it | Required? |
|---|---|---|
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens (read). **Also accept the Gemma licence** on the [model page](https://huggingface.co/google/gemma-3-27b-it) or the download 403s | **yes** |
| `OPENROUTER_API_KEY` | openrouter.ai → Keys. Put ~$5 of credit on it | **yes** |
| `TELEGRAM_BOT_TOKEN` | message @BotFather, `/newbot` | no, but you are blind without it |
| `TELEGRAM_CHAT_ID` | message your new bot once, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and read `result[0].message.chat.id` | as above |
| `HEALTHCHECK_URL` | healthchecks.io → new check, **period 300 s**, copy the ping URL | no, but nothing else can tell you the pod died |
| `RUNPOD_API_KEY` | RunPod → Settings → API Keys, set `api.runpod.io/graphql` to **Read/Write** | no, only for auto-stop |

> **Never paste these into a screenshot, a chat, or an issue.** The Telegram bot token in
> particular is the access credential for the entire chat history, and Telegram cloud chats are
> not end-to-end encrypted. If one leaks: BotFather `/revoke`, RunPod key delete, OpenRouter key
> delete, HF token revoke.

---

## Step 6 — Preflight

**Do this before spending a GPU-hour.** It loads the model and runs the rig checks, then exits.
No judge calls, no measurement.

```bash
cd "/workspace/Emergent-Introspection/Steering Optimization"
python -m m2.run --concepts Garlic --preflight
```

First run downloads ~54 GB — **15–25 minutes**. Subsequent runs load from `/workspace/hf` in
about two minutes.

**Then confirm the model actually landed on the volume**, before you pay for a run:

```bash
du -sh /workspace/hf
```

Expect **~54 G**. If it is small or the directory does not exist, `HF_HOME` was not set in the
shell that did the download and the model went to container disk, where it will not survive a
stop. Fix step 2 and re-run the preflight; better to lose two minutes here than to discover it
after a pod migration.

**What you should see:**

```
credentials
  HF_TOKEN               set
  OPENROUTER_API_KEY     set
  ...
importing m2
  public surface: 77 names present
loading model
  gemma3_27b  62 layers  padding=left
rig checks
  r5_reference_norm            PASS
  r7_forced_prompts            PASS
  r14_hook_liveness            PASS
--preflight: ... Nothing measured, no judge calls spent.
```

**If R14 fails, stop.** It means the injection hook is not steering, every forward-pass measure
would read exactly zero, and a mean of zero with a variance of zero passes most checks designed
to catch a weak effect. Do not run a paid sweep on that.

---

## Step 7 — Start the watchdog

**A second terminal** (File → New → Terminal). Not the notebook, not the same shell.

```bash
export RUNPOD_API_KEY=rpa_...
nohup bash "/workspace/Emergent-Introspection/Steering Optimization/pod_watchdog.sh" > /workspace/watchdog.log 2>&1 &
```

Stops the pod after ~20 min of ≤5% GPU **while VRAM ≥ 10 GB**. The VRAM condition is what stops
a long judge wait — during which the GPU is legitimately idle — from tripping it.

---

## Step 8 — Run

Back in the first terminal:

```bash
cd "/workspace/Emergent-Introspection/Steering Optimization"
nohup python -m m2.run --concepts Garlic --no-stop-pod > /workspace/m2_garlic.out 2>&1 &
tail -f /workspace/m2_garlic.out
```

`nohup` is the point — the run survives a dropped SSH session and a closed browser. `Ctrl-C`
stops the `tail`, not the run. To re-attach: `tail -f /workspace/m2.out`.

**Why start with Garlic.** Garlic is the highest-detection concept Macar et al. published on
Gemma3-27B — **100%** at their reference configuration (L37, α=4, 100 trials) — so it is the
strongest possible "before" number, and it is citable. Origami is your control: v1 measured its
full dose–response, so the pipeline should reproduce a known answer. Alternatives: Chocolate
(99%), Trees (97%).

**One concept at a time, and `--no-stop-pod`.** Without that flag the pod stops the moment the
concept finishes and you cannot inspect the result or start the next one without a restart.
Roughly 50-60 minutes each. Watch the board in the log, or your phone.

### Useful variants

```bash
# one concept
python -m m2.run --concepts Garlic

# a list from a file (one per line, # comments allowed)
python -m m2.run --concepts-file my_concepts.txt

# also test whether spreading the dose over k layers lowers detection (+15 min/concept)
python -m m2.run --concepts Garlic --multilayer

# change a constant; this gets its own run folder and cannot resume into the old grid
python -m m2.run --concepts Garlic --set D2_MAX=0.25 --set SCAN_DOSES=0.15,0.30,0.45

# keep the loose run folders after delivery
python -m m2.run --concepts Garlic --no-wipe
```

---

## Step 9 — Read the result

```bash
cat /workspace/m2_runs/garlic_*/operating_point.json | python -m json.tool | head -40
```

`operating_point.json` is the deliverable: the winning `(layer, α, r)`, every metric with its
standard error, the control verdicts, and the frontier of near-optimal alternatives.

The three numbers that matter:

- **E5** — concept influence, 0–10, judged against the model's own unsteered reply. Needs ≥ 4.
- **D2** — forced identification rate. Needs ≤ 0.20. **This is the constraint the whole pipeline
  exists to satisfy.**
- **S4** — `min(S1, S2, S3)`: integrity, degeneracy, capability. Needs ≥ 0.70.

**If no cell qualifies, that is a result, not a failure.** The frontier is still there, and the
escalation ladder distinguishes "no operating point exists at these constraints" from "the vector
is dead". `operating_point.json` records which.

Everything else in the folder exists so a number can be traced to the generation that produced
it — `judge_e5.jsonl`, `judge_s1.jsonl`, `judge_d2.jsonl`, `D2_transcripts.jsonl`,
`cis_transcripts.jsonl`, `scan.jsonl`, `verified.jsonl`, `controls.jsonl`.

---

## Step 10 — Use the result

```bash
cd "/workspace/Emergent-Introspection/Steering Optimization"
python
```

```python
from m2 import model, config, steer
model.load_model(config.CONFIG)

steer.use("/workspace/m2_runs/garlic_abc123")      # the folder from step 8
steer.compare("Tell me a short story.")            # steered vs unsteered, side by side
steer.session()                                    # interactive; blank line to exit
steer.sweep("Describe a landscape.", [0.10, 0.15, 0.20, 0.30])
```

For a downstream experiment, four numbers are the whole dependency:

```python
with steer.steering(concept="Garlic", layer=37, r=0.20):
    ...        # anything generated in here is steered
```

---

## Step 11 — Get the results off the pod and stop it

Bundles are delivered to Telegram automatically as each concept finishes. To pull them by hand:
JupyterLab file browser → `/workspace/m2_runs/` → right-click the `.zip` → Download.

The pod auto-stops after a clean batch if `RUNPOD_API_KEY` is set. Otherwise: RunPod console →
**Stop**. **Stop, not Terminate** — stop keeps the volume, terminate destroys it.

---

## Troubleshooting

| What you see | What it means | Fix |
|---|---|---|
| `Username for 'https://github.com':` on the harness clone | wrong URL — GitHub prompts for auth instead of 404ing on a repo that does not exist | the URL is `safety-research/introspection-mechanisms`. Re-run `setup_pod.sh` |
| `Could not open requirements file` | the clone above failed, so there is nothing to install | same |
| `No module named pytest` | step 3 not run, or run in a different shell | `bash .../m2/setup_pod.sh` |
| `No module named m2` | wrong directory | `cd "/workspace/Emergent-Introspection/Steering Optimization"` |
| `refusing to start: HF_TOKEN, OPENROUTER_API_KEY not set` | credentials missing — there is no TTY under `nohup`, so nothing can prompt | step 4, in the same shell |
| 403 downloading the model | Gemma licence not accepted | accept it on the model page with the same HF account |
| Re-downloads 54 GB after a restart | `HF_HOME` on container disk | `export HF_HOME=/workspace/hf` — `setup_pod.sh` sets and persists this |
| `UNIMPORTABLE vectors: No module named 'torch'` | step 3 not run | `bash .../m2/setup_pod.sh` |
| **R14 fails** | the injection hook is not steering | **stop.** Every forward-pass measure would read zero |
| R5 fails | extraction broken, or this concept sits far from the norm distribution | checked at the reference layer only; >1σ is normal per-concept variation |
| Judge FPR warning after Phase 0 | the judge invents influence on unsteered pairs | it puts a floor under every E5 in the run — fix before spending GPU time |
| CUDA out of memory | card smaller than 80 GB, or something else on the GPU | `nvidia-smi`; the model needs ~54 GB resident |
| Run died; will re-running redo the work? | no | re-run the same command. Rows are skipped at row level and **no judge call is ever paid for twice** |
| No Telegram messages | token or chat id wrong | `python -c "from m2 import monitor; monitor.notify_test()"` |

---

## One-page summary

```bash
# 1. pod: A100 80GB, 150 GB volume at /workspace, PyTorch template, Jupyter terminal

# 2. code
GH=YOUR_GITHUB_TOKEN
git clone https://x-access-token:$GH@github.com/PquePC/Emergent-Introspection.git /workspace/Emergent-Introspection
git -C /workspace/Emergent-Introspection remote set-url origin https://github.com/PquePC/Emergent-Introspection.git
git -C /workspace/Emergent-Introspection checkout M2 && unset GH

# 3. install  (expect: 53 passed)
bash "/workspace/Emergent-Introspection/Steering Optimization/m2/setup_pod.sh"

# 4. credentials  (note the leading spaces)
 export HF_TOKEN=... OPENROUTER_API_KEY=...
 export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... HEALTHCHECK_URL=... RUNPOD_API_KEY=...

# 5. preflight  (expect: R5, R7, R14 all PASS)
cd "/workspace/Emergent-Introspection/Steering Optimization"
python -m m2.run --concepts Garlic --preflight

# 6. watchdog, in a SECOND terminal
nohup bash "/workspace/Emergent-Introspection/Steering Optimization/pod_watchdog.sh" > /workspace/watchdog.log 2>&1 &

# 7. run
nohup python -m m2.run --concepts Garlic --no-stop-pod > /workspace/m2_garlic.out 2>&1 &
tail -f /workspace/m2_garlic.out

# 8. read
cat /workspace/m2_runs/garlic_*/operating_point.json | python -m json.tool
```
