# M2 — quickstart

**The only guide you need.** Copy-paste, in order, from an empty RunPod account to a finished
experiment.

Total: ~15 min of setup, ~20 min of model download, then **~50–60 min per concept** (measured,
not estimated). Cost: roughly **$1.60–2.50 per concept** — GPU dominates, judges are ~$0.50.

---

## Coming back to a pod you already used?

Skip to this. Migrating pods is routine — the network volume survives, the container does not.

```bash
export HF_HOME=/workspace/hf
cd "/workspace/Emergent-Introspection/Steering Optimization"
python -m m2.setup --repair
```

That reinstalls what the container took, pulls any code updates, tells you what each concept has
already measured, and names anything only you can fix. Then re-export your credentials
(§5) and go to §8.

If `/workspace/Emergent-Introspection` is missing, the volume did not follow — start at §1.

---

## 1. Rent the pod

RunPod → **Deploy** → **Pods**.

| Setting | Value | Why |
|---|---|---|
| **GPU** | **1× A100 80GB** or **H100 80GB** | Gemma3-27B in bf16 is **~54 GB of weights alone**. A 48 GB card physically cannot hold it, and splitting across two cards is not supported — the injection hook assumes one device. |
| **Template** | any recent **PyTorch** template | ships CUDA + torch |
| **Volume Disk** | **150 GB** (100 GB minimum) | 54 GB model + ~10 GB deps + run outputs |
| **Volume Mount Path** | `/workspace` | everything below assumes this |
| **Container Disk** | **40 GB** | pip wheels and CUDA libs |

Prefer a **network volume** if you expect to move hosts — it survives termination and can be
attached to a new pod **in the same datacentre**.

Deploy → wait for **Running** → **Connect → Jupyter Lab** → **File → New → Terminal**.

> **Stop, never Terminate**, unless you mean it. Stop keeps the volume; terminate destroys a
> pod volume. A *network* volume survives either.

---

## 2. `HF_HOME`, before anything else

**In every shell, every time.** This is the single most expensive mistake available here:

```bash
export HF_HOME=/workspace/hf
```

The default cache is on container disk, so without this the 54 GB model downloads to somewhere
that evaporates on the next stop. `m2.setup` writes it to `~/.bashrc` too, **but `~/.bashrc` is
itself on container disk** and does not survive a migration. If you memorise one line, this one.

---

## 3. Get the code

GitHub token with read access to this repo (github.com → Settings → Developer settings →
Personal access tokens → Fine-grained, read-only on `Emergent-Introspection`).

```bash
GH=YOUR_GITHUB_TOKEN
git clone https://x-access-token:$GH@github.com/PquePC/Emergent-Introspection.git /workspace/Emergent-Introspection
git -C /workspace/Emergent-Introspection remote set-url origin https://github.com/PquePC/Emergent-Introspection.git
git -C /workspace/Emergent-Introspection checkout M2
unset GH
```

The `remote set-url` afterwards keeps the token out of `.git/config`.

---

## 4. Set everything up

```bash
cd "/workspace/Emergent-Introspection/Steering Optimization"
python -m m2.setup --repair
```

**This is the install step.** It clones the upstream harness from
`safety-research/introspection-mechanisms`, installs every dependency, pulls any code updates,
and runs the offline tests. It imports nothing heavy, so it works when the missing thing *is*
the dependencies.

It also reports what it cannot fix. Expect this on a fresh pod:

```
[  ok  ] persistent volume        /workspace, 148 GB free
[ FIX  ] HF_HOME                  unset - not under /workspace
[ FIX  ] model cache              absent - the preflight will download ~54 GB (15-25 min)
[  ok  ] project repo             M2, up to date | bd22136 ...
[ FIX  ] upstream harness         not found on any searched path
[ FIX  ] python packages          missing: torch, transformers, ...
[BLOCK ] credentials (required)   missing: HF_TOKEN, OPENROUTER_API_KEY
[  ok  ] run data                 no previous runs - starting clean
```

Three states, no ambiguity:

| | meaning |
|---|---|
| `ok` | present and usable |
| `FIX` | `--repair` handles it |
| `BLOCK` | only you can — credentials, GPU, the volume itself |

`--repair` never deletes a measured row.

**Read-only any time:** `python -m m2.setup`. Also available as `python -m m2.run --setup`.

---

## 5. Credentials

**Prefix every line with a space** so it stays out of `~/.bash_history`.

```bash
 export HF_TOKEN=hf_...
 export OPENROUTER_API_KEY=sk-or-v1-...
 export TELEGRAM_BOT_TOKEN=...
 export TELEGRAM_CHAT_ID=...
 export HEALTHCHECK_URL=https://hc-ping.com/...
 export RUNPOD_API_KEY=rpa_...
```

| Variable | Where | Required? |
|---|---|---|
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens (read). **Also accept the Gemma licence** on the [model page](https://huggingface.co/google/gemma-3-27b-it) or the download 403s | **yes** |
| `OPENROUTER_API_KEY` | openrouter.ai → Keys. ~$5 of credit is plenty | **yes** |
| `TELEGRAM_BOT_TOKEN` | @BotFather → `/newbot` | no, but you are blind without it |
| `TELEGRAM_CHAT_ID` | message your bot once, then `https://api.telegram.org/bot<TOKEN>/getUpdates` → `result[0].message.chat.id` | as above |
| `HEALTHCHECK_URL` | healthchecks.io → new check, **period 300 s** | no, but nothing else can tell you the pod died |
| `RUNPOD_API_KEY` | RunPod → Settings → API Keys, `api.runpod.io/graphql` = **Read/Write** | no, only for auto-stop |

> **Never paste these into a screenshot, a chat or an issue.** The Telegram bot token is the
> access credential for the entire chat history, and Telegram cloud chats are not end-to-end
> encrypted. If one leaks: BotFather `/revoke`, delete the RunPod and OpenRouter keys, revoke
> the HF token.

---

## 6. Confirm setup is green

```bash
python -m m2.setup
```

Everything should read `ok` except the model cache, which is still absent until the preflight
downloads it. You want to see:

```
READY. Next:
  cd "/workspace/Emergent-Introspection/Steering Optimization"
  python -m m2.run --concepts Garlic --preflight
```

If anything still says `BLOCK`, fix that first — it will not get better by running.

---

## 7. Preflight

**Before spending a GPU-hour.** Loads the model, runs the rig checks, exits. No judge calls.

```bash
python -m m2.run --concepts Garlic --preflight
```

First run downloads ~54 GB — **15–25 minutes**. After that it loads in about two minutes.

```
RIG CHECKS: 2 pass, 0 FAIL, 5 SKIPPED
  skipped: R4 rig check (stored); R5 reference-layer vector norm; R8 ...; R14 ...
  summary                      PASS
```

**Read the FAIL count, not the pass count.** `0 FAIL` and `summary PASS` is the green light.

**Five of seven checks skip here, and that is correct.** R5 and R14 read the concept vectors, and
extraction *is* Phase 0 — nothing has been extracted at preflight time. Phase 0 runs both itself,
after extraction and before the first measurement; `hook_liveness` raises there, so a dead hook
aborts the run rather than being reported. R8 and R15's second half read `verified.jsonl`, which
exists from Phase 4. R4 cross-checks a stored M1.5 `rig_status.json` and skips on a fresh volume.

What the preflight actually proves: the model loads, `padding=left`, the GPU is real, and **R7** —
that the forced-ID prompt still matches the upstream repo's, token position included.

**If R14 fails in Phase 0, stop.** The injection hook is not steering; every forward-pass measure
would read exactly zero, and a mean of zero with a variance of zero passes most checks designed to
catch a weak effect.

Then confirm the model landed on the volume — two minutes now beats discovering it after the
next migration:

```bash
python -m m2.setup      # "model cache  54 GB at /workspace/hf/hub"
```

---

## 8. Run one concept

Watchdog first, in a **second terminal** (a hung process holds the GIL, so detection must live
outside it):

```bash
export RUNPOD_API_KEY=rpa_...
nohup bash "/workspace/Emergent-Introspection/Steering Optimization/pod_watchdog.sh" > /workspace/watchdog.log 2>&1 &
```

Then, back in the first:

```bash
cd "/workspace/Emergent-Introspection/Steering Optimization"
nohup python -m m2.run --concepts Garlic --no-stop-pod > /workspace/m2_garlic.out 2>&1 &
tail -f /workspace/m2_garlic.out
```

**`--no-stop-pod` matters.** Without it the pod stops the moment the concept finishes and you
cannot read the result or start the next one without a restart.

`nohup` means the run survives a dropped SSH session and a closed browser. `Ctrl-C` stops the
`tail`, not the run — re-attach with `tail -f /workspace/m2_garlic.out`.

**Why Garlic first.** It is the highest-detection concept Macar et al. published on Gemma3-27B —
**100%** at their reference configuration (L37, α=4, 100 trials) — so it is the strongest
possible "before" number, and it is citable. Alternatives: Chocolate (99%), Trees (97%).
**Origami** is the natural second: v1 measured its full dose–response, so the pipeline should
reproduce a known answer.

### Variants

```bash
python -m m2.run --concepts-file my_concepts.txt              # one per line, # comments ok
python -m m2.run --concepts Garlic --multilayer               # k in {2,3,5}, +15 min
python -m m2.run --concepts Garlic --no-wipe                  # keep the loose run folder
python -m m2.run --concepts Garlic --set D2_MAX=0.25          # own run folder, cannot resume
                                    --set SCAN_DOSES=0.15,0.30,0.45
```

---

## 9. Read the result

```bash
cat /workspace/m2_runs/garlic_*/operating_point.json | python -m json.tool | head -40
```

Three numbers decide it:

- **E5** — concept influence, 0–10, judged against the model's own unsteered reply. Needs **≥ 4**.
- **D2** — forced identification rate. Needs **≤ 0.20**. *This is the constraint the pipeline
  exists to satisfy.*
- **S4** — `min(S1, S2, S3)`: integrity, degeneracy, capability. Needs **≥ 0.70**.

Then read **gate 5** in the gates report — the Spearman ρ of the cheap D3 proxy against real D2.
Below 0.70 and the shortlist's residual ranking was unreliable, which changes how much you should
trust which cells got measured.

**If no cell qualifies, that is a result, not a failure.** The frontier is still reported, and the
escalation ladder distinguishes "no operating point exists at these constraints" from "the vector
is dead". `operating_point.json` records which.

Everything else in the folder exists so a number traces to the generation that produced it:
`judge_e5.jsonl`, `judge_s1.jsonl`, `judge_d2.jsonl`, `D2_transcripts.jsonl`,
`cis_transcripts.jsonl`, `scan.jsonl`, `verified.jsonl`, `controls.jsonl`, `provenance.jsonl`.

---

## 10. Use the result

```python
from m2 import model, config, steer
model.load_model(config.CONFIG)

steer.use("/workspace/m2_runs/garlic_abc123")      # the folder from §9
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

## 11. Next concept, and finishing up

```bash
nohup python -m m2.run --concepts Origami --no-stop-pod > /workspace/m2_origami.out 2>&1 &
```

Bundles go to Telegram as each concept finishes. To pull one by hand: JupyterLab file browser →
`/workspace/m2_runs/` → right-click the `.zip` → Download.

When done: RunPod console → **Stop** (not Terminate). The pod auto-stops after a batch if
`RUNPOD_API_KEY` is set and you did not pass `--no-stop-pod`.

---

## Troubleshooting

**First move for almost everything below:** `python -m m2.setup`. It names the problem and
whether `--repair` can fix it.

| What you see | What it means | Fix |
|---|---|---|
| `No module named pytest` / `torch` / `m2` | dependencies gone (container disk) or wrong directory | `cd` to the project dir, then `python -m m2.setup --repair` |
| `Username for 'https://github.com':` on a clone | wrong URL — GitHub prompts for auth rather than 404ing | the harness is `safety-research/introspection-mechanisms`; `--repair` uses the right one |
| `Could not open requirements file` | the clone above failed | same |
| `refusing to start: HF_TOKEN, OPENROUTER_API_KEY not set` | no TTY under `nohup`, so nothing can prompt | §5, in the same shell |
| 403 downloading the model | Gemma licence not accepted | accept it on the model page with the same HF account |
| Re-downloads 54 GB after a restart | `HF_HOME` was unset when it first downloaded | §2. The setup detects a model stranded outside `HF_HOME` and tells you how to move it |
| Run folder looks empty after a migration | volume did not follow, or a hard kill truncated a file | `python -m m2.setup` prints per-concept row counts and repairs truncated JSONL |
| Preflight says `5 SKIPPED` | R5/R14 need vectors, R8/R15b need Phase 4 rows, R4 needs an M1.5 folder | expected. `0 FAIL` and `summary PASS` is the green light; Phase 0 runs R5 and R14 for real |
| **R14 fails** | the injection hook is not steering | **stop.** Every forward-pass measure would read zero |
| R5 fails | extraction broken, or this concept is far from the norm distribution | reference layer only; >1σ is normal per-concept variation |
| Judge FPR warning after Phase 0 | the judge invents influence on unsteered pairs | it puts a floor under every E5 — fix before spending GPU time |
| `status board unavailable` | board could not be built | fixed in `276e3be`; `--repair` pulls it |
| CUDA out of memory | card under 80 GB, or something else on the GPU | `nvidia-smi`; the model needs ~54 GB resident |
| Run died — will re-running redo the work? | no | re-run the same command. Rows skip at row level and **no judge call is ever paid for twice** |
| No Telegram messages | token or chat id wrong | `python -c "from m2 import monitor; monitor.notify_test()"` |

---

## One page

```bash
# 1. pod: A100 80GB, 150 GB volume at /workspace, PyTorch template, Jupyter terminal

# 2. every shell, always
export HF_HOME=/workspace/hf

# 3. code
GH=YOUR_GITHUB_TOKEN
git clone https://x-access-token:$GH@github.com/PquePC/Emergent-Introspection.git /workspace/Emergent-Introspection
git -C /workspace/Emergent-Introspection remote set-url origin https://github.com/PquePC/Emergent-Introspection.git
git -C /workspace/Emergent-Introspection checkout M2 && unset GH

# 4. set up / check everything
cd "/workspace/Emergent-Introspection/Steering Optimization"
python -m m2.setup --repair

# 5. credentials  (leading spaces)
 export HF_TOKEN=... OPENROUTER_API_KEY=...
 export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... HEALTHCHECK_URL=... RUNPOD_API_KEY=...

# 6. confirm green
python -m m2.setup

# 7. preflight  (expect 0 FAIL and summary PASS; R5/R8/R14 skip until they have data)
python -m m2.run --concepts Garlic --preflight

# 8. watchdog, SECOND terminal
nohup bash "/workspace/Emergent-Introspection/Steering Optimization/pod_watchdog.sh" > /workspace/watchdog.log 2>&1 &

# 9. run
nohup python -m m2.run --concepts Garlic --no-stop-pod > /workspace/m2_garlic.out 2>&1 &
tail -f /workspace/m2_garlic.out

# 10. read
cat /workspace/m2_runs/garlic_*/operating_point.json | python -m json.tool
```

**Returning to a migrated pod:** steps 2, 4, 5, then 9.
