# Runbook — bare pod to a measured operating point

**Authority on: how to run the pipeline.** The only operating guide; it absorbed the three that
overlapped it (`RUNBOOK.md`, `M2 — HOW TO RUN.md` and the operational half of
`m2/README.md`). For what each measure *means*, see [`SPECIFICATION.md`](SPECIFICATION.md).

---


**The only guide you need.** Copy-paste, in order, from an empty RunPod account to a finished
experiment.

Total: ~15 min of setup, ~20 min of model download, then **~1h20m per concept**. Cost: roughly
**$2–3 per concept** — GPU dominates, judges are well under $1.

> **Branch.** The pipeline runs on **`pareto`**, not `main`. `main` carries the superseded
> layer-then-dose selection, without tasks 21, 24 and 26. Set `M2_BRANCH=pareto` **before** running
> setup: a branch mismatch is reported as `BLOCKED` and is never switched for you, because a setup
> tool must not decide which code your run executes.

---

## Coming back to a pod you already used?

Skip to this. Migrating pods is routine — the network volume survives, the container does not.

```bash
export HF_HOME=/workspace/hf M2_BRANCH=pareto M2_VOLUME_GB=150
cd "/workspace/steering-optimization"
python -m m2.setup
```

That automatically reinstalls the Python packages the container took, tells you what each concept
has already measured, and names anything only you can fix. If it reports a non-package `FIX`,
review it and use `--repair` explicitly. Then re-export your credentials (§5) and go to §8.

If `/workspace/steering-optimization` is missing, the volume did not follow — start at §1.

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

## 2. The environment, before anything else

**Set these in the RunPod deploy form's Environment Variables section**, not by hand:

| Variable | Value | Why |
|---|---|---|
| `HF_HOME` | `/workspace/hf` | the default cache is on container disk, so without this the 54 GB model downloads somewhere that evaporates on the next stop |
| `M2_BRANCH` | `pareto` | the branch the pipeline runs on; setup reports `BLOCKED` if you are elsewhere |
| `M2_VOLUME_GB` | `150` | inert unless RunPod's allocation API cannot be read, in which case it prevents a stall. Never overrides a working API reading |

By hand, if you are already on a running pod:

```bash
export HF_HOME=/workspace/hf M2_BRANCH=pareto M2_VOLUME_GB=150
```

**Why the deploy form is better than `export`, and not just tidier.** `m2.setup` writes `HF_HOME`
to `~/.bashrc`, but `~/.bashrc` is on container disk and does not survive a migration — and a
**non-interactive `ssh pod "command"` never sources it at all**, so an agent driving the pod over
SSH would silently run without these. Variables set at deploy time live in the container
environment and are inherited by every process, interactive or not.

---

## 3. Get the code

GitHub token with read access to this repo (github.com → Settings → Developer settings →
Personal access tokens → Fine-grained, read-only on `steering-optimization`).

```bash
GH=YOUR_GITHUB_TOKEN
git clone https://x-access-token:$GH@github.com/PquePC/steering-optimization.git /workspace/steering-optimization
git -C /workspace/steering-optimization remote set-url origin https://github.com/PquePC/steering-optimization.git
unset GH
```

The `remote set-url` afterwards keeps the token out of `.git/config`.

Then get on the branch the pipeline runs on. A clone lands on `main`, which is not it:

```bash
cd "/workspace/steering-optimization" && git checkout pareto && git log --oneline -1
```

---

## 4. Set everything up

```bash
cd "/workspace/steering-optimization"
python -m m2.setup --repair
```

**This is the install step.** It clones the upstream harness from
`safety-research/introspection-mechanisms`, installs every dependency, pulls any code updates,
and runs the offline tests. It imports nothing heavy, so it works when the missing thing *is*
the dependencies. Missing Python packages install on every setup invocation; `--repair` is present
here because a fresh pod also needs the harness and the other reversible setup actions.

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
| `FIX` | packages install automatically; `--repair` handles the other reversible items |
| `BLOCK` | only you can — credentials, GPU, the volume itself |

`--repair` never deletes a measured row.

**Routine setup/check:** `python -m m2.setup`. It changes only a missing Python environment unless
you add `--repair`. The same behavior is available as `python -m m2.run --setup`.

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
| `RUNPOD_API_KEY` | RunPod → Settings → API Keys. Used by the watchdog's auto-stop and by setup's volume-allocation read | no |

**There is no `OPENAI_API_KEY` and there must not be one.** Gate 11 scores transcripts with the
upstream repo's `LLMJudge`, which is built on the OpenAI SDK and accepts a key but no base URL —
so it used to demand an OpenAI credential. `gates._construct_repo_judge` now passes
`OPENROUTER_API_KEY` explicitly and scopes `OPENAI_BASE_URL` to OpenRouter across both
construction and evaluation, because that judge builds a fresh client inside every batch. One
provider, one key. It also makes gate 11 a cleaner comparison: both judges reach the same model,
so the only variable left is the rubric, which is what the gate is actually asking about.

The credentials can also go in the deploy form alongside §2's variables — same reasoning, and it
means they are never re-pasted into a shell. Leave `RUNPOD_API_KEY` out of the pod if an agent
will be driving it; nothing in the pipeline needs it there.

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
  cd "/workspace/steering-optimization"
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

The first thing it prints is gate 11's constructor check. It **exits** on failure rather than
warning, because the earlier version passed here and then skipped gate 11 thirty-eight minutes
into a run:

```
repo judge (Gate 11): PASS - OpenRouter route constructed without making a judge call
```

Then:

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
nohup bash "/workspace/steering-optimization/pod_watchdog.sh" > /workspace/watchdog.log 2>&1 &
```

Then, back in the first:

```bash
cd "/workspace/steering-optimization"
nohup python -m m2.run --concepts Garlic --no-stop-pod > /workspace/m2_garlic.out 2>&1 &
tail -f /workspace/m2_garlic.out
```

**`--no-stop-pod` matters.** Without it the pod stops the moment the concept finishes and you
cannot read the result or start the next one without a restart.

### Where the ~1h20m goes

| Phase | Basis | Time |
|---|---|---|
| CAL | measured | 1.5 min |
| SCAN, 147 cells | measured 10.9 s/cell | 26.8 min |
| knee search, 42 cells (task 24) | same rate | 7.6 min |
| measured-`d2` selection, <=30 cells (task 26) | measured ~13 s/cell | ~6.5 min |
| BISECT, 8 candidates | measured 68.7 s | 9.2 min |
| VERIFY / REFINE / CONFIRM / CONTROLS | **priors — never observed to completion** | ~18 min |

### What to read first when it lands

- **Phase 2's frontier table.** It ranks on **measured `d2`**, not on `d3` — task 25 showed `d3`
  reads the model's preamble rather than its answer, and inverted the ranking. If every eligible
  cell reads `d2` = 1.000 there is no covert cell at these doses, and reporting that is the
  correct outcome, not a failure.
- **The eligibility tier beside it.** Tier 0 means three prompts in twelve showed influence;
  tier 1 means it relaxed to two. An operating point found at tier 1 is a different claim.
- **Gate 5 should FAIL.** It correlates `d3` against measured `d2`, and those anti-correlate. A
  pass would be more surprising than a failure.

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
| `No module named pytest` / `torch` / `m2` | dependencies gone (container disk) or wrong directory | `cd` to the project dir, then `python -m m2.setup`; missing packages install automatically |
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
git clone https://x-access-token:$GH@github.com/PquePC/steering-optimization.git /workspace/steering-optimization
git -C /workspace/steering-optimization remote set-url origin https://github.com/PquePC/steering-optimization.git
unset GH

# 4. set up / check everything
cd "/workspace/steering-optimization"
python -m m2.setup --repair

# 5. credentials  (leading spaces)
 export HF_TOKEN=... OPENROUTER_API_KEY=...
 export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... HEALTHCHECK_URL=... RUNPOD_API_KEY=...

# 6. confirm green
python -m m2.setup

# 7. preflight  (expect 0 FAIL and summary PASS; R5/R8/R14 skip until they have data)
python -m m2.run --concepts Garlic --preflight

# 8. watchdog, SECOND terminal
nohup bash "/workspace/steering-optimization/pod_watchdog.sh" > /workspace/watchdog.log 2>&1 &

# 9. run
nohup python -m m2.run --concepts Garlic --no-stop-pod > /workspace/m2_garlic.out 2>&1 &
tail -f /workspace/m2_garlic.out

# 10. read
cat /workspace/m2_runs/garlic_*/operating_point.json | python -m json.tool
```

**Returning to a migrated pod:** steps 2, 4, 5, then 9.

---

# Beyond the quickstart

### 4b. The notebook

**Kernel → Restart & Run All.** In order: control panel → Setup 1–6 → RUN ALL.

Setup 5 runs the rig checks, and none of them is decoration:

| Check | What it catches |
|---|---|
| **R5** vector norm at the reference layer | broken extraction. ±2σ of Macar's 4664 ± 982, at the **reference layer only** — bug 19 declared a working rig broken by applying the band at every depth |
| **R7** forced-ID prompt equivalence | a batched D2 path that drifted from what the repo would send. Calls the repo's own function with the generator swapped for a recorder |
| **R14** hook liveness, both paths | bug 26 — the repo's hook silently declines to steer whenever `start_pos` is set, and measured an unsteered model at all 30 cells of a real v1 run |

**If R14 fails, stop.** Every forward-pass measure would read exactly zero, and a mean of zero
with a variance of zero passes most checks designed to catch a weak effect.

Per concept, RUN ALL executes:

| Phase | What | Judge calls | Wall |
|---|---|---:|---|
| 0 · calibrate | vectors, `‖v_L‖`, `‖h_L‖`, dose map, unsteered baselines, `cap_base` | 2 | ~2 min |
| 1 · full-depth scan | every layer ≥ `D_MIN` × 2 doses: E6, D3, S3, S2 | **0** | ~7–10 min |
| 2 · shortlist | local maxima + stratified + residual | **0** | free |
| 3 · dose bisection | bracket then bisect the sanity boundary | **0** | ~1 min |
| 4 · verify | E5 (E5), S1 (S1), D2+D4 (B) on ~10 cells | 490 | ~8–10 min |
| 5 · refine | layer ±1, ±2 and one dose step either side | 490 | ~8 min |
| 6 · confirm | winner on **held-out** prompts at `N_CONFIRM`, no adaptive stopping | ~150 | ~4 min |
| controls | §9.1 random direction, §9.2 forced-ID capability | ~75 | ~4 min |

**~35 min and ~1,210 judge calls per concept.** Judge cost is well under a dollar; wall time is
the binding constraint.

**Phases 1–5 are screening — their numbers are not reportable. Only Phase 6 output is.**

---


## 6. What arrives on your phone

Every message carries the **whole board**, not a summary line — a message that prompts a
follow-up you cannot make is a bad message when the only way to look deeper is to open a laptop.

Notification points: run started (and whether the dead man's switch is armed), each phase
completed, **the Phase 0 judge-FPR gate result**, **the Phase 2 shortlist**, **the Phase 4
qualifying set**, any phase failure, verdict changes, **the operating point**, run finished with
`operating_point.json` attached, and a slow beat every 600 s.

Four verdicts, which say what to **do**:

| Banner | Meaning |
|---|---|
| RECOVERED — nothing needs you | — |
| RUNNING SLOW — no action needed | slower than expected; the ETA already accounts for it |
| NEEDS YOU WHEN IT FINISHES | something died; the rest is still worth having |
| **STOP THE POD** | it will not recover; stop paying |

Three M2-specific triggers for *needs you*: the **judge FPR breaching `JUDGE_FPR_MAX`** (fires as
soon as Phase 0 completes, before GPU time is spent on a run whose numbers cannot be trusted);
an **empty qualifying set after Phase 4** (not broken — the frontier and the escalation ladder
are still the answer to "does an operating point exist for this concept"); and **both controls
rejecting the winner** (that verdict *is* the finding).

**What never leaves the pod in an alert:** exception messages and tracebacks. An API error can
quote its request payload back at you, and under M2 that payload is a steered generation or a
judge prompt containing one — arriving unbidden, in a message you did not choose to send.
Exceptions are mapped to classified labels; anything unmatched degrades to the class name alone,
which is safe by construction.

**Push only.** The pod talks; it never listens. There is no command channel, and none should be
added.

---


## 8. Failures, resuming, and the fatal abort

A phase that dies does not take the concept with it; a concept that dies does not take the batch
with it. `FATAL_CONSECUTIVE_D4S` compromised concepts in a row aborts the batch — that is
structural (OOM, judge auth, a bad install), not bad luck.

**Just re-run to resume.** The run folder is deterministic in `(concept, config)`, and:

- Phase 1 skips `(L, r)` already in `scan.jsonl`
- Phase 3 skips candidates in `bisect.jsonl`
- Phases 4–5 skip cells in `verified.jsonl`
- Phase 6 skips if `confirm.jsonl` is complete
- a concept whose archive already exists is skipped entirely

**No judge call is ever paid for twice.** The cache key is
`(phase, layer, r, prompt_id, judge_id, vec_fingerprint)`. Both trailing fields are load-bearing:
without `vec_fingerprint`, switching concepts in a live kernel returns the previous concept's
result (v1 bug 23); without `judge_id`, S1 returns E5's row and every S1 becomes a silent copy
of E5/10.

---

## 9. ⚠️ Before running harmful concepts

Everything M2 currently runs on is benign, and a transcript of a model steered toward *silk*
carries no dual-use risk. **The harmful arm is a different object** — those transcripts are what
a refusal-ablated model said with `weapon`, `poison` or `assault` injected, the exact artifact
`CLAUDE.md` hard rule 3 names.

The pipeline enforces this rather than trusting it: `transcripts_allowed()` refuses to put
transcripts in a bundle for any concept not on `m2.config.BENIGN_CONCEPTS` unless
`EXPORT_TRANSCRIPTS_OVERRIDE=True` is passed **at the call site**. It is deliberately not a
config key, an environment variable or a module global — all three are inheritable, and the batch
driver would carry a `True` set for the benign arm straight into the harmful arm when the concept
list changed.

Two facts worth having explicit before that point:

- **Telegram cloud chats are not end-to-end encrypted.** Messages and documents sit on Telegram's
  servers, and the bot token is the access credential.
- `CLAUDE.md` requires explicit approval before *any* upload to an external host. The transcript
  export policy is approved **for benign concepts only**.

Vectors, activations and weights are excluded from every bundle unconditionally. They regenerate
from a published config in minutes — **regeneration is the backup.**

Also note **acceptance gate 10**: M2 tunes on benign concepts, and Macar's gate analysis is
entirely benign (`cos(d_detect, d_refusal) = −0.09`). Validate the optimum on an arm-3 concept
before committing the harmful arm. If it moves, fit per arm — and the per-arm difference becomes
a result.

---

## 10. When a number looks wrong

Read `DEBUG-LOG.md` §6 first. Ten patterns, each of which produced more than one bug. The two
that cost the most time:

> **Read the code, not a summary of the paper.** Bugs 7, 19 and 25 all came from trusting a
> description over the source — including a function whose name and docstring both said "batch"
> over a body that was a `for` loop.

> **"The cell ran without error" is not evidence the cell did anything.**

`CONTRACT.md` §6 maps all 20 structural defences to the bug each one exists to prevent. If a
number is wrong, the fastest route is usually to find which defence should have caught it and
check whether it is still wired in — `python -m pytest m2/tests/test_offline.py -q` checks 46 of
those invariants in under a second.
