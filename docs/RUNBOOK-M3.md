# Runbook — M3, bare pod to a measured surface

`docs/RUNBOOK.md` is the **M2** runbook. It sends you to the `pareto` branch, asks for a GitHub
token, and sets up a Telegram notifier and a dead-man's switch. **None of that applies to M3.**
M3 is terminal-only: no notifier, no watchdog, no auto-stop. Use this page instead.

---

## What you need to bring

This is the whole list. Nothing on your own machine is required — no files, no keys, no state.

| | Where to get it | Needed because |
|---|---|---|
| **A RunPod account** | you have one | rents the GPU |
| **`HF_TOKEN`** | huggingface.co → Settings → Access Tokens (read scope) | Gemma3-27B is gated |
| **Accept the Gemma licence** | [the model page](https://huggingface.co/google/gemma-3-27b-it), once, on that HF account | without it the download 403s even with a valid token |
| **`OPENROUTER_API_KEY`** | openrouter.ai → Keys, ~$5 of credit | every measurement in M3 is judged |

**No GitHub credential.** Both repositories clone anonymously over HTTPS — this one is public,
and the upstream harness (`safety-research/introspection-mechanisms`) always was. The M2 runbook's
fine-grained token step is left over from when this repo was private and is no longer needed.

**Nothing else carries over.** The concept vectors and the residual norms are regenerated in
Phase 0 on every run, by design — regeneration is the backup. There is no state on your current
machine that a pod needs.

---

## 1. Rent the pod

RunPod → **Deploy** → **Pods**.

| Setting | Value | Why |
|---|---|---|
| **GPU** | **1× A100 80GB** or **H100 80GB** | Gemma3-27B in bf16 is ~54 GB of weights alone. A 48 GB card cannot hold it, and splitting across two is not supported — the injection hook assumes one device |
| **Template** | any recent **PyTorch** template | ships CUDA + torch |
| **Volume Disk** | **150 GB** (100 GB minimum) | 54 GB model + ~10 GB deps + run outputs |
| **Volume Mount Path** | `/workspace` | everything below assumes it |
| **Container Disk** | **40 GB** | pip wheels and CUDA libs |

In the deploy form's **Environment Variables** section, set these — not by hand afterwards. A
non-interactive `ssh pod "command"` never sources `~/.bashrc`, so anything exported by hand is
invisible to it.

| Variable | Value | If you skip it |
|---|---|---|
| `HF_HOME` | `/workspace/hf` | the 54 GB model lands on container disk and is lost on every stop |
| `M2_BRANCH` | `m3` | setup reports `BLOCK ... expected 'main'` and refuses to proceed |
| `M2_VOLUME_GB` | `150` | nothing, unless RunPod's allocation API is unreadable |
| `HF_TOKEN` | `hf_...` | the model will not download |
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | nothing can be scored |

`M3_RUNS_DIR` is **not** in the list: it defaults to `/workspace/m3_runs`, which is already
outside the repository, and setting it is only for moving results somewhere else deliberately.

### 1.1 The first command on any pod — make the variables persistent

Whether you set them in the deploy form or not, run this before anything else. It writes them to
the **volume**, where they survive a stop/start, and arms every future shell to load them. A
variable exported by hand into one terminal is invisible to the next terminal, to `nohup`, and to
the pod after a restart — which is how a run dies at hour two, or how `m2.setup` blocks on a
branch you already set.

`unset HISTFILE` first, so the credentials never reach `~/.bash_history` at all. That is stronger
than the leading-space trick, which silently does nothing unless `HISTCONTROL` includes
`ignorespace`.

```bash
unset HISTFILE
```

Paste this, **replace the two placeholder tokens before pressing enter**:

```bash
cat > /workspace/env.sh <<'EOF'
export HF_HOME=/workspace/hf
export M2_BRANCH=m3
export M2_VOLUME_GB=150
export HF_TOKEN=hf_PASTE_YOURS
export OPENROUTER_API_KEY=sk-or-v1-PASTE_YOURS
EOF
chmod 600 /workspace/env.sh
grep -q 'workspace/env.sh' ~/.bashrc || echo '[ -f /workspace/env.sh ] && . /workspace/env.sh' >> ~/.bashrc
. /workspace/env.sh
```

Check all five arrived, without printing the secrets:

```bash
for v in HF_HOME M2_BRANCH M2_VOLUME_GB HF_TOKEN OPENROUTER_API_KEY; do eval "val=\$$v"; if [ -n "$val" ]; then echo "$v ok"; else echo "$v MISSING"; fi; done
```

Five `ok` lines and you are clear to install. Any `MISSING` means the file did not load — fix it
here, because every failure below traces back to this.

**After a pod stop/start.** `/workspace/env.sh` survives, because the volume does. `~/.bashrc`
usually does **not**, because `/root` is container disk and is rebuilt. So on a restarted pod the
file is still there but nothing loads it: re-run the block above (it is idempotent — the `grep -q`
guard keeps `~/.bashrc` from collecting duplicates), or at minimum `. /workspace/env.sh` in each
new shell.

**The two that actually bite:**

**`HF_HOME`** is the expensive one. Without it the 54 GB model downloads to *container* disk
instead of the volume, and evaporates the next time you stop the pod — so you pay to download it
again on every restart.

**`M2_BRANCH=m3`** is the one that stops setup dead. It defaults to `main`, so on the `m3` branch
setup reports:

```
[BLOCK ] project repo    on branch 'm3', expected 'main'
```

That is **not** a problem with the branch — `m3` is the branch you want, and the message says so
itself: *"if 'm3' is the one you want, export `M2_BRANCH=m3` and re-check"*. It is setup refusing
to guess which line of work you meant, because which code runs is your decision and provenance
carries no git sha. It is never switched for you.

Seeing this block means §1.1 was skipped or its file is not loaded in **this** shell. Do not
export the variable inline to get past it — the next shell will block again, and so will the run
you launch from it. Run §1.1, then re-run setup; it reads `ok`.

`M2_VOLUME_GB` is inert unless RunPod's allocation API cannot be read, in which case it prevents
a stall. It never overrides a working reading.

Deploy → wait for **Running** → **Connect → Jupyter Lab** → **File → New → Terminal**.

> **Stop, never Terminate.** Stop keeps the volume; terminate destroys it. A *network* volume
> survives either, and is worth it if you expect to move hosts.

---

## 2. Get the code

```bash
git clone https://github.com/PquePC/steering-optimization.git /workspace/steering-optimization
cd /workspace/steering-optimization && git checkout m3 && git log --oneline -1
```

---

## 3. Install

```bash
cd /workspace/steering-optimization && python -m m2.setup --repair
```

Still `m2.setup` — it is the installer for the repository, not for a pipeline. It clones the
upstream harness, installs every dependency, and runs the offline tests. Expect `FIX` lines on a
fresh pod; expect **no `BLOCK`** lines by the end.

Every credential or branch `BLOCK` here means the same thing: this shell has not loaded
`/workspace/env.sh`. Go back to §1.1 rather than exporting anything inline — an inline export
fixes the one command in front of you and leaves the run to fail later.

---

## 4. Price it before you spend anything

```bash
cd /workspace/steering-optimization && python -m m3.run --concept Garlic --dry-run
```

Loads no model and spends nothing. You should see 49 layers, 196 cells, `<= $3.60` in judges, and:

```
boundary      12 probes descends 2.5 -> below the floor 0.05 at x0.7
              then <= 3 bisection probes inside the ladder's bracket, stopping within 10% of the boundary
```

If that line instead says **WARNING**, the boundary ladder cannot reach its own bracket floor and
layers it stopped short on will be recorded as `probes_exhausted` rather than as broken. It tells
you the number to set. At the shipped defaults it does not warn.

---

## 5. Run

```bash
cd /workspace/steering-optimization && nohup python -m m3.run --concept Garlic > /workspace/m3.out 2>&1 &
```

```bash
tail -f /workspace/m3.out
```

First run also downloads the model: ~54 GB, 15–25 minutes, before anything is printed about layers.

**Phase 0 is the preflight.** There is no separate `--preflight` in M3 — Phase 0 loads the model,
extracts the vectors, measures the residual norms and runs the R14 hook-liveness check. If it
prints `R14 pass` with two non-zero magnitudes, the injection hook is live and the run is real.
A dead hook returns a clean, plausible, completely empty surface, which is why this check exists.

Killed at any point, the run resumes: rerun the same command and cells already on disk are not
re-measured.

---

## 6. What to look for

| Phase | Healthy | Wrong |
|---|---|---|
| 0 | `R14 pass  start_pos 3.59e+01  all-pos 3.56e+01` | either magnitude ~0 — the hook is dead, stop |
| 1 | `dose_max` **varies** layer to layer, most outcomes `ok` | every layer the same number — the search resolved nothing per-layer, and it says so |
| 1 | — | a `WARNING` about probes exhausted above the floor |
| 2 | `ident`, `eff`, `coh`, `cap` all populated | `-` in a column means every judge call for it failed |
| 2 | `[N judge errors]` absent | present on many cells — check the API key and credit |
| end | `read-this bundle: N of M disagreements + 15 null` | `0 of 0` — be suspicious, check the judges ran |

Roughly 80 minutes of measurement plus the model download (the estimate line prints the exact
worst case; the boundary bisection usually stops early, so real runs come in under it).

---

## 7. Get the results off the pod

### 7.1 What the run saves

Everything, by design: every generation, every judge reply verbatim, and every mechanical verdict
beside them. A debug pass done weeks later should never need the pod back.

| File | One row per | Carries |
|---|---|---|
| `cells.jsonl` | cell | every aggregate: the four judged rates/means with Wilson intervals, both `identification` views, the mechanical measures |
| `responses_transcripts.jsonl` | battery response | the generation, its channel and unit, the parsed verdicts, degeneracy and its reason |
| `judge_calls.jsonl` | judge call | the **payload sent** and the **raw reply**, plus attempts, latency, cache state, parse error |
| `boundaries.jsonl` | layer | `dose_max`, `lowest_failing_dose`, the outcome, and each probe's aggregate |
| `boundary_transcripts.jsonl` | Phase 1 probe response | the generation and the judge's verbatim reply, with the three legs — `coherence`, `on_task`, `answered` — recorded **separately** |
| `null_transcripts.jsonl` | null-arm response | the α=0 control battery, `NULL_REPEATS` times |
| `unjudged_transcripts.jsonl` | orphaned response | only if judging raised: generations already paid for, kept rather than discarded |
| `norms.jsonl` | layer | `‖v‖`, `‖h‖` and the dose map every alpha comes from |
| `summary.json` | run | the **complete config** (all 27 settings), the layer list, hook liveness, timings, skipped layers |
| `provenance.jsonl` | run | git commit, branch and dirty flag, GPU, host, library versions |
| `read_this.md` | — | the disagreement bundle, for reading by eye |
| `lab.log` | — | the run's own log lines |

`boundary_transcripts.jsonl` is new as of `2026-08-19`. Before it, Phase 1 kept only per-probe
averages: the 2026-08-15 run discarded **840 generations and 840 paid judge calls**, 22% of all
the model output it produced, and `dose_max` — the number every cell's dose is a fraction of —
was the one figure in the run with no readable evidence behind it. If a layer's boundary looks
wrong, that file is where you find out which of the three legs rejected the dose and what the
model actually said.

### 7.2 Take the console log with you

The board printed to stdout is **not** in the run folder, and this project has already lost
numbers that existed only in a terminal. Copy it in before bundling:

```bash
cp /workspace/m3.out /workspace/m3_runs/garlic_*/console.log
```

The export bundle is built *before* you do that, so it will not contain the log. Send the **whole
run folder** instead — it needs no rebuild and picks up anything you copied in:

```bash
cd /workspace/m3_runs && tar czf garlic_full.tgz garlic_*/ && ls -lh garlic_full.tgz
```

### 7.3 Send it

```bash
runpodctl send /workspace/m3_runs/garlic_full.tgz
```

That prints a one-time code; run the matching `runpodctl receive <code>` on the machine you want
it on. The run also builds an export bundle at the end, which is the gated artefact — it withholds
transcripts for a concept that is not benign, and includes everything for one that is:

```bash
ls -la /workspace/m3_runs/export_garlic_*.zip
```

**Do this the moment the run finishes**, before any pod housekeeping. Task 28 (portable resume)
is unbuilt and a RunPod volume has already eaten a run's output once.

Worth reading first, on the pod:

```bash
cat /workspace/m3_runs/garlic_*/read_this.md
```

That file is selected by **disagreement**, not at random: judge-versus-detector contradictions in
both directions, responses judged influential that never name the concept, the covert `leaked`
class, failed judge calls, and the whole α=0 null arm. Every deep defect this project has found was
found by a person reading raw generations, and none by a gate, a rate or a judge.

---

## 8. Spot-checking a cell by hand — `freerun`

Once a run has produced a results table, this is how you see the model behave at one coordinate
from it, with your own prompts:

```bash
python -m m3.freerun --concept Garlic --layer 29 --dose 0.114438
```

`--dose` is copy-pasteable straight out of `cells.jsonl` or a `FINDINGS-*.md`. It loads the model,
re-extracts the vector, converts the dose to an alpha through the same `alpha_for` the sweep uses,
runs the full battery at that cell, prints every statistic **measured in that process**, and then
asks you for prompts — answering each one `n` times unsteered and `n` times steered, side by side.

Nothing is read from the previous run's numbers, deliberately: re-measuring makes this an
independent check on the sweep rather than a viewer for it.

Useful flags: `--n 5` (responses per arm), `--alpha 6.1` (instead of a dose), `--no-battery`
(skip the opening battery), `--prompt "..."` (ask and exit, repeatable, no interactive loop),
`--no-judge` (mechanical measures only). It refuses an unreachable dose rather than clamping it,
and refuses the harmful arm.

---

## 9. Stopping

```bash
runpodctl stop pod $RUNPOD_POD_ID
```

Or the RunPod console → **Stop**. Nothing stops the pod for you: M3 has no watchdog and no
auto-stop, deliberately. **An idle A100 bills at the same rate as a busy one.**

---

## Harmful concepts

Not from this runbook, and not from this pipeline as it stands. `config.HARMFUL_CONCEPTS` names
them and `m3` refuses them by name at every entry point. That is the only concept filter there
is — any other concept runs, and exports in full. Read the parent repository's ethics register
before changing that; see `CLAUDE.md`.
