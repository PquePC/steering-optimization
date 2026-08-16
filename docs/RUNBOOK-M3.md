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

**Already on a running pod?** Set them by hand, in *every* shell, before anything else. The
leading space keeps the credentials out of `~/.bash_history`:

```bash
export HF_HOME=/workspace/hf M2_BRANCH=m3 M2_VOLUME_GB=150
```

```bash
 export HF_TOKEN=hf_... OPENROUTER_API_KEY=sk-or-v1-...
```

**The two that actually bite:**

**`HF_HOME`** is the expensive one. Without it the 54 GB model downloads to *container* disk
instead of the volume, and evaporates the next time you stop the pod — so you pay to download it
again on every restart.

**`M2_BRANCH=m3`** is the one that stops setup dead. It defaults to `main`, so on the `m3` branch
setup reports:

```
[BLOCK ] project repo    on branch 'm3', expected 'main'
```

That is **not** a problem with the branch — `m3` is the branch you want. It is setup refusing to
guess which line of work you meant, because which code runs is your decision and provenance
carries no git sha. Export `M2_BRANCH=m3` and re-run; it reads `ok`. It is never switched for you.

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
fresh pod; expect **no `BLOCK`** lines by the end. If credentials show `BLOCK`, they did not reach
the container — check the deploy form rather than exporting them by hand.

If you did set them by hand, prefix each line with a space so it stays out of `~/.bash_history`:

```bash
 export HF_TOKEN=hf_... OPENROUTER_API_KEY=sk-or-v1-...
```

---

## 4. Price it before you spend anything

```bash
cd /workspace/steering-optimization && python -m m3.run --concept Garlic --dry-run
```

Loads no model and spends nothing. You should see 49 layers, 196 cells, `<= $2.43` in judges, and:

```
boundary      12 probes descends 2.5 -> below the floor 0.05 at x0.7
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

Roughly 70 minutes of measurement plus the model download.

---

## 7. Get the results off the pod

The run writes to `/workspace/m3_runs/`, outside the repository, and builds a bundle at the end:

```bash
ls -la /workspace/m3_runs/export_garlic_*.zip
```

```bash
runpodctl send /workspace/m3_runs/export_garlic_*.zip
```

That prints a one-time code; run the matching `runpodctl receive <code>` on the machine you want
it on. The bundle carries every transcript, for any concept.

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
