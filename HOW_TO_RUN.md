# Measurement Lab — how to run it (all features)

A run sweeps a grid of (layer × steering strength) cells for each concept and measures, per
cell: **Detection** (does the model notice the injected thought?), **Effectiveness** (how
strongly the concept is injected), and **Sanity** (is the model still coherent/capable?).
`BATCH_MODE` runs a whole list of concepts unattended, archives each, and can stop the pod
when done.

---

## 0. Get it onto the pod / keep it updated (`sync.sh`)
Instead of deleting and re-uploading the notebook, clone the repo once and use `sync.sh`.

**First time** (in a pod terminal) — HTTPS + a GitHub token (fine-grained, read-only on this
repo is enough):
```bash
GH=YOUR_TOKEN; git clone https://x-access-token:$GH@github.com/PquePC/steering-optimization.git \
  /workspace/steering-optimization \
  && git -C /workspace/steering-optimization remote set-url origin \
       https://github.com/PquePC/steering-optimization.git \
  && printf '%s' "$GH" > /workspace/.gh_token && chmod 600 /workspace/.gh_token
```
(or, if the pod has an SSH key: `git clone git@github.com:PquePC/steering-optimization.git /workspace/steering-optimization`)

**Before each run / after any code change:**
```bash
bash "/workspace/steering-optimization/sync.sh"
```
It force-matches GitHub (`fetch` + `reset --hard`, discarding the disposable cell outputs the
notebook writes into itself), fixes line endings / +x on the scripts, and prints what changed.
Then in Jupyter: **File → Reload Notebook from Disk** (or re-open it) and re-run. The token
lives only in `/workspace/.gh_token`, never in git config. Open the notebook from
`/workspace/steering-optimization/measurement_lab.ipynb`.

The notebook's Setup 2 still clones `introspection-mechanisms` and pip-installs deps on its
own (idempotently), so `sync.sh` only has to update this repo.

## 1. Configure — the CONTROL PANEL cell (everything you edit is here)
- **CONCEPTS** — the list to sweep, one per line. Setup + rig check run once; each concept
  gets its own vectors, sweep, probe and `runs/<concept>_<hash>` folder.
- **BATCH_MODE** — `True` = the RUN ALL CONCEPTS cell sweeps the whole list; `False` = old
  single-concept run of `CONCEPTS[0]`.
- **PROBE_QUESTIONS** — asked at the 3 best configs after each concept, steered vs unsteered.
  **PROBE_TOP_K** (how many configs), **PROBE_MAX_TOKENS** (answer length),
  **SEND_PROBE_TO_TELEGRAM** (raw generations — benign concepts only).
- **WIPE_AFTER_EACH** — delete the loose run folder after archiving (keeps the .zip).
- **TELEGRAM_WARNINGS_ONLY** — `False` = normal; `True` = only messages that need you.
- **KILL_POD_WHEN_DONE / KILL_POD_ON_FATAL / KILL_GRACE_SECONDS / FATAL_CONSECUTIVE_FAILS**
  — auto-stop (see §5).
- **AUTORUN / DEBUG_*** — `AUTORUN=True` is the normal "Run All" mode.
- **CONFIG** — the science grid. Commonly edited: `layer_fractions`, `ref_fraction`,
  `alphas`, `n_trials`. The rest are analysis-gate thresholds (leave unless you mean it).

After editing: re-run the **CONTROL PANEL**, then **Setup 4**, and — if you changed the
concept/grid — **Setup 7** and **Setup 8**. A plain **Run All** does all of this in order.

## 2. Credentials — Setup 1
Prompts (each optional-skippable with Enter, except HF/OpenRouter which the run needs):
- **HF token** — huggingface.co → Settings → Access Tokens (read).
- **OpenRouter API key** — openrouter.ai → Keys (the judge runs here).
- **Telegram bot token** — message @BotFather, `/newbot`, copy the token.
- **Telegram chat id** — message your new bot once, open
  `https://api.telegram.org/bot<TOKEN>/getUpdates`, read `result[0].message.chat.id`.
- **RunPod API key** — console → Settings → API Keys → +API Key, set
  **api.runpod.io/graphql = Read/Write**, Create, copy. Only needed for pod auto-stop.

After Setup 5, run `notify_test()` once in a fresh cell and confirm the Telegram message
arrives before walking away.

## 3. Run it
**Run All** (Kernel → Restart & Run All). In `BATCH_MODE` this executes, in order:
Control panel → Setup 1–8 (creds, install, model, config, helpers, primitives, vectors) →
**M0 rig check** (once — validates the steering rig reproduces the published detection rate)
→ measure definitions → **RUN ALL CONCEPTS** driver.

Per concept the driver does: `set_concept` → `prepare_concept` (vectors, baselines) →
`run_all_measures` (the sweep + aggregate results) → behaviour probe → archive the full
folder → wipe loose files → free VRAM → next concept.

Watch the **status board** (rewrites in place in the driver's output) and/or your phone.

## 4. Telegram — what you'll receive
**Normal (`TELEGRAM_WARNINGS_ONLY=False`):** per concept — a *start* message, a *"still
running"* check-in every ~10 min, a *finish* message with the aggregate results zip, and the
behaviour probe (if `SEND_PROBE_TO_TELEGRAM`). Plus one *batch finished* line at the end.
Per-measure pings were removed (they spammed).

**Quiet (`TELEGRAM_WARNINGS_ONLY=True`):** only things needing you — measure/concept
failures, structural aborts, and **stalls** (a stuck generation, caught when a measure
hasn't finished a cell in >6× its normal time). Mild slowdowns stay silent; a real hang
still alerts. Routine start/beat/finish/results/probe are suppressed.

## 5. Auto-stop the pod + the hang watchdog
**Auto-stop (opt-in, in the Control panel):** `KILL_POD_WHEN_DONE` stops the pod after a
clean batch; `KILL_POD_ON_FATAL` stops it if a fatal abort happens (see §7). Both wait
`KILL_GRACE_SECONDS` after the final message so exports/sends finish. It issues a **STOP,
not a terminate** — the volume and every zip survive, and you can restart the pod later with
the data intact. Needs the RunPod API key (§2) or an authenticated `runpodctl`.

**Kernel-hang watchdog** (for the Jupyter "kernel unresponsive" case, which the notebook
itself cannot catch): run in a **terminal**, not the notebook —
```bash
export RUNPOD_API_KEY=YOUR_KEY
nohup bash "$(find / -name pod_watchdog.sh 2>/dev/null | head -1)" > /workspace/watchdog.log 2>&1 &
```
Tail: `tail -f /workspace/watchdog.log`. Stop: `pkill -f pod_watchdog.sh`. It stops the pod
after ~20 min of near-0% GPU while the model is still loaded. Open a terminal via JupyterLab
(File → New → Terminal) or the RunPod console (Connect → Web Terminal).

## 6. Where the results are
On the pod, per concept: `/workspace/runs/lab_<concept>_<hash>.zip` — the **full** archive
including raw responses (transcripts) and vectors, for later analysis. Inside:
- `results_summary.csv` / `cell_summary.jsonl` — the joined per-cell Detection / Effectiveness / Sanity table.
- `measures/*.jsonl` — per-measure score rows; `measures/*_transcripts.jsonl` — raw generations.
- `probe/` — the behaviour-probe transcript.
- `config.json`, `norms.jsonl`, `rig_status.json`, logs.

Telegram gets only the **aggregate** results zip (no vectors/transcripts) plus the probe.

## 7. Failures and fatal abort
A single measure failing doesn't stop a concept (the rest still runs). A concept that raises,
has a dead steering vector (S14 skip), or loses ≥2 measures (structural — OOM / judge / bad
install) counts as "compromised." **`FATAL_CONSECUTIVE_FAILS` compromised concepts in a row**
(or any batch-level exception) aborts the whole batch — and stops the pod if
`KILL_POD_ON_FATAL`. Isolated blips that recover don't trip it.

## 8. Getting results off the pod
Because auto-stop is a STOP (not terminate), the zips persist on the volume — restart the pod
and they're still there. To pull them to your laptop: JupyterLab file browser → right-click
the `.zip` → Download, or `runpodctl send /workspace/runs/lab_<concept>_<hash>.zip`.

## 9. Resuming after an interruption
Just re-run. The run folder is `runs/<concept>_<hash>` (deterministic), and `sweep_measure`
skips (layer, alpha) cells already recorded — so a dropped kernel resumes where it stopped.
A different concept or a changed CONFIG gets its own folder and never overwrites an earlier run.

## 10. Single-concept / manual use
Set `BATCH_MODE=False` for a classic top-to-bottom run of `CONCEPTS[0]` (the RUN ALL cell
sweeps and exports as before; the X-cells run). Any time after setup you can call
`probe("your question", layer=46, alpha=3)` from a fresh cell to eyeball steered vs
unsteered output on demand.

## 11. ⚠️ Before running harmful concepts
The pipeline currently assumes benign concepts. Before adding weapon/poison/assault:
- set **`SEND_PROBE_TO_TELEGRAM = False`** (probe outputs are raw generations),
- review the per-concept pod archiving (it stores raw responses + vectors) and pull+wipe at
  session end — don't keep a long-lived volume of them.
See the note in `Pipeline v1 Implementation Plan.md` and `docs/risks-and-ethics.md`.
