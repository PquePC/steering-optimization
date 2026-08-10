# M2 — how to run it

The operating-point finder. Give it a concept (or a list), and for each one it finds the
`(layer, α)` at which the concept **visibly influences generated output**, the model **cannot
name it under forced identification**, and the model **is still working** — together with the
frontier of near-optimal alternatives, the two controls that rule out the known artifacts, and a
confirmation run at fixed N on held-out prompts.

Science lives in [`M2 — Specification.md`](M2%20—%20Specification.md); rationale and the decision
log in [`M2 — Pipeline Plan.md`](M2%20—%20Pipeline%20Plan.md). This document is how to operate it.

---

## 0. What is where

| Path | What it is |
|---|---|
| `m2/` | the pipeline. Plain `.py` modules — this is the codebase |
| `m2/CONTRACT.md` | module boundaries, exact names, file formats, and the 20 defences with the bug behind each |
| `m2/tests/test_offline.py` | 46 invariants that run with no GPU, no model and no judge key |
| `m2_pipeline.ipynb` | thin driver: control panel, setup, one Run All cell |
| `pod_watchdog.sh` | the out-of-process hang watchdog |
| `sync.sh` | pull this repo onto the pod |

**The notebook contains no logic.** That is deliberate: `DEBUG LOG.md` bug 24 was a notebook
whose cells silently collapsed to one physical line and did nothing while reporting success.

**Environment:** RunPod, 1× A100 80GB, `/workspace` persistent volume, Gemma3-27B bf16, judge
`openai/gpt-4.1-mini` via OpenRouter.

---

## 1. Get it onto the pod

First time, in a pod terminal:

```bash
GH=YOUR_TOKEN; git clone https://x-access-token:$GH@github.com/PquePC/Emergent-Introspection.git /workspace/Emergent-Introspection && git -C /workspace/Emergent-Introspection remote set-url origin https://github.com/PquePC/Emergent-Introspection.git && printf '%s' "$GH" > /workspace/.gh_token && chmod 600 /workspace/.gh_token
```

Before every run, and after any code change:

```bash
bash "/workspace/Emergent-Introspection/Steering Optimization/sync.sh"
```

Then in Jupyter: **File → Reload Notebook from Disk**. Open
`/workspace/Emergent-Introspection/Steering Optimization/m2_pipeline.ipynb`.

Unlike v1, a code change is now a `.py` edit that `sync.sh` pulls and the kernel re-imports —
the notebook itself rarely changes, and its cell outputs are no longer part of the source.

Run the offline tests once after any edit. They need nothing but pytest:

```bash
python -m pytest "Steering Optimization/m2/tests/test_offline.py" -q
```

---

## 2. Configure — the CONTROL PANEL cell

Everything you edit is in that one cell.

- **`CONCEPTS`** — one per line. Setup and rig checks run **once**; each concept then gets its
  own vectors, baselines, scan, verification, controls and `runs/<concept>_<hash>` folder.
- **`BATCH_MODE`** — `True` sweeps the whole list unattended; `False` runs `CONCEPTS[0]`.
- **`WIPE_AFTER_EACH`** — delete the loose run folder **after delivery is confirmed**, never
  before. See §7.
- **`EXPORT_TRANSCRIPTS_OVERRIDE`** — leave `False`. See §9.
- **`TELEGRAM_WARNINGS_ONLY`** — `True` for quiet.
- **`KILL_POD_WHEN_DONE` / `KILL_POD_ON_FATAL`** — auto-stop (STOP, not terminate).
- **`OVERRIDES`** — the science constants, commented out with their spec rationale. Everything
  not overridden comes from `m2.config.CONSTANTS`. The ones the spec marks load-bearing change
  what the pipeline *concludes*: `SCAN_DOSES`, `E5_FLOOR`, `D2_MAX`, `S4_MIN`, `D_MIN`.

**Changing any constant changes `config_hash`, which changes the run folder.** A rerun with a
new grid never overwrites an earlier run, and never silently resumes into it.

---

## 3. Credentials — Setup 1

Each is optional-skippable except HF and OpenRouter, which the run needs.

| Variable | Where from | If unset |
|---|---|---|
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens (read) | model will not load |
| `OPENROUTER_API_KEY` | openrouter.ai → Keys | judges cannot run |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | @BotFather `/newbot`; then message the bot and read `result[0].message.chat.id` from `getUpdates` | run is unattended-blind — watch `status.txt` |
| `HEALTHCHECK_URL` | healthchecks.io, **period 300 s** | a pod that dies outright will not report it; you would just stop hearing from it |
| `RUNPOD_API_KEY` | console → Settings → API Keys, `api.runpod.io/graphql = Read/Write` | stop the pod by hand |

Keys stay in this process. Nothing is written to disk or into the pod environment.

**Run Setup 6's `notify_test()` and wait for the message before walking away.** Transport errors
are swallowed on purpose — a broken alert channel must never break a run — so a channel that
silently does not work looks exactly like one with nothing to report.

---

## 4. Run it

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

## 5. The watchdog

Run in a **terminal**, not the notebook. A hung kernel holds the GIL, so the notebook's own
heartbeat thread stops too — detection has to live outside the process.

```bash
export RUNPOD_API_KEY=YOUR_KEY
nohup bash "$(find / -name pod_watchdog.sh 2>/dev/null | head -1)" > /workspace/watchdog.log 2>&1 &
```

Stops the pod after ~20 min of ≤5% GPU **while VRAM ≥ 10 GB** — the VRAM condition is what keeps
a long judge wait, during which the GPU is legitimately idle, from tripping it.

Tail with `tail -f /workspace/watchdog.log`; stop with `pkill -f pod_watchdog.sh`.

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

## 7. Where the results are

On the pod: `/workspace/m2_runs/<concept>_<hash>/`, archived to a bundle per concept.

| File | Contents |
|---|---|
| `operating_point.json` | **the answer** — `(L, α, r)`, every metric with SE, control verdicts, the frontier |
| `scan.jsonl` | one row per `(L, r)` from Phase 1 |
| `shortlist.json` | Phase 2 candidates, with the reason each was selected |
| `bisect.jsonl` | Phase 3 bracket, steps, chosen dose |
| `verified.jsonl` | Phases 4–5 — E5, S1, D2, D4, sanity, residual, covertness margin |
| `confirm.jsonl` | Phase 6 at `N_CONFIRM` on held-out prompts |
| `controls.jsonl` | §9.1, §9.2, §9.3 |
| `judge_e5/a2/b.jsonl` | every judge call: prompt, raw response, parsed fields |
| `D2_transcripts.jsonl` | every forced-ID generation |
| `cis_transcripts.jsonl`, `unsteered/*.jsonl` | every steered and baseline generation |
| `mmlu_items.json`, `norms.jsonl`, `dose_map.json` | the pinned S3 set and the calibration |

**The delivery order is archive → send → verify the send succeeded → only then wipe.** v1
archived, wiped, then attempted the send; a Telegram outage made two concepts' results
unrecoverable. On failure the loose folder is kept, the concept is marked `undelivered` in a
manifest, and delivery is retried at the end of the batch. A delivery failure must never destroy
data.

Auto-stop is a **STOP, not a terminate** — the volume and every bundle survive a restart.

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

Read `DEBUG LOG.md` §6 first. Ten patterns, each of which produced more than one bug. The two
that cost the most time:

> **Read the code, not a summary of the paper.** Bugs 7, 19 and 25 all came from trusting a
> description over the source — including a function whose name and docstring both said "batch"
> over a body that was a `for` loop.

> **"The cell ran without error" is not evidence the cell did anything.**

`m2/CONTRACT.md` §6 maps all 20 structural defences to the bug each one exists to prevent. If a
number is wrong, the fastest route is usually to find which defence should have caught it and
check whether it is still wired in — `python -m pytest m2/tests/test_offline.py -q` checks 46 of
those invariants in under a second.
