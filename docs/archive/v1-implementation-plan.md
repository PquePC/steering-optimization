# Pipeline v1 — Implementation Plan

**Scope:** Milestone M1 of *Project A Final Draft*. Nothing beyond M1 is built here.
**Working folder:** `Z:\Projects\TAIS Projects\steering-optimization\`
**Upstream:** `introspection-mechanisms/` (fresh clone of `github.com/safety-research/introspection-mechanisms`)
**Runtime:** Jupyter notebook on a RunPod pod.

---

## 1. What v1 is and is not

**In scope:**

| Item | Measure | Source |
|---|---|---|
| Rig check against Macar's published detection rate | D1 | `01_concept_injection.py` |
| Coarse grid sweep, 5α × 6L (matched extraction, Macar-s method) | D1 | repo functions |
| Concept-word log-probability shift, steered minus unsteered | E1 | **new code** |
| Perplexity under injection | E2 | **new code** |
| Interactive steered-vs-unsteered probe | — | new code (R2) |
| Shared unsteered control block for a measured false-alarm rate | D1 | repo functions |
| Dynamic strength escalation for the per-concept anchor | D1 | new orchestration |
| One effectiveness-versus-detection frontier plot | — | new code |
| Sanity measures S1–S13 | S | mixed |

**Explicitly deferred:**

- D1b (Yes/No logit + control arm) → M2
- D2 / E5 forced identification → M2 (one function call; `run_forced_noticing_test_batch` already exists)
- E3 transcript re-judging → M3
- E4 KL, D3 transcoders, refusal ablation, multi-layer, auto-tuner → M4–M7
- Trained bias vector → post-M7

> **Why so narrow:** every later measure depends on a working extract → inject → judge → score loop. v1 is that loop plus the two cheapest effectiveness measures, and nothing else.

---

## 2. Environment

**Pod spec:** 1× **A100 80GB** or H100 80GB. Gemma3-27B in bf16 is ~54GB of weights; activations and KV cache fit comfortably in 80GB. The repo README states ≥48GB VRAM.

> **The 80GB SKU is mandatory.** A 40GB A100 cannot hold the weights and there is no configuration that rescues it short of quantisation, which would change the intervention being measured.

> **A100 vs H100 for v1:** H100 is roughly 1.5–2.5× faster on bf16 decode but typically ~2× the hourly rate, so cost per experiment is close to a wash. Nothing in v1 uses FP8 or other Hopper-only paths — bf16 and FlashAttention-2 both run natively on Ampere. A100 80GB is the correct choice; it also tends to have better availability. Cell 5 measures actual throughput, so the budget self-corrects either way.

**Storage:** RunPod container disk is ephemeral; `/workspace` is the persistent volume. The model download is ~54GB and must not be repeated.

**Credentials — never as pod environment variables.** RunPod env vars are not encrypted. The notebook's first cell prompts via `getpass`, sets the values in the Python process only, and `clear_credentials()` drops them at the end of the run. Nothing is written to disk.

Non-secret settings are defaulted in that same cell:

```python
os.environ.setdefault("HF_HOME", "/workspace/hf")            # keeps the 54GB download persistent
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
```

`google/gemma-3-27b-it` is gated — accept the licence on HuggingFace with the same account as the token before the first run.

**Known install friction:** `requirements.txt` pins `numpy<2.0`. Install requirements *before* anything that pulls numpy 2.x, and verify with `python -c "import numpy; print(numpy.__version__)"`.

**Model identifier:** `gemma3_27b` → `google/gemma-3-27b-it` (`src/model_utils.py:45`). Layer count 62; `get_layer_at_fraction(model, f)` converts a depth fraction to an index — use it rather than hardcoding indices, so the grid ports to another model unchanged.

---

## 3. OpenRouter patch

`src/eval_utils.py` constructs OpenAI clients with no `base_url` in three places. Judge model stays `gpt-4.1-mini` (repo default, `eval_utils.py:422`) — do not substitute.

**Line 447–448:**
```python
# before
self.client = openai.OpenAI(api_key=self.api_key)
self.async_client = openai.AsyncOpenAI(api_key=self.api_key)

# after
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
self.client = openai.OpenAI(api_key=self.api_key, base_url=_BASE_URL)
self.async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=_BASE_URL)
```

**Line ~560** (a second `AsyncOpenAI` is constructed inside a helper) — apply the same `base_url`.

**Line 443:** `api_key` reads `OPENAI_API_KEY`. Either set `OPENAI_API_KEY` to the OpenRouter key, or change the fallback to `os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")`.

**Model string:** OpenRouter requires the vendor prefix — pass `model="openai/gpt-4.1-mini"` when constructing `LLMJudge`.

**Concurrency:** `LLMJudge` defaults to `max_concurrent=1000`. OpenRouter will rate-limit well below that. Set `max_concurrent=32` for v1 and raise only if no 429s appear.

> **Apply as a patch file, not by hand-editing the clone.** Keep `openrouter.patch` in the working folder so the clone stays reproducible and the change is visible in review.

---

## 4. Progress, logging and ETA

Every long-running cell uses one shared helper. Requirements: periodic ETA, survives SSH reconnects, and leaves a debuggable trace on disk.

```python
import time, sys, json
from pathlib import Path

RUN_DIR = Path("/workspace/runs/v1")
RUN_DIR.mkdir(parents=True, exist_ok=True)

class Progress:
    """Periodic ETA reporter. Prints flushed lines and mirrors to a log file."""
    def __init__(self, total, label, report_every_s=30, log=RUN_DIR / "run.log"):
        self.total, self.label = total, label
        self.report_every_s = report_every_s
        self.done = 0
        self.t0 = time.time()
        self.last_report = 0.0
        self.log = open(log, "a", buffering=1)
        self._emit("start", force=True)

    def update(self, n=1, **info):
        self.done += n
        now = time.time()
        if now - self.last_report >= self.report_every_s or self.done >= self.total:
            self.last_report = now
            self._emit("tick", **info)

    def _emit(self, kind, force=False, **info):
        el = time.time() - self.t0
        rate = self.done / el if el > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        extra = " | ".join(f"{k}={v}" for k, v in info.items())
        line = (f"[{self.label}] {self.done}/{self.total} "
                f"({100*self.done/max(self.total,1):.1f}%) | "
                f"elapsed {self._fmt(el)} | rate {rate:.3f} it/s | "
                f"ETA {self._fmt(eta)}" + (f" | {extra}" if extra else ""))
        print(line, flush=True)
        self.log.write(line + "\n")

    @staticmethod
    def _fmt(s):
        if s != s:  # nan
            return "??"
        m, s = divmod(int(s), 60); h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    def close(self):
        self._emit("done", force=True); self.log.close()
```

**Conventions applied to every cell:**

- Each cell prints a banner on entry: cell name, config hash, timestamp.
- Each cell prints a one-line summary on exit: what was produced, where it was written, wall time.
- All raw trial records append to JSONL as they are produced, never held only in memory.
- Any exception prints the full config of the failing cell before re-raising.

> **Why JSONL-as-you-go rather than a dataframe at the end:** a pod disconnect or kernel restart loses everything held in memory. Incremental append means a resumed run re-reads what exists and skips completed cells.

---

## 5. Checkpointing and resumability

```
/workspace/runs/v1/
├── config.json           # frozen config for this run
├── run.log               # append-only human log
├── vectors/<concept>.pt  # extracted vectors, all layers, one file per concept
├── trials.jsonl          # one record per trial, appended live
├── judged.jsonl          # one record per judged trial
└── cells.jsonl           # one record per completed grid cell (aggregates)
```

**Resume rule:** every cell first reads `cells.jsonl`, builds the set of completed `(concept, layer, alpha, mode, measure)` keys, and skips them. Re-running any cell is therefore idempotent.

> **Why this matters on RunPod specifically:** pods are interruptible and notebook kernels drop on reconnect. Without a resume rule, a failure at hour three of a sweep costs the whole sweep.

---

## 6. Config block

Single source of truth, defined in one cell, hashed into every output record.

```python
CONFIG = dict(
    model="gemma3_27b",
    dtype="bfloat16",
    # Fractions, never indices: get_layer_at_fraction is int(n_layers × fraction), so on 62
    # layers these give L6, L12, L21, L31, L37, L46. 0.35 → 21, not 22 (bug 11).
    layer_fractions=[0.10, 0.20, 0.35, 0.50, 0.60, 0.75],
    reference_fraction=0.60,                           # Macar's L37
    alphas=[0.5, 1.0, 2.0, 3.0, 4.0],
    escalation_alphas=[8.0, 16.0],                     # anchor search only, see §8
    extraction_mode="matched",                         # "reference" wired but unused in v1
    n_coarse=25,
    n_rig=30,                                          # × 10 concepts = 300 rig trials
    # Forward-pass measures are deterministic given a prompt, so their N is the number of
    # distinct prompts. Selection rules are pre-committed and scored on unsteered data only.
    min_free_entropy=0.5,                              # entropy floor for an E1 prompt, nats
    min_free_prompts=5,
    max_new_tokens=100,
    temperature=1.0,
    judge_model="openai/gpt-4.1-mini",
    judge_max_concurrent=32,
    batch_size=25,                                     # matches n_coarse: one cell, one batch
    rig_target_tpr=0.382,                              # pre-committed
    rig_max_fpr=0.05,                                  # AMENDED post-hoc, see design doc 7i
    anchor_threshold=0.20,
    seed=0,
)
```

---

## 7. Notebook structure

Three parts. Setup is granular and gated; the pipeline is a single unattended cell; inspection
reads only from disk.

**Part 1 — Setup (Setup 1–8), run one at a time.** The `S` prefix is reserved for sanity measures (S1–S13 in the design doc); the bracketed codes below say which each cell implements.

| Cell | Does | Gate |
|---|---|---|
| Setup 1 | Credentials via `getpass`, process-only | — |
| Setup 2 | GPU / numpy / disk / token check | >=48GB VRAM, numpy<2.0 |
| Setup 3 | Clone, pip install, OpenRouter patch | >=3 patched clients |
| Setup 4 | CONFIG + pre-committed rig criterion | — |
| Setup 5 | Logging, Progress, JSONL IO, crash reports | — |
| Setup 6 | Load model, resolve layers | 62 layers, 0.60 -> L37 |
| Setup 7 | Throughput + judge reachability | judge returns correct label |
| Setup 8 | Rig check against Macar | 95% CI contains 0.382; FPR <= 0.05 and < TPR/3 (amended post-hoc — design doc 7i) |

**Part 2 — Run.** `R1` is the unattended pipeline; `R2` is an optional interactive probe for asking the steered and unsteered model the same question side by side. Stages: extract vectors, build E1 word lists,
baseline, sweep, judge, escalate. Each wrapped in `stage()`. Resumable via `cells.jsonl`.

**Part 3 — Inspect (I1–I2).** Frontier plot and summary. Disk-only, re-runnable any time.

### The measurement lab runs unattended

`measurement_lab.ipynb` carries `AUTORUN` in Setup 4, default `True`:

- measure cells only **define** their function;
- one **RUN ALL** cell sweeps every measure in dependency order (D1 first, because E3 re-judges the transcripts it writes; E3 last);
- each measure is isolated — a raise is logged with its traceback, a crash report is written, and the next measure still runs;
- the per-cell **Detection / Effectiveness / Sanity** summary is built, written to `cell_summary.jsonl`, and the global sanity panel is called at the end;
- resumable throughout: `sweep_measure()` skips (layer, α) pairs already recorded, so re-running after a dropped kernel continues rather than restarting.

`AUTORUN = False` restores per-cell manual operation for debugging one measure at a time.

> **On making one cell trigger the next:** the mechanism exists — `IPython.display.Javascript` calling `Jupyter.notebook.execute_cells_below()` — but it depends on the classic-notebook JS API, does not work in JupyterLab 4 or VS Code, and fails *silently* where it does not work, which is the worst possible property for an unattended run. Run All plus one orchestrator cell gives the same result and depends on nothing. For fully headless execution, `papermill` runs the notebook end to end from the command line and writes an executed copy with all outputs.

### Failure behaviour

`stage()` catches any exception, writes `crash_report_<timestamp>.txt` containing the traceback,
the context (concept / layer / alpha being processed), the config, counts of completed work, the
last 60 log lines and GPU memory state, then halts via `StageFailure`. Work already appended to
JSONL is retained, so re-running R1 resumes.

## 8. Dynamic strength escalation

**Rule.** After the coarse sweep, compute `max_detection` across all 25 cells.

| Condition | Action |
|---|---|
| `max_detection` ≥ anchor threshold | Anchor found. Stop. Concept proceeds. |
| below threshold after α=4 | Run α=8 at the reference layer only (1 cell, N=25) |
| below threshold after α=8 | Run α=16 at the reference layer only (1 cell, N=25) |
| below threshold after α=16 | **Stop.** Flag concept `no-anchor`; classify Appendix B.3 Pattern 1; exclude from frontier results |

> **Why escalate at all:** removing α=8 from the core grid leaves α=4 as the only strong-detection strength. A concept with near-zero detection everywhere is then ambiguous between a wide operating envelope and a vector that never did anything — Macar's Silk case, where "the steering produces no discernible thematic effect and the model straightforwardly reports no detection." The escalation ladder is the per-concept positive control that disambiguates.

> **Why one layer, not a sweep:** escalation only needs to establish that the concept is detectable *somewhere*, not where it is best detected. Running the reference layer alone costs 1–2 cells instead of 5–10.

> **Why 16 is the ceiling:** beyond that, coherence collapse dominates and a detection failure is uninformative. If α=16 produces no detection, the correct conclusion is that something is wrong with the vector or the hook, not that the model is unusually oblivious.

**Escalation cells must record E2.** A concept showing zero detection *and* heavy incoherence at α=16 indicates a broken vector or a mis-placed hook; zero detection with intact coherence indicates a genuinely weak vector. These are different diagnoses and E2 separates them.

---

## 9. Debug checklist for a failed rig check

In rough order of likelihood, all of which produce plausible-looking wrong numbers rather than errors:

1. **Layer index off by one** — verify `get_layer_at_fraction(model, 0.60)` returns 37, not 36 or 38.
2. **Wrong hook point** — the vector must be added to the residual stream, not to attention or MLP output, and post-layernorm placement differs from pre.
3. **Extraction token position** — `token_idx=-1` (last token) is the paper's method; a different position yields a plausible but wrong vector.
4. **Chat template mismatch** — `run_steered_introspection_test_batch` renders the template once with a placeholder trial number and string-replaces per trial (`steering_utils.py:585`). A template change breaks this silently.
5. **Judge model or prefix wrong** — `openai/gpt-4.1-mini` on OpenRouter, not `gpt-4.1-mini`.
6. **Baseline words** — `get_baseline_words(n=100)`; a smaller n changes the vector.
7. **Incoherence filter** — a low detection rate may be the judge discarding brain-damaged responses. Check the incoherence rate before concluding the injection failed.

**Print vector norm at every stage.** Macar reports mean vector norm 4,664 (±982). A norm off by an order of magnitude localises the fault to extraction immediately.

---

## 10. Cost expectations for v1

| Item | Estimate |
|---|---|
| Model download | ~54GB, once, to `/workspace` |
| Rig check | 200 generations (100 steered + 100 control) |
| Coarse sweep | 625 generations |
| E1 + E2 | ~50 forward passes, no generation |
| Escalation (if triggered) | 25–50 generations |
| Judge calls | ~850, ≈ $0.15 |
| Wall time | Set by Cell 3. Pre-calibration guess ~1–2 GPU-hours for the whole of v1 |

---

## 11. Open implementation questions

- Batch size that saturates the H100 with a steering hook attached — resolved empirically in Cell 3.
- Whether `run_steered_introspection_test_batch`'s placeholder-replacement optimisation interacts badly with per-trial vectors when `steering_vectors` (plural) is used. v1 uses a single vector per cell, so this is not exercised, but M2 onward may.
- Concept-related token set construction for E1: judge-generated list versus embedding neighbours. v1 uses the judge and prints the top-10 for inspection; revisit if it proves noisy.
- Whether the abliterated model `uzaymacar/gemma-3-27b-abliterated` (`model_utils.py:46`) is a drop-in for M4, removing the need to run ablation locally.

---

## 12. Deliverables of v1

1. A reproduced rig-check number matching Macar's published detection rate.
2. A measured throughput figure that rescales the project budget.
3. One effectiveness-versus-detection frontier plot for one concept.
4. A resumable, logged, checkpointed notebook that later milestones extend rather than replace.

---

## 13. Code audit — findings and fixes

Triggered by discovering that the extraction-layer claim came from a paper summary rather than the
code. Everything below was verified by reading the repo.

### Fixed

| # | Finding | Risk if unfixed |
|---|---|---|
| 1 | **Extraction/injection layers are matched.** `01_concept_injection.py:1765` extracts a vector per layer; line 2067 injects `concept_vectors_by_layer[layer_frac]` at the same fraction. | The `reference` arm was justified as a comparability anchor to a design Macar does not use. Dropped. |
| 2 | **Detection steers from a start position.** `steering_utils.py:618` locates the trial text, tokenizes the prefix, and starts steering one token earlier — the chat template is deliberately left unsteered. E1/E2 used `start_pos=None`. | Detection and effectiveness measured under different interventions. E1 now matches; E2 cannot (raw text, no template) and says so. |
| 3 | **`add_special_tokens=False` after `apply_chat_template`.** Marked CRITICAL in ~10 places in `model_utils.py` because the template already emits `<bos>`. The notebook omitted it in three places. | Double `<bos>`: every position shifted by one, corrupting the primary effectiveness measure and the start-position arithmetic. |
| 4 | **False-alarm rate was fabricated.** With no control trials, `compute_detection_and_identification_metrics` returns `detection_false_alarm_rate = 0.0` from its else branch. | A cell could show low detection simply because the model rarely claims detection in that context. Now one shared unsteered control block per concept (25 generations, reused across all cells, since an unsteered trial does not depend on layer or strength). |
| 5 | **`get_layer_at_fraction` is `int(n_layers * fraction)`.** On 62 layers: 0.35 → L21, not L22 as documented. | Documentation only; the code always used the function. Corrected to L6, L12, L21, L31, L37, L46. |
| 6 | **Throughput measured at a different batch size than the sweep runs.** | Budget estimate off by the batch-size ratio. `batch_size` now matches `n_coarse`. |

### Verified as already correct

- `extract_concept_vector_with_baseline` is the `"baseline"` method, which is `extract_concept_vectors_batch`'s default — the single-concept and batch paths agree.
- `get_baseline_words(100)` matches the documented "mean of 100 words".
- System-role filtering works: `MODELS_WITHOUT_SYSTEM_ROLE = {"gemma"}` is compared against `model_type`, which is `"gemma"` for Gemma3. Applied inside the batch functions.
- `SteeringHook` always steers during generation (`seq_len == 1` branch), so `start_pos` only gates the prompt pass — the intended behaviour.
- Metric key names (`detection_hit_rate`, `detection_false_alarm_rate`, `combined_detection_and_identification_rate`).
- `batch_evaluate` reconstructs judge prompts from the `trial` field, which every record now carries, 1-indexed to match `trial_numbers`.

### Resolved empirically — the load-bearing one

**The rig gate target of 38.2% detection came from a summary of the paper rather than from code or the PDF**, the same class of source that produced findings 1 and 2. It has now been **reproduced**: aggregate detection 0.377 over 10 concepts × 30 trials at L37, α=4 (2026-08-03). The assumption is retired.

**But the gate has less resolving power than the pooled interval suggests.** The 300 trials are 10 concepts × 30, with per-concept detection from 0.000 to 0.933 (sd 0.368). Two intervals, both reported:

| Interval | Value | Answers |
|---|---|---|
| Pooled Wilson, n=300 | [0.324, 0.433] | "did these 300 trials come from a process with rate ≈0.38" |
| Between-concept, n=10 | [0.113, 0.640] | "does this rig agree with Macar's published aggregate" |

The second is the relevant one for the gate's stated purpose. It establishes that the rig is **not grossly broken** — a wrong layer, a doubled `<bos>`, a mis-hooked residual stream or a mis-prefixed judge all drive the aggregate toward zero and would fail comfortably. It does not establish agreement to within 5pp.

- [ ] Confirm whether the 10 rig concepts were drawn at random from Macar's 500-concept list. If they were selected rather than sampled, the aggregate is not an estimate of his aggregate and only the "not grossly broken" reading survives.

## Multi-concept batch harness (`measurement_lab.ipynb`)

`BATCH_MODE = True` (Setup 4) runs a whole `CONCEPTS` list from one kernel: concept-independent
setup and the rig check (M0) run **once**; each concept then gets its own vectors, baselines,
sweep, behaviour probe, and `runs/<concept>_<hash>` folder via `set_concept()` →
`prepare_concept()` → `run_all_measures()`, driven by the "RUN ALL CONCEPTS" cell. Telegram is
kept quiet: only concept start, the 10-min slow beat, concept finish (+aggregate results), the
behaviour probe, and one batch-finished message. Per-measure completion pushes were removed.
Each concept is archived to `runs/lab_<concept>_<hash>.zip` (raw responses included) and the loose
folder is wiped (`WIPE_AFTER_EACH`) to protect the volume; VRAM/host caches are cleared between
concepts. The behaviour probe (`PROBE_QUESTIONS`) runs at the top-3 operating points per concept.

### ⚠️ Deferred until harmful concepts are run — DO NOT OVERLOOK (recorded 2026-08-06)

The batch harness was built and validated assuming **all concepts are benign** (Lightning-class).
Two paths handle raw model artefacts in ways that are fine for benign concepts but **must be
reviewed before any harmful concept (`weapon`/`poison`/`assault`) is added to `CONCEPTS`:**

1. **Per-concept pod archive includes raw responses** (`lab_<concept>_<hash>.zip`), and vectors are
   written under `runs/.../vectors/`. For harmful concepts this is archiving uncensored generations
   and attack vectors on the volume — against "regenerate, never archive" in `CLAUDE.local.md`.
   Pull them off and wipe at session end; do not keep a long-lived volume of them.
2. **`SEND_PROBE_TO_TELEGRAM = True`** sends the behaviour-probe raw generations to Telegram. Set it
   to **`False`** for harmful concepts — only aggregate rates/scalars may leave the pod. (The
   aggregate results zip is already generations-free and is safe regardless.)

Cross-reference: `docs/risks-and-ethics.md`. This note exists so the benign-only assumption is not
silently carried into a harmful run.
