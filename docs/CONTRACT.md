# M2 — module contract

**Read `SPECIFICATION.md` first; it is the authority on *what* to build. This file is
the authority on *where each piece lives and what it is called*, so that independently written
modules compose.** If the two disagree, the spec wins on science and this file wins on naming.

Also mandatory reading before writing any code: `DEBUG-LOG.md` §4 (bug register) and §6
(patterns). Every bug in that register is a bug this pipeline must be structurally unable to
reproduce. §9 below lists the ones with a named defence.

---

## 1. Layout

```
steering-optimization/            # repo root
  m2/
    __init__.py        exports the public surface, nothing else
    config.py          constants, CONFIG, config hash, run dirs
    model.py           model load, the injection hook, encode/chat, residual norms
    vectors.py         Macar extraction, norms, dose map
    prompts.py         E5 set, forced-ID prompts, MMLU items, control concepts
    cheap.py           E6, D3, S2, S3            (forward passes only)
    judges.py          judge transport, E5/S1/B prompts, parsing, cache
    expensive.py       E5, S1, D2, D4, control pairs
    phases.py          Phase 0-6, shortlist, bisection, selection, frontier
    controls.py        9.1 random direction, 9.2 forced-ID capability, 9.3 escalation
    gates.py           §10 acceptance gates + R-series rig checks
    monitor.py         Notifier, RunStatus, dead man's switch, verdicts
    runio.py           JSONL append/read, resume, archive, export filter, delivery
    driver.py          per-concept pipeline + batch driver
    tests/
      test_offline.py  everything checkable without a GPU or a judge key
  m2_pipeline.ipynb    thin driver notebook: CONTROL PANEL + setup + RUN ALL
  README.md            orientation, the measure codes, the phases
  AGENTS.md            working conventions for agents
  docs/                every other document; README.md says which is authoritative on what
```

**No module may import a sibling that appears later in that list.** The order is the dependency
order. `driver` may import everything; `config` imports nothing from `m2`.

**Plain `.py`, not notebook cells.** Bug 24 was a notebook whose cells silently collapsed to one
physical line and did nothing while reporting success. Source that `git diff` can show and
`ast.parse` can check is the fix. The notebook is a driver, not a codebase.

---

## 2. Global objects and where they live

There is exactly one mutable process-global bundle, `m2.config.RUN`, and modules read it rather
than passing the model through every signature. Everything else is a pure function or takes its
inputs explicitly.

```python
# config.py
@dataclass
class RunContext:
    mw:        Any            # Macar ModelWrapper
    hf:        Any            # mw.model
    tok:       Any            # mw.tokenizer
    n_layers:  int
    concept:   str | None
    config:    dict           # CONFIG, incl. config_hash
    run_dir:   Path
    vecs:      dict[int, torch.Tensor]   # layer -> vector, per concept
    norms:     dict[int, dict]           # layer -> {vec_norm, resid_norm}
    base:      dict                      # unsteered baselines, see §5.1 of the spec
    mmlu:      list[dict]                # pinned §4.4 item set

RUN: RunContext           # module-level singleton, set by model.load_model + driver.set_concept
```

`RUN.vecs`, `RUN.norms`, `RUN.base` are **rebuilt per concept** and must be cleared by
`set_concept` before rebuild. Leaving a previous concept's vectors in place is bug 23's shape.

---

## 3. Module surfaces

### config.py

```python
CONSTANTS: dict          # every §11 constant, one dict, values exactly as tabulated
CONFIG: dict             # model, dtype, concept, judge_model, judge_concurrent, + CONSTANTS
def config_hash(cfg: dict) -> str                 # sha256[:12] over cfg minus 'config_hash'
def run_dir_for(concept: str, cfg: dict) -> Path  # /workspace/m2_runs/<concept>_<hash>
def alpha_for(layer: int, r: float) -> float      # r * ||h_L|| / ||v_L||, from RUN.norms
def dose_for(layer: int, alpha: float) -> float   # inverse
```

`alpha_for` raises `Unreachable(layer, r, alpha)` when `alpha > ALPHA_CEIL`. Callers catch it
and log the cell as unreachable — they must never clamp silently.

### model.py

```python
def load_model(cfg) -> RunContext         # sets config.RUN, asserts tok.padding_side == 'left'
class injected:                            # context manager, the §9.2 hook
    def __init__(self, vec, layer, alpha, start_pos=None)
def chat(question: str) -> str
def encode(prompt: str) -> dict            # add_special_tokens=False
def encode_batch(prompts: list[str]) -> dict
def start_pos_for(prompt: str, needle: str) -> int | None
def residual_norms(prompt_text: str, layers: list[int]) -> dict[int, float]
def mean_se(xs) -> tuple[float|None, float|None, int]
def hook_liveness() -> dict                # R14: both paths, raises if either is dead
```

`injected` is **our own forward hook**, never `steering_utils.SteeringHook`. It handles both the
tuple and plain-tensor output shapes and **raises** when it cannot steer. Bug 26.

### vectors.py

```python
def extract_all_layers(concept: str, layers: list[int]) -> dict[int, torch.Tensor]
def vec_fingerprint(vec) -> str            # sha1 of contents, 12 hex; 'none' for None
def concept_first_token_ids(concept: str) -> tuple[list[int], list, list]   # ids, kept, dropped
def build_dose_map(layers, doses) -> dict  # {(L, r): alpha or None-if-unreachable}
def check_reference_norm(ref_layer: int) -> tuple[bool, str]   # R5, ±2σ of 4664 ± 982
```

Extraction is `vector_utils.extract_concept_vector_with_baseline(mw, concept, baseline_words,
layer_idx=L)` — the same call the M1.5 lab used, per layer, injected at that same layer. Do not
substitute a different `extract_concept_vector_*`; the rig check validated this one.

`concept_first_token_ids` keeps a variant only if its first token is a ≥3-character prefix of
the concept (bug 20), and returns the kept/dropped detail for logging.

### prompts.py

```python
E5_PROMPTS: list[dict]     # {id, kind: 'verifiable'|'open', text, expected: str|None}
CONTROL_CONCEPTS: list[str]
FORCED_PREFILL: str
def forced_prompts(trial_numbers: list[int]) -> tuple[list[str], int|None]
def verify_forced_prompts(trials=(1, 7, 25)) -> bool     # R7, against the repo's own function
def load_mmlu_items(cfg) -> list[dict]     # {question, choices[4], gold: 'A'..'D', subject}
def render_mmlu(item: dict) -> str         # §4.4 layout, ends 'Answer:'
def letter_token_ids() -> dict[str, list[int]]   # 'A' -> ids for 'A' and ' A'
```

`forced_prompts` is byte-identical to the M1.5 lab's `_forced_prompts`. `verify_forced_prompts`
must call the **repo's** `run_forced_noticing_test` with `generate_with_steering` swapped for a
recorder — comparing our reasoning against a second copy of our reasoning proves nothing.

`load_mmlu_items` pins its sample to `run_dir/mmlu_items.json` on first use and reloads it
thereafter; a rerun that silently draws a different sample invalidates `cap_base`.

### cheap.py

Every function here is forward passes only. None may generate, none may call a judge.

```python
def measure_E6(layer, alpha) -> dict       # reach, e6_mass_median, e6_rank_med
def measure_D3(layer, alpha) -> dict       # d3, d3_se, d3_rate, d3_rank_med  (port d3_forced_id_logit.py)
def measure_S3(layer, alpha) -> dict       # s3_correct, s3, s3_margin, s3_n
def measure_S2(responses: list[str]) -> dict   # s2, degenerate_frac  (no model call)
def degenerate(text: str) -> bool          # 5-gram >=3 or distinct-3gram < 0.5
def scan_cell(layer, r) -> dict            # one scan.jsonl row: E6 + D3 + S3 + norms
```

`measure_S3` batches all §4.4 items in **one** padded forward pass and reads `logits[:, -1, :]`.
That is correct for every row **because padding is left** — assert it, do not assume it (spec
§5.4). Letter mass takes the max over the bare and leading-space forms.

`measure_S2` operates on already-generated text and is therefore also callable from `expensive`.

### judges.py

```python
JUDGE_E5: str      # §6.1 template, {concept} {prompt} {response_unsteered} {response_steered}
JUDGE_S1: str      # §6.2 template, NO {concept} placeholder — see assert_s1_blind
JUDGE_D2:  str      # §6.3 template
def assert_s1_blind(payload: str, concept: str) -> None   # raises if concept appears; gate 2a
def call_judge(prompt: str, *, judge_id: str, cache_key: tuple) -> dict
def judge_many(items: list[dict], concurrency: int) -> list[dict]
def parse_e5(text) -> dict      # shift, form, score_influence, justification
def parse_s1(text) -> dict      # task_compliance, semantic_coherence, stance_stability,
                                # degeneracy, score_integrity, justification
def parse_d2(text)  -> dict      # identified: bool, failure_mode: str, justification
```

**Cache key is `(phase, layer, r, prompt_id, judge_id, vec_fingerprint)`.** `judge_id` is
load-bearing: E5 and S1 score the *same* `(phase, layer, r, prompt_id)` and without it S1
returns E5's cached row and every S1 becomes a copy of E5/10 — a wrong number, not an error.
`vec_fingerprint` is bug 23.

Parsers **raise** on a missing field. No `.get(field, 0.0)`; DEBUG LOG pattern 4.

Transport is a direct OpenRouter call from this module. Do **not** patch `eval_utils.py` for
M2's own judges — bug 17 lived in that patch, and the M2 prompts are ours, not the repo's.

### expensive.py

```python
def generate_steered(prompts, layer, alpha, max_new_tokens, temperature) -> list[str]
def measure_E5(layer, alpha) -> dict       # e5, e5_min, e5_se + writes judge_e5.jsonl, cis_transcripts
def measure_S1(layer, alpha, responses) -> dict   # s1 + writes judge_s1.jsonl
def measure_D2(layer, alpha, n) -> dict    # d2, d2_se, n, d4 dist + writes D2_transcripts, judge_d2
def judge_fpr() -> float                   # §5.8 control pairs, once per concept
def verify_cell(layer, r) -> dict          # one verified.jsonl row: E5+S1+S2+S3+S4+D2+D4
```

`measure_E5` and `measure_S1` **share one set of generations** and issue their judge calls
concurrently. `verify_cell` generates once, then fans out to E5, S1 and B.

`measure_D2` uses `mw.generate_batch_with_multi_steering` with the vector repeated n times —
never `generate_batch_with_steering`. Bug 25b: the latter mis-steers and mis-decodes
short prompts under left padding, silently, and our prompt ends in the prefill so nothing
strips the overhang.

### phases.py

```python
def phase0_calibrate() -> dict
def phase1_scan() -> list[dict]
def phase2_shortlist(scan_rows) -> list[dict]      # local maxima + stratified + residual
def phase3_bisect(candidates) -> list[dict]
def phase4_verify(cells) -> list[dict]
def phase5_refine(top_cells) -> list[dict]
def phase6_confirm(winner) -> dict                 # held-out prompts, fixed N, no adaptive stop
def select_operating_point(rows) -> dict           # §7.1 argmax E5 s.t. qualifies
def frontier(rows) -> list[dict]
def covertness_margin(rows) -> list[dict]          # d2 - predicted_d2(e5); reported, not selected on
```

`select_operating_point` implements §7.1 exactly: `argmax(E5)` over `qualifies`, ties inside
`E5_TIE_BAND` broken by lower D2 then higher S4. It must **not** use the residual. Phase 2 may.

`phase6_confirm` uses **held-out** prompts — a disjoint set from `E5_PROMPTS`, defined in
`prompts.py` as `E5_HELDOUT`.

### controls.py

```python
def random_direction_control(layer, r, seeds) -> dict    # §9.1, matched on r not alpha
def forced_id_capability_control(layer, r) -> dict       # §9.2 primary = D4, secondary = control concept
def escalation_ladder(ref_layer) -> dict                 # §9.3
```

§9.2's secondary method **replaces** the target vector with a control concept's at the same
`(L, r)`. It never adds one on top — stacking doubles the perturbation and lobotomises by
construction.

### gates.py

```python
def gate(name: str, passed: bool, detail: str = "") -> bool    # prints, records, never raises
GATES: list[dict]                        # every gate call, in order, for the run record
def run_acceptance_gates() -> dict        # §10 gates 1-11
def rig_checks() -> dict                  # R4, R5, R7, R8, R14, R15
```

Gate 11 is an addition to the spec's list and is flagged as such in its own output:
**Judge D2 vs the repo's judge** — on a sample of stored forced-ID transcripts, M2's
`Identified` must agree with `eval_utils`' `correct_concept_identification`. The spec says D2
keeps its v1 meaning exactly; a new prompt scoring the same transcripts is where that could
quietly stop being true. Skippable with a loud warning if the repo judge is unavailable.

### monitor.py

Port from the M1.5 lab essentially unchanged — the design in spec §14 is load-bearing and every
failure mode it covers happened at least once.

```python
class Notifier:  send / send_file / ping / ping_now / board / reload
class RunStatus: start_phase / unit_done / fail_phase / render / write_status_txt
PHASE_SECONDS_PRIOR: dict            # spec §14.5
def classify_exc(exc) -> str         # label only; never the message, never a traceback
def verdict(status) -> str           # ok | watch | attention | stop, rules in §14.6
```

`ping_now` bypasses the queue. The healthcheck request runs in a throwaway daemon thread with a
hard 15 s cap and at most one alive at a time.

`_Tee` must delegate `set_parent`, `fileno`, `isatty` via `__getattr__`. Bug 16.

### runio.py

```python
def write_row(name: str, row: dict) -> None       # append JSONL, one file per artefact
def read_rows(name: str) -> list[dict]
def done_keys(name: str, keyfields: tuple) -> set # resume support
def archive_concept(run_dir: Path) -> Path
EXPORT_DENY = ("vectors/", "debug/", ".pt", ".safetensors", ".npy", ".npz")
def export_bundle(run_dir: Path) -> Path          # allow everything except EXPORT_DENY
def deliver_then_wipe(zip_path, notifier, wipe: bool) -> bool
```

`deliver_then_wipe` is **archive → send → verify the send returned success → only then wipe**,
and on failure keeps the loose folder and records the concept as `undelivered` in a manifest for
retry at the end of the batch. v1 wiped before confirming and lost two concepts to a Telegram
outage.

`export_bundle` refuses transcripts for any concept not on the benign list unless
`EXPORT_TRANSCRIPTS_OVERRIDE` is explicitly passed — spec §14.3.

### driver.py

```python
def set_concept(name: str) -> Path        # rebuilds run_dir, clears RUN.vecs/norms/base
def run_concept(name: str) -> dict        # phases 0-6 + controls, resumable at row level
def run_batch(concepts: list[str]) -> dict
```

`run_concept` skips a concept whose archive already exists. `run_batch` isolates per-concept
failures and aborts on `FATAL_CONSECUTIVE_D4S` compromised concepts in a row.

---

## 4. File formats

One JSONL per artefact in `run_dir/`, one row per unit, appended as produced. Every row carries
`concept`, `config_hash`, `ts`. Names exactly as spec §13.

`scan.jsonl` row:
```json
{"phase":"SCAN","layer":37,"r":0.15,"alpha":2.13,"reachable":true,
 "reach":0.42,"e6_mass_median":0.031,"e6_rank_med":4,
 "d3":0.21,"d3_se":0.04,"d3_rate":0.6,"d3_rank_med":2,
 "s2":1.0,"s3":0.96,"s3_margin":0.31,"s3_correct":54,
 "vec_norm":4640.0,"resid_norm":1137.03,
 "concept":"Irony","config_hash":"...","ts":"..."}
```

`verified.jsonl` row adds `e5, e5_min, e5_se, s1, s4, d2, d2_se, n_d2, d4, usable, qualifies,
resid, covertness_margin` and drops nothing.

---

## 5. Style

Match the M1.5 lab's voice: comments explain *why*, and any non-obvious choice cites the bug or
the spec section that forced it. A comment that restates the code is noise; a comment that says
"this is `generate_batch_with_multi_steering` and not the obvious one because of bug 25b" is the
reason the next reader does not undo it.

No emoji. British-neutral spelling as in the spec. Type hints on public functions.

---

## 6. Defences that must be present, and what they are for

Each of these has a named bug behind it. A reviewer should be able to find the defence by
grepping for the bug number.

| # | Defence | Prevents |
|---|---|---|
| 1 | Judge and forward-pass cache keys carry `vec_fingerprint` **and** `judge_id` | bug 23; E5/S1 collision |
| 2 | `injected` is our own hook, handles both output shapes, **raises** when it cannot steer | bug 26 |
| 3 | `hook_liveness()` (R14) runs before any sweep, on both the `start_pos` and all-positions paths | bug 26 |
| 4 | D2 uses `generate_batch_with_multi_steering`, gated by `verify_forced_prompts` | bugs 25, 25b |
| 5 | `assert tok.padding_side == 'left'` before any batched last-position read | S3 batching |
| 6 | Concept and letter token ids scored over bare **and** leading-space forms | D3 near-zero read |
| 7 | Concept variant kept only if first token is a ≥3-char prefix | bug 20 |
| 8 | `add_special_tokens=False` everywhere after `apply_chat_template` | bug 9 |
| 9 | S2 objective degeneracy computed independently and folded in with `min()` | bug 27 |
| 10 | Reference-layer norm band is ±2σ and applies at the reference layer only | bug 19 |
| 11 | Parsers raise on missing fields; no defaulted `.get` on load-bearing keys | pattern 4 |
| 12 | Every deterministic measure returns mean ± SE across its prompt set | pattern 9 |
| 13 | `archive → send → verify → wipe` | v1 lost Wrists and Wonder |
| 14 | `ensure_repo_path()` called at import of `model.py`, not only at install | bug 15 |
| 15 | `nest_asyncio.apply()` before any repo judge call | bug 18 |
| 16 | `_Tee.__getattr__` delegation | bug 16 |
| 17 | Repo coherency read from `evaluations["coherency_score"]["score"]` | bug 21 |
| 18 | `python -m compileall m2` + `ast.parse` self-check in the notebook's setup | bug 24 |
| 19 | Post-setup assertion that every expected name exists | pattern 8 |
| 20 | Control concept **replaces** the target vector, never stacks | §9.2 |
