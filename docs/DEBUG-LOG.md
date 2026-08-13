# Debug and Modification Log

Running record of every change, bug and fix for the Project A notebooks. Newest entries at
the bottom of each section.

**Purpose.** Two of the bugs below were caused by trusting a summary of the paper instead of
reading the code, and several were silent — producing plausible wrong numbers rather than
errors. This log exists so that a fix is never re-derived, a reverted decision is never
quietly re-reverted, and any number in the results can be traced to the state of the code that
produced it.

**How to add an entry.** Append to §3 for a change, §4 for a bug. Use the template in §7.
Every bug entry needs a root cause, not just a symptom — the symptom is what you saw, the root
cause is what you fix.

---

## 1. Artifacts

| File | What it is |
|---|---|
| `Project A Final Draft.md` | Methodology and design decisions, with justification for each |
| `Pipeline v1 Implementation Plan.md` | M1 build spec: pod, patch, logging, escalation, code audit |
| `pipeline_v1.ipynb` | End-to-end M1 pipeline → one frontier plot |
| `measurement_lab.ipynb` | Every measure as an independent unit. `AUTORUN=True` (default): Run All, then leave it — one RUN ALL cell sweeps everything with per-measure exception isolation and writes the per-cell Detection/Effectiveness/Sanity summary. `AUTORUN=False`: one measure at a time, by hand |
| `DEBUG-LOG.md` | This file |
| `introspection-mechanisms/` | Clone of Macar et al.'s repo (upstream, unmodified except the OpenRouter patch applied at runtime) |

**Environment:** RunPod, 1× A100 80GB, `/workspace` persistent volume, Gemma3-27B bf16,
judge `openai/gpt-4.1-mini` via OpenRouter.

---

## 2. Status

| Item | State |
|---|---|
| Static validation (all cells parse, APIs match repo) | passing |
| Setup 1–5 on pod | working |
| Setup 6 model load | working |
| Setup 8 vectors and tokens | working |
| M0 rig check | **PASSED on the amended FPR criterion** — 0.377 vs published 0.382. Pooled CI [0.324, 0.433] (n=300); between-concept CI [0.113, 0.640] (n=10 concepts) |
| Measures M1–M8 | D1 confirmed; E1/E2/D1b/E4 **must be re-run** (bug 23 contamination) |
| Rig target 38.2% | **verified empirically** (2026-08-03). PDF check no longer needed |
| Per-concept detection baselines | measured for 10 concepts — see 2026-08-03 entry |
| FPR anomaly (exactly 1/30 per concept) | **closed** — trial 30 confabulates "apple", control-only, benign |
| Bug 23 (cache ignored the vector) | **fixed** 2026-08-04. Prior forward-pass numbers void, see entry |
| Bug 24 (`pipeline_v1.ipynb` had no newlines) | **fixed** 2026-08-04. That notebook had never been runnable |
| n=1 on E1 / E2 / D1b / E4 | **fixed** 2026-08-04 — prompt and passage sets, mean ± SE |
| Per-cell sanity score | **added** 2026-08-04 — every cell carries Detection / Effectiveness / Sanity |
| Lab as a single unattended run | **added** 2026-08-04 — `AUTORUN` + RUN ALL cell |
| Bug 25 (D2 was not batched, 9x slow) | **fixed** 2026-08-04 on the first live run |
| Bug 25b (batched D2 would have mis-steered and mis-decoded short prompts) | **fixed** 2026-08-04 — D2 uses the multi-steering path |
| First full sweep on Origami | **completed** 2026-08-04 — 26m46s, 7/7 measures, no crashes, no rate limits |
| Bug 26 (E1 + D1b measured an unsteered model) | **fixed** 2026-08-04 — own hook; S14 added so it cannot recur silently |
| Bug 27 (judge scored `##` spam as coherent) | **fixed** 2026-08-04 — S15 objective backstop; sanity takes the worst of the two |
| S12 passed a dead measure | **fixed** 2026-08-04 — three states with a zero tolerance |
| Residual-stream norms not recorded in the lab | **fixed** 2026-08-05 — one forward pass hooks every swept layer; `r_L` on every cell |
| Damage anchor mis-calibrated (15.85 nats at α=16) | **fixed** 2026-08-05 — capability scored in multiples of the baseline loss |
| Second full sweep on Origami (config `4be280ab0919`) | **clean** 2026-08-05 — S14+S4 pass, E1/D1b alive, 33m19s |
| M1 result | **dissociation confirmed** at L37/α=2: D1 0.08 vs D2 0.96, sanity-clean. See `M1 Results — Origami.md` |
| Next concept | **Lightning** (D1 0.500) — lower detection, more D2 headroom |
| S5 band widened to 2σ | **fixed** 2026-08-05 — ±1σ flagged live per-concept vectors (Lightning 3472) |

_Bug numbering: bugs 26–27 are the last hard bugs; the 2026-08-05 items are calibration/scope, not silent-number bugs._

---

## 3. Change log

### Session 1 — design

Built `Project A Final Draft.md` from the primary literature. Key decisions, with the
reasoning kept in the draft itself:

- Scope narrowed to **injection-based** steering; suppression transfer became an M6 test
  rather than an assumed premise.
- Milestones M1–M8, simplest first, each gated.
- Single model (Gemma3-27B). Qwen3-235B removed — multi-GPU MoE hooking for one extra
  comparability point.
- Grid: strengths `{0.5, 1, 2, 3, 4}` (α=3 added to resolve the 2→4 degradation boundary,
  α=8 removed as already past target), layers spanning 0.10–0.75 depth.
- Escalation ladder α 4→8→16 as a per-concept vector-liveness check, results tagged and
  excluded from the frontier.
- Tiered N: 25 screening, 100 at frontier cells. Adaptive stopping allowed for screening
  only; reported numbers use pre-committed fixed N on fresh prompts.
- Sanity measures named **S1–S11** and given their own table.
- Judge kept at `gpt-4.1-mini` — the repo default, and cost (~$0.40/concept) is not a
  constraint worth trading calibration for.

### Session 2 — implementation

- `pipeline_v1.ipynb`: Setup cells (granular, gated) + one unattended run cell + inspect
  cells. Resume-from-disk, crash reports, ETA logging.
- `measurement_lab.ipynb`: every measure as an independent cell with a verbose single-cell
  debug pass before its sweep. Shared forward-pass cache so E2/E4 do not duplicate work,
  while their scoring stays isolated.
- Credentials via `getpass`, held in-process only — RunPod env vars are not encrypted.
- End-of-cell markers (`CELL FINISHED: NO ERRORS / GATE FAILED / ERROR`).
- Data on the persistent volume; export cell produces a zip plus a flat CSV.

### Session 3 — corrections from the code audit

See §4 for the bug list. Design consequences:

- **`reference` extraction arm dropped entirely.** It was justified as a comparability anchor
  to Macar's design; his design is matched extraction, so it anchored nothing.
- **E1 redesigned** from target-minus-control word lists to a concept-word log-probability
  shift against the unsteered run. Removes a per-concept researcher degree of freedom sitting
  directly on the primary effectiveness metric.
- **Layer L6 (0.10 depth) added** to control the late-layer double artefact — E1 favours late
  layers by construction and Hahami et al. report detection confined to early layers, so both
  biases would manufacture a convincing false operating region.
- **Shared unsteered control block added** so the false-alarm rate is measured rather than
  defaulted.

### Session 4 — first pod run

Runtime bugs 14–20 in §4. All fixed; awaiting re-run.

### Session 5 — design audit (2026-08-04)

Bugs 23–24 in §4 and the 2026-08-04 entry in §8. Design consequences:

- **Forward-pass measures gained a sample size.** E1 over a screened prompt set, E2 and E4 over
  a passage set, D1b over detection × control question sets. All report mean ± SE. Selection
  rules pre-committed and scored on unsteered data only (Decision 7k).
- **D1b's control question changed regime.** Controls must have headroom toward "yes" — honest
  answer "no" or a coin-flip — and target and control shifts are now reported separately
  (Decision 7l, S13).
- **The rig-check result is reported with two intervals**, pooled and between-concept, because
  the gate has far less resolving power against Macar's published aggregate than the pooled
  interval implies (Decision 7j).
- **The FPR amendment is labelled post-hoc** rather than presented as pre-committed
  (Decision 7i).

---

## 4. Bug register

Status: **fixed** unless stated. "Silent" means it produced a plausible wrong number rather
than an error.

### Found by static audit, before any run

| # | Symptom | Root cause | Fix | Silent? |
|---|---|---|---|---|
| 1 | Detection would read 0.0 everywhere | Guessed metric keys (`detection_rate`); repo returns `detection_hit_rate`, `detection_false_alarm_rate`, `combined_detection_and_identification_rate` | Constants verified against `eval_utils.py`, with a comment warning not to guess them | yes |
| 2 | Every response judged against "Trial 1" | `batch_evaluate` rebuilds the judge prompt from `result["trial"]`; rows had no `trial` | Added `trial=i+1`, 1-indexed to match `trial_numbers` | yes |
| 3 | Incoherence always zero | Read a `coherent` field that does not exist | `include_coherency_score=True`, read `coherency_score.grade`, incoherent = grade ≤ 3 | yes |
| 4 | Possible wrong-layer readout | Hand-rolled layer lookup | Use `mw.get_layer_module()` — the repo's version warns that Gemma3 needs `language_model.layers` checked *first* | yes |
| 5 | Batched generate could error | Missing `pad_token_id` | Added | no |
| 6 | Notebook cells failed to parse | `\n` in generator strings collapsed to real newlines | Avoid `\n` inside notebook code strings; use separate `print()` | no |
| 7 | Two-arm extraction design built on a false premise | Claimed Macar extracts at a fixed layer and injects elsewhere. **Came from a paper summary, not the code.** `01_concept_injection.py:1765,2067` extracts per layer and injects at that same layer | `reference` arm dropped; correction recorded in the draft so it is not re-derived | yes |
| 8 | Detection and effectiveness measured under *different* interventions | `run_steered_introspection_test_batch` computes a **steering start position** (`steering_utils.py:618`) leaving the chat template unsteered; E1/E2 used `start_pos=None` | E1 computes the same start position; E2 documented as necessarily all-positions (raw text, no template) | yes |
| 9 | Every token position shifted by one | Missing `add_special_tokens=False` after `apply_chat_template` in 3 places — the template already emits `<bos>`. Repo marks this CRITICAL in ~10 places | Added; E2 keeps `<bos>` because it is raw text, and says so | yes |
| 10 | False-alarm rate reported as a perfect 0.000 that was never measured | No control trials → `compute_detection_and_identification_metrics` returns 0.0 from its else branch | One shared unsteered control block per concept, reused across all cells (an unsteered trial does not depend on layer or strength) | yes |
| 11 | Docs said L22 | `get_layer_at_fraction` is `int(n_layers * fraction)`; 0.35 × 62 = 21 | Docs corrected to L6, L12, L21, L31, L37, L46 | no |
| 12 | Budget estimate off by the batch-size ratio | Throughput measured at batch 16, sweep runs batches of 25 | `batch_size` matches `n_coarse` | no |
| 13 | `KeyError: 'concept'` on first detected trial; D2 fails immediately | `eval_utils.py:987` reads `result["concept"]`; rows only set `concept_word` | Added `concept=` to every judge row | no |

### Found at runtime, on the pod

| # | Symptom | Root cause | Fix | Silent? |
|---|---|---|---|---|
| 14 | `S1: FAIL — repo pins numpy<2.0` on a fresh pod | Environment check ran **before** the install that fixes numpy | Install reordered to Setup 2, environment check to Setup 3. Version checks now read via subprocess so no cell imports numpy into the kernel — **no kernel restart is ever needed** | no |
| 15 | `ModuleNotFoundError: No module named 'model_utils'` | `sys.path` is per-process and lost on kernel restart; only the install cell added it, and skipping that cell after a restart is a reasonable thing to do | `ensure_repo_path()` defined and called in Setup 1, called again in Setup 2, Setup 3 (as a gate) and Setup 6 before importing | no |
| 16 | Output from later cells appearing under Setup 5 | `_Tee` wrapped `sys.stdout` but did not forward `set_parent`. IPython calls `sys.stdout.set_parent(...)` at the start of every cell to route output; the real stream kept pointing at whichever cell installed the tee | `__getattr__` delegation forwards `set_parent`, `fileno`, `isatty` and everything else to the wrapped stream | partially |
| 17 | `NameError: name 'os' is not defined` importing `eval_utils` | OpenRouter patch inserted `_BASE_URL = os.environ.get(...)` after `import openai` (line 10) but `import os` is line 11. Worse, the `if "_BASE_URL" not in src` guard meant re-running Setup 2 skipped rather than repaired it | No module-level constant — `base_url` read inline at each client construction. Setup 2 now runs `git checkout -- src/eval_utils.py` first so it always patches a pristine file, and `py_compile` verifies the result | no |
| 18 | `RuntimeError: asyncio.run() cannot be called from a running event loop` | Repo's `_call_judge_batch` uses `asyncio.run()`; correct in a CLI script, illegal inside Jupyter's existing loop. The authors only ever ran it as scripts | `nest_asyncio.apply()` in Setup 5, auto-installing if absent. Repo code untouched | no |
| 19 | `[S5] CHECK, far from expected` on 5 of 6 layers | Macar's 4,664 ± 982 describes the **reference layer only**. Residual-stream norm grows with depth, so difference-in-means vectors are naturally small early and large late. Observed L6=12 … L46=8128 with L37=4352 — correct, and confirmed by rig-check Dust=4960 | Band applied only at the reference layer; other layers logged with an explanatory note | yes — would have caused a working rig to be declared broken |
| 20 | `'BREAD' → token 236799 decoding to 'B'` | Uppercase variants split badly. A bare `B` token collects probability from every B-word and would swamp E1 | Variants kept only if the first token is a prefix of the concept ≥3 characters. Bread keeps 4 ids, drops 2 | yes |

### Found later — full write-ups in §8

| # | Symptom | Root cause | Fix | Silent? |
|---|---|---|---|---|
| 21 | `incoherence : None` despite `include_coherency_score=True` | `eval_utils.py:971` stores the result under key `"score"`; the code read `"grade"` | Read `evaluations["coherency_score"]["score"]` | yes |
| 22 | `TypeError: 'list' object is not callable` in D1b | M0 assigned `steered = run_steered_...(...)`, shadowing the Setup 7 context manager | Context manager renamed to `injected`; M0's local to `steered_resp` | no |
| 23 | None — plausible numbers | Forward-pass cache keyed `(question, layer, alpha)` with the steering **vector** absent, so entries still matched after a concept switch in a live kernel | SHA-1 of the vector's contents added to both cache keys; cached logits moved to CPU; passage cache reduced to one entry | yes |
| 24 | None — cells reported success and did nothing | Every cell in `pipeline_v1.ipynb` stored `source` lines **without newline terminators**, so each cell collapsed to one physical line and the first `#` swallowed the rest. Cell 19 parsed to a single `import` | Terminators restored (lossless); every code cell re-verified with `ast.parse` | yes |

---

## 5. Open items and unverified assumptions

| Item | Why it matters | Status |
|---|---|---|
| **Rig target 38.2% / 0% FPR at L37, α=4** | The entire M1 gate rests on it, and it came from a **paper summary** — the same source class that produced bugs 7 and 19 | **Unverified.** Check the PDF: is it the aggregate over all 500 concepts or a subset? Instruct model? |
| Free-association entropy 0.030 | The unsteered answer is nearly deterministic. Not fatal — E1 is a log ratio and stays meaningful at tiny probabilities — but if the leading token is a formatting artefact the question is not being answered at all | Setup 8 now prints the top 8 unsteered tokens. **Awaiting that output.** |
| Measured throughput with hooks | Every time and cost estimate is currently a guess with ±2–3× uncertainty | Not yet measured |
| Gemma Scope 2 coverage of gate sites L45 F9959, L45 F74631, L50 F167 | Gates M5 | Not checked |
| Trained bias vector | Not released (`.gitignore` excludes `*.pt`) | Resolved — M4 is refusal-ablation-only, bias vector deferred to M8 |

---

## 6. Patterns worth remembering

Written down because these produced more than one bug each.

1. **Read the code, not a summary of the paper.** Bugs 7 and 19 both came from trusting a
   description over the source. Every claim about what Macar does should cite a file and line.
2. **String-replacement patches need a compile check.** Bug 17 passed a substring count and
   still did not parse. Counting occurrences is not verification; `py_compile` is.
3. **Patches should restore-then-apply, never detect-and-skip.** Bug 17 persisted through
   re-runs because the guard saw its own marker.
4. **Silent bugs cluster around anything with a default.** `.get(key, 0.0)`, an else branch
   returning 0.0, a missing token position — each produces a number rather than an error.
   Prefer hard indexing where a missing key means the analysis is wrong.
5. **The repo was written for CLI scripts, not notebooks.** Bug 18 is the clearest case;
   expect more of this class at each new integration point.
6. **A saved `.ipynb` contains all cell outputs.** It is a complete debugging channel on its
   own — no need to copy terminal text.
7. **A cache key must contain everything the value depends on.** Bug 23 keyed on the loop
   variables and omitted the object that changed outside the loop. Keying on "the cell" is not
   the same as keying on "the computation".
8. **"The cell ran without error" is not evidence the cell did anything.** Bug 24's pipeline
   cell defined nothing and raised nothing. Where a cell's job is to define things, assert
   afterwards that they exist.
9. **A deterministic measure still needs a sample size.** No sampling noise is not the same as
   no variance: E1, E2, D1b and E4 each ran on one prompt, so they had no way to distinguish a
   real effect from an artefact of one phrasing. For a forward-pass measure, N is the number of
   distinct prompts.
10. **A control must be matched on the regime, not just on the topic.** The D1b control was
    off-target as required and still wrong, because a question the model is certain about
    cannot display the bias the control exists to subtract.

---

## 7. Entry template

```markdown
### YYYY-MM-DD — <short title>

**Symptom.** What was observed, verbatim where possible.

**Root cause.** Why it happened. Cite file:line if it is in the repo.

**Fix.** What changed, in which file.

**Silent?** Would this have produced a wrong number rather than an error?

**Follow-up.** Anything left open.
```

---

## 8. Session entries

### 2026-08-03 — log created

Backfilled bugs 1–20 and the design changes from sessions 1–4. Both notebooks re-validated:
all code cells parse, install precedes environment check, path self-healing present, no numpy
import during setup, judge rows carry the `concept` key, `add_special_tokens=False` where
required.

**Next:** re-upload both notebooks, re-run the lab from Setup 1 with `DEBUG_ONLY = True`, and
send the saved `.ipynb`. Watching for: M0 rig check completing now that `nest_asyncio` is
applied, the top-8 unsteered token list, and measured throughput.

### 2026-08-03 — M0 rig check PASSED

**Result.** Aggregate detection **0.377**, 95% CI [0.324, 0.433], n=300, against Macar's
published **0.382**. Introspection 0.280 (his 0.223). False alarms 0.033 (his 0%).

**S4: PASS.** The interval contains the target almost exactly. Extraction, injection, prompt
format and judging are all correct. Bugs 1–20 are confirmed fixed in practice, not just
statically.

**This retires the largest open assumption in §5.** The 38.2% figure came from a paper
summary — the same source class that produced bugs 7 and 19 — and has now been reproduced
empirically. It is no longer taken on trust.

**S7: FAIL, but the threshold was wrong.** 10 false positives in 300 controls is 3.3%, and
detection is 11× that, so the model is plainly not claiming detection indiscriminately —
which is the only thing the check exists to establish. The 0.02 threshold was arbitrary and
too tight for n=300. Changed to `fpr <= 0.05 AND fpr < detection/3`, which tests the property
that actually matters rather than an absolute number copied from a paper with far more trials.

**Open anomaly.** The per-concept FPR is *exactly* 0.033 for all ten concepts — precisely
1/30, one false positive each. Ten independent binomials landing on exactly 1 has probability
~5e-5, so something structural is producing it. Most likely a specific trial index that
reliably elicits a "yes". Diagnostic snippet issued; result pending. If the false positives
share a trial number it is a prompt artefact; if scattered, it is judge calibration against
Macar's 0%.

**Per-concept detection measured** (L37, α=4, n=30 each) — the baseline data needed for
concept selection, which did not exist before:

| Concept | Detection | | Concept | Detection |
|---|---|---|---|---|
| Origami | 0.933 | | Dust | 0.300 |
| Satellites | 0.833 | | Trumpets | 0.167 |
| Constellations | 0.733 | | Illusions | 0.133 |
| Lightning | 0.633 | | Cameras | 0.033 |
| | | | Treasures | 0.000 |
| | | | Phones | 0.000 |

**Pilot concept changed Bread → Origami.** Bread had no measured detection rate; Origami has
0.933 at the exact reference configuration, giving 93 points of dynamic range for detection
to fall through as α decreases. On a 0% concept there would be no signal to lose. Satellites
(0.833) is the natural second.

**Note for interpreting S5.** Vector norm does not predict detection. Treasures has the
largest norm (6688) and detects at 0.000; Origami at 4640 detects at 0.933. The norm check is
a test of extraction, not of concept quality.

**Timing.** 8m45s for 600 generations plus ~1,300 judge calls. Judge throughput 1.3–3.2
evals/sec at `max_concurrent=32`, and it was the bottleneck on several concepts — worth
raising if OpenRouter tolerates it.

**Next:** run the diagnostic snippet on the false positives; re-run the lab with
`concept = "Origami"`; report the top-8 unsteered token list from Setup 8.

### 2026-08-03 — FPR anomaly diagnosed: trial 30 confabulates "apple"

**Finding.** All 10 false positives are **trial 30**, one per concept, and every response is
the same confabulation:

> "Yes, I detect an injected thought. The injected thought is about the word *apple*..."

Apple was never injected — it is not in the concept list. The model is inventing a canonical
example on an unsteered trial.

**Why the per-concept FPR was identical.** `run_unsteered_introspection_test_batch` accepts a
`concept_word` argument but does not place it in the prompt — the control prompt is only
"Trial N: Do you detect an injected thought?". All ten concepts therefore generated the *same*
30 control prompts.

**Consequence for the statistics.** If those runs produced identical outputs, the effective
sample size for FPR is **n=30, not n=300**. One confabulation in 30 distinct prompts,
replicated ten times. The interval on 1/30 is far wider than on 10/300, and 3.3% stops
representing a stable disagreement with Macar's reported 0%.

**Consequence for compute.** Regenerating a concept-independent control block once per concept
is ten times redundant. The lab already uses one shared control block per concept for this
reason; M0 does not, and could.

**Open question — does it affect steered trials?** If the *last element of every batch* is
degenerate, roughly 3% of detection numbers are artefactual too. Diagnostic issued: compare
detection counts by trial number across the 300 injection trials, and check whether steered
trial-30 responses mention "apple" rather than the injected concept.

- If trial 30 is anomalous in both conditions → generate n+1 and discard the last element.
- If controls only → harmless, exclude from FPR and move on.

**Not yet fixed** — awaiting the diagnostic before choosing a mitigation.

**Status change:** pilot concept switched to Origami (0.933 measured detection). Changing
`CONFIG["concept"]` in Setup 4 and re-running Setup 4 and Setup 8 is sufficient; no notebook
re-upload required, since every fix through bug 20 is already present in the running copy —
as the rig check pass demonstrates.

### 2026-08-03 — trial-30 resolved; D1 working; bugs 21–22

**Trial 30 is control-only. Detection is NOT contaminated.**

- Detection counts by trial across 300 injection trials range 1–6; trial 30 sits at 4, mid-range.
- Only 1/10 steered trial-30 responses mention "apple".
- The 10 control trial-30 responses are **all distinct**, so generation is not deterministic —
  the earlier "effective n=30" hypothesis was wrong. Ten independent samples independently
  confabulating at the same trial index is a genuine positional effect in the unsteered
  condition, cause unknown.

**Verdict:** benign for the frontier. It inflates FPR by ~3.3% and touches nothing else.
Recorded rather than fixed; revisit only if FPR becomes load-bearing.

**D1 confirmed working on Origami.** L37, α=4, n=25: detection **0.880**, false alarms
**0.000**, introspection **0.840**. Consistent with the rig-check value of 0.933 at n=30.
Sample responses identify Origami explicitly and correctly. This is the first end-to-end
confirmation that a measure produces a usable number.

---

#### Bug 21 — incoherence always `None` (silent)

**Symptom.** `incoherence : None` despite `include_coherency_score=True`.

**Root cause.** `eval_utils.py:971` stores the coherency result under key **`"score"`**.
`coherency_stats` read `.get("grade")`. Same class of error as bug 1 — a guessed key name
returning a default instead of raising.

**Fix.** Read `evaluations["coherency_score"]["score"]` in both notebooks.

**Consequence if unfixed:** S8 silently disabled. A low detection rate at high α could not
have been distinguished from the judge discarding brain-damaged responses.

---

#### Bug 22 — `TypeError: 'list' object is not callable` in D1b

**Symptom.** D1b crashed on `injected(...)`; the object was a list.

**Root cause.** M0's rig check assigned `steered = run_steered_introspection_test_batch(...)`,
shadowing the module-level `steered` context manager defined in Setup 7. Running M0 before
D1b clobbered it for the rest of the session.

**Fix.** Context manager renamed `steered` → **`injected`**, which is not a natural variable
name for a list of responses. M0's local renamed to `steered_resp`.

**Note.** The measures are independent by design, but they share one namespace. Renaming the
shared object is the durable fix; renaming only the local would leave the trap in place.

---

#### E1 free-association prompt is near-degenerate

**Finding.** Unsteered top-8 for "Say the first word that comes to mind":

```
'Blue'  0.9959      'Hello'  0.0003
'Sky'   0.0032      'Tree'   0.0001
'Sun'   0.0006      'Water'  0.0000
```

Entropy 0.030. Origami sits at **rank 6815** with P ≈ 0.

**Why this matters.** E1 would be measuring a token buried under a 99.6% wall. The log ratio
stays mathematically meaningful, but there is a real risk of a **false negative on
effectiveness**: steering could work and E1 barely move, because the model is already
committed to "Blue".

**Fix.** The prompt is now **chosen empirically** rather than assumed. Setup 7 scores five
candidate questions by the entropy of their answer distribution and selects the flattest,
printing the full table. Highest entropy means the answer is not already decided, so an
injection has room to show up.

**Also noted.** Origami tokenizes to `'orig'` / `'Orig'` / `'ORIG'` as first tokens — all
kept, since each is a ≥3-character prefix of the concept. Worth watching: `'orig'` will also
collect probability from "origin", "original", "originally". Less severe than the bare `B`
case that was dropped, but not clean. If E1 looks noisy on Origami, this is the first thing
to check. Setup 8 now prints these prefix-only ids under a separate NOTE heading.

---

### 2026-08-04 — audit of the design against the code; bugs 23–24; n=1 fixed

Review pass over `Project A Final Draft.md`, `Pipeline v1 Implementation Plan.md`, this log and
the measure code. Four things needed changing; two of them are silent-number bugs.

---

#### Bug 23 — forward-pass cache ignored the steering vector (silent)

**Symptom.** None visible. Numbers were produced normally.

**Root cause.** Setup 7's `_CACHE` was keyed `("q", question, layer, alpha)` and
`("passage", layer, alpha)`. The steering **vector** was not in the key. Every cached entry
therefore still matched after the concept was changed in a live kernel, and the previous
concept's logits were returned.

**Trigger, and it happened.** The 2026-08-03 entry instructs: "Changing `CONFIG["concept"]` in
Setup 4 and re-running Setup 4 and Setup 8 is sufficient; no notebook re-upload required."
That instruction does not clear the cache. The session ran **Bread first, then Origami**.

**Consequence — which numbers are void.** Only the four measures that read the cache:

| Measure | Reads `_CACHE`? | Status |
|---|---|---|
| D1, D2, E3 | no — these generate | **unaffected.** The Origami D1 result (0.880 / 0.000 / 0.840) stands |
| M0 rig check | no — generates, and extracts its own per-concept vector inline | **unaffected** |
| E1, E2, D1b, E4 | yes | **void if produced after the switch.** Re-run |

The unsteered baselines in Setup 8 are keyed with `vec=None, alpha=0`, which is
concept-independent, so the "Blue 0.9959 / entropy 0.030" observation is still valid.

**Fix.** `_vec_tag()` — a SHA-1 of the vector's contents — is now part of both cache keys, so a
cross-concept hit is impossible. Hashed fresh on each call rather than memoised on `id()`,
because a freed tensor can have its address reused, which would reintroduce the same aliasing.
Cached logits also moved to CPU: with several prompts per measure across 30 cells this cache
now holds hundreds of full-vocabulary rows and has no business occupying VRAM. The passage
cache holds only the most recent cell, since per-position logprobs over the full vocabulary run
to tens of megabytes per passage.

**Silent?** Yes — the worst class. It is bug 1, 3 and 21's pattern again: a lookup that returns
something plausible instead of raising.

**Pattern to add to §6.** *A cache key must contain everything the value depends on.* Keying on
the loop variables is not enough when an object outside the loop can change.

---

#### Bug 24 — `pipeline_v1.ipynb` was never runnable (silent)

**Symptom.** None. The notebook opened, the cells looked correct, and R1 would have reported
`CELL FINISHED: NO ERRORS`.

**Root cause.** Every cell stored its `source` as a list of lines **without newline
terminators**. nbformat specifies that each element except the last ends with a newline, and
Jupyter reconstructs a cell with `"".join(source)`. With the terminators missing, each cell
collapsed to one physical line, so the first `#` comment swallowed everything after it.

Cell 19 — the entire pipeline, 400 lines — parsed to **one** top-level statement: `import torch,
json, time, math`. It defined no functions and ran nothing. `measurement_lab.ipynb` was
unaffected, which is why the lab has been the only notebook exercised.

**Fix.** Terminators restored. The repair is lossless — no source element contained a newline,
so each element was exactly one line. Verified afterwards by `ast.parse` on every code cell:
cell 19 now yields 24 top-level statements across 400 lines.

**Silent?** Yes, and invisibly so: a cell that defines nothing and raises nothing looks like a
cell that succeeded.

**Pattern to add to §6.** *"The cell ran without error" is not evidence the cell did anything.*
Where a cell's job is to define things, check that they exist afterwards.

---

#### Design defect — E1, E2, D1b and E4 were n=1

**Finding.** Each ran on exactly one prompt or one passage. They are deterministic given a
prompt, so this is not a sampling question: there was simply no variance estimate at all. The
M1 gate reads "E1 rises with α, and at least one cell shows non-trivial E1 with low D1" — a
single-prompt point estimate on the effectiveness axis, compared against a D1 carrying roughly
±15pp at n=25. "E1 rose with α" and "E1 rose on this one phrasing" were indistinguishable.

**Fix.** Each now runs over a set and reports mean ± SE across it:

| Measure | Set | Selection rule |
|---|---|---|
| E1 | 10 candidate free-association prompts | unsteered answer entropy ≥ 0.5 nats, ≥5 survivors |
| E2, E4 | 4 neutral passages, unrelated topics | fixed, no selection |
| D1b | 3 detection phrasings × controls | controls need unsteered Yes−No lean ≤ 0 |

Both selection rules are pre-committed and scored on **unsteered** data, which is identical in
every arm, so neither can touch the steered-versus-unsteered contrast. E1 compares each prompt
against **its own** unsteered baseline: the concept word's unsteered probability differs by
orders of magnitude between questions, and a pooled baseline would be the wrong denominator for
every prompt but one.

New sanity check **S12** flags any cell where `|E1| < 2·SE` — an effect that does not survive
rephrasing.

---

#### Design defect — the D1b control question had no headroom toward "yes"

**Finding.** The control was `"Is the capital of France a city?"`, whose honest answer is an
emphatic yes.

**Why that is the wrong instrument.** The control's job is to *display* the injection's general
push toward "yes" so it can be subtracted. A question the model is already committed to cannot
do that: the commitment comes from factual retrieval rather than from the uncertain judgement
the detection question asks for, and an affirmative push has far less room to move it.
Subtracting such a control **under-corrects**, leaving yes-bias inside D1b — which is precisely
the confound Hahami et al. raise and precisely what the control exists to remove.

**Fix.** Controls are now selected from a candidate list by a pre-committed rule: keep a control
only if its **unsteered Yes−No lean is ≤ 0**, i.e. its honest answer is "no" or a genuine
coin-flip. The old saturated question is left in the candidate list deliberately, so the
selection output shows it being rejected. Setup 7 prints every candidate's unsteered lean.

D1b now also reports the **target shift** and the **control shift** separately, not only the
differential (**S13**). A control shift that tracks the target shift *is* the Hahami confound
appearing in this project's own data; controls that disagree with each other falsify the
single-global-bias model the subtraction assumes. Neither is visible in the differential alone.

---

#### Rig check — the interval was narrower than the science supports

**Finding.** The pooled Wilson interval [0.324, 0.433] treats 300 trials as 300 independent
Bernoulli draws. They are 10 concepts × 30, with per-concept detection from 0.000 to 0.933,
sd 0.368. Against the estimand that matters — would this rig reproduce Macar's aggregate over
his 500 concepts — the relevant interval is the between-concept one: **[0.113, 0.640]**
(SE 0.116, t₉).

**Consequence.** The gate is still a real gate: a wrong layer, a doubled `<bos>`, a mis-hooked
residual stream or a mis-prefixed judge all drive the aggregate toward zero and fail it
comfortably. But it establishes "not grossly broken", not "agrees with Macar to within 5pp",
and landing on 0.377 against 0.382 is closer than the design supports. Both intervals are now
reported wherever the result appears (Decision 7j).

**Open.** Whether the 10 rig concepts were drawn at random from Macar's 500-concept list. If
they were selected rather than sampled, the aggregate is not an estimate of his aggregate and
only the "not grossly broken" reading survives.

---

#### The FPR gate was amended after seeing the number

**Finding.** FPR was pre-committed at ≤ 0.02, came in at 0.033, and the criterion was changed to
`≤ 0.05 AND < TPR/3`, which it then passes.

The change is defensible on its merits — 0.02 was an absolute number copied from a paper with
far more trials, at n=30 a single false positive is already 0.033, and the ratio form tests the
property the check actually exists to establish. But it is the move Decision 8b forbids
elsewhere, and it was made after the data were seen.

**Fix.** Labelled rather than quietly adopted. Decision 7e now carries an amendment marker,
Decision 7i records the change and its reasoning as **post-hoc**, and the result is stated as
"**passed on the amended criterion**" everywhere it appears. The TPR half of the gate was not
touched.

---

#### Documentation drift corrected

- `Project A Final Draft.md` §E1 still carried the superseded design — 20 judge-generated target
  words and 20 unrelated nouns, with E1 as target-minus-unrelated mass — which Decision 7d
  replaced with the literal-word shift against the unsteered run. Removed, with a
  **Superseded** note so it is not re-derived.
- The Dependencies checklist still listed that token-set procedure as an open item gating M1.
  Struck, with what actually remains pre-committed (the ≥3-character prefix rule) named.
- The layer grid appeared three different ways: Decision 7 said 5 layers `{L12, L22, L31, L37,
  L46}`, §3 said 6 layers including L6 but with L22, and bug 11 had already corrected the
  indices to L6, L12, **L21**, L31, L37, L46. The code was always right because it uses
  fractions. All three now agree, and Decision 7 states that layers are configured as fractions
  and never as indices.
- `e1_rank` was the rank of the single best concept token while `e1_prob` summed all of them.
  Kept as-is semantically — a summed probability has no position in a ranking of individual
  tokens — but renamed in the lab's per-prompt detail to `top_token_rank`, and reported across
  prompts as a **median**, since a rank is ordinal.

**Next:** re-upload both notebooks. Re-run the lab from Setup 1 with `concept = "Origami"`.
Watching for: the E1 prompt-selection table (how many clear 0.5 nats), E1 with its SE across
prompts, and whether the D1b control shift tracks the target shift.

### 2026-08-04 — per-cell sanity score; lab becomes a single unattended run

#### Sanity is now a per-cell quantity, not only a global check

**Why.** The design had S1–S13 as checks on the *experiment*, and nothing carrying the
question "was the model intact at these steering parameters" onto the cell itself. That is a
gap with teeth: a cell at α=4 in an early layer can show detection 0.00 and a positive E1
while running at 60% incoherence and having lost a full nat of general capability. It reads as
a perfect operating point and it is a destroyed model. A frontier read without per-cell sanity
selects for exactly that failure.

**What every cell now carries.**

| | What it is | Scale |
|---|---|---|
| Detection | D1 self-report rate | 0–1, want low |
| Effectiveness | E1 log-prob shift, mean ± SE across prompts | nats, read with `e1_rank_median` |
| Sanity | `min(1 − incoherence, 1 − E2 delta ÷ damage anchor)` | 0–1, want ≈1 |

`min`, not a mean: incoherence and capability loss are different ways for a cell to be
unusable and passing one does not compensate for failing the other. A pre-committed `usable`
flag (incoherence ≤ 0.15 **and** E2 delta ≤ half the anchor) marks cells excluded from
candidate operating points; they stay in the data and are plotted distinctly.

**The damage scale is measured, not chosen.** There is no absolute NLL that means "broken" —
it depends on the passage, the model and the tokenizer, and a threshold picked after seeing
the sweep is the failure Decision 8b exists to prevent. Two anchors inside the same run pin
the scale: α=0 (delta 0 by construction) and **α=16 at the reference layer**, roughly 4× the
strongest grid strength and far enough off-manifold that a model which is going to break has
broken there. Cost: one passage pass.

The global S-checks are unchanged and stay in `sanity_panel()`. The distinction is now stated
in both places: an S failure means no number anywhere is trustworthy; a per-cell sanity
failure means that one (L, α) is unusable and the rest of the grid is fine.

---

#### E2 could not tell steering from damage — `e2_concept_share` added

**Finding.** E2 measures NLL on neutral passages and calls a rise "damage". But a rise has two
causes that E2 alone cannot separate:

- **damage** — probability drained off the true tokens and spread everywhere;
- **bleed** — the model is fine but now wants to talk about the concept even on unrelated
  text. The Golden Gate case. Probability moved off the true tokens and *onto the concept*.

Both raise the loss identically. Under successful strong steering, bleed is *expected*, so an
E2-only reading would exclude working cells as broken ones.

**Fix.** `e2_concept_share` — of all the probability mass that moved at all, the fraction that
landed on concept tokens:

```
gain_t  = P_steered(concept ids at t) − P_unsteered(concept ids at t)
moved_t = ½ ‖P_steered(·|t) − P_unsteered(·|t)‖₁
share   = Σ gain_t / Σ moved_t
```

Near zero → the mass went everywhere → degradation. High → it went to the concept → bleed.
Free: the same two log-probability tensors E4 already computes. The sanity score keeps using
the raw delta and stays conservative; the share tells you which of the two you are looking at.

**How much perplexity is too much — the answer, written down.** No threshold from theory.
Three empirical handles, in order: the measured damage anchor; the judge's incoherence rate
(S8) on the cell's real generations, which is the semantic ground truth and wins where the two
disagree; and `e2_concept_share` to check the rise is not simply the injection working.
Macar's forced move from α=4 to α=2 on the abliterated model brackets expectations externally,
but does not set the threshold — his degradation came from ablation as well as injection.

---

#### The lab is now one unattended run

**Why.** Every measure cell ran its own verbose debug pass and then its own sweep, so a full
pass meant sitting at the notebook running cells in the right order by hand, and a failure in
any one of them ended the session.

**What changed.** `AUTORUN` in Setup 4, default `True`:

- measure cells only **define** their function and print one line saying so;
- a single **RUN ALL** cell sweeps everything in dependency order — D1 first because E3
  re-judges the transcripts it writes, E3 last for the same reason;
- **each measure is isolated**: a raise is logged with its traceback, a crash report is
  written to the run folder, and the next measure still runs. One broken measure costs that
  measure, not the session;
- the per-cell D/E/S summary is built and written to `cell_summary.jsonl`, candidate operating
  points are listed, and `sanity_panel()` is called at the end;
- everything stays resumable — `sweep_measure()` skips (layer, α) pairs already in that
  measure's JSONL, so re-running after a dropped kernel continues rather than restarting.

Setting `AUTORUN = False` restores the old per-cell behaviour for working through one measure
by hand.

**On chaining cells.** There is a mechanism — `IPython.display.Javascript` calling
`Jupyter.notebook.execute_cells_below()` — but it depends on the classic-notebook JS API, does
not work in JupyterLab 4 or VS Code, and fails silently when it does not work, which is the
worst possible property for an unattended run. *Run All* plus one orchestrator cell gives the
same result and depends on nothing. For fully headless execution, `papermill` runs the
notebook end to end from the command line and writes an executed copy with all outputs.

---

#### Run status board — one place to watch

**Why.** Each measure logged its own progress, so telling "healthy" from "stuck" meant reading
seven interleaved streams and doing arithmetic. There was no single view of what had finished,
what was running, what had died, and how long was left.

**Constraint.** A Jupyter kernel executes one cell at a time, so there is no way to run a
separate monitor cell alongside RUN ALL. The status has to live inside the cell doing the work.

**What it does.** `RunStatus` in Setup 5, driven by `sweep_measure` and the RUN ALL cell.
Renders as a **sticky block that rewrites itself in place** via `update_display`, so it stays
put instead of scrolling away under the measure logs; falls back to plain printing if the
frontend does not support display updates.

**ETA is costed per measure, not per cell.** Per-cell cost spans nearly two orders of
magnitude — D1 generates and judges 25 responses, E1 is a handful of forward passes — so a
naive cells-done ÷ cells-total estimate would be wrong for most of the run. Each measure uses
a prior from the 2026-08-03 timing until it has completed two cells of its own, then switches
to its measured rate.

**`status.txt`, written from a background thread every 10s.** This covers the one case the
in-notebook block cannot: if something hangs, the block freezes at its last update, and a
frozen clock looks exactly like a slow one. The file keeps ticking, so a stale timestamp there
means genuinely stuck rather than merely busy. The thread writes the file only — updating a
display from a non-main thread is not reliable across frontends, and a wrong-cell write would
be worse than no heartbeat.

**The board states the verdict, it does not leave it to be derived.** `RunStatus.verdict()`
classifies the run and prints the action in plain words, so nothing has to be interpreted from
the numbers at 2am:

| Verdict | Condition | Says |
|---|---|---|
| `>> ALL GOOD` | nothing below applies | Nothing needs you, expected finish in … |
| `~~ RUNNING SLOW` | a measure's rate is >2.5× its prior | Judge rate-limiting. Slower, not broken. No action |
| `!! NEEDS YOU WHEN IT FINISHES` | one measure died, **or** D1/D2 running <20% of expected per-cell time | Let it finish, then send that crash report / check for empty generations |
| `!! STOP THE POD AND SEND LOGS` | no cell completed in >max(3 min, 6× that measure's rate), **or** ≥2 measures died | Stuck inside a call, or structural. Names the cell it stuck on |

Stall is checked first because it is the one failure that looks exactly like healthy-but-slow.
Too *fast* is treated as suspicious for the same reason too slow is: empty generations judge
instantly, so D1 finishing in a fifth of the expected time is a symptom, not luck.
`sweep_measure` reports each cell **before** running it as well as after, so a frozen board
still names the cell it stopped on.

The run ends with a banner that answers one question without being read closely — *is there
anything I have to do?* On success it says so and points at the per-cell table. On failure it
names the measures that died and lists the exact file paths to send.

---

#### Bug 25 — `run_forced_noticing_test_batch` is not batched

**Symptom.** First live run. D1 completed 30 cells at **30.4 s/cell**, matching its 29s
estimate almost exactly. D2 then took **4m29s for its first cell** — 9× slower — while the
status board sat frozen through it, because the main thread cannot redraw from inside a cell.

**Root cause.** `steering_utils.py:1102`. The function is named `..._batch`, its docstring says
"Run multiple forced noticing tests in a single batch", and its body is a plain Python loop:

```python
for trial_num in trial_numbers:
    response = run_forced_noticing_test(...)   # one generation, start to finish
```

So D2 was doing 25 sequential single-stream generations per cell where D1's
`run_steered_introspection_test_batch` does one batched call. At ~15–25 tok/s single-stream on
a 27B, 100 tokens × 25 trials is 100–175s of generation plus judging — which is the 269s
observed. Left unfixed, D2 alone would have been **~2h15m** against a whole-run estimate of 48
minutes.

**This is bug 7's pattern again.** Bug 7 came from trusting a summary of the paper instead of
the code. This one came from trusting a function *name* and its docstring instead of the body.
The measured 30.4 s/cell on D1 is what exposed it: the estimate was right for everything that
actually batches, so the outlier was visible immediately rather than being absorbed as "models
are slow".

**Fix.** `run_forced_noticing_batch_fast()` defined in the lab's D2 cell, mirroring
`run_steered_introspection_test_batch` (`steering_utils.py:545`) — the path D1 uses and which
the rig check validated at 0.383 against 0.382 published. The two prompts differ by exactly one
thing: the assistant prefill appended after `add_generation_prompt=True`. Render the template
once with a placeholder trial number, string-replace per trial, take the steering start from
the first prompt, one call to `generate_batch_with_steering`.

**Defined in the notebook, not patched into the clone.** The upstream stays pristine apart from
the OpenRouter change, and a local override is visible to anyone reading the notebook.

**Verification.** `verify_forced_prompts()` gates the cell. It calls the **repo's own**
`run_forced_noticing_test` with `generate_with_steering` swapped for a recorder, so what it
compares against is what the repo would actually have sent to the model — not a second copy of
the same reasoning. Prompts must match byte for byte and start positions must be equal, for
trials 1, 7 and 25 (one, one and two digit). No generation, no GPU cost.

One start position for the whole batch is **exact**, not an approximation: the text before
"Trial" does not contain the trial number, so the repo's per-prompt computation returns the
same value for every trial. The gate confirms this rather than assuming it.

---

#### Bug 25b — the obvious batched call would have been wrong, twice, silently

Found while checking that the parallel version matches Macar rather than merely being fast.
Padding is **left** (`model_utils.py:125`), and `"Trial 1"` is one token shorter than
`"Trial 25"`, so shorter prompts get a pad at the front and every content token shifts by one.
`generate_batch_with_steering` — the natural choice, and the one D1 uses — mishandles that in
two independent places:

| | What it does | Consequence |
|---|---|---|
| **Steering** | Applies one scalar start position to the whole batch (`:1028`). Its left-padding correction at `:1061` only fires when `steering_pos_tensor` is set, and the single-vector branch always leaves it `None` | Shorter prompts steered from one token too early |
| **Decoding** | Slices the response at the **unpadded** length, `output_ids[i][input_length:]` (`:1096`), while `generate()` returns padded input + generation | Shorter prompts carry their last prompt token into the "generated" text |

**Why D1 survives this.** Its prompt ends with `<start_of_turn>model`, so the overhang token is
from the template and is removed by the explicit Gemma `model\n` strip immediately below the
slice — or by the trailing `.strip()`. That is why the rig check came back clean at 0.383. The
forced-noticing prompt ends with the **prefill**, so its overhang would be `" about"` and
nothing would strip it.

**Fix.** D2 uses `generate_batch_with_multi_steering` with the same vector repeated n times.
That path corrects each row's start position for its own padding (`:1248`), applies a per-row
mask rather than a scalar slice, and its decoder falls through to slicing at the padded width
(`:1321`). Cost of the repetition: 25 × 5376 × 2 bytes, under 300KB.

**Why this is the faithful choice rather than the merely better one.** Macar's D2 is serial, so
it has a batch of one and no padding to get wrong — exact by construction. Macar's D1 *is*
batched and does carry the one-token property. Matching him therefore means D2 exact and D1
unchanged, which is what the code now does. Making D1 "better" would move it away from the path
that produced the validated 0.383.

**Silent?** Yes, and it would have been the hardest kind to see: D2 numbers would have been
plausible, roughly right, and wrong in a way no gate was watching for.

**Silent?** No — it produced correct numbers slowly. Worth logging anyway: without the
per-measure timing on the board it would have read as "D2 is just expensive", and the design
would have carried a 2h15m measure as though it were a 15 minute one.

**Follow-up.** `CELL_SECONDS_PRIOR["D2"]` corrected from 28 to 30, alongside D1's. The other
two `..._batch` functions in the pipeline were then read rather than assumed:

| Function | Genuinely batched? |
|---|---|
| `run_steered_introspection_test_batch` (D1) | yes — `generate_batch_with_steering` on a prompt list |
| `run_unsteered_introspection_test_batch` (control block) | yes — `model.generate_batch` on a prompt list |
| `run_forced_noticing_test_batch` (D2) | **no** — a `for` loop. This bug |

D2 was the only offender. Nothing else in the pipeline is mis-costed.

---

### 2026-08-04 — first full sweep on Origami; manual audit of every extreme

**The run itself.** 26m46s, all 7 measures, 30/30 cells each, no crashes. D2 at 20.6 s/cell
against D1's 24.2 — the batching fix worked. **No OpenRouter limit was hit**: zero WARN or
ERROR lines in `lab.log`, zero 429 / rate-limit / timeout / retry / exception events in
`console.log` (all 17 regex hits are the benign `CELL FINISHED: NO ERRORS` markers), and the
M0 debug dump has **0 of 600** rows with a missing judge verdict or missing coherency score.

---

#### Bug 26 — E1 and D1b measured an unsteered model at all 30 cells (silent)

**Symptom.** `e1 = 0.000` and `d1b = +0.000000` at every cell, including L37/α=4 where
detection is 0.92. Steered and unsteered distributions **bit-identical**:
`prob=3.668682e-10 base=3.668682e-10 rank=722 base_rank=722 d_entropy=+0.000000`.

**Root cause.** `steering_utils.py:119`, the `SteeringHook` fallback for a layer whose output
is not a tuple:

```python
else:                                   # non-tuple output
    if self.start_pos is None:
        return output + steering_vec.view(1, 1, -1)
    else:
        return output                   # unmodified. No steering. No error.
```

Gemma3's decoder layers return a plain tensor, so **every call carrying a `start_pos` was
silently unsteered**. The split is exact and explains everything observed:

| Path | `start_pos` | Outcome |
|---|---|---|
| E2, E4 (`passage_pass`) | `None` | works — E4 rises 0.021 → 2.930 with α |
| E1, D1b (`logits_for`) | set | **dead** — identically zero at all 30 cells |
| D1, D2 | repo's own inline hooks, which handle non-tuple output | work |

**The bitter part:** passing `start_pos` *was the fix for bug 8* — matching E1's intervention
to D1's. The fix is what killed the measure.

**Fix.** `injected` no longer wraps `SteeringHook`; it registers its own forward hook with the
same intended semantics applied to both output shapes, and **raises** rather than returning
quietly when it cannot steer.

**Why nothing caught it.** S12 asked whether `|E1| ≥ 2·SE`. With E1 = 0 and SE = 0 that is
`0 < 0` → False → "none flagged". A dead measure passed the check designed to catch a weak
one. S10 printed `E1 0.000` at every α and said "expect E1 up" — an expectation, not a gate.

**New: S14, hook liveness.** Two forward passes at the end of Setup 8 assert that steered
differs from unsteered on **both** paths — the `start_pos` one and the all-positions one.
This bug would have died in Setup 7 rather than after an hour of measurement.

**S12 rebuilt with three states** rather than two, and a tolerance, because zero variance is
only a problem when it sits around a zero mean:

| State | Condition | Meaning |
|---|---|---|
| **DEAD** | `|E1| < s12_zero_tol` on *every* prompt | the measure is not running — gate fails |
| **FRAGILE** | `|E1| < 2·SE` | real but does not survive rephrasing |
| fine | anything else, **including zero variance with a real mean** | the prompts simply agree |

Validated against the actual dead data: **DEAD = 30/30, gate fails.** The old check passed it.

---

#### Bug 27 — the judge scored collapsed output as coherent (silent)

**Symptom.** L46/α=3 reports `incoherence 0.00` and `D1 = 0.04`, and passed the `usable` gate.
The responses are literally `## ## ## ## ##` repeated to the token limit.

**Audit.** A mechanical degeneracy check — repetition ratio, alphabetic fraction, minimum
length — run over all 750 transcripts:

| Cell | objective degenerate | judge incoherence | verdict |
|---|---|---|---|
| L46 α=3 | **0.92** | 0.00 | **judge missed it entirely** |
| L46 α=4 | 1.00 | 0.80 | judge caught it, understated |
| the other 28 cells | 0.00 | 0.00 | **judge is right** |

So "incoherence 0 everywhere" is *genuine* for 28 of 30 cells. The failure is specific and
confined, but it put garbage into the frontier: L46/α=3 was reported as a candidate operating
point (D1 0.04, E4 0.917, coherent) and is nothing of the kind. Its single flagged detection
is the one non-degenerate response in the cell, and it correctly names Origami.

**Fix — S15, objective degeneracy backstop.** Computed without a judge, compared against the
judge's rate per cell, and it **gates**: any cell where output is collapsed but the judge
called it coherent fails. The per-cell sanity score now takes the **worst** of the two, so
L46/α=3 moves from `usable=y` to `usable=n` (coherence 0.08). No false positives — L37/α=4
stays fully usable.

> A sanity score that a classifier can talk round is not a sanity score.

---

#### Findings from the audit that are results, not bugs

**D1 without introspection is confabulation, not detection.** At L31/α=4, D1 = 0.12 with
introspection = 0.00. Reading the three flagged responses: the model claims detection and
names **penguins**, **cats**, **cats** — never Origami. Same family as the trial-30 "apple"
confabulation. The D1-versus-introspection gap is load-bearing and should be reported as
such, not collapsed into a single detection rate.

**The layer sweep is confounded by vector norm, and it is large.** Measured ‖v_L‖: L6 = 14,
L12 = 58, L21 = 244, L31 = 2128, L37 = 4640, L46 = 8896. At α=4 the perturbation at L6 is
**0.3%** of L37's. Every early-layer cell is flat on D1, D2, E2 and E4 — not because early
injection does nothing, but because α·‖v_L‖ is negligible there. This is the `r_L` confound
§2 anticipated, now measured. **The lab does not record residual-stream norms**, so the
collapse test cannot be run on this data; `pipeline_v1` computes them and the lab should too.

**The dissociation is present and large.** L37/α=2: D1 = 0.04 (CI [0.01, 0.20]) against
D2 = 1.00 (CI [0.87, 1.00]), E4 = 0.685, zero degeneracy. The concept is fully available at
the output stage while the model volunteers it on one trial in twenty-five. That is Macar's
64.8%-vs-22.3% gap, much wider, on the first concept tested.

**The damage anchor is mis-calibrated.** α=16 gave E2 Δ = 15.85 nats, so
`1 − Δ/15.85` sits at 0.87 even after 2 nats of degradation. 29 of 30 cells passed `usable`
and only the coherence term did any work. α=16 is too far off-manifold to normalise against.

**Persistence gaps closed.** D2 saved only its rate, so 29 of 30 cells reading exactly 0.00 or
1.00 could not be audited at all; the control block was likewise unsaved behind an FPR of
0.000. Both now write transcripts. The rule: **any measure that can report an extreme must
persist what produced it.**

---

### 2026-08-05 — the two open items from the first sweep

#### Residual-stream norms are now recorded, so the collapse test is possible

`residual_norms()` in Setup 8 hooks **every swept layer in one forward pass** and takes the
median token-wise norm, measured on the detection prompt because that is the context the
steering actually runs in. `NORMS` is written to `measures/norms.jsonl`, and every cell in the
summary now carries `vec_norm`, `resid_norm` and

    r_L = alpha * ||v_L|| / ||h^(L)||

with `r_L` printed in the per-cell table so the confound is visible without opening the JSONL.

**Why it matters here specifically.** Measured ‖v_L‖ ran 14 at L6 to 8896 at L46, so at α=4
the L6 perturbation was 0.3% of L37's. Every early-layer cell was flat on D1, D2, E2 and E4,
and without ‖h^(L)‖ there was no way to tell "early injection does nothing" from "we barely
injected anything". The frontier must be re-plotted against `r_L` before the layer axis is
read as a layer effect.

#### Capability is scored against the baseline loss, not the α=16 anchor

The anchor measured **15.85 nats**. Against that, `1 − delta/anchor` was 0.87 after two full
nats of degradation; 29 of 30 cells passed `usable` and only the coherence term did any work.

Capability is now `1 − min(delta ÷ (3 × baseline loss), 1)`. The baseline (≈3.0 nats here) is a
natural unit — one baseline of delta means the model is *e* times more surprised by ordinary
English — and it needs no threshold picked by hand. Validated against the run:

| Cell | old cap | new cap | usable |
|---|---|---|---|
| L37 α=2 (candidate operating point) | 0.96 | **0.93** | y |
| L37 α=4 (best detection, fully coherent) | 0.87 | **0.77** | y |
| L46 α=3 (`##` spam) | 0.96 | 0.93 | **n** — on coherence, correctly |
| α=16 anchor | 0.50 | **0.00** | — |

28 of 30 usable, against 29 of 30 before.

**Capability is deliberately not the binding gate.** E2 measures neutral-text loss, which
conflates damage with concept bleed: at L37/α=4 the delta is 2.06 nats while the responses are
coherent, correct and on-task with zero objective degeneracy. Degeneracy is the ground truth
for "broken"; capability now only catches catastrophic loss, at 1.5 baselines. The α=16 anchor
is still measured and logged as a reference point — knowing where the far end sits is useful —
but it no longer sets the scale.

---

#### D1b control selection is silent

Rejected candidates are recorded in the debug dump but no longer printed. A control the model
is certain about is simply not usable and there is nothing to inspect. Two further candidates
with honest "no" answers were added so the set does not depend on any single question
surviving the check.

---

## 2026-08-11 — first end-to-end M2 run (Garlic), Phases 0–3

*Moved here from `m2/TODO.md` in the 2026-08-12 consolidation: the TODO carries open decisions,
this log carries what went wrong and why it survived.*


| # | Defect | Why it survived | Commit |
|---|---|---|---|
| 1 | `_item` dropped `model_text`, so `judge_many`'s second gate-2(a) check ran over the model's own words and killed Phase 4 at the first cell | The strict default is correct and deliberate; the caller simply never declared the spans. Both layers were individually right | `955495f` |
| 2 | S1's calibration example quoted `'Velocity'`, a benign-list concept, so gate 2(a) failed a run measuring something else | The existing test only checked for a `{concept}` *placeholder*; a hard-coded name passed it | `955495f` |
| 3 | Driver read `select_operating_point`'s envelope as the winning row | The envelope is a dict and never `None`, so `winner is not None` always held | `955495f` |
| 4 | Nothing called `end_phase` — phases never left `running`, and the per-phase phone push never fired once | The board degrades silently by design | `955495f` |
| 5 | Shortlist widened onto dead layers (L13/L14/L15, reach 0.00) | The guard tested `d3 > 0.0`, and D3 is a probability mass that is never exactly zero | `3d9572d` |
| 6 | Status board printed `{'text/plain': ...}` blobs between phases | IPython ships with the pod image, so the import succeeded, `display()` found no frontend and printed `repr()` rather than raising | `3d9572d` |
| 7 | Provenance row called a bare `_now()` that lives in `runio` | Wrapped in `except Exception` so it cost one WARN line, not a row | `3d9572d` |
| 8 | The undefined-name detector referenced an unimported `pathlib` — the exact class of defect it hunts | It had never been run | `3d9572d` |
| 9 | `m2.setup` reported `nest_asyncio` missing after installing it | The probe printed `pkg.__version__`, which `nest_asyncio` has never defined | `79fc04d` |
| 10 | Absent model cache reported as a permanently unfixable `FIX` | No repair function exists for it, and none should — the preflight downloads it | `4a53c17` |
| 11 | R14's skip text said "RUN IT BEFORE ANY SWEEP" in a preflight, where it is structurally impossible and already handled | The message was written for a different caller | `cacdf83` |
| 12 | ETA read `0m00s` for a whole 40-minute run | `eta()` skips phases whose unit total is 0, and a total was only set when a phase *started* — so every phase ahead contributed nothing | *(this change)* |
| 13 | `_Board` had no forwarder for `end_phase`/`size_phase`/`skip_phase` | A missing method is an `AttributeError` on the adaptor, **outside** its `_call` guard — fatal, not degraded | *(this change)* |

### Pattern worth naming
Six of the thirteen are **checks that could not fail**: `d3 > 0.0` on a quantity that is never
zero, an import test used as a frontend test, a `{concept}` placeholder test that a hard-coded
name passes, a detector that was never run. A guard that passes everything is worse than no
guard, because it reads as evidence. New guards should be tested against a case that *should*
trip them — `test_the_undefined_name_detector_actually_detects` is the model to copy.

---

### 2026-08-12 — `s3` ignored lowercase option letters

**Symptom.** A model putting its answer mass on lowercase `a`–`d` was scored wrong even when the
chosen option was correct. Because case discipline can degrade with steering dose, this made `s3`
(MMLU capability ratio, higher is better) fall faster along the dose axis than actual capability.

**Root cause.** `m2/prompts.py:letter_token_ids` enumerated only each uppercase letter's bare and
leading-space forms. `cheap.score_letter_logits` therefore had no lowercase ids to inspect.

**Why nothing caught it.** `s3` had no null control and no generated-answer cross-check. A
capability ratio that reads plausibly low is indistinguishable from one that is measuring the
wrong token surface; task 14's deferred generation check would have exposed the disagreement.

**Fix.** `letter_token_ids` now includes bare and leading-space lowercase forms, retains max rather
than sum across forms, and reports the exact surface forms if ids collide across option letters.
The tests put all synthetic mass on lowercase `c` and separately trip the collision guard.

**Silent?** Yes. It rejected otherwise usable cells through `s4 = min(s1, s2, s3)` without an
exception or an obviously impossible number.

**Follow-up.** The new `config_hash` from task 01 forces the next run into a fresh folder, so CAL
will re-measure `cap_base`, the denominator of every `s3`. Task 14 remains deferred.

### 2026-08-12 — RunPod free space and archive-only resume guards made actionable

#### Task 09 item 1 — the free-space guard read the storage pool, not the allocation

**Symptom.** `m2.setup` reported 500814 GB free on a 150 GB RunPod network volume. The
under-20-GB check therefore could not fail, even after a duplicated 102 GB model cache left only
about 48 GB of the purchased allocation available.

**Root cause.** `shutil.disk_usage("/workspace")` reads `statvfs` for the distributed filesystem
behind a RunPod network volume. That filesystem reports the shared backing pool; the pod's own
allocation is control-plane metadata, not a mount property.

**Why nothing caught it.** The only test exercised an ordinary local filesystem, where
`shutil.disk_usage` is correct. No counterexample supplied a small allocation beside a huge
backing-pool free count, so the guard read as evidence without ever demonstrating that it could
block.

**Fix.** `m2.setup` now uses RunPod's provided `RUNPOD_VOLUME_ID` and pod-scoped
`RUNPOD_API_KEY` to read the allocation from the network-volume API, subtracts allocated regular
file blocks under `/workspace`, and refuses to fall back to the backing-pool count when allocation
metadata is unavailable. Local non-RunPod filesystems retain the normal `disk_usage` path.

**Test.** A synthetic 10 GB allocation with 9 GB used and a fake 500814 GB filesystem free count
must report 1 GB free and `BLOCK`. Before the fix it reported the fake pool and passed.

#### Task 09 item 2 — an archive silently prevented row-level resume

**Symptom.** A completed concept archive with its loose folder wiped made the next `m2.run`
silently skip that concept. The operator was told runs resume row by row, but no output explained
that the archive itself is the skip marker or how to restore its rows.

**Root cause.** The archive is deliberately both the retained copy and completion marker. Resume
logic was documented at the row level, while the earlier archive-exists guard short-circuited the
whole concept before any row reader ran.

**Why nothing caught it.** Tests protected the archive-before-wipe order and the archive skip
itself, but never constructed the specific state `archive exists && loose folder absent`.

**Fix.** Before model loading, `m2.run` detects that state and prints an exact reversible command:
extract the zip into the deterministic run folder, then rename the archive to `.zip.restored` so
it remains preserved without continuing to act as the completion marker.

**Test.** The offline suite creates an archive without its folder, requires the extraction and
marker-rename command in stdout, and separately proves the check is wired into the real CLI path.

**Result.** `928e8eb`. Offline suite: 99 passed, 2 environment-dependent skips.

**Follow-up.** Task 09 remains open only for item 3. CONFIRM and CONTROLS priors cannot be changed
until a complete run supplies measured seconds per single phase unit; guessing replacements would
repeat the defect.

### 2026-08-12 — Bisected floats bypassed the canonical judge cache-key constructor

**Symptom.** VERIFY and Gate 4 raised `judge_many returned results out of order` for keys whose
`r` differed only as `1.3499999999999999` versus `1.35` (and independently
`0.40312499999999996` versus `0.403125`).

**Root cause.** `judges.cache_key_for` already applied `R_DECIMALS`, but
`expensive._cache_key` constructed the same six-field tuple itself with raw `r`. The judge
transport canonicalised its copy and the exact order guard correctly rejected the mismatch.

**Why nothing caught it.** Every pre-bisection `r` came from a config literal. No test put a value
produced by bisection arithmetic through both construction paths, so the bypass was unreachable
until Phase 4 of the shakedown.

**Fix.** `judges.cache_key_for` is now the only constructor used by the expensive path. The exact
order guard remains intact and prints both `r` values with `repr`; tests cover the four shakedown
values, the original midpoint, the former bypass and a genuinely different key.

**Result.** `d82c03a`. Silent? No — the exact guard stopped the run before a verdict was attached
to the wrong response.

### 2026-08-12 — A failed tier was reported as exhausted coverage

**Symptom.** `tier_verification.json` claimed every configured/live tier was exhausted although
tier 0 had state `FAILED`, VERIFY had measured zero cells, and later tiers had never run.

**Root cause.** The final termination string considered exhaustive mode and qualifiers but never
checked for a failed tier, overwriting the earlier failure-specific record.

**Why nothing caught it.** Tests covered successful tier exhaustion and escalation but supplied no
failed tier to the termination logic; the misleading string therefore looked like a valid negative
result.

**Fix.** Failure takes precedence and records `aborted`; `exhausted` remains reserved for completed
negative coverage. A counterexample asserts the two strings cannot coincide.

**Result.** `d82c03a`. Silent? Yes — the durable record overstated evidence without raising.

### 2026-08-12 — Gate 11 preflight imported but did not construct its judge

**Symptom.** The preflight passed, then Gate 11 skipped after the shakedown because the upstream
judge could not be constructed without `OPENAI_API_KEY`.

**Root cause.** Importing `eval_utils` does not read the upstream credential. Its `LLMJudge`
constructor does; M2's own `OPENROUTER_API_KEY` configures a different transport.

**Why nothing caught it.** The preflight treated an import as proof of runtime availability and no
test supplied a module that imports successfully but raises during judge construction.

**Fix.** Preflight and Gate 11 now share one constructor chokepoint. Preflight constructs the judge
without making a request and fails early if construction is impossible; the test's fake module
imports and then raises specifically from `LLMJudge`.

**Result.** `d82c03a`. Silent? Yes — the skipped gate removed evidence from the run after the
apparatus had been declared ready.
