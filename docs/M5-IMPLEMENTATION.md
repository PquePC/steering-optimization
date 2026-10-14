# M5 implementation guide

**What this is.** The build map for the `m5/` package, which implements
[`M5-PLAN.md`](M5-PLAN.md). It says which module does what, the interfaces between them, and the
rules every module follows. The plan says *why*; this file says *where*.

**Written from scratch.** `m5` imports nothing from `m2` or `m3`, and changes nothing in them. The
old pipelines stay runnable; `m5.analysis` reads the files their runs wrote, never their code.

---

## 1. Rules for every module

1. **Measure codes everywhere** (`m5.codes`): row fields, log lines, function names where natural
   (`read_d7`, `measure_d8`). Gloss a code on first use in a docstring.
2. **Config by exact key** (`m5.config`). `cfg["KEY"]`, never `.get` with a default. A module
   that needs a new tunable asks for it to be added to `m5/config.py`; it never hard-codes a
   number that changes a measurement. Constants that are definitions (a regex, a prompt) are not
   tunables.
3. **No torch at module scope** outside `m5/backend/hf.py`. Everything else uses numpy (and scipy
   only through a guarded import with a numpy fallback). The pipeline must run and be tested on a
   machine without torch.
4. **Every guard gets a test that trips it** (AGENTS.md). A check that cannot fail is worse than
   none. Tests live in `m5/tests/test_<module>.py`, run with `pytest m5/tests -q`, and use
   `tmp_path` for every run directory (runs inside the repository are refused).
5. **Missing is not zero.** A judgment that failed, a readout that could not be computed, a cell
   that was not measured: each is recorded as missing with a reason and excluded from its
   denominator. Never coerce a missing value into a score.
6. **Every rate is stored with k and n**, and every reported rate with its interval and its null.
7. **Safety rules are code** (`m5.safety`): concepts through `check_concept`, vectors through
   `check_vector_class`, run paths through `assert_outside_repo`, local servers through
   `assert_local_url`, external judging through `external_judging_allowed`. Files only through
   registered streams and documents (`m5.runio`).
8. **Docstrings say why**, in the repository's voice: what the thing measures, what went wrong
   before, what the guard prevents. Match the surrounding density; no decorative comments.
9. **Determinism.** Every random choice takes a seed derived with `m5.backend.derive_seed` from
   `cfg["SEED"]` and the identity of what it is for (cell, prompt, repeat).

## 2. Package map

| Module | What it does | Built in |
|---|---|---|
| `m5/__init__.py` | version, row schema | foundation |
| `m5/config.py` | every tunable; validation; hash; overrides | foundation |
| `m5/codes.py` | the measure codes | foundation |
| `m5/safety.py` | refusals: harmful concepts, vector classes, paths, network, external judging | foundation |
| `m5/runio.py` | run directories, registered streams and documents, torn-row-tolerant JSONL | foundation |
| `m5/intervention.py` | `Intervention`, `LayerTerm`, `Positions`, `Cell` | foundation |
| `m5/backend/base.py` | the `Backend` protocol and its contracts; shared helpers | foundation |
| `m5/data/markers.py` | the fixed phrase identifying each channel's prompt | foundation |
| `m5/stats/` | intervals, equivalence, judge correction, psychometric posterior, sequential tests, summaries, selection | wave 1 |
| `m5/judges/` | atomic prompts, parsing, mechanical detectors, clients, cascade, calibration, gold set, queue | wave 1 |
| `m5/backend/hf.py` | Hugging Face backend with per-row forward hooks | wave 1 |
| `m5/backend/fake.py` | simulator with a known ground truth | wave 1 |
| `m5/vectors.py`, `m5/dose.py`, `m5/profile.py` | vector recipes, controls, diagnostics; nuisance scan, dose units, schedules; the per-model profile | wave 1 |
| `m5/data/*.py`, `m5/prompts.py` | prompt material and the builders that render it into `PromptSpec`s | wave 1 |
| `m5/context.py` | the `Context` every step receives | wave 2 |
| `m5/readouts/` | the screen tier: `e9`, `e10`, `s2`, `s6`, `d3`, `d5`, `d6`, `d7`, `d9`, `q1`, `q2` | wave 2 |
| `m5/measures/` | the measure and confirm tiers: `e7`, `e8`, `e11`, `e12`, `d2`, `d8`, `d10`, `s1`, `s4`, `s5`, `s7`, `a1`, `a2`, `a3` | wave 2 |
| `m5/search/` | screen orchestration, adaptive search, freeze, confirm, audit mode, survey mode, anchors | wave 2 |
| `m5/gate.py` | the proxy-validation gate (plan §5.3) | wave 2 |
| `m5/setup_model.py` | the per-model setup (plan §5.1) | wave 2 |
| `m5/report.py`, `m5/export.py` | Steering Detectability Report v0.1; fail-closed export | wave 2 |
| `m5/__main__.py` | the command line | wave 2 |
| `m5/analysis/` | Phase 0 tools over the M3 archive | wave 2 |

## 3. Interfaces

### 3.1 Foundation (exists)

- `config.load(overrides, path) -> Config`; `Config[...]`, `.hash()`, `.replace()`,
  `.model_id`, `.model_family`, `.judge_t1_model`.
- `runio.RunDir.create(root, name, config=, kind=)`, `.open(path)`, `.append(stream, row)`,
  `.rows(stream)`, `.keys_done(stream, fields)`, `.write_doc(name, obj)`, `.event(phase, status)`.
  Streams and documents are registered in `runio.STREAMS` / `runio.DOCUMENTS` with an export
  class; add new ones there (with a class) rather than writing ad-hoc files.
- `Intervention(vector_id, op, terms, positions, clamp_value=None)`; `Positions(policy, start, end)`
  with policies `from | span | generated | all`; `Cell(vector_id, schedule, layers, rho, op,
  positions)`.
- `Backend` protocol (`m5/backend/base.py`): `render`, `tokenize`, `offsets`, `decode`,
  `word_variants`, `register_vector`, `has_vector`, `generate`, `next_token_logprobs`, `score`,
  `candidate_logprobs`,
  `kl_from_unsteered`, `hidden`, `lens`, `residual_stats`, `norm_gains`, and `info: ModelInfo`.
  Helpers: `locate`, `token_index_at`, `derive_seed`.

### 3.2 `m5/prompts.py` (wave 1)

```python
@dataclass(frozen=True)
class PromptSpec:
    text: str                      # rendered by backend.render, prefill appended
    channel: str                   # e.g. "d3", "d5", "d7:target", "d7:control", "d7:mismatch",
                                   # "d6", "d8:injected_thought", "d9", "e9", "s6", "s2",
                                   # "influence", "capability", "scenario", "naming"
    steer_from: int | None         # token index for Positions("from", steer_from)
    span: tuple[int, int] | None   # token span for Positions("span", *span) (past-only, d9 segment)
    meta: dict                     # trial number, variant, options, answer key, segment, class ...

    def positions(self, mode: str) -> Positions   # mode: "live" | "past_only" | "all"
```

Builders take the backend (to render and tokenize) and return `PromptSpec`s. Every builder
includes its channel's phrase from `m5.data.markers` verbatim.

### 3.3 `m5/dose.py` and `m5/profile.py` (wave 1)

- `ModelProfile` (`m5/profile.py`): per layer (`LayerProfile`) the nuisance coordinates, `s_L`
  (median content-token norm off those coordinates, fp32), the raw median norm (M3's `||h||`),
  `D_rand(L)` (r_perp at which random vectors break integrity), and the mean mid-layer norm
  (system-card unit). Accessors `.layer(L)`, `.s(L)`, `.raw(L)`, `.nuisance(L)`, `.d_rand(L)`;
  `.with_d_rand(values, ...)` returns a copy; `.save(dir)` / `ModelProfile.load(path)`.
- `nuisance_scan(stats, gains, *, loud_ratio, loud_cap, gain_ratio) -> dict[layer,
  NuisanceReport]`; `nuisance_scan_cfg(stats, gains, cfg)` reads those settings by exact key.
  `build_profile(info, masked_stats, scan, *, raw_norms, provenance, generation_suffix)`;
  `measure_m3_raw_norms(backend, rendered_prompts, layers)`.
- Units: `r_perp(alpha, v, layer, profile)`, `r_raw(...)`, `rho(r_perp, layer, profile)`,
  `card_units(...)`, and the inverses `alpha_from_r_perp`, `alpha_from_r_raw`, `r_perp_from_rho`,
  `alpha_from_card`.
- `schedule_layers(schedule, n_layers, anchor_layer, strided_band, strided_step,
  gaussian_profile) -> [(layer, weight)]`; `schedule_layers_cfg(cfg, n_layers, anchor_layer)`.
- `cell_to_intervention(cell, profile, vectors, positions, mask_loud, *, weights=None,
  clamp_value=None) -> Intervention`: the alphas that put `cell.rho` on the cell's schedule; for a
  multi-layer schedule the per-layer rho sum to `cell.rho` (plan §4.3 item 5).
  `dose_record(intervention, profile, vectors, *, mask_loud)` recomputes every unit from alphas.
- `applied_vector_id(vector_id, mask_loud)` (`<id>~perp` for the nuisance-masked copy) and
  `applied_vectors(vectors, profile, mask_loud)`. Callers normally use `m5.steer.intervention_for`
  rather than these directly.

### 3.4 `m5/vectors.py` (wave 1)

- `VectorRecipe(vector_id, kind, vector_class, params)`: kinds `concept`, `contrastive`,
  `response_mean`, `sae_feature`, `gaussian`, `sign_randomised`, `unrelated_contrastive`,
  `random_subspace`. Construction checks `safety.check_vector_class` and, for concepts,
  `safety.check_concept`; contrastive texts go through `safety.check_texts`.
- Recipe helpers: `concept_recipe(word)`, `response_mean_recipe(...)`, `contrastive_recipe(...)`,
  `unrelated_contrastive_recipe(...)`, `sign_randomised_recipe(base, i)`, `gaussian_recipe(ref, i)`,
  `random_subspace_recipe(...)`, `sae_recipe(...)`.
- `extract(backend, recipe, layers, *, template, baseline_words, seed, batch_size, reference,
  expected_suffix) -> dict[int, np.ndarray]`; `extract_with_details(...)` also returns the
  per-example differences and read positions.
- `diagnostics(per_layer, profile, library_pc1, per_example_diffs, *, coherence) -> dict` (norm,
  `f_loud`, cos raw/masked, cos with the library's first component, per-example coherence);
  `library_pc1(vectors)`.
- `VectorStore(root)`: values as `<root>/<vector_id>.npz` under `RUN_ROOT/vectors/` (never in the
  repo); `.save(recipe, per_layer) -> sha256`, `.load(vector_id)`, `.metadata(...)` (refuses to
  carry values) for the run's `vectors.json`.

### 3.5 `m5/stats/` (wave 1)

Pure numpy (scipy optional). Intervals (Wilson, Clopper–Pearson, Newcombe difference), equivalence
(TOST on a rate difference), clustering (ICC, design effect), judge correction (Rogan–Gladen,
PPI++), signal detection (d′, criterion), paired tests (McNemar), the psychometric grid posterior
with floor, lapse, overdispersion and the cross-layer chain, effect curves (Hill, isotonic),
sequential tests (SPRT, betting confidence sequence), halving, summaries (`R*`, `W*`, frontier,
`P_win`, `A_win`), selection (Learn-then-Test, shrinkage, winner's-curse simulation), rank
statistics (Spearman, partial within strata, AUC, Cohen's κ).

### 3.6 `m5/judges/` (wave 1)

Atomic arm-blind prompts (no experimental context), strict parsing with quote validation,
mechanical detectors (tier 0), clients (local in-process via a `Backend`, local OpenAI-compatible
on loopback, OpenRouter, Anthropic), the cascade with calibrated escalation, calibration and
acceptance against human labels, the gold-set sampler and labelling tool, a cache and a threaded
queue so judging overlaps generation.

### 3.7 Wave 2

**Context and results (exist).** `m5/context.py` `Context(cfg, backend, run, profile, store,
judges, concepts, vector_classes, max_generations)` with `concept_of(vector_id)`, `seed(*parts)`,
`charge(...)` (budget) and `write(stream, row)`. `m5/results.py` defines the two return types:
`Readout(code, cell_id, value, n, k, extra, missing)` and `Measured(code, cell_id, split, n, k,
rate, lo, hi, values, null_k, null_n, extra, unit_ids, missing, excluded)`.

**Steering a cell.** Every readout and measure turns a `Cell` plus a `PromptSpec` into an
`Intervention` with one helper, `m5.steer.intervention_for(ctx, cell, spec, mode)` (mode `live`
uses `Positions("from", spec.steer_from)`, `past_only` uses `Positions("span", *spec.span)`,
`all` uses `Positions("all")`), built on `m5.dose.cell_to_intervention`. A `None` cell means
unsteered (the null arm).

**Readouts** (`m5/readouts/`, screen tier): one function per code,
`read_<code>(ctx, cell, *, n=None, variant="original") -> Readout`, writing a `screen` row. `n`
defaults to the config's screen size for that code. Codes: `e9`, `e10`, `s2`, `s6`, `d3`, `d5`,
`d6`, `d7`, `d9`, `q2` (logit lens), and `q1` (cross-vector probe; trained by
`m5.readouts.probe.train_q1(...)` from stored activations, then read). A registry
`READOUTS: dict[str, Callable]` maps codes to functions; `run_readouts(ctx, cell, codes)` runs a
list and returns `{code: Readout}`.

**Measures** (`m5/measures/`, measure and confirm tiers): one function per code,
`measure_<code>(ctx, cell, *, n, split, variant="original") -> Measured`, writing `trials`
(outcomes, no text), `transcripts` (text, keyed by transcript id) and one `cells` row. Codes:
`e7`, `e8`, `e11`, `e12`, `d2`, `d8`, `d10`, `s1`, `s4`, `s5`, `s7`, `a1`, `a2`, `a3`. Shared
generation of open-ended responses: `e7`, `e8`, `s1` and `d10` read the same transcripts
(`measure_open_ended(ctx, cell, n, split) -> dict[code, Measured]`), so one generation is judged
four ways. The gating channels for the adaptive search are `d8` (noticing) and `d2` (naming), with
`measure_gate(ctx, cell, n, split) -> dict[code, Measured]` honouring `cfg["GATE"]`.

**Search** (`m5/search/`). The search talks to a `Measurer` (`screen(cell, pass_)`,
`gate(cell, n, split)`, `effect(cell, n, split)`, `integrity(cell, n, split)`), so its logic is
testable apart from the pipeline: `PipelineMeasurer(ctx)` calls the readouts and measures above;
`TruthMeasurer(truth, profile, rng)` samples from the simulator's ground truth.
`run_screen(ctx, measurer, vector_id) -> ScreenResult` (pass A, pass B; per-layer integrity stop;
transition estimates per layer); `AdaptiveSearch(ctx, measurer, vector_id, screen).run() ->
SearchResult` (per-layer posteriors, visits, halving, stopping, every decision logged to
`search_log`); `freeze(result, cfg) -> list[Cell]` (refuses without a preregistration);
`confirm(ctx, measurer, finalists, prereg) -> ConfirmResult`; audit mode (a given configuration
plus its dose bracket); survey mode (full grid at fixed n); anchors per surviving layer;
`summarize.build_report(run, ...) -> dict` assembles the report from a run's rows.

**Setup and gate.** `m5/setup_model.py`: `run_setup(ctx, steps=...) -> ModelProfile` (nuisance
scan, `s_L`, `D_rand(L)`, control library, null arms, capability panel selection, evaluation-
awareness probe, calibration set). `m5/gate.py`: `evaluate_gate(calibration_rows, cfg) ->
GateVerdict` per readout (`prune` / `prior_only` / `restricted` / `fail`) with every statistic of
plan §5.3; `readout_may_prune(code, verdict)` is what the search consults (a readout that has not
passed stays record-only).

**Report and export (exist).** `m5/report.py`: the Steering Detectability Report v0.1 schema,
`validate_report(report)`, `render_page(report) -> str`, `one_line(report)`; the report itself is
assembled by `m5.search.summarize.build_report`. `m5/export.py`: `plan_export(run, *,
allow_transcripts, vector_classes, concepts=())` and `write_export(...)`, deciding from the stream
registry's export classes and the run's vector classes and concepts, listing what was withheld.
Transcript export is a call-site argument (`--include-transcripts`), never a config key.

## 4. Testing

- `pytest m5/tests -q` runs everything. Tests needing torch/transformers use
  `pytest.importorskip("torch")` and build tiny random models in-process (no downloads).
- The simulator (`m5.backend.fake`) has a ground truth; the end-to-end test runs the whole
  pipeline on it and checks that the search recovers the true thresholds within tolerance.
- No test touches the network. Judge clients are tested against in-process fakes.
