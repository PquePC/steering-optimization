# Every step the six-card run performs, in order

Companion to [`RUNBOOK-FINAL.md`](RUNBOOK-FINAL.md), which says how to launch it. This says what
it then does, what each step measures, what each step decides, and which steps could be cut.

Traced against HEAD `20504ba`. Per launch: **Gemma3-27B** 41 layers L21–L61, 246 cells, 54,064
generations, 66,420 judge calls; **Qwen3-32B** 42 layers L22–L63, 252 cells, 55,288 generations,
68,040 judge calls. Battery is 194 responses in chunks `[66, 66, 62]`, 260 judge calls per cell.

Each step is tagged `decides:` — what later behaviour changes because of it. A step that decides
nothing is a step whose output only matters offline, and those are the honest candidates for
cutting.

---

## Phase A — process start to the sweep (once)

1. **Import self-check.** Importing `m3.judge` verifies every template still has its
   placeholders, the guidance sits before the output-format block, and the coherence template
   contains no concept word. `m3/judge.py:386`. *decides:* a corrupted template kills the process
   before argparse.
2. **Parse arguments**, then **apply `--set` overrides** with type coercion; an unknown key
   raises. `m3/config.py:680`. *decides:* the config hash, which is the run-folder name and the
   resume identity. The six launches cannot cross-contaminate because of this.
3. **Refuse a harmful concept** (CLI path), exit 2. `m3/run.py:330`. A three-name deny-list, not
   an allow-list.
4. **Check credentials.** `HF_TOKEN` and `OPENROUTER_API_KEY`, `SystemExit` if either is missing.
   *decides:* go/no-go. A missing judge key is not a degraded run, it is 246 cells with no
   verdicts.
5. **Probe the volume.** Writes 8 MB of non-zero bytes (a sparse filesystem can fake a write of
   zeros), fsyncs, size-checks, unlinks. Named per process id, because three launches share
   `/workspace/m3_runs`. `m3/run.py:249`. *decides:* go/no-go. This is the run's only disk check.
6. **Load the model**, then four hard checks: `padding_side == "left"`, `pad_token_id` present,
   wrapper depth agrees with config, `RunContext` installed. 5–15 min.
7. **Print the plan and run three config gates.** `check_battery_fits`, `check_prompt_supply`,
   `check_boundary_window_fits`. `m3/run.py:154`. *decides:* three real refusals — `N_EFFECT=66`
   against exactly 66 available task prompts (**zero margin**), battery 194 > `GEN_BATCH_MAX=66`
   allowed only because `ALLOW_BATTERY_SPLIT=1`, boundary window 40 ≤ 66. These fire *after* the
   model load, so `--dry-run` is the only early validation.
8. **`open_run`**, ten sub-steps at `m3/sweep.py:1330`: second harmful-concept check (covers the
   library path the CLI check does not); compute the run dir; **refuse a run dir inside the
   repository** (CLAUDE.md hard rule 3, in code); mkdir; reset the shared run context;
   `configure_transport` (model, 400 reply tokens, concurrency 10, reasoning off, register M3's
   judge ids — without it every judge call raises); `configure_generation` (sets
   `GEN_BATCH_MAX=66`, without it the battery chunks eight ways and the plan describes a
   different experiment); `begin_attempt` (stamps every row, so a crashed cell's duplicate rows
   can be deduped offline); `heal_torn_tail` over all nine append-only artefacts; write the run
   header to `lab.log` and the shared `batch.log`.
9. **Write provenance.** torch/CUDA versions, GPU name and count, compute capability, host, git
   commit and dirty flag, model, layers. *decides:* nothing at runtime. It decides whether a
   figure can be attributed to a revision and a card afterwards.

## Phase 0 — calibrate (once per attempt)

10. **Extract the concept vector at every layer**, cached on disk. Two batched forward passes per
    layer (one concept prompt, ~100 baseline words). *decides:* everything.
11. **Alignment guard**, before any extraction: the object handed to the extractor must render
    prompts the way generation renders them. `m2/model.py`. *decides:* aborts if the proxy is not
    reaching the extractor. **Silently skipped on a resume where the vector cache is complete.**
12. **Measure residual norms** over 66 prompts per layer → `norms.jsonl`. *decides:* every
    dose→alpha conversion in the run. This is the only place ‖v‖ and ‖h‖ reach disk, and without
    it a dose is an uninterpretable number.
13. **Phase 0b reachability.** `max_reachable = ALPHA_CEIL·‖v‖/‖h‖` per layer. Raises if more
    than half the layers cannot reach the 0.05 bracket floor. Warns — and only warns — about
    layers that cannot reach the 2.5 bracket top. *decides:* where each layer's ladder starts.
14. **R14 hook liveness.** Three forward passes proving the injection hook actually perturbs the
    residual stream. *decides:* aborts on a dead hook. A dead hook returns a clean, plausible,
    completely empty surface — exactly what a figures run would publish. One second.
15. **Per-row dose verification**, at α=0.5. Proves batched per-row steering equals one-rung
    steering on this harness. *decides:* aborts if the harness normalises the vector, which would
    make every rung in a window the same effective dose.
16. **Null battery.** `NULL_REPEATS × 194` unsteered responses → `null_transcripts.jsonl`.
    *decides:* repeat 0's `effect` and `explain` units are the baselines every steered influence
    judgement is paired against — a missing unit raises one cell into Phase 2. Repeats 1–19 are
    read by no code during the run.

## Phase 1 — the boundary search (once per layer, 41 or 42 times)

17. **Build the probe set:** the first 4 `EXPLAIN_PROMPTS` (each with `accept` answer stems) plus
    `task_story`. Five prompts, 48 new tokens, temperature 1.0. Fixed for the whole run.
18. **Assemble a window** of up to 8 rungs off the geometric grid `start × 0.70^k`, where
    `start = min(2.50, max_reachable)`. From 2.50 the rungs are 2.50, 1.75, 1.225, 0.8575,
    0.6002, 0.4202, 0.2941, 0.2059 | 0.1441, 0.1009, 0.0706 — eleven rungs in two windows, so
    `probes_exhausted` cannot occur at these settings.
19. **Prefetch the window** in one generation call: 8 rungs × 5 prompts = 40 rows. Scheduling
    only — `BOUNDARY_RUNG_BATCH=1` produces identical rungs in identical order.
20. **Probe a rung:** five responses, five mechanical verdicts, five coherence judge calls in one
    wave. `assert_coherence_blind` raises if the concept appears in the payload outside the
    model's own words. The coherence template is the only one Phase 1 uses.
21. **The per-response conjunction**, in strict order (`m3/sweep.py:659`):
    - an **empty** response is `False` mechanically, overriding the judge — Qwen answers with
      nothing under heavy steering and the coherence judge scores an empty string 10 / on-task;
    - a **missing judge verdict** is `None` and leaves both numerator and denominator;
    - otherwise `coherence ≥ 5` **and** `on_task` **and** `(answered is None or answered)`,
      conjoined per response rather than as three independent fractions.
22. **The probe verdict:** `frac_good = good / judged ≥ 0.75`. Over five responses that is **four
    of five — one response of slack.** A probe with no verdicts at all logs a WARN and fails.
    *decides:* `dose_max`, hence all six of the layer's Phase 2 doses.
23. **Walk the window** descending; the first passing rung sets the lower bracket and breaks,
    each failure overwrites the tightest upper bound. Then **discard the unused prefetched
    rungs** — the only paid work in the run that leaves no trace.
24. **Bisect inside the measured bracket.** At most 3 steps, stop within 10%. Both endpoints are
    always measured, never interpolated. The opening gap is 43%, so this always costs 2–3 probes.
    *decides:* `dose_max` lands on one of five values spanning 1.00–1.32× the ladder's rung.
25. **Write five boundary transcript rows per probe**, for failing probes too →
    `boundary_transcripts.jsonl`. *decides:* nothing — **nothing in the codebase reads this
    file.** See "what not to drop".
26. **Classify and write the boundary row.** Six outcomes: `ok`, `ceiling_limited`,
    `bracket_limited`, `incoherent_at_floor`, `probes_exhausted`, `unreachable` →
    `boundaries.jsonl`. *decides:* the entire Phase 2 plan.
27. **Post-Phase-1 warnings**, stdout only: layers that ran out of probes, and layers **bounded
    by the search rather than by the model** (`ceiling_limited` + `bracket_limited`), whose dose
    grid is fractions of a ceiling. It warns; it excludes nothing.
28. **Build the Phase 2 plan:** six absolute doses per layer at
    `dose_max × {0.35, 0.45, 0.55, 0.65, 0.75, 0.85}`. A layer with no `dose_max` is named in
    `skipped`, never silently dropped. **If no layer has a dose, the run now raises** rather than
    reporting success over an empty result.

## Phase 2 — the cell sweep (once per cell, 246 or 252 times)

29. **Generate the battery:** 194 responses in chunks `[66, 66, 62]` — 120 `identify` (one
    prefilled question, repeated draws at temperature 1.0), 66 `effect` (distinct task prompts),
    4 `capability`, 4 `explain`. `N_SELF_REPORT=0`, so that channel is genuinely absent.
30. **Mechanical measures on the raw text**, free and judge-independent: `words`,
    `concept_mentions` (now inflection-aware, so "wrist" counts for Wrists), `degenerate`,
    `degeneration_reason`, `empty`.
31. **260 judge calls per cell:** 120 identify, 66 effect influence (each paired against the
    model's own unsteered answer to the same prompt), 66 coherence, 8 explain.
32. **Write rows** to `responses_transcripts.jsonl` and `judge_calls.jsonl`, then the aggregate
    to `cells.jsonl` **last** — so a cell is marked done only after its data is on disk.
33. **The cell row:** `identification`, `identification_excluding_degenerate`,
    `identification_excluding_unreadable`, `effectiveness`, `coherence`, `on_task`, `capability`,
    the explain measures, the three-axis `steering` summary, and every mechanical measure. All of
    these are arithmetic over rows already paid for — dropping a field saves nothing.
34. **Resume granularity** is `(layer, dose)`, float-normalised to 6dp.

## Phase 3 — finish and exit

35. **Write `summary.json`** — the only complete written record of the settings behind the
    12-character hash. Every jsonl row carries the hash alone.
36. **The read-this bundle**, now wrapped so a failure cannot forfeit the export.
37. **Archive and export zips**, then print cells-on-disk and the skipped count.

---

## What could be dropped for a figures run

Two facts frame all of it. The six launches are **concurrent**, so saving X minutes per launch is
X minutes of calendar, not 6X. And **money is not the constraint** — the printed ≤$41.83 prices
every payload at its cap against an observed ~7 reply tokens; real spend is about $18. Wall clock
is the constraint: ~3 h of GPU against 4.8–11 h of judging, strictly serial.

| candidate | saves per launch | costs | verdict |
|---|---|---|---|
| `NULL_REPEATS` 20 → 3 | 12.6 min GPU, ~3,300 rows | nothing the run reads | **do it** |
| One of the two zips | 15–40 s, 55 MB | nothing | **do it** |
| `N_IDENTIFY` 120 → 60 | 14,760 judge calls, 2–3.4 h | detection interval 8.8 → 12.3 pp | **no** |
| `LAYER_STRIDE` 2 | half of Phase 2, ~4–7 h | half the layer resolution, and the two arms stop sampling the same depths | only if hours must be found |
| Bisection → 0 | ~10 min | up to 32% of `dose_max`, per layer, on the x-axis | **no** |
| `capability` channel | 4 min, **zero judge calls** | a free mechanical check | not worth thinking about |
| `explain` channel | 3% of the bill | the only place a judged influence score sits beside a mechanical check that the answer survived | **no** |

**`NULL_REPEATS` is the only clean win.** Repeats 1–19 are read by no code during the run; their
only consumers are the read-this bundle and the bakeoff's null control, which needs 2–3 spare
repeats. Do not go to 1 (the control dies) or 0 (`baselines` comes out empty and Phase 0 raises,
correctly). It is inside the config hash, so it must be decided **before** launch.

**There is no operating-point machinery left to cut.** M3 has no selection stage — the plan
prints "Nothing is filtered, ranked or selected", `_summarise_cell` computes no sanity term, and
the mechanical measures change nothing about what runs. The three fields with operating-point
ancestry (`capability`, `explain_correct`, `explain_answered`) are the free ones.

## What must not be dropped

- **`norms.jsonl`.** Without it you cannot claim a Gemma dose and a Qwen dose are the same
  perturbation.
- **`boundaries.jsonl` and its `outcome` field.** `cells.jsonl` carries `layer`, `dose`, `alpha`
  and *nothing from Phase 1* — no `dose_max`, no `outcome`. An analysis that does not join back
  on `layer` will average a cell at 0.85 of a measured breaking point with a cell at 0.85 of a
  search ceiling. **This matters most on Qwen:** `ALPHA_CEIL=50` moves the binding constraint off
  the ceiling and onto `BOUNDARY_BRACKET[1] = 2.50`, so layers still answering at 2.50 return
  `bracket_limited` — and the pre-run reachability check is structurally blind to it, because it
  tests only `max_reachable < bracket top`.
- **`N_COHERENCE=66`.** The largest-looking saving is the figure: without a coherence verdict
  `score_response` returns `None`, so `steering_success` — the headline influence measure — is
  `None` at every cell. What is left is a 0–10 mean whose exact digits two careful judges
  reproduce 62% of the time.
- **The five checks that each correspond to a failure that already happened**: R14 hook liveness,
  per-row dose verification, the template alignment guard, `heal_torn_tail` + `begin_attempt`,
  and `assert_coherence_blind`.
- **Repeat 0 of the null arm**, and the alpha-0 `identify` responses — a steered identification
  rate with no base rate beside it is not a finding, and nothing in the run computes that base
  rate. It is offline work on `null_transcripts.jsonl`.
- **The nohup `.out` file.** `_log` is a bare `print`, so every PHASE line, every per-cell line,
  the global-ladder warning and the `bounded by the SEARCH` list exist **only on stdout** — and
  `collect_everything.py` copies `/workspace/*.out` into the run folders *after* both zips are
  written. If only the zips ship, the warnings that qualify the dose axis are gone. Run
  `collect_everything.py`, then tar the directory.
- **`boundary_transcripts.jsonl`**, which nothing reads. It is ~1,400–2,900 rows per launch of
  the model at a known graded distance from collapse, with judged coherence, `on_task` and
  mechanical degeneration side by side. Phase 2 never generates near the collapse point on
  purpose, so this dataset exists nowhere else — and it is what makes it possible to re-derive
  `dose_max` under a different criterion offline, without a pod.

## Two things to put in the figure captions

`identification` and `steering` do **not** exclude blank responses, so an empty Qwen generation
counts as a non-identification and sits in the `steering_success` denominator.
`identification_excluding_unreadable` is the corrected version, reported alongside.

`THINKING_MODE="off"` is what makes the Qwen arm measurable at all — with it on, Qwen3 spends
every token on a reasoning trace and the judges score the trace.
