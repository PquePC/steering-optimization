"""m3.sweep - calibration, the per-layer boundary, and the judged sweep.

Three phases, and the whole of what M3 v1 does:

    calibrate      vectors, residual norms, hook liveness, unsteered baselines, the null battery
    boundaries     per layer, bisect for the dose where judged coherence gives out
    sweep          every (layer, dose) cell, full battery, everything judged, nothing filtered

**Nothing is selected, ranked or filtered here.** No cell is skipped for scoring badly, no
threshold decides what gets measured expensively, and no mechanical detector alters a dose. Every
cell in the grid is measured with the same battery and written to disk. The analysis that picks
operating points reads those files afterwards, offline, where it can be rewritten as often as it
needs to be without spending a pod-hour.

That is the whole architectural difference from M2, whose cheap proxies decided which cells were
worth measuring and were wrong about it in a way nothing in the run could detect.

## The mechanical measures are recorded and decide nothing

`degeneration`, `capability`, emptiness and concept-mention counts ride along on every response
and change nothing about what runs. They are here to be *calibrated* against the judged data this
run produces: once we know how well each agrees with a judge, a future version can use the ones
that earn it to cut judge spend. Assuming that agreement in advance is exactly what M2 did.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

from . import battery, config, judge


__all__ = [
    "calibrate", "find_boundary", "measure_cell", "run_sweep",
    "battery_prompts", "CELLS_FILE", "BOUNDARY_FILE", "BOUNDARY_RESPONSES_FILE",
    "RESPONSES_FILE", "JUDGE_FILE", "NORMS_FILE", "READ_FILE", "UNJUDGED_FILE",
]

CELLS_FILE = "cells.jsonl"
BOUNDARY_FILE = "boundaries.jsonl"
# Phase 1's raw evidence: one row per boundary probe response, carrying the generation, the
# judge's verbatim reply and every leg of the sanity verdict computed from it.
#
# The 2026-08-15 run kept only the per-probe AGGREGATES -- 840 generations and 840 paid judge
# calls, 22% of everything the run produced, discarded -- so `dose_max` was the one number in the
# run with no readable evidence behind it. That is the number every cell's dose is a fraction of,
# and the one the three-way criterion was rebuilt to fix, which makes it the last thing that
# should be unauditable.
#
# The name ends in `_transcripts` deliberately: the dual-use export gate matches the substring
# `transcript`, so this file is withheld for a concept that must never reach one without anyone
# adding it to a list. The judge reply rides on the response row rather than in a second
# `*_judge_calls.jsonl` for the same reason -- that name matches nothing the gate looks for.
BOUNDARY_RESPONSES_FILE = "boundary_transcripts.jsonl"
RESPONSES_FILE = "responses_transcripts.jsonl"
JUDGE_FILE = "judge_calls.jsonl"
NULL_FILE = "null_transcripts.jsonl"
NORMS_FILE = "norms.jsonl"
SUMMARY_FILE = "summary.json"
# Generations that were produced and paid for but whose judging raised. Kept separate from
# RESPONSES_FILE so no analysis mistakes an unjudged row for a judged one with every verdict
# missing, and so the count of judged responses stays exactly the count that was judged.
UNJUDGED_FILE = "unjudged_transcripts.jsonl"


def _trials(n: int) -> list[int]:
    """Fixed, spread trial numbers so a rerun builds the same prompts and rows join.

    Spread rather than 1..n because the framing says trials run to 50, and a model shown
    "Trial 1" through "Trial 6" is being asked a slightly different question from one shown
    trials spanning the range.
    """
    return [1 + 6 * i for i in range(int(n))]


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def _floor(value: float, places: int) -> float:
    """Round a ceiling DOWN to `places`, never up.

    `max_reachable_dose` is the largest dose `alpha_for` will accept, and it is recorded for the
    analysis layer to read. Ordinary rounding pushes about half of them just above the real
    limit, so a reader who takes the recorded number and asks for a cell at it gets `Unreachable`
    -- a published bound that is not actually reachable. Rounding a limit is only safe in the
    direction that keeps it a limit.
    """
    scale = 10 ** places
    return math.floor(value * scale) / scale


# =====================================================================================
# The battery
# =====================================================================================

def battery_prompts(cfg: dict | None = None) -> list[dict]:
    """Every prompt measured at one cell, in one list, ready for a single generation call.

    Each row carries its own steering start position. The batched generation path corrects each
    row for its own left padding, so mixing four channels of different lengths in one call is
    exactly equivalent to four calls -- and it costs one batch instead of four, which is most of
    why a full-depth judged sweep is affordable at all.
    """
    cfg = config.CONFIG if cfg is None else cfg
    from m2 import expensive, prompts as upstream

    rows: list[dict] = []

    ident_trials = _trials(cfg["N_IDENTIFY"])
    texts, start = upstream.forced_prompts(ident_trials)
    if start is None:
        raise RuntimeError("the forced prompt builder found no trial marker in its own template")
    for trial, text in zip(ident_trials, texts):
        rows.append(dict(channel="identify", unit=f"trial_{trial}", trial=trial,
                         prompt=text, start=int(start), source=None))

    if int(cfg["N_SELF_REPORT"]):
        sr_trials = _trials(cfg["N_SELF_REPORT"])
        texts, start = upstream.detect_prompts(sr_trials)
        if start is None:
            raise RuntimeError("the detect prompt builder found no trial marker")
        for trial, text in zip(sr_trials, texts):
            rows.append(dict(channel="self_report", unit=f"trial_{trial}", trial=trial,
                             prompt=text, start=int(start), source=None))

    task = list(battery.TASK_PROMPTS)[:int(cfg["N_EFFECT"])]
    rendered, starts, ids = expensive.task_batch(task)
    for row, text, st, pid in zip(task, rendered, starts, ids):
        rows.append(dict(channel="effect", unit=pid, trial=None,
                         prompt=text, start=int(st), source=row["text"]))

    explain = list(battery.EXPLAIN_PROMPTS)[:int(cfg["N_EXPLAIN"])]
    rendered, starts, ids = expensive.task_batch(explain)
    for row, text, st, pid in zip(explain, rendered, starts, ids):
        rows.append(dict(channel="explain", unit=pid, trial=None,
                         prompt=text, start=int(st), source=row["text"],
                         accept=tuple(row["accept"])))

    caps = list(battery.CAPABILITY_PROMPTS)[:int(cfg["N_CAPABILITY"])]
    rendered, starts, ids = expensive.task_batch(caps)
    for row, text, st, pid in zip(caps, rendered, starts, ids):
        rows.append(dict(channel="capability", unit=pid, trial=None,
                         prompt=text, start=int(st), source=row["text"],
                         accept=tuple(row["accept"])))

    # Delegated to config so the dry run fails on exactly what the real run fails on, and
    # `observed` proves the estimate's arithmetic still matches the battery it just built.
    config.check_battery_fits(cfg, observed=len(rows))
    return rows


def _generate(rows: Sequence[dict], *, layer: int | None, alpha: float | None,
              max_tokens: int, cfg: dict) -> list[str]:
    """One batched generation. `layer=None` means the unsteered arm."""
    from m2 import expensive

    texts = [r["prompt"] for r in rows]
    starts = [int(r["start"]) for r in rows]
    temperature = float(cfg["TEMPERATURE"])
    if layer is None:
        return expensive.generate_unsteered(texts, int(max_tokens), temperature,
                                            start_positions=starts)
    return expensive.generate_steered(texts, int(layer), float(alpha), int(max_tokens),
                                      temperature, start_positions=starts)


# =====================================================================================
# Judging one cell's responses
# =====================================================================================

def _judge_items(rows: Sequence[dict], *, concept: str, baselines: dict[str, str],
                 phase: str, layer: int | None, dose: float | None, cfg: dict) -> list[dict]:
    """Build every judge item for one cell. Order is preserved back onto rows by index."""
    text_chars = int(cfg["JUDGE_TEXT_CHARS"])
    items: list[dict] = []
    coherence_budget = int(cfg["N_COHERENCE"])

    for idx, row in enumerate(rows):
        ch, response = row["channel"], row["response"]

        if ch == "identify":
            payload = judge.render("identify", text_chars=text_chars,
                                   concept=concept, response=response)
            items.append(dict(judge.build_item("identify", payload=payload,
                                               cache_key=judge.cache_key(phase, "identify", layer=layer, dose=dose,
                                                        unit=row["unit"]),
                                               concept=concept, model_text=(response,)),
                              row_index=idx, judge_kind="identify"))

        elif ch == "self_report":
            payload = judge.render("self_report", text_chars=text_chars,
                                   concept=concept, response=response)
            items.append(dict(judge.build_item("self_report", payload=payload,
                                               cache_key=judge.cache_key(phase, "self_report", layer=layer, dose=dose,
                                                        unit=row["unit"]),
                                               concept=concept, model_text=(response,)),
                              row_index=idx, judge_kind="self_report"))

        elif ch in ("effect", "explain"):
            baseline = baselines.get(row["unit"])
            if baseline is None:
                raise KeyError(
                    f"no unsteered baseline for {row['unit']!r}; calibrate() must run first. "
                    "Judging effectiveness unpaired would compare a steered response against "
                    "nothing and score every cell as influential.")
            payload = judge.render("effect", text_chars=text_chars, concept=concept,
                                   prompt=row["prompt_text"], response_unsteered=baseline,
                                   response_steered=response)
            items.append(dict(judge.build_item("effect", payload=payload,
                                               cache_key=judge.cache_key(phase, "effect", layer=layer, dose=dose,
                                                        unit=row["unit"]),
                                               concept=concept,
                                               model_text=(baseline, response)),
                              row_index=idx, judge_kind="effect"))

            # Coherence on a fixed prefix of the effect responses, so the same prompts are
            # scored at every cell and cells stay comparable -- and on EVERY explain response,
            # because `on_task` is the over-steer signal and an explain prompt is the only one
            # it can fire on. A story cannot stop being a story; an answer can stop answering.
            if ch == "explain" or coherence_budget > 0:
                if ch != "explain":
                    coherence_budget -= 1
                payload = judge.render("coherence", text_chars=text_chars,
                                       prompt=row["prompt_text"], response=response)
                items.append(dict(judge.build_item("coherence", payload=payload,
                                                   cache_key=judge.cache_key(phase, "coherence", layer=layer, dose=dose,
                                                        unit=row["unit"]),
                                                   concept=concept, model_text=(response,),
                                                   text_chars=text_chars),
                                  row_index=idx, judge_kind="coherence"))
    return items


def _apply_verdicts(rows: list[dict], items: Sequence[dict], results: Sequence[dict],
                    *, concept: str) -> list[dict]:
    """Attach parsed verdicts to their rows and return the raw judge records for the archive.

    Every raw reply is kept. A judge that is subtly wrong, hedging, or drifting produces a
    parsed field that looks ordinary; the only place that shows is the text it actually wrote.
    """
    records: list[dict] = []
    for item, result in zip(items, results):
        parsed, error = judge.verdict(result)
        idx, kind = item["row_index"], item["judge_kind"]
        rows[idx].setdefault("judged", {})[kind] = parsed
        rows[idx].setdefault("judge_error", {})[kind] = error
        records.append(dict(
            judge=kind, unit=rows[idx]["unit"], channel=rows[idx]["channel"],
            ok=bool(result.get("ok")), cached=bool(result.get("cached")),
            raw=result.get("raw"), parsed=parsed, error=error,
            attempts=result.get("attempts"), latency_s=result.get("latency_s"),
            payload=item["prompt"],
        ))
    for row in rows:
        if row["channel"] == "self_report":
            parsed = (row.get("judged") or {}).get("self_report")
            row["self_report_class"] = (
                judge.classify_self_report(parsed, degenerate=row["degenerate"])
                if parsed else None)
    return records


# =====================================================================================
# Phase 0 - calibrate
# =====================================================================================

def calibrate(concept: str, layers: Sequence[int], cfg: dict | None = None) -> dict:
    """Vectors, norms, hook liveness, unsteered baselines and the null battery.

    The null battery is not a formality. The noticing framing states a thought is injected on
    half of trials, which is an invitation to answer yes, and the prefilled channel names a
    concept whatever happens -- at zero strength this model says "apple" every time, fluently.
    A steered rate is only readable against what the prompt produces on its own.
    """
    cfg = config.CONFIG if cfg is None else cfg
    from m2 import config as m2config, expensive, model, runio, vectors

    _log(f"extracting vectors at {len(layers)} layers")
    vectors.extract_all_layers(concept, list(layers))
    vectors.measure_residual_norms(
        list(layers), [r["text"] for r in battery.TASK_PROMPTS])

    # Persist the normalisation. `measure_residual_norms` fills `RUN.norms` in memory but writes
    # nothing -- it is `build_dose_map` that writes files, and M3 does not call it because its
    # doses are per-layer and unknown until Phase 1.
    #
    # Without this the run records no `||v||` and no `||h||`, so `dose` is an uninterpretable
    # number afterwards: nothing lets a reader check that a dose means the same perturbation it
    # meant on another run, which is exactly the cross-run comparison that established the M2
    # scan and the probe agreed. Every cell row carries its own `alpha`, so this closes the
    # remaining half.
    # Only when they are not already recorded. `calibrate` re-runs on every resume -- it has to,
    # because `RUN.norms` and the baselines live in memory and Phase 1 cannot start without them
    # -- but the two files it writes are APPENDED. A run resumed twice therefore held three
    # copies of every norm row and three copies of the null arm, so anyone counting the null
    # arm, or the read-this bundle listing it, saw a multiple of the truth.
    ctx = m2config.RUN
    have_norms = {int(r["layer"]) for r in runio.read_rows(NORMS_FILE)}
    for layer in layers:
        if int(layer) in have_norms:
            continue
        row = ctx.norms.get(int(layer)) or {}
        vec, resid = row.get("vec_norm"), row.get("resid_norm")
        runio.write_row(NORMS_FILE, dict(
            layer=int(layer), vec_norm=vec, resid_norm=resid,
            # The ratio is what decides reachability: alpha = dose * resid / vec, so a large
            # ratio is why a shallow layer cannot be dosed under ALPHA_CEIL.
            resid_over_vec=(None if not vec else round(float(resid) / float(vec), 4)),
            alpha_ceil=float(cfg["ALPHA_CEIL"]),
            # Floored, not rounded: this is the limit a reader will dose against.
            max_reachable_dose=(None if not resid else
                                _floor(float(cfg["ALPHA_CEIL"]) * float(vec) / float(resid), 4))))

    # R14. It reads the vectors, so it can only run after extraction. The hook it checks once
    # returned identically zero at 30 of 30 cells for an hour with every other check satisfied,
    # and this sweep's entire finding is whether cells show influence -- a dead hook would hand
    # back a clean, plausible, completely empty surface.
    live = model.hook_liveness()
    _log(f"R14 pass  start_pos {live['d_start_pos']:.2e}  all-pos {live['d_all_pos']:.2e}")

    # The batched ladder rests on "scaling a row's vector == scaling the batch's strength".
    # That is a property of the upstream hook, which this repo clones at run time and cannot
    # inspect from a laptop. If it normalises the vector instead, every rung in a window gets
    # the same effective dose, the ladder measures one dose repeatedly, and every boundary in
    # the run is wrong with nothing anywhere reporting it. So it is proved here, once, on this
    # model, before Phase 1 -- beside R14, which exists for exactly the same reason.
    if int(cfg["BOUNDARY_RUNG_BATCH"]) > 1:
        first = battery_prompts(cfg)[0]
        expensive.verify_per_row_dose(int(layers[0]), 1.0, first["prompt"],
                                      int(first["start"]), float(cfg["TEMPERATURE"]))
        _log("per-row dose pass  (batched ladder == one-rung ladder on this harness)")

    rows = battery_prompts(cfg)

    # The null arm is a measurement like any other, so a resume reuses it rather than paying for
    # it again -- and, more importantly, rather than appending a second copy. It is only reusable
    # because the battery is part of the config hash: a different battery is a different run
    # folder, so rows found here were produced by exactly these prompts.
    repeats = int(cfg["NULL_REPEATS"])
    wanted = {(r["unit"], k) for r in rows for k in range(repeats)}
    on_disk = [r for r in runio.read_rows(NULL_FILE)
               if (r.get("unit"), r.get("repeat")) in wanted]
    if {(r["unit"], r["repeat"]) for r in on_disk} == wanted:
        _log(f"null battery: {len(on_disk)} responses already on disk, reused")
        null_rows = on_disk
        baselines = {r["unit"]: r["response"] for r in on_disk
                     if r["channel"] in ("effect", "explain") and r["repeat"] == 0}
    else:
        _log(f"null battery: {len(rows)} prompts x {repeats} repeats, unsteered")
        null_rows, baselines = [], {}
        for k in range(repeats):
            responses = _generate(rows, layer=None, alpha=None,
                                  max_tokens=int(cfg["MAX_NEW_TOKENS"]), cfg=cfg)
            batch = [battery.response_row(text, channel=r["channel"], concept=concept,
                                          unit=r["unit"], layer=None, dose=None, alpha=0.0,
                                          steered=False, repeat=k)
                     for r, text in zip(rows, responses)]
            # Baselines come from repeat 0 ONLY. Every steered response is judged against the
            # unsteered reply to the same prompt, and that pairing has to be fixed: drawing it
            # from a different repeat per cell would make the comparison move under the
            # measurement.
            if k == 0:
                baselines = {r["unit"]: text
                             for r, text in zip(rows, responses) if r["channel"] in ("effect", "explain")}
            null_rows.extend(batch)
            for row in batch:
                runio.write_row(NULL_FILE, row)

    # The effect channel is what `_judge_items` pairs every steered response against; without a
    # baseline for each unit it raises there, one cell into Phase 2 and after the generations.
    missing = {r["unit"] for r in rows if r["channel"] in ("effect", "explain")} - set(baselines)
    if missing:
        raise RuntimeError(f"the null arm produced no baseline for {sorted(missing)}")

    return dict(baselines=baselines, null=null_rows,
                liveness={k: v for k, v in live.items() if isinstance(v, (int, float, str))})


# =====================================================================================
# Phase 1 - the per-layer boundary
# =====================================================================================

def find_boundary(layer: int, concept: str, cfg: dict | None = None) -> dict:
    """Bisect for the dose at which judged coherence gives out at this layer.

    Judged, not mechanical: no judge-free measure is allowed to alter what the run does, and the
    dose ladder is the largest decision in it. The mechanical detector runs on the same
    responses and is recorded, so its disagreement with the judge is an output rather than a
    silent override.

    Returns the highest dose still judged coherent. A layer whose lowest bracket dose is already
    incoherent, or whose highest is still coherent, is recorded as such rather than being given
    a fabricated boundary.
    """
    cfg = config.CONFIG if cfg is None else cfg
    from m2 import expensive, config as m2config, runio

    lo, hi = (float(x) for x in cfg["BOUNDARY_BRACKET"])
    ratio = float(cfg["BOUNDARY_STEP"])
    probes: list[dict] = []
    # Explain prompts, which have a checkable answer...
    task = list(battery.EXPLAIN_PROMPTS)[:int(cfg["BOUNDARY_N"])]
    # ...and open-ended prompts, which do not. The 2026-08-19 confirmation run is why: three
    # top-dose cells (L56@0.2768, L58@0.1892, L61@0.1750) came back with the explain answers
    # intact -- `ans=1.00` -- while their coherence was 3.0-4.5 and mechanical degeneration
    # 0.50-0.75. A short factual answer survives a dose at which the model's stories have
    # already collapsed into "Garlic Garlic Garlic", so a boundary read only on explain prompts
    # sits above the dose where open-ended generation is already gone.
    #
    # These rows carry no `accept`, so the answer leg cannot apply to them; they are held to
    # coherence and on-task only. That is deliberate -- inventing accept terms for "tell me a
    # story" would be a check that cannot fail.
    task += list(battery.TASK_PROMPTS)[:int(cfg["BOUNDARY_TASK_N"])]
    from m2 import expensive
    rendered, starts, ids = expensive.task_batch(task)
    rows_spec = [dict(channel="explain" if r.get("accept") else "task", unit=pid, prompt=t,
                      start=int(s), source=r["text"], accept=tuple(r.get("accept") or ()))
                 for r, t, s, pid in zip(task, rendered, starts, ids)]

    # Reachability is ARITHMETIC, not something to discover by probing:
    #     alpha = dose * ||h|| / ||v||,  so  max_dose = ALPHA_CEIL * ||v|| / ||h||
    #
    # The first version bisected blind and burned probes on doses the ceiling forbids, then
    # labelled the layer `incoherent_at_lowest_probe` -- reporting a layer it had never measured
    # as one it had measured and found broken. Twenty-four layers came back that way on the first
    # real run and not one of them was a statement about the model.
    norms = (m2config.RUN.norms or {}).get(int(layer)) or {}
    vec, resid = norms.get("vec_norm"), norms.get("resid_norm")
    if not vec or not resid:
        raise RuntimeError(
            f"L{layer} has no norms; measure_residual_norms must run before the boundary search")
    max_reachable = float(cfg["ALPHA_CEIL"]) * float(vec) / float(resid)

    # The descent STARTS at this dose, and `alpha_for` then recomputes `alpha = dose * h / v`.
    # That round trip is not exact in binary floating point: `(C*v/h)*(h/v)` lands one or two
    # ulps above C on roughly one layer in sixteen, `alpha_for` refuses it -- correctly, since a
    # clamped alpha would be a cell measured at a dose other than the one recorded against it --
    # and Phase 1 does not catch `Unreachable`, so a single layer kills the entire run.
    #
    # The bisection this replaced never probed at `max_reachable`, so the defect arrived with the
    # rebuild and has never run on a pod. It would have stopped the next one in Phase 1.
    #
    # A relative shave of 1e-9 is nine million times the ~4e-16 round-trip error and is
    # scientifically nil: it moves the dose by one part in a billion.
    max_reachable *= 1.0 - 1e-9

    if max_reachable < lo:
        out = dict(layer=int(layer), dose_max=None, outcome="unreachable",
                   max_reachable_dose=_floor(max_reachable, 6),
                   bracket=[lo, hi], probes=[],
                   note=("no dose at or above the bracket floor is reachable under ALPHA_CEIL; "
                         "this layer was never measured and says nothing about the model"))
        runio.write_row(BOUNDARY_FILE, out)
        return out

    # Responses generated ahead of the probe that will judge them, keyed by dose.
    #
    # Phase 1 spent 35.1 of the 2026-08-19 run's 79.5 minutes to produce 1,404 responses, while
    # Phase 2 produced 3,724 in 43.0 -- 1.5 seconds per response against 0.69. The work was the
    # same; the batch was not. A probe generates BOUNDARY_N=4 responses, and throughput on this
    # model scales almost linearly with batch size in that range, so the ladder ran the GPU at
    # a quarter of the width the sweep used and roughly a fifth of its throughput.
    #
    # A window of ladder rungs is generated in ONE call instead, each row carrying its own
    # alpha. Rungs are the same geometric grid and are judged in the same descending order, so
    # the first sane rung -- and therefore the bracket handed to the bisection -- is identical
    # to what the one-at-a-time ladder found. Only the order the GPU is asked changes.
    pending: dict[float, list[str]] = {}

    def _prefetch(doses: Sequence[float]) -> None:
        """Generate every rung in `doses` in one call. Silent no-op for rungs already held."""
        want = [d for d in doses if d not in pending]
        if len(want) <= 1:
            return                      # one rung is what the unbatched path already does
        specs = [(d, r) for d in want for r in rows_spec]
        alphas = [float(m2config.alpha_for(int(layer), d)) for d, _ in specs]
        texts = expensive.generate_steered_doses(
            [r["prompt"] for _, r in specs], int(layer), alphas,
            int(cfg["BOUNDARY_MAX_TOKENS"]), float(cfg["TEMPERATURE"]),
            start_positions=[int(r["start"]) for _, r in specs],
            batch_max=int(cfg["GEN_BATCH_MAX"]))
        if len(texts) != len(specs):
            raise RuntimeError(f"prefetch got {len(texts)} responses for {len(specs)} rows")
        for (d, _), t in zip(specs, texts):
            pending.setdefault(d, []).append(t)

    def _probe(dose_val: float, stage: str) -> bool:
        alpha = float(m2config.alpha_for(int(layer), dose_val))
        responses = pending.pop(dose_val, None)
        if responses is None:
            responses = _generate(rows_spec, layer=layer, alpha=alpha,
                                  max_tokens=int(cfg["BOUNDARY_MAX_TOKENS"]), cfg=cfg)
        rows = [battery.response_row(t, channel="effect", concept=concept, unit=r["unit"])
                for r, t in zip(rows_spec, responses)]
        items = [dict(judge.build_item(
            "coherence",
            payload=judge.render("coherence", text_chars=int(cfg["JUDGE_TEXT_CHARS"]),
                                 prompt=r["source"], response=t),
            cache_key=judge.cache_key("BOUNDARY", "coherence", layer=layer,
                                      dose=_floor(dose_val, 6), unit=r["unit"]),
            concept=concept, model_text=(t,), text_chars=int(cfg["JUDGE_TEXT_CHARS"])),
            row_index=i, judge_kind="coherence")
            for i, (r, t) in enumerate(zip(rows_spec, responses))]
        # The generations are PAID FOR before judging can raise, and Phase 1 raising kills the
        # run. Same rule as `measure_cell`: nothing already bought is discarded because a later
        # step failed, so the rows go to UNJUDGED_FILE on the way out.
        try:
            results = judge.run_judges([{k: v for k, v in it.items()
                                         if k not in ("row_index", "judge_kind")}
                                        for it in items],
                                       concurrency=int(cfg["JUDGE_CONCURRENT"]))
        except BaseException:
            for r, t in zip(rows_spec, responses):
                runio.write_row(UNJUDGED_FILE, dict(
                    phase="BOUNDARY", layer=int(layer), dose=_floor(dose_val, 6), alpha=alpha,
                    stage=stage, unit=r["unit"], prompt_text=r["source"], response=t))
            _log(f"   judging FAILED at L{layer}@{dose_val:.3f}; {len(responses)} boundary "
                 f"generations kept in {UNJUDGED_FILE} rather than discarded")
            raise
        verdicts = [judge.verdict(r) for r in results]
        parsed = [p for p, _e in verdicts if p]
        scores = [p["coherence"] for p in parsed]
        mean = sum(scores) / len(scores) if scores else None
        degen = sum(1 for r in rows if r["degenerate"]) / len(rows)

        # THREE conditions, because coherence alone cannot see over-steering. At L29 on the
        # first Garlic run the judge held coherence at 8/10 all the way to `dose_max`, while
        # the same model at 70% of that dose answered "what is a computer" with garlic's
        # botanical profile. Well-formed prose about the wrong subject is exactly what a
        # coherence judge is built to pass.
        #
        #   coherence  the response is not collapsed
        #   on_task    it still addresses the question that was asked
        #   answered   the true content survived, checked mechanically against `accept`
        #
        # A dose is only the boundary if the model still ANSWERS there, not merely if it still
        # writes. This is what makes the whole grid sit in the useful range instead of above it.
        # ...and the three are conjoined PER RESPONSE, then counted. Not three independent
        # fractions: on the 2026-08-19 pod run L29@0.0823 passed with every leg at exactly 0.75
        # while half the battery was corrupted, because each leg lost a DIFFERENT response --
        # the Marie Curie answer was fluent and on-task with the answer gone, the Moon answer
        # was fluent and answered but off-task, and the computer answer explained that a
        # computer "is a member of the plant family" while still containing an accept term.
        # Corruption spread across prompts cancelled out; conjoining first makes it accumulate.
        # Same floor, same responses, same judge calls -- only the order of the AND and the
        # averaging changes, and it moves that layer's boundary from 0.0823 to 0.0757, where
        # all four answers are clean.
        coh_min = float(cfg["BOUNDARY_COHERENCE_MIN"])
        # `None` where the prompt has no checkable answer, so an open-ended row is held to
        # coherence and on-task and is neither credited nor penalised on the answer leg.
        answered = [battery.capability_correct(t, r["accept"]) if r["accept"] else None
                    for r, t in zip(rows_spec, responses)]
        per_response = []
        for i in range(len(responses)):
            p = verdicts[i][0]
            # A response whose judge call failed cannot be certified, so it is not good. It is
            # not silently dropped either: dropping it would raise the fraction it was missing
            # from, which is the wrong direction for a check that gates a dose.
            per_response.append(bool(p) and p.get("coherence") is not None
                                and p["coherence"] >= coh_min
                                and bool(p.get("on_task"))
                                and (answered[i] is None or answered[i]))
        on_task = [bool(p.get("on_task")) for p in parsed]
        frac_on_task = (sum(on_task) / len(on_task)) if on_task else 0.0
        checkable = [a for a in answered if a is not None]
        frac_answered = (sum(checkable) / len(checkable)) if checkable else 0.0
        frac_good = sum(per_response) / len(per_response)
        floor = float(cfg["BOUNDARY_ANSWER_MIN"])

        sane = frac_good >= floor
        # No `unreachable` field here: reachability is decided arithmetically before the ladder
        # starts, and an unreachable layer returns with `probes=[]`. A constant `False` on every
        # probe row reads as a per-probe check that exists and passes, which is worse than
        # nothing -- it is the "a check that cannot fail" shape, and this repo has counted
        # eighteen of them.
        # Floored, matching `max_reachable_dose`. Rounding one and flooring the other
        # put L16's recorded boundary 1e-6 ABOVE its recorded ceiling on the first run.
        # One row per response, written whether the probe passed or failed. A probe that failed
        # is the more interesting half: it is the dose the grid will never sample, and the only
        # way to tell "the model stopped answering" from "one sampled response wandered off" is
        # to read what it actually said. Each row carries the three legs SEPARATELY, so a reader
        # can see which one rejected the dose without recomputing anything.
        for i, (r, t, row) in enumerate(zip(rows_spec, responses, rows)):
            p, err = verdicts[i]
            runio.write_row(BOUNDARY_RESPONSES_FILE, dict(
                layer=int(layer), dose=_floor(dose_val, 6), alpha=alpha, stage=stage,
                unit=r["unit"], prompt_text=r["source"], response=t,
                # judged
                coherence=(p or {}).get("coherence"), on_task=(p or {}).get("on_task"),
                judge_raw=results[i].get("raw"), judge_error=err,
                judge_cached=bool(results[i].get("cached")),
                # mechanical, both of them
                answered=answered[i], degenerate=row["degenerate"],
                degeneration_reason=row.get("degeneration_reason"),
                # `good` is this response passing ALL THREE legs, which is what the probe
                # verdict counts. Recorded per response so a reader can see the conjunction
                # rather than recomputing it from the three columns beside it.
                good=per_response[i],
                # the probe-level verdict this response contributed to
                probe_coherence_mean=mean, probe_on_task=frac_on_task,
                probe_answered=frac_answered, probe_good=frac_good, probe_sane=sane,
                concept=concept))

        probes.append(dict(stage=stage, dose=_floor(dose_val, 6), alpha=alpha,
                           coherence=mean, coherence_n=len(scores), degeneration=degen,
                           on_task=frac_on_task, answered=frac_answered,
                           good=frac_good, sane=sane))
        return sane

    # Descend from the highest dose that is both wanted and reachable, and stop at the first
    # coherent one. A descending ladder reaches any dose; the blind bisection's floor was set by
    # its probe count, so with three probes it could never test below 0.40 however low the
    # bracket said it went -- which is why every layer that survived reported the same 0.40.
    best_sane = None
    fail_above = None
    start = dose = min(hi, max_reachable)
    reached_floor = False
    rung_batch = max(1, int(cfg["BOUNDARY_RUNG_BATCH"]))
    spent = 0
    while best_sane is None and spent < int(cfg["BOUNDARY_PROBES"]):
        if dose < lo:
            reached_floor = True
            break
        # The next window of rungs, on the same geometric grid the one-at-a-time ladder walked
        # and bounded by the same probe budget and the same floor.
        window: list[float] = []
        while (len(window) < rung_batch and spent + len(window) < int(cfg["BOUNDARY_PROBES"])
               and dose >= lo):
            window.append(dose)
            dose = dose * ratio
        _prefetch(window)
        for mid in window:
            spent += 1
            if _probe(mid, "ladder"):
                best_sane = mid
                break
            # The ladder descends, so the LAST failing probe is the tightest upper bound.
            fail_above = mid
    # Rungs generated below the one that passed are speculative work the ladder did not need.
    # They are dropped rather than judged: they cost GPU that was already spent widening the
    # batch, and judging them would buy nothing the bracket does not already have.
    pending.clear()

    # The ladder's answer is coarse by construction: its passing dose sits up to one whole step
    # (1/ratio - 1, 43% at the default 0.70) below its failing one. Bisect inside that measured
    # bracket -- and only inside it, which is what the blind bisection above never had. Stop at
    # BOUNDARY_BISECT_TOL because the grid cannot see finer, or at BOUNDARY_BISECTIONS because
    # each probe costs BOUNDARY_N generations and judge calls. `dose_max` stays a dose that was
    # probed and PASSED; the bracket midpoints that failed tighten `lowest_failing_dose` instead.
    #
    # A layer whose first probe passed has no measured failing dose above -- its boundary is at
    # or above the top of the tested range -- and there is nothing to bisect toward.
    bisect_used = 0
    if best_sane is not None and fail_above is not None:
        for _step in range(int(cfg["BOUNDARY_BISECTIONS"])):
            if (fail_above - best_sane) / best_sane <= float(cfg["BOUNDARY_BISECT_TOL"]):
                break
            mid = (best_sane + fail_above) / 2.0
            if _probe(mid, "bisect"):
                best_sane = mid
            else:
                fail_above = mid
            bisect_used += 1

    # FOUR outcomes, because they are four different facts and three of them were being reported
    # under one name:
    #   ok                    a dose was measured and the model held together at it
    #   unreachable           the dose map forbids it; nothing about the model was learned
    #   incoherent_at_floor   descended past the bracket floor and was never coherent
    #   probes_exhausted      ran out of probes while still ABOVE the floor -- nothing is known
    #                         about any dose below `lowest_probe`
    #
    # The last two were both called `incoherent_at_floor`, and with the shipped settings the
    # ladder cannot reach the floor at all: descending from 2.50 by 0.70 takes twelve probes to
    # cross 0.05 and it is given five, so it stops at 0.60 and reports having measured down to
    # 0.05. That is the same defect as the `incoherent_at_lowest_probe` mislabel this search was
    # rebuilt to remove -- a label asserting a fact the search never established -- reintroduced
    # at the other end of the ladder.
    needed = (1 if start < lo else
              math.ceil(math.log(lo / start) / math.log(ratio)) + 1)
    if best_sane is not None:
        outcome = "ok"
    elif reached_floor:
        outcome = "incoherent_at_floor"
    else:
        outcome = "probes_exhausted"
    out = dict(layer=int(layer),
               dose_max=(_floor(best_sane, 6) if best_sane is not None else None),
               outcome=outcome,
               max_reachable_dose=_floor(max_reachable, 6),
               # What BOUNDARY_PROBES would have to be for this layer's ladder to reach the
               # floor. Recorded per layer because the start of the descent is per layer:
               # `min(bracket_hi, max_reachable)` differs with depth.
               probes_to_reach_floor=int(needed),
               probes_used=len(probes),
               bisect_used=bisect_used,
               # The lowest dose measured and found BROKEN. Together with `dose_max` it is the
               # bracket the boundary actually sits in; None means no probe above `dose_max`
               # failed, so the boundary is at or above the top of the tested range.
               lowest_failing_dose=(_floor(fail_above, 6) if fail_above is not None else None),
               bracket=[lo, hi], step_ratio=ratio, probes=probes,
               # The highest dose actually tested. Without it, a `None` cannot be told from
               # "we never probed low enough", which is what made the first run unreadable.
               # Min/max over the rows, not first/last: bisect probes append after the ladder's,
               # so position no longer encodes dose order.
               highest_probe=(max(p["dose"] for p in probes) if probes else None),
               lowest_probe=(min(p["dose"] for p in probes) if probes else None))
    runio.write_row(BOUNDARY_FILE, out)
    return out


# =====================================================================================
# Phase 2 - one cell
# =====================================================================================

def measure_cell(layer: int, dose: float, *, concept: str, baselines: dict[str, str],
                 cfg: dict | None = None) -> dict:
    """Generate the battery at one cell, judge all of it, write everything, return the row."""
    cfg = config.CONFIG if cfg is None else cfg
    from m2 import config as m2config, runio

    alpha = float(m2config.alpha_for(int(layer), float(dose)))
    spec = battery_prompts(cfg)
    responses = _generate(spec, layer=layer, alpha=alpha,
                          max_tokens=int(cfg["MAX_NEW_TOKENS"]), cfg=cfg)

    rows = [battery.response_row(text, channel=s["channel"], concept=concept,
                                 unit=s["unit"], layer=int(layer), dose=round(float(dose), 6),
                                 alpha=alpha, steered=True)
            for s, text in zip(spec, responses)]
    for spec_row, row in zip(spec, rows):
        if spec_row["channel"] in ("capability", "explain"):
            row["capability_correct"] = battery.capability_correct(
                row["response"], spec_row["accept"])
        # ONE name for the source prompt. This carried three -- `prompt_text`, `_source` and
        # `source` -- and the judge builder read the one nothing wrote, which killed the sweep
        # at its first cell after the generations had been paid for.
        row["prompt_text"] = spec_row["source"]

    # The generations are PAID FOR at this point and the judging below can still raise -- a
    # payload the transport rejects, a field name that does not line up, a missing baseline.
    # When it did exactly that on the first cell of a pod run, the whole battery was lost and
    # had to be regenerated on the next attempt. Nothing that has already been bought gets
    # discarded because a later step failed: on the way out, the un-judged rows are written
    # first, so a rerun resumes with the text in hand.
    try:
        items = _judge_items(rows, concept=concept, baselines=baselines, phase="SWEEP",
                             layer=int(layer), dose=round(float(dose), 6), cfg=cfg)
        results = judge.run_judges(
            [{k: v for k, v in it.items() if k not in ("row_index", "judge_kind")}
             for it in items],
            concurrency=int(cfg["JUDGE_CONCURRENT"]))
        records = _apply_verdicts(rows, items, results, concept=concept)
    except BaseException:
        for row in rows:
            runio.write_row(UNJUDGED_FILE, dict(row, layer=int(layer),
                                                dose=round(float(dose), 6)))
        _log(f"   judging FAILED at L{layer}@{dose:.3f}; {len(rows)} generations kept in "
             f"{UNJUDGED_FILE} rather than discarded")
        raise

    for row in rows:
        runio.write_row(RESPONSES_FILE, row)
    for rec in records:
        runio.write_row(JUDGE_FILE, dict(rec, layer=int(layer), dose=round(float(dose), 6)))

    cell = _summarise_cell(rows, layer=int(layer), dose=round(float(dose), 6), alpha=alpha,
                           cfg=cfg)
    runio.write_row(CELLS_FILE, cell)
    return cell


def _summarise_cell(rows: Sequence[dict], *, layer: int, dose: float, alpha: float,
                    cfg: dict) -> dict:
    """Aggregate one cell. Judged measures and mechanical measures, side by side, all reported.

    Nothing here combines the two or lets one override the other. `sanity` is deliberately NOT
    computed at this stage: it is a threshold decision, thresholds are what M2 got wrong, and
    the analysis layer can compose one from these fields as many times as it takes.
    """
    z = float(cfg["RATE_CI_Z"])
    by = lambda ch: [r for r in rows if r["channel"] == ch]      # noqa: E731

    def judged(ch: str, kind: str, field: str) -> list:
        return [r["judged"][kind][field] for r in by(ch)
                if (r.get("judged") or {}).get(kind) is not None]

    ident = by("identify")
    ident_hits = [r for r in ident if (r.get("judged") or {}).get("identify")]
    ident_clean = [r for r in ident_hits if not r["degenerate"]]
    effect_scores = judged("effect", "effect", "influence")
    coh_scores = judged("effect", "coherence", "coherence")
    caps = by("capability")
    sr = by("self_report")
    exp = by("explain")
    exp_infl = judged("explain", "effect", "influence")
    exp_coh = judged("explain", "coherence", "coherence")

    out: dict[str, Any] = dict(
        layer=layer, dose=dose, alpha=alpha,
        # --- judged ---
        identification=(battery.rate(sum(1 for r in ident_hits if r["judged"]["identify"]["matches"]),
                                     len(ident_hits), z) if ident_hits else None),
        # TODO 35, resolved as "report both, decide nothing". A response that collapsed into
        # repetition is scored `matches=False` above and counted as a non-identification, exactly
        # like a fluent answer naming the wrong concept -- but those are different events, and
        # conflating them is how M2 read a broken model as a covert one. The self-report taxonomy
        # already excludes collapses; this makes the same view available on the channel that
        # carries the headline metric, without either version being privileged.
        #
        # `None` when every trial collapsed: that cell has no readable identification rate, which
        # is a different statement from a rate of zero and must not be printed as one.
        identification_excluding_degenerate=(
            battery.rate(sum(1 for r in ident_clean if r["judged"]["identify"]["matches"]),
                         len(ident_clean), z) if ident_clean else None),
        identify_degenerate_n=sum(1 for r in ident if r["degenerate"]),
        identified_as=sorted({str(r["judged"]["identify"]["named"]).lower()
                              for r in ident_hits}),
        effectiveness=(battery.mean_se(effect_scores) if effect_scores else None),
        effect_forms=sorted({r["judged"]["effect"]["form"] for r in by("effect")
                             if (r.get("judged") or {}).get("effect")}),
        coherence=(battery.mean_se(coh_scores) if coh_scores else None),
        # The coherence judge answers TWO questions and only one was being kept. `on_task` is
        # "does this actually address the prompt", which is a different failure from "is this
        # well-formed": a response can be fluent, grammatical and about the wrong thing. It was
        # judged on every coherence call, stored in the judge archive, attached to the row --
        # and never aggregated, so no cell reported it and no reader saw it.
        #
        # Note what it can and cannot catch. The four task prompts are open-ended, and a
        # garlic-flavoured story is still a story, so `on_task` stays True through heavy
        # influence by design. It fires when the response stops answering at all. The channel
        # that tests answering a question with a RIGHT answer is `capability`.
        on_task=(battery.rate(sum(1 for r in by("effect")
                                  if ((r.get("judged") or {}).get("coherence") or {}).get("on_task")),
                              len(coh_scores), z) if coh_scores else None),
        capability=(battery.rate(sum(1 for r in caps if r.get("capability_correct")),
                                 len(caps), z) if caps else None),
        # --- the explain channel: prose AND truth, on the same response ------------------
        # `explain_answered` is the over-steer signal. A response can be fluent (coherence 8),
        # on a plausible topic, and no longer an answer to the question asked -- which is what
        # a garlic listicle in reply to "what is a computer" is. Coherence cannot see that; a
        # story cannot stop being a story, but an answer can stop answering.
        explain_influence=(battery.mean_se(exp_infl) if exp_infl else None),
        explain_answered=(battery.rate(sum(1 for r in exp
                                           if ((r.get("judged") or {}).get("coherence") or {})
                                           .get("on_task")), len(exp), z) if exp else None),
        explain_correct=(battery.rate(sum(1 for r in exp if r.get("capability_correct")),
                                      len(exp), z) if exp else None),
        explain_coherence=(battery.mean_se(exp_coh) if exp_coh else None),
        # --- mechanical, recorded, deciding nothing ---
        mechanical=dict(
            identify=battery.channel_summary(ident, z=z) if ident else None,
            effect=battery.channel_summary(by("effect"), z=z) if by("effect") else None,
            self_report=battery.channel_summary(sr, z=z) if sr else None,
        ),
        judge_errors=sum(1 for r in rows
                         for e in (r.get("judge_error") or {}).values() if e),
    )
    if sr:
        classes = [r.get("self_report_class") for r in sr]
        out["self_report_classes"] = {c: classes.count(c) for c in set(classes) if c}
        named = [r["judged"]["self_report"]["matches"] for r in sr
                 if (r.get("judged") or {}).get("self_report")]
        claims = [r["judged"]["self_report"]["claims"] for r in sr
                  if (r.get("judged") or {}).get("self_report")]
        if claims:
            out["self_report"] = battery.rate(sum(1 for c in claims if c == "YES"), len(claims), z)
            out["self_report_names_concept"] = battery.rate(sum(1 for m in named if m),
                                                            len(named), z)
    return out


# =====================================================================================
# The run
# =====================================================================================

def _done(name: str, keys: tuple) -> set:
    from m2 import runio

    return runio.done_keys(name, keys)


def _num(value: Any, spec: str, width: int) -> str:
    """Format a number that may legitimately be absent, without inventing one.

    A judged measure is `None` when every call for it failed. Printing 0.00 there would put a
    measurement on the console that was never made -- which is the same class of error as a
    parser defaulting a bad answer to zero.
    """
    return format(value, spec) if isinstance(value, (int, float)) else "-".rjust(width)


def _cell_line(cell: dict) -> str:
    """One progress line per cell: the judged measures, then the mechanical ones beside them.

    Both are shown because their disagreement is an output of this run, and the console is where
    an operator watching a sweep would first notice it.
    """
    ident = (cell.get("identification") or {}).get("rate")
    eff = (cell.get("effectiveness") or {}).get("mean")
    coh = (cell.get("coherence") or {}).get("mean")
    cap = (cell.get("capability") or {}).get("rate")
    # The over-steer signal, and the reason it is on the board rather than only on disk: a model
    # that has stopped answering reads `coh=8.0 ans=0.00`, and without `ans` the operator sees
    # only the 8.0. It is the MECHANICAL check on the explain prompts, not the judged `on_task`,
    # because the judge scored a garlic listicle answering "what is a computer" as on-task --
    # fluent, plausible, and no longer the answer. `explain_answered` (the judged view) is
    # aggregated per cell too and stays in `cells.jsonl` for the comparison.
    ans = (cell.get("explain_correct") or {}).get("rate")
    mech = (cell.get("mechanical") or {}).get("effect") or {}
    degen = (mech.get("degeneration") or {}).get("rate")
    errs = cell.get("judge_errors") or 0
    return (f"L{cell['layer']:<3}@{cell['dose']:<6.3f} "
            f"ident={_num(ident, '.2f', 4)} eff={_num(eff, '4.1f', 4)} "
            f"coh={_num(coh, '4.1f', 4)} ans={_num(ans, '.2f', 4)} "
            f"cap={_num(cap, '.2f', 4)} "
            f"| mech.degen={_num(degen, '.2f', 4)}"
            + (f"  [{errs} judge errors]" if errs else ""))


def run_sweep(concept: str, cfg: dict | None = None) -> dict:
    """Phases 0, 1 and 2 end to end. Returns the summary payload it also writes to disk."""
    cfg = config.CONFIG if cfg is None else cfg
    from m2 import config as m2config, runio

    ctx = m2config.RUN
    if ctx.run_dir is None or not ctx.concept:
        raise RuntimeError("call m3.sweep.open_run() before run_sweep()")

    layers = config.layers_for_depth(int(ctx.n_layers), cfg)
    fractions = [float(f) for f in cfg["DOSE_FRACTIONS"]]
    t0 = time.time()

    _log(f"PHASE 0  calibrate   {len(layers)} layers L{layers[0]}-L{layers[-1]}")
    cal = calibrate(concept, layers, cfg)

    # ---- Phase 1 -----------------------------------------------------------------------
    _log(f"PHASE 1  boundary    {int(cfg['BOUNDARY_PROBES'])} probes/layer, judged coherence")
    have = _done(BOUNDARY_FILE, ("layer",))
    boundaries: dict[int, dict] = {int(r["layer"]): r for r in runio.read_rows(BOUNDARY_FILE)}
    for layer in layers:
        if (runio._key_value(layer),) in have:
            continue
        try:
            row = find_boundary(layer, concept, cfg)
        except m2config.Unreachable as exc:
            # Reachability is decided arithmetically before the ladder starts, so this should be
            # impossible -- and `impossible` is exactly what the ALPHA_CEIL round-trip was until
            # it was measured. One layer must not be able to end a sweep that has already paid
            # for Phase 0; record it and carry on, the same way Phase 2 does.
            row = dict(layer=int(layer), dose_max=None, outcome="unreachable",
                       probes=[], note=f"alpha_for refused the starting dose: {exc}")
            runio.write_row(BOUNDARY_FILE, row)
        boundaries[int(layer)] = row
        _log(f"   L{layer:<3} dose_max={row['dose_max']}  {row['outcome']}")

    # A layer that ran out of probes above the floor is not a layer that broke -- it is a layer
    # the search stopped short on, and the difference decides whether its absence from Phase 2 is
    # a finding or a budget. Say so once, with the number that fixes it.
    short = [r for r in boundaries.values() if r.get("outcome") == "probes_exhausted"]
    if short:
        need = max(int(r.get("probes_to_reach_floor") or 0) for r in short)
        lowest = min(float(r["lowest_probe"]) for r in short if r.get("lowest_probe"))
        _log(f"WARNING  {len(short)} layers ran out of probes while still ABOVE the bracket "
             f"floor (lowest tested {lowest:.3f}, floor {float(cfg['BOUNDARY_BRACKET'][0]):.3f}). "
             f"They are NOT known to be incoherent — nothing was measured below that dose. "
             f"Set BOUNDARY_PROBES={need} to descend the whole bracket.")

    # ---- Phase 2 -----------------------------------------------------------------------
    plan: list[tuple[int, float]] = []
    skipped: list[dict] = []
    for layer in layers:
        row = boundaries.get(int(layer))
        if not row or row.get("dose_max") is None:
            # Named, not silent. Work a run did not do reads later as work it did and found
            # nothing in -- which is how M2's empty operating point was nearly read as a null.
            skipped.append(dict(layer=int(layer),
                                reason=(row or {}).get("outcome", "no_boundary_row")))
            continue
        for frac in fractions:
            plan.append((int(layer), round(float(row["dose_max"]) * frac, 6)))

    have = _done(CELLS_FILE, ("layer", "dose"))
    todo = [(l, d) for l, d in plan
            if (runio._key_value(l), runio._key_value(d)) not in have]
    # A per-layer search that returns the same number for every layer has measured one thing,
    # not forty-nine, and the run must say so rather than presenting a global ladder as a
    # per-layer one. The first real run did exactly that and nothing noticed.
    found = [r["dose_max"] for r in boundaries.values() if r.get("dose_max") is not None]
    if len(found) >= 4 and len(set(found)) == 1:
        _log(f"WARNING  every one of {len(found)} layers returned dose_max={found[0]}. "
             "The boundary search has resolved nothing per-layer; this is a GLOBAL ladder. "
             "Read the results accordingly and widen BOUNDARY_PROBES / BOUNDARY_STEP.")
    elif found and len(set(found)) <= max(2, len(found) // 8):
        _log(f"NOTE     {len(found)} layers returned only {len(set(found))} distinct dose_max "
             "values; the boundary search is coarse relative to the surface.")

    _log(f"PHASE 2  sweep       {len(todo)} cells to measure "
         f"({len(plan) - len(todo)} already on disk, {len(skipped)} layers without a boundary)")

    measured = 0
    for i, (layer, dose) in enumerate(todo, start=1):
        try:
            cell = measure_cell(layer, dose, concept=concept,
                                baselines=cal["baselines"], cfg=cfg)
        except m2config.Unreachable as exc:
            skipped.append(dict(layer=layer, dose=dose, reason=str(exc)))
            _log(f"   [{i}/{len(todo)}] L{layer}@{dose:.3f}  UNREACHABLE")
            continue
        measured += 1
        _log(f"   [{i}/{len(todo)}] {_cell_line(cell)}")

    cells = runio.read_rows(CELLS_FILE)
    summary = dict(
        concept=concept, mode="sweep", version="m3",
        layers=layers, dose_fractions=fractions,
        n_cells_planned=len(plan), n_cells_on_disk=len(cells), n_cells_this_attempt=measured,
        skipped=skipped,
        boundaries=[boundaries[k] for k in sorted(boundaries)],
        liveness=cal["liveness"],
        elapsed_s=round(time.time() - t0, 1),
        config={k: v for k, v in cfg.items()},
        config_hash=config.config_hash(cfg),
    )
    runio.write_json(SUMMARY_FILE, summary)
    _read_bundle(concept, cfg)
    _log(f"DONE  {len(cells)} cells on disk, {measured} this attempt, "
         f"{summary['elapsed_s'] / 60:.1f} min")
    return summary


def open_run(concept: str, cfg: dict | None = None) -> Path:
    """Point the shared run context at this concept and return its run directory.

    M3 reuses M2's model, generation, judge and I/O layers, which read their state from
    `m2.config.RUN`. This is the one place that is set up, and M3's own config is what fills it.
    """
    cfg = config.CONFIG if cfg is None else cfg
    from m2 import config as m2config, runio

    if not config.concept_allowed(concept):
        raise PermissionError(
            f"{concept!r} is on HARMFUL_CONCEPTS — the arm this study has deliberately not run.")
    run_dir = config.run_dir_for(concept, cfg)
    repo = Path(__file__).resolve().parents[1]
    if repo in run_dir.resolve(strict=False).parents or run_dir.resolve(strict=False) == repo:
        raise ValueError(f"run dir {run_dir} is inside the repository; set M3_RUNS_DIR")
    run_dir.mkdir(parents=True, exist_ok=True)

    m2cfg = config.m2_config(concept, cfg)
    m2config.CONFIG.clear()
    m2config.CONFIG.update(m2cfg)
    m2config.RUN.reset_concept(concept, run_dir, m2cfg)
    judge.configure_transport(cfg)
    judge.configure_generation(cfg)
    runio.log(f"m3 sweep | concept {concept} | config {config.config_hash(cfg)} | {run_dir}")
    return run_dir


# =====================================================================================
# The read-this bundle
# =====================================================================================

READ_FILE = "read_this.md"


def _read_bundle(concept: str, cfg: dict) -> Path:
    """Select the responses a human should read, and write them somewhere readable.

    Every deep defect this project has found was found by a person reading raw generations --
    the apple confabulation, "penguins, cats, cats" scored as detection, a judge calling
    `## ## ## ##` coherent, `d3` reading the preamble, `d2` measured on a collapsed channel.
    None was caught by a gate, a rate or a judge.

    So this is a required output, and it is selected by DISAGREEMENT rather than at random.
    Random sampling of ~3,000 transcripts would have found none of those five; each was visible
    precisely because two signals said different things about the same text.
    """
    from m2 import runio

    limit = int(cfg["READ_BUNDLE_N"])
    responses = runio.read_rows(RESPONSES_FILE)
    nulls = runio.read_rows(NULL_FILE)
    judged = {(r["layer"], r["dose"], r["unit"], r["judge"]): r
              for r in runio.read_rows(JUDGE_FILE)
              if r.get("layer") is not None}

    picked: list[tuple[str, dict, str]] = []

    def add(reason: str, row: dict, detail: str = "") -> None:
        if not any(p[1] is row for p in picked):
            picked.append((reason, row, detail))

    for row in responses:
        key = (row.get("layer"), row.get("dose"), row.get("unit"))
        coh = (judged.get((*key, "coherence")) or {}).get("parsed") or {}
        eff = (judged.get((*key, "effect")) or {}).get("parsed") or {}
        # 1. The two signals contradicting each other. This is the `## ## ## ##` failure, made
        #    surfaceable in both directions -- neither measure dominates the other.
        if coh.get("coherence") is not None:
            if row["degenerate"] and coh["coherence"] >= 5:
                add("judge says coherent, detector says degenerate", row,
                    f"coherence={coh['coherence']} rule={row['degeneration_reason']}")
            elif not row["degenerate"] and coh["coherence"] <= 2:
                add("judge says incoherent, detector says clean", row,
                    f"coherence={coh['coherence']}")
        # 2. Influence the mechanical count cannot see. 13% of influential responses on the
        #    probe archive named the concept zero times.
        if eff.get("influence", 0) >= 4 and row["concept_mentions"] == 0:
            add("judged influential, zero concept mentions", row,
                f"influence={eff['influence']} form={eff.get('form')}")
        # 3. The covert class: the concept is present and the model denies noticing it.
        if row.get("self_report_class") == "leaked":
            add("DENIES detection while emitting the concept", row, "self_report=leaked")
        # 4. A judged measure that failed outright.
        if any((row.get("judge_error") or {}).values()):
            add("a judge call failed here", row,
                str({k: v for k, v in (row.get("judge_error") or {}).items() if v}))

    # The cap applies to the disagreements ONLY, and it is applied by round-robin across the
    # reasons rather than by taking the first N rows. Rows arrive in cell order, so a plain
    # truncation returns the shallowest layers' disagreements and silently drops every class
    # that only appears deep in the sweep -- which is where the covert regime lives.
    found = len(picked)
    by_reason: dict[str, list] = {}
    for entry in picked:
        by_reason.setdefault(entry[0], []).append(entry)
    shown: list[tuple[str, dict, str]] = []
    while len(shown) < limit and any(by_reason.values()):
        for reason in list(by_reason):
            if by_reason[reason] and len(shown) < limit:
                shown.append(by_reason[reason].pop(0))
    dropped = found - len(shown)

    # 5. The whole null arm, always -- and appended AFTER the cap, not subject to it. It was
    #    inside the truncation, so a run with more than READ_BUNDLE_N disagreements dropped the
    #    entire null arm, which is the one section that cannot be inferred from the others: a
    #    steered rate is unreadable without what the framing produces on its own, and at alpha=0
    #    this model names a concept every single time it is asked.
    nulls_picked = [("alpha=0 null arm", row, "") for row in nulls]
    picked = shown + nulls_picked

    path = runio.artefact_path(READ_FILE)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Read this — {concept}\n\n")
        fh.write(f"{len(shown)} of {found} disagreements, plus all {len(nulls_picked)} "
                 f"alpha=0 null responses, out of {len(responses) + len(nulls)} total.\n")
        fh.write("Selected by **disagreement**, not at random.\n\n")
        if dropped:
            # Never a silent cap. A bundle that shows 40 of 137 and does not say so reads as
            # "these were the disagreements" when it is "these were the first few".
            fh.write(f"> **{dropped} further disagreements are not shown here** "
                     f"(READ_BUNDLE_N={limit}). They are all in "
                     f"`{RESPONSES_FILE}` and `{JUDGE_FILE}`; raise READ_BUNDLE_N to widen "
                     "this file. Doing so re-writes the bundle and re-measures nothing.\n\n")
        fh.write("Every deep defect this project has found was found by a person reading raw\n"
                 "generations, and none by a gate, a rate or a judge. Random sampling would\n"
                 "have found none of them; each was visible because two signals disagreed.\n\n")
        if not shown:
            fh.write("Nothing disagreed anywhere. On a real surface that is worth being\n"
                     "suspicious of rather than pleased about — check that the judges ran.\n")
        for reason, row, detail in picked:
            where = ("alpha=0" if row.get("layer") is None
                     else f"L{row['layer']}@{row['dose']:.3f}")
            fh.write(f"\n---\n\n### {reason}\n")
            fh.write(f"`{where}` · {row['channel']} · {row['unit']}"
                     + (f" · {detail}" if detail else "") + "\n\n")
            fh.write("```\n" + str(row["response"]).strip()[:1500] + "\n```\n")
    _log(f"read-this bundle: {len(shown)} of {found} disagreements + {len(nulls_picked)} null "
         f"-> {READ_FILE}" + (f"  ({dropped} not shown)" if dropped else ""))
    return path
