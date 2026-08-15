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

import time
from pathlib import Path
from typing import Any, Sequence

from . import battery, config, judge


__all__ = [
    "calibrate", "find_boundary", "measure_cell", "run_sweep",
    "battery_prompts", "CELLS_FILE", "BOUNDARY_FILE", "RESPONSES_FILE", "JUDGE_FILE",
    "NORMS_FILE",
]

CELLS_FILE = "cells.jsonl"
BOUNDARY_FILE = "boundaries.jsonl"
RESPONSES_FILE = "responses_transcripts.jsonl"
JUDGE_FILE = "judge_calls.jsonl"
NULL_FILE = "null_transcripts.jsonl"
NORMS_FILE = "norms.jsonl"
SUMMARY_FILE = "summary.json"


def _trials(n: int) -> list[int]:
    """Fixed, spread trial numbers so a rerun builds the same prompts and rows join.

    Spread rather than 1..n because the framing says trials run to 50, and a model shown
    "Trial 1" through "Trial 6" is being asked a slightly different question from one shown
    trials spanning the range.
    """
    return [1 + 6 * i for i in range(int(n))]


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


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

    caps = list(battery.CAPABILITY_PROMPTS)[:int(cfg["N_CAPABILITY"])]
    rendered, starts, ids = expensive.task_batch(caps)
    for row, text, st, pid in zip(caps, rendered, starts, ids):
        rows.append(dict(channel="capability", unit=pid, trial=None,
                         prompt=text, start=int(st), source=row["text"],
                         accept=tuple(row["accept"])))

    if len(rows) > int(cfg["GEN_BATCH_MAX"]):
        raise ValueError(
            f"the battery is {len(rows)} prompts but GEN_BATCH_MAX is {cfg['GEN_BATCH_MAX']}. "
            "It would be split across two generation calls, roughly doubling the sweep's GPU "
            "time. Reduce a channel or raise the batch cap deliberately.")
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

        elif ch == "effect":
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

            # Coherence on a fixed prefix of the effect responses, not a random sample, so the
            # same prompts are scored at every cell and cells stay comparable.
            if coherence_budget > 0:
                coherence_budget -= 1
                payload = judge.render("coherence", text_chars=text_chars,
                                       prompt=row["prompt_text"], response=response)
                items.append(dict(judge.build_item("coherence", payload=payload,
                                                   cache_key=judge.cache_key(phase, "coherence", layer=layer, dose=dose,
                                                        unit=row["unit"]),
                                                   concept=concept, model_text=(response,)),
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
    from m2 import config as m2config, model, runio, vectors

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
    ctx = m2config.RUN
    for layer in layers:
        row = ctx.norms.get(int(layer)) or {}
        vec, resid = row.get("vec_norm"), row.get("resid_norm")
        runio.write_row(NORMS_FILE, dict(
            layer=int(layer), vec_norm=vec, resid_norm=resid,
            # The ratio is what decides reachability: alpha = dose * resid / vec, so a large
            # ratio is why a shallow layer cannot be dosed under ALPHA_CEIL.
            resid_over_vec=(None if not vec else round(float(resid) / float(vec), 4)),
            alpha_ceil=float(cfg["ALPHA_CEIL"]),
            max_reachable_dose=(None if not resid else
                                round(float(cfg["ALPHA_CEIL"]) * float(vec) / float(resid), 4))))

    # R14. It reads the vectors, so it can only run after extraction. The hook it checks once
    # returned identically zero at 30 of 30 cells for an hour with every other check satisfied,
    # and this sweep's entire finding is whether cells show influence -- a dead hook would hand
    # back a clean, plausible, completely empty surface.
    live = model.hook_liveness()
    _log(f"R14 pass  start_pos {live['d_start_pos']:.2e}  all-pos {live['d_all_pos']:.2e}")

    rows = battery_prompts(cfg)
    _log(f"null battery: {len(rows)} prompts, unsteered")
    responses = _generate(rows, layer=None, alpha=None,
                          max_tokens=int(cfg["MAX_NEW_TOKENS"]), cfg=cfg)

    null_rows = [battery.response_row(text, channel=r["channel"], concept=concept,
                                      unit=r["unit"], layer=None, dose=None, alpha=0.0,
                                      steered=False)
                 for r, text in zip(rows, responses)]
    baselines = {r["unit"]: text for r, text in zip(rows, responses) if r["channel"] == "effect"}

    for row in null_rows:
        runio.write_row(NULL_FILE, row)

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
    from m2 import config as m2config, runio

    lo, hi = (float(x) for x in cfg["BOUNDARY_BRACKET"])
    ratio = float(cfg["BOUNDARY_STEP"])
    probes: list[dict] = []
    task = list(battery.TASK_PROMPTS)[:int(cfg["BOUNDARY_N"])]
    from m2 import expensive
    rendered, starts, ids = expensive.task_batch(task)
    rows_spec = [dict(channel="effect", unit=pid, prompt=t, start=int(s), source=r["text"])
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

    if max_reachable < lo:
        out = dict(layer=int(layer), dose_max=None, outcome="unreachable",
                   max_reachable_dose=round(max_reachable, 6),
                   bracket=[lo, hi], probes=[],
                   note=("no dose at or above the bracket floor is reachable under ALPHA_CEIL; "
                         "this layer was never measured and says nothing about the model"))
        runio.write_row(BOUNDARY_FILE, out)
        return out

    # Descend from the highest dose that is both wanted and reachable, and stop at the first
    # coherent one. A descending ladder reaches any dose; the bisection's floor was set by its
    # probe count, so with three probes it could never test below 0.40 however low the bracket
    # said it went -- which is why every layer that survived reported the same 0.40.
    best_sane = None
    dose = min(hi, max_reachable)
    for _step in range(int(cfg["BOUNDARY_PROBES"])):
        if dose < lo or best_sane is not None:
            break
        alpha = float(m2config.alpha_for(int(layer), dose))
        mid = dose
        dose = dose * ratio

        responses = _generate(rows_spec, layer=layer, alpha=alpha,
                              max_tokens=int(cfg["BOUNDARY_MAX_TOKENS"]), cfg=cfg)
        rows = [battery.response_row(t, channel="effect", concept=concept, unit=r["unit"])
                for r, t in zip(rows_spec, responses)]
        items = [dict(judge.build_item(
            "coherence",
            payload=judge.render("coherence", text_chars=int(cfg["JUDGE_TEXT_CHARS"]),
                                 prompt=r["source"], response=t),
            cache_key=judge.cache_key("BOUNDARY", "coherence", layer=layer,
                                      dose=round(mid, 6), unit=r["unit"]),
            concept=concept, model_text=(t,)), row_index=i, judge_kind="coherence")
            for i, (r, t) in enumerate(zip(rows_spec, responses))]
        results = judge.run_judges([{k: v for k, v in it.items()
                                     if k not in ("row_index", "judge_kind")} for it in items],
                                   concurrency=int(cfg["JUDGE_CONCURRENT"]))
        scores = [p["coherence"] for p, _e in (judge.verdict(r) for r in results) if p]
        mean = sum(scores) / len(scores) if scores else None
        degen = sum(1 for r in rows if r["degenerate"]) / len(rows)

        sane = mean is not None and mean >= float(cfg["BOUNDARY_COHERENCE_MIN"])
        probes.append(dict(dose=round(mid, 6), alpha=alpha, unreachable=False,
                           coherence=mean, coherence_n=len(scores), degeneration=degen,
                           sane=sane))
        if sane:
            best_sane = mid

    # Three outcomes, and they are three different facts about three different things:
    #   ok                    a dose was measured and the model held together at it
    #   unreachable           the dose map forbids it; nothing about the model was learned
    #   incoherent_at_floor   measured all the way down to the bracket floor and never coherent
    out = dict(layer=int(layer),
               dose_max=(round(best_sane, 6) if best_sane is not None else None),
               outcome=("ok" if best_sane is not None else "incoherent_at_floor"),
               max_reachable_dose=round(max_reachable, 6),
               bracket=[lo, hi], step_ratio=ratio, probes=probes,
               # The highest dose actually tested. Without it, a `None` cannot be told from
               # "we never probed low enough", which is what made the first run unreadable.
               highest_probe=(probes[0]["dose"] if probes else None),
               lowest_probe=(probes[-1]["dose"] if probes else None))
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
        if spec_row["channel"] == "capability":
            row["capability_correct"] = battery.capability_correct(
                row["response"], spec_row["accept"])
        # ONE name for the source prompt. This carried three -- `prompt_text`, `_source` and
        # `source` -- and the judge builder read the one nothing wrote, which killed the sweep
        # at its first cell after the generations had been paid for.
        row["prompt_text"] = spec_row["source"]

    items = _judge_items(rows, concept=concept, baselines=baselines, phase="SWEEP",
                         layer=int(layer), dose=round(float(dose), 6), cfg=cfg)
    results = judge.run_judges(
        [{k: v for k, v in it.items() if k not in ("row_index", "judge_kind")} for it in items],
        concurrency=int(cfg["JUDGE_CONCURRENT"]))
    records = _apply_verdicts(rows, items, results, concept=concept)

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
    effect_scores = judged("effect", "effect", "influence")
    coh_scores = judged("effect", "coherence", "coherence")
    caps = by("capability")
    sr = by("self_report")

    out: dict[str, Any] = dict(
        layer=layer, dose=dose, alpha=alpha,
        # --- judged ---
        identification=(battery.rate(sum(1 for r in ident_hits if r["judged"]["identify"]["matches"]),
                                     len(ident_hits), z) if ident_hits else None),
        identified_as=sorted({str(r["judged"]["identify"]["named"]).lower()
                              for r in ident_hits}),
        effectiveness=(battery.mean_se(effect_scores) if effect_scores else None),
        effect_forms=sorted({r["judged"]["effect"]["form"] for r in by("effect")
                             if (r.get("judged") or {}).get("effect")}),
        coherence=(battery.mean_se(coh_scores) if coh_scores else None),
        capability=(battery.rate(sum(1 for r in caps if r.get("capability_correct")),
                                 len(caps), z) if caps else None),
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
    mech = (cell.get("mechanical") or {}).get("effect") or {}
    degen = (mech.get("degeneration") or {}).get("rate")
    errs = cell.get("judge_errors") or 0
    return (f"L{cell['layer']:<3}@{cell['dose']:<6.3f} "
            f"ident={_num(ident, '.2f', 4)} eff={_num(eff, '4.1f', 4)} "
            f"coh={_num(coh, '4.1f', 4)} cap={_num(cap, '.2f', 4)} "
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
        row = find_boundary(layer, concept, cfg)
        boundaries[int(layer)] = row
        _log(f"   L{layer:<3} dose_max={row['dose_max']}  {row['outcome']}")

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

    if not config.is_benign(concept, cfg):
        raise PermissionError(
            f"{concept!r} is not on BENIGN_CONCEPTS. This mode writes every generation to disk "
            "and exports them; a non-benign concept must be structurally unable to enter it.")
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
