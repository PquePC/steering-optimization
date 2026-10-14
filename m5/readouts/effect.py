"""m5.readouts.effect - the judge-free effect readouts of the screen tier (plan §3.1).

    e9   teacher-forced concept lift. On neutral prompts (`m5.data.influence.NEUTRAL_PROMPTS`),
         the model's own unsteered greedy continuation is generated once and held fixed; along
         its first E9_POSITIONS positions, the log-probability mass of the concept's tokens is
         read steered and unsteered (`Backend.candidate_logprobs`), and the mean per-position
         difference is the lift. The change in mean predictive entropy (`Backend.score`) is
         reported beside it, because a damaged model's distribution flattens, which raises every
         rare token's probability -- the concept's included. A lift with a large entropy rise is
         the flattening Sridharan describes, not the concept.
    e10  behavioural propensity. On A/B items in CAA's format (`m5.data.contrastive`), the shift
         in logit(the behaviour's answer) - logit(the other) from unsteered to steered. One dose
         per readout; the slope across doses is fitted by the search (plan §3.1: "as a slope over
         +- doses").

Neither consults a judge. e9 needs a concept word, so it is missing for nameless vectors; e10
needs A/B items for a behaviour, so it is missing for a vector whose class has none unless the
caller names the behaviour (a control vector read on the sycophancy items is e10's own control).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from m5 import prompts
from m5.backend.base import ScoreRequest
from m5.data import contrastive
from m5.data.influence import NEUTRAL_PROMPTS
from m5.intervention import Cell
from m5.readouts._common import (GENERATED_ONLY, NULL_CONCEPT, UNREADABLE_OPTIONS,
                                 candidate_logprobs, check_args, concept_tokens, emit,
                                 logsumexp_rows, mean_se, nameless, option_logprobs, options,
                                 own_continuation, resolve_mode, score_requests, take)
from m5.results import Readout
from m5.steer import intervention_for

E9_MODES = ("all", "generated")
E10_MODES = ("live", "all", "generated")

# The A/B item sets embedded in this repository, by the vector class they measure. Others (CAA's
# files, read with `m5.data.contrastive.load_caa` from outside the repository) are passed as
# `items=`.
E10_SETS: dict[str, contrastive.ABSet] = {"sycophancy": contrastive.SYCOPHANCY_AB}


def read_e9(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str | None = None, pass_: str | None = None) -> Readout:
    """e9 (teacher-forced concept lift, nats per position; higher = more effect).

    n is the number of neutral prompts: SCREEN_A_E9_PROMPTS in pass A (and when no pass is
    given), SCREEN_B_E9_PROMPTS in pass B. Steering covers the prompt and the teacher-forced
    continuation (mode "all"), or the continuation only ("generated")."""
    code = "e9"
    check_args(code, variant=variant, variant_aware=False, pass_=pass_)
    mode = resolve_mode(code, cell, mode, kind="open", allowed=E9_MODES)

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    if cell is None:
        return out(missing=NULL_CONCEPT)
    concept = ctx.concept_of(cell.vector_id)
    if concept is None:
        return out(missing=nameless(code, cell.vector_id))
    toks = concept_tokens(ctx.backend, concept)
    if not toks.ids:
        return out(missing=f"no variant of {concept!r} has a usable first token (dropped "
                           f"{len(toks.dropped)} split variants)")
    size = n if n is not None else ctx.cfg["SCREEN_B_E9_PROMPTS" if pass_ == "B" else "SCREEN_A_E9_PROMPTS"]
    chosen = take(code, NEUTRAL_PROMPTS, size, what="neutral prompts")
    horizon = int(ctx.cfg["E9_POSITIONS"])

    excluded: dict[str, str] = {}
    units = []
    for p in chosen:
        spec = prompts.neutral_for_e9(ctx.backend, p)
        cont = own_continuation(ctx, code, spec.text, horizon, ctx.seed(code, "continuation", p.id),
                                unit=p.id, greedy=True)
        if not cont.text:
            excluded[p.id] = "the unsteered greedy continuation is empty; nothing to teacher-force"
            continue
        units.append((p, spec, cont))
    if not units:
        return out(missing="every neutral prompt was excluded", extra={"excluded": len(excluded),
                                                                       "excluded_units": excluded})

    ivs = [intervention_for(ctx, cell, spec, mode) for _, spec, _ in units]
    reqs_s = [ScoreRequest(spec.text, cont.text, iv) for (_, spec, cont), iv in zip(units, ivs)]
    reqs_u = [ScoreRequest(spec.text, cont.text, None) for _, spec, cont in units]
    cand_s = candidate_logprobs(ctx, reqs_s, toks.ids)
    cand_u = candidate_logprobs(ctx, reqs_u, toks.ids)
    sc_s = score_requests(ctx, reqs_s)
    sc_u = score_requests(ctx, reqs_u)

    per: list[dict[str, Any]] = []
    for k, (p, _, cont) in enumerate(units):
        m = min(horizon, len(cand_s[k]), len(cand_u[k]))
        mass_s = logsumexp_rows(cand_s[k][:m])
        mass_u = logsumexp_rows(cand_u[k][:m])
        lift = float(np.mean(mass_s - mass_u)) if m else math.nan
        d_ent = (float(np.mean(sc_s[k].entropies[:m]) - np.mean(sc_u[k].entropies[:m]))
                 if m else math.nan)
        if not (math.isfinite(lift) and math.isfinite(d_ent)):
            excluded[p.id] = "a non-finite log-probability or entropy along the continuation"
            continue
        per.append({"prompt_id": p.id, "lift": lift, "entropy_change": d_ent, "positions": m,
                    "null_logmass": float(np.mean(mass_u)),
                    "retokenised": tuple(sc_u[k].token_ids) != cont.token_ids,
                    "transcript_id": cont.transcript_id})
    if not per:
        return out(missing="no prompt gave a finite lift", extra={"excluded": len(excluded),
                                                                  "excluded_units": excluded})
    value, se = mean_se([r["lift"] for r in per])
    d_ent, d_ent_se = mean_se([r["entropy_change"] for r in per])
    null, null_se = mean_se([r["null_logmass"] for r in per])
    return out(value=value, n=len(per), extra={
        "se": se, "entropy_change": d_ent, "entropy_change_se": d_ent_se,
        "null_logmass": null, "null_logmass_se": null_se, "concept": concept,
        "token_ids": list(toks.ids), "dropped_variants": len(toks.dropped),
        "positions": horizon, "per_prompt": per,
        "excluded": len(excluded), "excluded_units": excluded})


def read_e10(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
             mode: str | None = None, pass_: str | None = None, behaviour: str | None = None,
             items: contrastive.ABSet | None = None) -> Readout:
    """e10 (behavioural propensity shift, logits; higher = more of the behaviour).

    The items are `items` when given, else the embedded set for `behaviour`, which defaults to the
    vector's own class. n is the number of items (all of the set by default). Steering starts
    after the user's instruction (CAA's arrangement, `m5.prompts.ab_item`) unless the cell's
    position policy says otherwise."""
    code = "e10"
    check_args(code, variant=variant, variant_aware=False, pass_=pass_)
    mode = resolve_mode(code, cell, mode, kind="marker", allowed=E10_MODES)

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    if cell is None:
        return out(missing="the null arm names no vector and so no behaviour; e10's unsteered gaps "
                           "are reported beside every steered reading (extra['null_gap'])")
    if mode == "generated":
        return out(missing=GENERATED_ONLY)
    if items is None:
        if behaviour is None and cell.vector_id in ctx.vector_classes:
            behaviour = ctx.vector_classes[cell.vector_id]
        if behaviour not in E10_SETS:
            return out(missing=f"no embedded A/B items for behaviour {behaviour!r}; name one of "
                               f"{sorted(E10_SETS)} or pass items= (m5.data.contrastive.load_caa)")
        items = E10_SETS[behaviour]
    elif behaviour is not None and items.vector_class != behaviour:
        raise ValueError(f"e10: the items of {items.set_id!r} measure {items.vector_class!r}, "
                         f"not {behaviour!r}")
    behaviour = items.vector_class
    chosen = take(code, items.items, n, what=f"A/B items of {items.set_id}")
    specs = [prompts.ab_item(ctx.backend, it) for it in chosen]
    opts = options(ctx.backend, ["A", "B"])
    ivs = [intervention_for(ctx, cell, s, mode) for s in specs]
    texts = [s.text for s in specs]
    lo_s, _ = option_logprobs(ctx, texts, ivs, opts)
    lo_u, _ = option_logprobs(ctx, texts, [None] * len(texts), opts)

    per, excluded = [], {}
    for k, it in enumerate(chosen):
        m, o = opts.index(it.matching_letter), opts.index(it.not_matching.strip("()"))
        with np.errstate(invalid="ignore"):
            gap_s = float(lo_s[k, m] - lo_s[k, o])
            gap_u = float(lo_u[k, m] - lo_u[k, o])
        if not (math.isfinite(gap_s) and math.isfinite(gap_u)):
            excluded[it.id] = UNREADABLE_OPTIONS
            continue
        per.append({"item_id": it.id, "shift": gap_s - gap_u, "gap": gap_s, "null_gap": gap_u,
                    "matching": it.matching_letter})
    if not per:
        return out(missing="no item gave a finite logit gap", extra={"excluded": len(excluded),
                                                                    "excluded_units": excluded})
    value, se = mean_se([r["shift"] for r in per])
    gap, gap_se = mean_se([r["gap"] for r in per])
    null_gap, null_gap_se = mean_se([r["null_gap"] for r in per])
    return out(value=value, n=len(per), extra={
        "se": se, "gap": gap, "gap_se": gap_se, "null_gap": null_gap, "null_gap_se": null_gap_se,
        "behaviour": behaviour, "item_set": items.set_id, "per_item": per,
        "multi_token_forms": opts.multi_token_forms, "excluded": len(excluded),
        "excluded_units": excluded})
