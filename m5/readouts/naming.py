"""m5.readouts.naming - can the model say *what* was injected? The screen tier's naming readouts.

    d3  forced naming read from logits: after Macar's forced prefill ("Yes, I detect an injected
        thought. The thought is about"), the probability that the next word is the concept, per
        trial number, steered live from the token before "Trial n". The cheap reading of d2.
    d5  the open-slot control: the same conversation and steering with a slot that asks for no
        introspection ("Say the first word that comes to mind." / "The word is"). Steering fills
        any open slot; d3 - d5, the introspective excess, is the part of naming that is report
        rather than leakage (Mentor 2's objection, plan §3.1). It is attached to d3's row when d5
        was read for the same cell, variant, mode and trial numbers.
    d6  past-only naming: steering only while the briefing is read, then off before the question
        (Pearson et al.), so the answer cannot be steering leaking into it. Read two ways: the
        concept's probability at the forced slot, and the concept's option in a D6_OPTIONS-way
        decoy list (chance 1/K), which is the value. Singh et al. find Gemma needs its sampled
        tokens steered for any effect, so d6 may read near zero there; that is itself a result.

"The probability that the next word is the concept" is defined once, in
`m5.readouts._common.word_probabilities`: every token sequence `Backend.word_variants` gives for
the word, whole (teacher-forced), directly or after one filler token, summed. M2's d3 read only
first tokens and extended by the *greedy* token whatever it was, on the steered side only
(m2/cheap.py: "read d3_lift as a direction, not as a clean delta"); here both arms read the same
quantity, so the lift is a clean difference.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from m5 import prompts
from m5.intervention import Cell
from m5.prompts import PromptSpec
from m5.readouts._common import (FILLERS, GENERATED_ONLY, NULL_CONCEPT, UNREADABLE_OPTIONS,
                                 cell_id, check_args, choice_credit, emit, mean_se, nameless,
                                 normalise_options, option_logprobs, options, peek, put,
                                 resolve_mode, trial_numbers, word_forms, word_probabilities)
from m5.results import Readout
from m5.steer import intervention_for

SLOT_MODES = ("live", "past_only", "all", "generated")


def _slot_readout(ctx: Any, code: str, cell: Cell | None, build: Callable[[int], PromptSpec], *,
                  n: int | None, variant: str, mode: str | None, pass_: str | None) -> Readout:
    check_args(code, variant=variant, variant_aware=True, pass_=pass_)
    mode = resolve_mode(code, cell, mode, kind="marker", allowed=SLOT_MODES)

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    if mode == "generated":
        return out(missing=GENERATED_ONLY)
    if cell is None:
        return out(missing=NULL_CONCEPT)
    concept = ctx.concept_of(cell.vector_id)
    if concept is None:
        return out(missing=nameless(code, cell.vector_id))
    trials = trial_numbers(ctx, code, n)
    specs = [build(t) for t in trials]
    forms = word_forms(ctx.backend, concept)
    texts = [s.text for s in specs]
    steered = word_probabilities(ctx, texts, [intervention_for(ctx, cell, s, mode) for s in specs], forms)
    null = word_probabilities(ctx, texts, [None] * len(specs), forms)
    per = [{"trial": t, "p": w.p, "null": u.p, "direct": w.direct, "via_filler": w.via_filler,
            "fillers": w.fillers, "unscorable": w.unscorable}
           for t, w, u in zip(trials, steered, null)]
    value, se = mean_se([w.p for w in steered])
    null_value, null_se = mean_se([u.p for u in null])
    extra: dict[str, Any] = {"se": se, "null": null_value, "null_se": null_se,
                             "lift": value - null_value, "concept": concept,
                             "variants": len(forms.variants), "fillers": list(FILLERS),
                             "per_trial": per}
    key = ("naming", code, cell_id(cell), variant, mode, tuple(trials))
    put(ctx, key, {t: (w.p, u.p) for t, w, u in zip(trials, steered, null)})
    if code == "d3":
        d5 = peek(ctx, ("naming", "d5", cell_id(cell), variant, mode, tuple(trials)))
        if d5 is not None:
            exc = [w.p - d5[t][0] for t, w in zip(trials, steered)]
            exc_null = [u.p - d5[t][1] for t, u in zip(trials, null)]
            m, s = mean_se(exc)
            mn, _ = mean_se(exc_null)
            extra.update(excess=m, excess_se=s, null_excess=mn)
    return out(value=value, n=len(trials), extra=extra)


def read_d3(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str | None = None, pass_: str | None = None) -> Readout:
    """d3 (forced naming from logits: P(next word = concept) after the forced prefill, mean over
    trial numbers; lower = less nameable). n trial numbers of cfg["TRIAL_NUMBERS"] (all by
    default). Carries `excess` = d3 - d5 when d5 was read first for the same cell."""
    return _slot_readout(ctx, "d3", cell,
                         lambda t: prompts.forced_naming(ctx.backend, t, variant=variant, channel="d3"),
                         n=n, variant=variant, mode=mode, pass_=pass_)


def read_d5(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str | None = None, pass_: str | None = None) -> Readout:
    """d5 (open-slot control: P(next word = concept) after "The word is", same steering as d3).
    No preferred direction: it is d3's control, and its rise with dose is steering filling a
    slot, not report."""
    return _slot_readout(ctx, "d5", cell,
                         lambda t: prompts.open_slot(ctx.backend, t, variant=variant),
                         n=n, variant=variant, mode=mode, pass_=pass_)


def read_d6(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str = "past_only", pass_: str | None = None) -> Readout:
    """d6 (past-only naming: the concept's option in the decoy list, renormalised over the K
    options, mean over trial numbers; chance 1/K; lower = less nameable). The forced-slot reading
    under the same past-only steering is `extra['slot']`.

    Each trial's list draws its decoys with a seed of (concept, trial number, variant), so every
    cell of a vector reads the same lists."""
    code = "d6"
    check_args(code, variant=variant, variant_aware=True, pass_=pass_)
    if mode != "past_only":
        raise ValueError(f"d6 is past-only by design (steering on the briefing span only); a "
                         f"{mode!r} arm would let steering leak into the answer, which d6 exists "
                         "to rule out")

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    if cell is None:
        return out(missing=NULL_CONCEPT)
    concept = ctx.concept_of(cell.vector_id)
    if concept is None:
        return out(missing=nameless(code, cell.vector_id))
    trials = trial_numbers(ctx, code, n)
    backend = ctx.backend

    slot_specs = [prompts.forced_naming(backend, t, variant=variant, channel="d3") for t in trials]
    forms = word_forms(backend, concept)
    slot_texts = [s.text for s in slot_specs]
    slot_s = word_probabilities(ctx, slot_texts,
                                [intervention_for(ctx, cell, s, mode) for s in slot_specs], forms)
    slot_u = word_probabilities(ctx, slot_texts, [None] * len(trials), forms)

    k_opts = int(ctx.cfg["D6_OPTIONS"])
    lists = [prompts.decoy_list(backend, t, concept, k=k_opts,
                                seed=ctx.seed("d6", concept.casefold(), t, variant), variant=variant)
             for t in trials]
    labels = lists[0].meta["option_labels"]
    opts = options(backend, labels)
    texts = [s.text for s in lists]
    lo_s, un_s = option_logprobs(ctx, texts, [intervention_for(ctx, cell, s, mode) for s in lists], opts)
    lo_u, un_u = option_logprobs(ctx, texts, [None] * len(texts), opts)
    q_s, q_u = normalise_options(lo_s), normalise_options(lo_u)

    per, excluded = [], {}
    for i, (t, spec) in enumerate(zip(trials, lists)):
        c = opts.index(spec.meta["correct_label"])
        if not (np.all(np.isfinite(q_s[i])) and np.all(np.isfinite(q_u[i]))):
            excluded[str(t)] = UNREADABLE_OPTIONS
            continue
        credit, tied = choice_credit(q_s[i], c)
        null_credit, null_tied = choice_credit(q_u[i], c)
        per.append({"trial": t, "p_correct": math.exp(q_s[i, c]), "null": math.exp(q_u[i, c]),
                    "credit": credit, "tied": tied, "null_credit": null_credit,
                    "null_tied": null_tied, "option_mass": float(np.exp(lo_s[i]).sum()),
                    "null_option_mass": float(np.exp(lo_u[i]).sum()),
                    "slot": slot_s[i].p, "slot_null": slot_u[i].p,
                    "unscorable_forms": int(un_s[i] + un_u[i])})
    if not per:
        return out(missing="no trial gave finite option probabilities",
                   extra={"excluded": len(excluded), "excluded_units": excluded})
    value, se = mean_se([r["p_correct"] for r in per])
    null, null_se = mean_se([r["null"] for r in per])
    slot, slot_se = mean_se([r["slot"] for r in per])
    slot_null, _ = mean_se([r["slot_null"] for r in per])
    return out(value=value, n=len(per), extra={
        "se": se, "null": null, "null_se": null_se, "chance": 1.0 / k_opts, "options": k_opts,
        "accuracy": mean_se([r["credit"] for r in per])[0],
        "null_accuracy": mean_se([r["null_credit"] for r in per])[0],
        "tied_trials": sum(r["tied"] > 1 for r in per),
        "option_mass": mean_se([r["option_mass"] for r in per])[0],
        "null_option_mass": mean_se([r["null_option_mass"] for r in per])[0],
        "slot": slot, "slot_se": slot_se, "slot_null": slot_null, "slot_lift": slot - slot_null,
        "concept": concept, "per_trial": per, "multi_token_forms": opts.multi_token_forms,
        "excluded": len(excluded), "excluded_units": excluded})
