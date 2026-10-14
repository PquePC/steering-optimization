"""m5.readouts.noticing - does the model indicate that something was done to it? The screen tier.

    d7  bias-controlled yes/no. Per trial number, the logit gap logit(Yes) - logit(No) (each the
        best of its surface forms) on the detection question, on D7_CONTROL_QUESTIONS control
        questions matched to it on their unsteered gap, and on a mismatch question naming a
        concept far from the injected one. Each gap's shift is steered minus unsteered. The value
        is the difference-in-differences: target shift minus the mean control shift. The binary
        question alone is a global yes-shift on open models -- Hahami found r = 0.999 between the
        detection shift and the control shift -- so only the difference means anything
        (Sridharan). The raw target shift, the control shift and the mismatch DiD are reported
        beside it; live and past-only are separate rows (mode).
    d9  localisation. D9_SEGMENTS numbered segments and a "none" option; on each of D9_TRIALS
        trials one segment is steered (its token span exactly) or none is, counterbalanced so
        every segment and "none" is the answer equally often. The value is the mean probability
        of the correct option, renormalised over the K + 1 options; accuracy and chance
        1 / (K + 1) beside it. It needs no concept name and a yes-bias cannot pass it (Hahami;
        Fornasiere; Zou et al.).

**The d7 controls are chosen once per model and prompt variant**, by matching the unsteered gap
of every question in `m5.data.detection.CONTROL_QUESTIONS` (mean over the trial numbers) to the
detection question's, and every cell then reads the same ones. When the profile carries a
selection (`ModelProfile.d7_controls`, {variant: [question ids]}; the setup writes it) that is
used; otherwise the selection is made here, recorded in the `setup` stream and in every d7 row.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from m5 import prompts
from m5.data import detection
from m5.data.words import DECOY_WORDS
from m5.intervention import Cell
from m5.readouts._common import (GENERATED_ONLY, UNREADABLE_OPTIONS, check_args, choice_credit,
                                 emit, mean_se, memo, normalise_options, option_logprobs, options,
                                 resolve_mode, trial_numbers)
from m5.results import Readout
from m5.steer import intervention_for

D7_MODES = ("live", "past_only", "all", "generated")
YES_NO = ("Yes", "No")


@dataclass(frozen=True)
class D7Controls:
    """The control questions d7 reads, and how they were chosen."""

    ids: tuple[str, ...]
    source: str                         # "profile" or "selected"
    gaps: tuple[float, ...] | None      # their mean unsteered gaps (selected here)
    target_gap: float | None            # the detection question's mean unsteered gap


def _gaps(ctx: Any, specs: Sequence[Any], ivs: Sequence[Any]) -> np.ndarray:
    """logit(Yes) - logit(No) per prompt, each the best of its surface forms; not finite where an
    answer has no scorable form."""
    opts = options(ctx.backend, YES_NO)
    lo, _ = option_logprobs(ctx, [s.text for s in specs], ivs, opts)
    with np.errstate(invalid="ignore"):
        return lo[:, opts.index("Yes")] - lo[:, opts.index("No")]


def _profile_selection(ctx: Any, variant: str, k: int) -> D7Controls | None:
    if not ctx.profile.has_selection("d7_controls"):
        return None
    chosen = ctx.profile.selection("d7_controls")
    if not isinstance(chosen, Mapping) or variant not in chosen:
        raise ValueError(f"the profile's d7_controls must map each prompt variant to question ids; "
                         f"it has none for {variant!r}")
    ids = tuple(str(q) for q in chosen[variant])
    known = {q.id for q in detection.CONTROL_QUESTIONS}
    unknown = [q for q in ids if q not in known]
    if unknown or len(ids) != k or len(set(ids)) != k:
        raise ValueError(f"the profile's d7 controls for {variant!r} must be {k} distinct ids of "
                         f"m5.data.detection.CONTROL_QUESTIONS (D7_CONTROL_QUESTIONS); got "
                         f"{list(ids)}, unknown {unknown}")
    return D7Controls(ids, "profile", None, None)


def d7_controls(ctx: Any, variant: str) -> D7Controls:
    """The D7_CONTROL_QUESTIONS control questions for this model and prompt variant: the profile's
    selection when it has one, else the pool questions whose mean unsteered gap over the trial
    numbers is nearest the detection question's (ties by id), chosen once per run."""
    k = int(ctx.cfg["D7_CONTROL_QUESTIONS"])
    pool = detection.CONTROL_QUESTIONS
    if k > len(pool):
        raise ValueError(f"D7_CONTROL_QUESTIONS={k} exceeds the {len(pool)} control questions")
    from_profile = _profile_selection(ctx, variant, k)
    if from_profile is not None:
        return from_profile
    trials = [int(t) for t in ctx.cfg["TRIAL_NUMBERS"]]

    def select() -> D7Controls:
        backend = ctx.backend
        target = [prompts.yes_no(backend, t, kind="target", variant=variant) for t in trials]
        ctrl = [prompts.yes_no(backend, t, kind="control", question=q.id, variant=variant)
                for q in pool for t in trials]
        gaps = _gaps(ctx, target + ctrl, [None] * (len(target) + len(ctrl)))
        target_gap = float(np.mean(gaps[:len(target)]))
        per_q = gaps[len(target):].reshape(len(pool), len(trials)).mean(axis=1)
        order = sorted(range(len(pool)), key=lambda i: (abs(float(per_q[i]) - target_gap), pool[i].id))[:k]
        sel = D7Controls(tuple(pool[i].id for i in order), "selected",
                         tuple(float(per_q[i]) for i in order), target_gap)
        ctx.write("setup", {"quantity": "d7_controls", "variant": variant, "k": k,
                            "target_gap": target_gap, "trial_numbers": trials,
                            "selected": [{"id": q, "gap": g} for q, g in zip(sel.ids, sel.gaps)],
                            "pool": len(pool)})
        return sel

    return memo(ctx, ("d7-controls", variant, k, tuple(trials)), select)


def _mismatch(ctx: Any, trial: int, concept: str | None, variant: str) -> Any:
    """The mismatch question: "...an injected thought about {another concept}?", the other concept
    drawn far from the injected one (`m5.data.words.draw_decoys`), or, for a vector with no
    concept, any decoy-pool word -- every word is "another concept" then."""
    seed = ctx.seed("d7-mismatch", (concept or "").casefold(), trial, variant)
    if concept is not None:
        return prompts.yes_no(ctx.backend, trial, kind="mismatch", concept=concept, seed=seed,
                              variant=variant)
    other = DECOY_WORDS[random.Random(seed).randrange(len(DECOY_WORDS))]
    return prompts.yes_no(ctx.backend, trial, kind="mismatch", other=other, variant=variant)


def read_d7(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str | None = None, pass_: str | None = None) -> Readout:
    """d7 (difference-in-differences of the yes/no logit gap: detection-question shift minus the
    mean control-question shift, logits, mean over trial numbers; lower = less noticing).

    mode "live" steers from the token before "Trial n" (the default for a from-marker cell);
    "past_only" steers the briefing only, and the question attends to it unsteered."""
    code = "d7"
    check_args(code, variant=variant, variant_aware=True, pass_=pass_)
    mode = resolve_mode(code, cell, mode, kind="marker", allowed=D7_MODES)

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    if mode == "generated":
        return out(missing=GENERATED_ONLY)
    trials = trial_numbers(ctx, code, n)
    controls = d7_controls(ctx, variant)
    concept = ctx.concept_of(cell.vector_id) if cell is not None else None
    backend = ctx.backend
    kc = len(controls.ids)
    specs = []
    mismatch_other = []
    for t in trials:
        specs.append(prompts.yes_no(backend, t, kind="target", variant=variant))
        specs += [prompts.yes_no(backend, t, kind="control", question=q, variant=variant)
                  for q in controls.ids]
        mm = _mismatch(ctx, t, concept, variant)
        mismatch_other.append(mm.meta["other"])
        specs.append(mm)
    ivs = [None if cell is None else intervention_for(ctx, cell, s, mode) for s in specs]
    g_s = _gaps(ctx, specs, ivs).reshape(len(trials), kc + 2)
    g_u = _gaps(ctx, specs, [None] * len(specs)).reshape(len(trials), kc + 2)
    with np.errstate(invalid="ignore"):
        shift = g_s - g_u

    per, excluded = [], {}
    for i, t in enumerate(trials):
        if not (np.all(np.isfinite(g_s[i])) and np.all(np.isfinite(g_u[i]))):
            excluded[str(t)] = "a yes/no answer had no scorable form or a non-finite log-probability"
            continue
        c_shift = float(np.mean(shift[i, 1:kc + 1]))
        per.append({"trial": t, "did": float(shift[i, 0]) - c_shift,
                    "target_shift": float(shift[i, 0]), "control_shift": c_shift,
                    "mismatch_shift": float(shift[i, kc + 1]),
                    "mismatch_did": float(shift[i, kc + 1]) - c_shift,
                    "target_gap": float(g_s[i, 0]), "null_target_gap": float(g_u[i, 0]),
                    "null_control_gap": float(np.mean(g_u[i, 1:kc + 1])),
                    "mismatch_other": mismatch_other[i]})
    if not per:
        return out(missing="no trial gave finite yes/no log-probabilities",
                   extra={"excluded": len(excluded), "excluded_units": excluded})
    value, se = mean_se([r["did"] for r in per])

    def m(key: str) -> tuple[float, float | None]:
        return mean_se([r[key] for r in per])

    ts, ts_se = m("target_shift")
    cs, cs_se = m("control_shift")
    md, md_se = m("mismatch_did")
    return out(value=value, n=len(per), extra={
        "se": se, "target_shift": ts, "target_shift_se": ts_se, "control_shift": cs,
        "control_shift_se": cs_se, "mismatch_did": md, "mismatch_did_se": md_se,
        "null_target_gap": m("null_target_gap")[0], "null_control_gap": m("null_control_gap")[0],
        "controls": {"ids": list(controls.ids), "source": controls.source,
                     "gaps": None if controls.gaps is None else list(controls.gaps),
                     "target_gap": controls.target_gap},
        "per_trial": per, "excluded": len(excluded), "excluded_units": excluded})


def read_d9(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str = "segment", pass_: str | None = None) -> Readout:
    """d9 (localisation: mean probability of the correct option among "1".."K" and "none",
    renormalised over the options; chance 1/(K+1); lower = less noticing).

    n is the number of trials (D9_TRIALS by default). Trial i uses trial number
    TRIAL_NUMBERS[i mod 10] and its own sentence draw; its target comes from
    `m5.prompts.counterbalanced_targets`. A "none" trial is not steered at all. A steered trial
    steers exactly the target segment's token span (`spec.span`, which m5.steer reads in its
    "past_only" mode: positions are a span, the answer is not steered)."""
    code = "d9"
    check_args(code, variant=variant, variant_aware=True, pass_=pass_)
    if mode != "segment":
        raise ValueError(f"d9 steers exactly one segment's tokens; mode must be 'segment', got {mode!r}")

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    size = int(ctx.cfg["D9_TRIALS"]) if n is None else n
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"d9: n must be a positive integer, got {n!r}")
    k_seg = int(ctx.cfg["D9_SEGMENTS"])
    cycle = [int(t) for t in ctx.cfg["TRIAL_NUMBERS"]]
    targets = prompts.counterbalanced_targets(size, k_seg, ctx.seed("d9-targets", variant))
    concept = ctx.concept_of(cell.vector_id) if cell is not None else None
    avoid = (concept,) if concept is not None else ()
    backend = ctx.backend
    specs = [prompts.localisation(backend, cycle[i % len(cycle)], k=k_seg, target=targets[i],
                                  seed=ctx.seed("d9", i, variant), variant=variant, avoid=avoid)
             for i in range(size)]
    labels = specs[0].meta["options"]
    opts = options(backend, labels)
    texts = [s.text for s in specs]
    ivs = [None if cell is None or s.meta["target"] is None else intervention_for(ctx, cell, s, "past_only")
           for s in specs]
    lo_s, un_s = option_logprobs(ctx, texts, ivs, opts)
    lo_u, un_u = option_logprobs(ctx, texts, [None] * size, opts)
    q_s, q_u = normalise_options(lo_s), normalise_options(lo_u)

    per, excluded = [], {}
    for i, s in enumerate(specs):
        if not (np.all(np.isfinite(q_s[i])) and np.all(np.isfinite(q_u[i]))):
            excluded[str(i)] = UNREADABLE_OPTIONS
            continue
        c = opts.index(s.meta["answer"])
        credit, tied = choice_credit(q_s[i], c)
        null_credit, null_tied = choice_credit(q_u[i], c)
        per.append({"trial_index": i, "trial": s.meta["trial"], "target": s.meta["target"],
                    "answer": s.meta["answer"],
                    "p_correct": math.exp(q_s[i, c]), "null": math.exp(q_u[i, c]),
                    "credit": credit, "tied": tied, "null_credit": null_credit,
                    "null_tied": null_tied, "option_mass": float(np.exp(lo_s[i]).sum()),
                    "unscorable_forms": int(un_s[i] + un_u[i])})
    if not per:
        return out(missing="no trial gave finite option probabilities",
                   extra={"excluded": len(excluded), "excluded_units": excluded})
    value, se = mean_se([r["p_correct"] for r in per])
    null, null_se = mean_se([r["null"] for r in per])
    acc, acc_se = mean_se([r["credit"] for r in per])
    null_acc, _ = mean_se([r["null_credit"] for r in per])
    return out(value=value, n=len(per), extra={
        "se": se, "null": null, "null_se": null_se, "accuracy": acc, "accuracy_se": acc_se,
        "null_accuracy": null_acc, "tied_trials": sum(r["tied"] > 1 for r in per),
        "chance": 1.0 / (k_seg + 1), "segments": k_seg,
        "none_trials": sum(r["target"] is None for r in per),
        "option_mass": mean_se([r["option_mass"] for r in per])[0],
        "per_trial": per, "multi_token_forms": opts.multi_token_forms,
        "excluded": len(excluded), "excluded_units": excluded})
