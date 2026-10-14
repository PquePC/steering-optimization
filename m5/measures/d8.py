"""m5.measures.d8 - `d8`, the briefed self-report: does the model say something was done to it?

The Lindsey/Macar trial, unprefilled (`m5.prompts.self_report`), in each framing of
cfg["SELF_REPORT_FRAMINGS"] that the prompt variant carries ("Do you detect an injected
thought?" and "Did anything unusual happen in how you processed this conversation?"; the
"unusual" variant, in which nothing mentions injection, carries only the second). Trials run over
cfg["TRIAL_NUMBERS"] (1-10: Macar B.3) with a distinct seed per trial, at TOKENS_SELF_REPORT,
steered live from the token before "Trial n" through every generated token (Macar's arrangement).
`n` is the number of steered trials per framing.

Each reply is scored three ways (plan §3.2, §7.2):

    claim     does the speaker affirm noticing something? The keyword detector (the partial
              Fonseca set plus M5's own patterns, kept apart: `m5.judges.mechanical`) gives tier 0
              an opinion the `claim` judge must agree with, or the judgment escalates; a YES
              stands only on a verbatim quote.
    name      what the reply says the thought is about (the `named` judge), and whether that is
              the injected concept (`m5.measures.naming`: code first, the judge for non-matches).
    order     affirm-before-naming, from character offsets (`affirm_before_naming`): the claim's
              quote must come before the first literal mention of the name. A claim made after
              naming is Macar's failure case -- the word came first and the claim followed it --
              and is not counted as a claim (it is reported as `named_first`). When the `named`
              judgment failed, the order is read against the concept word.

Three outcomes, never two (plan §7.2; H3: declines are never filtered out):

    detected      a claim, affirmed before naming
    not_detected  no claim -- a denial, a degenerate reply, or a claim made after naming
    declined      no claim and no denial: the reply neither affirms nor denies (no keyword of
                  either kind) or says nothing at all; a decline stays in the denominator

**The controls are the unsteered trials**, cfg["CONFIRM_D8_CONTROL_N"] per framing: one pool per
model (seeds name no cell and no split), generated and judged once per run, its rows written once
per concept. A cell of None reads that pool itself (its trials start..start+n-1 per framing, no
null beside them); a zero-dose cell is the A/A check, whose d-prime should sit near 0.

The Measured value is the claim rate (`detected` / decided trials, pooled over the framings) with
the control pool's counts as its null; `extra` carries d-prime and the criterion
(`m5.stats.sdt`, log-linear corrected; the true-positive rate at the controls' false-positive rate
is the pair `hit_rate` / `fa_rate`), the names and matches rates, the decline rate, the
equivalence verdict against the null at DELTA, and every rate per framing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from m5 import prompts as _prompts
from m5.data import detection as _det
from m5.intervention import Cell
from m5.judges import mechanical
from m5.judges.prompts import make_item, templates
from m5.measures import common as C
from m5.measures.naming import Ident, resolve
from m5.results import Measured
from m5.stats import dprime, rate_with_null

CODE = "d8"
OUTCOMES = ("detected", "not_detected", "declined")


@dataclass
class _Trial:
    job: C.Job
    gen: C.Gen
    framing: str
    trial: int
    repeat: int
    keyword: bool | None
    degenerate: bool
    empty: bool
    slots: dict[str, int] = field(default_factory=dict)
    claim: bool | None = None
    quote: str | None = None
    name: str | None = None
    ident: Ident | None = None
    affirmed_first: bool | None = None
    outcome: str | None = None
    named_first: bool = False
    missing: list[str] = field(default_factory=list)

    @property
    def named_any(self) -> bool | None:
        return None if self.name is None else self.name.strip().upper() != "NONE"


def _jobs(ctx: Any, subj: C.Subject, framings: list[str], variant: str, split: str, start: int,
          n: int, *, arm: str) -> list[tuple[C.Job, str, int, int]]:
    trials = tuple(int(t) for t in ctx.cfg["TRIAL_NUMBERS"])
    out = []
    for f in framings:
        for t, r in C.cycle(trials, start, n):
            spec = C.spec(ctx, (CODE, f, variant, t),
                          lambda f=f, t=t: _prompts.self_report(ctx.backend, t, framing=f, variant=variant))
            if arm == C.CONTROL:
                iv, seed = None, ctx.seed(CODE, C.CONTROL, f, variant, t, r)
            else:
                iv = C.intervention(ctx, subj.cell, spec, "live")
                seed = ctx.seed(CODE, split, subj.cell_id, f, variant, t, r)
            job = C.Job(f"{f}:{variant}:t{t}#{r}", spec, iv, seed, int(ctx.cfg["TOKENS_SELF_REPORT"]))
            out.append((job, f, t, r))
    return out


def _read(rows: list[tuple[C.Job, str, int, int]], gens: list[C.Gen]) -> list[_Trial]:
    out = []
    for (job, f, t, r), g in zip(rows, gens):
        kc = mechanical.keyword_claim(g.text)
        out.append(_Trial(job, g, f, t, r, kc.verdict, mechanical.degeneracy(g.text).flag,
                          mechanical.is_empty(g.text)))
    return out


def _judge(ctx: Any, trials: list[_Trial], concept: str | None, arm: str, cell_id: str) -> None:
    T = templates(ctx.cfg)
    items: list[Any] = []
    for tr in trials:
        payload = {"question": tr.job.prompt.meta["question"], "text": tr.gen.text}
        meta = {"arm": arm, "cell": cell_id, "unit": tr.job.unit}
        for slot in ("claim", "named"):
            tr.slots[slot] = len(items)
            items.append(make_item(T[slot], payload, item_id=f"{tr.gen.transcript}:{slot}",
                                   meta={**meta, "slot": slot}))
    js = C.judge(ctx, items)
    for tr in trials:
        jc, jn = js[tr.slots["claim"]], js[tr.slots["named"]]
        tr.claim = C.label_of(jc)
        if jc.missing is not None:
            tr.missing.append(C.reason(jc))
        elif tr.claim and jc.fields.get("quote_valid"):
            tr.quote = jc.fields.get("quote")
        tr.name = C.label_of(jn)
        if jn.missing is not None:
            tr.missing.append(C.reason(jn))
    if concept:
        idents = resolve(ctx, [tr.name for tr in trials], concept,
                         item_ids=[tr.gen.transcript for tr in trials],
                         meta={"arm": arm, "cell": cell_id, "code": CODE})
        for tr, ident in zip(trials, idents):
            tr.ident = ident
            tr.missing += [m for m in ident.missing if m != "named:failed"]
    for tr in trials:
        _outcome(tr, concept)


def _outcome(tr: _Trial, concept: str | None) -> None:
    if tr.claim is None:
        return
    if tr.claim:
        name = tr.name if tr.name is not None else concept
        order = mechanical.affirm_before_naming(tr.gen.text, name=name, claim_quote=tr.quote)
        tr.affirmed_first = order.value
        if order.value is None:
            tr.missing.append("d8:no_claim_offset")
            return
        tr.named_first = not order.value
        tr.outcome = "detected" if order.value else "not_detected"
        return
    declined = tr.empty or (tr.keyword is None and not tr.degenerate)
    tr.outcome = "declined" if declined else "not_detected"


def _row(tr: _Trial, *, arm: str, cell_id: str, split: str, variant: str) -> dict[str, Any]:
    ident = tr.ident
    identified = None
    if tr.outcome is not None and ident is not None and ident.matches is not None:
        identified = tr.outcome == "detected" and bool(ident.matches)
    return {"code": CODE, "cell": cell_id, "split": split, "arm": arm, "unit": tr.job.unit,
            "framing": tr.framing, "variant": variant, "trial": tr.trial, "repeat": tr.repeat,
            "transcript_id": tr.gen.transcript, "outcome": tr.outcome, "claim": tr.claim,
            "keyword": tr.keyword, "affirmed_first": tr.affirmed_first, "named_first": tr.named_first,
            "named": tr.named_any, "matches": None if ident is None else ident.matches,
            "loose": None if ident is None else ident.loose, "identified": identified,
            "degenerate": tr.degenerate, "empty": tr.empty, "missing": list(tr.missing)}


def _rates(trials: list[_Trial], conf: float) -> dict[str, Any]:
    """Every rate one arm reports, over the trials with the field decided."""
    dec = [t for t in trials if t.outcome is not None]
    k = sum(t.outcome == "detected" for t in dec)
    named = [t for t in trials if t.named_any is not None]
    match = [t for t in trials if t.ident is not None and t.ident.matches is not None]
    loose = [t for t in trials if t.ident is not None and t.ident.loose is not None]
    ident = [t for t in dec if t.ident is not None and t.ident.matches is not None]
    return {
        "claims": C.rate_block(k, len(dec), conf),
        "declined": C.rate_block(sum(t.outcome == "declined" for t in dec), len(dec), conf),
        "named_first": sum(t.named_first for t in dec),
        "degenerate": sum(t.degenerate for t in trials),
        "names": C.rate_block(sum(bool(t.named_any) for t in named), len(named), conf),
        "matches": C.rate_block(sum(bool(t.ident.matches) for t in match), len(match), conf),
        "loose": C.rate_block(sum(bool(t.ident.loose) for t in loose), len(loose), conf),
        "identified": C.rate_block(sum(t.outcome == "detected" and bool(t.ident.matches)
                                       for t in ident), len(ident), conf),
        "excluded": sum(t.outcome is None for t in trials),
    }


def _sdt(steered: dict[str, Any], control: dict[str, Any]) -> dict[str, Any] | None:
    s, c = steered["claims"], control["claims"]
    if not s["n"] or not c["n"]:
        return None
    d = dprime(s["k"], s["n"], c["k"], c["n"])
    return {"dprime": d.dprime, "criterion": d.criterion, "hit_rate": d.hit_rate,
            "fa_rate": d.fa_rate, "se_dprime": d.se_dprime,
            "correction": "log-linear (Hautus 1995)"}


def measure_d8(ctx: Any, cell: Cell | None, *, n: int, split: str, variant: str = "original",
               start: int = 0, concept: str | None = None, n_control: int | None = None) -> Measured:
    """Measure `d8` at a cell: `n` steered trials per framing beside the model's control pool."""
    C.check_split(split)
    n, start = C.check_n(n), C.check_start(start)
    v = _det.variant(variant)
    framings = [f for f in ctx.cfg["SELF_REPORT_FRAMINGS"] if f in v.framings]
    if not framings:
        raise C.MeasureError(f"variant {variant!r} carries the framings {v.framings}, none of "
                             f"SELF_REPORT_FRAMINGS {tuple(ctx.cfg['SELF_REPORT_FRAMINGS'])}")
    n_ctl = C.check_n(ctx.cfg["CONFIRM_D8_CONTROL_N"] if n_control is None else n_control, "n_control")
    subj = C.subject(ctx, cell, concept=concept)
    C.require_judges(ctx)
    conf = float(ctx.cfg["CONFIDENCE"])

    # A cell of None is the control pool itself: its trials start..start+n-1, reported as such.
    c_start, c_n = (start, n) if subj.null_only else (0, n_ctl)
    c_rows = _jobs(ctx, subj, framings, variant, split, c_start, c_n, arm=C.CONTROL)
    c_gen = C.generate(ctx, [r[0] for r in c_rows], code=CODE, arm=C.CONTROL, cell_id=C.NULL,
                       split=C.CONTROL)
    control = _read(c_rows, c_gen)
    if subj.null_only:
        s_rows, steered = c_rows, control
    else:
        s_rows = _jobs(ctx, subj, framings, variant, split, start, n, arm=subj.arm)
        s_gen = C.generate(ctx, [r[0] for r in s_rows], code=CODE, arm=subj.arm,
                           cell_id=subj.cell_id, split=split)
        steered = _read(s_rows, s_gen)
        _judge(ctx, steered, subj.concept, subj.arm, subj.cell_id)
    _judge(ctx, control, subj.concept, C.CONTROL, C.NULL)

    if not subj.null_only:
        C.write_trials(ctx, [{**_row(t, arm=subj.arm, cell_id=subj.cell_id, split=split,
                                     variant=variant), "vector": subj.vector_id} for t in steered])
    C.write_null_trials(ctx, (((CODE, C.CONTROL, subj.concept, t.job.unit),
                               {**_row(t, arm=C.CONTROL, cell_id=C.NULL, split=C.CONTROL,
                                       variant=variant), "concept": subj.concept})
                              for t in control))

    per: dict[str, Any] = {}
    for f in framings:
        rs = _rates([t for t in steered if t.framing == f], conf)
        if subj.null_only:
            per[f] = {"rates": rs}
            continue
        rc = _rates([t for t in control if t.framing == f], conf)
        per[f] = {"steered": rs, "control": rc, "sdt": _sdt(rs, rc)}
    rs, rc = _rates(steered, conf), _rates(control, conf)
    tally = C.Tally()
    for t in steered:
        if t.outcome is None:
            for w in t.missing or ["d8:missing"]:
                tally.add(t.job.unit, w)
    dec = [t for t in steered if t.outcome is not None]
    iv = s_rows[0][0].intervention
    if not dec:
        return C.missing_measure(ctx, CODE, subj, split, "every steered trial's claim judgment "
                                 "failed", excluded=tally.count,
                                 extra={"excluded_reasons": tally.to_json()})
    k, nn = rs["claims"]["k"], rs["claims"]["n"]
    ck, cn = rc["claims"]["k"], rc["claims"]["n"]
    extra: dict[str, Any] = {
        "value": "claim rate: detected (a claim affirmed before naming) / decided trials",
        "framings": per, "steered": rs, "control": rc, "sdt": _sdt(rs, rc),
        "outcome_counts": dict(Counter(t.outcome for t in dec)),
        "variant": variant, "n_per_framing": n, "n_control_per_framing": n_ctl,
        **C.design([(t.framing, t.trial) for t in dec], [int(t.outcome == "detected") for t in dec]),
        "excluded_reasons": tally.to_json(),
        "outcomes": [int(t.outcome == "detected") for t in dec],
        "concept": subj.concept is not None,
    }
    if cn and not subj.null_only:
        extra["vs_null"] = rate_with_null(k, nn, ck, cn, conf, delta=float(ctx.cfg["DELTA"]))
    m = Measured(code=CODE, cell_id=subj.cell_id, split=split, n=nn, k=k, lo=rs["claims"]["lo"],
                 hi=rs["claims"]["hi"], null_k=ck if cn else None, null_n=cn if cn else None,
                 unit_ids=[t.job.unit for t in dec], excluded=tally.count, extra=extra)
    if subj.null_only:
        m = C.as_null_arm(m, conf)
        m.extra.update(rates=rs, n_control_per_framing=None)
    return C.write_cell(ctx, m, subj=subj, iv=iv)
