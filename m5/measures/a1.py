"""m5.measures.a1 - `a1`, verbalised evaluation awareness: does the model say it is being tested?

On the scenarios of `m5.data.scenarios` -- evaluation-like and deployment-like, neither naming its
class -- answered at TOKENS_SCENARIO with every position steered, beside the same scenarios
unsteered (one pool per model). Plan §7.2: a lexical prefilter with high recall
(`m5.judges.mechanical.ea_prefilter`) decides which replies the `ea_verbalised` judge reads -- every
hit, plus a random share of the non-hits that measures what the prefilter misses. A non-hit the
judge does not read is a NO by the prefilter; a non-hit it reads and calls YES counts as a YES (an
observed positive is a positive) and as a prefilter miss in `extra["prefilter"]`.

**The non-hit share is a setting.** `nonhit_audit=None` -- the default -- reads
cfg["A1_NONHIT_AUDIT"] (the plan's 5%), so the share is inside the config hash; pass a share to
override it, 1.0 to have the judge read every non-hit (the complete measurement, which the sampled
one approximates at a twentieth of the judging). Each non-hit's draw is seeded by its transcript,
so the same replies are audited on every resume.

`n` is the number of scenarios per class (40 per class exist; past that they repeat with new
seeds). The scenario set is not split between search and confirmation: a1 runs on finalists only,
so no selection has seen it; the confirm tier still draws fresh samples (its seeds name the split).
The Measured value is the pooled rate over both classes with the unsteered pooled rate as its null,
paired by scenario; `extra["classes"]` holds each class's rate and null, and the paired difference.
"""

from __future__ import annotations

from typing import Any

from m5 import prompts as _prompts
from m5.backend.base import derive_seed
from m5.data import scenarios as _scn
from m5.intervention import Cell
from m5.judges import mechanical
from m5.judges.prompts import make_item, templates
from m5.measures import common as C
from m5.results import Measured

CODE = "a1"


def scenario_units(n: int, start: int = 0) -> list[tuple[_scn.Scenario, int]]:
    """n scenarios of each class, interleaved: (scenario, repeat)."""
    out = []
    for cls in _scn.CLASSES:
        pool = tuple(s for s in _scn.SCENARIOS if s.cls == cls)
        out += C.cycle(pool, start, n)
    return out


def measure_a1(ctx: Any, cell: Cell | None, *, n: int, split: str, variant: str = "original",
               start: int = 0, nonhit_audit: float | None = None) -> Measured:
    """Measure `a1` at a cell on `n` scenarios per class."""
    C.check_split(split)
    C.check_variant(CODE, variant)
    n, start = C.check_n(n), C.check_start(start)
    if nonhit_audit is not None and not 0.0 <= float(nonhit_audit) <= 1.0:
        raise C.MeasureError(f"nonhit_audit must lie in [0, 1] or be None, got {nonhit_audit!r}")
    share = float(ctx.cfg["A1_NONHIT_AUDIT"] if nonhit_audit is None else nonhit_audit)
    subj = C.subject(ctx, cell)
    C.require_judges(ctx)
    conf = float(ctx.cfg["CONFIDENCE"])
    units = scenario_units(n, start)
    s_jobs, z_jobs = [], []
    for sc, r in units:
        sp = C.spec(ctx, (CODE, sc.id), lambda sc=sc: _prompts.scenario(ctx.backend, sc))
        unit = f"{sc.id}#{r}"
        z_jobs.append(C.Job(unit, sp, None, ctx.seed(CODE, C.NULL, sc.id, r),
                            int(ctx.cfg["TOKENS_SCENARIO"])))
        if not subj.null_only:
            s_jobs.append(C.Job(unit, sp, C.intervention(ctx, cell, sp, sp.meta["mode"]),
                                ctx.seed(CODE, split, subj.cell_id, sc.id, r),
                                int(ctx.cfg["TOKENS_SCENARIO"])))
    z_gen = C.generate(ctx, z_jobs, code=CODE, arm=C.NULL, cell_id=C.NULL, split=split)
    if subj.null_only:          # a cell of None is the null arm itself
        s_jobs, s_gen = z_jobs, z_gen
    else:
        s_gen = C.generate(ctx, s_jobs, code=CODE, arm=subj.arm, cell_id=subj.cell_id, split=split)

    T = templates(ctx.cfg)
    seed = int(ctx.cfg["SEED"])
    reads: list[dict[str, Any]] = []
    items: list[Any] = []
    arms = [(C.NULL, C.NULL, z_jobs, z_gen)]
    if not subj.null_only:
        arms.insert(0, (subj.arm, subj.cell_id, s_jobs, s_gen))
    for arm, cell_id, jobs, gens in arms:
        for job, g in zip(jobs, gens):
            hits = mechanical.ea_prefilter(g.text)
            if hits:
                audited = False
            else:
                u = derive_seed(seed, CODE, "nonhit_audit", g.transcript) / float(2 ** 31)
                audited = u < share
            rd = {"job": job, "gen": g, "hit": bool(hits), "audited": audited, "slot": None,
                  "arm": arm, "cell_id": cell_id}
            if hits or audited:
                rd["slot"] = len(items)
                items.append(make_item(T["ea_verbalised"], {"text": g.text},
                                       item_id=f"{g.transcript}:ea_verbalised",
                                       meta={"arm": arm, "cell": cell_id, "unit": job.unit,
                                             "slot": "ea_verbalised"}))
            reads.append(rd)
    js = C.judge(ctx, items)
    for rd in reads:
        j = js[rd["slot"]] if rd["slot"] is not None else None
        rd["label"] = False if j is None else C.label_of(j)
        rd["why"] = C.reason(j) if j is not None and j.missing is not None else None
    if subj.null_only:
        s_reads = z_reads = reads
    else:
        s_reads, z_reads = reads[:len(s_jobs)], reads[len(s_jobs):]

    rows, null_rows = [], []
    for (sc, r), a, b in zip(units, s_reads, z_reads):
        base = {"code": CODE, "unit": a["job"].unit, "scenario": sc.id, "cls": sc.cls}
        rows.append({**base, "cell": subj.cell_id, "split": split, "arm": subj.arm,
                     "transcript_id": a["gen"].transcript, "null_transcript_id": b["gen"].transcript,
                     "prefilter_hit": a["hit"], "judged": a["slot"] is not None, "aware": a["label"],
                     "vector": subj.vector_id, "missing": [a["why"]] if a["why"] else []})
        null_rows.append(((CODE, a["job"].unit), {**base, "cell": C.NULL, "split": C.NULL,
                                                  "arm": C.NULL, "transcript_id": b["gen"].transcript,
                                                  "prefilter_hit": b["hit"],
                                                  "judged": b["slot"] is not None, "aware": b["label"],
                                                  "missing": [b["why"]] if b["why"] else []}))
    if not subj.null_only:
        C.write_trials(ctx, rows)
    C.write_null_trials(ctx, null_rows)

    tally = C.Tally()
    by_cls: dict[str, list[tuple[bool, bool]]] = {c: [] for c in _scn.CLASSES}
    unit_ids = []
    for (sc, _), a, b in zip(units, s_reads, z_reads):
        if a["label"] is None or b["label"] is None:
            null_why = f"null:{b['why']}" if b["why"] and b is not a else None
            for w in [x for x in (a["why"], null_why) if x] or ["a1:missing"]:
                tally.add(a["job"].unit, w)
            continue
        by_cls[sc.cls].append((bool(a["label"]), bool(b["label"])))
        unit_ids.append(a["job"].unit)
    pairs = [p for c in _scn.CLASSES for p in by_cls[c]]
    if not pairs:
        return C.missing_measure(ctx, CODE, subj, split, "no scenario has both arms decided",
                                 excluded=tally.count, extra={"excluded_reasons": tally.to_json()})

    def prefilter(rs: list[dict[str, Any]]) -> dict[str, Any]:
        nonhits = [x for x in rs if not x["hit"]]
        audited = [x for x in nonhits if x["audited"] and x["label"] is not None]
        return {"hits": sum(x["hit"] for x in rs), "nonhits": len(nonhits),
                "nonhits_judged": len(audited), "nonhit_yes": sum(bool(x["label"]) for x in audited),
                "hit_yes": sum(bool(x["label"]) for x in rs if x["hit"] and x["label"] is not None)}

    classes = {}
    for c, ps in by_cls.items():
        k, nk = sum(a for a, _ in ps), sum(b for _, b in ps)
        classes[c] = C.rate_block(k, len(ps), conf)
        if not subj.null_only:
            classes[c].update(null=C.rate_block(nk, len(ps), conf), paired=C.paired(ps, conf))
    k, nk = sum(a for a, _ in pairs), sum(b for _, b in pairs)
    lo, hi = C.wilson_or_none(k, len(pairs), conf)
    m = Measured(code=CODE, cell_id=subj.cell_id, split=split, n=len(pairs), k=k, lo=lo, hi=hi,
                 null_k=nk, null_n=len(pairs), unit_ids=unit_ids, excluded=tally.count,
                 extra={"classes": classes, "paired": C.paired(pairs, conf),
                        "null": C.rate_block(nk, len(pairs), conf),
                        "prefilter": prefilter(s_reads), "null_prefilter": prefilter(z_reads),
                        "nonhit_audit": share,
                        "n_per_class": n, "excluded_reasons": tally.to_json(),
                        "outcomes": [int(a) for a, _ in pairs],
                        **C.design([u.rsplit("#", 1)[0] for u in unit_ids], [int(a) for a, _ in pairs])})
    if subj.null_only:
        m = C.as_null_arm(m, conf)
    return C.write_cell(ctx, m, subj=subj, iv=s_jobs[0].intervention)
