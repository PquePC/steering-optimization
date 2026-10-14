"""m5.measures.d2 - `d2`, forced naming, generated: Macar's prefill, then what fills the slot.

Kept for comparability with Macar et al. and with M3 (plan §3.2): the trial over
cfg["TRIAL_NUMBERS"], the reply prefilled with "Yes, I detect an injected thought. The thought is
about" (`m5.prompts.forced_naming`), steered live from the token before "Trial n" through the
generated tokens, TOKENS_NAMING long. Its null is the same trials unsteered, one pool per model.

**Identification is decided in code where code can decide it** (plan §7.2, decision 7). The slot
-- the generated text up to its first sentence end, with "the word", "the concept of" and articles
stripped -- is matched against the concept in code (exact, inflection, curated synonym); only a
slot code cannot match goes to the judges: `named` records what the reply names, and
`m5.measures.naming` matches it (code again, then `same_concept`, then `related` for the loose
rule). The reported rate is the primary rule, `matches`.

Every trial with a decided identification is in the denominator, including replies that collapsed
(degenerate) and replies with nothing after the prefill (unreadable): both are the model failing
to name under steering, as Macar and M3 count them. The corrected rates M3 reported beside it are
in `extra`, each with its own n: `excl_degenerate` and `excl_unreadable` (M3's
`identification_excluding_degenerate` / `_excluding_unreadable`). A trial whose judgment failed is
missing, excluded everywhere and counted. `extra["words"]` holds what was named (the identified
words), steered and unsteered, as counts.

A vector with no concept word has no d2 (naming drops out for nameless vectors): the measure is
missing with that reason.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from m5 import prompts as _prompts
from m5.intervention import Cell
from m5.judges import matching, mechanical
from m5.judges.prompts import make_item, templates
from m5.measures import common as C
from m5.measures.naming import resolve
from m5.results import Measured
from m5.stats import rate_with_null

CODE = "d2"
_SENTENCE_END = re.compile(r"[.!?\n]")


@dataclass
class _Trial:
    job: C.Job
    gen: C.Gen
    trial: int
    repeat: int
    degenerate: bool
    empty: bool
    code_how: str | None = None
    word: str | None = None
    matches: bool | None = None
    loose: bool | None = None
    how: str = ""
    missing: list[str] = field(default_factory=list)

    @property
    def reply(self) -> str:
        return self.job.prompt.meta["prefill"] + self.gen.text


def slot_candidates(generated: str) -> list[str]:
    """What fills the slot, as code reads it: the text up to the first sentence end, and its
    first word, each normalised (frames such as "the word" and articles removed)."""
    slot = matching.normalise(_SENTENCE_END.split(generated, 1)[0])
    out = [slot] if slot else []
    if slot and " " in slot:
        out.append(slot.split()[0])
    return out


def _jobs(ctx: Any, subj: C.Subject, variant: str, split: str, start: int, n: int, *,
          arm: str) -> list[tuple[C.Job, int, int]]:
    out = []
    for t, r in C.cycle(tuple(int(x) for x in ctx.cfg["TRIAL_NUMBERS"]), start, n):
        spec = C.spec(ctx, (CODE, variant, t),
                      lambda t=t: _prompts.forced_naming(ctx.backend, t, variant=variant, channel=CODE))
        if arm == C.NULL:
            iv, seed = None, ctx.seed(CODE, C.NULL, variant, t, r)
        else:
            iv = C.intervention(ctx, subj.cell, spec, "live")
            seed = ctx.seed(CODE, split, subj.cell_id, variant, t, r)
        out.append((C.Job(f"{variant}:t{t}#{r}", spec, iv, seed, int(ctx.cfg["TOKENS_NAMING"])), t, r))
    return out


def _score(ctx: Any, rows: list[tuple[C.Job, int, int]], gens: list[C.Gen], concept: str, arm: str,
           cell_id: str) -> list[_Trial]:
    trials = []
    for (job, t, r), g in zip(rows, gens):
        tr = _Trial(job, g, t, r, mechanical.degeneracy(g.text).flag, mechanical.is_empty(g.text))
        if tr.empty:  # nothing after the prefill: nothing named, and nothing for a judge to read
            tr.word, tr.matches, tr.loose, tr.how = "NONE", False, False, "unreadable"
        else:
            for cand in slot_candidates(g.text):
                how = matching.code_match(cand, concept)
                if how:
                    tr.word, tr.matches, tr.loose, tr.how = cand, True, True, f"code:{how}"
                    break
        trials.append(tr)
    todo = [tr for tr in trials if tr.matches is None]
    if todo:
        T = templates(ctx.cfg)
        items = [make_item(T["named"], {"question": tr.job.prompt.meta["question"], "text": tr.reply},
                           item_id=f"{tr.gen.transcript}:named",
                           meta={"arm": arm, "cell": cell_id, "unit": tr.job.unit, "slot": "named"})
                 for tr in todo]
        js = C.judge(ctx, items)
        names = []
        for tr, j in zip(todo, js):
            names.append(C.label_of(j))
            if j.missing is not None:
                tr.missing.append(C.reason(j))
        idents = resolve(ctx, names, concept, item_ids=[tr.gen.transcript for tr in todo],
                         meta={"arm": arm, "cell": cell_id, "code": CODE})
        for tr, name, ident in zip(todo, names, idents):
            tr.word = name
            tr.matches, tr.loose, tr.how = ident.matches, ident.loose, ident.how
            tr.missing += [m for m in ident.missing if m != "named:failed"]
    return trials


def _row(tr: _Trial, *, arm: str, cell_id: str, split: str, variant: str) -> dict[str, Any]:
    return {"code": CODE, "cell": cell_id, "split": split, "arm": arm, "unit": tr.job.unit,
            "variant": variant, "trial": tr.trial, "repeat": tr.repeat,
            "transcript_id": tr.gen.transcript, "identified": tr.matches, "loose": tr.loose,
            "how": tr.how, "named": None if tr.word is None else tr.word.upper() != "NONE",
            "degenerate": tr.degenerate, "unreadable": tr.empty, "missing": list(tr.missing)}


def _words(trials: list[_Trial]) -> dict[str, int]:
    return C.counter_dict(Counter((t.word or "").casefold() or "?" for t in trials
                                  if t.matches is not None), top=12)


def _block(trials: list[_Trial], conf: float) -> dict[str, Any]:
    dec = [t for t in trials if t.matches is not None]
    nd = [t for t in dec if not t.degenerate]
    rd = [t for t in dec if not t.empty]
    lo = [t for t in dec if t.loose is not None]
    return {"identified": C.rate_block(sum(bool(t.matches) for t in dec), len(dec), conf),
            "excl_degenerate": C.rate_block(sum(bool(t.matches) for t in nd), len(nd), conf),
            "excl_unreadable": C.rate_block(sum(bool(t.matches) for t in rd), len(rd), conf),
            "loose": C.rate_block(sum(bool(t.loose) for t in lo), len(lo), conf),
            "degenerate": sum(t.degenerate for t in dec), "unreadable": sum(t.empty for t in dec),
            "how": C.counter_dict(Counter(t.how for t in dec))}


def measure_d2(ctx: Any, cell: Cell | None, *, n: int, split: str, variant: str = "original",
               start: int = 0, concept: str | None = None) -> Measured:
    """Measure `d2` at a cell: `n` forced-naming trials beside the same trials unsteered."""
    C.check_split(split)
    n, start = C.check_n(n), C.check_start(start)
    subj = C.subject(ctx, cell, concept=concept)
    if subj.concept is None:
        why = (f"vector {subj.vector_id!r} has no concept word; d2 asks for a name (naming drops "
               "out for nameless vectors)") if cell is not None else \
            "no concept to identify: pass concept= to measure d2 on the null arm"
        return C.missing_measure(ctx, CODE, subj, split, why)
    C.require_judges(ctx)
    conf = float(ctx.cfg["CONFIDENCE"])
    z_rows = _jobs(ctx, subj, variant, split, start, n, arm=C.NULL)
    z_gen = C.generate(ctx, [r[0] for r in z_rows], code=CODE, arm=C.NULL, cell_id=C.NULL, split=split)
    null = _score(ctx, z_rows, z_gen, subj.concept, C.NULL, C.NULL)
    if subj.null_only:          # a cell of None is the null arm itself
        s_rows, steered = z_rows, null
    else:
        s_rows = _jobs(ctx, subj, variant, split, start, n, arm=subj.arm)
        s_gen = C.generate(ctx, [r[0] for r in s_rows], code=CODE, arm=subj.arm,
                           cell_id=subj.cell_id, split=split)
        steered = _score(ctx, s_rows, s_gen, subj.concept, subj.arm, subj.cell_id)
        C.write_trials(ctx, [{**_row(t, arm=subj.arm, cell_id=subj.cell_id, split=split,
                                     variant=variant), "vector": subj.vector_id} for t in steered])
    C.write_null_trials(ctx, (((CODE, subj.concept, t.job.unit),
                               {**_row(t, arm=C.NULL, cell_id=C.NULL, split=C.NULL, variant=variant),
                                "concept": subj.concept}) for t in null))
    tally = C.Tally()
    for t in steered:
        if t.matches is None:
            for w in t.missing or ["d2:missing"]:
                tally.add(t.job.unit, w)
    bs, bz = _block(steered, conf), _block(null, conf)
    k, nn = bs["identified"]["k"], bs["identified"]["n"]
    zk, zn = bz["identified"]["k"], bz["identified"]["n"]
    iv = s_rows[0][0].intervention
    if not nn:
        return C.missing_measure(ctx, CODE, subj, split, "every steered trial's identification "
                                 "failed", excluded=tally.count,
                                 extra={"excluded_reasons": tally.to_json()})
    dec = [t for t in steered if t.matches is not None]
    extra: dict[str, Any] = {
        "value": "identified (decision 7 rule a, matches) / decided trials; degenerate and "
                 "unreadable replies count as not identified",
        "excl_degenerate": bs["excl_degenerate"], "excl_unreadable": bs["excl_unreadable"],
        "loose": bs["loose"], "degenerate": bs["degenerate"], "unreadable": bs["unreadable"],
        "how": bs["how"], "null": bz, "words": _words(steered), "null_words": _words(null),
        "variant": variant, **C.design([t.trial for t in dec], [int(bool(t.matches)) for t in dec]),
        "excluded_reasons": tally.to_json(),
        "outcomes": [int(bool(t.matches)) for t in dec]}
    if zn and not subj.null_only:
        extra["vs_null"] = rate_with_null(k, nn, zk, zn, conf, delta=float(ctx.cfg["DELTA"]))
    m = Measured(code=CODE, cell_id=subj.cell_id, split=split, n=nn, k=k, lo=bs["identified"]["lo"],
                 hi=bs["identified"]["hi"], null_k=zk if zn else None, null_n=zn if zn else None,
                 unit_ids=[t.job.unit for t in dec], excluded=tally.count, extra=extra)
    if subj.null_only:
        m = C.as_null_arm(m, conf)
    return C.write_cell(ctx, m, subj=subj, iv=iv)
