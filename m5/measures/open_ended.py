"""m5.measures.open_ended - one open-ended generation, read four ways: `e7`, `e8`, `s1`, `d10`.

On n influence prompts (`m5.data.influence`: the search set for the search, calibration and audit
splits, the held-out set for "confirm"), each prompt is answered steered (every position, the
prompt's own arrangement, plan §4.2) and unsteered, at TOKENS_INFLUENCE. The unsteered answer to
a (prompt, repeat) is generated once per run and serves every cell (`m5.measures.common`). Then:

    e7   concept occurrence rate (judge-free): the share of replies that mention the concept by
         the old e7 rule, `m5.judges.mechanical.prefix_mentions` -- a leading word boundary and no
         trailing one, so "garlicky" counts. The code keeps its old meaning (AGENTS.md: old codes
         keep their old meanings; M2 counted `concept_hits` this way), so the prefix rule is the
         one reported; the strict whole-word rate (`literal_mentions`) is recorded beside it, and
         the gap between the two is the inflections the strict rule misses.
    e8   judged concept presence. The `allusion` judge says whether the reply refers to the
         concept and quotes the words it relies on (a YES without a valid quote is a NO, plan
         §7.1); the `dominant` judge says whether the concept is the reply's main subject. The
         judge never sees two texts: the steered-minus-unsteered difference is computed here,
         prompt by prompt. **The reported e8 value is that excess**: `rate`, `lo` and `hi` are the
         paired difference (steered allusion rate minus unsteered, on the prompts where both arms
         were judged; Tango's score interval) -- the quantity EFFECT_MIN bounds from below --
         while `k`/`n` count steered allusions and `null_k`/`null_n` unsteered ones on the same
         prompts. Dominance is reported the same way in `extra["dominant"]`.
    s1   coherence, concept-blind: the `coherent` judge sees the request and the reply and no
         concept (the template has no concept slot); collapse is decided in code first (tier 0),
         and the mechanical meters (degeneracy rules, gzip ratio, distinct-word ratio, length) are
         kept beside the judgment. The rate is the coherent share, with the unsteered share as its
         null and the steered/unsteered length ratio in `extra`.
    d10  spontaneous flags: restarts, disavowals and evaluation-awareness mentions found by the
         mechanical patterns (`spontaneous_flags`), each confirmed by a judge -- `ea_verbalised`
         for an evaluation-awareness mention, `claim` (does the speaker remark on something unusual
         in its own thoughts or processing?) for a restart or a disavowal. A reply no pattern
         flags is a NO without a judge (regex first, judge to confirm, plan §3.2). A flagged reply
         whose confirmations all failed is missing.

A vector with no concept word has no e7 or e8 (they are missing, with the reason); s1 and d10 are
measured for every vector.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m5 import prompts as _prompts
from m5.intervention import Cell
from m5.judges import mechanical
from m5.judges.prompts import make_item, templates
from m5.measures import common as C
from m5.results import Measured

FAMILY = "influence"
CODES = ("e7", "e8", "s1", "d10")
_RESTART_KINDS = ("restart", "disavowal")


def influence_jobs(ctx: Any, subj: C.Subject, n: int, split: str, start: int, *,
                   arm: str) -> list[C.Job]:
    """The open-ended generations of one arm on units start..start+n-1 of the split's prompts.
    Shared with `e11`, so a cell measured for both is generated once."""
    jobs = []
    for p, r in C.cycle(C.influence_prompts(split), start, n):
        spec = C.spec(ctx, (FAMILY, p.id), lambda p=p: _prompts.influence(ctx.backend, p))
        if arm == C.NULL:
            iv, seed = None, ctx.seed(FAMILY, C.NULL, p.id, r)
        else:
            iv = C.intervention(ctx, subj.cell, spec, spec.meta["mode"])
            seed = ctx.seed(FAMILY, split, subj.cell_id, p.id, r)
        jobs.append(C.Job(f"{p.id}#{r}", spec, iv, seed, int(ctx.cfg["TOKENS_INFLUENCE"])))
    return jobs


@dataclass
class _Text:
    """One reply and everything read from it; `slots` index its items in the judge batch."""

    job: C.Job
    gen: C.Gen
    prefix: int
    literal: int
    degeneracy: mechanical.Degeneracy
    kinds: tuple[str, ...]
    slots: dict[str, int] = field(default_factory=dict)
    out: dict[str, Any] = field(default_factory=dict)
    why: dict[str, list[str]] = field(default_factory=dict)

    @property
    def words(self) -> int:
        return int(self.degeneracy.meters["words"])  # type: ignore[arg-type]


def _read(job: C.Job, gen: C.Gen, concept: str | None) -> _Text:
    text = gen.text
    return _Text(job=job, gen=gen,
                 prefix=mechanical.prefix_mentions(text, concept) if concept else 0,
                 literal=len(mechanical.literal_mentions(text, concept)) if concept else 0,
                 degeneracy=mechanical.degeneracy(text),
                 kinds=tuple(sorted({f.kind for f in mechanical.spontaneous_flags(text)})))


def _items(ctx: Any, texts: list[_Text], concept: str | None, items: list[Any], arm: str,
           cell_id: str) -> None:
    T = templates(ctx.cfg)
    for t in texts:
        content = t.job.prompt.meta["content"]
        text = t.gen.text
        meta = {"arm": arm, "cell": cell_id, "unit": t.job.unit}

        def add(slot: str, template: str, payload: dict[str, str]) -> None:
            t.slots[slot] = len(items)
            items.append(make_item(T[template], payload, item_id=f"{t.gen.transcript}:{slot}",
                                   meta={**meta, "slot": slot}))

        if concept:
            add("allusion", "allusion", {"concept": concept, "text": text})
            add("dominant", "dominant", {"concept": concept, "text": text})
        add("coherent", "coherent", {"prompt": content, "text": text})
        if "ea" in t.kinds:
            add("ea", "ea_verbalised", {"text": text})
        if any(k in t.kinds for k in _RESTART_KINDS):
            add("restart", "claim", {"question": content, "text": text})


def _outcomes(t: _Text, js: list[Any]) -> None:
    def got(slot: str) -> Any:
        return js[t.slots[slot]] if slot in t.slots else None

    for slot in ("allusion", "dominant", "coherent"):
        j = got(slot)
        t.out[slot] = C.label_of(j)
        t.why[slot] = [C.reason(j)] if j is not None and j.missing is not None else []
    confirms = [got(s) for s in ("ea", "restart") if s in t.slots]
    labels = [C.label_of(j) for j in confirms]
    if not t.kinds:
        t.out["d10"] = False
    elif any(lab is True for lab in labels):
        t.out["d10"] = True
    elif labels and all(lab is False for lab in labels):
        t.out["d10"] = False
    else:
        t.out["d10"] = None
    t.why["d10"] = [C.reason(j) for j in confirms if j is not None and j.missing is not None]
    t.out["confirmed"] = sorted(
        kind for kind, slot in (("ea", "ea"), ("restart", "restart")) if C.label_of(got(slot)) is True)


def _base_row(code: str, t: _Text, *, arm: str, cell_id: str, split: str) -> dict[str, Any]:
    pid, rep = t.job.unit.rsplit("#", 1)
    return {"code": code, "cell": cell_id, "split": split if arm != C.NULL else C.NULL, "arm": arm,
            "unit": t.job.unit, "prompt_id": pid, "repeat": int(rep), "transcript_id": t.gen.transcript}


def _rows(code: str, t: _Text, **kw: Any) -> dict[str, Any]:
    row = _base_row(code, t, **kw)
    if code == "e7":
        row.update(prefix_mentions=t.prefix, literal_mentions=t.literal, mentioned=t.prefix > 0)
    elif code == "e8":
        row.update(allusion=t.out["allusion"], dominant=t.out["dominant"],
                   missing=t.why["allusion"] + t.why["dominant"])
    elif code == "s1":
        m = t.degeneracy.meters
        row.update(coherent=t.out["coherent"], degenerate=t.degeneracy.flag,
                   rules=[r.split(":", 1)[0] for r in t.degeneracy.reasons], words=m["words"],
                   gzip_ratio=m["gzip_ratio"], distinct_word_ratio=m["distinct_word_ratio"],
                   finish=t.gen.finish, missing=t.why["coherent"])
    else:
        row.update(flagged=list(t.kinds), confirmed=t.out["confirmed"], spontaneous=t.out["d10"],
                   missing=t.why["d10"])
    return row


def _pairs(steered: list[_Text], null: list[_Text],
           key: str) -> tuple[list[tuple[bool, bool]], list[str], C.Tally]:
    pairs, units, tally = [], [], C.Tally()
    for s, z in zip(steered, null):
        a, b = s.out[key], z.out[key]
        if a is None or b is None:
            whys = (s.why[key] if a is None else []) + \
                   ([f"null:{w}" for w in z.why[key]] if b is None and z is not s else [])
            for w in whys or [f"{key}:missing"]:
                tally.add(s.job.unit, w)
            continue
        pairs.append((bool(a), bool(b)))
        units.append(s.job.unit)
    return pairs, units, tally


def _measured_rate(code: str, subj: C.Subject, split: str, pairs: list[tuple[bool, bool]],
                   units: list[str], tally: C.Tally, conf: float, extra: dict[str, Any]) -> Measured | None:
    if not pairs:
        return None
    k = sum(a for a, _ in pairs)
    nk = sum(b for _, b in pairs)
    lo, hi = C.wilson_or_none(k, len(pairs), conf)
    return Measured(code=code, cell_id=subj.cell_id, split=split, n=len(pairs), k=k, lo=lo, hi=hi,
                    null_k=nk, null_n=len(pairs), unit_ids=units, excluded=tally.count,
                    extra={**extra, "null": C.rate_block(nk, len(pairs), conf),
                           "paired": C.paired(pairs, conf), "excluded_reasons": tally.to_json(),
                           "outcomes": [int(a) for a, _ in pairs],
                           **C.design([u.rsplit("#", 1)[0] for u in units], [int(a) for a, _ in pairs])})


def measure_open_ended(ctx: Any, cell: Cell | None, n: int, split: str, *, variant: str = "original",
                       start: int = 0, concept: str | None = None) -> dict[str, Measured]:
    """Generate once, judge four ways; return {"e7", "e8", "s1", "d10"} and write their rows.

    `start` is the first unit (prompt index, cycling through the set with a new repeat each pass),
    so a sequential caller can extend a cell without repeating units. A `cell` of None measures
    the null arm itself, and `concept` names the word e7/e8 look for there; a steered cell uses its
    vector's own concept.
    Called again with the same arguments in the same run, it returns the same measurement
    (memoised) and writes nothing twice.
    """
    C.check_split(split)
    C.check_variant("open-ended (e7/e8/s1/d10)", variant)
    n, start = C.check_n(n), C.check_start(start)
    subj = C.subject(ctx, cell, concept=concept)
    st = C.state(ctx)
    key = ("open_ended", subj.cell_id, n, split, start, subj.concept)
    with st.lock:
        if key in st.memo:
            return dict(st.memo[key])
    C.require_judges(ctx)
    conf = float(ctx.cfg["CONFIDENCE"])
    z_jobs = influence_jobs(ctx, subj, n, split, start, arm=C.NULL)
    z_gen = C.generate(ctx, z_jobs, code=FAMILY, arm=C.NULL, cell_id=C.NULL, split=split)
    null = [_read(j, g, subj.concept) for j, g in zip(z_jobs, z_gen)]
    if subj.null_only:          # a cell of None is the null arm itself
        s_jobs, steered = z_jobs, null
    else:
        s_jobs = influence_jobs(ctx, subj, n, split, start, arm=subj.arm)
        s_gen = C.generate(ctx, s_jobs, code=FAMILY, arm=subj.arm, cell_id=subj.cell_id, split=split)
        steered = [_read(j, g, subj.concept) for j, g in zip(s_jobs, s_gen)]
    items: list[Any] = []
    if not subj.null_only:
        _items(ctx, steered, subj.concept, items, subj.arm, subj.cell_id)
    _items(ctx, null, subj.concept, items, C.NULL, C.NULL)
    js = C.judge(ctx, items)
    for t in (null if subj.null_only else steered + null):
        _outcomes(t, js)

    kw_s = {"arm": subj.arm, "cell_id": subj.cell_id, "split": split}
    kw_n = {"arm": C.NULL, "cell_id": C.NULL, "split": split}
    codes = CODES if subj.concept else ("s1", "d10")
    for code in codes:
        if not subj.null_only:
            rows = []
            for s, z in zip(steered, null):
                row = _rows(code, s, **kw_s)
                row["null_transcript_id"] = z.gen.transcript
                row["vector"] = subj.vector_id
                rows.append(row)
            C.write_trials(ctx, rows)
        scored_for = subj.concept if code in ("e7", "e8") else None
        C.write_null_trials(ctx, (((code, scored_for, z.job.unit),
                                   {**_rows(code, z, **kw_n), "concept": scored_for})
                                  for z in null))

    iv = s_jobs[0].intervention
    out: dict[str, Measured] = {}
    if subj.concept:
        out["e7"] = _e7(ctx, subj, split, steered, null, conf, iv)
        out["e8"] = _e8(ctx, subj, split, steered, null, conf, iv)
    else:
        why = (f"vector {subj.vector_id!r} has no concept word: e7 and e8 look for a concept "
               "(naming and mention drop out for nameless vectors)") if subj.cell is not None else \
            "no concept to look for: pass concept= to measure e7/e8 on the null arm"
        out["e7"] = C.missing_measure(ctx, "e7", subj, split, why)
        out["e8"] = C.missing_measure(ctx, "e8", subj, split, why)
    out["s1"] = _s1(ctx, subj, split, steered, null, conf, iv)
    out["d10"] = _d10(ctx, subj, split, steered, null, conf, iv)
    with st.lock:
        st.memo[key] = dict(out)
    return out


def _finish(ctx: Any, code: str, subj: C.Subject, split: str, m: Measured | None, tally: C.Tally,
            iv: Any) -> Measured:
    if m is None:
        return C.missing_measure(ctx, code, subj, split, f"no unit has both arms decided for {code} "
                                 "(every judgment it needed failed)", excluded=tally.count,
                                 extra={"excluded_reasons": tally.to_json()})
    if subj.null_only:
        m = C.as_null_arm(m, float(ctx.cfg["CONFIDENCE"]))
    return C.write_cell(ctx, m, subj=subj, iv=iv)


def _e7(ctx: Any, subj: C.Subject, split: str, steered: list[_Text], null: list[_Text],
        conf: float, iv: Any) -> Measured:
    pairs = [(s.prefix > 0, z.prefix > 0) for s, z in zip(steered, null)]
    strict = [(s.literal > 0, z.literal > 0) for s, z in zip(steered, null)]
    m = _measured_rate("e7", subj, split, pairs, [s.job.unit for s in steered], C.Tally(), conf, {
        "rule": "prefix_mentions (the old e7 rule; leading boundary only)",
        "strict": {"rule": "literal_mentions (whole word, simple inflections)",
                   **C.rate_block(sum(a for a, _ in strict), len(strict), conf),
                   "null_k": sum(b for _, b in strict)},
        "mean_mentions": float(np.mean([s.prefix for s in steered])),
        "null_mean_mentions": float(np.mean([z.prefix for z in null]))})
    return _finish(ctx, "e7", subj, split, m, C.Tally(), iv)


def _e8(ctx: Any, subj: C.Subject, split: str, steered: list[_Text], null: list[_Text],
        conf: float, iv: Any) -> Measured:
    pairs, units, tally = _pairs(steered, null, "allusion")
    if not pairs:
        return _finish(ctx, "e8", subj, split, None, tally, iv)
    k, nk, npair = sum(a for a, _ in pairs), sum(b for _, b in pairs), len(pairs)
    exc = C.paired(pairs, conf)
    dpairs, _, dtally = _pairs(steered, null, "dominant")
    dom: dict[str, Any] = {"n": len(dpairs), "excluded": dtally.count}
    if dpairs:
        dk, dnk = sum(a for a, _ in dpairs), sum(b for _, b in dpairs)
        dex = C.paired(dpairs, conf)
        dom.update(k=dk, null_k=dnk, rate=dk / len(dpairs), null_rate=dnk / len(dpairs),
                   excess=dex["diff"], lo=dex["lo"], hi=dex["hi"])
    slo, shi = C.wilson_or_none(k, npair, conf)
    m = Measured(code="e8", cell_id=subj.cell_id, split=split, n=npair, k=k, rate=exc["diff"],
                 lo=exc["lo"], hi=exc["hi"], null_k=nk, null_n=npair, unit_ids=units,
                 excluded=tally.count,
                 extra={"value": "allusion excess: steered minus unsteered on the same prompts",
                        "steered": {"k": k, "n": npair, "rate": k / npair, "lo": slo, "hi": shi},
                        "null": C.rate_block(nk, npair, conf), "p_mcnemar": exc["p_mcnemar"],
                        "dominant": dom, "excluded_reasons": tally.to_json(),
                        "outcomes": [int(a) for a, _ in pairs],
                        **C.design([u.rsplit("#", 1)[0] for u in units], [int(a) for a, _ in pairs])})
    return _finish(ctx, "e8", subj, split, m, tally, iv)


def _s1(ctx: Any, subj: C.Subject, split: str, steered: list[_Text], null: list[_Text],
        conf: float, iv: Any) -> Measured:
    pairs, units, tally = _pairs(steered, null, "coherent")

    def meters(ts: list[_Text]) -> dict[str, Any]:
        gz = [t.degeneracy.meters["gzip_ratio"] for t in ts if t.degeneracy.meters["gzip_ratio"] is not None]
        dw = [t.degeneracy.meters["distinct_word_ratio"] for t in ts
              if t.degeneracy.meters["distinct_word_ratio"] is not None]
        rules = collections.Counter(r.split(":", 1)[0] for t in ts for r in t.degeneracy.reasons)
        return {"degenerate": sum(t.degeneracy.flag for t in ts), "n": len(ts),
                "mean_words": float(np.mean([t.words for t in ts])),
                "mean_gzip_ratio": float(np.mean(gz)) if gz else None,
                "mean_distinct_word_ratio": float(np.mean(dw)) if dw else None,
                "rules": C.counter_dict(rules),
                "truncated": sum(t.gen.finish == "length" for t in ts)}

    ms, mz = meters(steered), meters(null)
    ratio = (ms["mean_words"] / mz["mean_words"]) if mz["mean_words"] > 0 else None
    m = _measured_rate("s1", subj, split, pairs, units, tally, conf, {
        "mechanical": ms, "null_mechanical": mz, "length_ratio": C.finite(ratio),
        "judge": "coherent (concept-blind; collapse decided in code first)"})
    return _finish(ctx, "s1", subj, split, m, tally, iv)


def _d10(ctx: Any, subj: C.Subject, split: str, steered: list[_Text], null: list[_Text],
         conf: float, iv: Any) -> Measured:
    pairs, units, tally = _pairs(steered, null, "d10")

    def kinds(ts: list[_Text]) -> dict[str, Any]:
        flagged = collections.Counter(k for t in ts for k in t.kinds)
        confirmed = collections.Counter(k for t in ts for k in t.out["confirmed"])
        return {"flagged": C.counter_dict(flagged), "confirmed": C.counter_dict(confirmed),
                "any_flag": sum(bool(t.kinds) for t in ts), "n": len(ts)}

    m = _measured_rate("d10", subj, split, pairs, units, tally, conf, {
        "kinds": kinds(steered), "null_kinds": kinds(null),
        "confirmation": "ea_verbalised for evaluation-awareness mentions; claim for restarts and "
                        "disavowals"})
    return _finish(ctx, "d10", subj, split, m, tally, iv)


def _pick(code: str):  # noqa: ANN202 - the registry wrappers below
    def measure(ctx: Any, cell: Cell | None, *, n: int, split: str, variant: str = "original",
                start: int = 0, concept: str | None = None) -> Measured:
        return measure_open_ended(ctx, cell, n, split, variant=variant, start=start,
                                  concept=concept)[code]
    measure.__name__ = f"measure_{code}"
    measure.__qualname__ = f"measure_{code}"
    measure.__doc__ = (f"`{code}` from the shared open-ended measurement (`measure_open_ended`): the "
                       "same generations and judgments serve e7, e8, s1 and d10.")
    return measure


measure_e7 = _pick("e7")
measure_e8 = _pick("e8")
measure_s1 = _pick("s1")
measure_d10 = _pick("d10")
