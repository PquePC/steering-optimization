"""m5.analysis.nearmiss - the identification near-misses on disk, scored by decision 7's rules
(Phase 0, item 6).

M3 scored identification with the judge's strict `matches`. The README described a
root-matching rule that never existed in code; Wrists reads 8.8% strict and 20.2% loose, the
difference being "wristwatches" and "hand" at a handful of cells. Decision 7 (plan §13) is to be
made once, with the numbers: this module produces them, from what the judge recorded as *named*
on every identify trial, under each rule of `m5.judges.matching`:

    m3_judge    the archived verdict (the judge's strict boolean, as the run's judge applied it)
    strict      code: the same word after normalisation (case, accents, a leading article)
    lemma       code: strict, or a simple inflection of it ("wrist" for "wrists")
    synonym     code: lemma, or a curated undisputed synonym / spelling variant
    root        code: synonym, or a shared root of at least four letters ("wristwatches")
    same_concept / related   the judges' decision 7 (a) and (b) answers for the names code could
                not decide, when a `pair_judge` is passed; without one they are counted as
                undecided -- missing, never a NO
    semantic    cosine similarity of name and concept under an `embed` function the caller
                passes; None when none is passed (the CLI has none: semantic scores need an
                embedding model, which this package does not load)

A name of NONE (the judge's "no concept named") fails every rule. Trials without a verdict are
excluded from every denominator and counted.

**Text.** The named strings are the model's words. Aggregate rows refer to them by `name_id`
(a hash); the strings themselves, with their counts and their verdict under every rule, are
returned as transcript rows (`texts`), which the CLI writes to the run's `transcripts` stream.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m5.analysis import concept_name_id
from m5.analysis.m3_archive import M3Archive
from m5.judges import matching
from m5.stats import paired_rate_difference, wilson

CODE_RULES = ("strict", "lemma", "synonym", "root")
RULES = ("m3_judge",) + CODE_RULES + ("same_concept", "related")

# pair_judge(template_id, named, concept) -> True / False / None (None: the judge did not answer)
PairJudge = Callable[[str, str, str], "bool | None"]


@dataclass(frozen=True)
class Scored:
    """One identify trial under every rule (None = undecided, excluded from that rule's n)."""

    transcript_id: str
    cell: str
    layer: int
    fraction: float | None
    named: str | None
    how: str
    verdicts: Mapping[str, bool | None]
    semantic: float | None


def score_name(named: str | None, concept: str, *, m3_match: bool,
               pair_judge: PairJudge | None = None,
               embed: Callable[[str], Sequence[float]] | None = None) -> tuple[str, dict[str, bool | None], float | None]:
    """(how, verdicts by rule, semantic) for one named string."""
    ident = matching.identify(named, concept, embed=embed)
    how = ident.how
    v: dict[str, bool | None] = {"m3_judge": bool(m3_match)}
    if how == "none_named":
        for r in RULES[1:]:
            v[r] = False
        return how, v, None
    code = how if how in ("exact", "inflection", "synonym") else None
    v["strict"] = code == "exact"
    v["lemma"] = code in ("exact", "inflection")
    v["synonym"] = code is not None
    v["root"] = code is not None or matching.shares_root(str(named), concept)
    if code is not None:
        v["same_concept"] = True
        v["related"] = True
    else:
        same = pair_judge("same_concept", str(named), concept) if pair_judge else None
        if same is None:
            v["same_concept"] = None
            v["related"] = True if v["root"] else None
        else:
            final = matching.identify(named, concept, same_concept=bool(same))
            v["same_concept"] = bool(final.matches)
            if final.loose is not None:
                v["related"] = bool(final.loose)
            else:
                rel = pair_judge("related", str(named), concept) if pair_judge else None
                v["related"] = None if rel is None else bool(
                    matching.identify(named, concept, same_concept=bool(same), related=bool(rel)).loose)
    return how, v, ident.semantic


def _memo(fn: Callable[[str], Sequence[float]]) -> Callable[[str], Sequence[float]]:
    cache: dict[str, Sequence[float]] = {}

    def call(text: str) -> Sequence[float]:
        if text not in cache:
            cache[text] = fn(text)
        return cache[text]
    return call


def _memo_pairs(fn: PairJudge) -> PairJudge:
    cache: dict[tuple[str, str, str], bool | None] = {}

    def call(template: str, named: str, concept: str) -> bool | None:
        key = (template, matching.normalise(named), concept)
        if key not in cache:
            cache[key] = fn(template, named, concept)
        return cache[key]
    return call


@dataclass
class NearMissResult:
    rows: list[dict[str, Any]]
    texts: list[dict[str, Any]]
    scored: list[Scored] = field(default_factory=list)


def _rate(k: int, n: int, conf: float) -> dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None}
    lo, hi = wilson(k, n, conf)
    return {"k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi}


JUDGED_RULES = ("same_concept", "related")


def _rule_rates(items: Sequence[Scored], conf: float, *, judged: bool) -> dict[str, Any]:
    """k/n/interval per rule over the trials that rule decided. Without a pair judge the judged
    rules decide only what code already matched, so their rate over "decided" trials would be a
    selected 1.0: they are reported missing instead, with every trial undecided."""
    out = {}
    for r in RULES:
        if r in JUDGED_RULES and not judged:
            out[r] = {"k": None, "n": 0, "rate": None, "lo": None, "hi": None,
                      "undecided": len(items), "reason": "no_pair_judge"}
            continue
        decided = [s.verdicts[r] for s in items if s.verdicts.get(r) is not None]
        row = _rate(sum(1 for x in decided if x), len(decided), conf)
        row["undecided"] = len(items) - len(decided)
        out[r] = row
    return out


def _semantic(items: Sequence[Scored], thresholds: Sequence[float]) -> dict[str, Any] | None:
    vals = [s.semantic for s in items if s.semantic is not None and s.named is not None]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    out: dict[str, Any] = {"n": int(arr.size), "mean": float(arr.mean()),
                           "median": float(np.median(arr))}
    if thresholds:
        out["above"] = [{"threshold": float(t), "k": int(np.sum(arr >= t)), "n": int(arr.size)}
                        for t in thresholds]
    return out


def near_misses(archive: M3Archive, cfg: Any, *, pair_judge: PairJudge | None = None,
                embed: Callable[[str], Sequence[float]] | None = None,
                semantic_thresholds: Sequence[float] = ()) -> NearMissResult:
    """Score every identify trial with a verdict under every rule; rates per cell and per
    concept, paired differences between rules, and the distinct names (as texts)."""
    conf = float(cfg["CONFIDENCE"])
    concept = archive.concept
    # Thousands of trials name a handful of things: ask the judge and the embedding once per name.
    pair_judge = _memo_pairs(pair_judge) if pair_judge is not None else None
    embed = _memo(embed) if embed is not None else None
    scored: list[Scored] = []
    excluded = 0
    for c in archive.cells:
        for t in archive.trials(c, "identify"):
            if t.matches is None:
                excluded += 1
                continue
            how, v, sem = score_name(t.named, concept, m3_match=t.matches, pair_judge=pair_judge,
                                     embed=embed)
            scored.append(Scored(t.transcript_id, c.cell_id, c.layer, c.fraction, t.named, how, v, sem))
    rows: list[dict[str, Any]] = []
    by_cell: dict[str, list[Scored]] = {}
    for s in scored:
        by_cell.setdefault(s.cell, []).append(s)
    for cell_id, items in by_cell.items():
        rows.append({"kind": "nearmiss_cell", "concept": concept, "cell": cell_id,
                     "layer": items[0].layer, "fraction": items[0].fraction, "n": len(items),
                     "rules": _rule_rates(items, conf, judged=pair_judge is not None),
                     "semantic": _semantic(items, semantic_thresholds),
                     "rescued_by_root": sum(1 for s in items if not s.verdicts["m3_judge"] and s.verdicts["root"]),
                     "judge_only": sum(1 for s in items if s.verdicts["m3_judge"] and not s.verdicts["synonym"])})
    # paired differences against the archived judge, on the same trials
    paired = []
    for r in RULES[1:]:
        if r in JUDGED_RULES and pair_judge is None:
            continue
        both = [(1.0 if s.verdicts[r] else 0.0, 1.0 if s.verdicts["m3_judge"] else 0.0)
                for s in scored if s.verdicts.get(r) is not None]
        if both:
            d = paired_rate_difference(np.array(both), conf)
            paired.append({"rule": r, "versus": "m3_judge", "n": d["n"], "diff": d["diff"],
                           "lo": d["lo"], "hi": d["hi"], "rule_only": d["b"], "m3_judge_only": d["c"],
                           "p_mcnemar": d["p_mcnemar"]})
    # "NONE" (the judge's "no concept named") is counted under how == "none_named", not as a name
    names = Counter(matching.normalise(s.named) for s in scored if s.how != "none_named")
    rows.append({"kind": "nearmiss_summary", "concept": concept, "n_trials": len(scored),
                 "excluded_no_verdict": excluded,
                 "rules": _rule_rates(scored, conf, judged=pair_judge is not None),
                 "semantic": _semantic(scored, semantic_thresholds),
                 "embed": embed is not None, "pair_judge": pair_judge is not None,
                 "paired_vs_m3_judge": paired, "n_distinct_names": len(names),
                 "how": dict(Counter(s.how for s in scored))})
    # distinct names: counts in the aggregate stream by id, the strings themselves as texts
    texts = []
    per_name: dict[str, list[Scored]] = {}
    for s in scored:
        if s.how != "none_named":
            per_name.setdefault(matching.normalise(s.named), []).append(s)
    for nm, items in sorted(per_name.items(), key=lambda kv: -len(kv[1])):
        nid = concept_name_id(concept, nm)
        verdicts = {r: items[0].verdicts.get(r) for r in RULES[1:]}
        rows.append({"kind": "nearmiss_name", "concept": concept, "name_id": nid,
                     "count": len(items), "m3_judge_yes": sum(1 for s in items if s.verdicts["m3_judge"]),
                     "verdicts": verdicts, "how": items[0].how,
                     "semantic": None if items[0].semantic is None else float(items[0].semantic),
                     "cells": len({s.cell for s in items})})
        texts.append({"id": f"m3name/{archive.run_id}/{nid}", "text": nm, "kind": "named_string",
                      "concept": concept, "count": len(items), "name_id": nid})
    return NearMissResult(rows=rows, texts=texts, scored=scored)
