"""m5.analysis.truncation - the truncation curve behind TOKENS_INFLUENCE (Phase 0, item 3).

M3 generated 100 tokens for every influence prompt; M5 sets each channel's length from evidence
(plan §8.1, "channel-specific generation lengths"). The evidence is this curve: the archived
influence transcripts cut at 24, 48, 72 and 100 tokens and re-judged at each length with M5's
judges, so the judged rates can be compared length by length. If the rate at 48 tokens is the
rate at 100 within its interval, the extra 52 tokens buy nothing and TOKENS_INFLUENCE can drop.
No GPU: the transcripts are on disk.

**Tokens.** The cut is made in the subject's own tokens when a tokenizer is passed in: a callable
returning the character span `(start, end)` of each token of a text (`Backend.offsets` has
exactly this shape). Without one the cut falls back to whitespace-separated words, which is
**approximate**: a word is about 1.3 Gemma tokens in English prose, so "24 words" is roughly 31
tokens and a 100-token response is often shorter than 100 words (the 100 cut is then the whole
text). Every row says which was used (`tokenizer`). A length above what the run generated
(MAX_NEW_TOKENS of its settings) is refused: an archived reply cannot be lengthened.

**Judging.** A judge callable is passed in -- normally `m5.judges.Cascade.judge`, which records
every call in the run's internal `judge_calls` stream -- taking `m5.judges.JudgeItem`s and
returning one result per item with `.label` and `.missing`. Items are built from M5's arm-blind
templates (`allusion` is `e8`'s question; `dominant` and `coherent` may be added) through
`m5.judges.make_item`, which refuses a harmful-arm concept like every other entry point. Whether
any text leaves the machine is the cascade's decision (JUDGE_EXTERNAL, decision 12), not this
module's.

**What is reported**, per template and length: the judged rate with k, n, its Wilson interval and
the excluded (failed or undecided) judgments; per dose fraction as well; and, paired on the same
transcripts, the change against the longest length (Tango's interval and McNemar's exact test,
`m5.stats.paired_rate_difference`) -- the comparison that says what each truncation loses.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from m5.analysis.m3_archive import ArchiveTrial, M3Archive
from m5.backend.base import derive_seed
from m5.judges.prompts import make_item, templates
from m5.stats import paired_rate_difference, wilson

# The lengths of plan §8.1, in tokens: definitions of the curve, not tunables.
PLAN_LENGTHS = (24, 48, 72, 100)
TEMPLATES_FOR = {"allusion": ("concept", "text"), "dominant": ("concept", "text"),
                 "coherent": ("prompt", "text")}

Offsets = Callable[[str], Sequence[tuple[int, int]]]
_WORD = re.compile(r"\S+")


def whitespace_offsets(text: str) -> list[tuple[int, int]]:
    """The fallback 'tokenizer': one span per whitespace-separated word. Approximate (module
    docstring): words are not the subject's tokens."""
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def truncate(text: str, n_tokens: int, offsets: Offsets) -> tuple[str, int, bool]:
    """(text up to the end of its n-th token, tokens kept, whether it was cut at all)."""
    if n_tokens < 1:
        raise ValueError("a truncation keeps at least one token")
    spans = list(offsets(text))
    if len(spans) <= n_tokens:
        return text, len(spans), False
    end = int(spans[n_tokens - 1][1])
    return text[:end], n_tokens, True


@dataclass
class TruncationResult:
    rows: list[dict[str, Any]]
    n_items: int
    n_judged: int


def _sample(trials: list[ArchiveTrial], max_items: int | None, seed: int) -> list[ArchiveTrial]:
    if max_items is None or len(trials) <= max_items:
        return trials
    rng = np.random.default_rng(seed)
    keep = sorted(rng.choice(len(trials), size=max_items, replace=False).tolist())
    return [trials[i] for i in keep]


def truncation_curve(archive: M3Archive, cfg: Any, *, judge: Callable[[list[Any]], list[Any]],
                     tokenizer: Offsets | None, lengths: Sequence[int] = PLAN_LENGTHS,
                     template_ids: Sequence[str] = ("allusion",), channels: Sequence[str] = ("effect",),
                     max_items: int | None = None) -> TruncationResult:
    """Cut every archived influence transcript at each length, judge each cut, tabulate.

    `max_items` (optional) judges a seeded random subset of transcripts -- the same subset at
    every length, so the comparison stays paired.
    """
    lengths = tuple(sorted({int(x) for x in lengths}))
    if not lengths or lengths[0] < 1:
        raise ValueError("lengths must be positive token counts")
    generated = archive.settings.get("MAX_NEW_TOKENS")
    if generated is not None and lengths[-1] > int(generated):
        raise ValueError(f"the run generated at most {generated} tokens; an archived reply cannot "
                         f"be judged at {lengths[-1]}")
    bound = templates(cfg)
    for tid in template_ids:
        if tid not in TEMPLATES_FOR:
            raise ValueError(f"template {tid!r} is not one the truncation curve uses: {sorted(TEMPLATES_FOR)}")
    offsets = tokenizer or whitespace_offsets
    tok_name = "subject_tokenizer" if tokenizer is not None else "whitespace_words_approximate"
    trials = [t for c in archive.cells for t in archive.trials(c)
              if t.channel in channels and not t.empty]
    trials = _sample(trials, max_items, derive_seed(int(cfg["SEED"]), "analysis", "truncation",
                                                     archive.run_id))
    items = []
    meta: list[tuple[str, int, ArchiveTrial, bool]] = []
    for t in trials:
        text = archive.text(t.transcript_id) or ""
        for L in lengths:
            cut, _, was_cut = truncate(text, L, offsets)
            for tid in template_ids:
                if TEMPLATES_FOR[tid][0] == "concept":
                    payload = {"concept": archive.concept, "text": cut}
                else:
                    prompt = archive.prompt_text(t.unit)
                    if not prompt:
                        continue
                    payload = {"prompt": prompt, "text": cut}
                # a reply shorter than L is the same item at every longer length: the judge's
                # cache answers it once (the item key is content-addressed)
                items.append(make_item(bound[tid], payload, item_id=f"{t.transcript_id}|{tid}|{L}",
                                       meta={"transcript_id": t.transcript_id, "length": L,
                                             "template": tid, "phase0": "truncation"}))
                meta.append((tid, L, t, was_cut))
    results = judge(items) if items else []
    if len(results) != len(items):
        raise RuntimeError(f"the judge returned {len(results)} results for {len(items)} items")
    conf = float(cfg["CONFIDENCE"])
    # outcome[(template, transcript)][length] = label or None
    outcome: dict[tuple[str, str], dict[int, bool | None]] = {}
    cut_counts: dict[tuple[str, int], int] = {}
    frac_of: dict[str, float | None] = {}
    for (tid, L, t, was_cut), res in zip(meta, results):
        label = None if getattr(res, "missing", None) is not None else getattr(res, "label", None)
        outcome.setdefault((tid, t.transcript_id), {})[L] = None if label is None else bool(label)
        cut_counts[(tid, L)] = cut_counts.get((tid, L), 0) + int(was_cut)
        cell = archive.cell(t.layer, t.dose) if t.layer is not None else None
        frac_of[t.transcript_id] = None if cell is None else cell.fraction
    rows: list[dict[str, Any]] = []
    longest = lengths[-1]
    for tid in template_ids:
        per = {tr: v for (tt, tr), v in outcome.items() if tt == tid}
        for L in lengths:
            vals = [v.get(L) for v in per.values() if L in v]
            decided = [x for x in vals if x is not None]
            k, n = sum(decided), len(decided)
            row: dict[str, Any] = {"kind": "truncation_length", "concept": archive.concept,
                                   "template": tid, "length": L, "tokenizer": tok_name,
                                   "k": k, "n": n, "excluded": len(vals) - n,
                                   "texts_cut": cut_counts.get((tid, L), 0)}
            if n:
                lo, hi = wilson(k, n, conf)
                row.update(rate=k / n, lo=lo, hi=hi)
            else:
                row.update(rate=None, lo=None, hi=None)
            pairs = [(float(v[L]), float(v[longest])) for v in per.values()
                     if v.get(L) is not None and v.get(longest) is not None]
            if L != longest and pairs:
                d = paired_rate_difference(np.array(pairs), conf)
                row["versus_longest"] = {"length": longest, "n": d["n"], "diff": d["diff"],
                                         "lo": d["lo"], "hi": d["hi"], "p_mcnemar": d["p_mcnemar"],
                                         "only_short": d["b"], "only_long": d["c"]}
            rows.append(row)
            for f in sorted({frac_of[tr] for tr in per if frac_of.get(tr) is not None}):
                fv = [v.get(L) for tr, v in per.items() if frac_of.get(tr) == f and L in v]
                fd = [x for x in fv if x is not None]
                fk, fn = sum(fd), len(fd)
                frow = {"kind": "truncation_fraction", "concept": archive.concept, "template": tid,
                        "length": L, "fraction": f, "k": fk, "n": fn, "excluded": len(fv) - fn,
                        "tokenizer": tok_name}
                if fn:
                    lo, hi = wilson(fk, fn, conf)
                    frow.update(rate=fk / fn, lo=lo, hi=hi)
                rows.append(frow)
    return TruncationResult(rows=rows, n_items=len(items),
                            n_judged=sum(1 for r in results if getattr(r, "missing", None) is None))
