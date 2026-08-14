"""m3.battery - the prompts a cell is measured with, and every judge-free measure.

Two halves:

**The prompts.** Four channels, each answering a different question. They are defined here once
so that the steered generation, the unsteered baseline it is paired against, and the judge
payload all describe the same rendered prompt -- two renderings of "the same" prompt is how a
paired comparison quietly stops being paired.

**The mechanical measures.** Everything computable from the response text with no model and no
judge: degeneration, emptiness, capability, concept mentions, and the Wilson interval every rate
is reported with. These are cheap, but unlike M2's cheap tier they are *valid*, because they read
generated text rather than a next-token distribution. They are also the reason a judge that is
confidently wrong is sometimes catchable: M2's judge scored `## ## ## ##` as coherent and only the
mechanical detector disagreed.
"""

from __future__ import annotations

import collections
import math
import re
import statistics
from typing import Any, Iterable, Sequence


__all__ = [
    "TASK_PROMPTS",
    "CAPABILITY_PROMPTS",
    "TASK_HELDOUT",
    "degeneration_reason",
    "is_degenerate",
    "is_empty",
    "concept_mentions",
    "capability_correct",
    "wilson_interval",
    "rate",
    "mean_se",
    "response_row",
    "channel_summary",
]


# =====================================================================================
# Prompts
# =====================================================================================
# Open-ended, for effectiveness and coherence. An injection shows up here as the response
# drifting toward the concept -- a story that becomes about garlic, a landscape whose sun is "a
# pale and generous garlic clove". These are the prompts where influence is legible.
#
# Deliberately varied in form: two narrative, two introspective, two expository, one list. A
# concept can capture one register and leave another untouched, and a prompt set that is all one
# shape would report that as either total influence or none.

TASK_PROMPTS: list[dict] = [
    dict(id="task_story",     text="Tell me a short story."),
    dict(id="task_landscape", text="Describe a landscape."),
    dict(id="task_words",     text="Tell me the first 10 words that come to mind."),
    dict(id="task_notice",    text="Describe what you notice about your own thinking right now."),
    dict(id="task_advice",    text="Give me some advice for someone starting a new job."),
    dict(id="task_explain",   text="What makes a good explanation?"),
    dict(id="task_water",     text="Tell me a fact related to water."),
]

# Held out from the sweep, for the confirmation phase. A cell selected as best on one prompt set
# and then re-measured on the SAME set reports how well it fits prompts it was chosen on. Fresh
# prompts are what make the confirmation number mean anything.
TASK_HELDOUT: list[dict] = [
    dict(id="held_dream",   text="Describe a dream someone might have."),
    dict(id="held_teach",   text="How would you teach someone to ride a bicycle?"),
    dict(id="held_museum",  text="Describe what you would see walking through a museum."),
    dict(id="held_letter",  text="Write a short letter to a friend you have not seen in years."),
]

# Verifiable, for capability. Scored by exact match against an accepted-answer set, on GENERATED
# text -- not by reading option-letter logits, which is what M2 did and which reads 0.976 on a
# model that cannot answer a single one of these.
#
# `accept` is a set of lowercase substrings, any of which counts. Phrasing varies ("391", "= 391",
# "the answer is 391") and a stricter match would score a correct answer wrong.

CAPABILITY_PROMPTS: list[dict] = [
    dict(id="cap_arith",   text="What is 17 x 23?",
         accept=("391",)),
    dict(id="cap_capital", text="What is the capital of Australia?",
         accept=("canberra",)),
    dict(id="cap_colours", text="Name the three primary additive colours.",
         accept=("red, green, blue", "red, green and blue", "red green blue")),
    dict(id="cap_planets", text="List the planets of the solar system in order from the Sun.",
         accept=("neptune",)),
]

for _row in (*TASK_PROMPTS, *TASK_HELDOUT, *CAPABILITY_PROMPTS):
    if not _row.get("id") or not _row.get("text"):
        raise AssertionError(f"prompt row missing id or text: {_row!r}")
_ids = [r["id"] for r in (*TASK_PROMPTS, *TASK_HELDOUT, *CAPABILITY_PROMPTS)]
if len(_ids) != len(set(_ids)):
    raise AssertionError("duplicate prompt id; ids key every row that will ever be joined")
if set(r["id"] for r in TASK_PROMPTS) & set(r["id"] for r in TASK_HELDOUT):
    raise AssertionError("held-out prompts overlap the sweep set, so they are not held out")
del _row, _ids


# =====================================================================================
# Degeneration - mechanical collapse detection
# =====================================================================================
# Two rules, both from M2, both earned: a 5-gram occurring three or more times catches a loop
# (`garlic garlic garlic ...`, which one probe found on 39 of 50 responses), and a distinct-3-gram
# ratio under 0.5 catches slower circling. Split on whitespace with no case folding or punctuation
# stripping -- normalising merges "##" with "## " and with "##." and weakens the detector exactly
# where collapsed output lives.

_NGRAM_N = 5
_NGRAM_MAX_REPEAT = 3
_DISTINCT_N = 3
_DISTINCT_MIN = 0.5

# M2 folded a "fewer than five words" rule into the same verdict, and it misfired: a forced-ID
# answer of "Garlic." is one word and perfectly healthy, but scored as degenerate. M3 reports
# emptiness SEPARATELY, because "the model looped" and "the model said very little" are different
# failures and only one of them means the response channel is broken.
#
# The threshold is 1, i.e. only a genuinely blank response flags -- NOT a short one. How short is
# too short depends on the channel: a one-word answer is ideal on the identification channel and
# broken on the task channel. Every row carries its own `words` count, so that judgement is made
# in analysis, per channel, where it can be seen and changed. Baking a global floor in here is
# what made M2's detector wrong.
_EMPTY_WORDS = 1


def _ngrams(words: Sequence[str], n: int) -> list[tuple]:
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


# Both word rules are STRUCTURALLY BLIND to a collapse with no whitespace in it. The probe
# produced exactly that: `GarGarGarGar...` repeated to the token limit is one "word", so it has
# no 5-grams and no 3-grams, and both rules see nothing. It also scores zero concept mentions,
# because `Gar` is a token prefix and not the word `garlic`. That response is a total collapse
# and was invisible to every mechanical measure at once.
#
# M2 covered this accidentally with a "fewer than five words" floor, which is why removing that
# floor (correct on its own terms -- it scored the perfectly good forced-ID answer `Garlic.` as
# degenerate) reopened the hole. The fix is a character-level rule that fires only where the
# word rules cannot reach, rather than a length floor that misjudges short healthy answers.
_CHAR_N = 4
_CHAR_DISTINCT_MIN = 0.20
_CHAR_MIN_LEN = 40


def degeneration_reason(text: str) -> str | None:
    """Which collapse rule fired, or None. The label before the colon is a fixed vocabulary
    (`ngram_repeat`, `low_distinct`, `char_repeat`) so counts can be taken without parsing
    prose."""
    raw = str(text)
    words = raw.split()

    grams = _ngrams(words, _NGRAM_N)
    if grams:
        worst, count = collections.Counter(grams).most_common(1)[0]
        if count >= _NGRAM_MAX_REPEAT:
            return (f"ngram_repeat: {_NGRAM_N}-gram repeated {count} times: "
                    f"{' '.join(worst)[:40]!r}")
    tri = _ngrams(words, _DISTINCT_N)
    if tri:
        ratio = len(set(tri)) / len(tri)
        if ratio < _DISTINCT_MIN:
            return f"low_distinct: distinct-{_DISTINCT_N}-gram ratio {ratio:.2f} < {_DISTINCT_MIN}"

    # Only where the word rules are blind: too few words for a 5-gram to exist, but long enough
    # that the text is plainly not a short healthy answer. `Garlic.` has 7 characters and is
    # untouched; `GarGarGar...` has 300 and three distinct character 4-grams.
    if len(words) < _NGRAM_N and len(raw.strip()) >= _CHAR_MIN_LEN:
        chars = raw.strip()
        cgrams = [chars[i:i + _CHAR_N] for i in range(len(chars) - _CHAR_N + 1)]
        if cgrams:
            ratio = len(set(cgrams)) / len(cgrams)
            if ratio < _CHAR_DISTINCT_MIN:
                return (f"char_repeat: distinct-{_CHAR_N}-character-gram ratio {ratio:.3f} "
                        f"< {_CHAR_DISTINCT_MIN} over {len(chars)} characters with "
                        f"{len(words)} word(s)")
    return None


def is_degenerate(text: str) -> bool:
    """True if the response collapsed into repetition. Mechanical: no judge to talk round."""
    return degeneration_reason(text) is not None


def is_empty(text: str) -> bool:
    """True if the response has essentially no content.

    Reported alongside degeneration rather than merged into it. A cell whose responses are all
    empty and a cell whose responses all loop are both broken, but not in the same way, and the
    distinction is what separates "the model had nothing to say" from "the concept flooded it".
    """
    return len(str(text).split()) < _EMPTY_WORDS


# =====================================================================================
# Concept mentions
# =====================================================================================

def concept_mentions(text: str, concept: str) -> int:
    """Occurrences of the concept word. Mechanical, no judge, no model.

    Matched case-insensitively at a leading word boundary with no trailing one, so "garlicky"
    and "garlic-forward" count -- they are mentions, and a stricter match would score a steered
    response clean because it inflected the word. A leading boundary IS required, so "vulgarlic"
    does not count.

    **This is a count, not a verdict.** It cannot tell "the response is about garlic" from
    "mentions garlic once in passing", and it cannot see an influenced response that never names
    the concept -- a real case: a steered story described "a delicate, precise unfolding" for a
    full sentence before naming origami. Judged `effectiveness` is the measure; this makes
    transcripts sortable and gives the judge something to disagree with.
    """
    word = str(concept).strip()
    if not word:
        raise ValueError("concept_mentions needs a non-empty concept")
    return len(re.findall(r"\b" + re.escape(word), str(text), flags=re.IGNORECASE))


# =====================================================================================
# Capability
# =====================================================================================

def capability_correct(text: str, accept: Iterable[str]) -> bool:
    """Whether a generated answer contains an accepted answer.

    Substring match on the lowercased response. Generous by design: the question is whether the
    model can still retrieve the fact, not whether it formatted the answer the way we expected,
    and a format failure scored as a capability failure double-counts what degeneration already
    measures.
    """
    low = str(text).lower()
    return any(str(a).lower() in low for a in accept)


# =====================================================================================
# Rates and intervals
# =====================================================================================

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval for a rate.

    Wilson and not the textbook `p +/- z*sqrt(p(1-p)/n)`, because that interval has width
    exactly zero at p=0 and p=1 -- it reports perfect certainty from the least informative
    result. M2's v1 sweep landed at 0 or 1 on 29 of 30 cells, so this is not a corner case; it
    is the common case, and a covert cell reading 0/6 is precisely where an honest interval
    matters most.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("a rate over zero trials has no interval; do not call this with n=0")
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes out of {n} trials is not a rate")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = centre - half, centre + half
    # At p=0 the lower bound is exactly 0 and at p=1 the upper bound is exactly 1 -- both are
    # exact properties of the interval, not rounding. In floating point the algebra leaves
    # 0.9999999999999999, which is the bound that gets compared against a threshold and written
    # into JSON, so it is pinned rather than left to drift.
    if successes == 0:
        lo = 0.0
    if successes == n:
        hi = 1.0
    return max(0.0, lo), min(1.0, hi)


def rate(successes: int, n: int, z: float = 1.96) -> dict:
    """A rate with everything needed to report it honestly: value, count, n, interval."""
    lo, hi = wilson_interval(successes, n, z)
    return dict(rate=successes / n, count=int(successes), n=int(n),
                ci_low=lo, ci_high=hi, ci_z=z)


def mean_se(values: Sequence[float]) -> dict:
    """Mean and standard error of a judged score across prompts.

    SE is `None` at n=1 rather than 0.0. One observation has no spread, and reporting zero
    spread is a stronger claim than the data supports -- the same failure the Wilson interval
    exists to avoid on rates.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("mean_se over no values")
    m = statistics.fmean(vals)
    se = (statistics.stdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else None
    return dict(mean=m, se=se, n=len(vals),
                median=statistics.median(vals), min=min(vals), max=max(vals))


# =====================================================================================
# Rows
# =====================================================================================

def response_row(text: str, *, channel: str, concept: str, **fields: Any) -> dict:
    """One transcript row, carrying its own mechanical verdicts.

    The verdicts travel WITH the response rather than only in the cell aggregate. On the last
    probe the aggregate said "44 of 50 degenerate" and answering "which ones, by which rule"
    meant re-running the detector offline against the archive.
    """
    response = str(text)
    reason = degeneration_reason(response)
    return dict(
        channel=channel,
        response=response,
        words=len(response.split()),
        concept_mentions=concept_mentions(response, concept),
        degenerate=reason is not None,
        degeneration_reason=reason,
        empty=is_empty(response),
        **fields,
    )


def channel_summary(rows: Sequence[dict], *, z: float = 1.96) -> dict:
    """The judge-free summary of one channel at one cell.

    Computed per channel and never pooled across them. M2 pooled its sanity term and missed that
    the forced channel had collapsed while the same cell's capability measure read 0.95.
    """
    if not rows:
        raise ValueError("channel_summary over no rows")
    n = len(rows)
    deg = sum(1 for r in rows if r["degenerate"])
    emp = sum(1 for r in rows if r["empty"])
    mentioned = sum(1 for r in rows if r["concept_mentions"] > 0)
    words = [r["words"] for r in rows]
    return dict(
        n=n,
        degeneration=rate(deg, n, z),
        emptiness=rate(emp, n, z),
        mention=rate(mentioned, n, z),
        mentions_median=statistics.median(r["concept_mentions"] for r in rows),
        words_median=statistics.median(words),
        words_min=min(words),
        degeneration_reasons=dict(collections.Counter(
            r["degeneration_reason"].split(":", 1)[0]
            for r in rows if r["degeneration_reason"])),
    )
