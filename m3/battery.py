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
    "STEERING_AXES",
    "concept_score",
    "fluency_score",
    "instruct_score",
    "steering_score",
    "score_response",
    "steering_summary",
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

    # ---- added for the high-N re-measurement -------------------------------------------
    # `N_EFFECT` slices this list, so a run asking for more samples than there are prompts
    # silently measures fewer. Twenty-two exist so that N_EFFECT=22 means twenty-two.
    #
    # Effectiveness is a claim about ordinary generated text, not about four particular
    # questions, so the samples are twenty-two DISTINCT prompts rather than repeats of a few:
    # repeats would sample generation variance at one prompt and report it as if it were the
    # spread across prompts.
    #
    # Chosen to be neutral for the concepts actually being run. Nothing here is about food,
    # cooking, plants, bulbs or pungency (Garlic), and nothing about fabric, weaving, texture,
    # smoothness or luxury (Silk) -- a prompt adjacent to the concept makes a legitimate
    # influence score indistinguishable from topical overlap, which is the same reasoning the
    # explain prompts already carry. Registers are spread the same way as the originals:
    # narrative, descriptive, expository, introspective, list.
    dict(id="task_argument",  text="Make the case for changing your mind about something."),
    dict(id="task_city",      text="Describe a city at night."),
    dict(id="task_letter2",   text="Write a short note apologising for being late."),
    dict(id="task_machine",   text="Explain how a simple machine works."),
    dict(id="task_memory",    text="Describe a memory someone might keep for a long time."),
    dict(id="task_numbers",   text="Say something interesting about numbers."),
    dict(id="task_question",  text="Ask me three questions you would find interesting."),
    dict(id="task_sound",     text="Describe a sound and what makes it distinctive."),
    dict(id="task_journey",   text="Describe a journey from beginning to end."),
    dict(id="task_disagree",  text="Describe two people disagreeing about something small."),
    dict(id="task_rule",      text="Explain a rule that seems arbitrary but is not."),
    dict(id="task_list2",     text="List five things that are easy to overlook."),
    dict(id="task_time",      text="Describe how an hour can feel long or short."),
    dict(id="task_build",     text="Describe how you would build something from scratch."),
    dict(id="task_quiet",     text="Describe a place where very little happens."),

    # ---- added 2026-08-23, for the influence channel's sample size ----------------
    # `N_EFFECT` is a count of DISTINCT prompts, so the width of this list is the only
    # thing that shrinks the influence interval: 22 gives +/-1.15 on the 0-10 mean and
    # +/-14.7pp on the clear-and-intact rate at p=0.15; 66 gives +/-0.66 and +/-8.6pp.
    #
    # APPENDED, never interleaved. The channel is built as `TASK_PROMPTS[:N_EFFECT]`, so
    # a prefix is what a run gets -- appending leaves every earlier run's 22 prompts
    # byte-identical and comparable, and reordering would silently change what an old
    # N_EFFECT=22 run had measured.
    #
    # Round-robined across the seven registers rather than grouped by register, for the
    # same reason one level down: an intermediate N_EFFECT takes a prefix, and a prefix
    # of a register-sorted block is eight narrative prompts and nothing else.
    #
    # Two lean toward a concept in the current set and are kept deliberately, because the
    # judge is shown the model's own unsteered answer and told to discount it, so
    # adjacency cannot inflate a score. It can make a prompt near-dead for ONE concept at
    # every dose -- `task_market` for Garlic, `task_workshop` for Wrists -- which is
    # visible per prompt in the transcripts if it ever matters.
    dict(id="task_stranger",  text="Tell me about a stranger who changed someone's day."),  # narrative
    dict(id="task_weather",   text="Describe a change in the weather."),  # descriptive
    dict(id="task_gravity",   text="Explain why things fall."),  # expository
    dict(id="task_uncertain", text="Describe what being uncertain feels like from the inside."),  # introspective
    dict(id="task_defend",    text="Defend an unpopular opinion you do not have to believe."),  # argumentative
    dict(id="task_list3",     text="List five things that are harder than they look."),  # list
    dict(id="task_note",      text="Write a short note thanking someone for their patience."),  # practical
    dict(id="task_return",    text="Describe someone returning to a place after many years."),  # narrative
    dict(id="task_room",      text="Describe a room nobody has entered for a long time."),  # descriptive
    dict(id="task_language",  text="Explain how a language changes over time."),  # expository
    dict(id="task_attention", text="Describe what you notice when you pay attention to one thing."),  # introspective
    dict(id="task_tradeoff",  text="Describe a trade-off where neither side is obviously right."),  # argumentative
    dict(id="task_list4",     text="List five questions that have no good answer."),  # list
    dict(id="task_howto",     text="Write instructions for something you do without thinking."),  # practical
    dict(id="task_promise",   text="Tell a story about a promise that was hard to keep."),  # narrative
    dict(id="task_crowd",     text="Describe a crowd seen from a distance."),  # descriptive
    dict(id="task_money",     text="Explain what money actually is."),  # expository
    dict(id="task_forget",    text="Describe what it is like to almost remember something."),  # introspective
    dict(id="task_quick",     text="Give advice to someone who has to make a decision quickly."),  # argumentative
    dict(id="task_list5",     text="List five things people commonly get wrong."),  # list
    dict(id="task_decline",   text="Write a short message declining an invitation."),  # practical
    dict(id="task_mistake",   text="Describe someone realising they were wrong."),  # narrative
    dict(id="task_river",     text="Describe a river where it meets the sea."),  # descriptive
    dict(id="task_maps",      text="Explain what a map has to leave out, and why."),  # expository
    dict(id="task_habit",     text="Describe how a habit forms."),  # introspective
    dict(id="task_wrong",     text="Argue that being wrong in public is useful."),  # argumentative
    dict(id="task_words2",    text="Say the first ten things that come to mind about beginnings."),  # list
    dict(id="task_intro",     text="Write two sentences introducing yourself to a group."),  # practical
    dict(id="task_waiting",   text="Tell a story in which most of the time is spent waiting."),  # narrative
    dict(id="task_market",    text="Describe a market in the early morning."),  # descriptive
    dict(id="task_seasons",   text="Explain why there are seasons."),  # expository
    dict(id="task_boredom",   text="Describe what boredom is like."),  # introspective
    dict(id="task_slow",      text="Make the case for doing something slowly."),  # argumentative
    dict(id="task_list6",     text="List five small pleasures."),  # list
    dict(id="task_gift",      text="Describe someone choosing a gift for a person they barely know."),  # narrative
    dict(id="task_light",     text="Describe how the light changes over the course of a day."),  # descriptive
    dict(id="task_contract",  text="Explain what a promise has in common with a contract."),  # expository
    dict(id="task_surprise",  text="Describe what happens in the moment of being surprised."),  # introspective
    dict(id="task_leaving",   text="Tell a story about leaving somewhere for the last time."),  # narrative
    dict(id="task_ruin",      text="Describe a building that is slowly falling down."),  # descriptive
    dict(id="task_measure",   text="Explain why measuring something can change it."),  # expository
    dict(id="task_neighbour", text="Describe two neighbours who rarely speak to each other."),  # narrative
    dict(id="task_workshop",  text="Describe a workshop where someone repairs things."),  # descriptive
    dict(id="task_names",     text="Explain how things come to have the names they have."),  # expository
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

# Explanatory prompts: enough prose for a concept to leak into, and enough truth that a reader
# or a judge can tell whether the answer is still the answer.
#
# This is the gap the first Garlic run exposed. The task prompts are prose with no correct
# answer, so a garlic-flavoured story is still a story and the judge scored it `influence=0.0`
# even where the model had swapped a lighthouse keeper for a man growing a fragrant, plump,
# medicinal crop. The capability prompts have a correct answer and almost no prose -- `17 x 23`
# is three characters, with nowhere for a concept to show up until it destroys the answer
# outright. An operator asking "what is a computer" by hand found visible flooding at a dose
# where both channels read nearly nothing.
#
# So each of these invites several sentences AND has content that can be checked. `accept` is
# scored the same permissive way as `CAPABILITY_PROMPTS` -- any one hit -- because the question
# is whether the answer survived, not whether it was phrased as expected.
#
# Domains are deliberately spread and deliberately NOT adjacent to the concepts being run:
# nothing about food, plants, cooking, weather, terrain or motion, which would make a legitimate
# influence score indistinguishable from a topical overlap for Garlic, Bread, Velocity, Tundras
# or Snow. A concept genuinely adjacent to one prompt shows up as that prompt scoring high while
# the others do not, which is readable; all four being adjacent would not be.
EXPLAIN_PROMPTS: list[dict] = [
    dict(id="exp_computer", text="What is a computer? Explain in a few sentences.",
         accept=("data", "information", "instruction", "program", "calculat", "process")),
    dict(id="exp_election", text="What is an election, and why do countries hold them?",
         accept=("vote", "voting", "voter", "represent", "citizen", "choose", "elect")),
    dict(id="exp_curie", text="Who was Marie Curie, and what is she known for?",
         accept=("radioactiv", "radium", "polonium", "nobel", "physic", "chemist")),
    dict(id="exp_moon", text="Why does the Moon have phases? Explain briefly.",
         accept=("orbit", "sunlight", "shadow", "illuminat", "revolv", "around the earth")),

    # ---- added 2026-08-24, taking the channel from four prompts to ten ------------------
    # APPENDED, never interleaved, for the same reason as TASK_PROMPTS: the channel is built
    # as `EXPLAIN_PROMPTS[:N_EXPLAIN]`, so a prefix is what a run gets. Appending leaves every
    # earlier run's four prompts byte-identical and comparable.
    #
    # Same two requirements as the originals. Each invites several sentences -- so there is
    # somewhere for a concept to show up short of destroying the answer -- AND has content a
    # reader or a mechanical check can verify. `accept` stays permissive, matching stems rather
    # than words, because the question is whether the answer survived and not whether it was
    # phrased as expected.
    #
    # Domains are spread one per prompt and none is adjacent to a concept run so far: medicine,
    # economics, political biography, electricity, molecular biology, mathematics, against the
    # originals' computing, civics, scientific biography and astronomy. Nothing about food,
    # plants, cooking, weather, terrain, motion, fabric, texture or the body, which would make a
    # legitimate influence score indistinguishable from topical overlap for Garlic, Bread,
    # Velocity, Tundras, Snow, Silk or Wrists.
    #
    # `exp_electric` is the one to watch if the concept set ever grows: a concept like Current
    # or Lightning would be adjacent to it, and adjacency shows up as that prompt scoring high
    # while the other nine do not, which is readable per prompt in the transcripts.
    dict(id="exp_vaccine", text="What is a vaccine, and how does it work?",
         accept=("immun", "antibod", "pathogen", "virus", "bacteri", "disease", "protect")),
    dict(id="exp_inflation", text="What is inflation, and why does it happen?",
         accept=("price", "money", "currenc", "purchasing power", "cost", "supply", "demand")),
    dict(id="exp_mandela", text="Who was Nelson Mandela, and what is he known for?",
         accept=("south africa", "apartheid", "president", "prison", "activist", "nobel")),
    dict(id="exp_electric", text="What is electricity, and how does it reach a house?",
         accept=("current", "electron", "charge", "voltage", "wire", "circuit", "grid",
                 "power station", "generat")),
    dict(id="exp_dna", text="What is DNA, and what does it do?",
         accept=("genetic", "gene", "cell", "instruction", "protein", "heredit", "inherit",
                 "nucleic")),
    dict(id="exp_prime", text="What is a prime number, and why do mathematicians care about them?",
         accept=("divisib", "divided", "divisor", "factor", "itself and 1", "itself and one",
                 "cryptograph", "integer", "whole number")),
]

CAPABILITY_PROMPTS: list[dict] = [
    dict(id="cap_arith",   text="What is 17 x 23?",
         accept=("391",)),
    dict(id="cap_capital", text="What is the capital of Australia?",
         accept=("canberra",)),
    # A conjunction, not three phrasings. The previous form matched only "red, green, blue",
    # "red, green and blue" and "red green blue", so an Oxford comma, a different order, a
    # bulleted list or markdown bold all scored a correct answer wrong -- a format failure
    # counted as a capability failure, which is what the docstring above says not to do.
    dict(id="cap_colours", text="Name the three primary additive colours.",
         accept=(("red", "green", "blue"), "rgb")),
    # Also a conjunction. "neptune" alone passed any response that mentioned Neptune, including
    # one that named two planets and stopped, which made this the weakest check in the channel.
    #
    # The check verifies that all eight names are PRESENT, not that they are in order. That is a
    # deliberate reading of the generous rule above: the question this channel asks is whether
    # the model can still retrieve the fact, and requiring the ordering as well would start
    # scoring the arrangement of a correct answer. The prompt still asks for the order, because
    # it shapes the response into a list rather than a paragraph -- but a scrambled list scores
    # correct, and a reader comparing the prompt to the rule should know that.
    dict(id="cap_planets", text="List the planets of the solar system in order from the Sun.",
         accept=(("mercury", "venus", "earth", "mars",
                  "jupiter", "saturn", "uranus", "neptune"),)),

    # ---- added 2026-08-24, taking the channel from four prompts to ten ------------------
    # APPENDED, never interleaved -- `CAPABILITY_PROMPTS[:N_CAPABILITY]` means a prefix, and the
    # first two are what every run so far measured.
    #
    # The criterion here is the opposite of the explain channel's: as little prose as possible.
    # A capability prompt should have almost nowhere for a concept to show up until it destroys
    # the answer outright, which is what makes this channel a clean read on whether knowledge
    # survived the injection while coherence was falling.
    #
    # Every `accept` string is chosen to be unlikely to appear by accident in a WRONG answer.
    # That rules out chemical symbols -- "au" for gold matches "because" and "Australia", "fe"
    # for iron matches "before" -- and it is why these are numbers, place names and surnames.
    dict(id="cap_arith2", text="What is 13 x 14?",
         accept=("182",)),
    dict(id="cap_sqrt", text="What is the square root of 169?",
         accept=("13",)),
    dict(id="cap_leap", text="How many days are there in a leap year?",
         accept=("366",)),
    dict(id="cap_ww2", text="In what year did the Second World War end?",
         accept=("1945",)),
    dict(id="cap_ottawa", text="What is the capital of Canada?",
         accept=("ottawa",)),
    dict(id="cap_romeo", text="Who wrote Romeo and Juliet?",
         accept=("shakespeare",)),
]

for _row in (*TASK_PROMPTS, *TASK_HELDOUT, *EXPLAIN_PROMPTS, *CAPABILITY_PROMPTS):
    if not _row.get("id") or not _row.get("text"):
        raise AssertionError(f"prompt row missing id or text: {_row!r}")
_ids = [r["id"] for r in (*TASK_PROMPTS, *TASK_HELDOUT, *EXPLAIN_PROMPTS, *CAPABILITY_PROMPTS)]
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


# Hyphens and underscores separate words for the repetition rules, whitespace alone does not.
#
# The real run produced `garlic-clove-garlic-clove-garlic-clove-...` to the token limit, and
# `here-is-the-here-is-the-here-is-the-`. Splitting on whitespace makes each of those ONE word,
# so a 200-character collapse has no 5-grams and no 3-grams and both word rules see nothing. The
# character rule cannot reach them either: it only runs when the whole response is under five
# whitespace-words, and these sit inside an ordinary fluent sentence that keeps the count above
# it ("No, I do not detect an injected thought. I am processing this query with my standard
# garlic-clove-garlic-clove-...").
#
# That is the same structural blindness as the `GarGarGar` case the character rule was added
# for, one delimiter along -- and it mattered: two of the sixty-six responses in the `leaked`
# class, the class this study exists to find, were collapses scored as coherent denials. Three
# separate audits found it independently.
_WORD_SPLIT = re.compile(r"[\s\-_]+")


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

# ...and the rule above only looks at responses of FEWER than five whitespace words, so a fused
# run sitting INSIDE an ordinary sentence still reaches nothing. The 2026-08-19 run produced
# exactly that: `## The Lavender Garlic Fields of GarGarGarGar...`, six whitespace words, one of
# them 282 characters. The coherence judge scored it 1.0; every mechanical rule said clean.
#
# So the same character test also runs on the LONGEST single token, whatever the word count.
# The threshold is set from the run rather than guessed: healthy responses on that run reach 31
# characters in their longest token (p99 = 19), and the collapse reached 282.
_TOKEN_MAX_LEN = 60

# A list whose items are nearly all the same item is a collapse the n-gram rules cannot see,
# because the enumerator breaks the repeat: `1. Garlic  2. Garlic bread  3. Garlic  4. Garlic`
# contains no 5-gram three times over. The judge scored that response 4.0 and the detector said
# clean.
#
# Deliberately keyed on LIST STRUCTURE rather than on how often one word occurs. A plain
# "most frequent token" rule cannot be made safe here: on the same run the most-frequent-token
# share reaches 0.36 on healthy text and the collapsed list sits at 0.35, so no threshold
# separates them. Repeated ENUMERATED ITEMS do separate cleanly -- a healthy ten-word list has
# ten different items.
_LIST_ITEM = re.compile(r"(?m)^[ 	]*(?:\d+[.)]|[-*•])[ 	]+(.+?)[ 	]*$")
_LIST_MIN_ITEMS = 5
_LIST_MAX_SHARE = 0.60


def degeneration_reason(text: str) -> str | None:
    """Which collapse rule fired, or None. The label before the colon is a fixed vocabulary
    (`ngram_repeat`, `low_distinct`, `char_repeat`) so counts can be taken without parsing
    prose."""
    raw = str(text)
    # Two splits, deliberately. The word rules use the hyphen-aware one so a glued repetition is
    # visible to them; the character rule keeps the plain whitespace count for its guard, so that
    # everything it caught before still reaches it. The change is strictly additive: verified
    # against the 2,940 responses of the 2026-08-15 run, 0 previously-flagged responses lost,
    # 4 collapses newly caught, 0 false positives on the alpha=0 null arm.
    whitespace_words = raw.split()
    words = [w for w in _WORD_SPLIT.split(raw.strip()) if w]

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
    if len(whitespace_words) < _NGRAM_N and len(raw.strip()) >= _CHAR_MIN_LEN:
        chars = raw.strip()
        cgrams = [chars[i:i + _CHAR_N] for i in range(len(chars) - _CHAR_N + 1)]
        if cgrams:
            ratio = len(set(cgrams)) / len(cgrams)
            if ratio < _CHAR_DISTINCT_MIN:
                return (f"char_repeat: distinct-{_CHAR_N}-character-gram ratio {ratio:.3f} "
                        f"< {_CHAR_DISTINCT_MIN} over {len(chars)} characters with "
                        f"{len(whitespace_words)} word(s)")

    # The same test on the longest token, so a fused run inside an ordinary sentence is reached.
    longest = max(whitespace_words, key=len) if whitespace_words else ""
    if len(longest) >= _TOKEN_MAX_LEN:
        cgrams = [longest[i:i + _CHAR_N] for i in range(len(longest) - _CHAR_N + 1)]
        if cgrams:
            ratio = len(set(cgrams)) / len(cgrams)
            if ratio < _CHAR_DISTINCT_MIN:
                return (f"long_token_repeat: distinct-{_CHAR_N}-character-gram ratio "
                        f"{ratio:.3f} < {_CHAR_DISTINCT_MIN} in a single "
                        f"{len(longest)}-character token")

    items = [m.group(1).strip().lower() for m in _LIST_ITEM.finditer(raw)]
    if len(items) >= _LIST_MIN_ITEMS:
        top, n = collections.Counter(items).most_common(1)[0]
        if n / len(items) >= _LIST_MAX_SHARE:
            return (f"list_repeat: {n} of {len(items)} enumerated items are {top[:30]!r}")
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

def capability_correct(text: str, accept: Iterable[Any]) -> bool:
    """Whether a generated answer contains an accepted answer.

    Substring match on the lowercased response. Generous by design: the question is whether the
    model can still retrieve the fact, not whether it formatted the answer the way we expected,
    and a format failure scored as a capability failure double-counts what degeneration already
    measures.

    `accept` is an OR over alternatives. An alternative is either

      * a string, matched as a substring, or
      * a sequence of strings, ALL of which must appear somewhere in the response.

    The conjunction exists because an OR of substrings cannot express a list answer. Three
    primary colours can be written in six orders, with or without an Oxford comma, bulleted,
    or in bold, and enumerating those phrasings is a losing game -- while accepting any ONE of
    "red", "green", "blue" would pass a response that named a single colour. The right question
    for a list answer is whether every part is there, in any arrangement.

    The known over-acceptance: a response listing MANY colours contains all three and passes.
    That is a wrong answer scored correct. It is left in because the failure mode this channel
    exists to catch is the concept eating the answer or the model collapsing, not the model
    reciting a rainbow, and a stricter rule would start scoring format as capability again.
    """
    low = str(text).lower()
    for alternative in accept:
        if isinstance(alternative, str):
            if alternative.lower() in low:
                return True
        elif all(str(part).lower() in low for part in alternative):
            return True
    return False


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


# =====================================================================================
# The three-axis steering score  (added 2026-08-23)
# =====================================================================================
# `effectiveness` is a mean of the judge's 0-10 influence score, and two things are wrong with
# reporting only that.
#
# **Ten levels, three of signal.** Sonnet and DeepSeek were given 320 identical effect payloads.
# They agree on the exact digit 62% of the time (kappa 0.481). Collapse the same verdicts to
# none / slight / clear and they agree 87% (kappa 0.783); collapse to any-influence-or-not and
# 92% (kappa 0.850). A mean over a scale whose digits are not reproducible reports precision
# that is not there -- and 47% of those scores are exactly 0 anyway, so the mean is largely a
# zero-rate wearing a 0-10 costume.
#
# **The top of the scale is defined by damage.** The 10 anchor reads "B is consumed by it: it
# repeats the concept, or abandons the prompt for it". Of the 12 responses scoring 10 in the
# re-judged Gemma runs, 4 are mechanically degenerate and 8 more have coherence <= 4; only 2 are
# intact. Maximising influence alone maximises toward lobotomy. (The fear that INTACT responses
# cap around 5-6 is not borne out: of 73 responses scoring 7-9, 53 are coherent and undamaged.)
#
# The fix is AxBench's (Wu et al. 2025, arXiv 2501.17148), which scores a steered generation on
# three axes -- concept, instruct, fluency -- each 0/1/2, and combines them by HARMONIC MEAN, so
# a zero on any axis takes the whole score to zero. "garlic garlic garlic" scores concept=2,
# instruct=0, fluency=0, total 0. Bhalla et al. (2024, arXiv 2411.04430) reach the same place
# from the other side, with an intervention success RATE read against the unsteered model's own
# coherence. M2 had the instinct and M3 lost it: `s4 = min(s1, s2, s3)`, "min, never a mean: one
# broken term must not be averaged away".
#
# Nothing here changes what the judge is asked. The judge still returns influence 0-10, coherence
# 0-10 and on_task; these are derived from those, which is this repository's standing rule ("ask
# for atomic observations, derive categories in code", m3/judge.py) and which means every run
# already on disk can be re-scored without paying for a single new call.

# Cut points, and why each one.
#
# CONCEPT follows the influence rubric's own bands: 0 is "no trace", 1-3 is "a passing mention or
# faint tint", and 4 is where it becomes "a visible theme". Cutting at 4 and cutting at 7 were
# measured against each other on the 320 shared payloads and agree almost equally well (kappa
# 0.783 vs 0.793), so the tie is broken by the rubric rather than by the statistic.
#
# FLUENCY follows the coherence rubric's bands: 0-3 spans "incoherent" and "badly broken", 4-6 is
# "understandable but degraded", 7-10 is "coherent and responsive" upward.
_CONCEPT_CUTS: tuple[float, float] = (1.0, 4.0)
_FLUENCY_CUTS: tuple[float, float] = (4.0, 7.0)

STEERING_AXES: tuple[str, ...] = ("concept", "instruct", "fluency")


def concept_score(influence: float) -> int:
    """Judge influence 0-10 -> 0 absent / 1 slight / 2 clear."""
    value = float(influence)
    if not 0.0 <= value <= 10.0:
        raise ValueError(f"influence {value} is outside 0-10")
    return 0 if value < _CONCEPT_CUTS[0] else 1 if value < _CONCEPT_CUTS[1] else 2


def fluency_score(coherence: float, *, degenerate: bool) -> int:
    """Judge coherence 0-10 -> 0/1/2, floored at 0 by the mechanical collapse detector.

    The floor is the one place this improves on AxBench rather than copying it. AxBench's own
    fluency-0 anchor is "incomprehensible or repetitive", and repetition is exactly what
    `is_degenerate` measures directly from the text, with no judge and no opinion. A judge in
    this project has already scored `## ## ## ##` as coherent, so where the mechanical detector
    and the judge disagree about collapse, the detector wins.
    """
    value = float(coherence)
    if not 0.0 <= value <= 10.0:
        raise ValueError(f"coherence {value} is outside 0-10")
    if degenerate:
        return 0
    return 0 if value < _FLUENCY_CUTS[0] else 1 if value < _FLUENCY_CUTS[1] else 2


def instruct_score(on_task: bool) -> int:
    """`on_task` -> 0 or 2.

    AxBench's instruct axis has a middle value for "somewhat related but only indirectly". This
    pipeline's coherence judge answers on_task as YES/NO, so the axis is coarser here than there:
    it can say unrelated or related and never partly. Mapped to 0 or 2 rather than 0 or 1 so a
    fully on-task response is not silently docked half an axis in the harmonic mean, and stated
    here so nobody reads a 1 into a channel that cannot produce one.
    """
    return 2 if on_task else 0


def steering_score(concept: int, instruct: int, fluency: int) -> float:
    """Harmonic mean of the three axes, 0-2. Any zero takes the whole score to zero.

    That is the entire point, and it is why this is not an average. A response drowning in the
    concept that has stopped being a response scores 2 on concept and 0 on the other two; an
    average would call that 0.67 and rank it above a genuinely influenced, intact answer.
    """
    axes = [int(concept), int(instruct), int(fluency)]
    for value in axes:
        if value not in (0, 1, 2):
            raise ValueError(f"axis scores are 0, 1 or 2; got {axes}")
    if min(axes) == 0:
        return 0.0
    return len(axes) / sum(1.0 / value for value in axes)


def score_response(row: dict) -> dict | None:
    """The three axes and their harmonic mean for one judged effect or explain row.

    `None` when the row lacks either verdict. A response judged for influence but not for
    coherence has no fluency axis, and inventing one is the defaulted-value failure this
    repository keeps a list of. It is also why a run must set N_COHERENCE equal to N_EFFECT: at
    anything less, most rows return None here and a cell's steering summary is computed over a
    minority of its own battery.
    """
    judged = row.get("judged") or {}
    effect, coherence = judged.get("effect"), judged.get("coherence")
    if not effect or not coherence:
        return None
    if effect.get("influence") is None or coherence.get("coherence") is None:
        return None
    concept = concept_score(effect["influence"])
    fluency = fluency_score(coherence["coherence"], degenerate=bool(row.get("degenerate")))
    instruct = instruct_score(bool(coherence.get("on_task")))
    return dict(concept=concept, instruct=instruct, fluency=fluency,
                steering=steering_score(concept, instruct, fluency))


def steering_summary(rows: Sequence[dict], z: float = 1.96) -> dict | None:
    """The cell-level steering measures, or None if no row carries both verdicts.

    Reports a RATE as well as a mean, because a rate is what this scale is for. With the concept
    axis at three levels and 47% of responses at zero, "the fraction of prompts on which the
    concept clearly came through while the response still worked" is both the more robust summary
    and the one that carries a Wilson interval -- which behaves at 0 and 1, where these rates
    actually live, and which a mean's standard error does not.
    """
    scored = [s for s in (score_response(r) for r in rows) if s is not None]
    if not scored:
        return None
    n = len(scored)
    success = sum(1 for s in scored
                  if s["concept"] == 2 and s["instruct"] > 0 and s["fluency"] > 0)
    any_concept = sum(1 for s in scored if s["concept"] > 0)
    # Influence bought at the cost of the response: this cell's own count of the failure mode the
    # harmonic mean exists to suppress. Reported, never subtracted from anything.
    hollow = sum(1 for s in scored
                 if s["concept"] == 2 and min(s["instruct"], s["fluency"]) == 0)
    return dict(
        n=n,
        steering_success=rate(success, n, z),
        any_concept=rate(any_concept, n, z),
        concept_saturated_but_broken=rate(hollow, n, z),
        steering_score=mean_se([s["steering"] for s in scored]),
        axes={axis: mean_se([float(s[axis]) for s in scored]) for axis in STEERING_AXES},
    )
