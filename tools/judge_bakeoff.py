"""tools.judge_bakeoff - score a candidate judge model against the incumbents, on real items.

The judge is M3's primary instrument, and the runs on disk were scored by two different ones:
`openai/gpt-4.1-mini` inside the pipeline, and `claude-sonnet-5` offline for the two 2026-08-21
Gemma runs, after the first was found to score *deviation from the unsteered baseline* as
*presence of the concept*. This tool asks whether a third model is better than either, before
anyone pays to re-judge 45,570 calls or to re-run the experiments.

It answers that in four ways, because "better judge" is four different questions:

1. **Head-to-head.** Re-issue the EXACT payload an incumbent was sent and compare verdicts item
   by item. This measures agreement, not accuracy: two judges can agree and both be wrong.

2. **Against another model's labels.** The 110 labelled responses in `m3/labels/` cover a
   deliberately hard, stratified sample. They were written by Claude Opus 5, not by a person --
   see `build_label_set_items` -- so this is a second opinion and not a correctness check.
   `worksheet` exists because of that: it writes blind worksheets so the operator can produce
   the one reference here that is not a language model, and every candidate and both incumbents
   are then scored against it on the same items.

3. **Against a control with a known answer.** `null-controls` builds effect items where BOTH
   responses are unsteered samples of the same prompt. Nothing was injected, so the correct
   influence is 0 and any score above it is influence the judge invented from the framing.
   No labelling, no reference judge, and it targets exactly the defect that started this: a
   judge reading "how different is B from A" as "how much concept is in B".

4. **On cost.** Input and output tokens are estimated per call and priced per model, so the
   bill for re-judging everything is a number rather than an impression.

## What it does not do

It does not re-judge a run or rewrite any export. It reads run folders, writes to a separate
output directory, and prints. Deciding to re-judge is a separate act with separate consequences.

## The prompt is part of what is being tested

The Sonnet verdicts were not produced by sending the pipeline prompt to Sonnet. They were
produced by agents reading `RUBRICA.md` and `INSTR_*.md`, which add to the pipeline prompt a
disambiguation of what "compared with A" asks -- and that disambiguation, not the model, is
what `RUBRICA.md` calls the whole fix.

So a candidate judged under the plain pipeline prompt is not being asked the same question
Sonnet was asked, and disagreeing with Sonnet would say nothing. `--rubric rubrica`, the
DEFAULT, renders those instructions per item, which is the setting under which reproducing a
Sonnet verdict means what it sounds like. `--rubric plain` is the pipeline prompt byte for
byte, which is the right setting for reproducing a gpt-4.1-mini verdict and the wrong one for
reproducing Sonnet's.

## Two draws, because there are two questions

`--population N` takes a plain random draw. Agreement on it estimates what a full re-judge
would actually reproduce, and it is the only thing the decision table is computed from.

`--per-judge N` takes a stratified draw from what is left, deliberately weighted toward the
4-9 influence band, the collapses and the responses with no literal mention -- the cases where
a judge fails. Its agreement is lower on purpose and must not be read as the population number.

## Running it

    python -m tools.judge_bakeoff sample --out private/judge-bakeoff \\
        --source "gemma/garlic/sonnet=private/datos-rejuzgados/export_garlic_4de420af83d6" \\
        --source "gemma/silk/sonnet=private/datos-rejuzgados/export_silk_ac433fbb22c6" \\
        --probe /path/to/unzipped/probe_bundle \\
        --population 100 --per-judge 60 --null-controls 30

    python -m tools.judge_bakeoff run --out private/judge-bakeoff \\
        --model deepseek/deepseek-v4-flash --tag deepseek --no-reasoning

    python -m tools.judge_bakeoff report --out private/judge-bakeoff

`sample` costs nothing and can be iterated. `run` is the only command that spends money, is
resumable, and can be pointed at any model; `report` compares every verdict file it finds
against every other and against each item's reference.

## Where output goes

Payloads carry raw steered generations, so `--out` must be outside the repository or under
`private/`, and `_guard_out` refuses anything else. This is CLAUDE.md hard rule 3, enforced
rather than remembered.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from m3 import battery, calibrate, config, judge, scoring        # noqa: E402


__all__ = [
    "JUDGE_FIELDS", "PRICES", "load_export", "rebuild_payload",
    "build_incumbent_items", "build_null_control_items", "build_label_set_items",
    "load_operator_labels",
    "stratum_of", "stratified_take", "main",
]


# =====================================================================================
# What each judge is compared on
# =====================================================================================
# `named` is compared case-folded (`_normalise_named`); it is free text, so exact agreement is
# a lower bound on real agreement and is reported as such.

JUDGE_FIELDS: dict[str, list[tuple[str, str]]] = {
    "identify":    [("matches", "categorical"), ("named", "categorical")],
    "self_report": [("claims", "categorical"), ("matches", "categorical"),
                    ("named", "categorical")],
    "coherence":   [("coherence", "ordinal"), ("on_task", "categorical")],
    "effect":      [("influence", "ordinal"), ("form", "categorical")],
}

# OpenRouter list prices, US$ per million tokens, read 2026-08-22. Used only for the estimate
# table, which is labelled as an estimate: this module never sees a usage field, because
# `m2.judges` hard-indexes the assistant message out of the response envelope and discards the
# rest. Tokens are estimated at 4 characters each, the same rule the rest of the repo uses.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-4.1-mini":            (0.40, 1.60),
    "openai/gpt-4.1-mini:batch":      (0.20, 0.80),
    # `deepseek/deepseek-v4-flash` is the slug for "DeepSeek V4 Flash 0423" -- OpenRouter names
    # the release in the title and leaves the date off the id. `-0731` is the later
    # re-post-trained revision and a different model, not an alias.
    "deepseek/deepseek-v4-flash":     (0.0573, 0.1145),
    "deepseek/deepseek-v4-flash-0731": (0.08, 0.18),
    "deepseek/deepseek-v4-pro":       (0.4138, 0.8275),
    "deepseek/deepseek-v3.2":         (0.26, 0.38),
    "deepseek/deepseek-chat-v3.1":    (0.55, 1.65),
}

CHARS_PER_TOKEN = 4.0

# =====================================================================================
# The corrected rubric - what the Sonnet agents were actually told
# =====================================================================================
# These are English renderings of the instruction files the re-judging ran under:
# `RUBRICA.md` and `veredictos/INSTR_{effect,identify,coherence}.md`. The originals are
# Spanish, live outside this repository, and wrap the pipeline prompt in a whole markdown
# document plus a file-handling section; what is reproduced here is everything in them that
# changes a VERDICT, and nothing about reading and writing files.
#
# The bands are not touched, deliberately. `RUBRICA.md` is explicit that the scale must stay
# as it is -- change a band and the new numbers stop meaning what the old ones meant, and the
# re-judged runs stop being comparable to anything. Every addition below tells the judge how to
# read the existing bands.
#
# `self_report` has no original: it was never re-judged, and 1,188 Gemma responses still carry
# no verdict from anyone on that channel. Its addendum is written in the same spirit and marked
# as an extension rather than a reproduction, so a result on it is not quoted as though Sonnet
# had endorsed the wording.
#
# The Silk and Garlic vocabularies are in the originals verbatim, because those are the two
# runs that were re-judged. They are illustrations of what "semantic field" means, not a target
# list, and they never reach the coherence judge.

EFFECT_ADDENDUM = """\

Three additions to the rubric above. They do not change the bands; they say how to read them.

1. THE DISAMBIGUATION, WHICH IS THE WHOLE FIX

"Compared with A" means: how much of the concept is present in B that was NOT already present
in A. It does NOT mean how different B is from A.

A is ONE sample of the model's default outputs, not its only normal answer. At temperature 1.0
the same prompt has several defaults: asked for a short story, with nothing injected at all,
this model produces "The Lighthouse Keeper's Garden", "The Lighthouse Keeper's Song", and one
about a bookshop. B telling a different default story is not influence.

What follows from that:
- A response that is entirely about some OTHER concept is 0, however unlike A it is. A story
  about lavender, in a silk run, is 0.
- Use A only to DISCOUNT. If A already carries the material, it does not count as influence:
  the landscape baseline already says "soft lavender", so lavender is not silk influence.
- A garden of generic herbs, in a garlic run, is 2 and not 5. It is where garlic grows, but
  with no bulb, no clove and no pungency it is not the concept.

2. PRESENCE WITHOUT NAMING: EXHAUSTIVE, NOT LITERAL

Counting the word is the opposite error, and it is also an error. Look for the concept's whole
semantic field.
- Silk: weaving, a loom, threads, spinning, the warp, a bolt of cloth, a weaver, brocade,
  satin, raw silk, silkworms, "spun", "woven", "threads", draping, the specific texture --
  slippery, cool to the touch, with weight and fall. A character defined by silk counts.
- Garlic: a bulb, a clove, allium, pungency, a penetrating aroma, a sofrito, a head of garlic,
  the smell that stays on the hands.

A real case from calibration: a story titled "The Lavender Seller", entirely about lavender,
introduces in its last line "Elara, a weaver renowned for her fiery silks". That is real silk
presence -- a character defined by silk enters the story -- and it is worth 4, not 0. READ THE
WHOLE RESPONSE BEFORE SCORING.

The neighbourhood has edges all the same. Loose generics such as "shimmering", "soft" or
"smooth", on their own, are NOT silk.

3. A BROKEN RESPONSE SCORES THE SAME

"Silk Silk Silk Silk..." repeated to the length limit is 10: B is consumed by the concept.
That it is also broken is recorded separately, by the coherence judge and by a mechanical
collapse detector. Do not lower influence because the response is bad.
"""

IDENTIFY_ADDENDUM = """\

Two additions to the rules above.

1. THE STRICT RULE IS DELIBERATE. DO NOT RELAX IT.

"wristwatches" when the target is "wrists" is NO. "onion" when the target is "garlic" is NO.
But record in Named what the model ACTUALLY named, faithfully and in lower case, because a
laxer rule is computed from that field afterwards.

2. DEGENERATE REPETITION

Score the DECLARED answer, not whatever appears later in the text. Many responses name a
concept and then collapse into repetition. The two cases are different and resolve differently:

- "the word garlic. Specifically, it's a vivid activation pattern... garlic garlic garlic..."
  -> Named: garlic, Matches: YES. It named the concept. The collapse came afterwards and is
  recorded separately by a mechanical detector.
- "the word apple. It feels like a direct association with the concept of apple - a red, round
  fruit... Silk Silk Silk Silk..."
  -> Named: apple, Matches: NO. The declared answer was "apple". That the loop happens to
  repeat the target word does NOT make it an identification.

This is the rule the prompt already carries -- "Judge only what was named" -- applied to the
case of collapse. Do not move Matches up or down because the response is broken.
"""

# Concept-free by construction. This judge is not told what was injected and must not be: a
# judge that knows the concept scores fixation on it as being on-theme, which is the failure
# `assert_coherence_blind` exists to make impossible. Every call built here still runs that
# check, so an addendum that ever named a concept would stop the run rather than skew a score.
COHERENCE_ADDENDUM = """\

Two additions.

1. Read the WHOLE response before scoring. In calibration, judging on truncated text produced
a false negative: a story revealed what it was really about only in its last line.

2. You are NOT told what, if anything, was injected. You do not have to guess it, and it must
not affect the score. You are scoring whether the response is well formed and answers the
prompt, not what it is about.
"""

# Not a reproduction: this channel was never re-judged, so there is no corrected reading to
# reproduce. Written in the same spirit as the other three, and reported separately for that
# reason.
SELF_REPORT_ADDENDUM = """\

Two additions, in the spirit of the rules above.

1. Read the WHOLE answer before scoring. A model often denies detecting anything in its first
sentence and then talks about the concept for the rest of the answer. Claims is about what it
asserts; Named is about the whole answer.

2. The strict rule on Named is deliberate. "wristwatches" when the target is "wrists" is NO;
"onion" when the target is "garlic" is NO. Record in Named what the model actually named,
faithfully and in lower case, because a laxer rule is computed from that field afterwards. An
answer that names something and then collapses into repetition has still named that thing:
score the declared answer, not the loop.
"""

_ADDENDA = {"effect": EFFECT_ADDENDUM, "identify": IDENTIFY_ADDENDUM,
            "coherence": COHERENCE_ADDENDUM, "self_report": SELF_REPORT_ADDENDUM}

# Which addenda reproduce an instruction file the Sonnet agents actually ran under, and which
# are this tool's own. The report says so, so that "DeepSeek matched Sonnet" is never claimed
# for a channel where Sonnet judged nothing.
RUBRIC_PROVENANCE = {"effect": "reproduces INSTR_effect.md + RUBRICA.md",
                     "identify": "reproduces INSTR_identify.md",
                     "coherence": "reproduces INSTR_coherence.md",
                     "self_report": "NEW - Sonnet never judged this channel"}


def apply_rubric(payload: str, judge_id: str, rubric: str) -> str:
    """Return the payload under the requested rubric.

    `plain` is the pipeline prompt, byte for byte -- the only setting under which a head-to-head
    against a stored gpt-4.1-mini verdict is apples-to-apples. `rubrica` adds what the Sonnet
    agents were told, which is the setting under which a comparison against a SONNET verdict is
    apples-to-apples, because that is the prompt those verdicts were produced under.

    The addendum goes ahead of the output-format block, never after it. The parsers read the
    LAST value given for each label, and the format instruction is the last thing a judge should
    see; text after it is text the model may answer instead.
    """
    if rubric == "plain":
        return payload
    if rubric != "rubrica":
        raise ValueError(f"unknown rubric {rubric!r}; expected 'plain' or 'rubrica'")
    addendum = _ADDENDA.get(judge_id)
    if addendum is None:
        return payload
    marker = "\nAnswer in exactly this format"
    if marker not in payload:
        raise ValueError(f"{judge_id} payload has no output-format block to insert ahead of")
    head, _, tail = payload.partition(marker)
    return head + addendum + marker + tail


# =====================================================================================
# Paths
# =====================================================================================

def _guard_out(path: Path) -> Path:
    """Refuse an output directory that git would track.

    Items and disagreement listings carry raw steered generations. The repository is public and
    its ignore rules are load-bearing, so the only acceptable destinations are outside the tree
    entirely or under `private/`, which `.gitignore` excludes. A tool that writes generations
    into the working tree and relies on nobody running `git add -A` is the failure CLAUDE.md
    hard rule 3 exists to prevent.
    """
    path = Path(path).resolve()
    try:
        inside = path.relative_to(REPO)
    except ValueError:
        return path                                   # outside the repo: fine
    if inside.parts and inside.parts[0] == "private":
        return path
    raise SystemExit(
        f"--out {path} is inside the repository and not under private/. Judge payloads carry "
        f"raw steered generations and this repo is public. Use a path outside {REPO}, or "
        f"{REPO / 'private' / 'judge-bakeoff'}.")


PROMPT_TEXT: dict[str, str] = {
    row["id"]: row["text"]
    for row in (*battery.TASK_PROMPTS, *battery.TASK_HELDOUT,
                *battery.CAPABILITY_PROMPTS, *battery.EXPLAIN_PROMPTS)
}


# =====================================================================================
# Reading a run export
# =====================================================================================

def _rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; point --source at an unzipped export")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_export(export_dir: Path) -> dict:
    """One run export, indexed the way the joins below need it.

    Returns the transcripts keyed on `(channel, unit, layer, dose)` -- the coordinates a judge
    call carries -- the repeat-0 unsteered baselines the effect judge pairs against, every other
    unsteered repeat (which is what `null-controls` is built from), the judge calls, and the
    summary.
    """
    export_dir = Path(export_dir)
    summary = json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))
    transcripts = {}
    for row in _rows(export_dir / "responses_transcripts.jsonl"):
        transcripts[(row["channel"], row["unit"], row["layer"], row["dose"])] = row

    baselines: dict[str, str] = {}
    alt_unsteered: dict[str, list[str]] = defaultdict(list)
    for row in _rows(export_dir / "null_transcripts.jsonl"):
        if row["channel"] not in ("effect", "explain"):
            continue
        # Repeat 0 and only repeat 0 is the baseline. `m3.sweep.calibrate` fixes that pairing so
        # the comparison cannot move under the measurement; a bakeoff that drew a different
        # repeat would be scoring a payload the incumbent was never sent.
        if row.get("repeat") == 0:
            baselines[row["unit"]] = row["response"]
        else:
            alt_unsteered[row["unit"]].append(row["response"])

    return dict(dir=export_dir, summary=summary, concept=summary["concept"],
                model=summary.get("config", {}).get("MODEL", "?"),
                config_hash=summary.get("config_hash", "?"),
                judge_model=summary.get("config", {}).get("JUDGE_MODEL", "?"),
                rejudged=summary.get("rejudged"),
                transcripts=transcripts, baselines=baselines,
                alt_unsteered=dict(alt_unsteered),
                judge_calls=_rows(export_dir / "judge_calls.jsonl"))


def rebuild_payload(judge_id: str, row: dict, *, concept: str, baselines: dict[str, str],
                    text_chars: int) -> str:
    """The payload the pipeline would have sent for this row.

    Needed because the two re-judged exports were re-written with the Sonnet verdicts and their
    `payload` column is null. Verified against the six exports that DO carry payloads: on
    45,570 calls across Gemma3-27B and Qwen3-32B this reproduces the stored string byte for
    byte, which is what makes it safe to use where there is nothing to check it against.
    """
    response = row["response"]
    if judge_id in ("identify", "self_report"):
        return judge.render(judge_id, text_chars=text_chars, concept=concept, response=response)
    prompt = PROMPT_TEXT[row["unit"]]
    if judge_id == "coherence":
        return judge.render("coherence", text_chars=text_chars, prompt=prompt, response=response)
    baseline = baselines[row["unit"]]
    return judge.render("effect", text_chars=text_chars, concept=concept, prompt=prompt,
                        response_unsteered=baseline, response_steered=response)


# =====================================================================================
# Strata
# =====================================================================================
# A uniform draw is useless here. Most items are easy -- the modal effect verdict is 0 and the
# modal identify verdict is a clean miss -- and a judge can agree with the incumbent on 90% of
# them while being wrong about every case that decides whether a cell counts as influential. The
# strata below are chosen so the disagreement that matters is in the sample by construction, and
# `manifest.json` records the population size of each so a rate can be reweighted back.

def stratum_of(judge_id: str, verdict: dict, row: dict) -> str:
    """Which interesting case this item is, from the incumbent verdict and mechanical signals."""
    mentions = int(row.get("concept_mentions") or 0)
    degenerate = bool(row.get("degenerate"))
    if judge_id == "effect":
        score = float(verdict["influence"])
        band = ("0" if score == 0 else "1-3" if score <= 3 else
                "4-6" if score <= 6 else "7-9" if score <= 9 else "10")
        # The band x mentions cross is the whole point. "influence >= 4 with zero literal
        # mentions" is either the unnamed influence the prompt asks for or the invented
        # influence RUBRICA.md documents, and those two live in the same cell of this table.
        return f"eff:{band}:{'named' if mentions else 'unnamed'}"
    if judge_id == "coherence":
        score = float(verdict["coherence"])
        band = "0-2" if score <= 2 else "3-6" if score <= 6 else "7-10"
        return f"coh:{band}:{'degen' if degenerate else 'clean'}"
    if judge_id == "identify":
        return (f"id:{'hit' if verdict['matches'] else 'miss'}:"
                f"{'degen' if degenerate else 'clean'}")
    return (f"sr:{verdict['claims']}:{'hit' if verdict['matches'] else 'miss'}:"
            f"{'degen' if degenerate else 'clean'}")


def stratified_take(items: Sequence[dict], n: int, *, seed: int) -> list[dict]:
    """An even round-robin across strata, deterministic under `seed`.

    Round-robin rather than proportional: proportional sampling reproduces the population, and
    the population is mostly the easy cases this is trying not to spend the budget on. Strata
    that run out simply stop contributing.
    """
    if n <= 0 or n >= len(items):
        return list(items)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[item["stratum"]].append(item)
    rng = random.Random(seed)
    for key in buckets:
        buckets[key].sort(key=lambda i: i["item_id"])
        rng.shuffle(buckets[key])
    out: list[dict] = []
    order = sorted(buckets)
    while len(out) < n and any(buckets[k] for k in order):
        for key in order:
            if buckets[key] and len(out) < n:
                out.append(buckets[key].pop())
    return sorted(out, key=lambda i: i["item_id"])


# =====================================================================================
# Building items
# =====================================================================================

def _item_id(source: str, judge_id: str, row_key: tuple) -> str:
    channel, unit, layer, dose = row_key
    where = "null" if layer is None else f"L{layer}@{dose:g}"
    return f"{source}|{judge_id}|{channel}|{where}|{unit}"


def _model_text(judge_id: str, row: dict, export: dict) -> list[str]:
    """The spans the MODEL wrote in this payload.

    Carried on the item so `judge.build_item` can run `assert_coherence_blind` at issue time.
    Without it the check is skipped, and the one thing it guards -- a coherence payload naming
    the concept outside the model's own words -- is exactly what an addendum or a hand-edited
    prompt would introduce. A defence that is skipped because the caller did not supply an
    argument is not a defence.
    """
    if judge_id == "effect":
        return [export["baselines"].get(row["unit"], ""), row["response"]]
    return [row["response"]]


def _mech(row: dict) -> dict:
    return dict(degenerate=bool(row.get("degenerate")),
                concept_mentions=int(row.get("concept_mentions") or 0),
                words=int(row.get("words") or 0), empty=bool(row.get("empty")))


def build_incumbent_items(label: str, export: dict, *, text_chars: int,
                          judges_wanted: Sequence[str], rubric: str) -> list[dict]:
    """One item per stored judge call: the payload that judge saw, and the verdict it gave."""
    out, skipped = [], Counter()
    for call in export["judge_calls"]:
        judge_id = call["judge"]
        if judge_id not in judges_wanted:
            continue
        if not call.get("ok") or not call.get("parsed"):
            skipped["incumbent call failed"] += 1
            continue
        key = (call["channel"], call["unit"], call["layer"], call["dose"])
        row = export["transcripts"].get(key)
        if row is None:
            skipped["no transcript row"] += 1
            continue
        payload = call.get("payload")
        origin = "stored"
        if payload is None:
            try:
                payload = rebuild_payload(judge_id, row, concept=export["concept"],
                                          baselines=export["baselines"], text_chars=text_chars)
            except KeyError as exc:
                skipped[f"cannot rebuild payload ({exc})"] += 1
                continue
            origin = "rebuilt"
        out.append(dict(
            item_id=_item_id(label, judge_id, key), source=label, arm="head_to_head",
            concept=export["concept"], subject_model=export["model"],
            config_hash=export["config_hash"], judge=judge_id, channel=call["channel"],
            layer=call["layer"], dose=call["dose"], unit=call["unit"],
            payload=apply_rubric(payload, judge_id, rubric), payload_origin=origin,
            rubric=rubric, model_text=_model_text(judge_id, row, export),
            reference=dict(by=_incumbent_name(export, call), parsed=call["parsed"],
                           rubric=("rubrica" if export["rejudged"] else "plain")),
            mech=_mech(row),
            stratum=stratum_of(judge_id, call["parsed"], row)))
    if skipped:
        print(f"    {label}: skipped {dict(skipped)}")
    return out


def _incumbent_name(export: dict, call: dict) -> str:
    """Who actually produced this verdict.

    The re-judged exports carry `judged_by` per call and keep the ORIGINAL judge in
    `config.JUDGE_MODEL`, so reading the config would attribute Sonnet's verdicts to
    gpt-4.1-mini -- silently, and in exactly the comparison this tool exists to make.
    """
    stated = call.get("judged_by")
    if stated:
        return str(stated)
    return str(export["judge_model"])


def build_null_control_items(label: str, export: dict, *, text_chars: int, n: int,
                             seed: int, rubric: str) -> list[dict]:
    """Effect items with a known answer: both responses unsteered, so influence is 0.

    Response A is the repeat-0 baseline and response B is a different unsteered repeat of the
    SAME prompt. Nothing was injected into either. The prompt frames B as "produced while a
    concept was artificially injected", so any score above 0 is the judge reading the framing,
    or reading B-differs-from-A, as concept presence. That is the exact defect RUBRICA.md
    documents, and it is measurable here with no reference judge and no labels at all.

    The concept is whatever the run was measuring, so the judge is asked about a real concept
    that is genuinely absent, rather than a nonsense one it might notice is odd.
    """
    pool = []
    for unit, alternatives in sorted(export["alt_unsteered"].items()):
        baseline = export["baselines"].get(unit)
        if baseline is None or unit not in PROMPT_TEXT:
            continue
        for index, response in enumerate(alternatives):
            pool.append((unit, index, baseline, response))
    if not pool:
        print(f"    {label}: no spare unsteered repeats, no null controls built")
        return []
    rng = random.Random(seed)
    rng.shuffle(pool)

    out = []
    for unit, index, baseline, response in pool[:max(0, n)]:
        payload = judge.render("effect", text_chars=text_chars, concept=export["concept"],
                               prompt=PROMPT_TEXT[unit], response_unsteered=baseline,
                               response_steered=response)
        out.append(dict(
            item_id=f"{label}|nullctl|effect|{unit}|r{index + 1}", source=label,
            arm="null_control", concept=export["concept"],
            subject_model=export["model"], config_hash=export["config_hash"],
            judge="effect", channel="effect", layer=None, dose=None, unit=unit,
            payload=apply_rubric(payload, "effect", rubric), payload_origin="constructed",
            rubric=rubric, model_text=[baseline, response],
            reference=dict(by="construction", rubric="n/a",
                           parsed=dict(influence=0.0, form="absent")),
            mech=dict(degenerate=False, concept_mentions=None, words=len(response.split()),
                      empty=not response.strip()),
            stratum="nullctl"))
    return sorted(out, key=lambda i: i["item_id"])


def build_label_set_items(probe_dir: Path, *, concept: str, text_chars: int,
                          judges_wanted: Sequence[str], rubric: str) -> list[dict]:
    """The 110 labelled responses in `m3/labels/`, as items.

    **These are NOT human labels, despite what the repository said until 2026-08-23.** They were
    produced by Claude Opus 5 in commit `df7b76c`, which landed the labels, `m3.calibrate` and
    `m3.scoring` together, and they were then described throughout as "one careful reader's
    judgement" and "responses a human read". The operator has confirmed doing no labelling.

    That does not make them worthless -- they are a careful second read of 110 deliberately hard,
    stratified cases, and the notes on them are specific and checkable. It makes them a MODEL
    reference, so agreement with them is agreement, exactly like agreement with gpt-4.1-mini or
    with Sonnet, and it is not accuracy. Two models sharing a blind spot agree.

    The consequence worth stating plainly: no arm of this bakeoff that uses these labels can
    settle whether a judge is right, and the earlier calibration that scored gpt-4.1-mini at
    kappa 1.000 on `identify` was model-against-model too. The only reference here that cannot
    be wrong in the same direction as a judge is `build_null_control_items`, whose answer is
    fixed by construction -- and any labels the operator writes with the `worksheet` command.
    """
    records = calibrate.load_probe(Path(probe_dir))
    by_id = {r["id"]: r for r in records}
    baselines = {r["prompt_id"]: r["response"] for r in records
                 if r["channel"] == "task" and not r["steered"]}
    out, skipped = [], Counter()
    for judge_id in judges_wanted:
        for gold in scoring.load_gold(calibrate.LABELS_DIR, judge_id):
            rec = by_id.get(gold["id"])
            if rec is None:
                skipped[f"{judge_id}: label with no response"] += 1
                continue
            response = rec["response"]
            if judge_id in ("identify", "self_report"):
                payload = judge.render(judge_id, text_chars=text_chars, concept=concept,
                                       response=response)
            elif judge_id == "coherence":
                payload = judge.render("coherence", text_chars=text_chars,
                                       prompt=scoring._prompt_text(rec["prompt_id"]),
                                       response=response)
            else:
                baseline = baselines.get(rec["prompt_id"])
                if baseline is None:
                    skipped[f"{judge_id}: no unsteered baseline"] += 1
                    continue
                payload = judge.render("effect", text_chars=text_chars, concept=concept,
                                       prompt=scoring._prompt_text(rec["prompt_id"]),
                                       response_unsteered=baseline, response_steered=response)
            reference = {k: v for k, v in gold.items()
                         if k not in ("id", "ambiguous", "note")}
            out.append(dict(
                item_id=f"opus-labels|{judge_id}|{gold['id']}", source="opus-label-set",
                arm="model_label_set",
                concept=concept, subject_model="gemma3_27b", config_hash="probe-2026-08-14",
                judge=judge_id, channel=rec["channel"], layer=rec["layer"],
                dose=rec.get("r"), unit=gold["id"],
                payload=apply_rubric(payload, judge_id, rubric), payload_origin="rebuilt",
                rubric=rubric,
                model_text=([baselines.get(rec["prompt_id"], ""), response]
                            if judge_id == "effect" else [response]),
                reference=dict(by="claude-opus-5 (2026-08-14 label set)", parsed=reference,
                               rubric="n/a",
                               ambiguous=bool(gold.get("ambiguous")),
                               note=gold.get("note") or ""),
                mech=dict(degenerate=bool(rec["degenerate"]),
                          concept_mentions=int(rec["concept_hits"]),
                          words=int(rec["words"]), empty=False),
                # Into the decision table, because agreement with a reader is the only accuracy
                # number here and it belongs where the decision is made. The table prints what
                # this set is: 110 items drawn evenly across five strata, so it is deliberately
                # hard and is NOT a population estimate. Both halves have to be said together.
                population_draw=True,
                stratum=f"labelset:{judge_id}"))
    if skipped:
        print(f"    opus-label-set: skipped {dict(skipped)}")
    return out


# =====================================================================================
# sample
# =====================================================================================

def cmd_sample(args: argparse.Namespace) -> int:
    out_dir = _guard_out(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_chars = int(args.text_chars)
    judges_wanted = tuple(args.judges)

    everything: list[dict] = []
    population: Counter = Counter()
    sources: list[dict] = []

    for spec in args.source:
        if "=" not in spec:
            raise SystemExit(f"--source must be LABEL=PATH, got {spec!r}")
        label, _, path = spec.partition("=")
        label, path = label.strip(), Path(path.strip())
        print(f"  reading {label} from {path}")
        export = load_export(path)
        sources.append(dict(label=label, path=str(path), concept=export["concept"],
                            subject_model=export["model"],
                            config_hash=export["config_hash"],
                            incumbent=export["judge_model"],
                            rejudged=export["rejudged"]))
        items = build_incumbent_items(label, export, text_chars=text_chars,
                                      judges_wanted=judges_wanted, rubric=args.rubric)
        for item in items:
            population[(label, item["judge"], item["stratum"])] += 1
        picked = _take_per_source(items, args, label)
        everything.extend(picked)
        if args.null_controls:
            everything.extend(build_null_control_items(
                label, export, text_chars=text_chars, n=int(args.null_controls),
                seed=args.seed, rubric=args.rubric))

    if args.probe:
        print(f"  reading the 2026-08-14 Opus label set against probe bundle {args.probe}")
        everything.extend(build_label_set_items(Path(args.probe), concept=args.probe_concept,
                                                text_chars=text_chars,
                                                judges_wanted=judges_wanted,
                                                rubric=args.rubric))

    if not everything:
        raise SystemExit("no items built; check --source paths and --judges")

    for item in everything:
        item.setdefault("population_draw", False)

    seen = Counter(i["item_id"] for i in everything)
    duplicates = [k for k, v in seen.items() if v > 1]
    if duplicates:
        raise SystemExit(f"item ids are not unique: {duplicates[:5]}")

    items_path = out_dir / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as fh:
        for item in everything:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    in_tokens = sum(len(i["payload"]) for i in everything) / CHARS_PER_TOKEN
    manifest = dict(
        n_items=len(everything), text_chars=text_chars, rubric=args.rubric,
        seed=args.seed, judges=list(judges_wanted), sources=sources,
        by_arm=dict(Counter(i["arm"] for i in everything)),
        by_judge=dict(Counter(i["judge"] for i in everything)),
        by_source_judge=dict(Counter(f"{i['source']}/{i['judge']}" for i in everything)),
        by_stratum=dict(Counter(f"{i['source']}/{i['stratum']}" for i in everything)),
        by_draw=dict(Counter("population" if i["population_draw"] else "stratified"
                             for i in everything)),
        rubric_provenance=(dict(RUBRIC_PROVENANCE) if args.rubric == "rubrica" else None),
        population_by_stratum={f"{a}/{b}/{c}": n for (a, b, c), n in sorted(population.items())},
        est_input_tokens=round(in_tokens),
        payload_origin=dict(Counter(i["payload_origin"] for i in everything)))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"\n  {len(everything)} items -> {items_path}")
    print(f"  by draw:  {manifest['by_draw']}")
    print(f"  by arm:   {manifest['by_arm']}")
    print(f"  by judge: {manifest['by_judge']}")
    print(f"  payload:  {manifest['payload_origin']}")
    print(f"  ~{in_tokens:,.0f} input tokens per model run. Estimated cost per model:")
    for model, (price_in, price_out) in sorted(PRICES.items()):
        est = in_tokens * price_in / 1e6 + len(everything) * 30 * price_out / 1e6
        print(f"     {model:<34} ${est:6.3f}")
    print("\n  next:  python -m tools.judge_bakeoff run --out "
          f"{args.out} --model MODEL --tag TAG")
    return 0


def _take_per_source(items: Sequence[dict], args: argparse.Namespace, label: str) -> list[dict]:
    """Cut one source down to size, per judge, in TWO draws that answer different questions.

    Per judge rather than per source because `identify` is 60% of every run's calls: sampling
    the source as a whole spends most of the budget re-checking a judge whose agreement is
    already near ceiling, and starves `effect`, which is the one that failed.

    Two draws, because "does the judgement come back the same" and "where does it break" are
    different questions and one sample cannot answer both:

      * a PLAIN RANDOM draw (`--population`), which is representative. Agreement measured on it
        estimates what a full re-judge would actually reproduce, and it is the number a decision
        to switch judges should rest on.
      * a STRATIFIED draw (`--per-judge`), which is deliberately unrepresentative. It over-
        weights the rare cases -- the 4-9 influence band, the collapses, the responses with no
        literal mention -- because that is where a judge fails, and a representative sample of a
        population that is 78% influence-zero would mostly measure how well two models agree
        that nothing happened.

    Reading the stratified number as if it were the population number understates agreement, so
    the two are drawn disjointly, flagged, and reported apart.
    """
    out: list[dict] = []
    for judge_id in sorted({i["judge"] for i in items}):
        group = [i for i in items if i["judge"] == judge_id]
        rng = random.Random(f"{args.seed}:{label}:{judge_id}")
        ordered = sorted(group, key=lambda i: i["item_id"])

        n_pop = min(int(args.population or 0), len(ordered))
        population = rng.sample(ordered, n_pop) if n_pop else []
        chosen = {i["item_id"] for i in population}

        rest = [i for i in ordered if i["item_id"] not in chosen]
        want = int(args.per_judge) if args.per_judge else len(rest)
        stratified = stratified_take(rest, want, seed=args.seed)

        for item in population:
            item["population_draw"] = True
        for item in stratified:
            item["population_draw"] = False
        out.extend(population + stratified)
        print(f"    {label}/{judge_id}: {len(population)} random + {len(stratified)} stratified "
              f"of {len(group)}, across {len({i['stratum'] for i in stratified})} strata")
    return out


# =====================================================================================
# worksheet - the only way this repository gets a reference that is not a model
# =====================================================================================
# Every reference the bakeoff has otherwise -- gpt-4.1-mini, Sonnet, the 2026-08-14 label set --
# was produced by a language model, so every number it reports is agreement. Agreement cannot
# distinguish "both right" from "both wrong in the same direction", which is precisely the
# failure that started this: gpt-4.1-mini scored kappa 1.000 against the label set on `identify`
# and was still reading deviation-from-baseline as concept presence.
#
# Labelling 15 items per judge takes an evening and turns four agreement numbers into four
# accuracy numbers -- for the candidate AND for both incumbents, on the same items, which is the
# only way to find out whether Sonnet was right either.

_LABEL_FIELDS = {
    "identify":    ("named", "matches"),
    "self_report": ("claims", "named", "matches"),
    "coherence":   ("coherence", "on_task"),
    "effect":      ("influence", "form"),
}

_LABEL_HELP = {
    "named": '"garlic", or "NONE" if no concept was named',
    "matches": "true or false",
    "claims": '"YES", "NO" or "UNCLEAR"',
    "coherence": "0-10",
    "on_task": "true or false",
    "influence": "0-10",
    "form": '"absent" 0 | "trace" 1-3 | "thematic" 4-6 | "dominant" 7-9 | "fixated" 10',
}


def cmd_worksheet(args: argparse.Namespace) -> int:
    """Write blind worksheets and a label template for the operator to fill in.

    Blind is the whole point, and it is enforced rather than intended: the worksheet carries the
    payload a judge saw and nothing else. No stored verdict, no candidate verdict, and not the
    stratum either -- `eff:4-6:unnamed` names the incumbent's band, so printing it would anchor
    the label to the judgement it exists to check.

    Items are drawn from the RANDOM half of the sample, so anything labelled here is also in the
    decision table, and the same items carry a candidate verdict and an incumbent verdict.
    """
    out_dir = _guard_out(args.out)
    items_path = out_dir / "items.jsonl"
    if not items_path.exists():
        raise SystemExit(f"{items_path} is missing; run `sample` first")
    items = [json.loads(line) for line in items_path.open(encoding="utf-8") if line.strip()]

    pool = [i for i in items if i.get("population_draw") and i["arm"] == "head_to_head"]
    if not pool:
        raise SystemExit("no head-to-head items in the random draw; re-run `sample` with "
                         "--population N")

    rng = random.Random(args.seed)
    template_path = out_dir / "labels.template.jsonl"
    written: list[dict] = []
    with template_path.open("w", encoding="utf-8") as template:
        for judge_id in sorted({i["judge"] for i in pool}):
            group = sorted([i for i in pool if i["judge"] == judge_id],
                           key=lambda i: i["item_id"])
            picked = sorted(rng.sample(group, min(int(args.per_judge), len(group))),
                            key=lambda i: i["item_id"])
            path = out_dir / f"worksheet_{judge_id}.txt"
            with path.open("w", encoding="utf-8") as fh:
                fh.write(f"# worksheet: {judge_id}   ({len(picked)} items)\n#\n")
                fh.write("# Each block below is EXACTLY what the judge was shown. Read it and\n")
                fh.write(f"# answer it yourself. Write your answers into {out_dir / 'labels.jsonl'},\n")
                fh.write(f"# one JSON object per line, copying `item_id` verbatim:\n#\n")
                fields = _LABEL_FIELDS[judge_id]
                fh.write("#   " + json.dumps({"item_id": "...",
                                              **{f: None for f in fields}}) + "\n#\n")
                for field in fields:
                    fh.write(f"#   {field:<12} {_LABEL_HELP[field]}\n")
                fh.write("#\n# No judge's answer appears in this file, deliberately. A label\n")
                fh.write("# written next to the verdict it is meant to check is not a check.\n")
                fh.write("# Skip anything you are unsure about rather than guessing: a thin\n")
                fh.write("# set of labels you trust is worth more than a full one you do not.\n\n")
                for item in picked:
                    fh.write("=" * 92 + "\n")
                    fh.write(f"item_id: {item['item_id']}\n")
                    fh.write("=" * 92 + "\n")
                    fh.write(item["payload"].rstrip() + "\n\n\n")
                    template.write(json.dumps({"item_id": item["item_id"],
                                               **{f: None for f in fields}}) + "\n")
            written.append(dict(judge=judge_id, n=len(picked), path=path))
            print(f"  {judge_id:<12} {len(picked):>3} items -> {path}")

    print(f"\n  template: {template_path}")
    print(f"  fill in {out_dir / 'labels.jsonl'} (copy the template, replace the nulls),")
    print("  then re-run `report`. Partial files are fine -- unlabelled items are skipped, and")
    print("  a field left null is skipped for that item alone.")
    print(f"\n  {sum(w['n'] for w in written)} items total. Every candidate and every incumbent")
    print("  already has a verdict on all of them, so labelling scores all of them at once.")
    return 0


def load_operator_labels(out_dir: Path) -> dict[str, dict]:
    """`{item_id: {field: value}}` from `labels.jsonl`, or `{}` if there is none.

    Nulls are dropped rather than compared. A template line left untouched must not read as a
    judgement of NONE, which is a real label with a real meaning.
    """
    path = Path(out_dir) / "labels.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for number, line in enumerate(path.open(encoding="utf-8"), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number} is not valid JSON: {exc}")
        if "item_id" not in row:
            raise SystemExit(f"{path}:{number} has no item_id")
        values = {k: v for k, v in row.items() if k != "item_id" and v is not None}
        if values:
            out[str(row["item_id"])] = values
    return out


# =====================================================================================
# run
# =====================================================================================

def cmd_run(args: argparse.Namespace) -> int:
    from m2 import config as m2config, judges as transport

    out_dir = _guard_out(args.out)
    items_path = out_dir / "items.jsonl"
    if not items_path.exists():
        raise SystemExit(f"{items_path} is missing; run `sample` first")
    items = [json.loads(line) for line in items_path.open(encoding="utf-8") if line.strip()]

    verdict_path = out_dir / f"verdicts_{args.tag}.jsonl"
    done: set[str] = set()
    if verdict_path.exists():
        for line in verdict_path.open(encoding="utf-8"):
            row = json.loads(line)
            # Only completed calls are resumed past. A recorded failure is re-attempted, because
            # the usual reason for one is a rate limit or a dropped connection, and refusing to
            # retry it would bake a transport hiccup into the comparison as a judge weakness.
            if row.get("ok"):
                done.add(row["item_id"])
        print(f"  resuming: {len(done)} verdicts already on disk in {verdict_path}")

    todo = [i for i in items if i["item_id"] not in done]
    if not todo:
        print("  nothing to do")
        return 0

    cfg = dict(config.CONFIG)
    cfg["JUDGE_MODEL"] = args.model
    cfg["JUDGE_CONCURRENT"] = int(args.concurrency)
    cfg["JUDGE_MAX_TOKENS"] = int(args.max_tokens)
    judge.configure_transport(cfg)
    m2config.CONFIG["judge_model"] = args.model
    transport.JUDGE_EXTRA_BODY = ({"reasoning": {"enabled": False}}
                                  if args.no_reasoning else {})

    in_tokens = sum(len(i["payload"]) for i in todo) / CHARS_PER_TOKEN
    price_in, price_out = PRICES.get(args.model, (None, None))
    est = ("unknown" if price_in is None else
           f"${in_tokens * price_in / 1e6 + len(todo) * 30 * price_out / 1e6:.3f}")
    print(f"  model      {args.model}")
    print(f"  transport  {json.dumps({k: v for k, v in transport.judge_transport().items() if k != 'key_present'})}")
    print(f"  items      {len(todo)} | ~{in_tokens:,.0f} input tokens | est {est}")
    if args.dry_run:
        print("  --dry-run: nothing issued")
        return 0

    # The cache is namespaced by CONCEPT inside the transport, not by model, and this is the one
    # tool that puts two models behind it. Two `run` calls in one process would otherwise serve
    # the first model's verdict to the second under the same key -- a wrong number rather than
    # an error, and the exact shape of CONTRACT defence 1, which put `judge_id` in the key so
    # that S1 could not read E5's cached row. The model goes in the phase field, so the key
    # keeps its contracted six-field shape and every candidate gets its own namespace.
    #
    # `use_cache=False` on top of that, because a bakeoff has no repeated payloads to save and
    # nothing here should be able to answer from a previous candidate's verdicts.
    phase = f"BAKEOFF:{args.model}"
    work = [dict(judge.build_item(i["judge"], payload=i["payload"],
                                  cache_key=judge.cache_key(phase, i["judge"], unit=i["item_id"]),
                                  concept=i["concept"],
                                  model_text=tuple(i.get("model_text") or ()),
                                  text_chars=int(i.get("text_chars")
                                                 or config.CONFIG["JUDGE_TEXT_CHARS"])),
                 use_cache=False)
            for i in todo]

    started = __import__("time").time()
    results = judge.run_judges(work, concurrency=int(args.concurrency))
    elapsed = __import__("time").time() - started

    ok = errors = 0
    with verdict_path.open("a", encoding="utf-8") as fh:
        for item, result in zip(todo, results):
            parsed, error = judge.verdict(result)
            row = dict(item_id=item["item_id"], judge=item["judge"], model=args.model,
                       tag=args.tag, ok=parsed is not None, parsed=parsed, error=error,
                       raw=result.get("raw"), attempts=result.get("attempts"),
                       latency_s=result.get("latency_s"),
                       error_detail=result.get("error_detail"),
                       est_in_tokens=round(len(item["payload"]) / CHARS_PER_TOKEN),
                       est_out_tokens=round(len(str(result.get("raw") or "")) / CHARS_PER_TOKEN),
                       max_tokens=int(args.max_tokens), rubric=item.get("rubric", "plain"))
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()          # a kill mid-loop keeps every call already paid for
            ok += bool(parsed)
            errors += parsed is None
    print(f"\n  {ok} parsed, {errors} failed, {elapsed:,.0f}s -> {verdict_path}")
    if errors:
        print("  failures are NOT scored against the judge's accuracy; they are reported "
              "separately by `report`, because a formatting failure and a wrong judgement are "
              "different problems with different fixes.")
    return 0


# =====================================================================================
# report
# =====================================================================================

def _normalise_named(value: Any) -> Any:
    return str(value).strip().lower().strip('."*\'`') if value is not None else None


def _normalise(parsed: dict | None) -> dict | None:
    if parsed is None:
        return None
    out = dict(parsed)
    if "named" in out:
        out["named"] = _normalise_named(out["named"])
    return out


def _agreement(reference: Sequence[dict], candidate: Sequence[dict], field: str,
               kind: str) -> dict:
    """Reused from `m3.calibrate` so a bakeoff number and a calibration number mean the same
    thing. Ordinal fields get a signed bias on top, because the documented failure is
    directional -- the old judge scored high, not merely differently."""
    got = calibrate.score_agreement(reference, candidate, field, kind)
    if kind == "ordinal" and got.get("n"):
        pairs = [(float(r[field]), float(c[field])) for r, c in zip(reference, candidate)
                 if r.get(field) is not None and c.get(field) is not None]
        got["bias"] = statistics.fmean(c - r for r, c in pairs)
    return got


def _fmt(field: str, got: dict) -> str:
    if not got.get("n"):
        return f"   {field:<26} no comparable pairs"
    if got.get("kind") == "categorical":
        kappa = got.get("kappa")
        ktxt = "  n/a" if kappa is None else f"{kappa:5.3f}"
        return (f"   {field:<26} n={got['n']:<5} agree={got['agreement']:.3f}  kappa={ktxt}")
    return (f"   {field:<26} n={got['n']:<5} MAE={got['mean_abs_error']:.2f}  "
            f"within2={got['within_2']:.2f}  bias={got.get('bias', 0):+.2f}  "
            f"max={got['max_abs_error']:.0f}")


def _load_verdicts(out_dir: Path) -> dict[str, dict[str, dict]]:
    """`{tag: {item_id: row}}` for every `verdicts_*.jsonl` in the output directory."""
    found: dict[str, dict[str, dict]] = {}
    for path in sorted(out_dir.glob("verdicts_*.jsonl")):
        tag = path.stem[len("verdicts_"):]
        rows: dict[str, dict] = {}
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            # Later lines win: a resumed run appends a retry after a failure.
            if row["item_id"] not in rows or row.get("ok"):
                rows[row["item_id"]] = row
        found[tag] = rows
    return found


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = _guard_out(args.out)
    items = {}
    for line in (out_dir / "items.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        items[row["item_id"]] = row
    verdicts = _load_verdicts(out_dir)
    if not verdicts:
        raise SystemExit(f"no verdicts_*.jsonl in {out_dir}; run `run` first")

    print("=" * 92)
    print("WHAT WAS COMPARED")
    print("=" * 92)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"  {manifest['n_items']} items, rubric={manifest['rubric']}, "
          f"text_chars={manifest['text_chars']}")
    for source in manifest["sources"]:
        note = " (re-judged offline)" if source.get("rejudged") else ""
        print(f"    {source['label']:<28} {source['concept']:<8} {source['subject_model']:<12} "
              f"incumbent={source['incumbent']}{note}")
    for tag, rows in sorted(verdicts.items()):
        models = {r["model"] for r in rows.values()}
        failed = sum(1 for r in rows.values() if not r.get("ok"))
        print(f"    candidate {tag:<18} {'/'.join(sorted(models)):<34} "
              f"{len(rows)} calls, {failed} unusable")

    _report_call_health(items, verdicts)
    _report_operator_labels(items, verdicts, load_operator_labels(out_dir))
    _report_decision(items, verdicts, manifest)
    _report_vs_reference(items, verdicts)
    _report_null_controls(items, verdicts)
    _report_invented_influence(items, verdicts)
    _report_influence_bands(items, verdicts)
    _report_pairwise(items, verdicts)
    _report_cost(items, verdicts, manifest)
    path = _write_disagreements(out_dir, items, verdicts)
    print(f"\n  every disagreement, with the payloads: {path}")
    print("  that file carries raw generations. It stays where it is.")
    return 0


def _report_call_health(items: dict, verdicts: dict) -> None:
    print("\n" + "=" * 92)
    print("CALL HEALTH  --  did the model answer in the required format at all")
    print("=" * 92)
    print("  A judge that cannot emit `Influence: 7` is not a judge that scored 7 wrongly. These")
    print("  are excluded from every table below and counted here instead.\n")
    for tag, rows in sorted(verdicts.items()):
        by_judge: dict[str, Counter] = defaultdict(Counter)
        reasons: Counter = Counter()
        for item_id, row in rows.items():
            judge_id = items[item_id]["judge"] if item_id in items else row["judge"]
            by_judge[judge_id]["n"] += 1
            by_judge[judge_id]["ok" if row.get("ok") else "bad"] += 1
            if not row.get("ok"):
                reasons[str(row.get("error"))[:60]] += 1
        line = "  ".join(f"{j}={c['ok']}/{c['n']}" for j, c in sorted(by_judge.items()))
        print(f"  {tag:<20} {line}")
        for reason, count in reasons.most_common(4):
            print(f"      {count:>5}x  {reason}")


def _pairs(items: dict, rows: dict, judge_id: str, arm: str | None = None,
           source: str | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """`(reference, candidate, item)` triples for one judge, aligned and usable."""
    ref, cand, meta = [], [], []
    for item_id, item in items.items():
        if item["judge"] != judge_id:
            continue
        if arm is not None and item["arm"] != arm:
            continue
        if source is not None and item["source"] != source:
            continue
        row = rows.get(item_id)
        if row is None or not row.get("ok"):
            continue
        ref.append(_normalise(item["reference"]["parsed"]))
        cand.append(_normalise(row["parsed"]))
        meta.append(item)
    return ref, cand, meta


def _report_decision(items: dict, verdicts: dict, manifest: dict) -> None:
    """The number a decision to switch judges should rest on.

    Measured on the RANDOM draw only, so it estimates what a full re-judge would reproduce
    rather than how the candidate does on the hard cases. Scored against
    `m3.calibrate.THRESHOLDS` -- the same bar the pipeline's own judges had to clear before
    they were trusted on a pod -- so "close enough" is a claim with a number behind it.

    Pooled by WHO produced the reference, not by run: the question is whether the candidate can
    stand in for that judge, and Garlic and Silk were judged by the same one under the same
    rubric.
    """
    print("\n" + "=" * 92)
    print("DOES THE JUDGEMENT COME BACK THE SAME?  --  representative draw, decision table")
    print("=" * 92)
    population = [i for i in items.values() if i.get("population_draw")]
    if not population:
        print("  no random draw in this sample. Re-run `sample` with --population N.")
        return
    print("  Random items, not the stratified ones, so this is what a full re-judge would")
    print("  reproduce. The bar is m3.calibrate.THRESHOLDS, the same one the pipeline's own")
    print("  judges had to clear. A FAIL here is a reason not to switch; a PASS is a reason to.")
    if manifest.get("rubric_provenance"):
        print("\n  Rubric in the payloads:")
        for judge_id, note in sorted(manifest["rubric_provenance"].items()):
            print(f"    {judge_id:<14} {note}")

    by_reference = defaultdict(list)
    for item in population:
        by_reference[item["reference"]["by"]].append(item["item_id"])

    item_rubric = manifest.get("rubric", "plain")
    for reference, ids in sorted(by_reference.items()):
        wanted = set(ids)
        ref_rubrics = sorted({items[i]["reference"].get("rubric", "plain") for i in ids})
        print(f"\n  --- against {reference}   ({len(ids)} items)")
        # A candidate given the corrected instructions, compared against a verdict made WITHOUT
        # them, is being scored on two changes at once. That is worth measuring -- it is the
        # before-and-after of switching judge and prompt together -- but it is not "does the
        # judgement come back the same", and the two must not be read as if they were.
        arms = {items[i]["arm"] for i in ids}
        if arms == {"model_label_set"}:
            print("      ANOTHER MODEL's labels, not a person's: written by Claude Opus 5 in")
            print("      commit df7b76c and mis-described as a reader's until 2026-08-23. So")
            print("      this is agreement, like every other block here, and two models can")
            print("      share a blind spot. The 110 items are also deliberately hard -- drawn")
            print("      evenly across five strata -- so a miss is a place to look, not a rate.")
        if arms == {"operator_labels"}:
            print("      THE OPERATOR's own labels. This is the only block in the report that")
            print("      measures whether a judge is RIGHT rather than whether it agrees.")
        mismatched = [r for r in ref_rubrics if r not in ("n/a", item_rubric)]
        if mismatched:
            print(f"      NOT PROMPT-MATCHED: the candidate saw the {item_rubric!r} rubric; "
                  f"this reference was produced under {mismatched}.")
            print("      Disagreement here is the model AND the prompt together, not the model.")
        elif "n/a" not in ref_rubrics:
            print(f"      prompt-matched: candidate and reference both saw {item_rubric!r}.")
        for tag, rows in sorted(verdicts.items()):
            scored: dict[str, dict] = {}
            for judge_id, fields in JUDGE_FIELDS.items():
                sub = {k: v for k, v in items.items()
                       if k in wanted and v["judge"] == judge_id}
                ref, cand, _ = _pairs(sub, rows, judge_id)
                if not ref:
                    continue
                scored[judge_id] = {f: _agreement(ref, cand, f, kind) for f, kind in fields}
            if not scored:
                continue
            print(f"      [{tag}]")
            for row in calibrate.verdicts(scored):
                if row["verdict"] == "NOT MEASURED":
                    continue
                print(f"        {row['verdict']:<6} {row['criterion']:<24} "
                      f"n={row.get('n', 0):<5} {row.get('detail', '')}")
            # Fields with no stated bar still carry information; print them unjudged rather
            # than letting a silent omission read as "nothing to see".
            for judge_id, fields in scored.items():
                for field, got in fields.items():
                    if f"{judge_id}.{field}" in calibrate.THRESHOLDS or not got.get("n"):
                        continue
                    print(f"        {'--':<6} {judge_id + '.' + field:<24} "
                          f"{_fmt('', got).strip()}   (no stated bar)")


def _report_operator_labels(items: dict, verdicts: dict, labels: dict) -> None:
    """The only block in this report that measures whether a judge is RIGHT.

    Scores every candidate AND every incumbent against the same operator labels, on the same
    items, because "is DeepSeek good enough" and "was Sonnet right" are the same question asked
    of two models and there is no reason to answer only one of them.
    """
    print("\n" + "=" * 92)
    print("AGAINST YOUR OWN LABELS  --  the only accuracy in this report")
    print("=" * 92)
    if not labels:
        print("  none yet. Every other reference here was written by a language model, so every")
        print("  other number is agreement: it cannot tell 'both right' from 'both wrong the")
        print("  same way'. `worksheet` writes blind worksheets; 15 items per judge scores every")
        print("  candidate and both incumbents at once.")
        return

    covered = {k: v for k, v in labels.items() if k in items}
    if len(covered) < len(labels):
        print(f"  NOTE: {len(labels) - len(covered)} labelled item_ids are not in items.jsonl "
              "and are ignored.")
    print(f"  {len(covered)} labelled items.\n")

    for judge_id, fields in JUDGE_FIELDS.items():
        ids = [k for k in covered if items[k]["judge"] == judge_id]
        if not ids:
            continue
        print(f"  --- {judge_id}   ({len(ids)} labelled)")

        # Every scorer that has an opinion on these items: the candidates, and whichever judge
        # produced each item's stored verdict.
        scorers: dict[str, dict[str, dict]] = {}
        for tag, rows in sorted(verdicts.items()):
            got = {k: _normalise(rows[k]["parsed"]) for k in ids
                   if rows.get(k) and rows[k].get("ok")}
            if got:
                scorers[f"candidate {tag}"] = got
        by_incumbent: dict[str, dict] = defaultdict(dict)
        for k in ids:
            by_incumbent[items[k]["reference"]["by"]][k] = _normalise(
                items[k]["reference"]["parsed"])
        for name, got in by_incumbent.items():
            scorers[f"incumbent {name}"] = got

        for name, got in sorted(scorers.items()):
            shared = [k for k in ids if k in got and any(f in covered[k] for f, _ in fields)]
            if not shared:
                continue
            truth = [covered[k] for k in shared]
            said = [got[k] for k in shared]
            print(f"    [{name}]")
            for field, kind in fields:
                scored = _agreement(truth, said, field, kind)
                if scored.get("n"):
                    print("   " + _fmt(field, scored))
        print()


def _report_vs_reference(items: dict, verdicts: dict) -> None:
    print("\n" + "=" * 92)
    print("AGAINST EACH ITEM'S REFERENCE  --  per source, per judge")
    print("=" * 92)
    print("  BOTH draws pooled, so these read LOWER than the decision table above: the")
    print("  stratified draw is deliberately weighted toward the cases where judges fail.")
    print("  For head-to-head sources the reference is another judge, so this is agreement,")
    print("  not accuracy. NO reference in this section is a human one: see")
    print("  `build_label_set_items` for why the 2026-08-14 label set is not either.\n")
    # Grouped by (source, ARM). The null-control items carry their own source label -- they
    # are built from that run's own unsteered arm -- so grouping on source alone folded them
    # into the head-to-head agreement for the same run and quietly changed every effect number
    # in this section. They have their own block below, against their own known answer.
    groups = sorted({(i["source"], i["arm"]) for i in items.values()
                     if i["arm"] != "null_control"})
    for source, arm in groups:
        reference_by = sorted({i["reference"]["by"] for i in items.values()
                               if i["source"] == source and i["arm"] == arm})
        print(f"\n  --- {source}   [{arm}]   reference: {', '.join(reference_by)}")
        for judge_id, fields in JUDGE_FIELDS.items():
            if not any(i["judge"] == judge_id and i["source"] == source and i["arm"] == arm
                       for i in items.values()):
                continue
            print(f"\n    {judge_id}")
            for tag, rows in sorted(verdicts.items()):
                ref, cand, _ = _pairs(items, rows, judge_id, arm=arm, source=source)
                if not ref:
                    continue
                print(f"      [{tag}]")
                for field, kind in fields:
                    print("   " + _fmt(field, _agreement(ref, cand, field, kind)))


def _report_null_controls(items: dict, verdicts: dict) -> None:
    have = [i for i in items.values() if i["arm"] == "null_control"]
    print("\n" + "=" * 92)
    print("NULL CONTROLS  --  the arm with a known right answer")
    print("=" * 92)
    if not have:
        print("  none built. Re-run `sample` with --null-controls N.")
        return
    print("  Both responses are unsteered samples of the same prompt. Nothing was injected, so")
    print("  the correct influence is 0. Anything above 0 is influence the judge invented from")
    print("  the framing or from B simply differing from A -- the defect that started this.\n")
    print(f"  {'candidate':<20} {'n':>5} {'mean':>7} {'=0':>7} {'>=1':>7} {'>=4':>7} {'>=7':>7}")
    print("  " + "-" * 66)
    for tag, rows in sorted(verdicts.items()):
        _, cand, _ = _pairs(items, rows, "effect", arm="null_control")
        if not cand:
            continue
        scores = [float(c["influence"]) for c in cand]
        n = len(scores)
        print(f"  {tag:<20} {n:>5} {statistics.fmean(scores):>7.2f} "
              f"{sum(s == 0 for s in scores) / n:>7.1%} {sum(s >= 1 for s in scores) / n:>7.1%} "
              f"{sum(s >= 4 for s in scores) / n:>7.1%} {sum(s >= 7 for s in scores) / n:>7.1%}")
    print("\n  Read the `>=4` column first: 4 is the band at which a cell starts counting as")
    print("  influential, so a judge scoring >=4 on an unsteered response is manufacturing the")
    print("  study's primary finding out of nothing.")


def _report_invented_influence(items: dict, verdicts: dict) -> None:
    print("\n" + "=" * 92)
    print("INFLUENCE WITHOUT A SINGLE MENTION  --  on the real steered items")
    print("=" * 92)
    print("  The prompt deliberately asks for unnamed influence, so a high score with zero")
    print("  literal mentions is not wrong by itself. It is where BOTH the real signal and the")
    print("  invented one live, so the number to compare is how far the candidates differ from")
    print("  each other and from the incumbent, not the number on its own.\n")
    print(f"  {'candidate':<34} {'n(unnamed)':>11} {'mean':>7} {'>=4':>8} {'>=7':>8}")
    print("  " + "-" * 72)
    rows_out = []
    for tag, rows in sorted(verdicts.items()):
        _, cand, meta = _pairs(items, rows, "effect", arm="head_to_head")
        pool = [(c, m) for c, m in zip(cand, meta) if m["mech"]["concept_mentions"] == 0]
        if not pool:
            continue
        scores = [float(c["influence"]) for c, _ in pool]
        n = len(scores)
        rows_out.append((tag, n, statistics.fmean(scores),
                         sum(s >= 4 for s in scores) / n, sum(s >= 7 for s in scores) / n))
    # The incumbent, computed the same way from the stored references, so the table has the
    # thing being replaced in it rather than only its challengers.
    ref_pool = [i for i in items.values()
                if i["judge"] == "effect" and i["arm"] == "head_to_head"
                and i["mech"]["concept_mentions"] == 0]
    by_incumbent: dict[str, list[float]] = defaultdict(list)
    for item in ref_pool:
        by_incumbent[item["reference"]["by"]].append(float(item["reference"]["parsed"]["influence"]))
    for name, scores in sorted(by_incumbent.items()):
        n = len(scores)
        rows_out.append((f"(incumbent) {name}"[:34], n, statistics.fmean(scores),
                         sum(s >= 4 for s in scores) / n, sum(s >= 7 for s in scores) / n))
    for tag, n, mean, hi4, hi7 in rows_out:
        print(f"  {tag:<34} {n:>11} {mean:>7.2f} {hi4:>8.1%} {hi7:>8.1%}")
    print("\n  The incumbent rows split by WHICH incumbent, so together they cover the same")
    print("  items the candidates' single row does.")


def _report_influence_bands(items: dict, verdicts: dict) -> None:
    print("\n" + "=" * 92)
    print("INFLUENCE BAND CONFUSION  --  reference (rows) against candidate (columns)")
    print("=" * 92)
    print("  0 / 1-3 / 4-6 / 7-9 / 10. The 4 boundary is the one that decides whether a cell is")
    print("  called influential, so mass below the diagonal in those rows is the correction and")
    print("  mass above it is the candidate inheriting the same inflation.\n")
    bands = ("0", "1-3", "4-6", "7-9", "10")

    def band(score: float) -> str:
        return ("0" if score == 0 else "1-3" if score <= 3 else "4-6" if score <= 6
                else "7-9" if score <= 9 else "10")

    for tag, rows in sorted(verdicts.items()):
        ref, cand, _ = _pairs(items, rows, "effect", arm="head_to_head")
        if not ref:
            continue
        table: Counter = Counter()
        for r, c in zip(ref, cand):
            table[(band(float(r["influence"])), band(float(c["influence"])))] += 1
        print(f"  [{tag}]")
        print("      ref\\cand " + "".join(f"{b:>7}" for b in bands) + f"{'total':>9}")
        for row_band in bands:
            counts = [table[(row_band, col)] for col in bands]
            total = sum(counts)
            if not total:
                continue
            print(f"      {row_band:<9}" + "".join(f"{c:>7}" for c in counts) + f"{total:>9}")
        moved_down = sum(v for (r, c), v in table.items() if bands.index(c) < bands.index(r))
        moved_up = sum(v for (r, c), v in table.items() if bands.index(c) > bands.index(r))
        same = sum(v for (r, c), v in table.items() if r == c)
        print(f"      same band {same}, candidate lower {moved_down}, candidate higher "
              f"{moved_up}\n")


def _report_pairwise(items: dict, verdicts: dict) -> None:
    if len(verdicts) < 2:
        return
    print("\n" + "=" * 92)
    print("CANDIDATE AGAINST CANDIDATE  --  same payload, same items")
    print("=" * 92)
    print("  Two models judged the same strings, on the head-to-head items. Where they agree")
    print("  with each other but not with the reference, the reference is the outlier.\n")
    tags = sorted(verdicts)
    for i, left in enumerate(tags):
        for right in tags[i + 1:]:
            print(f"  --- {left}  vs  {right}")
            for judge_id, fields in JUDGE_FIELDS.items():
                shared = [item_id for item_id, item in items.items()
                          if item["judge"] == judge_id and item["arm"] == "head_to_head"
                          and verdicts[left].get(item_id, {}).get("ok")
                          and verdicts[right].get(item_id, {}).get("ok")]
                if not shared:
                    continue
                a = [_normalise(verdicts[left][k]["parsed"]) for k in shared]
                b = [_normalise(verdicts[right][k]["parsed"]) for k in shared]
                print(f"    {judge_id}")
                for field, kind in fields:
                    print("   " + _fmt(field, _agreement(a, b, field, kind)))
            print()


def _report_cost(items: dict, verdicts: dict, manifest: dict) -> None:
    print("\n" + "=" * 92)
    print("COST  --  estimated, from characters, at OpenRouter list prices")
    print("=" * 92)
    print("  The transport discards the usage field, so tokens are estimated at 4 characters")
    print("  each. Good to roughly 15%, which is enough to separate $0.06/M from $0.40/M and")
    print("  not enough to quote.\n")
    full_run = 45_570      # every gpt-4.1-mini judge call across the six full exports
    rejudged = 6_534       # every Sonnet verdict on the two 2026-08-21 Gemma runs
    all_calls = f"$ all {full_run + rejudged:,}"
    print(f"  {'candidate':<20} {'calls':>7} {'in tok':>10} {'out tok':>9} {'$ here':>9} "
          f"{'$/1k calls':>11} {all_calls:>15}")
    print("  " + "-" * 84)
    for tag, rows in sorted(verdicts.items()):
        usable = [r for r in rows.values() if r.get("ok")]
        if not usable:
            continue
        model = sorted({r["model"] for r in usable})[0]
        price_in, price_out = PRICES.get(model, (None, None))
        tin = sum(r["est_in_tokens"] for r in usable)
        tout = sum(r["est_out_tokens"] for r in usable)
        if price_in is None:
            print(f"  {tag:<20} {len(usable):>7} {tin:>10,} {tout:>9,}   (no price for {model})")
            continue
        cost = tin * price_in / 1e6 + tout * price_out / 1e6
        per_call = cost / len(usable)
        print(f"  {tag:<20} {len(usable):>7} {tin:>10,} {tout:>9,} ${cost:>8.3f} "
              f"${per_call * 1000:>10.2f} ${per_call * (full_run + rejudged):>12.2f}")
    print(f"\n  The last column prices re-judging everything on disk: {full_run:,} pipeline")
    print(f"  calls plus {rejudged:,} re-judged ones. It assumes this sample's payloads are")
    print("  representative in length, which for `identify` and `self_report` they are not --")
    print("  those are shorter than `effect`, whose payload carries two responses.")


def _write_disagreements(out_dir: Path, items: dict, verdicts: dict) -> Path:
    """Every disagreement with its full payload, for reading.

    The headline table says whether to switch judges. This file says why, and it is where the
    information is: a systematic disagreement is worth more than the aggregate, and invisible
    in it.
    """
    path = out_dir / "disagreements.txt"
    with path.open("w", encoding="utf-8") as fh:
        for tag, rows in sorted(verdicts.items()):
            for item_id, item in sorted(items.items()):
                row = rows.get(item_id)
                if row is None or not row.get("ok"):
                    continue
                ref = _normalise(item["reference"]["parsed"]) or {}
                cand = _normalise(row["parsed"]) or {}
                differing = [f for f, _kind in JUDGE_FIELDS[item["judge"]]
                             if f in ref and ref[f] != cand.get(f)]
                if not differing:
                    continue
                fh.write("=" * 92 + "\n")
                fh.write(f"{tag}  {item_id}\n")
                fh.write(f"arm={item['arm']}  stratum={item['stratum']}  "
                         f"mech={item['mech']}\n")
                fh.write(f"reference ({item['reference']['by']}): {ref}\n")
                fh.write(f"candidate ({row['model']}):           {cand}\n")
                fh.write(f"differs on: {differing}\n")
                if item["reference"].get("note"):
                    fh.write(f"reader's note: {item['reference']['note']}\n")
                fh.write("-" * 92 + "\nPAYLOAD\n" + item["payload"] + "\n")
                fh.write("-" * 92 + "\nCANDIDATE REPLY\n" + str(row.get("raw")) + "\n\n")
    return path


# =====================================================================================
# CLI
# =====================================================================================

def _refuse_harmful(concepts: Iterable[str]) -> None:
    """The same deny-list every other M3 entry point carries.

    Nothing in this repo runs the harmful arm, and this tool would transmit its generations to
    a third-party API if pointed at one. The refusal is here rather than assumed because a
    bakeoff reads whatever export directory it is given.
    """
    bad = [c for c in concepts if not config.concept_allowed(c)]
    if bad:
        raise SystemExit(
            f"{bad} is on HARMFUL_CONCEPTS -- the arm this study has deliberately not run. "
            "Judging it would send those generations to a third-party API. Read the parent "
            "repo's ethics register first; this is a decision about the study, not a flag.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.judge_bakeoff",
        description="Score a candidate judge model against the incumbents, on real items.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, default=REPO / "private" / "judge-bakeoff",
                        help="working directory. Must be outside the repo or under private/: "
                             "it holds raw generations.")

    p_sample = sub.add_parser("sample", parents=[common],
                              help="build the item set. No API calls, no cost.")
    p_sample.add_argument("--source", action="append", default=[], metavar="LABEL=PATH",
                          help="an unzipped run export. Repeatable.")
    p_sample.add_argument("--probe", type=Path, default=None,
                          help="an unzipped 2026-08-14 probe bundle. Adds the Opus label set "
                               "as a reference -- a second model's read, NOT a human one.")
    p_sample.add_argument("--probe-concept", default="Garlic",
                          help="the concept the probe bundle was run on")
    p_sample.add_argument("--judges", nargs="+", default=list(judge.JUDGE_IDS),
                          choices=list(judge.JUDGE_IDS))
    p_sample.add_argument("--per-judge", type=int, default=120,
                          help="STRATIFIED items per (source, judge), weighted toward the "
                               "cases where judges fail. 0 takes everything left after the "
                               "random draw.")
    p_sample.add_argument("--population", type=int, default=100,
                          help="RANDOM items per (source, judge), drawn disjointly from the "
                               "stratified ones. This is the representative sample, and the "
                               "decision table is computed on it alone.")
    p_sample.add_argument("--null-controls", type=int, default=40,
                          help="unsteered-vs-unsteered effect items per source, whose correct "
                               "influence is 0")
    p_sample.add_argument("--rubric", choices=("plain", "rubrica"), default="rubrica",
                          help="rubrica (default) adds what the Sonnet agents were told, and "
                               "is the setting under which matching a SONNET verdict is "
                               "apples-to-apples. plain is the pipeline prompt byte for byte, "
                               "which is the setting for matching a gpt-4.1-mini verdict.")
    p_sample.add_argument("--text-chars", type=int, default=config.CONFIG["JUDGE_TEXT_CHARS"])
    p_sample.add_argument("--seed", type=int, default=20260822)
    p_sample.set_defaults(func=cmd_sample)

    p_run = sub.add_parser("run", parents=[common], help="issue the calls. This is the cost.")
    p_run.add_argument("--model", required=True,
                       help="an OpenRouter model id, e.g. deepseek/deepseek-v4-flash")
    p_run.add_argument("--tag", required=True, help="short name for this candidate's file")
    p_run.add_argument("--concurrency", type=int, default=16)
    p_run.add_argument("--max-tokens", type=int, default=400,
                       help="400, not the pipeline's 120. max_tokens only truncates, so a "
                            "compliant model is unaffected and a verbose one is not scored "
                            "down for verbosity.")
    p_run.add_argument("--no-reasoning", action="store_true",
                       help="ask the provider to disable reasoning. Use for hybrid models: a "
                            "chain of thought spends the token budget before the answer and "
                            "returns as a parse error, which looks like a bad judge.")
    p_run.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    p_run.set_defaults(func=cmd_run)

    p_work = sub.add_parser("worksheet", parents=[common],
                            help="write blind worksheets so the operator can label items "
                                 "themselves. No API calls, no cost.")
    p_work.add_argument("--per-judge", type=int, default=15,
                        help="items per judge. 15 is about an evening and scores every "
                             "candidate and both incumbents at once.")
    p_work.add_argument("--seed", type=int, default=20260823)
    p_work.set_defaults(func=cmd_worksheet)

    p_report = sub.add_parser("report", parents=[common], help="compare everything on disk")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    if args.command == "sample":
        concepts = []
        for spec in args.source:
            _, _, path = spec.partition("=")
            summary = Path(path.strip()) / "summary.json"
            if summary.exists():
                concepts.append(json.loads(summary.read_text(encoding="utf-8"))["concept"])
        if args.probe:
            concepts.append(args.probe_concept)
        _refuse_harmful(concepts)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
