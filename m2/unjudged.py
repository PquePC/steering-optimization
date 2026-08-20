"""m2.unjudged - the judge-free validation probe (task 29).

**On the module name.** This is the probe the `--probe-cells` flag runs, but it is NOT called
`m2.probe`: `m2.steer` already exports a public name `probe` (an interactive helper), and a
`m2/probe.py` would make `from m2 import probe` ambiguous -- resolving to whichever of the two
the package's `__getattr__` reached first, and silently shadowing the other. `unjudged` also
names the property that makes the module reusable outside the mid band: nothing in it is scored.


**What this exists to settle.** The 2026-08-14 Garlic run measured real `d2` on 38 cells and
every one read 1.000. That looked like "detection saturates wherever influence exists". It is
not: `SHORTLIST` only ever offered `d2` the cells that cleared `e6`, and `e6` (`cheap.measure_E6`)
is the concept's probability mass at the FIRST generated token of an answer to "What is 17 x 23?".
A model that is thoroughly steered but still answering the question opens with "The" or "17", so
`e6` reads 0.000 there and the cell was never eligible. `e6` only turns positive once the
injection is strong enough to hijack the opening token of an unrelated answer, which is the
collapse regime -- and `s3`, the scan-time sanity term, is a four-way argmax over option-letter
logits that cannot see generative collapse at all. Effectiveness that only fires at collapse,
plus sanity that is blind to collapse, selects exactly the broken cells and calls them sane.

Every `d2`-measured cell in that run was at L47 or deeper. The band M1 already showed carries a
clean dissociation -- L37 at r ~ 0.20, `D2` 0.96 / `D1` 0.08 / sanity 0.93 / KL 0.69, transcripts
verified -- was never measured at all. This module measures it, with the selection gate removed.

**Judge-free by construction.** Nothing here calls a judge. `no_judges()` patches the judge
entry points to raise for the duration of the probe, so a judge call is a crash rather than a
silent charge, and every number this writes is either a forward-pass scalar or a mechanical
count. Reading the transcripts is the operator's job; that is the point of the mode.

**Three response channels, because they are three different questions.**

    detect   the noticing question with NO prefill -- `prompts.detect_prompts`. "Do you detect
             an injected thought? If so, what is it about?" and the model answers for itself.
             This is the only channel where a volunteered "Yes, I detect an injected thought!
             It's about garlic" can appear, and it is what the probe is for.
    forced   the same conversation WITH the prefill -- `prompts.forced_prompts`, D2's prompt.
             Kept so this run is comparable with every earlier `d2` number, minus the judge.
    task     the twelve E5 prompts, generated steered. This is influence in ordinary output,
             which is what `e6` was standing in for and failing to measure.

**The unsteered null arm is mandatory, not optional.** The noticing framing tells the model that
a thought is injected on half of trials; it will sometimes answer yes with nothing injected. M1
recorded exactly that ("the word apple", never the target). A detect-channel transcript is
uninterpretable without the same channel at alpha = 0 on the same prompts, so `run_probe` always
measures it and refuses to report without it.

**Scalars recorded, none of them gating.** `e6`, `d3`, `s3` come along per cell because they are
nearly free and because the whole point is to show what `e6` reads at cells where the other
channels show influence. Nothing in this module filters, ranks or selects on any of them. There
is no shortlist here.

Deliberately self-contained, like `m2.autopsy`: no pipeline module imports it, heavy imports stay
inside the functions, and it changes no production measure.
"""

from __future__ import annotations

import contextlib
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterator, Sequence


__all__ = [
    "DEFAULT_CELLS",
    "REFERENCE_CELLS",
    "PROBE_N",
    "PROBE_TRIALS",
    "parse_cells",
    "concept_hits",
    "no_judges",
    "prepare",
    "measure_null",
    "measure_cell",
    "run_probe",
    "record_provenance",
    "export_bundle",
]

# ---------------------------------------------------------------------------------------
# The cell list
# ---------------------------------------------------------------------------------------
# L37-L46 is the band the 2026-08-14 run never measured `d2` on. Its SCAN rows read `d3` 0.93 to
# 1.00 with `s3` 0.71 to 0.93 -- the concept at the output stage with the model intact -- and
# `e6` reach 0.000 at every one of them, which is why none was ever eligible.
#
# The doses bracket M1's L37 operating point (r_L ~ 0.20, alpha = 2) on both sides. 0.15 and 0.30
# are the two SCAN doses that already have rows to compare against; 0.20 and 0.25 fill the gap
# that contains the operating point and has never been sampled at any layer.
PROBE_LAYERS: tuple[int, ...] = tuple(range(37, 47))
PROBE_DOSES: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)

DEFAULT_CELLS: tuple[tuple[int, float], ...] = tuple(
    (layer, dose) for layer in PROBE_LAYERS for dose in PROBE_DOSES)

# Two anchors from the 2026-08-14 low-dose autopsy, carried so this run can be read against a
# known result rather than only against itself:
#
#   L57@0.22  the one cell in that probe that was fully sane -- s2_forced 5/5 with d2 5/5.
#             If it comes back sane and identifying, the generation path still behaves.
#   L59@0.30  d2 5/5 with s2_forced 0/5: every response a repetition loop. If it comes back
#             degenerate, the degeneracy detector still behaves.
#
# They are anchors, NOT a pass/fail control. Neither is load-bearing for the mid-band reading,
# which is why nothing here raises on them -- unlike task 25, whose positive control gated the
# whole dump and, as item 27 records, was a criterion a broken model passed anyway.
REFERENCE_CELLS: tuple[tuple[int, float], ...] = ((57, 0.22), (59, 0.30))

# Trials per cell per generated channel. Five was task 25's screening resolution and it left
# every rate with a Wilson interval half the unit wide; eight is still cheap (the probe's cost is
# dominated by the twelve task generations) and gives the transcripts enough draws to show
# whether a response shape is typical or a one-off.
PROBE_N: int = 8

# Fixed trial numbers, so a rerun of the same cell produces the same prompts and the rows join.
# Spread across the 1-50 range the framing mentions rather than 1..8, matching how `cheap.D3_TRIALS`
# samples it.
PROBE_TRIALS: tuple[int, ...] = (1, 7, 13, 19, 25, 31, 37, 43)

if len(PROBE_TRIALS) != PROBE_N:                        # import-time, so it cannot ship wrong
    raise AssertionError(
        f"PROBE_TRIALS has {len(PROBE_TRIALS)} entries but PROBE_N is {PROBE_N}. The trial list "
        "is what the prompts are built from, so a disagreement means the run generates a "
        "different number of responses from the one every rate is divided by.")

CELLS_FILE = "probe_cells.jsonl"
DETAIL_FILE = "probe_detail.jsonl"
DETECT_FILE = "probe_detect_transcripts.jsonl"
FORCED_FILE = "probe_forced_transcripts.jsonl"
TASK_FILE = "probe_task_transcripts.jsonl"
NULL_FILE = "probe_null_transcripts.jsonl"
SUMMARY_FILE = "probe_summary.json"

# The layer half admits `L` because `_LAYER_ONE`/`_LAYER_RANGE` do, and a segment filter that
# is stricter than the parser it feeds rejects `L37@0.2` with "invalid segment" while the
# documentation, the CLI help and the parser all say that form is fine.
_SEGMENT = re.compile(r"^(?P<layers>[0-9Ll,\-\s]+)@(?P<doses>[0-9.,\s]+)$")
_LAYER_RANGE = re.compile(r"^L?(\d+)\s*-\s*L?(\d+)$", re.IGNORECASE)
_LAYER_ONE = re.compile(r"^L?(\d+)$", re.IGNORECASE)


def _cell_key(layer: int, r: float) -> tuple[int, float]:
    """Stable identity for a cell. Rounded, because item 24's raw floats reach the output.

    `0.45` arrived at by arithmetic is `0.44999999999999996`, and two rows keyed on that and on
    the literal `0.45` are two cells as far as every join is concerned.
    """
    return int(layer), round(float(r), 6)


def _parse_layers(text: str) -> list[int]:
    out: list[int] = []
    for item in text.split(","):
        piece = item.strip()
        if not piece:
            continue
        span = _LAYER_RANGE.fullmatch(piece)
        if span is not None:
            lo, hi = int(span.group(1)), int(span.group(2))
            if hi < lo:
                raise ValueError(f"layer range {piece!r} runs backwards")
            out.extend(range(lo, hi + 1))
            continue
        one = _LAYER_ONE.fullmatch(piece)
        if one is None:
            raise ValueError(f"invalid layer {piece!r}; expected 37, L37 or 37-46")
        out.append(int(one.group(1)))
    if not out:
        raise ValueError("no layers given")
    return out


def _parse_doses(text: str) -> list[float]:
    out: list[float] = []
    for item in text.split(","):
        piece = item.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError as exc:
            raise ValueError(f"invalid dose {piece!r}: not a number") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid dose {piece!r}: r must be finite and above zero")
        out.append(value)
    if not out:
        raise ValueError("no doses given")
    return out


def parse_cells(text: str | None, *, with_references: bool = True) -> list[tuple[int, float]]:
    """Parse `37-46@0.15,0.20;57@0.22` into an ordered, deduplicated cell list.

    A blank value means `DEFAULT_CELLS`. Segments are separated by `;`, and each is
    `<layers>@<doses>` where layers may be single (`37`, `L37`) or a range (`37-46`) and doses
    are a comma list. The grid form is what makes a forty-cell sweep expressible without forty
    hand-typed cells getting one of them wrong.

    `with_references` appends `REFERENCE_CELLS` unless they are already present. They are
    appended, not prepended, so the mid-band cells are measured first and a run that has to be
    cut short still has the thing it was for.
    """
    if text is None or not str(text).strip():
        cells = list(DEFAULT_CELLS)
    else:
        cells = []
        seen: set[tuple[int, float]] = set()
        for raw in str(text).split(";"):
            segment = raw.strip()
            if not segment:
                continue
            match = _SEGMENT.fullmatch(segment)
            if match is None:
                raise ValueError(
                    f"invalid probe segment {segment!r}; expected LAYERS@DOSES, for example "
                    "37-46@0.15,0.30")
            for layer in _parse_layers(match.group("layers")):
                for dose in _parse_doses(match.group("doses")):
                    key = _cell_key(layer, dose)
                    if key not in seen:
                        seen.add(key)
                        cells.append(key)
        if not cells:
            raise ValueError("probe cell list parsed to nothing")

    if with_references:
        present = {_cell_key(*cell) for cell in cells}
        for cell in REFERENCE_CELLS:
            if _cell_key(*cell) not in present:
                cells.append(_cell_key(*cell))
    return [_cell_key(*cell) for cell in cells]


# ---------------------------------------------------------------------------------------
# Judge-free enforcement
# ---------------------------------------------------------------------------------------

@contextlib.contextmanager
def no_judges() -> Iterator[None]:
    """Make any judge call raise for the duration of the block.

    The probe does not call a judge, so this patches nothing that is reached on the happy path.
    It exists because "this mode does not use judges" is a claim about every branch of every
    function it transitively calls, and the only version of that claim worth having is one that
    fails loudly. A judge call here would spend money, write a `judge_*.jsonl` the operator
    would then read as evidence, and quietly reintroduce the scoring layer this run was set up
    to do without.

    The originals are restored in a `finally`, so an exception inside the block leaves the
    process able to run the pipeline afterwards rather than carrying a poisoned `expensive`.
    """
    from . import expensive, judges

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "m2.unjudged runs judge-free: something reached a judge entry point. No probe "
            "measurement is scored, so this is a code path that does not belong in this "
            "mode rather than a configuration to relax.")

    saved = {
        (expensive, "_issue"): expensive._issue,
        (judges, "call_judge"): judges.call_judge,
        (judges, "judge_many"): judges.judge_many,
    }
    for (module, name) in saved:
        setattr(module, name, _refuse)
    try:
        yield
    finally:
        for (module, name), original in saved.items():
            setattr(module, name, original)


# ---------------------------------------------------------------------------------------
# Mechanical, judge-free response statistics
# ---------------------------------------------------------------------------------------

def concept_hits(text: str, concept: str) -> int:
    """Occurrences of the concept word in one response. Mechanical, no judge, no model.

    Matched case-insensitively at a leading word boundary with NO trailing boundary, so
    "garlicky" and "garlic-forward" both count -- they are mentions of the concept, and a
    stricter match would score a steered response as clean because it inflected the word.

    This is a **count, not a verdict.** It cannot tell "the response is about garlic" from "the
    response mentions garlic once in passing", and it cannot see an influenced response that
    never names the concept (M1's origami cell described "a delicate, precise unfolding" for a
    whole sentence before naming it). It is here to make the transcripts sortable, not to
    replace reading them. Judge E5 is what grades influence, and this run does not run it.
    """
    word = str(concept).strip()
    if not word:
        raise ValueError("concept_hits needs a non-empty concept")
    # A PLURAL concept also matches its singular, because the leading-boundary-only rule above
    # is blind to it otherwise. `\bWrists` does not match "wrist", "wristwatch" or "wristband" --
    # the letters after "wrist" are not "s". The 2026-08-20 Qwen run recorded concept_mentions=0
    # for all 624 Wrists effect responses, including one reading "My wristwatch," -- the concept
    # appearing in the output and being counted as absent.
    #
    # A trailing "ss" is left alone: stripping it would make "Glass" match "glasnost".
    forms = [word]
    low = word.lower()
    if len(word) > 3 and low.endswith("s") and not low.endswith("ss"):
        forms.append(word[:-1])
    alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return len(re.findall(r"\b(?:" + alt + r")", str(text), flags=re.IGNORECASE))


def _channel_stats(responses: Sequence[str], concept: str) -> dict:
    """The judge-free summary of one channel's responses at one cell.

    `s2` here is `cheap.measure_S2`, the same mechanical degeneracy detector the pipeline folds
    into `s4` -- computed per channel rather than pooled, because item 27's finding was
    precisely that the forced channel had collapsed while `s3` on the same cell read 0.95. One
    number over three different kinds of response would hide that again.
    """
    from . import cheap

    responses = [str(text) for text in responses]
    s2 = cheap.measure_S2(responses)
    hits = [concept_hits(text, concept) for text in responses]
    words = [len(text.split()) for text in responses]
    mentioned = sum(1 for h in hits if h > 0)
    n = len(responses)
    hit_ci_low, hit_ci_high = cheap.wilson_interval(mentioned, n)
    return dict(
        n=n,
        s2=s2["s2"],
        s2_count=s2["s2_count"],
        s2_ci_low=s2["s2_ci_low"],
        s2_ci_high=s2["s2_ci_high"],
        s2_reasons=s2["s2_reasons"],
        mention_rate=mentioned / n,
        mention_count=mentioned,
        mention_ci_low=hit_ci_low,
        mention_ci_high=hit_ci_high,
        hits_median=statistics.median(hits),
        hits_max=max(hits),
        words_median=statistics.median(words),
        words_min=min(words),
    )


def _per_response_rows(responses: Sequence[str], *, concept: str, channel: str,
                       labels: Sequence[Any], label_field: str, **fixed: Any) -> list[dict]:
    """One transcript row per response, carrying its own mechanical verdicts.

    The degeneracy reason travels WITH the response rather than only in the cell aggregate. On
    the last probe the aggregate said "44 of 50 degenerate" and answering "which ones, and by
    which rule" needed the detector re-run offline against the archive.
    """
    from . import cheap

    rows: list[dict] = []
    for label, text in zip(labels, responses):
        response = str(text)
        reason = cheap.degeneracy_reason(response)
        rows.append(dict(
            channel=channel,
            **{label_field: label},
            response=response,
            concept_hits=concept_hits(response, concept),
            words=len(response.split()),
            degenerate=reason is not None,
            degeneracy_reason=reason,
            **fixed,
        ))
    return rows


# ---------------------------------------------------------------------------------------
# Guards, shared with the task 25 diagnostic's policy
# ---------------------------------------------------------------------------------------

def _assert_outside_repo(path: Path, *, repo_root: Path | None = None) -> Path:
    """Refuse any probe output path inside the repository working tree.

    Same rule as `autopsy._assert_outside_repo`, restated rather than imported because the two
    modules are both disposable and neither should acquire a dependency on the other. A run
    writes model generations; `.gitignore` is load-bearing (CLAUDE.md hard rule 3) and a path
    inside the tree is one edit away from being committed.
    """
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve(strict=False)
    target = Path(path).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError:
        return target
    raise ValueError(
        f"probe output path {str(target)!r} is inside repository {str(root)!r}; "
        "the probe may write only to the external run directory")


def _require_transcripts_allowed(concept: str, cfg: dict) -> str:
    """The same two-argument transcript gate as `runio.export_bundle`, with no override.

    This mode's entire output is transcripts. A non-benign concept must be structurally unable
    to enter it, so there is deliberately no override parameter to pass.
    """
    from . import runio

    allowed, reason = runio.transcripts_allowed(concept, cfg)
    if not allowed:
        raise PermissionError(f"probe refused for {concept!r}: {reason}")
    return reason


def prepare(concepts: Sequence[str], cfg: dict, cell_text: str | None, *,
            log_path: Path | None = None,
            with_references: bool = True) -> list[tuple[int, float]]:
    """Validate the standalone mode before model loading creates any output."""
    if len(concepts) != 1:
        raise ValueError(
            f"the probe measures exactly one benign concept, got {len(concepts)}: {list(concepts)}")
    concept = str(concepts[0])
    _require_transcripts_allowed(concept, cfg)
    cells = parse_cells(cell_text, with_references=with_references)

    from . import config

    concept_cfg = dict(cfg)
    concept_cfg["concept"] = concept
    _assert_outside_repo(config.run_dir_for(concept, concept_cfg))
    if log_path is not None:
        _assert_outside_repo(Path(log_path))
    return cells


# ---------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------

def _gen_settings() -> tuple[int, float]:
    """`(max_new_tokens, temperature)` for every generated channel.

    Both are read from CONFIG rather than fixed here, so this probe generates under the same
    settings as the run it is meant to be compared with. In particular MAX_NEW_TOKENS stays at
    the run's value: `s2`'s 5-gram rule is more likely to fire in longer text, so a probe that
    quietly generated 200 tokens would report a degeneracy rate that no earlier cell's number
    can be read against.
    """
    from . import config

    return int(config.CONFIG["MAX_NEW_TOKENS"]), float(config.CONFIG["TEMPERATURE"])


_BUILDERS = {"detect": "detect_prompts", "forced": "forced_prompts"}


def _noticing_channel(channel: str, *, layer: int | None, alpha: float | None,
                      trials: Sequence[int]) -> list[str]:
    """Generate one noticing channel, steered or unsteered, through the correct batched path.

    `channel` names the builder rather than taking a callable, so the two channels can only be
    the two `m2.prompts` builders that share `_noticing_prompts` and no caller can pass in a
    third construction of the same conversation.

    `layer is None` means the unsteered null arm and routes to `generate_unsteered`, which uses
    the same multi-steering decoder at strength zero. `generate_steered` refuses alpha = 0
    outright, so there is no way to record an unsteered generation with a steered label.
    """
    from . import expensive, prompts as prompt_assets

    if channel not in _BUILDERS:
        raise ValueError(f"unknown noticing channel {channel!r}; expected one of {sorted(_BUILDERS)}")
    builder = getattr(prompt_assets, _BUILDERS[channel])
    max_new, temperature = _gen_settings()
    prompts_text, start = builder(list(trials))
    if start is None:
        raise RuntimeError(
            "the noticing builder could not locate its own trial marker, so there is no "
            "steering start position. Steering from 0 would steer the framing (bug 8).")
    starts = [int(start)] * len(prompts_text)
    if layer is None:
        return expensive.generate_unsteered(
            prompts_text, max_new, temperature, start_positions=starts)
    return expensive.generate_steered(
        prompts_text, int(layer), float(alpha), max_new, temperature, start_positions=starts)


def _task_channel(*, layer: int | None, alpha: float | None) -> tuple[list[str], list[str]]:
    """Generate the twelve E5 prompts, steered or unsteered. Returns `(ids, responses)`."""
    from . import expensive, prompts as prompt_assets

    max_new, temperature = _gen_settings()
    rendered, starts, ids = expensive.task_batch(prompt_assets.E5_PROMPTS)
    if layer is None:
        responses = expensive.generate_unsteered(
            rendered, max_new, temperature, start_positions=starts)
    else:
        responses = expensive.generate_steered(
            rendered, int(layer), float(alpha), max_new, temperature, start_positions=starts)
    return ids, responses


# ---------------------------------------------------------------------------------------
# The null arm
# ---------------------------------------------------------------------------------------

def measure_null(concept: str) -> dict:
    """The unsteered baseline for all three channels. Written once per run, before any cell.

    **This is not a formality.** The noticing framing states that a thought is injected on half
    of trials, which is an invitation to answer yes; M1 section 4 recorded the model inventing
    "the word apple" under exactly this prompt with nothing injected. Without this arm, a
    detect-channel transcript reading "Yes, I detect an injected thought" is not evidence of
    anything -- it could be the prompt talking. With it, the steered rate is readable against
    the rate the framing produces on its own.

    It also gives the task channel its unsteered reference: the twelve E5 responses here are
    what a steered response at any cell should be compared against when reading whether the
    output drifted.
    """
    from . import runio

    trials = list(PROBE_TRIALS)
    detect = _noticing_channel("detect", layer=None, alpha=None, trials=trials)
    forced = _noticing_channel("forced", layer=None, alpha=None, trials=trials)
    task_ids, task = _task_channel(layer=None, alpha=None)

    fixed = dict(layer=None, r=None, alpha=0.0, steered=False)
    for row in _per_response_rows(detect, concept=concept, channel="detect",
                                  labels=trials, label_field="trial", **fixed):
        runio.write_row(NULL_FILE, row)
    for row in _per_response_rows(forced, concept=concept, channel="forced",
                                  labels=trials, label_field="trial", **fixed):
        runio.write_row(NULL_FILE, row)
    for row in _per_response_rows(task, concept=concept, channel="task",
                                  labels=task_ids, label_field="prompt_id", **fixed):
        runio.write_row(NULL_FILE, row)

    out = dict(
        detect=_channel_stats(detect, concept),
        forced=_channel_stats(forced, concept),
        task=_channel_stats(task, concept),
    )
    print("\nNULL ARM (unsteered, alpha=0) - what the framing produces with nothing injected")
    for channel in ("detect", "forced", "task"):
        stats = out[channel]
        print(f"   {channel:<7} n={stats['n']:<3} mention_rate={stats['mention_rate']:.3f}  "
              f"s2={stats['s2']:.3f}  words_med={stats['words_median']:g}")
    return out


# ---------------------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------------------

def _cheap_scalars(layer: int, alpha: float) -> dict:
    """`e6`, `d3` and `s3` at one cell. Recorded, never used to filter anything here.

    `e6` is included precisely because it is the measure that mis-selected the last run. Having
    it beside a task-channel mention rate measured on the same cell is what turns "e6 is the
    wrong axis" from an argument into a table.
    """
    from . import cheap

    e6 = cheap.measure_E6(int(layer), float(alpha))
    d3 = cheap.measure_D3(int(layer), float(alpha))
    s3 = cheap.measure_S3(int(layer), float(alpha))
    return dict(e6=e6, d3=d3, s3=s3)


def measure_cell(layer: int, r: float, *, concept: str) -> dict:
    """Every channel and every scalar at one cell, with nothing judged and nothing filtered."""
    from . import config, runio

    alpha = float(config.alpha_for(int(layer), float(r)))
    trials = list(PROBE_TRIALS)

    scalars = _cheap_scalars(layer, alpha)
    detect = _noticing_channel("detect", layer=layer, alpha=alpha, trials=trials)
    forced = _noticing_channel("forced", layer=layer, alpha=alpha, trials=trials)
    task_ids, task = _task_channel(layer=layer, alpha=alpha)

    fixed = dict(layer=int(layer), r=_cell_key(layer, r)[1], alpha=alpha, steered=True)
    for name, channel, responses, labels, field in (
            (DETECT_FILE, "detect", detect, trials, "trial"),
            (FORCED_FILE, "forced", forced, trials, "trial"),
            (TASK_FILE, "task", task, task_ids, "prompt_id")):
        for row in _per_response_rows(responses, concept=concept, channel=channel,
                                      labels=labels, label_field=field, **fixed):
            runio.write_row(name, row)

    row = dict(
        layer=int(layer),
        r=_cell_key(layer, r)[1],
        alpha=alpha,
        # The three scalars, flattened to the names every earlier artefact uses so this file
        # joins straight onto `scan.jsonl` and `selection_d2.jsonl` without a translation step.
        e6_reach=scalars["e6"]["reach"],
        e6_mass_median=scalars["e6"]["e6_mass_median"],
        e6_rank_med=scalars["e6"]["e6_rank_med"],
        d3=scalars["d3"]["d3"],
        d3_rank_med=scalars["d3"]["d3_rank_med"],
        s3=scalars["s3"]["s3"],
        s3_margin=scalars["s3"]["s3_margin"],
        detect=_channel_stats(detect, concept),
        forced=_channel_stats(forced, concept),
        task=_channel_stats(task, concept),
    )
    runio.write_row(CELLS_FILE, row)
    runio.write_row(DETAIL_FILE, dict(
        layer=int(layer), r=row["r"], alpha=alpha,
        e6_per_prompt=scalars["e6"]["e6_per_prompt"],
        d3_per_trial=scalars["d3"].get("d3_per_trial"),
        s3_per_item=scalars["s3"]["s3_per_item"]))
    return row


def _print_cell(row: dict) -> None:
    """One line per cell, with the comparison the probe exists to make on it.

    `e6` and the task-channel mention rate sit next to each other deliberately: they are two
    readings of "is this cell influential", and the claim under test is that they disagree.
    """
    cell = f"L{int(row['layer'])}@{float(row['r']):.2f}"
    print(f"   {cell:>10}  a={float(row['alpha']):6.3f}  "
          f"e6={float(row['e6_reach']):.3f}  d3={float(row['d3']):.3f}  "
          f"s3={float(row['s3']):.3f}  |  "
          f"task_mention={row['task']['mention_rate']:.3f} s2={row['task']['s2']:.3f}  |  "
          f"detect_mention={row['detect']['mention_rate']:.3f} s2={row['detect']['s2']:.3f}  |  "
          f"forced_mention={row['forced']['mention_rate']:.3f} s2={row['forced']['s2']:.3f}")


def _print_header() -> None:
    print(f"   {'cell':>10}  {'alpha':>7}  {'e6':>6} {'d3':>6} {'s3':>6}  |  "
          "task(mention/s2)  |  detect(mention/s2)  |  forced(mention/s2)")


# ---------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------

def run_probe(cells: Sequence[tuple[int, float]]) -> dict:
    """Measure every cell, judge-free, and return the summary payload.

    Order of work: vectors, then the `s3` baseline, then the null arm, then the cells. The null
    arm goes before any steered measurement so that a run killed part-way still has the
    baseline its transcripts must be read against.
    """
    from . import config

    ctx = config.RUN
    if ctx.run_dir is None or not ctx.concept:
        raise RuntimeError("call model.load_model and driver.set_concept before the probe")
    concept = str(ctx.concept)
    _require_transcripts_allowed(concept, ctx.config)
    _assert_outside_repo(Path(ctx.run_dir))

    from . import cheap, model, prompts as prompt_assets, runio, vectors

    parsed = [_cell_key(*cell) for cell in cells]
    if not parsed:
        raise ValueError("run_probe was given no cells")
    layers = sorted({int(layer) for layer, _r in parsed})
    doses = sorted({float(r) for _layer, r in parsed})

    with no_judges():
        print(f"\nextracting vectors for {len(layers)} layers")
        vectors.extract_all_layers(concept, layers)
        # write=True, unlike the task 25 diagnostic: `norms.jsonl` and `dose_map.json` are
        # scalars, they carry `||v_L||` and `||h_L||` for every cell measured here, and without
        # them a reader cannot check that a given r means the same perturbation it meant on the
        # 2026-08-14 run. Vectors themselves are never written -- `.pt` is on EXPORT_DENY and
        # in `.gitignore` (CLAUDE.md hard rule 3).
        vectors.build_dose_map(
            layers, doses, calib_prompts=[row["text"] for row in prompt_assets.E5_PROMPTS])

        # R14, here rather than in `gates.rig_checks()`. It reads the concept vectors, so it
        # SKIPS at rig-check time when nothing has been extracted yet; on the pipeline path
        # `phase0_calibrate` runs it after extraction, and this mode never enters Phase 0.
        #
        # It is the single most load-bearing check in the run. Bug 26 was the repo's hook
        # declining to steer whenever `start_pos` was set, and it produced identically zero
        # readings at all 30 cells of a real run for an hour with every other check satisfied.
        # This probe's whole finding is whether the mid band shows influence, so a dead hook
        # would hand back a clean, plausible, entirely empty null across 42 cells. It raises.
        liveness = model.hook_liveness()
        print(f"R14        : pass - start_pos {liveness['d_start_pos']:.2e}, "
              f"all-positions {liveness['d_all_pos']:.2e} "
              f"(L{liveness['layer']} alpha={liveness['alpha']:.3f})")

        # `_s3_pass` hard-indexes `RUN.mmlu` and `load_mmlu_items` deliberately does not
        # assign it -- on the pipeline path `phase0_calibrate` does. This mode never enters
        # Phase 0, so it has to pin the item set itself or every cell raises on its sanity
        # term after its generations have already been paid for.
        if not getattr(ctx, "mmlu", None):
            ctx.mmlu = prompt_assets.load_mmlu_items(ctx.config)
        print(f"pinned {len(ctx.mmlu)} MMLU items")

        # `measure_S3` is a ratio against this and hard-indexes it. Without this call every
        # cell would raise on its sanity term after the generations had been paid for.
        print("measuring the unsteered MMLU baseline")
        baseline = cheap.measure_S3_baseline()
        print(f"   cap_base = {baseline['cap_base']}/{baseline['s3_n']} "
              f"({baseline['s3_acc_base']:.3f})")

        # The null arm is re-measured on a resume rather than skipped. It is three batches, it
        # is what every transcript is read against, and a resumed run may be on a different pod
        # -- bf16 kernels differ between GPU architectures, so a baseline carried over from
        # another card would be the one row in the file that came from a different instrument.
        null = measure_null(concept)

        # Row-level resume, per spec 14.8. The 2026-08-14 run lost a phase to a transient volume
        # EIO and the work already on disk was unusable to the next attempt; this is ~35 minutes
        # of GPU and the same fault would cost all of it.
        done = runio.done_keys(CELLS_FILE, ("layer", "r"))
        todo = [cell for cell in parsed
                if (runio._key_value(cell[0]), runio._key_value(cell[1])) not in done]
        if len(todo) < len(parsed):
            print(f"\nresuming: {len(parsed) - len(todo)} cell(s) already in {CELLS_FILE}")

        print(f"\n{len(todo)} cells, {PROBE_N} trials per noticing channel, "
              f"{len(prompt_assets.E5_PROMPTS)} task prompts each")
        _print_header()
        rows: list[dict] = []
        unreachable: list[dict] = []
        for layer, r in todo:
            try:
                row = measure_cell(layer, r, concept=concept)
            except config.Unreachable as exc:
                # An unreachable cell is a fact about the dose map, not a failure. Record it by
                # name: item 26's lesson is that work a run silently did not do reads later as
                # work it did and found nothing in.
                unreachable.append(dict(layer=int(layer), r=_cell_key(layer, r)[1],
                                        reason=str(exc)))
                print(f"   L{layer}@{r:.2f}  UNREACHABLE: {exc}")
                continue
            rows.append(row)
            _print_cell(row)

    # Read the cell rows back off disk rather than reporting `rows`. On a resumed run `rows`
    # holds only what THIS attempt measured, and a summary that counted those would report a
    # complete 42-cell probe as a 12-cell one -- the same class of error as a board numerator
    # that counts something narrower than its label says.
    measured = runio.read_rows(CELLS_FILE)
    summary = dict(
        concept=concept,
        mode="probe",
        judged=False,
        n_cells_requested=len(parsed),
        n_cells_measured=len(measured),
        n_cells_this_attempt=len(rows),
        probe_n=PROBE_N,
        probe_trials=list(PROBE_TRIALS),
        cap_base=baseline["cap_base"],
        s3_n=baseline["s3_n"],
        null=null,
        unreachable=unreachable,
        cells=measured,
    )
    runio.write_json(SUMMARY_FILE, summary)
    runio.log(
        f"probe complete for {concept}: {len(measured)}/{len(parsed)} cells measured "
        f"({len(rows)} this attempt), judge-free; no cell was selected, ranked or filtered",
        "INFO")
    _print_closing(summary)
    return summary


def record_provenance() -> dict:
    """Write this probe's `provenance.jsonl` row before any measurement.

    Kept here rather than added to `m2.driver` because `driver.run_concept` writes its own row
    on the pipeline path and this mode never enters it -- the last standalone bundle came back
    with no provenance at all, which is why the run's own git commit had to be guessed. Never
    raises: a missing version must not cost a run.
    """
    from . import model, runio

    try:
        prov = model.provenance()
        prov["ts"] = runio._now()
        prov["phase_entered_at"] = "probe"
        prov["mode"] = "probe"
        prov["judged"] = False
        runio.write_row("provenance.jsonl", prov)
        runio.log(f"provenance: {prov.get('gpu', '?')} | torch {prov.get('torch', '?')} | "
                  f"commit {str(prov.get('git_commit'))[:12]}"
                  f"{' (DIRTY)' if prov.get('git_dirty') else ''}")
        return prov
    except Exception as exc:                        # noqa: BLE001 - never cost a run
        runio.log(f"provenance not recorded ({type(exc).__name__})", "WARN")
        return {}


def export_bundle() -> Path:
    """Archive the probe's run directory and export it, transcripts included.

    No `EXPORT_TRANSCRIPTS_OVERRIDE`: `prepare` and `run_probe` have both already refused any
    concept that is not on `BENIGN_CONCEPTS`, and this mode's whole output is transcripts, so
    there is nothing here an override could usefully unlock.
    """
    from . import config, runio

    run_dir = Path(config.RUN.run_dir)
    runio.archive_concept(run_dir)
    return runio.export_bundle(run_dir)


def _print_closing(summary: dict) -> None:
    """State what the run did and did not establish, in the output the operator reads first."""
    rows = summary["cells"]
    print("\n" + "=" * 78)
    print(f"PROBE COMPLETE - {summary['n_cells_measured']}/{summary['n_cells_requested']} cells")
    print("=" * 78)
    if summary["unreachable"]:
        print(f"   {len(summary['unreachable'])} cell(s) unreachable above ALPHA_CEIL: "
              + ", ".join(f"L{u['layer']}@{u['r']:.2f}" for u in summary["unreachable"]))
    blind = [r for r in rows
             if float(r["e6_reach"]) == 0.0 and float(r["task"]["mention_rate"]) > 0.0]
    if blind:
        print(f"   {len(blind)} cell(s) read e6 = 0.000 while the task channel mentioned the "
              "concept in ordinary output")
    print("   Nothing here is judged. Influence is a mention COUNT, not Judge E5, and detection "
          "is\n   a transcript, not a rate. Read the transcripts before quoting any number.")
