"""m2.expensive - the generation + judge tier: E5, S1, D2/D4, and the judge-FPR control.

Everything here costs a generation, a judge call, or both, so nothing here runs over the
whole scan: spec section 5 splits the pipeline into a cheap tier (forward passes only,
`m2.cheap`) that crosses every layer, and this tier, which runs on the Phase 2 shortlist.

  generate_steered     one batched steered generation, per-row start positions   (bug 25b)
  generate_unsteered   the same, unsteered - NOT `mw.generate_batch`             (see below)
  measure_E5           spec 5.6 - Judge E5 against the cached unsteered reference A
  measure_S1           spec 5.7 - Judge S1 on the SAME responses, concept withheld
  measure_D2           spec 5.9 - N_D2 forced-ID generations, Judge D2, D4 distribution
  judge_fpr            E5 null on every open prompt, once per concept
  judge_s1_null        S1 null on unsteered/unsteered pairs, checked against objective S2
  judge_d2_null        unsteered forced-ID baseline, reported and never selected away
  verify_cell          one verified.jsonl row: generate once, fan out to E5, S1 and B

Three structural points, each of which is a bug that already happened once:

1. **Every batched generation goes through `generate_batch_with_multi_steering`.** Never
   `generate_batch_with_steering`, and never `generate_batch`. Bug 25b: under left padding
   the single-vector path applies one scalar start position to the whole batch and slices
   the decode at the UNPADDED length, so shorter prompts are steered one token too early
   AND carry a prompt token into the "generated" text. Both failures are silent. See the
   docstring on `generate_steered`.

2. **E5 and S1 issue in ONE `judge_many` call.** Spec 5.7 prices the second judge at "24
   calls per cell instead of 12 ... E5 and S1 for a given prompt are independent and issue
   concurrently, so wall time is unchanged". Calling `measure_E5` and then `measure_S1`
   would serialise them and make the split cost wall time as well as tokens. `verify_cell`
   is therefore the path Phase 4/5 uses; the two single-measure functions exist for
   diagnosis and for the phases that genuinely need only one of them.

3. **The concept is withheld from S1 structurally, and the withholding is asserted per
   call.** See `s1_blind_frame` for what exactly is asserted and why it cannot be the raw
   payload.

Layout order (CONTRACT section 1): this module may import config, model, vectors, prompts,
cheap and judges, and nothing later. That is why it appends its own JSONL rather than
calling `runio.write_row` - `runio` comes later in the dependency order. The rows match
CONTRACT section 4 (`concept`, `config_hash`, `ts` on every row) so `runio.read_rows` reads
these files unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from . import cheap
from . import config
from . import judges
from . import model
from . import prompts as prompt_assets
from . import vectors

__all__ = [
    # CONTRACT section 3 surface
    "generate_steered",
    "measure_E5",
    "measure_S1",
    "measure_D2",
    "judge_fpr",
    "judge_s1_null",
    "judge_d2_null",
    "judge_nulls",
    "verify_cell",
    # additions, documented in the module docstring and in each function
    "generate_unsteered",
    "task_batch",
    "reference_A",
    "unsteered_samples",
    "s1_blind_frame",
    "phase_scope",
    "current_phase",
    "assert_templates_sane",
    # constants other modules read
    "JUDGE_E5_ID", "JUDGE_S1_ID", "JUDGE_D2_ID",
    "JUDGE_A1_FILE", "JUDGE_A2_FILE", "JUDGE_B_FILE",
    "CIS_FILE", "D2_FILE",
    "FAILURE_MODES", "DAMAGE_MODES", "RETRIEVAL_MODES",
    "GEN_BATCH_MAX",
    "BASE_FPR_KEY", "BASE_S1_NULL_KEY", "BASE_D2_NULL_KEY",
]


# =====================================================================================
# Constants
# =====================================================================================

# Artefact file names. Spec section 13 names all five; `cis_transcripts.jsonl` and
# `D2_transcripts.jsonl` are the two the spec flags as REQUIRED. v1 stored only the D2 rate
# and 29 of 30 cells read exactly 0.00 or 1.00 with nothing behind them, so not one of those
# cells could be audited afterwards. Section 9.2's primary control (the D4 failure-mode
# distribution) cannot run at all without the transcripts.
JUDGE_A1_FILE: str = "judge_e5.jsonl"
JUDGE_A2_FILE: str = "judge_s1.jsonl"
JUDGE_B_FILE: str = "judge_d2.jsonl"
CIS_FILE: str = "cis_transcripts.jsonl"
D2_FILE: str = "D2_transcripts.jsonl"

# `judge_id` is load-bearing in the judge cache key (CONTRACT defence 1): E5 and S1 score
# the same (phase, layer, r, prompt_id) and without it S1 returns E5's cached row, making
# every S1 a copy of E5/10 - a wrong number, not an error. These three strings are the only
# values this module ever passes.
JUDGE_E5_ID: str = "E5"
JUDGE_S1_ID: str = "S1"
JUDGE_D2_ID: str = "D2"

# Generation batch width. v1 ran forced-ID at 25 (bug 25's fix, validated on the pod at
# 20.6 s/cell), so 25 is the widest batch this model is known to sustain with a steering
# hook attached. Phase 6 asks for N_CONFIRM = 100 forced-ID trials; issuing those as one
# batch of 100 would be an untested VRAM footprint on a measurement that only runs once per
# concept. Chunking is scientifically neutral here precisely BECAUSE the multi-steering path
# corrects each row's start position for its own padding: chunk composition changes the
# padding width and nothing else.
GEN_BATCH_MAX: int = 25

# Judge D2's Failure_Mode vocabulary (spec 6.3), plus one slot of our own.
#
# `unrecognised` is not in the spec's list: it is where a parsed-but-unexpected mode string
# goes. A judge that answers off-menu must be visible in the distribution rather than
# quietly dropped or coerced into one of the real modes - D4 IS the section 9.2 control, so
# a silently reshaped distribution is a silently wrong control verdict.
RETRIEVAL_MODES: tuple[str, ...] = ("wrong_concept", "vague")
DAMAGE_MODES: tuple[str, ...] = ("degenerate", "saturated", "empty")
FAILURE_MODES: tuple[str, ...] = ("n/a",) + RETRIEVAL_MODES + DAMAGE_MODES + ("unrecognised",)

# Where the Phase 0 unsteered completions live on RUN.base. Spec 5.1 step 4: three unsteered
# completions per E5 prompt; sample 1 is the paired reference A used by every E5 and S1 call,
# samples 2-3 supply the control pairs of 5.8.
BASE_UNSTEERED_KEY: str = "unsteered"
BASE_FPR_KEY: str = "judge_fpr"
BASE_S1_NULL_KEY: str = "judge_s1_null"
BASE_D2_NULL_KEY: str = "judge_d2_null"

# The two sentinels that stand in for the model-authored text in the S1 blindness frame.
# Chosen to contain no English word that could be a concept, so the frame check can never
# fail on the sentinel itself.
_A2_FRAME_A: str = "<<< reference response elided for the blindness check >>>"
_A2_FRAME_B: str = "<<< judged response elided for the blindness check >>>"


# =====================================================================================
# Phase label
# =====================================================================================
# The first element of the judge cache key (CONTRACT section 3). It is a LABEL: it names
# which phase paid for a judge call, so judge_e5.jsonl can be read per phase and so a Phase 6
# confirmation is never confused with the Phase 4 screening run that chose the cell.
#
# It is deliberately NOT the thing that keeps two different generations apart. That job
# belongs to the payload fingerprint in `_cache_key` below, because generations are sampled
# at temperature 1.0 and re-running a cell produces different text under the same
# (phase, layer, r, trial). So a mislabelled phase costs a mislabelled row, never a wrong
# number - which is why this one is allowed to have a default at all (DEBUG LOG pattern 4 is
# about defaults that become NUMBERS).

_PHASE: str = "VERIFY"


def current_phase() -> str:
    """The phase label currently stamped on judge calls and transcript rows."""
    return _PHASE


@contextmanager
def phase_scope(name: str) -> Iterator[str]:
    """Label every judge call and transcript row made inside the block with `name`.

    Used by `phases.py` so that Phase 4, Phase 5, Phase 6 and the controls each own their
    rows in judge_e5.jsonl / judge_d2.jsonl without every measure signature growing a
    `phase=` argument.
    """
    global _PHASE
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"phase label must be a non-empty string, got {name!r}")
    previous = _PHASE
    _PHASE = name.strip()
    try:
        yield _PHASE
    finally:
        _PHASE = previous


def _phase_of(explicit: str | None) -> str:
    return _PHASE if explicit is None else str(explicit)


# =====================================================================================
# Run context, stamping, and JSONL append
# =====================================================================================

def _run() -> Any:
    """The process-global RunContext, or a loud error.

    Read through this accessor rather than `from .config import RUN`: `model.load_model`
    rebinds the attribute on the config module, and a from-import would capture the pre-load
    placeholder forever (bug 23's family - a stale reference that returns something
    plausible instead of raising).
    """
    run = getattr(config, "RUN", None)
    if run is None or run.mw is None:
        raise RuntimeError(
            "m2.config.RUN is not loaded - call m2.model.load_model(CONFIG) and "
            "m2.driver.set_concept(name) before any expensive measurement.")
    return run


def _concept() -> str:
    """The concept currently under measurement. Hard read: no concept, no judge call.

    Every E5 and B payload names the concept, so a `None` here would send the literal string
    'None' to the judge and score influence toward it.
    """
    run = _run()
    if not run.concept:
        raise RuntimeError(
            "RUN.concept is not set - call m2.driver.set_concept(name) first. Judge E5 and "
            "Judge D2 both name the concept in their payload; there is no unnamed variant.")
    return str(run.concept)


def _cfg(key: str) -> Any:
    """One CONFIG value, hard-indexed. A missing knob must raise, never default."""
    return _run().config[key]


def _d2_null_fields() -> dict:
    """The report-only unsteered baseline copied beside every real D2 reading."""
    detail = (_run().base or {}).get(BASE_D2_NULL_KEY)
    if detail is None:
        return dict(d2_null=None, d2_null_ci_low=None, d2_null_ci_high=None,
                    d2_null_n=None, d2_null_reference=float(_cfg("D2_NULL_REFERENCE")))
    return {key: detail[key] for key in
            ("d2_null", "d2_null_ci_low", "d2_null_ci_high", "d2_null_n",
             "d2_null_reference")}


def _now() -> str:
    """UTC timestamp for the `ts` field every row carries (CONTRACT section 4)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp(row: dict) -> dict:
    """Add concept / config_hash / ts. Hard indexing: an unidentified row is worthless."""
    run = _run()
    out = dict(row)
    out["concept"] = run.concept
    out["config_hash"] = run.config["config_hash"]
    out["ts"] = _now()
    return out


def _append_row(name: str, row: dict) -> Path:
    """Append one JSON object to `run_dir/<name>`.

    `runio.write_row` does exactly this, but `runio` comes AFTER this module in the CONTRACT
    dependency order, so - as `vectors.py` already does - the row shape is matched rather
    than the function reused. `default=str` so a stray tensor or Path in a diagnostic field
    cannot lose a whole transcript to a serialisation error; the load-bearing fields are all
    primitives by construction.
    """
    run = _run()
    if run.run_dir is None:
        raise RuntimeError(
            "RUN.run_dir is not set - call m2.driver.set_concept(name) before measuring. "
            "Writing transcripts into the process working directory is how two concepts' "
            "rows end up in one file.")
    path = Path(run.run_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_stamp(row), ensure_ascii=False, default=str) + "\n")
    return path


# =====================================================================================
# Vector, dose and cache identity
# =====================================================================================

def _resolve_vector(layer: int, alpha: float, vec: Any, r: float | None) -> tuple[Any, float, str]:
    """`(vector, r, vec_fingerprint)` for a cell, given alpha and an optional vector override.

    Two cases, and conflating them is a trap worth naming:

      vec is None   The concept's own vector at this layer. `r` is then DERIVED from alpha
                    through `config.dose_for`, which uses ||v_L|| for that same vector - so
                    the derived dose is exact.

      vec given     A foreign direction: section 9.1's random unit vector, or section 9.2's
                    control concept. `config.dose_for` would divide by the CONCEPT vector's
                    norm and return a dose the model was never steered at, so `r` must be
                    supplied by the caller and this function refuses to guess it.

    `vec_fingerprint` is the content hash from `vectors.py` - the same function the judge
    cache key uses, never a second implementation (bug 23, CONTRACT defence 1).
    """
    run = _run()
    layer = int(layer)
    alpha = float(alpha)
    if vec is None:
        # Hard index: a cell whose vector was never extracted is a bug, not a cell to skip.
        vector = run.vecs[layer]
        dose = float(config.dose_for(layer, alpha)) if r is None else float(r)
    else:
        vector = vec
        if r is None:
            raise ValueError(
                f"a vector override was passed for L{layer} without an explicit r. The dose "
                "map divides by ||v_L|| of the CONCEPT vector, so deriving r from a foreign "
                "direction's alpha would label the row with a dose the model never saw "
                "(spec section 9.1 matches on r, not alpha).")
        dose = float(r)
    return vector, dose, vectors.vec_fingerprint(vector)


def _payload_fp(payload: str) -> str:
    """Content fingerprint of one rendered judge payload."""
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _cache_key(phase: str, layer: int | None, r: float | None, unit_id: str,
               judge_id: str, vec_fp: str, payload: str) -> tuple:
    """The CONTRACT section 3 cache key: `(phase, layer, r, prompt_id, judge_id, vec_fp)`.

    Shape and field order exactly as contracted. The one thing worth reading closely is what
    goes in the `prompt_id` slot: `"<unit>@<payload fingerprint>"` rather than the bare unit
    id.

    DEBUG LOG pattern 7 - a cache key must contain everything the value depends on. Every
    payload in this module embeds a GENERATION, and generations are sampled at
    TEMPERATURE = 1.0. Re-running a cell (a crash and resume, gate 8's deliberate
    re-judging, Phase 6 re-measuring the winner) produces different text under an identical
    (phase, layer, r, trial, judge_id, vec_fingerprint). Without the fingerprint the cache
    would hand back the previous run's verdict on a response the judge never saw - a
    plausible wrong number, which is the exact class this pipeline is built to exclude.

    With it, two identical payloads share a cache entry (the same question to the judge) and
    two different payloads never collide. The cost is that a resumed cell re-judges rather
    than reusing; at well under a dollar per concept (spec section 12) that is the right side
    of the trade.
    """
    return (phase, layer, r, f"{unit_id}@{_payload_fp(payload)}", judge_id, vec_fp)


# =====================================================================================
# Judge templates
# =====================================================================================

def _render(template: str, name: str, **fields: Any) -> str:
    """`template.format(**fields)` with a failure that names the template and the fields.

    A KeyError out of `str.format` says only which placeholder was missing. Since the
    templates live in `judges.py` and the fields are built here, the useful message is both
    halves at once - this is the seam where a renamed placeholder shows up.
    """
    try:
        return template.format(**fields)
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"could not render {name}: {type(exc).__name__} {exc}. This module supplies "
            f"{sorted(fields)}; the template in m2.judges expects something else. Read the "
            "template rather than adding a placeholder here.") from exc


def assert_templates_sane() -> None:
    """Structural checks on the three judge templates. Called at import of this module.

    Gate 2's whole design is that S1 cannot be influenced by the concept because it is never
    told it (spec 5.7: "Separation must be structural, not instructed"). A `{concept}`
    placeholder in JUDGE_S1 would make that false the moment somebody filled it in, so its
    absence is checked here rather than trusted - and checked at import, so the failure lands
    at setup and not forty minutes into a sweep (DEBUG LOG pattern 8).
    """
    if "{concept}" in judges.JUDGE_S1:
        raise RuntimeError(
            "m2.judges.JUDGE_S1 contains a {concept} placeholder. Judge S1 scores integrity "
            "and must never learn the injected concept (spec 5.7, gate 2a); a template with "
            "that placeholder makes the blindness a matter of who calls it.")
    for name, template, needed in (
        ("JUDGE_E5", judges.JUDGE_E5,
         ("{concept}", "{prompt}", "{response_unsteered}", "{response_steered}")),
        ("JUDGE_S1", judges.JUDGE_S1,
         ("{prompt}", "{response_unsteered}", "{response_steered}")),
        ("JUDGE_D2", judges.JUDGE_D2, ("{concept}", "{response}")),
    ):
        missing = [f for f in needed if f not in template]
        if missing:
            raise RuntimeError(
                f"m2.judges.{name} is missing the placeholders {missing}. This module fills "
                f"{list(needed)} (spec section 6); a template without one of them would be "
                "judged with that field silently absent.")


assert_templates_sane()


def s1_blind_frame(task_prompt: str) -> str:
    """The S1 payload with both model-authored responses replaced by sentinels.

    **This, and not the full payload, is what `judges.assert_s1_blind` is asserted on, and
    the reason is worth spelling out because the naive reading of gate 2a is impossible.**

    Gate 2a reads "the S1 payload contains the concept string zero times - assert it, do not
    eyeball it". Taken literally over the whole payload that assertion fires on every cell
    where the steering worked: a strongly influenced response B says "velocity" repeatedly,
    and that is E5's signal, not a leak. The pipeline would die exactly when the science
    succeeded.

    What can honestly be asserted is that the concept appears nowhere in the part WE author:
    the instructions, the rubric, the calibration example, the field names, and the task
    prompt. Spec 6.2 says precisely that - "Not in the instructions, not in the rubric, not
    in the calibration example, not in a field name" - and lists the model-authored fields
    separately as what S1 receives. This frame is that payload with the two model-authored
    fields elided, so the check is over exactly the authored surface and cannot be broken by
    the model's own words.

    The task prompt IS included: it is ours, and a concept that appears in it (concept
    "Water" against the E5 prompt "Tell me a fact related to water") would genuinely
    un-blind S1 for that prompt. That must raise.

    The count of concept mentions in response B is still recorded on every judge_s1.jsonl
    row (`concept_mentions_in_b`) so gate 2 can report the real number rather than assert a
    zero that cannot exist.
    """
    return _render(judges.JUDGE_S1, "JUDGE_S1", prompt=task_prompt,
                   response_unsteered=_A2_FRAME_A, response_steered=_A2_FRAME_B)


def _count_mentions(text: str, needle: str) -> int:
    """Case-insensitive occurrences of `needle` in `text`. Diagnostic only."""
    if not needle:
        return 0
    return str(text).lower().count(needle.lower())


# =====================================================================================
# Judge transport adapters
# =====================================================================================
# `judges.call_judge` / `judges.judge_many` return dicts whose exact key names are not fixed
# by the CONTRACT. These two adapters are the only place that shape is read. Neither
# defaults: each tries the plausible names and RAISES naming all of them if none is present,
# so a transport change is a loud failure at the first judge call rather than a run of empty
# strings parsed into zeros (DEBUG LOG pattern 4).

_TEXT_KEYS: tuple[str, ...] = ("text", "raw", "raw_response", "response", "content")
_ERROR_KEYS: tuple[str, ...] = ("error", "exception", "err")


def _judge_error(result: dict) -> str | None:
    """A transport-level failure label, or None. Presence test, not a default."""
    for key in _ERROR_KEYS:
        if key in result and result[key]:
            return f"{key}:{str(result[key])[:200]}"
    return None


def _judge_text(result: dict, judge_id: str) -> str:
    """The judge's raw reply text."""
    for key in _TEXT_KEYS:
        if key in result:
            return str(result[key])
    raise KeyError(
        f"judge {judge_id} result carries none of {list(_TEXT_KEYS)}; its keys are "
        f"{sorted(result)}. m2.judges' return shape has changed - read call_judge before "
        "adding a key here.")


def _issue(items: list[dict]) -> list[dict]:
    """Send a batch of judge items concurrently and return the results in input order.

    ONE call, so that E5, S1 and B all fly at once. Spec 5.7 is explicit that the second
    judge costs tokens and not wall time only because E5 and S1 are independent and issue
    concurrently; two sequential `judge_many` calls would spend the wall time anyway.

    Order is the association between a result and the response it scored, so it is checked
    where it can be: if the results echo their `cache_key`, every one is compared against the
    item that asked for it. If they do not, input order is all there is - recorded here as an
    assumption on `judges.judge_many` rather than left implicit.
    """
    if not items:
        return []
    results = judges.judge_many(items, int(_cfg("judge_concurrent")))
    if len(results) != len(items):
        raise RuntimeError(
            f"judge_many returned {len(results)} results for {len(items)} items. Results are "
            "matched to responses by position; a length mismatch means that association is "
            "gone, and a judged response cannot be re-attached by guessing.")
    for item, result in zip(items, results):
        if isinstance(result, dict) and "cache_key" in result:
            if tuple(result["cache_key"]) != tuple(item["cache_key"]):
                raise RuntimeError(
                    "judge_many returned results out of order: item "
                    f"{item['cache_key']!r} came back as {result['cache_key']!r}. Every "
                    "score would be attached to the wrong response.")
    return list(results)


def _item(unit: dict) -> dict:
    """One work item for `judges.judge_many`, in the shape `call_judge` takes.

    **`model_text` is not optional in practice, whatever its default says.** `judge_many`
    re-runs gate 2(a) on every S1 item as a second line of defence, and its default is to
    exclude nothing - the strict direction, so a caller who forgets cannot silently disable
    the check. This builder used to send `{prompt, judge_id, cache_key}` and nothing else, so
    that second check ran over the whole payload including the model's own words. Garlic at a
    saturated cell answered "Garlic Garlic Garlic ..."; the check saw the concept in the
    payload, could not tell it came from the model, and killed Phase 4 at the first cell.

    The spans are read off the unit rather than passed in, because the unit is what carries
    the exact strings that were rendered into the payload - a span assembled anywhere else
    could differ by a strip() and silently fail to match.
    """
    item = dict(prompt=unit["payload"], judge_id=unit["judge_id"],
                cache_key=unit["cache_key"])
    if "concept" in unit:
        item["concept"] = unit["concept"]
    spans = tuple(str(unit[key]) for key in ("response_unsteered", "response_steered",
                                             "response")
                  if key in unit and unit[key])
    if spans:
        item["model_text"] = spans
    return item


def _parse_or_fail(result: dict, judge_id: str, parser: Any) -> tuple[dict | None, str, str | None]:
    """`(parsed, raw_text, failure_label)`.

    A judge failure is any of: a transport error, a reply the parser rejects, or v1's
    `ERROR:` sentinel (lab_cells.py cell 28 recognised exactly that prefix). All three are
    reported and excluded from their denominator; none of them is ever scored as a zero or as
    a non-identification, because "the judge did not answer" and "the judge answered no" are
    different facts and only one of them is evidence (spec 5.9).
    """
    err = _judge_error(result)
    try:
        raw = _judge_text(result, judge_id)
    except KeyError:
        if err is not None:
            return None, "", err
        raise
    if err is not None:
        return None, raw, err
    if raw.startswith("ERROR:"):
        return None, raw, f"transport:{raw[:120]}"
    try:
        parsed = parser(raw)
    except Exception as exc:                      # noqa: BLE001 - reported, never swallowed
        # Parsers RAISE on a missing field by contract (defence 11). Catching here converts
        # that into an excluded unit with a label, which is what the metric needs; it never
        # converts it into a score.
        return None, raw, f"parse:{type(exc).__name__}:{str(exc)[:120]}"
    return parsed, raw, None


def _field(parsed: dict, name: str, judge_id: str) -> Any:
    """One parsed field, hard-indexed.

    No `.get(name, 0.0)`. A parser that stopped returning `Score_Influence` must break E5,
    not set it to zero at every cell - which is what a defaulted read looks like from the
    outside, and is indistinguishable from a genuinely inert layer (DEBUG LOG pattern 4).
    """
    if name not in parsed:
        raise KeyError(
            f"judge {judge_id} parse result has no {name!r}; it has {sorted(parsed)}. "
            "m2.judges' parser and this module disagree about the field name.")
    return parsed[name]


def _score_0_10(value: Any, name: str, judge_id: str) -> float:
    """A 0-10 judge score as a float, range-checked.

    Out of range means the judge answered off-rubric or the parser mis-read it; either way
    the number is not on the scale E5_FLOOR and S4_MIN are defined against, so it must not
    silently enter a mean.
    """
    score = float(value)
    if not math.isfinite(score) or not (0.0 <= score <= 10.0):
        raise ValueError(f"judge {judge_id} returned {name}={value!r}, outside the 0-10 scale")
    return score


# =====================================================================================
# Prompt rendering and the unsteered reference
# =====================================================================================

def task_batch(prompt_rows: Sequence[dict]) -> tuple[list[str], list[int], list[str]]:
    """Render a prompt set for generation: `(prompts, start_positions, ids)`.

    One place that renders E5_PROMPTS / E5_HELDOUT, so the steered generation, the unsteered
    reference A, and the judge payload all describe the same rendered prompt. Two renderings
    of "the same" prompt is how a paired comparison quietly stops being paired.

    The start position is per row and is derived from that row's own question, leaving the
    chat template unsteered exactly as Macar's detection test does (`steering_utils.py:618`).
    A row whose needle cannot be found RAISES: `model.start_pos_for` returns None there, and
    the two things a caller could do with a None - steer from 0, or steer everywhere - are
    both bug 8 (detection and effectiveness measured under different interventions).
    """
    rendered: list[str] = []
    starts: list[int] = []
    ids: list[str] = []
    for row in prompt_rows:
        # Hard indexing: prompts.py asserts this row shape at import, so a missing key here
        # means the caller passed something that is not a prompt row.
        pid, text = str(row["id"]), str(row["text"])
        prompt = model.chat(text)
        start = model.start_pos_for(prompt, text)
        if start is None:
            raise RuntimeError(
                f"prompt {pid!r}: its own text is not findable in its rendered chat template, "
                "so there is no steering start position. Steering from 0 or steering "
                "everywhere would both steer the framing, which is not the intervention the "
                "detection test applies (bug 8).")
        rendered.append(prompt)
        starts.append(int(start))
        ids.append(pid)
    return rendered, starts, ids


def unsteered_samples(prompt_id: str) -> list[str]:
    """The Phase 0 unsteered completions for one prompt, `[0]` being the reference A.

    **Shape other modules must honour:** `RUN.base["unsteered"]` is
    `dict[prompt_id, list[str]]`, sample 1 first - spec 5.1 step 4, "Generate 3 unsteered
    completions per E5 prompt. Sample 1 becomes the paired reference A used by every Judge E5
    and S1 call; samples 2-3 supply the control pairs". A bare string is accepted as a
    one-sample list, because a Phase 0 that stored only the reference is a legible thing to
    have done; anything else raises with the expected shape named.
    """
    run = _run()
    if BASE_UNSTEERED_KEY not in run.base:
        raise RuntimeError(
            f"RUN.base[{BASE_UNSTEERED_KEY!r}] is missing: Phase 0 (spec 5.1 step 4) has not "
            "generated the unsteered completions, so there is no reference A to pair against. "
            "Every Judge E5 and S1 call is paired (spec 5.6) - there is no unpaired variant.")
    store = run.base[BASE_UNSTEERED_KEY]
    if prompt_id not in store:
        raise KeyError(
            f"no unsteered baseline for prompt {prompt_id!r}; Phase 0 recorded "
            f"{sorted(store)}. If this is a Phase 6 held-out prompt, Phase 6 must generate "
            "baselines for E5_HELDOUT before measuring on it.")
    samples = store[prompt_id]
    if isinstance(samples, str):
        return [samples]
    samples = list(samples)
    if not samples:
        raise ValueError(f"unsteered baseline for {prompt_id!r} is empty")
    return [str(s) for s in samples]


def reference_A(prompt_id: str) -> str:
    """The paired unsteered reference A for one prompt (sample 1 of spec 5.1 step 4)."""
    return unsteered_samples(prompt_id)[0]


# =====================================================================================
# Generation
# =====================================================================================

def _check_start_positions(prompts: Sequence[str], starts: Sequence[int]) -> None:
    """Every start position must be a real index inside its own row. Raises otherwise.

    BUG 26, on the batched path. `model.injected` raises when `start_pos >= seq_len` because
    a hook that steers nothing while reporting success is how 30 cells read exactly 0.000 for
    an hour. The repo's multi-steering hook has no such guard: its mask is
    `pos_range >= steering_pos` (model_utils.py:1211), so an out-of-range start simply makes
    the mask all-False for that row and the prompt is processed unsteered - while the
    generation phase (seq_len == 1, applied unconditionally at :1200) still steers. That is a
    partially unsteered row with no error anywhere.

    Lengths are measured with add_special_tokens=False (bug 9): these prompts came through
    `apply_chat_template`, which has already emitted <bos>.
    """
    if len(starts) != len(prompts):
        raise ValueError(
            f"{len(starts)} start positions for {len(prompts)} prompts - the multi-steering "
            "path takes one per row (model_utils.py:1167) and there is no sensible fill.")
    tok = _run().tok
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    for i, (start, length) in enumerate(zip(starts, lengths)):
        if start is None:
            raise ValueError(
                f"row {i}: start position is None. Passing None to the multi-steering path "
                "steers ALL positions including the chat template, which is bug 8's shape.")
        if int(start) < 0 or int(start) >= int(length):
            raise ValueError(
                f"row {i}: start position {start} is outside its prompt's {length} tokens, so "
                "the prompt would be processed unsteered while its generation was steered. "
                "The repo's multi-steering hook does not raise on this (bug 26's shape).")


def _generate(prompts: Sequence[str], layer: int, vector: Any, strength: float,
              starts: Sequence[int], max_new_tokens: int, temperature: float) -> list[str]:
    """One or more `generate_batch_with_multi_steering` calls, chunked, order preserved."""
    run = _run()
    prompts = list(prompts)
    starts = [int(s) for s in starts]
    out: list[str] = []
    for lo in range(0, len(prompts), GEN_BATCH_MAX):
        chunk = prompts[lo:lo + GEN_BATCH_MAX]
        chunk_starts = starts[lo:lo + GEN_BATCH_MAX]
        got = run.mw.generate_batch_with_multi_steering(
            prompts=chunk,
            layer_idx=int(layer),
            # The same vector repeated, one per row. 25 x 5376 x 2 bytes is under 300KB -
            # the repetition is what buys the correct per-row padding handling below.
            steering_vectors=[vector] * len(chunk),
            strength=float(strength),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            steering_start_positions=list(chunk_starts),
        )
        if len(got) != len(chunk):
            raise RuntimeError(
                f"generate_batch_with_multi_steering returned {len(got)} responses for "
                f"{len(chunk)} prompts; responses are matched to prompts by position.")
        out.extend(str(g) for g in got)
    return out


def generate_steered(prompts: list[str], layer: int, alpha: float, max_new_tokens: int,
                     temperature: float, *, start_positions: Sequence[int],
                     vec: Any = None) -> list[str]:
    """Batched steered generation. **`generate_batch_with_multi_steering`, never the obvious one.**

    Uses `generate_batch_with_multi_steering` with the vector repeated `len(prompts)` times,
    even though every prompt gets the same vector. The reason is left padding
    (model_utils.py:125) and it is worth spelling out, because both failures are silent and
    this is bug 25b.

    Prompts differ in length - "Trial 1" is one token shorter than "Trial 25", and the E5 set
    ranges from "What is 17 x 23?" to a full sentence. Left padding puts that difference at
    the FRONT, so every content token in a shorter prompt sits one or more indices later than
    in the longest one. `generate_batch_with_steering`, the natural choice and the one D1
    uses, mishandles that in two independent places:

      steering  It applies one scalar start position to the whole batch
                (model_utils.py:1028). Its left-padding correction at :1061 only fires when
                `steering_pos_tensor` is set, which in that function is never - the
                single-vector branch always leaves it None. So shorter prompts are steered
                from one token too early.

      decoding  It then slices the response at the UNPADDED length
                (`output_ids[i][input_length:]`, :1096) while `generate()` returns the padded
                input plus the generation, so shorter prompts carry their last prompt tokens
                into the "generated" text. For D1 that overhang comes from
                `<start_of_turn>model` and is removed by the explicit Gemma strip just below
                the slice - which is why the M1.5 rig check came back clean at 0.383. **Our
                forced-ID prompt ends with the PREFILL**, so its overhang would be `" about"`
                and nothing strips it.

    `generate_batch_with_multi_steering` fixes both: it corrects each row's start position for
    its own padding (:1248) and applies a per-row mask, and its decoder falls through to
    slicing at the padded width (:1321). That makes this exactly equivalent to Macar's serial
    path, which is what D2 must match - his forced-noticing is serial, so it has a batch of
    one and no padding to get wrong.

    `start_positions` is keyword-only and REQUIRED, one per prompt. There is no default,
    because both possible defaults are wrong: `None` steers the chat template as well (bug 8),
    and a scalar is bug 25b's steering half restated. `_check_start_positions` then proves
    each one is a real index inside its own row (bug 26 on the batched path).

    `vec` overrides the concept vector for section 9.1's random direction and section 9.2's
    control concept; it defaults to `RUN.vecs[layer]`, which is a hard index and raises when
    the layer was never extracted.
    """
    prompts = list(prompts)
    if not prompts:
        raise ValueError("generate_steered() was given no prompts")
    if not float(alpha):
        # An alpha of zero with a steering hook attached is an unsteered generation wearing a
        # steered label - the shape of bug 26, where an hour of measurement read exactly zero
        # and every gate was satisfied. Unsteered generation has its own function.
        raise ValueError(
            "generate_steered() called with alpha=0. That is an unsteered generation; call "
            "generate_unsteered() so the row cannot be recorded as steered.")
    # Hard index when there is no override: a cell whose vector was never extracted is a bug,
    # not a cell to skip. No dose is computed here - generation does not need `r`, and asking
    # `_resolve_vector` for one would force a caller with an override to invent a dose it has
    # no use for.
    vector = _run().vecs[int(layer)] if vec is None else vec
    _check_start_positions(prompts, start_positions)
    return _generate(prompts, int(layer), vector, float(alpha), start_positions,
                     int(max_new_tokens), float(temperature))


def generate_unsteered(prompts: list[str], max_new_tokens: int, temperature: float, *,
                       start_positions: Sequence[int]) -> list[str]:
    """Batched UNSTEERED generation, for the spec 5.1 step 4 baselines. Not `generate_batch`.

    **Do not use `mw.generate_batch` for prompts of unequal length.** It carries bug 25b's
    decoding half verbatim: `new_tokens = output_ids[i][input_length:]` (model_utils.py:960)
    slices at the UNPADDED length while `generate()` returns the padded input plus the
    generation, so every row shorter than the longest in its batch has prompt tokens
    prepended to its "response". v1 survived that because its batched unsteered block used
    the introspection prompts, which differ only by a trial number - one token of overhang,
    from the template, removed by the Gemma `model\\n` strip. The twelve E5 prompts differ by
    many tokens, and the corrupted text would become the reference A that EVERY Judge E5 and
    S1 call in the entire run is paired against.

    So the baselines go down the same multi-steering path as everything else, at strength 0
    with a zero vector. The decoder there slices at the padded width (:1321) and is correct
    for every row. Adding an exactly-zero vector is exactly a no-op in floating point - it is
    asserted below rather than assumed - so this is a genuinely unsteered generation that
    merely borrows the correct decoder.
    """
    prompts = list(prompts)
    if not prompts:
        raise ValueError("generate_unsteered() was given no prompts")
    run = _run()
    zero = torch.zeros(int(run.mw.d_model))
    if bool(zero.any()):                       # cheap, and it is the whole claim of this path
        raise RuntimeError("the zero vector is not zero - refusing to call this unsteered")
    _check_start_positions(prompts, start_positions)
    # layer 0 and strength 0: the hook fires, adds zero, and changes nothing. The start
    # positions are passed through unchanged so this call takes byte-for-byte the same code
    # path as the steered one.
    return _generate(prompts, 0, zero, 0.0, start_positions,
                     int(max_new_tokens), float(temperature))


# =====================================================================================
# 5.6 - E5, Judge E5
# =====================================================================================

def _e5_units(prompt_ids: Sequence[str], task_texts: Sequence[str], responses: Sequence[str],
              *, concept: str, phase: str, layer: int | None, r: float | None,
              vec_fp: str) -> list[dict]:
    """One E5 unit per prompt: payload, cache key, and the fields its row needs."""
    units: list[dict] = []
    for pid, text, steered in zip(prompt_ids, task_texts, responses):
        baseline = reference_A(pid)
        # Responses are NOT truncated. MAX_NEW_TOKENS is 100, so they are short by
        # construction, and silently trimming what the judge sees would change the
        # measurement without changing its name.
        payload = _render(judges.JUDGE_E5, "JUDGE_E5", concept=concept, prompt=text,
                          response_unsteered=baseline, response_steered=steered)
        units.append(dict(
            unit_id=pid, task_prompt=text, response_unsteered=baseline,
            response_steered=steered, payload=payload, judge_id=JUDGE_E5_ID,
            cache_key=_cache_key(phase, layer, r, pid, JUDGE_E5_ID, vec_fp, payload),
        ))
    return units


def _score_e5(units: Sequence[dict], results: Sequence[dict], ctx: dict) -> dict:
    """Parse the E5 replies, write judge_e5.jsonl, and aggregate into E5.

    Returns `e5` (mean Score_Influence), `e5_min` and `e5_se` across prompts - spec 5.6, and
    DEBUG LOG pattern 9 / CONTRACT defence 12: the SE is across PROMPTS, which is the only
    variance a per-cell judge mean has, and four v1 measures ran at n=1 with no way to tell a
    real effect from an artefact of one phrasing.
    """
    scores: list[float] = []
    per_prompt: list[dict] = []
    errors: list[dict] = []

    for unit, result in zip(units, results):
        parsed, raw, failure = _parse_or_fail(result, JUDGE_E5_ID, judges.parse_e5)
        row = dict(
            measure="E5", judge_id=JUDGE_E5_ID, phase=ctx["phase"], layer=ctx["layer"],
            r=ctx["r"], alpha=ctx["alpha"], vec_fingerprint=ctx["vec_fingerprint"],
            prompt_id=unit["unit_id"], cache_key=list(unit["cache_key"]),
            task_prompt=unit["task_prompt"],
            response_unsteered=unit["response_unsteered"],
            response_steered=unit["response_steered"],
            payload=unit["payload"], raw=raw, judge_error=failure,
        )
        if failure is None:
            score = _score_0_10(_field(parsed, "score_influence", JUDGE_E5_ID),
                                "score_influence", JUDGE_E5_ID)
            row.update(shift=_field(parsed, "shift", JUDGE_E5_ID),
                       form=_field(parsed, "form", JUDGE_E5_ID),
                       score_influence=score,
                       justification=_field(parsed, "justification", JUDGE_E5_ID))
            scores.append(score)
            per_prompt.append(dict(id=unit["unit_id"], score_influence=score,
                                   shift=row["shift"], form=row["form"]))
        else:
            errors.append(dict(id=unit["unit_id"], error=failure))
        _append_row(JUDGE_A1_FILE, row)

    if not scores:
        raise RuntimeError(
            f"E5 at L{ctx['layer']} r={ctx['r']}: all {len(units)} Judge E5 calls failed "
            f"({[e['error'] for e in errors][:3]}). A cell with no judge is not a cell with "
            "no influence - refusing to record one as the other.")

    mean, se, n = model.mean_se(scores)
    return dict(
        e5=mean, e5_min=min(scores), e5_max=max(scores), e5_se=se, e5_n=n,
        e5_judge_errors=len(errors), e5_judge_error_detail=errors,
        e5_per_prompt=per_prompt,
    )


def measure_E5(layer: int, alpha: float, *, prompt_set: Sequence[dict] | None = None,
               phase: str | None = None, r: float | None = None, vec: Any = None) -> dict:
    """Spec 5.6. E5 is THE effectiveness metric: Judge E5 over the E5 prompt set.

    One steered completion per prompt, then one Judge E5 call each against the cached
    unsteered reference A. Paired because the model has strong native habits - nearly every
    unsteered story in the M1.5 probes is the same lighthouse keeper - so an unpaired judge
    cannot separate a native tendency from an induced one, and abstract concepts (irony,
    karma, skepticism) influence manner with no word to count.

    Writes `judge_e5.jsonl` (every call: payload, raw reply, parsed fields) and
    `cis_transcripts.jsonl` (every steered task-prompt generation, spec 14.3 channel 2).

    `prompt_set` is E5_PROMPTS by default and E5_HELDOUT for Phase 6 - screening chose the
    operating point BY maximising E5 on E5_PROMPTS, so re-measuring on those same twelve is a
    fitted value, not a confirmation (spec section 8, Phase 6).

    Returns e5 / e5_min / e5_se, plus the `responses` list, so `measure_S1` and
    `cheap.measure_S2` can score the same generations rather than new ones. Note that
    `verify_cell`, not this function, is Phase 4/5's entry point: calling this and then
    `measure_S1` serialises the two judges and spends the wall time spec 5.7 says the split
    does not cost.
    """
    rows = list(prompt_assets.E5_PROMPTS if prompt_set is None else prompt_set)
    phase_label = _phase_of(phase)
    vector, dose, vec_fp = _resolve_vector(int(layer), float(alpha), vec, r)
    task_prompts, starts, ids = task_batch(rows)
    texts = [str(row["text"]) for row in rows]

    responses = generate_steered(
        task_prompts, int(layer), float(alpha),
        int(_cfg("MAX_NEW_TOKENS")), float(_cfg("TEMPERATURE")),
        start_positions=starts, vec=vector if vec is not None else None)

    _write_cis(rows, responses, layer=int(layer), alpha=float(alpha), r=dose,
               phase=phase_label, vec_fp=vec_fp)

    ctx = dict(phase=phase_label, layer=int(layer), r=dose, alpha=float(alpha),
               vec_fingerprint=vec_fp)
    units = _e5_units(ids, texts, responses, concept=_concept(), phase=phase_label,
                      layer=int(layer), r=dose, vec_fp=vec_fp)
    results = _issue([_item(u) for u in units])

    out = _score_e5(units, results, ctx)
    out.update(layer=int(layer), alpha=float(alpha), r=dose, phase=phase_label,
               vec_fingerprint=vec_fp, responses=responses, prompt_ids=ids,
               prompt_texts=texts)
    return out


def _write_cis(prompt_rows: Sequence[dict], responses: Sequence[str], *, layer: int,
               alpha: float, r: float, phase: str, vec_fp: str) -> None:
    """Append every steered task-prompt generation to `cis_transcripts.jsonl`.

    Written immediately after generation and BEFORE any judge call, so a judge outage does
    not cost the generations that the GPU has already paid for - and so the transcripts exist
    for the diagnosis if the judging is what failed. Spec 14.3 lists this file as part of the
    export bundle for exactly that reason: every diagnosis in the M1.5 review required reading
    generations.
    """
    for row, response in zip(prompt_rows, responses):
        _append_row(CIS_FILE, dict(
            measure="E5", phase=phase, layer=layer, alpha=alpha, r=r,
            prompt_id=str(row["id"]), kind=str(row["kind"]), prompt=str(row["text"]),
            response=str(response), vec_fingerprint=vec_fp,
        ))


# =====================================================================================
# 5.7 - S1, Judge S1
# =====================================================================================

def _s1_units(prompt_ids: Sequence[str], task_texts: Sequence[str], responses: Sequence[str],
              *, concept: str, phase: str, layer: int | None, r: float | None,
              vec_fp: str) -> list[dict]:
    """One S1 unit per prompt. The blindness of every payload is asserted here.

    `judges.assert_s1_blind` is called on the FRAME of each payload rather than on the
    payload itself - see `s1_blind_frame` for why the literal reading of gate 2a is
    impossible. The frame varies per prompt (it carries the task text), so this is a
    per-payload assertion and not a single check hoisted out of the loop.
    """
    units: list[dict] = []
    for pid, text, steered in zip(prompt_ids, task_texts, responses):
        baseline = reference_A(pid)
        frame = s1_blind_frame(text)
        judges.assert_s1_blind(frame, concept)
        payload = _render(judges.JUDGE_S1, "JUDGE_S1", prompt=text,
                          response_unsteered=baseline, response_steered=steered)
        units.append(dict(
            unit_id=pid, task_prompt=text, response_unsteered=baseline,
            response_steered=steered, payload=payload, judge_id=JUDGE_S1_ID,
            # Carried so `_item` can declare it to judge_many's second gate-2(a) check.
            # Without it that check sees the concept in the model's own words and raises.
            concept=concept,
            cache_key=_cache_key(phase, layer, r, pid, JUDGE_S1_ID, vec_fp, payload),
            # Gate 2's honest diagnostic: the concept CAN appear in B - that is E5's signal -
            # and the number is recorded so the gate can report it instead of asserting a
            # zero that a working cell cannot produce.
            concept_mentions_in_b=_count_mentions(steered, concept),
            concept_mentions_in_a=_count_mentions(baseline, concept),
        ))
    return units


def _score_s1(units: Sequence[dict], results: Sequence[dict], ctx: dict) -> dict:
    """Parse the S1 replies, write judge_s1.jsonl, and aggregate into S1 = mean/10."""
    scores: list[float] = []
    per_prompt: list[dict] = []
    errors: list[dict] = []

    for unit, result in zip(units, results):
        parsed, raw, failure = _parse_or_fail(result, JUDGE_S1_ID, judges.parse_s1)
        row = dict(
            measure=ctx.get("measure", "S1"), judge_id=JUDGE_S1_ID,
            phase=ctx["phase"], layer=ctx["layer"],
            r=ctx["r"], alpha=ctx["alpha"], vec_fingerprint=ctx["vec_fingerprint"],
            prompt_id=unit["unit_id"], cache_key=list(unit["cache_key"]),
            task_prompt=unit["task_prompt"],
            response_unsteered=unit["response_unsteered"],
            response_steered=unit["response_steered"],
            payload=unit["payload"], raw=raw, judge_error=failure,
            concept_mentions_in_b=unit["concept_mentions_in_b"],
            concept_mentions_in_a=unit["concept_mentions_in_a"],
            a2_blind=True,          # asserted in _s1_units before the call was built
        )
        if failure is None:
            score = _score_0_10(_field(parsed, "score_integrity", JUDGE_S1_ID),
                                "score_integrity", JUDGE_S1_ID)
            row.update(task_compliance=_field(parsed, "task_compliance", JUDGE_S1_ID),
                       semantic_coherence=_field(parsed, "semantic_coherence", JUDGE_S1_ID),
                       stance_stability=_field(parsed, "stance_stability", JUDGE_S1_ID),
                       degeneracy=_field(parsed, "degeneracy", JUDGE_S1_ID),
                       score_integrity=score,
                       justification=_field(parsed, "justification", JUDGE_S1_ID))
            scores.append(score)
            per_prompt.append(dict(id=unit["unit_id"], score_integrity=score,
                                   task_compliance=row["task_compliance"],
                                   semantic_coherence=row["semantic_coherence"],
                                   degeneracy=row["degeneracy"]))
        else:
            errors.append(dict(id=unit["unit_id"], error=failure))
        _append_row(JUDGE_A2_FILE, row)

    if not scores:
        raise RuntimeError(
            f"S1 at L{ctx['layer']} r={ctx['r']}: all {len(units)} Judge S1 calls failed "
            f"({[e['error'] for e in errors][:3]}). S1 is the primary sanity term; a cell "
            "with no integrity reading must not be scored as intact.")

    # S1 = mean(Score_Integrity) / 10, on 0-1 so that S4 = min(S1, S2, S3) compares like with
    # like (spec 5.7 and section 7).
    normalised = [s / 10.0 for s in scores]
    mean, se, n = model.mean_se(normalised)
    return dict(
        s1=mean, s1_min=min(normalised), s1_se=se, s1_n=n,
        s1_judge_errors=len(errors), s1_judge_error_detail=errors,
        s1_per_prompt=per_prompt,
    )


def measure_S1(layer: int, alpha: float, responses: Sequence[str], *,
               prompt_set: Sequence[dict] | None = None, phase: str | None = None,
               r: float | None = None, vec: Any = None) -> dict:
    """Spec 5.7. S1 is the primary sanity term: Judge S1 on the SAME responses E5 scored.

    Takes `responses` rather than generating, because E5 and S1 must score the same text: if
    each generated its own, influence and integrity would describe two different samples and
    the velocity L37 alpha=3.0 case - high influence AND low integrity on ONE response - could
    not be stated at all.

    The concept appears nowhere in the payload. That is structural, not instructed: v1 and the
    first M2 draft put both halves in a single call and asked the judge to hold them apart by
    instruction, which is the weakest available guarantee because the model scoring integrity
    had, in the same context, just been told the target concept and just scored how strongly it
    appeared. Withholding it makes a concept-driven integrity penalty impossible by
    construction and turns gate 2 into a property of the design (spec 5.7).

    Writes `judge_s1.jsonl`. Returns `s1` = mean(Score_Integrity)/10, with its SE and min.
    """
    rows = list(prompt_assets.E5_PROMPTS if prompt_set is None else prompt_set)
    responses = [str(x) for x in responses]
    if len(responses) != len(rows):
        raise ValueError(
            f"measure_S1 got {len(responses)} responses for {len(rows)} prompts. S1 must "
            "score exactly the responses E5 scored, in the same order - a length mismatch "
            "means they are not the same set.")
    phase_label = _phase_of(phase)
    _vector, dose, vec_fp = _resolve_vector(int(layer), float(alpha), vec, r)
    ids = [str(row["id"]) for row in rows]
    texts = [str(row["text"]) for row in rows]

    ctx = dict(phase=phase_label, layer=int(layer), r=dose, alpha=float(alpha),
               vec_fingerprint=vec_fp)
    units = _s1_units(ids, texts, responses, concept=_concept(), phase=phase_label,
                      layer=int(layer), r=dose, vec_fp=vec_fp)
    results = _issue([_item(u) for u in units])

    out = _score_s1(units, results, ctx)
    out.update(layer=int(layer), alpha=float(alpha), r=dose, phase=phase_label,
               vec_fingerprint=vec_fp)
    return out


# =====================================================================================
# 5.9 - D2 and D4, Judge D2
# =====================================================================================

def _d2_units(trials: Sequence[int], responses: Sequence[str], *, concept: str, phase: str,
             layer: int | None, r: float | None, vec_fp: str) -> list[dict]:
    """One Judge D2 unit per forced-ID trial."""
    units: list[dict] = []
    for trial, response in zip(trials, responses):
        payload = _render(judges.JUDGE_D2, "JUDGE_D2", concept=concept, response=response)
        unit_id = f"trial_{int(trial)}"
        units.append(dict(
            unit_id=unit_id, trial=int(trial), response=str(response), payload=payload,
            judge_id=JUDGE_D2_ID,
            cache_key=_cache_key(phase, layer, r, unit_id, JUDGE_D2_ID, vec_fp, payload),
        ))
    return units


def _d4_distribution(modes: Sequence[str]) -> dict:
    """The section 9.2 primary control: the failure-mode distribution over non-identified trials.

    Every key in FAILURE_MODES is present with an explicit 0.0 even when unobserved. A dict
    whose keys depend on what happened forces every consumer into `.get(mode, 0)`, and
    DEBUG LOG pattern 4 is that silent bugs cluster around exactly that - here it would be the
    difference between "no damage modes were seen" and "the damage modes are missing from this
    row", which is the whole reading of the control.

    `n/a` is excluded from the denominator: it is what Judge D2 answers when it DID identify
    the concept, so counting it would dilute the very distribution 9.2 reads.
    """
    failures = [m for m in modes if m != "n/a"]
    counts = {mode: 0 for mode in FAILURE_MODES}
    for mode in modes:
        counts[mode] = counts[mode] + 1
    n = len(failures)
    dist = {mode: (counts[mode] / n if n else 0.0) for mode in FAILURE_MODES if mode != "n/a"}
    damage = sum(dist[mode] for mode in DAMAGE_MODES)
    retrieval = sum(dist[mode] for mode in RETRIEVAL_MODES)
    mode_ci = ({mode: dict(zip(("low", "high"), cheap.wilson_interval(counts[mode], n)), n=n)
                for mode in FAILURE_MODES if mode != "n/a"} if n else {})
    damage_count = sum(counts[mode] for mode in DAMAGE_MODES)
    retrieval_count = sum(counts[mode] for mode in RETRIEVAL_MODES)
    damage_ci = cheap.wilson_interval(damage_count, n) if n else (None, None)
    retrieval_ci = cheap.wilson_interval(retrieval_count, n) if n else (None, None)
    dominant = max(dist, key=lambda m: dist[m]) if n else None
    return dict(d4=dist, d4_counts=counts, d4_n=n, d4_dominant=dominant,
                d4_ci=mode_ci,
                d4_damage_frac=damage, d4_damage_count=damage_count,
                d4_damage_frac_ci_low=damage_ci[0], d4_damage_frac_ci_high=damage_ci[1],
                d4_retrieval_frac=retrieval, d4_retrieval_count=retrieval_count,
                d4_retrieval_frac_ci_low=retrieval_ci[0],
                d4_retrieval_frac_ci_high=retrieval_ci[1],
                # The reading of the table in spec 9.2, computed once here so controls.py and
                # the batch driver's FATAL_CONSECUTIVE_D4S rule cannot disagree about it.
                d4_reading=("damage" if n and damage > retrieval else
                            ("retrieval" if n else None)))


def _score_d2(units: Sequence[dict], results: Sequence[dict], ctx: dict) -> dict:
    """Parse the Judge D2 replies, write judge_d2.jsonl and D2_transcripts.jsonl, aggregate D2/D4.

    `D2_transcripts.jsonl` is written for EVERY trial including the ones whose judge call
    failed, and it is written from a `finally`, so a judge outage halfway through cannot cost
    the generations. Spec 5.9 flags the file as a required change and gate 7 gates section
    9.2's primary control on it: v1 stored only the rate, 29 of 30 cells read exactly 0.00 or
    1.00, and not one of them could be audited afterwards.
    """
    identified = 0
    scored = 0
    modes: list[str] = []
    errors: list[dict] = []
    transcript_rows: list[dict] = []

    try:
        for unit, result in zip(units, results):
            parsed, raw, failure = _parse_or_fail(result, JUDGE_D2_ID, judges.parse_d2)
            measure = ctx.get("measure", "D2")
            judge_row = dict(
                measure=measure, judge_id=JUDGE_D2_ID, phase=ctx["phase"], layer=ctx["layer"],
                r=ctx["r"], alpha=ctx["alpha"], vec_fingerprint=ctx["vec_fingerprint"],
                trial=unit["trial"], cache_key=list(unit["cache_key"]),
                response=unit["response"], payload=unit["payload"], raw=raw,
                judge_error=failure,
            )
            transcript = dict(
                measure=measure, phase=ctx["phase"], layer=ctx["layer"], r=ctx["r"],
                alpha=ctx["alpha"], vec_fingerprint=ctx["vec_fingerprint"],
                trial=unit["trial"], response=unit["response"],
                identified=None, failure_mode=None, justification=None,
                judge_error=failure,
            )
            # Appended BEFORE the verdict is read, and mutated in place afterwards. A judge
            # whose reply violates the parser contract raises below, and the generation this
            # cell has already paid for must still reach D2_transcripts.jsonl (gate 7).
            transcript_rows.append(transcript)

            if failure is None:
                verdict = _field(parsed, "identified", JUDGE_D2_ID)
                if not isinstance(verdict, bool):
                    # parse_d2 is contracted to return a bool. A string "yes" counted as truthy
                    # would make every answer an identification, including "no".
                    raise TypeError(
                        f"parse_d2 returned identified={verdict!r} ({type(verdict).__name__}); "
                        "D2 counts identifications and needs a bool, not a truthy string.")
                mode = str(_field(parsed, "failure_mode", JUDGE_D2_ID)).strip().lower()
                unexpected = mode not in FAILURE_MODES
                if unexpected:
                    mode = "unrecognised"
                justification = _field(parsed, "justification", JUDGE_D2_ID)

                scored += 1
                identified += int(verdict)
                modes.append("n/a" if verdict else mode)
                judge_row.update(identified=verdict, failure_mode=mode,
                                 failure_mode_unexpected=unexpected,
                                 justification=justification)
                transcript.update(identified=verdict, failure_mode=mode,
                                  failure_mode_unexpected=unexpected,
                                  justification=justification)
            else:
                errors.append(dict(trial=unit["trial"], error=failure))

            _append_row(JUDGE_B_FILE, judge_row)
    finally:
        for row in transcript_rows:
            _append_row(D2_FILE, row)

    if scored == 0:
        raise RuntimeError(
            f"D2 at L{ctx['layer']} r={ctx['r']}: all {len(units)} Judge D2 calls failed "
            f"({[e['error'] for e in errors][:3]}). D2 is the selection criterion and the "
            "detection metric - an unjudged cell must not become a D2 of 0.00, which is "
            "precisely the reading the whole experiment is looking for.")

    d2 = identified / scored
    # Binomial SE on the SCORED denominator. Judge failures are excluded from it and reported
    # separately, never counted as non-identification (spec 5.9): "the judge did not answer"
    # and "the model did not name the concept" are different facts, and only the second is
    # evidence of covertness.
    d2_se = math.sqrt(d2 * (1.0 - d2) / scored)
    d2_ci_low, d2_ci_high = cheap.wilson_interval(identified, scored)

    out = dict(d2=d2, d2_se=d2_se, d2_ci_low=d2_ci_low, d2_ci_high=d2_ci_high,
               n_d2=scored, n=scored, d2_identified=identified,
               d2_judge_errors=len(errors), d2_judge_error_detail=errors,
               d2_n_generated=len(units))
    out.update(_d4_distribution(modes))
    return out


def measure_D2(layer: int, alpha: float, n: int, *, phase: str | None = None,
               r: float | None = None, vec: Any = None,
               trial_numbers: Sequence[int] | None = None) -> dict:
    """Spec 5.9. D2 is THE detection metric and the selection criterion; D4 rides along free.

    `n` forced-identification generations through `generate_steered` on
    `prompts.forced_prompts`, one Judge D2 call each.

        D2    = |{Identified = yes}| / N_scored
        d2_se = sqrt(D2(1-D2)/N_scored)
        D4    = the distribution over Failure_Mode

    D2 keeps its v1 name and meaning EXACTLY (spec 2.1), which is why the prompt comes from
    `prompts.forced_prompts` - a byte-identical port of the M1.5 lab's builder, gated by
    `prompts.verify_forced_prompts` (R7) against the repo's own function. One builder, so D3
    (which reads its concept mass off the same prompts) and D2 cannot drift apart.

    Writes `D2_transcripts.jsonl` and `judge_d2.jsonl`.

    `vec` and `r` together support the controls: section 9.1 injects a random unit direction
    at the same `r`, section 9.2 REPLACES the target with a control concept's vector at the
    same (L, r) - never stacks one on top, which would double the perturbation and lobotomise
    by construction.
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"measure_D2 needs at least one trial, got n={n}")
    trials = list(range(1, n + 1)) if trial_numbers is None else [int(t) for t in trial_numbers]
    if len(trials) != n:
        raise ValueError(f"{len(trials)} trial numbers given for n={n}")

    phase_label = _phase_of(phase)
    vector, dose, vec_fp = _resolve_vector(int(layer), float(alpha), vec, r)

    forced, start = prompt_assets.forced_prompts(trials)
    if start is None:
        # forced_prompts returns None only when "Trial <n>" is absent from its own rendered
        # template, which is a construction error. Steering from position 0 would steer the
        # interpretability-researcher framing as well, so detection would be measured under a
        # different intervention from everything else (bug 8).
        raise RuntimeError(
            "prompts.forced_prompts() could not locate the trial marker in its own prompt, so "
            "there is no steering start position. Run prompts.verify_forced_prompts() (R7) - "
            "D2 must not run on a prompt whose construction is unverified.")

    responses = generate_steered(
        forced, int(layer), float(alpha),
        int(_cfg("MAX_NEW_TOKENS")), float(_cfg("TEMPERATURE")),
        # One start position for the whole batch is EXACT here, not an approximation: the text
        # before "Trial" is identical for every trial number, so the repo's own per-prompt
        # computation returns the same value for all of them (R7 confirms this on one, one and
        # two digit trials). The multi-steering path then corrects each row for its own left
        # padding, which is what makes the batch equivalent to Macar's serial path (bug 25b).
        start_positions=[int(start)] * len(forced),
        vec=vector if vec is not None else None)

    ctx = dict(phase=phase_label, layer=int(layer), r=dose, alpha=float(alpha),
               vec_fingerprint=vec_fp)
    units = _d2_units(trials, responses, concept=_concept(), phase=phase_label,
                     layer=int(layer), r=dose, vec_fp=vec_fp)
    results = _issue([_item(u) for u in units])

    out = _score_d2(units, results, ctx)
    out.update(layer=int(layer), alpha=float(alpha), r=dose, phase=phase_label,
               vec_fingerprint=vec_fp, responses=responses, trials=trials)
    out.update(_d2_null_fields())
    return out


# =====================================================================================
# 5.8 - Judge E5 control pairs
# =====================================================================================

def _fpr_prompt_rows() -> list[dict]:
    """Every open E5 prompt, with CONFIG pinning the expected count.

    Open rather than verifiable, deliberately. Two unsteered samples of "What is 17 x 23?"
    are near-identical, so a judge scores them 0 without being tested at all; two unsteered
    samples of "Tell me a short story" differ substantially, which is exactly the condition
    under which a judge that expects to find influence invents some. The control is only worth
    its calls if it is the harder case.

    Derived from E5_PROMPTS rather than hard-coded ids, so editing the prompt set cannot leave
    this pointing at a prompt that no longer exists.
    """
    rows = [row for row in prompt_assets.E5_PROMPTS if row["kind"] == "open"]
    expected = int(_cfg("N_FPR_PAIRS"))
    if len(rows) != expected:
        raise RuntimeError(
            f"N_FPR_PAIRS={expected} but E5_PROMPTS has {len(rows)} open prompts. The E5 null "
            "must use ALL available open prompts: a stale count silently changes its noise floor")
    return rows


def judge_fpr() -> float:
    """Spec 5.8. Mean Score_Influence on unsteered/unsteered E5 pairs. Once per concept.

    Two extra Judge E5 calls where BOTH A and B are unsteered samples of the same prompt.
    Expected ~0. A non-zero value is the judge inventing influence because it expects to find
    some, and it puts a FLOOR under every E5 in the run - which is why it fires as soon as
    Phase 0 completes, before GPU time is spent on numbers that cannot be trusted (gate 3,
    spec 14.6 rule 5, `JUDGE_FPR_MAX`).

    A is sample 1 - the same reference every real E5 call is paired against - and B is sample
    2, so the control call is structurally identical to a real one in every respect except
    that B was never steered.

    Judged once per concept: with no steering the result cannot vary by cell. The value and
    its per-pair detail are cached on `RUN.base["judge_fpr"]`, so a second caller (gate 3, the
    status board, a resumed Phase 0) reads the number instead of re-spending the calls.

    Writes its calls to `judge_e5.jsonl` like any other E5 call - they are E5 calls, and
    keeping them in the same file is what lets gate 3 be checked against the same rows E5 was
    computed from.
    """
    run = _run()
    if BASE_FPR_KEY in run.base:
        return float(run.base[BASE_FPR_KEY]["fpr"])

    concept = _concept()
    phase_label = _phase_of(None)
    rows = _fpr_prompt_rows()

    units: list[dict] = []
    for row in rows:
        pid, text = str(row["id"]), str(row["text"])
        samples = unsteered_samples(pid)
        if len(samples) < 2:
            raise RuntimeError(
                f"control pair for {pid!r} needs two unsteered samples, Phase 0 stored "
                f"{len(samples)}. Spec 5.1 step 4 generates three per prompt: sample 1 is the "
                "reference A, samples 2-3 are these pairs. A pair built from one sample would "
                "compare a response with itself and score 0 by construction.")
        payload = _render(judges.JUDGE_E5, "JUDGE_E5", concept=concept, prompt=text,
                          response_unsteered=samples[0], response_steered=samples[1])
        unit_id = f"fpr_{pid}"
        units.append(dict(
            unit_id=unit_id, task_prompt=text, response_unsteered=samples[0],
            response_steered=samples[1], payload=payload, judge_id=JUDGE_E5_ID,
            # layer and r are None and the fingerprint is 'none': no vector was involved, and
            # naming a layer here would imply a steered measurement. `vectors.vec_fingerprint`
            # returns exactly 'none' for a None vector, so this is that function's own value
            # rather than a second convention.
            cache_key=_cache_key(phase_label, None, None, unit_id, JUDGE_E5_ID,
                                 vectors.vec_fingerprint(None), payload),
        ))

    results = _issue([_item(u) for u in units])

    scores: list[float] = []
    pairs: list[dict] = []
    errors: list[dict] = []
    for unit, result in zip(units, results):
        parsed, raw, failure = _parse_or_fail(result, JUDGE_E5_ID, judges.parse_e5)
        judge_row = dict(
            measure="JUDGE_FPR", judge_id=JUDGE_E5_ID, phase=phase_label, layer=None, r=None,
            alpha=None, vec_fingerprint=vectors.vec_fingerprint(None),
            prompt_id=unit["unit_id"], cache_key=list(unit["cache_key"]),
            task_prompt=unit["task_prompt"], response_unsteered=unit["response_unsteered"],
            response_steered=unit["response_steered"], payload=unit["payload"], raw=raw,
            judge_error=failure, control_pair=True,
        )
        if failure is None:
            score = _score_0_10(_field(parsed, "score_influence", JUDGE_E5_ID),
                                "score_influence", JUDGE_E5_ID)
            judge_row.update(shift=_field(parsed, "shift", JUDGE_E5_ID),
                             form=_field(parsed, "form", JUDGE_E5_ID),
                             score_influence=score,
                             justification=_field(parsed, "justification", JUDGE_E5_ID))
            scores.append(score)
            pairs.append(dict(id=unit["unit_id"], score_influence=score,
                              shift=judge_row["shift"]))
        else:
            errors.append(dict(id=unit["unit_id"], error=failure))
        _append_row(JUDGE_A1_FILE, judge_row)

    if not scores:
        raise RuntimeError(
            "judge_fpr: every control-pair call failed "
            f"({[e['error'] for e in errors]}). Gate 3 cannot be evaluated, and an unknown "
            "floor under E5 is not the same as a zero floor.")

    mean, se, n = model.mean_se(scores)
    ceiling = float(_cfg("JUDGE_FPR_MAX"))
    within = sum(1 for score in scores if score <= ceiling)
    within_ci_low, within_ci_high = cheap.wilson_interval(within, n)
    run.base[BASE_FPR_KEY] = dict(fpr=mean, fpr_se=se, fpr_n=n, pairs=pairs,
                                  judge_errors=len(errors), judge_error_detail=errors,
                                  ceiling=ceiling,
                                  # The primary null reading is a MEAN judge score and keeps
                                  # mean +/- SE. This companion is the actual binomial rate:
                                  # how many prompt-level scores cleared the ceiling.
                                  within_ceiling_count=within,
                                  within_ceiling_rate=within / n,
                                  within_ceiling_rate_ci_low=within_ci_low,
                                  within_ceiling_rate_ci_high=within_ci_high,
                                  within_ceiling_rate_n=n)
    print(f"judge FPR  : {mean:.2f} over {n} unsteered pairs"
          + (f" +/- {se:.2f} SE" if se is not None else "")
          + f"; prompt pass rate {within / n:.2f} (95% Wilson "
            f"[{within_ci_low:.2f}, {within_ci_high:.2f}], n={n})"
          + f"   (ceiling {float(_cfg('JUDGE_FPR_MAX')):g})")
    return float(mean)


def judge_s1_null() -> dict:
    """Judge S1 on unsteered B against unsteered A, cross-checked by objective S2.

    S1 alone cannot tell a harsh judge from genuinely poor model output. S2 makes the null
    gate admissible: low S1 while S2 says the same responses are mechanically healthy is an
    instrument disagreement and fails; low readings on both are model behaviour and are
    reported without aborting. The authored S1 frame remains concept-blind through
    `_s1_units`, exactly like every real S1 call.
    """
    run = _run()
    if BASE_S1_NULL_KEY in run.base:
        return dict(run.base[BASE_S1_NULL_KEY])

    rows = list(prompt_assets.E5_PROMPTS)
    ids = [str(row["id"]) for row in rows]
    texts = [str(row["text"]) for row in rows]
    responses = []
    for pid in ids:
        samples = unsteered_samples(pid)
        if len(samples) < 2:
            raise RuntimeError(
                f"S1 null for {pid!r} needs two unsteered samples, Phase 0 stored "
                f"{len(samples)}")
        responses.append(samples[1])

    phase_label = "CAL_NULL_S1"
    vec_fp = vectors.vec_fingerprint(None)
    units = _s1_units(ids, texts, responses, concept=_concept(), phase=phase_label,
                      layer=None, r=None, vec_fp=vec_fp)
    results = _issue([_item(unit) for unit in units])
    s1 = _score_s1(units, results,
                   dict(phase=phase_label, layer=None, r=None, alpha=None,
                        vec_fingerprint=vec_fp, measure="S1_NULL"))
    s2 = cheap.measure_S2(responses)
    minimum = float(_cfg("S1_NULL_MIN"))
    s1_low = float(s1["s1"]) < minimum
    s2_fine = float(s2["s2"]) >= minimum
    judge_fault = bool(s1_low and s2_fine)

    passing = sum(1 for row in s1["s1_per_prompt"]
                  if float(row["score_integrity"]) / 10.0 >= minimum)
    n = int(s1["s1_n"])
    pass_ci_low, pass_ci_high = cheap.wilson_interval(passing, n)
    detail = dict(
        s1_null=float(s1["s1"]), s1_null_se=s1["s1_se"], s1_null_n=n,
        s1_null_min=minimum, s1_null_pass_count=passing,
        s1_null_pass_rate=passing / n,
        s1_null_pass_rate_ci_low=pass_ci_low,
        s1_null_pass_rate_ci_high=pass_ci_high,
        s2=float(s2["s2"]), s2_n=int(s2["s2_n"]),
        s2_ci_low=s2["s2_ci_low"], s2_ci_high=s2["s2_ci_high"],
        s1_low=s1_low, s2_fine=s2_fine, judge_fault=judge_fault,
        passed=not judge_fault,
        reading=("judge_fault" if judge_fault else
                 ("model_output_degenerate" if s1_low else "healthy")),
        per_prompt=s1["s1_per_prompt"], s2_reasons=s2["s2_reasons"],
        judge_errors=s1["s1_judge_errors"],
    )
    run.base[BASE_S1_NULL_KEY] = detail
    print(f"S1 null    : {detail['s1_null']:.2f} +/- "
          f"{float(detail['s1_null_se'] or 0.0):.2f} SE; "
          f"prompt pass rate {detail['s1_null_pass_rate']:.2f} (95% Wilson "
          f"[{detail['s1_null_pass_rate_ci_low']:.2f}, "
          f"{detail['s1_null_pass_rate_ci_high']:.2f}], n={n}); "
          f"S2={detail['s2']:.2f} [{detail['s2_ci_low']:.2f}, "
          f"{detail['s2_ci_high']:.2f}], n={detail['s2_n']}; {detail['reading']}")
    return dict(detail)


def judge_d2_null() -> dict:
    """Unsteered forced-ID baseline, with transcripts; report-only at every reading.

    A nonzero null may be judge error or genuine model confabulation. The pipeline cannot
    distinguish those live, so this function records the generations and Wilson interval,
    flags values above one identification in N_D2, and NEVER returns a fatal verdict. Real
    cell selection continues to compare raw D2 with D2_MAX; the baseline is not subtracted.
    """
    run = _run()
    if BASE_D2_NULL_KEY in run.base:
        return dict(run.base[BASE_D2_NULL_KEY])

    n = int(_cfg("N_D2"))
    trials = list(range(1, n + 1))
    forced, start = prompt_assets.forced_prompts(trials)
    if start is None:
        raise RuntimeError("D2 null could not locate the forced-ID trial marker")
    responses = generate_unsteered(
        forced, int(_cfg("MAX_NEW_TOKENS")), float(_cfg("TEMPERATURE")),
        start_positions=[int(start)] * n)
    phase_label = "CAL_NULL_D2"
    vec_fp = vectors.vec_fingerprint(None)
    units = _d2_units(trials, responses, concept=_concept(), phase=phase_label,
                      layer=None, r=None, vec_fp=vec_fp)
    results = _issue([_item(unit) for unit in units])
    out = _score_d2(units, results,
                    dict(phase=phase_label, layer=None, r=None, alpha=None,
                         vec_fingerprint=vec_fp, measure="D2_NULL"))
    reference = float(_cfg("D2_NULL_REFERENCE"))
    derived = 1.0 / n
    if not math.isclose(reference, derived, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"D2_NULL_REFERENCE={reference} but 1/N_D2={derived}; the reporting line must "
            "remain one identification in the actual null sample")
    detail = dict(
        d2_null=float(out["d2"]), d2_null_se=out["d2_se"],
        d2_null_ci_low=out["d2_ci_low"], d2_null_ci_high=out["d2_ci_high"],
        d2_null_n=int(out["n_d2"]), d2_null_identified=int(out["d2_identified"]),
        d2_null_reference=reference, exceeds_reference=float(out["d2"]) > reference,
        report_only=True, passed=None,
        transcripts_file=D2_FILE, judge_errors=int(out["d2_judge_errors"]),
        d4=out["d4"], d4_ci=out["d4_ci"],
    )
    run.base[BASE_D2_NULL_KEY] = detail
    print(f"D2 null    : {detail['d2_null']:.2f} "
          f"[{detail['d2_null_ci_low']:.2f}, {detail['d2_null_ci_high']:.2f}] "
          f"n={detail['d2_null_n']} (reference {reference:.2f}; report only)")
    return dict(detail)


def judge_nulls() -> dict:
    """Measure all three Phase 0 judge nulls before any judged science phase runs."""
    judge_fpr()
    fpr = dict(_run().base[BASE_FPR_KEY])
    fpr["passed"] = float(fpr["fpr"]) <= float(_cfg("JUDGE_FPR_MAX"))
    return dict(e5=fpr, s1=judge_s1_null(), d2=judge_d2_null())


# =====================================================================================
# Verification - one verified.jsonl row
# =====================================================================================

def _finite(value: Any, name: str) -> float:
    """A metric that must exist and be a real number before it can enter min() or a gate."""
    if value is None:
        raise RuntimeError(
            f"{name} is None, so S4 = min(S1, S2, S3) and `qualifies` cannot be computed. "
            "A missing sanity term must not be treated as a passing one.")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{name} is {value!r}, which cannot enter a comparison")
    return out


def verify_cell(layer: int, r: float, *, phase: str | None = None,
                n_d2: int | None = None,
                prompt_set: Sequence[dict] | None = None) -> dict:
    """Phase 4/5. Generate ONCE, fan out to Judge E5, S1 and B, return the verified.jsonl row.

    The order matters and is the reason this function exists rather than three calls in a row:

      1. one steered generation over the task prompts  (12 responses)
      2. one steered generation over the forced-ID prompts  (N_D2 responses)
      3. **one** `judge_many` call carrying all 49 items - 12 E5, 12 S1, 25 B

    Spec 5.7 prices S1 at "24 judge calls per cell instead of 12 ... E5 and S1 for a given
    prompt are independent and issue concurrently, so wall time is unchanged". That is only
    true if they are in flight together. `measure_E5(...)` followed by `measure_S1(...)` would
    give the same numbers and spend the wall time the spec says the split does not cost, so
    Phase 4 and Phase 5 come through here.

    E5 and S1 score the SAME twelve responses - not two samples of them - which is what makes
    "high influence AND low integrity" (velocity L37 alpha=3.0, gate 2) a statement about one
    response rather than about two.

    Assembles, per CONTRACT section 4:
        e5, e5_min, e5_se, s1, s2, s3, s4 = min(S1, S2, S3), d2, d2_se, n_d2,
        the D4 distribution, usable, qualifies.

    `resid` and `covertness_margin` are NOT here: both are cross-cell quantities (a fit of D2
    against E5 over the verified set), so `phases.py` computes them once the set exists and
    merges them - along with the cell's Phase 1 scan fields - into the row it writes. This
    function returns the expensive half plus the sanity terms.

    Raises `config.Unreachable` for a cell above ALPHA_CEIL, unchanged from `alpha_for`.
    Callers log the cell as unreachable; nothing here clamps.
    """
    phase_label = _phase_of(phase)
    layer = int(layer)
    r = float(r)
    alpha = float(config.alpha_for(layer, r))     # raises Unreachable; never clamps
    rows = list(prompt_assets.E5_PROMPTS if prompt_set is None else prompt_set)
    trials_n = int(_cfg("N_D2")) if n_d2 is None else int(n_d2)

    # The vector itself is not needed here - generate_steered hard-indexes RUN.vecs for the
    # no-override case. What this call buys is the fingerprint for the cache key and the
    # round-trip check below.
    _vector, dose, vec_fp = _resolve_vector(layer, alpha, None, None)
    if not math.isclose(dose, r, rel_tol=1e-6, abs_tol=1e-9):
        # alpha_for and dose_for are inverses over the same RUN.norms row. A disagreement
        # means one of them has the formula flipped, which would rescale the whole grid
        # without erroring anywhere (spec section 3).
        raise RuntimeError(
            f"dose round-trip failed at L{layer}: r={r} -> alpha={alpha} -> r={dose}. "
            "config.alpha_for and config.dose_for disagree; do not measure.")

    max_new = int(_cfg("MAX_NEW_TOKENS"))
    temperature = float(_cfg("TEMPERATURE"))
    concept = _concept()

    # --- 1. generate once ------------------------------------------------------------
    task_prompts, starts, ids = task_batch(rows)
    texts = [str(row["text"]) for row in rows]
    task_responses = generate_steered(task_prompts, layer, alpha, max_new, temperature,
                                      start_positions=starts)
    _write_cis(rows, task_responses, layer=layer, alpha=alpha, r=r, phase=phase_label,
               vec_fp=vec_fp)

    trials = list(range(1, trials_n + 1))
    forced, start = prompt_assets.forced_prompts(trials)
    if start is None:
        raise RuntimeError(
            "prompts.forced_prompts() could not locate the trial marker in its own prompt "
            "(see measure_D2). D2 must not run on an unverified prompt construction.")
    forced_responses = generate_steered(forced, layer, alpha, max_new, temperature,
                                        start_positions=[int(start)] * len(forced))

    # --- 2. one concurrent judge batch: E5 + S1 + B ----------------------------------
    ctx = dict(phase=phase_label, layer=layer, r=r, alpha=alpha, vec_fingerprint=vec_fp)
    a1_units = _e5_units(ids, texts, task_responses, concept=concept, phase=phase_label,
                         layer=layer, r=r, vec_fp=vec_fp)
    a2_units = _s1_units(ids, texts, task_responses, concept=concept, phase=phase_label,
                         layer=layer, r=r, vec_fp=vec_fp)
    b_units = _d2_units(trials, forced_responses, concept=concept, phase=phase_label,
                       layer=layer, r=r, vec_fp=vec_fp)

    all_units = list(a1_units) + list(a2_units) + list(b_units)
    results = _issue([_item(u) for u in all_units])

    n_a1, n_a2 = len(a1_units), len(a2_units)
    e5 = _score_e5(a1_units, results[:n_a1], ctx)
    s1 = _score_s1(a2_units, results[n_a1:n_a1 + n_a2], ctx)
    d2 = _score_d2(b_units, results[n_a1 + n_a2:], ctx)

    # --- 3. the cheap sanity terms on the SAME responses -----------------------------
    # S2 is computed on the steered task generations this cell just produced, not on a
    # separate sample: CONTRACT defence 9 folds an objective degeneracy floor in with min(),
    # and it only floors this cell if it describes this cell's text. It stays independent of
    # the judge (bug 27) - it catches loop-collapse the judge may score as coherent, and the
    # judge catches fluent fixation this cannot see.
    s2 = cheap.measure_S2(task_responses)
    s3 = cheap.measure_S3(layer, alpha)

    s1_v = _finite(s1["s1"], "S1")
    s2_v = _finite(s2["s2"], "S2")
    s3_v = _finite(s3["s3"], "S3")
    # Minimum, not mean (spec section 7): three different ways to be unusable, and passing one
    # must not compensate for failing another.
    s4 = min(s1_v, s2_v, s3_v)

    e5_v = _finite(e5["e5"], "E5")
    d2_v = _finite(d2["d2"], "D2")
    usable = s4 >= float(_cfg("S4_MIN"))
    qualifies = bool(usable and e5_v >= float(_cfg("E5_FLOOR")) and d2_v <= float(_cfg("D2_MAX")))

    row = dict(
        phase=phase_label, layer=layer, r=r, alpha=alpha, reachable=True,
        vec_fingerprint=vec_fp,
        # effectiveness (spec 5.6)
        e5=e5["e5"], e5_min=e5["e5_min"], e5_se=e5["e5_se"], e5_n=e5["e5_n"],
        # sanity (spec 5.7, 5.5, 5.4 and section 7)
        s1=s1_v, s1_se=s1["s1_se"], s1_min=s1["s1_min"],
        s2=s2_v, s2_ci_low=s2["s2_ci_low"], s2_ci_high=s2["s2_ci_high"],
        s2_n=s2["s2_n"], degenerate_frac=s2["degenerate_frac"],
        degenerate_frac_ci_low=s2["degenerate_frac_ci_low"],
        degenerate_frac_ci_high=s2["degenerate_frac_ci_high"],
        s3=s3_v, s3_correct=s3["s3_correct"], s3_margin=s3["s3_margin"],
        s3_n=s3["s3_n"], s3_acc=s3["s3_acc"],
        s3_acc_ci_low=s3["s3_acc_ci_low"], s3_acc_ci_high=s3["s3_acc_ci_high"],
        s4=s4, s4_term=("S1" if s4 == s1_v else ("S2" if s4 == s2_v else "S3")),
        # detection (spec 5.9)
        d2=d2_v, d2_se=d2["d2_se"], d2_ci_low=d2["d2_ci_low"],
        d2_ci_high=d2["d2_ci_high"], n_d2=d2["n_d2"],
        d2_identified=d2["d2_identified"],
        d4=d2["d4"], d4_counts=d2["d4_counts"], d4_n=d2["d4_n"],
        d4_ci=d2["d4_ci"],
        d4_dominant=d2["d4_dominant"], d4_damage_frac=d2["d4_damage_frac"],
        d4_damage_count=d2["d4_damage_count"],
        d4_damage_frac_ci_low=d2["d4_damage_frac_ci_low"],
        d4_damage_frac_ci_high=d2["d4_damage_frac_ci_high"],
        d4_retrieval_frac=d2["d4_retrieval_frac"],
        d4_retrieval_count=d2["d4_retrieval_count"], d4_reading=d2["d4_reading"],
        d4_retrieval_frac_ci_low=d2["d4_retrieval_frac_ci_low"],
        d4_retrieval_frac_ci_high=d2["d4_retrieval_frac_ci_high"],
        # the section 7 verdicts
        usable=bool(usable), qualifies=qualifies,
        # what was excluded, so no denominator in this row is anonymous
        e5_judge_errors=e5["e5_judge_errors"], s1_judge_errors=s1["s1_judge_errors"],
        d2_judge_errors=d2["d2_judge_errors"],
        e5_per_prompt=e5["e5_per_prompt"], s1_per_prompt=s1["s1_per_prompt"],
        prompt_ids=ids,
    )
    row.update(_d2_null_fields())

    print(f"verify     : L{layer} r={r:.3f} a={alpha:.2f}  "
          f"E5={e5_v:.2f} +/- {float(e5['e5_se'] or 0.0):.2f} SE  "
          f"S4={s4:.2f} ({row['s4_term']})  "
          f"D2={d2_v:.2f} [{d2['d2_ci_low']:.2f}, {d2['d2_ci_high']:.2f}] "
          f"n={d2['n_d2']} (unsteered {row['d2_null']:.2f} "
          f"[{row['d2_null_ci_low']:.2f}, {row['d2_null_ci_high']:.2f}], "
          f"n={row['d2_null_n']})  "
          f"{'QUALIFIES' if qualifies else ('usable' if usable else 'unusable')}")
    return row
