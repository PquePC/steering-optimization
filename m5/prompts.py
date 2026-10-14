"""m5.prompts - every prompt M5 sends, rendered once, with the token positions its steering uses.

A `PromptSpec` is the rendered text (chat template applied by `Backend.render`, prefill appended),
the channel it serves, and the token positions an intervention needs:

    steer_from  live steering: Positions("from", steer_from) -- that token and every later prompt
                token and every generated token (Lindsey / Macar)
    span        Positions("span", *span) -- the past-only briefing (d6, d7) or one d9 segment
    meta        everything a readout or scorer needs (trial, variant, options, answer key, segment
                spans, ids), JSON-serialisable so it can be written to a row as it stands

`spec.positions(mode)` maps "live" / "past_only" / "all" (and "generated") to `Positions` through
`m5.steer.positions_for`, the one place that mapping is defined.

## The conversation (d2, d3, d5, d6, d7, d8, d9)

Macar et al.'s harness, verbatim (`m5.data.detection`): the researcher briefing as the first user
turn, "Ok." as the model's turn, then "Trial {n}: ..." as the second user turn, and the
generation prompt, with any prefill appended directly after it. Each channel changes only the
second user turn and the prefill, so its numbers are comparable with the others'.

**No system turn, for any model.** Macar's harness drops the empty system message for Gemma
because "passing an empty system message causes the chat template to prepend extra newlines to the
first user message, changing the tokenization and significantly increasing false positive rates
(especially for abliterated models)" (upstream/introspection_mechanisms/src/steering_utils.py:17-20;
read_macar_protocol.md §1.1). For other models the harness keeps it, which renders an empty system
turn (Qwen3: `<|im_start|>system\\n<|im_end|>`) in front of the briefing. M5 sends no system turn to
any model, so every model reads the same conversation and nothing precedes the briefing but the
template's own header. On Gemma this is byte-identical to the harness; on Qwen it departs from it
by exactly that empty turn.

**Where live steering starts: the token immediately before "Trial {n}".** Macar's rule
(steering_utils.py:448-461): find "Trial {n}" in the rendered prompt, tokenize everything before
it, and start at the last of those tokens -- on Gemma the "\\n" after `<start_of_turn>user`. The
briefing and the "Ok." turn are never steered live. When a tokenizer merges that whitespace into
the marker's own first token, the merged token is the start and `meta["steer_from_rule"]` says so;
any other shape raises (a steering start that silently moves is the defect class that withdrew the
old Qwen arm). `steer_from` is always >= 1: index 0 is the BOS sink.

**The past-only span is the briefing**: from the first token of the first user turn's content to
the end of the "Ok." turn, ending where the second user turn's header begins. A past-only
intervention steers the briefing; the question attends to its steered keys and values but is not
itself steered (Pearson et al.'s design; plan §3.1 `d6`, `d7`).

**Trial numbers are 1-10** (Macar B.3: "Trial 30" reliably confabulates "apples"; M3 reached 175).
Every builder refuses others, booleans included.

## Variants

`variant` is a rung of the elicitation ladder (`m5.data.detection.VARIANTS`): "original" (Macar),
"primer" (Macar's briefing plus Pearson et al.'s mechanism-informative primer) and "unusual" (an
injection-free first turn; d8's unusual-processing framing only). "alternative" and "skeptical"
raise: Macar's exact texts are in no local source. "bias_vector" raises: it is a trained vector,
not a prompt.

d8's two framings (`cfg["SELF_REPORT_FRAMINGS"]`) are the question, not the briefing: under
"original" and "primer" both are asked after the briefing ("briefed self-report", plan §3.2);
"unusual" is the version in which nothing at all mentions injection.

## Everything else

Open-ended prompts (influence, e9, s2, s5 capability items, a1 scenarios, s6 text) are one user
turn steered with Positions("all"), so they carry no `steer_from` or `span`. The e10 A/B item is
the exception: it marks `steer_from` at the end of the instruction, CAA's arrangement, and
`meta["mode"]` says "live". Every random choice (decoys, the concept's place in a list, segment
draws) takes the caller's seed through `derive_seed`, so a spec is a pure function of its arguments.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from m5.backend.base import derive_seed, token_index_at
from m5.data import capability as _cap
from m5.data import contrastive as _con
from m5.data import detection as _det
from m5.data import influence as _inf
from m5.data import markers
from m5.data import scenarios as _scn
from m5.data import words as _words
from m5.intervention import Positions
from m5.safety import check_concept

# Macar et al. B.3 (vault macar-2026-...md:550-552): trial numbers 1-10, where FPR 0 was measured.
TRIAL_RANGE = range(1, 11)


@dataclass(frozen=True)
class PromptSpec:
    text: str                      # rendered by backend.render, prefill appended
    channel: str                   # e.g. "d3", "d5", "d7:target", "d8:injected_thought", "e9"
    steer_from: int | None         # token index for Positions("from", steer_from)
    span: tuple[int, int] | None   # token span for Positions("span", *span)
    meta: dict                     # trial, variant, options, answer key, segment spans, ids ...

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("a prompt spec needs its rendered text")
        if not self.channel:
            raise ValueError("a prompt spec needs its channel")
        n = self.meta.get("n_tokens")
        if self.steer_from is not None:
            if isinstance(self.steer_from, bool) or not isinstance(self.steer_from, int) or self.steer_from < 1:
                raise ValueError(f"{self.channel}: steer_from must be an int >= 1 (index 0 is the BOS "
                                 f"sink), got {self.steer_from!r}")
            if n is not None and self.steer_from >= n:
                raise ValueError(f"{self.channel}: steer_from {self.steer_from} is beyond the prompt ({n} tokens)")
        if self.span is not None:
            start, end = self.span
            if not 1 <= start < end:
                raise ValueError(f"{self.channel}: span must satisfy 1 <= start < end, got {self.span}")
            if n is not None and end > n:
                raise ValueError(f"{self.channel}: span end {end} is beyond the prompt ({n} tokens)")

    def positions(self, mode: str) -> Positions:
        """Positions for mode "live" | "past_only" | "all" | "generated"; raises when this spec
        marks no position for the mode."""
        from m5.steer import positions_for  # local: m5.steer may import this module's type later
        return positions_for(self, mode)


# =====================================================================================
# Shared machinery
# =====================================================================================

def _check_trial(trial: Any) -> int:
    if isinstance(trial, bool) or not isinstance(trial, int) or trial not in TRIAL_RANGE:
        raise ValueError(f"trial number must be an int in 1-10 (Macar B.3: higher numbers confabulate), "
                         f"got {trial!r}")
    return trial


def _tokens(backend: Any, text: str) -> list[tuple[int, int]]:
    ids = backend.tokenize(text)
    offsets = backend.offsets(text)
    if len(ids) != len(offsets):
        raise ValueError(f"backend.offsets gives {len(offsets)} spans for {len(ids)} tokens "
                         "(Backend contract 1)")
    return [tuple(o) for o in offsets]


def _find_once(text: str, needle: str, what: str) -> int:
    i = text.find(needle)
    if i < 0:
        raise ValueError(f"{what} {needle[:60]!r} is not in the rendered prompt")
    if text.find(needle, i + 1) >= 0:
        raise ValueError(f"{what} {needle[:60]!r} occurs more than once in the rendered prompt")
    return i


def _render(backend: Any, messages: Sequence[Mapping[str, str]], prefill: str = "") -> str:
    for m in messages:
        if m["role"] not in ("user", "assistant"):
            raise ValueError("M5 prompts carry no system turn (module docstring)")
    return backend.render(list(messages), add_generation_prompt=True) + prefill


def _trial_start(text: str, offsets: list[tuple[int, int]], trial: int) -> tuple[int, str]:
    """Macar's steering start: the token immediately before "Trial {n}" (see the docstring)."""
    if text.count(markers.TRIAL_MARKER) != 1:
        raise ValueError(f"the rendered prompt must contain {markers.TRIAL_MARKER!r} exactly once; "
                         f"it contains it {text.count(markers.TRIAL_MARKER)} times")
    pos = _find_once(text, f"{markers.TRIAL_MARKER}{trial}:", "trial marker")
    tok = token_index_at(offsets, pos)
    first = offsets[tok][0]
    if first == pos:
        start, rule = tok - 1, "token_before_marker"
    elif first < pos and text[first:pos].isspace():
        start, rule = tok, "marker_token_with_leading_whitespace"
    else:
        raise ValueError(f"'Trial {trial}' does not begin on a token boundary in this template; "
                         "Macar's steering start is undefined here")
    if start < 1:
        raise ValueError(f"steering would start at token {start}, the BOS sink or before; the template "
                         "puts nothing before the question")
    return start, rule


def _turn_header(text: str, first_content: int, second_content: int) -> int:
    """Character index where the second user turn's header begins: the longest common suffix of the
    text before the first user turn's content and the text before the second's is the template's
    user-turn header (Gemma `<start_of_turn>user\\n`, Qwen `<|im_start|>user\\n`)."""
    a, b = text[:first_content], text[:second_content]
    n = 0
    while n < len(a) and a[len(a) - 1 - n] == b[len(b) - 1 - n]:
        n += 1
    if n == 0:
        raise ValueError("cannot find the user-turn header in this template")
    return second_content - n


def _briefing_span(text: str, offsets: list[tuple[int, int]], briefing: str,
                   question_content: str) -> tuple[int, int]:
    b0 = _find_once(text, briefing, "briefing")
    q0 = _find_once(text, question_content, "question turn")
    start = token_index_at(offsets, b0)
    end_char = _turn_header(text, b0, q0)
    end = token_index_at(offsets, end_char)
    if offsets[end][0] < end_char and not text[offsets[end][0]:end_char].isspace():
        raise ValueError("the end of the 'Ok.' turn does not fall on a token boundary")
    if not 1 <= start < end:
        raise ValueError(f"empty or misplaced briefing span [{start}, {end})")
    return start, end


def _briefed(variant: str, channel: str) -> _det.Variant:
    v = _det.variant(variant)
    if "injected_thought" not in v.framings:
        raise ValueError(f"{channel}: variant {variant!r} is the injection-free conversation (d8's "
                         f"unusual-processing framing only); {channel}'s prompt names an injected thought")
    return v


def _salience_sentence(salience: str | None) -> str | None:
    if salience is None:
        return None
    check_concept(salience)
    return _det.SALIENCE_TEMPLATE.format(word=salience.lower())


def _conversation(backend: Any, *, builder: str, channel: str, trial: int, briefing: str,
                  briefing_id: str, variant: str, question: str, prefill: str, marker: str,
                  live: bool, past: bool, salience: str | None = None,
                  extra: Mapping[str, Any] | None = None) -> PromptSpec:
    """Render briefing / "Ok." / question (+ prefill) and mark the live start and past-only span."""
    sentence = _salience_sentence(salience)
    content = question if sentence is None else f"{sentence}\n\n{question}"
    messages = [{"role": "user", "content": briefing},
                {"role": "assistant", "content": _det.MACAR_ACK},
                {"role": "user", "content": content}]
    text = _render(backend, messages, prefill)
    if marker not in text:
        raise AssertionError(f"{channel}: the prompt lost its marker {marker!r}")  # pragma: no cover
    offsets = _tokens(backend, text)
    steer_from, rule = _trial_start(text, offsets, trial) if live else (None, None)
    span = _briefing_span(text, offsets, briefing, content) if past else None
    meta = {"builder": builder, "variant": variant, "briefing": briefing_id, "trial": trial,
            "question": content, "prefill": prefill, "marker": marker, "salience": salience,
            "n_tokens": len(offsets), "steer_from_rule": rule}
    meta.update(extra or {})
    return PromptSpec(text, channel, steer_from, span, meta)


def _resolve_framing(v: _det.Variant, framing: str | None) -> str:
    if framing is None:
        return v.framings[0] if len(v.framings) == 1 else "injected_thought"
    f = _det.canonical_framing(framing)
    if f not in v.framings:
        raise ValueError(f"variant {v.name!r} carries the framing(s) {v.framings}, not {f!r}: the "
                         "'unusual' variant is the one in which nothing mentions injection")
    return f


def _display(word: str) -> str:
    return word[:1].upper() + word[1:]


# =====================================================================================
# The detection-style channels
# =====================================================================================

def detection_trial(backend: Any, trial: int, *, variant: str = "original", prefill: str | None = None,
                    framing: str | None = None, salience: str | None = None) -> PromptSpec:
    """The Lindsey/Macar trial.

    prefill None      d8: unprefilled self-report, channel "d8:<framing>"
    prefill "forced"  d3 (and d2): Macar's forced-identification prefill, channel "d3"
    prefill "yes_no"  d7's target question, the same as `yes_no(kind="target")`

    `framing` is "injected_thought" or "unusual_processing" ("unusual" is accepted); by default the
    variant's own. `salience` (a concept) puts the salience control's sentence before the question.
    """
    trial = _check_trial(trial)
    if prefill == "yes_no":
        if framing not in (None, "injected_thought"):
            raise ValueError("the yes/no target question is the injected-thought framing")
        return yes_no(backend, trial, kind="target", variant=variant, salience=salience)
    if prefill not in (None, "forced"):
        raise ValueError(f"unknown prefill {prefill!r}; one of None, 'forced', 'yes_no'")
    v = _det.variant(variant)
    f = _resolve_framing(v, framing)
    if prefill == "forced":
        if f != "injected_thought":
            raise ValueError("the forced prefill names an injected thought; it belongs to the "
                             "injected-thought framing")
        return _conversation(backend, builder="detection_trial", channel="d3", trial=trial,
                             briefing=v.briefing, briefing_id=v.briefing_id, variant=v.name,
                             question=_det.MACAR_QUESTION.format(n=trial), prefill=_det.FORCED_PREFILL,
                             marker=markers.FORCED_PREFILL, live=True, past=True, salience=salience,
                             extra={"framing": f})
    marker = markers.DETECTION_QUESTION if f == "injected_thought" else markers.UNUSUAL_QUESTION
    return _conversation(backend, builder="detection_trial", channel=f"d8:{f}", trial=trial,
                         briefing=v.briefing, briefing_id=v.briefing_id, variant=v.name,
                         question=_det.FRAMINGS[f].format(n=trial), prefill="", marker=marker,
                         live=True, past=True, salience=salience, extra={"framing": f})


def self_report(backend: Any, trial: int, *, framing: str | None = None, variant: str = "original",
                salience: str | None = None) -> PromptSpec:
    """d8: the briefed self-report, unprefilled, in one of the two framings."""
    return detection_trial(backend, trial, variant=variant, prefill=None, framing=framing, salience=salience)


def forced_naming(backend: Any, trial: int, *, variant: str = "original", channel: str = "d3",
                  salience: str | None = None) -> PromptSpec:
    """d2 / d3: the trial with Macar's forced prefill. d2 generates from it, d3 reads the concept's
    probability right after it; d6's slot reading uses the same spec with mode "past_only"."""
    if channel not in ("d2", "d3"):
        raise ValueError(f"forced naming serves d2 or d3, not {channel!r}")
    _briefed(variant, channel)
    spec = detection_trial(backend, trial, variant=variant, prefill="forced", salience=salience)
    return dataclasses.replace(spec, channel=channel, meta={**spec.meta, "builder": "forced_naming"})


def open_slot(backend: Any, trial: int, *, variant: str = "original") -> PromptSpec:
    """d5: the same conversation and steering start as d3, with a slot that asks for no
    introspection. Introspective excess = d3 - d5."""
    trial = _check_trial(trial)
    v = _briefed(variant, "d5")
    return _conversation(backend, builder="open_slot", channel="d5", trial=trial, briefing=v.briefing,
                         briefing_id=v.briefing_id, variant=v.name,
                         question=_det.OPEN_SLOT_QUESTION.format(n=trial),
                         prefill=markers.OPEN_SLOT_PREFILL, marker=markers.OPEN_SLOT_REQUEST,
                         live=True, past=True)


def decoy_list(backend: Any, trial: int, concept: str, *, k: int, seed: int, variant: str = "original",
               decoys: Sequence[str] | None = None) -> PromptSpec:
    """d6: past-only naming as a K-way list (chance 1/K). `k` is `cfg["D6_OPTIONS"]`.

    The concept sits at a seeded position among k - 1 decoys drawn far from it
    (`m5.data.words.draw_decoys`) unless `decoys` are given. Past-only by design, so there is no
    `steer_from`: a live decoy list would let steering leak into the answer, which is what d6
    exists to rule out.
    """
    trial = _check_trial(trial)
    check_concept(concept)
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= _det.MAX_OPTIONS:
        raise ValueError(f"d6 needs 2 <= k <= {_det.MAX_OPTIONS} options (each label one digit token), got {k!r}")
    v = _briefed(variant, "d6")
    if decoys is None:
        decoys = _words.draw_decoys(concept, k - 1, seed)
    decoys = list(decoys)
    if len(decoys) != k - 1:
        raise ValueError(f"d6 with k={k} needs {k - 1} decoys, got {len(decoys)}")
    for d in decoys:
        check_concept(d)
        if _words.related(d, concept):
            raise ValueError(f"decoy {d!r} is related to the concept {concept!r}")
    if len({d.casefold() for d in decoys}) != len(decoys):
        raise ValueError("decoys must be distinct")
    pos = random.Random(derive_seed(seed, "d6", str(concept).casefold(), k)).randrange(k)
    options = [_display(w) for w in (*decoys[:pos], concept, *decoys[pos:])]
    lines = "\n".join(f"{i}. {w}" for i, w in enumerate(options, start=1))
    question = f"{_det.DECOY_LIST_HEAD.format(n=trial)}\n\n{lines}\n\n{_det.DECOY_LIST_TAIL}"
    return _conversation(backend, builder="decoy_list", channel="d6", trial=trial, briefing=v.briefing,
                         briefing_id=v.briefing_id, variant=v.name, question=question,
                         prefill=markers.DECOY_LIST_PREFILL, marker=markers.DECOY_LIST_QUESTION,
                         live=False, past=True,
                         extra={"concept": concept, "options": options,
                                "option_labels": [str(i) for i in range(1, k + 1)],
                                "correct_option": pos + 1, "correct_label": str(pos + 1),
                                "decoys": [_display(d) for d in decoys], "k": k, "chance": 1 / k})


YES_NO_KINDS = ("target", "control", "mismatch", "match")


def yes_no(backend: Any, trial: int, *, kind: str = "target", question: str | int | None = None,
           concept: str | None = None, other: str | None = None, seed: int | None = None,
           variant: str = "original", salience: str | None = None) -> PromptSpec:
    """d7: one yes/no question after the briefing, answered after "Answer:".

    kind "target"    the detection question
         "control"   a question with a known answer (`question`: id or index into
                     `m5.data.detection.CONTROL_QUESTIONS`); `meta["expected"]` is its answer
         "mismatch"  "...an injected thought about {other}?", `other` far from the injected
                     `concept` (drawn with `seed` when not given)
         "match"     the same question naming the injected `concept` (Sridharan's congruent arm)

    Live and past-only from the same spec: `steer_from` and `span` are both marked.
    """
    trial = _check_trial(trial)
    if kind not in YES_NO_KINDS:
        raise ValueError(f"unknown yes/no kind {kind!r}; one of {YES_NO_KINDS}")
    v = _briefed(variant, "d7")
    extra: dict[str, Any] = {"kind": kind, "expected": None, "question_id": None,
                             "concept": concept, "other": None}
    if kind != "control" and question is not None:
        raise ValueError(f"a {kind} question is fixed; `question` selects control questions only")
    if kind == "target":
        text, marker = _det.YES_NO_TARGET.format(n=trial), markers.DETECTION_QUESTION
    elif kind == "control":
        if question is None:
            raise ValueError("a control question must be selected (`question`: id or index)")
        cq = _det.control_question(question)
        text, marker = _det.YES_NO_CONTROL.format(n=trial, question=cq.text), markers.YES_NO_INSTRUCTION
        extra.update(expected=cq.expected or None, question_id=cq.id, about=cq.about)
    else:
        if concept is not None:
            check_concept(concept)
        if kind == "match":
            if concept is None:
                raise ValueError("a match question names the injected concept: pass `concept`")
            named = concept
        else:
            if other is None:
                if concept is None or seed is None:
                    raise ValueError("a mismatch question needs `other`, or `concept` and `seed` to draw one")
                other = _words.draw_decoys(concept, 1, seed)[0]
            check_concept(other)
            if concept is not None and _words.related(other, concept):
                raise ValueError(f"the mismatch concept {other!r} is related to the injected {concept!r}")
            named = other
        extra["other"] = other
        text, marker = _det.YES_NO_NAMED.format(n=trial, concept=named.lower()), markers.YES_NO_INSTRUCTION
    return _conversation(backend, builder="yes_no", channel=f"d7:{kind}", trial=trial, briefing=v.briefing,
                         briefing_id=v.briefing_id, variant=v.name, question=text,
                         prefill=markers.YES_NO_PREFILL, marker=marker, live=True, past=True,
                         salience=salience, extra=extra)


def localisation(backend: Any, trial: int, *, k: int, target: int | None, seed: int,
                 variant: str = "original", pool: str = "m5", texts: Sequence[str] | None = None,
                 avoid: Sequence[str] = ()) -> PromptSpec:
    """d9: K numbered segments in one user turn; which one, if any, carried the injection.

    `k` is `cfg["D9_SEGMENTS"]`; `target` is the steered segment (1..k) or None for a "none" trial.
    `meta["segment_spans"][i]` is segment i+1's token span, so an intervention steers exactly one
    segment; `span` is the target's. The sentences are drawn from `pool` with `seed` alone, so
    sub-trials that differ only in `target` share them (Hahami et al.'s design); sentences related
    to any word in `avoid` (pass the concept) are never drawn. Options are "1".."k" and "none";
    chance is 1/(k+1).
    """
    trial = _check_trial(trial)
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= _det.MAX_OPTIONS:
        raise ValueError(f"d9 needs 2 <= k <= {_det.MAX_OPTIONS} segments (each label one digit token), got {k!r}")
    if target is not None and (isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= k):
        raise ValueError(f"target must be None or a segment number in 1..{k}, got {target!r}")
    v = _briefed(variant, "d9")
    briefing = _det.LOCALISATION_BRIEFING if v.name == "original" else \
        "\n\n".join((_det.LOCALISATION_BRIEFING, _det.PRIMER))
    drawn_from = pool if texts is None else None
    if texts is None:
        if pool not in _det.SEGMENT_POOLS:
            raise ValueError(f"unknown segment pool {pool!r}; one of {sorted(_det.SEGMENT_POOLS)}")
        usable = [s for s in _det.SEGMENT_POOLS[pool]
                  if not any(_words.related(w, a) for a in avoid for w in _inf.normalise(s).split())]
        if len(usable) < k:
            raise ValueError(f"pool {pool!r} has only {len(usable)} sentences clear of {list(avoid)}")
        rng = random.Random(derive_seed(seed, "d9", pool, k))
        texts = [usable.pop(rng.randrange(len(usable))) for _ in range(k)]
    texts = [str(t) for t in texts]
    if len(texts) != k:
        raise ValueError(f"d9 with k={k} needs {k} segments, got {len(texts)}")
    for t in texts:
        if not t.strip() or "\n" in t or markers.TRIAL_MARKER in t:
            raise ValueError(f"a segment must be one non-empty line without the trial marker: {t!r}")
    blocks = "\n\n".join(f"{_det.SEGMENT_LABEL.format(i=i)}\n{t}" for i, t in enumerate(texts, start=1))
    question = f"{_det.LOCALISATION_HEAD.format(n=trial, k=k)}\n\n{blocks}\n\n{_det.LOCALISATION_TAIL}"
    spec = _conversation(backend, builder="localisation", channel="d9", trial=trial, briefing=briefing,
                         briefing_id="localisation" if v.name == "original" else "localisation_primer",
                         variant=v.name, question=question, prefill=markers.SEGMENT_PREFILL,
                         marker=markers.SEGMENT_QUESTION, live=False, past=False)
    offsets = _tokens(backend, spec.text)
    q0 = _find_once(spec.text, question, "question turn")
    spans: list[tuple[int, int]] = []
    for i, t in enumerate(texts, start=1):
        head = f"{_det.SEGMENT_LABEL.format(i=i)}\n"
        c0 = q0 + question.index(head + t) + len(head)
        c1 = c0 + len(t)
        s = token_index_at(offsets, c0)
        e = token_index_at(offsets, c1 - 1) + 1
        if spec.text[offsets[s][0]:c0].strip() or spec.text[c1:offsets[e - 1][1]].strip():
            raise ValueError(f"segment {i}'s tokens reach past its text into the labels; a "
                             "one-segment intervention would not be one segment")
        spans.append((s, e))
    options = [str(i) for i in range(1, k + 1)] + [markers.SEGMENT_NONE]
    meta = {**spec.meta, "k": k, "target": target, "segments": texts, "segment_spans": spans,
            "options": options, "answer": str(target) if target is not None else markers.SEGMENT_NONE,
            "chance": 1 / (k + 1), "pool": drawn_from}
    return dataclasses.replace(spec, span=spans[target - 1] if target is not None else None, meta=meta)


def counterbalanced_targets(n_trials: int, k: int, seed: int) -> list[int | None]:
    """d9 targets for `n_trials` trials (`cfg["D9_TRIALS"]`): every segment and "none" (None) equally
    often, up to the remainder, in a seeded order."""
    if n_trials < 1 or k < 1:
        raise ValueError("need at least one trial and one segment")
    cycle: list[int | None] = [*range(1, k + 1), None]
    targets = [cycle[i % len(cycle)] for i in range(n_trials)]
    rng = random.Random(derive_seed(seed, "d9-targets", n_trials, k))
    for i in range(len(targets) - 1, 0, -1):
        j = rng.randrange(i + 1)
        targets[i], targets[j] = targets[j], targets[i]
    return targets


# =====================================================================================
# Single-turn prompts
# =====================================================================================

def _single_turn(backend: Any, *, builder: str, channel: str, content: str, prefill: str = "",
                 extra: Mapping[str, Any] | None = None, steer_from_answer: bool = False) -> PromptSpec:
    text = _render(backend, [{"role": "user", "content": content}], prefill)
    offsets = _tokens(backend, text)
    steer_from = None
    if steer_from_answer:
        end = _find_once(text, content, "user content") + len(content)
        steer_from = next(i for i, (s, _) in enumerate(offsets) if s >= end)
    meta = {"builder": builder, "content": content, "prefill": prefill, "n_tokens": len(offsets),
            "mode": "live" if steer_from_answer else "all"}
    meta.update(extra or {})
    return PromptSpec(text, channel, steer_from, None, meta)


def _open_prompt(prompt: _inf.OpenPrompt | str | int, default_set: str) -> tuple[str, _inf.OpenPrompt]:
    if isinstance(prompt, _inf.OpenPrompt):
        for name, ps in _inf.SETS.items():
            if prompt in ps:
                return name, prompt
        return "custom", prompt
    if isinstance(prompt, bool):
        raise TypeError("a prompt is an OpenPrompt, an id or an index, not a boolean")
    if isinstance(prompt, int):
        return default_set, _inf.prompt_set(default_set)[prompt]
    return _inf.find(prompt)


def influence(backend: Any, prompt: _inf.OpenPrompt | str) -> PromptSpec:
    """e7 / e8 / s1 / d10: one open-ended prompt (an `OpenPrompt` or a search/held-out id),
    steered at every position."""
    split, p = _open_prompt(prompt, "search")
    if split not in ("search", "heldout", "custom"):
        raise ValueError(f"{p.id} belongs to the {split!r} set, not to the influence prompts")
    return _single_turn(backend, builder="influence", channel="influence", content=p.text,
                        extra={"prompt_id": p.id, "split": split, "register": p.register})


def neutral_for_e9(backend: Any, prompt: _inf.OpenPrompt | str | int) -> PromptSpec:
    """e9: a neutral question. The model's own unsteered continuation is appended at run time and
    scored teacher-forced; its first token will sit at index `meta["answer_start"]`."""
    split, p = _open_prompt(prompt, "neutral")
    spec = _single_turn(backend, builder="neutral_for_e9", channel="e9", content=p.text,
                        extra={"prompt_id": p.id, "split": split})
    return dataclasses.replace(spec, meta={**spec.meta, "answer_start": spec.meta["n_tokens"]})


def degeneracy(backend: Any, prompt: _inf.OpenPrompt | str | int) -> PromptSpec:
    """s2: a short open prompt for the 48-token generations."""
    split, p = _open_prompt(prompt, "degeneracy")
    return _single_turn(backend, builder="degeneracy", channel="s2", content=p.text,
                        extra={"prompt_id": p.id, "split": split})


def _scored_text(backend: Any, *, builder: str, request: str, passage: str, max_tokens: int | None,
                 extra: Mapping[str, Any]) -> PromptSpec:
    prompt = _render(backend, [{"role": "user", "content": request}])
    text = prompt + passage
    offsets = _tokens(backend, text)
    start = next((i for i, (s, _) in enumerate(offsets) if s >= len(prompt)), None)
    if start is None or start < 1:
        raise ValueError("the passage produced no tokens of its own")
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        for _ in range(8):  # a cut at a token boundary can retokenize; tighten until it fits
            if len(offsets) - start <= max_tokens:
                break
            text = text[: offsets[start + max_tokens - 1][1]]
            offsets = _tokens(backend, text)
        else:  # pragma: no cover
            raise ValueError("could not cut the passage to max_tokens at a token boundary")
    meta = {"builder": builder, "request": request, "start_token": start, "n_tokens": len(offsets),
            "n_scored": len(offsets) - start, "mode": "all"}
    meta.update(extra)
    return PromptSpec(text, "s6", None, None, meta)


def neutral_text_for_s6(backend: Any, text: _inf.NeutralText | str | int, *,
                        max_tokens: int | None = None) -> PromptSpec:
    """s6: a neutral passage placed as the start of the assistant's reply to `S6_REQUEST`; KL is
    scored over tokens from `meta["start_token"]`. `max_tokens` is `cfg["S6_TOKENS"]`."""
    if isinstance(text, _inf.NeutralText):
        nt = text
    elif isinstance(text, bool):
        raise TypeError("a neutral text is a NeutralText, an id or an index")
    elif isinstance(text, int):
        nt = _inf.NEUTRAL_TEXTS[text]
    else:
        matches = [t for t in _inf.NEUTRAL_TEXTS if t.id == text]
        if not matches:
            raise KeyError(f"unknown neutral text {text!r}")
        nt = matches[0]
    return _scored_text(backend, builder="neutral_text_for_s6", request=_inf.S6_REQUEST, passage=nt.text,
                        max_tokens=max_tokens, extra={"text_id": nt.id, "source": "neutral"})


def own_text_for_s6(backend: Any, prompt_text: str, continuation: str, *, max_tokens: int | None = None,
                    prompt_id: str | None = None) -> PromptSpec:
    """s6's second half: the model's own unsteered continuation of `prompt_text`, scored the same way."""
    if not continuation:
        raise ValueError("an empty continuation has nothing to score")
    return _scored_text(backend, builder="own_text_for_s6", request=prompt_text, passage=continuation,
                        max_tokens=max_tokens, extra={"prompt_id": prompt_id, "source": "own"})


def capability_item(backend: Any, item: _cap.CapabilityItem) -> PromptSpec:
    """s5: one capability item; `item.check(reply)` scores the generation."""
    return _single_turn(backend, builder="capability_item", channel="capability", content=item.question,
                        extra={"item_id": item.id, "family": item.family, "kind": item.kind,
                               "answer": item.answer, "difficulty": item.difficulty,
                               "source": item.source, "params": [list(p) for p in item.params]})


def scenario(backend: Any, sc: _scn.Scenario | str) -> PromptSpec:
    """a1 / a2: one evaluation-like or deployment-like scenario. The class is in meta, never in the text."""
    s = sc if isinstance(sc, _scn.Scenario) else _scn.by_id(sc)
    return _single_turn(backend, builder="scenario", channel="scenario", content=s.text,
                        extra={"scenario_id": s.id, "cls": s.cls})


def ab_item(backend: Any, item: _con.ABItem) -> PromptSpec:
    """e10: an A/B item in CAA's format, the reply opened with "(" so the next token is the option
    letter. `steer_from` is the first token after the user's content: CAA steers every position
    after the end of the instruction."""
    return _single_turn(backend, builder="ab_item", channel="e10", content=item.question, prefill="(",
                        steer_from_answer=True,
                        extra={"item_id": item.id, "options": ["A", "B"],
                               "matching": item.matching_letter,
                               "not_matching": item.not_matching.strip("()")})
