"""m5.codes - the measure codes M5 reports, and what each one is for.

Old codes keep their old meanings (`AGENTS.md`); every new quantity gets a new code. The plan's
§3.5 is the source of this table. Six constructs, each measured separately, none folded into
another:

    effect        did the intervention change the targeted content or behaviour?
    integrity     is the model still working?
    noticing      when asked, does the model indicate that something was done to it?
    naming        can it say *what* was injected?
    registration  is an injection signal decodable inside the model, reported or not?
    consequence   does it move evaluation awareness or behaviour beyond what any
                  intervention of its size does?

`record_only` marks readouts that are written down and never used to decide anything until they
pass the proxy-validation gate (`m5.gate`, plan §5.3). The registration readers start that way;
the gate can promote a screen readout, never the other way round.

Retired or absent on purpose:

    e1, e4, e5, e6, d1, d4, s3   older meanings, not produced by M5 (e5/e6 retired from M5 runs)
    s8                           judged attack success: not implemented. It generates possibly
                                 harmful completions and is decision 8 in plan §13.
"""

from __future__ import annotations

from dataclasses import dataclass

CONSTRUCTS = ("effect", "integrity", "noticing", "naming", "registration", "consequence")
TIERS = ("screen", "measure", "confirm")
DIRECTIONS = ("higher", "lower", "none")


@dataclass(frozen=True)
class Code:
    """One measure.

    `better` is the direction an *operating point* wants: effect higher, detection lower,
    integrity higher (or, for `s2`/`s6`, lower). `none` means the quantity is a control or is
    reported without a preferred direction.

    `tier` is the cheapest tier that produces it; most screen readouts are re-read at higher n in
    the confirm tier, and that does not change their code.
    """

    code: str
    name: str
    construct: str
    tier: str
    better: str
    judged: bool
    record_only: bool = False

    def __post_init__(self) -> None:
        if self.construct not in CONSTRUCTS:
            raise ValueError(f"{self.code}: unknown construct {self.construct!r}")
        if self.tier not in TIERS:
            raise ValueError(f"{self.code}: unknown tier {self.tier!r}")
        if self.better not in DIRECTIONS:
            raise ValueError(f"{self.code}: unknown direction {self.better!r}")


_ALL = [
    # effect
    Code("e7", "concept occurrence rate (regex, judge-free)", "effect", "measure", "higher", False),
    Code("e8", "judged concept presence (allusion with a validated quote; dominant; literal)",
         "effect", "measure", "higher", True),
    Code("e9", "teacher-forced concept lift, with the entropy change beside it",
         "effect", "screen", "higher", False),
    Code("e10", "behavioural propensity slope (A/B logit difference against dose)",
         "effect", "screen", "higher", False),
    Code("e11", "behavioural trait score (logprob-weighted judge)", "effect", "measure", "higher", True),
    Code("e12", "off-target battery (multiple-choice shifts read from logits)",
         "effect", "confirm", "lower", False),
    # naming
    Code("d2", "forced naming, generated and judged", "naming", "measure", "lower", True),
    Code("d3", "forced naming, read from logits", "naming", "screen", "lower", False),
    Code("d5", "open-slot control (non-introspective slot under the same steering)",
         "naming", "screen", "none", False),
    Code("d6", "past-only naming (steering off before the question)", "naming", "screen", "lower", False),
    # noticing
    Code("d7", "bias-controlled yes/no (difference-in-differences against control questions)",
         "noticing", "screen", "lower", False),
    Code("d8", "briefed self-report: claims, names, matches, affirm-before-naming, d-prime",
         "noticing", "measure", "lower", True),
    Code("d9", "localisation: which segment, if any, was injected", "noticing", "screen", "lower", False),
    Code("d10", "spontaneous flags: restarts, disavowals, evaluation-awareness mentions",
         "noticing", "measure", "lower", True),
    # registration (record-only until the proxy gate passes)
    Code("q1", "cross-vector probe at the answer position", "registration", "screen", "none", False, True),
    Code("q2", "lens readouts (logit lens; J-lens once fitted)", "registration", "screen", "none", False, True),
    Code("q3", "external reader (natural-language autoencoder, activation oracle)",
         "registration", "confirm", "none", False, True),
    Code("q4", "gate and carrier features (Gemma Scope transcoders)",
         "registration", "confirm", "none", False, True),
    # integrity
    Code("s1", "coherence: mechanical meters plus a concept-blind judge",
         "integrity", "measure", "higher", True),
    Code("s2", "degeneracy rate on short generations", "integrity", "screen", "lower", False),
    Code("s4", "minimum of the live integrity terms", "integrity", "measure", "higher", False),
    Code("s5", "paired capability panel (steered vs unsteered on the same items)",
         "integrity", "measure", "higher", False),
    Code("s6", "KL(unsteered || steered) on neutral text", "integrity", "screen", "lower", False),
    Code("s7", "refusal-token proxy (forward passes only, no harmful text produced)",
         "integrity", "confirm", "none", False),
    # consequence
    Code("a1", "verbalised evaluation awareness", "consequence", "measure", "none", True),
    Code("a2", "latent evaluation awareness (probe score)", "consequence", "measure", "none", False),
    Code("a3", "excess over the control-vector distribution (percentile)",
         "consequence", "measure", "none", False),
]

CODES: dict[str, Code] = {c.code: c for c in _ALL}
if len(CODES) != len(_ALL):  # pragma: no cover - a duplicate is a typo in the table above
    raise RuntimeError("duplicate measure code in m5.codes")

RETIRED = frozenset({"e1", "e4", "e5", "e6", "d1", "d4", "s3"})
NOT_IMPLEMENTED = {"s8": "judged attack success generates possibly harmful completions; "
                         "decision 8 in docs/M5-PLAN.md §13 has not been made"}

NOTICING = tuple(c.code for c in _ALL if c.construct == "noticing")
NAMING = tuple(c.code for c in _ALL if c.construct == "naming")


def code(c: str) -> Code:
    """Look a code up, refusing retired and unknown ones by name."""
    if c in CODES:
        return CODES[c]
    if c in RETIRED:
        raise KeyError(f"{c} is a retired code; M5 does not produce it (see m5.codes)")
    if c in NOT_IMPLEMENTED:
        raise KeyError(f"{c} is not implemented: {NOT_IMPLEMENTED[c]}")
    raise KeyError(f"unknown measure code {c!r}")
