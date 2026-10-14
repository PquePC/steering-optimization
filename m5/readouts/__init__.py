"""m5.readouts - the screen tier (plan §3.1): forward passes and short mechanical generations.

One function per code, `read_<code>(ctx, cell, *, n=None, variant="original", mode=None,
pass_=None) -> m5.results.Readout`, each writing one `screen` row (numbers and ids, never text):

    effect        e9   teacher-forced concept lift, the entropy change beside it      effect.py
                  e10  behavioural propensity shift on A/B items                       effect.py
    integrity     s2   degenerate share of short generations (tier-0 rules)            integrity.py
                  s6   KL(unsteered || steered) on neutral and own text, the concept
                       removed and the bleed share beside it                           integrity.py
                  s7   refusal-token proxy (gated: S7_ENABLED, prompts outside the
                       repository)                                                     integrity.py
    naming        d3   forced naming from logits                                       naming.py
                  d5   open-slot control; d3's row carries the excess d3 - d5          naming.py
                  d6   past-only naming: the forced slot and a K-way decoy list        naming.py
    noticing      d7   bias-controlled yes/no (difference-in-differences)              noticing.py
                  d9   localisation: which segment, if any                             noticing.py
    registration  q1   cross-vector probe at the answer position (record-only)         probe.py
                  q2   logit lens (record-only)                                        lens.py

Shared by all of them (`_common`): a `None` cell is the unsteered null arm; every steered
quantity is computed beside its unsteered counterpart on the same prompts, whose values are cached
per prompt per run; options are scored by their best surface form; a word's probability is
summed over its whole token sequences; mean and SE are across distinct prompts; missing is never
zero; every pass is charged to the context.

`mode` (an `m5.steer` mode) defaults to what the cell's own position policy means for the prompt
(`Cell.positions`; "from_marker" is Macar's live start on detection prompts and every position on
open prompts). d6 is past-only and d9 one segment by design. `pass_` ("A" / "B") sets the default
sizes of e9 and s6 (SCREEN_A_* / SCREEN_B_*) and is recorded on every row.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from m5.codes import code as _code
from m5.intervention import Cell
from m5.readouts.effect import E10_SETS, read_e9, read_e10
from m5.readouts.integrity import REFUSAL_OPENERS, load_s7_prompts, read_s2, read_s6, read_s7
from m5.readouts.lens import read_q2
from m5.readouts.naming import read_d3, read_d5, read_d6
from m5.readouts.noticing import D7Controls, d7_controls, read_d7, read_d9
from m5.readouts.probe import Q1Example, Q1Probe, q1_examples, read_q1, train_q1, trained_probe
from m5.results import Readout

READOUTS: dict[str, Callable[..., Readout]] = {
    "e9": read_e9, "e10": read_e10,
    "s2": read_s2, "s6": read_s6, "s7": read_s7,
    "d3": read_d3, "d5": read_d5, "d6": read_d6,
    "d7": read_d7, "d9": read_d9,
    "q1": read_q1, "q2": read_q2,
}

# Readouts whose prompts are the detection conversation, and so have elicitation variants.
VARIANT_AWARE = frozenset({"d3", "d5", "d6", "d7", "d9", "q1", "q2"})

# d5 before d3, so d3's row can carry the introspective excess d3 - d5.
_ORDER = {"d5": 0, "d3": 1}


def _parse(key: str) -> tuple[str, str | None]:
    base, _, mode = key.partition(":")
    _code(base)  # refuses retired and unknown codes by name
    if base not in READOUTS:
        raise KeyError(f"{base} is not a screen readout (m5.readouts.READOUTS: {sorted(READOUTS)}); "
                       "measure-tier codes live in m5.measures")
    return base, (mode or None)


def run_readouts(ctx: Any, cell: Cell | None, codes: Sequence[str], **kw: Any) -> dict[str, Readout]:
    """Run the readouts named in `codes` on one cell; return {code as given: Readout}.

    A code may carry a mode after a colon ("d7:past_only"), so one call reads d7's live and
    past-only rows; such a code's mode overrides `mode`. `n` may be a mapping {code: n} (codes as
    given) or one int for every readout. `variant` goes only to the readouts whose prompts have
    variants (VARIANT_AWARE). Any other keyword goes to the readouts that take it, and one that no
    requested readout takes raises: a misspelt option must not be silently dropped."""
    keys = list(codes)
    if len(set(keys)) != len(keys):
        raise ValueError(f"run_readouts: repeated codes in {keys}")
    parsed = {k: _parse(k) for k in keys}
    n_arg = kw.pop("n", None)
    if isinstance(n_arg, Mapping):
        unknown = sorted(set(n_arg) - set(keys))
        if unknown:
            raise KeyError(f"run_readouts: n names codes that were not requested: {unknown}")
    variant = kw.pop("variant", "original")
    mode_arg = kw.pop("mode", None)
    params = {k: inspect.signature(READOUTS[b]).parameters for k, (b, _) in parsed.items()}
    unused = [name for name in kw if not any(name in p for p in params.values())]
    if unused:
        raise TypeError(f"run_readouts: no requested readout takes {unused}")
    out: dict[str, Readout] = {}
    for key in sorted(keys, key=lambda k: (_ORDER.get(parsed[k][0], 2), keys.index(k))):
        base, mode = parsed[key]
        args: dict[str, Any] = {name: v for name, v in kw.items() if name in params[key]}
        args["n"] = n_arg.get(key) if isinstance(n_arg, Mapping) else n_arg
        if base in VARIANT_AWARE:
            args["variant"] = variant
        if mode is not None or mode_arg is not None:
            args["mode"] = mode if mode is not None else mode_arg
        out[key] = READOUTS[base](ctx, cell, **args)
    return {k: out[k] for k in keys}


__all__ = ["READOUTS", "VARIANT_AWARE", "run_readouts", "read_e9", "read_e10", "read_s2", "read_s6",
           "read_s7", "read_d3", "read_d5", "read_d6", "read_d7", "read_d9", "read_q1", "read_q2",
           "train_q1", "q1_examples", "trained_probe", "Q1Example", "Q1Probe", "d7_controls",
           "D7Controls", "E10_SETS", "REFUSAL_OPENERS", "load_s7_prompts"]
