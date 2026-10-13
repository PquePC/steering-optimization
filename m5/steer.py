"""m5.steer - turn a cell and a prompt into the intervention that steers that prompt.

Every readout and measure goes through `intervention_for`, so the mapping from "this cell" to
"these alphas at these token positions" exists once. Three position modes:

    live       Positions("from", spec.steer_from): from the token before "Trial n" onward,
               including every generated token (Lindsey / Macar)
    past_only  Positions("span", *spec.span): only the briefing the prompt marks as past; the
               question and the answer are not steered (Pearson et al.)
    all        Positions("all"): every token after BOS (open-ended prompts with no marker)

A `None` cell is the unsteered null arm and returns `None`.
"""

from __future__ import annotations

from typing import Any

from m5.intervention import Cell, Intervention, Positions

MODES = ("live", "past_only", "all", "generated")


def positions_for(spec: Any, mode: str) -> Positions:
    """The Positions a prompt spec implies under a mode; raises when the spec cannot support it."""
    if mode == "live":
        if spec.steer_from is None:
            raise ValueError(f"{spec.channel}: no steering start marked; use mode 'all'")
        return Positions("from", spec.steer_from)
    if mode == "past_only":
        if spec.span is None:
            raise ValueError(f"{spec.channel}: no past-only span marked")
        return Positions("span", *spec.span)
    if mode == "all":
        return Positions("all")
    if mode == "generated":
        return Positions("generated")
    raise ValueError(f"unknown steering mode {mode!r}; one of {MODES}")


def ensure_registered(ctx: Any, vector_id: str) -> dict:
    """Load a vector from the store and register it with the backend once; return its values."""
    if ctx.store is None:
        raise ValueError("the context has no vector store")
    vectors = ctx.store.load(vector_id)
    if not ctx.backend.has_vector(vector_id):
        ctx.backend.register_vector(vector_id, vectors)
    return vectors


def intervention_for(ctx: Any, cell: Cell | None, spec: Any, mode: str = "live") -> Intervention | None:
    """The intervention that puts `cell` on `spec` under `mode`, or None for the null arm."""
    if cell is None:
        return None
    from m5 import dose  # local: keeps this module importable before dose exists in a test

    vectors = ensure_registered(ctx, cell.vector_id)
    result = dose.cell_to_intervention(cell, ctx.profile, vectors, positions_for(spec, mode),
                                       mask_loud=ctx.cfg["MASK_LOUD_IN_VECTOR"])
    return result[0] if isinstance(result, tuple) else result
