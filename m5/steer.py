"""m5.steer - turn a cell and a prompt into the intervention that steers that prompt.

Every readout and measure goes through `intervention_for`, so the mapping from "this cell" to
"these alphas at these token positions" exists once. Position modes:

    live       Positions("from", spec.steer_from): from the token before "Trial n" onward,
               including every generated token (Lindsey / Macar)
    past_only  Positions("span", *spec.span): only the briefing the prompt marks as past; the
               question and the answer are not steered (Pearson et al.)
    all        Positions("all"): every token after BOS (open-ended prompts with no marker)
    generated  Positions("generated"): generated tokens only

A `None` cell is the unsteered null arm and returns `None`.

Vectors are registered with the backend under the id the intervention names: the vector's own
id, or `<id>~perp` for its nuisance-masked copy when MASK_LOUD_IN_VECTOR is on
(`m5.dose.applied_vector_id`), so a masked and an unmasked run can never steer with each other's
values.
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
    """Load a vector, register the values the run applies (masked or not) once, return the
    stored (unmasked) values, which the dose arithmetic reads."""
    from m5 import dose

    if ctx.store is None:
        raise ValueError("the context has no vector store")
    vectors = ctx.store.load(vector_id)
    mask = ctx.cfg["MASK_LOUD_IN_VECTOR"]
    applied_id = dose.applied_vector_id(vector_id, mask)
    if not ctx.backend.has_vector(applied_id):
        ctx.backend.register_vector(applied_id, dose.applied_vectors(vectors, ctx.profile, mask))
    return vectors


def schedule_weights(ctx: Any, cell: Cell) -> dict[int, float] | None:
    """The per-layer shares of a cell's schedule, checked against the cell's own layers.

    Single, strided and all-layer cells use equal shares by definition (None lets
    `m5.dose.cell_to_intervention` apply them); a Gaussian cell needs its profile's shares.
    """
    if cell.schedule != "gaussian":
        return None
    from m5 import dose

    pairs = dose.schedule_layers_cfg(ctx.cfg, ctx.backend.info.n_layers, None)
    weights = {int(L): float(w) for L, w in pairs}
    if set(weights) != set(cell.layers):
        raise ValueError(f"{cell.cell_id()}: its layers do not match the configured Gaussian schedule")
    return weights


def intervention_for(ctx: Any, cell: Cell | None, spec: Any, mode: str = "live") -> Intervention | None:
    """The intervention that puts `cell` on `spec` under `mode`, or None for the null arm."""
    if cell is None:
        return None
    from m5 import dose

    vectors = ensure_registered(ctx, cell.vector_id)
    return dose.cell_to_intervention(cell, ctx.profile, vectors, positions_for(spec, mode),
                                     ctx.cfg["MASK_LOUD_IN_VECTOR"],
                                     weights=schedule_weights(ctx, cell))
