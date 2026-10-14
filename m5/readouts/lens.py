"""m5.readouts.lens - q2, the lens readouts: is the concept *poised to be said* inside the model?

Record-only (`m5.codes`: q2 decides nothing until it passes the proxy-validation gate, plan §5.3).

**Logit lens.** On the forced-naming prompt (the same spec d3 reads), the residual at the answer
position (the prompt's last token, right after "The thought is about") is sent through the final
norm and the unembedding at every layer from the cell's first injection layer to the last
(`Backend.lens`). Per layer, the log-probability mass of the concept's first tokens is read
steered and unsteered; the per-layer lift, averaged over trial numbers, is the curve, and its
maximum over layers is the value. Pearson et al. find the signal attenuated in the final layers,
which is why the curve is kept rather than only its last point. Generation positions (plan §3.1:
"at the answer position and at generation positions") are not read: the lens reads a prompt
position, and generating a reply to read would move this readout out of the cheap tier.

**J-lens.** Not implemented. When the profile carries a fitted J-lens (`ModelProfile.j_lens`,
{layer: [k, d_model] matrix whose rows span the verbalisable subspace}; the optional setup step of
plan §5.1), `f_J` -- the share of the vector's squared norm (nuisance coordinates zeroed) inside
that subspace at each injected layer -- is reported, since it needs only the vector. The J-lens
concept score itself needs the unembedding, which the Backend protocol does not expose, so it is
missing with that reason either way.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from m5 import prompts
from m5.intervention import Cell
from m5.readouts._common import (GENERATED_ONLY, NULL_CONCEPT, check_args, concept_tokens, emit,
                                 logsumexp_rows, mean_se, nameless, peek, put, resolve_mode,
                                 trial_numbers)
from m5.results import Readout
from m5.steer import intervention_for

Q2_MODES = ("live", "past_only", "all", "generated")

J_LENS_SCORE_MISSING = ("the J-lens concept score needs the unembedding, which the Backend "
                        "protocol does not expose")
J_LENS_UNFITTED = ("J-lens not fitted: the profile carries no j_lens selection (plan §5.1, optional "
                   "setup step), so f_J is not computed")


def _null_lens(ctx: Any, texts: list[str], token_ids: tuple[int, ...]) -> np.ndarray:
    """[len(texts), n_layers, len(token_ids)] unsteered lens log-probs at the last prompt token,
    every layer, computed once per prompt per run."""
    n_layers = ctx.backend.info.n_layers
    todo = [t for t in dict.fromkeys(texts) if peek(ctx, ("lens-null", t, token_ids)) is None]
    if todo:
        ctx.charge(forward=len(todo))
        res = ctx.backend.lens(todo, [None] * len(todo), list(range(n_layers)), -1, token_ids)
        for t, arr in zip(todo, res):
            put(ctx, ("lens-null", t, token_ids), np.asarray(arr, dtype=np.float64))
    return np.stack([peek(ctx, ("lens-null", t, token_ids)) for t in texts])


def _j_lens_bases(ctx: Any) -> dict[int, np.ndarray]:
    """The fitted J-lens bases the profile names: `{"path", "sha256"}` of an .npz under RUN_ROOT
    with one array `L<layer>` per layer. Matrices never live in the profile itself."""
    from m5.safety import assert_outside_repo
    sel = ctx.profile.selection("j_lens")
    path = assert_outside_repo(Path(sel["path"]), what="J-lens file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sel["sha256"]:
        raise ValueError(f"q2: {path} does not match the J-lens digest the profile records")
    with np.load(path) as z:
        return {int(k[1:]): np.asarray(z[k]) for k in z.files}


def _f_j(ctx: Any, cell: Cell) -> dict[str, Any]:
    if not ctx.profile.has_selection("j_lens"):
        return {"missing": J_LENS_UNFITTED, "score_missing": J_LENS_SCORE_MISSING}
    fitted = _j_lens_bases(ctx)
    vectors = ctx.store.load(cell.vector_id)
    out: dict[str, Any] = {}
    for L in cell.layers:
        if L not in fitted:
            out[str(L)] = None
            continue
        basis = np.asarray(fitted[L], dtype=np.float64)
        if basis.ndim != 2 or basis.shape[1] != vectors[L].shape[-1]:
            raise ValueError(f"q2: the J-lens matrix at layer {L} has shape {basis.shape}; expected "
                             f"[k, {vectors[L].shape[-1]}]")
        q, _ = np.linalg.qr(basis.T)
        v = np.asarray(vectors[L], dtype=np.float64).copy()
        v[list(ctx.profile.nuisance(L))] = 0.0
        nv = float(v @ v)
        out[str(L)] = float(np.sum((q.T @ v) ** 2) / nv) if nv > 0 else None
    return {"f_J": out, "score_missing": J_LENS_SCORE_MISSING}


def read_q2(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str | None = None, pass_: str | None = None) -> Readout:
    """q2 (logit-lens lift of the concept's tokens at the answer position: the maximum over layers,
    from the first injected layer to the last, of the mean steered-minus-unsteered log-mass;
    nats; record-only). n trial numbers (all by default)."""
    code = "q2"
    check_args(code, variant=variant, variant_aware=True, pass_=pass_)
    mode = resolve_mode(code, cell, mode, kind="marker", allowed=Q2_MODES)

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    if mode == "generated":
        return out(missing=GENERATED_ONLY)
    if cell is None:
        return out(missing=NULL_CONCEPT)
    concept = ctx.concept_of(cell.vector_id)
    if concept is None:
        return out(missing=nameless(code, cell.vector_id))
    toks = concept_tokens(ctx.backend, concept)
    if not toks.ids:
        return out(missing=f"no variant of {concept!r} has a usable first token")
    trials = trial_numbers(ctx, code, n)
    specs = [prompts.forced_naming(ctx.backend, t, variant=variant, channel="d3") for t in trials]
    texts = [s.text for s in specs]
    n_layers = ctx.backend.info.n_layers
    first = min(cell.layers)
    layers = list(range(first, n_layers))
    ivs = [intervention_for(ctx, cell, s, mode) for s in specs]
    ctx.charge(forward=len(specs))
    lens_s = np.asarray(ctx.backend.lens(texts, ivs, layers, -1, toks.ids), dtype=np.float64)
    lens_u = _null_lens(ctx, texts, toks.ids)[:, first:, :]
    mass_s = logsumexp_rows(lens_s)            # [prompts, layers]
    mass_u = logsumexp_rows(lens_u)
    lift = mass_s - mass_u
    keep = np.all(np.isfinite(lift), axis=1)
    excluded = {str(t): "a non-finite lens log-probability" for t, ok in zip(trials, keep) if not ok}
    if not keep.any():
        return out(missing="no trial gave a finite lens curve",
                   extra={"excluded": len(excluded), "excluded_units": excluded})
    lift, mass_u = lift[keep], mass_u[keep]
    curve = []
    for j, L in enumerate(layers):
        m, se = mean_se(lift[:, j])
        curve.append({"layer": L, "lift": m, "se": se, "null_logmass": float(np.mean(mass_u[:, j]))})
    peak = int(np.argmax([c["lift"] for c in curve]))
    return out(value=curve[peak]["lift"], n=int(keep.sum()), extra={
        "se": curve[peak]["se"], "peak_layer": curve[peak]["layer"],
        "last_layer_lift": curve[-1]["lift"], "curve": curve, "concept": concept,
        "token_ids": list(toks.ids), "position": "last prompt token (after the forced prefill)",
        "j_lens": _f_j(ctx, cell), "excluded": len(excluded), "excluded_units": excluded})
