"""m2.multilayer - optional arm: spread the injection across k layers instead of one.

THE HYPOTHESIS. A single-layer injection at the dose that produces real influence is often
also detectable. Splitting the same total dose across k layers might keep the influence
(because the total perturbation is unchanged) while lowering forced identification (because no
single layer is perturbed hard enough to trip whatever notices it).

WHAT THE LITERATURE PREDICTS, and it is not encouraging. `Steering Awareness - Detecting
Activation Steering from Within` finds that steering awareness is "mechanistically explained by
a **distributed transport into a shared direction in the final layer**": the perturbation
introduced at the injection layer rotates monotonically toward one detection direction as it
propagates, `cos(delta_c, d_detect)` rising from ~0.2 at the injection layer to near 1 by the
final layer, with standard deviation < 0.05 across 18 held-out concepts (their Figure 8).

If that is right, k injections do not hide from each other. They each rotate into the *same*
final-layer direction and **add there**, so at matched total dose the projection onto the
detection direction is roughly preserved - and the contiguous-window variant is the worst case,
because layers close together have almost the same amount of depth left to rotate through.

So this arm is a **test of a specific prediction**, not an optimisation expected to work:

  - if D2 falls at matched E5, the distributed-transport account is incomplete, and that is a
    substantially more interesting result than the operating point itself;
  - if D2 does not fall, that is a clean confirmation of the account on a model and a concept
    family it was not fitted on.

Either outcome is publishable. What would not be defensible is running it, finding nothing, and
filing it as a failed engineering attempt - so `compare_to_single` writes the matched
single-layer baseline into every row.

The wider steering literature is also lukewarm on multi-layer generally: single-layer steering
is usually preferred as "computationally more efficient, introduc[ing] fewer hyperparameters,
and avoid[ing] the need to consider interactions between interventions at multiple locations",
and multi-vector work reports that stacking interventions "can compound destabilisation". The
sanity terms are what will catch that here, and S4 = min(S1, S2, S3) is deliberately unforgiving.

DEPENDENCY POSITION. After `expensive` (whose judging machinery it reuses) and before `gates`.
Rows are written with a local `_append_row` matching the shape the other modules use, since
`runio` comes later in the CONTRACT order.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import cheap
from . import config
from . import expensive
from . import model
from . import prompts
from . import vectors


# k values the arm sweeps. The user's three, in the order they get more aggressive.
DEFAULT_K: tuple[int, ...] = (2, 3, 5)

# Placement strategies.
#   "window"  contiguous layers centred on the best single layer (L37 -> 35..39 at k=5).
#             Tests "does smearing the same site help?"
#   "top_k"   the k best distinct layers from the scan, which may be far apart.
#             Tests "does using several genuinely different sites help?"
PLACEMENTS: tuple[str, ...] = ("window", "top_k")

# Dose split conventions. Given a total dose R over k layers:
#   "equal"  each layer gets R/k. Total is conserved IF the contributions add coherently -
#            which is exactly what the distributed-transport account says happens. This is the
#            matched comparison and the default.
#   "sqrt"   each layer gets R/sqrt(k). Total is conserved if the contributions were
#            ORTHOGONAL. Included as the alternative null: if transport is not coherent, this
#            is the matched one instead, and running both brackets the truth.
#   "full"   each layer gets R. k times the perturbation - not a matched comparison, kept only
#            because it is the obvious thing someone will want to try and it should be labelled
#            rather than discovered by accident.
SPLITS: tuple[str, ...] = ("equal", "sqrt", "full")

MULTILAYER_FILE = "multilayer.jsonl"
_R_PLACES = 4


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for m2.multilayer") from exc
    return torch


def _run() -> Any:
    ctx = getattr(config, "RUN", None)
    if ctx is None:
        raise RuntimeError("m2.config.RUN is not set - load the model and set a concept first")
    return ctx


def _cfg() -> dict:
    ctx = getattr(config, "RUN", None)
    if ctx is not None and ctx.config:
        return ctx.config
    return config.CONFIG


def _const(key: str) -> Any:
    cfg = _cfg()
    if key not in cfg:
        raise KeyError(f"{key} missing from CONFIG (spec section 11)")
    return cfg[key]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_row(name: str, row: dict) -> Path:
    ctx = _run()
    if ctx.run_dir is None:
        raise RuntimeError("RUN.run_dir is not set - call m2.driver.set_concept(name) first")
    path = Path(ctx.run_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(row)
    out.setdefault("concept", ctx.concept)
    out.setdefault("config_hash", _cfg().get("config_hash"))
    out.setdefault("ts", _ts())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
    return path


# =====================================================================================
# building a plan
# =====================================================================================

def dose_per_layer(total_r: float, k: int, split: str) -> float:
    """The dose each of `k` layers receives, under one of the SPLITS conventions."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    total_r = float(total_r)
    if split == "equal":
        return total_r / k
    if split == "sqrt":
        return total_r / math.sqrt(k)
    return total_r


def choose_layers(k: int, placement: str, *, best_layer: int,
                  ranked_layers: Sequence[int] | None = None,
                  n_layers: int | None = None) -> list[int]:
    """The k layers a plan will steer.

    `window` centres on `best_layer` and takes the k nearest layers, clipped to the model. For
    even k the window is taken one layer deeper rather than shallower, because detection is
    concentrated late (L37 in the v1 collapse test) and the deeper side is where the question
    is interesting.

    `top_k` takes the first k entries of `ranked_layers` - the caller's ranking, normally the
    Phase 2 shortlist ordered by verified E5 or by scan reachability.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    ctx = _run()
    n_layers = int(n_layers if n_layers is not None else ctx.n_layers)

    if placement == "top_k":
        if not ranked_layers:
            raise ValueError("placement='top_k' needs ranked_layers (the shortlist, in order)")
        chosen = list(dict.fromkeys(int(x) for x in ranked_layers))[:k]
        if len(chosen) < k:
            raise ValueError(
                f"only {len(chosen)} distinct ranked layers available, need {k}. Widen the "
                "shortlist or drop this k rather than repeating a layer - steering one layer "
                "twice is a dose change dressed up as a placement change.")
        return sorted(chosen)

    half = (k - 1) // 2
    lo = int(best_layer) - half
    layers = [lo + i for i in range(k)]
    # Clip into range by sliding, not by truncating: a truncated window would silently run at
    # a smaller k than the row claims.
    if layers[0] < 0:
        layers = [x - layers[0] for x in layers]
    if layers[-1] > n_layers - 1:
        layers = [x - (layers[-1] - (n_layers - 1)) for x in layers]
    if layers[0] < 0:
        raise ValueError(f"k={k} does not fit in a {n_layers}-layer model")
    return layers


def build_plan(concept: str, layers: Sequence[int], r_each: float) -> dict:
    """`{layer: (vector, alpha)}` plus a record of how it was derived.

    Each layer's alpha is computed from ITS OWN norms, because a layer's vector norm and
    residual norm both change with depth - handing every layer the same alpha would inject
    wildly different doses and make the arm a test of the normalisation instead (spec 3).
    """
    ctx = _run()
    layers = [int(x) for x in layers]
    missing = [x for x in layers if x not in ctx.vecs]
    if missing:
        ctx.vecs.update(vectors.extract_all_layers(concept, missing))
    for layer in layers:
        if layer not in ctx.norms:
            raise KeyError(
                f"no calibrated norms at layer {layer} - Phase 0 measures them, and without "
                f"||h_{layer}|| the dose at this layer is undefined")

    plan, alphas, unreachable = {}, {}, []
    ceil = float(_const("ALPHA_CEIL"))
    for layer in layers:
        vec = ctx.vecs[layer]
        alpha = vectors.alpha_from_norms(float(r_each),
                                         float(ctx.norms[layer]["vec_norm"]),
                                         float(ctx.norms[layer]["resid_norm"]))
        if alpha > ceil:
            unreachable.append(dict(layer=layer, alpha=alpha))
            continue
        plan[layer] = (vec, float(alpha))
        alphas[layer] = float(alpha)

    return dict(plan=plan, alphas=alphas, layers=layers, r_each=float(r_each),
                unreachable=unreachable,
                vec_fingerprints={l: vectors.vec_fingerprint(ctx.vecs[l]) for l in layers})


def plan_id(concept: str, layers: Sequence[int], r_each: float, split: str,
            placement: str) -> str:
    """A stable label for a plan, used as the judge-cache unit prefix and in the row."""
    body = "-".join(str(int(x)) for x in sorted(layers))
    return f"ml{len(list(layers))}_{placement}_{split}_L{body}_r{r_each:.4f}"


# =====================================================================================
# generation under a multi-layer plan
# =====================================================================================

def steering_mask(prompts_text: Sequence[str], starts: Sequence[int]) -> Any:
    """`[batch, seq]` bool mask: True where each row should be steered.

    Built from the left-padded encoding, so row i's start position is shifted by its own pad
    count. This is the correction `generate_batch_with_multi_steering` applies internally at
    `model_utils.py:1248` for the one layer it owns; multi-layer has no equivalent, so it is
    computed once here and handed to every hook.

    Getting this wrong is silent: shorter prompts would be steered from a token too early, and
    the numbers would be plausible. That is bug 25b.
    """
    torch = _require_torch()
    ctx = _run()
    if ctx.tok.padding_side != "left":
        raise RuntimeError(
            f"padding_side is {ctx.tok.padding_side!r}, not 'left'. The mask arithmetic below "
            "assumes left padding, as the rest of the pipeline does.")
    enc = ctx.tok(list(prompts_text), return_tensors="pt", padding=True,
                  add_special_tokens=False)
    attn = enc["attention_mask"]
    batch, seq = attn.shape
    mask = torch.zeros((batch, seq), dtype=torch.bool)
    for i in range(batch):
        pad = int(seq - int(attn[i].sum()))
        start = int(starts[i]) if starts[i] is not None else 0
        if pad + start >= seq:
            raise ValueError(
                f"row {i}: start {start} + padding {pad} >= sequence {seq}; nothing would be "
                "steered on this row")
        mask[i, pad + start:] = True
    return mask


def generate_multi(prompts_text: Sequence[str], plan: dict, starts: Sequence[int],
                   max_new_tokens: int, temperature: float) -> list[str]:
    """Batched generation with every layer in `plan` steered.

    Macar's `generate_batch_with_multi_steering` handles one layer, so this drives
    `model.generate` directly with our own hooks. Two things it must get right, both of which
    that function gets right and the single-vector one does not (bug 25b):

      - per-row steering start, via `steering_mask` above;
      - decoding sliced at the **padded** width, not the unpadded length. Our prompts can end
        in the forced-ID prefill, so an overhang token would be " about" and nothing would
        strip it.
    """
    torch = _require_torch()
    ctx = _run()
    tok, hf = ctx.tok, ctx.hf

    enc = tok(list(prompts_text), return_tensors="pt", padding=True,
              add_special_tokens=False).to(hf.device)
    mask = steering_mask(prompts_text, starts).to(hf.device)
    padded_width = int(enc["input_ids"].shape[1])

    with model.injected_multi(plan, mask=mask):
        with torch.no_grad():
            out_ids = hf.generate(
                **enc,
                max_new_tokens=int(max_new_tokens),
                do_sample=temperature > 0,
                temperature=float(temperature) if temperature > 0 else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )

    # Slice at the PADDED width: generate() returns padded input followed by generation.
    return [tok.decode(row[padded_width:], skip_special_tokens=True).strip()
            for row in out_ids]


# =====================================================================================
# measuring a plan
# =====================================================================================

def measure_plan(concept: str, layers: Sequence[int], r_each: float, *,
                 split: str, placement: str, total_r: float,
                 n_d2: int | None = None) -> dict:
    """E5, S1, S2, S3 and D2 for one multi-layer plan. Same judges, same prompts, same N.

    Deliberately reuses `expensive`'s unit builders and scorers rather than re-deriving them:
    the arm is only interpretable if its numbers are produced by the identical instrument the
    single-layer cells were measured with.
    """
    cfg = _cfg()
    n_d2 = int(n_d2 if n_d2 is not None else _const("N_D2"))
    built = build_plan(concept, layers, r_each)
    if built["unreachable"]:
        return dict(ok=False, reason="unreachable", **{k: built[k] for k in
                    ("layers", "alphas", "r_each", "unreachable")},
                    k=len(layers), split=split, placement=placement, total_r=total_r)

    label = plan_id(concept, layers, r_each, split, placement)
    max_new = int(_const("MAX_NEW_TOKENS"))
    temp = float(_const("TEMPERATURE"))

    # --- the task prompts: E5 and S1 share one set of generations (spec 5.6, 5.7) ----------
    rows = list(prompts.E5_PROMPTS)
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    chat_prompts = [model.chat(t) for t in texts]
    starts = [model.start_pos_for(p, t) for p, t in zip(chat_prompts, texts)]
    responses = generate_multi(chat_prompts, built["plan"], starts, max_new, temp)

    fp = "+".join(built["vec_fingerprints"][l] for l in sorted(built["layers"]))
    ctxrec = dict(phase="MULTILAYER", layer=None, r=float(total_r), alpha=None,
                  vec_fingerprint=fp)

    with expensive.phase_scope("MULTILAYER"):
        e5_units = expensive._e5_units(ids, texts, responses, concept=concept,
                                       phase="MULTILAYER", layer=None, r=float(total_r),
                                       vec_fp=fp)
        s1_units = expensive._s1_units(ids, texts, responses, concept=concept,
                                       phase="MULTILAYER", layer=None, r=float(total_r),
                                       vec_fp=fp)
        # --- forced identification -------------------------------------------------------
        trials = list(range(1, n_d2 + 1))
        forced, forced_start = prompts.forced_prompts(trials)
        forced_responses = generate_multi(forced, built["plan"],
                                          [forced_start] * len(forced), max_new, temp)
        d2_units = expensive._d2_units(trials, forced_responses, concept=concept,
                                       phase="MULTILAYER", layer=None, r=float(total_r),
                                       vec_fp=fp)

        all_units = list(e5_units) + list(s1_units) + list(d2_units)
        results = expensive._issue([expensive._item(u["payload"], u["judge_id"], u["cache_key"])
                                    for u in all_units])

    n_e5, n_s1 = len(e5_units), len(s1_units)
    e5 = expensive._score_e5(e5_units, results[:n_e5], ctxrec)
    s1 = expensive._score_s1(s1_units, results[n_e5:n_e5 + n_s1], ctxrec)
    d2 = expensive._score_d2(d2_units, results[n_e5 + n_s1:], ctxrec)

    s2 = cheap.measure_S2(list(responses) + list(forced_responses))
    # S3 under a multi-layer plan needs the same batched forward pass with the plan applied.
    s3 = _s3_multi(built["plan"])

    s4 = min(float(s1["s1"]), float(s2["s2"]), float(s3["s3"]))
    usable = s4 >= float(cfg["S4_MIN"])
    qualifies = bool(usable and float(e5["e5"]) >= float(cfg["E5_FLOOR"])
                     and float(d2["d2"]) <= float(cfg["D2_MAX"]))

    row = dict(
        ok=True, plan_id=label, k=len(built["layers"]), layers=built["layers"],
        alphas=built["alphas"], split=split, placement=placement,
        total_r=float(total_r), r_each=float(r_each), vec_fingerprint=fp,
        e5=e5["e5"], e5_min=e5["e5_min"], e5_se=e5["e5_se"],
        s1=s1["s1"], s2=s2["s2"], s3=s3["s3"], s3_margin=s3["s3_margin"], s4=s4,
        s4_term=("S1" if s4 == s1["s1"] else ("S2" if s4 == s2["s2"] else "S3")),
        d2=d2["d2"], d2_se=d2["d2_se"], n_d2=d2["n_d2"],
        d4=d2["d4"], d4_dominant=d2["d4_dominant"],
        d4_damage_frac=d2["d4_damage_frac"], d4_reading=d2["d4_reading"],
        usable=usable, qualifies=qualifies,
    )
    _append_row(MULTILAYER_FILE, row)
    print(f"multilayer : k={row['k']} {placement}/{split} L{built['layers']} "
          f"r_each={r_each:.3f}  E5={row['e5']:.2f}  S4={s4:.2f}  D2={row['d2']:.2f}  "
          f"{'QUALIFIES' if qualifies else ('usable' if usable else 'unusable')}")
    return row


def _s3_multi(plan: dict) -> dict:
    """S3 with a multi-layer plan applied. Same items, same letter-logit read as spec 5.4."""
    torch = _require_torch()
    ctx = _run()
    items = ctx.mmlu
    texts = [prompts.render_mmlu(item) for item in items]
    enc = ctx.tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(ctx.hf.device)
    if ctx.tok.padding_side != "left":
        raise RuntimeError("S3 batches and reads logits[:, -1, :]; that needs left padding")
    # start_pos is None here (raw text, no chat template), so every position is steered - the
    # same convention single-layer S3 uses.
    with model.injected_multi(plan, mask=None):
        with torch.no_grad():
            logits = ctx.hf(**enc).logits[:, -1, :].float()
    scored = cheap.score_letter_logits(logits, items)

    # Same ratio form as single-layer S3 (spec 5.4): the standing letter/position prior is
    # constant across dose, so dividing by the unsteered count on the same 57 items cancels it.
    if cheap.CAP_BASE_KEY not in ctx.base:
        raise RuntimeError(
            f"RUN.base has no {cheap.CAP_BASE_KEY!r} - Phase 0 must run "
            "cheap.measure_S3_baseline() before the multi-layer arm, or its S3 is not "
            "comparable with any single-layer cell.")
    cap_base = int(ctx.base[cheap.CAP_BASE_KEY])
    return dict(s3_correct=int(scored["correct"]),
                s3=int(scored["correct"]) / max(cap_base, 1),
                s3_margin=scored["margin"], s3_margin_se=scored["margin_se"],
                s3_n=scored["n"], cap_base=cap_base)


# =====================================================================================
# the arm
# =====================================================================================

def run_arm(winner: dict, *, ks: Sequence[int] = DEFAULT_K,
            placements: Sequence[str] = PLACEMENTS,
            splits: Sequence[str] = ("equal",),
            ranked_layers: Sequence[int] | None = None,
            n_d2: int | None = None) -> dict:
    """Sweep k x placement x split at the winner's total dose, and compare to the winner.

    Every row is written with the single-layer baseline it should be read against, because the
    only interesting quantity here is the DIFFERENCE: multi-layer D2 at matched E5.
    """
    ctx = _run()
    concept = str(winner.get("concept") or ctx.concept)
    best_layer = int(winner["layer"])
    total_r = float(winner["r"])

    baseline = dict(layer=best_layer, r=total_r,
                    e5=winner.get("e5"), s4=winner.get("s4"), d2=winner.get("d2"))

    rows: list[dict] = []
    for k in ks:
        for placement in placements:
            try:
                layers = choose_layers(k, placement, best_layer=best_layer,
                                       ranked_layers=ranked_layers)
            except ValueError as exc:
                print(f"multilayer : k={k} {placement} skipped - {exc}")
                continue
            for split in splits:
                r_each = dose_per_layer(total_r, k, split)
                row = measure_plan(concept, layers, r_each, split=split,
                                   placement=placement, total_r=total_r, n_d2=n_d2)
                row["baseline"] = baseline
                if row.get("ok"):
                    row["d_e5"] = (None if baseline["e5"] is None
                                   else float(row["e5"]) - float(baseline["e5"]))
                    row["d_d2"] = (None if baseline["d2"] is None
                                   else float(row["d2"]) - float(baseline["d2"]))
                    row["d_s4"] = (None if baseline["s4"] is None
                                   else float(row["s4"]) - float(baseline["s4"]))
                rows.append(row)

    scored = [r for r in rows if r.get("ok")]
    improved = [r for r in scored
                if r.get("d_d2") is not None and r["d_d2"] < 0
                and r.get("d_e5") is not None and r["d_e5"] > -1.0
                and r.get("usable")]

    verdict = ("spreading_helps" if improved else
               ("no_improvement" if scored else "not_measured"))
    out = dict(concept=concept, baseline=baseline, ks=list(ks),
               placements=list(placements), splits=list(splits),
               n_plans=len(rows), n_scored=len(scored), n_improved=len(improved),
               best=(min(improved, key=lambda r: r["d2"]) if improved else None),
               rows=rows, verdict=verdict,
               prediction=("distributed transport into a shared final-layer direction predicts "
                           "no D2 reduction at matched total dose; a reduction would be the "
                           "more interesting result"))
    _append_row(MULTILAYER_FILE, dict(row_type="arm_summary",
                                      **{k: out[k] for k in
                                         ("baseline", "n_plans", "n_scored", "n_improved",
                                          "verdict")}))
    print("=" * 78)
    print(f"MULTILAYER ARM - {len(scored)}/{len(rows)} plans measured, "
          f"{len(improved)} lowered D2 without losing E5")
    print(f"   verdict: {verdict}")
    print("=" * 78)
    return out
