"""m5.dose - what "strength" means in M5 (plan §4.3): the nuisance scan, the dose units, the
layer schedules, and the alphas a search cell implies.

## Why a new unit

M3's dose `r = alpha ||v|| / ||h||` has four known problems:

- **Gemma-3's loud coordinate.** One residual coordinate in every Gemma-3 size is about 8e4 tall
  against about 2e2 for its neighbours, on every token (the loud-channel repository, which traced
  the broken difference-of-means abliteration on Gemma-3 to it). It dominates ||h||, and it
  dominates any difference-of-means vector, because a difference of means weights each coordinate
  by its variance. The median-token norm M3 used does not remove it.
- **Sink positions** (BOS, template tokens) inflate norm statistics.
- **Fixed units do not transfer across models**: the Opus 4.6 card caps inhibition at 0.1 of the
  mean mid-layer norm, and the same recipe on GLM-5 was incoherent above 0.025 (Read et al.).
- **Multi-layer schedules have no single-layer dose.**

## The units (every one logged for every cell)

    r_perp = ||alpha v_perp|| / s_L      primary (decision 6). v_perp is v with the layer's
                                         nuisance coordinates zeroed; s_L the median content-token
                                         norm off those coordinates (`m5.profile`)
    r_raw  = alpha ||v|| / raw_norm_L    M3's dose, with raw_norm_L M3's own ||h|| definition
                                         (`measure_m3_raw_norms`), so every M5 number can be set
                                         beside an archived M3 one
    rho    = r_perp / D_rand(L)          the behavioural anchor the search moves in: 1.0 is where
                                         norm-matched random vectors break integrity at that layer
    card   = alpha ||v_L|| / N_mid       the system-card unit: a fraction of the mean activation
                                         norm at the middle layer (`card_units`)

`r_perp` is defined on v_perp whether or not the vector is *applied* masked
(`MASK_LOUD_IN_VECTOR`): the dose counts what the model can feel off the nuisance coordinates.
Masked application is a separate choice, made once the masked-vs-unmasked comparison (plan §4.3
item 3) decides it, because massive dimensions can carry causal signal (Roh et al.).

## Multi-layer doses

A schedule's cumulative dose is the sum of its per-layer doses (plan §4.3 item 5): cumulative
r_perp = sum of r_perp_L, cumulative rho = sum of rho_L. The schedule's weights are each layer's
share of the total steering multiplier, `alpha_L = w_L * A` -- the convention of both sources M5
reproduces: Hua et al. split the total strength equally over layers 10, 14, ..., 30, and the
system cards apply one global factor to every layer's vector "to preserve relative magnitudes
across layers". With equal weights a strided or all-layer cell is therefore exactly one global
scale, and maps onto a card strength. `A` is solved so the cumulative rho equals the cell's rho.
Splitting the dose itself equally (equal r_perp per layer) would give every layer a different
alpha, which no practitioner runs and no card strength describes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m5.backend.base import Backend, ModelInfo, ResidualStats
from m5.intervention import SCHEDULES, Cell, Intervention, LayerTerm, Positions
from m5.profile import (PROVENANCE_REQUIRED, LayerProfile, ModelProfile, layer_ranges,
                        middle_layer)

# Suffix of the vector id under which the nuisance-masked copy of a vector is registered. An
# intervention names exactly what is applied, so a masked cell can never silently steer with the
# unmasked values: if the masked copy was not registered, the backend refuses the id.
MASKED_SUFFIX = "~perp"

# Weights must sum to one; this absorbs float rounding only.
_WEIGHT_SUM_TOL = 1e-9


class DoseError(ValueError):
    """A dose cannot be computed: a denominator is missing, a vector is absent, or an input is
    invalid. The message names every layer affected."""


# =====================================================================================
# 1. The nuisance scan (plan §4.3 item 1)
# =====================================================================================

@dataclass(frozen=True)
class GainCheck:
    """One RMSNorm gain vector's flags: coordinates whose |gain| exceeds `gain_ratio` x the
    vector's median |gain|, largest first, at most the scan's cap."""

    name: str
    coords: tuple[int, ...]
    n_above: int
    max_ratio: float
    median_gain: float

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "coords": list(self.coords), "n_above": self.n_above,
                "max_ratio": self.max_ratio, "median_gain": self.median_gain}


@dataclass(frozen=True)
class NuisanceReport:
    """One layer's scan: the activation flags (which define the mask) and, beside them, the
    weights-only cross-check with its agreement statistics.

    `gains` is None when no gains were supplied or the architecture reports none for this layer:
    the cross-check is then missing, which is not the same as "agrees" or "flags nothing".
    """

    layer: int
    coords: tuple[int, ...]
    ratios: tuple[float, ...]
    n_above: int
    median_abs_mean: float
    max_ratio: float
    gains: tuple[GainCheck, ...] | None
    agreement: Mapping[str, Any] | None
    n_tokens: int

    @property
    def gain_union(self) -> tuple[int, ...] | None:
        if self.gains is None:
            return None
        return tuple(sorted({c for g in self.gains for c in g.coords}))

    def to_json(self) -> dict[str, Any]:
        return {"layer": self.layer, "coords": list(self.coords), "ratios": list(self.ratios),
                "n_above": self.n_above, "median_abs_mean": self.median_abs_mean,
                "max_ratio": self.max_ratio, "n_tokens": self.n_tokens,
                "gains": None if self.gains is None else [g.to_json() for g in self.gains],
                "agreement": None if self.agreement is None else dict(self.agreement)}


def _flag(values: np.ndarray, ratio: float, cap: int,
          what: str) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Coordinates whose value exceeds `ratio` x the median value, largest first, capped.

    Returns (coords, their ratios, how many passed before the cap, the median, the largest ratio).
    A zero median makes every ratio infinite; that is degenerate input, not a finding.
    """
    med = float(np.median(values))
    if not (math.isfinite(med) and med > 0):
        raise DoseError(f"{what}: the median is {med}; ratios to it are undefined (degenerate "
                        "statistics -- a real residual or gain vector never has a zero median)")
    ratios = values / med
    above = np.flatnonzero(ratios > ratio)
    order = above[np.argsort(-ratios[above], kind="stable")]
    kept = order[:cap]
    return kept, ratios[kept], int(above.size), med, float(ratios.max())


def _agreement(act: Sequence[int], gain: Sequence[int]) -> dict[str, Any]:
    a, g = set(act), set(gain)
    union = a | g
    return {"both": sorted(a & g), "activation_only": sorted(a - g), "gain_only": sorted(g - a),
            "jaccard": (len(a & g) / len(union)) if union else None,
            "activation_recall": (len(a & g) / len(a)) if a else None}


def nuisance_scan(stats: Mapping[int, ResidualStats],
                  gains: Mapping[int, Mapping[str, np.ndarray]] | None, *,
                  loud_ratio: float, loud_cap: int, gain_ratio: float) -> dict[int, NuisanceReport]:
    """Flag each layer's nuisance coordinates from unsteered baseline statistics.

    A coordinate is flagged when its |mean| over the baseline content tokens exceeds `loud_ratio`
    x the layer's median |mean|; at most `loud_cap` per layer, largest first (plan §4.3 item 1:
    50x and 16). This is the loud-channel repository's class-blind rule, chosen from neutral
    text only -- choosing coordinates from a vector's own difference would remove its signal by
    construction.

    The weights-only cross-check reads the block's RMSNorm gains (`Backend.norm_gains`, which
    must report the multiplier each coordinate receives -- for Gemma's RMSNorm that is
    1 + weight). The loud-channel repository found the post-FFN writer gain, as a multiple of the
    layer median, separates the models that need the correction (4.5x-6.9x on Gemma-3) from those
    that do not (2.5x-2.7x on Gemma-2) before any forward pass. It is reported beside the
    activation flags, never used to set the mask: a coordinate is written loud at a few blocks and
    then carried by every later layer, so agreement is informative, not expected to be exact.
    """
    if not (math.isfinite(loud_ratio) and loud_ratio > 0):
        raise DoseError(f"loud_ratio must be finite and > 0, got {loud_ratio}")
    if isinstance(loud_cap, bool) or not isinstance(loud_cap, int) or loud_cap < 1:
        raise DoseError(f"loud_cap must be an integer >= 1, got {loud_cap!r}")
    if not (math.isfinite(gain_ratio) and gain_ratio > 0):
        raise DoseError(f"gain_ratio must be finite and > 0, got {gain_ratio}")
    if not stats:
        raise DoseError("no residual statistics to scan")
    if gains is not None:
        extra = sorted(set(gains) - set(stats))
        missing = sorted(set(stats) - set(gains))
        if extra or missing:
            raise DoseError(f"gains do not cover the scanned layers: missing "
                            f"{layer_ranges(missing)}, extra {layer_ranges(extra)}")
    d_model: int | None = None
    out: dict[int, NuisanceReport] = {}
    for layer in sorted(stats):
        st = stats[layer]
        if st.layer != layer:
            raise DoseError(f"statistics keyed as layer {layer} are for layer {st.layer}")
        mean = np.asarray(st.mean, dtype=np.float64)
        if mean.ndim != 1 or not np.all(np.isfinite(mean)):
            raise DoseError(f"layer {layer}: mean must be a finite 1-D array, got shape {mean.shape}")
        if d_model is None:
            d_model = mean.size
        elif mean.size != d_model:
            raise DoseError(f"layer {layer}: d_model {mean.size} differs from {d_model}")
        coords, ratios, n_above, med, max_ratio = _flag(np.abs(mean), loud_ratio, loud_cap,
                                                         f"layer {layer} |mean|")
        checks: tuple[GainCheck, ...] | None = None
        agreement: dict[str, Any] | None = None
        layer_gains = None if gains is None else gains[layer]
        if layer_gains:  # None (not supplied) and {} (architecture reports none) both mean missing
            found = []
            for name in sorted(layer_gains):
                g = np.abs(np.asarray(layer_gains[name], dtype=np.float64))
                if g.shape != (mean.size,) or not np.all(np.isfinite(g)):
                    raise DoseError(f"layer {layer}: gain {name!r} must be finite with shape "
                                    f"({mean.size},), got {g.shape}")
                gc, _, g_above, g_med, g_max = _flag(g, gain_ratio, loud_cap,
                                                     f"layer {layer} gain {name!r}")
                found.append(GainCheck(name, tuple(int(c) for c in gc), g_above, g_max, g_med))
            checks = tuple(found)
            agreement = _agreement([int(c) for c in coords],
                                   sorted({c for g in checks for c in g.coords}))
        out[layer] = NuisanceReport(layer=layer, coords=tuple(int(c) for c in coords),
                                    ratios=tuple(float(r) for r in ratios), n_above=n_above,
                                    median_abs_mean=med, max_ratio=max_ratio, gains=checks,
                                    agreement=agreement, n_tokens=int(st.n_tokens))
    return out


def nuisance_scan_cfg(stats: Mapping[int, ResidualStats],
                      gains: Mapping[int, Mapping[str, np.ndarray]] | None,
                      cfg: Mapping[str, Any]) -> dict[int, NuisanceReport]:
    """`nuisance_scan` with its settings read from the config by exact key."""
    return nuisance_scan(stats, gains, loud_ratio=cfg["LOUD_RATIO"], loud_cap=cfg["LOUD_CAP"],
                         gain_ratio=cfg["GAIN_RATIO"])


# =====================================================================================
# 2. s_L, M3's ||h||, and the profile
# =====================================================================================

@dataclass(frozen=True)
class NormSummary:
    """Content-token norms of one layer, from one `residual_stats` pass with the scan's mask."""

    s_L: float           # median per-token norm with the nuisance coordinates zeroed
    content_norm: float  # median per-token norm of the same tokens, unmasked
    mean_norm: float     # mean per-token norm, unmasked (the card's "average activation norm")
    n_tokens: int


def norms_from_stats(stats: ResidualStats, nuisance: Sequence[int]) -> NormSummary:
    """Medians of the backend's per-token norms over content tokens (it already skipped BOS).

    The tokens must be content only: pass raw text (BOS + text, `skip_first=1`) rather than
    chat-rendered prompts, whose template tokens `skip_first` cannot remove (plan §4.3: s_L is
    taken with "BOS and template excluded"). Sink positions inflate norm statistics.

    `stats` must come from `residual_stats(..., masks={layer: nuisance})`. ResidualStats does not
    record which mask produced `masked_norms`, so two consistency checks stand in for it:

    - every masked norm is at most its raw norm (zeroing coordinates cannot raise a norm);
    - the mean of (norm^2 - masked^2) -- exactly the mean of sum_c h_c^2 over the masked
      coordinates -- is at least sum_c (mean |h_c|)^2 (Jensen). Statistics computed without the
      mask, or with a mask that misses the loud coordinate, fail this by orders of magnitude;
      s_L would otherwise silently equal the raw norm and the new unit would be M3's under
      another name.
    """
    where = f"layer {stats.layer}"
    norms = np.asarray(stats.norms, dtype=np.float64)
    if norms.ndim != 1 or norms.size == 0 or norms.size != stats.n_tokens:
        raise DoseError(f"{where}: {norms.size} per-token norms for n_tokens={stats.n_tokens}")
    if not np.all(np.isfinite(norms)) or np.any(norms < 0):
        raise DoseError(f"{where}: per-token norms must be finite and non-negative")
    coords = [int(c) for c in nuisance]
    if stats.masked_norms is None:
        if coords:
            raise DoseError(f"{where}: the scan flagged {coords} but the statistics carry no masked "
                            "norms; call residual_stats with masks={layer: nuisance}")
        masked = norms
    else:
        masked = np.asarray(stats.masked_norms, dtype=np.float64)
        if masked.shape != norms.shape or not np.all(np.isfinite(masked)) or np.any(masked < 0):
            raise DoseError(f"{where}: masked norms must be finite, non-negative, one per token")
        scale = float(np.max(norms)) if norms.size else 0.0
        if np.any(masked > norms * (1 + 1e-5) + 1e-6 * scale):
            raise DoseError(f"{where}: a masked norm exceeds its raw norm; the two arrays do not "
                            "describe the same tokens")
        removed = float(np.mean(norms ** 2 - masked ** 2))
        mean_abs = np.asarray(stats.mean_abs, dtype=np.float64)
        floor = float(np.sum(mean_abs[coords] ** 2)) if coords else 0.0
        # fp32 accumulation of two norms over d_model coordinates, squared: generous, because a
        # false alarm here would stop a setup, and the defect it guards against is ~all of norm^2
        noise = 1e-4 * float(np.mean(norms ** 2))
        if removed < 0.99 * floor - noise:
            raise DoseError(f"{where}: the masked norms remove {removed:.6g} of squared norm per "
                            f"token, but the flagged coordinates {coords} alone carry at least "
                            f"{floor:.6g}. The statistics were computed with a different mask.")
        if not coords and removed > noise:
            raise DoseError(f"{where}: nothing was flagged, yet the masked norms differ from the "
                            "raw norms; the statistics were computed with a mask")
    return NormSummary(s_L=float(np.median(masked)), content_norm=float(np.median(norms)),
                       mean_norm=float(np.mean(norms)), n_tokens=int(norms.size))


@dataclass(frozen=True)
class RawNorm:
    """M3's ||h_L|| for one layer, kept for comparability with every archived M3 dose."""

    value: float        # mean over prompts of the per-prompt lower median
    se: float | None    # standard error of that mean (sd with ddof=1 / sqrt(n)); None for n = 1
    n_prompts: int
    n_tokens: int       # every position of every prompt, BOS and template included


def lower_median(x: Any) -> float:
    """The lower of the two middle values for an even count, the middle one for an odd count.

    `torch.median`'s semantics, which M3's ||h|| used; numpy's median averages the two middle
    values instead, and on a short prompt the difference is a visible shift in r_raw.
    """
    a = np.sort(np.asarray(x, dtype=np.float64).ravel())
    if a.size == 0:
        raise DoseError("the median of no values is undefined")
    return float(a[(a.size - 1) // 2])


def m3_raw_norm(per_prompt_norms: Sequence[Any]) -> RawNorm:
    """M3's ||h||: the mean over prompts of each prompt's lower median per-token norm.

    A mean of medians -- not a median of medians, and not a pooled median -- exactly as
    `m2/model.py` computed it (one prompt per forward pass, every position, fp32 norms).
    """
    if not per_prompt_norms:
        raise DoseError("M3's ||h|| needs at least one prompt")
    medians, n_tokens = [], 0
    for i, n in enumerate(per_prompt_norms):
        a = np.asarray(n, dtype=np.float64)
        if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a < 0):
            raise DoseError(f"prompt #{i}: per-token norms must be a non-empty finite 1-D array "
                            "of non-negative values")
        medians.append(lower_median(a))
        n_tokens += a.size
    m = np.asarray(medians)
    se = float(np.std(m, ddof=1) / math.sqrt(m.size)) if m.size > 1 else None
    return RawNorm(value=float(np.mean(m)), se=se, n_prompts=int(m.size), n_tokens=n_tokens)


def measure_m3_raw_norms(backend: Backend, prompts: Sequence[str],
                         layers: Sequence[int]) -> dict[int, RawNorm]:
    """Measure M3's ||h_L|| on chat-rendered prompts, one prompt per call.

    `ResidualStats.norms` pools every token of every prompt in a call, so the per-prompt median
    M3 took needs one `residual_stats` call per prompt -- which is also M3's batch size of one.
    `skip_first=0` keeps the BOS, as M3 did. For archived comparability pass M3's own prompt set
    (its task prompts, chat-rendered: 66 since 2026-08-23, 7 or 22 in older runs) and record it in
    the profile's provenance.

    Refuses a prompt that does not start with the BOS token when the model has one: M3 refused raw
    text here, because it would be measured in a context steering never runs in.
    """
    if not prompts:
        raise DoseError("M3's ||h|| needs at least one prompt")
    layers = [int(L) for L in layers]
    if not layers:
        raise DoseError("no layers to measure")
    bos = backend.info.bos_token_id
    per_layer: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    for i, p in enumerate(prompts):
        ids = backend.tokenize(p)
        if bos is not None and (not ids or ids[0] != bos):
            raise DoseError(f"prompt #{i} does not start with BOS ({bos}); M3's ||h|| was measured "
                            "on chat-rendered prompts, so render it with Backend.render first")
        stats = backend.residual_stats([p], layers, masks=None, skip_first=0)
        for L in layers:
            if L not in stats:
                raise DoseError(f"the backend returned no statistics for layer {L}")
            st = stats[L]
            norms = np.asarray(st.norms, dtype=np.float64)
            if st.layer != L or norms.size != len(ids):
                raise DoseError(f"prompt #{i}, layer {L}: {norms.size} per-token norms for a "
                                f"{len(ids)}-token prompt; M3's ||h|| counts every position")
            per_layer[L].append(norms)
    return {L: m3_raw_norm(v) for L, v in per_layer.items()}


def build_profile(info: ModelInfo, stats: Mapping[int, ResidualStats],
                  scan: Mapping[int, NuisanceReport], *, raw_norms: Mapping[int, RawNorm],
                  provenance: Mapping[str, Any],
                  generation_suffix: str | None = None) -> ModelProfile:
    """Assemble the profile from the scan, the masked content-token statistics, and M3's ||h||.

    `stats` is the second pass over the scan's prompts, `residual_stats(prompts, layers,
    masks=<scan coords>)`: its `masked_norms` give s_L, its `norms` give the unmasked median of the
    same tokens and the middle layer's mean norm (the card unit). `raw_norms` come from
    `measure_m3_raw_norms` on M3's prompts. `provenance` names both prompt sets and the scan
    settings (`m5.profile.PROVENANCE_REQUIRED`). A layer without any of the three is refused:
    a missing norm must not become a default.

    The two passes must be over the same tokens -- the provenance records one prompt set for the
    scan and s_L -- so the token count and the median |mean| of each layer are compared.
    """
    if set(stats) != set(scan):
        raise DoseError(f"statistics cover layers {layer_ranges(list(stats))} but the scan covers "
                        f"{layer_ranges(list(scan))}")
    for L in sorted(scan):
        same_n = stats[L].n_tokens == scan[L].n_tokens
        med = float(np.median(np.abs(np.asarray(stats[L].mean, dtype=np.float64))))
        if not (same_n and math.isclose(med, scan[L].median_abs_mean, rel_tol=1e-3)):
            raise DoseError(f"layer {L}: the masked statistics ({stats[L].n_tokens} tokens) are not "
                            f"over the same prompts as the scan ({scan[L].n_tokens} tokens)")
    missing_raw = sorted(set(scan) - set(raw_norms))
    if missing_raw:
        raise DoseError(f"M3's ||h|| is missing for layers {layer_ranges(missing_raw)}")
    absent = [k for k in PROVENANCE_REQUIRED if k not in provenance]
    if absent:
        raise DoseError(f"provenance lacks {absent}; a profile must say which prompts and which "
                        "scan settings produced it")
    n_raw = {r.n_prompts for r in raw_norms.values()}
    if n_raw != {provenance["raw_norm_n_prompts"]}:
        raise DoseError(f"M3's ||h|| was measured on {sorted(n_raw)} prompts, but the provenance "
                        f"says {provenance['raw_norm_n_prompts']}")
    layers: dict[int, LayerProfile] = {}
    mean_mid: float | None = None
    mid = middle_layer(info.n_layers)
    for L in sorted(scan):
        rep = scan[L]
        ns = norms_from_stats(stats[L], rep.coords)
        if L == mid:
            mean_mid = ns.mean_norm
        layers[L] = LayerProfile(layer=L, nuisance=rep.coords, loud_ratios=rep.ratios,
                                 n_above=rep.n_above, median_abs_mean=rep.median_abs_mean,
                                 s_L=ns.s_L, content_norm=ns.content_norm, n_tokens=ns.n_tokens,
                                 raw_norm=raw_norms[L].value, raw_norm_se=raw_norms[L].se,
                                 gain_flags=rep.gain_union, gain_agreement=rep.agreement)
    return ModelProfile(model_id=info.hf_id, revision=info.revision, n_layers=info.n_layers,
                        d_model=info.d_model, layers=layers, mid_layer=mid,
                        mean_mid_norm=mean_mid, provenance=provenance,
                        chat_template_hash=info.chat_template_hash or None,
                        generation_suffix=generation_suffix)


# =====================================================================================
# 3. Unit conversions
# =====================================================================================

def _vec(v: Any, d_model: int, what: str = "vector") -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.ndim != 1 or a.size != d_model:
        raise DoseError(f"{what} must be 1-D of length d_model={d_model}, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise DoseError(f"{what} has non-finite values")
    return a


def _scalar(x: float, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float, np.floating, np.integer)):
        raise DoseError(f"{what} must be a number, got {x!r}")
    out = float(x)
    if not (math.isfinite(out) and out >= 0):
        raise DoseError(f"{what} must be finite and >= 0 (the sign is the operation), got {x}")
    return out


def _layer(profile: ModelProfile, layer: int) -> LayerProfile:
    if layer not in profile.layers:
        raise DoseError(f"s_L for layer {layer} is not in the profile of {profile.model_id} "
                        f"(profiled: {layer_ranges(list(profile.layers))})")
    return profile.layers[layer]


def mask_coords(v: Any, coords: Sequence[int]) -> np.ndarray:
    """v with `coords` zeroed (a copy): v_perp for a layer's nuisance coordinates."""
    a = np.array(v, dtype=np.float64)
    if a.ndim != 1:
        raise DoseError(f"expected a 1-D vector, got shape {a.shape}")
    idx = [int(c) for c in coords]
    if any(c < 0 or c >= a.size for c in idx):
        raise DoseError(f"coordinates {idx} fall outside a vector of length {a.size}")
    a[idx] = 0.0
    return a


def f_loud(v: Any, coords: Sequence[int]) -> float:
    """||v_M||^2 / ||v||^2: the share of the vector's squared norm on the nuisance coordinates.

    Recorded per vector (plan §4.3 item 3) because a difference-of-means vector on Gemma-3 can be
    mostly loud coordinate, which would make its raw norm meaningless as a size.
    """
    a = np.asarray(v, dtype=np.float64)
    total = float(a @ a)
    if not total > 0:
        raise DoseError("f_loud of a zero vector is undefined")
    m = a - mask_coords(a, coords)  # v_M: the part the mask removes
    return float(m @ m) / total


def cos_raw_masked(v: Any, coords: Sequence[int]) -> float:
    """cos(v, v_perp). Equal to sqrt(1 - f_loud) because v_perp and v_M are orthogonal; both are
    recorded so each can be checked against the other."""
    a = np.asarray(v, dtype=np.float64)
    p = mask_coords(a, coords)
    na, npp = float(np.linalg.norm(a)), float(np.linalg.norm(p))
    if not na > 0:
        raise DoseError("cos(raw, masked) of a zero vector is undefined")
    return float(a @ p) / (na * npp) if npp > 0 else 0.0


def perp_norm(v: Any, layer: int, profile: ModelProfile) -> float:
    """||v_perp|| at this layer."""
    lp = _layer(profile, layer)
    return float(np.linalg.norm(mask_coords(_vec(v, profile.d_model), lp.nuisance)))


def r_perp(alpha: float, v: Any, layer: int, profile: ModelProfile) -> float:
    """The primary dose: ||alpha v_perp|| / s_L."""
    a = _scalar(alpha, "alpha")
    return a * perp_norm(v, layer, profile) / _layer(profile, layer).s_L


def alpha_from_r_perp(r: float, v: Any, layer: int, profile: ModelProfile) -> float:
    n = perp_norm(v, layer, profile)
    if not n > 0:
        raise DoseError(f"layer {layer}: the vector lies entirely on nuisance coordinates; no "
                        "alpha gives it an r_perp")
    return _scalar(r, "r_perp") * _layer(profile, layer).s_L / n


def r_raw(alpha: float, v: Any, layer: int, profile: ModelProfile) -> float:
    """M3's dose: alpha ||v|| / ||h||, with ||h|| M3's own definition (`measure_m3_raw_norms`),
    kept so an M5 cell can be placed on an archived M3 dose axis."""
    a = _scalar(alpha, "alpha")
    return a * float(np.linalg.norm(_vec(v, profile.d_model))) / _layer(profile, layer).raw_norm


def alpha_from_r_raw(r: float, v: Any, layer: int, profile: ModelProfile) -> float:
    n = float(np.linalg.norm(_vec(v, profile.d_model)))
    if not n > 0:
        raise DoseError("a zero vector has no alpha for any r_raw")
    return _scalar(r, "r_raw") * _layer(profile, layer).raw_norm / n


def _d_rand(profile: ModelProfile, layer: int) -> float:
    d = _layer(profile, layer).D_rand
    if d is None:
        raise DoseError(f"D_rand({layer}) has not been measured for {profile.model_id}; rho is "
                        "undefined there (run the random-vector integrity boundary, plan §5.1)")
    return d


def rho(r_perp_value: float, layer: int, profile: ModelProfile) -> float:
    """The behavioural anchor: r_perp / D_rand(L). 1.0 is where norm-matched random vectors break
    integrity at this layer, which makes doses comparable across layers, vectors and models (the
    "equal dose is not equal perturbation" caveat in RESULTS-QWEN.md)."""
    return _scalar(r_perp_value, "r_perp") / _d_rand(profile, layer)


def r_perp_from_rho(rho_value: float, layer: int, profile: ModelProfile) -> float:
    return _scalar(rho_value, "rho") * _d_rand(profile, layer)


def _mid_norm(profile: ModelProfile) -> float:
    if profile.mean_mid_norm is None:
        raise DoseError(f"the profile of {profile.model_id} has no mean norm at the middle layer "
                        f"({profile.mid_layer}); card units are undefined")
    return profile.mean_mid_norm


def card_units(alpha: float, v: Any, layer: int, profile: ModelProfile) -> float:
    """alpha ||v_L|| / N_mid: the norm added at `layer`, as a fraction of the mean activation norm
    at the middle layer (the system-card unit).

    The Fable 5 card's convention, quoted: per-layer vectors "were normalized using a single
    scaling factor chosen such that a strength of 1.0 corresponded to the average activation norm
    in the middle layer of the model. Scaling was applied uniformly across all layers to preserve
    relative magnitudes across layers." Read here as: one alpha at every layer, with strength
    s = alpha ||v_mid|| / N_mid -- so `card_units(alpha, v_mid, profile.mid_layer, profile)` is
    the card's strength of a uniform-alpha schedule (0.01 in that card's inhibition experiments;
    the Opus 4.6 card caps inhibition at 0.1). At any other layer the same expression reads as
    the fraction of N_mid added there. N_mid is the mean (the card's "average") unmasked token
    norm over content tokens, so it includes any loud coordinate -- the card's unit is what
    practitioners run, not M5's.

    `v` is the vector as applied (pass the masked vector when it is applied masked).
    """
    a = _scalar(alpha, "alpha")
    _layer(profile, layer)
    return a * float(np.linalg.norm(_vec(v, profile.d_model))) / _mid_norm(profile)


def alpha_from_card(units: float, v: Any, layer: int, profile: ModelProfile) -> float:
    """The alpha giving `units` of the card unit at `layer`. For a system-card strength s, pass
    the middle layer's vector: one alpha, then applied at every layer."""
    _layer(profile, layer)
    n = float(np.linalg.norm(_vec(v, profile.d_model)))
    if not n > 0:
        raise DoseError("a zero vector has no alpha for any card strength")
    return _scalar(units, "card units") * _mid_norm(profile) / n


# =====================================================================================
# 4. Layer schedules (plan §4.2)
# =====================================================================================

def layer_depth(layer: int, n_layers: int) -> float:
    """Depth fraction of layer L: L / (n_layers - 1), so 0.0 is the first block's output and
    1.0 the last. Config bands (`STRIDED_BAND`, `GAUSSIAN_PROFILE`) are in this unit, which
    means the same place in any model."""
    if n_layers < 2:
        raise DoseError(f"depth fractions need at least 2 layers, got {n_layers}")
    if not 0 <= layer < n_layers:
        raise DoseError(f"layer {layer} is outside a model of {n_layers} layers")
    return layer / (n_layers - 1)


def band_layers(band: Sequence[float], n_layers: int) -> list[int]:
    """Every layer whose depth fraction lies in [low, high]."""
    if len(band) != 2 or not 0.0 <= band[0] < band[1] <= 1.0:
        raise DoseError(f"a depth band is (low, high) with 0 <= low < high <= 1, got {tuple(band)}")
    lo, hi = float(band[0]), float(band[1])
    out = [L for L in range(n_layers) if lo - 1e-12 <= layer_depth(L, n_layers) <= hi + 1e-12]
    if not out:
        raise DoseError(f"no layer of a {n_layers}-layer model lies in the depth band {tuple(band)}")
    return out


def schedule_layers(schedule: str, n_layers: int, anchor_layer: int | None,
                    strided_band: Sequence[float], strided_step: int,
                    gaussian_profile: Sequence[float]) -> list[tuple[int, float]]:
    """(layer, weight) pairs; the weights sum to 1 and are each layer's share of the schedule's
    total steering multiplier (see the module docstring for why multipliers, not doses).

        single    [(anchor, 1.0)]
        strided   every `strided_step`-th layer of the depth band, equal shares (Hua et al.).
                  The anchor, when given, sets the phase and must lie in the band; None starts
                  at the band's first layer.
        all       every layer, equal shares: one global scale (the system cards)
        gaussian  every layer, shares following exp(-(depth - centre)^2 / (2 width^2))

    'all' and 'gaussian' take no anchor: a search that varied the anchor would otherwise produce
    identical cells under different names.
    """
    if schedule not in SCHEDULES:
        raise DoseError(f"unknown schedule {schedule!r}; one of {SCHEDULES}")
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 2:
        raise DoseError(f"n_layers must be an integer >= 2, got {n_layers!r}")
    if anchor_layer is not None and not 0 <= anchor_layer < n_layers:
        raise DoseError(f"anchor layer {anchor_layer} is outside a model of {n_layers} layers")
    if schedule == "single":
        if anchor_layer is None:
            raise DoseError("a single-layer schedule needs its anchor layer")
        return [(int(anchor_layer), 1.0)]
    if schedule == "strided":
        if isinstance(strided_step, bool) or not isinstance(strided_step, int) or strided_step < 1:
            raise DoseError(f"strided_step must be an integer >= 1, got {strided_step!r}")
        band = band_layers(strided_band, n_layers)
        if anchor_layer is not None and anchor_layer not in band:
            raise DoseError(f"anchor layer {anchor_layer} is outside the strided band "
                            f"{tuple(strided_band)} (layers {layer_ranges(band)})")
        start = band[0] if anchor_layer is None else int(anchor_layer)
        layers = [L for L in band if (L - start) % strided_step == 0]
        return [(L, 1.0 / len(layers)) for L in layers]
    if anchor_layer is not None:
        raise DoseError(f"the {schedule!r} schedule has no anchor layer; pass None")
    if schedule == "all":
        return [(L, 1.0 / n_layers) for L in range(n_layers)]
    if len(gaussian_profile) != 2:
        raise DoseError(f"gaussian_profile is (centre, width), got {tuple(gaussian_profile)}")
    centre, width = float(gaussian_profile[0]), float(gaussian_profile[1])
    if not (0 < centre < 1 and width > 0 and math.isfinite(width)):
        raise DoseError(f"gaussian_profile needs centre in (0, 1) and width > 0, got "
                        f"{tuple(gaussian_profile)}")
    raw = [math.exp(-0.5 * ((layer_depth(L, n_layers) - centre) / width) ** 2)
           for L in range(n_layers)]
    total = math.fsum(raw)
    return [(L, w / total) for L, w in enumerate(raw)]


def schedule_layers_cfg(cfg: Mapping[str, Any], n_layers: int,
                        anchor_layer: int | None) -> list[tuple[int, float]]:
    """`schedule_layers` for the configured schedule, its settings read by exact key."""
    return schedule_layers(cfg["SCHEDULE"], n_layers, anchor_layer, cfg["STRIDED_BAND"],
                           cfg["STRIDED_STEP"], cfg["GAUSSIAN_PROFILE"])


# =====================================================================================
# 5. Cells -> interventions, and the dose bookkeeping
# =====================================================================================

def applied_vector_id(vector_id: str, mask_loud: bool) -> str:
    """The id an intervention names: the vector itself, or its nuisance-masked copy."""
    if MASKED_SUFFIX in vector_id:
        raise DoseError(f"{vector_id!r} already names a masked copy")
    return vector_id + MASKED_SUFFIX if mask_loud else vector_id


def is_masked_id(vector_id: str) -> bool:
    return vector_id.endswith(MASKED_SUFFIX)


def applied_vectors(vectors: Mapping[int, np.ndarray], profile: ModelProfile,
                    mask_loud: bool) -> dict[int, np.ndarray]:
    """The per-layer values to register under `applied_vector_id(vector_id, mask_loud)`."""
    out: dict[int, np.ndarray] = {}
    for L, v in sorted(vectors.items()):
        a = _vec(v, profile.d_model, f"vector at layer {L}")
        out[L] = (mask_coords(a, _layer(profile, L).nuisance) if mask_loud else a).astype(np.float32)
    return out


def _cell_weights(cell: Cell, weights: Mapping[int, float] | None) -> dict[int, float]:
    if weights is None:
        if cell.schedule == "gaussian":
            raise DoseError("a Gaussian cell needs its weights: pass the schedule_layers output "
                            "(the profile cannot be recovered from the layer list)")
        return {L: 1.0 / len(cell.layers) for L in cell.layers}
    w = {int(L): float(x) for L, x in weights.items()}
    if set(w) != set(cell.layers):
        raise DoseError(f"weights cover layers {layer_ranges(list(w))}, the cell has "
                        f"{layer_ranges(list(cell.layers))}")
    if any(not (math.isfinite(x) and x >= 0) for x in w.values()):
        raise DoseError("weights must be finite and >= 0")
    if abs(math.fsum(w.values()) - 1.0) > _WEIGHT_SUM_TOL:
        raise DoseError(f"weights must sum to 1, got {math.fsum(w.values())!r}")
    return w


def _vectors_at(layers: Sequence[int], vectors: Mapping[int, np.ndarray],
                profile: ModelProfile) -> dict[int, np.ndarray]:
    missing = [L for L in layers if L not in vectors]
    if missing:
        raise DoseError(f"the vector has no entry at scheduled layers {layer_ranges(missing)}")
    return {L: _vec(vectors[L], profile.d_model, f"vector at layer {L}") for L in layers}


def _rho_per_alpha(layers: Sequence[int], vecs: Mapping[int, np.ndarray],
                   profile: ModelProfile) -> dict[int, float]:
    """rho added at each layer by alpha = 1: ||v_L_perp|| / (s_L D_rand(L))."""
    no_profile = [L for L in layers if L not in profile.layers]
    if no_profile:
        raise DoseError(f"s_L is missing for layers {layer_ranges(no_profile)}: they are not in "
                        f"the profile of {profile.model_id}")
    no_drand = [L for L in layers if profile.layers[L].D_rand is None]
    if no_drand:
        raise DoseError(f"D_rand is missing for layers {layer_ranges(no_drand)}; a rho-based cell "
                        "needs the random-vector integrity boundary at every scheduled layer")
    out: dict[int, float] = {}
    for L in layers:
        lp = profile.layers[L]
        n = float(np.linalg.norm(mask_coords(vecs[L], lp.nuisance)))
        if not n > 0:
            raise DoseError(f"layer {L}: the vector lies entirely on nuisance coordinates; it has "
                            "no r_perp to scale")
        out[L] = n / (lp.s_L * lp.D_rand)  # type: ignore[operator]
    return out


def cell_to_intervention(cell: Cell, profile: ModelProfile, vectors: Mapping[int, np.ndarray],
                         positions: Positions, mask_loud: bool,
                         op_alpha_for_project: float = 1.0, *,
                         weights: Mapping[int, float] | None = None,
                         clamp_value: float | None = None) -> Intervention:
    """The intervention that puts `cell.rho` on the cell's schedule.

    add / sub   alpha_L = w_L * A, with A solved so that sum_L rho_L = cell.rho, where
                rho_L = alpha_L ||v_L_perp|| / (s_L D_rand(L)). The cumulative r_perp is the sum
                of the per-layer r_perp (plan §4.3 item 5); `dose_record` reports both.
    project     directional ablation is not a dose: its size depends on each token's h . v^.
                alpha is the ablation fraction `op_alpha_for_project` at every scheduled layer
                (1.0 removes the component; above 1 would reflect it, a different operation),
                and `cell.rho` is not read.
    clamp       sets the component along v^ to `clamp_value` (residual units) at every scheduled
                layer. No rule for choosing that value exists yet, so a clamp cell without one
                raises NotImplementedError rather than inventing it.

    `weights` are the `schedule_layers` output for this cell; without them single, strided and
    all-layer cells use equal shares (their definition), and Gaussian cells raise. With
    `mask_loud` the intervention names the masked copy (`applied_vector_id`), which the caller
    registers with `applied_vectors`.
    """
    if not isinstance(positions, Positions):
        raise DoseError(f"positions must be a Positions, got {type(positions).__name__}")
    w = _cell_weights(cell, weights)
    vid = applied_vector_id(cell.vector_id, mask_loud)
    vecs = _vectors_at(cell.layers, vectors, profile)
    if cell.op != "clamp" and clamp_value is not None:
        raise DoseError("clamp_value is only for clamp cells")
    if cell.op == "clamp":
        if clamp_value is None:
            raise NotImplementedError(
                "a clamp cell needs clamp_value (the target component along v^, in residual "
                "units); M5 has no rule for choosing it yet, so none is invented")
        c = float(clamp_value)
        if not math.isfinite(c):
            raise DoseError(f"clamp_value must be finite, got {clamp_value}")
        return Intervention(vid, "clamp", tuple(LayerTerm(L, 1.0) for L in cell.layers),
                            positions, clamp_value=c)
    if cell.op == "project":
        frac = _scalar(op_alpha_for_project, "ablation fraction")
        if frac > 1.0:
            raise DoseError(f"the ablation fraction must lie in [0, 1], got {frac}; above 1 the "
                            "component is reflected, not removed")
        for L, v in vecs.items():
            direction = mask_coords(v, _layer(profile, L).nuisance) if mask_loud else v
            if not np.linalg.norm(direction) > 0:
                raise DoseError(f"layer {L}: a zero direction cannot be projected out")
        return Intervention(vid, "project", tuple(LayerTerm(L, frac) for L in cell.layers),
                            positions)
    k = _rho_per_alpha(cell.layers, vecs, profile)
    # > 0: the weights sum to 1 and every k_L > 0 (`_rho_per_alpha` refuses a zero v_perp)
    scale = cell.rho / math.fsum(w[L] * k[L] for L in cell.layers)
    return Intervention(vid, cell.op, tuple(LayerTerm(L, w[L] * scale) for L in cell.layers),
                        positions)


@dataclass(frozen=True)
class LayerDose:
    layer: int
    alpha: float
    r_perp: float | None
    r_raw: float | None
    rho: float | None
    card: float | None

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DoseRecord:
    """Every unit of one intervention, per layer and cumulative, for the logs.

    None means not defined or not measurable, with the reason in `notes`: projection and clamp
    have no additive dose; rho needs D_rand at every layer; the card strength needs one alpha at
    every layer, the middle layer's vector, and N_mid.
    """

    vector_id: str
    op: str
    mask_loud: bool
    per_layer: tuple[LayerDose, ...]
    r_perp: float | None
    r_raw: float | None
    rho: float | None
    card_strength: float | None
    notes: tuple[str, ...] = field(default=())

    def to_json(self) -> dict[str, Any]:
        return {"vector_id": self.vector_id, "op": self.op, "mask_loud": self.mask_loud,
                "per_layer": [d.to_json() for d in self.per_layer], "r_perp": self.r_perp,
                "r_raw": self.r_raw, "rho": self.rho, "card_strength": self.card_strength,
                "notes": list(self.notes)}


def dose_record(intervention: Intervention, profile: ModelProfile,
                vectors: Mapping[int, np.ndarray], *, mask_loud: bool) -> DoseRecord:
    """Recompute every dose unit from an intervention's alphas.

    Works for any intervention -- one from `cell_to_intervention`, or one an audit built from a
    card strength -- so the bookkeeping never trusts the arithmetic that produced the alphas.
    `vectors` are the unmasked values; `mask_loud` says whether the masked copy is applied, and
    must agree with the id the intervention names.
    """
    if is_masked_id(intervention.vector_id) != mask_loud:
        raise DoseError(f"intervention names {intervention.vector_id!r} but mask_loud={mask_loud}; "
                        "the record would describe a different vector from the one applied")
    layers = list(intervention.layers)
    vecs = _vectors_at(layers, vectors, profile)
    notes: list[str] = []
    if intervention.op in ("project", "clamp"):
        notes.append(f"{intervention.op} has no additive dose; alpha is "
                     + ("the ablation fraction" if intervention.op == "project" else "unused"))
        per = tuple(LayerDose(t.layer, t.alpha, None, None, None, None) for t in intervention.terms)
        return DoseRecord(intervention.vector_id, intervention.op, mask_loud, per, None, None,
                          None, None, tuple(notes))
    mid_norm = profile.mean_mid_norm
    if mid_norm is None:
        notes.append(f"no mean norm at the middle layer ({profile.mid_layer}): card units undefined")
    per_layer: list[LayerDose] = []
    rho_missing: list[int] = []
    for t in intervention.terms:
        lp = _layer(profile, t.layer)
        v_perp = mask_coords(vecs[t.layer], lp.nuisance)
        applied = v_perp if mask_loud else vecs[t.layer]
        rp = t.alpha * float(np.linalg.norm(v_perp)) / lp.s_L
        rr = t.alpha * float(np.linalg.norm(applied)) / lp.raw_norm
        if lp.D_rand is None:
            rho_missing.append(t.layer)
        rh = None if lp.D_rand is None else rp / lp.D_rand
        card = None if mid_norm is None else t.alpha * float(np.linalg.norm(applied)) / mid_norm
        per_layer.append(LayerDose(t.layer, t.alpha, rp, rr, rh, card))
    if rho_missing:
        notes.append(f"D_rand missing at layers {layer_ranges(rho_missing)}: cumulative rho undefined")
    alphas = [t.alpha for t in intervention.terms]
    card_strength: float | None = None
    if mid_norm is not None:
        uniform = max(alphas) - min(alphas) <= 1e-12 * max(1.0, max(alphas))
        if not uniform:
            notes.append("alphas differ across layers: no single system-card strength")
        elif profile.mid_layer not in vectors:
            notes.append(f"the vector has no entry at the middle layer ({profile.mid_layer}): "
                         "the card strength is undefined")
        else:
            v_mid = _vec(vectors[profile.mid_layer], profile.d_model, "middle-layer vector")
            if mask_loud:
                v_mid = mask_coords(v_mid, _layer(profile, profile.mid_layer).nuisance)
            card_strength = alphas[0] * float(np.linalg.norm(v_mid)) / mid_norm
    return DoseRecord(
        vector_id=intervention.vector_id, op=intervention.op, mask_loud=mask_loud,
        per_layer=tuple(per_layer),
        r_perp=math.fsum(d.r_perp for d in per_layer),  # type: ignore[misc]
        r_raw=math.fsum(d.r_raw for d in per_layer),  # type: ignore[misc]
        rho=None if rho_missing else math.fsum(d.rho for d in per_layer),  # type: ignore[misc]
        card_strength=card_strength, notes=tuple(notes))
