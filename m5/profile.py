"""m5.profile - what M5 knows about one checkpoint before it steers anything (plan §5.1).

A `ModelProfile` is measured once per checkpoint and read by every later step. Per layer (layer L
is the output of decoder block L, `m5.intervention`) it holds:

    nuisance     the coordinates the nuisance scan flagged (`m5.dose.nuisance_scan`). On Gemma-3
                 one coordinate is ~8e4 tall against ~2e2 for its neighbours, on every token, and
                 it dominates both ||h|| and any difference-of-means vector (loud-channel repo).
    s_L          the median per-token residual norm over content tokens with those coordinates
                 zeroed, from fp32 statistics: the denominator of the primary dose r_perp.
    content_norm the median *unmasked* norm over the same content tokens. s_L <= content_norm
                 always, and their ratio is how much of the typical token's norm the mask removes.
    raw_norm     M3's ||h||, kept for comparability: per prompt, the lower median over *every*
                 token position (BOS and chat-template tokens included) of the fp32 norm, then the
                 mean of those medians over the prompt set, at batch size 1
                 (`m5.dose.measure_m3_raw_norms`). It is the denominator of r_raw, so every M5
                 dose can be set beside an archived M3 dose. Its token set differs from s_L's, so
                 the two are not ordered.
    D_rand       the r_perp at which norm-matched random vectors break integrity at that layer;
                 None until measured. The denominator of rho, the unit the search moves in.
    gain_flags   the weights-only cross-check from the block's RMSNorm gains. None when the
                 architecture reports no gains: missing, which is not the same as "none flagged".

For the whole model it holds the mean activation norm at the middle layer (the system-card unit,
`m5.dose.card_units`), the chat template's hash and generation prompt (so a vector extracted under
a different template than the one generation uses is refused -- the defect that withdrew the old
Qwen arm, `m5.vectors`), and the provenance of the statistics.

Redo it when the checkpoint changes: another checkpoint has other loud coordinates and another
integrity boundary, and the refusal-ablated arm, if it is ever run, is another checkpoint.

The profile holds scalars and coordinate indices only -- never vectors or activations -- so it is
an `aggregate` document (`m5.runio.DOCUMENTS["profile.json"]`). It still lives with the runs,
outside the repository: `save` and `load` refuse a path inside the tree.

Every invariant is checked when a profile is built, whether by the setup or from JSON, so an
inconsistent profile (s_L above the unmasked norm of the same tokens, a flagged coordinate outside
the model, a cap the scan did not respect) cannot exist long enough to be used.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from m5.runio import write_json_atomic
from m5.safety import assert_outside_repo

PROFILE_SCHEMA = "m5.profile.v1"
PROFILE_FILE = "profile.json"

# Per-model choices the setup makes once and every later stage reads (plan §5.1), beside the dose
# denominators. Each is plain JSON describing the choice -- ids, seeds, file paths and digests --
# never weights or vector values (those live under RUN_ROOT, named by path and sha256 here):
#
#   d7_controls   {variant: [control question ids]} matched on unsteered yes/no gap (step 6)
#   capability    the s5 items kept for this model and how they were chosen (step 5)
#   ea_probe      the a2 evaluation-awareness probe: path, sha256, layers, held-out AUC (step 7)
#   j_lens        a fitted J-lens for q2: path, sha256, layers (optional)
SELECTION_KEYS = ("d7_controls", "capability", "ea_probe", "j_lens")
_MAX_NUMERIC_LIST = 512   # longer numeric lists than any id or layer list look like weights

# What a profile must record so a reader knows what its numbers mean: the prompts behind the scan
# and s_L, the scan settings, and the prompts behind M3's ||h|| (a different set: M3 measured it
# on its chat-rendered task prompts, 66 since 2026-08-23, 7 or 22 in older runs).
PROVENANCE_REQUIRED = ("prompt_set", "prompt_hash", "n_prompts", "skip_first",
                       "loud_ratio", "loud_cap", "gain_ratio",
                       "raw_norm_prompt_set", "raw_norm_prompt_hash", "raw_norm_n_prompts")

# s_L <= content_norm holds exactly (the masked norm of every token is at most its unmasked norm,
# and the median preserves that order); this slack only absorbs fp32 rounding.
_ORDER_SLACK = 1e-6


class ProfileError(ValueError):
    """A profile is inconsistent, incomplete, or asked for something it does not hold."""


# =====================================================================================
# Small validators. Each raises ProfileError naming the field, never coerces a bad value.
# =====================================================================================

def _int(v: Any, what: str, *, lo: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ProfileError(f"{what} must be an integer, got {v!r}")
    if lo is not None and v < lo:
        raise ProfileError(f"{what} must be >= {lo}, got {v}")
    return v


def _pos_float(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ProfileError(f"{what} must be a number, got {v!r}")
    out = float(v)
    if not (math.isfinite(out) and out > 0):
        raise ProfileError(f"{what} must be finite and > 0, got {v!r}")
    return out


def _opt_pos_float(v: Any, what: str) -> float | None:
    return None if v is None else _pos_float(v, what)


def _coords(v: Any, what: str) -> tuple[int, ...]:
    if isinstance(v, (str, bytes)) or not isinstance(v, Sequence):
        raise ProfileError(f"{what} must be a list of coordinates, got {v!r}")
    out = tuple(_int(c, f"{what} entry", lo=0) for c in v)
    if len(set(out)) != len(out):
        raise ProfileError(f"{what} repeats a coordinate: {list(out)}")
    return out


def _json_safe(obj: Any, what: str) -> Any:
    try:
        return json.loads(json.dumps(obj))
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{what} is not JSON-serialisable: {exc}") from None


def _no_values(obj: Any, where: str) -> None:
    """Refuse a long numeric list anywhere in `obj`: weights and vector values are stored under
    RUN_ROOT and named here by path and digest, never inlined into an exportable document."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _no_values(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        numeric = sum(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj)
        if numeric > _MAX_NUMERIC_LIST:
            raise ProfileError(f"{where} is a list of {numeric} numbers; weights and vector values "
                               "are stored under RUN_ROOT and referenced by path and sha256")
        for i, x in enumerate(obj):
            _no_values(x, f"{where}[{i}]")


def _exact_keys(d: Mapping[str, Any], expected: Sequence[str], what: str) -> None:
    missing = sorted(set(expected) - set(d))
    unknown = sorted(set(d) - set(expected))
    if missing or unknown:
        raise ProfileError(f"{what}: missing keys {missing}, unknown keys {unknown}")


def layer_ranges(layers: Sequence[int]) -> str:
    """'0-5, 8, 10-12' -- for error messages that name many layers."""
    ls = sorted(set(layers))
    if not ls:
        return "none"
    parts, start, prev = [], ls[0], ls[0]
    for x in ls[1:] + [None]:  # type: ignore[operator]
        if x is not None and x == prev + 1:
            prev = x
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if x is not None:
            start = prev = x
    return ", ".join(parts)


# =====================================================================================
# Per layer
# =====================================================================================

_LAYER_KEYS = ("layer", "nuisance", "loud_ratios", "n_above", "median_abs_mean", "s_L",
               "content_norm", "n_tokens", "raw_norm", "raw_norm_se", "D_rand", "gain_flags",
               "gain_agreement")


@dataclass(frozen=True)
class LayerProfile:
    """One layer's dose denominators and the scan that produced them.

    `loud_ratios[i]` is |mean| of `nuisance[i]` over the layer's median |mean|, largest first.
    `n_above` counts every coordinate above the scan ratio *before* the cap, so a layer where
    the cap cut the list is visible (n_above > len(nuisance)) rather than silently truncated.
    `n_tokens` counts the content tokens behind s_L and content_norm. `raw_norm_se` is the
    standard error of M3's mean of per-prompt medians (None from a single prompt).
    """

    layer: int
    nuisance: tuple[int, ...]
    loud_ratios: tuple[float, ...]
    n_above: int
    median_abs_mean: float
    s_L: float
    content_norm: float
    n_tokens: int
    raw_norm: float
    raw_norm_se: float | None
    D_rand: float | None = None
    gain_flags: tuple[int, ...] | None = None
    gain_agreement: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        where = f"layer {self.layer}"
        _int(self.layer, f"{where}: layer", lo=0)
        object.__setattr__(self, "nuisance", _coords(self.nuisance, f"{where}: nuisance"))
        ratios = tuple(_pos_float(r, f"{where}: loud ratio") for r in self.loud_ratios)
        object.__setattr__(self, "loud_ratios", ratios)
        if len(ratios) != len(self.nuisance):
            raise ProfileError(f"{where}: {len(ratios)} loud ratios for {len(self.nuisance)} "
                               "nuisance coordinates")
        if any(b > a for a, b in zip(ratios, ratios[1:])):
            raise ProfileError(f"{where}: nuisance coordinates must be listed largest first")
        _int(self.n_above, f"{where}: n_above", lo=len(self.nuisance))
        object.__setattr__(self, "median_abs_mean",
                           _pos_float(self.median_abs_mean, f"{where}: median_abs_mean"))
        object.__setattr__(self, "s_L", _pos_float(self.s_L, f"{where}: s_L"))
        object.__setattr__(self, "content_norm",
                           _pos_float(self.content_norm, f"{where}: content_norm"))
        if self.s_L > self.content_norm * (1 + _ORDER_SLACK):
            raise ProfileError(f"{where}: s_L={self.s_L:.6g} exceeds the unmasked norm of the same "
                               f"tokens ({self.content_norm:.6g}); zeroing coordinates cannot raise "
                               "a norm, so the two statistics came from different data")
        _int(self.n_tokens, f"{where}: n_tokens", lo=1)
        object.__setattr__(self, "raw_norm", _pos_float(self.raw_norm, f"{where}: raw_norm"))
        if self.raw_norm_se is not None:
            se = self.raw_norm_se
            if isinstance(se, bool) or not isinstance(se, (int, float)) or not (
                    math.isfinite(se) and se >= 0):
                raise ProfileError(f"{where}: raw_norm_se must be finite and >= 0, got {se!r}")
            object.__setattr__(self, "raw_norm_se", float(se))
        object.__setattr__(self, "D_rand", _opt_pos_float(self.D_rand, f"{where}: D_rand"))
        if self.gain_flags is not None:
            object.__setattr__(self, "gain_flags", _coords(self.gain_flags, f"{where}: gain_flags"))
        if self.gain_agreement is not None:
            if self.gain_flags is None:
                raise ProfileError(f"{where}: gain agreement without gain flags")
            object.__setattr__(self, "gain_agreement",
                               _json_safe(dict(self.gain_agreement), f"{where}: gain_agreement"))

    def to_json(self) -> dict[str, Any]:
        return {"layer": self.layer, "nuisance": list(self.nuisance),
                "loud_ratios": list(self.loud_ratios), "n_above": self.n_above,
                "median_abs_mean": self.median_abs_mean, "s_L": self.s_L,
                "content_norm": self.content_norm, "n_tokens": self.n_tokens,
                "raw_norm": self.raw_norm, "raw_norm_se": self.raw_norm_se, "D_rand": self.D_rand,
                "gain_flags": None if self.gain_flags is None else list(self.gain_flags),
                "gain_agreement": None if self.gain_agreement is None else dict(self.gain_agreement)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "LayerProfile":
        if not isinstance(d, Mapping):
            raise ProfileError(f"a layer entry must be an object, got {type(d).__name__}")
        _exact_keys(d, _LAYER_KEYS, f"layer entry {d.get('layer')!r}")
        return cls(layer=d["layer"], nuisance=d["nuisance"], loud_ratios=d["loud_ratios"],
                   n_above=d["n_above"], median_abs_mean=d["median_abs_mean"], s_L=d["s_L"],
                   content_norm=d["content_norm"], n_tokens=d["n_tokens"],
                   raw_norm=d["raw_norm"], raw_norm_se=d["raw_norm_se"], D_rand=d["D_rand"],
                   gain_flags=d["gain_flags"], gain_agreement=d["gain_agreement"])


# =====================================================================================
# The model
# =====================================================================================

_PROFILE_KEYS = ("schema", "model_id", "revision", "n_layers", "d_model", "mid_layer",
                 "mean_mid_norm", "chat_template_hash", "generation_suffix", "provenance",
                 "d_rand_provenance", "selections", "layers")


def middle_layer(n_layers: int) -> int:
    """The layer the system-card unit is anchored to: the output of block n_layers // 2.

    The Fable 5 card says "the middle layer of the model" without an index; for an even depth
    (Gemma-3-27B: 62, Qwen3-32B: 64) this is the first block of the upper half.
    """
    return _int(n_layers, "n_layers", lo=1) // 2


@dataclass(frozen=True)
class ModelProfile:
    """Per-checkpoint dose denominators. Build with `m5.dose.build_profile` or `from_json`.

    `layers` may cover a subset of the model (a profile is usable on the layers it measured);
    every lookup of a layer it does not hold raises rather than guessing.
    """

    model_id: str
    revision: str | None
    n_layers: int
    d_model: int
    layers: Mapping[int, LayerProfile]
    mid_layer: int
    mean_mid_norm: float | None
    provenance: Mapping[str, Any]
    chat_template_hash: str | None = None
    generation_suffix: str | None = None
    d_rand_provenance: Mapping[str, Any] | None = field(default=None)
    selections: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ProfileError("model_id is required")
        if self.revision is not None and not isinstance(self.revision, str):
            raise ProfileError(f"revision must be text or None, got {self.revision!r}")
        _int(self.n_layers, "n_layers", lo=1)
        _int(self.d_model, "d_model", lo=1)
        _int(self.mid_layer, "mid_layer", lo=0)
        if self.mid_layer >= self.n_layers:
            raise ProfileError(f"mid_layer {self.mid_layer} is outside the model's "
                               f"{self.n_layers} layers")
        object.__setattr__(self, "mean_mid_norm",
                           _opt_pos_float(self.mean_mid_norm, "mean_mid_norm"))
        for name in ("chat_template_hash", "generation_suffix"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or not v):
                raise ProfileError(f"{name} must be non-empty text or None, got {v!r}")
        prov = _json_safe(dict(self.provenance), "provenance")
        missing = [k for k in PROVENANCE_REQUIRED if k not in prov]
        if missing:
            raise ProfileError(f"provenance lacks {missing}: a profile must say which prompts and "
                               "which scan settings produced it")
        _int(prov["n_prompts"], "provenance.n_prompts", lo=1)
        _int(prov["skip_first"], "provenance.skip_first", lo=0)
        _int(prov["raw_norm_n_prompts"], "provenance.raw_norm_n_prompts", lo=1)
        loud_ratio = _pos_float(prov["loud_ratio"], "provenance.loud_ratio")
        loud_cap = _int(prov["loud_cap"], "provenance.loud_cap", lo=1)
        _pos_float(prov["gain_ratio"], "provenance.gain_ratio")
        object.__setattr__(self, "provenance", prov)
        if self.d_rand_provenance is not None:
            object.__setattr__(self, "d_rand_provenance",
                               _json_safe(dict(self.d_rand_provenance), "d_rand_provenance"))

        if not isinstance(self.layers, Mapping) or not self.layers:
            raise ProfileError("a profile needs at least one layer")
        layers: dict[int, LayerProfile] = {}
        for key, lp in self.layers.items():
            if not isinstance(lp, LayerProfile):
                raise ProfileError(f"layer {key!r}: expected a LayerProfile")
            if _int(key, "layer key", lo=0) != lp.layer:
                raise ProfileError(f"layer key {key} holds the profile of layer {lp.layer}")
            if lp.layer >= self.n_layers:
                raise ProfileError(f"layer {lp.layer} is outside the model's {self.n_layers} layers")
            bad = [c for c in lp.nuisance + (lp.gain_flags or ()) if c >= self.d_model]
            if bad:
                raise ProfileError(f"layer {lp.layer}: coordinates {bad} are outside d_model "
                                   f"{self.d_model}")
            if len(lp.nuisance) > loud_cap:
                raise ProfileError(f"layer {lp.layer}: {len(lp.nuisance)} nuisance coordinates "
                                   f"exceed the cap {loud_cap}")
            if lp.n_above > len(lp.nuisance) and len(lp.nuisance) != loud_cap:
                raise ProfileError(f"layer {lp.layer}: {lp.n_above} coordinates passed the scan "
                                   f"but only {len(lp.nuisance)} were kept, below the cap")
            if any(r <= loud_ratio for r in lp.loud_ratios):
                raise ProfileError(f"layer {lp.layer}: a nuisance coordinate's ratio is not above "
                                   f"the scan's loud_ratio {loud_ratio}")
            layers[lp.layer] = lp
        object.__setattr__(self, "layers", dict(sorted(layers.items())))
        if (self.mean_mid_norm is None) != (self.mid_layer not in self.layers):
            raise ProfileError("mean_mid_norm is measured exactly when the middle layer "
                               f"({self.mid_layer}) is profiled")
        if not isinstance(self.selections, Mapping):
            raise ProfileError("selections must be a mapping")
        unknown = sorted(set(self.selections) - set(SELECTION_KEYS))
        if unknown:
            raise ProfileError(f"unknown profile selections {unknown}; known: {SELECTION_KEYS}")
        sel = _json_safe(dict(self.selections), "selections")
        _no_values(sel, "selections")
        object.__setattr__(self, "selections", sel)

    # ---------------------------------------------------------------- lookups

    def layer(self, layer: int) -> LayerProfile:
        try:
            return self.layers[layer]
        except KeyError:
            raise ProfileError(
                f"layer {layer} is not in the profile of {self.model_id} (profiled: "
                f"{layer_ranges(list(self.layers))}); its s_L, nuisance coordinates and D_rand are "
                "unknown") from None

    def s(self, layer: int) -> float:
        """s_L: the primary dose denominator at this layer."""
        return self.layer(layer).s_L

    def raw(self, layer: int) -> float:
        """M3's ||h|| at this layer, kept for comparability (mean of per-prompt lower medians over
        every token position, unmasked)."""
        return self.layer(layer).raw_norm

    def nuisance(self, layer: int) -> tuple[int, ...]:
        return self.layer(layer).nuisance

    def d_rand(self, layer: int) -> float:
        """D_rand(L), or a refusal naming the setup step that measures it."""
        v = self.layer(layer).D_rand
        if v is None:
            raise ProfileError(
                f"D_rand({layer}) has not been measured for {self.model_id}; rho is undefined there. "
                "Run the random-vector integrity boundary (plan §5.1) first.")
        return v

    def layers_with_d_rand(self) -> list[int]:
        return [L for L, lp in self.layers.items() if lp.D_rand is not None]

    # ---------------------------------------------------------------- updates

    def with_d_rand(self, values: Mapping[int, float],
                    provenance: Mapping[str, Any]) -> "ModelProfile":
        """A copy with D_rand(L) set for the given layers.

        D_rand is measured after the scan, possibly in several sessions. Values measured under
        different settings (seeds, ladder, integrity criterion) would be different units under
        one name, so a second call must carry the same provenance as the first.
        """
        if not values:
            raise ProfileError("no D_rand values given")
        prov = _json_safe(dict(provenance), "D_rand provenance")
        if not prov:
            raise ProfileError("D_rand needs its provenance (seeds, ladder, criterion)")
        if self.d_rand_provenance is not None and dict(self.d_rand_provenance) != prov:
            raise ProfileError("D_rand provenance differs from the values already in the profile "
                               f"({dict(self.d_rand_provenance)} vs {prov}); measurements under "
                               "different settings cannot share a profile")
        layers = dict(self.layers)
        for L, v in values.items():
            layers[L] = replace(self.layer(L), D_rand=_pos_float(v, f"D_rand({L})"))
        return replace(self, layers=layers, d_rand_provenance=prov)

    def has_selection(self, name: str) -> bool:
        if name not in SELECTION_KEYS:
            raise ProfileError(f"unknown profile selection {name!r}; known: {SELECTION_KEYS}")
        return name in self.selections

    def selection(self, name: str) -> Any:
        """A setup choice by exact name; raises when the setup has not made it, so a stage that
        needs it cannot quietly run without it (callers that may choose for themselves check
        `has_selection` first and record what they chose)."""
        if not self.has_selection(name):
            raise ProfileError(f"the profile has no {name!r} selection; run the setup step that "
                               "makes it (plan §5.1)")
        return self.selections[name]

    def with_selection(self, name: str, value: Mapping[str, Any]) -> "ModelProfile":
        """A copy with one setup choice recorded (replacing an earlier one of the same name)."""
        if name not in SELECTION_KEYS:
            raise ProfileError(f"unknown profile selection {name!r}; known: {SELECTION_KEYS}")
        if not isinstance(value, Mapping) or not value:
            raise ProfileError(f"selection {name!r} must be a non-empty mapping")
        return replace(self, selections={**dict(self.selections), name: dict(value)})

    # ---------------------------------------------------------------- serialisation

    def to_json(self) -> dict[str, Any]:
        return {"schema": PROFILE_SCHEMA, "model_id": self.model_id, "revision": self.revision,
                "n_layers": self.n_layers, "d_model": self.d_model, "mid_layer": self.mid_layer,
                "mean_mid_norm": self.mean_mid_norm,
                "chat_template_hash": self.chat_template_hash,
                "generation_suffix": self.generation_suffix,
                "provenance": dict(self.provenance),
                "d_rand_provenance": (None if self.d_rand_provenance is None
                                      else dict(self.d_rand_provenance)),
                "selections": dict(self.selections),
                "layers": {str(L): lp.to_json() for L, lp in self.layers.items()}}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "ModelProfile":
        if not isinstance(d, Mapping):
            raise ProfileError(f"a profile must be a JSON object, got {type(d).__name__}")
        if d.get("schema") != PROFILE_SCHEMA:
            raise ProfileError(f"profile schema {d.get('schema')!r} is not {PROFILE_SCHEMA!r}; "
                               "refusing to read it as one")
        _exact_keys(d, _PROFILE_KEYS, "profile")
        raw_layers = d["layers"]
        if not isinstance(raw_layers, Mapping):
            raise ProfileError("profile.layers must be an object keyed by layer")
        layers: dict[int, LayerProfile] = {}
        for k, v in raw_layers.items():
            try:
                L = int(k)
            except (TypeError, ValueError):
                raise ProfileError(f"layer key {k!r} is not an integer") from None
            layers[L] = LayerProfile.from_json(v)
        return cls(model_id=d["model_id"], revision=d["revision"], n_layers=d["n_layers"],
                   d_model=d["d_model"], layers=layers, mid_layer=d["mid_layer"],
                   mean_mid_norm=d["mean_mid_norm"], provenance=d["provenance"],
                   chat_template_hash=d["chat_template_hash"],
                   generation_suffix=d["generation_suffix"],
                   d_rand_provenance=d["d_rand_provenance"], selections=d["selections"])

    def key(self) -> str:
        """16 hex characters identifying this exact profile, for stamping rows that used it."""
        payload = json.dumps(self.to_json(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def save(self, directory: str | Path, *, overwrite: bool = False) -> Path:
        """Write `<directory>/profile.json`. Refuses the repository, and an existing profile
        unless `overwrite` (a silently replaced profile changes the unit of every earlier rho)."""
        d = assert_outside_repo(directory, what="profile directory")
        p = d / PROFILE_FILE
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} already holds a profile; pass overwrite=True to replace it")
        write_json_atomic(p, self.to_json())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ModelProfile":
        """Read a profile from a file, or from `<directory>/profile.json`."""
        p = assert_outside_repo(path, what="profile")
        if p.is_dir():
            p = p / PROFILE_FILE
        return cls.from_json(json.loads(p.read_text(encoding="utf-8")))
