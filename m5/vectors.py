"""m5.vectors - the vectors M5 steers with, the controls every claim is made against, and what is
recorded about each (plan §4.1, §4.4, §3.4, §5.1).

## Kinds

    concept                last-token residual of CONCEPT_TEMPLATE rendered as a user turn with
                           the generation prompt, minus the mean over the baseline words of the
                           same, per layer (Lindsey / Macar)
    response_mean          the mean over the model's own reply to the same template, minus the
                           baseline words' reply means (Persona Vectors find the response average
                           strongest at equal norm)
    contrastive            the mean difference over prompt pairs at a fixed token: the last token,
                           or a marker such as a trailing colon (Hua et al.; CAA)
    sae_feature            one decoder direction from a .npy/.npz file outside the repository, at
                           one layer (Gemma Scope)
    gaussian               control: matched in mean, variance and norm to a reference vector, per
                           layer
    sign_randomised        control: a contrastive construction with each pair's order flipped by a
                           seeded coin. SteerCheck's strong control: random directions alone are
                           weak, being near-orthogonal to everything; this one keeps the
                           per-example structure and destroys only the consistent direction
    unrelated_contrastive  control: pairs with no semantic content, such as the Fable 5 card's
                           "The train arrived at the station." vs "The train left the station."
    random_subspace        control: a rank-k orthonormal basis, for projection-ablation controls

## Safety

Every recipe declares a `vector_class` and is refused unless `m5.safety` allows it; concept words
go through `check_concept`, every contrastive text through `check_texts`, and an SAE file path
must lie outside the repository. A recipe's params are read-only once checked, and `extract` and
`VectorStore.save` run the checks again. Vector values are written only by `VectorStore`, under a
root outside the repository; the metadata the run's `vectors.json` carries never holds values.

## The read position (why the checks exist)

The old Qwen arm was withdrawn because extraction rendered the prompt without the template switch
generation used: the vector was read at the newline after `assistant`, and injected into passes
whose last prompt token was the newline after `</think>` -- four tokens later, a state the model
never entered during extraction (RESULTS-QWEN.md). Here a 'last' read renders the user turn with
the backend's own `render` (the one generation uses), and then checks that the token read is the
generation prompt's last token: the template appended a generation prompt; it equals the one the
generation path uses (`expected_suffix`, recorded in the profile); the last token ends exactly at
the end of the text and lies inside that suffix; and it decodes to that suffix's tail. Any
failure raises `ExtractionError`, never reads somewhere else.

Activations are accumulated in float64 and stored as float32. The only extraction check on the
result is finite and non-zero: a model-specific norm band was once used as an instrument check and
was wrong (DEBUG-LOG R5).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from m5 import safety
from m5.backend.base import Backend, GenRequest, derive_seed, token_index_at
from m5.dose import cos_raw_masked, f_loud, mask_coords
from m5.profile import ModelProfile, layer_ranges
from m5.runio import utc_now

KINDS = ("concept", "response_mean", "contrastive", "sae_feature",
         "gaussian", "sign_randomised", "unrelated_contrastive", "random_subspace")
CONTROL_KINDS = frozenset({"gaussian", "sign_randomised", "unrelated_contrastive",
                           "random_subspace"})
CONCEPT_KINDS = frozenset({"concept", "response_mean"})
PAIR_KINDS = frozenset({"contrastive", "unrelated_contrastive", "sign_randomised"})
RENDERS = ("user_turn", "raw")   # a chat user turn with the generation prompt, or BOS + raw text
LAST = "last"

# The Fable 5 card's control pair, quoted: "contrasting the last-token activations of 'The train
# arrived at the station.' vs. 'The train left the station.'"
CARD_CONTROL_PAIR = ("The train arrived at the station.", "The train left the station.")

# Vector ids name files and appear in cell ids, so they are restricted to a safe alphabet; '~'
# is reserved for the masked copy (`m5.dose.MASKED_SUFFIX`) and '|' separates cell-id fields.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# kind -> (required params, optional params)
_PARAMS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "concept": (frozenset({"word"}), frozenset()),
    "response_mean": (frozenset({"word", "max_new_tokens", "temperature"}), frozenset()),
    "contrastive": (frozenset({"pairs", "position", "render"}), frozenset()),
    "unrelated_contrastive": (frozenset({"pairs", "position", "render"}), frozenset()),
    "sign_randomised": (frozenset({"pairs", "position", "render", "base_id", "index"}), frozenset()),
    "sae_feature": (frozenset({"path", "layer"}), frozenset({"index", "key"})),
    "gaussian": (frozenset({"reference_id", "index"}), frozenset()),
    "random_subspace": (frozenset({"rank", "index"}), frozenset()),
}

# vectors.json is an aggregate document and may leave the machine; vector values must never be in
# it (CLAUDE.md hard rule 3). Any numeric list this long in metadata is refused as looking like
# values. The longest legitimate list is a layer list (64 layers on Qwen3-32B).
_MAX_NUMERIC_LIST = 256


class RecipeError(ValueError):
    """A recipe is malformed: unknown kind, missing or unknown params, a class the kind forbids."""


class ExtractionError(RuntimeError):
    """Extraction could not be done as specified: the read position failed its check, the backend
    broke its contract, or an input made the vector undefined."""


# =====================================================================================
# Recipes
# =====================================================================================

def _check_id(v: Any, what: str) -> str:
    if not isinstance(v, str) or not _ID.match(v):
        raise RecipeError(f"{what} {v!r} must match {_ID.pattern}")
    return v


def _int_param(v: Any, what: str, lo: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise RecipeError(f"{what} must be an integer >= {lo}, got {v!r}")
    return v


def _text(v: Any, what: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise RecipeError(f"{what} must be non-empty text, got {v!r}")
    return v


def _pairs(v: Any, what: str, min_pairs: int) -> tuple[tuple[str, str], ...]:
    if isinstance(v, (str, bytes)) or not isinstance(v, Sequence):
        raise RecipeError(f"{what}: pairs must be a list of [positive, negative] texts")
    out = []
    for i, p in enumerate(v):
        if isinstance(p, (str, bytes)) or not isinstance(p, Sequence) or len(p) != 2:
            raise RecipeError(f"{what}: pair #{i} must be [positive, negative]")
        a, b = _text(p[0], f"{what} pair #{i} positive"), _text(p[1], f"{what} pair #{i} negative")
        if a == b:
            raise RecipeError(f"{what}: pair #{i} has identical texts; its difference is zero")
        out.append((a, b))
    if len(out) < min_pairs:
        raise RecipeError(f"{what}: needs at least {min_pairs} pair(s), got {len(out)}")
    return tuple(out)


def _check_params(kind: str, vid: str, p: Mapping[str, Any]) -> dict[str, Any]:
    req, opt = _PARAMS[kind]
    missing, unknown = sorted(req - set(p)), sorted(set(p) - req - opt)
    if missing or unknown:
        raise RecipeError(f"{kind} recipe {vid!r}: missing params {missing}, unknown params {unknown}")
    out: dict[str, Any] = {}
    if kind in CONCEPT_KINDS:
        out["word"] = _text(p["word"], f"{vid}: word")
        if kind == "response_mean":
            out["max_new_tokens"] = _int_param(p["max_new_tokens"], f"{vid}: max_new_tokens", 1)
            t = p["temperature"]
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not (
                    math.isfinite(t) and t >= 0):
                raise RecipeError(f"{vid}: temperature must be finite and >= 0, got {t!r}")
            out["temperature"] = float(t)
    if kind in PAIR_KINDS:
        pairs = _pairs(p["pairs"], vid, 2 if kind == "sign_randomised" else 1)
        position = _text(p["position"], f"{vid}: position")
        if p["render"] not in RENDERS:
            raise RecipeError(f"{vid}: render must be one of {RENDERS}, got {p['render']!r}")
        if position != LAST:
            for i, pair in enumerate(pairs):
                for side, t in zip(("positive", "negative"), pair):
                    if position not in t:
                        raise RecipeError(f"{vid}: marker {position!r} does not occur in pair #{i} "
                                          f"{side} text {t!r}")
        out.update(pairs=pairs, position=position, render=p["render"])
        if kind == "sign_randomised":
            out["base_id"] = _check_id(p["base_id"], f"{vid}: base_id")
            out["index"] = _int_param(p["index"], f"{vid}: index", 0)
    if kind == "sae_feature":
        path = _text(p["path"], f"{vid}: path")
        suffix = Path(path).suffix
        if suffix not in (".npy", ".npz"):
            raise RecipeError(f"{vid}: SAE weights must be .npy or .npz, got {path!r}")
        index = p["index"] if "index" in p else None
        key = p["key"] if "key" in p else None
        if index is not None:
            _int_param(index, f"{vid}: index", 0)
        if suffix == ".npz" and not (isinstance(key, str) and key):
            raise RecipeError(f"{vid}: an .npz file needs the array's key (e.g. 'W_dec')")
        if suffix == ".npy" and key is not None:
            raise RecipeError(f"{vid}: a .npy file holds one array; key is only for .npz")
        out.update(path=path, layer=_int_param(p["layer"], f"{vid}: layer", 0), index=index, key=key)
    if kind == "gaussian":
        out["reference_id"] = _check_id(p["reference_id"], f"{vid}: reference_id")
        out["index"] = _int_param(p["index"], f"{vid}: index", 0)
    if kind == "random_subspace":
        out["rank"] = _int_param(p["rank"], f"{vid}: rank", 1)
        out["index"] = _int_param(p["index"], f"{vid}: index", 0)
    return out


def _plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


@dataclass(frozen=True)
class VectorRecipe:
    """What a vector is, as data: built from a dict (`from_json`) or a helper below.

    The class must suit the kind: control kinds are class 'control' and nothing else is; concept
    kinds are class 'concept'. A recipe that passes construction has passed every safety check.
    """

    vector_id: str
    kind: str
    vector_class: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        _check_id(self.vector_id, "vector_id")
        if self.kind not in KINDS:
            raise RecipeError(f"unknown vector kind {self.kind!r}; one of {KINDS}")
        safety.check_vector_class(self.vector_class)
        if (self.kind in CONTROL_KINDS) != (self.vector_class == "control"):
            raise RecipeError(f"{self.vector_id}: a {self.kind!r} recipe cannot have class "
                              f"{self.vector_class!r}; control kinds and the 'control' class go "
                              "together, so every control is counted as one")
        if self.kind in CONCEPT_KINDS and self.vector_class != "concept":
            raise RecipeError(f"{self.vector_id}: a {self.kind!r} recipe has class 'concept'")
        if not isinstance(self.params, Mapping):
            raise RecipeError(f"{self.vector_id}: params must be a mapping")
        params = _check_params(self.kind, self.vector_id, self.params)
        object.__setattr__(self, "params", MappingProxyType(params))
        self.recheck()

    def recheck(self) -> None:
        """Run the safety checks again (at extraction and at storage, not only at construction)."""
        safety.check_vector_class(self.vector_class)
        p = self.params
        if self.kind in CONCEPT_KINDS:
            safety.check_concept(p["word"])
        if self.kind in PAIR_KINDS:
            safety.check_texts([t for pair in p["pairs"] for t in pair],
                               what=f"contrastive text of {self.vector_id}")
        if self.kind == "sae_feature":
            safety.assert_outside_repo(p["path"], what="SAE weights")
        # last, so a harmful word is refused for what it is before the id derived from it
        safety.check_texts([self.vector_id], what="vector id")

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_json(), sort_keys=True))

    def to_json(self) -> dict[str, Any]:
        return {"vector_id": self.vector_id, "kind": self.kind, "vector_class": self.vector_class,
                "params": _plain(self.params)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "VectorRecipe":
        keys = {"vector_id", "kind", "vector_class", "params"}
        if not isinstance(d, Mapping) or set(d) != keys:
            raise RecipeError(f"a recipe has exactly the keys {sorted(keys)}")
        return cls(d["vector_id"], d["kind"], d["vector_class"], d["params"])


def _slug(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", word.casefold()).strip("_") or "x"


def concept_recipe(word: str, *, vector_id: str | None = None) -> VectorRecipe:
    return VectorRecipe(vector_id or f"concept-{_slug(word)}", "concept", "concept", {"word": word})


def response_mean_recipe(word: str, *, max_new_tokens: int, temperature: float,
                         vector_id: str | None = None) -> VectorRecipe:
    return VectorRecipe(vector_id or f"respmean-{_slug(word)}", "response_mean", "concept",
                        {"word": word, "max_new_tokens": max_new_tokens, "temperature": temperature})


def contrastive_recipe(vector_id: str, vector_class: str, pairs: Sequence[Sequence[str]], *,
                       position: str, render: str) -> VectorRecipe:
    return VectorRecipe(vector_id, "contrastive", vector_class,
                        {"pairs": pairs, "position": position, "render": render})


def unrelated_contrastive_recipe(vector_id: str, pairs: Sequence[Sequence[str]], *,
                                 position: str, render: str) -> VectorRecipe:
    return VectorRecipe(vector_id, "unrelated_contrastive", "control",
                        {"pairs": pairs, "position": position, "render": render})


def sign_randomised_recipe(base: VectorRecipe, index: int) -> VectorRecipe:
    """The same construction as `base` with each pair's order flipped by a seeded coin."""
    if base.kind not in ("contrastive", "unrelated_contrastive"):
        raise RecipeError(f"sign randomisation needs a contrastive recipe, got {base.kind!r}")
    p = base.params
    return VectorRecipe(f"{base.vector_id}.sign{index}", "sign_randomised", "control",
                        {"pairs": p["pairs"], "position": p["position"], "render": p["render"],
                         "base_id": base.vector_id, "index": index})


def gaussian_recipe(reference_id: str, index: int) -> VectorRecipe:
    return VectorRecipe(f"{reference_id}.gauss{index}", "gaussian", "control",
                        {"reference_id": reference_id, "index": index})


def random_subspace_recipe(vector_id: str, rank: int, index: int) -> VectorRecipe:
    return VectorRecipe(vector_id, "random_subspace", "control", {"rank": rank, "index": index})


def sae_recipe(vector_id: str, vector_class: str, path: str | Path, layer: int, *,
               index: int | None = None, key: str | None = None) -> VectorRecipe:
    params: dict[str, Any] = {"path": str(path), "layer": layer}
    if index is not None:
        params["index"] = index
    if key is not None:
        params["key"] = key
    return VectorRecipe(vector_id, "sae_feature", vector_class, params)


# =====================================================================================
# Rendering and the read-position checks
# =====================================================================================

def user_turn(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


@dataclass(frozen=True)
class ReadPosition:
    """Where one activation is read: the rendered prompt, the absolute token index, the text of
    that token, and (for a user turn) the generation prompt the template appended."""

    text: str
    index: int
    token_text: str
    suffix: str | None


def _tokens(backend: Backend, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    ids = [int(i) for i in backend.tokenize(text)]
    offs = [(int(s), int(e)) for s, e in backend.offsets(text)]
    if not ids:
        raise ExtractionError("the prompt tokenised to nothing")
    if len(ids) != len(offs):
        raise ExtractionError(f"{len(ids)} tokens but {len(offs)} offsets: the backend's tokenize "
                              "and offsets disagree")
    return ids, offs


def _render_user_turn(backend: Backend, user_text: str,
                      expected_suffix: str | None) -> tuple[str, str]:
    """(rendered prompt, generation suffix), after checking the suffix is real and expected."""
    full = backend.render(user_turn(user_text), add_generation_prompt=True)
    bare = backend.render(user_turn(user_text), add_generation_prompt=False)
    if user_text not in full:
        raise ExtractionError("the rendered prompt does not contain the user text verbatim; the "
                              "template altered the content")
    if not full.startswith(bare):
        raise ExtractionError("the template's generation prompt is not an extension of the "
                              "rendered user turn, so the token read cannot be verified")
    suffix = full[len(bare):]
    if not suffix:
        raise ExtractionError("the template appended no generation prompt: the last token would be "
                              "the end of the user turn, not where the reply begins")
    if expected_suffix is not None and suffix != expected_suffix:
        raise ExtractionError(
            f"the generation prompt rendered here is {suffix!r}, but the generation path uses "
            f"{expected_suffix!r}. Reading a vector at a different token from the one generation "
            "steers is the defect that withdrew the old Qwen arm (RESULTS-QWEN.md).")
    return full, suffix


def generation_suffix(backend: Backend, probe: str = "Tell me about silk") -> str:
    """The text the chat template appends for the assistant's turn, e.g. '<start_of_turn>model\\n'
    for Gemma-3 or the thinking-off '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' for Qwen3.
    Recorded in the profile at setup; every later extraction must see the same."""
    return _render_user_turn(backend, probe, None)[1]


def read_last_of_user_turn(backend: Backend, user_text: str, *,
                           expected_suffix: str | None) -> ReadPosition:
    """The last token of `user_text` rendered as a user turn with the generation prompt -- the
    exact prompt the generation path renders -- after checking it is the token intended."""
    full, suffix = _render_user_turn(backend, user_text, expected_suffix)
    ids, offs = _tokens(backend, full)
    start, end = offs[-1]
    if end != len(full):
        raise ExtractionError(f"the last token ends at character {end} of {len(full)}: position -1 "
                              "would not read the end of the generation prompt")
    if start < len(full) - len(suffix):
        raise ExtractionError("the last token begins inside the user turn; it is not the generation "
                              "prompt's own token")
    piece = full[start:end]
    decoded = backend.decode([ids[-1]])
    if decoded != piece and not (decoded and suffix.endswith(decoded)):
        raise ExtractionError(f"the token read decodes to {decoded!r}, not to the generation "
                              f"prompt's last token {piece!r}")
    return ReadPosition(full, len(ids) - 1, piece, suffix)


def _raw(backend: Backend, text: str) -> str:
    bos = backend.info.bos_token_id
    return text if bos is None else backend.decode([bos]) + text


def _check_bos(backend: Backend, ids: Sequence[int]) -> None:
    bos = backend.info.bos_token_id
    if bos is not None and ids[0] != bos:
        raise ExtractionError(f"raw text does not start with BOS ({bos}); the model would be read in "
                              "a context it never sees")


def read_last_of_raw(backend: Backend, text: str) -> ReadPosition:
    """The last token of BOS + `text` (no chat template), checked to end the text."""
    rendered = _raw(backend, text)
    ids, offs = _tokens(backend, rendered)
    _check_bos(backend, ids)
    start, end = offs[-1]
    if end != len(rendered):
        raise ExtractionError(f"the last token ends at character {end} of {len(rendered)}")
    piece = rendered[start:end]
    decoded = backend.decode([ids[-1]])
    if decoded != piece and not (decoded and rendered.endswith(decoded)):
        raise ExtractionError(f"the token read decodes to {decoded!r}, not to the text's last "
                              f"token {piece!r}")
    return ReadPosition(rendered, len(ids) - 1, piece, None)


def read_marker(backend: Backend, rendered: str, content: str, marker: str) -> ReadPosition:
    """The token holding the first character of the *last* occurrence of `marker` inside
    `content` (the trailing colon of Hua et al.'s prompts), searched only within the content so a
    marker in the template can never be read instead."""
    start = rendered.find(content)
    if start < 0:
        raise ExtractionError("the rendered prompt does not contain the content verbatim")
    rel = content.rfind(marker)
    if rel < 0:
        raise ExtractionError(f"marker {marker!r} does not occur in {content!r}")
    char = start + rel
    ids, offs = _tokens(backend, rendered)
    idx = token_index_at(offs, char)
    s, e = offs[idx]
    if not s <= char < e:
        raise ExtractionError(f"the marker's first character falls between tokens (token {idx} "
                              f"spans [{s}, {e}))")
    piece = rendered[s:e]
    decoded = backend.decode([ids[idx]])
    if decoded != piece and marker[0] not in decoded:
        raise ExtractionError(f"the token read decodes to {decoded!r}, which does not hold the "
                              f"marker {marker!r}")
    return ReadPosition(rendered, idx, piece, None)


def _read_pair_text(backend: Backend, text: str, position: str, render: str,
                    expected_suffix: str | None) -> ReadPosition:
    if render == "user_turn":
        if position == LAST:
            return read_last_of_user_turn(backend, text, expected_suffix=expected_suffix)
        full, suffix = _render_user_turn(backend, text, expected_suffix)
        r = read_marker(backend, full, text, position)
        return ReadPosition(r.text, r.index, r.token_text, suffix)
    if position == LAST:
        return read_last_of_raw(backend, text)
    rendered = _raw(backend, text)
    _check_bos(backend, _tokens(backend, rendered)[0])
    return read_marker(backend, rendered, text, position)


# =====================================================================================
# Forward passes and accumulation
# =====================================================================================

def _iter_hidden(backend: Backend, texts: Sequence[str], positions: Sequence[Sequence[int]],
                 layers: Sequence[int], batch_size: int) -> Iterator[np.ndarray]:
    """One float64 array [len(positions_i), len(layers), d_model] per text, in order."""
    d = backend.info.d_model
    for i in range(0, len(texts), batch_size):
        chunk, pos = list(texts[i:i + batch_size]), [list(p) for p in positions[i:i + batch_size]]
        res = backend.hidden(chunk, [None] * len(chunk), list(layers), pos)
        if len(res) != len(chunk):
            raise ExtractionError(f"hidden() returned {len(res)} arrays for {len(chunk)} prompts")
        for arr, p in zip(res, pos):
            a = np.asarray(arr, dtype=np.float64)
            if a.shape != (len(p), len(layers), d):
                raise ExtractionError(f"hidden() returned shape {a.shape}, expected "
                                      f"{(len(p), len(layers), d)}")
            if not np.all(np.isfinite(a)):
                raise ExtractionError("hidden() returned non-finite residuals")
            yield a


class _Diffs:
    """Running mean of per-example differences, and the running sum of their unit vectors, from
    which the mean pairwise cosine follows without storing the examples:
    sum_{i != j} u_i . u_j = ||sum u||^2 - n."""

    def __init__(self, n_layers: int, d: int) -> None:
        self.total = np.zeros((n_layers, d))
        self.units = np.zeros((n_layers, d))
        self.n = 0

    def add(self, diff: np.ndarray) -> None:
        norms = np.linalg.norm(diff, axis=1)
        if np.any(norms == 0):
            raise ExtractionError("an example's difference is exactly zero at some layer: its two "
                                  "prompts are identical up to the token read")
        self.total += diff
        self.units += diff / norms[:, None]
        self.n += 1

    def mean(self) -> np.ndarray:
        if self.n == 0:
            raise ExtractionError("no examples were accumulated")
        return self.total / self.n

    def coherence(self) -> np.ndarray | None:
        if self.n < 2:
            return None
        sq = np.einsum("ld,ld->l", self.units, self.units)
        return (sq - self.n) / (self.n * (self.n - 1))


def _fill(template: str, word: str) -> str:
    if "{word}" not in template:
        raise ExtractionError(f"template {template!r} has no {{word}} placeholder; every concept "
                              "would read the same prompt and the vector would be zero")
    try:
        text = template.format(word=word)
    except (KeyError, IndexError, ValueError) as exc:
        raise ExtractionError(f"template {template!r} cannot be filled: {exc}") from None
    return text


def _baseline(words: Sequence[str] | None, concept: str) -> list[str]:
    if words is None or isinstance(words, str) or len(words) == 0:
        raise ExtractionError("a concept vector needs its baseline words")
    seen: dict[str, str] = {}
    for w in words:
        if not isinstance(w, str) or not w.strip():
            raise ExtractionError(f"baseline word {w!r} is not a word")
        safety.check_concept(w)
        k = w.strip().casefold()
        if k in seen:
            raise ExtractionError(f"baseline word {w!r} repeats {seen[k]!r}; a repeated word is "
                                  "weighted twice in the baseline mean (Macar's list repeats "
                                  "'Butterflies'), so de-duplicate the list explicitly")
        seen[k] = w
    if concept.strip().casefold() in seen:
        raise ExtractionError(f"the concept {concept!r} is one of its own baseline words")
    return list(words)


def _check_layers(layers: Sequence[int], n_layers: int) -> list[int]:
    out = sorted({int(L) for L in layers})
    if not out:
        raise ExtractionError("no layers requested")
    bad = [L for L in out if not 0 <= L < n_layers]
    if bad:
        raise ExtractionError(f"layers {bad} are outside the model's {n_layers} layers")
    return out


def _sha(texts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]


# =====================================================================================
# Extraction
# =====================================================================================

@dataclass(frozen=True)
class Extraction:
    """An extracted vector with what is known about how it was read.

    `coherence` is per layer the mean pairwise cosine of the per-example differences (None when
    there is one example, or the kind has none). `record` is JSON-safe and holds no values.
    """

    vectors: dict[int, np.ndarray]
    coherence: dict[int, float] | None
    n_examples: int
    record: dict[str, Any]


def _concept(backend: Backend, recipe: VectorRecipe, layers: list[int], template: str | None,
             baseline_words: Sequence[str] | None, batch_size: int,
             expected_suffix: str | None) -> tuple[_Diffs, dict[str, Any]]:
    word = recipe.params["word"]
    if template is None:
        raise ExtractionError("a concept vector needs its template (CONCEPT_TEMPLATE)")
    words = _baseline(baseline_words, word)
    reads = [read_last_of_user_turn(backend, _fill(template, w), expected_suffix=expected_suffix)
             for w in [word, *words]]
    tokens = sorted({r.token_text for r in reads})
    if len(tokens) != 1:
        raise ExtractionError(f"the prompts end on different tokens {tokens}; one template must "
                              "end every prompt on the same generation-prompt token")
    acc = _Diffs(len(layers), backend.info.d_model)
    h_word: np.ndarray | None = None
    for i, a in enumerate(_iter_hidden(backend, [r.text for r in reads], [[r.index] for r in reads],
                                       layers, batch_size)):
        if i == 0:
            h_word = a[0]
        else:
            acc.add(h_word - a[0])  # type: ignore[operator]
    return acc, {"template": template, "n_baseline_words": len(words),
                 "baseline_hash": _sha(words), "read_tokens": tokens,
                 "generation_suffix": reads[0].suffix}


def _response_mean(backend: Backend, recipe: VectorRecipe, layers: list[int],
                   template: str | None, baseline_words: Sequence[str] | None, seed: int,
                   batch_size: int, expected_suffix: str | None) -> tuple[_Diffs, dict[str, Any]]:
    p = recipe.params
    if template is None:
        raise ExtractionError("a response-mean vector needs its template (CONCEPT_TEMPLATE)")
    words = [p["word"], *_baseline(baseline_words, p["word"])]
    rendered = [_render_user_turn(backend, _fill(template, w), expected_suffix) for w in words]
    replies = []
    for i in range(0, len(words), batch_size):
        # Plain sampling at the recipe's temperature (0 = greedy), stated here rather than inherited
        # from GenRequest's defaults: the sampler is part of what the vector is.
        reqs = [GenRequest(prompt=r, intervention=None, max_new_tokens=p["max_new_tokens"],
                           temperature=p["temperature"], top_p=1.0, top_k=0,
                           seed=derive_seed(seed, "response_mean", recipe.vector_id, w))
                for (r, _), w in zip(rendered[i:i + batch_size], words[i:i + batch_size])]
        got = backend.generate(reqs)
        if len(got) != len(reqs):
            raise ExtractionError(f"generate() returned {len(got)} results for {len(reqs)} requests")
        replies.extend(got)
    texts, spans = [], []
    for (prompt, _), reply, w in zip(rendered, replies, words):
        if not reply.text.strip():
            raise ExtractionError(f"the reply to {w!r} is empty; a response mean needs reply tokens")
        full = prompt + reply.text
        ids_p, ids_f = backend.tokenize(prompt), backend.tokenize(full)
        if list(ids_f[:len(ids_p)]) != list(ids_p):
            raise ExtractionError(f"appending the reply to {w!r} re-tokenised the end of the prompt; "
                                  "the reply span cannot be located")
        if len(ids_f) == len(ids_p):
            raise ExtractionError(f"the reply to {w!r} adds no tokens")
        texts.append(full)
        spans.append(list(range(len(ids_p), len(ids_f))))
    acc = _Diffs(len(layers), backend.info.d_model)
    m_word: np.ndarray | None = None
    for i, a in enumerate(_iter_hidden(backend, texts, spans, layers, batch_size)):
        m = a.mean(axis=0)
        if i == 0:
            m_word = m
        else:
            acc.add(m_word - m)  # type: ignore[operator]
    return acc, {"template": template, "n_baseline_words": len(words) - 1,
                 "baseline_hash": _sha(words[1:]), "generation_suffix": rendered[0][1],
                 "reply_tokens": int(sum(len(s) for s in spans)),
                 "max_new_tokens": p["max_new_tokens"], "temperature": p["temperature"],
                 "top_p": 1.0, "top_k": 0}


def coin_signs(n: int, seed: int, base_id: str, index: int) -> tuple[np.ndarray, int]:
    """One seeded fair coin per pair; a draw with every sign equal is redrawn (it would be the
    vector itself or its negation, not a control), and the attempt number is recorded."""
    if n < 2:
        raise ExtractionError("sign randomisation needs at least two pairs")
    for attempt in range(64):
        rng = np.random.default_rng(derive_seed(seed, "sign_randomised", base_id, index, attempt))
        s = rng.integers(0, 2, size=n) * 2 - 1
        if s.min() != s.max():
            return s.astype(np.float64), attempt
    raise ExtractionError("64 draws in a row gave every pair the same sign")  # p = 2**-64 at n = 2


def _pairs_vector(backend: Backend, recipe: VectorRecipe, layers: list[int], seed: int,
                  batch_size: int, expected_suffix: str | None) -> tuple[_Diffs, dict[str, Any]]:
    p = recipe.params
    reads = [_read_pair_text(backend, t, p["position"], p["render"], expected_suffix)
             for pair in p["pairs"] for t in pair]
    record: dict[str, Any] = {"position": p["position"], "render": p["render"],
                              "n_pairs": len(p["pairs"]),
                              "read_tokens": sorted({r.token_text for r in reads}),
                              "generation_suffix": reads[0].suffix}
    signs = np.ones(len(p["pairs"]))
    if recipe.kind == "sign_randomised":
        signs, attempt = coin_signs(len(p["pairs"]), seed, p["base_id"], p["index"])
        record["signs"] = {"n_flipped": int(np.sum(signs < 0)), "attempt": attempt}
    acc = _Diffs(len(layers), backend.info.d_model)
    it = _iter_hidden(backend, [r.text for r in reads], [[r.index] for r in reads], layers,
                      batch_size)
    for s in signs:
        h_pos, h_neg = next(it)[0], next(it)[0]
        acc.add(s * (h_pos - h_neg))
    return acc, record


def gaussian_like(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A Gaussian vector with exactly the reference's coordinate mean and variance (ddof=0), and
    therefore its norm: ||g||^2 = d (mean^2 + var) = ||reference||^2."""
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 1 or not np.all(np.isfinite(ref)):
        raise ExtractionError("the reference must be a finite 1-D vector")
    sd = float(ref.std())
    if not sd > 0:
        raise ExtractionError("a constant reference has no variance to match")
    z = rng.standard_normal(ref.size)
    z = (z - z.mean()) / z.std()
    return float(ref.mean()) + sd * z


def random_basis(d_model: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    """[rank, d_model] with orthonormal rows, uniformly distributed (QR of a Gaussian matrix with
    the sign convention that makes the factorisation unique)."""
    if not 1 <= rank <= d_model:
        raise ExtractionError(f"rank must lie in [1, {d_model}], got {rank}")
    q, r = np.linalg.qr(rng.standard_normal((d_model, rank)))
    return (q * np.sign(np.diag(r))).T


def _load_sae(recipe: VectorRecipe, layers: list[int], d_model: int) -> dict[int, np.ndarray]:
    p = recipe.params
    if layers != [p["layer"]]:
        raise ExtractionError(f"an SAE feature lives at layer {p['layer']} only; requested "
                              f"{layer_ranges(layers)}. The SAE's site must be the residual at the "
                              "output of block L (M5's layer convention)")
    path = safety.assert_outside_repo(p["path"], what="SAE weights")
    if not path.is_file():
        raise ExtractionError(f"SAE weights not found at {path}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            if p["key"] not in f.files:
                raise ExtractionError(f"{path.name} has no array {p['key']!r} (has {f.files})")
            arr = np.asarray(f[p["key"]], dtype=np.float64)
    else:
        arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if arr.ndim == 1:
        if p["index"] is not None:
            raise ExtractionError("a 1-D array is one direction; index is only for a decoder matrix")
        v = arr
    elif arr.ndim == 2:
        if p["index"] is None:
            raise ExtractionError("a decoder matrix [n_features, d_model] needs the feature index")
        if not 0 <= p["index"] < arr.shape[0]:
            raise ExtractionError(f"feature {p['index']} is outside the {arr.shape[0]} features")
        v = arr[p["index"]]
    else:
        raise ExtractionError(f"SAE weights must be 1-D or 2-D, got shape {arr.shape}")
    if v.size != d_model:
        raise ExtractionError(f"the SAE direction has {v.size} coordinates; the model has {d_model}")
    return {p["layer"]: v}


def extract_with_details(backend: Backend, recipe: VectorRecipe, layers: Sequence[int], *,
                         template: str | None = None,
                         baseline_words: Sequence[str] | None = None, seed: int, batch_size: int,
                         reference: Mapping[int, np.ndarray] | None = None,
                         expected_suffix: str | None = None) -> Extraction:
    """Build `recipe`'s vector at `layers` through the Backend protocol only.

    `template` and `baseline_words` are the concept kinds' (cfg["CONCEPT_TEMPLATE"] and the list
    cfg["BASELINE_WORDS"] names); `reference` is the per-layer vector a Gaussian control matches;
    `expected_suffix` is the generation prompt recorded in the profile (`profile.generation_suffix`),
    so a vector read under any other template is refused. `seed` is cfg["SEED"]; every random
    choice derives its own seed from it and the identity of what it is for.
    """
    recipe.recheck()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ExtractionError(f"batch_size must be an integer >= 1, got {batch_size!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ExtractionError(f"seed must be an integer, got {seed!r}")
    info = backend.info
    ls = _check_layers(layers, info.n_layers)
    d = info.d_model
    record: dict[str, Any] = {"vector_id": recipe.vector_id, "kind": recipe.kind,
                              "vector_class": recipe.vector_class, "layers": ls, "seed": seed,
                              "model": info.hf_id, "revision": info.revision,
                              "chat_template_hash": info.chat_template_hash, "created": utc_now()}
    coherence: dict[int, float] | None = None
    n_examples = 0
    if recipe.kind in CONCEPT_KINDS or recipe.kind in PAIR_KINDS:
        if recipe.kind == "concept":
            acc, extra = _concept(backend, recipe, ls, template, baseline_words, batch_size,
                                  expected_suffix)
        elif recipe.kind == "response_mean":
            acc, extra = _response_mean(backend, recipe, ls, template, baseline_words, seed,
                                        batch_size, expected_suffix)
        else:
            acc, extra = _pairs_vector(backend, recipe, ls, seed, batch_size, expected_suffix)
        record.update(extra)
        mean = acc.mean()
        vectors = {L: mean[i] for i, L in enumerate(ls)}
        coh = acc.coherence()
        coherence = None if coh is None else {L: float(coh[i]) for i, L in enumerate(ls)}
        n_examples = acc.n
    elif recipe.kind == "gaussian":
        if reference is None:
            raise ExtractionError("a Gaussian control needs the reference vector it matches")
        missing = [L for L in ls if L not in reference]
        if missing:
            raise ExtractionError(f"the reference has no entry at layers {layer_ranges(missing)}")
        p = recipe.params
        vectors = {}
        for L in ls:
            ref = np.asarray(reference[L], dtype=np.float64)
            if ref.shape != (d,):
                raise ExtractionError(f"reference at layer {L} has shape {ref.shape}, not ({d},)")
            rng = np.random.default_rng(derive_seed(seed, "gaussian", p["reference_id"],
                                                    p["index"], L))
            vectors[L] = gaussian_like(ref, rng)
        record["reference_id"] = p["reference_id"]
    elif recipe.kind == "random_subspace":
        p = recipe.params
        vectors = {L: random_basis(d, p["rank"], np.random.default_rng(
            derive_seed(seed, "random_subspace", recipe.vector_id, p["index"], L))) for L in ls}
        record["rank"] = p["rank"]
    else:  # sae_feature
        vectors = _load_sae(recipe, ls, d)
        record["sae"] = {"file": Path(recipe.params["path"]).name, "key": recipe.params["key"],
                         "index": recipe.params["index"]}
    out: dict[int, np.ndarray] = {}
    for L, v in vectors.items():
        if not np.all(np.isfinite(v)) or not np.linalg.norm(v) > 0:
            raise ExtractionError(f"the vector at layer {L} is not finite and non-zero")
        out[L] = np.asarray(v, dtype=np.float32)
    record["n_examples"] = n_examples
    return Extraction(vectors=out, coherence=coherence, n_examples=n_examples, record=record)


def extract(backend: Backend, recipe: VectorRecipe, layers: Sequence[int], *,
            template: str | None = None, baseline_words: Sequence[str] | None = None,
            seed: int, batch_size: int, reference: Mapping[int, np.ndarray] | None = None,
            expected_suffix: str | None = None) -> dict[int, np.ndarray]:
    """The per-layer vectors only (float32); see `extract_with_details`."""
    return extract_with_details(backend, recipe, layers, template=template,
                                baseline_words=baseline_words, seed=seed, batch_size=batch_size,
                                reference=reference, expected_suffix=expected_suffix).vectors


# =====================================================================================
# Diagnostics (plan §4.4): candidate per-vector predictors of detectability
# =====================================================================================

@dataclass(frozen=True)
class LibraryPC1:
    """The first principal component of a vector library at one layer, and the share of the
    library's variance along it (Macar et al. found ~18% for their concept vectors)."""

    direction: np.ndarray
    explained: float
    n_vectors: int


def library_pc1(vectors: Sequence[np.ndarray] | Mapping[str, np.ndarray]) -> LibraryPC1:
    """PCA (centred) of vectors from one layer. The sign is fixed so the vectors' mean projects
    non-negatively, which makes the component reproducible; the cosine reported is absolute."""
    vs = list(vectors.values()) if isinstance(vectors, Mapping) else list(vectors)
    if len(vs) < 2:
        raise ExtractionError("a library needs at least two vectors for a principal component")
    x = np.stack([np.asarray(v, dtype=np.float64) for v in vs])
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ExtractionError("library vectors must be finite and of one length")
    xc = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    total = float(np.sum(s ** 2))
    if not total > 0:
        raise ExtractionError("every library vector is identical; there is no variance")
    pc = vt[0]
    proj = float(x.mean(axis=0) @ pc)
    if proj < 0 or (proj == 0 and pc[np.argmax(np.abs(pc))] < 0):
        pc = -pc
    return LibraryPC1(direction=pc, explained=float(s[0] ** 2) / total, n_vectors=len(vs))


def example_coherence(diffs: Any) -> float:
    """Mean pairwise cosine of per-example differences [n, d] (Braun et al.'s predictor of
    steerability): near 1 when every example pulls the same way, near 0 when they disagree."""
    x = np.asarray(diffs, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ExtractionError("coherence needs at least two per-example differences [n, d]")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ExtractionError("a per-example difference is zero; its direction is undefined")
    u = (x / norms[:, None]).sum(axis=0)
    n = x.shape[0]
    return float((u @ u - n) / (n * (n - 1)))


def _abs_cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not (na > 0 and nb > 0):
        raise ExtractionError("cosine with a zero vector is undefined")
    return abs(float(a @ b)) / (na * nb)


def diagnostics(per_layer: Mapping[int, np.ndarray], profile: ModelProfile,
                library_pc1: Mapping[int, np.ndarray | LibraryPC1] | None = None,
                per_example_diffs: Mapping[int, np.ndarray] | None = None, *,
                coherence: Mapping[int, float] | None = None) -> dict[int, dict[str, Any]]:
    """Per layer: the norm and its off-nuisance part, f_loud and cos(raw, masked), the dose each
    unit of alpha buys (r_perp, r_raw, rho, card), |cos| with the library's first component, and
    the per-example coherence -- from `per_example_diffs`, or precomputed (`Extraction.coherence`).

    These are recorded, not used to decide anything: any predictor of detectability has to beat
    Macar et al.'s baselines (transcoder features AUC 0.90, raw vector 0.86), and vector norm
    alone does not predict it (rho = -0.15). A value that cannot be computed is None, with the
    reason under `missing`.
    """
    if per_example_diffs is not None and coherence is not None:
        raise ExtractionError("pass per-example differences or precomputed coherence, not both")
    out: dict[int, dict[str, Any]] = {}
    for L, v in sorted(per_layer.items()):
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 2:  # a subspace basis
            out[L] = {"rank": int(a.shape[0]), "orthonormality_error":
                      float(np.max(np.abs(a @ a.T - np.eye(a.shape[0]))))}
            continue
        if L not in profile.layers:
            raise ExtractionError(f"layer {L} is not in the profile; its diagnostics need s_L and "
                                  "the nuisance coordinates")
        lp = profile.layers[L]
        if a.shape != (profile.d_model,) or not np.all(np.isfinite(a)):
            raise ExtractionError(f"vector at layer {L} must be finite with shape "
                                  f"({profile.d_model},)")
        norm = float(np.linalg.norm(a))
        if not norm > 0:
            raise ExtractionError(f"vector at layer {L} is zero")
        norm_perp = float(np.linalg.norm(mask_coords(a, lp.nuisance)))
        missing: dict[str, str] = {}
        row: dict[str, Any] = {
            "norm": norm, "norm_perp": norm_perp, "n_nuisance": len(lp.nuisance),
            "f_loud": f_loud(a, lp.nuisance), "cos_raw_masked": cos_raw_masked(a, lp.nuisance),
            "r_perp_per_alpha": norm_perp / lp.s_L, "r_raw_per_alpha": norm / lp.raw_norm}
        if lp.D_rand is None:
            row["rho_per_alpha"] = None
            missing["rho_per_alpha"] = "D_rand not measured at this layer"
        else:
            row["rho_per_alpha"] = norm_perp / lp.s_L / lp.D_rand
        if profile.mean_mid_norm is None:
            row["card_per_alpha"] = None
            missing["card_per_alpha"] = "no mean norm at the middle layer"
        else:
            row["card_per_alpha"] = norm / profile.mean_mid_norm
        pc = None if library_pc1 is None else library_pc1.get(L)
        if pc is None:
            row["cos_pc1"] = None
            missing["cos_pc1"] = "no library component given for this layer"
        else:
            row["cos_pc1"] = _abs_cos(a, pc.direction if isinstance(pc, LibraryPC1) else
                                      np.asarray(pc, dtype=np.float64))
        if per_example_diffs is not None and L in per_example_diffs:
            row["coherence"] = example_coherence(per_example_diffs[L])
        elif coherence is not None and L in coherence:
            row["coherence"] = float(coherence[L])
        else:
            row["coherence"] = None
            missing["coherence"] = "no per-example differences for this layer"
        row["missing"] = missing
        out[L] = row
    return out


# =====================================================================================
# Storage
# =====================================================================================

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_no_values(obj: Any, where: str = "metadata") -> None:
    """Refuse anything in vector metadata that could carry vector values: arrays, or numeric
    lists longer than any legitimate list (a layer list)."""
    if isinstance(obj, np.ndarray):
        raise safety.Refused(f"{where} holds an array; vector values never go into metadata")
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            assert_no_values(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        numeric = sum(isinstance(x, (int, float, np.number)) and not isinstance(x, bool)
                      for x in obj)
        if numeric > _MAX_NUMERIC_LIST:
            raise safety.Refused(f"{where} is a list of {numeric} numbers; that looks like vector "
                                 "values, which never go into metadata")
        for i, x in enumerate(obj):
            assert_no_values(x, f"{where}[{i}]")


class VectorStore:
    """Per-layer vector values as `<root>/<vector_id>.npz` (arrays `L<layer>`), root outside the
    repository -- normally RUN_ROOT/vectors. Values never go anywhere else: `metadata` builds the
    JSON-safe description the run's `vectors.json` carries, and refuses to carry values."""

    def __init__(self, root: str | Path) -> None:
        self.root = safety.assert_outside_repo(root, what="vector store")

    def path(self, vector_id: str) -> Path:
        _check_id(vector_id, "vector_id")
        return self.root / f"{vector_id}.npz"

    def has(self, vector_id: str) -> bool:
        return self.path(vector_id).is_file()

    def save(self, recipe: VectorRecipe, per_layer: Mapping[int, np.ndarray], *,
             overwrite: bool = False) -> str:
        """Write the values atomically; return the file's sha256 (record it in the metadata)."""
        recipe.recheck()
        if not per_layer:
            raise ExtractionError("nothing to store")
        arrays: dict[str, np.ndarray] = {}
        for L, v in sorted(per_layer.items()):
            if isinstance(L, bool) or not isinstance(L, (int, np.integer)) or L < 0:
                raise ExtractionError(f"layer key {L!r} is not a layer")
            a = np.asarray(v, dtype=np.float32)
            if a.ndim not in (1, 2) or not np.all(np.isfinite(a)):
                raise ExtractionError(f"layer {L}: values must be a finite 1-D vector or 2-D basis")
            arrays[f"L{int(L)}"] = a
        p = self.path(recipe.vector_id)
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} exists; a stored vector is replaced only with overwrite=True")
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=p.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return _sha256_file(p)

    def load(self, vector_id: str, *, sha256: str | None = None) -> dict[int, np.ndarray]:
        """The stored values; with `sha256`, refuse a file that is not the one recorded."""
        p = self.path(vector_id)
        if not p.is_file():
            raise FileNotFoundError(f"no stored vector {vector_id!r} under {self.root}")
        if sha256 is not None and _sha256_file(p) != sha256:
            raise ExtractionError(f"{p.name} does not match the sha256 recorded for it")
        out: dict[int, np.ndarray] = {}
        with np.load(p, allow_pickle=False) as f:
            for k in f.files:
                if not re.fullmatch(r"L\d+", k):
                    raise ExtractionError(f"{p.name} holds an unexpected array {k!r}")
                out[int(k[1:])] = np.asarray(f[k])
        return dict(sorted(out.items()))

    def metadata(self, recipe: VectorRecipe, per_layer: Mapping[int, np.ndarray], *,
                 sha256: str | None, extraction: Extraction | None = None,
                 diagnostics: Mapping[int, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """The JSON-safe description for the run's `vectors.json`: recipe, layers, norms,
        extraction record, diagnostics, file digest. Never the values."""
        norms = {str(L): float(np.linalg.norm(np.asarray(v, dtype=np.float64)))
                 for L, v in sorted(per_layer.items())}
        md: dict[str, Any] = {
            "vector_id": recipe.vector_id, "recipe": recipe.to_json(),
            "layers": [int(L) for L in sorted(per_layer)],
            "shape": list(np.asarray(next(iter(per_layer.values()))).shape),
            "dtype": "float32", "norms": norms, "sha256": sha256,
            "extraction": None if extraction is None else extraction.record,
            "coherence": (None if extraction is None or extraction.coherence is None
                          else {str(L): c for L, c in extraction.coherence.items()}),
            "diagnostics": (None if diagnostics is None else
                            {str(L): dict(row) for L, row in diagnostics.items()}),
            "recorded": utc_now()}
        assert_no_values(md)
        return json.loads(json.dumps(md, default=_json_scalar))


def _json_scalar(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot put a {type(obj).__name__} into vector metadata")
