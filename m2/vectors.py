"""m2.vectors - concept-vector extraction, norms, and the dose map.

This module owns everything between "a concept word" and "an alpha you can inject":

  extract_all_layers      Macar extraction, one call per layer, cached to disk
  vec_fingerprint         content hash of a steering vector (bug 23)
  concept_first_token_ids first-token ids for E6/D3 mass, bug-20 filtered
  check_reference_norm    R5, finite and non-zero at the reference layer
  build_dose_map          alpha = r * ||h_L|| / ||v_L||, writes dose_map.json + norms.jsonl

**The public API is in r, never in alpha.** Spec section 3: at fixed alpha the real dose varies
>20x across layers and non-monotonically, so comparing layers at fixed alpha measures the
normalisation rather than the layer. Every function here that takes a dose takes `r`; alpha is
*derived*, never supplied. `build_dose_map` is the only place alpha comes into existence, and it
records a cell as unreachable rather than clamping when the alpha it would need exceeds
ALPHA_CEIL - a clamped cell is a silently different experiment (pattern 4).

Layout order (CONTRACT section 1): this module may import `config` and `model` and nothing else
from `m2`. That is why it writes its own JSONL rather than calling `runio.write_row` - `runio`
comes later in the dependency order. The row shape matches CONTRACT section 4 exactly
(`concept`, `config_hash`, `ts` on every row), so `runio.read_rows` reads these files unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from . import config
# Imported for its symbols AND for its import-time side effect: model.py calls
# ensure_repo_path() at import (defence 14 / bug 15), which is what puts Macar's src/ on
# sys.path. `import vector_utils` below therefore happens inside the functions, after this.
from . import model


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Macar's published baseline-word count. Spec section 11 does not tabulate this, so it is
# declared here rather than defaulted out of CONFIG; `config.CONFIG` may override it with an
# explicit "n_baseline_words" key, and either way the value that was actually used is stored
# in the vector cache and re-validated on load (pattern 7 - a cache key must contain
# everything the value depends on).
N_BASELINE_WORDS: int = 100

# Bug 20: 'BREAD' first-tokenizes to a bare 'B', which collects probability from every word
# beginning with B and would swamp any concept-mass measure. A variant survives only if its
# first token is a prefix of the concept at least this many characters long.
MIN_PREFIX_CHARS: int = 3

# Doses are dict keys, and 0.1 + 0.05 != 0.15 in binary floating point. Every (layer, r) key
# goes through dose_key() so a caller that arrives at the same dose by a different arithmetic
# route still hits the same cell.
R_DECIMALS: int = 6

NORMS_FILE: str = "norms.jsonl"
DOSE_MAP_FILE: str = "dose_map.json"

# ||h_L|| is a property of the model and the calibration prompts, not of the concept - in the
# M1.5 data it was identical across all six concepts at every layer (L6: 1137.03), which is
# why spec section 5.1 says to cache it across concepts. Process-local, keyed on everything
# the value depends on: config hash (model + dtype), the layer set, the exact prompt texts,
# and whether the chat template was applied.
_RESID_CACHE: dict[tuple, dict[int, dict]] = {}


# --------------------------------------------------------------------------------------
# Small internals
# --------------------------------------------------------------------------------------

def _run():
    """The process-global RunContext, or a loud error.

    Never returns a stand-in. A module that quietly proceeds without a model is how a
    measure ends up reporting a number nothing produced.
    """
    run = config.RUN
    if run is None:
        raise RuntimeError("m2.config.RUN is not set - call m2.model.load_model(CONFIG) and "
                           "m2.driver.set_concept(name) before using m2.vectors")
    return run


def _ts() -> str:
    """UTC timestamp for the `ts` field every row carries (CONTRACT section 4)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp(extra: dict) -> dict:
    """Add concept / config_hash / ts to a row. Hard indexing: a missing config_hash means
    the run is not identified and the row is worthless, so it must raise."""
    run = _run()
    row = dict(extra)
    row["concept"] = run.concept
    row["config_hash"] = run.config["config_hash"]
    row["ts"] = _ts()
    return row


def _atomic_write_text(path: Path, text: str) -> Path:
    """Write via a temp file and os.replace.

    A crash halfway through a direct write leaves a truncated final line, and a truncated
    JSONL line is a parse error in every reader downstream. Replace is atomic on both
    platforms this ever runs on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _norm_f32(vec: "torch.Tensor") -> float:
    """L2 norm computed in float32.

    Explicitly not the bf16 norm. bf16 carries 8 mantissa bits, so a norm near 4664 is
    quantised to steps of 32 (~0.7%), and alpha is *derived* from this number - the error
    would propagate into every dose in the grid.
    """
    return float(vec.detach().to(torch.float32).norm())


def _as_layer_list(layers: Iterable[int]) -> list[int]:
    """Normalise and validate a layer argument."""
    run = _run()
    out = [int(layer) for layer in layers]
    if not out:
        raise ValueError("empty layer list")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate layers in {out}")
    bad = [layer for layer in out if layer < 0 or layer >= run.n_layers]
    if bad:
        raise ValueError(f"layers {bad} outside [0, {run.n_layers}) for this model")
    return out


def dose_key(layer: int, r: float) -> tuple[int, float]:
    """The canonical (layer, r) key used by the dose map. Rounded to R_DECIMALS."""
    return (int(layer), round(float(r), R_DECIMALS))


# --------------------------------------------------------------------------------------
# Fingerprint  (bug 23)
# --------------------------------------------------------------------------------------

def vec_fingerprint(vec: "torch.Tensor | None") -> str:
    """Content fingerprint of a steering vector. 'none' for an unsteered pass.

    Hashed fresh each call rather than memoised on id(): a freed tensor can have its address
    reused, which would reintroduce exactly the aliasing this is here to prevent. Hashing
    ~10KB costs microseconds against a forward pass.

    Bug 23. Every cache key that can outlive a concept switch - the forward-pass cache and
    the judge cache - must identify the steering VECTOR, not just the grid cell. Keyed on
    (question, layer, alpha) alone, every entry still matched after switching concepts in a
    live kernel, so the previous concept's logits came back silently: a wrong number, not an
    error.
    """
    if vec is None:
        return "none"
    payload = vec.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
    return hashlib.sha1(payload).hexdigest()[:12]


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------

def baseline_words() -> list[str]:
    """Macar's baseline word list, at the count this pipeline declares.

    Membership test rather than `.get(key, default)`: an override is honoured if CONFIG
    carries one, and its absence falls back to this module's declared constant rather than
    to a silent zero (pattern 4).
    """
    from vector_utils import get_baseline_words   # after model.py put src/ on sys.path

    cfg = _run().config
    n = int(cfg["n_baseline_words"]) if "n_baseline_words" in cfg else N_BASELINE_WORDS
    words = list(get_baseline_words(n))
    if len(words) < n:
        raise RuntimeError(f"asked for {n} baseline words, repo returned {len(words)} - "
                           f"DEFAULT_BASELINE_WORDS is shorter than this pipeline assumes")
    return words


def _extractor_signature_check(fn) -> dict:
    """Read the repo's own defaults instead of restating them here.

    Pattern 1: read the code, not a summary. Two of these defaults are load-bearing and
    would be silent if they changed under us:

      normalize=False - `normalize=True` would set ||v_L|| = 1 at every layer, which does not
                        error, does not look wrong, and destroys the entire dose map, since
                        r = alpha*||v||/||h|| is defined on the unnormalised vector.
      token_idx=-1    - the last position. Correct for every row because ModelWrapper sets
                        padding_side='left' (model_utils.py:125).

    The template is not asserted, only recorded: it is part of the cache identity, so a repo
    change invalidates cached vectors instead of silently mixing two extraction recipes.
    """
    import inspect

    params = inspect.signature(fn).parameters
    if "normalize" not in params or "token_idx" not in params or "template" not in params:
        raise RuntimeError("extract_concept_vector_with_baseline has an unexpected signature; "
                           "read vector_utils.py before changing this call")
    if params["normalize"].default is not False:
        raise RuntimeError("vector_utils.extract_concept_vector_with_baseline now defaults to "
                           "normalize=True. ||v_L|| would be 1 at every layer and every dose in "
                           "the grid would be wrong without erroring. Pass normalize=False "
                           "explicitly before running anything.")
    if params["token_idx"].default != -1:
        raise RuntimeError("extract_concept_vector_with_baseline no longer extracts at the last "
                           "token; the M1.5 rig check validated token_idx=-1")
    return dict(template=str(params["template"].default),
                token_idx=int(params["token_idx"].default),
                normalize=bool(params["normalize"].default))


def _cache_identity(concept: str, words: Sequence[str], meta: dict) -> dict:
    """Everything the cached vectors depend on, so a stale file cannot be reused.

    Pattern 7: a cache key must contain everything the value depends on. config_hash covers
    the model and dtype; the rest covers the extraction recipe, which lives partly in this
    module and partly in the repo and is therefore not inside config_hash.
    """
    return dict(
        concept=concept,
        config_hash=_run().config["config_hash"],
        n_baseline_words=len(words),
        baseline_sha=_sha1_text("|".join(words)),
        extractor="vector_utils.extract_concept_vector_with_baseline",
        template=meta["template"],
        token_idx=meta["token_idx"],
        normalize=meta["normalize"],
    )


def _vector_cache_path(concept: str) -> Path:
    """run_dir/vectors/<concept>.pt.

    This file never leaves the pod: CLAUDE.md hard rule 3 and runio.EXPORT_DENY exclude
    `vectors/` and `*.pt`. It exists so a resumed run does not re-extract, not as an archive -
    regeneration is the backup.
    """
    return _run().run_dir / "vectors" / f"{concept}.pt"


def _load_vector_cache(concept: str, identity: dict) -> dict[int, "torch.Tensor"]:
    """Reload cached vectors, or return {} with a printed reason.

    Any mismatch discards the cache. Reusing vectors extracted under a different recipe is
    bug 23's shape: plausible numbers, wrong provenance.
    """
    path = _vector_cache_path(concept)
    if not path.exists():
        return {}
    try:
        # weights_only=True where available: this payload is tensors and primitives only, so
        # there is no reason to allow arbitrary unpickling of a file on a rented pod.
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            blob = torch.load(path, map_location="cpu")   # torch too old for the kwarg
    except Exception as exc:                              # noqa: BLE001 - re-extraction is cheap
        print(f"vectors    : cache unreadable ({type(exc).__name__}), re-extracting")
        return {}

    stored = {k: blob[k] for k in identity if k in blob}
    if stored != identity:
        differing = sorted(k for k in identity if blob.get(k) != identity[k])
        print(f"vectors    : cache identity mismatch on {differing} - re-extracting")
        return {}
    if "vecs" not in blob:
        print("vectors    : cache has no 'vecs' payload - re-extracting")
        return {}
    return {int(layer): vec for layer, vec in blob["vecs"].items()}


def _validate_vector(concept: str, layer: int, vec: "torch.Tensor") -> float:
    """Shape / finiteness / non-degeneracy checks. Returns the float32 norm."""
    run = _run()
    if vec.dim() != 1:
        raise ValueError(f"{concept} L{layer}: expected a 1-D vector, got shape {tuple(vec.shape)}")
    d_model = int(run.mw.d_model)
    if vec.shape[0] != d_model:
        raise ValueError(f"{concept} L{layer}: vector dim {vec.shape[0]} != model hidden dim "
                         f"{d_model} - wrong model or a cache from another run")
    norm = _norm_f32(vec)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{concept} L{layer}: vector norm is {norm} - extraction is broken, and "
                         f"every alpha derived from it would be infinite or zero")
    return norm


def extract_all_layers(concept: str, layers: list[int]) -> dict[int, "torch.Tensor"]:
    """Extract the concept vector at every requested layer and cache the result.

    One `vector_utils.extract_concept_vector_with_baseline(mw, concept, baseline_words,
    layer_idx=L)` call per layer - the same call the M1.5 lab used (cell 19), which is the one
    the rig check validated. Do not substitute another `extract_concept_vector_*`.

    **Extraction is per layer and injection happens at that same layer.** There is no
    reference-layer indirection. Bug 7 was exactly that mistake, made from a paper summary:
    Macar's `01_concept_injection.py:1765,2067` extracts per layer and injects at that layer.

    Cost note: this is 2 batched forward passes per layer (1 concept prompt + ~100 baseline
    prompts), so ~100 short passes for a 50-layer scan - a minute or two, paid once per
    concept thanks to the on-disk cache.

    Fills `RUN.vecs` and the `vec_norm` half of `RUN.norms`. It never clears `RUN.vecs`:
    clearing per concept is `driver.set_concept`'s job (CONTRACT section 2), and this function
    refuses to run when the requested concept is not the one `RUN` is set to, so a stale
    concept's vectors cannot be merged in (bug 23's shape).
    """
    from vector_utils import extract_concept_vector_with_baseline

    run = _run()
    if run.concept is not None and run.concept != concept:
        raise ValueError(f"asked to extract {concept!r} while RUN.concept is {run.concept!r} - "
                         f"call driver.set_concept first; mixing concepts in RUN.vecs is bug 23")

    layers = _as_layer_list(layers)
    words = baseline_words()
    meta = _extractor_signature_check(extract_concept_vector_with_baseline)
    identity = _cache_identity(concept, words, meta)

    vecs = _load_vector_cache(concept, identity)
    cached = sorted(set(vecs) & set(layers))
    missing = [layer for layer in layers if layer not in vecs]
    if cached:
        print(f"vectors    : {len(cached)} of {len(layers)} layers loaded from cache")

    if missing:
        t0 = time.time()
        print(f"vectors    : extracting {concept!r} at {len(missing)} layers "
              f"(L{min(missing)}-L{max(missing)}), {len(words)} baseline words")
        for i, layer in enumerate(missing, start=1):
            vecs[layer] = extract_concept_vector_with_baseline(
                run.mw, concept, words, layer_idx=layer)
            if i % 8 == 0 or i == len(missing):
                print(f"             {i}/{len(missing)} layers, {time.time()-t0:.0f}s")
        # Save the union, so a later call for extra layers (phase 5 neighbourhoods) does not
        # discard what an earlier call paid for.
        path = _vector_cache_path(concept)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = dict(identity)
        blob["vecs"] = {int(k): v for k, v in vecs.items()}
        blob["layers"] = sorted(int(k) for k in vecs)
        torch.save(blob, path)
        print(f"vectors    : extracted and saved ({time.time()-t0:.0f}s)")

    out: dict[int, "torch.Tensor"] = {}
    for layer in layers:
        vec = vecs[layer]                      # hard index: a missing layer must raise
        norm = _validate_vector(concept, layer, vec)
        out[layer] = vec
        run.vecs[layer] = vec
        entry = run.norms[layer] if layer in run.norms else {}
        entry["vec_norm"] = norm
        entry["vec_fingerprint"] = vec_fingerprint(vec)
        run.norms[layer] = entry
    return out


# --------------------------------------------------------------------------------------
# Concept token ids  (bug 20, defences 6 and 7)
# --------------------------------------------------------------------------------------

def variant_verdict(concept: str, decoded: str) -> tuple[bool, str]:
    """Pure: should a surface variant be kept, given what its FIRST token decodes to?

    Bug 20. 'BREAD' first-tokenizes to id 236799, which decodes to a bare 'B'. That token
    collects probability from every word beginning with B, so including it does not add
    concept signal - it adds a wall of unrelated mass on top of the measure. A variant
    survives only if its first token is a prefix of the concept of at least
    MIN_PREFIX_CHARS characters.

    Kept pure (str in, verdict out) so tests/test_offline.py can exercise the rule without a
    tokenizer, a model or a GPU.
    """
    clean = decoded.strip().lower()
    if not clean:
        return False, "first token decodes to whitespace"
    if len(clean) < MIN_PREFIX_CHARS:
        return False, f"first token {clean!r} is under {MIN_PREFIX_CHARS} characters (bug 20)"
    if not concept.lower().startswith(clean):
        return False, f"first token {clean!r} is not a prefix of {concept.lower()!r}"
    return True, "prefix of the concept"


def concept_first_token_ids(concept: str) -> tuple[list[int], list, list]:
    """First-token ids for the concept, over bare AND leading-space surface forms.

    Returns `(ids, kept, dropped)`. `kept` and `dropped` are lists of dicts so the caller can
    log exactly which variants survived and why the others did not; `dropped` entries carry a
    `reason`.

    Two defences, both from real failures:

    - **Both forms are scored** (defence 6). After a word like "about", the tokenizer
      distinguishes " velocity" from "velocity". Scoring only the bare form reads near-zero
      everywhere, which is indistinguishable from a genuinely covert cell - spec section 5.3.
    - **Bug 20's prefix filter** (defence 7), via variant_verdict.

    Raises when nothing survives. An empty id list is not an empty result: `probs[[]].sum()`
    is 0.0, so every downstream E6 / D3 reading would be a silent zero, which is precisely
    the "wrong number rather than an error" class this pipeline is built to exclude
    (pattern 4). The message carries the full kept/dropped table so the caller can log it.
    """
    tok = _run().tok
    ids: set[int] = set()
    kept: list[dict] = []
    dropped: list[dict] = []

    # Same enumeration as M1.5 cell 19's [S6] block: three casings, each bare and
    # leading-space. Plurals and possessives share the singular's first token, so they are
    # already covered.
    for form in (concept.lower(), concept.capitalize(), concept.upper()):
        for variant in (form, " " + form):
            enc = tok.encode(variant, add_special_tokens=False)   # bug 9: never add <bos> here
            if not enc:
                dropped.append(dict(variant=variant, token_id=None, decodes_to=None,
                                    reason="variant encodes to zero tokens"))
                continue
            token_id = int(enc[0])
            decoded = tok.decode([token_id])
            ok, reason = variant_verdict(concept, decoded)
            if ok:
                ids.add(token_id)
                kept.append(dict(variant=variant, token_id=token_id, decodes_to=decoded,
                                 whole_word=decoded.strip().lower() == concept.lower(),
                                 reason=reason))
            else:
                dropped.append(dict(variant=variant, token_id=token_id, decodes_to=decoded,
                                    reason=reason))

    if not ids:
        raise ValueError(
            f"no usable first-token ids for concept {concept!r}: every surface form was "
            f"dropped {dropped}. A zero-length id set makes every concept-mass reading a "
            f"silent 0.0; pick a concept that tokenizes cleanly or revisit bug 20's filter.")

    # A kept token that is a strict prefix rather than the whole word also collects
    # probability from unrelated completions - 'orig' picks up origin, original, originally.
    # Flagged, not dropped: it is the concept's own leading token and excluding it would lose
    # real signal.
    for entry in kept:
        entry["prefix_only"] = not entry["whole_word"]
    return sorted(ids), kept, dropped


# --------------------------------------------------------------------------------------
# R5 - reference-layer vector norm  (bug 19)
# --------------------------------------------------------------------------------------

def reference_norm_verdict(vec_norm: float, ref_layer: int) -> tuple[bool, str]:
    """Pure R5: the vector norm must be finite and non-zero; always report the number."""
    norm = float(vec_norm)
    passed = math.isfinite(norm) and norm > 0.0
    detail = f"L{int(ref_layer)} vector norm {norm:.6g}"
    return (True, detail + "; finite and non-zero") if passed else (
        False, detail + "; extraction is broken because the norm is zero or non-finite")


def check_reference_norm(ref_layer: int) -> tuple[bool, str]:
    """R5. At the reference layer, require only a finite, non-zero norm and report it.

    A model-specific magnitude band is not an instrument check: norms scale with architecture
    and depth. Zero/NaN/Inf is the portable failure that proves extraction returned no usable
    direction; R14 and the escalation ladder own behavioural validation.
    """
    run = _run()
    vec = run.vecs[int(ref_layer)]    # hard index: R5 on a layer we never extracted is a bug
    return reference_norm_verdict(_norm_f32(vec), int(ref_layer))


# --------------------------------------------------------------------------------------
# Residual-stream norms and the dose map
# --------------------------------------------------------------------------------------

def alpha_from_norms(r: float, vec_norm: float, resid_norm: float) -> float:
    """Pure: alpha = r * ||h_L|| / ||v_L||  (spec section 3).

    Duplicated deliberately against `config.alpha_for`, which is the module the rest of the
    pipeline calls. build_dose_map runs both and refuses to continue if they disagree: the
    formula and its inverse differ only in which norm is on top, an inversion that would
    rescale the entire grid without erroring anywhere.
    """
    r = float(r)
    if r < 0.0:
        raise ValueError(f"negative dose r={r}")
    if not (vec_norm > 0.0 and math.isfinite(vec_norm)):
        raise ValueError(f"||v_L|| = {vec_norm} - extraction is broken")
    if not (resid_norm > 0.0 and math.isfinite(resid_norm)):
        raise ValueError(f"||h_L|| = {resid_norm} - the residual-norm pass is broken")
    return r * resid_norm / vec_norm


def dose_from_alpha(alpha: float, vec_norm: float, resid_norm: float) -> float:
    """Pure inverse: r = alpha * ||v_L|| / ||h_L||."""
    if not (vec_norm > 0.0 and math.isfinite(vec_norm)):
        raise ValueError(f"||v_L|| = {vec_norm} - extraction is broken")
    if not (resid_norm > 0.0 and math.isfinite(resid_norm)):
        raise ValueError(f"||h_L|| = {resid_norm} - the residual-norm pass is broken")
    return float(alpha) * vec_norm / resid_norm


def measure_residual_norms(layers: list[int],
                           calib_prompts: Sequence[str],
                           *,
                           apply_chat_template: bool = True) -> dict[int, dict]:
    """||h_L|| over the calibration prompts, written into `RUN.norms`.

    Spec section 5.1 step 2 and section 3: `||h_L||` is the mean L2 norm of the residual
    stream at layer L over the calibration prompts (the 12 E5 prompts). Within a prompt the
    aggregate is whatever `model.residual_norms` returns for that layer - the M1.5 lab used
    the median over token positions - and across prompts it is the mean, reported with its SE
    (pattern 9: a deterministic measure still needs a sample size, and N here is the number
    of distinct prompts).

    `calib_prompts` are raw user questions; the chat template is applied here, because that
    is the context steering actually runs in. A prompt that is already templated raises
    rather than being templated twice.

    The prompt set is passed in rather than imported: `prompts.py` comes after this module in
    the CONTRACT dependency order.
    """
    run = _run()
    layers = _as_layer_list(layers)
    texts = [str(p) for p in calib_prompts]
    if not texts:
        raise ValueError("no calibration prompts - ||h_L|| would be undefined")

    if apply_chat_template:
        specials = [s for s in getattr(run.tok, "all_special_tokens", []) or [] if s]
        for text in texts:
            hit = [s for s in specials if s in text]
            if hit:
                raise ValueError(f"calibration prompt already contains special tokens {hit} - "
                                 f"pass raw questions, or apply_chat_template=False")

    key = (run.config["config_hash"], tuple(layers),
           _sha1_text("\x00".join(texts)), bool(apply_chat_template))
    if key in _RESID_CACHE:
        cached = _RESID_CACHE[key]
        print(f"resid norms: reused across concepts ({len(texts)} prompts, cached)")
    else:
        t0 = time.time()
        per_prompt: list[dict[int, float]] = []
        for text in texts:
            prompt = model.chat(text) if apply_chat_template else text
            per_prompt.append(model.residual_norms(prompt, layers))
        cached = {}
        for layer in layers:
            xs = [d[layer] for d in per_prompt]    # hard index: a missing layer must raise
            mean, se, n = model.mean_se(xs)
            if mean is None or not math.isfinite(mean) or mean <= 0.0:
                raise ValueError(f"L{layer}: ||h|| came out {mean} over {n} prompts")
            cached[layer] = dict(resid_norm=mean, resid_norm_se=se, resid_n=n)
        _RESID_CACHE[key] = cached
        print(f"resid norms: measured over {len(texts)} prompts, {len(layers)} layers "
              f"({time.time()-t0:.0f}s)")

    for layer in layers:
        entry = run.norms[layer] if layer in run.norms else {}
        entry.update(cached[layer])
        run.norms[layer] = entry
    return {layer: dict(cached[layer]) for layer in layers}


def write_norms_jsonl(layers: list[int]) -> Path:
    """Write run_dir/norms.jsonl - one row per layer, ||v_L|| and ||h_L||.

    Rewritten whole rather than appended, because it is a complete per-concept snapshot: a
    resumed run that appended would accumulate duplicate and possibly contradictory rows for
    the same layer. One JSON object per line, so runio.read_rows reads it unchanged.
    """
    run = _run()
    layers = _as_layer_list(layers)
    lines = []
    for layer in layers:
        entry = run.norms[layer]                  # hard index: an unmeasured layer must raise
        lines.append(json.dumps(_stamp(dict(
            measure="norms",
            layer=layer,
            vec_norm=entry["vec_norm"],
            vec_fingerprint=entry["vec_fingerprint"],
            resid_norm=entry["resid_norm"],
            resid_norm_se=entry["resid_norm_se"],
            resid_n=entry["resid_n"],
        ))))
    return _atomic_write_text(run.run_dir / NORMS_FILE, "\n".join(lines) + "\n")


def _alpha_for_cell(layer: int, r: float) -> tuple[float | None, str | None]:
    """alpha for one (layer, r), or (None, reason) when the cell is unreachable.

    Calls `config.alpha_for` - the single source of truth the rest of the pipeline uses - and
    cross-checks it against this module's own arithmetic. A disagreement means one of the two
    has the formula inverted, which would rescale every dose in the grid silently, so it
    raises rather than picking a winner.

    Unreachable is recorded, never clamped. Clamping alpha to ALPHA_CEIL would quietly run a
    different dose than the one the row claims (spec section 8, phase 1).
    """
    run = _run()
    ceil = float(config.CONSTANTS["ALPHA_CEIL"])
    entry = run.norms[layer]
    mine = alpha_from_norms(r, entry["vec_norm"], entry["resid_norm"])

    try:
        theirs = float(config.alpha_for(layer, r))
    except config.Unreachable:
        if mine <= ceil:
            raise RuntimeError(
                f"config.alpha_for(L{layer}, r={r}) declared the cell unreachable but this "
                f"module computes alpha={mine:.3f} <= ALPHA_CEIL={ceil}. One of the two has "
                f"the dose formula inverted; do not run until they agree.")
        return None, f"alpha {mine:.2f} > ALPHA_CEIL {ceil}"

    if not math.isclose(mine, theirs, rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError(
            f"dose formula disagreement at L{layer}, r={r}: config.alpha_for gives {theirs!r}, "
            f"vectors.alpha_from_norms gives {mine!r}. alpha = r*||h||/||v|| (spec section 3); "
            f"an inverted pair rescales the whole grid without erroring.")
    if theirs > ceil:
        # Defence in depth: alpha_for is contracted to raise here, so reaching this line means
        # it does not. Record unreachable anyway rather than injecting an over-ceiling dose.
        return None, f"alpha {theirs:.2f} > ALPHA_CEIL {ceil}"
    return theirs, None


def build_dose_map(layers: list[int],
                   doses: Sequence[float],
                   *,
                   calib_prompts: Sequence[str] | None = None,
                   write: bool = True) -> dict:
    """The inverse dose map: `{(layer, r): alpha}`, with `None` for unreachable cells.

    alpha = r * ||h_L|| / ||v_L||  (spec section 3). **All layer comparison happens in r**;
    alpha exists only so the hook has a number to multiply by. At fixed alpha the real dose
    varies >20x across layers and non-monotonically (irony: L12 r=0.142 -> L21 r=0.043), so a
    fixed-alpha scan measures the normalisation rather than the layer.

    Requires `extract_all_layers` to have run (||v_L||) and `measure_residual_norms` to have
    run (||h_L||); pass `calib_prompts` to have the second done here. Missing norms raise -
    they are never treated as absent-therefore-zero.

    Writes `dose_map.json` and `norms.jsonl` into `run_dir`. Keys are `dose_key(layer, r)`.
    """
    run = _run()
    layers = _as_layer_list(layers)
    doses = [float(d) for d in doses]
    if not doses:
        raise ValueError("no doses given")
    if any(d < 0.0 for d in doses):
        raise ValueError(f"negative dose in {doses}")

    if calib_prompts is not None:
        measure_residual_norms(layers, calib_prompts)

    for layer in layers:
        entry = run.norms[layer] if layer in run.norms else {}
        for field in ("vec_norm", "resid_norm"):
            if field not in entry:
                raise RuntimeError(
                    f"L{layer} has no {field}. Run extract_all_layers (||v||) and "
                    f"measure_residual_norms (||h||) before build_dose_map - an absent norm "
                    f"must not become a dose.")

    dose_map: dict[tuple[int, float], float | None] = {}
    cells: list[dict] = []
    n_unreachable = 0
    for layer in layers:
        entry = run.norms[layer]
        for r in doses:
            alpha, reason = _alpha_for_cell(layer, r)
            dose_map[dose_key(layer, r)] = alpha
            if alpha is None:
                n_unreachable += 1
            cells.append(dict(layer=layer, r=round(r, R_DECIMALS), alpha=alpha,
                              reachable=alpha is not None, reason=reason,
                              vec_norm=entry["vec_norm"], resid_norm=entry["resid_norm"]))

    print("")
    print("dose map - alpha = r * ||h_L|| / ||v_L||   (all comparison happens in r)")
    header = f"   {'layer':>6}{'||v||':>10}{'||h||':>10}"
    for r in doses:
        header += f"{'a@r=' + format(r, '.3g'):>12}"
    print(header)
    for layer in layers:
        entry = run.norms[layer]
        line = (f"   {'L' + str(layer):>6}{entry['vec_norm']:>10.0f}"
                f"{entry['resid_norm']:>10.0f}")
        for r in doses:
            alpha = dose_map[dose_key(layer, r)]
            line += f"{'UNREACHABLE' if alpha is None else format(alpha, '.3f'):>12}"
        print(line)
    print(f"   {len(cells)} cells, {n_unreachable} unreachable at "
          f"ALPHA_CEIL={float(config.CONSTANTS['ALPHA_CEIL'])} (recorded, never clamped)")

    if write:
        payload = _stamp(dict(
            alpha_ceil=float(config.CONSTANTS["ALPHA_CEIL"]),
            formula="alpha = r * resid_norm / vec_norm",
            layers=layers,
            doses=[round(d, R_DECIMALS) for d in doses],
            n_cells=len(cells),
            n_unreachable=n_unreachable,
            cells=cells,
        ))
        _atomic_write_text(run.run_dir / DOSE_MAP_FILE, json.dumps(payload, indent=2) + "\n")
        write_norms_jsonl(layers)
        print(f"   wrote {DOSE_MAP_FILE} and {NORMS_FILE}")

    return dose_map
