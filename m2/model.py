"""m2.model - model load, the injection hook, encode/chat, residual norms.

Everything in this module is the layer between M2 and the model: loading it through Macar's
`ModelWrapper`, getting a steering vector into the residual stream, turning a question into
tokens, and reading the residual-stream norms the dose map is built from.

Ported from the v1 measurement lab (`measurement_lab.ipynb`, Setup 6/7/8), which is
pod-validated on this exact model and this exact repo integration. Where this file departs
from the lab the comment says which bug or which section of `M2 - Specification.md` forced
it, so the next reader does not undo it.

Two things in here exist only because they already went wrong once:

  * `injected` is OUR forward hook, never `steering_utils.SteeringHook`. Bug 26.
  * `hook_liveness()` (R14) proves the hook is alive on BOTH injection paths before a sweep
    is allowed to spend an hour measuring an unsteered model. Bug 26 again.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

# torch is always present on the pod. It is guarded here only so that the pure helpers
# (`mean_se`) and this module's names remain importable in an offline checkout where the
# tests run without a GPU stack. This is NOT a silent default: every function that needs a
# tensor goes through `_run()` or `_require_torch()`, both of which raise.
try:  # pragma: no cover - environment dependent
    import torch
except ModuleNotFoundError:  # pragma: no cover - offline checkout
    torch = None  # type: ignore[assignment]

from . import config

__all__ = [
    "ensure_repo_path",
    "load_model",
    "injected",
    "chat",
    "encode",
    "encode_batch",
    "start_pos_for",
    "mean_se",
    "vec_tag",
    "logits_for",
    "cache_clear",
    "residual_norms",
    "hook_liveness",
]


# =====================================================================================
# Repo import path
# =====================================================================================

def ensure_repo_path(verbose: bool = False) -> tuple[Path, list[str]]:
    """Put Macar's `src/` and `experiments/` on `sys.path`, idempotently.

    BUG 15. `sys.path` is per-process and is lost on a kernel restart, while the files stay
    installed on the volume. Skipping the install cell after a restart is a reasonable thing
    to do - the install really is done - so the path has to heal itself at every entry point
    rather than only where the clone happens.

    Returns the repo root and the subdirectories this call added (empty if already on path).
    """
    repo = Path(os.environ.get("WORK_DIR", "/workspace/steering-opt")) / "introspection-mechanisms"
    added: list[str] = []
    for sub in ("src", "experiments"):
        d = str(repo / sub)
        if (repo / sub).is_dir() and d not in sys.path:
            sys.path.insert(0, d)
            added.append(sub)
    if verbose:
        if not repo.exists():
            print(f"repo path  : {repo} (not cloned yet)")
        else:
            print(f"repo path  : {repo}"
                  + (f"  (added {', '.join(added)})" if added else "  (already on path)"))
    return repo, added


# BUG 15, defence 14. Called at IMPORT of this module, not only where the repo is installed:
# `import m2.model` in a fresh kernel must be sufficient to make `from model_utils import ...`
# work inside load_model() below.
ensure_repo_path()


# =====================================================================================
# Access to the process-global run context
# =====================================================================================

def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "torch is not installed in this interpreter - m2.model's GPU paths cannot run "
            "here. Only the pure helpers (mean_se) are usable offline.")


def _run() -> Any:
    """The one mutable process-global, `config.RUN` (CONTRACT section 2).

    Read through this accessor rather than `from .config import RUN`: `load_model` rebinds
    the attribute on the config module, and a from-import would capture the pre-load `None`
    forever. The `getattr` default here is not a load-bearing default - it raises on the
    next line rather than becoming a usable value (DEBUG LOG pattern 4).
    """
    ctx = getattr(config, "RUN", None)
    if ctx is None:
        raise RuntimeError(
            "m2.config.RUN is not set - call m2.model.load_model(CONFIG) before any "
            "measurement. Nothing in this module works without it.")
    return ctx


def _input_device() -> Any:
    """Device tokenised inputs must land on.

    `ModelWrapper._get_input_device()` is `next(model.parameters()).device`, which is what
    every repo call site uses and is correct under `device_map='auto'` sharding. Falling
    back to `hf.device` is the v1 lab's path and evaluates to the same thing; it is here so
    a private-method rename upstream degrades to the validated behaviour instead of an
    AttributeError.
    """
    ctx = _run()
    getter = getattr(ctx.mw, "_get_input_device", None)
    if getter is not None:
        return getter()
    return ctx.hf.device


# =====================================================================================
# Model load
# =====================================================================================

def load_model(cfg: dict) -> Any:
    """Load the model through Macar's wrapper and populate `config.RUN`.

    Ported from the lab's Setup 6 (cell 15). Returns the `RunContext` and also installs it
    as `m2.config.RUN`, which is where every other module reads it from.

    Every key read from `cfg` is hard-indexed. A missing 'model' or 'dtype' must be a
    KeyError, not a default that quietly loads the wrong thing (DEBUG LOG pattern 4).
    """
    _require_torch()

    # BUG 15 again: this cell can be re-run after a kernel restart without the install cell.
    ensure_repo_path()
    from model_utils import load_model as _repo_load_model  # noqa: WPS433 (deferred by design)

    mw = _repo_load_model(cfg["model"], dtype=cfg["dtype"])
    hf, tok = mw.model, mw.tokenizer

    # DEFENCE 5. `model_utils.py:125` sets padding_side='left', and S3 depends on it: it
    # batches all 57 MMLU items in one padded forward pass and reads logits[:, -1, :], which
    # is the next-token position for every row ONLY under left padding (spec section 5.4).
    # An explicit raise rather than `assert`, because `python -O` strips asserts and this
    # check failing silently would corrupt every S3 in the run.
    if tok.padding_side != "left":
        raise RuntimeError(
            f"tokenizer padding_side is {tok.padding_side!r}, must be 'left'. "
            "S3's batched last-position read (spec 5.4) and bug 25b's decode overhang both "
            "depend on it. model_utils.py:125 sets it; something has overwritten it.")

    # Bug 5: batched generation needs a pad token id. The wrapper sets it from eos; check
    # rather than trust, because a None here surfaces much later as an opaque generate error.
    if tok.pad_token_id is None:
        raise RuntimeError("tokenizer has no pad_token_id - batched generation cannot run")

    # Bug 4: the layer count must agree with the stack `get_layer_module` actually indexes.
    # `mw.n_layers` counts `model.language_model.layers` for Gemma3 (checked FIRST, per the
    # repo's own warning); the config value is the paper's number. If they disagree, every
    # depth fraction in the spec points somewhere else.
    n_layers = int(mw.n_layers)
    cfg_layers = (getattr(hf.config, "num_hidden_layers", None)
                  or getattr(getattr(hf.config, "text_config", None), "num_hidden_layers", None))
    if cfg_layers is not None and int(cfg_layers) != n_layers:
        raise RuntimeError(
            f"layer-count mismatch: wrapper reports {n_layers}, config reports {cfg_layers}. "
            "get_layer_module would be indexing a different stack than the config describes.")

    concept = cfg["concept"]
    run_dir = None
    if concept:
        run_dir = config.run_dir_for(concept, cfg)
        Path(run_dir).mkdir(parents=True, exist_ok=True)

    ctx = config.RunContext(
        mw=mw,
        hf=hf,
        tok=tok,
        n_layers=n_layers,
        concept=concept,
        config=cfg,
        run_dir=run_dir,
        vecs={},        # per concept; driver.set_concept clears and rebuilds these three
        norms={},
        base={},
        mmlu=[],
    )
    # Rebinding the module attribute, so that `config.RUN` is the singleton every module
    # sees. Assigning to a local imported name would leave other modules on the old value.
    config.RUN = ctx

    print(f"model      : {cfg['model']} ({cfg['dtype']})")
    print(f"layers     : {n_layers}")
    print(f"padding    : {tok.padding_side}  (left is required - spec 5.4)")
    if torch.cuda.is_available():
        print("VRAM       : %.1f GB" % (torch.cuda.memory_allocated() / 1e9))
    print(f"run_dir    : {run_dir}")
    return ctx


# =====================================================================================
# The injection hook
# =====================================================================================

class injected:
    """Apply the injection for the duration of a `with` block.

    `start_pos` leaves the chat template unsteered, matching Macar's detection test
    (`steering_utils.py:618` computes the same position). Passing `alpha=0` or `vec=None`
    gives a clean unsteered pass, which is how every baseline in the pipeline is taken.

    BUG 26 - THIS MUST NOT BECOME `steering_utils.SteeringHook`. That class's fallback for a
    layer whose output is not a tuple (`steering_utils.py:119`) reads:

        else:                                   # non-tuple output
            if self.start_pos is None:
                return output + steering_vec.view(1, 1, -1)
            else:
                return output                   # <- unmodified. No steering. No error.

    Gemma3's decoder layers return a plain tensor, so every call carrying a start_pos was
    silently unsteered. In the 2026-08-04 run that read exactly 0.000 at all 30 cells, on
    the two measures that used it, and no gate caught it - a mean of zero with a variance of
    zero satisfies "the effect is separable from noise".

    So the hook is ours: the same semantics the repo intends, applied to BOTH output shapes,
    and it RAISES rather than returning quietly whenever it cannot steer. A measure that
    reads zero everywhere must fail loudly, not pass.
    """

    def __init__(self, vec: Any, layer: int, alpha: float, start_pos: int | None = None) -> None:
        self.vec = vec
        self.layer = int(layer)
        self.alpha = float(alpha)
        self.start_pos = start_pos
        self.handle = None
        self.fired = 0       # hook invocations; checked on exit, see __exit__

    def _hook(self, module: Any, inp: Any, out: Any) -> Any:
        self.fired += 1
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        v = (self.vec * self.alpha).to(h.device).to(h.dtype)
        if v.shape[0] != h.shape[-1]:
            raise ValueError(f"steering vector dim {v.shape[0]} != hidden dim {h.shape[-1]} "
                             f"- wrong layer or wrong model")
        seq = h.shape[1]
        if seq == 1 or self.start_pos is None:
            # seq==1 is the generation phase: steer every new token, as the repo does.
            out_h = h + v.view(1, 1, -1)
        elif h.shape[0] > 1:
            # BUG 25b. Padding is left, so a scalar start position is only correct when every
            # row has the same unpadded length - and "Trial 1" is one token shorter than
            # "Trial 25". `generate_batch_with_steering` gets exactly this wrong, silently,
            # steering shorter prompts one token too early. Refuse rather than repeat it:
            # batch with start_pos=None (raw text, as S3 does), or go through
            # `generate_batch_with_multi_steering`, which corrects each row for its own
            # padding (model_utils.py:1248).
            raise ValueError(
                f"batched forward pass (batch={h.shape[0]}) with start_pos={self.start_pos}: "
                "a scalar start position mis-steers left-padded rows of unequal length. "
                "This is bug 25b. Use start_pos=None, or the multi-steering path.")
        elif self.start_pos < seq:
            out_h = h.clone()
            out_h[:, self.start_pos:, :] += v.view(1, 1, -1)
        else:
            raise ValueError(f"start_pos {self.start_pos} >= seq_len {seq}: nothing would be "
                             f"steered, and silently measuring an unsteered model is how "
                             f"bug 26 happened")
        return ((out_h,) + out[1:]) if is_tuple else out_h

    def __enter__(self) -> "injected":
        if self.vec is not None and self.alpha:
            self.handle = _run().mw.get_layer_module(self.layer).register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            # BUG 26's remaining silent path: a hook that is registered on a module the
            # forward pass never reaches steers nothing and reports nothing. Only checked
            # when the block exited cleanly - raising here during an exception would mask
            # the real error.
            if exc_type is None and self.fired == 0:
                raise RuntimeError(
                    f"injection hook at layer {self.layer} never fired: no forward pass "
                    f"reached it, so nothing was steered. This is bug 26's shape.")
        return False


# =====================================================================================
# Prompt plumbing
# =====================================================================================

class injected_multi:
    """Inject at SEVERAL layers at once, for the multi-layer arm (`m2.multilayer`).

    `plan` maps layer -> (vector, alpha). One hook per layer, all live for the block, each
    with the same semantics as `injected`. The hypothesis being tested is that spreading a
    fixed total dose across k layers keeps influence while lowering forced identification.

    **Per-row mask instead of a scalar start position.** `injected` refuses a batched forward
    pass with a scalar `start_pos` because left padding makes it wrong (bug 25b), and for
    generation the repo's `generate_batch_with_multi_steering` solves that internally - but
    only for the one layer it owns. There is no multi-layer equivalent, so the mask is built by
    the caller from the attention mask (`multilayer.steering_mask`) and passed in explicitly:
    `mask[i, t]` is True where row i should be steered. That is the same correction the repo
    applies at `model_utils.py:1248`, done once here rather than per layer.

    During generation `seq == 1` and every new token is steered, matching both `injected` and
    the repo.
    """

    def __init__(self, plan: dict, mask: Any = None) -> None:
        if not plan:
            raise ValueError("injected_multi needs at least one layer")
        self.plan = {int(layer): (vec, float(alpha)) for layer, (vec, alpha) in plan.items()}
        for layer, (vec, alpha) in self.plan.items():
            if vec is None or not alpha:
                raise ValueError(
                    f"layer {layer} has vec={vec is not None} alpha={alpha}: an inert layer in "
                    "a multi-layer plan silently reduces k, so the row would be labelled k "
                    "when the model saw k-1")
        self.mask = mask
        self.handles: list = []
        self.fired: dict[int, int] = {layer: 0 for layer in self.plan}

    def _make_hook(self, layer: int):
        vec, alpha = self.plan[layer]

        def _hook(module: Any, inp: Any, out: Any) -> Any:
            self.fired[layer] += 1
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            v = (vec * alpha).to(h.device).to(h.dtype)
            if v.shape[0] != h.shape[-1]:
                raise ValueError(f"steering vector dim {v.shape[0]} != hidden dim "
                                 f"{h.shape[-1]} at layer {layer}")
            seq = h.shape[1]
            if seq == 1 or self.mask is None:
                out_h = h + v.view(1, 1, -1)
            else:
                m = self.mask
                if tuple(m.shape) != (h.shape[0], seq):
                    raise ValueError(
                        f"steering mask {tuple(m.shape)} does not match hidden states "
                        f"{(h.shape[0], seq)} at layer {layer}. A mismatched mask steers the "
                        "wrong positions silently, which is bug 25b by another route.")
                if not bool(m.any()):
                    raise ValueError(
                        f"steering mask selects no position at layer {layer}: nothing would be "
                        "steered, and silently measuring an unsteered model is bug 26")
                out_h = h.clone()
                out_h[m.to(h.device)] += v.view(1, -1)
            return ((out_h,) + out[1:]) if is_tuple else out_h

        return _hook

    def __enter__(self) -> "injected_multi":
        mw = _run().mw
        for layer in sorted(self.plan):
            self.handles.append(
                mw.get_layer_module(layer).register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if exc_type is None:
            dead = [layer for layer, n in self.fired.items() if n == 0]
            if dead:
                raise RuntimeError(
                    f"layers {dead} never fired: hooks were registered on modules the forward "
                    f"pass did not reach, so the plan steered fewer layers than it claims. "
                    f"This is bug 26's remaining silent path.")
        return False


def chat(question: str) -> str:
    """Render a user question through the chat template.

    The returned string already contains <bos>, which is why everything that tokenises it
    passes add_special_tokens=False (bug 9).
    """
    return _run().tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)


def start_pos_for(prompt: str, needle: str) -> int | None:
    """Token index where `needle` begins, so the template before it stays unsteered.

    Mirrors `steering_utils.py:618`: find the needle, tokenise the prefix, step back one
    token. add_special_tokens=False because apply_chat_template already emitted <bos>
    (bug 9); with it, every position here would be off by one.

    Returns None when the needle is not in the prompt - callers must treat that as a
    construction error, never as "steer everywhere".
    """
    if needle not in prompt:
        return None
    before = prompt[:prompt.find(needle)]
    tok = _run().tok
    return max(0, len(tok(before, add_special_tokens=False)["input_ids"]) - 1)


def encode(prompt: str) -> dict:
    """Tokenise ONE chat-template prompt. No extra <bos> - the template has one (bug 9).

    For raw (non-templated) text use `encode_batch(..., add_special_tokens=True)`, which
    takes the flag explicitly.
    """
    ctx = _run()
    return ctx.tok(prompt, return_tensors="pt", add_special_tokens=False).to(_input_device())


def encode_batch(prompts: list[str], *, add_special_tokens: bool) -> dict:
    """Left-padded batch tokenisation, for S3's single batched forward pass (spec 5.4).

    `add_special_tokens` is REQUIRED and deliberately has no default. This is bug 9: the
    correct value depends on where the prompt came from, and either default is silently
    wrong for half the callers.

        add_special_tokens=False  - the prompt came through `apply_chat_template`, which
                                    already emits <bos>. Passing True doubles it and shifts
                                    every token position by one. Used by any batched
                                    chat-rendered caller.
        add_special_tokens=True   - the prompt is RAW text with no template. The MMLU items
                                    of spec 4.4 are exactly this ("no chat template, no CoT
                                    instruction, no system prompt"), so `measure_S3` passes
                                    True.

    Left padding is re-checked here rather than trusted from load time, because it is what
    makes `logits[:, -1, :]` the next-token position for every row regardless of item length
    (defence 5). Truncation is off: right-side truncation would silently drop the trailing
    "Answer:" that S3's last-position read depends on, and an over-length item should be a
    visible problem rather than a corrupted row.
    """
    ctx = _run()
    if ctx.tok.padding_side != "left":
        raise RuntimeError(
            f"padding_side is {ctx.tok.padding_side!r} at batch-encode time, must be 'left' - "
            "a batched logits[:, -1, :] read is wrong for every short row under right padding.")
    return ctx.tok(prompts, return_tensors="pt", padding=True, truncation=False,
                   add_special_tokens=add_special_tokens).to(_input_device())


# =====================================================================================
# Statistics
# =====================================================================================

def mean_se(xs: Iterable[float]) -> tuple[float | None, float | None, int]:
    """Mean, standard error of the mean, and n. SE is None below two points.

    DEBUG LOG pattern 9, defence 12: every deterministic measure returns this. The SE is
    across PROMPTS, so it answers "would this number survive a different way of asking?" -
    which is the only variance a deterministic measure has. Four v1 measures ran at n=1 and
    could not tell a real effect from an artefact of one phrasing.

    Pure: no torch, no RUN. Offline-testable.
    """
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return None, None, 0
    m = sum(xs) / n
    if n < 2:
        return m, None, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n), n


# =====================================================================================
# Forward-pass cache
# =====================================================================================
# BUG 23. The key must identify the steering VECTOR, not just the grid cell. Keyed on
# (question, layer, alpha) alone, every entry still matched after switching concepts in a
# live kernel, so the previous concept's logits came back silently - a wrong number, not an
# error. The fingerprint below makes a cross-concept hit impossible. Defence 1.

_CACHE: dict = {}

# Each entry is one full-vocabulary float32 row: Gemma3's 262k vocab is ~1 MB. The cap keeps
# the cache under ~0.5 GB of host RAM across a ~100-cell scan; eviction is correctness-neutral
# because a miss just recomputes the pass.
_CACHE_MAX = 512


def vec_tag(vec: Any) -> str:
    """Content fingerprint of a steering vector. 'none' for an unsteered pass.

    Hashed fresh on each call rather than memoised on `id()`: a freed tensor can have its
    address reused, which would reintroduce exactly the aliasing this exists to prevent.
    Hashing ~20 KB costs microseconds against a forward pass.

    `vectors.vec_fingerprint` must be this function (import it) rather than a second
    implementation - two fingerprints that can disagree are bug 23 with extra steps.
    """
    if vec is None:
        return "none"
    return hashlib.sha1(
        vec.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()[:12]


# The lab's private name, kept so a ported cell reads unchanged.
_vec_tag = vec_tag


def logits_for(question: str, vec: Any, layer: int, alpha: float) -> Any:
    """Final-position logits for a chat-rendered question. Cached on CPU.

    The steering start position is derived from the question itself, so the framing stays
    unsteered exactly as in the detection test (bug 8).

    Cached on CPU deliberately: across a full scan this cache holds hundreds of
    full-vocabulary rows, and they have no business occupying VRAM the model needs.
    """
    ctx = _run()
    key = ("q", question, vec_tag(vec), int(layer), float(alpha))
    if key in _CACHE:
        return _CACHE[key]
    prompt = chat(question)
    with torch.no_grad():
        with injected(vec, layer, alpha, start_pos=start_pos_for(prompt, question)):
            out = ctx.hf(**encode(prompt)).logits[0, -1, :].float().cpu()
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))      # FIFO; dicts keep insertion order
    _CACHE[key] = out
    return out


def cache_clear() -> int:
    """Drop the forward-pass cache. Returns the number of entries dropped.

    Not needed for correctness since bug 23's fix - the fingerprint makes a cross-concept
    hit impossible - but `driver.set_concept` should still call it so a long batch does not
    carry the previous concept's rows in host RAM for no reason.
    """
    n = len(_CACHE)
    _CACHE.clear()
    return n


# =====================================================================================
# Residual-stream norms
# =====================================================================================

def _assert_chat_rendered(prompt_text: str) -> None:
    """Raise unless the text carries the template's <bos>.

    `encode` adds no special tokens (bug 9), so raw text passed here would be tokenised
    without <bos> and measured in a context the steering never actually runs in. ||h_L||
    sets alpha for every cell in the run (`alpha = r*||h_L||/||v_L||`), so a quietly wrong
    norm here mis-doses the whole pipeline. Loud is the only acceptable failure mode.
    """
    bos = getattr(_run().tok, "bos_token", None)
    if bos and bos not in prompt_text:
        raise ValueError(
            "residual_norms expects a chat-rendered prompt (call chat(question) first): "
            f"{bos!r} is absent, so encode() would tokenise this without <bos> and the "
            "norms would describe a context the steering never runs in.")


def residual_norms(prompt_text: str, layers: list[int]) -> dict[int, float]:
    """Median token-wise residual-stream norm at each requested layer, from ONE forward pass.

    ||h_L|| is the denominator of the effective dose `r = alpha*||v_L||/||h_L||` (spec
    section 3), which is the unit the whole pipeline searches in. It is a property of the
    model and the prompts, not of the concept - in the M1.5 data it was identical across all
    six concepts at every layer - so it is measured once and cached across concepts
    (spec 5.1).

    Why it matters: measured ||v_L|| ran 14 at L6 to 8896 at L46, so at a fixed alpha the L6
    perturbation was 0.3% of L37's. Every early-layer cell read flat - not because early
    injection does nothing, but because alpha*||v_L|| was negligible there. Without ||h_L||
    those two cannot be told apart.

    `prompt_text` must already be chat-rendered. Every layer in `layers` must be seen; a
    layer whose module never fired raises rather than going missing from the dict, because a
    missing norm downstream becomes an unreachable cell or a wrong alpha, not an error.
    """
    ctx = _run()
    _assert_chat_rendered(prompt_text)

    seen: dict[int, float] = {}
    handles = []

    def _make(idx: int):
        # The norm is reduced to a scalar INSIDE the hook. Keeping the hidden states of ~50
        # layers alive until the pass ends would hold ~10x the v1 lab's 6-layer footprint in
        # VRAM for no reason.
        def _fn(module, inp, out, _i=idx):
            h = out[0] if isinstance(out, tuple) else out
            seen[_i] = float(h.detach().float().norm(dim=-1).median())
        return _fn

    try:
        for idx in layers:
            handles.append(ctx.mw.get_layer_module(int(idx)).register_forward_hook(_make(int(idx))))
        with torch.no_grad():
            # use_cache=False mirrors the repo's own activation-extraction pass
            # (model_utils.py:427) - there is nothing to cache in a single forward.
            ctx.hf(**encode(prompt_text), use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = sorted(set(int(i) for i in layers) - set(seen))
    if missing:
        raise RuntimeError(
            f"residual_norms: layers {missing} never fired during the forward pass - "
            "their modules were not reached, so their norms are unknown. A missing norm "
            "must not become a default (DEBUG LOG pattern 4).")
    return {int(i): seen[int(i)] for i in layers}


# =====================================================================================
# R14 - hook liveness
# =====================================================================================
# Local to this module on purpose: prompts.py is later in the CONTRACT dependency order, and
# R14 has to be runnable the moment vectors exist. Any question would do - all this probe
# needs is a chat-rendered prompt whose needle is findable so start_pos is a real index.
_LIVENESS_QUESTION = "Tell me a fact related to water."

# Depth of the probe layer. 0.60 is Macar's reference depth (L37 on Gemma3-27B) and the depth
# at which the v1 lab's liveness check was validated.
_LIVENESS_DEPTH = 0.60

# Dose for the probe when RUN.norms is available: the top of SCAN_DOSES, i.e. the strongest
# intervention Phase 1 will actually apply. Attesting the hook at a dose the sweep never uses
# would attest the wrong thing.
_LIVENESS_R = 0.30

# The v1 anchor, used only when RUN.norms is not built yet. alpha=4 at the reference layer is
# the value the lab's S14 ran at and is known to move the logits on this model.
_LIVENESS_ALPHA_FALLBACK = 4.0


def hook_liveness(layer: int | None = None, alpha: float | None = None,
                  question: str | None = None, threshold: float = 1e-3) -> dict:
    """R14 (v1's S14): prove that injection actually changes the model. Raises if it does not.

    RUN THIS BEFORE ANY SWEEP. The 2026-08-04 run produced identically zero readings at every
    one of 30 cells because the repo's SteeringHook declined to steer whenever start_pos was
    set (bug 26), and nothing caught it for an hour of measurement. Two forward passes here
    would have killed it at setup.

    Checks BOTH injection paths, because they fail independently and bug 26 killed exactly
    one of them:

        start_pos path   - used wherever the framing must stay unsteered (D3, E6, E5, D2)
        all-positions    - used wherever there is no template to protect (S3's raw MMLU items)

    Deliberately does NOT go through `logits_for`: a liveness check that can be served from a
    cache attests the cache, not the hook.

    Returns a dict describing exactly what was probed (including where the probe alpha came
    from, so the number is never anonymous). Raises RuntimeError if either path is dead.
    """
    ctx = _run()
    if not ctx.vecs:
        raise RuntimeError(
            "R14 needs RUN.vecs - extract the concept vectors before checking hook liveness.")

    if layer is None:
        # The extracted layer closest to Macar's reference depth.
        target = _LIVENESS_DEPTH * ctx.n_layers
        layer = min(ctx.vecs, key=lambda L: abs(L - target))
    layer = int(layer)
    vec = ctx.vecs[layer]          # hard index: a probe on a layer we have no vector for is a bug

    alpha_source = "caller"
    if alpha is None:
        try:
            # One dose map for the whole pipeline - `config.alpha_for` is it. Duplicating the
            # formula here is how two dose maps end up disagreeing.
            alpha = float(config.alpha_for(layer, _LIVENESS_R))
            alpha_source = f"alpha_for(L{layer}, r={_LIVENESS_R})"
        except Exception as exc:      # noqa: BLE001 - reported, never swallowed
            # Norms not built yet, or this cell is unreachable at that dose. Neither is a
            # reason for a liveness probe to fail; fall back to the v1 anchor and SAY SO in
            # the returned dict, so no reader has to guess which dose was probed.
            alpha = _LIVENESS_ALPHA_FALLBACK
            alpha_source = f"fallback alpha={_LIVENESS_ALPHA_FALLBACK} ({type(exc).__name__})"
    alpha = float(alpha)

    q = _LIVENESS_QUESTION if question is None else question
    prompt = chat(q)
    sp = start_pos_for(prompt, q)
    if sp is None:
        raise RuntimeError(
            f"R14: the probe question is not findable in its own rendered prompt, so there "
            f"is no start_pos to test. Question: {q!r}")

    enc = encode(prompt)
    with torch.no_grad():
        base = ctx.hf(**enc).logits[0, -1, :].float()
        with injected(vec, layer, alpha, start_pos=sp):
            lg_sp = ctx.hf(**enc).logits[0, -1, :].float()
        with injected(vec, layer, alpha, start_pos=None):
            lg_all = ctx.hf(**enc).logits[0, -1, :].float()

    d_start_pos = float((lg_sp - base).abs().max())
    d_all_pos = float((lg_all - base).abs().max())
    identical = bool(torch.equal(lg_sp, lg_all))

    out = dict(
        check="R14",
        layer=layer,
        alpha=alpha,
        alpha_source=alpha_source,
        start_pos=sp,
        vec_fingerprint=vec_tag(vec),
        d_start_pos=d_start_pos,
        d_all_pos=d_all_pos,
        threshold=threshold,
        paths_distinct=(not identical),
        passed=(d_start_pos > threshold and d_all_pos > threshold),
    )

    if not out["passed"]:
        raise RuntimeError(
            f"R14 FAIL - steering changed nothing at L{layer} alpha={alpha:.3f} "
            f"(start_pos path {d_start_pos:.2e}, all-positions path {d_all_pos:.2e}, "
            f"threshold {threshold:.0e}). Every forward-pass measure would read exactly "
            f"zero. This is bug 26. Do not run the sweep.")

    # If start_pos > 0 the two paths steer different prefixes, so their final-position logits
    # cannot be bit-identical - attention over the earlier positions differs. Identical output
    # means start_pos was ignored and the framing is being steered too, which is bug 8's shape
    # (detection and effectiveness measured under different interventions).
    if sp > 0 and identical:
        raise RuntimeError(
            f"R14 FAIL - the start_pos and all-positions paths returned bit-identical logits "
            f"at start_pos={sp}. start_pos is being ignored, so the chat template is steered "
            f"as well and this is not the intervention the detection test applies (bug 8).")

    return out
