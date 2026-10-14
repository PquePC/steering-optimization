"""m5.backend.hf - the real subject: Hugging Face transformers, steered through forward hooks.

`HFBackend` implements every method of `m5.backend.base.Backend` for Gemma-3 (the 4B/12B/27B "-it"
checkpoints are the multimodal `Gemma3ForConditionalGeneration`; text-only checkpoints are
`Gemma3ForCausalLM`) and Qwen3 (`Qwen3ForCausalLM`). It is the only module in M5 that imports torch
at module scope.

## Design: the backend owns the plan, the hooks only carry it out

Every public method validates its whole request before any forward pass, then, for each batch it
runs, builds a `_Pass`: per steered layer the rows it steers, each row's vector, operation and
dose, and a boolean mask of the positions to edit in the *padded* batch; per read layer the
callbacks that copy what a method needs. One forward hook per decoder layer that some row of the
call steers or reads is registered for the duration of the call and removed in `finally`. The hook
holds no state: it looks up the current pass, checks the output has exactly the planned shape,
applies the edit, runs the readers, and counts that it fired. After every forward pass each hook
must have fired exactly once.

What each piece guards against (`docs/DEBUG-LOG.md`; the build notes on the old hooks):

* **Bare tensors and tuples.** In transformers 5.19 both `Gemma3DecoderLayer` and
  `Qwen3DecoderLayer` return a bare tensor; older versions returned a tuple. Upstream's
  `SteeringHook` returned the output unmodified for a tensor whenever a start position was set,
  and 30 of 30 cells read exactly 0.000 for an hour (bug 26). The hook here edits both shapes,
  returns the same structure, and raises on anything else.
* **A hook that never fires** steers nothing and reports nothing (bug 26's other path): checked
  after every pass.
* **Left padding.** Positions are unpadded token indices of `tokenize(text)`; the backend shifts
  each row by its own pad count. Upstream's batch path applied one start to every row, steering
  shorter prompts one token early (bug 25b). Position ids are passed explicitly
  (`cumsum(mask) - 1`), so a padded row sees exactly the positions it sees alone.
* **Per-row interventions** (contract 2): each row has its own vector, operation, layers, doses
  and positions; one batch can mix cells from different layers (plan §8.1).
* **Exact tokens** (contract 1): the forward pass is fed `tokenize(text)` and nothing else, and a
  prompt must begin with exactly one BOS when the template writes one (bug 9: a doubled BOS shifted
  every position by one; a missing one measures a context steering never runs in).
* **No leaks** (contract 3): a fresh KV cache per generation batch, no cached logits anywhere, and
  no hook survives a call (bug 23: a cache keyed without the vector returned the previous concept's
  numbers).
* **Per-row samplers** (contract 4): each row draws from its own `torch.Generator` seeded from
  `GenRequest.seed`, so its sample does not depend on what else is in the batch. HF `generate`
  draws every row from one stream, and the three Qwen null arms came out byte-identical.
  Temperature, top-k and top-p are applied per request and never read from `generation_config`.
* **fp32 arithmetic** (contract 5): every edit is computed in fp32 on an fp32 copy of the residual
  and cast back once; log-probabilities, entropies and KL in fp32; residual statistics in fp64.
* **One device.** Trap 13 of the old pipelines: `device_map="auto"` shards across every visible
  card and the per-row plan assumes one device. `load` places the model on one device and the
  constructor refuses a model that spans several.

## Conventions every method shares

* **Layer L** is the output of decoder block L (`m5.intervention`). Reads at L see the output after
  any steering at L. This is read through the hook, never through `output_hidden_states`: HF
  installs its capture hooks once and permanently, so a hook registered later edits the output
  *after* HF recorded it, and in transformers 5.x `hidden_states[-1]` is the post-final-norm state,
  not block n-1's output.
* **Generated tokens.** In `generate` they are the sampled tokens; in teacher forcing (`score`,
  `candidate_logprobs`, `forced_logprobs`) the continuation's tokens stand in for them. Either way
  they are steered exactly when the intervention's policy steers generated tokens (`from`,
  `generated`, `all`), so teacher forcing scores what generation would have computed for those
  tokens.
* **A pass that steers nothing raises.** An intervention whose positions select no token of a
  forward pass (`generated` in a prompt-only pass, a `from` start beyond the prompt) is a
  construction error, never an unsteered measurement under a steered label.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from m5.backend.base import (
    GenRequest,
    GenResult,
    KLResult,
    ModelInfo,
    ResidualStats,
    ScoreRequest,
    ScoreResult,
)
from m5.intervention import Intervention

__all__ = ["HFBackend", "Forced"]

# Where the decoder stack and its final norm live, tried in order. The multimodal Gemma-3 wrapper
# comes first: upstream warns that `model.model.layers` "may exist but be wrong" there (bug 4).
_STACK_PATHS: tuple[tuple[str, str], ...] = (
    # Gemma3ForConditionalGeneration, transformers >= 4.52
    ("model.language_model.layers", "model.language_model.norm"),
    # Gemma3ForConditionalGeneration, older releases
    ("language_model.model.layers", "language_model.model.norm"),
    # Gemma3ForCausalLM, Qwen3ForCausalLM
    ("model.layers", "model.norm"),
)

# The token that closes a turn, by family: generation must stop there even when a checkpoint's
# generation_config forgets it.
_END_OF_TURN = {"gemma": "<end_of_turn>", "qwen": "<|im_end|>"}

# Hugging Face model types `load` knows how to steer, and the class each loads as.
_MODEL_CLASSES = {
    "gemma3": "Gemma3ForConditionalGeneration",
    "gemma3_text": "Gemma3ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
}
_FAMILY_OF_TYPE = {"gemma3": "gemma", "gemma3_text": "gemma", "qwen3": "qwen"}

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Memory bounds for transient fp32 work. Not tunables: they change how a computation is split, and
# its result by floating-point noise at most (rows of the edit are independent; a matmul over fewer
# rows may block its kernel differently).
_EDIT_CHUNK_ELEMENTS = 1 << 26      # fp32 elements per temporary in the edit (256 MB)
_LOGIT_CHUNK_ELEMENTS = 1 << 26     # fp32 logits per chunk when scoring many positions

# Contexts that word_variants tokenises a variant after, in order of preference. A newline is where
# a reply line starts; the others are running text for tokenizers that merge across a newline.
_VARIANT_CONTEXTS = ("\n", "The word is", "Answer:")


# =====================================================================================
# Plan objects
# =====================================================================================

@dataclass(frozen=True)
class _Vector:
    vec: torch.Tensor    # [d] fp32 on the model's device
    unit: torch.Tensor   # [d] fp32, vec / ||vec||
    norm: float          # ||vec||, computed in fp64


@dataclass(frozen=True)
class _LayerEdit:
    """One layer's edit for one forward pass: h <- h + a*v + (c + b*(h . u)) u at masked positions.

    add: a = alpha; sub: a = -alpha; project: b = -alpha; clamp: b = -1, c = clamp_value. One
    formula for the four operations of `m5.intervention`, so rows with different operations share
    one batched computation.
    """

    rows: torch.Tensor   # [R] long, batch rows edited at this layer
    vec: torch.Tensor    # [R, d] fp32
    unit: torch.Tensor   # [R, d] fp32
    a: torch.Tensor      # [R] fp32
    b: torch.Tensor      # [R] fp32
    c: torch.Tensor      # [R] fp32
    mask: torch.Tensor   # [R, width] bool, padded positions edited


@dataclass
class _Pass:
    """Everything the hooks need for one forward pass. Built by the backend before the pass."""

    batch: int
    width: int
    edits: dict[int, _LayerEdit] = field(default_factory=dict)
    readers: dict[int, list[Callable[[torch.Tensor], None]]] = field(default_factory=dict)
    final_readers: list[Callable[[torch.Tensor], None]] = field(default_factory=list)
    fired: dict[Any, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _Row:
    """One row of a forward pass: its input ids, how many are prompt, its steering."""

    ids: list[int]
    n_prompt: int
    iv: Intervention | None
    mask: list[bool] | None   # unpadded, len(ids): positions steered in a pass over `ids`


@dataclass(frozen=True)
class _GenRow:
    ids: list[int]
    iv: Intervention | None
    mask: list[bool] | None   # prompt positions steered in the prefill
    max_new: int
    temperature: float
    top_k: int
    top_p: float
    seed: int


@dataclass(frozen=True)
class Forced:
    """Teacher-forced reading of given continuation ids: log p of each forced token (nats) and the
    most probable token of the distribution it was read from (the argmax a greedy sampler takes)."""

    logprobs: np.ndarray   # [m] float64
    argmax: np.ndarray     # [m] int64


# =====================================================================================
# Small helpers
# =====================================================================================

def _attr_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _find_stack(model: Any) -> tuple[str, torch.nn.ModuleList, str, torch.nn.Module]:
    """The decoder layers and final norm, found by trying each known path; raises listing them."""
    tried = []
    for layers_path, norm_path in _STACK_PATHS:
        layers = _attr_path(model, layers_path)
        if not isinstance(layers, torch.nn.ModuleList) or len(layers) == 0:
            tried.append(layers_path)
            continue
        norm = _attr_path(model, norm_path)
        if not isinstance(norm, torch.nn.Module):
            raise RuntimeError(f"{type(model).__name__}: decoder layers at {layers_path} but no "
                               f"final norm at {norm_path}")
        return layers_path, layers, norm_path, norm
    raise RuntimeError(f"cannot find the decoder layers of {type(model).__name__}; tried "
                       f"{', '.join(tried)}. Add its path to _STACK_PATHS rather than guessing: a "
                       "hook on the wrong stack steers nothing (bug 4, bug 26).")


def _find_lm_head(model: Any) -> torch.nn.Module:
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is None:
        head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if head is None or not torch.is_tensor(weight) or weight.dim() != 2:
        raise RuntimeError(f"{type(model).__name__}: no output embedding (lm_head) to unembed with")
    return head


def _one_device(model: Any) -> torch.device:
    devices = {p.device for p in model.parameters()} | {b.device for b in model.buffers()}
    if len(devices) != 1:
        raise RuntimeError(
            f"the model spans {sorted(str(d) for d in devices)}. The per-row steering plan and the "
            "registered vectors live on one device; load the model on one device "
            "(device_map={'': device}, never 'auto', which shards across every visible card).")
    return devices.pop()


def _add_ids(out: set[int], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for v in value:
            _add_ids(out, v)
        return
    out.add(int(value))


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


# =====================================================================================
# The backend
# =====================================================================================

class HFBackend:
    """Hugging Face backend with per-row forward hooks. See the module docstring.

    Build it with `HFBackend.load(cfg)` for a real checkpoint, or directly from a model and a fast
    tokenizer (the tests build tiny random models in-process). `batch_size` is the most rows any
    forward pass or generation batch holds.
    """

    def __init__(self, model: Any, tokenizer: Any, *, name: str, hf_id: str, family: str,
                 revision: str | None = None, batch_size: int = 16) -> None:
        if not _is_int(batch_size) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if not getattr(tokenizer, "is_fast", False):
            raise TypeError("HFBackend needs a fast tokenizer: offsets() reads character spans from "
                            "the Rust tokenizer, and every steering position is located through them")
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            raise ValueError("the tokenizer has no chat template (a single string); every prompt is "
                             "rendered through it")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.family = str(family)
        self.device = _one_device(model)
        self._layers_path, self._layers, self._norm_path, self._final_norm = _find_stack(model)
        self._lm_head = _find_lm_head(model)

        config = model.config
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        n_layers = len(self._layers)
        declared = getattr(text_config, "num_hidden_layers", None)
        if declared is not None and int(declared) != n_layers:
            raise RuntimeError(f"{self._layers_path} holds {n_layers} layers but the config declares "
                               f"{declared}: the hooks would index a different stack than the config "
                               "describes (bug 4)")
        d_model = int(text_config.hidden_size)
        head_w = self._lm_head.weight
        if int(head_w.shape[1]) != d_model:
            raise RuntimeError(f"lm_head reads {head_w.shape[1]} dimensions, the residual has {d_model}")
        vocab = int(head_w.shape[0])
        # Softcapping is applied exactly when the loaded class's own forward applies it: in
        # transformers 5.19 Gemma3ForCausalLM reads `config.final_logit_softcapping`, while
        # Gemma3ForConditionalGeneration (whose composite config has no such attribute) applies
        # none even if its text_config names one. The released Gemma-3 checkpoints set it to null.
        # `liveness` checks the lens against the model's own logits on every run.
        cap = getattr(config, "final_logit_softcapping", None)
        self._softcap = None if cap is None else float(cap)

        self._thinking_switch = "enable_thinking" in template
        self._stop_ids = self._find_stop_ids(model, tokenizer, vocab)
        self._stop_set = frozenset(self._stop_ids)
        self._pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None \
            else min(self._stop_ids)
        self._bos: int | None = None
        self._lock = threading.RLock()
        self._vectors: dict[str, dict[int, _Vector]] = {}
        self._pass: _Pass | None = None
        self._handles: list[Any] = []
        self._hook_keys: tuple[Any, ...] = ()
        self._n_layers = n_layers
        self._d_model = d_model
        self._vocab = vocab
        self._bos = self._template_bos(tokenizer)

        self.info = ModelInfo(
            name=str(name), hf_id=str(hf_id), revision=revision, family=self.family,
            n_layers=n_layers, d_model=d_model, vocab_size=vocab,
            dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
            bos_token_id=self._bos, eos_token_ids=tuple(self._stop_ids),
            chat_template_hash=hashlib.sha256(template.encode("utf-8")).hexdigest()[:16],
        )

    # ---------------------------------------------------------------- construction checks

    def _find_stop_ids(self, model: Any, tokenizer: Any, vocab: int) -> list[int]:
        """generation_config's eos ids, the tokenizer's eos, and the family's end-of-turn token.

        The end of turn is required by name for Gemma and Qwen: a checkpoint whose
        generation_config omits it would otherwise generate past the reply into a new turn.
        """
        ids: set[int] = set()
        _add_ids(ids, getattr(getattr(model, "generation_config", None), "eos_token_id", None))
        _add_ids(ids, tokenizer.eos_token_id)
        eot = _END_OF_TURN.get(self.family)
        if eot is not None:
            if eot not in tokenizer.get_vocab():
                raise ValueError(f"a {self.family} model must stop at {eot!r}, which this tokenizer "
                                 "does not have")
            ids.add(int(tokenizer.convert_tokens_to_ids(eot)))
        if not ids:
            raise ValueError("no stop token: generation_config, the tokenizer and the family name none")
        bad = [i for i in ids if not 0 <= i < vocab]
        if bad:
            raise ValueError(f"stop ids {bad} are outside the model's vocabulary ({vocab})")
        return sorted(ids)

    def _template_bos(self, tokenizer: Any) -> int | None:
        """The BOS every rendered prompt begins with, or None when the template writes none.

        Raises when the template writes two (bug 9 in the template itself), when it writes none
        although the tokenizer would add one to raw text (the model expects a BOS that no rendered
        prompt would carry), and when the thinking switch fails to close the reasoning block.
        """
        probe = self.render([{"role": "user", "content": "Hello."}])
        ids = self.tokenize(probe)
        bos = tokenizer.bos_token_id
        if self._thinking_switch and probe.rfind("</think>") < probe.rfind("<think>"):
            raise RuntimeError("enable_thinking=False left an open <think> block in the generation "
                               "prompt; the model would reason before answering, outside the window "
                               "every channel reads (the old Qwen arm)")
        if bos is not None and ids and ids[0] == bos:
            if len(ids) > 1 and ids[1] == bos:
                raise RuntimeError("the chat template writes two BOS tokens (bug 9)")
            return int(bos)
        if bos is not None:
            with_specials = tokenizer("Hello.", add_special_tokens=True)["input_ids"]
            if with_specials and with_specials[0] == bos:
                raise RuntimeError("the tokenizer adds a BOS to raw text but the chat template writes "
                                   "none, so rendered prompts would run without the BOS the model "
                                   "expects")
        return None

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, cfg: Any) -> "HFBackend":
        """Load the configured subject on one device.

        Reads MODEL / MODEL_ID, MODEL_REVISION, DTYPE, DEVICE, ATTN_IMPLEMENTATION and
        GEN_BATCH_MAX. Gemma-3 "-it" checkpoints (model type `gemma3`) load as the multimodal
        Gemma3ForConditionalGeneration and are used text-only; `gemma3_text` as Gemma3ForCausalLM;
        Qwen3 as Qwen3ForCausalLM. Any other model type is refused: the hooks, the norm gains and
        the stop tokens are only known for these. DEVICE 'auto' means one CUDA device if there is
        one, else the CPU; it never shards. The resolved commit is recorded in `info.revision`.
        """
        import transformers
        from transformers import AutoConfig, AutoTokenizer

        hf_id = cfg.model_id
        family = cfg.model_family
        revision = cfg["MODEL_REVISION"] or None
        dtype = _DTYPES[cfg["DTYPE"]]
        device = cls._pick_device(cfg["DEVICE"])
        hf_config = AutoConfig.from_pretrained(hf_id, revision=revision)
        model_type = getattr(hf_config, "model_type", None)
        if model_type not in _MODEL_CLASSES:
            raise ValueError(f"{hf_id}: model type {model_type!r} is not one this backend steers "
                             f"({', '.join(sorted(_MODEL_CLASSES))})")
        if _FAMILY_OF_TYPE[model_type] != family:
            raise ValueError(f"{hf_id} is a {_FAMILY_OF_TYPE[model_type]} model but the config "
                             f"names family {family!r}; the judge-family rule depends on it")
        model_cls = getattr(transformers, _MODEL_CLASSES[model_type])
        tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        kwargs: dict[str, Any] = dict(revision=revision, dtype=dtype,
                                      attn_implementation=cfg["ATTN_IMPLEMENTATION"])
        if importlib.util.find_spec("accelerate") is not None:
            kwargs["device_map"] = {"": device}
        elif device != "cpu":
            raise RuntimeError("placing the model on a GPU needs `accelerate` (pip install accelerate); "
                               "without it transformers cannot take a device_map")
        # Without accelerate the weights load on the CPU, which is then the one device asked for.
        model = model_cls.from_pretrained(hf_id, **kwargs)
        resolved = getattr(model.config, "_commit_hash", None) or revision
        name = cfg["MODEL_ID"] or cfg["MODEL"]
        backend = cls(model, tokenizer, name=name, hf_id=hf_id, family=family, revision=resolved,
                      batch_size=cfg["GEN_BATCH_MAX"])
        if not cfg["MODEL_ID"]:
            # A registry model: its published depth must be what loaded, or every layer index in
            # the profile, the screen and the report would name a different block.
            from m5.config import MODELS
            expected = MODELS[cfg["MODEL"]]["n_layers"]
            if backend.info.n_layers != expected:
                raise ValueError(f"{hf_id} loaded with {backend.info.n_layers} decoder layers; the "
                                 f"registry (m5.config.MODELS) says {expected}")
        return backend

    @staticmethod
    def _pick_device(device: str) -> str:
        if device == "cpu":
            return "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda but no CUDA device is visible")
        if device in ("cuda", "auto"):
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        raise ValueError(f"unknown DEVICE {device!r}")

    def provenance(self) -> dict[str, Any]:
        """What produced this backend's numbers: classes, paths, versions, device, sampling defaults
        the checkpoint ships (recorded, never used), and the visible GPU count (trap 13)."""
        import transformers
        try:
            import tokenizers
            tok_version = tokenizers.__version__
        except Exception:  # noqa: BLE001 - a version string must never cost a run
            tok_version = None
        text_config = self.model.config.get_text_config() \
            if hasattr(self.model.config, "get_text_config") else self.model.config
        gc = getattr(self.model, "generation_config", None)
        out: dict[str, Any] = dict(
            model_class=type(self.model).__name__,
            layers_path=self._layers_path,
            final_norm_path=self._norm_path,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            final_logit_softcapping=self._softcap,
            sliding_window=getattr(text_config, "sliding_window", None),
            layer_types=list(getattr(text_config, "layer_types", None) or []),
            tokenizer_size=len(self.tokenizer),
            pad_token_id=self._pad_id,
            thinking_switch=self._thinking_switch,
            checkpoint_generation_config={
                k: getattr(gc, k, None) for k in ("eos_token_id", "temperature", "top_k", "top_p",
                                                   "do_sample")},
            device=str(self.device),
            visible_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            torch=torch.__version__, transformers=transformers.__version__, tokenizers=tok_version,
            info=self.info.to_json(),
        )
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            out["gpu_name"] = props.name
            out["gpu_capability"] = f"{props.major}.{props.minor}"
        return out

    # ---------------------------------------------------------------- text

    def render(self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = True) -> str:
        """Apply the chat template; every prompt in M5 comes through here, extraction included.

        Two rules, both from the old pipelines:

        * Gemma: an *empty* system message is dropped. The template folds a system message into the
          first user turn, and an empty one adds blank lines there, which raised false-positive
          rates (upstream `_filter_messages_for_model`). A non-empty one is kept and folded as the
          template does (upstream dropped those as well, silently).
        * When the template has a thinking switch (Qwen3), it is rendered with
          `enable_thinking=False`, always: the old Qwen arm extracted vectors from one rendering
          and injected into another, four tokens apart, and was withdrawn.
        """
        msgs = []
        for i, m in enumerate(messages):
            try:
                role, content = m["role"], m["content"]
            except KeyError as exc:
                raise ValueError(f"message {i} lacks {exc.args[0]!r}") from None
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"message {i}: unknown role {role!r}")
            if not isinstance(content, str):
                raise TypeError(f"message {i}: content must be text")
            if self.family == "gemma" and role == "system" and content == "":
                continue
            msgs.append({"role": role, "content": content})
        if not msgs:
            raise ValueError("nothing to render")
        kwargs = {"enable_thinking": False} if self._thinking_switch else {}
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=add_generation_prompt, **kwargs)
        if not isinstance(text, str):
            raise TypeError(f"apply_chat_template returned {type(text).__name__}, not text")
        return text

    def tokenize(self, text: str) -> list[int]:
        """Exactly the ids the model is fed for `text`: no special tokens are added (contract 1)."""
        return [int(i) for i in self.tokenizer(text, add_special_tokens=False)["input_ids"]]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(s), int(e)) for s, e in enc["offset_mapping"]]

    def decode(self, ids: Sequence[int]) -> str:
        """The text of `ids`, special tokens kept and no cleanup, so decode(tokenize(t)) == t.
        `GenResult.text` is the variant with special tokens skipped."""
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    def word_variants(self, word: str) -> list[tuple[int, ...]]:
        """Token sequences of " word", "word", " Word", "Word" as the next words of a reply.

        Each variant is tokenised after a neutral context and the context's tokens are cut off, so
        a SentencePiece tokenizer cannot mark the variant as text-initial; a context whose tokens
        are not a prefix of the joint tokenisation (a merge across the cut) is skipped, and the
        suffix must decode back to the variant. "Word" upper-cases the first letter only ("DNA"
        stays "DNA"). Duplicates are dropped, first occurrence kept.
        """
        if not isinstance(word, str) or not word or word != word.strip():
            raise ValueError(f"word_variants needs a word without surrounding space, got {word!r}")
        cap = word[:1].upper() + word[1:]
        out: list[tuple[int, ...]] = []
        for variant in (" " + word, word, " " + cap, cap):
            ids = self._isolate(variant)
            if ids not in out:
                out.append(ids)
        return out

    def _isolate(self, variant: str) -> tuple[int, ...]:
        for ctx in _VARIANT_CONTEXTS:
            ctx_ids = self.tokenize(ctx)
            joint = self.tokenize(ctx + variant)
            if joint[:len(ctx_ids)] != ctx_ids or len(joint) == len(ctx_ids):
                continue
            suffix = tuple(joint[len(ctx_ids):])
            if self.decode(suffix) == variant:
                return suffix
        raise ValueError(f"cannot isolate the tokens of {variant!r} after any neutral context "
                         f"{_VARIANT_CONTEXTS}")

    # ---------------------------------------------------------------- vectors

    def register_vector(self, vector_id: str, per_layer: Mapping[int, np.ndarray]) -> None:
        """Copy a vector's per-layer values to the model's device, in fp32.

        Kept in fp32 only: every operation is computed in fp32 on an fp32 copy of the residual and
        cast back once, so the vector is never rounded to the model dtype before it is scaled
        (legacy vectors were bf16 end to end). Re-registering an id with different values raises:
        one name with two contents is bug 23's shape; `unregister_vector` first.
        """
        if not isinstance(vector_id, str) or not vector_id:
            raise ValueError("vector_id must be a non-empty string")
        if not per_layer:
            raise ValueError(f"vector {vector_id!r}: no layers given")
        stored: dict[int, _Vector] = {}
        for layer, values in per_layer.items():
            if not _is_int(layer) or not 0 <= int(layer) < self._n_layers:
                raise ValueError(f"vector {vector_id!r}: layer {layer!r} is not in [0, {self._n_layers})")
            arr = np.asarray(values)
            if arr.ndim != 1 or arr.shape[0] != self._d_model:
                raise ValueError(f"vector {vector_id!r} layer {layer}: shape {arr.shape}, the residual "
                                 f"is ({self._d_model},)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"vector {vector_id!r} layer {layer}: non-finite values")
            vec = torch.tensor(np.array(arr, dtype=np.float32), device=self.device)
            norm = float(torch.linalg.vector_norm(vec.double()))
            if norm == 0.0:
                raise ValueError(f"vector {vector_id!r} layer {layer}: zero vector; it steers nothing "
                                 "and has no direction to project on")
            stored[int(layer)] = _Vector(vec=vec, unit=(vec.double() / norm).float(), norm=norm)
        with self._lock:
            old = self._vectors.get(vector_id)
            if old is not None:
                same = set(old) == set(stored) and all(torch.equal(old[k].vec, stored[k].vec)
                                                       for k in stored)
                if same:
                    return
                raise ValueError(f"vector {vector_id!r} is already registered with different values; "
                                 "unregister it first (one name, two contents is bug 23)")
            self._vectors[vector_id] = stored

    def has_vector(self, vector_id: str) -> bool:
        return vector_id in self._vectors

    def unregister_vector(self, vector_id: str) -> None:
        with self._lock:
            if vector_id not in self._vectors:
                raise KeyError(f"vector {vector_id!r} is not registered")
            del self._vectors[vector_id]

    def vector_values(self, vector_id: str, layer: int) -> np.ndarray:
        """The registered fp32 values of one layer (liveness checks the edit against them)."""
        try:
            return self._vectors[vector_id][int(layer)].vec.detach().cpu().numpy().copy()
        except KeyError:
            raise KeyError(f"vector {vector_id!r} has no registered layer {layer}") from None

    # ---------------------------------------------------------------- validation

    def _check_ids(self, ids: Sequence[int], what: str) -> None:
        """A forward-pass input begins with exactly one BOS when the template writes one (bug 9)."""
        if not ids:
            raise ValueError(f"{what}: no tokens")
        if self._bos is not None:
            if ids[0] != self._bos:
                raise ValueError(f"{what}: does not begin with the BOS the chat template writes. Render "
                                 "it with render(), or prefix raw text with decode([info.bos_token_id]); "
                                 "without it the model runs in a context nothing else measures")
            if len(ids) > 1 and ids[1] == self._bos:
                raise ValueError(f"{what}: begins with two BOS tokens; every position would be off by "
                                 "one (bug 9)")

    def _check_intervention(self, iv: Any, what: str) -> Intervention | None:
        if iv is None:
            return None
        if not isinstance(iv, Intervention):
            raise TypeError(f"{what}: expected an Intervention or None, got {type(iv).__name__}")
        vecs = self._vectors.get(iv.vector_id)
        if vecs is None:
            raise KeyError(f"{what}: vector {iv.vector_id!r} is not registered")
        for term in iv.terms:
            if term.layer >= self._n_layers:
                raise ValueError(f"{what}: layer {term.layer} is beyond the model "
                                 f"({self._n_layers} layers)")
            if term.layer not in vecs:
                raise ValueError(f"{what}: vector {iv.vector_id!r} has no value at layer {term.layer}")
        return iv

    @staticmethod
    def _steer_mask(iv: Intervention | None, n_prompt: int, n_total: int, what: str) -> list[bool] | None:
        """Which unpadded positions of a pass over n_total tokens (the first n_prompt the prompt,
        the rest generated or teacher-forced) the intervention steers."""
        if iv is None:
            return None
        pos = iv.positions
        if pos.policy == "from" and pos.start >= n_prompt:
            raise ValueError(f"{what}: steering starts at token {pos.start} but the prompt has "
                             f"{n_prompt} tokens. An out-of-range start leaves the prompt unsteered "
                             "while generated tokens are steered (an old silent defect)")
        try:
            mask = pos.prompt_mask(n_prompt)
        except ValueError as exc:
            raise ValueError(f"{what}: {exc}") from None
        return mask + [pos.steers_generated] * (n_total - n_prompt)

    def _rows(self, prompts: Sequence[str], interventions: Sequence[Any], what: str) -> list[_Row]:
        """Prompt-only rows for a forward pass, validated; an intervention must steer something."""
        if len(prompts) != len(interventions):
            raise ValueError(f"{what}: {len(prompts)} prompts but {len(interventions)} interventions")
        rows = []
        for i, (text, iv) in enumerate(zip(prompts, interventions)):
            label = f"{what} row {i}"
            ids = self.tokenize(text)
            self._check_ids(ids, label)
            iv = self._check_intervention(iv, label)
            mask = self._steer_mask(iv, len(ids), len(ids), label)
            if mask is not None and not any(mask):
                raise ValueError(f"{label}: positions {iv.positions.to_json()} steer no token of a "
                                 "forward pass over the prompt; a pass that cannot steer must not "
                                 "read as a steered measurement (bug 26)")
            rows.append(_Row(ids=ids, n_prompt=len(ids), iv=iv, mask=mask))
        return rows

    def _token_ids(self, token_ids: Sequence[int], what: str) -> list[int]:
        out = []
        for t in token_ids:
            if not _is_int(t) or not 0 <= int(t) < self._vocab:
                raise ValueError(f"{what}: token id {t!r} is outside the vocabulary ({self._vocab})")
            out.append(int(t))
        return out

    def _layer_list(self, layers: Sequence[int], what: str) -> list[int]:
        out = []
        for layer in layers:
            if not _is_int(layer) or not 0 <= int(layer) < self._n_layers:
                raise ValueError(f"{what}: layer {layer!r} is not in [0, {self._n_layers})")
            out.append(int(layer))
        if not out:
            raise ValueError(f"{what}: no layers")
        if len(set(out)) != len(out):
            raise ValueError(f"{what}: duplicate layers {out}")
        return out

    # ---------------------------------------------------------------- hooks and passes

    def _layer_module(self, layer: int) -> torch.nn.Module:
        return self._layers[layer]

    @contextmanager
    def _hooked(self, layers: set[int] | Sequence[int], *, final: bool = False) -> Iterator[None]:
        """Register one hook per listed layer (and on the final norm) for the duration of a call."""
        if self._handles:
            raise RuntimeError("steering hooks from an earlier call are still attached")
        keys: list[Any] = sorted(set(int(x) for x in layers))
        try:
            for layer in keys:
                self._handles.append(self._layer_module(layer).register_forward_hook(self._hook(layer)))
            if final:
                self._handles.append(self._final_norm.register_forward_hook(self._hook("final")))
                keys.append("final")
            self._hook_keys = tuple(keys)
            yield
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            self._hook_keys = ()
            self._pass = None

    def _hook(self, key: Any) -> Callable[..., Any]:
        def hook(module: Any, args: Any, output: Any) -> Any:
            return self._on_output(key, output)
        return hook

    def _on_output(self, key: Any, output: Any) -> Any:
        p = self._pass
        if p is None:
            raise RuntimeError(f"steering hook {key!r} fired outside a planned forward pass")
        if isinstance(output, tuple):
            if not output or not torch.is_tensor(output[0]):
                raise TypeError(f"hook {key!r}: a tuple output must lead with the residual tensor")
            h, rest = output[0], output[1:]
        elif torch.is_tensor(output):
            h, rest = output, None
        else:
            raise TypeError(f"hook {key!r}: decoder output of type {type(output).__name__} cannot be "
                            "steered; refusing rather than passing it through unsteered (bug 26)")
        if h.dim() != 3 or h.shape[0] != p.batch or h.shape[1] != p.width or h.shape[2] != self._d_model:
            raise RuntimeError(f"hook {key!r}: output {tuple(h.shape)} does not match the planned "
                               f"({p.batch}, {p.width}, {self._d_model}); a plan that does not match "
                               "its batch steers the wrong positions silently (bug 25b)")
        p.fired[key] = p.fired.get(key, 0) + 1
        if key == "final":
            for read in p.final_readers:
                read(h)
            return None
        edit = p.edits.get(key)
        if edit is not None:
            h = self._edit(h, edit)
        for read in p.readers.get(key, ()):
            read(h)
        if edit is None:
            return None
        return (h,) + tuple(rest) if rest is not None else h

    @staticmethod
    def _edit(h: torch.Tensor, e: _LayerEdit) -> torch.Tensor:
        """Apply one layer's edit: fp32 arithmetic, cast back once, unmasked positions untouched."""
        out = h.clone()
        n_rows, width, d = int(e.rows.shape[0]), int(h.shape[1]), int(h.shape[2])
        step = max(1, _EDIT_CHUNK_ELEMENTS // max(1, width * d))
        for s in range(0, n_rows, step):
            rows = e.rows[s:s + step]
            x = h.index_select(0, rows).float()
            unit = e.unit[s:s + step]
            proj = torch.einsum("rtd,rd->rt", x, unit)
            coef = e.c[s:s + step, None] + e.b[s:s + step, None] * proj
            delta = (e.a[s:s + step, None, None] * e.vec[s:s + step, None, :]
                     + coef[..., None] * unit[:, None, :])
            y = torch.where(e.mask[s:s + step, :, None], x + delta, x)
            out.index_copy_(0, rows, y.to(h.dtype))
        return out

    def _place(self, mask: list[bool], pad: int, width: int) -> list[bool]:
        """An unpadded row mask moved into left-padded coordinates (bug 25b lives or dies here)."""
        placed = [False] * pad + list(mask)
        if len(placed) != width:
            raise RuntimeError(f"row mask of {len(mask)} positions with {pad} padding does not fill "
                               f"{width} columns")
        return placed

    def _edits(self, entries: Sequence[tuple[int, Intervention, list[bool]]]) -> dict[int, _LayerEdit]:
        """Per layer, the rows it edits with their vectors, operations, doses and masks."""
        by_layer: dict[int, list[tuple[int, Intervention, float, list[bool]]]] = {}
        for row, iv, mask in entries:
            if not any(mask):
                continue
            for term in iv.terms:
                by_layer.setdefault(term.layer, []).append((row, iv, float(term.alpha), mask))
        out: dict[int, _LayerEdit] = {}
        dev = self.device
        for layer, items in sorted(by_layer.items()):
            a, b, c = [], [], []
            for _, iv, alpha, _ in items:
                if iv.op == "add":
                    a.append(alpha); b.append(0.0); c.append(0.0)
                elif iv.op == "sub":
                    a.append(-alpha); b.append(0.0); c.append(0.0)
                elif iv.op == "project":
                    a.append(0.0); b.append(-alpha); c.append(0.0)
                elif iv.op == "clamp":
                    # m5.intervention's clamp sets the component to c; alpha names the layer only.
                    a.append(0.0); b.append(-1.0); c.append(float(iv.clamp_value))
                else:  # pragma: no cover - Intervention validates op
                    raise ValueError(f"unknown operation {iv.op!r}")
            vecs = [self._vectors[iv.vector_id][layer] for _, iv, _, _ in items]
            out[layer] = _LayerEdit(
                rows=torch.tensor([r for r, _, _, _ in items], dtype=torch.long, device=dev),
                vec=torch.stack([v.vec for v in vecs]),
                unit=torch.stack([v.unit for v in vecs]),
                a=torch.tensor(a, dtype=torch.float32, device=dev),
                b=torch.tensor(b, dtype=torch.float32, device=dev),
                c=torch.tensor(c, dtype=torch.float32, device=dev),
                mask=torch.tensor([m for _, _, _, m in items], dtype=torch.bool, device=dev),
            )
        return out

    def _batch(self, seqs: Sequence[Sequence[int]]
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], int]:
        """Left-padded ids, attention mask and position ids (real tokens count from 0 in every row)."""
        lens = [len(s) for s in seqs]
        width = max(lens)
        ids = torch.full((len(seqs), width), self._pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, width - len(s):] = torch.tensor(list(s), dtype=torch.long)
            attn[b, width - len(s):] = 1
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        return ids.to(self.device), attn.to(self.device), pos.to(self.device), \
            [width - n for n in lens], width

    def _run(self, p: _Pass, ids: torch.Tensor, attn: torch.Tensor, pos: torch.Tensor, *,
             past: Any = None, use_cache: bool = False) -> Any:
        """One forward pass under plan `p`; every attached hook must fire exactly once."""
        if self._pass is not None:
            raise RuntimeError("a forward pass is already running")
        p.fired = {k: 0 for k in self._hook_keys}
        self._pass = p
        try:
            out = self.model(input_ids=ids, attention_mask=attn, position_ids=pos,
                             past_key_values=past, use_cache=use_cache, logits_to_keep=1)
        finally:
            self._pass = None
        bad = {k: n for k, n in p.fired.items() if n != 1}
        if bad:
            raise RuntimeError(f"hooks fired {bad} times in one forward pass, expected once each: a hook "
                               "the pass never reaches steers nothing (bug 26), one reached twice "
                               "steers twice")
        return out

    def _chunks(self, n: int) -> Iterator[list[int]]:
        for s in range(0, n, self.batch_size):
            yield list(range(s, min(n, s + self.batch_size)))

    @staticmethod
    def _steer_layers(rows: Sequence[Any]) -> set[int]:
        return {t.layer for r in rows if r.iv is not None for t in r.iv.terms}

    def _final_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Logits from the final norm's output, computed as the loaded class computes them."""
        logits = self._lm_head(h)
        if self._softcap is not None:
            logits = logits / self._softcap
            logits = torch.tanh(logits)
            logits = logits * self._softcap
        return logits.float()

    def _logit_chunk(self) -> int:
        return max(1, _LOGIT_CHUNK_ELEMENTS // self._vocab)

    # ---------------------------------------------------------------- generation

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResult]:
        """Sample each request with its own sampler, steering applied per row.

        A fresh KV cache per batch (contract 3); temperature, top-k and top-p applied per request
        exactly as HF's warpers apply them (temperature, then top-k keeping ties with the k-th
        logit, then top-p, which is skipped at 1.0; temperature 0 is greedy); one CPU
        `torch.Generator` per row seeded from `seed` (contract 4). Stops at any stop id
        (`info.eos_token_ids`) or at max_new_tokens.
        """
        rows: list[_GenRow] = []
        for i, r in enumerate(requests):
            label = f"generate row {i}"
            if not isinstance(r, GenRequest):
                raise TypeError(f"{label}: expected a GenRequest")
            if not _is_int(r.top_k) or r.top_k < 0:
                raise ValueError(f"{label}: top_k must be an integer >= 0 (0 = off), got {r.top_k!r}")
            if not (isinstance(r.top_p, (int, float)) and 0.0 < r.top_p <= 1.0):
                raise ValueError(f"{label}: top_p must lie in (0, 1], got {r.top_p!r}")
            if not math.isfinite(r.temperature):
                raise ValueError(f"{label}: temperature must be finite")
            if not _is_int(r.seed):
                raise ValueError(f"{label}: seed must be an integer")
            ids = self.tokenize(r.prompt)
            self._check_ids(ids, label)
            iv = self._check_intervention(r.intervention, label)
            mask = self._steer_mask(iv, len(ids), len(ids), label)
            if iv is not None and not any(mask) and not iv.positions.steers_generated:
                raise ValueError(f"{label}: positions {iv.positions.to_json()} steer nothing")
            rows.append(_GenRow(ids=ids, iv=iv, mask=mask, max_new=int(r.max_new_tokens),
                                temperature=float(r.temperature), top_k=int(r.top_k),
                                top_p=float(r.top_p), seed=int(r.seed)))
        results: list[GenResult] = []
        with self._lock, torch.no_grad(), self._hooked(self._steer_layers(rows)):
            for chunk in self._chunks(len(rows)):
                for toks, finish, _, _ in self._generate_chunk([rows[i] for i in chunk]):
                    text = self.tokenizer.decode(toks, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
                    results.append(GenResult(text=text, token_ids=tuple(toks), finish=finish))
        return results

    def _decode_entries(self, rows: Sequence[_GenRow]) -> list[tuple[int, Intervention, list[bool]]]:
        """Rows whose generated tokens are steered at every decode step."""
        return [(b, r.iv, [True]) for b, r in enumerate(rows)
                if r.iv is not None and r.iv.positions.steers_generated]

    def _generate_chunk(self, rows: Sequence[_GenRow], forced: Sequence[Sequence[int]] | None = None
                        ) -> list[tuple[list[int], str, list[float], list[int]]]:
        """Prefill, then one token per step with a KV cache made for this batch alone.

        With `forced`, each step feeds the given token instead of sampling: the same prefill, the
        same cache, the same hooks and the same per-step plan as generation, which is what lets
        liveness attest the path generation uses (trap 14) rather than a sibling of it.
        Returns per row: tokens, finish reason, log p of each chosen token, and each step's argmax.
        """
        B = len(rows)
        ids, attn, pos, pads, width = self._batch([r.ids for r in rows])
        prefill = _Pass(batch=B, width=width, edits=self._edits(
            [(b, r.iv, self._place(r.mask, pads[b], width)) for b, r in enumerate(rows)
             if r.iv is not None]))
        decode_edits = self._edits(self._decode_entries(rows))
        out = self._run(prefill, ids, attn, pos, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :].float()
        n_steps = [len(forced[b]) for b in range(B)] if forced is not None else [r.max_new for r in rows]
        gens = None if forced is not None else \
            [torch.Generator(device="cpu").manual_seed(r.seed) for r in rows]
        tokens: list[list[int]] = [[] for _ in range(B)]
        lps: list[list[float]] = [[] for _ in range(B)]
        tops: list[list[int]] = [[] for _ in range(B)]
        finish = ["length"] * B
        done = [n == 0 for n in n_steps]
        cur = [len(r.ids) for r in rows]
        step = 0
        while not all(done):
            if forced is not None:
                chosen = [forced[b][step] if not done[b] else self._pad_id for b in range(B)]
            else:
                chosen = self._choose(logits, rows, gens, done)
            logp = torch.log_softmax(logits, dim=-1)
            chosen_t = torch.tensor(chosen, dtype=torch.long, device=logits.device)
            step_lp = logp.gather(1, chosen_t[:, None]).squeeze(1).tolist()
            step_top = logits.argmax(-1).tolist()   # what the greedy sampler takes
            for b in range(B):
                if done[b]:
                    continue
                tokens[b].append(int(chosen[b]))
                lps[b].append(float(step_lp[b]))
                tops[b].append(int(step_top[b]))
                if forced is None and chosen[b] in self._stop_set:
                    done[b], finish[b] = True, "eos"
                elif len(tokens[b]) >= n_steps[b]:
                    done[b] = True
            if all(done):
                break
            feed = torch.tensor([[tokens[b][-1] if not done[b] else self._pad_id] for b in range(B)],
                                dtype=torch.long, device=self.device)
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=attn.device)], dim=1)
            step_pos = torch.tensor([[c] for c in cur], dtype=torch.long, device=self.device)
            out = self._run(_Pass(batch=B, width=1, edits=decode_edits), feed, attn, step_pos,
                            past=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1, :].float()
            cur = [c + 1 for c in cur]
            step += 1
        del cache
        return [(tokens[b], finish[b], lps[b], tops[b]) for b in range(B)]

    def _choose(self, logits: torch.Tensor, rows: Sequence[_GenRow], gens: Sequence[torch.Generator],
                done: Sequence[bool]) -> list[int]:
        """One token per active row: greedy at temperature 0, else temperature/top-k/top-p and an
        inverse-CDF draw from the row's own generator over a fixed (stable, descending) order."""
        B = int(logits.shape[0])
        chosen = [self._pad_id] * B
        greedy = [b for b in range(B) if not done[b] and rows[b].temperature == 0.0]
        sample = [b for b in range(B) if not done[b] and rows[b].temperature > 0.0]
        dev = logits.device
        if greedy:
            idx = torch.tensor(greedy, dtype=torch.long, device=dev)
            for b, t in zip(greedy, logits.index_select(0, idx).argmax(-1).tolist()):
                chosen[b] = int(t)
        if sample:
            idx = torch.tensor(sample, dtype=torch.long, device=dev)
            order, probs, n_keep = self._filter(
                logits.index_select(0, idx),
                torch.tensor([rows[b].temperature for b in sample], dtype=torch.float32, device=dev),
                torch.tensor([rows[b].top_k for b in sample], dtype=torch.long, device=dev),
                torch.tensor([rows[b].top_p for b in sample], dtype=torch.float32, device=dev))
            cdf = probs.cumsum(-1)
            u = torch.tensor([float(torch.rand((), generator=gens[b], dtype=torch.float64))
                              for b in sample], dtype=torch.float64, device=dev)
            pick = torch.searchsorted(cdf, (u * cdf[:, -1])[:, None], right=True).squeeze(1)
            pick = torch.minimum(pick, n_keep - 1)
            for b, t in zip(sample, order.gather(1, pick[:, None]).squeeze(1).tolist()):
                chosen[b] = int(t)
        return chosen

    @staticmethod
    def _filter(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor,
                top_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The sampling distribution per row, in the order the draw walks it.

        Temperature, then top-k keeping every token tied with the k-th logit (0 or >= vocab = off),
        then top-p over what top-k kept (1.0 = off), at least one token always kept: HF's
        Temperature/TopK/TopP warpers. Returns the descending stable order of each row, the fp64
        probabilities in that order (zero beyond the kept prefix), and the kept count.
        """
        V = logits.shape[-1]
        srt, order = torch.sort(logits / temperature[:, None], dim=-1, descending=True, stable=True)
        k = torch.where((top_k > 0) & (top_k < V), top_k, torch.full_like(top_k, V))
        keep = srt >= srt.gather(1, (k - 1)[:, None])
        probs = torch.softmax(srt.masked_fill(~keep, float("-inf")), dim=-1)
        before = probs.cumsum(-1) - probs
        keep = keep & ((before < top_p[:, None]) | (top_p[:, None] >= 1.0))
        keep[:, 0] = True
        final = torch.softmax(srt.masked_fill(~keep, float("-inf")).double(), dim=-1)
        return order, final, keep.sum(-1)

    # ---------------------------------------------------------------- next-token reads

    def next_token_logprobs(self, prompts: Sequence[str], interventions: Sequence[Intervention | None],
                            token_ids: Sequence[int]) -> np.ndarray:
        """log p of each candidate as the next token: log-softmax in fp32 of the model's own logits
        at each prompt's last token (left padding puts every row's last token in the last column)."""
        rows = self._rows(prompts, interventions, "next_token_logprobs")
        tids = self._token_ids(token_ids, "next_token_logprobs")
        out = np.empty((len(rows), len(tids)), dtype=np.float64)
        if not rows:
            return out
        tid_t = torch.tensor(tids, dtype=torch.long, device=self.device)
        with self._lock, torch.no_grad(), self._hooked(self._steer_layers(rows)):
            for chunk in self._chunks(len(rows)):
                sub = [rows[i] for i in chunk]
                ids, attn, pos, pads, width = self._batch([r.ids for r in sub])
                p = _Pass(batch=len(sub), width=width, edits=self._edits(
                    [(b, r.iv, self._place(r.mask, pads[b], width)) for b, r in enumerate(sub)
                     if r.iv is not None]))
                res = self._run(p, ids, attn, pos)
                lp = torch.log_softmax(res.logits[:, -1, :].float(), dim=-1)
                out[chunk[0]:chunk[-1] + 1] = lp.index_select(1, tid_t).double().cpu().numpy()
        return out

    # ---------------------------------------------------------------- teacher forcing

    def _split_in_context(self, prompt: str, continuation: str, what: str) -> tuple[list[int], int]:
        """Tokenise prompt + continuation together; the continuation's tokens are those that start
        at or after len(prompt) characters.

        Tokenisation in context is the choice because it is what the model reads: a continuation
        tokenised alone can differ from the same text after the prompt (SentencePiece marks the
        first word of a text differently; BPE can merge across the join). Two cases raise rather
        than guess: a token straddling the join, and a prompt whose own tokens change in context
        (then its steering positions would point at different tokens).
        """
        if not continuation:
            raise ValueError(f"{what}: empty continuation")
        full = prompt + continuation
        enc = self.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = [int(i) for i in enc["input_ids"]]
        offs = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        cut = len(prompt)
        k = next((i for i, (s, _) in enumerate(offs) if s >= cut), len(ids))
        if k == 0:
            raise ValueError(f"{what}: empty prompt; the first continuation token has no position to "
                             "be predicted from")
        if k == len(ids):
            raise ValueError(f"{what}: the continuation adds no token after the prompt")
        if k > 0 and offs[k - 1][1] > cut:
            raise ValueError(f"{what}: token {k - 1} spans the prompt/continuation join; score a "
                             "continuation that starts on a token boundary")
        if ids[:k] != self.tokenize(prompt):
            raise ValueError(f"{what}: the prompt tokenises differently in front of this continuation")
        return ids, k

    def _forced_rows(self, prompts: Sequence[str], interventions: Sequence[Any],
                     continuations: Sequence[Sequence[int]], what: str) -> list[_Row]:
        if not (len(prompts) == len(interventions) == len(continuations)):
            raise ValueError(f"{what}: prompts, interventions and continuations differ in length")
        rows = []
        for i, (text, iv, cont) in enumerate(zip(prompts, interventions, continuations)):
            label = f"{what} row {i}"
            ids = self.tokenize(text)
            self._check_ids(ids, label)
            cont = self._token_ids(cont, label)
            if not cont:
                raise ValueError(f"{label}: empty continuation")
            iv = self._check_intervention(iv, label)
            mask = self._steer_mask(iv, len(ids), len(ids) + len(cont), label)
            if mask is not None and not any(mask):
                raise ValueError(f"{label}: positions {iv.positions.to_json()} steer nothing")
            rows.append(_Row(ids=ids + cont, n_prompt=len(ids), iv=iv, mask=mask))
        return rows

    def _teacher_forced(self, rows: Sequence[_Row], candidates: Sequence[int] | None = None
                        ) -> list[dict[str, np.ndarray]]:
        """One prefill pass per batch; the distributions predicting each continuation token.

        Reads the final norm's output at those positions and unembeds them in chunks, so a long
        continuation over a 262k vocabulary never materialises a [batch, length, vocab] tensor.
        """
        results: list[dict[str, np.ndarray]] = []
        cand_t = None if candidates is None else \
            torch.tensor(list(candidates), dtype=torch.long, device=self.device)
        with self._lock, torch.no_grad(), self._hooked(self._steer_layers(rows), final=True):
            for chunk in self._chunks(len(rows)):
                sub = [rows[i] for i in chunk]
                ids, attn, pos, pads, width = self._batch([r.ids for r in sub])
                rr, tt, targets, counts = [], [], [], []
                for b, r in enumerate(sub):
                    m = len(r.ids) - r.n_prompt
                    rr += [b] * m
                    tt += [pads[b] + r.n_prompt - 1 + j for j in range(m)]
                    targets += r.ids[r.n_prompt:]
                    counts.append(m)
                rr_t = torch.tensor(rr, dtype=torch.long, device=self.device)
                tt_t = torch.tensor(tt, dtype=torch.long, device=self.device)
                captured: list[torch.Tensor] = []
                p = _Pass(batch=len(sub), width=width, edits=self._edits(
                    [(b, r.iv, self._place(r.mask, pads[b], width)) for b, r in enumerate(sub)
                     if r.iv is not None]),
                    final_readers=[lambda h: captured.append(h[rr_t, tt_t])])
                self._run(p, ids, attn, pos)
                final = captured[0]
                tgt = torch.tensor(targets, dtype=torch.long, device=self.device)
                lp_parts, ent_parts, top_parts, cand_parts = [], [], [], []
                step = self._logit_chunk()
                for s in range(0, final.shape[0], step):
                    logits = self._final_logits(final[s:s + step])
                    lp = torch.log_softmax(logits, dim=-1)
                    lp_parts.append(lp.gather(1, tgt[s:s + step, None]).squeeze(1).double().cpu())
                    ent_parts.append((-(lp.exp() * lp).sum(-1)).double().cpu())
                    top_parts.append(logits.argmax(-1).cpu())
                    if cand_t is not None:
                        cand_parts.append(lp.index_select(1, cand_t).double().cpu())
                lps = torch.cat(lp_parts).numpy()
                ents = torch.cat(ent_parts).numpy()
                tops = torch.cat(top_parts).numpy().astype(np.int64)
                cands = torch.cat(cand_parts).numpy() if cand_t is not None else None
                s = 0
                for b, m in enumerate(counts):
                    item = {"ids": np.array(sub[b].ids[sub[b].n_prompt:], dtype=np.int64),
                            "logprobs": lps[s:s + m], "entropies": ents[s:s + m], "argmax": tops[s:s + m]}
                    if cands is not None:
                        item["candidates"] = cands[s:s + m]
                    results.append(item)
                    s += m
        return results

    def _score_rows(self, requests: Sequence[ScoreRequest], what: str) -> list[_Row]:
        rows = []
        for i, r in enumerate(requests):
            label = f"{what} row {i}"
            if not isinstance(r, ScoreRequest):
                raise TypeError(f"{label}: expected a ScoreRequest")
            ids, k = self._split_in_context(r.prompt, r.continuation, label)
            self._check_ids(ids, label)
            iv = self._check_intervention(r.intervention, label)
            mask = self._steer_mask(iv, k, len(ids), label)
            if mask is not None and not any(mask):
                raise ValueError(f"{label}: positions {iv.positions.to_json()} steer nothing")
            rows.append(_Row(ids=ids, n_prompt=k, iv=iv, mask=mask))
        return rows

    def score(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        """Teacher forcing: log p and predictive entropy (nats) of each continuation token.

        The continuation is tokenised in context (see `_split_in_context`), and its tokens are
        steered as generated tokens, so the score is what generation would compute for them.
        """
        rows = self._score_rows(requests, "score")
        return [ScoreResult(token_ids=tuple(int(t) for t in r["ids"]),
                            logprobs=tuple(float(x) for x in r["logprobs"]),
                            entropies=tuple(float(x) for x in r["entropies"]))
                for r in self._teacher_forced(rows)]

    def candidate_logprobs(self, requests: Sequence[ScoreRequest],
                           token_ids: Sequence[int]) -> list[np.ndarray]:
        """Per request [len(continuation tokens), len(token_ids)]: log p of each candidate under the
        distribution predicting continuation token t -- the distributions `score` reads. e9
        (teacher-forced concept lift) reads the concept's tokens along a fixed continuation."""
        rows = self._score_rows(requests, "candidate_logprobs")
        tids = self._token_ids(token_ids, "candidate_logprobs")
        return [r["candidates"] for r in self._teacher_forced(rows, tids)]

    def forced_logprobs(self, prompts: Sequence[str], interventions: Sequence[Intervention | None],
                        continuations: Sequence[Sequence[int]], *, path: str = "prefill") -> list[Forced]:
        """Teacher forcing on token ids, through one of the two paths the backend computes with.

        `prefill`: one forward pass over prompt + continuation (how `score` reads). `decode`: the
        generation loop itself -- prefill, KV cache, one token per step with the decode-phase plan
        -- feeding the given ids instead of sampling. The two must agree; liveness checks that they
        do, which is how generation's own hooks are attested (trap 14).
        """
        rows = self._forced_rows(prompts, interventions, continuations, "forced_logprobs")
        if path == "prefill":
            return [Forced(logprobs=r["logprobs"], argmax=r["argmax"]) for r in self._teacher_forced(rows)]
        if path != "decode":
            raise ValueError(f"path must be 'prefill' or 'decode', got {path!r}")
        gen_rows = [_GenRow(ids=r.ids[:r.n_prompt], iv=r.iv,
                            mask=None if r.mask is None else r.mask[:r.n_prompt],
                            max_new=len(r.ids) - r.n_prompt, temperature=0.0, top_k=0, top_p=1.0, seed=0)
                    for r in rows]
        out: list[Forced] = []
        with self._lock, torch.no_grad(), self._hooked(self._steer_layers(gen_rows)):
            for chunk in self._chunks(len(gen_rows)):
                sub = [gen_rows[i] for i in chunk]
                forced = [rows[i].ids[rows[i].n_prompt:] for i in chunk]
                for _, _, lps, tops in self._generate_chunk(sub, forced=forced):
                    out.append(Forced(logprobs=np.array(lps, dtype=np.float64),
                                      argmax=np.array(tops, dtype=np.int64)))
        return out

    # ---------------------------------------------------------------- KL

    def kl_from_unsteered(self, texts: Sequence[str], interventions: Sequence[Intervention], *,
                          start_tokens: Sequence[int],
                          exclude_token_ids: Sequence[int] = ()) -> list[KLResult]:
        """s6 (KL(unsteered || steered) on neutral text): the mean over the next-token
        distributions at positions start_tokens[i] .. n-1 of each text (each predicting the token
        after it), from two passes with identical batches. The text is all prompt: its positions
        index tokenize(text), and a policy that steers only generated tokens steers nothing here
        and raises. The `without` variant removes `exclude_token_ids` from both distributions and
        renormalises before the KL. Per-position values below zero by rounding are clamped to zero.
        """
        if len(texts) != len(interventions) or len(texts) != len(start_tokens):
            raise ValueError("kl_from_unsteered: texts, interventions and start_tokens differ in length")
        for i, iv in enumerate(interventions):
            if iv is None:
                raise ValueError(f"kl_from_unsteered row {i}: no intervention; the unsteered arm's KL "
                                 "is zero by definition")
        rows = self._rows(texts, interventions, "kl_from_unsteered")
        starts = []
        for i, (r, s) in enumerate(zip(rows, start_tokens)):
            if not _is_int(s) or not 0 <= int(s) < len(r.ids):
                raise ValueError(f"kl_from_unsteered row {i}: start {s!r} is not a position of the "
                                 f"text ({len(r.ids)} tokens)")
            starts.append(int(s))
        excl = sorted(set(self._token_ids(exclude_token_ids, "kl_from_unsteered")))
        keep = None
        if excl:
            keep = torch.ones(self._vocab, dtype=torch.bool, device=self.device)
            keep[torch.tensor(excl, dtype=torch.long, device=self.device)] = False
        results: list[KLResult] = []
        with self._lock, torch.no_grad(), self._hooked(self._steer_layers(rows), final=True):
            for chunk in self._chunks(len(rows)):
                sub = [rows[i] for i in chunk]
                sub_starts = [starts[i] for i in chunk]
                ids, attn, pos, pads, width = self._batch([r.ids for r in sub])
                rr, tt, counts = [], [], []
                for b, r in enumerate(sub):
                    n = len(r.ids) - sub_starts[b]
                    rr += [b] * n
                    tt += [pads[b] + sub_starts[b] + j for j in range(n)]
                    counts.append(n)
                rr_t = torch.tensor(rr, dtype=torch.long, device=self.device)
                tt_t = torch.tensor(tt, dtype=torch.long, device=self.device)
                cap_u: list[torch.Tensor] = []
                cap_s: list[torch.Tensor] = []
                self._run(_Pass(batch=len(sub), width=width,
                                final_readers=[lambda h: cap_u.append(h[rr_t, tt_t])]), ids, attn, pos)
                self._run(_Pass(batch=len(sub), width=width, edits=self._edits(
                    [(b, r.iv, self._place(r.mask, pads[b], width)) for b, r in enumerate(sub)]),
                    final_readers=[lambda h: cap_s.append(h[rr_t, tt_t])]), ids, attn, pos)
                kl_all, kl_wo = [], []
                step = self._logit_chunk()
                for s in range(0, cap_u[0].shape[0], step):
                    lu = torch.log_softmax(self._final_logits(cap_u[0][s:s + step]), dim=-1)
                    ls = torch.log_softmax(self._final_logits(cap_s[0][s:s + step]), dim=-1)
                    kl_all.append(self._kl(lu, ls).double().cpu())
                    if keep is not None:
                        kl_wo.append(self._kl(self._renorm(lu, keep), self._renorm(ls, keep),
                                              keep).double().cpu())
                kl_all_np = torch.cat(kl_all).numpy()
                kl_wo_np = torch.cat(kl_wo).numpy() if keep is not None else None
                s = 0
                for n in counts:
                    results.append(KLResult(
                        mean_kl=float(kl_all_np[s:s + n].mean()), n_positions=int(n),
                        mean_kl_without=None if kl_wo_np is None else float(kl_wo_np[s:s + n].mean())))
                    s += n
        return results

    @staticmethod
    def _renorm(lp: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        masked = lp.masked_fill(~keep, float("-inf"))
        return masked - torch.logsumexp(masked, dim=-1, keepdim=True)

    @staticmethod
    def _kl(lp_p: torch.Tensor, lp_q: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        terms = lp_p.exp() * (lp_p - lp_q)
        if keep is not None:
            terms = torch.where(keep, terms, torch.zeros((), dtype=terms.dtype, device=terms.device))
        return terms.sum(-1).clamp(min=0.0)

    # ---------------------------------------------------------------- residual reads

    def _positions(self, n: int, positions: Sequence[int], what: str) -> list[int]:
        out = []
        for p in positions:
            if not _is_int(p) or not -n <= int(p) < n:
                raise ValueError(f"{what}: position {p!r} is outside the prompt ({n} tokens)")
            out.append(int(p) % n)
        return out

    def hidden(self, prompts: Sequence[str], interventions: Sequence[Intervention | None],
               layers: Sequence[int], positions: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Per prompt [len(positions_i), len(layers), d_model] fp32: the output of block L at those
        positions, after any steering at L (what vector extraction and the q1 cross-vector probe
        read). Negative positions count from the end of the prompt."""
        rows = self._rows(prompts, interventions, "hidden")
        layer_list = self._layer_list(layers, "hidden")
        if len(positions) != len(rows):
            raise ValueError(f"hidden: {len(rows)} prompts but {len(positions)} position lists")
        pos_lists = [self._positions(len(r.ids), ps, f"hidden row {i}")
                     for i, (r, ps) in enumerate(zip(rows, positions))]
        out: list[np.ndarray] = []
        with self._lock, torch.no_grad(), self._hooked(self._steer_layers(rows) | set(layer_list)):
            for chunk in self._chunks(len(rows)):
                sub = [rows[i] for i in chunk]
                ids, attn, pos, pads, width = self._batch([r.ids for r in sub])
                rr, tt = [], []
                for b, i in enumerate(chunk):
                    rr += [b] * len(pos_lists[i])
                    tt += [pads[b] + q for q in pos_lists[i]]
                rr_t = torch.tensor(rr, dtype=torch.long, device=self.device)
                tt_t = torch.tensor(tt, dtype=torch.long, device=self.device)
                store: dict[int, torch.Tensor] = {}
                readers = {L: [lambda h, L=L: store.__setitem__(L, h[rr_t, tt_t].float().cpu())]
                           for L in layer_list}
                p = _Pass(batch=len(sub), width=width, readers=readers, edits=self._edits(
                    [(b, r.iv, self._place(r.mask, pads[b], width)) for b, r in enumerate(sub)
                     if r.iv is not None]))
                self._run(p, ids, attn, pos)
                stacked = torch.stack([store[L] for L in layer_list], dim=1)  # [P, layers, d]
                s = 0
                for i in chunk:
                    n = len(pos_lists[i])
                    out.append(stacked[s:s + n].numpy().astype(np.float32, copy=True))
                    s += n
        return out

    def reference_hidden(self, prompt: str, layers: Sequence[int]) -> np.ndarray:
        """[n_tokens, len(layers), d_model] fp32 from HF's own `output_hidden_states[L + 1]`, for
        one unpadded prompt with no hook attached: an independent witness of which module the
        hooks treat as layer L (bug 4 was a lookup that could index the wrong stack, and no check
        inside the hook can see that). The last layer is refused because transformers 5.x ties
        `hidden_states[-1]` to the post-final-norm state, not block n-1's output."""
        layer_list = self._layer_list(layers, "reference_hidden")
        if self._n_layers - 1 in layer_list:
            raise ValueError("reference_hidden: HF reports the last layer after the final norm; "
                             f"layer {self._n_layers - 1} has no independent reference")
        ids = self.tokenize(prompt)
        self._check_ids(ids, "reference_hidden")
        with self._lock, torch.no_grad():
            if self._handles:
                raise RuntimeError("reference_hidden must run with no steering hook attached")
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            out = self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                             output_hidden_states=True, logits_to_keep=1)
        states = out.hidden_states
        if states is None or len(states) != self._n_layers + 1:
            raise RuntimeError(f"output_hidden_states returned {None if states is None else len(states)} "
                               f"states for {self._n_layers} layers")
        return np.stack([states[L + 1][0].float().cpu().numpy() for L in layer_list], axis=1)

    def lens(self, prompts: Sequence[str], interventions: Sequence[Intervention | None],
             layers: Sequence[int], position: int, token_ids: Sequence[int]) -> np.ndarray:
        """Logit lens, for q2 (lens readouts): [n_prompts, len(layers), len(token_ids)] log-probs
        when the residual after block L at `position` goes through the final norm, the unembedding
        and (when the loaded class applies one) the final softcap, in the model's dtype as the
        model computes them. At the last layer this is the model's own next-token distribution at
        that position."""
        rows = self._rows(prompts, interventions, "lens")
        layer_list = self._layer_list(layers, "lens")
        tids = self._token_ids(token_ids, "lens")
        tid_t = torch.tensor(tids, dtype=torch.long, device=self.device)
        out = np.empty((len(rows), len(layer_list), len(tids)), dtype=np.float64)
        pos_of = [self._positions(len(r.ids), [position], f"lens row {i}")[0] for i, r in enumerate(rows)]
        with self._lock, torch.no_grad(), self._hooked(self._steer_layers(rows) | set(layer_list)):
            for chunk in self._chunks(len(rows)):
                sub = [rows[i] for i in chunk]
                ids, attn, pos, pads, width = self._batch([r.ids for r in sub])
                rr_t = torch.arange(len(sub), dtype=torch.long, device=self.device)
                tt_t = torch.tensor([pads[b] + pos_of[i] for b, i in enumerate(chunk)],
                                    dtype=torch.long, device=self.device)
                store: dict[int, torch.Tensor] = {}
                readers = {L: [lambda h, L=L: store.__setitem__(L, h[rr_t, tt_t].clone())]
                           for L in layer_list}
                p = _Pass(batch=len(sub), width=width, readers=readers, edits=self._edits(
                    [(b, r.iv, self._place(r.mask, pads[b], width)) for b, r in enumerate(sub)
                     if r.iv is not None]))
                self._run(p, ids, attn, pos)
                for j, L in enumerate(layer_list):
                    lp = torch.log_softmax(self._final_logits(self._final_norm(store[L])), dim=-1)
                    out[chunk[0]:chunk[-1] + 1, j, :] = lp.index_select(1, tid_t).double().cpu().numpy()
        return out

    def residual_stats(self, prompts: Sequence[str], layers: Sequence[int], *,
                       masks: Mapping[int, Sequence[int]] | None = None,
                       skip_first: int = 1) -> dict[int, ResidualStats]:
        """Unsteered residual statistics over every real token of every prompt except the first
        `skip_first` (the BOS sink), padding never counted: means accumulated in fp64 and returned
        in fp32, per-token L2 norms in prompt order, and, when `masks` is given (it must then name
        every requested layer, possibly with no coordinates), the norms with those coordinates
        zeroed -- the inputs of the nuisance scan and of s_L (plan §4.3)."""
        layer_list = self._layer_list(layers, "residual_stats")
        if not _is_int(skip_first) or skip_first < 0:
            raise ValueError(f"residual_stats: skip_first must be an integer >= 0, got {skip_first!r}")
        coords: dict[int, torch.Tensor] = {}
        if masks is not None:
            missing = [L for L in layer_list if L not in masks]
            if missing:
                raise ValueError(f"residual_stats: masks name no coordinates for layers {missing}; "
                                 "pass an empty list for a layer with none")
            for L in layer_list:
                cs = [int(c) for c in masks[L]]
                bad = [c for c in cs if not 0 <= c < self._d_model]
                if bad:
                    raise ValueError(f"residual_stats: layer {L} mask coordinates {bad} out of range")
                coords[L] = torch.tensor(sorted(set(cs)), dtype=torch.long, device=self.device)
        rows = self._rows(prompts, [None] * len(prompts), "residual_stats")
        def zeros() -> torch.Tensor:
            return torch.zeros(self._d_model, dtype=torch.float64, device=self.device)

        sums = {L: zeros() for L in layer_list}
        abs_sums = {L: zeros() for L in layer_list}
        norms: dict[int, list[torch.Tensor]] = {L: [] for L in layer_list}
        masked: dict[int, list[torch.Tensor]] = {L: [] for L in layer_list}
        count = 0
        with self._lock, torch.no_grad(), self._hooked(set(layer_list)):
            for chunk in self._chunks(len(rows)):
                sub = [rows[i] for i in chunk]
                ids, attn, pos, pads, width = self._batch([r.ids for r in sub])
                rr, tt = [], []
                for b, r in enumerate(sub):
                    q = list(range(skip_first, len(r.ids)))
                    rr += [b] * len(q)
                    tt += [pads[b] + j for j in q]
                if not rr:
                    continue
                rr_t = torch.tensor(rr, dtype=torch.long, device=self.device)
                tt_t = torch.tensor(tt, dtype=torch.long, device=self.device)

                def read(h: torch.Tensor, L: int) -> None:
                    x = h[rr_t, tt_t].double()
                    sums[L] += x.sum(0)
                    abs_sums[L] += x.abs().sum(0)
                    norms[L].append(torch.linalg.vector_norm(x, dim=-1).float().cpu())
                    if L in coords:
                        xm = x.clone()
                        if coords[L].numel():
                            xm[:, coords[L]] = 0.0
                        masked[L].append(torch.linalg.vector_norm(xm, dim=-1).float().cpu())

                readers = {L: [lambda h, L=L: read(h, L)] for L in layer_list}
                self._run(_Pass(batch=len(sub), width=width, readers=readers), ids, attn, pos)
                count += len(rr)
        if count == 0:
            raise ValueError(f"residual_stats: no token is counted (every prompt has <= {skip_first} "
                             "tokens)")
        out: dict[int, ResidualStats] = {}
        for L in layer_list:
            out[L] = ResidualStats(
                layer=L, n_tokens=count,
                mean=(sums[L] / count).float().cpu().numpy(),
                mean_abs=(abs_sums[L] / count).float().cpu().numpy(),
                norms=torch.cat(norms[L]).numpy().astype(np.float32, copy=False),
                masked_norms=(torch.cat(masked[L]).numpy().astype(np.float32, copy=False)
                              if L in coords else None))
        return out

    def norm_gains(self, layer: int) -> dict[str, np.ndarray]:
        """The effective gain of each RMSNorm in block L, by attribute name, as fp32 [d_model].

        Effective means what multiplies the normalised residual: Gemma's RMSNorm multiplies by
        (1 + weight), Qwen3's by weight. It is read by applying the module to a constant vector
        (whose normalisation is exactly one), so the parametrisation never has to be special-cased
        and the weights-only nuisance scan compares like with like across families. Norms of
        another width (per-head q/k norms) are not residual gains and are left out.
        """
        if not _is_int(layer) or not 0 <= int(layer) < self._n_layers:
            raise ValueError(f"norm_gains: layer {layer!r} is not in [0, {self._n_layers})")
        block = self._layers[int(layer)]
        out: dict[str, np.ndarray] = {}
        with self._lock, torch.no_grad():
            for name, module in block.named_children():
                weight = getattr(module, "weight", None)
                if "rmsnorm" not in type(module).__name__.lower() or not torch.is_tensor(weight) \
                        or weight.dim() != 1 or int(weight.shape[0]) != self._d_model:
                    continue
                probe = torch.full((1, self._d_model), 1000.0, dtype=torch.float32, device=weight.device)
                out[name] = module(probe)[0].float().cpu().numpy().astype(np.float32, copy=True)
        return out
