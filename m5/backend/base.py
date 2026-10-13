"""m5.backend.base - the one seam between M5 and a model.

Everything that touches weights goes through `Backend`. Two implementations exist:

    m5.backend.hf    Hugging Face transformers + torch forward hooks (the real subject)
    m5.backend.fake  a simulator with a known ground truth (tests, and validating the search)

Every measure, readout and search step is written against this protocol only, so the whole
pipeline runs end to end on the simulator, and the simulator's ground truth tells us whether the
search recovers what is really there (plan §5.2, "what this gives up, and the mitigation").

## Contracts every implementation must honour

1. **Exact tokens.** `tokenize(text)` returns exactly the ids the model is fed for `text`, adding
   nothing (the chat template already writes BOS). Every position an `Intervention` names is an
   index into that list.
2. **Batch independence.** A row's result does not depend on what else is in its batch, beyond
   floating-point noise. In particular each row's intervention -- vector, operation, layers,
   positions -- applies to that row only (per-row layer ids, plan §8.1).
3. **No steering leaks across requests.** Nothing computed under one intervention (KV cache,
   cached logits) is reused for a request with a different intervention (M2 bug 23 class).
4. **Determinism.** Same request and seed, same result.
5. **fp32 statistics.** Norms, means, KL and entropies are accumulated in fp32 or wider.
6. **Layer L** is the output of decoder block L, 0-based (`m5.intervention`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from m5.intervention import Intervention


@dataclass(frozen=True)
class ModelInfo:
    name: str
    hf_id: str
    revision: str | None
    family: str
    n_layers: int
    d_model: int
    vocab_size: int
    dtype: str
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]
    chat_template_hash: str

    def to_json(self) -> dict:
        return dict(self.__dict__, eos_token_ids=list(self.eos_token_ids))


@dataclass(frozen=True)
class GenRequest:
    """One generation. `prompt` is fully rendered (template applied, any prefill appended)."""

    prompt: str
    intervention: Intervention | None
    max_new_tokens: int
    temperature: float
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")


@dataclass(frozen=True)
class GenResult:
    text: str                    # the new tokens only, decoded, special tokens skipped
    token_ids: tuple[int, ...]   # the new tokens, including a final stop token if one was sampled
    finish: str                  # "eos" (a stop token was sampled) or "length"


@dataclass(frozen=True)
class ScoreRequest:
    """Teacher forcing: score `continuation` after `prompt` under `intervention`."""

    prompt: str
    continuation: str
    intervention: Intervention | None


@dataclass(frozen=True)
class ScoreResult:
    token_ids: tuple[int, ...]   # the continuation's tokens (tokenized in context, see below)
    logprobs: tuple[float, ...]  # log p(token_i | prompt, continuation_<i), nats
    entropies: tuple[float, ...]  # entropy of the predictive distribution at each position, nats

    @property
    def total(self) -> float:
        return float(sum(self.logprobs))


@dataclass(frozen=True)
class KLResult:
    mean_kl: float               # mean over scored positions of KL(unsteered || steered), nats
    n_positions: int
    mean_kl_without: float | None  # the same with `exclude_token_ids` removed and renormalised


@dataclass(frozen=True)
class ResidualStats:
    layer: int
    n_tokens: int
    mean: np.ndarray             # [d] mean of h over the counted tokens, fp32
    mean_abs: np.ndarray         # [d] mean of |h|, fp32
    norms: np.ndarray            # [n_tokens] per-token L2 norms, fp32
    masked_norms: np.ndarray | None  # per-token norms with the masked coordinates zeroed


@runtime_checkable
class Backend(Protocol):
    """See the module docstring for the contracts. Shapes are documented per method."""

    info: ModelInfo

    # ---------------------------------------------------------------- text
    def render(self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = True) -> str:
        """Apply the chat template. Messages are {'role': 'system'|'user'|'assistant', 'content'}.
        With add_generation_prompt the result ends where the assistant's reply begins, so a
        prefill is plain string concatenation."""
        ...

    def tokenize(self, text: str) -> list[int]: ...

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span [start, end) of each token of tokenize(text)."""
        ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def word_variants(self, word: str) -> list[tuple[int, ...]]:
        """Distinct token sequences the word can take as the next words of a reply:
        with and without a leading space, lower case and capitalised. Used by every
        concept-probability readout so 'garlic', ' Garlic' and ' garlic' all count."""
        ...

    # ---------------------------------------------------------------- vectors
    def register_vector(self, vector_id: str, per_layer: Mapping[int, np.ndarray]) -> None:
        """Make a vector available to interventions. Values are copied to the model's device."""
        ...

    def has_vector(self, vector_id: str) -> bool: ...

    # ---------------------------------------------------------------- forward passes
    def generate(self, requests: Sequence[GenRequest]) -> list[GenResult]: ...

    def next_token_logprobs(self, prompts: Sequence[str],
                            interventions: Sequence[Intervention | None],
                            token_ids: Sequence[int]) -> np.ndarray:
        """[n_prompts, len(token_ids)] log-probabilities of each candidate as the next token."""
        ...

    def score(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]: ...

    def kl_from_unsteered(self, texts: Sequence[str], interventions: Sequence[Intervention],
                          *, start_tokens: Sequence[int],
                          exclude_token_ids: Sequence[int] = ()) -> list[KLResult]:
        """Teacher-forced KL(unsteered || steered) per text over positions >= start_tokens[i]."""
        ...

    def hidden(self, prompts: Sequence[str], interventions: Sequence[Intervention | None],
               layers: Sequence[int], positions: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Per prompt, an array [len(positions_i), len(layers), d_model] of residuals (fp32).
        Negative positions count from the end of the prompt."""
        ...

    def lens(self, prompts: Sequence[str], interventions: Sequence[Intervention | None],
             layers: Sequence[int], position: int, token_ids: Sequence[int]) -> np.ndarray:
        """Logit lens: [n_prompts, len(layers), len(token_ids)] log-probs of the candidates when
        the residual at `position` after block L goes straight through the final norm and the
        unembedding."""
        ...

    def residual_stats(self, prompts: Sequence[str], layers: Sequence[int], *,
                       masks: Mapping[int, Sequence[int]] | None = None,
                       skip_first: int = 1) -> dict[int, ResidualStats]:
        """Unsteered residual statistics over every token of every prompt except the first
        `skip_first` (the BOS sink) -- the inputs to the nuisance scan and to s_L."""
        ...

    def norm_gains(self, layer: int) -> dict[str, np.ndarray]:
        """The block's RMSNorm gain vectors by name, for the weights-only nuisance scan.
        Empty when the architecture has none to report."""
        ...


# =====================================================================================
# Helpers every implementation and every prompt builder shares
# =====================================================================================

def token_index_at(offsets: Sequence[tuple[int, int]], char_index: int) -> int:
    """The index of the first token whose span ends after `char_index`, i.e. the token that
    contains that character or, if it falls in a gap, the next token."""
    for i, (_, end) in enumerate(offsets):
        if end > char_index:
            return i
    raise ValueError(f"character {char_index} is beyond the last token")


def locate(backend: Backend, text: str, needle: str, *, occurrence: int = -1) -> int:
    """Token index of the first token of `needle` in `text` (the last occurrence by default).

    Raises when the needle is absent: a steering start that silently falls back to some other
    position is the defect class that withdrew the old Qwen arm.
    """
    starts = []
    i = text.find(needle)
    while i != -1:
        starts.append(i)
        i = text.find(needle, i + 1)
    if not starts:
        raise ValueError(f"marker {needle!r} not found in the rendered prompt")
    return token_index_at(backend.offsets(text), starts[occurrence])


def derive_seed(base: int, *parts: object) -> int:
    """A 31-bit seed from a base seed and any identifying parts (cell, prompt, repeat)."""
    h = hashlib.sha256(repr((base,) + parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))
