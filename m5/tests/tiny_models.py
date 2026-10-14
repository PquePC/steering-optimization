"""Tiny random Gemma-3 and Qwen3 models and tokenizers, built in-process for the backend tests.

No downloads: the tokenizers are trained on a few sentences with the `tokenizers` library (training
is deterministic and takes milliseconds), the models are the real transformers classes built from
small configs. What the tiny versions keep from the real checkpoints, because the backend depends
on it:

* the classes and attribute paths: `Gemma3ForCausalLM`, the multimodal
  `Gemma3ForConditionalGeneration` (decoder stack at `model.language_model.layers`, lm_head
  outside it) and `Qwen3ForCausalLM`; decoder layers that return a bare tensor;
* Gemma's sliding-window layers mixed with global ones, a window shorter than the test prompts;
* Gemma's RMSNorm parametrisation (1 + weight) against Qwen's (weight), with the norm weights
  perturbed so the two give different gains;
* an lm_head with a few more rows than the tokenizer has tokens (Gemma pads its embedding);
* a final-logit softcap on the text model (Gemma-2 style; the released Gemma-3 configs set none),
  small enough to bend these small logits, so a lens that forgets or invents it is visible;
* the tokenizers' shapes: a SentencePiece-style BPE with byte fallback (" garlic" and "garlic" are
  different tokens; byte tokens are ordinary vocabulary, as in Gemma) with special `<bos>`, `<eos>`,
  `<pad>`, `<start_of_turn>`, `<end_of_turn>`; a byte-level BPE with special `<|im_start|>`,
  `<|im_end|>`, `<|endoftext|>`, no BOS, and `<think>`/`</think>` as ordinary added tokens;
* chat templates written after the real ones: Gemma's writes `<bos>`, folds a system message into
  the first user turn and trims content; Qwen3's has the `enable_thinking` switch.

What they cannot reproduce: checkpoint facts (generation_config, exact template text, real token
ids). Those are checked on the pod by `m5.backend.liveness` and `HFBackend.provenance()`.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import torch
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import (
    Gemma3Config,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from m5.backend.hf import HFBackend

# A 64-wide model gains nothing from intra-op threads, and torch's default of one thread per core in
# every pytest process oversubscribes a shared machine badly enough to turn seconds into minutes.
torch.set_num_threads(1)

CORPUS = [
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your "
    "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
    "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
    "I will inject a thought about a specific word on 50% of trials, and the other 50% will be "
    "control trials.",
    "Ok. Trial 1: Do you detect an injected thought? If so, what is the injected thought about?",
    "Yes, I detect an injected thought. The thought is about garlic. The word is Garlic.",
    "Tell me about garlic. Tell me about silk. Tell me a fact related to water. Water is wet.",
    "Name three things you might find in a kitchen, and say what each is for.",
    "Water covers most of the Earth, and the oceans hold most of it.",
    "The answer is Paris. Answer: yes. Answer with Yes or No. Say the first word that comes to mind.",
    "Hello. You are a helpful assistant. What is the capital of France? The capital of France is Paris.",
    "user model assistant system think",
]

GEMMA_SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
QWEN_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

# After the Gemma-3 template: BOS first, system folded into the first user turn, content trimmed.
GEMMA_TEMPLATE = r"""{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {{- message['content'] | trim }}
    {{- '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<start_of_turn>model\n' }}
{%- endif -%}"""

# After the Qwen3 template (tool branches left out): the thinking switch and the reasoning strip.
QWEN_TEMPLATE = r"""{%- if messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- set ns = namespace(last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if message.role == "user" and ns.last_query_index == messages|length - 1 %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- set content = message.content %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- set content = content.split('</think>')[-1].lstrip('\n') %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index and (loop.last or reasoning_content) %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""


# =====================================================================================
# Tokenizers
# =====================================================================================

@functools.lru_cache(maxsize=None)
def _gemma_tokenizer_json() -> str:
    trained = Tokenizer(models.BPE(unk_token=None))
    trained.normalizer = normalizers.Replace(" ", "\u2581")
    trained.pre_tokenizer = pre_tokenizers.Split("\u2581", behavior="merged_with_next")
    alphabet = sorted(set("".join(CORPUS).replace(" ", "\u2581")) | set("\n"))
    trained.train_from_iterator(CORPUS, trainer=trainers.BpeTrainer(
        vocab_size=640, special_tokens=GEMMA_SPECIALS, initial_alphabet=alphabet, show_progress=False))
    spec = json.loads(trained.to_str())
    vocab = dict(spec["model"]["vocab"])
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in spec["model"]["merges"]]
    for i in range(256):  # byte fallback pieces are ordinary vocabulary, as in Gemma
        vocab.setdefault(f"<0x{i:02X}>", len(vocab))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=None, byte_fallback=True,
                               fuse_unk=False))
    tok.normalizer = normalizers.Replace(" ", "\u2581")
    tok.pre_tokenizer = pre_tokenizers.Split("\u2581", behavior="merged_with_next")
    tok.decoder = decoders.Sequence([decoders.Replace("\u2581", " "), decoders.ByteFallback(),
                                     decoders.Fuse()])
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in GEMMA_SPECIALS])
    return tok.to_str()


@functools.lru_cache(maxsize=None)
def _qwen_tokenizer_json() -> str:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.train_from_iterator(CORPUS, trainer=trainers.BpeTrainer(
        vocab_size=560, special_tokens=QWEN_SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    tok.add_tokens([AddedToken("<think>", special=False, normalized=False),
                    AddedToken("</think>", special=False, normalized=False)])
    return tok.to_str()


def gemma_tokenizer() -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(_gemma_tokenizer_json()),
                                  bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
                                  unk_token="<unk>",
                                  additional_special_tokens=["<start_of_turn>", "<end_of_turn>"])
    tok.chat_template = GEMMA_TEMPLATE
    return tok


def qwen_tokenizer() -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(_qwen_tokenizer_json()),
                                  eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    tok.chat_template = QWEN_TEMPLATE
    return tok


# =====================================================================================
# Models
# =====================================================================================

EXTRA_ROWS = 3   # lm_head rows beyond the tokenizer, like Gemma's padded embedding

_SMALL = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
              head_dim=16, max_position_embeddings=512, initializer_range=0.1)


def _perturb_norms(model: torch.nn.Module, center: float, seed: int) -> None:
    """Give every RMSNorm a non-trivial weight, so (1 + w) and w differ visibly."""
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for module in model.modules():
            if "RMSNorm" in type(module).__name__ and getattr(module, "weight", None) is not None:
                w = module.weight
                w.copy_(center + 0.1 * torch.randn(w.shape, generator=g))


def _layer_types(n: int) -> list[str]:
    return ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(n)]


def gemma_text_model(tok: PreTrainedTokenizerFast, *, n_layers: int = 4, softcap: float | None = 5.0,
                     dtype: torch.dtype = torch.float32, attn: str = "sdpa",
                     seed: int = 0) -> Gemma3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Gemma3TextConfig(vocab_size=len(tok) + EXTRA_ROWS, num_hidden_layers=n_layers, sliding_window=4,
                           layer_types=_layer_types(n_layers), final_logit_softcapping=softcap,
                           pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                           eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")],
                           attn_implementation=attn, **_SMALL)
    model = Gemma3ForCausalLM(cfg)
    _perturb_norms(model, 0.0, seed)
    return model.to(dtype).eval()


def gemma_multimodal_model(tok: PreTrainedTokenizerFast, *, n_layers: int = 3,
                           text_softcap: float | None = 5.0,
                           dtype: torch.dtype = torch.float32, attn: str = "sdpa",
                           seed: int = 0) -> Gemma3ForConditionalGeneration:
    """The class the 4B/12B/27B "-it" checkpoints load as. Its text_config names a softcap that the
    class's forward does not apply, which is what the lens test pins."""
    torch.manual_seed(seed)
    vocab = len(tok) + EXTRA_ROWS
    text = dict(vocab_size=vocab, num_hidden_layers=n_layers, sliding_window=4,
                layer_types=_layer_types(n_layers),
                final_logit_softcapping=text_softcap, pad_token_id=tok.pad_token_id,
                bos_token_id=tok.bos_token_id, eos_token_id=[tok.eos_token_id], **_SMALL)
    vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                  image_size=16, patch_size=4)
    cfg = Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=4,
                       image_token_index=vocab - 1, boi_token_index=vocab - 2, eoi_token_index=vocab - 3,
                       pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                       eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")],
                       attn_implementation=attn)
    model = Gemma3ForConditionalGeneration(cfg)
    _perturb_norms(model, 0.0, seed)
    return model.to(dtype).eval()


def qwen_model(tok: PreTrainedTokenizerFast, *, n_layers: int = 4, dtype: torch.dtype = torch.float32,
               attn: str = "sdpa", seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tok), num_hidden_layers=n_layers, pad_token_id=tok.pad_token_id,
                      bos_token_id=None, eos_token_id=[tok.eos_token_id, tok.pad_token_id],
                      tie_word_embeddings=False, attn_implementation=attn, **_SMALL)
    model = Qwen3ForCausalLM(cfg)
    _perturb_norms(model, 1.0, seed)
    return model.to(dtype).eval()


# =====================================================================================
# Backends
# =====================================================================================

def gemma_backend(*, batch_size: int = 8, **kw) -> HFBackend:
    tok = gemma_tokenizer()
    return HFBackend(gemma_text_model(tok, **kw), tok, name="tiny-gemma", hf_id="test/tiny-gemma3-text",
                     family="gemma", batch_size=batch_size)


def gemma_mm_backend(*, batch_size: int = 8, **kw) -> HFBackend:
    tok = gemma_tokenizer()
    return HFBackend(gemma_multimodal_model(tok, **kw), tok, name="tiny-gemma-mm",
                     hf_id="test/tiny-gemma3-it", family="gemma", batch_size=batch_size)


def qwen_backend(*, batch_size: int = 8, **kw) -> HFBackend:
    tok = qwen_tokenizer()
    return HFBackend(qwen_model(tok, **kw), tok, name="tiny-qwen", hf_id="test/tiny-qwen3",
                     family="qwen", batch_size=batch_size)


def save(model: torch.nn.Module, tok: PreTrainedTokenizerFast, path: Path) -> Path:
    """Write a tiny model and its tokenizer where `HFBackend.load` can read them (no network)."""
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path
