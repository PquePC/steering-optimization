"""The Hugging Face backend on tiny random Gemma-3 and Qwen3 models: every Backend contract, every
guard tripped, and the liveness checks shown to pass on live hooks and to fail on broken ones.

The models are the real transformers classes built in-process from small configs
(`m5.tests.tiny_models`), so these tests exercise the attribute paths, the bare-tensor layer
outputs, sliding-window masks under left padding, the KV-cached decode loop and the chat templates
without a download. fp32 on CPU: where two computations should agree, they agree to ~1e-6.
"""

from __future__ import annotations

import math
import types
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from m5 import config  # noqa: E402
from m5.backend import liveness  # noqa: E402
from m5.backend.base import Backend, GenRequest, ScoreRequest, locate  # noqa: E402
from m5.backend.hf import HFBackend, _Pass  # noqa: E402
from m5.intervention import Intervention, LayerTerm, Positions  # noqa: E402
from m5.tests import tiny_models as tm  # noqa: E402

Q1 = "Tell me a fact related to water."
Q2 = "Name three things you might find in a kitchen, and say what each is for."
Q3 = "Hello."
BUILDERS = {"gemma": tm.gemma_backend, "gemma_mm": tm.gemma_mm_backend, "qwen": tm.qwen_backend}


# ===================================================================================== helpers

@pytest.fixture
def gemma():
    return tm.gemma_backend()


@pytest.fixture
def qwen():
    return tm.qwen_backend()


@pytest.fixture(params=sorted(BUILDERS))
def be(request):
    return BUILDERS[request.param]()


def texts(be, qs=(Q1, Q2, Q3)):
    return [be.render([{"role": "user", "content": q}]) for q in qs]


def starts(be, ts, qs=(Q1, Q2, Q3)):
    return [locate(be, t, q) for t, q in zip(ts, qs)]


def register(be, layers, vid="v", seed=0, scale=1.0):
    """A random direction per layer, scaled to `scale` x the median residual norm there."""
    st = be.residual_stats(texts(be), sorted(layers))
    rng = np.random.default_rng(seed)
    values = {}
    for L in layers:
        d = rng.standard_normal(be.info.d_model)
        values[L] = (d / np.linalg.norm(d) * scale * float(np.median(st[L].norms))).astype(np.float32)
    be.register_vector(vid, values)
    return values


def add(layer, alpha, pos, vid="v", op="add", clamp=None):
    return Intervention(vid, op, (LayerTerm(layer, alpha),), pos, clamp_value=clamp)


def all_ids(be):
    return list(range(be.info.vocab_size))


def every_position(be, ts):
    return [list(range(len(be.tokenize(t)))) for t in ts]


def hook_count(be):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in be.model.modules())


# ===================================================================================== construction

def test_backend_satisfies_the_protocol_and_reports_the_model(be):
    assert isinstance(be, Backend)
    info = be.info
    assert info.n_layers == len(be._layers) == be.model.config.get_text_config().num_hidden_layers
    assert info.d_model == 64 and info.dtype == "float32"
    assert info.vocab_size == be._lm_head.weight.shape[0] >= len(be.tokenizer)
    eot = "<end_of_turn>" if info.family == "gemma" else "<|im_end|>"
    assert be.tokenizer.convert_tokens_to_ids(eot) in info.eos_token_ids
    assert info.bos_token_id == (be.tokenizer.bos_token_id if info.family == "gemma" else None)
    assert len(info.chat_template_hash) == 16
    assert set(info.to_json()) >= {"name", "hf_id", "revision", "n_layers", "eos_token_ids"}


def test_decoder_stack_is_found_at_each_known_path():
    assert tm.gemma_backend()._layers_path == "model.layers"
    mm = tm.gemma_mm_backend()
    assert mm._layers_path == "model.language_model.layers"   # the -it checkpoints' class
    assert mm._norm_path == "model.language_model.norm"
    assert tm.qwen_backend()._layers_path == "model.layers"


def test_construction_guards_trip():
    tok = tm.gemma_tokenizer()
    model = tm.gemma_text_model(tok)
    model.model.blocks = model.model.layers
    del model.model.layers
    with pytest.raises(RuntimeError, match="tried model.language_model.layers"):
        HFBackend(model, tok, name="x", hf_id="x", family="gemma")

    model = tm.gemma_text_model(tok)
    model.config.num_hidden_layers = 5                     # bug 4: a different stack than declared
    with pytest.raises(RuntimeError, match="bug 4"):
        HFBackend(model, tok, name="x", hf_id="x", family="gemma")

    model = tm.gemma_text_model(tok)
    model.lm_head.weight = torch.nn.Parameter(torch.empty(model.lm_head.weight.shape, device="meta"))
    with pytest.raises(RuntimeError, match="one device"):
        HFBackend(model, tok, name="x", hf_id="x", family="gemma")

    with pytest.raises(TypeError, match="fast tokenizer"):
        HFBackend(tm.gemma_text_model(tok), types.SimpleNamespace(is_fast=False), name="x", hf_id="x",
                  family="gemma")
    no_template = tm.gemma_tokenizer()
    no_template.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        HFBackend(tm.gemma_text_model(no_template), no_template, name="x", hf_id="x", family="gemma")
    with pytest.raises(ValueError, match="batch_size"):
        HFBackend(tm.gemma_text_model(tok), tok, name="x", hf_id="x", family="gemma", batch_size=0)
    bad_eot = tm.gemma_tokenizer()
    with pytest.raises(ValueError, match="must stop at"):
        HFBackend(tm.gemma_text_model(bad_eot), bad_eot, name="x", hf_id="x", family="qwen")


def test_template_guards_trip():
    """Two BOS from the template (bug 9); a BOS the tokenizer adds that the template omits; a
    thinking switch that leaves the reasoning block open (the old Qwen arm)."""
    from tokenizers import processors

    tok = tm.gemma_tokenizer()
    tok.chat_template = "{{ bos_token }}" + tm.GEMMA_TEMPLATE
    with pytest.raises(RuntimeError, match="two BOS"):
        HFBackend(tm.gemma_text_model(tok), tok, name="x", hf_id="x", family="gemma")

    tok = tm.gemma_tokenizer()
    tok._tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", tok.bos_token_id)])
    tok.chat_template = tm.GEMMA_TEMPLATE.replace("{{ bos_token }}", "", 1)
    with pytest.raises(RuntimeError, match="adds a BOS to raw text"):
        HFBackend(tm.gemma_text_model(tok), tok, name="x", hf_id="x", family="gemma")

    tok = tm.qwen_tokenizer()
    tok.chat_template = tm.QWEN_TEMPLATE.replace(r"'<think>\n\n</think>\n\n'", r"'<think>\n'")
    with pytest.raises(RuntimeError, match="open <think>"):
        HFBackend(tm.qwen_model(tok), tok, name="x", hf_id="x", family="qwen")


# ===================================================================================== text

def test_gemma_render_drops_an_empty_system_turn_and_writes_one_bos(gemma):
    user = [{"role": "user", "content": Q1}]
    with_empty = gemma.render([{"role": "system", "content": ""}] + user)
    assert with_empty == gemma.render(user)                 # upstream's filter, empty case
    assert with_empty.startswith("<bos><start_of_turn>user\n")
    assert with_empty.endswith("<end_of_turn>\n<start_of_turn>model\n")
    folded = gemma.render([{"role": "system", "content": "Be brief."}] + user)
    assert "<start_of_turn>user\nBe brief.\n\n" + Q1 in folded   # non-empty: folded, not lost
    ids = gemma.tokenize(with_empty)
    assert ids[0] == gemma.info.bos_token_id and ids[1] != gemma.info.bos_token_id
    convo = [{"role": "system", "content": ""}, {"role": "user", "content": "Framing."},
             {"role": "assistant", "content": "Ok."}, {"role": "user", "content": "Trial 1: question?"}]
    text = gemma.render(convo) + "Yes, I detect an injected thought. The thought is about"
    assert text.count("<bos>") == 1 and "<start_of_turn>model\nOk.<end_of_turn>" in text
    assert gemma.render(user, add_generation_prompt=False).endswith("<end_of_turn>\n")


def test_qwen_render_turns_thinking_off_and_keeps_the_system_turn(qwen):
    text = qwen.render([{"role": "system", "content": ""}, {"role": "user", "content": Q1}])
    assert text.startswith("<|im_start|>system\n<|im_end|>\n")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert qwen.info.bos_token_id is None


def test_render_refuses_malformed_messages(gemma):
    for bad in ([{"role": "tool", "content": "x"}], [{"role": "user", "content": 3}],
                [{"content": "x"}], [], [{"role": "system", "content": ""}]):
        with pytest.raises((ValueError, TypeError)):
            gemma.render(bad)


def test_tokenize_adds_nothing_and_offsets_and_decode_round_trip(be):
    for t in texts(be) + ["plain text without a template"]:
        ids = be.tokenize(t)
        assert ids == list(be.tokenizer(t, add_special_tokens=False)["input_ids"])
        offs = be.offsets(t)
        assert len(offs) == len(ids)
        assert all(0 <= s <= e <= len(t) for s, e in offs)
        assert be.decode(ids) == t
    t = texts(be)[0]
    i = locate(be, t, Q1)
    s, e = be.offsets(t)[i]
    assert s <= t.index(Q1) < e and be.decode(be.tokenize(t)[i:]).startswith(Q1[:2])


def test_forward_inputs_must_begin_with_exactly_one_bos(gemma):
    register(gemma, [1])
    good = texts(gemma)[0]
    raw = "Water is wet."
    for text in (raw, "<bos>" + good):
        with pytest.raises(ValueError, match="BOS"):
            gemma.next_token_logprobs([text], [None], [1])
        with pytest.raises(ValueError, match="BOS"):
            gemma.generate([GenRequest(text, None, 2, 1.0, top_p=1.0, top_k=0, seed=0)])
        with pytest.raises(ValueError, match="BOS"):
            gemma.residual_stats([text], [1])
        with pytest.raises(ValueError, match="BOS"):
            gemma.score([ScoreRequest(text, " more", None)])
    with_bos = gemma.decode([gemma.info.bos_token_id]) + raw   # raw text carries the BOS explicitly
    assert gemma.next_token_logprobs([with_bos], [None], [1]).shape == (1, 1)


def test_word_variants(be):
    variants = be.word_variants("garlic")
    assert [be.decode(v) for v in variants] == [" garlic", "garlic", " Garlic", "Garlic"]
    assert len(set(variants)) == 4
    assert [be.decode(v) for v in be.word_variants("Paris")] == [" Paris", "Paris"]   # deduplicated
    assert be.decode(be.word_variants("dNA")[3]) == "DNA"
    for bad in ("", " garlic", "garlic "):
        with pytest.raises(ValueError):
            be.word_variants(bad)


def test_word_variants_refuse_a_variant_that_does_not_round_trip(gemma):
    gemma.decode = lambda ids: "garbled"
    with pytest.raises(ValueError, match="cannot isolate"):
        gemma.word_variants("garlic")


# ===================================================================================== vectors

def test_register_vector_guards(gemma):
    d, n = gemma.info.d_model, gemma.info.n_layers
    good = np.ones(d, dtype=np.float32)
    for bad in ({}, {0: np.ones(d + 1)}, {0: np.full(d, np.nan)}, {0: np.zeros(d)}, {n: good}, {-1: good},
                {0: np.ones((2, d))}):
        with pytest.raises(ValueError):
            gemma.register_vector("w", bad)
    gemma.register_vector("w", {0: good, 2: 2 * good})
    gemma.register_vector("w", {0: good, 2: 2 * good})                       # identical: a no-op
    with pytest.raises(ValueError, match="bug 23"):
        gemma.register_vector("w", {0: 3 * good, 2: 2 * good})
    assert gemma.has_vector("w") and not gemma.has_vector("nope")
    np.testing.assert_array_equal(gemma.vector_values("w", 2), 2 * good)
    gemma.unregister_vector("w")
    gemma.register_vector("w", {0: 3 * good})
    t = texts(gemma)[:1]
    with pytest.raises(KeyError, match="not registered"):
        gemma.next_token_logprobs(t, [add(0, 1.0, Positions("all"), vid="nope")], [1])
    with pytest.raises(ValueError, match="no value at layer 1"):
        gemma.next_token_logprobs(t, [add(1, 1.0, Positions("all"), vid="w")], [1])
    with pytest.raises(ValueError, match="beyond the model"):
        gemma.next_token_logprobs(t, [add(n, 1.0, Positions("all"), vid="w")], [1])
    with pytest.raises(TypeError):
        gemma.next_token_logprobs(t, ["add"], [1])
    with pytest.raises(KeyError):
        gemma.unregister_vector("nope")


# ===================================================================================== the edit

def _manual(op, h, v, alpha, c=None):
    h = h.astype(np.float64)
    u = v / np.linalg.norm(v)
    proj = h @ u
    if op == "add":
        return h + alpha * v
    if op == "sub":
        return h - alpha * v
    if op == "project":
        return h - alpha * proj[:, None] * u
    return h + (c - proj)[:, None] * u


@pytest.mark.parametrize("op", ["add", "sub", "project", "clamp"])
@pytest.mark.parametrize("kind", ["gemma", "qwen"])
def test_operations_match_the_documented_math(kind, op):
    be = BUILDERS[kind]()
    L = 1
    v = register(be, [L])[L].astype(np.float64)
    ts = texts(be)
    st = starts(be, ts)
    pos = every_position(be, ts)
    c = 3.0 if op == "clamp" else None
    alpha = 0.0 if op == "clamp" else 0.7
    ivs = [add(L, alpha, Positions("from", s), op=op, clamp=c) for s in st]
    hu = be.hidden(ts, [None] * 3, [L], pos)
    hs = be.hidden(ts, ivs, [L], pos)
    for i, s in enumerate(st):
        np.testing.assert_array_equal(hs[i][:s], hu[i][:s])                 # untouched before start
        want = _manual(op, hu[i][s:, 0, :], v, alpha, c)
        np.testing.assert_allclose(hs[i][s:, 0, :], want, rtol=1e-5, atol=1e-4)
        if op == "clamp":
            np.testing.assert_allclose(hs[i][s:, 0, :] @ (v / np.linalg.norm(v)), c, atol=1e-4)
        if op == "project":
            assert not np.allclose(hs[i][s:], hu[i][s:])


def test_multi_layer_terms_compose_in_depth_order(gemma):
    v = register(gemma, [0, 2])
    ts = texts(gemma)
    st = starts(gemma, ts)
    pos = every_position(gemma, ts)
    both = [Intervention("v", "add", (LayerTerm(2, 0.5), LayerTerm(0, 0.8)), Positions("from", s))
            for s in st]
    first = [add(0, 0.8, Positions("from", s)) for s in st]
    h_both = gemma.hidden(ts, both, [0, 2], pos)
    h_first = gemma.hidden(ts, first, [0, 2], pos)
    hu = gemma.hidden(ts, [None] * 3, [0], pos)
    for i, s in enumerate(st):
        np.testing.assert_allclose(h_both[i][s:, 0], hu[i][s:, 0] + 0.8 * v[0], rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(h_both[i][s:, 1], h_first[i][s:, 1] + 0.5 * v[2], rtol=1e-5, atol=1e-4)


def test_bfloat16_edit_is_computed_in_fp32_and_cast_once():
    be = tm.gemma_backend(dtype=torch.bfloat16)
    L = 1
    v = register(be, [L])[L]
    ts = texts(be)[:1]
    s = starts(be, ts)[0]
    pos = every_position(be, ts)
    hu = be.hidden(ts, [None], [L], pos)[0][:, 0]
    hs = be.hidden(ts, [add(L, 0.37, Positions("from", s))], [L], pos)[0][:, 0]
    want = (torch.tensor(hu[s:]) + 0.37 * torch.tensor(v)).to(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(hs[s:], want)


# ===================================================================================== positions

def test_span_steers_the_past_only(be):
    """At the steered layer only the span moves; at a later layer the positions after it move
    (they attend to steered keys and values) while the positions before it cannot."""
    L, L2 = 0, be.info.n_layers - 1
    register(be, [L])
    ts = texts(be)[:2]
    st = starts(be, ts, (Q1, Q2))
    pos = every_position(be, ts)
    ivs = [add(L, 1.0, Positions("span", s, s + 3)) for s in st]
    hu = be.hidden(ts, [None, None], [L, L2], pos)
    hs = be.hidden(ts, ivs, [L, L2], pos)
    for i, s in enumerate(st):
        e = s + 3
        np.testing.assert_array_equal(hs[i][:s, 0], hu[i][:s, 0])
        np.testing.assert_array_equal(hs[i][e:, 0], hu[i][e:, 0])          # not steered at L
        assert np.all(np.abs(hs[i][s:e, 0] - hu[i][s:e, 0]).max(-1) > 1e-3)
        np.testing.assert_array_equal(hs[i][:s, 1], hu[i][:s, 1])          # causal
        assert np.all(np.abs(hs[i][e:, 1] - hu[i][e:, 1]).max(-1) > 1e-4)  # attends to the span


def _changed(a, b, tol=1e-4):
    """Positions where two [positions, layers, d] reads differ by more than fp32 noise (relative to
    the residual's own scale: a padded row and an unpadded one differ by ~1e-7 of it, a steered
    position by the order of the residual itself)."""
    scale = np.maximum(np.abs(b).max(axis=(-1, -2)), 1.0)
    return [int(p) for p in np.nonzero(np.abs(a - b).max(axis=(-1, -2)) > tol * scale)[0]]


def test_policies_steer_exactly_their_positions(gemma):
    L = 1
    register(gemma, [L])
    t = texts(gemma)[:1]
    n = len(gemma.tokenize(t[0]))
    s = starts(gemma, t)[0]
    pos = every_position(gemma, t)
    hu = gemma.hidden(t, [None], [L], pos)[0]
    for positions, want in ((Positions("all"), list(range(1, n))),
                            (Positions("from", s), list(range(s, n))),
                            (Positions("span", 2, 5), [2, 3, 4])):
        hs = gemma.hidden(t, [add(L, 1.0, positions)], [L], pos)[0]
        assert _changed(hs, hu) == want


def test_left_padding_maps_each_row_onto_its_own_positions(be):
    """Prompts of three lengths in one batch, each steered from its own start: every row equals the
    same prompt run alone, and the positions that moved are exactly start..n-1. Upstream's batch
    path steered shorter prompts one token early (bug 25b); `_place` shifted the same way is
    caught by this comparison."""
    L = 1
    register(be, [L])
    ts = texts(be)
    st = starts(be, ts)
    pos = every_position(be, ts)
    assert len({len(p) for p in pos}) == 3
    ivs = [add(L, 1.0, Positions("from", s)) for s in st]
    batch = be.hidden(ts, ivs, [L], pos)
    for i in range(3):
        alone = be.hidden([ts[i]], [ivs[i]], [L], [pos[i]])[0]
        np.testing.assert_allclose(batch[i], alone, rtol=1e-5, atol=1e-5)
        unsteered = be.hidden([ts[i]], [None], [L], [pos[i]])[0]
        assert _changed(batch[i], unsteered) == list(range(st[i], len(pos[i])))

    def one_early(mask, pad, width):
        return [False] * max(pad - 1, 0) + list(mask) + [False] * (1 if pad else 0)

    be._place = one_early
    shifted = be.hidden(ts, ivs, [L], pos)
    assert any(not np.allclose(shifted[i], batch[i], atol=1e-4) for i in range(3))


def test_positions_that_cannot_be_mapped_raise(gemma):
    register(gemma, [1])
    t = texts(gemma)[:1]
    n = len(gemma.tokenize(t[0]))
    with pytest.raises(ValueError, match="out-of-range start"):
        gemma.next_token_logprobs(t, [add(1, 1.0, Positions("from", n))], [1])
    with pytest.raises(ValueError, match="out-of-range start"):
        gemma.generate([GenRequest(t[0], add(1, 1.0, Positions("from", n)), 2, 1.0, top_p=1.0, top_k=0, seed=0)])
    with pytest.raises(ValueError, match="beyond the prompt"):
        gemma.hidden(t, [add(1, 1.0, Positions("span", 1, n + 1))], [1], [[0]])
    gen = add(1, 1.0, Positions("generated"))
    for call in (lambda: gemma.next_token_logprobs(t, [gen], [1]),
                 lambda: gemma.hidden(t, [gen], [1], [[0]]),
                 lambda: gemma.lens(t, [gen], [1], -1, [1]),
                 lambda: gemma.kl_from_unsteered(t, [gen], start_tokens=[1])):
        with pytest.raises(ValueError, match="steer no token"):
            call()
    bos_only = gemma.decode([gemma.info.bos_token_id])
    with pytest.raises(ValueError, match="steer no token"):
        gemma.next_token_logprobs([bos_only], [add(1, 1.0, Positions("all"))], [1])


# ===================================================================================== batching and leaks

def _mixed_rows(be, ts, st):
    c = float(np.median(be.residual_stats(ts, [1]).get(1).norms))
    return [
        (ts[0], None),
        (ts[1], add(1, 1.0, Positions("from", st[1]))),
        (ts[2], add(2, 0.8, Positions("all"), vid="w")),
        (ts[0], add(1, 0.0, Positions("from", st[0]))),
        (ts[1], add(1, 0.6, Positions("span", st[1], st[1] + 3), op="sub")),
        (ts[2], add(1, 1.0, Positions("from", st[2]), op="project")),
        (ts[0], add(1, 0.0, Positions("from", st[0]), op="clamp", clamp=0.5 * c)),
        (ts[1], Intervention("v", "add", (LayerTerm(1, 0.4), LayerTerm(2, 0.4)), Positions("from", st[1]))),
    ]


def test_mixed_batch_rows_equal_the_same_rows_alone(be):
    """Contract 2: each row's vector, operation, layers, dose and positions are its own."""
    register(be, [1, 2])
    register(be, [2], vid="w", seed=1)
    ts = texts(be)
    st = starts(be, ts)
    rows = _mixed_rows(be, ts, st)
    prompts, ivs = [p for p, _ in rows], [iv for _, iv in rows]
    ids = all_ids(be)
    batch = be.next_token_logprobs(prompts, ivs, ids)
    pos = [list(range(len(be.tokenize(p)))) for p in prompts]
    hid = be.hidden(prompts, ivs, [1, 2], pos)
    scores = be.score([ScoreRequest(p, " Water is wet.", iv) for p, iv in rows])
    for i, (p, iv) in enumerate(rows):
        np.testing.assert_allclose(batch[i], be.next_token_logprobs([p], [iv], ids)[0], atol=1e-5)
        np.testing.assert_allclose(hid[i], be.hidden([p], [iv], [1, 2], [pos[i]])[0], rtol=1e-5, atol=1e-5)
        alone = be.score([ScoreRequest(p, " Water is wet.", iv)])[0]
        assert alone.token_ids == scores[i].token_ids
        np.testing.assert_allclose(alone.logprobs, scores[i].logprobs, atol=1e-5)
    unsteered = be.next_token_logprobs([ts[1]], [None], ids)[0]
    assert np.abs(batch[1] - unsteered).max() > 1e-3                         # the steering is real


def test_chunking_does_not_change_results():
    small, wide = tm.gemma_backend(batch_size=2), tm.gemma_backend(batch_size=8)
    for be in (small, wide):
        register(be, [1])
    ts = texts(small)
    st = starts(small, ts)
    ivs = [add(1, 1.0, Positions("from", s)) for s in st]
    np.testing.assert_allclose(small.next_token_logprobs(ts, ivs, all_ids(small)),
                               wide.next_token_logprobs(ts, ivs, all_ids(wide)), atol=1e-5)
    reqs = [GenRequest(t, iv, 6, 1.0, top_k=0, seed=10 + i, top_p=1.0) for i, (t, iv) in enumerate(zip(ts, ivs))]
    assert [g.token_ids for g in small.generate(reqs)] == [g.token_ids for g in wide.generate(reqs)]


def test_nothing_leaks_from_a_steered_call_into_the_next(gemma):
    """Contract 3. A steered call between two unsteered ones changes nothing, no hook outlives a
    call, and an exception inside a forward pass still removes every hook."""
    register(gemma, [1])
    ts = texts(gemma)
    st = starts(gemma, ts)
    ivs = [add(1, 1.0, Positions("from", s)) for s in st]
    ids = all_ids(gemma)
    hooks = hook_count(gemma)
    fresh = gemma.next_token_logprobs(ts, [None] * 3, ids)
    reqs = [GenRequest(t, None, 6, 1.0, top_k=0, seed=i, top_p=1.0) for i, t in enumerate(ts)]
    gen_fresh = [g.token_ids for g in gemma.generate(reqs)]
    gemma.next_token_logprobs(ts, ivs, ids)
    gemma.generate([replace(r, intervention=iv) for r, iv in zip(reqs, ivs)])
    gemma.kl_from_unsteered(ts, ivs, start_tokens=st)
    np.testing.assert_array_equal(gemma.next_token_logprobs(ts, [None] * 3, ids), fresh)
    assert [g.token_ids for g in gemma.generate(reqs)] == gen_fresh
    assert hook_count(gemma) == hooks and gemma._handles == [] and gemma._pass is None

    def boom(h, e):
        raise RuntimeError("edit failed")

    gemma._edit = boom
    with pytest.raises(RuntimeError, match="edit failed"):
        gemma.generate([replace(reqs[0], intervention=ivs[0])])
    assert hook_count(gemma) == hooks and gemma._handles == [] and gemma._pass is None
    np.testing.assert_array_equal(gemma.next_token_logprobs(ts, [None] * 3, ids), fresh)


# ===================================================================================== generation

def test_generation_is_reproducible_per_seed_whatever_the_batch(be):
    """Contract 4: a row's sample depends on its request and seed, not on its batch-mates."""
    register(be, [1])
    ts = texts(be)
    st = starts(be, ts)
    a = GenRequest(ts[0], add(1, 1.0, Positions("from", st[0])), 10, 1.0, top_p=0.9, top_k=40, seed=7)
    others = [GenRequest(ts[1], None, 4, 1.0, seed=3, top_p=1.0, top_k=0),
              GenRequest(ts[2], add(1, 0.5, Positions("all")), 12, 0.7, seed=9, top_p=1.0, top_k=0)]
    alone = be.generate([a])[0]
    mixed = be.generate([others[0], a, others[1]])
    assert mixed[1] == alone and be.generate([a])[0] == alone
    assert len(alone.token_ids) == 10 or alone.finish == "eos"
    draws = {be.generate([replace(a, seed=s)])[0].token_ids for s in range(4)}
    assert len(draws) > 1                                                     # the seed is used


def test_greedy_generation_takes_the_argmax_of_each_step(gemma):
    register(gemma, [1])
    ts = texts(gemma)[:2]
    st = starts(gemma, ts, (Q1, Q2))
    ivs = [add(1, 1.0, Positions("from", s)) for s in st]
    gens = gemma.generate([GenRequest(t, iv, 8, 0.0, seed=s, top_p=1.0, top_k=0) for s, (t, iv) in enumerate(zip(ts, ivs))])
    traced = gemma.forced_logprobs(ts, ivs, [list(g.token_ids) for g in gens], path="decode")
    for g, tr in zip(gens, traced):
        assert list(tr.argmax) == list(g.token_ids)


def test_stop_tokens_and_length(gemma, monkeypatch):
    t = texts(gemma)[0]
    eot = gemma.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    calls = {"n": 0}
    real_choose = HFBackend._choose

    def stop_on_third(self, logits, rows, gens, done):
        calls["n"] += 1
        chosen = real_choose(self, logits, rows, gens, done)
        return [eot if calls["n"] == 3 else (c if c not in self._stop_set else 7) for c in chosen]

    monkeypatch.setattr(HFBackend, "_choose", stop_on_third)
    res = gemma.generate([GenRequest(t, None, 10, 1.0, seed=1, top_p=1.0, top_k=0)])[0]
    assert res.finish == "eos" and len(res.token_ids) == 3 and res.token_ids[-1] == eot
    assert "<end_of_turn>" not in res.text
    calls["n"] = 100
    res = gemma.generate([GenRequest(t, None, 5, 1.0, seed=1, top_p=1.0, top_k=0)])[0]
    assert res.finish == "length" and len(res.token_ids) == 5


def test_sampling_filter_matches_hf_warpers():
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(5, 60, generator=g) * 3
    logits[4] = torch.round(logits[4])                                      # ties at the top-k edge
    temps = torch.tensor([1.0, 0.7, 1.3, 1.0, 1.0])
    ks = torch.tensor([0, 10, 0, 25, 7])
    ps = torch.tensor([1.0, 1.0, 0.8, 0.6, 1.0])
    order, probs, n_keep = HFBackend._filter(logits, temps, ks, ps)
    mine = torch.zeros_like(probs).scatter(1, order, probs)
    for i in range(5):
        x = logits[i:i + 1].clone()
        x = TemperatureLogitsWarper(float(temps[i]))(None, x)
        if ks[i] > 0:
            x = TopKLogitsWarper(int(ks[i]))(None, x)
        if ps[i] < 1:
            x = TopPLogitsWarper(float(ps[i]))(None, x)
        np.testing.assert_allclose(mine[i].numpy(), torch.softmax(x.double(), -1)[0].numpy(), atol=1e-6)
        assert int(n_keep[i]) == int((mine[i] > 0).sum())
    assert int(n_keep[4]) >= 7                                               # ties kept, as HF does


def test_sampler_draws_follow_the_distribution(gemma):
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    row = types.SimpleNamespace(temperature=1.0, top_k=0, top_p=1.0)
    gen = [torch.Generator().manual_seed(3)]
    counts = np.zeros(4)
    for _ in range(3000):
        counts[gemma._choose(logits, [row], gen, [False])[0]] += 1
    np.testing.assert_allclose(counts / 3000, [0.5, 0.3, 0.15, 0.05], atol=0.03)


def test_generated_tokens_are_steered_in_decoding(be):
    """Teacher forcing through the generation loop (KV cache, decode-phase plan) equals one prefill
    pass for every policy; under 'generated' the first token is unsteered and the rest move. A
    span's effect on the continuation is not required: it reaches later tokens only through
    attention, and a sliding-window layer above the steered one can put it out of reach (the
    3-layer multimodal model has exactly that)."""
    register(be, [1])
    ts = texts(be)[:2]
    st = starts(be, ts, (Q1, Q2))
    cont = [be.tokenize(" Water covers most of the Earth.")[:5]] * 2
    un = be.forced_logprobs(ts, [None, None], cont, path="decode")
    for positions in ([Positions("from", s) for s in st], [Positions("span", s, s + 3) for s in st],
                      [Positions("generated")] * 2, [Positions("all")] * 2):
        ivs = [add(1, 1.0, p) for p in positions]
        pre = be.forced_logprobs(ts, ivs, cont, path="prefill")
        dec = be.forced_logprobs(ts, ivs, cont, path="decode")
        for a, b, u in zip(pre, dec, un):
            np.testing.assert_allclose(a.logprobs, b.logprobs, atol=1e-4)
            np.testing.assert_array_equal(a.argmax, b.argmax)
            if positions[0].policy == "generated":
                assert a.logprobs[0] == pytest.approx(u.logprobs[0], abs=1e-6)
                assert np.abs(b.logprobs[1:] - u.logprobs[1:]).max() > 1e-3
            elif positions[0].policy in ("from", "all"):
                assert np.abs(b.logprobs - u.logprobs).max() > 1e-3
    with pytest.raises(ValueError, match="path"):
        be.forced_logprobs(ts, [None, None], cont, path="sideways")


# ===================================================================================== teacher forcing

def test_score_tokenises_the_continuation_in_context(be):
    register(be, [1])
    t = texts(be)[0]
    s = starts(be, [t])[0]
    iv = add(1, 1.0, Positions("from", s))
    k = len(be.tokenize(t))
    for cont in (" garlic.", "Water is wet."):
        res = be.score([ScoreRequest(t, cont, None), ScoreRequest(t, cont, iv)])
        want = tuple(be.tokenize(t + cont)[k:])
        assert res[0].token_ids == want == res[1].token_ids
        forced = be.forced_logprobs([t, t], [None, iv], [list(want)] * 2, path="prefill")
        for r, f in zip(res, forced):
            np.testing.assert_allclose(r.logprobs, f.logprobs, atol=1e-6)
            assert r.total == pytest.approx(sum(r.logprobs))
            assert all(0.0 <= h <= math.log(be.info.vocab_size) + 1e-6 for h in r.entropies)
        first = be.next_token_logprobs([t, t], [None, iv], [want[0]])[:, 0]
        np.testing.assert_allclose([r.logprobs[0] for r in res], first, atol=1e-5)


def test_score_guards_trip(gemma):
    t = texts(gemma)[0]
    with pytest.raises(ValueError, match="empty continuation"):
        gemma.score([ScoreRequest(t, "", None)])
    with pytest.raises(ValueError, match="empty prompt"):
        gemma.score([ScoreRequest("", t, None)])
    assert len(gemma.tokenize("Tell")) == 1
    with pytest.raises(ValueError, match="spans the prompt/continuation join"):
        gemma.score([ScoreRequest(t + "Te", "ll me more", None)])
    real = gemma.tokenize
    gemma.tokenize = lambda text: real(text)[:-1] if text == t else real(text)
    with pytest.raises(ValueError, match="tokenises differently"):
        gemma.score([ScoreRequest(t, " more", None)])


def test_candidate_logprobs_read_the_distributions_score_reads(be):
    register(be, [1])
    ts = texts(be)[:2]
    st = starts(be, ts, (Q1, Q2))
    reqs = [ScoreRequest(ts[0], " Water is wet.", add(1, 1.0, Positions("from", st[0]))),
            ScoreRequest(ts[1], " garlic", None)]
    scores = be.score(reqs)
    cands = sorted({t for s in scores for t in s.token_ids} | {1, 2, 3})
    arrays = be.candidate_logprobs(reqs, cands)
    nxt = be.next_token_logprobs([r.prompt for r in reqs], [r.intervention for r in reqs], cands)
    for s, arr, row in zip(scores, arrays, nxt):
        assert arr.shape == (len(s.token_ids), len(cands))
        for j, tok in enumerate(s.token_ids):
            assert arr[j, cands.index(tok)] == pytest.approx(s.logprobs[j], abs=1e-6)
        np.testing.assert_allclose(arr[0], row, atol=1e-5)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        be.candidate_logprobs(reqs, [be.info.vocab_size])


# ===================================================================================== KL

def test_kl_is_zero_for_a_null_intervention_and_matches_a_manual_computation(gemma):
    register(gemma, [1])
    ts = texts(gemma)[:2]
    st = starts(gemma, ts, (Q1, Q2))
    zero = gemma.kl_from_unsteered(ts, [add(1, 0.0, Positions("from", s)) for s in st], start_tokens=st,
                                   exclude_token_ids=[3, 4])
    assert all(r.mean_kl == 0.0 and r.mean_kl_without == 0.0 for r in zero)
    ivs = [add(1, 1.0, Positions("from", s)) for s in st]
    excl = [5, 9, 40]
    res = gemma.kl_from_unsteered(ts, ivs, start_tokens=st, exclude_token_ids=excl)
    assert gemma.kl_from_unsteered(ts, ivs, start_tokens=st)[0].mean_kl_without is None
    last, ids = gemma.info.n_layers - 1, all_ids(gemma)
    keep = np.ones(len(ids), dtype=bool)
    keep[excl] = False
    for i, (t, s) in enumerate(zip(ts, st)):
        n = len(gemma.tokenize(t))
        assert res[i].n_positions == n - s and res[i].mean_kl > 0
        kls, kls_wo = [], []
        for q in range(s, n):
            lu = gemma.lens([t], [None], [last], q, ids)[0, 0]
            ls = gemma.lens([t], [ivs[i]], [last], q, ids)[0, 0]
            kls.append(max(0.0, float(np.sum(np.exp(lu) * (lu - ls)))))
            ru = lu[keep] - np.log(np.exp(lu[keep]).sum())
            rs = ls[keep] - np.log(np.exp(ls[keep]).sum())
            kls_wo.append(max(0.0, float(np.sum(np.exp(ru) * (ru - rs)))))
        assert res[i].mean_kl == pytest.approx(np.mean(kls), rel=1e-4, abs=1e-6)
        assert res[i].mean_kl_without == pytest.approx(np.mean(kls_wo), rel=1e-4, abs=1e-6)


def test_kl_guards_trip(gemma):
    register(gemma, [1])
    t = texts(gemma)[:1]
    n = len(gemma.tokenize(t[0]))
    iv = add(1, 1.0, Positions("all"))
    with pytest.raises(ValueError, match="no intervention"):
        gemma.kl_from_unsteered(t, [None], start_tokens=[1])
    for bad in (-1, n):
        with pytest.raises(ValueError, match="not a position"):
            gemma.kl_from_unsteered(t, [iv], start_tokens=[bad])
    with pytest.raises(ValueError, match="outside the vocabulary"):
        gemma.kl_from_unsteered(t, [iv], start_tokens=[1], exclude_token_ids=[gemma.info.vocab_size])
    with pytest.raises(ValueError, match="differ in length"):
        gemma.kl_from_unsteered(t, [iv], start_tokens=[1, 2])


# ===================================================================================== residual reads

def test_layer_l_is_the_output_of_block_l(be):
    """hidden(L) is HF's hidden_states[L + 1] below the last layer. At the last layer HF reports
    the post-final-norm state instead (transformers 5.x), which is why every read goes through the
    hook: hidden(n-1) is block n-1's output, and only its final norm equals HF's last entry."""
    t = texts(be)[0]
    n_layers = be.info.n_layers
    pos = every_position(be, [t])
    mine = be.hidden([t], [None], list(range(n_layers)), pos)[0]
    ids = torch.tensor([be.tokenize(t)])
    with torch.no_grad():
        hs = be.model(input_ids=ids, output_hidden_states=True).hidden_states
    for L in range(n_layers - 1):
        np.testing.assert_allclose(mine[:, L], hs[L + 1][0].numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(be.reference_hidden(t, list(range(n_layers - 1))), mine[:, :-1],
                               rtol=1e-5, atol=1e-5)
    assert not np.allclose(mine[:, -1], hs[-1][0].numpy(), atol=1e-3)
    with torch.no_grad():
        normed = be._final_norm(torch.tensor(mine[:, -1])).numpy()
    np.testing.assert_allclose(normed, hs[-1][0].numpy(), rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="no independent reference"):
        be.reference_hidden(t, [n_layers - 1])


def test_hidden_positions_and_guards(gemma):
    t = texts(gemma)[:1]
    n = len(gemma.tokenize(t[0]))
    full = gemma.hidden(t, [None], [0, 2], every_position(gemma, t))[0]
    part = gemma.hidden(t, [None], [2, 0], [[-1, 0, n - 2]])[0]
    assert part.shape == (3, 2, gemma.info.d_model) and part.dtype == np.float32
    np.testing.assert_array_equal(part[0, 0], full[n - 1, 1])
    np.testing.assert_array_equal(part[1, 1], full[0, 0])
    for bad in ([[n]], [[-n - 1]], [[0.5]]):
        with pytest.raises(ValueError, match="outside the prompt"):
            gemma.hidden(t, [None], [0], bad)
    for layers in ([], [0, 0], [gemma.info.n_layers], [True]):
        with pytest.raises(ValueError):
            gemma.hidden(t, [None], layers, [[0]])
    with pytest.raises(ValueError, match="position lists"):
        gemma.hidden(t, [None], [0], [[0], [1]])


def test_lens_at_the_last_layer_is_the_models_own_distribution(be):
    """Through the final norm, the unembedding and the softcap exactly when the loaded class applies
    one: Gemma3ForCausalLM does (the tiny one is capped at 5); Gemma3ForConditionalGeneration does
    not, though its text_config names a cap. Flipping that choice breaks the equality."""
    register(be, [1])
    ts = texts(be)
    st = starts(be, ts)
    ids, last = all_ids(be), be.info.n_layers - 1
    for ivs in ([None] * 3, [add(1, 1.0, Positions("from", s)) for s in st]):
        nt = be.next_token_logprobs(ts, ivs, ids)
        np.testing.assert_allclose(be.lens(ts, ivs, [last], -1, ids)[:, 0], nt, atol=1e-5)
    be._softcap = 5.0 if be._softcap is None else None
    flipped = be.lens(ts, [None] * 3, [last], -1, ids)[:, 0]
    assert np.abs(flipped - be.next_token_logprobs(ts, [None] * 3, ids)).max() > 1e-3


def test_lens_at_an_earlier_layer_matches_a_manual_unembedding(gemma):
    t = texts(gemma)[:1]
    ids = all_ids(gemma)
    h = gemma.hidden(t, [None], [1], [[5]])[0][0, 0]
    with torch.no_grad():
        logits = gemma._lm_head(gemma._final_norm(torch.tensor(h)[None]))[0]
        logits = 5.0 * torch.tanh(logits / 5.0)
    want = torch.log_softmax(logits.double(), -1).numpy()
    np.testing.assert_allclose(gemma.lens(t, [None], [1], 5, ids)[0, 0], want, atol=1e-5)


def test_residual_stats_match_a_manual_computation(be):
    ts = texts(be)
    pos = every_position(be, ts)
    layers = [0, 2]
    masks = {0: [3, 7], 2: []}
    st = be.residual_stats(ts, layers, masks=masks, skip_first=1)
    hid = be.hidden(ts, [None] * 3, layers, pos)
    for j, L in enumerate(layers):
        x = np.concatenate([h[1:, j].astype(np.float64) for h in hid])
        r = st[L]
        assert r.layer == L and r.n_tokens == x.shape[0] == sum(len(p) - 1 for p in pos)
        np.testing.assert_allclose(r.mean, x.mean(0), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(r.mean_abs, np.abs(x).mean(0), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(r.norms, np.linalg.norm(x, axis=1), rtol=1e-5)   # prompt order
        xm = x.copy()
        xm[:, masks[L]] = 0.0
        np.testing.assert_allclose(r.masked_norms, np.linalg.norm(xm, axis=1), rtol=1e-5)
        assert r.mean.dtype == np.float32 and r.norms.dtype == np.float32
    assert be.residual_stats(ts, [1])[1].masked_norms is None
    assert be.residual_stats(ts, [1], skip_first=0)[1].n_tokens == sum(len(p) for p in pos)


def test_residual_stats_guards_trip(gemma):
    ts = texts(gemma)
    with pytest.raises(ValueError, match="layers \\[2\\]"):
        gemma.residual_stats(ts, [0, 2], masks={0: []})
    with pytest.raises(ValueError, match="out of range"):
        gemma.residual_stats(ts, [0], masks={0: [gemma.info.d_model]})
    with pytest.raises(ValueError, match="skip_first"):
        gemma.residual_stats(ts, [0], skip_first=-1)
    with pytest.raises(ValueError, match="no token is counted"):
        gemma.residual_stats([gemma.decode([gemma.info.bos_token_id])], [0], skip_first=1)


def test_norm_gains_are_the_effective_gains():
    """Gemma multiplies by (1 + weight), Qwen3 by weight: the gains returned are what multiplies."""
    for kind, names, effective in (
            ("gemma", ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                       "post_feedforward_layernorm"], lambda w: 1.0 + w),
            ("gemma_mm", ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                          "post_feedforward_layernorm"], lambda w: 1.0 + w),
            ("qwen", ["input_layernorm", "post_attention_layernorm"], lambda w: w)):
        be = BUILDERS[kind]()
        gains = be.norm_gains(1)
        assert sorted(gains) == sorted(names)
        for name in names:
            w = getattr(be._layers[1], name).weight.detach().numpy()
            np.testing.assert_allclose(gains[name], effective(w), rtol=1e-6)
            assert gains[name].dtype == np.float32
        with pytest.raises(ValueError):
            be.norm_gains(be.info.n_layers)


# ===================================================================================== hooks

@pytest.mark.parametrize("kind", ["gemma", "qwen"])
@pytest.mark.parametrize("last", [False, True])
def test_tuple_and_tensor_layer_outputs_are_both_steered(kind, last):
    """transformers 5.19 layers return a bare tensor; older ones returned a tuple, and upstream's
    hook skipped one of the two shapes without a word (bug 26). Both are steered alike."""
    be = BUILDERS[kind]()
    L = be.info.n_layers - 1 if last else 1
    register(be, [L])
    ts = texts(be)
    st = starts(be, ts)
    ivs = [add(L, 1.0, Positions("from", s)) for s in st]
    pos = every_position(be, ts)
    seen = []
    probe = be._layers[L].register_forward_hook(lambda m, a, o: seen.append(type(o)))
    be.next_token_logprobs(ts[:1], [None], [1])
    probe.remove()
    assert seen == [torch.Tensor]
    ids = all_ids(be)
    want_lp = be.next_token_logprobs(ts, ivs, ids)
    want_h = be.hidden(ts, ivs, [L], pos)

    layer = be._layers[L]
    plain = layer.forward
    layer.forward = lambda *a, **k: (plain(*a, **k), None)
    following = be._final_norm if last else be._layers[L + 1]
    unwrap = following.register_forward_pre_hook(
        lambda m, a, k: ((a[0][0],) + tuple(a[1:]), k) if a and isinstance(a[0], tuple) else None,
        with_kwargs=True)
    seen.clear()
    probe = layer.register_forward_hook(lambda m, a, o: seen.append(type(o)))
    got_lp = be.next_token_logprobs(ts, ivs, ids)
    probe.remove()
    assert seen == [tuple]
    np.testing.assert_allclose(got_lp, want_lp, atol=1e-6)
    for a, b in zip(be.hidden(ts, ivs, [L], pos), want_h):
        np.testing.assert_allclose(a, b, atol=1e-6)
    assert np.abs(got_lp - be.next_token_logprobs(ts, [None] * 3, ids)).max() > 1e-3
    unwrap.remove()


def test_an_unknown_layer_output_is_refused(gemma):
    register(gemma, [1])
    layer = gemma._layers[1]
    plain = layer.forward
    layer.forward = lambda *a, **k: {"h": plain(*a, **k)}
    with pytest.raises(TypeError, match="cannot be steered"):
        gemma.next_token_logprobs(texts(gemma)[:1], [add(1, 1.0, Positions("all"))], [1])


def test_a_hook_the_pass_never_reaches_is_caught(gemma):
    """Bug 26's remaining path: a hook on a module the forward pass does not call."""
    register(gemma, [1])
    gemma._layer_module = lambda layer: torch.nn.Identity()
    with pytest.raises(RuntimeError, match="fired"):
        gemma.next_token_logprobs(texts(gemma)[:1], [add(1, 1.0, Positions("all"))], [1])
    assert gemma._handles == []


def test_a_plan_that_does_not_match_its_batch_is_refused(gemma):
    with pytest.raises(RuntimeError, match="outside a planned forward pass"):
        gemma._on_output(0, torch.zeros(1, 3, gemma.info.d_model))
    gemma._pass = _Pass(batch=2, width=5)
    with pytest.raises(RuntimeError, match="bug 25b"):
        gemma._on_output(0, torch.zeros(2, 4, gemma.info.d_model))
    gemma._pass = None
    with gemma._hooked([0]):
        with pytest.raises(RuntimeError, match="still attached"):
            with gemma._hooked([1]):
                pass


# ===================================================================================== multimodal Gemma-3

def test_multimodal_gemma_is_steered_text_only():
    """The class the -it checkpoints load as: layers under model.language_model, lm_head outside."""
    be = tm.gemma_mm_backend()
    register(be, [1])
    ts = texts(be)
    st = starts(be, ts)
    ivs = [add(1, 1.0, Positions("from", s)) for s in st]
    gens = be.generate([GenRequest(t, iv, 6, 1.0, seed=i, top_p=1.0, top_k=0) for i, (t, iv) in enumerate(zip(ts, ivs))])
    assert all(len(g.token_ids) == 6 or g.finish == "eos" for g in gens)
    ids = all_ids(be)
    steered = be.next_token_logprobs(ts, ivs, ids)
    assert np.abs(steered - be.next_token_logprobs(ts, [None] * 3, ids)).max() > 1e-3


# ===================================================================================== loading

@pytest.mark.parametrize("kind", ["causal", "multimodal", "qwen3"])
def test_load_reads_a_local_checkpoint_onto_one_device(tmp_path, kind):
    build_model, build_tok, name = {
        "causal": (tm.gemma_text_model, tm.gemma_tokenizer, "tiny-gemma3-text"),
        "multimodal": (tm.gemma_multimodal_model, tm.gemma_tokenizer, "tiny-gemma3-it"),
        "qwen3": (tm.qwen_model, tm.qwen_tokenizer, "tiny-qwen3")}[kind]
    tok = build_tok()
    model = build_model(tok)
    path = str(tm.save(model, tok, tmp_path / name))
    cfg = config.load({"MODEL_ID": path, "DTYPE": "float32", "DEVICE": "cpu", "GEN_BATCH_MAX": 4})
    be = HFBackend.load(cfg)
    assert type(be.model).__name__ == type(model).__name__
    assert be.info.name == path and be.batch_size == 4 and str(be.device) == "cpu"
    ref = HFBackend(model, tok, name="ref", hf_id="ref", family=cfg.model_family)
    t = be.render([{"role": "user", "content": Q1}])
    np.testing.assert_array_equal(be.next_token_logprobs([t], [None], all_ids(be)),
                                  ref.next_token_logprobs([t], [None], all_ids(ref)))
    prov = be.provenance()
    assert prov["model_class"] == type(model).__name__ and prov["visible_gpus"] == 0
    assert prov["device"] == "cpu" and prov["layers_path"] == be._layers_path


def test_load_guards_trip(tmp_path):
    tok = tm.qwen_tokenizer()
    path = str(tm.save(tm.qwen_model(tok), tok, tmp_path / "tiny-gemma-but-qwen"))
    with pytest.raises(ValueError, match="is a qwen model"):
        HFBackend.load(config.load({"MODEL_ID": path, "DTYPE": "float32", "DEVICE": "cpu"}))
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="no CUDA device"):
            HFBackend.load(config.load({"MODEL_ID": path, "DEVICE": "cuda"}))


# ===================================================================================== liveness

@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_liveness_passes_on_live_hooks(kind):
    be = BUILDERS[kind]()
    hooks = hook_count(be)
    report = liveness.check_liveness(be)
    assert report["passed"] and not report["failed"]
    assert set(report["checks"]) == {"zero_dose", "dose_moves", "wrong_layer", "batch_independence",
                                     "decode_path", "lens_matches_logits", "generate_path"}
    assert report["checks"]["batch_independence"]["rows"] == 8
    assert not be.has_vector(liveness.LIVENESS_VECTOR_ID) and be._handles == []
    assert hook_count(be) >= hooks           # HF's own capture hooks may have been installed


def test_liveness_passes_in_bfloat16():
    report = liveness.check_liveness(tm.gemma_backend(dtype=torch.bfloat16))
    assert report["passed"] and report["dtype"] == "bfloat16"


def _drop_interventions(be):
    real = be.generate
    be.generate = lambda reqs: real([replace(r, intervention=None) for r in reqs])


def _leaky_edit(h, e):
    out = HFBackend._edit(h, e)
    bump = torch.zeros_like(out)
    bump[e.rows] = e.mask[..., None].to(out.dtype) * 0.05
    return out + bump


@pytest.mark.parametrize("breakage, check, dtype", [
    ("inert", "dose_moves", "float32"),
    ("leaky", "zero_dose", "float32"),
    ("one_module", "wrong_layer", "float32"),
    ("one_early", "batch_independence", "float32"),
    ("decode_unsteered", "decode_path", "float32"),
    ("no_softcap", "lens_matches_logits", "float32"),
    ("generate_drops_steering", "generate_path", "float32"),
    # the bounds that are looser in bfloat16 still fail on their defects
    ("one_early", "batch_independence", "bfloat16"),
    ("decode_unsteered", "decode_path", "bfloat16"),
    ("no_softcap", "lens_matches_logits", "bfloat16"),
])
def test_liveness_fails_when_the_hooks_are_broken(breakage, check, dtype):
    """Each check is shown to fail on the defect it exists for (AGENTS.md: a check that cannot
    fail is worse than none)."""
    be = tm.gemma_backend(dtype=getattr(torch, dtype))
    if breakage == "inert":
        be._edit = lambda h, e: h.clone()
    elif breakage == "leaky":
        be._edit = _leaky_edit
    elif breakage == "one_module":
        be._layer_module = lambda layer: be._layers[0]
    elif breakage == "one_early":
        be._place = lambda mask, pad, width: ([False] * max(pad - 1, 0) + list(mask)
                                              + [False] * (1 if pad else 0))
    elif breakage == "decode_unsteered":
        be._decode_entries = lambda rows: []
    elif breakage == "no_softcap":
        be._softcap = None
    elif breakage == "generate_drops_steering":
        _drop_interventions(be)
    with pytest.raises(liveness.LivenessError) as err:
        liveness.check_liveness(be)
    assert check in err.value.report["failed"], err.value.report["failed"]
    assert check in str(err.value)
    assert not be.has_vector(liveness.LIVENESS_VECTOR_ID)


def test_liveness_on_a_stored_vector_needs_a_real_dose(gemma):
    register(gemma, [1, 2], vid="concept")
    with pytest.raises(ValueError, match="pass alpha"):
        liveness.check_liveness(gemma, vector_id="concept")
    with pytest.raises(ValueError, match="too weak"):
        liveness.check_liveness(gemma, vector_id="concept", alpha=0.01)
    report = liveness.check_liveness(gemma, vector_id="concept", alpha=1.0, layer=2, other_layer=1)
    assert report["passed"] and report["vector"] == "concept"
    assert gemma.has_vector("concept")                 # a caller's vector is left registered
    with pytest.raises(ValueError, match="distinct layers"):
        liveness.check_liveness(gemma, layer=1, other_layer=1)
    with pytest.raises(KeyError, match="unknown liveness tolerance"):
        liveness.check_liveness(gemma, tolerances={"exactish": 1.0})
