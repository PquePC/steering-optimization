"""Every remaining guard in m5.judges, tripped once (AGENTS.md: a guard that cannot fail is worse
than none). The behavioural tests live beside each module's own test file."""

from __future__ import annotations

import dataclasses

import pytest

from m5.judges import calibration as Cal
from m5.judges import cascade as K
from m5.judges import clients as C
from m5.judges import gold as G
from m5.judges import macar as M
from m5.judges import mechanical, parse
from m5.judges import prompts as P
from m5.judges.queue import CachedClient, JudgeQueue, JudgeStore
from m5.tests._judges_fakes import (FakeAnthropic, FakeHTTP, FakeJudgeBackend, chat_ok,
                                    fake_anthropic_errors)


@pytest.fixture
def T(cfg):
    return P.templates(cfg)


def _item(T, text="a silk scarf", iid="1"):
    return P.make_item(T["dominant"], {"concept": "silk", "text": text}, item_id=iid)


# ===================================================================================== templates

def test_template_construction_guards():
    with pytest.raises(P.TemplateError, match="answer kind"):
        dataclasses.replace(P.DOMINANT, answer="likert")
    with pytest.raises(P.TemplateError, match="duplicate slot"):
        dataclasses.replace(P.DOMINANT, slots=(P.Slot("text", "A", "text"), P.Slot("text", "B", "text")))
    with pytest.raises(P.TemplateError, match="quote_from"):
        dataclasses.replace(P.DOMINANT, quote_from="text")
    with pytest.raises(P.TemplateError, match="not one of its slots"):
        dataclasses.replace(P.DOMINANT, text_slot="reply")
    with pytest.raises(P.TemplateError, match="kind"):
        P.Slot("x", "X", "image")
    with pytest.raises(P.TemplateError, match="no placeholder"):
        P.ALLUSION.bind(quote_max_words=15, temperature=0)
    with pytest.raises(P.TemplateError, match="unknown format"):
        P.DOMINANT.format_block("xml")
    with pytest.raises(P.TemplateDrift, match="calls itself"):
        P.check_templates({"wrong": P.DOMINANT}, {"wrong": (1, P.DOMINANT.text_hash)})


def test_parse_and_mechanical_argument_guards(T):
    with pytest.raises(ValueError, match="reply format"):
        parse.parse_reply(T["dominant"], "Answer: YES", fmt="yaml")
    assert parse.parse_reply(T["dominant"], None).missing == "empty_reply"
    with pytest.raises(ValueError, match="non-empty concept"):
        mechanical.concept_pattern("  ")
    with pytest.raises(ValueError, match="non-empty concept"):
        mechanical.prefix_mentions("x", "")


# ===================================================================================== replies

def test_a_reply_or_judgment_with_both_or_neither_label_and_reason_is_refused(T):
    base = dict(item_key="k", item_id="1", template_id="dominant", template_version=1,
                template_hash="h", client="c", model="m", signature="s")
    with pytest.raises(ValueError, match="label or a missing reason"):
        C.Reply(**base, label=None)
    with pytest.raises(ValueError, match="label or a missing reason"):
        C.Reply(**base, label=True, missing="x")
    with pytest.raises(ValueError, match="never both or neither"):
        K.Judgment(item_key="k", item_id="1", template_id="dominant", template_version=1, label=None,
                   missing=None, decided_by=None, stop="x", trail=(), fields={}, audit={}, meta={})
    assert C.Reply.from_row({**base, "label": True, "notes": ["a"], "kind": "reply", "tier": "t1"}).notes == ("a",)


# ===================================================================================== tier 1 setup

def test_backend_judge_refuses_a_tokenizer_without_single_token_answers(cfg):
    be = FakeJudgeBackend(dist=lambda p: {})
    be.word_variants = lambda w: [tuple(be.tokenize(" " + w)) + (999,)]       # always two tokens
    with pytest.raises(C.JudgeSetupError, match="no single-token YES/NO"):
        C.BackendJudge(be, cfg)


def test_backend_judge_refuses_a_token_that_is_both_answers(cfg):
    be = FakeJudgeBackend(dist=lambda p: {})
    be.word_variants = lambda w: [(7,)]
    with pytest.raises(C.JudgeSetupError, match="both YES-like and NO-like"):
        C.BackendJudge(be, cfg)


def test_backend_judge_refuses_a_token_that_reads_as_two_numbers(cfg):
    be = FakeJudgeBackend(dist=lambda p: {})
    real = be.tokenize
    be.tokenize = lambda text: [be.vocab["1"]] if text in ("1", "10") else real(text)
    with pytest.raises(C.JudgeSetupError, match="two numbers"):
        C.BackendJudge(be, cfg)


def test_anthropic_client_injection_needs_its_errors_namespace(cfg):
    with pytest.raises(C.JudgeSetupError, match="errors namespace"):
        C.AnthropicJudge(cfg.replace(JUDGE_EXTERNAL=True), vector_classes=["concept"],
                         client=FakeAnthropic(), max_tokens=1024)


def test_anthropic_tier_reads_its_own_token_ceiling_by_exact_key(cfg):
    """Thinking cannot be disabled on tier 3, so its ceiling is a separate key,
    JUDGE_T3_MAX_TOKENS, not the few-word JUDGE_MAX_TOKENS the other tiers use."""
    ext = cfg.replace(JUDGE_EXTERNAL=True, JUDGE_T3_MAX_TOKENS=3000)
    j = C.AnthropicJudge(ext, vector_classes=["concept"], client=FakeAnthropic(),
                         errors=fake_anthropic_errors())
    assert j.max_tokens == 3000 != ext["JUDGE_MAX_TOKENS"]


def test_an_unknown_sdk_exception_is_a_bug_and_is_raised(cfg, T):
    j = C.AnthropicJudge(cfg.replace(JUDGE_EXTERNAL=True), vector_classes=["concept"],
                         client=FakeAnthropic([KeyError("bug")]), errors=fake_anthropic_errors(),
                         max_tokens=1024)
    with pytest.raises(KeyError):
        j.judge([_item(T)])


def test_reasoning_tokens_despite_reasoning_off_are_noted(cfg, T):
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0001,
             "completion_tokens_details": {"reasoning_tokens": 40}}
    http = FakeHTTP([(200, chat_ok("Answer: YES", usage=usage), {})])
    j = C.OpenRouterJudge(cfg.replace(JUDGE_EXTERNAL=True), vector_classes=["concept"], api_key="k",
                          opener=http, sleep=lambda s: None)
    r = j.judge([_item(T)])[0]
    assert r.label is True and any(n.startswith("reasoning_tokens=40") for n in r.notes)


# ===================================================================================== ordering

class _Scripted:
    kind, model, signature, external = "backend", "Qwen/Qwen3-8B", "backend|Qwen/Qwen3-8B", False

    def __init__(self, fn):
        self.fn = fn

    def judge(self, items):
        return self.fn(items)


def _reply(item, label=True):
    return C.Reply(item_key=item.key, item_id=item.item_id, template_id="dominant",
                   template_version=1, template_hash="h", client="backend", model="m",
                   signature="s", label=label)


def test_replies_out_of_order_or_missing_are_refused(T):
    items = [_item(T, "one", "1"), _item(T, "two", "2")]
    flipped = CachedClient(_Scripted(lambda its: [_reply(i) for i in reversed(its)]),
                           JudgeStore.memory(), tier="t1")
    with pytest.raises(RuntimeError, match="out of order"):
        flipped.judge(items)
    short = CachedClient(_Scripted(lambda its: [_reply(its[0])]), JudgeStore.memory(), tier="t1")
    with pytest.raises(RuntimeError, match="replies for"):
        short.judge(items)
    with pytest.raises(RuntimeError, match="without a reply"):
        C._complete([None], "x")


def test_the_queue_refuses_a_judge_that_drops_items(T):
    with JudgeQueue(lambda batch: batch[:-1], workers=1, batch_size=2) as q:
        futs = q.submit([_item(T, "one", "1"), _item(T, "two", "2")])
        with pytest.raises(RuntimeError, match="results for"):
            futs[0].result()
    with pytest.raises(ValueError):
        JudgeQueue(lambda b: b, workers=0, batch_size=1)


# ===================================================================================== calibration, gold, macar

def test_calibration_argument_guards():
    with pytest.raises(ValueError, match="target"):
        Cal.selective_risk_band([0.9], [True], target=0.0, conf=0.95)
    with pytest.raises(ValueError, match="paired"):
        Cal.selective_risk_band([0.9, 0.1], [True], target=0.1, conf=0.95)
    with pytest.raises(ValueError, match="paired"):
        Cal.agreement([True], [True, False], conf=0.95)
    with pytest.raises(ValueError, match="conf"):
        Cal.agreement([True], [True], conf=1.0)
    with pytest.raises(ValueError, match="not a Platt"):
        Cal.PlattCalibrator.from_json({"kind": "isotonic"})


def test_gold_argument_guards(tmp_path):
    row = {"id": "t0", "text": "x", "concept": "silk", "dose_band": "b", "mech_flag": "f",
           "model": "m", "prompt": "p", "question": "q"}
    with pytest.raises(ValueError, match="double_share"):
        G.stratified_sample([row], fields=["allusion"], per_field=1, seed=0, double_share=1.5)
    with pytest.raises(ValueError, match="unknown gold field"):
        G.stratified_sample([row], fields=["vibes"], per_field=1, seed=0, double_share=0)
    with pytest.raises(ValueError, match="answer must be"):
        G.append_label(tmp_path, item_id="g1", field="allusion", labeller="A", answer="maybe")
    items, keys = G.stratified_sample([row], fields=["allusion"], per_field=1, seed=0, double_share=0)
    path = G.write_set(items, keys, tmp_path / "set.jsonl")
    G.append_label(tmp_path, item_id="g-unknown", field="allusion", labeller="A", answer="yes")
    with pytest.raises(KeyError, match="unknown item"):
        G.gold_rows(path, tmp_path)


def test_macar_argument_guards():
    with pytest.raises(ValueError, match="unknown Macar criterion"):
        M.make_macar_item("vibes", prompt="p", response="r", item_id="x")
