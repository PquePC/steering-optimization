"""m5.judges.clients against in-process fakes: a scripted Backend, a scripted HTTP opener and a
fake Anthropic SDK client. No network, no model, no SDK install.

Every failure class of read_judges §1.4 is exercised: retried transients (429 with retry-after,
5xx, transport errors), non-retried deterministic failures (empty content, length stops, 400),
run-fatal ones (402 -> BudgetExhausted carrying the replies already paid for; 401/404 ->
JudgeSetupError), refusals recorded as missing, and the safety gates (loopback only, no proxy
for local calls, external judges refused for non-benign runs and when JUDGE_EXTERNAL is off).
"""

from __future__ import annotations

import pytest

from m5 import safety
from m5.judges import clients as C
from m5.judges import macar
from m5.judges import prompts as P
from m5.tests._judges_fakes import (FakeAnthropic, FakeHTTP, FakeJudgeBackend, anthropic_message,
                                    chat_ok, fake_anthropic_errors)

TEXT = "I made garlic soup today, with plenty of garlic."


@pytest.fixture
def ext(cfg):
    """External tiers need JUDGE_EXTERNAL set explicitly (the default is off, decision 12)."""
    return cfg.replace(JUDGE_EXTERNAL=True)


@pytest.fixture
def T(cfg):
    return P.templates(cfg)


def _allusion(T, text=TEXT, iid="a"):
    return P.make_item(T["allusion"], {"concept": "garlic", "text": text}, item_id=iid)


# ===================================================================================== tier 1, in process

def _backend(**kw):
    def dist(p):
        if "garlic soup" in p:
            return {" YES": 0.6, "YES": 0.2, "Yes": 0.05, " NO": 0.1}
        if "onion" in p:
            return {" NO": 0.85, " YES": 0.1}
        if "rambling" in p:
            return {"The": 0.9, " YES": 0.05, " NO": 0.04}
        if "even" in p:
            return {" YES": 0.4, " NO": 0.4}
        return {"85": 0.5, "90": 0.3, "REF": 0.05}

    def gen(p):
        if p.endswith("Quote:"):
            return " garlic soup\n"
        if p.endswith("Name:"):
            return " garlic"
        return "REFUSAL" if "I refuse to answer" in p else "77"
    return FakeJudgeBackend(dist=kw.pop("dist", dist), gen=kw.pop("gen", gen), **kw)


def test_backend_judge_reads_p_yes_from_logprobs_and_generates_only_the_quote(cfg, T):
    be = _backend()
    j = C.BackendJudge(be, cfg)
    yes, no = j.judge([_allusion(T), _allusion(T, "I like onion rings.", "b")])
    assert yes.label is True and yes.fields["quote_valid"] and yes.fields["quote"] == "garlic soup"
    assert yes.p_yes == pytest.approx(0.85 / 0.95)
    assert no.label is False and no.p_yes == pytest.approx(0.1 / 0.95) and no.fields["quote"] is None
    # one log-prob pass for both items, one generation for the one YES (the quote), none for NO
    assert be.calls == [("logprobs", 2), ("generate", 1)]
    assert any(p.endswith("Answer: YES\nQuote:") for p in be.prompts)
    assert yes.cost_usd == 0.0 and not yes.external


def test_backend_judge_label_follows_its_probability_even_when_yes_mass_is_split(cfg, T):
    """' NO' alone (0.1) is no less likely than any single YES spelling here, yet YES-like mass
    is 0.85: the label is the mass comparison, and the quote is fetched after a forced YES."""
    be = _backend(dist=lambda p: {" YES": 0.3, "YES": 0.3, "Yes": 0.25, " NO": 0.12})
    r = C.BackendJudge(be, cfg).judge([_allusion(T)])[0]
    assert r.label is True and r.p_yes > 0.85


def test_backend_judge_off_format_and_ties_are_missing(cfg, T):
    j = C.BackendJudge(_backend(), cfg)
    off, tie = j.judge([_allusion(T, "rambling text", "c"), _allusion(T, "even odds", "d")])
    assert off.missing.startswith("off_format") and off.label is None
    assert tie.missing.startswith("tie")


def test_backend_judge_downgrades_a_yes_whose_quote_is_not_in_the_text(cfg, T):
    be = _backend(gen=lambda p: " onion soup\n" if p.endswith("Quote:") else " x")
    r = C.BackendJudge(be, cfg).judge([_allusion(T)])[0]
    assert r.label is False and r.fields["downgraded"] == "quote_invalid" and r.p_yes > 0.5


def test_backend_judge_names_and_persona_vector_scores(cfg, T):
    be = _backend()
    j = C.BackendJudge(be, cfg)
    name = P.make_item(T["named"], {"question": "What is it about?", "text": "It is garlic."}, item_id="n")
    trait = P.make_item(T["trait"], {"trait": "sycophantic", "description": "pleases the user",
                                     "prompt": "Is my essay good?", "response": "Brilliant!"}, item_id="t")
    n, t = j.judge([name, trait])
    assert n.label == "garlic"
    assert t.label == pytest.approx((85 * 0.5 + 90 * 0.3) / 0.8)
    assert t.fields["method"] == "logprob" and t.fields["number_mass"] == pytest.approx(0.8)


def test_backend_judge_falls_back_to_the_greedy_number_on_a_digit_splitting_tokenizer(cfg, T):
    j = C.BackendJudge(_backend(digit_split=True), cfg)
    assert j._numbers is None
    trait = P.make_item(T["trait"], {"trait": "t", "description": "d", "prompt": "p",
                                     "response": "r"}, item_id="t")
    r = j.judge([trait])[0]
    assert r.label == 77.0 and r.fields["method"] == "greedy_digit_split_tokenizer"


def test_backend_judge_low_number_mass_reads_refusal_greedily(cfg, T):
    be = _backend(dist=lambda p: {"REF": 0.9})
    trait = P.make_item(T["trait"], {"trait": "t", "description": "d", "prompt": "p",
                                     "response": "I refuse to answer."}, item_id="t")
    r = C.BackendJudge(be, cfg).judge([trait])[0]
    assert r.label == "REFUSAL" and r.notes == ("greedy_after_low_number_mass",)


def test_a_judge_from_the_subjects_family_is_refused(cfg):
    with pytest.raises(C.JudgeSetupError, match="family"):
        C.BackendJudge(_backend(hf_id="google/gemma-3-12b-it"), cfg)   # subject is Gemma


def test_a_judge_prompt_with_an_open_reasoning_block_is_refused(cfg, T):
    be = _backend()
    be.render = lambda messages, add_generation_prompt=True: "<user>x<assistant><think>\n"
    with pytest.raises(C.JudgeSetupError, match="reasoning block"):
        C.BackendJudge(be, cfg).judge([_allusion(T)])


# ===================================================================================== tier 1, loopback server

def test_local_endpoint_refuses_non_loopback_urls(cfg):
    for url in ("http://10.0.0.5:8000/v1", "http://0.0.0.0:8000/v1", "https://example.com/v1"):
        with pytest.raises(safety.Refused, match="loopback"):
            C.LocalOpenAIJudge(cfg, url=url)
    with pytest.raises(C.JudgeSetupError, match="JUDGE_T1_URL is empty"):
        C.LocalOpenAIJudge(cfg)


def test_local_endpoint_reads_p_yes_from_top_logprobs_with_thinking_off(cfg, T):
    lp = [{"token": "Answer", "logprob": -0.01, "top_logprobs": []},
          {"token": ":", "logprob": 0.0, "top_logprobs": []},
          {"token": " YES", "logprob": -0.1, "top_logprobs": [{"token": " YES", "logprob": -0.1},
                                                              {"token": " NO", "logprob": -2.5}]}]
    http = FakeHTTP([(200, chat_ok("Answer: YES\nQuote: garlic soup", logprobs=lp), {})])
    j = C.LocalOpenAIJudge(cfg, url="http://127.0.0.1:8000/v1", opener=http)
    r = j.judge([_allusion(T)])[0]
    assert r.label is True and 0.9 < r.p_yes < 0.93 and not r.external
    body = http.requests[0]["body"]
    assert http.requests[0]["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0 and body["logprobs"] is True and body["top_logprobs"] == 20


def test_the_loopback_opener_bypasses_proxies(monkeypatch):
    import urllib.request
    seen = {}

    def fake_build_opener(*handlers):
        seen["handlers"] = handlers
        return urllib.request.OpenerDirector()
    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    C.loopback_opener()
    proxies = [h for h in seen["handlers"] if isinstance(h, urllib.request.ProxyHandler)]
    assert proxies and proxies[0].proxies == {}


# ===================================================================================== tier 2

def _openrouter(ext, http, sleeps, **kw):
    return C.OpenRouterJudge(ext, vector_classes=["concept"], api_key="sk-test", opener=http,
                             sleep=sleeps.append, **kw)


def test_openrouter_body_turns_reasoning_off_and_records_the_provider(ext, T):
    http = FakeHTTP([(200, chat_ok("Answer: YES\nQuote: garlic soup"), {})])
    r = _openrouter(ext, http, []).judge([_allusion(T)])[0]
    body = http.requests[0]["body"]
    assert body["reasoning"] == {"enabled": False} and body["temperature"] == 0
    assert body["max_tokens"] == ext["JUDGE_MAX_TOKENS"] and "provider" not in body
    assert http.requests[0]["url"] == C.OPENROUTER_URL
    assert http.requests[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert r.label is True and r.provider == "DeepInfra" and r.cost_usd == 1e-05 and r.external


def test_openrouter_pins_the_provider_without_fallbacks_and_flags_a_mismatch(ext, T):
    http = FakeHTTP([(200, chat_ok("Answer: NO\nQuote: NONE", provider="DeepInfra"), {}),
                     (200, chat_ok("Answer: NO\nQuote: NONE", provider="Together"), {})])
    j = _openrouter(ext.replace(JUDGE_T2_PROVIDER="deepinfra/fp8", JUDGE_CONCURRENCY=1), http, [])
    ok, bad = j.judge([_allusion(T), _allusion(T, "other text", "b")])
    assert http.requests[0]["body"]["provider"] == {"order": ["deepinfra/fp8"], "allow_fallbacks": False}
    assert ok.label is False and ok.provider == "DeepInfra"
    assert bad.missing.startswith("provider_mismatch")


def test_openrouter_retries_429_with_retry_after_then_5xx_then_succeeds(ext, T):
    sleeps: list[float] = []
    http = FakeHTTP([(429, {"error": "rate"}, {"Retry-After": "3"}), (503, b"busy", {}),
                     C.TransportError("timed out"),
                     (200, chat_ok("Answer: YES\nQuote: garlic soup"), {})])
    r = _openrouter(ext, http, sleeps).judge([_allusion(T)])[0]
    assert r.label is True and r.attempts == 4 and sleeps[0] == 3.0 and len(sleeps) == 3


def test_openrouter_transients_that_outlast_the_retries_are_missing(ext, T):
    http = FakeHTTP([(503, b"busy", {})] * 3)
    r = _openrouter(ext.replace(JUDGE_RETRIES=2), http, []).judge([_allusion(T)])[0]
    assert r.missing.startswith("transient_exhausted") and r.attempts == 3


@pytest.mark.parametrize("response,reason", [
    ((200, chat_ok(""), {}), "empty_content"),
    ((200, chat_ok("Answer: YES\nQuo", finish="length"), {}), "truncated"),
    ((400, {"error": {"message": "bad"}}, {}), "http_400"),
    ((200, chat_ok("Answer: YES or NO"), {}), "template_echo"),
    ((200, b"<html>", {}), "unparseable"),
])
def test_deterministic_failures_are_missing_and_not_retried(ext, T, response, reason):
    http = FakeHTTP([response])
    r = _openrouter(ext, http, []).judge([_allusion(T)])[0]
    assert r.missing.startswith(reason) and r.label is None
    assert len(http.requests) == 1                    # read_judges J-05: no retry


def test_402_stops_the_run_and_carries_the_replies_already_paid_for(ext, T):
    http = FakeHTTP([(200, chat_ok("Answer: NO\nQuote: NONE"), {}),
                     (402, {"error": {"code": 402, "message": "credit"}}, {})])
    j = _openrouter(ext.replace(JUDGE_CONCURRENCY=1), http, [])
    with pytest.raises(C.BudgetExhausted) as info:
        j.judge([_allusion(T, "first text", "x"), _allusion(T, "second text", "y")])
    assert [r.item_id for r in info.value.completed] == ["x"]


def test_402_inside_a_200_body_also_stops_the_run(ext, T):
    http = FakeHTTP([(200, {"error": {"code": 402, "message": "insufficient credits"}}, {})])
    with pytest.raises(C.BudgetExhausted):
        _openrouter(ext, http, []).judge([_allusion(T)])


@pytest.mark.parametrize("status", [401, 403, 404])
def test_auth_and_model_errors_are_setup_errors(ext, T, status):
    with pytest.raises(C.JudgeSetupError):
        _openrouter(ext, FakeHTTP([(status, {"error": "no"}, {})]), []).judge([_allusion(T)])


def test_openrouter_needs_its_key_at_start(ext, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(C.JudgeSetupError, match="OPENROUTER_API_KEY"):
        C.OpenRouterJudge(ext, vector_classes=["concept"])


def test_external_judges_are_refused_by_default_and_for_non_benign_classes(cfg, ext):
    assert cfg["JUDGE_EXTERNAL"] is False                 # fail closed until decision 12
    with pytest.raises(safety.Refused):
        C.OpenRouterJudge(cfg, vector_classes=["concept"], api_key="k")
    with pytest.raises(safety.Refused):
        C.AnthropicJudge(cfg, vector_classes=["concept"], client=FakeAnthropic(),
                         errors=fake_anthropic_errors(), max_tokens=1024)
    with pytest.raises(safety.Refused):
        C.OpenRouterJudge(ext, vector_classes=["concept", "deception"], api_key="k")
    with pytest.raises(safety.Refused):
        C.MacarJudge(ext, vector_classes=[], api_key="k")


# ===================================================================================== tier 3

def _anthropic(ext, responses, sleeps=None, **kw):
    E = fake_anthropic_errors()
    client = FakeAnthropic(responses, **kw)
    j = C.AnthropicJudge(ext, vector_classes=["concept"], client=client, errors=E, max_tokens=2048,
                         sleep=(sleeps.append if sleeps is not None else (lambda s: None)))
    return j, client, E


def test_anthropic_request_shape(ext, T):
    j, client, _ = _anthropic(ext, [anthropic_message('{"answer": "YES", "quote": "garlic soup"}')])
    r = j.judge([_allusion(T)])[0]
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 2048
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    assert kw["output_config"]["effort"] == ext["JUDGE_T3_EFFORT"]
    assert kw["output_config"]["format"]["type"] == "json_schema"
    for banned in ("temperature", "top_p", "top_k", "thinking", "system"):
        assert banned not in kw
    assert kw["messages"][-1]["role"] == "user"          # no assistant prefill
    assert r.label is True and r.p_yes is None and r.served_model == "claude-opus-5-5"
    assert r.cost_usd == pytest.approx((400 * 4.0 + 30 * 20.0) / 1e6)


def test_a_refusal_is_a_missing_judgment_with_its_category_never_a_no(ext, T):
    j, _, _ = _anthropic(ext, [anthropic_message(None, stop="refusal", category="bio")])
    r = j.judge([_allusion(T)])[0]
    assert r.label is None and r.missing == "refusal: bio"
    assert r.fields["stop_details"]["category"] == "bio"


def test_anthropic_billing_error_stops_the_run(ext, T):
    E = fake_anthropic_errors()
    j, _, _ = _anthropic(ext, [E.APIStatusError("credit", status_code=402, type="billing_error")])
    with pytest.raises(C.BudgetExhausted):
        j.judge([_allusion(T)])


def test_anthropic_bad_request_is_missing_and_not_retried(ext, T):
    E = fake_anthropic_errors()
    j, client, _ = _anthropic(ext, [E.BadRequestError("prefill not allowed", status_code=400,
                                                      type="invalid_request_error")])
    r = j.judge([_allusion(T)])[0]
    assert r.missing.startswith("http_400") and len(client.calls) == 1


def test_anthropic_retries_rate_limits_overload_and_connection_errors(ext, T):
    E = fake_anthropic_errors()
    sleeps: list[float] = []
    j, client, _ = _anthropic(ext, [
        E.RateLimitError("slow down", status_code=429, type="rate_limit_error", headers={"retry-after": "7"}),
        E.APIStatusError("overloaded", status_code=529, type="overloaded_error"),
        E.APIConnectionError("reset"),
        anthropic_message('{"answer": "NO", "quote": "NONE"}')], sleeps)
    r = j.judge([_allusion(T)])[0]
    assert r.label is False and r.attempts == 4 and sleeps[0] == 7.0 and len(client.calls) == 4


def test_anthropic_max_tokens_stop_is_truncated(ext, T):
    j, _, _ = _anthropic(ext, [anthropic_message('{"answer": "YE', stop="max_tokens")])
    assert j.judge([_allusion(T)])[0].missing.startswith("truncated")


def test_tier_three_without_the_sdk_names_the_install(ext, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_anthropic(name, *a, **k):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_anthropic)
    with pytest.raises(C.JudgeSetupError, match="pip install anthropic"):
        C.AnthropicJudge(ext, vector_classes=["concept"], max_tokens=1024)


def test_anthropic_batch_path_keys_results_by_custom_id_and_has_no_fallbacks(ext, T):
    from types import SimpleNamespace as NS
    a, b, c = _allusion(T, iid="a"), _allusion(T, "I like onion rings.", "b"), _allusion(T, "more", "c")
    results = [  # any order
        NS(custom_id=b.key, result=NS(type="succeeded", message=anthropic_message('{"answer": "NO", "quote": "NONE"}'))),
        NS(custom_id=a.key, result=NS(type="succeeded", message=anthropic_message('{"answer": "YES", "quote": "garlic soup"}'))),
        NS(custom_id=c.key, result=NS(type="expired")),
    ]
    sleeps: list[float] = []
    j, client, _ = _anthropic(ext, [], sleeps, batch_results=results, polls_before_end=2)
    ra, rb, rc = j.judge_batch([a, b, c], poll_s=60.0, max_wait_s=600.0)
    assert ra.label is True and rb.label is False and rc.missing == "batch_expired"
    assert all("fallbacks" not in r["params"] and "betas" not in r["params"] for r in client.batch_requests)
    assert {r["custom_id"] for r in client.batch_requests} == {a.key, b.key, c.key}
    assert ra.cost_usd == pytest.approx((400 * 4.0 + 30 * 20.0) / 1e6 * 0.5)
    assert sleeps == [60.0, 60.0]


def test_anthropic_batch_billing_error_stops_the_run(ext, T):
    from types import SimpleNamespace as NS
    a = _allusion(T)
    results = [NS(custom_id=a.key, result=NS(type="errored", error=NS(type="error", error=NS(type="billing_error"))))]
    j, _, _ = _anthropic(ext, [], batch_results=results)
    with pytest.raises(C.BudgetExhausted):
        j.judge_batch([a], poll_s=1.0, max_wait_s=10.0)


# ===================================================================================== Macar

def test_macar_judge_sends_their_prompt_with_their_settings_and_returns_their_fields(ext):
    item = macar.make_macar_item("claims_detection", prompt=macar.trial_question(3),
                                 response="Yes, I notice an injected thought.", item_id="m1")
    http = FakeHTTP([(200, chat_ok("Thinking...\nAnswer: YES", model="openai/gpt-4.1-mini",
                                   provider="OpenAI"), {}),
                     (200, chat_ok("I cannot tell.", model="openai/gpt-4.1-mini"), {})])
    j = C.MacarJudge(ext.replace(JUDGE_CONCURRENCY=1), vector_classes=["concept"], api_key="k",
                     opener=http, sleep=lambda s: None)
    yes, unclear = j.judge([item, macar.make_macar_item("claims_detection", prompt=macar.trial_question(4),
                                                        response="Hmm.", item_id="m2")])
    body = http.requests[0]["body"]
    assert body["model"] == "openai/gpt-4.1-mini" and body["temperature"] == 0.0 and body["max_tokens"] == 500
    assert body["messages"] == [{"role": "user", "content": item.render()}]
    assert "reasoning" not in body
    assert yes.label is True and yes.fields == {"answer": "YES", "grade": 1}
    assert unclear.label is None and "scores it 0" in unclear.missing


def test_batch_mode_routes_judge_through_batches_under_its_own_signature(ext, T):
    from types import SimpleNamespace as NS
    a = _allusion(T)
    results = [NS(custom_id=a.key, result=NS(type="succeeded",
                                             message=anthropic_message('{"answer": "YES", "quote": "garlic soup"}')))]
    E = fake_anthropic_errors()
    client = FakeAnthropic([], batch_results=results)
    batched = C.AnthropicJudge(ext, vector_classes=["concept"], client=client, errors=E,
                               max_tokens=2048, batch=(1.0, 10.0), sleep=lambda s: None)
    direct = C.AnthropicJudge(ext, vector_classes=["concept"], client=client, errors=E, max_tokens=2048)
    assert batched.judge([a])[0].label is True and client.calls == []
    assert batched.signature != direct.signature and "batch" in batched.signature
