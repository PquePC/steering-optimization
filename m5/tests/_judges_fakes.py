"""In-process fakes for the m5.judges tests: no network, no model, no SDK.

    FakeJudgeBackend   a Backend whose next-token distribution and greedy continuations are
                       scripted per prompt (for BackendJudge)
    FakeClient         a judge client returning scripted replies (for the cascade and queue)
    FakeHTTP           an opener recording requests and replaying scripted HTTP responses
    fake_anthropic     a fake SDK client and errors namespace with the SDK's class hierarchy
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from m5.backend.base import GenResult, ModelInfo
from m5.judges.clients import HttpResult, Reply, missing_reply
from m5.judges.parse import Parsed


# ===================================================================================== backend

class FakeJudgeBackend:
    """Tokens: a small named vocabulary plus one token per character for everything else.

    `dist(prompt) -> {token string: probability}` scripts the next-token distribution;
    `gen(prompt) -> str` scripts greedy continuations. `digit_split=True` makes multi-digit
    numbers several tokens, like Qwen's tokenizer.
    """

    def __init__(self, *, hf_id: str = "Qwen/Qwen3-8B", dist: Callable[[str], Mapping[str, float]],
                 gen: Callable[[str], str] = lambda p: " NONE", digit_split: bool = False) -> None:
        words = ["YES", " YES", "Yes", " Yes", "yes", " yes", "NO", " NO", "No", " No", "no", " no",
                 "The", " The", "REF"]
        nums = [str(k) for k in range(10)] if digit_split else [str(k) for k in range(101)]
        self.vocab = {w: i + 1 for i, w in enumerate(words + nums)}
        self.inv = {i: w for w, i in self.vocab.items()}
        self.dist, self.gen = dist, gen
        self.calls: list[tuple[str, int]] = []
        self.prompts: list[str] = []
        self.info = ModelInfo(name="fake-judge", hf_id=hf_id, revision="r1", family="qwen",
                              n_layers=2, d_model=4, vocab_size=5000, dtype="float32",
                              bos_token_id=None, eos_token_ids=(0,), chat_template_hash="tmpl1")

    def render(self, messages, *, add_generation_prompt=True):
        return "<user>" + "".join(m["content"] for m in messages) + "<assistant>"

    def tokenize(self, text):
        if text in self.vocab:
            return [self.vocab[text]]
        return [1000 + ord(c) for c in text]

    def offsets(self, text):
        return [(i, i + 1) for i in range(len(text))]

    def decode(self, ids):
        return "".join(self.inv.get(i, chr(i - 1000)) for i in ids)

    def word_variants(self, word):
        forms = {word, " " + word, word.lower(), " " + word.lower(), word.capitalize(),
                 " " + word.capitalize()}
        return sorted({tuple(self.tokenize(f)) for f in forms})

    def next_token_logprobs(self, prompts, interventions, token_ids):
        self.calls.append(("logprobs", len(prompts)))
        out = np.full((len(prompts), len(token_ids)), -30.0)
        for r, p in enumerate(prompts):
            self.prompts.append(p)
            d = self.dist(p)
            for c, tid in enumerate(token_ids):
                w = self.inv.get(int(tid))
                if w is not None and d.get(w, 0.0) > 0:
                    out[r, c] = math.log(d[w])
        return out

    def generate(self, requests):
        self.calls.append(("generate", len(requests)))
        out = []
        for rq in requests:
            assert rq.temperature == 0.0 and rq.intervention is None
            self.prompts.append(rq.prompt)
            text = self.gen(rq.prompt)
            out.append(GenResult(text=text, token_ids=tuple(range(len(text.split()) + 1)), finish="eos"))
        return out


# ===================================================================================== clients

@dataclass
class FakeClient:
    """A judge client answering from `script(item) -> (label, p_yes, fields)` or a missing reason
    string. Records every batch it is asked."""

    kind: str
    model: str
    external: bool
    script: Callable[[Any], Any]
    signature: str = ""
    batches: list[list[str]] = field(default_factory=list)
    raise_on_call: BaseException | None = None

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = f"{self.kind}|{self.model}"

    @property
    def n_items(self) -> int:
        return sum(len(b) for b in self.batches)

    def judge(self, items: Sequence[Any]) -> list[Reply]:
        self.batches.append([it.item_id for it in items])
        if self.raise_on_call is not None:
            raise self.raise_on_call
        out = []
        for it in items:
            res = self.script(it)
            if isinstance(res, str):
                out.append(missing_reply(it, self, res))
                continue
            label, p, fields = res
            from m5.judges.clients import reply_from_parsed
            out.append(reply_from_parsed(it, self, Parsed(label, dict(fields)), p_yes=p,
                                         cost_usd=0.001 if self.external else 0.0))
        return out


# ===================================================================================== HTTP

class FakeHTTP:
    """Replays `responses` in order (status, body-object-or-bytes, headers); records requests."""

    def __init__(self, responses: Sequence[tuple[int, Any, Mapping[str, str]] | BaseException]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url, body, headers, timeout):
        self.requests.append({"url": url, "body": json.loads(body.decode("utf-8")),
                              "headers": dict(headers), "timeout": timeout})
        if not self.responses:
            raise AssertionError("FakeHTTP ran out of scripted responses")
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        status, obj, hdrs = r
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")
        return HttpResult(status, {k.lower(): v for k, v in hdrs.items()}, data)


def chat_ok(content: str, *, provider: str = "DeepInfra", finish: str = "stop",
            model: str = "deepseek/deepseek-v4-flash", logprobs: Any = None,
            usage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    choice: dict[str, Any] = {"message": {"role": "assistant", "content": content},
                              "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    return {"id": "gen-1", "model": model, "provider": provider, "choices": [choice],
            "usage": dict(usage or {"prompt_tokens": 120, "completion_tokens": 6, "cost": 0.00001})}


# ===================================================================================== anthropic

class APIError(Exception):
    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message


class APIStatusError(APIError):
    def __init__(self, message: str = "", *, status_code: int, type: str | None = None,
                 headers: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.type = type
        self.response = SimpleNamespace(headers=dict(headers or {}))


class BadRequestError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    pass


class InternalServerError(APIStatusError):
    pass


class APIConnectionError(APIError):
    pass


def fake_anthropic_errors() -> SimpleNamespace:
    """The SDK's hierarchy: APIError > APIStatusError > {BadRequestError, RateLimitError, ...};
    APIConnectionError is a sibling of APIStatusError. One set of classes per process, so a test
    can raise what the client catches."""
    return SimpleNamespace(APIError=APIError, APIStatusError=APIStatusError,
                           BadRequestError=BadRequestError, RateLimitError=RateLimitError,
                           InternalServerError=InternalServerError,
                           APIConnectionError=APIConnectionError)


def anthropic_message(text: str | None, *, stop: str = "end_turn", category: str | None = None,
                      model: str = "claude-opus-5-5", iterations: Any = None) -> SimpleNamespace:
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        model=model, stop_reason=stop, content=content if stop != "refusal" else [],
        stop_details=SimpleNamespace(type="refusal", category=category, explanation="x") if stop == "refusal" else None,
        usage=SimpleNamespace(input_tokens=400, output_tokens=30, cache_read_input_tokens=0,
                              cache_creation_input_tokens=0, iterations=iterations))


class FakeAnthropic:
    """`client.beta.messages.create(**kw)` answers from `responses` (a message or an exception per
    call); `client.messages.batches` is a scripted in-memory batch."""

    def __init__(self, responses: Sequence[Any] = (), *, batch_results: Sequence[Any] = (),
                 polls_before_end: int = 1) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.batch_requests: list[Any] = []
        self._batch_results = list(batch_results)
        self._polls = polls_before_end
        outer = self

        class _Messages:
            def create(self, **kw):
                outer.calls.append(kw)
                r = outer.responses.pop(0)
                if isinstance(r, BaseException):
                    raise r
                return r

        class _Batches:
            def create(self, *, requests):
                outer.batch_requests = list(requests)
                return SimpleNamespace(id="batch_1", processing_status="in_progress")

            def retrieve(self, bid):
                outer._polls -= 1
                return SimpleNamespace(id=bid, processing_status="ended" if outer._polls <= 0 else "in_progress")

            def results(self, bid):
                return list(outer._batch_results)

        self.beta = SimpleNamespace(messages=_Messages())
        self.messages = SimpleNamespace(batches=_Batches())
