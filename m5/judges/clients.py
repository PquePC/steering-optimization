"""m5.judges.clients - the judges themselves: one local in-process, one local server, two APIs, and
Macar's judge for the reproduction mode (plan §7.3).

Every client answers `judge(items) -> list[Reply]`, in input order, one Reply per item. A Reply
is either a label or a missing reason, never both (implementation guide §1 rule 5), and carries
what the old transport threw away: the served model, the provider, latency, token usage, cost,
the finish reason and the attempt count (read_judges J-06).

Failure classes, each from a defect in the old transport (read_judges §1.4, §6):

    BudgetExhausted   402 / billing_error. Raised, never recorded as a judgment: the old path
                      wrote every later cell with nulls, marked it done, and never re-judged it
                      (J-04, plan I4). The replies already paid for ride on the exception
                      (`completed`) so a cache can keep them (J-13).
    JudgeSetupError   401/403/404, a missing key or SDK, a non-loopback "local" URL, a judge from
                      the subject's family: wrong for every item, so raised at once (J-08, J-46).
    missing reply     a single item that cannot be judged: 400, empty content, a length stop
                      ("truncated", J-07), a refusal, an unparseable reply, transient errors that
                      outlasted the retries. Retries only for transient failures (429, 5xx,
                      timeouts, connection errors); a deterministic failure is not retried (J-05).

Tiers (plan §7.3): `BackendJudge` and `LocalOpenAIJudge` are tier 1 (on the pod); `OpenRouterJudge`
is tier 2 and `AnthropicJudge` tier 3, both external, both refusing to exist unless external
judging is allowed for the run's vector classes (decision 12). Every judge must come from a
different model family than the subject (self-preference bias, plan §7.2).
"""

from __future__ import annotations

import http.client
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

import numpy as np

from m5 import safety
from m5.backend.base import Backend, GenRequest
from m5.config import model_family
from m5.judges import macar
from m5.judges.parse import Parsed, parse_reply
from m5.judges.prompts import JudgeItem

# Persona Vectors' rule for reading a judge from its log-probabilities (R:chen/persona_vectors
# judge.py `_aggregate_0_100_score` and `_aggregate_binary_score`, commit b8e0f04): with less than
# 0.25 of the next-token probability on the answer tokens the judge was not answering in the
# format, and the reading is missing, not a number. A definition of their recipe, kept with it.
MIN_ANSWER_MASS = 0.25

# Transport constants: they change when an answer arrives, never what it is (not measurement
# settings; the old values, read_judges §1.3).
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529})
SETUP_STATUS = frozenset({401, 403, 404})
BUDGET_STATUS = frozenset({402})
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC_HOST_URL = "https://api.anthropic.com"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# US$ per million tokens (input, output), tied to the model id so a model change can never be
# priced at another model's rate (J-38). From the claude-api skill's model table (2026-10-06).
ANTHROPIC_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {"claude-opus-5-5": (4.0, 20.0)}
BATCH_DISCOUNT = 0.5


class JudgeError(RuntimeError):
    """A failure that stops judging. `completed` holds replies finished before it, if any."""

    def __init__(self, message: str, completed: Sequence["Reply"] = ()) -> None:
        super().__init__(message)
        self.completed = list(completed)


class BudgetExhausted(JudgeError):
    """Out of credit (402 / billing_error). Stop the run; nothing is recorded as judged."""


class JudgeSetupError(JudgeError):
    """A judge that cannot work for any item: auth, model, key, SDK, URL or family."""


class TransportError(Exception):
    """A network-level failure before any HTTP status (timeout, DNS, reset). Transient."""


# =====================================================================================
# Replies
# =====================================================================================

@dataclass(frozen=True)
class Reply:
    """One judge's answer to one item. See the module docstring."""

    item_key: str
    item_id: str
    template_id: str
    template_version: int
    template_hash: str
    client: str
    model: str
    signature: str
    label: Any
    fields: dict[str, Any] = field(default_factory=dict)
    p_yes: float | None = None
    raw: str | None = None
    served_model: str | None = None
    provider: str | None = None
    latency_s: float = 0.0
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    finish: str | None = None
    attempts: int = 1
    missing: str | None = None
    external: bool = False
    cached: bool = False
    notes: tuple[str, ...] = ()
    payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.label is None) != (self.missing is not None):
            raise ValueError(f"{self.template_id}: a reply has a label or a missing reason, not "
                             f"both or neither (label={self.label!r}, missing={self.missing!r})")

    @property
    def ok(self) -> bool:
        return self.missing is None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["notes"] = list(self.notes)
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Reply":
        names = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in row.items() if k in names}
        kw["notes"] = tuple(kw.get("notes") or ())
        return cls(**kw)


def _item_meta(item: Any) -> dict[str, Any]:
    """Identity fields for a Reply from an M5 JudgeItem or a Macar item."""
    if isinstance(item, JudgeItem):
        t = item.template
        return {"item_key": item.key, "item_id": item.item_id, "template_id": t.id,
                "template_version": t.version, "template_hash": t.hash,
                "payload": dict(item.payload)}
    return {"item_key": item.key, "item_id": item.item_id, "template_id": item.template_id,
            "template_version": 1, "template_hash": macar.prompt_hash(item.criterion),
            "payload": {"prompt": item.prompt, "response": item.response, "word": item.word}}


def reply_from_parsed(item: Any, client: "JudgeClient", parsed: Parsed, **extra: Any) -> Reply:
    notes = tuple(parsed.notes) + tuple(extra.pop("notes", ()))
    return Reply(**_item_meta(item), client=client.kind, model=client.model,
                 signature=client.signature, label=parsed.label, fields=dict(parsed.fields),
                 missing=parsed.missing, external=client.external, notes=notes, **extra)


def missing_reply(item: Any, client: "JudgeClient", reason: str, **extra: Any) -> Reply:
    return Reply(**_item_meta(item), client=client.kind, model=client.model,
                 signature=client.signature, label=None, missing=reason,
                 external=client.external, **extra)


def _complete(replies: Sequence[Reply | None], what: str) -> list[Reply]:
    """Every slot filled, in input order (read_judges J-28: results out of order are a wrong
    number, not an error, unless something checks)."""
    if any(r is None for r in replies):
        raise RuntimeError(f"{what}: an item was left without a reply")
    return list(replies)  # type: ignore[arg-type]


class JudgeClient(Protocol):
    kind: str
    model: str
    signature: str
    external: bool

    def judge(self, items: Sequence[Any]) -> list[Reply]: ...


def check_judge_family(judge_model: str, cfg: Any) -> None:
    """A judge never shares the subject's model family (self-preference bias, plan §7.2)."""
    fam = model_family(judge_model)
    if fam != "other" and fam == cfg.model_family:
        raise JudgeSetupError(f"judge {judge_model!r} is from the subject's family ({fam}); judges "
                              "prefer their own family's outputs (plan §7.2)")


# =====================================================================================
# Tier 1, in process: any Backend
# =====================================================================================

class BackendJudge:
    """A local judge model loaded in-process through the `Backend` protocol (tier 1).

    YES/NO templates are read from the next-token log-probabilities after the answer prefill
    ("Answer:"): p_yes = P(YES-like tokens) / P(YES-like + NO-like), summed over every
    single-token spelling (YES, Yes, yes, with and without a space). The label is p_yes > 0.5,
    so it can never disagree with its own probability. Generation is spent only where text is
    needed: the quote behind a YES (generated after a forced "Answer: YES\\nQuote:"), and names.

    Trait scores follow Persona Vectors: the expectation of the score under the judge's
    distribution over single-token integers 0-100. Their recipe needs every integer 0-100 to be
    one token (their judge.py docstring says only OpenAI's tokenizer guarantees it); a tokenizer
    that splits digits (Qwen) falls back to the greedy number, and the reply says so.
    """

    kind = "backend"
    external = False

    def __init__(self, backend: Backend, cfg: Any, *, lock: threading.Lock | None = None) -> None:
        info = backend.info
        check_judge_family(info.hf_id, cfg)
        self.backend = backend
        self.model = info.hf_id
        self.max_tokens = int(cfg["JUDGE_MAX_TOKENS"])
        self.batch = int(cfg["GEN_BATCH_MAX"])
        self.signature = (f"backend|{info.hf_id}|rev={info.revision}|template={info.chat_template_hash}"
                          f"|max_tokens={self.max_tokens}|greedy")
        self._lock = lock or threading.Lock()
        self._yes, self._no = self._answer_tokens()
        self._numbers = self._number_tokens()

    # ------------------------------------------------------------ token sets

    def _single_token_ids(self, words: Sequence[str]) -> set[int]:
        ids: set[int] = set()
        for w in words:
            for seq in self.backend.word_variants(w):
                if len(seq) == 1:
                    ids.add(int(seq[0]))
        return ids

    def _answer_tokens(self) -> tuple[frozenset[int], frozenset[int]]:
        yes = self._single_token_ids(("YES", "Yes", "yes"))
        no = self._single_token_ids(("NO", "No", "no"))
        if not yes or not no:
            raise JudgeSetupError(f"{self.model}: no single-token YES/NO; p_yes cannot be read")
        if yes & no:
            raise JudgeSetupError(f"{self.model}: a token is both YES-like and NO-like")
        return frozenset(yes), frozenset(no)

    def _number_tokens(self) -> dict[int, int] | None:
        table: dict[int, int] = {}
        for k in range(101):
            for form in (str(k), " " + str(k)):
                ids = self.backend.tokenize(form)
                if len(ids) == 1:
                    if table.get(ids[0], k) != k:
                        raise JudgeSetupError(f"{self.model}: token {ids[0]} reads as two numbers")
                    table[ids[0]] = k
        return table if set(table.values()) == set(range(101)) else None

    # ------------------------------------------------------------ helpers

    def _prompt(self, item: JudgeItem, prefill: str | None = None) -> str:
        rendered = self.backend.render(item.messages("lines"), add_generation_prompt=True)
        if rendered.count("<think>") > rendered.count("</think>"):
            raise JudgeSetupError(f"{self.model}: the rendered judge prompt opens a reasoning block; "
                                  "render the judge with thinking off (read_judges J-30)")
        return rendered + (item.template.prefill if prefill is None else prefill)

    def _generate(self, prompts: Sequence[str]) -> list[Any]:
        out = []
        for i in range(0, len(prompts), self.batch):
            reqs = [GenRequest(prompt=p, intervention=None, max_new_tokens=self.max_tokens,
                               temperature=0.0, top_p=1.0, top_k=0, seed=0)
                    for p in prompts[i:i + self.batch]]
            out.extend(self.backend.generate(reqs))
        return out

    def _logprobs(self, prompts: Sequence[str], ids: Sequence[int]) -> np.ndarray:
        rows = []
        for i in range(0, len(prompts), self.batch):
            chunk = prompts[i:i + self.batch]
            rows.append(np.asarray(self.backend.next_token_logprobs(chunk, [None] * len(chunk), ids),
                                   dtype=np.float64))
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, len(ids)))

    def _usage(self, prompt: str, gen: Any | None = None) -> dict[str, Any]:
        return {"input_tokens": len(self.backend.tokenize(prompt)),
                "output_tokens": 0 if gen is None else len(gen.token_ids)}

    # ------------------------------------------------------------ judging

    def judge(self, items: Sequence[JudgeItem]) -> list[Reply]:
        out: list[Reply | None] = [None] * len(items)
        by_kind: dict[str, list[int]] = {}
        for i, it in enumerate(items):
            by_kind.setdefault(it.template.answer, []).append(i)
        with self._lock:
            for kind, idx in by_kind.items():
                group = [items[i] for i in idx]
                t0 = time.monotonic()
                if kind in ("yesno", "yesno_quote"):
                    replies = self._judge_binary(group)
                elif kind == "name":
                    replies = self._judge_names(group)
                else:
                    replies = self._judge_scores(group)
                per = (time.monotonic() - t0) / max(1, len(group))
                for i, r in zip(idx, replies):
                    out[i] = replace(r, latency_s=per)
        return _complete(out, self.kind)

    def _judge_binary(self, items: Sequence[JudgeItem]) -> list[Reply]:
        yes, no = sorted(self._yes), sorted(self._no)
        ids = yes + no
        prompts = [self._prompt(it) for it in items]
        lp = self._logprobs(prompts, ids)
        p_tok = np.exp(lp)
        m_yes = p_tok[:, :len(yes)].sum(axis=1)
        m_no = p_tok[:, len(yes):].sum(axis=1)
        replies: list[Reply | None] = [None] * len(items)
        need_quote: list[int] = []
        for j, it in enumerate(items):
            mass = float(m_yes[j] + m_no[j])
            usage = self._usage(prompts[j])
            if not math.isfinite(mass) or mass < MIN_ANSWER_MASS:
                replies[j] = missing_reply(it, self, f"off_format: answer tokens carry {mass:.3f} of "
                                           "the next-token probability", usage=usage, cost_usd=0.0,
                                           fields={"answer_mass": mass})
                continue
            p = float(m_yes[j] / mass)
            if p == 0.5:
                replies[j] = missing_reply(it, self, "tie: P(YES) = P(NO)", p_yes=p, usage=usage,
                                           cost_usd=0.0, fields={"answer_mass": mass})
                continue
            label = p > 0.5
            if it.template.answer == "yesno_quote" and label:
                need_quote.append(j)
                replies[j] = None
                continue
            raw = f"Answer: {'YES' if label else 'NO'}"
            parsed = parse_reply(it.template, raw, source=it.quote_source)
            replies[j] = reply_from_parsed(it, self, parsed, p_yes=p, raw=None, usage=usage,
                                           cost_usd=0.0, notes=("logprob_read",))
            replies[j] = replace(replies[j], fields=dict(replies[j].fields, answer_mass=mass))
        if need_quote:
            # The YES is already decided by the probabilities; generation only fetches the
            # evidence, so it continues from a forced "Answer: YES" (greedy decoding could start
            # with a NO-like token when YES mass is split across spellings).
            qprompts = [prompts[j] + " YES\nQuote:" for j in need_quote]
            gens = self._generate(qprompts)
            for j, qp, gen in zip(need_quote, qprompts, gens):
                it = items[j]
                mass = float(m_yes[j] + m_no[j])
                raw = "Answer: YES\nQuote:" + gen.text
                parsed = parse_reply(it.template, raw, source=it.quote_source)
                r = reply_from_parsed(it, self, parsed, p_yes=float(m_yes[j] / mass), raw=raw,
                                      finish=gen.finish, usage=self._usage(qp, gen),
                                      cost_usd=0.0, notes=("logprob_read",))
                replies[j] = replace(r, fields=dict(r.fields, answer_mass=mass))
        return _complete(replies, self.kind)

    def _judge_names(self, items: Sequence[JudgeItem]) -> list[Reply]:
        prompts = [self._prompt(it) for it in items]
        gens = self._generate(prompts)
        out = []
        for it, p, gen in zip(items, prompts, gens):
            raw = it.template.prefill + gen.text
            parsed = parse_reply(it.template, raw)
            out.append(reply_from_parsed(it, self, parsed, raw=raw, finish=gen.finish,
                                         usage=self._usage(p, gen), cost_usd=0.0))
        return out

    def _judge_scores(self, items: Sequence[JudgeItem]) -> list[Reply]:
        prompts = [self._prompt(it) for it in items]
        replies: list[Reply | None] = [None] * len(items)
        greedy: list[tuple[int, str]] = []
        if self._numbers is not None:
            ids = sorted(self._numbers)
            values = np.array([self._numbers[t] for t in ids], dtype=np.float64)
            p_tok = np.exp(self._logprobs(prompts, ids))
            for j, it in enumerate(items):
                total = float(p_tok[j].sum())
                if total >= MIN_ANSWER_MASS:
                    score = float((p_tok[j] * values).sum() / total)
                    replies[j] = Reply(**_item_meta(it), client=self.kind, model=self.model,
                                       signature=self.signature, label=score,
                                       fields={"refusal": False, "score": score, "number_mass": total,
                                               "method": "logprob"},
                                       usage=self._usage(prompts[j]), cost_usd=0.0,
                                       notes=("logprob_read",))
                else:
                    greedy.append((j, "greedy_after_low_number_mass"))
        else:
            greedy = [(j, "greedy_digit_split_tokenizer") for j in range(len(items))]
        if greedy:
            gens = self._generate([prompts[j] for j, _ in greedy])
            for (j, method), gen in zip(greedy, gens):
                it = items[j]
                parsed = parse_reply(it.template, gen.text)
                r = reply_from_parsed(it, self, parsed, raw=gen.text, finish=gen.finish,
                                      usage=self._usage(prompts[j], gen), cost_usd=0.0,
                                      notes=(method,))
                replies[j] = replace(r, fields=dict(r.fields, method=method)) if r.ok else r
        return _complete(replies, self.kind)


# =====================================================================================
# HTTP plumbing shared by the OpenAI-compatible clients
# =====================================================================================

@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: Mapping[str, str]
    body: bytes


Opener = Callable[[str, bytes, Mapping[str, str], float], HttpResult]


def _urlopen(opener: urllib.request.OpenerDirector, url: str, body: bytes,
             headers: Mapping[str, str], timeout: float) -> HttpResult:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with opener.open(req, timeout=timeout) as resp:
            return HttpResult(resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read())
    except urllib.error.HTTPError as e:
        hdrs = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
        try:
            payload = e.read() or b""
        except Exception:
            payload = b""
        return HttpResult(e.code, hdrs, payload)
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException,
            OSError) as e:  # socket.timeout is TimeoutError since Python 3.10
        raise TransportError(f"{type(e).__name__}: {e}") from None


def urllib_opener() -> Opener:
    """HTTPS through the environment's proxy settings (the pod's egress path)."""
    od = urllib.request.build_opener()
    return lambda url, body, headers, timeout: _urlopen(od, url, body, headers, timeout)


def loopback_opener() -> Opener:
    """No proxy at all: a request to 127.0.0.1 must never be routed through an egress proxy,
    which would carry transcripts off the machine (CLAUDE.md, decision 12)."""
    od = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return lambda url, body, headers, timeout: _urlopen(od, url, body, headers, timeout)


def backoff_s(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(BACKOFF_CAP_S, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (attempt - 1)) * (1.0 + 0.25 * random.random())


def _snippet(body: bytes) -> str:
    return body[:300].decode("utf-8", "replace")


def _inner_error(obj: Any) -> tuple[int | None, str] | None:
    """OpenRouter can answer HTTP 200 with {"error": {...}} and no choices."""
    if isinstance(obj, dict) and "error" in obj and not obj.get("choices"):
        err = obj["error"] if isinstance(obj["error"], dict) else {"message": str(obj["error"])}
        code = err.get("code")
        try:
            code = int(code) if code is not None else None
        except (TypeError, ValueError):
            code = None
        return code, str(err.get("message", ""))[:300]
    return None


@dataclass
class _Outcome:
    obj: dict[str, Any] | None
    attempts: int
    missing: str | None
    detail: str | None = None


def post_json(url: str, body: Mapping[str, Any], headers: Mapping[str, str], *, opener: Opener,
              sleep: Callable[[float], None], retries: int, timeout: float) -> _Outcome:
    """POST with the retry policy of the module docstring. Raises BudgetExhausted and
    JudgeSetupError; returns a parsed JSON object or a missing reason otherwise."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last = "no attempt made"
    for attempt in range(1, retries + 2):
        try:
            res = opener(url, data, headers, timeout)
        except TransportError as e:
            last = f"transport: {e}"
            if attempt <= retries:
                sleep(backoff_s(attempt, None))
                continue
            return _Outcome(None, attempt, f"transient_exhausted: {last}")
        status = res.status
        obj: Any = None
        if status == 200:
            try:
                obj = json.loads(res.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return _Outcome(None, attempt, "unparseable: response body is not JSON",
                                _snippet(res.body))
            inner = _inner_error(obj)
            if inner is not None:
                status = inner[0] or 502
                last = f"inner error {inner[0]}: {inner[1]}"
        if status in BUDGET_STATUS:
            raise BudgetExhausted(f"HTTP 402 from {url}: judge account out of credit")
        if status in SETUP_STATUS:
            raise JudgeSetupError(f"HTTP {status} from {url}: {_snippet(res.body)[:120]}")
        if status in RETRY_STATUS:
            last = f"http_{status}"
            if attempt <= retries:
                sleep(backoff_s(attempt, res.headers.get("retry-after")))
                continue
            return _Outcome(None, attempt, f"transient_exhausted: {last}", _snippet(res.body))
        if status != 200:
            return _Outcome(None, attempt, f"http_{status}: request rejected", _snippet(res.body))
        if not isinstance(obj, dict):
            return _Outcome(None, attempt, "unparseable: response is not a JSON object")
        return _Outcome(obj, attempt, None)
    return _Outcome(None, retries + 1, f"transient_exhausted: {last}")


def _choice(obj: Mapping[str, Any]) -> tuple[str | None, str | None, Mapping[str, Any]]:
    choices = obj.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None, None, {}
    ch = choices[0]
    msg = ch.get("message") or {}
    return msg.get("content"), ch.get("finish_reason"), ch


def _run_parallel(fn: Callable[[Any], Reply], items: Sequence[Any], workers: int) -> list[Reply]:
    """Order-preserving parallel map that stops cleanly on a run-fatal error: workers that have
    not started skip their item, and the first fatal error is raised with every reply that did
    complete attached (so a cache can keep what was paid for)."""
    if not items:
        return []
    stop = threading.Event()
    fatal: list[JudgeError] = []
    results: list[Reply | None] = [None] * len(items)

    def run(i: int) -> None:
        if stop.is_set():
            return
        try:
            results[i] = fn(items[i])
        except JudgeError as e:
            stop.set()
            fatal.append(e)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        list(pool.map(run, range(len(items))))
    if fatal:
        e = fatal[0]
        raise type(e)(str(e), completed=[r for r in results if r is not None])
    return _complete(results, "parallel judge")


# =====================================================================================
# Tier 1 alternative: an OpenAI-compatible server on loopback (e.g. vLLM)
# =====================================================================================

def _top_masses(entry: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for alt in entry.get("top_logprobs") or []:
        tok = str(alt.get("token", ""))
        lp = alt.get("logprob")
        if lp is None:
            continue
        out[tok] = out.get(tok, 0.0) + math.exp(float(lp))
    return out


def p_yes_from_logprobs(content: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    """(p_yes, answer mass) from the first YES/NO token of a reply's top-logprobs."""
    for entry in content:
        tok = str(entry.get("token", "")).strip().strip("*_").upper()
        if tok in ("YES", "NO"):
            masses = _top_masses(entry)
            y = sum(p for t, p in masses.items() if t.strip().strip("*_").upper() == "YES")
            n = sum(p for t, p in masses.items() if t.strip().strip("*_").upper() == "NO")
            if y + n < MIN_ANSWER_MASS:
                return None, y + n
            return y / (y + n), y + n
    return None, None


def score_from_logprobs(content: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    """Persona Vectors' 0-100 aggregation on the first token: integers 0-100 among the top
    log-probabilities, weighted mean, None below MIN_ANSWER_MASS."""
    if not content:
        return None, None
    masses = _top_masses(content[0])
    total, acc = 0.0, 0.0
    for tok, p in masses.items():
        t = tok.strip()
        if t.isdigit() and 0 <= int(t) <= 100:
            total += p
            acc += int(t) * p
    if total < MIN_ANSWER_MASS:
        return None, total
    return acc / total, total


class LocalOpenAIJudge:
    """Tier 1 served by an OpenAI-compatible server on the loopback interface only (CLAUDE.md hard
    rule 2: an exposed endpoint on a rented pod is a public model). The request never goes through
    a proxy. Log-probabilities give p_yes (YES/NO templates) and Persona Vectors' expectation
    (trait); thinking is switched off in the chat template (read_judges J-30)."""

    kind = "local_openai"
    external = False

    def __init__(self, cfg: Any, *, url: str | None = None, model: str | None = None,
                 opener: Opener | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        base = cfg["JUDGE_T1_URL"] if url is None else url
        if not base:
            raise JudgeSetupError("JUDGE_T1_URL is empty: no local judge server; load the judge "
                                  "in-process with BackendJudge")
        safety.assert_local_url(base)
        self.endpoint = base.rstrip("/") + "/chat/completions"
        self.model = cfg.judge_t1_model if model is None else model
        check_judge_family(self.model, cfg)
        self.max_tokens = int(cfg["JUDGE_MAX_TOKENS"])
        self.retries = int(cfg["JUDGE_RETRIES"])
        self.timeout = float(cfg["JUDGE_TIMEOUT_S"])
        self.concurrency = int(cfg["JUDGE_CONCURRENCY"])
        self.opener = opener or loopback_opener()
        self.sleep = sleep
        self.signature = (f"local_openai|{self.model}|max_tokens={self.max_tokens}|temperature=0"
                          "|top_logprobs=20|thinking=off")

    def body(self, item: JudgeItem) -> dict[str, Any]:
        return {"model": self.model, "messages": item.messages("lines"), "temperature": 0,
                "max_tokens": self.max_tokens, "logprobs": True, "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False}}

    def _one(self, item: JudgeItem) -> Reply:
        t0 = time.monotonic()
        out = post_json(self.endpoint, self.body(item), {"Content-Type": "application/json",
                                                         "Accept": "application/json"},
                        opener=self.opener, sleep=self.sleep, retries=self.retries,
                        timeout=self.timeout)
        lat = time.monotonic() - t0
        if out.missing:
            return missing_reply(item, self, out.missing, attempts=out.attempts, latency_s=lat,
                                 fields={"error_detail": out.detail} if out.detail else {})
        obj = out.obj or {}
        content, finish, ch = _choice(obj)
        usage = obj.get("usage")
        common = dict(attempts=out.attempts, latency_s=lat, finish=finish, usage=usage,
                      served_model=obj.get("model"), cost_usd=0.0)
        if not content or not str(content).strip():
            return missing_reply(item, self, "empty_content", **common)
        if finish == "length":
            return missing_reply(item, self, "truncated: length stop", raw=content, **common)
        lp = ((ch.get("logprobs") or {}).get("content")) or []
        parsed = parse_reply(item.template, content, source=item.quote_source)
        if item.template.answer == "score":
            score, mass = score_from_logprobs(lp)
            if score is not None and parsed.ok and parsed.label != "REFUSAL":
                return reply_from_parsed(item, self, Parsed(score, {"refusal": False, "score": score,
                                                                    "number_mass": mass,
                                                                    "method": "logprob"}),
                                         raw=content, **common)
            return reply_from_parsed(item, self, parsed, raw=content, **common)
        p, mass = (None, None)
        if item.template.answer in ("yesno", "yesno_quote"):
            p, mass = p_yes_from_logprobs(lp)
        r = reply_from_parsed(item, self, parsed, raw=content, p_yes=p, **common)
        return replace(r, fields=dict(r.fields, answer_mass=mass)) if r.ok else r

    def judge(self, items: Sequence[JudgeItem]) -> list[Reply]:
        return _run_parallel(self._one, items, self.concurrency)


# =====================================================================================
# Tier 2: OpenRouter
# =====================================================================================

def _provider_slug(name: str | None) -> str:
    return "".join(ch for ch in str(name or "").split("/")[0].casefold() if ch.isalnum())


class OpenRouterJudge:
    """Tier 2 through OpenRouter (stdlib urllib). Reasoning is off in the outgoing body: the old
    pipeline was qualified with it off and run with it on, and 29 of 260 test calls came back
    empty or unparseable (read_judges J-30). Temperature 0. The provider is pinned with fallbacks
    disabled when JUDGE_T2_PROVIDER is set (plan §7.3, J3), and the provider that served each
    call is recorded either way (J-41)."""

    kind = "openrouter"
    external = True

    def __init__(self, cfg: Any, *, vector_classes: Sequence[str], model: str | None = None,
                 provider: str | None = None, api_key: str | None = None,
                 opener: Opener | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        safety.assert_external_judging_allowed(cfg["JUDGE_EXTERNAL"], vector_classes)
        self.url = safety.assert_external_judge_url(OPENROUTER_URL)
        self.model = cfg["JUDGE_T2_MODEL"] if model is None else model
        check_judge_family(self.model, cfg)
        self.provider = cfg["JUDGE_T2_PROVIDER"] if provider is None else provider
        key = os.environ.get("OPENROUTER_API_KEY", "") if api_key is None else api_key
        if not key.strip():
            raise JudgeSetupError("OPENROUTER_API_KEY is not set; tier 2 cannot run (checked at "
                                  "start, not mid-run: read_judges J-46)")
        self._key = key.strip()
        self.max_tokens = int(cfg["JUDGE_MAX_TOKENS"])
        self.retries = int(cfg["JUDGE_RETRIES"])
        self.timeout = float(cfg["JUDGE_TIMEOUT_S"])
        self.concurrency = int(cfg["JUDGE_CONCURRENCY"])
        self.opener = opener or urllib_opener()
        self.sleep = sleep
        self.signature = (f"openrouter|{self.model}|provider={self.provider or 'unpinned'}"
                          f"|max_tokens={self.max_tokens}|temperature=0|reasoning=off")

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                "Accept": "application/json"}

    def body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        b: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0,
                             "max_tokens": self.max_tokens, "reasoning": {"enabled": False},
                             "usage": {"include": True}}
        if self.provider:
            b["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        return b

    def _call(self, item: Any, messages: list[dict[str, str]]) -> tuple[Reply | None, dict[str, Any]]:
        t0 = time.monotonic()
        out = post_json(self.url, self.body(messages), self.headers(), opener=self.opener,
                        sleep=self.sleep, retries=self.retries, timeout=self.timeout)
        lat = time.monotonic() - t0
        if out.missing:
            return missing_reply(item, self, out.missing, attempts=out.attempts, latency_s=lat,
                                 fields={"error_detail": out.detail} if out.detail else {}), {}
        obj = out.obj or {}
        content, finish, _ = _choice(obj)
        usage = obj.get("usage") or {}
        served_provider = obj.get("provider")
        notes: list[str] = []
        reasoning = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        if reasoning:
            notes.append(f"reasoning_tokens={reasoning} despite reasoning off")
        common = dict(attempts=out.attempts, latency_s=lat, finish=finish, usage=usage,
                      served_model=obj.get("model"), provider=served_provider,
                      cost_usd=usage.get("cost"), notes=tuple(notes))
        if self.provider and _provider_slug(served_provider) != _provider_slug(self.provider):
            return missing_reply(item, self, f"provider_mismatch: pinned {self.provider!r}, served "
                                 f"{served_provider!r}", raw=content, **common), common
        if not content or not str(content).strip():
            return missing_reply(item, self, "empty_content", **common), common
        if finish == "length":
            return missing_reply(item, self, "truncated: length stop", raw=content, **common), common
        return None, dict(common, raw=content)

    def _one(self, item: JudgeItem) -> Reply:
        failed, common = self._call(item, item.messages("lines"))
        if failed is not None:
            return failed
        parsed = parse_reply(item.template, common["raw"], source=item.quote_source)
        return reply_from_parsed(item, self, parsed, **common)

    def judge(self, items: Sequence[JudgeItem]) -> list[Reply]:
        return _run_parallel(self._one, items, self.concurrency)


# =====================================================================================
# Tier 3: the Anthropic Messages API, official SDK
# =====================================================================================

def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def anthropic_cost(model: str | None, usage: Mapping[str, Any], *, batch: bool = False) -> float | None:
    if model not in ANTHROPIC_PRICES_USD_PER_MTOK:
        return None
    pin, pout = ANTHROPIC_PRICES_USD_PER_MTOK[model]
    cost = ((usage.get("input_tokens") or 0) * pin + (usage.get("output_tokens") or 0) * pout) / 1e6
    return cost * (BATCH_DISCOUNT if batch else 1.0)


class AnthropicJudge:
    """Tier 3 through the official `anthropic` SDK (imported only when this tier is built).

    Structured outputs (`output_config.format`) guarantee the first text block is JSON for the
    template's schema; effort comes from JUDGE_T3_EFFORT. Thinking cannot be disabled on this
    model and sampling parameters are rejected, so neither is sent; assistant prefill is not
    allowed. Server-side refusal fallbacks are on (`fallbacks="default"`), and a refusal that
    survives them is recorded as a missing judgment with its `stop_details.category` -- never as
    a NO. There are no log-probabilities on this API, so tier-3 labels are hard labels.

    Retries: the SDK retries 429/5xx itself (max_retries = JUDGE_RETRIES); errors that outlast it
    are classified here, most specific first, and transient ones retried up to JUDGE_RETRIES more
    times (honouring retry-after). Worst case (JUDGE_RETRIES + 1)^2 requests for one judgment.
    """

    kind = "anthropic"
    external = True

    def __init__(self, cfg: Any, *, vector_classes: Sequence[str], client: Any = None,
                 errors: Any = None, model: str | None = None, max_tokens: int | None = None,
                 batch: tuple[float, float] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """`batch=(poll_s, max_wait_s)` sends `judge()` through the Message Batches API (plan
        §7.3: tier 3 "via the batch API"); None uses the Messages API with refusal fallbacks."""
        safety.assert_external_judging_allowed(cfg["JUDGE_EXTERNAL"], vector_classes)
        safety.assert_external_judge_url(ANTHROPIC_HOST_URL)
        self.model = cfg["JUDGE_T3_MODEL"] if model is None else model
        check_judge_family(self.model, cfg)
        self.effort = cfg["JUDGE_T3_EFFORT"]
        # Thinking is always on for this model and counts against max_tokens, so tier 3 needs its
        # own ceiling (requested as config key JUDGE_T3_MAX_TOKENS; read by exact key).
        self.max_tokens = int(cfg["JUDGE_T3_MAX_TOKENS"]) if max_tokens is None else int(max_tokens)
        self.retries = int(cfg["JUDGE_RETRIES"])
        self.concurrency = int(cfg["JUDGE_CONCURRENCY"])
        self.sleep = sleep
        if client is None:
            try:
                import anthropic  # noqa: PLC0415 - only this tier needs it
            except ImportError:
                raise JudgeSetupError("tier 3 needs the official Anthropic SDK: pip install anthropic") from None
            client = anthropic.Anthropic(max_retries=self.retries, timeout=float(cfg["JUDGE_TIMEOUT_S"]))
            errors = anthropic
        elif errors is None:
            raise JudgeSetupError("an injected Anthropic client needs its errors namespace")
        self.client, self.errors = client, errors
        self.batch = batch
        # Batches reject `fallbacks`, so a batch judgment is a different request: its own key.
        route = "batch|fallbacks=none" if batch is not None else "fallbacks=default"
        self.signature = (f"anthropic|{self.model}|effort={self.effort}|max_tokens={self.max_tokens}"
                          f"|{route}|format=json_schema")

    def params(self, item: JudgeItem) -> dict[str, Any]:
        return {"model": self.model, "max_tokens": self.max_tokens,
                "messages": item.messages("json"),
                "output_config": {"effort": self.effort,
                                  "format": {"type": "json_schema",
                                             "schema": json.loads(json.dumps(item.template.json_schema))}}}

    # ------------------------------------------------------------ errors

    def _retry_after(self, exc: Any) -> str | None:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        if headers is None:
            return None
        try:
            return headers.get("retry-after")
        except Exception:
            return None

    def _classify(self, exc: BaseException) -> tuple[str, str]:
        """('retry' | 'missing', reason) for one SDK exception; raises the run-fatal ones."""
        E = self.errors
        if isinstance(exc, E.RateLimitError):
            return "retry", "http_429: rate_limit_error"
        if isinstance(exc, E.BadRequestError):
            return "missing", f"http_400: {getattr(exc, 'message', str(exc))[:200]}"
        if isinstance(exc, E.APIStatusError):
            status = getattr(exc, "status_code", None)
            typ = getattr(exc, "type", None)
            if status == 402 or typ == "billing_error":
                raise BudgetExhausted(f"Anthropic API: {typ or 'HTTP 402'}: out of credit")
            if status in (401, 403, 404) or typ in ("authentication_error", "permission_error",
                                                    "not_found_error"):
                raise JudgeSetupError(f"Anthropic API: HTTP {status} {typ}")
            if status == 529 or typ == "overloaded_error" or (status or 0) >= 500:
                return "retry", f"http_{status}: {typ}"
            return "missing", f"http_{status}: {typ}"
        if isinstance(exc, E.APIConnectionError):
            return "retry", f"transport: {type(exc).__name__}"
        raise exc

    # ------------------------------------------------------------ one judgment

    def _message_reply(self, item: JudgeItem, msg: Any, *, attempts: int, latency: float,
                       batch: bool = False) -> Reply:
        stop = _get(msg, "stop_reason")
        served = _get(msg, "model")
        u = _get(msg, "usage")
        usage = {k: _get(u, k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                         "cache_creation_input_tokens")} if u is not None else {}
        notes: list[str] = []
        iters = _get(u, "iterations") if u is not None else None
        if iters and any(_get(x, "type") == "fallback_message" for x in iters):
            notes.append("fallback_ran")
        common = dict(attempts=attempts, latency_s=latency, finish=stop, usage=usage,
                      served_model=served, cost_usd=anthropic_cost(served or self.model, usage, batch=batch),
                      notes=tuple(notes))
        if stop == "refusal":
            det = _get(msg, "stop_details")
            category = _get(det, "category") if det is not None else None
            return missing_reply(item, self, f"refusal: {category}",
                                 fields={"stop_details": {"category": category,
                                                          "explanation": _get(det, "explanation") if det is not None else None}},
                                 **common)
        if stop == "max_tokens":
            return missing_reply(item, self, "truncated: max_tokens", **common)
        text = None
        for block in _get(msg, "content") or []:
            if _get(block, "type") == "text":
                text = _get(block, "text")
                break
        if not text or not str(text).strip():
            return missing_reply(item, self, "empty_content", **common)
        parsed = parse_reply(item.template, text, fmt="json", source=item.quote_source)
        return reply_from_parsed(item, self, parsed, raw=text, **common)

    def _one(self, item: JudgeItem) -> Reply:
        params = self.params(item)
        last = "no attempt made"
        t0 = time.monotonic()
        for attempt in range(1, self.retries + 2):
            try:
                msg = self.client.beta.messages.create(**params, betas=[FALLBACK_BETA],
                                                       fallbacks="default")
            except Exception as exc:  # classified, most specific first; unknown ones re-raised
                action, why = self._classify(exc)
                last = why
                if action == "missing":
                    return missing_reply(item, self, why, attempts=attempt,
                                         latency_s=time.monotonic() - t0)
                if attempt <= self.retries:
                    self.sleep(backoff_s(attempt, self._retry_after(exc)))
                    continue
                return missing_reply(item, self, f"transient_exhausted: {last}", attempts=attempt,
                                     latency_s=time.monotonic() - t0)
            return self._message_reply(item, msg, attempts=attempt, latency=time.monotonic() - t0)
        return missing_reply(item, self, f"transient_exhausted: {last}", attempts=self.retries + 1)

    def judge(self, items: Sequence[JudgeItem]) -> list[Reply]:
        if self.batch is not None:
            poll_s, max_wait_s = self.batch
            return self.judge_batch(items, poll_s=poll_s, max_wait_s=max_wait_s)
        return _run_parallel(self._one, items, self.concurrency)

    # ------------------------------------------------------------ optional: Message Batches

    def judge_batch(self, items: Sequence[JudgeItem], *, poll_s: float, max_wait_s: float) -> list[Reply]:
        """The same judgments through the Message Batches API (half price, asynchronous).

        The batch endpoint rejects `fallbacks`, so a refusal here is final and recorded as
        missing. Results come back in any order and are keyed by custom_id (the item key). A
        batch that has not ended within `max_wait_s` leaves its items missing ("batch_not_ended"),
        to be re-judged on resume; billing errors stop the run as everywhere else.
        """
        if not items:
            return []
        requests, seen = [], set()
        for it in items:
            if it.key not in seen:
                seen.add(it.key)
                requests.append({"custom_id": it.key, "params": self.params(it)})
        t0 = time.monotonic()
        try:
            batch = self.client.messages.batches.create(requests=requests)
        except Exception as exc:
            action, why = self._classify(exc)
            return [missing_reply(it, self, f"batch_create_failed: {why}") for it in items]
        waited = 0.0
        while _get(batch, "processing_status") != "ended":
            if waited >= max_wait_s:
                return [missing_reply(it, self, f"batch_not_ended: {_get(batch, 'id')}") for it in items]
            self.sleep(poll_s)
            waited += poll_s
            batch = self.client.messages.batches.retrieve(_get(batch, "id"))
        results = {_get(r, "custom_id"): r for r in self.client.messages.batches.results(_get(batch, "id"))}
        lat = (time.monotonic() - t0) / len(items)
        out = []
        for it in items:
            r = results.get(it.key)
            if r is None:
                out.append(missing_reply(it, self, "batch_result_absent"))
                continue
            res = _get(r, "result")
            typ = _get(res, "type")
            if typ == "succeeded":
                out.append(self._message_reply(it, _get(res, "message"), attempts=1, latency=lat, batch=True))
                continue
            if typ == "errored":
                err = _get(res, "error")
                etype = _get(err, "type")
                inner = _get(err, "error")
                if inner is not None:
                    etype = _get(inner, "type") or etype
                if etype == "billing_error":
                    raise BudgetExhausted("Anthropic batch: billing_error", completed=out)
                out.append(missing_reply(it, self, f"batch_errored: {etype}"))
                continue
            out.append(missing_reply(it, self, f"batch_{typ}"))
        return out


# =====================================================================================
# Reproduction mode: Macar et al.'s judge
# =====================================================================================

class MacarJudge:
    """Macar et al.'s verbatim prompts and parser (m5.judges.macar) on their judge model, through
    OpenRouter, with their request settings (temperature 0, 500 tokens, one user message). Used
    only to reproduce their 38.2% under their own criteria (plan Phase 1). Returns their fields;
    an unparseable reply is missing here and scored both ways by `macar.macar_metrics`."""

    kind = "macar"
    external = True

    def __init__(self, cfg: Any, *, vector_classes: Sequence[str], model: str = macar.MACAR_JUDGE_MODEL,
                 api_key: str | None = None, opener: Opener | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        safety.assert_external_judging_allowed(cfg["JUDGE_EXTERNAL"], vector_classes)
        self.url = safety.assert_external_judge_url(OPENROUTER_URL)
        self.model = model
        check_judge_family(self.model, cfg)
        key = os.environ.get("OPENROUTER_API_KEY", "") if api_key is None else api_key
        if not key.strip():
            raise JudgeSetupError("OPENROUTER_API_KEY is not set; the Macar judge cannot run")
        self._key = key.strip()
        self.retries = int(cfg["JUDGE_RETRIES"])
        self.timeout = float(cfg["JUDGE_TIMEOUT_S"])
        self.concurrency = int(cfg["JUDGE_CONCURRENCY"])
        self.opener = opener or urllib_opener()
        self.sleep = sleep
        self.signature = (f"macar|{self.model}|max_tokens={macar.MACAR_MAX_TOKENS}"
                          f"|temperature={macar.MACAR_TEMPERATURE}|upstream={macar.UPSTREAM_COMMIT[:8]}")

    def body(self, item: macar.MacarItem) -> dict[str, Any]:
        return {"model": self.model, "messages": item.messages(), "temperature": macar.MACAR_TEMPERATURE,
                "max_tokens": macar.MACAR_MAX_TOKENS, "usage": {"include": True}}

    def _one(self, item: macar.MacarItem) -> Reply:
        t0 = time.monotonic()
        out = post_json(self.url, self.body(item), {"Authorization": f"Bearer {self._key}",
                                                    "Content-Type": "application/json",
                                                    "Accept": "application/json"},
                        opener=self.opener, sleep=self.sleep, retries=self.retries,
                        timeout=self.timeout)
        lat = time.monotonic() - t0
        if out.missing:
            return missing_reply(item, self, out.missing, attempts=out.attempts, latency_s=lat)
        obj = out.obj or {}
        content, finish, _ = _choice(obj)
        usage = obj.get("usage") or {}
        common = dict(attempts=out.attempts, latency_s=lat, finish=finish, usage=usage,
                      served_model=obj.get("model"), provider=obj.get("provider"),
                      cost_usd=usage.get("cost"))
        if not content or not str(content).strip():
            return missing_reply(item, self, "empty_content", **common)
        if finish == "length":
            return missing_reply(item, self, "truncated: length stop", raw=content, **common)
        label, fields_, why = macar.read(item.criterion, content)
        return Reply(**_item_meta(item), client=self.kind, model=self.model,
                     signature=self.signature, label=label, fields=fields_, missing=why,
                     raw=content, external=True, **common)

    def judge(self, items: Sequence[macar.MacarItem]) -> list[Reply]:
        return _run_parallel(self._one, items, self.concurrency)
