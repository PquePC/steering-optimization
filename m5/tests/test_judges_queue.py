"""m5.judges.queue: the reply store and cache, and the threaded queue."""

from __future__ import annotations

import threading
import time

import pytest

from m5 import runio, safety
from m5.judges import clients as C
from m5.judges import prompts as P
from m5.judges.cascade import Cascade
from m5.judges.queue import CachedClient, JudgeQueue, JudgeStore, cache_key
from m5.tests._judges_fakes import FakeClient


@pytest.fixture
def T(cfg):
    return P.templates(cfg)


def _item(T, text, iid):
    return P.make_item(T["dominant"], {"concept": "silk", "text": text}, item_id=iid)


def _client(label=True, name="Qwen/Qwen3-8B"):
    return FakeClient("backend", name, False, lambda it: (label, 0.97, {"answer": "YES" if label else "NO"}))


def test_a_repeated_item_is_a_cache_hit(T):
    store = JudgeStore.memory()
    fake = _client()
    cc = CachedClient(fake, store, tier="t1")
    first = cc.judge([_item(T, "a silk scarf", "trial-1")])[0]
    again = cc.judge([_item(T, "a silk scarf", "trial-2")])[0]
    assert fake.n_items == 1
    assert not first.cached and again.cached and again.item_id == "trial-2"
    assert again.label is first.label
    # duplicates inside one batch are judged once
    fake2 = _client()
    CachedClient(fake2, JudgeStore.memory(), tier="t1").judge(
        [_item(T, "same", "x"), _item(T, "same", "y"), _item(T, "other", "z")])
    assert fake2.n_items == 2


def test_the_key_covers_the_client_signature_and_the_bound_template(cfg, T):
    it = _item(T, "a silk scarf", "1")
    a, b = _client(), _client(name="mistralai/judge-x")
    assert cache_key(it, a) != cache_key(it, b)
    other = P.templates(cfg.replace(QUOTE_MAX_WORDS=9))
    q15 = P.make_item(T["allusion"], {"concept": "silk", "text": "x"}, item_id="1")
    q9 = P.make_item(other["allusion"], {"concept": "silk", "text": "x"}, item_id="1")
    assert cache_key(q15, a) != cache_key(q9, a)


def test_failures_are_recorded_but_never_cached(T):
    store = JudgeStore.memory()
    calls = []

    def script(it):
        calls.append(it.item_id)
        return "transient_exhausted: http_503" if len(calls) == 1 else (True, 0.9, {})
    cc = CachedClient(FakeClient("backend", "Qwen/Qwen3-8B", False, script), store, tier="t1")
    assert cc.judge([_item(T, "x", "1")])[0].missing
    assert cc.judge([_item(T, "x", "1")])[0].label is True          # re-judged, not served
    assert [r["missing"] is None for r in store.rows] == [False, True]


def test_a_resumed_run_reuses_its_replies_from_the_judge_calls_stream(cfg, run_root, T):
    run = runio.RunDir.create(run_root, "r1", config=cfg, kind="test")
    fake = _client()
    CachedClient(fake, JudgeStore.for_run(run), tier="t1").judge([_item(T, "a silk scarf", "1")])
    run.close()
    resumed = runio.RunDir.open(run.path)
    fake2 = _client()
    r = CachedClient(fake2, JudgeStore.for_run(resumed), tier="t1").judge([_item(T, "a silk scarf", "1")])[0]
    assert fake2.n_items == 0 and r.cached
    assert runio.STREAMS["judge_calls"][0] == "internal"


def test_a_directory_store_survives_reopening_and_refuses_the_repository(tmp_path, T):
    s1 = JudgeStore.at(tmp_path / "store")
    CachedClient(_client(), s1, tier="t1").judge([_item(T, "silk", "1")])
    s1.close()
    s2 = JudgeStore.at(tmp_path / "store")
    assert len(s2) == 1
    with pytest.raises(safety.Refused):
        JudgeStore.at(safety.REPO_ROOT / "judge_store")


def test_use_cache_false_rejudges_deliberately(T):
    store = JudgeStore.memory()
    fake = _client()
    CachedClient(fake, store, tier="t1").judge([_item(T, "silk", "1")])
    CachedClient(fake, store, tier="t1", use_cache=False).judge([_item(T, "silk", "1")])
    assert fake.n_items == 2


def test_replies_paid_for_before_a_402_are_kept(T):
    store = JudgeStore.memory()
    paid = C.Reply(item_key=_item(T, "a", "1").key, item_id="1", template_id="dominant",
                   template_version=1, template_hash="h", client="openrouter", model="m",
                   signature="openrouter|m", label=True)

    class Broke:
        kind, model, signature, external = "openrouter", "m", "openrouter|m", True

        def judge(self, items):
            raise C.BudgetExhausted("402", completed=[paid])
    with pytest.raises(C.BudgetExhausted):
        CachedClient(Broke(), store, tier="t2").judge([_item(T, "a", "1"), _item(T, "b", "2")])
    assert len(store) == 1 and store.rows[0]["item_id"] == "1"


def test_the_queue_overlaps_batches_keeps_order_and_accounts_local_vs_external(cfg, T):
    ext = cfg.replace(JUDGE_EXTERNAL=True, JUDGE_AUDIT_FRACTION=0.0)

    def t1(it):
        return (True, 0.6, {}) if "unsure" in it.payload["text"] else (True, 0.97, {})
    cascade = Cascade(ext, t1=FakeClient("backend", "Qwen/Qwen3-8B", False, t1),
                      t2=FakeClient("openrouter", "deepseek/deepseek-v4-flash", True, lambda it: (True, None, {})),
                      vector_classes=["concept"], store=JudgeStore.memory())
    items = [_item(T, ("unsure " if k % 5 == 0 else "") + f"text {k}", f"i{k}") for k in range(23)]
    with JudgeQueue(cascade.judge, workers=3, batch_size=4) as q:
        futures = q.submit(items)
        judgments = q.results(futures)
        acct = q.accounting()
    assert [j.item_id for j in judgments] == [it.item_id for it in items]
    assert acct["judgments"] == 23 and acct["judgments_with_external_calls"] == 5
    assert acct["judgments_local_only"] == 18 and acct["calls_external"] == 5
    assert acct["cost_usd"] == pytest.approx(0.005)


def test_a_fatal_error_latches_and_fails_every_later_batch(T):
    started = threading.Event()

    def judge(batch):
        started.set()
        if any(it.item_id == "boom" for it in batch):
            raise C.BudgetExhausted("402")
        time.sleep(0.01)
        return [object() for _ in batch]
    q = JudgeQueue(judge, workers=1, batch_size=1)
    futs = q.submit([_item(T, "x", "boom"), _item(T, "y", "later")])
    with pytest.raises(C.BudgetExhausted):
        futs[0].result()
    with pytest.raises(C.BudgetExhausted):
        futs[1].result()
    assert isinstance(q.stopped, C.BudgetExhausted)
    with pytest.raises(C.BudgetExhausted):
        q.submit([_item(T, "z", "after")])
    q.close()


def test_a_bug_fails_its_batch_loudly_without_latching(T):
    def judge(batch):
        if batch[0].item_id == "bad":
            raise ZeroDivisionError("bug")
        return ["ok"] * len(batch)
    with JudgeQueue(judge, workers=1, batch_size=1) as q:
        bad, good = q.submit([_item(T, "x", "bad"), _item(T, "y", "good")])
        with pytest.raises(ZeroDivisionError):
            bad.result()
        assert good.result() == "ok" and q.stopped is None
