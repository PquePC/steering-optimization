"""m5.judges.queue - the reply store and cache, and a thread-pool queue so judging overlaps
generation (plan §8.1, "overlapped judging"; read_judges J-40).

**The store** is the durable record of every judge reply and every final judgment, and the cache
is an index over it. In a run both go to the `judge_calls` stream (export class `internal`: the
rows embed transcripts, `m5.runio`), so no unclassified file is ever created (inventory H4). A
resumed run rebuilds the index from that stream and pays for nothing twice; the old M3 cache
lived in process memory, so a killed run re-paid every unfinished cell (read_judges J-26).

The cache key is content-addressed (read_judges J-25): the item key (template id, version,
bound-template hash, payload) plus the client's signature (model, provider pin, every setting
that shapes the request body). A missing `judge_id` once made S1 return E5's cached row -- "a
wrong number, not an error" -- and a missing model forced the old bake-off to smuggle the model
into another field. Only successful replies are indexed: a failure is recorded but never cached,
so a resumed run re-judges exactly the items that have no verdict (plan I4). `use_cache=False`
re-judges deliberately (stability gates, read_judges J-27).

**The queue** runs a judge function (normally `Cascade.judge`) on a thread pool. Measures submit
each batch of transcripts as it is generated and collect the futures later, so the GPU keeps
generating while judges answer. A run-fatal error (BudgetExhausted, JudgeSetupError) latches:
every pending and later batch fails with it, and nothing after it is recorded as judged.
"""

from __future__ import annotations

import collections
import hashlib
import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from m5 import SCHEMA, runio, safety
from m5.judges.clients import JudgeError, Reply


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def cache_key(item: Any, client: Any) -> str:
    blob = json.dumps(["m5.judges.cache/1", item.key, client.signature], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class JudgeStore:
    """Every reply and judgment, written as rows through `sink`; the cache index over replies."""

    def __init__(self, sink: Callable[[dict[str, Any]], None],
                 rows: Iterable[Mapping[str, Any]] = ()) -> None:
        self._sink = sink
        self._lock = threading.Lock()
        self._index: dict[str, Reply] = {}
        for row in rows:
            if row.get("kind") == "reply" and row.get("missing") is None and row.get("cache_key"):
                self._index[str(row["cache_key"])] = Reply.from_row(row)

    # ------------------------------------------------------------ constructors

    @classmethod
    def for_run(cls, run: runio.RunDir) -> "JudgeStore":
        """Rows go to the run's `judge_calls` stream; the cache is rebuilt from it on resume."""
        return cls(lambda row: run.append("judge_calls", row), run.rows("judge_calls"))

    @classmethod
    def at(cls, directory: str | Path) -> "JudgeStore":
        """A store in a directory outside the repository (Phase 0 work that has no run)."""
        d = safety.assert_outside_repo(directory, what="judge store directory")
        d.mkdir(parents=True, exist_ok=True)
        path = d / "judge_calls.jsonl"
        rows = runio.read_jsonl(path).rows
        fh = open(path, "a", encoding="utf-8", buffering=1)

        def sink(row: dict[str, Any]) -> None:
            stamped = {"schema": SCHEMA, "ts": runio.utc_now(), **row}
            fh.write(json.dumps(stamped, ensure_ascii=False, default=_json_default) + "\n")
            fh.flush()

        store = cls(sink, rows)
        store._handle = fh  # type: ignore[attr-defined]
        return store

    @classmethod
    def memory(cls) -> "JudgeStore":
        """No persistence: tests, and dry runs whose replies may be lost."""
        rows: list[dict[str, Any]] = []
        store = cls(rows.append)
        store.rows = rows  # type: ignore[attr-defined]
        return store

    # ------------------------------------------------------------ use

    def get(self, key: str) -> Reply | None:
        with self._lock:
            return self._index.get(key)

    def put_reply(self, reply: Reply, *, tier: str, key: str) -> None:
        row = {"kind": "reply", "tier": tier, "cache_key": key, **reply.to_row()}
        with self._lock:
            self._sink(row)
            if reply.ok:
                self._index[key] = reply

    def put_judgment(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            self._sink({"kind": "judgment", **row})

    def __len__(self) -> int:
        return len(self._index)

    def close(self) -> None:
        """Close the file a directory store appends to (a run's store closes with the run)."""
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None


class CachedClient:
    """A judge client behind the store. Cache hits are returned marked `cached`; misses are
    judged once per distinct key, recorded, and returned in input order."""

    def __init__(self, client: Any, store: JudgeStore, *, tier: str, use_cache: bool = True) -> None:
        self.client = client
        self.store = store
        self.tier = tier
        self.use_cache = use_cache
        self.kind = client.kind
        self.model = client.model
        self.signature = client.signature
        self.external = client.external

    def judge(self, items: Sequence[Any]) -> list[Reply]:
        keys = [cache_key(it, self.client) for it in items]
        out: list[Reply | None] = [None] * len(items)
        todo: dict[str, list[int]] = {}
        for i, (it, k) in enumerate(zip(items, keys)):
            hit = self.store.get(k) if self.use_cache else None
            if hit is not None and hit.item_key == it.key:
                out[i] = replace(hit, cached=True, item_id=it.item_id, latency_s=0.0)
            else:
                todo.setdefault(k, []).append(i)
        if todo:
            firsts = [idx[0] for idx in todo.values()]
            try:
                fresh = self.client.judge([items[i] for i in firsts])
            except JudgeError as e:
                for r in e.completed:  # paid for before the stop: keep them (read_judges J-13)
                    self.store.put_reply(r, tier=self.tier, key=cache_key(_KeyOnly(r.item_key), self.client))
                raise
            if len(fresh) != len(firsts):
                raise RuntimeError(f"{self.kind}: {len(fresh)} replies for {len(firsts)} items")
            for (k, idx), r in zip(todo.items(), fresh):
                if r.item_key != items[idx[0]].key:
                    raise RuntimeError(f"{self.kind}: reply for {r.item_key} returned in the place of "
                                       f"{items[idx[0]].key} (results out of order, read_judges J-28)")
                self.store.put_reply(r, tier=self.tier, key=k)
                for n, i in enumerate(idx):
                    out[i] = r if n == 0 else replace(r, item_id=items[i].item_id, cached=True,
                                                      latency_s=0.0)
        if any(r is None for r in out):
            raise RuntimeError(f"{self.kind}: an item was left without a reply")
        return list(out)  # type: ignore[arg-type]


class _KeyOnly:
    def __init__(self, key: str) -> None:
        self.key = key


# =====================================================================================
# The queue
# =====================================================================================

class JudgeQueue:
    """Run `judge(batch) -> list[result]` on a thread pool; one future per item.

    `workers` is how many batches are judged at once (throughput only). With a local tier-1
    judge on the same GPU, 1-2 workers are enough: the queue's job is to let generation continue
    while a batch is judged, not to run the GPU twice.
    """

    def __init__(self, judge: Callable[[list[Any]], list[Any]], *, workers: int,
                 batch_size: int) -> None:
        if workers < 1 or batch_size < 1:
            raise ValueError("workers and batch_size must be positive")
        self._judge = judge
        self._batch = batch_size
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="m5-judge")
        self._lock = threading.Lock()
        self._fatal: BaseException | None = None
        self._counts: collections.Counter[str] = collections.Counter()
        self._cost = 0.0

    @property
    def stopped(self) -> BaseException | None:
        return self._fatal

    def submit(self, items: Sequence[Any]) -> list[Future]:
        with self._lock:
            if self._fatal is not None:
                raise self._fatal
        futures: list[Future] = [Future() for _ in items]
        for start in range(0, len(items), self._batch):
            chunk = list(range(start, min(len(items), start + self._batch)))
            self._pool.submit(self._run, [items[i] for i in chunk], [futures[i] for i in chunk])
        return futures

    def _run(self, items: list[Any], futures: list[Future]) -> None:
        with self._lock:
            fatal = self._fatal
        if fatal is not None:
            for f in futures:
                f.set_exception(fatal)
            return
        try:
            results = self._judge(items)
            if len(results) != len(items):
                raise RuntimeError(f"judge returned {len(results)} results for {len(items)} items")
        except JudgeError as e:
            with self._lock:
                self._fatal = self._fatal or e
            for f in futures:
                f.set_exception(e)
            return
        except BaseException as e:  # a bug: fail this batch loudly, do not latch
            for f in futures:
                f.set_exception(e)
            return
        for f, r in zip(futures, results):
            self._account(r)
            f.set_result(r)

    def _account(self, judgment: Any) -> None:
        trail = getattr(judgment, "trail", ())
        with self._lock:
            self._counts["judgments"] += 1
            if getattr(judgment, "missing", None) is not None:
                self._counts["missing"] += 1
            decided = getattr(judgment, "decided_by", None)
            if decided:
                self._counts[f"decided_by_{decided}"] += 1
            external_fresh = [s for s in trail if s.external and not s.cached and s.called]
            if external_fresh:
                self._counts["judgments_with_external_calls"] += 1
            else:
                self._counts["judgments_local_only"] += 1
            for s in trail:
                if not s.called:
                    continue
                self._counts[f"calls_{'external' if s.external else 'local'}"] += 1
                if s.cached:
                    self._counts["calls_cached"] += 1
                if s.cost_usd and not s.cached:
                    self._cost += float(s.cost_usd)

    def accounting(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = dict(self._counts)
            out["cost_usd"] = round(self._cost, 6)
            return out

    @staticmethod
    def results(futures: Sequence[Future]) -> list[Any]:
        return [f.result() for f in futures]

    def judge_all(self, items: Sequence[Any]) -> list[Any]:
        return self.results(self.submit(items))

    def close(self, *, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> "JudgeQueue":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
