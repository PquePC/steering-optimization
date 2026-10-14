"""m5.measures.s5 - `s5`, the paired capability panel: does the steered model still get things right?

The same auto-verifiable items answered steered and unsteered (plan §3.2): a paired comparison,
so only the items one arm got right and the other wrong carry information, and the test is the
exact McNemar test (`m5.stats.paired`) with Tango's score interval on the paired difference.

**The panel** is the per-model choice the setup made (plan §5.1 step 5: about 600 candidates run
unsteered, the items the model passes S5_PASS_BAND of the time kept, so the panel can neither
saturate nor floor -- Mentor 1's objection), read from `ctx.profile.selection("capability")`.
The profile holds ids, seeds, paths and digests, never the items, so the panel is rebuilt here
from the source the selection names and checked against it. The shape, agreed with the setup:

    {"source": "generate", "seed": int, "n_candidates": int,      # m5.data.capability.generate
     "families": [...], "difficulties": [...],                     #   (optional: its defaults)
     "item_ids": [kept ids, in panel order],                       # the measure takes the first n
     "pass_rates": {id: unsteered pass rate},                      # how they were chosen
     "items_sha256": panel_digest(kept items),                     # optional; over item_ids
     "splits": {"search": [ids], "confirm": [ids]},                # optional per-split panels
     ...}                                                          # anything else: provenance

    a benchmark source instead: {"source": "mmlu_pro" | "gsm8k" | "gpqa_diamond",
     "path": file outside the repository, "sha256": the file's digest, "limit": int | None,
     "categories": [...] (mmlu_pro), "seed": int (gpqa_diamond, its option shuffle), ...}

    several sources: {"pools": [<source mappings>], "item_ids": [...], ...}

An id the rebuilt pool does not contain, a file whose digest differs, or a panel whose digest
differs raises: the panel the setup chose is the panel measured, or nothing is. **Without a
selection** -- and only then -- the items come from `generate`, seeded per split family (the
confirm tier draws a different panel from the search); `extra["panel"]` says so, an `events` row
records it once per run, and those items may saturate, which the selection exists to prevent.
`items=` overrides both.

Each item is answered at TOKENS_CAPABILITY with every position steered, and scored by its own
checker (`CapabilityItem.check`, the text after the last "Answer:"). A reply with no "Answer:" fails
-- under steering, losing the format is losing the instruction, and the unsteered arm is held to
the same rule -- and is counted apart from a wrong answer (`extra["no_answer"]`).

The Measured value is the steered pass rate (k / n) with the unsteered pass rate on the same items
as its null; `extra["paired"]` holds the paired difference (steered - unsteered), its interval and
the McNemar p-value, and `extra["relative"]` the steered rate as a share of the unsteered one (the
term `s4` reads).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from m5 import prompts as _prompts
from m5.data import capability as _cap
from m5.intervention import Cell
from m5.judges import mechanical
from m5.measures import common as C
from m5.results import Measured
from m5.safety import assert_outside_repo

CODE = "s5"
SELECTION = "capability"
SOURCES = ("generate", "mmlu_pro", "gsm8k", "gpqa_diamond")


def _item(x: Any) -> _cap.CapabilityItem:
    if isinstance(x, _cap.CapabilityItem):
        return x
    if isinstance(x, Mapping):
        return _cap.CapabilityItem(id=str(x["id"]), family=str(x["family"]), question=str(x["question"]),
                                   answer=str(x["answer"]), kind=str(x["kind"]),
                                   params=tuple(tuple(p) for p in x.get("params", ())),
                                   source=str(x.get("source", "generated")),
                                   difficulty=int(x.get("difficulty", 0)))
    raise TypeError(f"a capability item is a CapabilityItem or its JSON form, got {type(x).__name__}")


def panel_digest(items: Sequence[_cap.CapabilityItem]) -> str:
    """sha256 over the panel's (id, question, answer) in order: what `items_sha256` records."""
    blob = json.dumps([[it.id, it.question, it.answer] for it in items], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _need(src: Mapping[str, Any], key: str) -> Any:
    if key not in src:
        raise C.MeasureError(f"the capability selection's {src.get('source')!r} source does not record "
                             f"{key!r}; the panel cannot be rebuilt without it (m5.measures.s5)")
    return src[key]


def rebuild_pool(src: Mapping[str, Any]) -> tuple[_cap.CapabilityItem, ...]:
    """The candidate items one recorded source produces, exactly as the setup drew them."""
    kind = src.get("source")
    if kind == "generate":
        kw: dict[str, Any] = {}
        if "families" in src:
            kw["families"] = tuple(src["families"])
        if "difficulties" in src:
            kw["difficulties"] = tuple(int(d) for d in src["difficulties"])
        return _cap.generate(int(_need(src, "n_candidates")), int(_need(src, "seed")), **kw)
    if kind not in SOURCES:
        raise C.MeasureError(f"unknown capability source {kind!r}; one of {SOURCES}")
    path = assert_outside_repo(_need(src, "path"), what="capability benchmark file")
    if not path.is_file():
        raise C.MeasureError(f"the capability selection's {kind} file {path} does not exist")
    digest = C.sha256_file(path)
    if digest != _need(src, "sha256"):
        raise C.MeasureError(f"{path} has sha256 {digest[:12]}..., not the {str(src['sha256'])[:12]}... "
                             "the capability selection recorded: it is not the file the setup read")
    limit = src.get("limit")
    if kind == "mmlu_pro":
        load = _cap.load_mmlu_pro(path, limit=limit, categories=src.get("categories"))
    elif kind == "gpqa_diamond":
        load = _cap.load_gpqa_diamond(path, seed=int(_need(src, "seed")), limit=limit)
    else:
        load = _cap.load_gsm8k(path, limit=limit)
    return load.items


def selected_panel(selection: Mapping[str, Any], split: str) -> tuple[_cap.CapabilityItem, ...]:
    """The whole selected panel for a split, rebuilt from its sources and verified."""
    pools = selection["pools"] if "pools" in selection else [selection]
    by_id: dict[str, _cap.CapabilityItem] = {}
    for src in pools:
        for it in rebuild_pool(src):
            if it.id in by_id and by_id[it.id] != it:
                raise C.MeasureError(f"two capability sources produce different items under id {it.id!r}")
            by_id[it.id] = it
    if "splits" in selection:
        family = "confirm" if split == "confirm" else "search"
        if family not in selection["splits"]:
            raise C.MeasureError(f"the capability selection has no panel for split family {family!r}")
        ids = list(selection["splits"][family])
    else:
        ids = list(_need(selection, "item_ids"))
    absent = [i for i in ids if i not in by_id]
    if absent:
        raise C.MeasureError(f"the capability selection names items its sources do not produce: "
                             f"{absent[:5]} (the candidate pool changed since the setup chose them)")
    panel = tuple(by_id[i] for i in ids)
    if not panel:
        raise C.MeasureError("the capability selection is empty")
    if "items_sha256" in selection and "splits" not in selection \
            and panel_digest(panel) != selection["items_sha256"]:
        raise C.MeasureError("the rebuilt capability panel does not match the digest the selection "
                             "recorded (items_sha256): it is not the panel the setup chose")
    return panel


def panel_items(ctx: Any, n: int, split: str) -> tuple[tuple[_cap.CapabilityItem, ...], str]:
    """The first `n` items of the model's panel for this split, and where they came from."""
    C.check_split(split)
    n = C.check_n(n)
    profile = ctx.profile
    if profile is not None and profile.has_selection(SELECTION):
        st = C.state(ctx)
        key = ("s5-panel", split)
        with st.lock:
            panel = st.memo.get(key)
        if panel is None:
            panel = selected_panel(profile.selection(SELECTION), split)
            with st.lock:
                st.memo[key] = panel
        return panel[:n], "profile selection"
    family = "confirm" if split == "confirm" else "search"
    seed = ctx.seed(CODE, "panel", family)
    return _cap.generate(n, seed), f"generated ({family}, seed {seed}): no capability selection"


def measure_s5(ctx: Any, cell: Cell | None, *, n: int, split: str, variant: str = "original",
               start: int = 0, items: Sequence[Any] | None = None) -> Measured:
    """Measure `s5` at a cell on `n` panel items (each answered once per arm; past the panel's
    end the items repeat with new seeds)."""
    C.check_split(split)
    C.check_variant(CODE, variant)
    n, start = C.check_n(n), C.check_start(start)
    subj = C.subject(ctx, cell)
    if items is None:
        panel, source = panel_items(ctx, n, split)
    else:
        panel, source = tuple(_item(x) for x in items), "given"
    selected = source == "profile selection"
    if source.startswith("generated"):
        st = C.state(ctx)
        with st.lock:
            first = ("s5-fallback", split) not in st.memo
            st.memo[("s5-fallback", split)] = True
        if first and ctx.run is not None:   # recorded once per run: the panel was not chosen
            ctx.run.event("s5", "panel_fallback", split=split, panel=source)
    conf = float(ctx.cfg["CONFIDENCE"])
    s_jobs, z_jobs, units = [], [], []
    for it, r in C.cycle(panel, start, n):
        sp = C.spec(ctx, (CODE, it.id, it.question), lambda it=it: _prompts.capability_item(ctx.backend, it))
        unit = f"{it.id}#{r}"
        units.append((unit, it))
        z_jobs.append(C.Job(unit, sp, None, ctx.seed(CODE, C.NULL, it.id, r),
                            int(ctx.cfg["TOKENS_CAPABILITY"])))
        if not subj.null_only:
            s_jobs.append(C.Job(unit, sp, C.intervention(ctx, cell, sp, sp.meta["mode"]),
                                ctx.seed(CODE, split, subj.cell_id, it.id, r),
                                int(ctx.cfg["TOKENS_CAPABILITY"])))
    z_gen = C.generate(ctx, z_jobs, code=CODE, arm=C.NULL, cell_id=C.NULL, split=split)
    if subj.null_only:          # a cell of None is the null arm itself
        s_jobs, s_gen = z_jobs, z_gen
    else:
        s_gen = C.generate(ctx, s_jobs, code=CODE, arm=subj.arm, cell_id=subj.cell_id, split=split)

    def score(it: _cap.CapabilityItem, g: C.Gen) -> dict[str, Any]:
        answered = _cap.parse_answer(g.text, multiline=it.kind == "constraint") is not None
        return {"correct": bool(it.check(g.text)), "answered": answered,
                "degenerate": mechanical.degeneracy(g.text).flag, "finish": g.finish}

    pairs, rows, null_rows = [], [], []
    s_stats = {"no_answer": 0, "degenerate": 0, "truncated": 0}
    z_stats = {"no_answer": 0, "degenerate": 0, "truncated": 0}
    for (unit, it), sg, zg in zip(units, s_gen, z_gen):
        a, b = score(it, sg), score(it, zg)
        pairs.append((a["correct"], b["correct"]))
        for stats, x in ((s_stats, a), (z_stats, b)):
            stats["no_answer"] += not x["answered"]
            stats["degenerate"] += x["degenerate"]
            stats["truncated"] += x["finish"] == "length"
        base = {"code": CODE, "unit": unit, "item": it.id, "family": it.family, "kind": it.kind}
        rows.append({**base, "cell": subj.cell_id, "split": split, "arm": subj.arm,
                     "transcript_id": sg.transcript, "null_transcript_id": zg.transcript, **a,
                     "vector": subj.vector_id})
        null_rows.append(((CODE, unit), {**base, "cell": C.NULL, "split": C.NULL, "arm": C.NULL,
                                         "transcript_id": zg.transcript, **b}))
    if not subj.null_only:
        C.write_trials(ctx, rows)
    C.write_null_trials(ctx, null_rows)
    k = sum(a for a, _ in pairs)
    nk = sum(b for _, b in pairs)
    nn = len(pairs)
    lo, hi = C.wilson_or_none(k, nn, conf)
    pd = C.paired(pairs, conf)
    m = Measured(code=CODE, cell_id=subj.cell_id, split=split, n=nn, k=k, lo=lo, hi=hi, null_k=nk,
                 null_n=nn, unit_ids=[u for u, _ in units],
                 extra={"panel": source, "panel_selected": selected,
                        "paired": pd, "p_mcnemar": pd["p_mcnemar"] if pd else None,
                        "relative": (k / nk) if nk else None, "null": C.rate_block(nk, nn, conf),
                        "steered_failures": s_stats, "null_failures": z_stats,
                        "no_answer": s_stats["no_answer"],
                        "outcomes": [int(a) for a, _ in pairs],
                        **C.design([u.rsplit("#", 1)[0] for u, _ in units], [int(a) for a, _ in pairs])})
    if subj.null_only:
        m = C.as_null_arm(m, conf)
    return C.write_cell(ctx, m, subj=subj, iv=s_jobs[0].intervention)
