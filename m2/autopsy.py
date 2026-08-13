"""Disposable task-25 diagnostic for the cells that the d3 frontier selects.

This module is deliberately self-contained.  It may be deleted after one Garlic diagnostic,
so none of its helpers are imported by the measurement pipeline and it changes no production
measure.  Heavy imports stay inside :func:`run_autopsy`, keeping the ordinary offline import
surface usable without torch, transformers or the upstream repository.
"""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CELLS: tuple[tuple[int, float], ...] = (
    (57, 0.30),
    (58, 0.30),
    (59, 0.30),       # positive control: the concept must win with probability > 0.9
    (52, 0.60),
)
POSITIVE_CONTROL: tuple[int, float] = (59, 0.30)
TOP_K = 10

_CELL = re.compile(r"^L?(\d+)@([+]?(?:\d+(?:\.\d*)?|\.\d+))$", re.IGNORECASE)


def _cell_key(layer: int, r: float) -> tuple[int, float]:
    """Stable identity for the CLI's human-written doses."""
    return int(layer), round(float(r), 6)


def parse_cells(text: str | None) -> list[tuple[int, float]]:
    """Parse ``L57@0.30,...`` and require task 25's positive control.

    A blank value means the documented defaults; this is what lets ``--autopsy-cells`` run
    them while still allowing an explicit comma-separated override through the same one flag.
    """
    if text is None or not str(text).strip():
        cells = list(DEFAULT_CELLS)
    else:
        cells = []
        for raw in str(text).split(","):
            item = raw.strip()
            match = _CELL.fullmatch(item)
            if match is None:
                raise ValueError(
                    f"invalid autopsy cell {item!r}; expected LAYER@DOSE, for example 57@0.30")
            layer, r = int(match.group(1)), float(match.group(2))
            if not math.isfinite(r) or r < 0.0:
                raise ValueError(f"invalid autopsy dose in {item!r}: r must be finite and >= 0")
            key = _cell_key(layer, r)
            if key not in {_cell_key(*cell) for cell in cells}:
                cells.append((layer, r))

    if _cell_key(*POSITIVE_CONTROL) not in {_cell_key(*cell) for cell in cells}:
        raise ValueError(
            "task 25 requires positive-control cell L59@0.30; without it, a dump that shows "
            "nothing is indistinguishable from a broken dump")
    return cells


def _assert_outside_repo(path: Path, *, repo_root: Path | None = None) -> Path:
    """Refuse any task-25 output path inside the repository working tree."""
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve(strict=False)
    target = Path(path).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError:
        return target
    raise ValueError(
        f"task 25 output path {str(target)!r} is inside repository {str(root)!r}; "
        "the diagnostic may write only to the external run directory or stdout")


def _require_transcripts_allowed(concept: str, cfg: dict) -> str:
    """Apply the same two-argument transcript gate as ``runio.export_bundle``.

    There is intentionally no override parameter in this diagnostic.  It prints and persists
    model generations, so a non-benign concept must be structurally unable to enter it.
    """
    from . import runio

    allowed, reason = runio.transcripts_allowed(concept, cfg)
    if not allowed:
        raise PermissionError(f"task 25 autopsy refused for {concept!r}: {reason}")
    return reason


def prepare(concepts: Sequence[str], cfg: dict, cell_text: str | None, *,
            log_path: Path | None = None) -> list[tuple[int, float]]:
    """Validate the standalone mode before model loading creates any output."""
    if len(concepts) != 1:
        raise ValueError(
            f"task 25 autopsies exactly one benign concept, got {len(concepts)}: {list(concepts)}")
    concept = str(concepts[0])
    _require_transcripts_allowed(concept, cfg)
    cells = parse_cells(cell_text)

    from . import config

    concept_cfg = dict(cfg)
    concept_cfg["concept"] = concept
    _assert_outside_repo(config.run_dir_for(concept, concept_cfg))
    if log_path is not None:
        _assert_outside_repo(Path(log_path))
    return cells


def _print_variants(concept: str, kept: Sequence[dict], dropped: Sequence[dict]) -> None:
    """Print every surface candidate and the verdict that kept or dropped it."""
    print("1. VARIANT TABLE")
    print(f"   {'state':<8} {'variant':<16} {'token':>7} {'decoded':<16} reason")
    for state, entries in (("KEPT", kept), ("DROPPED", dropped)):
        for row in entries:
            token = "-" if row.get("token_id") is None else str(row["token_id"])
            decoded = repr(row.get("decodes_to"))
            print(f"   {state:<8} {repr(row.get('variant')):<16} {token:>7} "
                  f"{decoded:<16} {row.get('reason')}")
    print(f"   concept={concept!r}; kept={len(kept)}, dropped={len(dropped)}")


def _top_rows(probs: Any, tok: Any, *, k: int = TOP_K) -> list[dict]:
    """Return the top-k token ids, decoded strings and probabilities from one d3 tensor."""
    n = min(int(k), int(probs.numel()))
    values, indices = probs.topk(n)
    rows = []
    for value, token_id in zip(values.detach().cpu().tolist(),
                               indices.detach().cpu().tolist()):
        tid = int(token_id)
        rows.append(dict(token_id=tid, token=tok.decode([tid]), probability=float(value)))
    return rows


def _print_top_rows(rows: Sequence[dict], cids: set[int], dropped_ids: set[int]) -> None:
    print(f"      {'rank':>4} {'token_id':>9} {'probability':>12} {'mark':<9} decoded")
    for rank, row in enumerate(rows, start=1):
        token_id = int(row["token_id"])
        mark = "CID" if token_id in cids else ("DROPPED" if token_id in dropped_ids else "")
        print(f"      {rank:>4} {token_id:>9} {float(row['probability']):>12.8f} "
              f"{mark:<9} {row['token']!r}")


def _require_positive_control(top1: Sequence[dict]) -> None:
    """The L59 dump is evidence only if every fixed d3 trial validates the readout."""
    bad = [row for row in top1
           if not row.get("in_cids") or float(row.get("probability", 0.0)) <= 0.90]
    if bad:
        detail = [(row.get("trial"), row.get("token_id"), row.get("probability"),
                   row.get("in_cids")) for row in bad]
        raise RuntimeError(
            "task 25 positive control FAILED at L59@0.30: every forced-position trial must "
            f"put a CID token at rank 1 with probability > 0.9; failures={detail!r}. "
            "The dump is not validated, so no other autopsy output is evidence.")


def _validate_control_cell(layer: int, r: float, top1: Sequence[dict]) -> bool:
    """Run the mandatory L59 guard and say whether this was the control cell."""
    if _cell_key(layer, r) != _cell_key(*POSITIVE_CONTROL):
        return False
    _require_positive_control(top1)
    return True


def _require_allow_filler(enabled: bool) -> None:
    if not enabled:
        raise RuntimeError("task 25 requires cheap.ALLOW_FILLER=True to inspect d3's second read")


def _require_start_pos(start: int | None) -> int:
    if start is None:
        raise RuntimeError("d3 forced prompts have no steering start position")
    return int(start)


def _require_matching_ids(listed_ids: Sequence[int], cids: Sequence[int]) -> None:
    if sorted(int(i) for i in cids) != sorted(int(i) for i in listed_ids):
        raise RuntimeError(
            f"variant table ids {list(listed_ids)!r} disagree with d3 cids {list(cids)!r}")


def _require_d3_reading(value: float | None) -> float:
    if value is None:
        raise RuntimeError("d3 autopsy produced no trial readings")
    return float(value)


def _greedy_extension(enc: dict, probs: Any, *, layer: int, alpha: float,
                      start_pos: int, ctx: Any) -> tuple[int, Any]:
    """Append d3's greedy token and return the next distribution without rebuilding a prompt."""
    import torch

    from . import model

    nxt = int(torch.argmax(probs))
    ids2 = torch.cat(
        [enc["input_ids"], torch.tensor([[nxt]], device=enc["input_ids"].device)], dim=1)
    am2 = torch.cat(
        [enc["attention_mask"],
         torch.ones((1, 1), dtype=enc["attention_mask"].dtype,
                    device=enc["attention_mask"].device)], dim=1)
    with model.injected(ctx.vecs[int(layer)], int(layer), float(alpha), start_pos=start_pos):
        with torch.no_grad():
            logits = ctx.hf(input_ids=ids2, attention_mask=am2).logits[0, -1, :].float()
    return nxt, torch.softmax(logits, dim=-1)


def _dump_and_score_d3(layer: int, r: float, alpha: float, *, cids: Sequence[int],
                       dropped_ids: set[int]) -> tuple[dict, list[dict]]:
    """Print both d3 positions and derive d3 from those exact returned tensors."""
    from . import cheap, config, model

    ctx = config.RUN
    _require_allow_filler(cheap.ALLOW_FILLER)
    start, forwards = cheap._d3_forward(int(layer), float(alpha), cheap.D3_TRIALS)
    start = _require_start_pos(start)

    cid_set = {int(token_id) for token_id in cids}
    best_values: list[float] = []
    ranks: list[int] = []
    top1: list[dict] = []

    print("2. TOP-10 TOKEN DUMP")
    for trial, (enc, probs) in zip(cheap.D3_TRIALS, forwards):
        forced_rows = _top_rows(probs, ctx.tok)
        print(f"   trial {trial} - forced-ID position")
        _print_top_rows(forced_rows, cid_set, dropped_ids)

        forced_mass = float(probs[list(cid_set)].sum())
        forced_rank = int(cheap._rank_of_best(probs, list(cid_set)))
        ranks.append(forced_rank)
        first = forced_rows[0]
        top1.append(dict(trial=int(trial), token_id=int(first["token_id"]),
                         probability=float(first["probability"]),
                         in_cids=int(first["token_id"]) in cid_set))

        nxt, filler_probs = _greedy_extension(
            enc, probs, layer=layer, alpha=alpha, start_pos=start, ctx=ctx)
        print(f"   trial {trial} - after ALLOW_FILLER greedy token "
              f"{ctx.tok.decode([nxt])!r} (id {nxt})")
        _print_top_rows(_top_rows(filler_probs, ctx.tok), cid_set, dropped_ids)
        filler_mass = float(filler_probs[list(cid_set)].sum())

        # This is measure_D3's exact scoring rule: when the first token is already a CID,
        # there is no filler recovery to score.  We still show the extension above because
        # task 25 asks to inspect both positions, but it cannot inflate the recorded d3.
        best_values.append(max(forced_mass, filler_mass if nxt not in cid_set else 0.0))

    d3, d3_se, n = model.mean_se(best_values)
    d3 = _require_d3_reading(d3)
    threshold = float(ctx.config["D3_RATE_THRESH"])
    hits = sum(value > threshold for value in best_values)
    return (dict(d3=d3, d3_se=d3_se, d3_n=n,
                 d3_rate=hits / n, d3_rank_med=statistics.median(ranks),
                 layer=int(layer), r=float(r), alpha=float(alpha)), top1)


def _measure_real_d2(expensive: Any, *, layer: int, r: float, alpha: float,
                     trials: Sequence[int]) -> dict:
    """Run real d2 on the same small fixed trial set used by d3."""
    return expensive.measure_D2(
        int(layer), float(alpha), len(trials), phase="AUTOPSY", r=float(r),
        trial_numbers=list(trials))


def _print_summary_table(rows: Sequence[dict]) -> None:
    print("\n" + "=" * 78)
    print("TASK 25 FOUR-CELL SUMMARY")
    print("=" * 78)
    print(f"   {'cell':>12} {'d3':>10} {'d3_rate':>10} {'d3_rank_med':>13} {'d2':>8}")
    for row in rows:
        cell = f"L{int(row['layer'])}@{float(row['r']):.2f}"
        print(f"   {cell:>12} {float(row['d3']):>10.6f} {float(row['d3_rate']):>10.3f} "
              f"{float(row['d3_rank_med']):>13g} {float(row['d2']):>8.3f}")


def run_autopsy(cells: Sequence[tuple[int, float]]) -> list[dict]:
    """Build only the needed vectors, inspect four cells, and return their summary rows."""
    from . import config

    ctx = config.RUN
    if ctx.run_dir is None or not ctx.concept:
        raise RuntimeError("call model.load_model and driver.set_concept before task 25")
    _require_transcripts_allowed(str(ctx.concept), ctx.config)
    _assert_outside_repo(Path(ctx.run_dir))

    from . import cheap, expensive, prompts, runio, vectors

    parsed = parse_cells(",".join(f"{layer}@{r}" for layer, r in cells))
    layers = sorted({int(layer) for layer, _r in parsed})
    doses = sorted({float(r) for _layer, r in parsed})
    vectors.extract_all_layers(str(ctx.concept), layers)
    vectors.build_dose_map(
        layers, doses, calib_prompts=[row["text"] for row in prompts.E5_PROMPTS], write=False)

    summaries: list[dict] = []
    for layer, r in parsed:
        alpha = float(config.alpha_for(int(layer), float(r)))
        print("\n" + "=" * 78)
        print(f"TASK 25 AUTOPSY  concept={ctx.concept!r}  L{layer}@r={r:.2f}  alpha={alpha:.6f}")
        print("=" * 78)

        listed_ids, kept, dropped = vectors.concept_first_token_ids(str(ctx.concept))
        cids = cheap._concept_ids()
        _require_matching_ids(listed_ids, cids)
        _print_variants(str(ctx.concept), kept, dropped)
        dropped_ids = {int(row["token_id"]) for row in dropped
                       if row.get("token_id") is not None and int(row["token_id"]) not in cids}

        d3_row, top1 = _dump_and_score_d3(
            layer, r, alpha, cids=cids, dropped_ids=dropped_ids)
        if _validate_control_cell(layer, r, top1):
            print("   POSITIVE CONTROL PASS: every trial has a rank-1 CID token at p > 0.9")

        print("3. REAL d2 (forced-ID rate, lower is better)")
        d2_row = _measure_real_d2(
            expensive, layer=layer, r=r, alpha=alpha, trials=cheap.D3_TRIALS)
        for trial, response in zip(d2_row["trials"], d2_row["responses"]):
            print(f"   trial {trial}: {response}")

        summary = dict(d3_row, d2=float(d2_row["d2"]), n_d2=int(d2_row["n_d2"]))
        summaries.append(summary)
        print("4. SUMMARY")
        print(f"   L{layer}@{r:.2f}: d3={summary['d3']:.6f}  "
              f"d3_rate={summary['d3_rate']:.3f}  "
              f"d3_rank_med={summary['d3_rank_med']:g}  d2={summary['d2']:.3f}")

    _print_summary_table(summaries)
    runio.log(
        f"task 25 autopsy complete for {ctx.concept}: {len(summaries)} cells; "
        "diagnostic readings are not run results", "INFO")
    return summaries
