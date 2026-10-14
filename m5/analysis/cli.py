"""m5.analysis.cli - `python -m m5 analyze-m3 --archive DIR --what ...` (Phase 0).

    --archive DIR   an M3 run folder, an unzipped export, or a folder of run folders. Refused
                    before anything is read when it lies inside the repository: runs hold
                    generations, and the repository is public (CLAUDE.md hard rule 3).
    --what ...      any of refit, replay, coupling, near-misses, truncation, headline

Every invocation creates a new run (kind "analysis") under RUN_ROOT; results go to its `analysis`
stream (numbers and transcript ids, never text) and texts to its `transcripts` stream. The
reader's own record of what it read, dropped and inferred is written first. A short numeric
summary is printed; no transcript text ever is.

Choices the CLI makes, each written into the rows it produces:

- refit: on M3's dose axis (r), the floor's control from the null arm's literal mentions, R* with
  the effect curve, the winner's-curse simulation at BOOTSTRAP_REPS replications (no key sets
  that count yet);
- replay: without replacement (the search cannot buy more than M3 measured), compared on the
  fraction axis;
- coupling: both binnings, with the documented cut points (`coupling.coupling_cuts`);
- near-misses: no embedding and no pair judge (semantic scores and judged rules stay missing);
- truncation: the plan's lengths, the subject's tokenizer when it is available locally (the
  whitespace fallback otherwise, marked approximate), and the judge cascade the config allows --
  a loopback tier 1 at JUDGE_T1_URL, external tiers only when JUDGE_EXTERNAL allows them;
- headline: Silk's L29 at 85% of dose_max, for a Silk archive.

Exit status: 0 when everything asked for ran, 2 when something was refused or skipped (the
reason is printed and recorded as an `events` row).
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
import traceback
from collections.abc import Callable, Sequence
from typing import Any

from m5 import runio, safety
from m5.analysis import AnalysisSink

WHATS = ("refit", "replay", "coupling", "near-misses", "truncation", "headline")


def _say(msg: str) -> None:
    print(msg, flush=True)


def _fmt(x: Any, nd: int = 3) -> str:
    return "NA" if x is None else f"{x:.{nd}f}"


def main(args: Any, cfg: Any) -> int:
    archive_arg = getattr(args, "archive", None)
    whats = list(getattr(args, "what", None) or ["refit", "replay", "coupling"])
    bad = [w for w in whats if w not in WHATS]
    if bad:
        print(f"unknown analyses {bad}; choose from {WHATS}", file=sys.stderr)
        return 2
    try:
        root = safety.assert_outside_repo(archive_arg, what="M3 archive")
    except (safety.Refused, TypeError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    from m5.analysis import m3_archive

    try:
        archives = m3_archive.load_archives(root)
    except (m3_archive.ArchiveError, safety.Refused) as exc:
        print(f"cannot read the archive: {exc}", file=sys.stderr)
        return 2
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"analyze_m3_{stamp}_{os.urandom(3).hex()}"
    run = runio.RunDir.create(cfg["RUN_ROOT"], name, config=cfg, kind="analysis",
                              extra={"archive": str(root), "what": whats,
                                     "concepts": [a.concept for a in archives],
                                     "archive_runs": [a.run_id for a in archives]})
    sink = AnalysisSink(run)
    status = 0
    try:
        _say(f"analysis run: {run.path}")
        for a in archives:
            sink.rows(m3_archive.archive_rows(a))
            rep = a.report
            _say(f"[{a.concept}] {len(a.cells)} cells, {len(a.layers)} layers, grid "
                 f"{list(a.dose_fractions)}; duplicates dropped "
                 f"{sum(f.duplicates_dropped for f in rep.files.values())}, torn last lines "
                 f"{sum(int(f.torn_last_line) for f in rep.files.values())}, cells without a "
                 f"boundary {len(rep.cells_without_boundary)}; drift {rep.drift}")
        run.event("archive", "read", archives=len(archives), cells=sum(len(a.cells) for a in archives))
        steps: dict[str, Callable[[], int]] = {
            "refit": lambda: _refit(archives, cfg, sink),
            "replay": lambda: _replay(archives, cfg, sink, run),
            "coupling": lambda: _coupling(archives, cfg, sink),
            "near-misses": lambda: _near_misses(archives, cfg, sink),
            "truncation": lambda: _truncation(archives, cfg, sink, run),
            "headline": lambda: _headline(archives, sink),
        }
        for what in whats:
            run.event(what, "started")
            rc = steps[what]()
            run.event(what, "finished" if rc == 0 else "skipped", exit_status=rc)
            status = max(status, rc)
        _say(f"wrote {sink.rows_written} analysis rows and {sink.texts_written} transcript rows")
    finally:
        run.close()
    return status


# ------------------------------------------------------------------------------- the steps

def _refit(archives: Sequence[Any], cfg: Any, sink: AnalysisSink) -> int:
    from m5.analysis.refit import refit

    res = refit(archives, cfg, unit="dose", floor_control="null_literal", with_effect=True,
                winners_curse_reps=int(cfg["BOOTSTRAP_REPS"]))
    sink.rows(res.rows)
    q = float(cfg["Q"])
    skipped = [r for r in res.rows if r["kind"] == "refit_skipped"]
    for r in skipped:
        print(f"[{r['concept']}] refit skipped: {r['reason']}", file=sys.stderr)
    for run_id, fit in res.fits.items():
        n_fit = len(fit.chain_layers)
        ident = sum(1 for lf in fit.layers.values() if lf.identified)
        _say(f"[{fit.concept}] refit (dose axis): {n_fit} layers fitted, {ident} with u_D({q}) "
             f"identified, {len(fit.layers) - ident} not")
        rs = next((r for r in fit.rstar or () if abs(r["q"] - q) < 1e-12), None)
        if rs is not None:
            _say(f"[{fit.concept}] R*({q}) = {_fmt(rs['median'])} [{_fmt(rs['lo'])}, {_fmt(rs['hi'])}] "
                 f"(presence excess), best layer L{rs['best_layer']}, flags {rs['flags']}")
        for est, icc in res.iccs.get(run_id, {}).items():
            _say(f"[{fit.concept}] ICC {est}: {_fmt(icc.icc)} [{_fmt(icc.lo)}, {_fmt(icc.hi)}] "
                 f"over {icc.n_cells} cells x {icc.n_prompts} prompts")
    for r in res.rows:
        if r["kind"] == "refit_winners_curse" and (r["n_arms"], r["n"]) == (25, 22) and \
                abs(r["p"] - float(cfg["SPRT_P0"])) < 1e-12:
            _say(f"winner's curse, best of 25 at n=22, p={r['p']}: ICC {r['icc']:.3g} -> "
                 f"+{100 * r['inflation']:.1f} points")
    return 2 if skipped else 0


def _replay(archives: Sequence[Any], cfg: Any, sink: AnalysisSink, run: Any) -> int:
    from m5.analysis.replay import OffGrid, ReplayError, SearchUnavailable, replay_search

    rc = 0
    for a in archives:
        try:
            res = replay_search(a, cfg, mode="without_replacement", integrity="m3_flags",
                                floor_control="null_literal", run=run)
        except SearchUnavailable as exc:
            print(f"replay skipped: {exc}", file=sys.stderr)
            run.event("replay", "unavailable", reason="m5.search not importable")
            return 2
        except (OffGrid, ReplayError) as exc:
            print(f"[{a.concept}] replay refused: {exc}", file=sys.stderr)
            rc = 2
            continue
        except Exception as exc:  # noqa: BLE001 - the search is a separate module; report, go on
            traceback.print_exc(file=sys.stderr)
            print(f"[{a.concept}] replay failed inside the search ({type(exc).__name__}): {exc}",
                  file=sys.stderr)
            run.event("replay", "failed", concept=a.concept, error=type(exc).__name__)
            rc = 2
            continue
        sink.rows(res.rows)
        for label, c in (("search vs survey", res.search_vs_survey),
                         ("refit, visited vs full", res.comparison)):
            full, srch = c.get("full") or {}, c.get("search") or {}
            if not full and not srch:
                _say(f"[{a.concept}] replay ({label}): not available ({c.get('reason')})")
                continue
            _say(f"[{a.concept}] replay ({label}): R*({c['q']}) full {_fmt(full.get('median'))} "
                 f"[{_fmt(full.get('lo'))}, {_fmt(full.get('hi'))}] L{full.get('best_layer')} vs "
                 f"search {_fmt(srch.get('median'))} L{srch.get('best_layer')}; "
                 f"recovered {c.get('recovered')}")
        _say(f"[{a.concept}] replay spend: d2 trials {res.spend['d2_trials_drawn']} of "
             f"{res.spend['d2_trials_archived']}, influence prompts {res.spend['e8_prompts_drawn']} "
             f"of {res.spend['e8_prompts_archived']}")
    return rc


def _coupling(archives: Sequence[Any], cfg: Any, sink: AnalysisSink) -> int:
    from m5.analysis.coupling import coupling, coupling_cuts

    cuts = coupling_cuts(cfg)
    for bin_by in ("fraction", "log_dose"):
        rows = coupling(archives, cfg, bin_by=bin_by, **cuts)
        sink.rows(rows)
        for r in rows:
            if r["kind"] != "coupling_summary":
                continue
            w = r["rank_coupling"]
            nn, ni = r["named_not_influenced"], r["influenced_not_named"]
            _say(f"[{r['concept']}] coupling by {bin_by}: rho_mean {_fmt(w.get('rho_mean'))} "
                 f"[{_fmt(w.get('lo'))}, {_fmt(w.get('hi'))}] within bins "
                 f"(across bins {_fmt(r['spearman_across_bins'])}) over {r['n_cells']} cells; "
                 f"named-not-influenced {nn['k']}/{nn['n']}, influenced-not-named {ni['k']}/{ni['n']}")
    return 0


def _near_misses(archives: Sequence[Any], cfg: Any, sink: AnalysisSink) -> int:
    from m5.analysis.nearmiss import near_misses

    for a in archives:
        res = near_misses(a, cfg)
        sink.rows(res.rows)
        sink.texts(res.texts)
        summ = next(r for r in res.rows if r["kind"] == "nearmiss_summary")
        rates = ", ".join(f"{k} {_fmt(v['rate'])} (n {v['n']})" for k, v in summ["rules"].items()
                          if v["rate"] is not None)
        _say(f"[{a.concept}] identification under each rule: {rates} "
             f"({summ['n_trials']} trials, {summ['n_distinct_names']} distinct names)")
    return 0


def _truncation(archives: Sequence[Any], cfg: Any, sink: AnalysisSink, run: Any) -> int:
    from m5.analysis.truncation import truncation_curve
    from m5.config import MODELS

    rc = 0
    for a in archives:
        # The judge never shares the subject's family (plan §7.2): the subject is the archive's.
        acfg = cfg.replace(MODEL=a.model) if a.model in MODELS else cfg
        judge, why = _judge_for(acfg, run)
        if judge is None:
            print(f"truncation skipped: {why}", file=sys.stderr)
            run.event("truncation", "unavailable", reason="no judge")
            return 2
        tok = _local_tokenizer(a, cfg)
        try:
            res = truncation_curve(a, acfg, judge=judge, tokenizer=tok)
        except ValueError as exc:
            print(f"[{a.concept}] truncation refused: {exc}", file=sys.stderr)
            rc = 2
            continue
        sink.rows(res.rows)
        for r in res.rows:
            if r["kind"] == "truncation_length":
                _say(f"[{a.concept}] {r['template']} at {r['length']} ({r['tokenizer']}): "
                     f"{r['k']}/{r['n']} = {_fmt(r['rate'])}")
    return rc


def _headline(archives: Sequence[Any], sink: AnalysisSink) -> int:
    from m5.analysis.headline import (
        HEADLINE_CONCEPT,
        HEADLINE_FRACTION,
        HEADLINE_LAYER,
        HeadlineError,
        headline_cell,
    )

    done = 0
    for a in archives:
        if a.concept.lower() != HEADLINE_CONCEPT:
            _say(f"[{a.concept}] headline: the headline cell is {HEADLINE_CONCEPT.title()}'s "
                 f"(L{HEADLINE_LAYER} at {HEADLINE_FRACTION} of dose_max); skipped")
            continue
        try:
            res = headline_cell(a, layer=HEADLINE_LAYER, fraction=HEADLINE_FRACTION)
        except HeadlineError as exc:
            print(f"headline: {exc}", file=sys.stderr)
            return 2
        sink.rows(res.rows)
        sink.texts(res.texts)
        done += 1
        for r in res.rows:
            if r["kind"] == "headline_channel":
                extra = ""
                if "matches_k" in r:
                    extra = f" matches {r['matches_k']}/{r['matches_n']}"
                if "claims" in r:
                    extra += f" claims {r['claims']}"
                if "influence" in r and r["influence"]:
                    extra += f" influence mean {sum(r['influence']) / len(r['influence']):.2f}"
                _say(f"[{a.concept}] headline L{HEADLINE_LAYER}@{HEADLINE_FRACTION} {r['channel']}: "
                     f"n {r['n']}{extra}")
    return 0 if done else 2


# ------------------------------------------------------------------------------- judge, tokenizer

def _judge_for(cfg: Any, run: Any) -> tuple[Callable[[list[Any]], list[Any]] | None, str]:
    """The cascade the config allows, or None and why. Tier 1 only from a loopback server here:
    loading a judge model in-process is a GPU job, not a Phase-0 analysis."""
    from m5.judges import AnthropicJudge, Cascade, JudgeError, JudgeStore, LocalOpenAIJudge, OpenRouterJudge

    classes = ["concept"]
    try:
        t1 = LocalOpenAIJudge(cfg) if cfg["JUDGE_T1_URL"] else None
        external = safety.external_judging_allowed(cfg["JUDGE_EXTERNAL"], classes)
        t2 = OpenRouterJudge(cfg, vector_classes=classes) if external else None
        t3 = AnthropicJudge(cfg, vector_classes=classes) if external else None
    except (JudgeError, safety.Refused) as exc:
        return None, f"a judge could not be set up: {exc}"
    if t1 is None and t2 is None and t3 is None:
        return None, ("no judge is available: set JUDGE_T1_URL to a loopback judge server, or "
                      "JUDGE_EXTERNAL=1 once decision 12 allows it (transcripts would leave the "
                      "machine)")
    cascade = Cascade(cfg, t1=t1, t2=t2, t3=t3, vector_classes=classes,
                      store=JudgeStore.for_run(run))
    return cascade.judge, ""


def _local_tokenizer(archive: Any, cfg: Any) -> Callable[[str], list[tuple[int, int]]] | None:
    """The subject's tokenizer from the local Hugging Face cache (never downloaded here); None
    when it is not available, and the curve then uses the approximate whitespace fallback."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415 - optional, local files only
        from m5.config import MODELS
        hf_id = MODELS[archive.model]["hf_id"] if archive.model in MODELS else cfg.model_id
        tok = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
    except Exception:
        return None

    def offsets(text: str) -> list[tuple[int, int]]:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(span) for span in enc["offset_mapping"]]

    return offsets
