"""m5.analysis: the M3 archive reader and the Phase-0 analyses built on it.

Every fixture is a SYNTHETIC M3 run written into tmp_path in the on-disk format of the format
specification's example rows (§14): field names, key order and types as M3 writes them, values
made up. The hazards the specification names are built in on purpose -- a torn last line, a
re-measured cell appended twice, a cell whose layer has no boundary row, a boundary outcome in
the pre-c04ab0f vocabulary, a failed judge call -- and each test checks the reader or the
analysis does the documented thing with it. Distinctive marker words sit in every response and
every named string, so a test can prove no text reached an aggregate row.
"""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from m5 import config, runio, safety
from m5.analysis import STAMP_KEYS, AnalysisSink, check_aggregate, name_id
from m5.analysis import m3_archive as ar
from m5.intervention import Cell

FRACTIONS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
TASKS = (("task_story", "Tell me a short story."), ("task_landscape", "Describe a landscape."),
         ("task_words", "List some words you like."), ("task_notice", "What do you notice today?"))
EXPLAINS = (("exp_computer", "What is a computer?"), ("exp_election", "What is an election?"),
            ("exp_curie", "Who was Marie Curie?"), ("exp_moon", "Why does the moon have phases?"))
CAPS = (("cap_arith", "What is 17 x 23?"), ("cap_capital", "What is the capital of France?"))
MARK = "QUOKKAMARK"          # in every response text
NAME_MARK = "zebrafruit"     # a named string that must never reach an aggregate row


def logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@pytest.fixture
def small_cfg():
    """The default config with coarse grids and few draws, so a fit takes a fraction of a second."""
    return config.load({"PRIOR_S_POINTS": 9, "PRIOR_FLOOR_POINTS": 3, "PRIOR_LAPSE_POINTS": 3,
                        "PRIOR_ICC": (0.0, 0.05), "POSTERIOR_DRAWS": 300, "BOOTSTRAP_REPS": 30})


# =====================================================================================
# The synthetic-run writer
# =====================================================================================

class RunWriter:
    """Writes one M3 run folder (§14 shapes). `cell(...)` adds one measured cell's battery,
    judge calls and cell row; `finish()` writes summary, boundaries, norms and the null arm."""

    def __init__(self, root: Path, *, concept: str = "Garlic", hash12: str = "221fa4e080d9",
                 layers=(27, 29, 31), dose_max=None, fractions=FRACTIONS, n_identify: int = 30,
                 attempt: str | None = None, seed: int = 0) -> None:
        self.concept, self.hash = concept, hash12
        self.dir = root / f"{concept.lower()}_{hash12}"
        self.dir.mkdir(parents=True)
        self.layers = list(layers)
        self.dose_max = dose_max or {L: round(0.2 + 0.01 * L, 6) for L in layers}
        self.fractions = tuple(fractions)
        self.n_identify = n_identify
        self.attempt = attempt
        self.rng = np.random.default_rng(seed)
        self.files: dict[str, list[dict]] = {}

    def stamp(self, row: dict, attempt: str | None = "default") -> dict:
        row = dict(row, concept=self.concept, config_hash=self.hash, ts="2026-08-20T10:00:00+00:00")
        a = self.attempt if attempt == "default" else attempt
        if a is not None:
            row["attempt"] = a
        return row

    def add(self, name: str, row: dict) -> None:
        self.files.setdefault(name, []).append(row)

    def dose(self, layer: int, f: float) -> float:
        return round(self.dose_max[layer] * f, 6)

    def cell(self, layer: int, f: float, *, match=lambda i: False, named=lambda i: "lemon",
             influence=lambda j: 0.0, claims=lambda i: "NO", sr_named=lambda i: "NONE",
             sr_matches=lambda i: False, coherence=lambda j: 8.0, fail_identify=(),
             degenerate=(), attempt: str | None = "default", cell_row: bool = True,
             stored_override: dict | None = None, empty_effect=()) -> None:
        d = self.dose(layer, f)
        alpha = round(d * 40.0, 9)
        rows = []
        calls = []

        def base(ch, unit, text, extra=None):
            r = {"channel": ch, "response": text, "words": len(text.split()),
                 "concept_mentions": text.lower().count(self.concept.lower()),
                 "degenerate": False, "degeneration_reason": None, "empty": not text.split(),
                 "unit": unit, "layer": layer, "dose": d, "alpha": alpha, "steered": True}
            r.update(extra or {})
            return r

        for i in range(self.n_identify):
            unit = f"trial_{1 + 6 * i}"
            nm = named(i)
            text = f"the word \"{nm}\". It feels vivid {MARK}{i}."
            r = base("identify", unit, text, {"prompt_text": None})
            if i in degenerate:
                r.update(degenerate=True, degeneration_reason="ngram_repeat: 5-gram repeated 9 times: "
                         f"'{MARK} {MARK}'")
            if i in fail_identify:
                r["judged"] = {"identify": None}
                r["judge_error"] = {"identify": "judge response did not parse"}
                calls.append({"judge": "identify", "unit": unit, "channel": "identify", "ok": False,
                              "cached": False, "raw": "I cannot comply with this format.",
                              "parsed": None, "error": "judge response did not parse"})
            else:
                m = bool(match(i))
                r["judged"] = {"identify": {"named": nm, "matches": m}}
                r["judge_error"] = {"identify": None}
                calls.append({"judge": "identify", "unit": unit, "channel": "identify", "ok": True,
                              "cached": False, "raw": f"Named: {nm}\nMatches: {'YES' if m else 'NO'}",
                              "parsed": {"named": nm, "matches": m}, "error": None})
            rows.append(r)
        for i in range(3):
            unit = f"trial_{1 + 6 * i}"
            c = claims(i)
            text = f"{'Yes' if c == 'YES' else 'No'}, I think it is {sr_named(i)} {MARK}sr{i}."
            r = base("self_report", unit, text, {"prompt_text": None})
            m = bool(sr_matches(i))
            r["judged"] = {"self_report": {"claims": c, "named": sr_named(i), "matches": m}}
            r["judge_error"] = {"self_report": None}
            r["self_report_class"] = ("identified" if c == "YES" and m else "confabulated" if c == "YES"
                                      else "leaked" if m else "clean_denial")
            calls.append({"judge": "self_report", "unit": unit, "channel": "self_report", "ok": True,
                          "cached": False, "raw": f"Claims: {c}", "parsed": r["judged"]["self_report"],
                          "error": None})
            rows.append(r)
        for j, (unit, prompt) in enumerate(TASKS):
            infl = float(influence(j))
            text = "" if j in empty_effect else f"A story about {self.concept.lower()} {MARK}e{j} by the sea."
            r = base("effect", unit, text, {"prompt_text": prompt})
            r["judged"] = {"effect": {"influence": infl, "form": "thematic" if infl >= 4 else "trace"},
                           "coherence": {"coherence": float(coherence(j)), "on_task": True}}
            r["judge_error"] = {"effect": None, "coherence": None}
            calls.append({"judge": "effect", "unit": unit, "channel": "effect", "ok": True,
                          "cached": False, "raw": f"Influence: {infl}", "parsed": r["judged"]["effect"],
                          "error": None})
            rows.append(r)
        for j, (unit, prompt) in enumerate(EXPLAINS):
            text = f"An answer {MARK}x{j}."
            r = base("explain", unit, text, {"capability_correct": True, "prompt_text": prompt})
            r["judged"] = {"effect": {"influence": 1.0, "form": "trace"},
                           "coherence": {"coherence": 9.0, "on_task": True}}
            r["judge_error"] = {"effect": None, "coherence": None}
            rows.append(r)
        for j, (unit, prompt) in enumerate(CAPS):
            text = f"391 {MARK}c{j}"
            rows.append(base("capability", unit, text, {"capability_correct": j == 0,
                                                        "prompt_text": prompt}))
        for r in rows:
            self.add(ar.RESPONSES, self.stamp(r, attempt))
        for cl in calls:
            self.add(ar.JUDGE_CALLS, self.stamp(dict(cl, attempts=1, latency_s=0.001,
                                                     payload=f"RESPONSE {MARK}", layer=layer, dose=d),
                                                attempt))
        if cell_row:
            self.add(ar.CELLS, self.stamp(dict(self.stored(rows, layer, d, alpha), **(stored_override or {})),
                                          attempt))

    @staticmethod
    def stored(rows, layer, d, alpha):
        """The run's own cell row, M3's formulas (spec §8), for the fields the reader checks."""
        def rate(k, n):
            return None if n == 0 else {"rate": k / n, "count": k, "n": n, "ci_low": 0.0,
                                        "ci_high": 1.0, "ci_z": 1.96}
        ident = [r for r in rows if r["channel"] == "identify"]
        hits = [r for r in ident if r["judged"]["identify"]]
        clean = [r for r in hits if not r["degenerate"]]
        eff = [r["judged"]["effect"]["influence"] for r in rows
               if r["channel"] == "effect" and r["response"].strip()]
        sr = [r["judged"]["self_report"] for r in rows if r["channel"] == "self_report"]
        caps = [r for r in rows if r["channel"] == "capability"]
        return {"layer": layer, "dose": d, "alpha": alpha,
                "identification": rate(sum(r["judged"]["identify"]["matches"] for r in hits), len(hits)),
                "identification_excluding_degenerate": rate(
                    sum(r["judged"]["identify"]["matches"] for r in clean), len(clean)),
                "identify_degenerate_n": sum(1 for r in ident if r["degenerate"]),
                "identified_as": sorted({r["judged"]["identify"]["named"].lower() for r in hits}),
                "effectiveness": None if not eff else {"mean": sum(eff) / len(eff), "se": None,
                                                       "n": len(eff), "median": 0, "min": 0, "max": 0},
                "capability": rate(sum(1 for r in caps if r["capability_correct"]), len(caps)),
                "judge_errors": sum(1 for r in rows for e in (r.get("judge_error") or {}).values() if e),
                "self_report": rate(sum(1 for s in sr if s["claims"] == "YES"), len(sr)),
                "self_report_names_concept": rate(sum(1 for s in sr if s["matches"]), len(sr))}

    def boundary(self, layer: int, *, outcome="ok", failing=True, old_vocabulary=False):
        dm = self.dose_max[layer]
        reach = round(dm * (1.0 if old_vocabulary else 1.4), 6)
        return {"layer": layer, "dose_max": dm, "outcome": outcome,
                "max_reachable_dose": reach, "probes_to_reach_floor": 8, "probes_used": 4,
                "bisect_used": 2, "lowest_failing_dose": None if not failing else round(dm * 1.08, 6),
                "bracket": [0.05, 2.5], "step_ratio": 0.7,
                "probes": [{"stage": "ladder", "dose": dm, "alpha": 10.0, "coherence": 7.0,
                            "coherence_n": 5, "degeneration": 0.0, "on_task": 1.0, "answered": 1.0,
                            "good": 1.0, "sane": True, "unjudged": 0}],
                "highest_probe": dm, "lowest_probe": dm}

    def finish(self, *, boundaries=None, summary: bool = True, nulls: bool = True,
               null_repeats: int = 3, settings: dict | None = None) -> Path:
        bounds = [self.boundary(L) for L in self.layers] if boundaries is None else boundaries
        for b in bounds:
            self.add(ar.BOUNDARIES, self.stamp(b))
        for L in self.layers:
            self.add(ar.NORMS, self.stamp({"layer": L, "vec_norm": 1000.0, "resid_norm": 40000.0,
                                           "resid_over_vec": 40.0, "alpha_ceil": 16.0,
                                           "max_reachable_dose": 0.4}))
        if nulls:
            for k in range(null_repeats):
                for i in range(self.n_identify):
                    text = f"the word \"apple\" {MARK}n{i}"
                    self.add(ar.NULLS, self.stamp(
                        {"channel": "identify", "response": text, "words": len(text.split()),
                         "concept_mentions": 0, "degenerate": False, "degeneration_reason": None,
                         "empty": False, "unit": f"trial_{1 + 6 * i}", "layer": None, "dose": None,
                         "alpha": 0.0, "steered": False, "repeat": k}))
                for unit, _ in TASKS:
                    text = f"A plain story {MARK}nt"
                    self.add(ar.NULLS, self.stamp(
                        {"channel": "effect", "response": text, "words": 4, "concept_mentions": 0,
                         "degenerate": False, "degeneration_reason": None, "empty": False,
                         "unit": unit, "layer": None, "dose": None, "alpha": 0.0, "steered": False,
                         "repeat": k}))
        for name, rows in self.files.items():
            with open(self.dir / name, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if summary:
            cfg = {"MODEL": "gemma3_27b", "N_IDENTIFY": self.n_identify, "N_EFFECT": 4,
                   "N_SELF_REPORT": 3, "MAX_NEW_TOKENS": 100, "DOSE_FRACTIONS": list(self.fractions),
                   "JUDGE_ENABLED": 1, **(settings or {})}
            s = {"concept": self.concept, "mode": "sweep", "version": "m3", "layers": self.layers,
                 "dose_fractions": list(self.fractions), "n_cells_planned": 0, "n_cells_on_disk": 0,
                 "n_cells_this_attempt": 0, "skipped": [], "boundaries": bounds, "liveness": {},
                 "elapsed_s": 1.0, "config": cfg, "config_hash": self.hash,
                 "ts": "2026-08-20T10:00:00+00:00"}
            (self.dir / ar.SUMMARY).write_text(json.dumps(s, indent=2), encoding="utf-8")
        return self.dir


def hazard_run(tmp_path: Path) -> Path:
    """The specification's hazards in one run (see each test for which one it checks)."""
    w = RunWriter(tmp_path, layers=(27, 29, 31))
    for L in (27, 29):
        for f in FRACTIONS:
            if (L, f) == (29, 0.85):
                continue
            w.cell(L, f, match=lambda i, f=f: i < int(30 * f) - 9,
                   influence=lambda j, f=f: 10 * f, fail_identify=(3,) if (L, f) == (27, 0.35) else (),
                   degenerate=(0, 1) if (L, f) == (29, 0.45) else ())
    # a re-measured cell: the first battery (a crashed pass: every trial a miss) then the second
    w.cell(29, 0.85, match=lambda i: True, influence=lambda j: 9.0)
    # layer 31 has cells but will get no boundary row
    w.cell(31, 0.55, match=lambda i: i % 2 == 0, influence=lambda j: 3.0)
    bounds = [w.boundary(27, outcome="ok", failing=False, old_vocabulary=True),  # pre-c04ab0f "ok"
              w.boundary(29)]
    run = w.finish(boundaries=bounds)
    # the duplicate: the crashed pass's rows come FIRST in the files (no attempt stamps)
    resp = (run / ar.RESPONSES).read_text().split("\n")
    first = [json.loads(x) for x in resp if x and json.loads(x)["layer"] == 29
             and json.loads(x)["dose"] == w.dose(29, 0.85)]
    crashed = []
    for r in first:
        r = dict(r)
        if r["channel"] == "identify":
            r["judged"] = {"identify": {"named": "lemon", "matches": False}}
        crashed.append(json.dumps(r))
    cells = [json.loads(x) for x in (run / ar.CELLS).read_text().split("\n") if x]
    old_cell = next(c for c in cells if c["layer"] == 29 and c["dose"] == w.dose(29, 0.85))
    old_cell = dict(old_cell, identification={"rate": 0.0, "count": 0, "n": 30, "ci_low": 0.0,
                                              "ci_high": 0.1, "ci_z": 1.96})
    (run / ar.RESPONSES).write_text("\n".join(crashed) + "\n" + "\n".join(resp), encoding="utf-8")
    (run / ar.CELLS).write_text(json.dumps(old_cell) + "\n" + (run / ar.CELLS).read_text(),
                                encoding="utf-8")
    # a torn last line: the process died mid-append
    with open(run / ar.RESPONSES, "a", encoding="utf-8") as f:
        f.write('{"channel": "identify", "response": "half a ro')
    with open(run / ar.CELLS, "a", encoding="utf-8") as f:
        f.write('{"layer": 31, "dose": 0.')
    return run


# =====================================================================================
# The reader
# =====================================================================================

def test_reader_dedupes_joins_tolerates_a_torn_tail_and_excludes_failed_verdicts(tmp_path):
    run = hazard_run(tmp_path)
    a = ar.load_archive(run)
    rep = a.report
    assert rep.files[ar.RESPONSES].torn_last_line and rep.files[ar.CELLS].torn_last_line
    # the re-measured cell: the LAST copy wins, in the trials and in the stored row
    c = a.cell(29, fraction=0.85)
    assert c is not None and c.identification == (30, 30)
    assert c.stored["identification"]["count"] == 30 and not c.stored_mismatch
    assert rep.files[ar.RESPONSES].duplicates_dropped == 43
    assert rep.files[ar.CELLS].duplicates_dropped == 1
    assert len([x for x in a.cells if (x.layer, x.fraction) == (29, 0.85)]) == 1
    # the judge failure leaves the denominator and is counted
    c27 = a.cell(27, fraction=0.35)
    assert c27.ident_n == 29 and c27.ident_failed == 1
    bad = [t for t in a.trials(c27, "identify") if t.matches is None]
    assert len(bad) == 1 and bad[0].missing["identify"].startswith("judge_failed")
    # the join: fraction snapped, dose_max and outcome from the layer's boundary
    assert c27.dose_max == pytest.approx(0.47) and c27.fraction == 0.35
    # the older vocabulary: a ceiling-bound layer that read "ok"
    assert a.boundaries[27].outcome == "ceiling_limited"
    assert a.boundaries[27].outcome_raw == "ok"
    assert "outcome_vocabulary_pre_c04ab0f" in rep.drift
    # the cell without a boundary row is kept, measured, and placed nowhere on the grid
    c31 = a.cell(31, a.cells[-1].dose)
    assert c31 is not None and c31.fraction is None and not c31.has_boundary
    assert rep.cells_without_boundary == [c31.cell_id]
    # drift recorded explicitly
    assert {"no_attempt_stamps", "no_n_empty"} <= set(rep.drift)
    # degenerate rows stay in identification, leave the clean view
    c29 = a.cell(29, fraction=0.45)
    assert c29.ident_degenerate == 2 and c29.ident_clean_n == c29.ident_n - 2


def test_trial_numbers_are_parsed_from_the_unit_and_trials_carry_no_text(tmp_path):
    w = RunWriter(tmp_path, layers=(29,))
    w.cell(29, 0.85, match=lambda i: True)
    a = ar.load_archive(w.finish())
    ts = a.trials(a.cells[0], "identify")
    assert [t.trial for t in ts[:3]] == [1, 7, 13] and ts[-1].trial == 1 + 6 * 29
    assert ar.parse_trial("task_story") is None and ar.parse_trial("trial_x") is None
    for t in a.trials():
        assert not any(MARK in str(v) for v in vars(t).values())
    assert MARK in a.text(ts[0].transcript_id)
    sr = a.trials(a.cells[0], "self_report")
    assert [t.trial for t in sr] == [1, 7, 13] and ts[0].transcript_id != sr[0].transcript_id


def test_interior_corruption_raises_unless_allowed_and_then_is_counted(tmp_path):
    w = RunWriter(tmp_path, layers=(29,))
    w.cell(29, 0.85)
    run = w.finish()
    lines = (run / ar.RESPONSES).read_text().split("\n")
    lines.insert(5, "\x00\x00\x00 not json")
    (run / ar.RESPONSES).write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ar.ArchiveCorrupt, match="line 6"):
        ar.load_archive(run)
    a = ar.load_archive(run, allow_corrupt=True)
    assert a.report.files[ar.RESPONSES].corrupt_lines == [6]
    # a final line with its newline that does not parse is still a torn tail
    (run / ar.CELLS).write_text((run / ar.CELLS).read_text() + "\x00\x00\x00\n")
    assert ar.load_archive(run, allow_corrupt=True).report.files[ar.CELLS].torn_last_line


def test_records_are_split_on_newline_only(tmp_path):
    w = RunWriter(tmp_path, layers=(29,))
    w.cell(29, 0.85, named=lambda i: "lemon pie\u0085")
    a = ar.load_archive(w.finish())
    assert a.cells[0].ident_n == 30
    assert " " in a.trials(a.cells[0], "identify")[0].named


def test_attempt_stamps_decide_duplicates_when_present(tmp_path):
    w = RunWriter(tmp_path, layers=(29,), attempt="2026-08-24T12:00:00+00:00")
    w.cell(29, 0.85, match=lambda i: True)                                   # the later attempt...
    w.cell(29, 0.85, match=lambda i: False, attempt="2026-08-24T09:00:00+00:00")  # ...written after
    a = ar.load_archive(w.finish())
    assert a.cells[0].identification == (30, 30)
    assert "no_attempt_stamps" not in a.report.drift
    rows = [{"k": 1, "attempt": "unset", "v": 1}, {"k": 1, "attempt": None, "v": 2}]
    kept, dropped = ar.dedupe(rows, ("k",))
    assert dropped == 1 and kept[0]["v"] == 2   # "unset" ranks with absent: the later copy wins


def test_a_stored_cell_that_disagrees_with_its_trials_is_reported(tmp_path):
    w = RunWriter(tmp_path, layers=(29,))
    w.cell(29, 0.85, match=lambda i: True,
           stored_override={"identification": {"rate": 0.5, "count": 15, "n": 30}})
    a = ar.load_archive(w.finish())
    assert a.cells[0].stored_mismatch == ("identification",)
    assert a.report.stored_mismatches == [a.cells[0].cell_id]
    assert a.cells[0].identification == (30, 30)   # the trials are what is used


def test_unjudged_runs_read_as_missing_never_zero(tmp_path):
    w = RunWriter(tmp_path, layers=(29,))
    w.cell(29, 0.85, match=lambda i: True)
    run = w.finish(settings={"JUDGE_ENABLED": 0})
    rows = [json.loads(x) for x in (run / ar.RESPONSES).read_text().split("\n") if x]
    for r in rows:
        r.pop("judged", None)
        r.pop("judge_error", None)
    (run / ar.RESPONSES).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    a = ar.load_archive(run)
    c = a.cells[0]
    assert c.identification is None and c.ident_not_judged == 30 and c.ident_failed == 0
    assert c.influence == () and "judging_disabled" in a.report.drift
    from m5.analysis.refit import refit
    res = refit([a], config.load({"POSTERIOR_DRAWS": 100, "BOOTSTRAP_REPS": 10}), unit="dose",
                floor_control="null_literal", with_effect=True, winners_curse_reps=20)
    assert [r["reason"] for r in res.rows if r["kind"] == "refit_skipped"] == ["no_valid_d2_trial"]
    assert not res.fits


def test_the_reader_refuses_repo_paths_vectors_and_the_harmful_arm(tmp_path, monkeypatch):
    with pytest.raises(safety.Refused):
        ar.load_archive(safety.REPO_ROOT / "m5")
    w = RunWriter(tmp_path, layers=(29,))
    w.cell(29, 0.85)
    run = w.finish()
    (run / "vectors").mkdir()
    (run / "vectors" / "Garlic.pt").write_bytes(b"\x00not a tensor")
    with pytest.raises(safety.Refused, match="never read"):
        ar._guard_path(run / "vectors" / "Garlic.pt", run)
    with pytest.raises(safety.Refused):
        ar._guard_path(run / "read_this.md", run)
    opened = []
    real_open = open

    def spy(path, *args, **kw):
        opened.append(str(path))
        return real_open(path, *args, **kw)

    monkeypatch.setattr("builtins.open", spy)
    ar.load_archive(run)
    monkeypatch.setattr("builtins.open", real_open)
    assert opened and not any("vectors" in p or p.endswith(".pt") for p in opened)
    h = RunWriter(tmp_path / "h", concept="Poison", hash12="0123456789ab", layers=(29,))
    h.cell(29, 0.85)
    harmful = h.finish()
    opened.clear()
    monkeypatch.setattr("builtins.open", spy)
    with pytest.raises(safety.Refused, match="harmful arm"):
        ar.load_archive(harmful)
    monkeypatch.setattr("builtins.open", real_open)
    assert not any(p.endswith(ar.RESPONSES) or p.endswith(ar.NULLS) for p in opened)


def test_an_export_without_transcripts_falls_back_to_the_stored_rows(tmp_path, small_cfg):
    """Older exports may have withheld transcript files (spec §2.3): the run's own aggregates are
    then used for what they hold, and nothing per prompt is invented."""
    from m5.analysis.coupling import coupling, coupling_cuts
    from m5.analysis.refit import fit_concept

    w = RunWriter(tmp_path, layers=(27, 29))
    for L in (27, 29):
        for f in FRACTIONS:
            w.cell(L, f, match=lambda i, f=f: i < int(30 * f), influence=lambda j, f=f: 10 * f)
    run = w.finish()
    (run / ar.RESPONSES).unlink()
    (run / ar.NULLS).unlink()
    a = ar.load_archive(run)
    c = a.cell(29, fraction=0.65)
    assert c.source == "stored_row" and c.identification == (int(30 * 0.65), 30)
    assert c.influence == () and c.influence_mean == pytest.approx(6.5) and c.influence_n == 4
    assert "cells_from_stored_rows_only" in a.report.drift and a.trials(c) == []
    fit = fit_concept(a, small_cfg, unit="fraction", floor_control="null_literal", with_effect=True)
    assert fit.control is None and len(fit.chain_layers) == 2
    assert {e["reason"] for e in fit.excluded_from_rstar} == {"no_per_prompt_influence:stored_rows_only"}
    rows = coupling([a], small_cfg, bin_by="fraction", **coupling_cuts(small_cfg))
    assert next(r for r in rows if r["kind"] == "coupling_summary")["n_cells"] == 12


def test_rows_copied_from_another_run_are_dropped_and_counted(tmp_path):
    w = RunWriter(tmp_path, layers=(29,))
    w.cell(29, 0.85)
    run = w.finish()
    row = json.loads((run / ar.CELLS).read_text().split("\n")[0])
    row.update(config_hash="ffffffffffff", dose=0.123456)
    (run / ar.CELLS).write_text((run / ar.CELLS).read_text() + json.dumps(row) + "\n")
    a = ar.load_archive(run)
    assert a.report.files[ar.CELLS].foreign_rows == 1 and len(a.cells) == 1


def test_find_runs_takes_a_folder_of_runs_and_refuses_zips(tmp_path):
    for concept, h in (("Garlic", "aaaaaaaaaaaa"), ("Silk", "bbbbbbbbbbbb")):
        w = RunWriter(tmp_path / "root", concept=concept, hash12=h, layers=(29,))
        w.cell(29, 0.85)
        w.finish()
    (tmp_path / "root" / "garlic_cccccccccccc").mkdir()    # M3's stray empty folder
    got = ar.load_archives(tmp_path / "root")
    assert [a.concept for a in got] == ["Garlic", "Silk"]
    (tmp_path / "x.zip").write_bytes(b"PK")
    with pytest.raises(ar.ArchiveError, match="unzip"):
        ar.find_run_dirs(tmp_path / "x.zip")


def test_unknown_and_missing_outcomes_are_explicit():
    assert ar.normalise_outcome({"outcome": "probes_exhausted", "dose_max": 0.1}) == ("probes_exhausted", None)
    assert ar.normalise_outcome({"outcome": "weird"})[0] == "unknown"
    assert ar.normalise_outcome({"outcome": "ok", "dose_max": 2.5, "lowest_failing_dose": None,
                                 "max_reachable_dose": 9.0, "bracket": [0.05, 2.5]})[0] == "bracket_limited"
    assert ar.snap_fraction(0.123, 0.5, FRACTIONS) == (None, None)
    assert ar.snap_fraction(round(0.47 * 0.85, 6), 0.47, FRACTIONS)[0] == 0.85


# =====================================================================================
# The aggregate-row guard
# =====================================================================================

def test_check_aggregate_refuses_anything_that_could_carry_text():
    check_aggregate({"kind": "x", "concept": "Garlic", "layer": 3, "rate": 0.5, "flags": ["floor"],
                     "missing": {"identify": "judge_failed: x"}, "p_best": {29: 0.4}, "v": None})
    for bad in ({"kind": "x", "named": "lemon"},                    # a text field by name
                {"kind": "x", "comment": "the model said lemon"},   # a string under a non-id key
                {"kind": "x", "note": "two\nlines"},                # an identifier never spans lines
                {"kind": "x", "rate": float("nan")},                # missing is None, not NaN
                {"kind": "x", "inner": {"text": "hello"}},
                {"kind": "x", "values": ["lemon"]}):
        with pytest.raises(safety.Refused):
            check_aggregate(bad)


def test_archive_rows_are_aggregate(tmp_path):
    a = ar.load_archive(hazard_run(tmp_path))
    rows = ar.archive_rows(a)
    for r in rows:
        check_aggregate(r)
    blob = json.dumps(rows)
    assert MARK not in blob and "lemon" not in blob


# =====================================================================================
# Refit
# =====================================================================================

TRUE_S = 0.15


def curve_run(tmp_path, *, n_identify=60, seed=1):
    """Identification drawn from a known logistic in log dose per layer: two layers with the
    threshold inside the measured range, one that never rises, one already risen."""
    dm = {20: 0.30, 22: 0.40, 24: 0.50, 26: 0.35}
    mu = {20: math.log(dm[20] * 0.55), 22: math.log(dm[22] * 0.68),
          24: math.log(dm[24] * 4.0), 26: math.log(dm[26] * 0.08)}
    w = RunWriter(tmp_path, layers=tuple(dm), dose_max=dm, n_identify=n_identify, seed=seed)
    rng = np.random.default_rng(seed)
    for L in dm:
        for f in FRACTIONS:
            u = math.log(w.dose(L, f))
            p = logistic((u - mu[L]) / TRUE_S)
            hits = rng.random(n_identify) < p
            w.cell(L, f, match=lambda i, h=hits: bool(h[i]), named=lambda i, h=hits: "garlic" if h[i] else "apple",
                   influence=lambda j, u=u, L=L: round(10 * logistic((u - mu[L] + 0.2) / 0.2), 1))
    return w.finish(), mu


def test_refit_recovers_known_thresholds_and_flags_the_unidentifiable(tmp_path, small_cfg):
    from m5.analysis.refit import fit_concept, fit_rows

    run, mu = curve_run(tmp_path)
    a = ar.load_archive(run)
    fit = fit_concept(a, small_cfg, unit="dose", floor_control="null_literal", with_effect=False)
    assert fit.control == (0, 3 * 60)   # the null arm names "apple", never the concept
    q = small_cfg["Q"]
    for L in (20, 22):
        lf = fit.layers[L]
        t = next(x for x in lf.own["thresholds"] if x["q"] == q)
        true_ud = mu[L] + TRUE_S * math.log(q / (1 - q))
        assert lf.identified, lf.reason
        assert abs(t["median"] - true_ud) < 0.2, (L, t["median"], true_ud)
        assert t["lo"] <= true_ud <= t["hi"]
        assert abs(lf.own["mu_mean"] - mu[L]) < 0.15
        assert lf.own["s_lo"] <= TRUE_S <= lf.own["s_hi"]
        assert t["dose_lo"] <= math.exp(true_ud) <= t["dose_hi"]
        assert lf.coupled is not None   # coupled by the chain too
    assert (fit.layers[24].identified, fit.layers[24].reason) == (False, "above_range")
    assert (fit.layers[26].identified, fit.layers[26].reason) == (False, "below_range")
    rows = fit_rows(fit, small_cfg)
    for r in rows:
        check_aggregate(r)
    table = {r["layer"]: r["reason"] for r in rows if r["kind"] == "refit_identifiability"}
    assert table == {20: "identified", 22: "identified", 24: "above_range", 26: "below_range"}
    cells = [r for r in rows if r["kind"] == "refit_cell" and r["layer"] == 20]
    assert len(cells) == 6 and all(r["lo"] <= r["rate"] <= r["hi"] for r in cells)


def test_layers_without_cells_or_with_one_dose_get_a_reason(tmp_path, small_cfg):
    from m5.analysis.refit import fit_concept

    w = RunWriter(tmp_path, layers=(27, 29, 31))
    w.cell(29, 0.55, match=lambda i: i % 3 == 0)
    w.cell(31, 0.55, match=lambda i: i % 3 == 0)     # layer 31 will have no boundary row
    bounds = [w.boundary(29), {"layer": 13, "dose_max": None, "outcome": "unreachable",
                               "max_reachable_dose": 0.02, "bracket": [0.05, 2.5], "probes": []}]
    a = ar.load_archive(w.finish(boundaries=bounds))
    fit = fit_concept(a, small_cfg, unit="fraction", floor_control="none", with_effect=True)
    assert fit.layers[13].reason == "no_cells:boundary_unreachable"
    assert fit.layers[29].reason == "one_dose"
    assert fit.layers[31].reason == "cells_not_on_axis:no_boundary_row"
    on_dose = fit_concept(a, small_cfg, unit="dose", floor_control="none", with_effect=False)
    assert on_dose.layers[31].reason == "one_dose"     # on M3's own dose axis it is placed
    assert fit.control is None and fit.excluded_from_rstar == []
    assert fit.rstar is not None and fit.rstar[0]["best_layer"] == 29   # one dose still has an R_l


def test_crossed_icc_recovers_a_prompt_effect_and_is_none_when_undefined(small_cfg):
    from m5.analysis.refit import crossed_icc, prompt_icc

    rng = np.random.default_rng(3)
    cells, prompts, sd_p = 200, 30, 0.8
    eff = rng.normal(0, sd_p, prompts)
    base = rng.normal(0, 0.5, cells)
    latent = base[:, None] + eff[None, :]
    y = (rng.random((cells, prompts)) < 1 / (1 + np.exp(-latent))).astype(float)
    icc = crossed_icc(y)
    shuffled = np.array([rng.permutation(r) for r in y])     # destroys the prompt effect
    assert icc > 0.05 and crossed_icc(shuffled) < 0.02
    assert crossed_icc(np.zeros((5, 4))) is None and crossed_icc(np.ones((1, 4))) is None
    table = {c: {f"t{p}": y[c, p] for p in range(prompts)} for c in range(40)}
    table[0].pop("t3")                       # a failed verdict costs a cell or a prompt, not zero
    res = prompt_icc(table, estimand="d2", cfg=small_cfg, seed=1)
    assert res.icc is not None and res.lo <= res.icc <= res.hi
    assert res.n_prompts * res.n_cells == max(39 * 30, 40 * 29)
    flat = prompt_icc({c: {"a": 0.0, "b": 0.0} for c in range(5)}, estimand="d2", cfg=small_cfg, seed=1)
    assert flat.icc is None and flat.reason.startswith("fewer_than_2")


def test_refit_reports_icc_design_effects_and_a_calibrated_winners_curse(tmp_path, small_cfg):
    from m5.analysis.refit import refit

    run, _ = curve_run(tmp_path, n_identify=30)
    a = ar.load_archive(run)
    res = refit([a], small_cfg, unit="dose", floor_control="null_literal", with_effect=True,
                winners_curse_reps=300)
    for r in res.rows:
        check_aggregate(r)
    icc_rows = {r["estimand"]: r for r in res.rows if r["kind"] == "refit_icc"}
    assert set(icc_rows) == {"d2", "d8", "influence"}
    d2 = icc_rows["d2"]
    assert d2["icc"] is not None and d2["n_prompts"] == 30
    assert [x["n"] for x in d2["design_effects"]] == [small_cfg["SPRT_NMAX"], small_cfg["STOP_TRIALS"],
                                                      small_cfg["CONFIRM_D2_N"]]
    wc = [r for r in res.rows if r["kind"] == "refit_winners_curse"]
    plan = [r for r in wc if (r["n_arms"], r["n"]) == (25, 22) and r["icc"] == 0.0
            and r["p"] == small_cfg["SPRT_P0"]]
    assert len(plan) == 1 and 0.10 < plan[0]["inflation"] < 0.20     # the plan's +14.9 points
    assert any(r["source"].startswith("archive:") for r in wc)
    rstar = [r for r in res.rows if r["kind"] == "refit_rstar"]
    assert [r["q"] for r in rstar] == list(small_cfg["FRONTIER_Q"])
    assert all(r["lo"] <= r["median"] <= r["hi"] for r in rstar)
    assert any(r["kind"] == "refit_wstar" for r in res.rows)


# =====================================================================================
# Replay
# =====================================================================================

def two_level_run(tmp_path):
    """L29: every identify trial a match at 0.85, none at 0.35; L31 measured at 0.85 only."""
    w = RunWriter(tmp_path, layers=(29, 31))
    for f in FRACTIONS:
        w.cell(29, f, match=lambda i, f=f: f >= 0.8, fail_identify=(0,) if f == 0.55 else (),
               influence=lambda j, f=f: 10.0 if f >= 0.8 else 0.0,
               coherence=lambda j: 9.0 if j < 2 else 3.0, degenerate=(0,) if f == 0.85 else ())
    w.cell(31, 0.85, match=lambda i: i < 10)
    return ar.load_archive(w.finish())


def test_archive_measurer_resamples_the_cell_a_request_maps_to(tmp_path, small_cfg):
    from m5.analysis.replay import ArchiveMeasurer, replay_config

    a = two_level_run(tmp_path)
    cfg = replay_config(a, small_cfg)
    m = ArchiveMeasurer(a, cfg, mode="without_replacement", integrity="m3_flags")
    vid = a.vector_id
    hi = m.gate(Cell(vid, "single", (29,), 0.85), 8, "search")["d2"]
    lo = m.gate(Cell(vid, "single", (29,), 0.35), 8, "search")["d2"]
    assert (hi.k, hi.n) == (8, 8) and (lo.k, lo.n) == (0, 8)
    assert all(u.startswith(f"m3/{a.run_id}/L29@") and "/identify/" in u for u in hi.unit_ids)
    other = m.gate(Cell(vid, "single", (31,), 0.85), 30, "search")["d2"]
    assert (other.k, other.n) == (10, 30)
    # without replacement: 30 archived trials, 8 dealt, then 22, then nothing
    rest = m.gate(Cell(vid, "single", (29,), 0.85), 40, "search")["d2"]
    assert rest.n == 22 and rest.extra["short_by"] == 18
    done = m.gate(Cell(vid, "single", (29,), 0.85), 4, "search")["d2"]
    assert done.n == 0 and "already used" in done.missing
    # a failed verdict is dealt like any trial and excluded, never a miss
    mid = m.gate(Cell(vid, "single", (29,), 0.55), 30, "search")["d2"]
    assert mid.n == 29 and mid.excluded == 1
    # with replacement: unbounded, and recycling is recorded
    m2 = ArchiveMeasurer(a, cfg, mode="with_replacement", integrity="m3_flags")
    big = m2.gate(Cell(vid, "single", (29,), 0.85), 100, "search")["d2"]
    assert (big.k, big.n) == (100, 100) and big.extra["recycled"] > 0
    assert len(m.requests) == 6 and m.requests[0].transcript_ids == hi.unit_ids


def test_archive_measurer_refuses_off_grid_and_answers_missing_what_m3_never_measured(tmp_path, small_cfg):
    from m5.analysis.replay import ArchiveMeasurer, OffGrid, replay_config

    a = two_level_run(tmp_path)
    cfg = replay_config(a, small_cfg)
    assert cfg["GATE"] == "naming" and cfg["RHO_GRID"] == FRACTIONS
    m = ArchiveMeasurer(a, cfg, mode="without_replacement", integrity="m3_flags")
    assert m.writes_rows is False and m.source == "m3_archive"
    vid = a.vector_id
    for cell in (Cell(vid, "single", (29,), 0.5),          # between two measured fractions
                 Cell(vid, "single", (29,), 0.95),         # beyond the grid
                 Cell("m3:silk", "single", (29,), 0.85),   # another vector
                 Cell(vid, "single", (29,), 0.85, op="sub")):
        with pytest.raises(OffGrid):
            m.gate(cell, 4, "search")
        with pytest.raises(OffGrid):
            m.effect(cell, 4, "search")
        with pytest.raises(OffGrid):
            m.integrity(cell, 4, "search")
    # grid points M3 did not measure, and the null, are answered missing -- never filled
    for cell, why in ((Cell(vid, "single", (30,), 0.85), "did not measure layer 30"),
                      (Cell(vid, "single", (31,), 0.35), "did not measure layer 31"),
                      (Cell(vid, "single", (29,), 0.0), "never judged")):
        assert why in m.gate(cell, 4, "search")["d2"].missing
        assert why in m.effect(cell, 4, "search").missing
        assert all(why in x.missing for x in m.integrity(cell, 4, "search").values())
    screen = m.screen(Cell(vid, "single", (30,), 0.85), "A")   # never raises: nothing is fabricated
    assert set(screen) == {"e9", "s6", "s2", "d3", "d5", "q2"}
    assert all(r.value is None and "no forward-pass" in r.missing for r in screen.values())
    assert set(m.screen(Cell(vid, "single", (29,), 0.85), "B")) >= {"d6", "d7", "d9"}
    assert "never judged" in m.gate(None, 4, "null")["d2"].missing
    assert "held-out" in m.gate(Cell(vid, "single", (29,), 0.85), 4, "confirm")["d2"].missing
    integ = m.integrity(Cell(vid, "single", (29,), 0.85), 8, "search")
    assert set(integ) == {"s1", "s2"} and integ["s1"].n == 8 and integ["s2"].n == 8
    assert integ["s1"].k == 6       # coherence 9, 9, 3, 3 on effect rows; 9 on every explain row
    assert integ["s2"].k == 0       # the degenerate row of this cell is an identify row
    e = m.effect(Cell(vid, "single", (29,), 0.85), 4, "search")
    assert (e.k, e.n) == (4, 4) and e.values == [1.0] * 4 and e.code == "e8"
    # measure(): the confirm tier's entry point
    cell = Cell(vid, "single", (29,), 0.35)
    assert m.measure("d2", cell, 4, "search").n == 4
    assert "paired panel" in m.measure("s5", cell, 4, "search").missing
    assert "no measure behind" in m.measure("d7", cell, 4, "search").missing
    assert "original prompts" in m.measure("d8", cell, 4, "search", variant="primer").missing
    # integrity="boundary": M3's boundary is the integrity evidence, nothing else is offered
    b = ArchiveMeasurer(a, cfg, mode="without_replacement", integrity="boundary")
    assert all("boundary" in x.missing for x in b.integrity(Cell(vid, "single", (29,), 0.85), 4,
                                                            "search").values())
    with pytest.raises(ValueError):
        ArchiveMeasurer(a, cfg, mode="without_replacement", integrity="vibes")


def test_a_replay_needs_a_grid_of_two_doses_and_cells_on_it(tmp_path, small_cfg):
    from m5.analysis.replay import ReplayError, replay_config

    w = RunWriter(tmp_path / "one", layers=(29,), fractions=(0.5,))
    w.cell(29, 0.5)
    with pytest.raises(ReplayError, match="at least two"):
        replay_config(ar.load_archive(w.finish()), small_cfg)
    w = RunWriter(tmp_path / "cells", layers=(29,))
    w.cell(29, 0.85)
    off = ar.load_archive(w.finish(boundaries=[]))            # no boundary: no fraction anywhere
    with pytest.raises(ReplayError, match="fraction grid"):
        replay_config(off, small_cfg)


def test_replay_without_m5_search_says_so(tmp_path, small_cfg, monkeypatch):
    import importlib

    from m5.analysis import replay

    real = importlib.import_module

    def fake(name, *a, **k):
        if name.startswith("m5.search"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(name, *a, **k)

    monkeypatch.setattr(replay.importlib, "import_module", fake)
    with pytest.raises(replay.SearchUnavailable, match="m5.search"):
        replay.replay_search(two_level_run(tmp_path), small_cfg, mode="without_replacement",
                             integrity="m3_flags", floor_control="none")


class _StandInSearch:
    """Stands in for m5.search.AdaptiveSearch: visits every grid cell of the layers its screen
    kept, the way the documented interface is used (`AdaptiveSearch(ctx, measurer, vector_id,
    screen).run()`), and visits only `doses` when given."""

    doses = None

    def __init__(self, ctx, measurer, vector_id, screen):
        self.ctx, self.m, self.vid, self.layers = ctx, measurer, vector_id, screen

    def run(self):
        grid = self.doses or self.ctx.cfg["RHO_GRID"]
        for L in self.layers:
            for rho in grid:
                cell = Cell(self.vid, "single", (L,), rho)
                self.m.gate(cell, 60, "search")
                self.m.effect(cell, 4, "search")
                self.m.integrity(cell, 4, "search")
        return SimpleNamespace(visited=len(self.layers) * len(grid))


def _stand_in_screen(ctx, measurer, vector_id):
    from m5.dose import band_layers

    layers = band_layers(ctx.cfg["LAYER_BAND"], ctx.backend.info.n_layers)[::ctx.cfg["SCREEN_STRIDE"]]
    for L in layers:
        measurer.screen(Cell(vector_id, "single", (L,), ctx.cfg["RHO_GRID"][0]), "A")
    return layers


def test_replay_recovers_the_full_grid_when_the_search_visits_every_cell(tmp_path, small_cfg):
    from m5.analysis.replay import SearchAPI, replay_search

    run, _ = curve_run(tmp_path, n_identify=30)
    a = ar.load_archive(run)
    api = SearchAPI(_stand_in_screen, _StandInSearch, "stand-in")
    res = replay_search(a, small_cfg, mode="without_replacement", integrity="m3_flags",
                        floor_control="null_literal", search_api=api)
    c = res.comparison
    assert c["recovered"] is True and c["intervals_overlap"] is True
    assert res.search_vs_survey["available"] is False     # the stand-in has no survey
    assert res.spend["d2_trials_drawn"] == res.spend["d2_trials_archived"] == 4 * 6 * 30
    assert c["threshold_inside_share"] == 1.0
    for r in res.rows:
        check_aggregate(r)
    assert sum(r["kind"] == "replay_request" for r in res.rows) == len(res.measurer.requests)
    # the off-grid refusal reaches the caller with the replay's explanation
    from m5.analysis.replay import OffGrid

    class Wanders(_StandInSearch):
        doses = (0.5,)

    with pytest.raises(OffGrid, match="refuses rather than interpolate"):
        replay_search(a, small_cfg, mode="without_replacement", integrity="m3_flags",
                      floor_control="none", search_api=SearchAPI(_stand_in_screen, Wanders, "stand-in"))


def test_replay_with_a_sparse_search_still_compares(tmp_path, small_cfg):
    from m5.analysis.replay import SearchAPI, replay_search

    run, _ = curve_run(tmp_path, n_identify=30)
    a = ar.load_archive(run)

    class Sparse(_StandInSearch):
        doses = (0.35, 0.55, 0.85)

    res = replay_search(a, small_cfg, mode="without_replacement", integrity="boundary",
                        floor_control="none", search_api=SearchAPI(_stand_in_screen, Sparse, "stand-in"))
    assert res.spend["d2_trials_drawn"] == 4 * 3 * 30
    assert res.comparison["recovered"] in (True, False) and res.comparison["full"] is not None


def test_replay_with_the_real_search_compares_both_ways(tmp_path, small_cfg):
    """Integration with m5.search (written beside this module): the search and its own
    full-grid survey run through ArchiveMeasurers of the same settings, and the search's rows
    land in the analysis run."""
    pytest.importorskip("m5.search")
    from m5.analysis.replay import replay_search

    run_dir, _ = curve_run(tmp_path / "archive", n_identify=30)
    a = ar.load_archive(run_dir)
    run = runio.RunDir.create(tmp_path / "runs", "replay", config=small_cfg, kind="analysis")
    res = replay_search(a, small_cfg, mode="without_replacement", integrity="m3_flags",
                        floor_control="null_literal", run=run)
    run.close()
    svs, refit_cmp = res.search_vs_survey, res.comparison
    assert svs["available"] is True, svs
    for c in (svs, refit_cmp):
        assert isinstance(c["recovered"], bool) and c["full"]["best_layer"] in a.layers
        assert c["full"]["lo"] <= c["full"]["median"] <= c["full"]["hi"]
    assert svs["survey_n_gate"] == 30 and svs["survey_n_effect"] == 4
    assert 0 < res.spend["d2_trials_drawn"] <= res.spend["d2_trials_archived"]
    assert all(r.missing or r.layer in a.layers for r in res.measurer.requests)
    for r in res.rows:
        check_aggregate(r)
    logged = runio.RunDir.open(run.path).rows("search_log")
    assert any(r.get("event") == "visit" for r in logged)


# =====================================================================================
# Coupling
# =====================================================================================

def test_coupling_counts_decoupled_cells_within_bins(tmp_path, small_cfg):
    from m5.analysis.coupling import coupling, coupling_cuts

    w = RunWriter(tmp_path, layers=(21, 23, 25, 27, 29))
    for k, L in enumerate((21, 23, 25, 27, 29)):
        for f in FRACTIONS:
            if L == 21:   # named on most trials, no visible influence: Mentor 1's case
                w.cell(L, f, match=lambda i: i < 25, influence=lambda j: 0.0)
            elif L == 23:  # strong influence, never named: the window's shape
                w.cell(L, f, match=lambda i: False, influence=lambda j: 7.0)
            else:          # naming tracks influence across layers within each bin
                w.cell(L, f, match=lambda i, k=k, f=f: i < 3 * k + int(10 * f),
                       influence=lambda j, k=k, f=f: 1.0 + k + f)
    a = ar.load_archive(w.finish())
    cuts = coupling_cuts(small_cfg)
    assert cuts == {"naming_high": 0.5, "naming_low": small_cfg["Q"], "influence_high": 4.0,
                    "influence_low": 1.0}
    rows = coupling([a], small_cfg, bin_by="fraction", **cuts)
    for r in rows:
        check_aggregate(r)
    summ = next(r for r in rows if r["kind"] == "coupling_summary")
    assert summ["named_not_influenced"]["k"] == 6 and summ["named_not_influenced"]["n"] == 30
    assert summ["influenced_not_named"]["k"] == 6
    lo, hi = summ["named_not_influenced"]["lo"], summ["named_not_influenced"]["hi"]
    assert lo < 0.2 < hi
    bins = [r for r in rows if r["kind"] == "coupling_bin"]
    assert len(bins) == 6 and all(b["n_cells"] == 5 for b in bins)
    assert summ["rank_coupling"]["n_strata"] == 6
    rows2 = coupling([a], small_cfg, bin_by="log_dose", **cuts)
    assert sum(r["n_cells"] for r in rows2 if r["kind"] == "coupling_bin") == 30
    with pytest.raises(ValueError):
        coupling([a], small_cfg, bin_by="fraction", naming_high=0.1, naming_low=0.5,
                 influence_high=4.0, influence_low=1.0)


# =====================================================================================
# Near-misses
# =====================================================================================

def test_near_misses_are_scored_under_every_rule_and_names_stay_out_of_aggregates(tmp_path, small_cfg):
    from m5.analysis.nearmiss import near_misses

    names = ["Wrists", "wrist", "wristwatches", "hand", "NONE", NAME_MARK]
    w = RunWriter(tmp_path, concept="Wrists", hash12="2923c257797c", layers=(29,), n_identify=12)
    w.cell(29, 0.85, named=lambda i: names[i % 6], match=lambda i: i % 6 in (0, 1))
    a = ar.load_archive(w.finish())

    calls = []

    def pair_judge(template, named, concept):
        calls.append((template, named))
        if named == NAME_MARK:
            return None                              # the judge did not answer
        return {"same_concept": False, "related": named == "hand"}[template]

    res = near_misses(a, small_cfg, pair_judge=pair_judge,
                      embed=lambda s: [1.0, 0.0] if "wrist" in s else [0.0, 1.0],
                      semantic_thresholds=(0.5,))
    summ = next(r for r in res.rows if r["kind"] == "nearmiss_summary")
    rules = summ["rules"]
    assert (rules["m3_judge"]["k"], rules["strict"]["k"], rules["lemma"]["k"],
            rules["synonym"]["k"], rules["root"]["k"]) == (4, 2, 4, 4, 6)
    assert rules["same_concept"]["k"] == 4 and rules["same_concept"]["undecided"] == 2
    assert rules["related"]["k"] == 8 and rules["related"]["undecided"] == 2
    assert summ["semantic"]["above"][0]["k"] == 6
    cell = next(r for r in res.rows if r["kind"] == "nearmiss_cell")
    assert cell["rescued_by_root"] == 2 and cell["judge_only"] == 0
    for r in res.rows:
        check_aggregate(r)
    blob = json.dumps(res.rows)
    assert NAME_MARK not in blob and "wristwatches" not in blob
    assert {t["text"] for t in res.texts} >= {"wristwatches", NAME_MARK}
    ids = {r["name_id"] for r in res.rows if r["kind"] == "nearmiss_name"}
    assert ids == {t["name_id"] for t in res.texts} and len(ids) == 5   # NONE is not a name
    assert len(calls) == len(set(calls))   # one judgment per distinct name, not per trial
    from m5.analysis import concept_name_id
    assert concept_name_id("Wrists", "The Hand") == concept_name_id("Wrists", "hand") in ids
    no_judge = near_misses(a, small_cfg)
    s2 = next(r for r in no_judge.rows if r["kind"] == "nearmiss_summary")
    assert s2["rules"]["same_concept"]["undecided"] == 12 and s2["rules"]["same_concept"]["rate"] is None
    assert s2["rules"]["strict"]["rate"] == pytest.approx(1 / 6) and s2["semantic"] is None
    assert {x["rule"] for x in s2["paired_vs_m3_judge"]} == {"strict", "lemma", "synonym", "root"}


# =====================================================================================
# Truncation
# =====================================================================================

def test_truncation_cuts_judges_and_pairs_every_length(tmp_path, small_cfg):
    from m5.analysis.truncation import truncate, truncation_curve, whitespace_offsets

    w = RunWriter(tmp_path, concept="Silk", hash12="2923c257797c", layers=(29,))
    w.cell(29, 0.85)
    run = w.finish()
    rows = [json.loads(x) for x in (run / ar.RESPONSES).read_text().split("\n") if x]
    for r in rows:
        if r["channel"] == "effect":
            pos = {"task_story": 5, "task_landscape": 30, "task_words": 60, "task_notice": 90}[r["unit"]]
            words = [f"w{i}" for i in range(95)]
            words[pos] = "silk"
            r["response"] = " ".join(words)
    (run / ar.RESPONSES).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    a = ar.load_archive(run)
    seen = []

    def judge(items):
        seen.extend(items)
        out = []
        for it in items:
            if it.payload["text"].endswith("w23") and "silk" not in it.payload["text"]:
                out.append(SimpleNamespace(label=None, missing="t1_missing: test"))
            else:
                out.append(SimpleNamespace(label="silk" in it.payload["text"], missing=None))
        return out

    res = truncation_curve(a, small_cfg, judge=judge, tokenizer=None)
    by_len = {r["length"]: r for r in res.rows if r["kind"] == "truncation_length"}
    assert [by_len[L]["k"] for L in (24, 48, 72, 100)] == [1, 2, 3, 4]
    assert by_len[24]["excluded"] == 3 and by_len[24]["n"] == 1   # failed judgments leave n
    assert by_len[48]["versus_longest"]["only_long"] == 2
    assert all(r["tokenizer"] == "whitespace_words_approximate" for r in by_len.values())
    assert {it.template.id for it in seen} == {"allusion"} and len(seen) == 16
    for r in res.rows:
        check_aggregate(r)
    # a subject tokenizer is used when given: here two characters per token
    res2 = truncation_curve(a, small_cfg, judge=judge,
                            tokenizer=lambda t: [(i, min(i + 2, len(t))) for i in range(0, len(t), 2)],
                            lengths=(24,))
    r2 = res2.rows[0]
    assert r2["tokenizer"] == "subject_tokenizer" and (r2["k"], r2["n"], r2["excluded"]) == (1, 4, 0)
    assert truncate("a b c d", 2, whitespace_offsets) == ("a b", 2, True)
    assert truncate("a b", 5, whitespace_offsets) == ("a b", 2, False)
    with pytest.raises(ValueError, match="generated at most"):
        truncation_curve(a, small_cfg, judge=judge, tokenizer=None, lengths=(150,))


# =====================================================================================
# Headline
# =====================================================================================

def silk_run(tmp_path):
    w = RunWriter(tmp_path, concept="Silk", hash12="2923c257797c", layers=(27, 29))
    for L in (27, 29):
        for f in FRACTIONS:
            w.cell(L, f, match=lambda i: False, named=lambda i: NAME_MARK,
                   claims=lambda i: "YES", sr_named=lambda i: NAME_MARK,
                   influence=lambda j, f=f: 4.0 if f >= 0.8 else 1.0)
    return ar.load_archive(w.finish())


def test_headline_cell_collects_every_channel_and_keeps_text_apart(tmp_path):
    from m5.analysis.headline import HeadlineError, headline_cell

    a = silk_run(tmp_path)
    res = headline_cell(a, layer=29, fraction=0.85)
    kinds = [r["kind"] for r in res.rows]
    assert kinds[0] == "headline_cell" and kinds.count("headline_channel") == 5
    ch = {r["channel"]: r for r in res.rows if r["kind"] == "headline_channel"}
    assert ch["self_report"]["claims"] == {"YES": 3} and ch["self_report"]["matches_k"] == 0
    assert ch["identify"]["matches_n"] == 30 and ch["effect"]["influence"] == [4.0] * 4
    assert sum(r["kind"] == "headline_null" for r in res.rows) == 3 * (30 + 4)
    calls = next(r for r in res.rows if r["kind"] == "headline_judge_calls")
    assert calls["n"] == 30 + 3 + 4 and calls["ok"] == 37
    for r in res.rows:
        check_aggregate(r)
    blob = json.dumps(res.rows)
    assert MARK not in blob and NAME_MARK not in blob
    texts = {t["id"]: t["text"] for t in res.texts}
    assert all(MARK in texts[r["transcript_id"]] for r in res.rows if r["kind"] == "headline_trial")
    assert any(t["kind"] == "named_string" and t["text"] == NAME_MARK for t in res.texts)
    with pytest.raises(HeadlineError):
        headline_cell(a, layer=31, fraction=0.85)


# =====================================================================================
# The command line
# =====================================================================================

def _cli(archive, run_root, whats, cfg_overrides=None):
    from m5.analysis import cli

    cfg = config.load({"RUN_ROOT": str(run_root), "PRIOR_S_POINTS": 7, "PRIOR_FLOOR_POINTS": 3,
                       "PRIOR_LAPSE_POINTS": 3, "PRIOR_ICC": (0.0, 0.05), "POSTERIOR_DRAWS": 200,
                       "BOOTSTRAP_REPS": 20, **(cfg_overrides or {})})
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(SimpleNamespace(archive=archive, what=whats), cfg)
    return rc, out.getvalue(), err.getvalue()


def test_cli_refuses_an_archive_inside_the_repository_before_reading(tmp_path):
    root = tmp_path / "runs"
    rc, out, err = _cli(safety.REPO_ROOT / "docs", root, ["refit"])
    assert rc == 2 and "refused" in err and "inside the repository" in err
    assert not root.exists()


def test_cli_writes_aggregate_rows_and_transcripts_only(tmp_path):
    a = silk_run(tmp_path / "archive")
    root = tmp_path / "runs"
    rc, out, err = _cli(a.path, root, ["refit", "coupling", "near-misses", "headline", "truncation"])
    assert rc == 2 and "truncation skipped" in err and "no judge is available" in err
    runs = list(root.iterdir())
    assert len(runs) == 1
    run = runio.RunDir.open(runs[0])
    assert run.manifest["kind"] == "analysis"
    rows = run.rows("analysis")
    kinds = {r["kind"] for r in rows}
    assert {"archive_read", "refit_layer", "refit_icc", "coupling_summary", "nearmiss_summary",
            "headline_cell"} <= kinds
    for r in rows:
        check_aggregate({k: v for k, v in r.items() if k not in STAMP_KEYS})
    blob = (runs[0] / "analysis.jsonl").read_text()
    assert MARK not in blob and NAME_MARK not in blob
    assert MARK not in out and NAME_MARK not in out
    texts = run.rows("transcripts")
    assert texts and any(MARK in t["text"] for t in texts)
    events = run.rows("events")
    assert any(e["phase"] == "truncation" and e["status"] == "unavailable" for e in events)
    assert "R*(" in out and "headline L29@0.85" in out


def test_cli_replay_reports_a_missing_search_without_losing_the_rest(tmp_path, monkeypatch):
    import importlib

    from m5.analysis import replay

    real = importlib.import_module

    def fake(name, *a, **k):
        if name.startswith("m5.search"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(name, *a, **k)

    monkeypatch.setattr(replay.importlib, "import_module", fake)
    a = silk_run(tmp_path / "archive")
    rc, out, err = _cli(a.path, tmp_path / "runs", ["coupling", "replay"])
    assert rc == 2 and "replay skipped" in err and "m5.search" in err
    assert "coupling by fraction" in out


def test_sink_refuses_text_and_writes_transcripts_apart(tmp_path, cfg):
    run = runio.RunDir.create(tmp_path / "r", "x", config=cfg, kind="analysis")
    sink = AnalysisSink(run)
    sink.rows([{"kind": "x", "rate": 0.5}])
    with pytest.raises(safety.Refused):
        sink.rows([{"kind": "x", "rate": 0.5}, {"kind": "x", "response": "hello"}])
    assert len(run.rows("analysis")) == 1      # a refused batch writes nothing
    with pytest.raises(ValueError, match="stamps"):
        sink.rows([{"kind": "x", "config_hash": "abc"}])
    sink.texts([{"id": "t1", "text": "hello"}])
    assert run.rows("transcripts")[0]["text"] == "hello"
    assert name_id("lemon") == name_id("lemon") != name_id("lime")
