"""End-to-end execution of the M3 pipeline under the fake-GPU harness.

This file exists because ten defects were found in M3 and **none of them by a test**. The 250
unit tests pass throughout, because they exercise functions that get called; every defect lived
in a path nothing had executed. Four of the ten would have been caught here in under a second:
a judge id the transport rejects, a cache key of the wrong arity, a config key the bridge never
supplied, and a dict field written under one name and read under another.

These tests are slow by this suite's standards (a second or so) and worth every millisecond.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
import pathlib
from pathlib import Path

import pytest

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from m3 import config, judge, run as m3run              # noqa: E402
from m3.tests.fake_gpu import fake_gpu                   # noqa: E402


SMALL = ["LAYER_FRACTIONS=0.55,0.75", "LAYER_STRIDE=3",
         "DOSE_FRACTIONS=0.3,0.7", "BOUNDARY_PROBES=2"]


@pytest.fixture()
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("M3_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setenv("HF_TOKEN", "fake")
    saved = dict(config.CONFIG)
    config.apply_overrides(SMALL, config.CONFIG)
    yield tmp_path
    config.CONFIG.clear()
    config.CONFIG.update(saved)


def _load(d: Path, name: str) -> list[dict]:
    """Rows from the MOST RECENT run folder under `d`.

    Was `next(d.glob(...))`, which returns the first match in enumeration order — not creation
    order. A test that runs the pipeline twice with different settings creates two
    `garlic_<hash>` folders, and which one that read resolved to was decided by how the two
    hash strings happened to sort. It grades the right run today by accident, not by rule.
    """
    hits = sorted(d.glob(f"garlic_*/{name}"), key=lambda p: p.stat().st_mtime)
    assert hits, f"no run folder under {d} contains {name}"
    return [json.loads(l) for l in hits[-1].open(encoding="utf-8")]


def test_the_whole_pipeline_runs_end_to_end(run_dir):
    """`python -m m3.run` completes. This is the assertion the project did not have."""
    with fake_gpu() as calls:
        assert m3run.main(["--concept", "Garlic"]) == m3run.EXIT_OK
    assert calls["generations"] > 0 and calls["judge_calls"] > 0


def test_every_artefact_is_written_with_the_right_shape(run_dir):
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    d = next(run_dir.glob("garlic_*"))
    for name in ("cells.jsonl", "boundaries.jsonl", "responses_transcripts.jsonl",
                 "judge_calls.jsonl", "null_transcripts.jsonl", "norms.jsonl",
                 "summary.json", "provenance.jsonl"):
        assert (d / name).exists(), f"{name} was never written"

    cells = _load(run_dir, "cells.jsonl")
    battery = config.battery_size()
    resp = _load(run_dir, "responses_transcripts.jsonl")
    assert len(resp) == len(cells) * battery

    # channel split matches the configured battery exactly
    got = Counter(r["channel"] for r in resp)
    for channel, key in (("identify", "N_IDENTIFY"), ("self_report", "N_SELF_REPORT"),
                         ("effect", "N_EFFECT"), ("capability", "N_CAPABILITY")):
        assert got[channel] == len(cells) * config.CONFIG[key], channel


def test_no_measure_silently_produces_nothing(run_dir):
    """A `None` in a cell row is a judged measure whose every call failed. Under the harness
    every call succeeds, so a None here means the wiring dropped it."""
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    for cell in _load(run_dir, "cells.jsonl"):
        for field in ("identification", "effectiveness", "coherence", "capability"):
            assert cell[field] is not None, f"L{cell['layer']}@{cell['dose']} lost {field}"
        assert cell["judge_errors"] == 0


def test_every_paid_judge_reply_is_persisted(run_dir):
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    calls = _load(run_dir, "judge_calls.jsonl")
    assert calls and all(c.get("raw") for c in calls)
    assert {c["judge"] for c in calls} == {"identify", "self_report", "effect", "coherence"}


def test_norms_record_the_normalisation_every_dose_depends_on(run_dir):
    """Without these, `dose` is uninterpretable afterwards: nothing lets a reader check that a
    dose meant the same perturbation as on another run."""
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    norms = _load(run_dir, "norms.jsonl")
    assert norms
    for row in norms:
        assert row["vec_norm"] and row["resid_norm"]
        assert row["max_reachable_dose"] > 0


def test_a_second_run_resumes_instead_of_re_measuring(run_dir):
    """Row-level resume has never executed on a pod. It costs a full sweep if it is wrong."""
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    first = len(_load(run_dir, "cells.jsonl"))
    with fake_gpu() as calls:
        m3run.main(["--concept", "Garlic"])
    assert calls["judge_calls"] == 0, "a resumed run re-judged cells already on disk"
    assert len(_load(run_dir, "cells.jsonl")) == first, "resume duplicated rows"


def test_layers_without_a_boundary_are_named_not_dropped(run_dir):
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    summary = json.loads(next(run_dir.glob("garlic_*/summary.json")).read_text(encoding="utf-8"))
    measured = {c["layer"] for c in _load(run_dir, "cells.jsonl")}
    named = {s["layer"] for s in summary["skipped"]}
    assert set(summary["layers"]) == measured | named, "a layer vanished without being recorded"


def test_the_calibration_paths_run_end_to_end(run_dir):
    """`score` and `full` both spend money on a pod; both must execute offline first."""
    from m3 import calibrate, scoring
    probe = Path("Z:/Projects/TAIS Projects/temp/probe")
    if not probe.exists():
        pytest.skip("no probe bundle available locally")
    records = calibrate.load_probe(probe)
    with fake_gpu():
        # The exit code is the verdict, so this asserts the machinery rather than a constant:
        # the canned judge cannot reproduce a careful reader's labels, so `score` must come back
        # 1. A `run` that returned 0 here would be reporting a bar nothing was measured against.
        assert scoring.run("score", records, gold_dir=calibrate.LABELS_DIR,
                           concept="Garlic") == 1
        assert scoring.run_full(records, concept="Garlic",
                                out=run_dir / "judged_full.jsonl") == 0
    assert (run_dir / "judged_full.jsonl").exists()


def test_the_boundary_search_finds_a_known_per_layer_boundary(run_dir):
    """The first real run returned dose_max=0.4 for all 25 surviving layers -- one bit of
    information dressed as a per-layer measurement, because blindly bisecting a bracket has a
    floor set by the probe count. The descending ladder must recover a boundary that VARIES with
    depth -- and the in-bracket bisection must then land within BOUNDARY_BISECT_TOL of it, not a
    whole ladder step (43%) below it, or the top grid cell sits far under the regime the study
    is about."""
    from m3.tests.fake_gpu import true_boundary
    config.apply_overrides(["LAYER_FRACTIONS=0.21,1.0", "LAYER_STRIDE=6",
                            "DOSE_FRACTIONS=0.5"], config.CONFIG)
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    rows = _load(run_dir, "boundaries.jsonl")
    found = {r["layer"]: (r["dose_max"], r["max_reachable_dose"])
             for r in rows if r["dose_max"] is not None}
    assert len(found) >= 5
    assert len({d for d, _ in found.values()}) > 1, \
        "the search returned one value for every layer"
    for layer, (dose, reachable) in found.items():
        true = true_boundary(layer, 62)
        assert dose <= true * 1.01, f"L{layer}: reported {dose} above the true boundary {true}"
        # The boundary can never exceed what ALPHA_CEIL allows, so precision is judged against
        # whichever bound actually binds. 0.85 is the guarantee of the default tolerance (0.10)
        # with float margin, against the bare ladder's 0.70.
        bound = min(true, reachable)
        assert dose >= bound * 0.85, \
            f"L{layer}: reported {dose}, more than 15% below the reachable boundary {bound}"


def test_the_boundary_is_found_even_when_oversteering_stays_fluent(run_dir):
    """The failure mode the three-way criterion exists for, reproduced from the first real run:
    past the boundary the model writes well-formed, enthusiastic prose about garlic instead of
    the question (at L29 the real judge held coherence 8/10 while "what is a computer" was
    answered with garlic's botanical profile). The canned judge here is fooled the same way --
    it scores the fluent text coherent AND on-task -- so only the mechanical answer check can
    fail the probe. If the boundary still lands at the true value, it landed for the right
    reason. Under `oversteer="collapse"` every criterion sees the break, so that mode cannot
    tell whether `answered` is load-bearing; this one can."""
    from m3.tests.fake_gpu import true_boundary
    config.apply_overrides(["LAYER_FRACTIONS=0.21,1.0", "LAYER_STRIDE=6",
                            "DOSE_FRACTIONS=0.5"], config.CONFIG)
    with fake_gpu(oversteer="fluent"):
        m3run.main(["--concept", "Garlic"])
    rows = _load(run_dir, "boundaries.jsonl")
    found = {r["layer"]: (r["dose_max"], r["max_reachable_dose"])
             for r in rows if r["dose_max"] is not None}
    assert len(found) >= 5
    for layer, (dose, reachable) in found.items():
        true = true_boundary(layer, 62)
        assert dose <= true * 1.01, \
            f"L{layer}: {dose} above the true boundary {true} -- the fluent over-steer " \
            f"was scored sane, so the answer check is not load-bearing"
        assert dose >= min(true, reachable) * 0.85, \
            f"L{layer}: reported {dose}, far below {min(true, reachable)}"
    # The judge really was fooled: every recorded broken probe must have been failed by the
    # answer check, not by coherence. Otherwise this test is the collapse test wearing a wig.
    fooled = [p for r in rows for p in r["probes"] if not p["sane"]]
    assert fooled, "no probe ever landed past a boundary, so nothing was tested"
    assert all(p["answered"] < 0.75 for p in fooled)
    # Asserted PER RESPONSE, on the rows that have an answer to lose. The probe-level coherence
    # mean now also covers the open-ended row, which has no answer check and is not what this
    # test is about; averaging it in would let a real regression hide behind it.
    btx = _load(run_dir, "boundary_transcripts.jsonl")
    broken = {(r["layer"], p["dose"]) for r in rows for p in r["probes"] if not p["sane"]}
    checkable = [t for t in btx if (t["layer"], t["dose"]) in broken and t["answered"] is not None]
    assert checkable, "no boundary response with a checkable answer landed past a boundary"
    assert not any(t["answered"] for t in checkable),         "the answer check passed a response past the boundary"
    fooled_rows = [t for t in checkable
                   if t["coherence"] is not None and t["coherence"] >= 5.0 and t["on_task"]]
    assert len(fooled_rows) >= 0.75 * len(checkable), (
        f"only {len(fooled_rows)}/{len(checkable)} of the broken responses fooled the judge -- "
        "this is the collapse test wearing a wig, not the fluent one")


def test_an_unusable_alpha_ceiling_stops_the_run_before_it_spends(run_dir):
    """ALPHA_CEIL=16 was chosen against Gemma3-27B's norms and nothing makes it transfer.

    `dose = alpha * ||v|| / ||h||`, so a model whose residual stream is large relative to its
    concept vectors caps out below the bracket floor at every layer. Without this gate the run
    pays for the weights, the null arm and an hour of GPU, and returns a full grid of
    `unreachable` rows -- a result shaped exactly like "this concept has no effect".

    The assertion that matters is the SECOND one. Reporting the problem after the null battery
    would be a warning; reporting it before is the difference between losing two minutes and
    losing the judge spend.
    """
    from m3 import sweep

    config.apply_overrides(["ALPHA_CEIL=1.0"], config.CONFIG)
    with fake_gpu() as calls:
        # `main` catches and converts to an exit code, so the raise is observed as EXIT_FAILED.
        assert m3run.main(["--concept", "Garlic"]) == m3run.EXIT_FAILED
    assert sweep.Unreachable is not None
    assert calls["judge_calls"] == 0, "the gate fired only after the null arm had been paid for"
    assert calls["generations"] == 0, "the gate fired only after generation had been paid for"


def test_the_ceiling_the_gate_recommends_actually_works(run_dir, capfd):
    """A recommendation nobody has executed is a guess with a number in it.

    The message names the ALPHA_CEIL that would clear the floor everywhere. This runs with
    exactly that value and requires the sweep to complete -- so the arithmetic behind the
    advice is checked against the search that has to live with it, not just against itself.
    """
    from m3 import sweep

    config.apply_overrides(["ALPHA_CEIL=1.0"], config.CONFIG)
    with fake_gpu():
        assert m3run.main(["--concept", "Garlic"]) == m3run.EXIT_FAILED
    # The advice is read back out of the message the operator is actually shown, so this pins
    # the text they act on rather than a number recomputed alongside it.
    printed = capfd.readouterr()
    advised = float(re.search(r"--set ALPHA_CEIL=([0-9.]+)", printed.out + printed.err).group(1))

    config.apply_overrides([f"ALPHA_CEIL={advised}"], config.CONFIG)
    with fake_gpu():
        assert m3run.main(["--concept", "Garlic"]) == m3run.EXIT_OK
    bounds = _load(run_dir, "boundaries.jsonl")
    assert bounds and not [b for b in bounds if b["outcome"] == "unreachable"], (
        f"ALPHA_CEIL={advised} was advised as clearing the floor everywhere, and did not")


def test_a_workable_ceiling_is_reported_and_not_blocked(run_dir):
    """The gate must not refuse a run it should allow: a few unreachable shallow layers are
    ordinary, already recorded by name, and not a reason to stop."""
    from m3 import sweep

    with fake_gpu():
        assert m3run.main(["--concept", "Garlic"]) == m3run.EXIT_OK
        # Inside the context: `reachability` reads norms through run I/O, which needs the run
        # directory the sweep set up.
        got = sweep.reachability(config.layers_for_depth(62, config.CONFIG), config.CONFIG)
    assert got["n_layers"] > 0
    assert got["ceil_for_floor"] <= got["alpha_ceil"], (
        "the shipped ceiling does not clear the floor on the harness's own norms")


def test_damage_spread_across_prompts_is_not_cancelled_by_averaging_each_leg():
    """The real L29 probes from the 2026-08-19 pod run, replayed through the sanity rule.

    At dose 0.0823 each leg read exactly 0.75 -- coherence mean 7.0, on-task 3/4, answered 3/4 --
    and the dose passed. But each leg lost a DIFFERENT response: the Marie Curie answer was
    fluent and on-task with the answer gone, the Moon answer was fluent and answered but
    off-task, and the computer answer said a computer "is a member of the plant family" while
    still containing an accept term. Half the battery was corrupted and three independent
    averages could not see it.

    Conjoining the three per response, then counting, is the whole fix. Same responses, same
    judge calls, same floor -- only the order of the AND and the averaging changes.
    """
    coh_min, floor = 5.0, 0.75
    # (coherence, on_task, answered) per response, transcribed from boundary_transcripts.jsonl
    probes = {
        0.127153: [(2.0, False, False), (2.0, False, True), (5.0, True, False), (3.0, False, False)],
        0.089007: [(2.0, False, False), (8.0, True, True), (7.0, True, True), (8.0, True, False)],
        0.082331: [(8.0, True, True), (7.0, True, True), (8.0, True, False), (5.0, False, True)],
        0.075656: [(8.0, True, True), (8.0, True, True), (8.0, True, True), (8.0, True, True)],
        0.062305: [(8.0, True, True), (7.0, True, True), (8.0, True, True), (8.0, True, True)],
    }

    def per_leg(rs):                      # what the code used to do
        return (sum(c for c, _, _ in rs) / len(rs) >= coh_min
                and sum(1 for _, t, _ in rs if t) / len(rs) >= floor
                and sum(1 for _, _, a in rs if a) / len(rs) >= floor)

    def conjoined(rs):                    # what it does now
        return (sum(1 for c, t, a in rs if c >= coh_min and t and a) / len(rs)) >= floor

    assert per_leg(probes[0.082331]), "the regression this pins would not reproduce"
    assert not conjoined(probes[0.082331]), "the corrupted dose is still accepted"
    assert conjoined(probes[0.075656]) and conjoined(probes[0.062305]), \
        "a clean dose was rejected; the rule is now too strict to find any boundary"
    assert not any(conjoined(probes[d]) for d in (0.127153, 0.089007))

    highest_sane = max(d for d in probes if conjoined(probes[d]))
    assert highest_sane == 0.075656, f"L29 would take dose_max {highest_sane}, not 0.075656"


def test_batching_the_ladder_moves_no_boundary(tmp_path, monkeypatch):
    """BOUNDARY_RUNG_BATCH is scheduling. Every boundary must come out bit-identical.

    The batched ladder generates a window of rungs in one call instead of one rung per call.
    The rungs are the same geometric grid and are judged in the same descending order, so the
    first passing rung -- and the bracket handed to the bisection -- cannot change. If it ever
    does, the optimisation has become a change to the measurement, which is the one thing it is
    not allowed to be.

    Run in all three over-steer modes: `spread` in particular puts rungs that pass and rungs
    that fail inside the SAME window, which is the case a per-window verdict would get wrong.
    """
    import os
    from m3 import run as m3run

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setenv("HF_TOKEN", "fake")
    saved = dict(config.CONFIG)

    def boundaries(rung_batch, oversteer):
        os.environ["M3_RUNS_DIR"] = str(tmp_path / f"{oversteer}_{rung_batch}")
        cfg = config.CONFIG
        cfg.clear()
        cfg.update(config.SETTINGS)
        config.apply_overrides([f"BOUNDARY_RUNG_BATCH={rung_batch}", "LAYER_STRIDE=4",
                                "LAYER_FRACTIONS=0.21,1.0", "DOSE_FRACTIONS=0.5"], cfg)
        with fake_gpu(oversteer=oversteer) as calls:
            m3run.main(["--concept", "Garlic"])
        rd = next(pathlib.Path(os.environ["M3_RUNS_DIR"]).glob("garlic_*"))
        rows = [json.loads(x) for x in (rd / "boundaries.jsonl").read_text(encoding="utf-8")
                .splitlines() if x.strip()]
        return {r["layer"]: (r["dose_max"], r["lowest_failing_dose"], r["outcome"],
                             r["bisect_used"]) for r in rows}, dict(calls)

    # The shipped value, not a hardcoded one: the window must fit GEN_BATCH_MAX, and how many
    # rungs fit depends on the probe width. Pinning 6 here is what broke when the probe gained
    # its open-ended row.
    batch = int(config.SETTINGS["BOUNDARY_RUNG_BATCH"])
    for mode in ("fluent", "spread", "collapse"):
        one, c1 = boundaries(1, mode)
        many, c6 = boundaries(batch, mode)
        assert one, f"{mode}: no boundaries were produced, so this asserts nothing"
        assert one == many, (
            f"{mode}: batching the ladder changed the boundary at "
            f"{[k for k in one if one[k] != many.get(k)]}")
        # ...and it must actually have batched, or the equality above is trivially true.
        assert c6["batches"] < c1["batches"], f"{mode}: no fewer generation calls"
        # Speculative rungs are dropped rather than judged, so judge spend must not rise.
        assert c6["judge_calls"] <= c1["judge_calls"], f"{mode}: batching cost judge calls"

    config.CONFIG.clear()
    config.CONFIG.update(saved)


def test_the_boundary_rejects_damage_spread_across_prompts(run_dir):
    """The same defect, against the REAL search rather than a restatement of the rule.

    The test above reimplements both rules and compares them, so it documents the reasoning and
    pins nothing -- reverting `sweep.py` to per-leg averaging leaves it green. This one drives
    `find_boundary` through a model whose damage is spread across prompts, where every leg reads
    3/4 and half the battery is corrupt, and fails if the search accepts any dose above the
    model's true boundary.
    """
    from m3.tests.fake_gpu import true_boundary
    config.apply_overrides(["LAYER_FRACTIONS=0.21,1.0", "LAYER_STRIDE=6",
                            "DOSE_FRACTIONS=0.5"], config.CONFIG)
    with fake_gpu(oversteer="spread"):
        m3run.main(["--concept", "Garlic"])
    rows = [r for r in _load(run_dir, "boundaries.jsonl") if r["dose_max"] is not None]
    assert len(rows) >= 5, "no layer produced a boundary, so this asserts nothing"
    for r in rows:
        true = true_boundary(r["layer"], 62)
        assert r["dose_max"] <= true * 1.01, (
            f"L{r['layer']}: took dose_max {r['dose_max']} above the true boundary {true:.4f} -- "
            f"damage spread across prompts was averaged away leg by leg")

    # ...and the probe rows must show WHY: a rejected probe here is one where the individual
    # legs look survivable and the conjunction does not.
    spread = [p for r in rows for p in r["probes"]
              if not p["sane"] and p["on_task"] >= 0.75 and p["answered"] >= 0.75]
    assert spread, "the spread-damage shape never occurred, so the rule was not exercised"
    assert all(p["good"] < 0.75 for p in spread)


def test_every_boundary_generation_and_judge_reply_is_persisted(run_dir):
    """The 2026-08-15 run discarded 840 generations and 840 paid judge calls -- every one Phase 1
    made -- keeping only per-probe aggregates. `dose_max` is the number every cell's dose is a
    fraction of, and it was the one number in the run with no readable evidence behind it.

    Pinned as a COUNT against the probe rows, so dropping the write, or writing only the passing
    probes, or writing one row per probe instead of one per response, all fail here."""
    config.apply_overrides(["LAYER_FRACTIONS=0.21,1.0", "LAYER_STRIDE=6",
                            "DOSE_FRACTIONS=0.5"], config.CONFIG)
    with fake_gpu(oversteer="fluent"):
        m3run.main(["--concept", "Garlic"])
    bounds = _load(run_dir, "boundaries.jsonl")
    rows = _load(run_dir, "boundary_transcripts.jsonl")

    # One row per response, and a probe is BOUNDARY_N explain rows plus BOUNDARY_TASK_N
    # open-ended rows -- the open-ended ones are the only place an open-ended collapse is
    # visible to the boundary at all, so they must be persisted like the rest.
    width = int(config.CONFIG["BOUNDARY_N"]) + int(config.CONFIG["BOUNDARY_TASK_N"])
    expected = sum(len(b["probes"]) for b in bounds) * width
    assert expected > 0, "no probe ran, so this asserts nothing"
    assert len(rows) == expected, (
        f"{len(rows)} boundary responses on disk against {expected} probed "
        f"({sum(len(b['probes']) for b in bounds)} probes x {width} rows)")

    # Failing probes are the half worth reading: they are the doses the grid will never sample.
    assert any(not r["probe_sane"] for r in rows), "only passing probes were kept"
    for r in rows:
        assert r["response"], "a row carries no generation"
        assert r["judge_raw"], "a row carries no verbatim judge reply"
        # The three legs, separately. An aggregate cannot say which one rejected the dose.
        assert r["coherence"] is not None and r["on_task"] is not None
        # `answered` is a bool where the prompt HAS a checkable answer and None on the
        # open-ended row, which is held to coherence and on-task only. None is the honest
        # value: writing False there would record a failed answer check that never ran.
        assert isinstance(r["answered"], bool) or r["answered"] is None
        assert (r["answered"] is None) == (not r["accept"]) if "accept" in r else True
        assert r["stage"] in ("ladder", "bisect")
    assert any(r["answered"] is None for r in rows),         "no open-ended boundary row was kept, so the boundary still cannot see a story collapse"
    assert any(r["answered"] is not None for r in rows), "no checkable-answer row was kept"

    # The bundle is where the operator actually reads it from, and the export gate resolves by
    # NAME -- so a file the run writes is not necessarily a file the run ships.
    bundle = next(run_dir.glob("export_garlic_*.zip"))
    with zipfile.ZipFile(bundle) as z:
        names = [n.rsplit("/", 1)[-1] for n in z.namelist()]
    assert "boundary_transcripts.jsonl" in names, f"not in the export bundle: {names}"


def test_bisection_zero_reproduces_the_bare_ladder(run_dir):
    """BOUNDARY_BISECTIONS=0 must disable refinement entirely: no bisect-stage probes, and
    `dose_max` is exactly a ladder rung. The knob that turns a feature off is part of the
    feature."""
    config.apply_overrides(["LAYER_FRACTIONS=0.21,1.0", "LAYER_STRIDE=6",
                            "DOSE_FRACTIONS=0.5", "BOUNDARY_BISECTIONS=0"], config.CONFIG)
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    rows = _load(run_dir, "boundaries.jsonl")
    assert rows
    for r in rows:
        assert all(p["stage"] == "ladder" for p in r["probes"]), \
            f"L{r['layer']}: a bisect probe ran with BOUNDARY_BISECTIONS=0"
        assert r["bisect_used"] == 0


def test_an_unreachable_layer_is_not_reported_as_incoherent(run_dir):
    """24 layers came back `incoherent_at_lowest_probe` on the first real run and not one was a
    statement about the model -- the dose was simply forbidden by ALPHA_CEIL.

    ALPHA_CEIL=2.0 puts FOUR of these nine layers under the bracket floor, not all nine. The
    earlier version forbade every layer, which the Phase 0b reachability gate now refuses to
    start -- correctly, since a run whose entire grid is unreachable cannot answer anything. A
    minority still exercises the reporting this test is about, and additionally proves the gate
    lets a partially reachable run through.
    """
    config.apply_overrides(["LAYER_FRACTIONS=0.21,1.0", "LAYER_STRIDE=6",
                            "ALPHA_CEIL=2.0"], config.CONFIG)
    with fake_gpu():
        assert m3run.main(["--concept", "Garlic"]) == m3run.EXIT_OK
    rows = _load(run_dir, "boundaries.jsonl")
    blocked = [r for r in rows if r["outcome"] == "unreachable"]
    assert blocked, "no layer was unreachable, so the reporting path never ran"
    assert len(blocked) < len(rows), "every layer was unreachable; the gate should have stopped it"
    assert all(r["probes"] == [] for r in blocked), "an unreachable layer must cost no generations"
    assert not any(r["outcome"] == "incoherent_at_lowest_probe" for r in rows), \
        "a dose forbidden by ALPHA_CEIL was reported as a statement about the model"


def test_the_read_this_bundle_surfaces_judge_versus_mechanical_disagreement(run_dir):
    """Every deep defect this project found was found by reading raw generations, and none by a
    gate or a rate. The bundle is the required output that makes that possible, and it selects
    by disagreement because random sampling would have found none of them.

    The `null arm` half of this assertion used to live here too, and it was the weakest test in
    the suite: it passed only because this fixture's grid produces fewer findings than the cap,
    so the cap never bound. Reintroducing the truncation bug left it green. It now lives in
    `test_the_null_arm_survives_a_bundle_full_of_disagreements`, on a grid where the cap does
    bind — where it goes red. What is left here is the part this grid can actually test.
    """
    with fake_gpu(judge_disagree_every=5):
        m3run.main(["--concept", "Garlic"])
    text = next(run_dir.glob("garlic_*/read_this.md")).read_text(encoding="utf-8")
    reasons = {l for l in text.splitlines() if l.startswith("### ")}
    assert any("degenerate" in r or "incoherent" in r for r in reasons), \
        f"no judge/detector disagreement surfaced; got {reasons}"
    assert "```" in text, "the bundle must contain the actual response text"


def test_a_presentation_only_setting_does_not_split_the_run_folder(run_dir):
    """`--set READ_BUNDLE_N=60` changes how many transcripts get printed. If it also changed the
    config hash it would land in a new folder and re-measure the whole surface to do it."""
    base = config.config_hash(dict(config.SETTINGS))
    assert config.config_hash(dict(config.SETTINGS, READ_BUNDLE_N=60)) == base
    assert config.config_hash(dict(config.SETTINGS, N_IDENTIFY=99)) != base


# =====================================================================================
# Regressions from the part-3 read-through
# =====================================================================================

def test_the_null_arm_survives_a_bundle_full_of_disagreements(run_dir):
    """The null arm was inside the cap, appended last, and therefore first to be truncated.

    A steered rate is unreadable without what the framing produces on its own -- at alpha=0 this
    model names a concept every single time it is asked -- so it is the one section that cannot
    be reconstructed from the rest of the bundle. Any real run produces more than
    READ_BUNDLE_N disagreements, so on a real run it was always going to be dropped; the small
    grid these tests use is the only reason the existing assertion passed.
    """
    # The full battery and a wider grid: the shrunken one these tests share produces fewer
    # disagreements than the cap, so the cap would never bind and the test would assert nothing.
    config.CONFIG.clear()
    config.CONFIG.update(config.SETTINGS)
    config.apply_overrides(["READ_BUNDLE_N=1", "LAYER_FRACTIONS=0.55,1.00",
                            "LAYER_STRIDE=2", "DOSE_FRACTIONS=0.4,0.9"], config.CONFIG)
    # Not every 2nd call: the boundary phase is judged too, and a judge that lies half the time
    # now correctly yields no boundary at any layer, which starves the grid this test needs.
    # That is the product behaving right and the fixture being too adversarial to measure it.
    with fake_gpu(judge_disagree_every=4):
        m3run.main(["--concept", "Garlic"])
    text = next(run_dir.glob("garlic_*/read_this.md")).read_text(encoding="utf-8")
    nulls = [l for l in text.splitlines() if l.startswith("### ") and "null arm" in l]
    expected = config.battery_size() * config.CONFIG["NULL_REPEATS"]
    assert len(nulls) == expected, \
        "the null arm was truncated by READ_BUNDLE_N; it must sit outside the cap"
    assert "not shown here" in text, "a cap that drops findings must say how many"


def test_a_search_that_stops_above_the_floor_does_not_claim_to_have_reached_it(run_dir):
    """`incoherent_at_floor` and `probes_exhausted` are different facts.

    With the settings this shipped with, descending from 2.50 by 0.70 reaches 0.60 in five
    probes against a floor of 0.05 -- and reported `incoherent_at_floor`. That is the same
    mislabel as the `incoherent_at_lowest_probe` defect the search was rebuilt to remove: a
    label asserting something the search never established, at the other end of the ladder.
    """
    common = ["LAYER_FRACTIONS=1.0,1.0", "DOSE_FRACTIONS=0.5",
              "BOUNDARY_COHERENCE_MIN=9.9", "BOUNDARY_BRACKET=0.05,2.5"]
    config.apply_overrides(common + ["BOUNDARY_PROBES=3"], config.CONFIG)
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    short = _load(run_dir, "boundaries.jsonl")[-1]
    assert short["outcome"] == "probes_exhausted", short["outcome"]
    assert short["lowest_probe"] > short["bracket"][0], "it did reach the floor after all"
    assert short["probes_to_reach_floor"] > 3, "the row must say what would have sufficed"

    config.apply_overrides(common + ["BOUNDARY_PROBES=12"], config.CONFIG)
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    deep = _load(run_dir, "boundaries.jsonl")[-1]
    assert deep["outcome"] == "incoherent_at_floor", deep["outcome"]
    assert deep["lowest_probe"] < deep["bracket"][0] / deep["step_ratio"]


def test_the_shipped_settings_can_descend_the_whole_bracket(run_dir):
    """The defaults must satisfy their own arithmetic. They did not: five probes against a
    bracket needing twelve, so every layer that survived to the bottom was mislabelled."""
    est = m3run.estimate(62, dict(config.SETTINGS))
    assert est["ladder_reaches_floor"], (
        f"BOUNDARY_PROBES={config.SETTINGS['BOUNDARY_PROBES']} only reaches "
        f"{est['lowest_probe']:.3f}, not the floor "
        f"{config.SETTINGS['BOUNDARY_BRACKET'][0]}; {est['probes_needed']} are needed")


def test_no_layer_is_refused_the_dose_the_search_computed_for_it(run_dir):
    """`max_reachable = CEIL*v/h` is fed straight back to `alpha_for`, which recomputes
    `alpha = dose*h/v`. That round trip overshoots ALPHA_CEIL by an ulp on about one layer in
    sixteen; `alpha_for` refuses -- rightly, a clamped alpha is a cell measured at a dose other
    than the recorded one -- and Phase 1 does not catch it, so one layer ends the whole sweep.

    The bisection this replaced never probed at `max_reachable`, so the defect arrived with the
    rebuild and has never run on a pod.
    """
    from m2 import config as m2config
    from m3 import sweep

    config.apply_overrides(["LAYER_FRACTIONS=0.21,1.0", "LAYER_STRIDE=1",
                            "DOSE_FRACTIONS=0.5", "BOUNDARY_PROBES=1"], config.CONFIG)
    with fake_gpu():
        assert m3run.main(["--concept", "Garlic"]) == m3run.EXIT_OK
        for row in _load(run_dir, "boundaries.jsonl"):
            if row.get("max_reachable_dose") is None:
                continue
            # The invariant, stated directly: the highest dose the search will start at must be
            # one `alpha_for` accepts.
            m2config.alpha_for(row["layer"], row["max_reachable_dose"])
    assert len(_load(run_dir, "boundaries.jsonl")) == 49


def test_a_blindness_check_survives_a_clipped_response(run_dir):
    """`render` clips model text into the payload; the caller hands the check the unclipped
    span. When they differ the subtraction matches nothing, the model's own "garlic" stays in
    the haystack, and the check kills the cell claiming the payload leaks the concept.

    Latent at the shipped numbers (100 tokens is ~500 characters against a 1200-character cap)
    and armed by any change to either -- a tripwire that fires on a pod, not here.
    """
    long_response = ("A story about garlic. " * 200)
    payload = judge.render("coherence", text_chars=200,
                           prompt="Tell me a short story.", response=long_response)
    assert "truncated" in payload, "this test needs the clip to actually fire"
    judge.build_item("coherence", payload=payload, cache_key=judge.cache_key(
        "T", "coherence", layer=1, dose=0.1, unit="task_story"),
        concept="Garlic", model_text=(long_response,), text_chars=200)

    # And the check must still catch a genuine leak: the concept in the INSTRUCTIONS.
    leaky = payload.replace("Tell me a short story.", "Tell me a short story about garlic.")
    with pytest.raises(AssertionError, match="outside the model's own"):
        judge.build_item("coherence", payload=leaky, cache_key=judge.cache_key(
            "T", "coherence", layer=1, dose=0.1, unit="task_story"),
            concept="Garlic", model_text=(long_response,), text_chars=200)


def test_resuming_does_not_append_a_second_copy_of_phase_zero(run_dir):
    """`calibrate` re-runs on every resume -- it has to, the norms and baselines live in memory
    -- but it APPENDED. Two resumes meant three copies of the null arm, so anyone counting it,
    or reading it in the bundle, saw a multiple of the truth."""
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    first = (len(_load(run_dir, "norms.jsonl")), len(_load(run_dir, "null_transcripts.jsonl")))
    for _ in range(2):
        with fake_gpu():
            m3run.main(["--concept", "Garlic"])
    assert (len(_load(run_dir, "norms.jsonl")),
            len(_load(run_dir, "null_transcripts.jsonl"))) == first
    assert first[1] == config.battery_size() * config.CONFIG["NULL_REPEATS"]


def test_the_probe_loader_reads_the_field_name_the_archive_actually_carries(tmp_path):
    """M2 writes `degeneracy_reason`; M3's own detector calls the same thing
    `degeneration_reason`. The loader read the M3 name off an M2 file, with `.get`, so the
    collapse reason was `None` for every response in every archive — silently, with nothing
    raising and no output visibly wrong.

    This is the third time the two vocabularies have crossed at this boundary
    (`concept_hits`/`concept_mentions` was the first, and cost 1,720 paid judge calls). The
    existing regression test passes on this defect: it asserts the key exists on the loaded
    record, which it does — populated from a key that exists in no file.

    So this test asserts the VALUE arrives, and that a row carrying neither name raises rather
    than defaulting. A defaulted read of a misspelled key is invisible at every level.
    """
    from m3 import calibrate

    # Exactly the field names `m2.unjudged._per_response_rows` writes, verified against a real
    # 2026-08-14 bundle.
    raw = dict(channel="task", layer=40, r=0.30, prompt_id="task_story", trial=None,
               response="garlic garlic garlic", words=3, concept_hits=3, degenerate=True,
               degeneracy_reason="ngram_repeat: 5-gram repeated 4 times", steered=True,
               alpha=1.0, concept="Garlic", config_hash="deadbeef", ts="2026-08-14T00:00:00Z")
    for name in ("probe_forced_transcripts.jsonl", "probe_detect_transcripts.jsonl",
                 "probe_task_transcripts.jsonl", "probe_null_transcripts.jsonl"):
        (tmp_path / name).write_text(json.dumps(raw) + "\n", encoding="utf-8")

    (tmp_path / "probe_forced_transcripts.jsonl").write_text(
        json.dumps(dict(raw, channel="forced", trial=1, prompt_id=None)) + "\n",
        encoding="utf-8")
    (tmp_path / "probe_detect_transcripts.jsonl").write_text(
        json.dumps(dict(raw, channel="detect", trial=1, prompt_id=None)) + "\n",
        encoding="utf-8")
    (tmp_path / "probe_null_transcripts.jsonl").write_text(
        json.dumps(dict(raw, channel="task", layer=None, r=None, steered=False)) + "\n",
        encoding="utf-8")

    records = calibrate.load_probe(tmp_path)
    assert records, "the loader returned nothing"
    assert all(r["degeneration_reason"] == raw["degeneracy_reason"] for r in records), \
        "the collapse reason was dropped crossing the m2/m3 vocabulary boundary"

    # And a row with neither spelling must raise, not default.
    stripped = {k: v for k, v in raw.items() if k != "degeneracy_reason"}
    (tmp_path / "probe_task_transcripts.jsonl").write_text(
        json.dumps(stripped) + "\n", encoding="utf-8")
    with pytest.raises(KeyError, match="collapse-reason"):
        calibrate.load_probe(tmp_path)


def test_generations_are_kept_when_judging_raises(run_dir, monkeypatch):
    """The generations are paid for before the judging runs, and the judging can still raise --
    it did, on the first cell of a pod run, and the whole battery was lost. Nothing already
    bought is discarded because a later step failed."""
    from m3 import sweep

    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
        cfg = config.CONFIG
        layer = config.layers_for_depth(62, cfg)[0]
        cal = sweep.calibrate("Garlic", [layer], cfg)

        def boom(*a, **k):
            raise RuntimeError("the transport rejected this payload")

        monkeypatch.setattr(sweep.judge, "run_judges", boom)
        with pytest.raises(RuntimeError, match="rejected"):
            sweep.measure_cell(layer, 0.10, concept="Garlic",
                               baselines=cal["baselines"], cfg=cfg)

    kept = _load(run_dir, sweep.UNJUDGED_FILE)
    assert len(kept) == config.battery_size(), "the paid generations were discarded"
    assert all(r["response"] for r in kept)


def test_freerun_executes_end_to_end(run_dir, capsys):
    """`m3.freerun` loads a model, converts a dose to an alpha, runs the battery and generates
    both arms. It is the shortest path from a coordinate in a results table to the text the model
    actually produced, so it must not be the one path nobody executed before a pod."""
    from m3 import freerun

    with fake_gpu():
        assert freerun.main(["--concept", "Garlic", "--layer", "29", "--dose", "0.11",
                             "--n", "2", "--prompt", "Tell me a random word"]) == freerun.EXIT_OK
    out = capsys.readouterr().out
    assert "CELL STATISTICS" in out
    assert "identification (forced)" in out and "claims detection" in out
    assert "--- UNSTEERED ---" in out and "--- STEERED ---" in out
    assert "STATISTICS FOR THIS QUESTION" in out


def test_freerun_refuses_an_unreachable_dose_rather_than_clamping(run_dir, capsys):
    """A clamped alpha is a cell measured at a dose other than the one asked for. The sweep
    refuses; so must the tool people will use to spot-check the sweep."""
    from m3 import freerun

    with fake_gpu():
        rc = freerun.main(["--concept", "Garlic", "--layer", "29", "--dose", "99.0",
                           "--no-battery", "--prompt", "hi"])
    assert rc == freerun.EXIT_CONFIG
    assert "not reachable" in capsys.readouterr().out


def test_freerun_will_not_print_generations_for_the_harmful_arm(run_dir, capsys):
    from m3 import freerun

    with fake_gpu():
        rc = freerun.main(["--concept", "weapon", "--layer", "29", "--dose", "0.1",
                           "--prompt", "hi"])
    assert rc == freerun.EXIT_CONFIG
    assert "HARMFUL" in capsys.readouterr().out
