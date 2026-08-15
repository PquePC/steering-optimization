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
import sys
from collections import Counter
from pathlib import Path

import pytest

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from m3 import config, run as m3run                      # noqa: E402
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
    path = next(d.glob(f"garlic_*/{name}"))
    return [json.loads(l) for l in path.open(encoding="utf-8")]


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
        assert scoring.run("score", records, gold_dir=calibrate.LABELS_DIR,
                           concept="Garlic") == 0
        assert scoring.run_full(records, concept="Garlic",
                                out=run_dir / "judged_full.jsonl") == 0
    assert (run_dir / "judged_full.jsonl").exists()


def test_the_boundary_search_finds_a_known_per_layer_boundary(run_dir):
    """The first real run returned dose_max=0.4 for all 25 surviving layers -- one bit of
    information dressed as a per-layer measurement, because bisecting a bracket has a floor set
    by the probe count. The descending ladder must recover a boundary that VARIES with depth."""
    from m3.tests.fake_gpu import true_boundary
    config.apply_overrides(["LAYER_FRACTIONS=0.21,1.0", "LAYER_STRIDE=6",
                            "DOSE_FRACTIONS=0.5"], config.CONFIG)
    with fake_gpu():
        m3run.main(["--concept", "Garlic"])
    rows = _load(run_dir, "boundaries.jsonl")
    found = {r["layer"]: r["dose_max"] for r in rows if r["dose_max"] is not None}
    assert len(found) >= 5
    assert len(set(found.values())) > 1, "the search returned one value for every layer"
    for layer, dose in found.items():
        true = true_boundary(layer, 62)
        assert dose <= true * 1.01, f"L{layer}: reported {dose} above the true boundary {true}"
        assert dose >= true * 0.5, f"L{layer}: reported {dose}, far below the true {true}"


def test_an_unreachable_layer_is_not_reported_as_incoherent(run_dir):
    """24 layers came back `incoherent_at_lowest_probe` on the first real run and not one was a
    statement about the model -- the dose was simply forbidden by ALPHA_CEIL."""
    config.apply_overrides(["LAYER_FRACTIONS=0.21,1.0", "LAYER_STRIDE=6",
                            "BOUNDARY_BRACKET=1.5,2.5", "ALPHA_CEIL=1.0"], config.CONFIG)
    with fake_gpu() as calls:
        m3run.main(["--concept", "Garlic"])
    rows = _load(run_dir, "boundaries.jsonl")
    assert rows and all(r["outcome"] == "unreachable" for r in rows), \
        [r["outcome"] for r in rows]
    assert all(r["probes"] == [] for r in rows), "an unreachable layer must cost no generations"
