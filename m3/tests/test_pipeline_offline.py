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


def test_the_read_this_bundle_surfaces_judge_versus_mechanical_disagreement(run_dir):
    """Every deep defect this project found was found by reading raw generations, and none by a
    gate or a rate. The bundle is the required output that makes that possible, and it selects
    by disagreement because random sampling would have found none of them."""
    with fake_gpu(judge_disagree_every=5):
        m3run.main(["--concept", "Garlic"])
    text = next(run_dir.glob("garlic_*/read_this.md")).read_text(encoding="utf-8")
    reasons = {l for l in text.splitlines() if l.startswith("### ")}
    assert any("degenerate" in r or "incoherent" in r for r in reasons), \
        f"no judge/detector disagreement surfaced; got {reasons}"
    assert any("null arm" in r for r in reasons), "the null arm must always be included"
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
    with fake_gpu(judge_disagree_every=2):
        m3run.main(["--concept", "Garlic"])
    text = next(run_dir.glob("garlic_*/read_this.md")).read_text(encoding="utf-8")
    nulls = [l for l in text.splitlines() if l.startswith("### ") and "null arm" in l]
    assert len(nulls) == config.battery_size(), \
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
    assert first[1] == config.battery_size()


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
