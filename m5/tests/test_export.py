"""The export gate fails closed: each rule has a test that trips it."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from m5 import export, runio, safety


@pytest.fixture
def run(cfg, run_root):
    r = runio.RunDir.create(run_root, "exp", config=cfg, kind="test")
    r.append("cells", {"code": "d8", "cell": "silk|single|L29|rho=0.2|add|from_marker", "k": 1, "n": 30})
    r.append("transcripts", {"id": "t1", "text": "Yes, I notice something unusual about silk."})
    r.append("judge_calls", {"id": "j1", "prompt": "...", "reply": "YES"})
    r.write_doc("report.json", {"ok": True})
    r.close()
    np.savez(r.path / "silk_vectors.npz", L29=np.zeros(4))
    (r.path / "sub").mkdir()
    (r.path / "sub" / "x.json").write_text("{}")
    return r


def names(plan, root):
    return sorted(p.relative_to(root).as_posix() for p in plan.include)


def test_default_bundle_has_aggregates_only(run):
    plan = export.plan_export(run, allow_transcripts=False, vector_classes=["concept"])
    assert names(plan, run.path) == ["cells.jsonl", "config.json", "manifest.json", "report.json"]
    reasons = {p.name: r for p, r in plan.withheld}
    assert "ALLOW_TRANSCRIPT_EXPORT" in reasons["transcripts.jsonl"]
    assert "internal" in reasons["judge_calls.jsonl"]
    assert "fail closed" in reasons["silk_vectors.npz"]
    assert "subdirectory" in reasons["x.json"]


def test_transcripts_need_the_flag_and_benign_classes(run):
    plan = export.plan_export(run, allow_transcripts=True, vector_classes=["concept", "control"])
    assert "transcripts.jsonl" in names(plan, run.path)
    plan = export.plan_export(run, allow_transcripts=True, vector_classes=["concept", "refusal"])
    assert "transcripts.jsonl" not in names(plan, run.path)
    plan = export.plan_export(run, allow_transcripts=True, vector_classes=[])
    assert "transcripts.jsonl" not in names(plan, run.path)
    # judge calls never leave, whatever the flags
    assert "judge_calls.jsonl" not in names(plan, run.path)


def test_an_aggregate_stream_carrying_text_is_withheld(run):
    run.append("cells", {"code": "e8", "note": "x" * 1000})
    run.close()
    plan = export.plan_export(run, allow_transcripts=False, vector_classes=["concept"])
    assert "cells.jsonl" not in names(plan, run.path)
    assert any("treated as text" in r for _, r in plan.withheld)


def test_bundle_carries_its_manifest_and_refuses_the_repo(run, tmp_path):
    out = tmp_path / "bundle.zip"
    export.write_export(run, out, allow_transcripts=False, vector_classes=["concept"])
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("EXPORT_MANIFEST.json"))
        assert "transcripts.jsonl" not in zf.namelist()
    assert {w["file"] for w in manifest["withheld"]} >= {"transcripts.jsonl", "judge_calls.jsonl"}
    with pytest.raises(safety.Refused):
        export.write_export(run, safety.REPO_ROOT / "bundle.zip", allow_transcripts=False,
                            vector_classes=["concept"])


def test_a_harmful_concept_withholds_every_transcript(run):
    plan = export.plan_export(run, allow_transcripts=True, vector_classes=["concept"],
                              concepts=["Weapons"])
    assert "transcripts.jsonl" not in names(plan, run.path)
    assert any("harmful-arm" in r for _, r in plan.withheld)


def test_an_empty_bundle_is_refused_and_no_part_file_remains(cfg, run_root, tmp_path):
    empty = runio.RunDir.create(run_root, "empty", config=cfg, kind="test")
    for f in empty.path.iterdir():
        f.unlink()
    with pytest.raises(safety.Refused, match="nothing"):
        export.write_export(empty, tmp_path / "b.zip", allow_transcripts=False, vector_classes=["concept"])
    assert not list(tmp_path.glob("*.part"))


def test_transcript_export_is_not_a_config_setting(cfg):
    """Old invariant 142: the permission is a call-site argument, so it cannot be inherited."""
    assert not any("EXPORT" in k for k in cfg)
