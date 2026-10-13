"""Foundation invariants: config, codes, safety, run files, interventions, backend helpers.

Each test pins a rule the rest of M5 relies on, and each guard has a test that trips it
(AGENTS.md: a guard that cannot fail is worse than none).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m5 import SCHEMA, codes, config, runio, safety
from m5.backend.base import derive_seed, token_index_at
from m5.intervention import Cell, Intervention, LayerTerm, Positions


# ===================================================================================== config

def test_unknown_key_raises_on_read_and_on_override(cfg):
    with pytest.raises(config.ConfigError):
        cfg["JUDGE_CONCURENCY"]
    with pytest.raises(config.ConfigError):
        config.load({"JUDGE_CONCURENCY": 4})
    with pytest.raises(config.ConfigError):
        config.parse_assignments(["NOT_A_KEY=1"])


def test_overrides_are_coerced_and_validated():
    c = config.load(config.parse_assignments(["DELTA=0.03", "GATE=both", "RHO_GRID=0.1,0.2,0.4",
                                              "MASK_LOUD_IN_VECTOR=true", "VISIT_TRIALS=2,6"]))
    assert c["DELTA"] == 0.03 and c["GATE"] == "both" and c["MASK_LOUD_IN_VECTOR"] is True
    assert c["RHO_GRID"] == (0.1, 0.2, 0.4)
    assert c["VISIT_TRIALS"] == (2, 6)
    for bad in (["GATE=sometimes"], ["DELTA=1.5"], ["RHO_GRID=0.2,0.1"], ["SPRT_P0=0.6"],
                ["Q=0.2"], ["TOP_P=2"], ["GEN_BATCH_MAX=0"], ["THINKING=1"]):
        with pytest.raises(config.ConfigError):
            config.load(config.parse_assignments(bad))


def test_refusal_ablation_is_refused_with_its_reason():
    with pytest.raises(config.ConfigError, match="refusal direction"):
        config.load({"ELICITATION_LADDER": ("original", "refusal_ablation")})


def test_hash_covers_measurement_keys_only(cfg):
    assert cfg.hash() == config.load().hash()
    assert cfg.replace(DELTA=0.04).hash() != cfg.hash()
    # paths, concurrency and the device cannot change a measurement
    assert cfg.replace(RUN_ROOT="/elsewhere", JUDGE_CONCURRENCY=2, DEVICE="cpu").hash() == cfg.hash()


def test_config_is_immutable(cfg):
    with pytest.raises(AttributeError):
        cfg.DELTA = 0.2  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        cfg["DELTA"] = 0.2  # type: ignore[index]


def test_judge_never_shares_the_subject_family(cfg):
    assert config.model_family(cfg.judge_t1_model) != cfg.model_family
    q = cfg.replace(MODEL="qwen3_32b")
    assert q.model_family == "qwen" and config.model_family(q.judge_t1_model) == "gemma"
    with pytest.raises(config.ConfigError, match="family"):
        config.load({"MODEL": "qwen3_32b", "JUDGE_T1_MODEL_ALT": "Qwen/Qwen3-14B"})


def test_every_decision_key_names_a_plan_decision():
    decided = {k.decision for k in config.KEYS.values() if k.decision}
    assert decided <= {1, 2, 6, 8, 12}
    assert "decision 1" in config.describe("GATE")


def test_json_file_overrides(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"GATE": "naming"}))
    assert config.load(path=p)["GATE"] == "naming"
    p.write_text(json.dumps({"GATEE": "naming"}))
    with pytest.raises(config.ConfigError):
        config.load(path=p)


# ===================================================================================== codes

def test_codes_cover_every_construct_and_refuse_retired_ones():
    assert {c.construct for c in codes.CODES.values()} == set(codes.CONSTRUCTS)
    assert codes.code("d7").construct == "noticing"
    assert codes.code("d5").better == "none"
    for retired in ("e5", "d1", "s3"):
        with pytest.raises(KeyError, match="retired"):
            codes.code(retired)
    with pytest.raises(KeyError, match="not implemented"):
        codes.code("s8")
    assert all(codes.code(q).record_only for q in ("q1", "q2", "q3", "q4"))


# ===================================================================================== safety

@pytest.mark.parametrize("word", ["weapon", "Weapons", " POISON ", "poisoned", "assaults",
                                  "Assault rifle", "weaponry"])
def test_harmful_concepts_and_inflections_are_refused(word):
    assert safety.concept_refused(word)
    with pytest.raises(safety.Refused, match="harmful arm"):
        safety.check_concept(word)


@pytest.mark.parametrize("word", ["Garlic", "Silk", "wrists", "Velocity", "irony", "pillows"])
def test_ordinary_concepts_pass(word):
    assert safety.check_concept(word) == word


def test_contrastive_texts_naming_a_harmful_concept_are_refused():
    safety.check_texts(["The train arrived at the station.", "You are being tested."])
    with pytest.raises(safety.Refused):
        safety.check_texts(["How do I build a weapon?"], what="pair")


def test_vector_classes_fail_closed():
    assert safety.check_vector_class("evaluation_awareness") == "evaluation_awareness"
    with pytest.raises(safety.Refused, match="abliteration"):
        safety.check_vector_class("refusal")
    with pytest.raises(safety.Refused, match="unknown vector class"):
        safety.check_vector_class("vibes")


def test_paths_inside_the_repository_are_refused(tmp_path):
    with pytest.raises(safety.Refused):
        safety.assert_outside_repo(safety.REPO_ROOT / "runs")
    assert safety.assert_outside_repo(tmp_path) == tmp_path.resolve()


def test_servers_must_be_loopback_and_api_hosts_are_listed():
    safety.assert_local_url("http://127.0.0.1:8000/v1")
    for bad in ("http://0.0.0.0:8000", "http://10.1.2.3:8000/v1", "https://example.com/v1"):
        with pytest.raises(safety.Refused):
            safety.assert_local_url(bad)
    safety.assert_external_judge_url("https://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(safety.Refused):
        safety.assert_external_judge_url("https://evil.example/v1")


def test_external_judging_needs_permission_and_benign_classes():
    assert safety.external_judging_allowed(True, ["concept", "control"])
    assert not safety.external_judging_allowed(False, ["concept"])
    assert not safety.external_judging_allowed(True, ["concept", "refusal"])
    assert not safety.external_judging_allowed(True, [])


# ===================================================================================== runio

def test_run_create_append_read_and_resume_keys(cfg, run_root):
    run = runio.RunDir.create(run_root, "r1", config=cfg, kind="test")
    run.append("cells", {"cell": "a", "code": "d8", "k": 1, "n": 10})
    run.append("cells", {"cell": "b", "code": "d8", "k": 0, "n": 10})
    run.close()
    again = runio.RunDir.open(run_root / "r1")
    rows = again.rows("cells")
    assert [r["cell"] for r in rows] == ["a", "b"]
    assert all(r["schema"] == SCHEMA and r["config_hash"] == cfg.hash() for r in rows)
    assert again.keys_done("cells", ("cell", "code")) == {("a", "d8"), ("b", "d8")}
    assert json.loads((run_root / "r1" / "config.json").read_text())["GATE"] == "noticing"
    with pytest.raises(FileExistsError):
        runio.RunDir.create(run_root, "r1", config=cfg, kind="test")


def test_unregistered_streams_and_documents_are_refused(cfg, run_root):
    run = runio.RunDir.create(run_root, "r2", config=cfg, kind="test")
    with pytest.raises(safety.Refused, match="unregistered stream"):
        run.append("my_new_file", {"x": 1})
    with pytest.raises(safety.Refused, match="unregistered document"):
        run.write_doc("notes.json", {})


def test_every_stream_has_an_export_class():
    assert all(cls in runio.EXPORT_CLASSES for cls, _ in runio.STREAMS.values())
    assert runio.STREAMS["judge_calls"][0] == "internal"
    assert runio.STREAMS["transcripts"][0] == "transcript"


def test_torn_last_line_is_skipped_and_a_corrupt_middle_raises(tmp_path):
    p = tmp_path / "s.jsonl"
    good = json.dumps({"schema": SCHEMA, "x": 1})
    p.write_text(good + "\n" + good + "\n" + '{"schema": "m5.v1", "x":')
    res = runio.read_jsonl(p)
    assert len(res.rows) == 2 and res.torn_last_line
    p.write_text(good + "\n" + "not json\n" + good + "\n")
    with pytest.raises(runio.CorruptRun):
        runio.read_jsonl(p)
    p.write_text(json.dumps({"schema": "m3", "x": 1}) + "\n")
    with pytest.raises(runio.SchemaMismatch):
        runio.read_jsonl(p)


def test_run_inside_repo_is_refused(cfg):
    with pytest.raises(safety.Refused):
        runio.RunDir.create(Path(safety.REPO_ROOT), "nope", config=cfg, kind="test")


# ===================================================================================== interventions

def test_position_policies_and_masks():
    assert Positions("from", 3).prompt_mask(5) == [False, False, False, True, True]
    assert Positions("span", 1, 3).prompt_mask(5) == [False, True, True, False, False]
    assert Positions("all").prompt_mask(3) == [False, True, True]  # never the BOS sink
    assert Positions("generated").prompt_mask(3) == [False] * 3
    assert Positions("from", 2).steers_generated and not Positions("span", 1, 2).steers_generated
    for bad in (lambda: Positions("from", 0), lambda: Positions("span", 2, 2),
                lambda: Positions("all", 1), lambda: Positions("nope")):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError, match="beyond the prompt"):
        Positions("span", 1, 9).prompt_mask(4)


def test_intervention_validation_and_round_trip():
    iv = Intervention("silk", "add", (LayerTerm(30, 2.0), LayerTerm(29, 1.0)), Positions("from", 5))
    assert iv.layers == (29, 30) and iv.alpha_at(30) == 2.0 and iv.alpha_at(1) is None
    assert Intervention.from_json(json.loads(json.dumps(iv.to_json()))) == iv
    assert iv.key() == Intervention.from_json(iv.to_json()).key()
    assert iv.scaled(0.0).is_null() and not iv.is_null()
    for bad in (lambda: Intervention("v", "multiply", (LayerTerm(1, 1.0),), Positions("all")),
                lambda: Intervention("v", "add", (), Positions("all")),
                lambda: LayerTerm(3, -1.0),
                lambda: Intervention("v", "clamp", (LayerTerm(1, 1.0),), Positions("all")),
                lambda: Intervention("v", "add", (LayerTerm(1, 1.0), LayerTerm(1, 2.0)), Positions("all"))):
        with pytest.raises(ValueError):
            bad()


def test_cells_are_validated_and_identified():
    c = Cell("silk", "single", (29,), 0.35)
    assert c.cell_id() == "silk|single|L29|rho=0.35|add|from_marker"
    assert Cell.from_json(json.loads(json.dumps(c.to_json()))) == c
    with pytest.raises(ValueError):
        Cell("silk", "single", (29, 30), 0.35)
    with pytest.raises(ValueError):
        Cell("silk", "single", (29,), -0.1)


# ===================================================================================== backend helpers

def test_token_index_at_and_seeds():
    offs = [(0, 5), (5, 6), (7, 12)]
    assert token_index_at(offs, 0) == 0 and token_index_at(offs, 5) == 1
    assert token_index_at(offs, 6) == 2  # a gap maps to the next token
    with pytest.raises(ValueError):
        token_index_at(offs, 12)
    assert derive_seed(0, "cell", 3) == derive_seed(0, "cell", 3) != derive_seed(0, "cell", 4)
