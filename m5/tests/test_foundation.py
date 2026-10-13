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
                                  "Assault rifle", "weaponry", "wea\u200bpon", "w e a p o n",
                                  "poi-son", "\uff57\uff45\uff41\uff50\uff4f\uff4e"])
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


# ===================================================================================== static guards
# These scan the whole m5 package, so they also hold every module other agents add.

import ast  # noqa: E402

_PKG = Path(__file__).resolve().parents[1]


def _package_modules():
    for path in sorted(_PKG.rglob("*.py")):
        if "tests" in path.relative_to(_PKG).parts:
            continue
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_no_config_value_is_read_with_a_default():
    """Old invariant 87: `cfg.get(KEY, default)` is a different measurement under the same name."""
    offenders = []
    for path, tree in _package_modules():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"):
                target = node.func.value
                name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
                if name in ("cfg", "config", "CONFIG"):
                    offenders.append(f"{path.relative_to(_PKG)}:{node.lineno}")
    assert not offenders, f"config read with .get(): {offenders}"


def test_the_package_serves_nothing_and_uploads_nothing():
    """Hard rules 1-2: no listening sockets, servers, tunnels or upload clients in m5."""
    banned = {"socket", "socketserver", "http.server", "flask", "fastapi", "uvicorn", "gradio",
              "streamlit", "ngrok", "pyngrok", "huggingface_hub.upload", "boto3", "paramiko"}
    found = []
    for path, tree in _package_modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            found += [f"{path.relative_to(_PKG)}:{node.lineno} {n}" for n in names
                      if n in banned or n.split(".")[0] in {"flask", "fastapi", "uvicorn", "gradio"}]
    assert not found, f"server or upload imports in m5: {found}"


def test_gitignore_keeps_its_load_bearing_patterns():
    """Hard rule 3: the ignore rules keep vectors, activations and run data out of a public repo."""
    lines = {l.strip() for l in (safety.REPO_ROOT / ".gitignore").read_text().splitlines()}
    for pattern in ("vectors/", "activations/", "results/", "outputs/", "*.pt", "*.safetensors",
                    "*.npy", "*.npz", "private/", "m2_runs/"):
        assert pattern in lines, f".gitignore lost {pattern}"


def test_harmful_list_matches_the_old_pipelines_until_they_retire():
    """Old invariant 139: one list of harmful concepts. m5 keeps its own copy (it imports nothing
    from m2); this test reads m2's file, not its module, and fails if the two ever diverge."""
    src = (safety.REPO_ROOT / "m2" / "config.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        targets = node.targets if isinstance(node, ast.Assign) else \
            [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(getattr(t, "id", "") == "HARMFUL_CONCEPTS" for t in targets):
            old = tuple(ast.literal_eval(node.value))
            assert {c.casefold() for c in old} == set(safety.HARMFUL_CONCEPTS)
            return
    pytest.skip("m2/config.py no longer defines HARMFUL_CONCEPTS")


# ===================================================================================== run files, more

def test_records_split_on_newline_only(cfg, run_root):
    """Old invariant 102: U+2028/U+2029/U+0085 inside a text must not split a record."""
    run = runio.RunDir.create(run_root, "u", config=cfg, kind="test")
    text = "line separator para\u0085next"
    run.append("transcripts", {"id": "t", "text": text})
    run.close()
    assert runio.RunDir.open(run_root / "u").rows("transcripts")[0]["text"] == text


def test_a_torn_tail_is_quarantined_before_the_next_append(cfg, run_root):
    """Old invariant 103: crash, resume, read -- the fragment is kept aside, never glued on."""
    run = runio.RunDir.create(run_root, "t", config=cfg, kind="test")
    run.append("cells", {"cell": "a"})
    run.close()
    p = run.stream_path("cells")
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"schema": "m5.v1", "cell": "b", "k"')  # the crash
    resumed = runio.RunDir.open(run_root / "t")
    resumed.append("cells", {"cell": "c"})
    resumed.close()
    assert [r["cell"] for r in runio.RunDir.open(run_root / "t").rows("cells")] == ["a", "c"]
    assert '"cell": "b"' in (p.parent / "cells.jsonl.torn").read_text()


def test_a_complete_last_row_missing_its_newline_is_kept(cfg, run_root):
    run = runio.RunDir.create(run_root, "n", config=cfg, kind="test")
    run.append("cells", {"cell": "a"})
    run.close()
    p = run.stream_path("cells")
    p.write_text(p.read_text().rstrip("\n"))
    resumed = runio.RunDir.open(run_root / "n")
    resumed.append("cells", {"cell": "b"})
    resumed.close()
    assert [r["cell"] for r in runio.RunDir.open(run_root / "n").rows("cells")] == ["a", "b"]
    assert not (p.parent / "cells.jsonl.torn").exists()


def test_rows_carry_an_attempt_id_per_process(cfg, run_root):
    run = runio.RunDir.create(run_root, "a", config=cfg, kind="test")
    run.append("cells", {"cell": "x"})
    run.close()
    again = runio.RunDir.open(run_root / "a")
    again.append("cells", {"cell": "y"})
    again.close()
    rows = runio.RunDir.open(run_root / "a").rows("cells")
    assert rows[0]["attempt"] != rows[1]["attempt"]


def test_git_state_says_unknown_rather_than_clean_when_it_cannot_ask(tmp_path):
    assert runio.git_state(tmp_path)["dirty"] is None


def test_write_probe(tmp_path):
    runio.write_probe(tmp_path, megabytes=1)
    assert not list(tmp_path.glob(".write_probe_*"))
    with pytest.raises(safety.Refused):
        runio.write_probe(safety.REPO_ROOT, megabytes=1)


def test_cell_doses_are_canonical():
    a = Cell("silk", "single", (29,), 0.44999999999999996)
    b = Cell("silk", "single", (29,), 0.45)
    assert a == b and a.cell_id() == b.cell_id()
