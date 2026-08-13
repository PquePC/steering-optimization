"""Offline acceptance tests for task 25's disposable diagnostic."""

from __future__ import annotations

import inspect
from pathlib import Path
import pytest

from m2 import autopsy, config, run


def test_defaults_include_the_mandatory_positive_control_and_override_cannot_drop_it():
    assert autopsy.parse_cells("") == list(autopsy.DEFAULT_CELLS)
    with pytest.raises(ValueError, match="positive-control cell L59@0.30"):
        autopsy.parse_cells("57@0.30,58@0.30,52@0.60")
    with pytest.raises(ValueError, match="invalid autopsy cell"):
        autopsy.parse_cells("not-a-cell,59@0.30")


def test_autopsy_refuses_more_than_one_concept_before_any_other_work():
    with pytest.raises(ValueError, match="exactly one benign concept"):
        autopsy.prepare(["Garlic", "Origami"], dict(config.CONFIG), "")


def test_harmful_concept_raises_before_any_output_is_written(tmp_path, monkeypatch):
    output_root = tmp_path / "runs"
    monkeypatch.setenv("M2_RUNS_DIR", str(output_root))
    with pytest.raises(PermissionError, match="not on BENIGN_CONCEPTS"):
        run.main(["--concepts", "weapon", "--autopsy-cells", "--dry-run"])
    assert not output_root.exists()


def test_autopsy_cannot_write_a_run_or_log_inside_the_repository(tmp_path, monkeypatch):
    fake_repo = Path(autopsy.__file__).resolve().parents[1]
    monkeypatch.setenv("M2_RUNS_DIR", str(fake_repo / "forbidden-autopsy-runs"))
    with pytest.raises(ValueError, match="inside repository"):
        autopsy.prepare(["Garlic"], dict(config.CONFIG), "")

    monkeypatch.setenv("M2_RUNS_DIR", str(tmp_path / "external-runs"))
    with pytest.raises(ValueError, match="inside repository"):
        autopsy.prepare(["Garlic"], dict(config.CONFIG), "", log_path=fake_repo / "bad.log")


def test_dropped_variants_are_printed_with_their_verdict_reason(capsys):
    autopsy._print_variants(
        "Garlic",
        [dict(variant=" garlic", token_id=1, decodes_to=" garlic",
              reason="prefix of the concept")],
        [dict(variant="GARLIC", token_id=2, decodes_to="G",
              reason="first token 'g' is under 3 characters (bug 20)")])
    output = capsys.readouterr().out
    assert "DROPPED" in output
    assert "GARLIC" in output
    assert "under 3 characters" in output


def test_positive_control_guard_trips_on_non_cid_or_low_probability():
    with pytest.raises(RuntimeError, match="positive control FAILED"):
        autopsy._require_positive_control([
            dict(trial=1, token_id=10, probability=0.99, in_cids=False)])
    with pytest.raises(RuntimeError, match="positive control FAILED"):
        autopsy._require_positive_control([
            dict(trial=1, token_id=11, probability=0.90, in_cids=True)])
    autopsy._require_positive_control([
        dict(trial=1, token_id=11, probability=0.900001, in_cids=True)])


def test_positive_control_guard_is_wired_to_l59_and_not_to_other_cells():
    broken = [dict(trial=1, token_id=10, probability=0.99, in_cids=False)]
    assert autopsy._validate_control_cell(58, 0.30, broken) is False
    with pytest.raises(RuntimeError, match="positive control FAILED"):
        autopsy._validate_control_cell(59, 0.30, broken)
    assert autopsy._validate_control_cell(
        59, 0.30,
        [dict(trial=1, token_id=11, probability=0.99, in_cids=True)]) is True


def test_top_token_table_marks_kept_and_dropped_ids(capsys):
    autopsy._print_top_rows([
        dict(token_id=11, probability=0.999, token=" garlic"),
        dict(token_id=12, probability=0.0005, token="G"),
    ], {11}, {12})
    output = capsys.readouterr().out
    assert "1        11   0.99900000 CID" in output
    assert "2        12   0.00050000 DROPPED" in output


def test_internal_consistency_guards_each_trip():
    with pytest.raises(RuntimeError, match="ALLOW_FILLER=True"):
        autopsy._require_allow_filler(False)
    with pytest.raises(RuntimeError, match="no steering start position"):
        autopsy._require_start_pos(None)
    with pytest.raises(RuntimeError, match="disagree with d3 cids"):
        autopsy._require_matching_ids([1, 2], [1, 3])
    with pytest.raises(RuntimeError, match="no trial readings"):
        autopsy._require_d3_reading(None)


def test_run_requires_a_loaded_concept_context(monkeypatch):
    empty = config.RunContext()
    monkeypatch.setattr(config, "RUN", empty)
    with pytest.raises(RuntimeError, match="driver.set_concept"):
        autopsy.run_autopsy(autopsy.DEFAULT_CELLS)


def test_real_d2_uses_the_same_small_fixed_trials_as_d3():
    calls = []

    class FakeExpensive:
        @staticmethod
        def measure_D2(*args, **kwargs):
            calls.append((args, kwargs))
            return {"d2": 0.2}

    trials = (1, 7, 13, 19, 25)
    assert autopsy._measure_real_d2(
        FakeExpensive, layer=58, r=0.30, alpha=2.39, trials=trials)["d2"] == 0.2
    args, kwargs = calls[0]
    assert args == (58, 2.39, 5)
    assert kwargs == {"phase": "AUTOPSY", "r": 0.30, "trial_numbers": list(trials)}


def test_token_dump_reuses_d3_forward_without_rebuilding_the_prompt():
    source = inspect.getsource(autopsy._dump_and_score_d3)
    assert "cheap._d3_forward" in source
    assert "forced_prompts" not in source
    assert "model.encode" not in source


def test_four_cell_summary_keeps_d3_rate_rank_and_real_d2_side_by_side(capsys):
    rows = [dict(layer=layer, r=r, d3=0.01 * i, d3_rate=0.2 * i,
                 d3_rank_med=2 if layer != 59 else 1, d2=0.25 * i)
            for i, (layer, r) in enumerate(autopsy.DEFAULT_CELLS)]
    autopsy._print_summary_table(rows)
    output = capsys.readouterr().out
    assert all(name in output for name in ("d3", "d3_rate", "d3_rank_med", "d2"))
    assert output.count("L57@0.30") == 1
    assert output.count("L58@0.30") == 1
    assert output.count("L59@0.30") == 1
    assert output.count("L52@0.60") == 1


def test_run_exposes_one_optional_value_flag_with_documented_defaults():
    parser = run._parser()
    defaults = parser.parse_args(["--concepts", "Garlic", "--autopsy-cells"])
    override = parser.parse_args([
        "--concepts", "Garlic", "--autopsy-cells", "57@.3,58@.3,59@.3,52@.6"])
    assert defaults.autopsy_cells == ""
    assert override.autopsy_cells == "57@.3,58@.3,59@.3,52@.6"
