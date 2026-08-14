"""Offline acceptance tests for task 29's judge-free probe.

No GPU, no model, no judge key. The two tests that matter most here are structural rather than
behavioural -- that the probe cannot reach a judge, and that the prefilled and unprefilled
noticing prompts come from one builder -- because both are claims about every branch of the
module and neither can be established by running one happy path.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from m2 import config, prompts, run, unjudged


# =====================================================================================
# The cell list
# =====================================================================================

def test_default_cells_are_the_band_the_shortlist_gate_excluded():
    """L37-L46 x four doses. The 2026-08-14 run measured d2 at L47 and deeper only."""
    assert unjudged.DEFAULT_CELLS == tuple(
        (layer, dose) for layer in range(37, 47)
        for dose in (0.15, 0.20, 0.25, 0.30))
    assert len(unjudged.DEFAULT_CELLS) == 40
    layers = {layer for layer, _r in unjudged.DEFAULT_CELLS}
    assert max(layers) < 47, "the probe must not re-measure the band that already read d2 = 1.0"


def test_grid_syntax_expands_ranges_and_deduplicates():
    assert unjudged.parse_cells("37-39@0.15,0.30", with_references=False) == [
        (37, 0.15), (37, 0.30), (38, 0.15), (38, 0.30), (39, 0.15), (39, 0.30)]
    # A cell named twice is measured once; segments may overlap without inflating the plan.
    assert unjudged.parse_cells("37@0.15;37@0.15,0.30", with_references=False) == [
        (37, 0.15), (37, 0.30)]
    assert unjudged.parse_cells("L37@0.2;52@0.6", with_references=False) == [(37, 0.2), (52, 0.6)]


def test_reference_anchors_are_appended_last_and_can_be_dropped():
    """Appended, so a run cut short still has the mid-band cells it was for."""
    with_refs = unjudged.parse_cells(None)
    assert with_refs[-2:] == [(57, 0.22), (59, 0.30)]
    assert with_refs[:-2] == list(unjudged.DEFAULT_CELLS)
    assert unjudged.parse_cells(None, with_references=False) == list(unjudged.DEFAULT_CELLS)
    # Already present in the request -> not duplicated.
    assert unjudged.parse_cells("57@0.22;59@0.30").count((57, 0.22)) == 1


@pytest.mark.parametrize("text,match", [
    ("nonsense", "invalid probe segment"),
    ("37@0", "r must be finite and above zero"),
    ("37@-0.2", "invalid probe segment"),
    ("46-37@0.15", "runs backwards"),
    ("37@abc", "invalid probe segment"),
])
def test_malformed_cell_specs_raise_rather_than_being_dropped(text, match):
    with pytest.raises(ValueError, match=match):
        unjudged.parse_cells(text, with_references=False)


def test_doses_are_rounded_so_two_spellings_of_one_cell_join():
    """Item 24: 0.45 reached by arithmetic is 0.44999999999999996 and keys as a second cell."""
    assert unjudged.parse_cells("37@0.44999999999999996;37@0.45",
                                with_references=False) == [(37, 0.45)]


def test_trial_list_and_trial_count_cannot_disagree():
    assert len(unjudged.PROBE_TRIALS) == unjudged.PROBE_N
    assert len(set(unjudged.PROBE_TRIALS)) == unjudged.PROBE_N


# =====================================================================================
# The mechanical influence count
# =====================================================================================

def test_concept_hits_counts_inflections_and_ignores_case():
    """No trailing boundary: "garlicky" is a mention, and scoring it clean would be wrong."""
    assert unjudged.concept_hits("Garlicky garlic bread, no GARLIC left", "Garlic") == 3
    assert unjudged.concept_hits("a plain sentence", "Garlic") == 0
    # A leading boundary IS required, so an unrelated word containing the concept is not a hit.
    assert unjudged.concept_hits("vulgarlic is not a word", "Garlic") == 0
    with pytest.raises(ValueError, match="non-empty concept"):
        unjudged.concept_hits("text", "  ")


# =====================================================================================
# Judge-free, structurally
# =====================================================================================

_JUDGE_NAMES = frozenset({
    "_issue", "_item", "call_judge", "judge_many", "judge_fpr", "judge_nulls",
    "judge_s1_null", "judge_d2_null", "measure_D2", "measure_E5", "measure_S1",
    "verify_cell", "measure_sanity_anchor",
})


def _called_names(source: str) -> set[str]:
    """Every function name this module calls, whether bare or through an attribute."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            out.add(func.id)
        elif isinstance(func, ast.Attribute):
            out.add(func.attr)
    return out


def test_the_probe_module_never_calls_a_judged_measurement():
    """The claim "this mode calls no judge" is about every branch, so check the source.

    `no_judges()` turns a judge call into a crash at runtime, but only for code that runs. This
    catches a scored measurement added to a branch that a smoke test never enters -- including
    `measure_D2`, which is judged and is the obvious thing for someone to reach for when adding
    "just the detection number" to this mode later.
    """
    source = Path(unjudged.__file__).read_text(encoding="utf-8")
    called = _called_names(source)
    # `_refuse` is assigned OVER these names inside no_judges(); it never calls them.
    offenders = sorted(called & _JUDGE_NAMES)
    assert not offenders, (
        f"m2.unjudged calls judged measurement(s) {offenders}. This mode exists to produce "
        "transcripts with nothing scored; a judged call here spends money and writes a "
        "judge_*.jsonl the operator would then read as evidence.")


def test_the_probe_only_generates_through_the_padding_safe_batched_paths():
    """bug 25b: `generate_batch` mis-slices unequal-length prompts. Neither channel may use it."""
    source = Path(unjudged.__file__).read_text(encoding="utf-8")
    called = _called_names(source)
    assert "generate_batch" not in called
    assert "generate_batch_with_steering" not in called
    assert {"generate_steered", "generate_unsteered"} <= called


def test_no_judges_names_the_entry_points_it_guards():
    """The guard must patch the module-level entry points, not a local alias of them."""
    source = inspect.getsource(unjudged.no_judges)
    for name in ("_issue", "call_judge", "judge_many"):
        assert f'"{name}"' in source, f"no_judges() does not guard {name}"
    assert "finally" in source, "no_judges() must restore the originals even when the body raises"


# =====================================================================================
# The two noticing prompts
# =====================================================================================

def test_detect_and_forced_prompts_come_from_one_builder():
    """Only the prefill may differ. Two copies of the framing is how the arms stop matching.

    R7 (`prompts.verify_forced_prompts`) checks the prefilled prompt byte-for-byte against the
    repo's own function. Nothing checks the unprefilled one, so the guarantee it has to rest on
    is that it is the same construction minus one string.
    """
    forced = inspect.getsource(prompts.forced_prompts)
    detect = inspect.getsource(prompts.detect_prompts)
    assert "_noticing_prompts(trial_numbers, FORCED_PREFILL)" in forced
    assert '_noticing_prompts(trial_numbers, "")' in detect
    # The framing text lives in the shared builder and in no caller.
    for body in (forced, detect):
        assert "interpretability researcher" not in body


def test_the_shared_builder_still_carries_the_verbatim_framing_and_prefill():
    shared = inspect.getsource(prompts._noticing_prompts)
    assert "I am an interpretability researcher studying transformer-based language models" in shared
    assert "Do you detect an injected thought?" in shared
    assert "template += prefill" in shared
    assert prompts.FORCED_PREFILL == "Yes, I detect an injected thought. The thought is about"


# =====================================================================================
# The dual-use guards, same policy as the task 25 diagnostic
# =====================================================================================

def test_probe_refuses_more_than_one_concept_before_any_other_work():
    with pytest.raises(ValueError, match="exactly one benign concept"):
        unjudged.prepare(["Garlic", "Origami"], dict(config.CONFIG), "")


def test_harmful_concept_raises_before_any_output_is_written(tmp_path, monkeypatch):
    output_root = tmp_path / "runs"
    monkeypatch.setenv("M2_RUNS_DIR", str(output_root))
    with pytest.raises(PermissionError, match="not on BENIGN_CONCEPTS"):
        run.main(["--concepts", "weapon", "--probe-cells", "--dry-run"])
    assert not output_root.exists()


def test_probe_cannot_write_a_run_or_log_inside_the_repository(tmp_path, monkeypatch):
    fake_repo = Path(unjudged.__file__).resolve().parents[1]
    monkeypatch.setenv("M2_RUNS_DIR", str(fake_repo / "forbidden-probe-runs"))
    with pytest.raises(ValueError, match="inside repository"):
        unjudged.prepare(["Garlic"], dict(config.CONFIG), "")

    monkeypatch.setenv("M2_RUNS_DIR", str(tmp_path / "external-runs"))
    with pytest.raises(ValueError, match="inside repository"):
        unjudged.prepare(["Garlic"], dict(config.CONFIG), "", log_path=fake_repo / "bad.log")


def test_probe_has_no_transcript_override_parameter():
    """Its whole output is transcripts, so a non-benign concept must have no way in."""
    assert "EXPORT_TRANSCRIPTS_OVERRIDE" not in inspect.signature(unjudged.prepare).parameters
    assert "EXPORT_TRANSCRIPTS_OVERRIDE" not in inspect.signature(
        unjudged.export_bundle).parameters


def test_every_transcript_file_name_trips_the_fail_closed_export_filter():
    """`runio.is_transcript` is what withholds model text from a harmful concept's bundle."""
    from m2 import runio

    for name in (unjudged.DETECT_FILE, unjudged.FORCED_FILE,
                 unjudged.TASK_FILE, unjudged.NULL_FILE):
        assert runio.is_transcript(name), f"{name} would not be recognised as a transcript"
    # And the scalar files must NOT be, or a harmful-arm bundle would lose its numbers too.
    assert not runio.is_transcript(unjudged.CELLS_FILE)
    assert not runio.is_transcript(unjudged.SUMMARY_FILE)


# =====================================================================================
# CLI wiring
# =====================================================================================

def test_the_two_standalone_modes_are_mutually_exclusive(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("M2_RUNS_DIR", str(tmp_path / "runs"))
    assert run.main(["--concepts", "Garlic", "--probe-cells",
                     "--autopsy-cells", "--dry-run"]) == run.EXIT_CONFIG
    assert "two different standalone modes" in capsys.readouterr().out


def test_judge_free_mode_does_not_require_a_judge_key(monkeypatch):
    """A mode that cannot spend OPENROUTER_API_KEY must not refuse to start without one."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "x")
    assert run.check_environment(strict=True, required=("HF_TOKEN",))["missing_required"] == []
    # ...but the pipeline path still refuses, because Phase 4 would die forty minutes in.
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        run.check_environment(strict=True)


def test_check_environment_required_narrows_and_never_widens():
    with pytest.raises(ValueError, match="not in REQUIRED_ENV"):
        run.check_environment(strict=False, required=("HF_TOKEN", "NOT_A_CREDENTIAL"))


# =====================================================================================
# Provenance
# =====================================================================================

def test_git_state_records_the_commit_and_whether_the_tree_was_dirty():
    """TODO item 16: the 2026-08-14 bundle could not be traced to the code that made it."""
    from m2 import model

    state = model._git_state()
    assert set(state) == {"git_commit", "git_branch", "git_dirty"}
    if state["git_commit"] is not None:
        assert len(state["git_commit"]) == 40
        # None means "could not tell", which is a different claim from "clean".
        assert state["git_dirty"] in (True, False, None)
