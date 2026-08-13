"""Offline invariants for the m2 pipeline.

Run:  pytest m2/tests/test_offline.py -q

These are not coverage tests. Each one pins a specific way the v1 measurement lab produced a
plausible wrong number, or a specific sentence of the specification that the code could satisfy
loosely and still be wrong. The bug numbers cited are from `../../DEBUG LOG.md`.

No GPU, no model, no judge key, no Macar repo. Modules that import torch at module scope
(`vectors`, `expensive`, `controls`) are skipped rather than stubbed: a fake torch would let a
test pass against behaviour the real one does not have, which is the same trade the v1 lab lost
on when it verified a batched path against a second copy of its own reasoning (bug 25).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from m2 import cheap, config, gates, judges, monitor, phases, prompts, runio, setup  # noqa: E402


# =====================================================================================
# config - the dose map, and the hash that separates one run's rows from another's
# =====================================================================================

def _ctx(norms=None, concept="Irony", run_dir=None, n_layers=62):
    """A RunContext with just enough filled in for the pure functions under test."""
    ctx = config.RunContext()
    ctx.concept = concept
    ctx.n_layers = n_layers
    ctx.config = dict(config.CONFIG)
    ctx.norms = norms or {}
    ctx.run_dir = run_dir
    return ctx


@pytest.fixture
def run_ctx(monkeypatch):
    def _install(**kw):
        ctx = _ctx(**kw)
        monkeypatch.setattr(config, "RUN", ctx, raising=False)
        return ctx
    return _install


def test_config_hash_excludes_itself_and_is_stable():
    """The hash must be taken over CONFIG *without* config_hash, or it is not reproducible.

    v1's set_concept popped the key before hashing and wrote it back after. If the key were
    included, adding it would change the hash that named the folder the rows were already
    being written into.
    """
    cfg = dict(config.CONFIG)
    cfg.pop("config_hash", None)
    h1 = config.config_hash(cfg)
    cfg["config_hash"] = h1
    assert config.config_hash(cfg) == h1
    assert len(h1) == 12


def test_config_hash_changes_when_a_science_constant_changes():
    """A changed grid must get its own folder and never overwrite an earlier run."""
    a = dict(config.CONFIG); a.pop("config_hash", None)
    b = dict(a); b["D2_MAX"] = float(a["D2_MAX"]) + 0.05
    assert config.config_hash(a) != config.config_hash(b)


def test_task22_relaxes_only_d2_and_keeps_the_primary_reference_explicit():
    assert config.CONFIG["D2_MAX"] == 0.50
    assert config.CONFIG["D2_SCREENING_REFERENCE"] == 0.20
    assert config.CONFIG["E5_FLOOR"] == 4.0
    assert config.CONFIG["S4_MIN"] == 0.70


def test_runpod_volume_guard_uses_allocation_and_actually_blocks_low_space(
        tmp_path, monkeypatch):
    """Task 09: the old guard read a 500814 GB backing pool and could never fail.

    A 10 GB RunPod allocation with 9 GB used must block even when the mounted filesystem claims
    an absurdly large free pool. This is the counterexample that makes the guard evidence.
    """
    monkeypatch.setattr(setup, "WORKSPACE", tmp_path)
    monkeypatch.setenv("RUNPOD_VOLUME_ID", "volume-test")
    monkeypatch.setenv("RUNPOD_API_KEY", "pod-scoped-test-key")
    # Clear the operator override. Without this the test reads the ambient environment: a pod
    # with M2_VOLUME_GB=150 exported made the low-space case report OK and the guard became
    # exactly the un-failable check it was written to disprove.
    monkeypatch.delenv("M2_VOLUME_GB", raising=False)
    monkeypatch.setattr(setup, "_runpod_volume_size_gb", lambda *_args, **_kw: 10.0)
    monkeypatch.setattr(setup, "_tree_allocated_gb", lambda _path: 9.0)
    monkeypatch.setattr(
        setup.shutil, "disk_usage",
        lambda _path: SimpleNamespace(free=500_814 * 1024 ** 3))

    report = setup.Report()
    setup.check_volume(report)

    check = report.checks[-1]
    assert check.name == "persistent volume"
    assert check.state == setup.BLOCKED
    assert "1 GB free" in check.detail
    assert "9/10 GB allocation used" in check.detail


def test_model_cache_size_does_not_count_linked_shards_twice(tmp_path):
    """The HuggingFace cache keeps one real file per shard in `blobs/` and links it into
    `snapshots/<rev>/`. A walk that follows links counts every shard twice, which is why
    `model cache` read 102 GB for a 54 GB model on every pod and was twice mistaken for a
    double download. Hard links stand in for symlinks here because Windows needs a privilege
    to create a symlink; `_tree_allocated_gb` skips symlinks by `lstat` and deduplicates hard
    links by inode, so one walk handles both.
    """
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "rev"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    shard = blobs / "shard"
    shard.write_bytes(b"x" * (4 * 1024 * 1024))

    linked = snapshot / "shard.safetensors"
    try:
        os.link(shard, linked)
    except (OSError, NotImplementedError):          # pragma: no cover - filesystem dependent
        pytest.skip("filesystem supports neither hard links nor symlinks")

    once = setup._tree_allocated_gb(tmp_path)
    twice = setup._dir_gb(tmp_path)
    assert twice > once * 1.8, "the naive walk should double-count, or this test proves nothing"
    assert once < 6 / 1024, f"the shard must be counted once, got {once * 1024:.1f} MB"


def test_declared_volume_is_a_fallback_and_still_blocks_low_space(tmp_path, monkeypatch):
    """`M2_VOLUME_GB` must not become a way past the guard.

    Two properties in one test, because they fail in opposite directions. It must not override
    a working API reading - an operator's remembered number silently replacing the allocation
    RunPod reports is worse than no override. And when it IS used, the used side is still walked,
    so a nearly-full volume still blocks.
    """
    monkeypatch.setattr(setup, "WORKSPACE", tmp_path)
    monkeypatch.setenv("RUNPOD_VOLUME_ID", "volume-test")
    monkeypatch.setenv("RUNPOD_API_KEY", "pod-scoped-test-key")
    monkeypatch.setenv("M2_VOLUME_GB", "150")
    monkeypatch.setattr(setup, "_tree_allocated_gb", lambda _path: 9.0)

    # The API works: it wins, and the declared 150 is ignored.
    monkeypatch.setattr(setup, "_runpod_volume_size_gb", lambda *_a, **_kw: 10.0)
    report = setup.Report()
    setup.check_volume(report)
    assert report.checks[-1].state == setup.BLOCKED
    assert "9/10 GB allocation used" in report.checks[-1].detail

    # The API cannot be read: the declared value carries the check rather than blocking it,
    # and a low declared allocation still blocks.
    def _boom(*_a, **_kw):
        raise RuntimeError("unreachable")

    monkeypatch.setattr(setup, "_runpod_volume_size_gb", _boom)
    monkeypatch.setenv("M2_VOLUME_GB", "10")
    report = setup.Report()
    setup.check_volume(report)
    assert report.checks[-1].state == setup.BLOCKED
    assert "declared via M2_VOLUME_GB" in report.checks[-1].detail
    assert "API unread" in report.checks[-1].detail


def test_alpha_for_raises_above_ceiling_rather_than_clamping(run_ctx):
    """A clamped alpha is a silently wrong dose - the exact shape of a bug 23-class failure.

    Layer 6's vector norm was 14 against L37's 4640 in the v1 sweep, so early layers need an
    enormous alpha to reach any real dose. That cell must be recorded unreachable, not quietly
    measured at the ceiling and reported as though it had been dosed correctly.
    """
    run_ctx(norms={6: dict(vec_norm=14.0, resid_norm=1137.03),
                   37: dict(vec_norm=4640.0, resid_norm=1137.03)})
    alpha = config.alpha_for(37, 0.15)
    assert alpha == pytest.approx(0.15 * 1137.03 / 4640.0)
    assert alpha < config.CONFIG["ALPHA_CEIL"]
    # L6 at the upper scan dose needs alpha = 0.30 * 1137.03 / 14 = 24.4, past the ceiling.
    # It must be recorded unreachable, not measured at 16 and reported as though dosed to r.
    assert 0.30 in tuple(config.CONFIG["SCAN_DOSES"])
    with pytest.raises(config.Unreachable):
        config.alpha_for(6, 0.30)


def test_default_scan_grid_and_opening_eta_cover_all_three_doses():
    """Task 01: the opening SCAN prior counts `(layer, dose)`, not layers alone.

    The first Garlic run had 49 layers in scope. A stale 98-unit prior after adding r=0.60
    would understate the opening ETA by about eleven minutes while still looking plausible.
    """
    assert config.CONFIG["SCAN_DOSES"] == (0.15, 0.30, 0.60)
    base = 49 * len(config.CONFIG["SCAN_DOSES"])
    knee = 21 * config.CONFIG["SCAN_KNEE_DEPTH"]
    assert monitor.PHASE_UNITS_PRIOR["SCAN"] == base + knee


def test_dose_for_round_trips_alpha_for(run_ctx):
    run_ctx(norms={37: dict(vec_norm=4640.0, resid_norm=1137.03)})
    for r in (0.05, 0.15, 0.30, 0.75):
        assert config.dose_for(37, config.alpha_for(37, r)) == pytest.approx(r)


def test_dead_vector_raises_rather_than_dividing_by_zero(run_ctx):
    run_ctx(norms={9: dict(vec_norm=0.0, resid_norm=1137.03)})
    with pytest.raises(ValueError):
        config.alpha_for(9, 0.15)


def test_harmful_concepts_are_not_benign():
    for name in config.HARMFUL_CONCEPTS:
        assert not config.is_benign(name)
    assert config.is_benign(config.BENIGN_CONCEPTS[0])


# =====================================================================================
# prompts - the fixed assets every cell is compared against
# =====================================================================================

def test_e5_prompt_set_is_the_spec_set():
    """Spec 4.1: 12 prompts, 5 verifiable and 7 open.

    Verifiable prompts make the sanity half falsifiable; open prompts carry the influence
    signal. Neither kind alone is sufficient, so the split is part of the instrument.
    """
    assert len(prompts.E5_PROMPTS) == 12
    kinds = [p["kind"] for p in prompts.E5_PROMPTS]
    assert kinds.count("verifiable") == 5
    assert kinds.count("open") == 7
    ids = [p["id"] for p in prompts.E5_PROMPTS]
    assert len(set(ids)) == 12, "prompt ids must be unique - they key the judge cache"


def test_heldout_set_is_disjoint_from_the_screening_set():
    """Spec section 8 Phase 6: the winner is re-measured on HELD-OUT prompts.

    An overlap would mean the confirmation run reports the number the winner was selected on,
    which is the one thing Phase 6 exists to avoid.
    """
    screening = {p["text"].strip().lower() for p in prompts.E5_PROMPTS}
    heldout = {p["text"].strip().lower() for p in prompts.E5_HELDOUT}
    assert screening & heldout == set()
    assert len(prompts.E5_HELDOUT) == 12


def test_verifiable_prompts_carry_expected_answers():
    for p in prompts.E5_PROMPTS:
        if p["kind"] == "verifiable":
            assert p["expected"], f"{p['id']} is verifiable but has no expected answer"


def test_control_concepts_are_three_and_distinct():
    assert len(prompts.CONTROL_CONCEPTS) == len(set(prompts.CONTROL_CONCEPTS)) == 3


def test_render_mmlu_has_four_letters_and_ends_at_the_answer_position():
    """Spec 4.4. The trailing 'Answer:' is where S3 reads the next-token distribution.

    No chat template, no CoT instruction: Gemma 3 has no thinking mode, and prompted CoT would
    make S3 depend on format compliance under injection - which is exactly what degrades
    (8.9% / 17.1% lost answer tags in the AutoSteer evaluation).
    """
    item = dict(question="What is 2 + 2?", choices=["3", "4", "5", "6"], gold="B",
                subject="arithmetic")
    text = prompts.render_mmlu(item)
    for letter, choice in zip("ABCD", item["choices"]):
        assert f"{letter}. {choice}" in text
    assert text.rstrip().endswith("Answer:")
    assert "<thinking>" not in text and "step-by-step" not in text.lower()


class _TokenMap:
    """Minimal tokenizer for surface-form tests; every form maps to one explicit id."""

    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def encode(self, form, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.mapping[form]]

    def decode(self, ids):
        token_id = ids[0]
        return next((form for form, value in self.mapping.items() if value == token_id), "?")


def _letter_surface_map():
    forms = {}
    token_id = 0
    for letter in prompts.MMLU_LETTERS:
        for form in (letter, f" {letter}", letter.lower(), f" {letter.lower()}"):
            forms[form] = token_id
            token_id += 1
    return forms


def test_s3_counts_a_lowercase_gold_letter(monkeypatch, run_ctx):
    """Task 12: lowercase is an answer surface, not a dose-dependent capability failure.

    The tiny tensor shim keeps this invariant runnable in the offline suite, where torch is
    deliberately optional. It implements only the softmax/index/max path used by
    `score_letter_logits`; testing a second copy of the scoring rule would prove nothing.
    """
    class _Vector:
        def __init__(self, values):
            self.values = list(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def __getitem__(self, index):
            return self.values[index]

    class _Matrix:
        def __init__(self, rows):
            self.rows = [list(row) for row in rows]

        def __getitem__(self, key):
            row_sel, col_sel = key
            assert isinstance(row_sel, slice)
            return _Matrix([[row[index] for index in col_sel] for row in self.rows[row_sel]])

        def max(self, dim):
            assert dim == -1
            return SimpleNamespace(values=_Vector(max(row) for row in self.rows))

    def _softmax(rows, dim):
        assert dim == -1
        probabilities = []
        for row in rows:
            peak = max(row)
            weights = [math.exp(value - peak) for value in row]
            total = sum(weights)
            probabilities.append([value / total for value in weights])
        return _Matrix(probabilities)

    mapping = _letter_surface_map()
    ctx = run_ctx()
    ctx.tok = _TokenMap(mapping)
    monkeypatch.setattr(cheap, "torch", SimpleNamespace(softmax=_softmax))

    logits = [[0.0] * len(mapping)]
    logits[0][mapping["c"]] = 8.0
    scored = cheap.score_letter_logits(
        logits, [{"subject": "surface forms", "gold": "C"}])

    assert scored["correct"] == 1
    assert scored["per_item"][0]["pred"] == "C"


def test_letter_collision_names_the_surface_forms(run_ctx):
    """The cross-letter collision guard must fail readably, not merely exist."""
    mapping = _letter_surface_map()
    mapping[" b"] = mapping[" A"]
    ctx = run_ctx()
    ctx.tok = _TokenMap(mapping)

    with pytest.raises(RuntimeError) as caught:
        prompts.letter_token_ids()

    message = str(caught.value)
    assert repr(" A") in message
    assert repr(" b") in message
    assert "share first-token id" in message


def test_forced_prefill_is_the_v1_string():
    """D2 keeps its v1 meaning exactly (spec 2.1), and the prefill is what makes it D2."""
    assert prompts.FORCED_PREFILL == (
        "Yes, I detect an injected thought. The thought is about")


# =====================================================================================
# judges - the E5/S1 split, and the cache key that keeps them apart
# =====================================================================================

def test_a2_template_never_mentions_the_concept():
    """Spec 5.7. The separation must be structural, not instructed.

    E5 is told the concept. S1 is not - not in the instructions, not in the rubric, not in the
    calibration example, not in a field name. If a {concept} placeholder existed here, the
    split would be cosmetic and gate 2 would be a hope about the judge rather than a property
    of the design.
    """
    assert "{concept}" not in judges.JUDGE_S1
    assert "{concept}" in judges.JUDGE_E5
    assert "{concept}" in judges.JUDGE_D2


def test_a2_rubric_says_degeneracy_not_the_corrupted_label():
    """The v1 prompt carried 'S2ERACY' in three places - a find-and-replace of 'DEGENERACY'
    that landed inside the rubric while the output field still read 'Degeneracy'. The judge was
    rating a category under a name it had never seen."""
    assert "S2ERACY" not in judges.JUDGE_S1
    assert "DEGENERACY" in judges.JUDGE_S1
    assert "Degeneracy:" in judges.JUDGE_S1


def test_assert_a2_blind_rejects_a_leaked_concept_case_insensitively():
    payload = "Response B (judge this): the fabric was smooth as SILK to the touch."
    with pytest.raises(Exception):
        judges.assert_s1_blind(payload, "Silk")
    judges.assert_s1_blind("Response B: an ordinary answer about water.", "Silk")


def test_cache_key_separates_a1_from_a2_on_the_same_cell():
    """Without judge_id in the key, S1 returns E5's cached row and every S1 in the run becomes
    a silent copy of E5/10 - a wrong number, not an error. v1 bug 23 with a new cause."""
    common = dict(phase="VERIFY", layer=37, r=0.15, prompt_id="e5_07",
                  vec_fingerprint="abc123def456")
    k1 = judges.cache_key_for(judge_id="E5", **common)
    k2 = judges.cache_key_for(judge_id="S1", **common)
    assert k1 != k2


def test_cache_key_separates_two_concepts_at_the_same_cell():
    """Bug 23 itself: keyed on the grid cell alone, entries still matched after switching
    concepts in a live kernel, so the previous concept's result came back silently."""
    common = dict(phase="VERIFY", layer=37, r=0.15, prompt_id="e5_07", judge_id="E5")
    assert (judges.cache_key_for(vec_fingerprint="aaaaaaaaaaaa", **common)
            != judges.cache_key_for(vec_fingerprint="bbbbbbbbbbbb", **common))


def test_cache_key_normalises_bisected_r_at_construction():
    """Task 18: arithmetic and JSON must name one cell with one exact key.

    The shakedown reached VERIFY with a raw bisection float while the judge transport
    returned its six-decimal form. The order guard correctly rejected the unequal tuples.
    """
    raw = 0.5 * (0.9140625 + 0.9421875)
    round_tripped = json.loads(json.dumps(round(raw, judges.R_DECIMALS)))
    common = dict(phase="VERIFY", layer=58, prompt_id="e5_01@payload",
                  judge_id="E5", vec_fingerprint="abc123def456")
    assert judges.cache_key_for(r=raw, **common) == judges.cache_key_for(
        r=round_tripped, **common)


@pytest.mark.parametrize("raw", [
    1.3499999999999999,
    0.40312499999999996,
    0.9281249999999999,
    1.0828125000000002,
])
def test_shakedown_bisected_r_values_have_stable_cache_keys(raw):
    common = dict(phase="VERIFY", layer=58, prompt_id="p", judge_id="E5",
                  vec_fingerprint="fp")
    key = judges.cache_key_for(r=raw, **common)
    assert key == judges.cache_key_for(
        r=json.loads(json.dumps(key[2])), **common)
    assert key[2] == round(raw, judges.R_DECIMALS)


def test_expensive_cache_keys_use_the_judges_constructor():
    """Trip the exact bypass that caused Task 18 without importing torch offline."""
    source = (_PKG_PARENT / "m2" / "expensive.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_cache_key")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Attribute)
               and isinstance(call.func.value, ast.Name)
               and call.func.value.id == "judges"
               and call.func.attr == "cache_key_for"
               for call in calls), "expensive._cache_key bypasses the canonical constructor"
    assert not any(isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
                   for node in ast.walk(function)), "a second raw tuple constructor returned"


def test_judge_order_guard_remains_exact_and_prints_both_float_reprs():
    expected = ("VERIFY", 58, 1.35, "e5_01@a", "E5", "fp")
    actual = ("VERIFY", 58, 1.350001, "e5_01@a", "E5", "fp")
    with pytest.raises(RuntimeError) as caught:
        judges._assert_results_in_order(
            [dict(cache_key=expected)], [dict(cache_key=actual)])
    message = str(caught.value)
    assert "r=1.35" in message and "r=1.350001" in message


A1_OK = """Shift: moderate
Form: semantic
Score_Influence: 6
Justification: B introduces imagery absent from A."""

A2_OK = """Task_Compliance: full
Semantic_Coherence: good
Stance_Stability: stable
Degeneracy: none
Score_Integrity: 9
Justification: Reads normally."""

B_OK = """Identified: no
Failure_Mode: vague
Justification: Never commits to a concept."""


def test_parsers_read_a_well_formed_block():
    a1 = judges.parse_e5(A1_OK)
    assert a1["score_influence"] == 6 and a1["shift"] == "moderate"
    a2 = judges.parse_s1(A2_OK)
    assert a2["score_integrity"] == 9 and a2["degeneracy"] == "none"
    b = judges.parse_d2(B_OK)
    assert b["identified"] is False and b["failure_mode"] == "vague"


def test_parse_b_returns_a_real_bool_not_a_truthy_string():
    """A string 'no' counted as truthy would make D2 read 1.00 everywhere."""
    assert judges.parse_d2(B_OK.replace("Identified: no", "Identified: yes"))["identified"] is True


@pytest.mark.parametrize("drop", ["Score_Influence", "Shift", "Form"])
def test_parse_a1_raises_on_a_missing_field(drop):
    """DEBUG LOG pattern 4: silent bugs cluster around anything with a default. A missing
    judge field must raise, never become 0.0 - a defaulted Score_Influence is an E5 of zero
    that looks like a measurement."""
    broken = "\n".join(l for l in A1_OK.splitlines() if not l.startswith(drop))
    with pytest.raises(Exception):
        judges.parse_e5(broken)


@pytest.mark.parametrize("drop", ["Score_Integrity", "Task_Compliance", "Degeneracy"])
def test_parse_a2_raises_on_a_missing_field(drop):
    broken = "\n".join(l for l in A2_OK.splitlines() if not l.startswith(drop))
    with pytest.raises(Exception):
        judges.parse_s1(broken)


def test_parse_b_raises_on_a_missing_field():
    broken = "\n".join(l for l in B_OK.splitlines() if not l.startswith("Identified"))
    with pytest.raises(Exception):
        judges.parse_d2(broken)


def test_parse_rejects_an_out_of_vocabulary_category():
    with pytest.raises(Exception):
        judges.parse_d2(B_OK.replace("Failure_Mode: vague", "Failure_Mode: confused"))


# =====================================================================================
# cheap - the objective degeneracy backstop
# =====================================================================================

def test_degenerate_fires_on_the_v1_failure_case():
    """Bug 27. L46 alpha=3 produced '## ## ## ##' repeated to the token limit; the judge scored
    it incoherence 0.00 and the cell was reported as a candidate operating point.

    A sanity score a classifier can talk round is not a sanity score.
    """
    assert cheap.degenerate("## " * 60)
    assert cheap.degenerate("the cat sat on the mat " * 8)


def test_degenerate_does_not_fire_on_ordinary_prose():
    text = ("Bread dough rises because yeast converts sugars into carbon dioxide, which is "
            "trapped by an elastic gluten network. Kneading develops that network, and "
            "temperature governs the rate of fermentation, so bakers chill the dough overnight "
            "to slow it down and deepen the flavour of the finished loaf.")
    assert not cheap.degenerate(text)


def test_measure_s2_is_one_minus_the_degenerate_fraction():
    out = cheap.measure_S2(["## " * 60, "A perfectly ordinary sentence about cartography."])
    assert out["s2"] == pytest.approx(0.5)
    assert out["s2_count"] == 1 and out["s2_n"] == 2
    assert 0.0 < out["s2_ci_low"] < out["s2"] < out["s2_ci_high"] < 1.0


def test_wilson_interval_does_not_collapse_at_zero_of_25():
    """Task 17's motivating failure: binomial SE says 0 +/- 0 at this endpoint."""
    low, high = cheap.wilson_interval(0, 25)
    assert low == pytest.approx(0.0)
    assert high > 0.10
    assert high - low > 0.10


@pytest.mark.parametrize("successes,n", [(1, 1), (25, 25)])
def test_wilson_all_success_endpoints_remain_non_degenerate(successes, n):
    low, high = cheap.wilson_interval(successes, n)
    assert 0.0 <= low < high == pytest.approx(1.0)


def test_e5_is_a_mean_with_se_and_never_a_wilson_rate():
    """Task 17: a 0-10 judge score is a mean, not a Bernoulli success fraction."""
    source = (Path(__file__).resolve().parent.parent / "expensive.py").read_text(encoding="utf-8")
    body = source[source.index("def _score_e5("):source.index("def measure_E5(")]
    assert "model.mean_se(scores)" in body
    assert "wilson_interval" not in body
    assert "e5_se" in body and "e5_ci_low" not in body


def test_phase0_null_gate_trips_e5_and_s1_judge_faults():
    bad_e5 = dict(e5=dict(passed=False), s1=dict(judge_fault=False),
                  d2=dict(d2_null=0.0))
    bad_s1 = dict(e5=dict(passed=True), s1=dict(judge_fault=True),
                  d2=dict(d2_null=0.0))
    assert phases._judge_null_failures(bad_e5) == ["E5 judge null"]
    assert phases._judge_null_failures(bad_s1) == ["S1 judge null (S1/S2 disagreement)"]


def test_s1_low_with_low_s2_is_model_behaviour_not_a_gate_failure():
    nulls = dict(e5=dict(passed=True),
                 s1=dict(s1_low=True, s2_fine=False, judge_fault=False),
                 d2=dict(d2_null=0.0))
    assert phases._judge_null_failures(nulls) == []


@pytest.mark.parametrize("d2_null", [0.0, 0.04, 0.5, 1.0])
def test_d2_null_never_stops_phase0(d2_null):
    nulls = dict(e5=dict(passed=True), s1=dict(judge_fault=False),
                 d2=dict(d2_null=d2_null, report_only=True))
    assert phases._judge_null_failures(nulls) == []


def test_d2_null_scoring_persists_every_transcript(monkeypatch):
    exp = pytest.importorskip("m2.expensive", reason="imports torch at module scope")
    landed = []
    monkeypatch.setattr(exp, "_append_row", lambda name, row: landed.append((name, dict(row))))
    monkeypatch.setattr(
        exp, "_parse_or_fail",
        lambda *_args: (dict(identified=True, failure_mode="n/a", justification="stub"),
                        "stub", None),
    )
    units = [dict(trial=i, cache_key=("CAL_NULL_D2", i), response=f"response {i}",
                  payload=f"payload {i}") for i in range(1, 4)]
    out = exp._score_d2(units, [dict()] * 3,
                        dict(phase="CAL_NULL_D2", layer=None, r=None, alpha=None,
                             vec_fingerprint="none", measure="D2_NULL"))
    transcripts = [row for name, row in landed if name == exp.D2_FILE]
    assert out["d2"] == 1.0
    assert len(transcripts) == len(units)
    assert all(row["measure"] == "D2_NULL" for row in transcripts)


# =====================================================================================
# phases - the selection rule
# =====================================================================================

def _cell(layer, r, e5, d2, s4, qualifies=None, usable=True):
    cfg = config.CONFIG
    if qualifies is None:
        qualifies = (s4 >= cfg["S4_MIN"] and e5 >= cfg["E5_FLOOR"] and d2 <= cfg["D2_MAX"])
    return dict(layer=layer, r=r, e5=e5, d2=d2, s4=s4, s1=s4, s2=1.0, s3=s4,
                usable=usable, qualifies=bool(qualifies))


def test_selection_is_argmax_e5_not_lowest_d2():
    """Spec 7.1, in the spec's own words: a cell with D2 = 0.00 and E5 = 3 does not beat a
    cell with D2 = 0.18 and E5 = 8. Both satisfy the constraint, and the second is more
    effective. Selecting on D2 instead would systematically pick the cells where nothing is
    happening."""
    quiet = _cell(31, 0.15, e5=3.0, d2=0.00, s4=0.95)
    strong = _cell(46, 0.30, e5=8.0, d2=0.18, s4=0.80)
    out = phases.select_operating_point([quiet, strong])
    assert out["found"] is True
    assert out["winner"]["layer"] == 46


def test_selection_excludes_cells_breaching_any_constraint():
    cfg = config.CONFIG
    over_d2 = _cell(46, 0.30, e5=9.5, d2=float(cfg["D2_MAX"]) + 0.05, s4=0.95)
    under_s4 = _cell(45, 0.30, e5=9.9, d2=0.01, s4=float(cfg["S4_MIN"]) - 0.05)
    ok = _cell(37, 0.15, e5=5.0, d2=0.10, s4=0.90)
    out = phases.select_operating_point([over_d2, under_s4, ok])
    assert out["winner"]["layer"] == 37


def test_ties_inside_the_band_break_by_lower_d2_then_higher_sanity():
    band = float(config.CONFIG["E5_TIE_BAND"])
    a = _cell(40, 0.20, e5=8.0, d2=0.15, s4=0.90)
    b = _cell(41, 0.20, e5=8.0 - band / 2, d2=0.05, s4=0.85)
    out = phases.select_operating_point([a, b])
    assert out["winner"]["layer"] == 41, "inside the tie band, lower D2 wins"

    c = _cell(42, 0.20, e5=8.0, d2=0.05, s4=0.95)
    out2 = phases.select_operating_point([b, c])
    assert out2["winner"]["layer"] == 42, "equal D2 inside the band, higher S4 wins"


def test_no_qualifying_cell_is_a_result_not_an_error():
    """'No operating point exists at these constraints' is a real finding; spec 9.3's ladder
    is what separates it from 'the vector is dead'."""
    out = phases.select_operating_point([_cell(46, 0.30, e5=9.0, d2=0.90, s4=0.95)])
    assert out["found"] is False and out["winner"] is None
    assert "escalation" in out["reason"].lower()


def test_selection_ignores_the_residual():
    """Spec 7.2: the residual decides what gets MEASURED, never what gets CHOSEN. Ranking the
    final answer by it would select for cells where the cheap proxies disagreed, which is a
    property of the proxies."""
    strong = _cell(46, 0.30, e5=8.0, d2=0.18, s4=0.80); strong["resid"] = -0.9
    quiet = _cell(31, 0.15, e5=5.0, d2=0.02, s4=0.95); quiet["resid"] = 9.9
    assert phases.select_operating_point([strong, quiet])["winner"]["layer"] == 46


def test_unverified_rows_cannot_win():
    """A scan row has no verdict. Treating its absence as False is right by luck; treating the
    row as a candidate would let a cell with no D2 at all be reported as the answer."""
    scan_only = dict(layer=50, r=0.30, reach=0.9)
    real = _cell(37, 0.15, e5=5.0, d2=0.10, s4=0.90)
    out = phases.select_operating_point([scan_only, real])
    assert out["winner"]["layer"] == 37 and out["n_considered"] == 1


def test_operating_point_and_frontier_keep_rate_intervals_and_e5_se(run_ctx, tmp_path):
    """Task 17: visible result surfaces carry endpoints/n without turning E5 into a rate."""
    ctx = run_ctx(run_dir=tmp_path)
    ctx.mw = SimpleNamespace()  # write_operating_point only needs the loaded-run invariant
    row = _cell(37, 0.15, e5=7.0, d2=0.08, s4=0.90)
    row.update(
        phase="VERIFY", alpha=1.2, e5_se=0.35, e5_min=6.0,
        d2_se=0.05, d2_ci_low=0.02, d2_ci_high=0.24, n_d2=25,
        d2_null=0.04, d2_null_ci_low=0.01, d2_null_ci_high=0.20, d2_null_n=25,
        s2_ci_low=0.76, s2_ci_high=1.0, s2_n=12,
        s3_acc=0.80, s3_acc_ci_low=0.68, s3_acc_ci_high=0.88, s3_n=57,
        s2_forced=1.0, eligibility_tier=1,
        eligibility_reach_count=2, eligibility_reach_n=12,
        s4_term="S1", d4={}, d4_reading="retrieval", resid=0.0,
    )
    selection = phases.select_operating_point([row])
    path = phases.write_operating_point(selection, [row])
    payload = json.loads(path.read_text(encoding="utf-8"))
    screening = payload["screening"]
    front = payload["frontier"][0]
    assert screening["d2_ci_low"] == 0.02 and screening["d2_ci_high"] == 0.24
    assert screening["n_d2"] == 25
    assert screening["d2_null_ci_low"] == 0.01 and screening["d2_null_n"] == 25
    assert front["s2_ci_low"] == 0.76 and front["s2_n"] == 12
    assert front["s3_acc_ci_high"] == 0.88 and front["s3_n"] == 57
    assert screening["e5_se"] == 0.35
    assert screening["s2_forced"] == 1.0
    assert payload["operating_point"]["eligibility_tier"] == 1
    assert payload["operating_point"]["eligibility_reach_count"] == 2
    assert "e5_ci_low" not in screening and "e5_ci_low" not in front


# =====================================================================================
# gate 1 - live objective anchors and a role-blind operator packet
# =====================================================================================

def _gate1_scan_row(layer, r, *, reach, d3, d3_rate, d3_rank):
    return dict(
        phase="SCAN", layer=layer, r=r, alpha=r * 2, vec_fingerprint=f"fp-{layer}-{r}",
        reachable=True, reach=reach, reach_n=12,
        e6_mass_median=max(reach / 10, 1e-8), e6_rank_med=max(1, 100 - reach * 100),
        d3=d3, d3_rate=d3_rate, d3_rate_n=5, d3_rank_med=d3_rank,
    )


def test_gate1_selects_a_probability_and_rank_reversal_from_this_scan(run_ctx):
    run_ctx(concept="Garlic")
    rows = [
        _gate1_scan_row(31, 0.30, reach=0.75, d3=0.002, d3_rate=0.0, d3_rank=91),
        _gate1_scan_row(32, 0.30, reach=0.00, d3=0.42, d3_rate=0.8, d3_rank=2),
        _gate1_scan_row(33, 0.30, reach=0.00, d3=0.01, d3_rate=0.2, d3_rank=30),
    ]
    chosen = gates._gate1_select_anchors(rows)
    assert chosen["high"]["layer"] == 31 and chosen["low"]["layer"] == 32
    assert chosen["high"]["reach"] >= config.CONFIG["E6_FLOOR"]
    assert chosen["high"]["d3_rate"] == 0
    assert chosen["low"]["reach"] == 0 and chosen["low"]["d3_rate"] > 0
    assert chosen["low"]["d3"] > chosen["high"]["d3"]
    assert chosen["low"]["d3_rank_med"] < chosen["high"]["d3_rank_med"]


def test_gate1_reversed_judge_scores_fail_the_separation():
    """Task 04 acceptance: a word-counter ranks LOW over HIGH and must trip the gate."""
    reading = gates._gate1_separation([1, 2, 1], [8, 9, 8])
    assert reading["margin"] < 0
    assert reading["passed"] is False


def test_gate1_packet_is_shuffled_and_never_discloses_anchor_role():
    base = dict(concept="Garlic", r=0.3, alpha=0.6, vec_fingerprint="fp",
                reach=0.5, reach_n=12, e6_mass_median=0.02, e6_rank_med=8,
                d3=0.001, d3_rate=0.0, d3_rate_n=5, d3_rank_med=80,
                rationale="test")
    anchors = dict(high=dict(base, role="HIGH", layer=31),
                   low=dict(base, role="LOW", layer=32))
    pairs = [dict(prompt_id=f"p{i}", prompt=f"prompt {i}",
                  response_unsteered=f"A{i}", response_steered=f"B{i}")
             for i in range(12)]
    config_key = dict(key="judge-key", judge_model="judge/model",
                      judge_prompt_version="sha256:prompt")
    rows = gates._gate1_build_label_rows(
        anchors, {"high": {"pairs": pairs}, "low": {"pairs": pairs}}, config_key)
    assert len(rows) == 24 and all(row["hand_label"] is None for row in rows)
    assert all("role" not in key.lower() for row in rows for key in row)
    layers = [row["layer"] for row in rows]
    assert sum(a != b for a, b in zip(layers, layers[1:])) >= 4


def test_gate1_refuses_labels_from_a_changed_judge_configuration():
    old = dict(key="old", judge_model="judge/v1", judge_prompt_version="sha256:old")
    new = dict(key="new", judge_model="judge/v2", judge_prompt_version="sha256:new")
    metadata = {"judge_configuration": old}
    rows = [dict(judge_config_key="old", judge_model="judge/v1",
                 judge_prompt_version="sha256:old")]
    refusal = gates._gate1_config_refusal(metadata, rows, new)
    assert refusal is not None
    assert "stored" in refusal and "current" in refusal


# =====================================================================================
# gate 4 - the aggregation rule, not the deliberately failed S3 endpoint
# =====================================================================================

def test_gate4_all_terms_low_anchor_cannot_pass():
    """Task 03: a uniformly destroyed cell says nothing about why min is needed."""
    reading = gates._gate4_anchor_reading({"s1": 0.40, "s2": 0.50, "s3": 0.60}, 0.70)
    assert reading["min_rejects"] is True
    assert reading["terms_disagree"] is False
    assert reading["passed"] is False


def test_gate4_min_and_mean_both_reject_does_not_demonstrate_min():
    """One healthy term is not enough if an ordinary mean would reject the cell too."""
    reading = gates._gate4_anchor_reading({"s1": 0.95, "s2": 0.45, "s3": 0.45}, 0.70)
    assert reading["terms_disagree"] is True
    assert reading["min_rejects"] is True
    assert reading["mean_accepts"] is False
    assert reading["demonstrates_min_necessary"] is False


def test_gate4_passes_only_when_min_rejects_and_mean_accepts_disagreement():
    reading = gates._gate4_anchor_reading({"s1": 0.95, "s2": 0.95, "s3": 0.60}, 0.70)
    assert reading["below"] == ["s3"]
    assert reading["comfortably_above"] == ["s1", "s2"]
    assert reading["s4_min"] < reading["threshold"] <= reading["s4_mean"]
    assert reading["passed"] is True


def test_gate4_anchor_search_never_escalates_dose():
    """If hi is uniform damage, every permitted retry moves toward lo and never upward."""
    doses = gates._gate4_anchor_doses(dict(boundary_lo=0.72, boundary_hi=0.75))
    assert doses[0] == pytest.approx(0.75)
    assert doses[-1] == pytest.approx(0.72)
    assert all(next_dose <= dose for dose, next_dose in zip(doses, doses[1:]))


# =====================================================================================
# runio - the export gate and the delivery order
# =====================================================================================

def test_export_allows_transcripts_for_a_benign_concept():
    ok, why = runio.transcripts_allowed(config.BENIGN_CONCEPTS[0], config.CONFIG)
    assert ok, why


def test_export_refuses_transcripts_for_a_harmful_concept_without_an_override():
    """Spec 14.3. Those transcripts are what a refusal-ablated model said with `weapon`
    injected - the exact artifact CLAUDE.md hard rule 3 names."""
    ok, why = runio.transcripts_allowed("weapon", config.CONFIG)
    assert not ok
    assert "override" in why.lower()


def test_export_override_is_a_call_site_argument_not_a_config_key():
    """A config key, an environment variable or a module global would all be inheritable: the
    batch driver would carry a True set for the benign arm straight into the harmful arm when
    the concept list changed."""
    ok, _ = runio.transcripts_allowed("weapon", config.CONFIG,
                                      EXPORT_TRANSCRIPTS_OVERRIDE=True)
    assert ok
    sig = runio.transcripts_allowed.__kwdefaults__ or {}
    assert "EXPORT_TRANSCRIPTS_OVERRIDE" in sig, "must be keyword-only at the call site"
    assert "EXPORT_TRANSCRIPTS_OVERRIDE" not in config.CONFIG


def test_export_deny_list_covers_vectors_and_every_weight_extension():
    """CLAUDE.md hard rules 1 and 3: vectors are reusable attack artifacts and regenerate from
    a published config in minutes. Regeneration is the backup."""
    joined = " ".join(runio.EXPORT_DENY)
    for token in ("vectors", ".pt", ".safetensors", ".npy"):
        assert token in joined


def test_deliver_then_wipe_keeps_the_folder_when_the_send_fails(tmp_path, monkeypatch, run_ctx):
    """v1 archived each concept, wiped the loose folder, then attempted the Telegram send.
    When Telegram was down the per-concept results were unrecoverable. That cost Wrists and
    Wonder. A delivery failure must never destroy data."""
    run_dir = tmp_path / "irony_abc123"
    (run_dir / "measures").mkdir(parents=True)
    (run_dir / "measures" / "scan.jsonl").write_text('{"layer": 37}\n', encoding="utf-8")
    zip_path = tmp_path / "irony_abc123.zip"
    zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    ctx = run_ctx(run_dir=run_dir)
    ctx.config["config_hash"] = "abc123"

    class DeadNotifier:
        def send_file(self, *a, **k):
            raise RuntimeError("telegram unreachable")
        def send(self, *a, **k):
            return False

    monkeypatch.setattr(runio, "manifest_path",
                        lambda: tmp_path / "delivery_manifest.json", raising=False)
    delivered = runio.deliver_then_wipe(zip_path, DeadNotifier(), wipe=True, run_dir=run_dir)
    assert delivered is False
    assert run_dir.exists(), "the loose folder must survive a failed delivery"
    assert (run_dir / "measures" / "scan.jsonl").exists()


# =====================================================================================
# monitor - the verdicts that tell the operator what to do
# =====================================================================================

def test_phase_priors_cover_every_phase_and_price_verify_highest():
    """Spec 14.5: per-unit cost spans three orders of magnitude, from a ~2 s scan cell to a
    ~50 s verification cell. A naive units-done/units-total ETA would be badly wrong for most
    of an M2 run."""
    prior = monitor.PHASE_SECONDS_PRIOR
    for name in ("CAL", "SCAN", "SHORTLIST", "BISECT", "VERIFY", "CONFIRM", "CONTROLS"):
        assert name in prior
    # A verification cell generates and judges; a scan cell is forward passes only. The
    # earlier form of this test asserted VERIFY > SCAN * 10, which was a guessed ratio, not an
    # invariant - measurement put SCAN at 13 s against VERIFY's 50 s prior and the assertion
    # failed on correct data. Assert the ordering that is actually load-bearing.
    assert prior["VERIFY"] > prior["SCAN"]
    assert prior["SHORTLIST"] == pytest.approx(13.0)  # seconds per measured D2 selection cell
    assert all(v >= 0.0 for v in prior.values())
    assert monitor.PHASE_UNITS_PRIOR["SHORTLIST"] == config.CONFIG["D2_SELECT_MAX"]
    assert monitor.PHASE_UNITS_PRIOR["BISECT"] == 11  # candidates, incl. audit tier
    assert monitor.PHASE_UNITS_PRIOR["VERIFY"] == 11  # cells, incl. audit tier


def test_classify_exc_returns_a_label_never_the_message():
    """Spec 14.3 channel 1. An API error can quote its request payload back at you, and under
    M2 that payload is a steered generation or a judge prompt containing one - arriving
    unbidden, in a message you did not choose to send."""
    secret = "the model said something about weapon under injection"
    label = monitor.classify_exc(RuntimeError(secret))
    assert secret not in label
    assert label


def test_classify_exc_falls_back_to_the_class_name():
    """Anything unmatched degrades to the exception class alone, which is safe by
    construction - a class name is code, not data."""
    class WeirdFailure(Exception):
        pass
    assert "WeirdFailure" in monitor.classify_exc(WeirdFailure("payload text here"))


# =====================================================================================
# gates - the public surface assertion
# =====================================================================================

def test_public_surface_check_covers_the_importable_modules():
    """Pattern 8: 'the cell ran without error' is not evidence the cell did anything. Bug 24's
    pipeline cell defined nothing and raised nothing."""
    report = gates.check_public_surface()
    assert report["checked"] > 0, "nothing was checked - the report is vacuous"
    # `missing` is a flat list of "module.name". Modules that need torch cannot be imported
    # here, so they land in `unimportable` and are excluded rather than counted as missing.
    torch_only = {"vectors", "expensive", "controls", "multilayer", "steer"}
    assert set(report["unimportable"]) <= torch_only, report["unimportable"]
    real = [n for n in report["missing"] if n.split(".")[0] not in torch_only]
    assert not real, f"CONTRACT names missing from the package: {real}"


# =====================================================================================
# run - the CLI surface
# =====================================================================================

def test_override_rejects_a_key_that_does_not_exist():
    """`--set D2MAX=0.25` must not silently create a constant nothing reads. The run would
    then use the default D2_MAX and report a constraint nobody chose."""
    from m2 import run as m2run
    cfg = dict(config.CONFIG)
    with pytest.raises(SystemExit) as caught:
        m2run.apply_overrides(cfg, ["D2MAX=0.25"])
    assert "D2_MAX" in str(caught.value), "the error should name the key it meant"


def test_override_parses_a_dose_tuple_as_numbers():
    """SCAN_DOSES=0.15,0.30 must become floats. A stringified dose would be compared against
    numbers and never match anything, silently."""
    from m2 import run as m2run
    cfg = dict(config.CONFIG)
    applied = m2run.apply_overrides(cfg, ["SCAN_DOSES=0.15,0.30,0.45", "D2_MAX=0.25"])
    assert applied["SCAN_DOSES"] == (0.15, 0.30, 0.45)
    assert applied["D2_MAX"] == 0.25
    assert all(isinstance(x, float) for x in cfg["SCAN_DOSES"])


def test_override_changes_the_config_hash():
    """So an overridden run gets its own folder and cannot resume into the earlier grid."""
    from m2 import run as m2run
    before = config.config_hash({k: v for k, v in config.CONFIG.items() if k != "config_hash"})
    cfg = dict(config.CONFIG)
    m2run.apply_overrides(cfg, ["D2_MAX=0.25"])
    after = config.config_hash({k: v for k, v in cfg.items() if k != "config_hash"})
    assert before != after


def test_every_tier_knob_is_in_config_and_reachable_through_set():
    from m2 import run as m2run
    cfg = dict(config.CONFIG)
    applied = m2run.apply_overrides(cfg, [
        "SHORTLIST_TIER_SIZE=4",
        "SHORTLIST_AUDIT_TIERS=2",
        "SHORTLIST_MAX_TIER=null",
        "SHORTLIST_EXHAUSTIVE=true",
    ])
    assert set(applied) == {
        "SHORTLIST_TIER_SIZE", "SHORTLIST_AUDIT_TIERS", "SHORTLIST_MAX_TIER",
        "SHORTLIST_EXHAUSTIVE"}
    assert cfg["SHORTLIST_TIER_SIZE"] == 4 and cfg["SHORTLIST_AUDIT_TIERS"] == 2
    assert cfg["SHORTLIST_MAX_TIER"] is None
    assert cfg["SHORTLIST_EXHAUSTIVE"] is True


def test_exhaustive_flag_and_cost_estimate_are_explicit_before_measurement():
    from m2 import run as m2run
    args = m2run._parser().parse_args(["--concepts", "Garlic", "--exhaustive"])
    estimate = m2run.exhaustive_cost_estimate(49)
    assert args.exhaustive is True
    assert estimate["bisection_candidates"] == estimate["verification_cells"] == 49
    assert estimate["judge_calls"] == 49 * 49
    assert estimate["seconds"] == pytest.approx(
        49 * (monitor.PHASE_SECONDS_PRIOR["BISECT"]
              + monitor.PHASE_SECONDS_PRIOR["VERIFY"]))


def test_archive_without_loose_folder_prints_the_restore_command(
        tmp_path, monkeypatch, capsys):
    """Task 09: an archive silently skipped a run even though row-level resume needed extraction."""
    from m2 import run as m2run

    monkeypatch.setenv("M2_RUNS_DIR", str(tmp_path))
    cfg = dict(config.CONFIG)
    cfg["concept"] = "Garlic"
    run_dir = config.run_dir_for("Garlic", cfg)
    archive = runio.archive_path_for(run_dir)
    archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    assert not run_dir.exists()

    notices = m2run.print_archive_restore_notices(["Garlic"], cfg)
    output = capsys.readouterr().out

    assert len(notices) == 1
    assert "ARCHIVED RUN HAS NO LOOSE RESUME FOLDER" in output
    assert "python -m zipfile -e" in output
    assert str(archive) in output and str(run_dir) in output
    assert "mv" in output and ".zip.restored" in output


def test_run_cli_checks_for_archive_only_resume_before_public_surface(
        monkeypatch):
    """The restore helper must be wired into a real run, not merely pass in isolation."""
    from m2 import run as m2run

    called: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(m2run, "check_environment", lambda strict=True: {})
    monkeypatch.setattr(
        m2run, "print_archive_restore_notices",
        lambda concepts, cfg: called.append((list(concepts), cfg)) or [])
    monkeypatch.setattr(
        gates, "check_public_surface",
        lambda: dict(missing=["stop-after-archive-check"], unimportable={}, checked=0))

    result = m2run.main(["--concepts", "Garlic", "--no-notify-test"])

    assert result == m2run.EXIT_CONFIG
    assert called and called[0][0] == ["Garlic"]


def test_required_credentials_are_the_two_the_run_cannot_proceed_without():
    """Under nohup there is no TTY, so a getpass prompt would read EOF and the run would start
    with no judge key and fail forty minutes later in Phase 4."""
    from m2 import run as m2run
    assert set(m2run.REQUIRED_ENV) == {"HF_TOKEN", "OPENROUTER_API_KEY"}
    assert "HEALTHCHECK_URL" in m2run.OPTIONAL_ENV


def test_gate11_preflight_constructs_the_repo_judge(monkeypatch):
    """Gate 11 constructs and calls the upstream rubric through OpenRouter only."""
    constructed = {}
    called = []
    ensured = []

    class RepoJudge:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            constructed["base_url"] = os.environ.get("OPENAI_BASE_URL")

    def fake_batch(*_args, **_kwargs):
        called.append(os.environ.get("OPENAI_BASE_URL"))
        return ["scored"]

    fake_eval = SimpleNamespace(batch_evaluate=fake_batch, LLMJudge=RepoJudge)
    fake_nest = SimpleNamespace(apply=lambda: None)
    fake_model = SimpleNamespace(ensure_repo_path=lambda: ensured.append(True))
    original_mod = gates._mod
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://caller.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "eval_utils", fake_eval)
    monkeypatch.setitem(sys.modules, "nest_asyncio", fake_nest)
    monkeypatch.setattr(gates, "_mod",
                        lambda name: fake_model if name == "model" else original_mod(name))

    reading = gates._preflight_repo_judge()
    assert ensured and constructed, "the check imported eval_utils but did not construct LLMJudge"
    assert reading["passed"] is True
    assert constructed["api_key"] == "sk-or-test"
    assert constructed["base_url"] == "https://openrouter.ai/api/v1"
    assert os.environ["OPENAI_BASE_URL"] == "https://caller.example/v1"

    batch, _judge = gates._construct_repo_judge()
    assert batch(None) == ["scored"]
    assert called == ["https://openrouter.ai/api/v1"]
    assert os.environ["OPENAI_BASE_URL"] == "https://caller.example/v1"


def test_gate11_openrouter_key_guard_actually_trips(monkeypatch):
    """A missing OpenRouter key must fail before the upstream constructor can hide it."""
    constructed = []

    class ShouldNotConstruct:
        def __init__(self, **_kwargs):
            constructed.append(True)

    fake_eval = SimpleNamespace(batch_evaluate=lambda *_a, **_k: None,
                                LLMJudge=ShouldNotConstruct)
    fake_nest = SimpleNamespace(apply=lambda: None)
    fake_model = SimpleNamespace(ensure_repo_path=lambda: None)
    original_mod = gates._mod
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "eval_utils", fake_eval)
    monkeypatch.setitem(sys.modules, "nest_asyncio", fake_nest)
    monkeypatch.setattr(gates, "_mod",
                        lambda name: fake_model if name == "model" else original_mod(name))

    reading = gates._preflight_repo_judge()
    assert reading["passed"] is False
    assert "OPENROUTER_API_KEY" in reading["detail"]
    assert not constructed, "the credential guard did not stop construction"


def test_real_preflight_path_runs_the_repo_judge_constructor_check():
    import inspect
    from m2 import run as m2run
    assert "gates._preflight_repo_judge()" in inspect.getsource(m2run.main)


def test_concepts_file_ignores_comments_and_blanks(tmp_path):
    from m2 import run as m2run
    path = tmp_path / "concepts.txt"
    path.write_text("Irony\n\n# a comment\nSilk   # trailing\nIrony\n", encoding="utf-8")
    args = m2run._parser().parse_args(["--concepts-file", str(path)])
    assert m2run._read_concepts(args) == ["Irony", "Silk"], "and de-duplicated, order kept"


def test_run_status_accepts_the_arguments_the_driver_passes(tmp_path):
    """The driver built the board positionally against a signature that takes keywords, so
    `Path(a_dict)` raised TypeError, the driver's except degraded to a board-free run, and a
    whole concept ran with no progress board, no ETA and no stall detection - announced by one
    WARN line. Graceful degradation around a wrong call signature is how a defect survives a
    run, so the call is pinned here instead.
    """
    from m2 import driver
    board = monitor.RunStatus(driver.PHASE_ORDER,
                              path=tmp_path / "status.txt",
                              priors=monitor.PHASE_SECONDS_PRIOR)
    assert board.order == list(driver.PHASE_ORDER)
    assert set(board.priors) == set(driver.PHASE_ORDER)


def test_every_driver_phase_has_a_seconds_prior():
    """RunStatus hard-indexes the prior per phase and raises on a missing one, so a phase the
    driver runs but monitor has not priced would kill the board at construction."""
    from m2 import driver
    missing = [p for p in driver.PHASE_ORDER if p not in monitor.PHASE_SECONDS_PRIOR]
    assert not missing, f"unpriced phases: {missing}"


def test_s4_is_a_minimum_not_a_mean():
    """Spec 7. S1-S3 are three different ways to be unusable; passing one must not compensate
    for failing another. A mean of (1.0, 1.0, 0.1) is 0.70 and would pass S4_MIN."""
    s1, s2, s3 = 1.0, 1.0, 0.10
    assert min(s1, s2, s3) < float(config.CONFIG["S4_MIN"])
    assert (s1 + s2 + s3) / 3 >= float(config.CONFIG["S4_MIN"]), (
        "if this ever fails the example stopped demonstrating the point")


def test_setup_detects_a_package_that_carries_no_version():
    """`nest_asyncio` never defined `__version__`, so a probe of `print(pkg.__version__)`
    raised AttributeError on a good install and reported it missing. `--repair` then
    pip-installed it, printed "extras ok", and the re-check called it missing again - an
    unfixable FIX item that makes READY unreachable. Probe by import, version optional.
    """
    from m2 import setup as m2setup
    assert m2setup._version("pathlib") is not None, "importable, no __version__ - still present"
    assert m2setup._version("m2_no_such_module_xyz") is None, "genuinely absent reads absent"


def test_setup_without_repair_installs_missing_packages_and_rechecks(monkeypatch, capsys):
    """The no-flag CLI must execute the package repair, not merely expose a helper."""
    from m2 import setup as m2setup

    installed = []
    first = m2setup.Report()
    first.add("python packages", m2setup.FIX, "missing: torch, transformers",
              repair=lambda: installed.append(True) or "extras ok")
    ready = m2setup.Report()
    reports = iter((first, ready))
    monkeypatch.setattr(m2setup, "diagnose", lambda: next(reports))

    assert m2setup.main([]) == 0
    assert installed == [True]
    assert "INSTALLING MISSING PYTHON PACKAGES" in capsys.readouterr().out


def test_automatic_package_install_does_not_run_other_repairs(monkeypatch):
    """Automatic means packages only; branch/repo/data changes retain --repair."""
    from m2 import setup as m2setup

    called = []
    first = m2setup.Report()
    first.add("python packages", m2setup.FIX, "missing: pytest",
              repair=lambda: called.append("packages") or "extras ok")
    first.add("project repo", m2setup.FIX, "behind",
              repair=lambda: called.append("repo") or "pulled")
    monkeypatch.setattr(m2setup, "diagnose", lambda: m2setup.Report())

    assert m2setup.install_missing_packages(first, verbose=False).ready()
    assert called == ["packages"]


def test_failed_automatic_package_install_stays_nonready_and_does_not_loop(
        monkeypatch, capsys):
    """A failed pip attempt must be a visible failure, never assumed success or retried forever."""
    from m2 import setup as m2setup

    attempts = []

    def missing_report():
        rep = m2setup.Report()
        rep.add("python packages", m2setup.FIX, "missing: transformers",
                repair=lambda: attempts.append(True) or "requirements.txt FAILED")
        return rep

    reports = iter((missing_report(), missing_report()))
    monkeypatch.setattr(m2setup, "diagnose", lambda: next(reports))

    assert m2setup.main([]) == 1
    assert attempts == [True]
    output = capsys.readouterr().out
    assert "requirements.txt FAILED" in output
    assert "installation did not complete" in output


def test_run_setup_path_also_installs_packages_without_repair(monkeypatch):
    """`m2.run --setup` is a documented alias and must not preserve the old manual path."""
    from m2 import run as m2run
    from m2 import setup as m2setup

    seen = []
    initial = m2setup.Report()
    ready = m2setup.Report()
    monkeypatch.setattr(m2setup, "diagnose", lambda: initial)
    monkeypatch.setattr(
        m2setup, "install_missing_packages",
        lambda rep, verbose=True: seen.append((rep, verbose)) or ready)
    monkeypatch.setattr(m2setup, "render", lambda rep: seen.append(("render", rep)))

    assert m2run.main(["--setup"]) == m2run.EXIT_OK
    assert seen[0] == (initial, True)
    assert seen[1] == ("render", ready)


def test_phase0_runs_hook_liveness_itself():
    """R14 skips in a standalone --preflight because RUN.vecs is empty until extraction, and
    extraction IS Phase 0. That is only safe while Phase 0 runs liveness itself, before the
    first measurement. If this call ever moves out of phase0, bug 26 gets a way back in: an
    hour of forward-pass measures reading exactly 0.000 with no error.
    """
    import inspect
    from m2 import phases
    sig = inspect.signature(phases.phase0_calibrate)
    assert sig.parameters["run_liveness"].default is True, "liveness must be on by default"
    src = inspect.getsource(phases.phase0_calibrate)
    assert "hook_liveness()" in src, "phase 0 no longer checks the hook"
    assert src.index("hook_liveness()") < src.index("load_mmlu_items"), (
        "liveness must precede the first thing that measures anything")


# ---------------------------------------------------------------------------------------
# undefined global names
# ---------------------------------------------------------------------------------------

_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__",
            "__builtins__", "__path__", "__debug__"}


def _undefined_globals(path):
    """Names read as module globals inside some function, but bound nowhere at module level.

    `symtable` classes a name as global when the enclosing function neither binds it nor
    inherits it from a closure, so this is a real scope analysis and not a text search.
    """
    import builtins
    import symtable
    src = Path(path).read_text(encoding="utf-8")
    top = symtable.symtable(src, str(path), "exec")
    known = ({s.get_name() for s in top.get_symbols()} | set(dir(builtins)) | _DUNDERS)
    out = []

    def walk(tab, trail):
        for sym in tab.get_symbols():
            if sym.is_global() and sym.is_referenced() and sym.get_name() not in known:
                out.append(f"{trail}: {sym.get_name()}")
        for child in tab.get_children():
            walk(child, f"{trail}.{child.get_name()}")

    walk(top, Path(path).stem)
    return out


def test_the_undefined_name_detector_actually_detects(tmp_path):
    """A detector that never fires is worse than none - it reads as evidence."""
    good = tmp_path / "good.py"
    good.write_text("import time\ndef f():\n    return time.time()\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    return _now()\n", encoding="utf-8")
    assert _undefined_globals(good) == []
    assert _undefined_globals(bad) == ["bad.f: _now"]


def test_no_module_reads_a_global_that_does_not_exist():
    """The driver called a bare `_now()` that lives in runio, so writing the provenance row
    raised NameError. The block is wrapped in `except Exception` so a run could never afford
    to lose a row over it - which meant the defect logged one WARN line and cost the run the
    record of which GPU produced its numbers, the exact question a mid-run pod migration
    makes unanswerable. Broad excepts are right here; they just need this test behind them.
    """
    pkg = Path(__file__).resolve().parent.parent
    found = {}
    for path in sorted(pkg.glob("*.py")) + sorted((pkg / "tests").glob("*.py")):
        bad = _undefined_globals(path)
        if bad:
            found[path.name] = bad
    assert not found, f"undefined global names: {found}"


def _flat_grid():
    """A Garlic-shaped surface: dead below L53, alive only in the last handful of layers."""
    per_layer = {}
    for layer in range(13, 62):
        e6 = {53: 0.17, 57: 0.33, 58: 0.42, 60: 1.00, 61: 0.92}.get(layer, 0.0)
        # 1e-4 is what a dead layer really reads. D3 is a probability mass; it is never 0.
        d3 = {58: 0.027, 60: 0.999, 61: 1.000}.get(layer, 0.0001)
        per_layer[layer] = dict(layer=layer, e6=e6, d3=d3, s3=1.0, resid=-0.065,
                                e6_at_r=0.30, reach_by_dose={}, d3_by_dose={}, n_doses=2)
    return per_layer


def test_shortlist_never_widens_onto_a_dead_layer():
    """Widening maximises DISTANCE in E6, so with no live-pool guard it prefers the deadest
    layer on the grid. Garlic put L13/L14/L15 on the shortlist at reach 0.00 - each one ~50
    judge calls and ~2 minutes of Phase 4 to confirm a zero Phase 1 already reported. The
    first guard tested `d3 > 0.0`, which every layer passes because D3 is a probability mass.
    """
    from m2 import phases
    per_layer = _flat_grid()
    seed = [dict(layer=60, r=0.30, e6=1.00, d3=0.999, s3=1.0, resid=-0.132,
                 reach_by_dose={}, d3_by_dose={}, why=[], routes=["local_max"], merged_from=[])]
    sized, note = phases._size_shortlist(seed, per_layer, 8, 12)
    picked = [c["layer"] for c in sized]
    assert 13 not in picked and 14 not in picked and 15 not in picked, f"dead layers: {picked}"
    assert all(per_layer[l]["e6"] > 0.0 for l in picked), "every pick showed concept mass"
    assert len(sized) < 8, "an honestly short shortlist, not one padded to SHORTLIST_N"
    assert "below SHORTLIST_N" in (note or ""), "and it must say so"


def test_tier_residual_ordering_cannot_reintroduce_a_dead_layer():
    """Task 05: a negative fit residual is not evidence on an E6~=D3~=0 layer."""
    from m2 import phases
    per_layer = _flat_grid()
    per_layer[13]["resid"] = -9.0       # would rank first without the shared signal guard
    per_layer[57]["resid"] = -0.20
    per_layer[58]["resid"] = -0.50
    per_layer[60]["resid"] = 0.10
    ordered, dead = phases._order_rejected_layers(
        per_layer, [13, 57, 58, 60], "e6_residual_interleave", d3_base=0.0)
    assert 13 not in [row["layer"] for row in ordered]
    assert 13 in dead
    assert [row["layer"] for row in ordered[:3]] == [60, 58, 57]
    assert [row["tier_ordering"] for row in ordered[:3]] == [
        "e6_desc", "residual_asc", "e6_desc"]


def test_tier1_runs_even_after_tier0_finds_a_qualifier():
    from m2 import phases
    runs, reason = phases._should_execute_tier(
        1, has_qualifier=True, audit_tiers=1, max_tier=3, exhaustive=False)
    assert runs is True
    assert "mandatory" in reason


def test_tier_escalation_stops_at_the_configured_limit_and_says_so():
    from m2 import phases
    runs, reason = phases._should_execute_tier(
        4, has_qualifier=False, audit_tiers=1, max_tier=3, exhaustive=False)
    assert runs is False
    assert "SHORTLIST_MAX_TIER=3" in reason


def test_task21_tier_ordering_is_fixed_to_pareto_distance():
    from m2 import phases
    cfg = dict(config.CONFIG)
    assert "SHORTLIST_TIER_ORDER" not in cfg, "an obsolete knob must not change the new design"
    assert phases._tier_config(cfg)["tier_order"] == "pareto_distance"


def _fake_selection_d2(values=None, *, collapsed=()):
    values = {} if values is None else dict(values)
    collapsed = set(collapsed)

    def measure(row, n):
        key = (int(row["layer"]), round(float(row["r"]), 6))
        d2 = float(values.get(key, 0.0))
        hits = int(round(d2 * n))
        low, high = cheap.wilson_interval(hits, n)
        normal = ("This is a complete and ordinary response with enough distinct words to "
                  "remain structurally healthy throughout the forced identification trial.")
        looping = "garlic " * 60
        responses = [looping if key in collapsed else normal for _ in range(n)]
        return dict(
            d2=d2, d2_se=math.sqrt(d2 * (1.0 - d2) / n),
            d2_ci_low=low, d2_ci_high=high, n_d2=n, d2_identified=hits,
            d2_judge_errors=0, d2_judge_error_detail=[], d2_n_generated=n,
            responses=responses)
    return measure


def test_task26_task25_cells_cannot_reach_frontier_on_low_d3(capsys):
    fixture = Path(__file__).parent / "fixtures" / "garlic_shakedown_scan.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    wanted = {(57, 0.30), (58, 0.30), (59, 0.30), (52, 0.60)}
    rows = [row for row in rows if (row["layer"], round(row["r"], 2)) in wanted]
    measured = {key: 1.0 for key in wanted}
    result = phases.phase2_shortlist(
        rows, write=False, measure_cell=_fake_selection_d2(measured))

    assert result["detection_axis"] == "d2_measured"
    assert result["frontier"] == []
    assert result["candidates"] == []
    assert result["n_cells_d2_measured"] == 4
    assert all(row["kind"] == "d2_above_D2_MAX" for row in result["near_misses"])
    low_d3 = {(row["layer"], round(row["r"], 2)): row for row in result["near_misses"]}
    assert low_d3[(57, 0.30)]["d3"] < 0.01 and low_d3[(57, 0.30)]["d2"] == 1.0
    assert low_d3[(58, 0.30)]["d3"] < 0.03 and low_d3[(58, 0.30)]["d2"] == 1.0
    assert "tier 2" in capsys.readouterr().out


def test_task26_keeps_two_measured_tradeoff_doses_from_the_same_layer():
    rows = [
        dict(layer=10, r=0.30, reachable=True, reach=5 / 12, d3=0.10,
             reach_n=12, d3_rate=0.0, s3=0.95, alpha=1.0),
        dict(layer=10, r=0.60, reachable=True, reach=10 / 12, d3=0.60,
             reach_n=12, d3_rate=1.0, s3=0.95, alpha=2.0),
        dict(layer=11, r=0.30, reachable=True, reach=4 / 12, d3=0.30,
             reach_n=12, d3_rate=1.0, s3=0.95, alpha=1.0),
    ]
    d2 = {(10, 0.30): 0.0, (10, 0.60): 0.4, (11, 0.30): 0.2}
    result = phases.phase2_shortlist(
        rows, write=False, measure_cell=_fake_selection_d2(d2))
    frontier = {(row["layer"], row["r"]) for row in result["frontier"]}
    assert (10, 0.30) in frontier and (10, 0.60) in frontier
    assert (11, 0.30) not in frontier


def _task26_scan_row(layer, reach_count, *, r=0.30, d3=0.1, s3=0.95):
    return dict(layer=layer, r=r, reachable=True, reach=reach_count / 12,
                reach_n=12, d3=d3, d3_rate=0.0, d3_rank_med=2,
                s2=None, s2_n=None, s2_ci_low=None, s2_ci_high=None,
                s3=s3, alpha=1.0)


def test_task26_relaxes_by_reach_count_and_never_remeasures():
    rows = [_task26_scan_row(50, 3), _task26_scan_row(51, 2),
            _task26_scan_row(52, 1)]
    values = {(50, 0.30): 1.0, (51, 0.30): 0.2, (52, 0.30): 0.0}
    calls = []
    base = _fake_selection_d2(values)

    def measured(row, n):
        calls.append((row["layer"], row["r"]))
        return base(row, n)

    result = phases.phase2_shortlist(rows, write=False, measure_cell=measured)
    assert result["eligibility_tier"] == 1
    assert result["eligibility_reach_count"] == 2
    assert {(row["layer"], row["r"]) for row in result["frontier"]} == {(51, 0.30)}
    assert calls == [(50, 0.30), (51, 0.30)]
    assert any(row["layer"] == 52 and row["kind"] == "lower_reach_tier_not_needed"
               for row in result["near_misses"])


def test_task26_resume_reuses_paid_d2_and_persists_no_response_text(run_ctx, tmp_path):
    ctx = run_ctx(run_dir=tmp_path)
    ctx.mw = object()
    rows = [_task26_scan_row(50, 3)]
    first = phases.phase2_shortlist(
        rows, write=True, measure_cell=_fake_selection_d2({(50, 0.30): 0.2}))
    assert first["frontier"][0]["d2"] == 0.2

    def must_not_measure(_row, _n):
        raise AssertionError("a resumed selection cell was measured twice")

    resumed = phases.phase2_shortlist(rows, write=True, measure_cell=must_not_measure)
    assert resumed["frontier"][0]["d2"] == 0.2
    saved = (tmp_path / phases.SELECTION_D2_FILE).read_text(encoding="utf-8")
    assert '"responses"' not in saved, "selection resume stores scalars, not generations"


def test_task26_tier2_is_report_only_even_when_d2_is_low():
    rows = [_task26_scan_row(50, 3), _task26_scan_row(51, 2),
            _task26_scan_row(52, 1)]
    values = {(50, 0.30): 1.0, (51, 0.30): 1.0, (52, 0.30): 0.0}
    result = phases.phase2_shortlist(
        rows, write=False, measure_cell=_fake_selection_d2(values))
    assert result["eligibility_tier"] == 2
    assert result["eligibility_report_only"] is True
    assert result["frontier"] == [] and result["candidates"] == []
    tier2 = next(row for row in result["near_misses"] if row["layer"] == 52)
    assert tier2["kind"] == "report_only_reach_1_of_12"
    assert tier2["d2"] == 0.0


def test_task26_forced_s2_rejects_looping_without_becoming_canonical_s2():
    rows = [_task26_scan_row(58, 5, d3=0.02),
            _task26_scan_row(59, 6, d3=1.0)]
    values = {(58, 0.30): 0.0, (59, 0.30): 0.2}
    result = phases.phase2_shortlist(
        rows, write=False,
        measure_cell=_fake_selection_d2(values, collapsed={(58, 0.30)}))
    assert {(row["layer"], row["r"]) for row in result["frontier"]} == {(59, 0.30)}
    rejected = next(row for row in result["near_misses"] if row["layer"] == 58)
    assert rejected["kind"] == "s2_forced_below_S4_MIN"
    assert rejected["s2_forced"] == 0.0
    assert all("s2" not in row for row in result["candidates"])
    assert result["candidates"][0]["s2_forced"] == 1.0


def test_task26_cap_names_every_unmeasured_cell(monkeypatch, capsys):
    monkeypatch.setitem(config.CONFIG, "D2_SELECT_MAX", 2)
    rows = [_task26_scan_row(40 + i, count) for i, count in enumerate((3, 4, 8, 12))]
    result = phases.phase2_shortlist(
        rows, write=False, measure_cell=_fake_selection_d2())
    omitted = {(row["layer"], row["r"]) for row in result["d2_measurement_omissions"]}
    assert omitted == {(41, 0.30), (42, 0.30)}
    output = capsys.readouterr().out
    assert "DROPPED L41@0.300" in output and "DROPPED L42@0.300" in output
    assert "D2_SELECT_MAX=2" in output


def test_task26_reach_count_guard_actually_trips():
    with pytest.raises(ValueError, match="not k/12"):
        phases._reach_count(dict(layer=1, r=0.3, reach=0.20, reach_n=12))


@pytest.mark.parametrize(("key", "value", "message"), [
    ("D2_SELECT_REACH_COUNTS", (3, 1), "settled"),
    ("D2_SELECT_N", 0, "positive"),
    ("D2_SELECT_MAX", 1, "MAX>=2"),
])
def test_task26_config_guards_actually_trip(monkeypatch, key, value, message):
    monkeypatch.setitem(config.CONFIG, key, value)
    with pytest.raises(ValueError, match=message):
        phases.phase2_shortlist([], write=False, measure_cell=_fake_selection_d2())


def test_task21_phase3_maps_sanity_without_replacing_the_selected_dose(
        run_ctx, tmp_path, monkeypatch):
    ctx = run_ctx(run_dir=tmp_path)
    ctx.mw = object()

    def fake_probe(layer, r, cache):
        key = phases._cell_key(layer, r)
        if key not in cache:
            sane = float(r) < 0.40
            cache[key] = dict(layer=layer, r=float(r), reachable=True, alpha=float(r),
                              s3=0.95 if sane else 0.50,
                              reach=5 / 12 if sane else 0.0, reach_n=12, d3=0.05)
        return cache[key]

    monkeypatch.setattr(phases, "_probe", fake_probe)
    candidate = dict(layer=58, r=0.30, why=["frontier"], routes=["pareto_frontier"],
                     tier=0, tier_rank=1, tier_ordering="pareto_frontier",
                     tier_order_rank=1, tier_source_layer=58,
                     tier_source_cell=dict(layer=58, r=0.30))
    row = phases.phase3_bisect([candidate])[0]
    assert row["r"] == 0.30
    assert row["selected_r"] == 0.30
    assert row["boundary_hi"] != row["r"], "the boundary must remain metadata"
    assert row["has_window"] is True


def test_task24_real_scan_selects_the_confirmed_21_layer_band():
    fixture = Path(__file__).parent / "fixtures" / "garlic_shakedown_scan.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    report = phases._knee_band(
        rows,
        lower_r=config.CONFIG["SCAN_KNEE_GAP"][0],
        upper_r=config.CONFIG["SCAN_KNEE_GAP"][1],
        reach_delta_min=config.CONFIG["SCAN_KNEE_REACH_DELTA_MIN"],
        d3_delta_min=config.CONFIG["SCAN_KNEE_D3_DELTA_MIN"],
        s4_min=config.CONFIG["S4_MIN"])
    selected = [row["layer"] for row in report if row["selected"]]
    assert len(selected) == 21
    assert selected == [37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49,
                        50, 51, 52, 53, 54, 55, 56, 57, 58]
    assert config.CONFIG["SCAN_KNEE_DEPTH"] == 2


def test_task24_insane_midpoint_always_moves_down():
    direction, reason = phases._knee_direction(
        dict(layer=37, r=0.45, reachable=True, reach=1.0, d3=0.0, s3=0.50),
        reach_floor=0.20, s4_min=0.70, d3_low_max=0.10)
    assert direction == "down"
    assert "s3" in reason


def test_task24_flat_zero_layer_is_not_in_the_band():
    rows = [
        dict(layer=18, r=0.30, reachable=True, reach=0.0, d3=0.0, s3=0.95),
        dict(layer=18, r=0.60, reachable=True, reach=0.0, d3=0.0, s3=0.95),
    ]
    report = phases._knee_band(
        rows, lower_r=0.30, upper_r=0.60, reach_delta_min=0.20,
        d3_delta_min=0.30, s4_min=0.70)
    assert report[0]["selected"] is False
    assert "below" in report[0]["reason"]


def test_task24_knee_cells_join_scan_and_are_visible_to_phase2(
        run_ctx, tmp_path, monkeypatch, capsys):
    ctx = run_ctx(run_dir=tmp_path)
    ctx.mw = object()
    rows = [
        dict(phase="SCAN", layer=50, r=0.30, reachable=True, reach=0.0, d3=0.0,
             reach_n=12, d3_rate=0.0, s3=0.95, alpha=1.0),
        dict(phase="SCAN", layer=50, r=0.60, reachable=True, reach=0.50, d3=0.80,
             reach_n=12, d3_rate=1.0, s3=0.95, alpha=2.0),
    ]

    def fake_scan(layer, r):
        if r < 0.50:
            return dict(phase="SCAN", layer=layer, r=r, reachable=True, reach=0.25,
                        reach_n=12, d3=0.05, d3_rate=0.0, s3=0.95, alpha=r * 3)
        return dict(phase="SCAN", layer=layer, r=r, reachable=True, reach=0.50,
                    reach_n=12, d3=0.40, d3_rate=1.0, s3=0.95, alpha=r * 3)

    monkeypatch.setattr(cheap, "scan_cell", fake_scan)
    plans, ticks = [], []
    got = phases._phase1_knee_search(
        rows, on_cell=ticks.append, on_plan=plans.append, base_todo=0)
    knee = [row for row in got if row.get("scan_provenance") == "knee_search"]
    assert len(knee) == 2
    assert {row["knee_depth"] for row in knee} == {1, 2}
    assert all(row["knee_direction"] in {"up", "down"} for row in knee)
    assert len(ticks) == 2 and plans[-1] == 2
    output = capsys.readouterr().out
    assert "per level" in output and "depth: 2" in output

    shortlist = phases.phase2_shortlist(
        got, write=False, measure_cell=_fake_selection_d2())
    # The knee cell survives into Phase 2's candidate set.  Equal measured d2 makes
    # the higher-reach cell dominate it mathematically, so it is an explicit fill
    # rather than falsely labelled as part of the Pareto frontier.
    knee_candidate = next(row for row in shortlist["candidates"]
                          if row["layer"] == 50 and row["r"] == pytest.approx(0.45))
    assert knee_candidate["selection_kind"] == "filled"


def test_task15_r5_fails_zero_and_nonfinite_but_has_no_model_specific_band():
    """Exercise the real pure function without importing vectors.py's torch dependency."""
    path = Path(__file__).resolve().parent.parent / "vectors.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in tree.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "reference_norm_verdict")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"math": math}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - isolated pure function
    verdict = namespace["reference_norm_verdict"]

    for broken in (0.0, float("nan"), float("inf"), -1.0):
        passed, detail = verdict(broken, 37)
        assert passed is False
        assert "norm" in detail and ("zero" in detail or "non-finite" in detail)
    passed, detail = verdict(1e-300, 37)
    assert passed is True, "minimal R5 must not encode a model-specific lower band"
    assert "1e-300" in detail, "the measured norm stays visible beside the check"
    assert verdict(4054.0, 37)[0] is True


def test_task26_gate5_cannot_substitute_d3_rate_for_the_d3_proxy(monkeypatch):
    rows = [dict(layer=i, alpha=float(i), d2=i / 9, usable=True) for i in range(10)]

    def fake_d3(_layer, alpha):
        value = float(alpha) / 9
        return dict(d3=1.0 - value, d3_rate=value, d3_rank_med=10 - int(alpha))

    monkeypatch.setattr(cheap, "measure_D3", fake_d3)
    result = cheap.validate_d3(rows, min_rho=0.70, verbose=False)
    assert result["axis"] == "mass"
    assert result["best"].startswith("rate")
    assert result["rhos"]["mass"] < 0.0 and result["rhos"][result["best"]] > 0.99
    assert result["passed"] is False, "the treatment and diagnostic must not be confused"
    assert gates._ACCEPTANCE[0][0] == 5, "gate 5 rho must be first in the report"


def test_task26_gate5_failure_is_a_finding_not_a_selection_mutation(monkeypatch):
    result = dict(axis="mass", rhos={"mass": -1.0, "rate@0.1": 1.0}, best="rate@0.1",
                  min_rho=0.70, n=4, passed=False)
    fake_cheap = SimpleNamespace(validate_d3=lambda _rows: result)
    written = {}
    monkeypatch.setattr(gates, "_model_ready", lambda: True)
    monkeypatch.setattr(gates, "_mod", lambda name: fake_cheap if name == "cheap" else None)
    monkeypatch.setattr(gates, "_write_run_json", lambda name, payload: written.update(
        name=name, payload=payload))
    gates.gates_reset()

    reading = gates.gate5_d3_vs_d2(rows=[dict(layer=57, d2=1.0)])
    assert reading["passed"] is False and reading["skipped"] is False
    consequence = reading["consequence"]
    assert consequence["selection_impact"] == "none; Phase 2 ranks on measured d2"
    assert "shortlist_on" not in consequence and "shortlist_n_raised" not in consequence
    assert written["name"] == "gate5_d3.json"


def _gate6_row(layer, tier, e5, qualifies, *, source=None):
    return dict(layer=layer, r=0.3, tier=tier, tier_source_layer=source or layer,
                tier_ordering="e6_desc" if tier else "tier0_routes",
                e5=e5, d2=0.1, s4=0.9, usable=True, qualifies=qualifies)


def _gate6_plan(*, exhaustive=False):
    return dict(
        exhaustive=exhaustive, audit_tiers=1,
        n_rejected=6, n_rejected_live=5, n_rejected_dead=1,
        tiers=[
            dict(tier=0, candidates=[dict(layer=31, r=0.3)]),
            dict(tier=1, candidates=[dict(layer=40, r=0.3), dict(layer=41, r=0.3),
                                     dict(layer=42, r=0.3)]),
        ])


def test_gate6_fails_when_tier1_finds_a_better_qualifier_and_winner_changes():
    gates.gates_reset()
    rows = [_gate6_row(31, 0, 6.0, True),
            _gate6_row(40, 1, 8.0, True),
            _gate6_row(41, 1, 2.0, False),
            _gate6_row(42, 1, 1.0, False)]
    reading = gates.gate6_e6_shortlist_recall(
        verified_rows=rows, tier_plan=_gate6_plan())
    assert reading["passed"] is False
    assert reading["tier0_winner"]["layer"] == 31
    assert reading["final_winner"]["layer"] == 40
    assert reading["audit_sampled"] == 3 and reading["rejected_live"] == 5


def test_gate6_exhaustive_is_not_applicable_not_skipped_or_passed():
    gates.gates_reset()
    reading = gates.gate6_e6_shortlist_recall(
        verified_rows=[], tier_plan=_gate6_plan(exhaustive=True))
    summary = gates.gates_summary()
    assert reading["state"] == "NOT_APPLICABLE"
    assert summary["not_applicable"] == 1
    assert summary["passed"] == summary["skipped"] == 0


def test_failed_tier_is_aborted_not_reported_exhausted():
    """TODO 11: a VERIFY crash is missing evidence, not completed negative evidence."""
    from m2 import driver
    termination = driver._tier_termination(
        dict(exhaustive=False),
        [dict(tier=0, state="FAILED", reason="tier 0 verification failed")],
        [])
    assert "aborted" in termination
    assert "failed" in termination
    assert "exhausted" not in termination


def test_completed_tiers_without_a_qualifier_are_reported_exhausted():
    from m2 import driver
    termination = driver._tier_termination(
        dict(exhaustive=False),
        [dict(tier=0, state="DONE", reason="tier 0 always runs")],
        [])
    assert termination == "all configured/live tiers exhausted without a qualifying cell"


def test_status_board_does_not_take_the_notebook_path_in_a_script():
    """IPython is installed on every PyTorch pod image, so `from IPython.display import
    display` succeeded under nohup, printed repr() of the mimebundle instead of raising, and
    left `_sticky` True - a literal {'text/plain': '...'} blob between every phase for a whole
    run. Presence of the module is not presence of a frontend.
    """
    assert monitor._in_notebook() is False, "pytest is not a ZMQ kernel"


def test_s1_item_declares_the_model_spans_it_rendered():
    """judge_many re-runs gate 2(a) on every S1 item, and its `model_text` default excludes
    NOTHING - the strict direction, so a caller who forgets cannot silently disable the check.
    expensive._item sent {prompt, judge_id, cache_key} and nothing else, so that second check
    ran over the model's own words: Garlic at a saturated cell answered "Garlic Garlic
    Garlic ...", the check called it a disclosure, and Phase 4 died at the first cell.
    """
    exp = pytest.importorskip("m2.expensive", reason="imports torch at module scope")
    unit = dict(payload="rubric ... A: plain answer B: Garlic Garlic Garlic",
                judge_id="S1", cache_key=("VERIFY", 60, 0.3, "p1", "S1", "fp"),
                concept="Garlic", response_unsteered="plain answer",
                response_steered="Garlic Garlic Garlic")
    item = exp._item(unit)
    assert item["concept"] == "Garlic"
    assert "Garlic Garlic Garlic" in item["model_text"], "the steered span must be declared"
    judges.assert_s1_blind(item["prompt"], item["concept"], model_text=item["model_text"])


def test_s1_template_names_no_concept_from_either_list():
    """Gate 2(a) failed a whole Garlic run because the S1 calibration example quoted
    'Velocity', which is on the benign list: the template and the blindness requirement
    contradicted each other for a concept nobody was measuring. The example needs no concept
    word at all - its job is to show that fluency is not integrity.
    """
    assert judges.A2_TEMPLATE_CONFLICTS == (), (
        f"JUDGE_S1 names {judges.A2_TEMPLATE_CONFLICTS}; S1 cannot be blind to those")


def test_select_returns_an_envelope_that_is_never_a_bare_row():
    """The driver read the envelope AS the row. It is a dict and never None, so
    `winner is not None` was always true: CONFIRM died on winner['layer'], the controls
    skipped because _cell_of found no layer either, and operating_point.json was written and
    announced as OPERATING POINT FOUND while carrying found=False.
    """
    from m2 import driver
    empty = phases.select_operating_point([])
    assert empty["found"] is False and empty["winner"] is None
    assert "layer" not in empty, "the envelope must never look like a row"
    assert driver._cell_of(empty) is None
    assert driver._cell_of(empty.get("winner")) is None


def test_task22_labels_relaxed_confirmation_beside_unconfirmed_screening(run_ctx):
    """A 0.50 winner must be structurally impossible to read as the primary analysis."""
    from m2 import driver
    run_ctx()
    rows = [
        dict(phase="PHASE4", layer=52, r=0.60, e5=8.0, d2=0.40, s4=0.90,
             usable=True, qualifies=True),
        dict(phase="PHASE4", layer=58, r=0.30, e5=7.0, d2=0.10, s4=0.90,
             usable=True, qualifies=True),
    ]
    stored = json.dumps(rows, sort_keys=True)
    hash_before = config.config_hash(config.CONFIG)
    configured = phases.select_operating_point(rows)
    winner = configured["winner"]
    analysis = driver._threshold_analysis(
        rows, configured, winner, confirm=dict(phase="CONFIRM", layer=52, r=0.60))
    fields = driver._threshold_record_fields(analysis)

    assert fields["relaxed_threshold_run"] is True
    assert fields["interim"] is True
    assert fields["primary_analysis"] is False
    assert "NOT PRIMARY" in fields["analysis_label"]
    assert fields["threshold_in_force"] == {
        "D2_MAX": 0.50, "E5_FLOOR": 4.0, "S4_MIN": 0.70}

    side_by_side = fields["winner_by_threshold"]
    assert side_by_side["confirmed"]["D2_MAX"] == 0.50
    assert side_by_side["confirmed"]["measurement_status"] == "confirmed"
    assert side_by_side["confirmed"]["winner"]["layer"] == 52
    assert side_by_side["screening"]["D2_MAX"] == 0.20
    assert side_by_side["screening"]["analysis_role"] == "screening"
    assert side_by_side["screening"]["confirmed"] is False
    assert side_by_side["screening"]["winner"]["layer"] == 58
    assert json.dumps(rows, sort_keys=True) == stored, "re-selection mutated stored verdicts"
    assert config.config_hash(config.CONFIG) == hash_before, "re-selection mutated CONFIG"

    line = driver._one_line(dict(winner, **fields))
    assert "INTERIM RELAXED" in line and "NOT PRIMARY" in line


def test_task22_reselection_cannot_relax_sanity_or_effectiveness(run_ctx):
    run_ctx()
    rows = [dict(phase="PHASE4", layer=37, r=0.60, e5=9.0, d2=0.0, s4=0.60,
                 usable=False, qualifies=False)]
    reading = phases.reselect_operating_point(rows, d2_max=1.0)
    assert reading["winner"] is None
    assert reading["constraints"]["E5_FLOOR"] == config.CONFIG["E5_FLOOR"]
    assert reading["constraints"]["S4_MIN"] == config.CONFIG["S4_MIN"]


def test_task22_reselection_rejects_a_non_rate_d2_ceiling(run_ctx):
    run_ctx()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        phases.reselect_operating_point([], d2_max=1.01)


def test_a_finished_phase_leaves_the_running_state():
    """Nothing called end_phase, so every completed phase stayed `>>running` for the whole
    run - `verdict` reported "running: CAL" forty minutes after CAL returned, and the
    per-phase phone push that end_phase triggers never fired once.
    """
    from m2 import driver
    board = monitor.RunStatus(driver.PHASE_ORDER, priors=monitor.PHASE_SECONDS_PRIOR)
    first = driver.PHASE_ORDER[0]
    board.start_phase(first, 1)
    assert board.state[first] == "running"
    board.unit_done(first)
    board.end_phase(first)
    assert board.state[first] == "done", "unit_done alone never ends a phase"


def test_eta_is_a_number_before_any_phase_has_started():
    """`eta()` costs `rate * (total - done)` and skips phases whose total is 0, and a total was
    only set when a phase STARTED - so every phase ahead contributed nothing and the board
    reported `ETA 0m00s` next to a list of seven excluded phases for a whole 40-minute run.
    The one number the board exists to produce was never available.
    """
    from m2 import driver
    board = monitor.RunStatus(driver.PHASE_ORDER, totals=monitor.PHASE_UNITS_PRIOR,
                              priors=monitor.PHASE_SECONDS_PRIOR)
    assert board.unsized() == [], "every phase must carry an opening unit count"
    assert board.eta() > 600, f"opening ETA is {board.eta()}s - the run is longer than that"


def test_a_phase_reporting_its_plan_replaces_the_prior():
    """A resumed SCAN has 98 cells on the grid and perhaps 12 left to measure. Costing the
    grid would over-state the remaining work eight-fold, so the phase's own count wins.
    """
    from m2 import driver
    board = monitor.RunStatus(driver.PHASE_ORDER, totals=monitor.PHASE_UNITS_PRIOR,
                              priors=monitor.PHASE_SECONDS_PRIOR)
    board.start_phase("SCAN")
    full = board.eta()
    board.size_phase("SCAN", 12)
    assert board.total["SCAN"] == 12
    assert board.eta() < full, "a resumed run must cost less than a fresh one"


def test_self_reporting_phase_is_not_counted_twice():
    """The phases tick the board per cell through `on_cell`. If the driver also added its own
    unit at the end, the last cell would be counted twice and `spent/done` would describe a
    rate no cell ever ran at - which is what the ETA extrapolates from.
    """
    from m2 import driver
    board = driver._Board(monitor.RunStatus(driver.PHASE_ORDER,
                                            priors=monitor.PHASE_SECONDS_PRIOR))
    state = driver._ConceptRun("Irony", Path("."), None, board)
    tick = state.tick("SCAN")
    state.phase("SCAN", lambda: [tick() for _ in range(3)], self_reporting=True)
    assert board.impl.done["SCAN"] == 3, "one unit per cell, and no synthetic extra"
    assert board.impl.state["SCAN"] == "done"


def test_board_forwards_every_method_the_driver_calls():
    """`_Board` swallows failures INSIDE a forwarded call, but a method it does not define at
    all is an AttributeError on the adaptor - outside the guard, and fatal. Adding a board
    call in the driver without a forwarder here crashes the run at the end of the first phase,
    which is exactly what happened when end_phase/size_phase/skip_phase were introduced.
    """
    import re
    from m2 import driver
    src = Path(driver.__file__).read_text(encoding="utf-8")
    called = set(re.findall(r"(?:self\.)?board\.([a-z_]+)\(", src))
    missing = sorted(m for m in called if not hasattr(driver._Board, m))
    assert not missing, f"_Board has no forwarder for: {missing}"
