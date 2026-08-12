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

import dataclasses
import json
import sys
from pathlib import Path

import pytest

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from m2 import cheap, config, gates, judges, monitor, phases, prompts, runio  # noqa: E402


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
    assert prior["SHORTLIST"] == 0.0
    assert all(v >= 0.0 for v in prior.values())


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


def test_required_credentials_are_the_two_the_run_cannot_proceed_without():
    """Under nohup there is no TTY, so a getpass prompt would read EOF and the run would start
    with no judge key and fail forty minutes later in Phase 4."""
    from m2 import run as m2run
    assert set(m2run.REQUIRED_ENV) == {"HF_TOKEN", "OPENROUTER_API_KEY"}
    assert "HEALTHCHECK_URL" in m2run.OPTIONAL_ENV


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
