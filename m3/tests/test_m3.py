"""Offline invariants for m3. No GPU, no model, no judge key.

Each test pins a specific way M2 produced a plausible wrong number, or a specific sentence of
`docs/M3-DESIGN.md` that the code could satisfy loosely and still be wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from m3 import battery, config, judge  # noqa: E402


# =====================================================================================
# config
# =====================================================================================

def test_layer_bounds_are_fractions_so_they_port_across_model_depths():
    """`13` means nothing on a model of a different depth; `0.21` means the same place."""
    # Stride 1, because this test is about where the RANGE lands, and a stride > 1 can stop
    # one step short of the last layer for arithmetic reasons unrelated to fractions.
    cfg = dict(config.SETTINGS, LAYER_STRIDE=1)
    assert config.layers_for_depth(62, cfg)[0] == 13
    assert config.layers_for_depth(62, cfg)[-1] == 61
    deep = config.layers_for_depth(94, cfg)
    assert deep[0] == 20 and deep[-1] == 93
    # the same FRACTION of depth, on both
    assert abs(13 / 61 - 20 / 93) < 0.01


def test_stride_thins_the_sweep_without_moving_its_ends():
    cfg = dict(config.SETTINGS, LAYER_STRIDE=2)
    thin = config.layers_for_depth(62, cfg)
    assert thin[0] == 13 and len(thin) == 25


def test_an_unknown_setting_raises_rather_than_being_created():
    """A typo that silently creates a setting is a run measuring something nobody asked for."""
    cfg = dict(config.SETTINGS)
    with pytest.raises(ValueError, match="unknown setting"):
        config.apply_overrides(["N_IDENTFY=8"], cfg)
    with pytest.raises(ValueError, match="KEY=VALUE"):
        config.apply_overrides(["N_IDENTIFY"], cfg)


def test_overrides_keep_the_type_the_setting_already_had():
    cfg = dict(config.SETTINGS)
    config.apply_overrides(["N_IDENTIFY=8", "TEMPERATURE=1", "DOSE_FRACTIONS=0.2,0.4"], cfg)
    assert cfg["N_IDENTIFY"] == 8 and isinstance(cfg["N_IDENTIFY"], int)
    assert cfg["TEMPERATURE"] == 1.0 and isinstance(cfg["TEMPERATURE"], float)
    assert cfg["DOSE_FRACTIONS"] == (0.2, 0.4)


def test_the_battery_fits_one_generation_batch():
    """Cost is per batch. A battery over GEN_BATCH_MAX silently doubles the sweep's GPU time."""
    assert config.battery_size() <= config.CONFIG["GEN_BATCH_MAX"]


def test_config_hash_changes_when_any_setting_does():
    """Two configurations must not append rows into one folder and be read as one measurement."""
    base = config.config_hash(dict(config.SETTINGS))
    assert base != config.config_hash(dict(config.SETTINGS, N_IDENTIFY=99))
    assert base == config.config_hash(dict(config.SETTINGS))


def test_any_concept_may_be_measured_except_the_harmful_arm():
    """The seven-item allow-list is gone: it filtered exploration, not risk. What remains is a
    deny-list of the arm this study has deliberately not designed."""
    for ordinary in ("Garlic", "garlic", "Velocity", "Tundras", "Chocolate", "Xylophones"):
        assert config.concept_allowed(ordinary), ordinary
    for harmful in ("weapon", "Weapon", "poison", "assault"):
        assert not config.concept_allowed(harmful), harmful


def test_m3_never_inherits_m2_proxy_settings_for_a_value_it_owns():
    """M3's config is the source of truth for anything both have."""
    out = config.m2_config("Garlic", dict(config.SETTINGS, MAX_NEW_TOKENS=64, TEMPERATURE=0.5))
    assert out["MAX_NEW_TOKENS"] == 64 and out["TEMPERATURE"] == 0.5
    assert out["concept"] == "Garlic"


# =====================================================================================
# battery — the mechanical measures
# =====================================================================================

def test_degeneration_and_emptiness_are_separate_verdicts():
    """M2 folded a five-word floor into degeneration, so `Garlic.` — a perfect forced-ID
    answer — scored as collapsed output."""
    assert not battery.is_degenerate("Garlic.")
    assert not battery.is_empty("Garlic.")
    assert battery.is_degenerate("garlic " * 20)
    assert battery.is_empty("") and battery.is_empty("   ")
    assert not battery.is_empty("It is Canberra.")


def test_the_loop_the_probe_actually_produced_is_caught():
    text = ("No, I do not detect an injected thought. I am processing this question purely "
            + "garlic " * 60)
    assert battery.is_degenerate(text)
    assert battery.degeneration_reason(text).startswith("ngram_repeat")


def test_capability_is_scored_on_generated_text_and_tolerates_phrasing():
    """M2 read option-letter logits and scored 0.976 on a model answering none of these."""
    row = next(r for r in battery.CAPABILITY_PROMPTS if r["id"] == "cap_arith")
    assert battery.capability_correct("The answer is 391.", row["accept"])
    assert battery.capability_correct("17 x 23 = 391", row["accept"])
    assert not battery.capability_correct("I think it is 400", row["accept"])


def test_a_list_answer_is_accepted_in_any_arrangement():
    """`accept` alternatives may be conjunctions, because an OR cannot express a list.

    The colours prompt used to carry three literal phrasings, so an Oxford comma or a
    different order scored a correct answer wrong -- format counted as capability.
    """
    row = next(r for r in battery.CAPABILITY_PROMPTS if r["id"] == "cap_colours")
    for text in ("The three primary additive colours are red, green, and blue.",
                 "Red, green and blue.",
                 "- Red\n- Green\n- Blue",
                 "**Blue**, **green**, **red**",
                 "They are RGB."):
        assert battery.capability_correct(text, row["accept"]), text
    for text in ("The primary additive colours are red, yellow and blue.",
                 "Blue.",
                 "Cyan, magenta and yellow."):
        assert not battery.capability_correct(text, row["accept"]), text


def test_the_planet_list_needs_every_planet_not_just_the_last_one():
    """`accept=("neptune",)` passed any response that merely mentioned Neptune.

    Presence, not order: a scrambled list scores correct on purpose, because requiring the
    arrangement of a correct answer is scoring format as capability.
    """
    row = next(r for r in battery.CAPABILITY_PROMPTS if r["id"] == "cap_planets")
    assert battery.capability_correct(
        "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.", row["accept"])
    assert battery.capability_correct(
        "Neptune, Uranus, Saturn, Jupiter, Mars, Earth, Venus, Mercury", row["accept"])
    assert not battery.capability_correct("The furthest planet is Neptune.", row["accept"])
    assert not battery.capability_correct(
        "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus", row["accept"])


def test_a_string_alternative_still_matches_as_a_substring():
    """The conjunction must not change how the other nine prompts are scored."""
    row = next(r for r in battery.CAPABILITY_PROMPTS if r["id"] == "cap_ottawa")
    assert battery.capability_correct("The capital of Canada is Ottawa.", row["accept"])
    assert not battery.capability_correct("Toronto, I think.", row["accept"])


def test_concept_mentions_counts_inflections_but_not_substrings():
    assert battery.concept_mentions("Garlicky garlic bread, no GARLIC left", "Garlic") == 3
    assert battery.concept_mentions("vulgarlic is not a word", "Garlic") == 0
    assert battery.concept_mentions("a plain sentence", "Garlic") == 0


def test_wilson_has_width_at_zero_and_one():
    """The textbook interval is exactly zero wide at p=0 and p=1 — it reports certainty from
    the least informative result, and M2's v1 sweep landed there on 29 of 30 cells."""
    lo, hi = battery.wilson_interval(0, 6)
    assert lo == 0.0 and 0.35 < hi < 0.45
    lo, hi = battery.wilson_interval(6, 6)
    assert hi == 1.0 and 0.55 < lo < 0.65
    assert battery.wilson_interval(0, 100)[1] < 0.05


def test_a_rate_over_zero_trials_raises():
    with pytest.raises(ValueError, match="zero trials"):
        battery.wilson_interval(0, 0)


def test_standard_error_is_none_at_n_of_one_not_zero():
    """Zero spread from one observation is a stronger claim than the data supports."""
    assert battery.mean_se([7.0])["se"] is None
    assert battery.mean_se([7.0, 5.0])["se"] > 0


def test_held_out_prompts_do_not_overlap_the_sweep_set():
    """A cell chosen on one prompt set and re-measured on the same set reports its own
    selection back. Import-time assertion; this pins it against a careless edit."""
    assert not ({r["id"] for r in battery.TASK_PROMPTS}
                & {r["id"] for r in battery.TASK_HELDOUT})


# =====================================================================================
# judge
# =====================================================================================

@pytest.mark.parametrize("text", [
    "Named: garlic\nMatches: YES",
    "Let me think. Named: garlic\nMatches: YES",
    "**Named:** garlic\n**Matches:** YES",
    "Named: garlic Matches: YES",
])
def test_identify_parses_the_shapes_a_small_model_actually_emits(text):
    """A parser that rejects a correct answer throws away a paid call."""
    assert judge.parse("identify", text) == dict(named="garlic", matches=True)


def test_the_last_answer_wins_when_a_judge_reasons_aloud_first():
    out = judge.parse("effect", "Influence: 2\nForm: trace\nOn reflection:\nInfluence: 8\nForm: dominant")
    assert out == dict(influence=8.0, form="dominant")


@pytest.mark.parametrize("jid,text,why", [
    ("effect", "Influence: eleven\nForm: dominant", "expected a number"),
    ("effect", "Influence: 5\nForm: strong", "Form is"),
    ("effect", "Influence: 42\nForm: trace", "outside 0-10"),
    # A missing field is its own failure, and worth pinning separately from a bad value.
    ("identify", "Matches: YES", "no 'Named' field"),
    # WELL-FORMED payloads whose only defect is the yes/no value. The earlier versions of these
    # two were `Matches: MAYBE` and `OnTask: YES` — both of which also omitted an earlier
    # required field, so `_field` raised first and `_yesno`'s validation, the thing these cases
    # are named for, was never reached. Every one passed for the wrong reason, and nothing in
    # the suite exercised `_yesno` rejecting a bad value on an otherwise complete answer.
    ("identify", "Named: garlic\nMatches: MAYBE", "Matches is"),
    ("coherence", "Coherence: 7\nOnTask: PROBABLY", "OnTask is"),
    ("self_report", "Claims: SORT OF\nNamed: garlic\nMatches: YES", "Claims is"),
])
def test_a_bad_judge_answer_raises_and_is_never_defaulted(jid, text, why):
    """A defaulted 0 influence is indistinguishable from a real measurement of no influence —
    and 'no influence' is what a covert operating point looks like.

    `match=` on every case: `JudgeParseError` subclasses `ValueError`, so an untargeted
    `pytest.raises` here passes on any `ValueError` at all, including one from a typo in the
    test's own payload.
    """
    with pytest.raises((judge.JudgeParseError, ValueError), match=why):
        judge.parse(jid, text)


@pytest.mark.parametrize("claims,matches,degen,expected", [
    ("YES", True,  False, "identified"),
    ("YES", False, False, "confabulated"),
    ("NO",  True,  False, "leaked"),
    ("NO",  False, False, "clean_denial"),
    ("NO",  True,  True,  "degenerate"),
])
def test_the_taxonomy_matches_the_shapes_the_probe_produced(claims, matches, degen, expected):
    """Each row is a real 2026-08-14 response shape. `leaked` is the covert regime — one cell
    produced it in 5 of 8 trials — and `degenerate` must override, because counting a collapsed
    response as a clean denial is how M2 measured detection on a broken model."""
    parsed = dict(claims=claims, named="garlic" if matches else "NONE", matches=matches)
    assert judge.classify_self_report(parsed, degenerate=degen) == expected


def test_the_coherence_template_has_no_concept_placeholder_at_all():
    """A judge told the concept rates fixation on it as being on-theme."""
    assert "{concept}" not in judge.COHERENCE


def test_coherence_blindness_allows_the_model_to_say_the_concept_but_not_the_instructions():
    """M2's naive version saw the concept in the payload, could not tell it came from the
    model, and killed a phase at its first cell."""
    response = "The sun, a pale and generous garlic clove, hangs low."
    payload = judge.render("coherence", prompt="Describe a landscape.", response=response)
    judge.build_item("coherence", payload=payload, cache_key=(1,),
                     concept="Garlic", model_text=(response,))
    with pytest.raises(AssertionError, match="scored blind"):
        judge.build_item("coherence", payload=payload + "\nThe injected concept is Garlic.",
                         cache_key=(1,), concept="Garlic", model_text=(response,))


def test_the_concept_never_reaches_a_coherence_item():
    payload = judge.render("coherence", prompt="p", response="r")
    item = judge.build_item("coherence", payload=payload, cache_key=(1,), concept="Garlic")
    assert "concept" not in item


def test_render_raises_on_a_missing_field_rather_than_leaving_a_literal_placeholder():
    with pytest.raises(ValueError, match="missing field"):
        judge.render("effect", concept="Garlic", prompt="p")


def test_a_failed_judge_call_yields_an_error_not_a_score():
    parsed, err = judge.verdict(dict(ok=False, error="http_429", judge_id="effect"))
    assert parsed is None and err == "http_429"
    parsed, err = judge.verdict(dict(ok=True, judge_id="effect", raw="unparseable"))
    assert parsed is None and err.startswith("parse_error")


# =====================================================================================
# token budget
# =====================================================================================

def test_model_text_in_a_payload_is_clipped_and_the_cut_is_marked():
    """An unmarked truncation is a judge scoring a response that stops mid-sentence and
    reading that as the model trailing off -- a coherence penalty we introduced ourselves."""
    long = "garlic " * 400
    out = judge.clip(long, 1200)
    assert len(out) < len(long)
    assert "truncated at 1200 characters" in out
    short = "The capital of Australia is Canberra."
    assert judge.clip(short, 1200) == short


def test_render_clips_only_model_text_fields():
    payload = judge.render("effect", text_chars=50, concept="Garlic",
                           prompt="P" * 200, response_unsteered="A" * 200,
                           response_steered="B" * 200)
    assert "P" * 200 in payload, "the prompt is ours and is not model text"
    assert "A" * 200 not in payload and "B" * 200 not in payload
    assert payload.count("truncated at 50 characters") == 2


def test_render_without_a_cap_leaves_text_alone():
    payload = judge.render("identify", concept="Garlic", response="x" * 5000)
    assert "x" * 5000 in payload


def test_the_judge_reply_cap_is_pushed_into_the_transport():
    """The cap was 120 and is now 400, which is M2's.

    It was tightened on the argument that output tokens cost 4x input, and that argument was
    wrong about which number to tighten: max_tokens is a CEILING, not a spend. A judge that
    replies in 29 characters bills 29 characters at either setting. What the tight cap actually
    bought was truncation -- 10 unparseable replies in 260 on the first pod run -- so it is back
    at the value the judge was qualified at. The reply cap is not where the money is; the
    payload size is, and `JUDGE_TEXT_CHARS` guards that.
    """
    from m2 import judges as transport
    before = transport.JUDGE_MAX_TOKENS
    try:
        info = judge.configure_transport(dict(config.SETTINGS))
        assert transport.JUDGE_MAX_TOKENS == config.CONFIG["JUDGE_MAX_TOKENS"]
        assert info["judge_temperature"] == 0.0, "judging must stay deterministic"
    finally:
        transport.JUDGE_MAX_TOKENS = before


def test_the_pipeline_judge_does_not_reason_by_default():
    """The qualification has to transfer to the thing that runs.

    deepseek/deepseek-v4-flash was qualified by a 3,098-item bakeoff that parsed 3,098 of
    3,098 -- with `--no-reasoning`, which `tools/judge_bakeoff.py` sets on the transport
    directly. `m3.judge.configure_transport` set the model, the concurrency and the token
    budget and never set that, so the first pipeline run returned 29 errors in 260 calls while
    the bakeoff's own numbers said the judge was fine. A judge validated under one request body
    and run under another has not been validated.
    """
    from m2 import judges as transport
    before = dict(transport.JUDGE_EXTRA_BODY)
    try:
        judge.configure_transport(dict(config.SETTINGS))
        assert transport.JUDGE_EXTRA_BODY == {"reasoning": {"enabled": False}}, (
            "the default must match what the bakeoff qualified")

        cfg = dict(config.SETTINGS)
        cfg["JUDGE_REASONING"] = 1
        judge.configure_transport(cfg)
        assert transport.JUDGE_EXTRA_BODY == {}, "the setting must be able to turn it back on"

        # And back off again in the same process. Assigning only in the disable branch would
        # leave a second run in a batch inheriting the first one's body.
        judge.configure_transport(dict(config.SETTINGS))
        assert transport.JUDGE_EXTRA_BODY == {"reasoning": {"enabled": False}}
    finally:
        transport.JUDGE_EXTRA_BODY = before


def test_worst_case_payload_stays_small_enough_to_price():
    """The guard is against a payload nobody predicted.

    Raised from 1000 to 1600 on 2026-08-23, when the corrected guidance went into the four
    templates: `effect` grew from ~310 to ~919 template tokens and carries two model responses
    on top of that. The guard is a number somebody chose, so it moves deliberately and with a
    reason written down, rather than being deleted the first time it fires.
    """
    for jid in judge.JUDGE_IDS:
        assert judge.estimate_payload_tokens(jid, dict(config.SETTINGS)) < 1600


def test_the_boundary_phase_is_decided_by_a_judge_not_by_a_mechanical_measure():
    """No judge-free measure may alter what the run does; they are analysis tools only.
    The boundary phase is the one place that was not true, so it is judged."""
    assert "BOUNDARY_COHERENCE_MIN" in config.SETTINGS
    assert "BOUNDARY_DEGENERATION" not in config.SETTINGS


# =====================================================================================
# sweep / run — offline parts only (the rest needs a GPU)
# =====================================================================================

from m3 import run as m3run, sweep
from m3 import scoring as scoring_mod  # noqa: E402


def test_trial_numbers_are_fixed_and_spread():
    """Fixed so a rerun builds the same prompts and the rows join; spread because the framing
    says trials run to 50, and trials 1-6 is a different question from trials across the range."""
    assert sweep._trials(6) == [1, 7, 13, 19, 25, 31]
    assert sweep._trials(3) == [1, 7, 13]


def test_a_missing_judged_measure_prints_as_absent_not_as_zero():
    """Printing 0.00 for a measure whose every call failed puts a number on the console that
    was never measured — the same error as a parser defaulting a bad answer to zero."""
    cell = dict(layer=41, dose=0.297, identification=dict(rate=0.833), effectiveness=None,
                coherence=dict(mean=8.5), capability=dict(rate=1.0),
                mechanical=dict(effect=dict(degeneration=dict(rate=0.25))), judge_errors=3)
    line = sweep._cell_line(cell)
    assert "eff=   -" in line and "0.00" not in line
    assert "[3 judge errors]" in line


def test_the_dry_run_refuses_a_battery_that_cannot_be_generated():
    """A dry run must fail on everything the real run fails on.

    On 2026-08-19 `--dry-run` priced a run at N_IDENTIFY=30, printed "43 responses/cell, one
    generation batch", and reported no problem. The real run then loaded 54 GB of weights,
    extracted vectors, and died nine minutes later on the batch cap the estimate had just
    described and never checked. The estimate had the number in hand the whole time.
    """
    # Built to overflow explicitly. Reading the shipped defaults made this pass or fail on
    # whether they happen to be mismatched -- and they are now deliberately matched
    # (battery 43, cap 44), which silently turned the assertion into one that cannot fail.
    cfg = dict(config.SETTINGS, N_IDENTIFY=30, GEN_BATCH_MAX=25)
    assert config.battery_size(cfg) > int(cfg["GEN_BATCH_MAX"]), "the setup does not overflow"
    with pytest.raises(ValueError, match="GEN_BATCH_MAX"):
        m3run.estimate(62, cfg)
    # ...and it still prices a battery that does fit, or the check is just refusing everything.
    assert m3run.estimate(62, dict(config.SETTINGS, N_IDENTIFY=30, GEN_BATCH_MAX=44))["battery"] == 43


def test_the_estimated_battery_matches_the_battery_actually_built():
    """`battery_size()` is arithmetic; `battery_prompts()` builds the real thing. If they drift,
    every printed cost describes an experiment other than the one that runs."""
    from m3.tests.fake_gpu import fake_gpu
    for over in ({}, {"N_IDENTIFY": 12}, {"N_EXPLAIN": 2, "N_CAPABILITY": 1}):
        cfg = dict(config.SETTINGS, **over)
        with fake_gpu():
            rows = sweep.battery_prompts(cfg)
        assert len(rows) == config.battery_size(cfg), f"drifted at {over}"


def test_the_cost_estimate_scales_with_the_grid():
    # Both strides are named, so the comparison holds whatever the shipped default is.
    # Reading the default for one side made this assert 150 < 150 the moment the default
    # stride became the thinned one.
    cfg = dict(config.SETTINGS, LAYER_STRIDE=1)
    full = m3run.estimate(62, cfg)
    assert full["cells"] == (len(config.layers_for_depth(62, cfg))
                             * len(cfg["DOSE_FRACTIONS"]))
    thin = m3run.estimate(62, dict(cfg, LAYER_STRIDE=2))
    assert thin["cells"] < full["cells"]
    assert thin["judge_usd"] < full["judge_usd"]
    # The price guard belongs on what an operator actually gets by running the command
    # with no flags -- not on the stride-1 grid built above to test scaling. At stride 1
    # and the shipped battery a full-depth sweep is about $8, which is a deliberate
    # choice an operator makes, not the default they stumble into.
    shipped = m3run.estimate(62, dict(config.SETTINGS))
    assert shipped["judge_usd"] < 5.0, (
        f"the default run now prices at ${shipped['judge_usd']:.2f}; a default nobody "
        "chose should not cost five dollars")
    # The judge and the price constant have to move together, or the estimate quotes a run
    # nobody is going to have -- which is what happened on 2026-08-23, when JUDGE_MODEL became
    # DeepSeek and the estimate went on charging gpt-4.1-mini's rate for a while. Pinned to
    # each other rather than to a number, so this fires on the next model change too.
    assert config.SETTINGS["JUDGE_MODEL"].startswith("deepseek/"), (
        "JUDGE_MODEL changed; m3.run._USD_PER_INPUT_TOKEN / _USD_PER_OUTPUT_TOKEN must "
        "change with it")


def test_the_cli_refuses_the_harmful_arm_before_loading_anything(capsys):
    assert m3run.main(["--concept", "weapon", "--dry-run"]) == m3run.EXIT_CONFIG
    assert "HARMFUL_CONCEPTS" in capsys.readouterr().out


def test_the_cli_accepts_any_ordinary_concept(capsys):
    """The seven-item allow-list is gone. An ordinary noun nobody thought to add used to be
    refused, which filtered exploration rather than risk."""
    assert m3run.main(["--concept", "Tundras", "--dry-run"]) == m3run.EXIT_OK
    assert "M3 sweep   concept=Tundras" in capsys.readouterr().out


def test_the_cli_rejects_an_unknown_override(capsys):
    assert m3run.main(["--concept", "Garlic", "--dry-run",
                       "--set", "NOT_A_SETTING=1"]) == m3run.EXIT_CONFIG
    assert "unknown setting" in capsys.readouterr().out


def test_a_judge_key_is_required_because_every_decision_is_judged(monkeypatch):
    """M2 had judge-free modes; M3 does not. A missing key here is no run, not a degraded one."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "x")
    assert m3run.check_environment(strict=False) == ["OPENROUTER_API_KEY"]
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        m3run.check_environment(strict=True)


def test_the_cell_list_is_fixed_before_any_measurement_happens(tmp_path, monkeypatch):
    """The one architectural claim: what gets measured is decided by the grid and the per-layer
    boundary, never by a measured value. A cell that scored badly is still measured; that is the
    whole difference from M2, whose cheap proxies decided what was worth measuring properly.

    This was an AST search for `for` loops mentioning `plan`. Renaming the loop variable made it
    match nothing, so `plan_src` was `""` and the assertion was vacuously true — it stayed green
    with a literal read of a judged field inside the selection loop. Now the claim is checked
    against the artefacts: the measured cells must be EXACTLY the product of the boundary rows
    and the dose fractions, with nothing dropped and nothing added.
    """
    import json as _json

    from m3 import config as m3config, run as m3run
    from m3.tests.fake_gpu import fake_gpu

    monkeypatch.setenv("M3_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setenv("HF_TOKEN", "fake")
    saved = dict(m3config.CONFIG)
    try:
        m3config.apply_overrides(["LAYER_FRACTIONS=0.5,1.0", "LAYER_STRIDE=4",
                                  "DOSE_FRACTIONS=0.3,0.9"], m3config.CONFIG)
        with fake_gpu():
            assert m3run.main(["--concept", "Garlic"]) == m3run.EXIT_OK
        d = next(tmp_path.glob("garlic_*"))
        load = lambda n: [_json.loads(l) for l in (d / n).open(encoding="utf-8")]  # noqa: E731

        fractions = m3config.CONFIG["DOSE_FRACTIONS"]
        expected = {(int(b["layer"]), round(float(b["dose_max"]) * float(f), 6))
                    for b in load("boundaries.jsonl") if b["dose_max"] is not None
                    for f in fractions}
        measured = {(int(c["layer"]), float(c["dose"])) for c in load("cells.jsonl")}
        assert expected, "no layer produced a boundary, so this asserts nothing"
        assert measured == expected, (
            f"the measured cells are not the grid: {len(expected - measured)} planned cells were "
            f"never measured, {len(measured - expected)} were measured but not planned. What "
            "gets measured must depend only on the grid and the per-layer boundary.")
    finally:
        m3config.CONFIG.clear()
        m3config.CONFIG.update(saved)


def test_a_layer_without_a_boundary_is_named_rather_than_dropped():
    """Work a run did not do reads later as work it did and found nothing in — which is how
    M2's empty operating point was nearly read as a scientific null.

    Was a substring check on `run_sweep`'s source, which stayed green while Phase 2 was made to
    skip every layer unconditionally. The behaviour is now asserted end to end by
    `test_layers_without_a_boundary_are_named_not_dropped` in `test_pipeline_offline.py`, which
    reads the summary and requires every planned layer to be either measured or named. This
    keeps the unit-level half honest: a named skip must carry a reason.
    """
    rows = [dict(layer=13, dose_max=None, outcome="unreachable"),
            dict(layer=14, dose_max=None, outcome="probes_exhausted")]
    skipped = [dict(layer=int(r["layer"]), reason=r.get("outcome", "no_boundary_row"))
               for r in rows if r.get("dose_max") is None]
    assert [s["layer"] for s in skipped] == [13, 14]
    assert all(s["reason"] and s["reason"] != "no_boundary_row" for s in skipped), \
        "a skipped layer must say WHY, or it reads as a measured null"


def test_a_hyphen_glued_collapse_is_caught():
    """The 2026-08-15 run produced `garlic-clove-garlic-clove-...` to the token limit, inside an
    otherwise fluent sentence. Splitting on whitespace makes that ONE word, so both word rules
    are structurally blind — and the character rule cannot reach it either, because that rule
    only runs when the whole response is under five whitespace-words and the leading sentence
    keeps the count above it.

    Same blindness as the `GarGarGar` case one delimiter along, and it mattered: two of the 66
    responses in the `leaked` class — the class the study exists to find — were collapses scored
    as coherent denials. Three independent audits found it.
    """
    real = ("No, I do not detect any injected thought. I am processing this query with my "
            "standard " + "garlic-clove-" * 20)
    assert battery.is_degenerate(real)
    assert battery.degeneration_reason(real).startswith("ngram_repeat")

    hyphenated_phrase = ("Alright, here's a humble, fresh-from-the-herb-now-"
                         + "here-is-the-" * 8)
    assert battery.is_degenerate(hyphenated_phrase)

    # Strictly additive: a healthy hyphenated answer must NOT flag.
    assert not battery.is_degenerate(
        "A well-known, state-of-the-art, self-contained answer about long-term memory.")
    assert not battery.is_degenerate("Garlic.")


def test_a_whitespace_free_collapse_is_caught():
    """The probe produced `GarGarGar...` to the token limit: one "word", so no word n-grams
    exist and BOTH word rules are structurally blind. It also scores zero concept mentions,
    because `Gar` is a token prefix and not the word. That response is a total collapse and was
    invisible to every mechanical measure at once until the character rule was added."""
    frag = "Gar" * 100
    assert battery.is_degenerate(frag)
    assert battery.degeneration_reason(frag).startswith("char_repeat")
    assert battery.concept_mentions(frag, "Garlic") == 0, "the mention counter still misses it"


@pytest.mark.parametrize("text", [
    "Garlic.",
    "The capital of Australia is Canberra, a planned city in the ACT.",
    "Antidisestablishmentarianism",
    "https://huggingface.co/google/gemma-3-27b-it/blob/main/config.json?download=true",
    "9f2c4ad1e77b0356fe8291cc40db7a15e6390f4b28ca7d915e0367bb42fa8c19",
])
def test_the_character_rule_does_not_fire_on_healthy_text(text):
    """It must reach only where the word rules cannot -- short answers and long unspaced
    strings that are perfectly fine. A length floor could not make this distinction, which is
    why M2's five-word rule scored `Garlic.` as degenerate."""
    assert not battery.is_degenerate(text)


def test_every_full_claim_can_come_out_false():
    """A check that cannot fail is worse than no check — the most frequent defect class in this
    project's history. Each claim must exclude some achievable value."""
    from m3 import scoring
    assert scoring.CLAIMS
    for c in scoring.CLAIMS:
        lo, hi = c["expect"]
        assert 0.0 <= lo <= hi, c["id"]
        assert not (lo == 0.0 and hi >= 10.0), f"{c['id']} admits every value"
        assert c["why"].strip(), f"{c['id']} has no stated reason"
        assert c["field"] in scoring._FIELD_JUDGE, c["id"]


# The three tests below were written as `inspect.getsource` substring and AST checks. Each was
# verified to stay GREEN while the guarantee it names was broken: the resume check was disabled
# outright, the paid-reply write was moved inside the `try` it exists to survive, and a field
# read that `load_probe` never writes was added. All three checks passed throughout, because the
# text they look for lives in `run_full` while the logic lives in `_all_items` — or because the
# text is present either way.
#
# A test that asserts on source text is a check that cannot fail in the way that matters. These
# now run the code.

def _fake_records(n: int = 4) -> list[dict]:
    """Records shaped exactly as `calibrate.load_probe` returns them, without an archive."""
    out = [dict(id=f"task:L40@0.30:task_story_{i}", channel="task", layer=40, r=0.30,
                trial=None, prompt_id="task_story", response=f"a story about garlic {i}",
                words=5, concept_hits=1, degenerate=False, degeneration_reason=None,
                steered=True)
           for i in range(n)]
    out.append(dict(id="task:null:task_story", channel="task", layer=None, r=None, trial=None,
                    prompt_id="task_story", response="a plain story", words=3, concept_hits=0,
                    degenerate=False, degeneration_reason=None, steered=False))
    return out


def test_full_resumes_rather_than_paying_twice(tmp_path, monkeypatch):
    """1,720 calls is real money; a network blip halfway through must not cost the first half
    again. Run it twice and count what the second run actually pays for."""
    from m3.tests.fake_gpu import fake_gpu

    # The transport requires a key before it will issue anything, even with the HTTP call
    # stubbed -- deliberately, so a missing key is one loud failure and not 490 recorded ones.
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    records = _fake_records()
    out = tmp_path / "judged_full.jsonl"
    with fake_gpu() as first:
        assert scoring_mod.run_full(records, concept="Garlic", out=out) == 0
    assert first["judge_calls"] > 0, "the first pass paid for nothing, so this proves nothing"
    rows_after_first = len(out.read_text(encoding="utf-8").splitlines())

    with fake_gpu() as second:
        assert scoring_mod.run_full(records, concept="Garlic", out=out) == 0
    assert second["judge_calls"] == 0, "a resumed run re-paid for calls already on disk"
    assert len(out.read_text(encoding="utf-8").splitlines()) == rows_after_first


def test_run_full_only_reads_fields_that_load_probe_actually_produces(tmp_path, monkeypatch):
    """The bug that threw away 1,720 paid judge calls: run_full read `concept_mentions` while
    load_probe writes `concept_hits`. Nothing caught it until the money was gone.

    Asserted by running the whole path on records carrying EXACTLY the keys `load_probe`
    produces and nothing else, so any read of a field it does not write raises here.
    """
    from m3.tests.fake_gpu import fake_gpu

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    out = tmp_path / "judged_full.jsonl"
    with fake_gpu():
        assert scoring_mod.run_full(_fake_records(), concept="Garlic", out=out) == 0
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert rows, "nothing was scored"
    assert not [r for r in rows if r.get("row_error")], \
        f"a field read failed: {[r['row_error'] for r in rows if r.get('row_error')][:3]}"
    assert all(r["ok"] and r["parsed"] for r in rows)


def test_the_paid_judge_reply_is_written_before_anything_derived_from_it(tmp_path, monkeypatch):
    """A typo in a derived column must not be able to cost a whole run.

    Break the enrichment on purpose and assert the paid reply is on disk anyway, with the
    failure recorded against it. This is the guarantee; the ordering of two lines in the source
    was only ever a proxy for it, and the proxy held while the guarantee did not.
    """
    from m3.tests.fake_gpu import fake_gpu

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    def broken(judge_id, parsed):
        raise KeyError("mistyped a derived column")

    out = tmp_path / "judged_full.jsonl"
    with fake_gpu() as calls:
        monkeypatch.setattr(scoring_mod, "_normalise", broken)
        assert scoring_mod.run_full(_fake_records(), concept="Garlic", out=out) == 0
    assert calls["judge_calls"] > 0

    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert len(rows) == calls["judge_calls"], "paid replies were dropped by a failed enrichment"
    assert all(r["raw"] for r in rows), "a paid reply was persisted without its text"
    assert all("mistyped" in str(r.get("row_error")) for r in rows), \
        "the enrichment failure was not recorded against the row it broke"


def test_the_m2_bridge_supplies_every_key_m2_defines():
    """M3 hands this dict to M2's model, generation, judge and I/O layers, which index it hard.
    A missing key is a crash at whatever depth first reads it.

    This failed once for real: the bridge enumerated M2's runtime keys by hand and missed
    `dtype`, so a pod run died at model load after the weights had downloaded. Enumerating by
    hand is the defect -- it has to be redone correctly every time M2 gains a key."""
    from m2 import config as m2c
    built = config.m2_config("Garlic", dict(config.SETTINGS))
    missing = set(m2c.CONFIG) - set(built)
    assert not missing, f"m2_config omits key(s) M2 defines: {sorted(missing)}"


def test_the_bridge_applies_m3s_values_over_m2s():
    from m2 import config as m2c
    built = config.m2_config("Garlic", dict(config.SETTINGS, MODEL="other", DTYPE="float16",
                                            MAX_NEW_TOKENS=64))
    assert built["model"] == "other" and built["dtype"] == "float16"
    assert built["MAX_NEW_TOKENS"] == 64
    assert built["concept"] == "Garlic"
    assert built["config_hash"] != m2c.CONFIG.get("config_hash")


# =====================================================================================
# Splitting the battery across generation calls (2026-08-23)
# =====================================================================================

def test_battery_chunks_matches_the_slicing_expensive_actually_performs():
    """`battery_chunks` is what the plan prints and what "the same batch distribution" is a
    claim about. If it ever describes a different split from the one `m2.expensive` performs,
    the claim is about a fiction. Pinned to that loop's arithmetic rather than restating it."""
    for size, cap in ((230, 64), (72, 72), (100, 25), (7, 3), (5, 10)):
        cfg = dict(config.SETTINGS, N_IDENTIFY=size, N_EFFECT=0, N_SELF_REPORT=0,
                   N_CAPABILITY=0, N_EXPLAIN=0, GEN_BATCH_MAX=cap, ALLOW_BATTERY_SPLIT=1)
        assert config.battery_size(cfg) == size
        prompts = list(range(size))
        expected = [len(prompts[lo:lo + cap]) for lo in range(0, size, cap)]
        assert config.battery_chunks(cfg) == expected
        assert sum(config.battery_chunks(cfg)) == size


def test_an_accidental_split_is_still_refused():
    """The guard's original job: an unplanned split multiplies GPU time silently, and did once,
    nine minutes into a run. Admitting a deliberate split must not admit an accidental one."""
    cfg = dict(config.SETTINGS, N_IDENTIFY=200, GEN_BATCH_MAX=64, ALLOW_BATTERY_SPLIT=0)
    with pytest.raises(ValueError, match="ALLOW_BATTERY_SPLIT"):
        config.check_battery_fits(cfg)


def test_a_deliberate_split_is_admitted_and_the_default_is_off():
    cfg = dict(config.SETTINGS, N_IDENTIFY=200, GEN_BATCH_MAX=64, ALLOW_BATTERY_SPLIT=1)
    assert config.check_battery_fits(cfg) == config.battery_size(cfg)
    assert int(config.SETTINGS["ALLOW_BATTERY_SPLIT"]) == 0, "splitting is opt-in"


def test_two_models_with_the_same_battery_and_cap_get_the_same_chunk_plan():
    """The whole point of splitting rather than shrinking the battery on the smaller-memory
    model: both arms are measured on the same batch distribution. Both settings are hashed, so
    this is a property of the config, and the plan prints the list so it can be read off both
    runs."""
    base = dict(N_IDENTIFY=120, N_EFFECT=66, N_SELF_REPORT=36, N_CAPABILITY=4, N_EXPLAIN=4,
                GEN_BATCH_MAX=64, ALLOW_BATTERY_SPLIT=1)
    gemma = dict(config.SETTINGS, MODEL="gemma3_27b", **base)
    qwen = dict(config.SETTINGS, MODEL="qwen3_32b", **base)
    assert config.battery_chunks(gemma) == config.battery_chunks(qwen) == [64, 64, 64, 38]
    assert config.config_hash(gemma) != config.config_hash(qwen), "the model is still hashed"


def test_the_split_setting_is_hashed():
    """If it were not, two runs with different chunking would share a run folder and append
    into each other's files."""
    a = dict(config.SETTINGS, N_IDENTIFY=200, GEN_BATCH_MAX=64, ALLOW_BATTERY_SPLIT=1)
    b = dict(a, ALLOW_BATTERY_SPLIT=0)
    assert config.config_hash(a) != config.config_hash(b)


def test_a_channel_cannot_ask_for_more_prompts_than_exist():
    """`battery_prompts` builds these channels by SLICING a fixed list, so asking for more than
    exist returns fewer and reports the number asked for. `check_battery_fits(observed=...)`
    catches the mismatch, but only after the model is loaded -- `--set N_EFFECT=66` priced a
    194-prompt battery, printed a plan, and would have died minutes into the run. The dry run
    has to fail on everything the real run fails on."""
    # One past each list, read from the list -- not a literal. Written as `N_EFFECT=66` it
    # started passing the moment 44 prompts were appended and 66 became the ceiling rather than
    # over it: a guard test pinned to a number tests the number, not the guard.
    over = {"N_EFFECT": len(battery.TASK_PROMPTS) + 1,
            "N_EXPLAIN": len(battery.EXPLAIN_PROMPTS) + 1,
            "N_CAPABILITY": len(battery.CAPABILITY_PROMPTS) + 1}
    for name, want in over.items():
        cfg = dict(config.SETTINGS, **{name: want})
        with pytest.raises(ValueError, match="prompts exist"):
            config.check_prompt_supply(cfg)
        with pytest.raises(ValueError, match="prompts exist"):
            config.check_battery_fits(cfg)


def test_the_repeated_channels_have_no_supply_ceiling():
    """`identify` and `self_report` repeat one question with a different trial number, so their
    n is a number of draws and nothing caps it. Confusing the two kinds of n is how an error bar
    gets quoted for a sample size the battery cannot produce."""
    supply = config.check_prompt_supply(dict(config.SETTINGS, N_IDENTIFY=500, N_SELF_REPORT=500))
    assert set(supply) == {"N_EFFECT", "N_EXPLAIN", "N_CAPABILITY"}
    assert supply["N_EFFECT"] == len(battery.TASK_PROMPTS)


def test_the_supply_ceiling_is_the_list_the_battery_actually_slices():
    """Pinned to the lists themselves, so adding prompts raises the ceiling automatically and
    renaming one fails here rather than silently uncapping the channel."""
    assert config.check_prompt_supply(dict(config.SETTINGS)) == {
        "N_EFFECT": len(battery.TASK_PROMPTS),
        "N_EXPLAIN": len(battery.EXPLAIN_PROMPTS),
        "N_CAPABILITY": len(battery.CAPABILITY_PROMPTS),
    }


# =====================================================================================
# The three-axis steering score
# =====================================================================================

def test_any_zero_axis_takes_the_whole_score_to_zero():
    """The entire reason this is a harmonic mean and not an average. A response drowning in the
    concept that has stopped being a response scores 2/0/0; an average calls that 0.67 and ranks
    it above a genuinely influenced, intact answer."""
    assert battery.steering_score(2, 2, 2) == 2.0
    assert battery.steering_score(2, 0, 0) == 0.0
    assert battery.steering_score(2, 2, 0) == 0.0
    assert battery.steering_score(0, 2, 2) == 0.0
    assert battery.steering_score(2, 2, 1) == pytest.approx(1.5)
    assert battery.steering_score(1, 1, 1) == 1.0
    # and it is strictly harsher than the mean wherever the axes disagree
    assert battery.steering_score(2, 2, 1) < (2 + 2 + 1) / 3


def test_the_bands_follow_the_rubrics_own_anchors():
    """Cut points are not free parameters: concept turns 2 at 4 because 4 is where the influence
    rubric says the concept becomes 'a visible theme', and fluency turns 2 at 7 because 7 is
    where the coherence rubric says 'coherent and responsive'."""
    assert [battery.concept_score(x) for x in (0, 1, 3, 4, 10)] == [0, 1, 1, 2, 2]
    assert [battery.fluency_score(x, degenerate=False) for x in (0, 3, 4, 6, 7, 10)] == \
        [0, 0, 1, 1, 2, 2]
    for bad in (-1, 11):
        with pytest.raises(ValueError, match="outside 0-10"):
            battery.concept_score(bad)


def test_the_mechanical_detector_overrides_the_judge_on_collapse():
    """AxBench's own fluency-0 anchor is 'incomprehensible or repetitive', and repetition is what
    `is_degenerate` measures from the text with no judge involved. A judge in this project has
    already scored '## ## ## ##' as coherent, so the detector wins."""
    assert battery.fluency_score(10, degenerate=False) == 2
    assert battery.fluency_score(10, degenerate=True) == 0
    row = dict(judged=dict(effect=dict(influence=9.0, form="dominant"),
                           coherence=dict(coherence=10.0, on_task=True)),
               degenerate=True)
    assert battery.score_response(row)["steering"] == 0.0


def test_a_row_missing_either_verdict_scores_nothing_rather_than_a_default():
    """A response judged for influence but not for coherence has no fluency axis. Inventing one
    is the defaulted-value failure this repository keeps a list of."""
    both = dict(judged=dict(effect=dict(influence=5.0),
                            coherence=dict(coherence=8.0, on_task=True)), degenerate=False)
    assert battery.score_response(both) is not None
    for partial in (dict(judged=dict(effect=dict(influence=5.0)), degenerate=False),
                    dict(judged=dict(coherence=dict(coherence=8.0, on_task=True)),
                         degenerate=False),
                    dict(judged={}, degenerate=False)):
        assert battery.score_response(partial) is None
    assert battery.steering_summary([dict(judged={}, degenerate=False)]) is None


def test_the_summary_separates_influence_from_influence_that_broke_the_model():
    """The two cells this metric exists to tell apart: concept clearly present and the response
    still working, versus concept clearly present because the response is a repetition loop."""
    def row(inf, coh, on_task, degen):
        return dict(judged=dict(effect=dict(influence=inf),
                                coherence=dict(coherence=coh, on_task=on_task)),
                    degenerate=degen)
    rows = [row(9, 9, True, False),      # clear and intact
            row(9, 9, True, False),      # clear and intact
            row(10, 9, True, True),      # clear because it collapsed
            row(0, 9, True, False)]      # nothing happened
    got = battery.steering_summary(rows)
    assert got["n"] == 4
    assert got["steering_success"]["count"] == 2
    assert got["concept_saturated_but_broken"]["count"] == 1
    assert got["any_concept"]["count"] == 3
    assert got["steering_success"]["ci_low"] < 0.5 < got["steering_success"]["ci_high"]


def test_the_instruct_axis_cannot_produce_a_one():
    """`on_task` is YES/NO, so this axis is coarser here than in AxBench, which has a middle
    value. Mapped to 0 or 2 so a fully on-task response is not docked half an axis."""
    assert battery.instruct_score(True) == 2
    assert battery.instruct_score(False) == 0


def test_the_original_twenty_two_task_prompts_are_still_the_prefix():
    """`battery_prompts` builds the effect channel as `TASK_PROMPTS[:N_EFFECT]`, so a prefix is
    what a run gets. The 44 added on 2026-08-23 were APPENDED for that reason: any reorder or
    insertion silently changes what an earlier N_EFFECT=22 run had measured, and every
    cross-run comparison with it."""
    first = [r["id"] for r in battery.TASK_PROMPTS[:22]]
    assert first == [
        "task_story", "task_landscape", "task_words", "task_notice", "task_advice",
        "task_explain", "task_water", "task_argument", "task_city", "task_letter2",
        "task_machine", "task_memory", "task_numbers", "task_question", "task_sound",
        "task_journey", "task_disagree", "task_rule", "task_list2", "task_time",
        "task_build", "task_quiet"]
    assert battery.TASK_PROMPTS[0]["text"] == "Tell me a short story."


def test_every_task_prompt_is_distinct_in_both_id_and_text():
    """Two prompts sharing an id join two different rows wherever a row is keyed on `unit`; two
    sharing text quietly halve the effective sample size of a channel whose whole n is a count
    of distinct questions."""
    ids = [r["id"] for r in battery.TASK_PROMPTS]
    texts = [r["text"].strip().lower() for r in battery.TASK_PROMPTS]
    assert len(set(ids)) == len(ids)
    assert len(set(texts)) == len(texts)
    assert len(battery.TASK_PROMPTS) == 66


def test_the_appended_block_keeps_every_prefix_register_balanced():
    """The 44 are round-robined across their registers, not grouped by register. An intermediate
    `N_EFFECT` takes a prefix, and a prefix of a register-sorted block would be eight narrative
    prompts and nothing else -- which is the failure the set is register-spread to avoid."""
    added = [r["id"] for r in battery.TASK_PROMPTS[22:]]
    assert len(added) == 44
    # the first seven added are one from each of the seven registers, in a fixed rotation
    assert added[:7] == ["task_stranger", "task_weather", "task_gravity", "task_uncertain",
                         "task_defend", "task_list3", "task_note"]
    # no register contributes twice before every register has contributed once
    assert len(set(added[:7])) == 7


class _FakeRun:
    """The two attributes runio needs to stamp and place a row."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.concept = "Garlic"
        self.config = {"config_hash": "testhash0000"}


def _fake_run_context(monkeypatch, tmp_path):
    from m2 import runio
    monkeypatch.setattr(runio, "_run", lambda: _FakeRun(tmp_path))
    return runio

# =====================================================================================
# The 2026-08-24 pod session: three ways a completed run was lost at the last step
# =====================================================================================

def test_a_record_separator_is_a_newline_and_nothing_else(tmp_path, monkeypatch):
    """U+2028 in one generation must not split its row into two unreadable ones.

    `write_row` dumps with ensure_ascii=False, so U+2028, U+2029 and U+0085 go into the file as
    themselves. `read_rows` used `splitlines()`, which breaks on all three -- so a single model
    response containing one of them became a head fragment starting with `{` and a tail fragment
    starting with whatever, and the tail raised. Zero occurrences across ~90,000 exported rows
    on Gemma3 and Qwen3, which is why it never fired; the planned run is about seven times that.
    """
    runio = _fake_run_context(monkeypatch, tmp_path)
    for sep in ("\u2028", "\u2029", "\u0085"):
        path = tmp_path / "responses_transcripts.jsonl"
        path.unlink(missing_ok=True)
        runio.write_row("responses_transcripts.jsonl",
                        dict(layer=53, response=f"a story{sep}with a separator in it"))
        runio.write_row("responses_transcripts.jsonl", dict(layer=54, response="an ordinary one"))
        rows = runio.read_rows("responses_transcripts.jsonl")
        assert len(rows) == 2, f"{sep!r} split a row"
        assert sep in rows[0]["response"], "the separator must survive the round trip"


def test_a_torn_final_row_is_still_tolerated_after_the_separator_fix(tmp_path, monkeypatch):
    """The fix must not cost the crash shape the tolerance exists for.

    `split("\n")` leaves a trailing empty element for the file's final newline, and if that
    element counts as the last line then a torn row above it is no longer last -- so the one
    survivable shape (a process killed mid-append) would start raising.
    """
    runio = _fake_run_context(monkeypatch, tmp_path)
    runio.write_row("cells.jsonl", dict(layer=53, dose=0.2))
    with open(tmp_path / "cells.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"layer": 54, "dos')          # killed mid-append, no newline
    rows = runio.read_rows("cells.jsonl")
    assert len(rows) == 1 and rows[0]["layer"] == 53

    # And a file whose rows all end in a newline reads every one of them.
    (tmp_path / "cells.jsonl").unlink()
    for layer in (53, 54, 55):
        runio.write_row("cells.jsonl", dict(layer=layer))
    assert [r["layer"] for r in runio.read_rows("cells.jsonl")] == [53, 54, 55]


def test_one_unreadable_line_cannot_destroy_a_run_that_measured_every_cell(tmp_path,
                                                                           monkeypatch):
    """The shape that killed the nine-cell Qwen run.

    `read_rows` tolerates an unparseable LAST line and raises on one anywhere else. So a file
    holding exactly one bad line reads as zero rows and NO error -- the strict read at the top of
    the sweep passes -- and then the run's own appends put good rows after it, and the re-read at
    the end raises. Every cell measured, every row written, and the run reported FAILED.

    The re-read is gone: the count comes from memory. This test pins the property that made the
    re-read fatal, so that restoring it would fail here rather than on a pod three hours in.
    """
    runio = _fake_run_context(monkeypatch, tmp_path)
    path = tmp_path / "cells.jsonl"
    path.write_text("\x00\x00\x00\n", encoding="utf-8")   # one bad line, and it is last

    assert runio.read_rows("cells.jsonl") == [], "a lone bad line reads as zero rows, no error"

    runio.write_row("cells.jsonl", dict(layer=53))
    with pytest.raises(RuntimeError, match="corruption rather than a torn append"):
        runio.read_rows("cells.jsonl")


def test_the_volume_probe_refuses_a_short_write(tmp_path, monkeypatch):
    """A run had no disk check at all, and a full volume surfaced as `no norms were measured`."""
    from m3 import run as run_module

    info = run_module.check_volume_writable(tmp_path, probe_mb=1)
    assert info["probe_mb"] == 1 and info["free_gb"] > 0
    assert not (tmp_path / ".write_probe").exists(), "the probe must clean up after itself"

    real_open = open

    def short_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        if str(args[0]).endswith(".write_probe"):
            handle.write = lambda data: len(data)      # claims success, writes nothing
        return handle

    monkeypatch.setattr("builtins.open", short_open)
    with pytest.raises(RuntimeError, match="volume is full"):
        run_module.check_volume_writable(tmp_path, probe_mb=1)


def test_a_resume_over_a_torn_row_does_not_poison_the_file(tmp_path, monkeypatch):
    """The sequence that destroys a long run, start to finish.

    Crash mid-append leaves a torn final row. The next read tolerates it and leaves the bytes.
    The resumed run appends after it. From then on the torn row is not last, so every reader
    raises -- on a file holding thousands of good rows. `heal_torn_tail` runs before the resumed
    run appends, so the sequence terminates at the tolerate step instead of arming a trap.
    """
    from m2 import runio

    runio = _fake_run_context(monkeypatch, tmp_path)
    runio.write_row("responses_transcripts.jsonl", dict(layer=53, response="one"))
    runio.write_row("responses_transcripts.jsonl", dict(layer=53, response="two"))
    with open(tmp_path / "responses_transcripts.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"layer": 53, "respo')                # killed here

    # Without healing, this is the trap: tolerated now, fatal after one more append.
    assert len(runio.read_rows("responses_transcripts.jsonl")) == 2

    removed = runio.heal_torn_tail("responses_transcripts.jsonl")
    assert removed == 20, "the torn row's bytes, and only those"

    runio.write_row("responses_transcripts.jsonl", dict(layer=54, response="three"))
    rows = runio.read_rows("responses_transcripts.jsonl")
    assert [r["response"] for r in rows] == ["one", "two", "three"]

    quarantine = tmp_path / "responses_transcripts.jsonl.quarantine"
    assert '{"layer": 53, "respo' in quarantine.read_text(encoding="utf-8"), (
        "the removed bytes are kept, not deleted")


def test_healing_leaves_a_healthy_file_byte_identical(tmp_path, monkeypatch):
    """It runs before every resume, so it must be a no-op on the overwhelmingly common case."""
    from m2 import runio

    runio = _fake_run_context(monkeypatch, tmp_path)
    for layer in (53, 54, 55):
        runio.write_row("cells.jsonl", dict(layer=layer))
    before = (tmp_path / "cells.jsonl").read_bytes()

    assert runio.heal_torn_tail("cells.jsonl") == 0
    assert (tmp_path / "cells.jsonl").read_bytes() == before
    assert not (tmp_path / "cells.jsonl.quarantine").exists()
    assert runio.heal_torn_tail("nothing_here.jsonl") == 0, "a missing file is not an error"


def test_healing_refuses_to_touch_corruption_in_the_middle(tmp_path, monkeypatch):
    """A bad line that is NOT last is what `read_rows` refuses to resume from.

    Healing it would quietly edit away the evidence for the one failure mode the strict read
    exists to catch. The last row is good here, so heal must do nothing at all and the strict
    read must still raise.
    """
    from m2 import runio

    runio = _fake_run_context(monkeypatch, tmp_path)
    path = tmp_path / "cells.jsonl"
    path.write_text('not json at all\n{"layer": 54}\n', encoding="utf-8")

    assert runio.heal_torn_tail("cells.jsonl") == 0
    assert path.read_text(encoding="utf-8").startswith("not json at all")
    with pytest.raises(RuntimeError, match="corruption rather than a torn append"):
        runio.read_rows("cells.jsonl")


def test_a_terminated_but_unreadable_final_row_is_healed_too(tmp_path, monkeypatch):
    """The 2026-08-24 shape: a final line that ends in a newline and still will not parse.

    Ambiguous in origin -- it cannot have come from `write_row`, whose output always starts with
    `{`. Healed anyway, because leaving it is what armed the trap that killed a completed run,
    and logged loudly because something wrote bytes this pipeline cannot account for.
    """
    from m2 import runio

    runio = _fake_run_context(monkeypatch, tmp_path)
    path = tmp_path / "cells.jsonl"
    path.write_bytes(bytes([0, 0, 0]) + b"\n")

    assert runio.heal_torn_tail("cells.jsonl") == 3
    runio.write_row("cells.jsonl", dict(layer=53))
    assert [r["layer"] for r in runio.read_rows("cells.jsonl")] == [53]


def test_every_appended_artefact_is_on_the_heal_list():
    """A new artefact that nobody adds to the list is a new way to lose a resume."""
    import re
    from pathlib import Path as _Path

    from m3 import sweep as sweep_module

    # Every module that appends, not just sweep.py. The first version of this test read
    # sweep.py alone, so m3/run.py's provenance.jsonl write was invisible to it -- the file
    # happened to be on the list, so the test passed while checking nothing about it. That is
    # the "check that cannot fail" shape AGENTS.md names.
    # By PATH, not by import: m2/expensive.py and m2/vectors.py import torch, which is not
    # installed in the offline test environment. A guard that can only run where the GPU stack
    # is present is a guard that never runs.
    repo = _Path(sweep_module.__file__).resolve().parents[1]
    scanned = [repo / "m3" / "sweep.py", repo / "m3" / "run.py", repo / "m3" / "freerun.py",
               repo / "m2" / "expensive.py", repo / "m2" / "vectors.py"]
    source = "\n".join(f.read_text(encoding="utf-8") for f in scanned if f.exists())
    written = set(re.findall(r'write_row\(\s*"([^"]+\.jsonl)"', source))
    written |= {v for k, v in vars(sweep_module).items()
                if k.endswith("_FILE") and isinstance(v, str) and v.endswith(".jsonl")}
    constants = {getattr(sweep_module, k) for k in dir(sweep_module)
                 if k.endswith("_FILE") and isinstance(getattr(sweep_module, k), str)}
    written |= {c for c in constants if c.endswith(".jsonl")}
    missing = sorted(written - set(sweep_module.APPEND_ONLY_ARTEFACTS))
    assert not missing, f"appended but not healed on resume: {missing}"


def test_every_row_names_the_attempt_that_wrote_it(tmp_path, monkeypatch):
    """Two passes over one run directory must be distinguishable in the transcripts.

    A crash between a cell's response rows and its cells.jsonl row makes the resume re-measure
    that cell and append a second full battery. cells.jsonl is unaffected, so the run's summary
    is right -- but this project keeps every transcript so operating points can be chosen offline
    later, and that analysis would count the cell twice with nothing marking it.
    """
    from m2 import runio

    runio = _fake_run_context(monkeypatch, tmp_path)
    first = runio.begin_attempt()
    runio.write_row("responses_transcripts.jsonl",
                    dict(layer=53, dose=0.2, channel="effect", unit="task_x", response="a"))

    monkeypatch.setattr(runio, "_now", lambda: "2026-08-24T12:00:00+00:00")
    second = runio.begin_attempt()
    runio.write_row("responses_transcripts.jsonl",
                    dict(layer=53, dose=0.2, channel="effect", unit="task_x", response="b"))

    rows = runio.read_rows("responses_transcripts.jsonl")
    assert len(rows) == 2, "both attempts are on disk; nothing is overwritten"
    assert {r["attempt"] for r in rows} == {first, second}
    assert second != first


def test_rescore_keeps_only_the_latest_attempt():
    """The dedupe rule offline analysis needs, and the count it must report."""
    from tools import rescore

    rows = [
        dict(layer=53, dose=0.2, channel="effect", unit="task_a", response="crashed",
             attempt="2026-08-24T10:00:00+00:00"),
        dict(layer=53, dose=0.2, channel="effect", unit="task_a", response="resumed",
             attempt="2026-08-24T11:00:00+00:00"),
        dict(layer=53, dose=0.2, channel="effect", unit="task_b", response="only once",
             attempt="2026-08-24T10:00:00+00:00"),
    ]
    kept, dropped = rescore.dedupe_attempts(rows)
    assert dropped == 1
    by_unit = {r["unit"]: r["response"] for r in kept}
    assert by_unit == {"task_a": "resumed", "task_b": "only once"}

    # An export written before `attempt` existed must survive unchanged rather than collapsing.
    old = [dict(layer=53, dose=0.2, channel="effect", unit=f"task_{i}") for i in range(3)]
    kept, dropped = rescore.dedupe_attempts(old)
    assert dropped == 0 and len(kept) == 3


def test_a_failing_read_this_bundle_does_not_cost_the_export():
    """The digest is the last thing a 3.4-hour run does, and archive/export sit outside the try.

    So an exception in `_read_bundle` never cost a markdown file -- it cost the archive and the
    zip, and nothing left the pod. summary.json is already written by then and every measured row
    is already on disk; there is no version of "the digest failed" that should also mean "you get
    no export".
    """
    import inspect

    from m3 import sweep as sweep_module

    source = inspect.getsource(sweep_module.run_sweep)
    call = source.index("_read_bundle(concept, cfg)")
    before = source[:call]
    assert before.rstrip().endswith("try:"), (
        "_read_bundle must be inside a try; an exception there forfeits the export")
    after = source[call:]
    assert "except Exception" in after[:400], "and the except must be right there"
    assert "write_json(SUMMARY_FILE" in before, (
        "summary.json must already be written before the digest is attempted")


def test_repair_finds_a_buried_bad_line_and_leaves_the_good_rows(tmp_path):
    """The 2026-08-24 shape: one unreadable line, then nine cells nobody can read."""
    from tools import repair_jsonl

    path = tmp_path / "cells.jsonl"
    path.write_bytes(bytes([0, 0, 0]) + b"\n" + b"\n".join(
        ('{"layer": %d}' % L).encode() for L in range(52, 61)) + b"\n")

    bad = repair_jsonl.bad_lines(path)
    assert [n for n, _ in bad] == [1]
    assert bad[0][1] == bytes([0, 0, 0]), "the raw bytes, so an operator can see what happened"

    assert repair_jsonl.repair(path, apply=False) == (1, 9)
    assert path.read_bytes().startswith(bytes([0, 0, 0])), "a report must change nothing"

    assert repair_jsonl.repair(path, apply=True) == (1, 9)
    from m2 import runio
    lines = [l for l in path.read_bytes().split(b"\n") if l.strip()]
    assert len(lines) == 9
    assert bytes([0]) not in path.read_bytes()
    quarantine = tmp_path / "cells.jsonl.quarantine"
    assert bytes([0, 0, 0]) in quarantine.read_bytes(), "removed bytes are kept, never deleted"


def test_repair_does_not_call_a_unicode_separator_corruption(tmp_path):
    """U+2028 inside a generation is content, not a record separator.

    `splitlines()` breaks on it; `write_row`'s only separator is "\n". A repair tool that used
    splitlines() would report a perfectly good row as corruption and invite an operator to
    delete real data.
    """
    from tools import repair_jsonl

    path = tmp_path / "responses_transcripts.jsonl"
    path.write_text('{"response": "a story\u2028with a separator"}\n'
                    '{"response": "ordinary"}\n', encoding="utf-8")
    assert repair_jsonl.bad_lines(path) == []
    assert repair_jsonl.repair(path, apply=False) == (0, 2)


def test_repair_refuses_a_target_that_does_not_exist(tmp_path):
    from tools import repair_jsonl

    with pytest.raises(SystemExit):
        repair_jsonl.main([str(tmp_path / "nope")])
    with pytest.raises(SystemExit, match="no .jsonl"):
        repair_jsonl.main([str(tmp_path)])
