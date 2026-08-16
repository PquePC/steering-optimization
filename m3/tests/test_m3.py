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
    cfg = dict(config.SETTINGS)
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


def test_the_judge_reply_cap_is_tighter_than_m2s_and_is_pushed_into_the_transport():
    """Output tokens cost 4x input, so the reply cap is where the money is."""
    assert config.CONFIG["JUDGE_MAX_TOKENS"] < 400
    from m2 import judges as transport
    before = transport.JUDGE_MAX_TOKENS
    try:
        info = judge.configure_transport(dict(config.SETTINGS))
        assert transport.JUDGE_MAX_TOKENS == config.CONFIG["JUDGE_MAX_TOKENS"]
        assert info["judge_temperature"] == 0.0, "judging must stay deterministic"
    finally:
        transport.JUDGE_MAX_TOKENS = before


def test_worst_case_payload_stays_small_enough_to_price():
    """The guard is against a 4000-token payload nobody predicted."""
    for jid in judge.JUDGE_IDS:
        assert judge.estimate_payload_tokens(jid, dict(config.SETTINGS)) < 1000


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


def test_the_cost_estimate_scales_with_the_grid():
    cfg = dict(config.SETTINGS)
    full = m3run.estimate(62, cfg)
    assert full["cells"] == 49 * len(cfg["DOSE_FRACTIONS"])
    thin = m3run.estimate(62, dict(cfg, LAYER_STRIDE=2))
    assert thin["cells"] < full["cells"]
    assert thin["judge_usd"] < full["judge_usd"]
    assert full["judge_usd"] < 5.0, "a run this size should not cost five dollars"


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
