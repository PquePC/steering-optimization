"""Offline invariants for m3. No GPU, no model, no judge key.

Each test pins a specific way M2 produced a plausible wrong number, or a specific sentence of
`docs/M3-DESIGN.md` that the code could satisfy loosely and still be wrong.
"""

from __future__ import annotations

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


def test_transcripts_are_gated_on_a_benign_concept_list():
    assert config.is_benign("Garlic") and config.is_benign("garlic")
    assert not config.is_benign("weapon")


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


@pytest.mark.parametrize("jid,text", [
    ("effect", "Influence: eleven\nForm: dominant"),
    ("effect", "Influence: 5\nForm: strong"),
    ("effect", "Influence: 42\nForm: trace"),
    ("identify", "Matches: MAYBE"),
    ("coherence", "OnTask: YES"),
])
def test_a_bad_judge_answer_raises_and_is_never_defaulted(jid, text):
    """A defaulted 0 influence is indistinguishable from a real measurement of no influence —
    and 'no influence' is what a covert operating point looks like."""
    with pytest.raises((judge.JudgeParseError, ValueError)):
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
