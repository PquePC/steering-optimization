"""m5.judges.prompts: atomic, arm-blind, versioned questions (plan §7.1).

Each guard has a test that trips it: a rubric that leaks the experiment, a template edited
without a version bump, a concept handed to the concept-blind coherence template, a harmful
concept handed to a judge.
"""

from __future__ import annotations

import dataclasses

import pytest

from m5 import safety
from m5.judges import prompts as P

# The words the brief names, plus the old prompts' worst offenders (read_judges §2.3).
FORBIDDEN = ("inject", "injected", "injection", "steer", "steering", "experiment")


def _ours(t: P.Template) -> str:
    return " ".join([t.rubric, t.format_block("lines"), t.format_block("json"), t.prefill]
                    + [s.label for s in t.slots]).casefold()


def test_no_template_rubric_mentions_injection_steering_or_the_experiment(cfg):
    for t in P.templates(cfg).values():
        text = _ours(t)
        for w in FORBIDDEN:
            assert w not in text, f"{t.id} rubric contains {w!r}"


def test_the_blindness_check_trips_on_a_leaky_rubric():
    leaky = dataclasses.replace(P.ALLUSION, rubric=P.ALLUSION.rubric + " Response B was steered.")
    assert "steer" in P.blindness_violations(leaky)
    with pytest.raises(P.TemplateDrift, match="never told the experiment"):
        P.check_templates({"allusion": leaky}, P.PINNED)
    label_leak = dataclasses.replace(P.DOMINANT, slots=(P.Slot("concept", "CONCEPT", "word"),
                                                         P.Slot("text", "INJECTED TEXT", "text")))
    assert P.blindness_violations(label_leak) == ["inject"]


def test_the_payload_may_contain_the_subjects_own_words(cfg):
    """The stoplist is about what we write; a reply saying 'injected' is evidence, not context."""
    T = P.templates(cfg)
    it = P.make_item(T["claim"], {"question": "Do you detect an injected thought?",
                                  "text": "Yes, an injected thought about the sea."}, item_id="t1")
    assert "injected thought about the sea" in it.messages()[0]["content"]


def test_the_coherence_template_cannot_take_a_concept(cfg):
    T = P.templates(cfg)
    assert "concept" not in T["coherent"].slot_names
    with pytest.raises(P.TemplateError, match="concept-blind"):
        P.make_item(T["coherent"], {"prompt": "Tell me a story.", "text": "Once.", "concept": "silk"},
                    item_id="t1")


def test_rubric_first_payload_last_and_the_prefix_is_identical_across_items(cfg):
    T = P.templates(cfg)
    a = P.make_item(T["allusion"], {"concept": "garlic", "text": "one"}, item_id="1")
    b = P.make_item(T["allusion"], {"concept": "silk", "text": "two"}, item_id="2")
    prefix = T["allusion"].static_prefix("lines")
    for it in (a, b):
        content = it.messages()[0]["content"]
        assert content.startswith(prefix)
        assert content.rstrip().endswith("[END TEXT]")
    assert "garlic" not in prefix and "silk" not in prefix
    assert a.messages()[0]["content"] != b.messages()[0]["content"]


def test_one_user_message_and_no_system_turn(cfg):
    it = P.make_item(P.templates(cfg)["dominant"], {"concept": "silk", "text": "x"}, item_id="1")
    msgs = it.messages()
    assert [m["role"] for m in msgs] == ["user"]


def test_every_template_is_pinned_and_import_checks_held():
    assert set(P.PINNED) == set(P.TEMPLATES)
    P.check_templates(P.TEMPLATES, P.PINNED)


def test_editing_a_template_without_a_version_bump_and_new_pin_raises():
    edited = dataclasses.replace(P.DOMINANT, rubric=P.DOMINANT.rubric + " Be careful.")
    with pytest.raises(P.TemplateDrift, match="new version"):
        P.check_templates({"dominant": edited}, P.PINNED)
    with pytest.raises(P.TemplateDrift):
        P.check_templates({"dominant": dataclasses.replace(edited, version=2)}, P.PINNED)
    with pytest.raises(P.TemplateDrift, match="no pinned hash"):
        P.check_templates({"dominant": P.DOMINANT}, {})


def test_binding_the_quote_limit_changes_the_rendered_hash_not_the_text_hash(cfg):
    a = P.templates(cfg)["allusion"]
    b = P.templates(cfg.replace(QUOTE_MAX_WORDS=10))["allusion"]
    assert a.text_hash == b.text_hash and a.hash != b.hash
    assert f"at most {cfg['QUOTE_MAX_WORDS']} words" in a.render({"concept": "x", "text": "y"})
    with pytest.raises(P.TemplateError, match="unbound"):
        P.ALLUSION.render({"concept": "x", "text": "y"})
    with pytest.raises(P.TemplateError, match="bind the template"):
        P.make_item(P.ALLUSION, {"concept": "x", "text": "y"}, item_id="1")


def test_missing_unknown_empty_and_multiline_word_slots_raise(cfg):
    T = P.templates(cfg)
    with pytest.raises(P.TemplateError, match="missing slot"):
        P.make_item(T["allusion"], {"text": "y"}, item_id="1")
    with pytest.raises(P.TemplateError, match="unknown slot"):
        P.make_item(T["allusion"], {"concept": "x", "text": "y", "arm": "steered"}, item_id="1")
    with pytest.raises(P.TemplateError, match="cannot be empty"):
        P.make_item(T["allusion"], {"concept": "  ", "text": "y"}, item_id="1")
    with pytest.raises(P.TemplateError, match="one line"):
        P.make_item(T["same_concept"], {"word_a": "a\nb", "word_b": "c"}, item_id="1")
    with pytest.raises(P.TemplateError, match="item_id"):
        P.make_item(T["dominant"], {"concept": "x", "text": "y"}, item_id=" ")


def test_harmful_concepts_are_refused_at_the_judge_entry_point(cfg):
    T = P.templates(cfg)
    with pytest.raises(safety.Refused):
        P.make_item(T["allusion"], {"concept": "Weapons", "text": "x"}, item_id="1")
    with pytest.raises(safety.Refused):
        P.make_item(T["same_concept"], {"word_a": "toxin", "word_b": "poison"}, item_id="1")
    # what the subject named is its text, not our choice of concept: not refused
    P.make_item(T["same_concept"], {"word_a": "weapon", "word_b": "garlic"}, item_id="1")


def test_item_key_is_content_addressed_and_meta_never_reaches_the_prompt(cfg):
    T = P.templates(cfg)
    a = P.make_item(T["dominant"], {"concept": "silk", "text": "x"}, item_id="cell-7:steered",
                    meta={"arm": "steered", "dose": 0.3})
    b = P.make_item(T["dominant"], {"concept": "silk", "text": "x"}, item_id="other")
    assert a.key == b.key
    content = a.messages()[0]["content"]
    assert "steered" not in content and "0.3" not in content and "cell-7" not in content
    c = P.make_item(T["dominant"], {"concept": "silk", "text": "y"}, item_id="x")
    assert c.key != a.key
