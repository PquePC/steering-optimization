"""m5.judges.matching: decision 7's identification rules, computed in code wherever code can
decide, and routed to the judge everywhere else (plan §7.2, §13)."""

from __future__ import annotations

import pytest

from m5.judges import matching as M


@pytest.mark.parametrize("named,target,how", [
    ("Garlic", "garlic", "exact"), ("the word 'garlic'", "garlic", "exact"),
    ("Café", "cafe", "exact"), ("wrists", "wrist", "inflection"),
    ("wrist", "wrists", "inflection"), ("berries", "berry", "inflection"),
    ("boxes", "box", "inflection"), ("tomatoes", "tomato", "inflection"),
    ("glasses", "glass", "inflection"), ("mice", "mouse", "inflection"),
    ("garlic's", "garlic", "inflection"), ("couch", "sofa", "synonym"),
    ("automobiles", "car", "synonym"), ("amphitheatre", "amphitheater", "synonym"),
])
def test_code_matches_exact_inflection_and_curated_synonyms(named, target, how):
    assert M.code_match(named, target) == how
    ident = M.identify(named, target)
    assert ident.matches is True and ident.loose is True and ident.how == how and not ident.needs


@pytest.mark.parametrize("named,target", [("news", "new"), ("onion", "garlic"),
                                          ("pineapple", "apple"), ("hand", "wrists")])
def test_code_never_says_yes_to_what_is_not_the_same_word(named, target):
    assert M.code_match(named, target) is None


def test_non_latin_names_go_to_the_judge_never_matched_or_rejected_by_spelling():
    ident = M.identify("大蒜", "garlic")                 # garlic, in Chinese
    assert ident.matches is None and ident.how == "needs_judge:non_latin"
    assert ident.needs == (("same_concept", {"word_a": "大蒜", "word_b": "garlic"}),)
    assert M.has_non_latin("大蒜") and not M.has_non_latin("café")
    judged = M.identify("大蒜", "garlic", same_concept=True)
    assert judged.matches is True and judged.loose is True and judged.how == "judge_same"


def test_nothing_named_is_a_no_on_every_rule():
    for named in (None, "NONE", "none", ""):
        ident = M.identify(named, "garlic")
        assert ident.matches is False and ident.loose is False and ident.how == "none_named"


def test_loose_rule_uses_shared_roots_in_code_and_the_related_judge_otherwise():
    w = M.identify("wristwatches", "wrists")          # the Wrists case (RESULTS-WRISTS §3.1)
    assert w.loose is True and w.matches is None and w.needs[0][0] == "same_concept"
    w = M.identify("wristwatches", "wrists", same_concept=False)
    assert w.matches is False and w.loose is True and not w.needs
    h = M.identify("hand", "wrists", same_concept=False)
    assert h.loose is None and h.needs == (("related", {"word_a": "hand", "word_b": "wrists"}),)
    assert M.identify("hand", "wrists", same_concept=False, related=True).loose is True
    assert M.identify("hand", "wrists", same_concept=False, related=False).loose is False
    assert M.shares_root("garlic bread", "garlic") and not M.shares_root("pineapple", "apple")
    assert not M.shares_root("cart", "car")           # under ROOT_MIN_CHARS


def test_the_judge_is_asked_same_concept_first_and_related_only_if_needed():
    first = M.identify("onion", "garlic")
    assert [t for t, _ in first.needs] == ["same_concept"] and first.loose is None
    assert M.identify("onion", "garlic", same_concept=True).needs == ()


def test_semantic_score_is_continuous_and_absent_without_an_embedding():
    assert M.identify("onion", "garlic").semantic is None
    vecs = {"onion": [1.0, 1.0, 0.0], "garlic": [1.0, 0.9, 0.1]}
    s = M.identify("onion", "garlic", embed=lambda w: vecs[w]).semantic
    assert 0.9 < s < 1.0
    assert M.cosine([0.0, 0.0], [1.0, 0.0]) is None


def test_identify_needs_a_target():
    with pytest.raises(ValueError):
        M.identify("garlic", "  ")
