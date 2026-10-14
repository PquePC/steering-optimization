"""m5.judges.parse: strict readers. Anything unreadable is missing, never a coerced label
(read_judges J-01, J-10); a YES needs a quote that is really in the text (plan §7.1)."""

from __future__ import annotations

import pytest

from m5.judges import parse as Q
from m5.judges import prompts as P

SOURCE = "I really love “garlic bread” — it’s great.\n\nAnd   more   text here."


# The old parsers' coercions, each of which entered a rate as a fabricated answer (J-10).
@pytest.mark.parametrize("reply", [
    "Answer: YES or NO", "Answer: Yesterday", "Answer: Not sure", "Answer: NONE",
    "Answer: Nope, different", "Answer: <YES or NO>", "Answer:", "", "   ",
    "I think it does.", "Answer: Yes, there is no garlic", "Answer: YES and NO",
])
def test_malformed_yes_no_replies_are_missing_never_coerced(reply):
    p = Q.parse_yesno(reply)
    assert p.label is None and p.missing
    assert p.missing.split(":")[0] in ("unparseable", "template_echo", "empty_reply")


def test_markdown_prose_and_the_last_occurrence_are_handled():
    assert Q.parse_yesno("**Answer:** YES").label is True
    assert Q.parse_yesno("- Answer: no").label is False
    assert Q.parse_yesno("> 1. **Answer**: No.").label is False
    p = Q.parse_yesno("Let me think.\nAnswer: NO\nOn reflection:\nAnswer: YES")
    assert p.label is True and "conflicting_duplicates:answer" in p.notes
    p = Q.parse_yesno("Answer: No, it never mentions it.")
    assert p.label is False and "trailing_text" in p.notes
    assert Q.parse_yesno("YES.").label is True and "bare_answer" in Q.parse_yesno("YES.").notes


def test_parsed_is_a_label_or_a_reason_never_both():
    with pytest.raises(ValueError):
        Q.Parsed(label=None)
    with pytest.raises(ValueError):
        Q.Parsed(label=True, missing="x")


@pytest.fixture
def allusion(cfg):
    return P.templates(cfg)["allusion"]


def test_a_quote_not_in_the_text_turns_yes_into_no(allusion):
    p = Q.parse_reply(allusion, "Answer: YES\nQuote: onion soup", source=SOURCE)
    assert p.ok and p.label is False
    assert p.fields["downgraded"] == "quote_invalid" and p.fields["quote_check"] == "not_in_text"
    assert p.fields["answer"] == "YES"          # what the judge said is kept
    assert "quote_invalid:not_in_text" in p.notes


@pytest.mark.parametrize("reply,check", [
    ("Answer: YES", "absent"),
    ("Answer: YES\nQuote: NONE", "none"),
    ("Answer: YES\nQuote: ...", "none"),
    ("Answer: YES\nQuote: " + " ".join(["garlic"] * 16), "too_long"),
])
def test_absent_empty_or_overlong_quotes_are_invalid(allusion, reply, check):
    p = Q.parse_reply(allusion, reply, source=SOURCE)
    assert p.label is False and p.fields["quote_check"] == check


@pytest.mark.parametrize("quote", [
    '"garlic bread"', "“GARLIC BREAD”", "garlic   bread", "it's great", "it’s great.",
    "...really love “garlic bread”", "*garlic bread*", "and more text",
])
def test_quotes_match_after_normalising_case_whitespace_and_curly_quotes(allusion, quote):
    p = Q.parse_reply(allusion, f"Answer: YES\nQuote: {quote}", source=SOURCE)
    assert p.label is True, (quote, p.fields)
    assert p.fields["quote_valid"] and isinstance(p.fields["quote_at"], int)


def test_quote_offset_points_into_the_original_text():
    at = Q.locate("garlic BREAD", SOURCE)
    assert SOURCE[at:at + 6] == "garlic"
    assert Q.locate("more text", SOURCE) == SOURCE.index("more")
    assert Q.locate("absent words", SOURCE) is None


def test_a_no_needs_no_quote(allusion):
    p = Q.parse_reply(allusion, "Answer: NO\nQuote: NONE", source=SOURCE)
    assert p.label is False and "downgraded" not in p.fields


def test_quote_templates_refuse_to_parse_without_their_source(allusion):
    with pytest.raises(ValueError, match="source"):
        Q.parse_reply(allusion, "Answer: YES\nQuote: x")


@pytest.mark.parametrize("reply,label", [
    ("Name: garlic", "garlic"), ("Name: **Garlic.**", "Garlic"), ("Name: NONE", "NONE"),
    ("Name: none", "NONE"), ("Name: 大蒜", "大蒜"), ("Name: garlic bread", "garlic bread"),
])
def test_names_are_recorded_as_written(reply, label):
    assert Q.parse_name(reply).label == label


@pytest.mark.parametrize("reply", ["Name: the word or concept, or NONE", "Name:", "garlic",
                                   "Name: this is a long sentence about it", "Name: <word>"])
def test_unreadable_names_are_missing(reply):
    assert Q.parse_name(reply).missing


@pytest.mark.parametrize("reply,label", [("73", 73.0), ("Score: 55.5", 55.5), ("REFUSAL", "REFUSAL"),
                                         ("50.", 50.0), ("0", 0.0), ("100", 100.0)])
def test_scores(reply, label):
    p = Q.parse_score(reply)
    assert p.label == label
    assert p.fields["refusal"] is (label == "REFUSAL")


@pytest.mark.parametrize("reply", ["<0-100>", "4-6", "150", "I think 50", "", "fifty"])
def test_unreadable_scores_are_missing_not_zero(reply):
    p = Q.parse_score(reply)
    assert p.label is None and p.missing


def test_json_replies_are_validated_strictly(cfg, allusion):
    T = P.templates(cfg)
    ok = Q.parse_reply(allusion, '{"answer": "YES", "quote": "garlic bread"}', fmt="json", source=SOURCE)
    assert ok.label is True
    bad_quote = Q.parse_reply(allusion, '{"answer": "YES", "quote": "onion"}', fmt="json", source=SOURCE)
    assert bad_quote.label is False and bad_quote.fields["downgraded"] == "quote_invalid"
    for text in ('{"answer": "MAYBE", "quote": "x"}', '{"answer": "YES"}', "not json",
                 '{"answer": "YES", "quote": "x", "extra": 1}', "[1]"):
        assert Q.parse_reply(allusion, text, fmt="json", source=SOURCE).missing, text
    assert Q.parse_reply(T["named"], '{"name": "silk"}', fmt="json").label == "silk"
    assert Q.parse_reply(T["trait"], '{"refusal": false, "score": 42}', fmt="json").label == 42.0
    assert Q.parse_reply(T["trait"], '{"refusal": true, "score": 0}', fmt="json").label == "REFUSAL"
    assert Q.parse_reply(T["trait"], '{"refusal": false, "score": 420}', fmt="json").missing
