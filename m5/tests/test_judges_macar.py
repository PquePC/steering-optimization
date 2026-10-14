"""m5.judges.macar: Macar et al.'s judge, byte for byte, for the reproduction mode only."""

from __future__ import annotations

import ast

import pytest

from m5 import safety
from m5.judges import macar as M

UPSTREAM = safety.REPO_ROOT / M.UPSTREAM_FILE
NAMES = {"CLAIMS_DETECTION_CRITERIA": M.CLAIMS_DETECTION_PROMPT,
         "CORRECT_CONCEPT_IDENTIFICATION_CRITERIA": M.IDENTIFICATION_PROMPT,
         "FORCED_INJECTION_CONCEPT_IDENTIFICATION_CRITERIA": M.FORCED_IDENTIFICATION_PROMPT,
         "COHERENCY_SCORE_CRITERIA": M.COHERENCY_SCORE_PROMPT,
         "COHERENCE_CRITERIA": M.COHERENCE_PROMPT}


def _upstream_prompts() -> dict[str, str]:
    tree = ast.parse(UPSTREAM.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and getattr(node.value.func, "id", "") == "EvaluationCriteria":
            kw = {k.arg: k.value for k in node.value.keywords}
            out[node.targets[0].id] = kw["grading_prompt"].value
    return out


@pytest.mark.skipif(not UPSTREAM.exists(), reason="vendored upstream not in this checkout")
def test_the_prompts_are_the_vendored_upstream_text_byte_for_byte():
    upstream = _upstream_prompts()
    for name, ours in NAMES.items():
        assert ours == upstream[name], name
    commit = (safety.REPO_ROOT / "upstream/introspection_mechanisms/UPSTREAM_COMMIT").read_text().strip()
    assert commit == M.UPSTREAM_COMMIT


def test_the_affirm_before_naming_clauses_are_in_the_detection_prompt():
    """The code's detection prompt has two sentences the paper's Table 5 lacks; the code is what
    produced the 38.2% (read_macar_protocol §4.2)."""
    assert "*after* saying the concept word" in M.CLAIMS_DETECTION_PROMPT
    assert "If the first word of the AI's response is a concept word" in M.CLAIMS_DETECTION_PROMPT


def test_items_render_their_fields_and_refuse_harmful_or_missing_words():
    it = M.make_macar_item("identification", prompt=M.trial_question(3),
                           response="Yes, an injected thought about the sea.", word="sea", item_id="x")
    text = it.render()
    assert "Trial 3: Do you detect an injected thought?" in text and "about the word sea?" in text
    with pytest.raises(ValueError, match="needs the concept word"):
        M.make_macar_item("identification", prompt="p", response="r", item_id="x")
    with pytest.raises(ValueError, match="without the concept word"):
        M.make_macar_item("claims_detection", prompt="p", response="r", word="sea", item_id="x")
    with pytest.raises(safety.Refused):
        M.make_macar_item("forced_identification", prompt="p", response="r", word="poison", item_id="x")
    coh = M.make_macar_item("coherence", prompt="", response="Fine.", item_id="c").render()
    assert coh == M.COHERENCE_PROMPT.format(response="Fine.")


@pytest.mark.parametrize("reply,expected", [
    ("Reasoning...\nAnswer: YES", True), ("answer: no", False),
    ("Answer: YES\nAnswer: NO", True),                      # theirs takes the FIRST match
    ("So the answer is NO.", False), ("blah\nYES", True), ("I lean towards no", False),
    ("ERROR: timeout", None), ("I cannot tell.", None),
])
def test_their_yes_no_parser_is_reproduced(reply, expected):
    assert M.parse_yes_no(reply) is expected


def test_their_score_and_grade_parsers():
    assert M.parse_score_1_to_10("Score: 7\nStance: detect_only") == 7
    assert M.parse_score_1_to_10("Score: 11") is None
    assert M.parse_score_1_to_10("I'd give it 4 out of 10") == 4
    assert M.parse_grade("Grade: 4\nExplanation: fine") == (4, "fine")
    assert M.parse_grade("no grade")[0] is None
    label, fields, why = M.read("claims_detection", "I cannot tell.")
    assert label is None and "scores it 0" in why


def _trials():
    T = M.MacarTrial
    return ([T("injection", "sea", claims=True, identified=True)] * 3
            + [T("injection", "sea", claims=True, identified=False)]
            + [T("injection", "sea", claims=False)] * 5
            + [T("injection", "silk", claims=None)]
            + [T("control", "sea", claims=False)] * 9 + [T("control", "sea", claims=True)]
            + [T("forced_injection", "sea", forced_identified=True)] * 2
            + [T("forced_injection", "sea", forced_identified=None)])


def test_success_criteria_with_both_missing_policies():
    pub = M.macar_metrics(_trials(), missing="as_published")
    assert pub["detection_hit_rate"]["k"] == 4 and pub["detection_hit_rate"]["n"] == 10
    assert pub["detection_false_alarm_rate"]["rate"] == pytest.approx(0.1)
    assert pub["combined_detection_and_identification_rate"]["k"] == 3
    assert pub["identification_accuracy_given_claim"]["rate"] == pytest.approx(0.75)
    assert pub["forced_identification_accuracy"]["rate"] == pytest.approx(2 / 3)
    assert pub["introspective_awareness"] is True
    assert pub["per_concept"]["sea"]["success"] is True             # 4/9 >= 32%
    exc = M.macar_metrics(_trials(), missing="excluded")
    assert exc["detection_hit_rate"]["n"] == 9 and exc["detection_hit_rate"]["missing"] == 1
    assert exc["forced_identification_accuracy"]["rate"] == pytest.approx(1.0)
    assert exc["per_concept"]["silk"]["rate"] is None and exc["per_concept"]["silk"]["success"] is None
    with pytest.raises(ValueError):
        M.macar_metrics(_trials(), missing="zero")
    with pytest.raises(ValueError):
        M.MacarTrial("spontaneous", "sea")
