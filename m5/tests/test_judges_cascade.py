"""m5.judges.cascade: routing between tiers (plan §7.3), with scripted fake judges.

T0 decides what is mechanical; T1 decides when confident and agreeing with T0; the uncertain
band and T0/T1 disagreements go to T2; T2/T1 disagreements go to T3; a seeded audit fraction is
re-judged by T3; and when external judging is not allowed, escalated items stay undecided.
"""

from __future__ import annotations

import pytest

from m5.judges import cascade as K
from m5.judges import clients as C
from m5.judges import prompts as P
from m5.judges.calibration import PlattCalibrator
from m5.judges.queue import JudgeStore
from m5.tests._judges_fakes import FakeClient


@pytest.fixture
def ext(cfg):
    return cfg.replace(JUDGE_EXTERNAL=True, JUDGE_AUDIT_FRACTION=0.0)


@pytest.fixture
def T(cfg):
    return P.templates(cfg)


def _t1_script(it):
    t = it.payload.get("text", "")
    if "uncertain" in t:
        return (True, 0.6, {"answer": "YES"})
    if "contradict" in t:
        return (False, 0.02, {"answer": "NO"})
    if "fail" in t:
        return "transient_exhausted: http_503"
    if "badquote" in t:
        return (False, 0.95, {"answer": "YES", "downgraded": "quote_invalid"})
    if "garlic" in t or "sea" in t:
        return (True, 0.97, {"answer": "YES"})
    return (False, 0.03, {"answer": "NO"})


def _tiers(t2_label=True, t3_label=False):
    t1 = FakeClient("backend", "Qwen/Qwen3-8B", False, _t1_script)
    t2 = FakeClient("openrouter", "deepseek/deepseek-v4-flash", True, lambda it: (t2_label, None, {}))
    t3 = FakeClient("anthropic", "claude-opus-5-5", True, lambda it: (t3_label, None, {}))
    return t1, t2, t3


def _items(T, texts, template="allusion"):
    return [P.make_item(T[template], {"concept": "garlic", "text": t}, item_id=f"i{n}")
            for n, t in enumerate(texts)]


def _by_id(judgments):
    return {j.item_id: j for j in judgments}


def test_tier0_decides_empty_and_collapsed_text_without_any_judge(ext, T):
    t1, t2, t3 = _tiers()
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
    items = _items(T, ["", "   "]) + [
        P.make_item(T["coherent"], {"prompt": "Tell me a story.", "text": "## ## ## ## ## ## ## ##"}, item_id="c1"),
        P.make_item(T["coherent"], {"prompt": "Tell me a story.", "text": ""}, item_id="c2"),
        P.make_item(T["named"], {"question": "What is it about?", "text": " "}, item_id="n1"),
        P.make_item(T["same_concept"], {"word_a": "Wrists", "word_b": "wrist"}, item_id="s1"),
        P.make_item(T["trait"], {"trait": "t", "description": "d", "prompt": "p", "response": ""}, item_id="tr")]
    j = _by_id(c.judge(items))
    assert j["i0"].label is False and j["i0"].decided_by == "t0" and j["i0"].stop == "t0_mechanical"
    assert j["c1"].label is False and j["c1"].trail[0].note == "decided:degenerate"
    assert j["c2"].label is False and j["c2"].trail[0].note == "decided:empty_text"
    assert j["n1"].label == "NONE"
    assert j["s1"].label is True and j["s1"].trail[0].note == "decided:code_match:inflection"
    assert j["tr"].label is None and j["tr"].missing.startswith("empty_text")   # missing, not 0
    assert t1.n_items == t2.n_items == t3.n_items == 0


def test_tier1_decides_when_confident_and_agreeing_with_tier0(ext, T):
    t1, t2, t3 = _tiers()
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
    j = _by_id(c.judge(_items(T, ["I ate garlic soup.", "A story about the moon."])))
    assert j["i0"].label is True and j["i0"].stop == "t1_confident_agrees_t0"
    assert j["i0"].trail[0].note == "opinion:literal_mention"
    assert j["i1"].label is False and j["i1"].stop == "t1_confident"
    assert t2.n_items == 0 and t3.n_items == 0


def test_the_uncertain_band_escalates_to_tier2(ext, T):
    t1, t2, t3 = _tiers(t2_label=True)
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
    j = c.judge(_items(T, ["uncertain words"]))[0]
    assert [s.tier for s in j.trail] == ["t0", "t1", "t2"]
    assert j.trail[1].note == "uncertain" and j.stop == "t2_agrees_t1" and j.decided_by == "t2"
    assert t2.n_items == 1 and t3.n_items == 0


def test_t0_t1_disagreement_goes_to_t2_and_t2_t1_disagreement_to_t3(ext, T):
    # T0: literal "garlic" -> YES; T1: confident NO; T2: YES (disagrees with T1); T3 decides NO.
    t1, t2, t3 = _tiers(t2_label=True, t3_label=False)
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
    j = c.judge(_items(T, ["contradict garlic"]))[0]
    assert [(s.tier, s.note) for s in j.trail] == [("t0", "opinion:literal_mention"),
                                                     ("t1", "disagrees_t0"), ("t2", "disagrees_t1"),
                                                     ("t3", "resolves:t2_disagrees_t1")]
    assert j.label is False and j.decided_by == "t3"
    # T2 siding with T1 overrules T0 without T3
    t1, t2, t3 = _tiers(t2_label=False)
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
    j = c.judge(_items(T, ["contradict garlic"]))[0]
    assert j.stop == "t2_agrees_t1" and j.label is False and t3.n_items == 0


def test_a_tier1_yes_without_a_valid_quote_escalates(ext, T):
    t1, t2, t3 = _tiers(t2_label=False)
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
    j = c.judge(_items(T, ["badquote text"]))[0]
    assert j.trail[1].note == "quote_invalid" and j.decided_by == "t2"


def test_tier1_failure_escalates_and_tier2_alone_can_decide(ext, T):
    t1, t2, t3 = _tiers(t2_label=True)
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
    j = c.judge(_items(T, ["fail here"]))[0]
    assert j.trail[1].missing.startswith("transient_exhausted") and j.stop == "t2_alone"


def test_external_judging_not_allowed_leaves_escalated_items_undecided(cfg, T):
    """Default config: JUDGE_EXTERNAL is off. Tiers 2 and 3 are never called and nothing that
    needed them is resolved to T1's label or to NO."""
    assert cfg["JUDGE_EXTERNAL"] is False
    t1, t2, t3 = _tiers()
    c = K.Cascade(cfg, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
    assert not c.external_allowed
    j = _by_id(c.judge(_items(T, ["I ate garlic soup.", "uncertain words", "contradict garlic", "fail here"])))
    assert j["i0"].label is True and j["i0"].decided_by == "t1"     # local judging still works
    for k in ("i1", "i2", "i3"):
        assert j[k].label is None and j[k].stop == "external_disallowed"
        assert "decision 12" in j[k].missing
    assert j["i1"].trail[1].label is True                            # T1's answer is kept
    assert t2.n_items == 0 and t3.n_items == 0


def test_non_benign_vector_classes_also_keep_judging_local(ext, T):
    t1, t2, t3 = _tiers()
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept", "deception"],
                  store=JudgeStore.memory())
    j = c.judge(_items(T, ["uncertain words"]))[0]
    assert j.stop == "external_disallowed" and t2.n_items == 0


def test_no_tier1_escalates_everything_and_tier3_steps_in_without_tier2(ext, T):
    t1, _, t3 = _tiers(t3_label=True)
    c = K.Cascade(ext.replace(JUDGE_T1="none"), t1=t1, t2=None, t3=t3, vector_classes=["concept"],
                  store=JudgeStore.memory())
    j = c.judge(_items(T, ["a quiet sea"]))[0]
    assert t1.n_items == 0 and j.decided_by == "t3" and j.stop.startswith("t3_resolves:t2_absent")
    c = K.Cascade(ext, t1=None, t2=None, t3=None, vector_classes=["concept"], store=JudgeStore.memory())
    j = c.judge(_items(T, ["a quiet sea"]))[0]
    assert j.label is None and j.stop == "no_external_tier"


def test_a_tier3_refusal_is_missing_not_a_no(ext, T):
    t1, t2, _ = _tiers(t2_label=True)
    t3 = FakeClient("anthropic", "claude-opus-5-5", True, lambda it: "refusal: bio")
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
    j = c.judge(_items(T, ["contradict garlic"]))[0]
    assert j.label is None and j.missing.startswith("t3_missing: refusal: bio")


def test_the_audit_fraction_is_seeded_per_item_and_never_overrides(ext, T):
    cfg = ext.replace(JUDGE_AUDIT_FRACTION=0.25)
    texts = [f"a sentence about the sea number {k}" for k in range(200)]

    def run():
        t1, t2, t3 = _tiers(t3_label=False)          # T3 disagrees with every T1 YES
        c = K.Cascade(cfg, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=JudgeStore.memory())
        return c.judge(_items(T, texts)), t3

    first, t3 = run()
    audited = [j.item_id for j in first if j.audit["selected"]]
    assert 25 <= len(audited) <= 75                  # about a quarter of 200
    assert t3.n_items == len(audited)
    for j in first:
        assert j.label is True and j.decided_by == "t1"   # the audit records, never overrides
        if j.audit["selected"]:
            assert j.audit["agrees"] is False and j.trail[-1].note == "audit"
    second, _ = run()
    assert [j.item_id for j in second if j.audit["selected"]] == audited
    # the same items are selected whatever batch they arrive in
    c = K.Cascade(cfg, t1=_tiers()[0], t2=None, t3=_tiers()[2], vector_classes=["concept"],
                  store=JudgeStore.memory())
    assert [j.item_id for j in c.judge(_items(T, texts)[::-1]) if j.audit["selected"]] == audited[::-1]


def test_audits_are_skipped_with_a_reason_when_external_judging_is_off(cfg, T):
    t1, t2, t3 = _tiers()
    c = K.Cascade(cfg.replace(JUDGE_AUDIT_FRACTION=1.0), t1=t1, t2=t2, t3=t3,
                  vector_classes=["concept"], store=JudgeStore.memory())
    j = c.judge(_items(T, ["I ate garlic soup."]))[0]
    assert j.audit == {"selected": True, "skipped": "external judging not allowed"}
    assert t3.n_items == 0


def test_budget_exhaustion_propagates_and_no_judgment_is_recorded(ext, T):
    t1, _, t3 = _tiers()
    t2 = FakeClient("openrouter", "deepseek/deepseek-v4-flash", True, lambda it: (True, None, {}),
                    raise_on_call=C.BudgetExhausted("HTTP 402"))
    store = JudgeStore.memory()
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=store)
    with pytest.raises(C.BudgetExhausted):
        c.judge(_items(T, ["I ate garlic soup.", "uncertain words"]))
    kinds = [r["kind"] for r in store.rows]
    assert "judgment" not in kinds                    # nothing marked done (plan I4)
    assert kinds.count("reply") == 2                  # tier-1 replies, paid for, are kept


def test_tier1_must_be_local_and_no_tier_shares_the_subjects_family(ext):
    remote = FakeClient("openrouter", "deepseek/deepseek-v4-flash", True, lambda it: (True, 0.9, {}))
    with pytest.raises(C.JudgeSetupError, match="local"):
        K.Cascade(ext, t1=remote, vector_classes=["concept"], store=JudgeStore.memory())
    gemma = FakeClient("backend", "google/gemma-3-12b-it", False, lambda it: (True, 0.9, {}))
    with pytest.raises(C.JudgeSetupError, match="family"):
        K.Cascade(ext, t1=gemma, vector_classes=["concept"], store=JudgeStore.memory())


def test_a_calibrator_moves_the_decision_and_is_recorded(ext, T):
    t1 = FakeClient("backend", "Qwen/Qwen3-8B", False, lambda it: (True, 0.95, {"answer": "YES"}))
    t2 = FakeClient("openrouter", "deepseek/deepseek-v4-flash", True, lambda it: (True, None, {}))
    shrink = PlattCalibrator(a=0.3, b=0.0, n=100, template_id="dominant")   # 0.95 -> about 0.71
    c = K.Cascade(ext, t1=t1, t2=t2, vector_classes=["concept"], store=JudgeStore.memory(),
                  calibrators={"dominant": shrink})
    j = c.judge([P.make_item(T["dominant"], {"concept": "garlic", "text": "x"}, item_id="d")])[0]
    step = j.trail[1]
    assert step.p_raw == 0.95 and 0.1 < step.p < 0.9 and step.note == "uncertain"
    assert j.decided_by == "t2"


def test_every_judgment_records_its_trail_and_rows_reach_the_store(ext, T):
    t1, t2, t3 = _tiers()
    store = JudgeStore.memory()
    c = K.Cascade(ext, t1=t1, t2=t2, t3=t3, vector_classes=["concept"], store=store)
    out = c.judge(_items(T, ["I ate garlic soup.", "uncertain words"]))
    rows = [r for r in store.rows if r["kind"] == "judgment"]
    assert len(rows) == 2 and {r["item_id"] for r in rows} == {"i0", "i1"}
    for j, row in zip(out, rows):
        assert row["stop"] == j.stop and row["trail"][0]["tier"] == "t0"
        assert all(k in row for k in ("label", "missing", "decided_by", "audit", "external"))
    assert rows[1]["external"] is True and rows[0]["external"] is False
