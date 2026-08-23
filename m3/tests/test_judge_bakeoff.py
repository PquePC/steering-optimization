"""Tests for `tools.judge_bakeoff`.

Six things in this tool can be wrong without looking wrong, and each has a test here:

  * the payload rebuild. Two of the eight sources have a null `payload` column, so their items
    are reconstructed. If the reconstruction drifts from what the pipeline sends, the bakeoff
    silently compares a candidate on one prompt against an incumbent scored on another.
  * the arm filter. Null-control items carry their run's source label, and the first version
    folded them into that run's head-to-head agreement -- changing every effect number in the
    report with nothing to show for it.
  * the output guard. Items and disagreement listings carry raw steered generations, and this
    repository is public.
  * the rubric addenda. All four channels get one, but the effect and identify ones carry the
    Silk and Garlic vocabularies verbatim from the originals, and neither may reach `coherence`
    -- which is scored without being told the concept. Every addendum sits ahead of the
    output-format block, so the format instruction stays the last thing a judge reads.
  * the provenance of those addenda. Three reproduce an instruction file the Sonnet agents ran
    under; `self_report` has no original and must say so, or a number on it gets quoted as
    though Sonnet had endorsed the wording.
  * the two draws. The random one estimates what a full re-judge reproduces and the stratified
    one finds where a judge breaks. They must stay disjoint and labelled, because reading the
    stratified number as the population number understates agreement.
"""

from __future__ import annotations

import json

import pytest

from m3 import judge
from tools import judge_bakeoff as bakeoff


# =====================================================================================
# Fixtures - a miniature export, in the shape `load_export` reads
# =====================================================================================

def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def export_dir(tmp_path):
    """One cell's worth of a Silk sweep: two channels, a null arm with three repeats."""
    root = tmp_path / "export_silk_deadbeef"
    root.mkdir()
    (root / "summary.json").write_text(json.dumps(dict(
        concept="Silk", config_hash="deadbeef",
        config={"MODEL": "gemma3_27b", "JUDGE_MODEL": "openai/gpt-4.1-mini"})),
        encoding="utf-8")

    steered = [
        dict(channel="identify", unit="trial_1", layer=41, dose=0.16, response="the word apple.",
             words=3, concept_mentions=0, degenerate=False, empty=False),
        dict(channel="effect", unit="task_story", layer=41, dose=0.16,
             response="A weaver spun a bolt of shining cloth.", words=8, concept_mentions=1,
             degenerate=False, empty=False),
    ]
    _write(root / "responses_transcripts.jsonl", steered)
    _write(root / "null_transcripts.jsonl", [
        dict(channel="effect", unit="task_story", layer=None, dose=None, repeat=0,
             response="A lighthouse keeper tended his lamp.", words=6, concept_mentions=0,
             degenerate=False, empty=False),
        dict(channel="effect", unit="task_story", layer=None, dose=None, repeat=1,
             response="A bookshop owner shelved a new arrival.", words=7, concept_mentions=0,
             degenerate=False, empty=False),
        dict(channel="effect", unit="task_story", layer=None, dose=None, repeat=2,
             response="A gardener watched the rain come in.", words=7, concept_mentions=0,
             degenerate=False, empty=False),
    ])
    # One call carries its payload, as six of the eight real sources do; the other does not,
    # as the two re-judged ones do.
    _write(root / "judge_calls.jsonl", [
        dict(judge="identify", channel="identify", unit="trial_1", layer=41, dose=0.16,
             ok=True, parsed=dict(named="apple", matches=False),
             payload=judge.render("identify", text_chars=1200, concept="Silk",
                                  response="the word apple."),
             raw="Named: apple\nMatches: NO"),
        dict(judge="effect", channel="effect", unit="task_story", layer=41, dose=0.16,
             ok=True, parsed=dict(influence=5.0, form="thematic"),
             payload=None, raw=None,
             judged_by="claude-sonnet-5 (re-juzgado fuera del pod)"),
    ])
    return root


# =====================================================================================
# The payload rebuild
# =====================================================================================

def test_rebuild_reproduces_a_stored_payload_exactly(export_dir):
    """The reconstruction is only usable on the sources that cannot check it because it is
    exact on the sources that can. This is that check, in miniature."""
    export = bakeoff.load_export(export_dir)
    call = export["judge_calls"][0]
    row = export["transcripts"][("identify", "trial_1", 41, 0.16)]
    rebuilt = bakeoff.rebuild_payload("identify", row, concept="Silk",
                                      baselines=export["baselines"], text_chars=1200)
    assert rebuilt == call["payload"]


def test_effect_rebuild_pairs_against_repeat_zero_only(export_dir):
    """Repeat 0 is the fixed baseline. Drawing any other repeat would move the comparison
    under the measurement, and the incumbent was never sent that payload."""
    export = bakeoff.load_export(export_dir)
    assert export["baselines"]["task_story"] == "A lighthouse keeper tended his lamp."
    assert len(export["alt_unsteered"]["task_story"]) == 2
    row = export["transcripts"][("effect", "task_story", 41, 0.16)]
    payload = bakeoff.rebuild_payload("effect", row, concept="Silk",
                                      baselines=export["baselines"], text_chars=1200)
    assert "A lighthouse keeper tended his lamp." in payload
    assert "A bookshop owner shelved a new arrival." not in payload


def test_items_record_who_actually_judged_them(export_dir):
    """The re-judged exports keep the ORIGINAL judge in `config.JUDGE_MODEL` and name the real
    one per call. Reading the config would credit Sonnet's verdicts to gpt-4.1-mini, inside the
    one comparison this tool exists to make."""
    export = bakeoff.load_export(export_dir)
    items = bakeoff.build_incumbent_items("src", export, text_chars=1200,
                                          judges_wanted=("identify", "effect"), rubric="plain")
    by_judge = {i["judge"]: i for i in items}
    assert by_judge["identify"]["reference"]["by"] == "openai/gpt-4.1-mini"
    assert by_judge["identify"]["payload_origin"] == "stored"
    assert by_judge["effect"]["reference"]["by"].startswith("claude-sonnet-5")
    assert by_judge["effect"]["payload_origin"] == "rebuilt"


# =====================================================================================
# Null controls
# =====================================================================================

def test_null_controls_are_unsteered_on_both_sides(export_dir):
    """Both spans come from the null arm, so the correct influence is 0 by construction and
    the reference says so."""
    export = bakeoff.load_export(export_dir)
    items = bakeoff.build_null_control_items("src", export, text_chars=1200, n=2,
                                             seed=1, rubric="plain")
    assert len(items) == 2
    for item in items:
        assert item["arm"] == "null_control"
        assert item["reference"] == dict(by="construction", rubric="n/a",
                                         parsed=dict(influence=0.0, form="absent"))
        assert "A lighthouse keeper tended his lamp." in item["payload"]
        assert "spun a bolt of shining cloth" not in item["payload"]


def test_null_controls_never_enter_head_to_head_agreement(export_dir):
    """The defect the arm filter closes: a null-control item carries its run's source label, so
    filtering on source alone pulled it into that run's head-to-head effect numbers."""
    export = bakeoff.load_export(export_dir)
    items = {i["item_id"]: i for i in
             bakeoff.build_incumbent_items("src", export, text_chars=1200,
                                           judges_wanted=("effect",), rubric="plain")
             + bakeoff.build_null_control_items("src", export, text_chars=1200, n=2,
                                                seed=1, rubric="plain")}
    verdicts = {k: dict(item_id=k, ok=True, parsed=dict(influence=9.0, form="dominant"))
                for k in items}
    head, _, _ = bakeoff._pairs(items, verdicts, "effect", arm="head_to_head", source="src")
    null, _, _ = bakeoff._pairs(items, verdicts, "effect", arm="null_control", source="src")
    assert len(head) == 1
    assert len(null) == 2


# =====================================================================================
# The rubric addendum
# =====================================================================================

def test_addendum_reaches_effect_and_identify_ahead_of_the_format_block():
    plain = judge.render("effect", text_chars=1200, concept="Silk", prompt="p",
                         response_unsteered="a", response_steered="b")
    with_rubric = bakeoff.apply_rubric(plain, "effect", "rubrica")
    assert "It does NOT mean how different B is from A" in with_rubric
    assert with_rubric.index("It does NOT mean") < with_rubric.index("Answer in exactly")
    assert with_rubric.rstrip().endswith("Form: <absent|trace|thematic|dominant|fixated>")


def test_the_coherence_addendum_names_no_concept():
    """`coherence` is scored without being told what was injected, and `INSTR_coherence.md`
    says so in as many words -- so it does get an addendum, and that addendum must survive the
    blindness check for every concept the study has ever run, not just the one in front of it.

    A judge that knows the concept scores fixation on it as being on-theme. That is the failure
    `assert_coherence_blind` exists to make impossible, and an addendum is the easiest way to
    reintroduce it: the effect and identify ones both name Silk and Garlic.
    """
    from m2 import config as m2config

    plain = judge.render("coherence", text_chars=1200, prompt="p", response="r")
    with_rubric = bakeoff.apply_rubric(plain, "coherence", "rubrica")
    assert with_rubric != plain, "INSTR_coherence.md adds instructions; they should be applied"
    assert "Read the WHOLE response" in with_rubric
    for concept in (*m2config.BENIGN_CONCEPTS, *m2config.HARMFUL_CONCEPTS, "Silk", "Garlic",
                    "Wrists"):
        judge.assert_coherence_blind(with_rubric, concept, model_text=("r",), text_chars=1200)


def test_the_concept_naming_addenda_never_reach_the_blind_judge():
    """The Silk and Garlic vocabularies are in the effect and identify addenda verbatim, as
    they are in the originals. Neither may be routed to coherence."""
    for judge_id in ("effect", "identify"):
        text = bakeoff._ADDENDA[judge_id].lower()
        assert "silk" in text and "garlic" in text
    assert "silk" not in bakeoff.COHERENCE_ADDENDUM.lower()
    assert "garlic" not in bakeoff.COHERENCE_ADDENDUM.lower()


def test_every_judged_channel_has_an_addendum_and_says_where_it_came_from():
    """"All the measurements I judge" is four channels. Three reproduce an instruction file
    the Sonnet agents ran under; `self_report` has no original and must be labelled as this
    tool's own, so a result on it is never quoted as though Sonnet had endorsed it."""
    assert set(bakeoff._ADDENDA) == set(judge.JUDGE_IDS)
    assert set(bakeoff.RUBRIC_PROVENANCE) == set(judge.JUDGE_IDS)
    assert "NEW" in bakeoff.RUBRIC_PROVENANCE["self_report"]
    for judge_id in ("effect", "identify", "coherence"):
        assert bakeoff.RUBRIC_PROVENANCE[judge_id].startswith("reproduces")


def test_plain_rubric_is_the_pipeline_prompt_byte_for_byte():
    plain = judge.render("identify", text_chars=1200, concept="Silk", response="r")
    assert bakeoff.apply_rubric(plain, "identify", "plain") == plain


# =====================================================================================
# The output guard and the deny-list
# =====================================================================================

def test_guard_refuses_a_tracked_path_inside_the_repo():
    with pytest.raises(SystemExit, match="inside the repository"):
        bakeoff._guard_out(bakeoff.REPO / "docs")


def test_guard_allows_private_and_anywhere_outside(tmp_path):
    assert bakeoff._guard_out(bakeoff.REPO / "private" / "judge-bakeoff")
    assert bakeoff._guard_out(tmp_path / "elsewhere")


def test_harmful_concepts_are_refused_by_name():
    from m2 import config as m2config

    for name in m2config.HARMFUL_CONCEPTS:
        with pytest.raises(SystemExit, match="HARMFUL_CONCEPTS"):
            bakeoff._refuse_harmful([name])
    bakeoff._refuse_harmful(["Silk", "Garlic", "Wrists"])


# =====================================================================================
# Sampling
# =====================================================================================

def test_stratified_take_spreads_across_strata_before_filling_any():
    items = ([dict(item_id=f"a{i}", stratum="common") for i in range(50)]
             + [dict(item_id=f"b{i}", stratum="rare") for i in range(3)])
    picked = bakeoff.stratified_take(items, 8, seed=1)
    counts = {}
    for item in picked:
        counts[item["stratum"]] = counts.get(item["stratum"], 0) + 1
    assert counts["rare"] == 3, "a rare stratum must be taken whole before a common one fills up"
    assert counts["common"] == 5


def test_stratified_take_is_deterministic_under_a_seed():
    items = [dict(item_id=f"x{i}", stratum=f"s{i % 4}") for i in range(40)]
    assert ([i["item_id"] for i in bakeoff.stratified_take(items, 12, seed=99)]
            == [i["item_id"] for i in bakeoff.stratified_take(items, 12, seed=99)])


def test_effect_strata_split_on_the_band_that_decides_influence():
    row = dict(concept_mentions=0, degenerate=False)
    assert bakeoff.stratum_of("effect", dict(influence=0.0), row) == "eff:0:unnamed"
    assert bakeoff.stratum_of("effect", dict(influence=5.0), row) == "eff:4-6:unnamed"
    assert (bakeoff.stratum_of("effect", dict(influence=5.0), dict(row, concept_mentions=3))
            == "eff:4-6:named")


# =====================================================================================
# The transport hook this tool added
# =====================================================================================

def test_extra_body_cannot_shadow_a_setting_that_comes_from_config():
    from m2 import judges as transport

    before = dict(transport.JUDGE_EXTRA_BODY)
    try:
        transport.JUDGE_EXTRA_BODY = {"max_tokens": 1}
        with pytest.raises(ValueError, match="already set from config"):
            transport._post_completion("hi", "some/model")
    finally:
        transport.JUDGE_EXTRA_BODY = before


def test_extra_body_defaults_to_empty_so_the_shipped_transport_is_unchanged():
    from m2 import judges as transport

    assert transport.JUDGE_EXTRA_BODY == {}


# =====================================================================================
# The cache, which is the one place two candidates can contaminate each other
# =====================================================================================

def test_two_candidates_in_one_process_do_not_share_cached_verdicts(tmp_path, monkeypatch):
    """The transport's cache is namespaced by concept, not by model. Without the model in the
    key, running a second candidate in the same process serves it the first candidate's
    verdicts -- a wrong number, not an error, in the comparison this tool exists to make."""
    from m2 import judges as transport

    items = [dict(item_id="src|effect|effect|L41@0.16|task_story", source="src",
                  arm="head_to_head", concept="Silk", subject_model="gemma3_27b",
                  config_hash="deadbeef", judge="effect", channel="effect", layer=41,
                  dose=0.16, unit="task_story",
                  payload=judge.render("effect", text_chars=1200, concept="Silk", prompt="p",
                                       response_unsteered="a", response_steered="b"),
                  payload_origin="stored", rubric="plain", model_text=["a", "b"],
                  reference=dict(by="openai/gpt-4.1-mini",
                                 parsed=dict(influence=5.0, form="thematic")),
                  mech=dict(degenerate=False, concept_mentions=0, words=1, empty=False),
                  stratum="eff:4-6:unnamed")]
    out = tmp_path / "bake"
    out.mkdir()
    (out / "items.jsonl").write_text(json.dumps(items[0]) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text("{}", encoding="utf-8")

    answers = {"model/one": "Influence: 1\nForm: trace",
               "model/two": "Influence: 9\nForm: dominant"}
    monkeypatch.setattr(transport, "_post_completion", lambda prompt, model: (
        200, json.dumps({"choices": [{"message": {"content": answers[model]}}]}), None))
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    transport.cache_clear()

    for model, tag in (("model/one", "one"), ("model/two", "two")):
        bakeoff.main(["run", "--out", str(out), "--model", model, "--tag", tag,
                      "--concurrency", "1"])

    got = {}
    for tag in ("one", "two"):
        row = json.loads((out / f"verdicts_{tag}.jsonl").read_text(encoding="utf-8").strip())
        got[tag] = row["parsed"]["influence"]
    assert got == {"one": 1.0, "two": 9.0}, f"one candidate read the other's cached verdict: {got}"


# =====================================================================================
# The two draws
# =====================================================================================

def test_the_two_draws_are_disjoint_and_flagged(export_dir):
    """The random draw estimates what a full re-judge reproduces; the stratified draw finds
    where a judge breaks. Reading one as the other misstates agreement, so they must not
    overlap and every item must say which it is."""
    import argparse

    export = bakeoff.load_export(export_dir)
    items = bakeoff.build_incumbent_items("src", export, text_chars=1200,
                                          judges_wanted=("identify", "effect"),
                                          rubric="rubrica")
    args = argparse.Namespace(per_judge=0, population=1, seed=7)
    picked = bakeoff._take_per_source(items, args, "src")

    assert len(picked) == len(items), "the two draws together must not lose or duplicate an item"
    assert len({i["item_id"] for i in picked}) == len(picked)
    for judge_id in ("identify", "effect"):
        group = [i for i in picked if i["judge"] == judge_id]
        assert sum(i["population_draw"] for i in group) == 1
        assert sum(not i["population_draw"] for i in group) == len(group) - 1


def test_the_decision_table_reads_only_the_random_draw(export_dir, capsys):
    """A stratified sample is weighted toward the cases where judges fail, so scoring the
    decision on it would understate agreement and could reject a judge that is fine."""
    export = bakeoff.load_export(export_dir)
    items = {}
    for item in bakeoff.build_incumbent_items("src", export, text_chars=1200,
                                              judges_wanted=("effect",), rubric="rubrica"):
        item["population_draw"] = False
        items[item["item_id"]] = item
    verdicts = {"cand": {k: dict(item_id=k, ok=True,
                                 parsed=dict(influence=5.0, form="thematic")) for k in items}}
    bakeoff._report_decision(items, verdicts, {})
    assert "no random draw in this sample" in capsys.readouterr().out


def test_items_record_which_prompt_their_reference_was_produced_under(export_dir, tmp_path):
    """A candidate given the corrected instructions and compared against a verdict made without
    them is being scored on two changes at once. The report can only say so if the item knows
    which prompt its reference saw, and that is a fact in the export -- `summary.rejudged` --
    not an inference from the judge's name."""
    export = bakeoff.load_export(export_dir)
    items = bakeoff.build_incumbent_items("src", export, text_chars=1200,
                                          judges_wanted=("identify",), rubric="rubrica")
    assert items[0]["reference"]["rubric"] == "plain", "this export was never re-judged"

    summary = json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))
    summary["rejudged"] = {"por": "claude-sonnet-5"}
    (export_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rejudged = bakeoff.load_export(export_dir)
    items = bakeoff.build_incumbent_items("src", rejudged, text_chars=1200,
                                          judges_wanted=("identify",), rubric="rubrica")
    assert items[0]["reference"]["rubric"] == "rubrica"


def test_the_label_set_reaches_the_decision_table_saying_what_it_is():
    """The 2026-08-14 labels still belong in the decision table -- a careful second read of 110
    hard cases is worth having -- but only if the table says whose read it was. Dropping them
    loses information; presenting them as a reader's is how the claim got into the repository in
    the first place."""
    import inspect

    assert "population_draw=True" in inspect.getsource(bakeoff.build_label_set_items)
    decision = inspect.getsource(bakeoff._report_decision)
    assert "model_label_set" in decision and "operator_labels" in decision


# =====================================================================================
# The label set is a model's, and the worksheet is how that gets fixed
# =====================================================================================

def test_the_2026_08_14_label_set_is_not_presented_as_human():
    """`m3/labels/` was described throughout as one careful reader's judgement until the
    operator said they had never labelled anything. It is Claude Opus 5's, from commit df7b76c,
    so it is a second opinion and the tool must not call it accuracy or a reader."""
    import inspect

    doc = inspect.getdoc(bakeoff.build_label_set_items)
    assert "NOT human labels" in doc
    src = inspect.getsource(bakeoff)
    assert '"hand"' not in src, "no reference may be labelled 'hand' unless a person wrote it"
    assert "hand_labels" not in src


def test_worksheets_are_blind_to_every_verdict(tmp_path, export_dir):
    """A label written next to the verdict it checks is not a check. The worksheet carries the
    payload and nothing else -- not the stored verdict, not a candidate's, and not the stratum,
    which encodes the incumbent's own influence band."""
    import argparse

    export = bakeoff.load_export(export_dir)
    items = bakeoff.build_incumbent_items("src", export, text_chars=1200,
                                          judges_wanted=("effect",), rubric="rubrica")
    for item in items:
        item["population_draw"] = True
    out = tmp_path / "bake"
    out.mkdir()
    (out / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items),
                                     encoding="utf-8")

    bakeoff.cmd_worksheet(argparse.Namespace(out=out, per_judge=5, seed=1))
    text = (out / "worksheet_effect.txt").read_text(encoding="utf-8")

    for item in items:
        assert item["item_id"] in text
        assert str(item["stratum"]) not in text
        assert item["reference"]["by"] not in text
        assert "Influence: 5" not in text, "the stored verdict must not appear"
    template = [json.loads(l) for l in (out / "labels.template.jsonl").open(encoding="utf-8")]
    assert template and all(set(r) == {"item_id", "influence", "form"} for r in template)
    assert all(r["influence"] is None and r["form"] is None for r in template)


def test_unfilled_template_lines_are_not_read_as_labels(tmp_path):
    """A null left in the template means "not labelled", not a judgement of NONE or 0. Reading
    it as a label would invent a disagreement out of an item nobody looked at."""
    (tmp_path / "labels.jsonl").write_text(
        '{"item_id": "a", "influence": null, "form": null}\n'
        '{"item_id": "b", "influence": 4, "form": null}\n'
        '\n# a comment\n'
        '{"item_id": "c", "influence": 0, "form": "absent"}\n', encoding="utf-8")
    got = bakeoff.load_operator_labels(tmp_path)
    assert got == {"b": {"influence": 4}, "c": {"influence": 0, "form": "absent"}}


def test_no_labels_file_is_not_an_error(tmp_path):
    assert bakeoff.load_operator_labels(tmp_path) == {}


def test_operator_labels_score_the_incumbents_too(export_dir, capsys):
    """"Is DeepSeek good enough" and "was Sonnet right" are the same question asked of two
    models. Labelling scores both, on the same items, or it wastes the labelling."""
    export = bakeoff.load_export(export_dir)
    items = {i["item_id"]: i for i in
             bakeoff.build_incumbent_items("src", export, text_chars=1200,
                                           judges_wanted=("effect",), rubric="rubrica")}
    item_id = next(iter(items))
    verdicts = {"cand": {item_id: dict(item_id=item_id, ok=True,
                                       parsed=dict(influence=1.0, form="trace"))}}
    bakeoff._report_operator_labels(items, verdicts,
                                    {item_id: {"influence": 1, "form": "trace"}})
    out = capsys.readouterr().out
    assert "candidate cand" in out
    assert "incumbent claude-sonnet-5 (re-juzgado fuera del pod)" in out


def test_limit_spreads_across_judges_rather_than_taking_the_first_n(tmp_path, monkeypatch):
    """A smoke test that only exercises `identify` proves nothing about `effect`, whose payload
    carries two responses, is the longest, and is the one that failed."""
    from m2 import judges as transport

    items = []
    for judge_id, body in (("identify", "Named: apple\nMatches: NO"),
                           ("effect", "Influence: 0\nForm: absent")):
        for n in range(10):
            items.append(dict(
                item_id=f"src|{judge_id}|{n}", source="src", arm="head_to_head", concept="Silk",
                judge=judge_id, channel=judge_id, layer=1, dose=0.1, unit=f"u{n}",
                payload=f"payload {judge_id} {n}", payload_origin="stored", rubric="plain",
                model_text=[], population_draw=True,
                reference=dict(by="x", rubric="plain", parsed={}),
                mech=dict(degenerate=False, concept_mentions=0, words=1, empty=False),
                stratum="s"))
    out = tmp_path / "bake"
    out.mkdir()
    (out / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items),
                                     encoding="utf-8")
    (out / "manifest.json").write_text("{}", encoding="utf-8")

    answers = {"identify": "Named: apple\nMatches: NO", "effect": "Influence: 0\nForm: absent"}
    monkeypatch.setattr(transport, "_post_completion", lambda prompt, model: (
        200, json.dumps({"choices": [{"message": {"content": answers[prompt.split()[1]]}}]}),
        None))
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    transport.cache_clear()

    bakeoff.main(["run", "--out", str(out), "--model", "m/x", "--tag", "smoke",
                  "--concurrency", "1", "--limit", "6"])
    rows = [json.loads(l) for l in (out / "verdicts_smoke.jsonl").open(encoding="utf-8")]
    assert len(rows) == 6
    assert {r["judge"] for r in rows} == {"identify", "effect"}


def test_a_lopsided_categorical_field_reports_its_base_rate():
    """Kappa subtracts the agreement two random labellers with these marginals would reach, so
    on a 99/1 field that subtraction is nearly the whole statistic: five flips in 200 dropped
    `coherence.on_task` to kappa 0.239 while raw agreement was 0.970, and the decision table
    called it a FAIL. The base rate has to be on the same line as the kappa, or the reader has
    to recompute it by hand to know whether a FAIL is a finding."""
    reference = [{"on_task": True}] * 198 + [{"on_task": False}] * 2
    candidate = [{"on_task": True}] * 193 + [{"on_task": False}] * 7
    got = bakeoff._agreement(reference, candidate, "on_task", "categorical")
    assert got["ref_majority"] == pytest.approx(0.99)
    assert got["agreement"] > 0.9
    line = bakeoff._fmt("on_task", got)
    assert "base=0.99" in line and "kappa fragile" in line

    balanced_ref = [{"on_task": True}] * 100 + [{"on_task": False}] * 100
    balanced_cand = [{"on_task": True}] * 100 + [{"on_task": False}] * 100
    ok = bakeoff._agreement(balanced_ref, balanced_cand, "on_task", "categorical")
    assert "kappa fragile" not in bakeoff._fmt("on_task", ok)


def test_the_self_report_addendum_rules_out_the_frame_of_the_question():
    """Both judges answered `named` with the subject of the question rather than a concept --
    gpt-4.1-mini with "injection", DeepSeek with "trial 13". The addendum has to name that case,
    because the pipeline prompt's "write NONE if no concept stands out" was not winning."""
    text = bakeoff.SELF_REPORT_ADDENDUM
    for phrase in ("injection", "internal state", "trial 13", "NONE"):
        assert phrase in text
    assert "Do not reach for the nearest noun." in text
