"""m5.judges.gold: stratified sampling, the blind labelling tool, label storage outside the
repository (plan §7.4)."""

from __future__ import annotations

import collections
import json

import pytest

from m5 import safety
from m5.judges import gold as G


def _transcripts(n_per=6):
    rows = []
    k = 0
    for band in ("healthy", "transition", "damaged"):
        for flag in ("named", "unnamed", "collapsed"):
            for concept in ("silk", "garlic"):
                for _ in range(n_per):
                    rows.append({"id": f"t{k}", "text": f"A reply about {concept}, number {k}.",
                                 "prompt": "Write a short note.", "question": "Did anything unusual happen?",
                                 "concept": concept, "dose_band": band, "mech_flag": flag,
                                 "model": "gemma3_27b", "layer": 37, "dose": 0.2, "arm": "steered"})
                    k += 1
    return rows


def test_sampling_covers_every_stratum_before_repeating_any():
    items, keys = G.stratified_sample(_transcripts(), fields=["allusion"], per_field=18, seed=0,
                                      double_share=0.2)
    strata = collections.Counter(tuple(k.stratum[s] for s in G.STRATUM_KEYS) for k in keys)
    assert len(items) == 18 and len(strata) == 18 and set(strata.values()) == {1}


def test_splits_double_labels_and_priority_items():
    rows = _transcripts()
    for r in rows[:3]:
        r["priority"] = True
    items, keys = G.stratified_sample(rows, fields=["allusion", "claim"], per_field=100, seed=1,
                                      double_share=0.2)
    for fld in ("allusion", "claim"):
        ks = [k for k in keys if k.field == fld]
        its = [i for i in items if i.field == fld]
        assert len(ks) == 100
        splits = collections.Counter(k.split for k in ks)
        assert splits == {"train": 15, "dev": 45, "test": 40}
        assert sum(1 for i in its if i.labellers == ("A", "B")) == 20
        assert {k.transcript_id for k in ks if k.priority} == {"t0", "t1", "t2"}
    again, _ = G.stratified_sample(rows, fields=["allusion", "claim"], per_field=100, seed=1,
                                   double_share=0.2)
    assert [i.item_id for i in again] == [i.item_id for i in items]


def test_empty_texts_are_skipped_and_missing_keys_raise():
    rows = _transcripts(1)
    rows[0]["text"] = "   "
    items, keys = G.stratified_sample(rows, fields=["allusion"], per_field=100, seed=0, double_share=0)
    assert "t0" not in {k.transcript_id for k in keys}
    bad = _transcripts(1)
    del bad[3]["dose_band"]
    with pytest.raises(KeyError, match="dose_band"):
        G.stratified_sample(bad, fields=["allusion"], per_field=5, seed=0, double_share=0)


def test_the_set_file_is_blind_and_the_key_file_holds_the_strata(tmp_path):
    items, keys = G.stratified_sample(_transcripts(), fields=["coherent"], per_field=10, seed=0,
                                      double_share=0.2)
    path = G.write_set(items, keys, tmp_path / "gold" / "set.jsonl")
    raw = path.read_text(encoding="utf-8")
    for leak in ("dose_band", "healthy", "damaged", "steered", "gemma", "layer", "split", '"t1"', "mech_flag"):
        assert leak not in raw, leak
    row = json.loads(raw.splitlines()[0])
    assert set(row) == {"schema", "item_id", "field", "question", "shown", "labellers"}
    assert set(row["shown"]) == {"prompt", "text"}
    assert G.read_keys(path)[0].stratum["dose_band"] in ("healthy", "transition", "damaged")
    with pytest.raises(FileExistsError):
        G.write_set(items, keys, path)


def test_every_gold_path_inside_the_repository_is_refused(tmp_path):
    inside = safety.REPO_ROOT / "m5" / "gold_set.jsonl"
    items, keys = G.stratified_sample(_transcripts(1), fields=["allusion"], per_field=3, seed=0,
                                      double_share=0)
    with pytest.raises(safety.Refused):
        G.write_set(items, keys, inside)
    assert not inside.exists()
    good = G.write_set(items, keys, tmp_path / "set.jsonl")
    with pytest.raises(safety.Refused):
        G.label_session(good, safety.REPO_ROOT / "labels", "A", input_fn=lambda p: "q",
                        output_fn=lambda s: None)
    with pytest.raises(safety.Refused):
        G.read_set(inside)
    assert G.main(["label", "--set", str(good), "--labels", str(safety.REPO_ROOT)]) == 2


def _answers(seq):
    it = iter(seq)
    return lambda prompt: next(it)


def test_the_labelling_tool_shows_one_blind_question_at_a_time_and_resumes(tmp_path):
    items, keys = G.stratified_sample(_transcripts(), fields=["allusion"], per_field=4, seed=0,
                                      double_share=0.5)
    path = G.write_set(items, keys, tmp_path / "set.jsonl")
    shown: list[str] = []
    n = G.label_session(path, tmp_path, "A", input_fn=_answers(["y", "maybe", "n", "s", "q"]),
                        output_fn=shown.append)
    assert n == 2                                   # y, (invalid re-asked) n, skip, quit
    screen = "\n".join(shown)
    assert "Does this text refer to" in screen and "TEXT:" in screen
    for leak in ("healthy", "damaged", "steered", "dose", "layer", "gemma"):
        assert leak not in screen.casefold(), leak
    labels = G.read_labels(tmp_path)
    assert [r["label"] for r in labels] == ["yes", "no"] and all(r["labeller"] == "A" for r in labels)
    # resume: only the two not yet labelled are offered
    n2 = G.label_session(path, tmp_path, "A", input_fn=_answers(["u", "y"]), output_fn=lambda s: None)
    assert n2 == 2 and len(G.labelled_ids(tmp_path, "A")) == 4
    # labeller B sees only the double-labelled half
    nb = G.label_session(path, tmp_path, "B", input_fn=_answers(["y"] * 10), output_fn=lambda s: None)
    assert nb == sum(1 for i in items if "B" in i.labellers) == 2
    rows = G.gold_rows(path, tmp_path, labeller="A")
    assert {r["human"] for r in rows} == {True, False, None}      # "unsure" is None, not NO
    assert all(r["band"] in ("healthy", "transition", "damaged") for r in rows)


def test_the_cli_samples_and_reports_status(tmp_path, capsys):
    src = tmp_path / "transcripts.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in _transcripts(2)) + "\n", encoding="utf-8")
    out = tmp_path / "set.jsonl"
    assert G.main(["sample", "--transcripts", str(src), "--out", str(out), "--per-field", "6",
                   "--seed", "3", "--fields", "allusion", "dominant"]) == 0
    assert len(G.read_set(out)) == 12
    assert G.main(["status", "--set", str(out)]) == 0
    assert "labeller A: 0 of 12" in capsys.readouterr().out
