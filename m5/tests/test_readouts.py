"""m5.readouts on the simulator: each screen readout against the default truth's ground truth.

The simulator (`m5.backend.fake.default_truth`) says, per layer, where each construct switches on,
so every readout is checked to move the way the truth says with dose, to sit exactly at its null
at zero dose, to read its null on the same prompts (paired), to report missing with a reason
where it cannot read, and to trip each of its guards. The context is built once per module
through `m5.tests.sim` (the real dose, profile and vector code).

Layers used (24-layer default truth): L2 early (noticing and localisation only), L8-L10 mid
(noticing before the effect; introspective excess d3 > d5), L16 late (the operating window: the
effect far below noticing, no excess), L22 motor (naming without noticing).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math

import numpy as np
import pytest

from m5 import prompts
from m5 import readouts as R
from m5.data.contrastive import SYCOPHANCY_AB
from m5.export import _aggregate_carries_text
from m5.intervention import Cell
from m5.readouts import _common as C
from m5.readouts import effect, probe as P
from m5.safety import REPO_ROOT, Refused
from m5.steer import intervention_for
from m5.tests.sim import sim_context
from m5.vectors import extract, gaussian_recipe

GARLIC = "concept-garlic"


def cell(vid: str, layer: int, rho: float, **kw) -> Cell:
    return Cell(vid, "single", (layer,), rho, **kw)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    c = sim_context(tmp_path_factory.mktemp("readouts"))
    c.backend.register_ab_items({it.question: (it.matching, "sycophancy") for it in SYCOPHANCY_AB.items})
    layers = list(range(c.backend.info.n_layers))
    for ref in (GARLIC, "syc"):
        rec = gaussian_recipe(ref, 0)
        c.store.save(rec, extract(c.backend, rec, layers, seed=c.cfg["SEED"], batch_size=8,
                                  reference=c.store.load(ref)))
        c.vector_classes[rec.vector_id] = "control"
    return c


def fresh(ctx, **changes):
    """The same model and vectors with an empty cache (and any field changed)."""
    return dataclasses.replace(ctx, cache={}, **changes)


def true_r(ctx, c: Cell) -> float:
    """The simulator's own r_perp for a cell (its truth is in these units)."""
    spec = prompts.forced_naming(ctx.backend, 1)
    iv = intervention_for(ctx, c, spec, "live")
    return ctx.backend.r_perp_of(iv.vector_id, c.layer, iv.terms[0].alpha)


def last_row(ctx, code: str) -> dict:
    return [r for r in ctx.run.rows("screen") if r["code"] == code][-1]


# =====================================================================================
# e9
# =====================================================================================

def test_e9_lifts_with_the_effect_and_is_its_null_at_zero_dose(ctx):
    zero = R.read_e9(ctx, cell(GARLIC, 16, 0.0))
    assert zero.value == 0.0 and zero.extra["entropy_change"] == 0.0 and zero.n == ctx.cfg["SCREEN_A_E9_PROMPTS"]
    lifts = [R.read_e9(ctx, cell(GARLIC, 16, rho)).value for rho in (0.05, 0.15, 0.5)]
    assert 0.0 < lifts[0] < lifts[1] < lifts[2]
    truth = ctx.backend.truth
    effects = [truth.p("garlic", 16, "effect", true_r(ctx, cell(GARLIC, 16, rho))) for rho in (0.05, 0.15, 0.5)]
    assert effects[0] < 0.2 < 0.9 < effects[2]       # the doses span the truth's effect curve
    # the simulator's open text puts the concept's mass at M_ABSENT unsteered and raises it toward
    # M_MAX with the effect, so the lift is exactly log(steered mass / unsteered mass)
    from m5.backend.fake import _OpenLM
    lo, hi = _OpenLM.M_ABSENT, _OpenLM.M_MAX
    for lift, eff in zip(lifts, effects):
        assert lift == pytest.approx(math.log((lo + (hi - lo) * eff) / lo), abs=1e-3)
    high = R.read_e9(ctx, cell(GARLIC, 16, 0.5))
    assert [p["prompt_id"] for p in high.extra["per_prompt"]] == \
        [p["prompt_id"] for p in zero.extra["per_prompt"]]          # paired: the same prompts
    assert R.read_e9(ctx, cell(GARLIC, 16, 0.5), pass_="B").n == ctx.cfg["SCREEN_B_E9_PROMPTS"]


def test_e9_entropy_change_separates_flattening_from_the_concept(ctx):
    working = R.read_e9(ctx, cell(GARLIC, 16, 0.5))
    breaking = R.read_e9(ctx, cell(GARLIC, 16, 1.5))
    assert abs(working.value - breaking.value) < 0.2          # the same concept lift...
    assert working.extra["entropy_change"] < 0.3 < 1.0 < breaking.extra["entropy_change"]  # ...not the same model


def test_e9_is_missing_with_a_reason_for_nameless_vectors_and_the_null_arm(ctx):
    for r in (R.read_e9(ctx, cell("ea", 16, 0.5)), R.read_e9(ctx, None)):
        assert r.value is None and r.n == 0 and r.missing
    assert "no concept word" in R.read_e9(ctx, cell("syc", 16, 0.5)).missing


def test_e9_excludes_an_empty_continuation_and_a_non_finite_prompt(ctx, monkeypatch):
    real_cont, real_cand = effect.own_continuation, effect.candidate_logprobs
    first = {}

    def empty_first(ctx_, code, prompt, *a, unit, **kw):
        first.setdefault("unit", unit)
        if unit == first["unit"]:
            return C.Continuation("", (), None)
        return real_cont(ctx_, code, prompt, *a, unit=unit, **kw)

    monkeypatch.setattr(effect, "own_continuation", empty_first)
    r = R.read_e9(ctx, cell(GARLIC, 16, 0.5))
    assert r.n == ctx.cfg["SCREEN_A_E9_PROMPTS"] - 1 and r.extra["excluded"] == 1
    assert "empty" in r.extra["excluded_units"][first["unit"]]
    monkeypatch.setattr(effect, "own_continuation", real_cont)

    def poison_first(ctx_, reqs, cand):
        out = real_cand(ctx_, reqs, cand)
        if reqs and reqs[0].intervention is not None:
            out[0] = np.full_like(out[0], -np.inf)
        return out

    monkeypatch.setattr(effect, "candidate_logprobs", poison_first)
    r = R.read_e9(ctx, cell(GARLIC, 16, 0.5))
    assert r.n == ctx.cfg["SCREEN_A_E9_PROMPTS"] - 1 and r.extra["excluded"] == 1
    assert "non-finite" in next(iter(r.extra["excluded_units"].values()))


# =====================================================================================
# e10
# =====================================================================================

def test_e10_is_the_truths_propensity_shift(ctx):
    truth = ctx.backend.truth
    assert R.read_e10(ctx, cell("syc", 16, 0.0)).value == 0.0
    prev = 0.0
    for rho in (0.1, 0.3, 0.6):
        c = cell("syc", 16, rho)
        r = R.read_e10(ctx, c)
        assert r.value == pytest.approx(truth.e10_shift("syc", 16, true_r(ctx, c), "sycophancy"), abs=1e-6)
        assert r.value > prev
        prev = r.value
    assert r.n == len(SYCOPHANCY_AB.items) and r.extra["behaviour"] == "sycophancy"
    inhibited = R.read_e10(ctx, cell("syc", 16, 0.6, op="sub"))
    assert inhibited.value == pytest.approx(-r.value, abs=1e-6)


def test_e10_on_a_concept_vector_is_the_control_and_needs_a_named_behaviour(ctx):
    r = R.read_e10(ctx, cell(GARLIC, 16, 0.6), behaviour="sycophancy")
    assert abs(r.value) < 1e-9                                       # concept vectors move nothing
    missing = R.read_e10(ctx, cell(GARLIC, 16, 0.6))
    assert missing.value is None and "no embedded A/B items" in missing.missing
    assert R.read_e10(ctx, None).missing
    with pytest.raises(ValueError, match="measure 'sycophancy'"):
        R.read_e10(ctx, cell("syc", 16, 0.6), behaviour="honesty", items=SYCOPHANCY_AB)


# =====================================================================================
# s2
# =====================================================================================

def test_s2_rises_with_dose_and_its_null_is_paired_sample_by_sample(ctx):
    zero = R.read_s2(ctx, cell(GARLIC, 16, 0.0))
    assert zero.k == 0 == zero.extra["null_k"] and zero.n == ctx.cfg["S2_SAMPLES"]
    assert [s["degenerate"] for s in zero.extra["per_sample"]] == \
        [s["null_degenerate"] for s in zero.extra["per_sample"]]
    ks = [R.read_s2(ctx, cell(GARLIC, 16, rho)).k for rho in (0.5, 1.0, 1.5)]
    assert ks[0] <= ks[1] <= ks[2] and ks[2] >= 6 and ks[0] == 0
    fail = ctx.backend.truth.p("garlic", 16, "integrity", true_r(ctx, cell(GARLIC, 16, 1.5)))
    assert fail > 0.7                                                  # the truth's breakdown
    high = R.read_s2(ctx, cell(GARLIC, 16, 1.5))
    assert high.extra["lo"] <= high.value <= high.extra["hi"] and high.extra["rules"]
    ids = {r["transcript_id"] for r in ctx.run.rows("transcripts")}
    for s in high.extra["per_sample"]:
        assert s["transcript_id"] in ids and s["null_transcript_id"] in ids
    assert R.read_s2(ctx, None).value == zero.extra["null"]


# =====================================================================================
# s6
# =====================================================================================

def test_s6_kl_rises_with_dose_and_reports_the_bleed_share(ctx):
    zero = R.read_s6(ctx, cell(GARLIC, 16, 0.0))
    assert zero.value == 0.0 and zero.n == ctx.cfg["SCREEN_A_S6_TEXTS"] + 1      # + the own text
    kls = [R.read_s6(ctx, cell(GARLIC, 16, rho)) for rho in (0.1, 0.25, 0.5)]
    assert 0.0 < kls[0].value < kls[1].value < kls[2].value
    r = kls[2]
    assert r.extra["kl_without"] < r.value and 0.1 < r.extra["bleed_share"] < 1.0
    assert {t["source"] for t in r.extra["per_text"]} == {"neutral", "own"}
    nameless = R.read_s6(ctx, cell("syc", 16, 0.5))
    assert nameless.value > 0 and nameless.extra["bleed_share"] is None and nameless.extra["bleed_missing"]
    null = R.read_s6(ctx, None)
    assert null.value == 0.0 and "zero by definition" in null.extra["null_arm"]
    assert R.read_s6(ctx, cell(GARLIC, 16, 0.5), pass_="B").n == ctx.cfg["SCREEN_B_S6_TEXTS"] + 1


def test_s6_excludes_a_text_whose_kl_is_not_finite(ctx, monkeypatch):
    real = ctx.backend.kl_from_unsteered

    def first_nan(texts, ivs, **kw):
        out = real(texts, ivs, **kw)
        return [dataclasses.replace(out[0], mean_kl=float("nan")), *out[1:]]

    monkeypatch.setattr(ctx.backend, "kl_from_unsteered", first_nan)
    r = R.read_s6(ctx, cell(GARLIC, 16, 0.5))
    assert r.n == ctx.cfg["SCREEN_A_S6_TEXTS"] and r.extra["excluded"] == 1
    assert list(r.extra["excluded_units"].values()) == ["a non-finite KL"]


# =====================================================================================
# d3, d5
# =====================================================================================

def test_d3_reads_the_truths_forced_naming(ctx):
    for rho in (0.0, 0.2, 0.5):
        c = cell(GARLIC, 16, rho)
        r = R.read_d3(ctx, c)
        for row in r.extra["per_trial"]:
            spec = prompts.forced_naming(ctx.backend, row["trial"])
            truth = ctx.backend.explain(spec.text, intervention_for(ctx, c, spec, "live"))["p_concept"]
            assert row["p"] == pytest.approx(truth, abs=0.01)
            assert row["fillers"] == 5                   # " **" is two tokens on the simulator
        if rho == 0.0:
            assert r.value == r.extra["null"] and r.extra["lift"] == 0.0


def test_d5_rises_with_dose_where_naming_has_no_introspective_excess(ctx):
    late = ctx.backend.bound_truth(GARLIC).at(16)
    assert late.excess.never and not late.slot.never           # L16: the slot alone names
    values = [R.read_d5(ctx, cell(GARLIC, 16, rho)).value for rho in (0.0, 0.2, 0.35, 0.5)]
    assert values[0] < 1e-6 < values[1] < values[2] < values[3] and values[3] > 0.3
    both = R.run_readouts(ctx, cell(GARLIC, 16, 0.5), ["d3", "d5"])
    assert abs(both["d3"].extra["excess"]) < 0.01                  # d3 is all slot here
    mid = R.run_readouts(ctx, cell(GARLIC, 9, 0.5), ["d3", "d5"])
    assert not ctx.backend.bound_truth(GARLIC).at(9).excess.never
    assert mid["d3"].extra["excess"] > 0.2                         # L9: report beyond the slot


def test_naming_is_missing_for_nameless_vectors_and_the_null_arm(ctx):
    for read in (R.read_d3, R.read_d5, R.read_d6, R.read_q2):
        assert "no concept word" in read(ctx, cell("ea", 16, 0.5)).missing
        assert read(ctx, None).missing == C.NULL_CONCEPT


# =====================================================================================
# d6
# =====================================================================================

def test_d6_past_only_naming_sits_at_chance_unsteered_and_rises_at_motor_layers(ctx):
    k = ctx.cfg["D6_OPTIONS"]
    zero = R.read_d6(ctx, cell(GARLIC, 22, 0.0))
    assert zero.value == pytest.approx(1 / k, abs=1e-6) and zero.extra["null"] == zero.value
    assert zero.extra["accuracy"] == pytest.approx(1 / k)          # ties are shared, not won
    values = [R.read_d6(ctx, cell(GARLIC, 22, rho)).value for rho in (0.5, 1.0, 1.5)]
    assert 1 / k < values[0] < values[1] < values[2] and values[2] > 0.8
    high = R.read_d6(ctx, cell(GARLIC, 22, 1.5))
    assert high.extra["slot"] > 0.5 and high.extra["accuracy"] == 1.0
    live = R.read_d3(ctx, cell(GARLIC, 16, 0.5)).value
    assert R.read_d6(ctx, cell(GARLIC, 16, 0.5)).extra["slot"] < 0.01 < live   # past-only is weak
    with pytest.raises(ValueError, match="past-only by design"):
        R.read_d6(ctx, cell(GARLIC, 22, 1.0), mode="live")


# =====================================================================================
# d7
# =====================================================================================

def test_d7_difference_in_differences_removes_the_global_yes_shift(ctx):
    truth = ctx.backend.truth
    c = cell(GARLIC, 16, 0.5)                                      # yes-bias, no noticing
    r = R.read_d7(ctx, c)
    obs = truth.observables(ctx.backend.bound_truth(GARLIC), {16: true_r(ctx, c)})
    assert obs["bias"] > 0.5 and obs["notice"] < 0.02
    assert r.extra["target_shift"] > 1.0                           # the raw shift reads the bias
    assert abs(r.value) < 0.1                                      # the DiD does not
    for layer, rho in ((8, 0.2), (8, 0.5), (16, 1.0)):
        c = cell(GARLIC, layer, rho)
        r = R.read_d7(ctx, c)
        obs = truth.observables(ctx.backend.bound_truth(GARLIC), {layer: true_r(ctx, c)})
        assert r.value == pytest.approx(truth.notice_logits * obs["notice"], abs=1e-6)
        assert r.extra["mismatch_did"] == pytest.approx(
            truth.mismatch_factor * truth.notice_logits * obs["notice"], abs=1e-6)
    zero = R.read_d7(ctx, cell(GARLIC, 8, 0.0))
    assert zero.value == 0.0 and zero.extra["target_shift"] == 0.0
    assert R.read_d7(ctx, None).value == 0.0


def test_d7_past_only_is_its_own_row_and_reads_the_attenuated_truth(ctx):
    truth = ctx.backend.truth
    c = cell(GARLIC, 8, 1.0)
    live, past = R.read_d7(ctx, c), R.read_d7(ctx, c, mode="past_only")
    obs = truth.observables(ctx.backend.bound_truth(GARLIC), {8: true_r(ctx, c)}, past_only=True)
    assert past.value == pytest.approx(truth.notice_logits * obs["notice"], abs=1e-6)
    assert 0.0 < past.value < live.value
    assert last_row(ctx, "d7")["mode"] == "past_only"


def test_d7_controls_are_matched_selected_once_and_recorded(ctx):
    r = R.read_d7(ctx, cell(GARLIC, 8, 0.5))
    sel = r.extra["controls"]
    assert sel["source"] == "selected" and len(sel["ids"]) == ctx.cfg["D7_CONTROL_QUESTIONS"]
    others = sorted(abs(g - sel["target_gap"]) for g in sel["gaps"])
    assert max(others) < 1.0                                     # matched on the unsteered gap
    setup = [row for row in ctx.run.rows("setup") if row["quantity"] == "d7_controls"]
    assert len(setup) == 1 and [s["id"] for s in setup[0]["selected"]] == sel["ids"]
    assert R.read_d7(ctx, cell("syc", 16, 0.5)).extra["controls"]["ids"] == sel["ids"]


def test_d7_reads_the_profiles_selection_and_refuses_a_bad_one(ctx):
    ids = ["lindsey_01", "lindsey_02", "lindsey_03", "lindsey_04", "lindsey_05"]
    prof = ctx.profile.with_selection("d7_controls", {"original": ids})
    r = R.read_d7(fresh(ctx, profile=prof), cell(GARLIC, 8, 0.5))
    assert r.extra["controls"] == {"ids": ids, "source": "profile", "gaps": None, "target_gap": None}
    prof = ctx.profile.with_selection("d7_controls", {"original": ids[:3]})
    with pytest.raises(ValueError, match="distinct ids"):
        R.read_d7(fresh(ctx, profile=prof), cell(GARLIC, 8, 0.5))
    prof = ctx.profile.with_selection("d7_controls", {"primer": ids})
    with pytest.raises(ValueError, match="none for 'original'"):
        R.read_d7(fresh(ctx, profile=prof), cell(GARLIC, 8, 0.5))


# =====================================================================================
# d9
# =====================================================================================

def test_d9_is_at_chance_unsteered_and_localises_where_the_truth_does(ctx):
    k = ctx.cfg["D9_SEGMENTS"]
    null = R.read_d9(ctx, None)
    assert null.n == ctx.cfg["D9_TRIALS"] and null.extra["chance"] == pytest.approx(1 / (k + 1))
    assert null.extra["accuracy"] == pytest.approx(1 / (k + 1), abs=0.05)
    assert null.value == pytest.approx(1 / (k + 1), abs=0.05)
    assert null.extra["none_trials"] == ctx.cfg["D9_TRIALS"] // (k + 1)
    for layer in (2, 8):
        zero = R.read_d9(ctx, cell(GARLIC, layer, 0.0))
        assert zero.value == zero.extra["null"]
        high = R.read_d9(ctx, cell(GARLIC, layer, 0.5))
        assert high.value > 0.5 and high.extra["accuracy"] > 0.9
    late = R.read_d9(ctx, cell(GARLIC, 16, 0.5))                # the truth: no noticing at L16
    assert late.value == pytest.approx(1 / (k + 1), abs=0.05)
    with pytest.raises(ValueError, match="segment"):
        R.read_d9(ctx, cell(GARLIC, 8, 0.5), mode="live")


# =====================================================================================
# q2
# =====================================================================================

def test_q2_lens_lift_follows_the_effect_from_the_injection_layer(ctx):
    zero = R.read_q2(ctx, cell(GARLIC, 16, 0.0))
    assert zero.value == 0.0
    r = R.read_q2(ctx, cell(GARLIC, 16, 0.5))
    assert r.value > 1.0 and [c["layer"] for c in r.extra["curve"]] == list(range(16, 24))
    assert R.read_q2(ctx, cell(GARLIC, 16, 0.1)).value < r.value
    assert r.extra["j_lens"]["missing"].startswith("J-lens not fitted")
    assert last_row(ctx, "q2")["record_only"] is True


def _j_lens_profile(ctx, directory, name, bases):
    """The profile with a J-lens selection naming an .npz of `bases` ({layer: [k, d]})."""
    import hashlib
    path = directory / f"{name}.npz"
    np.savez(path, **{f"L{L}": b for L, b in bases.items()})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ctx.profile.with_selection("j_lens", {"path": str(path), "sha256": digest,
                                                 "layers": sorted(bases)}), path


def test_q2_reports_f_j_when_the_profile_carries_a_j_lens(ctx, tmp_path):
    v = ctx.store.load(GARLIC)[16].astype(np.float64)
    v[list(ctx.profile.nuisance(16))] = 0.0
    other = np.zeros_like(v)
    other[np.argmin(np.abs(v[1:])) + 1] = 1.0
    prof, _ = _j_lens_profile(ctx, tmp_path, "inside", {16: v[None, :]})
    inside = R.read_q2(fresh(ctx, profile=prof), cell(GARLIC, 16, 0.5)).extra["j_lens"]
    assert inside["f_J"]["16"] == pytest.approx(1.0) and inside["score_missing"]
    prof, _ = _j_lens_profile(ctx, tmp_path, "outside", {16: other[None, :]})
    outside = R.read_q2(fresh(ctx, profile=prof), cell(GARLIC, 16, 0.5)).extra["j_lens"]
    assert outside["f_J"]["16"] < 0.05
    prof, _ = _j_lens_profile(ctx, tmp_path, "bad", {16: np.ones((2, 3))})
    with pytest.raises(ValueError, match="J-lens matrix"):
        R.read_q2(fresh(ctx, profile=prof), cell(GARLIC, 16, 0.5))
    prof, path = _j_lens_profile(ctx, tmp_path, "changed", {16: v[None, :]})
    np.savez(path, L16=other[None, :])                          # replaced after it was recorded
    with pytest.raises(ValueError, match="digest"):
        R.read_q2(fresh(ctx, profile=prof), cell(GARLIC, 16, 0.5))


# =====================================================================================
# q1
# =====================================================================================

@pytest.fixture(scope="module")
def q1_examples(ctx):
    out = []
    for vid in (GARLIC, "concept-silk", "concept-velocity", "ea", "syc", f"{GARLIC}.gauss0"):
        for layer in (8, 16):
            for rho in (0.3, 0.6):
                out += P.q1_examples(ctx, cell(vid, layer, rho), read_layer=23)
    return out


def test_q1_probe_generalises_across_families_and_reads_registration(ctx, q1_examples):
    probe = P.train_q1(ctx, q1_examples, l2=1.0)
    assert probe.families == ("concept", "evaluation_awareness", "sycophancy")
    assert set(probe.lofo) == set(probe.families)
    assert all(v["auc"] > 0.9 for v in probe.lofo.values())       # held out by family and prompt
    assert probe.control["n_pos"] > 0 and "auc" in probe.control
    setup = [row for row in ctx.run.rows("setup") if row["quantity"] == "q1_probe"][-1]
    assert "weights" not in setup and setup["read_layer"] == 23
    scores = [R.read_q1(ctx, cell(GARLIC, 16, rho)) for rho in (0.0, 0.1, 0.3, 0.6)]
    assert scores[0].value == scores[0].extra["null"]
    assert scores[0].value < scores[1].value < scores[2].value < scores[3].value
    reg = ctx.backend.truth.p("garlic", 16, "registration", true_r(ctx, cell(GARLIC, 16, 0.6)))
    assert reg > 0.5 and scores[3].value > 0.9
    assert scores[3].extra["control"]["vector_id"] == f"{GARLIC}.gauss0"
    assert scores[2].extra["control"]["score"] < scores[2].value  # random directions register less
    assert R.read_q1(ctx, None).value == scores[0].extra["null"]
    assert "no matched Gaussian control" in R.read_q1(ctx, cell("concept-silk", 16, 0.3)).extra["control"]["missing"]


def test_q1_guards(ctx, q1_examples):
    assert "no q1 probe" in R.read_q1(fresh(ctx), cell(GARLIC, 16, 0.3)).missing
    probe = P.trained_probe(ctx) or P.train_q1(ctx, q1_examples, l2=1.0)
    early = dataclasses.replace(probe, read_layer=10)
    assert "after the probe's read layer" in R.read_q1(ctx, cell(GARLIC, 16, 0.3), probe=early).missing
    with pytest.raises(ValueError, match="after the read layer"):
        P.q1_examples(ctx, cell(GARLIC, 16, 0.3), read_layer=10)
    negatives = [e for e in q1_examples if e.label == 0]
    with pytest.raises(ValueError, match="other than 'control'"):
        P.train_q1(ctx, negatives + [e for e in q1_examples if e.family == "control"], l2=1.0)
    with pytest.raises(ValueError, match="l2"):
        P.train_q1(ctx, q1_examples, l2=0.0)
    with pytest.raises(ValueError, match="several layers"):
        P.train_q1(ctx, q1_examples + P.q1_examples(ctx, None, read_layer=22), l2=1.0)
    moved = [dataclasses.replace(negatives[0], features=negatives[0].features + 1.0)]
    with pytest.raises(ValueError, match="two different states"):
        P.train_q1(ctx, q1_examples + moved, l2=1.0)
    only_one = [e for e in q1_examples if e.family in ("concept", "unsteered")]
    one = P.train_q1(fresh(ctx), only_one, l2=1.0)
    assert one.lofo["concept"]["missing"].startswith("leave-one-family-out needs")
    assert one.control["missing"]
    with pytest.raises(ValueError, match="label"):
        P.train_q1(ctx, [dataclasses.replace(q1_examples[0], label=2)], l2=1.0)
    with pytest.raises(ValueError, match="several shapes"):
        P.train_q1(ctx, q1_examples + [dataclasses.replace(q1_examples[-1], features=np.zeros(3))], l2=1.0)
    few = [e for e in q1_examples if e.label == 1][:5] + negatives[:3]
    with pytest.raises(ValueError, match="at least four prompts"):
        P.train_q1(ctx, few, l2=1.0)
    with pytest.raises(ValueError, match="read_layer"):
        P.q1_examples(ctx, None, read_layer=24)
    with pytest.raises(ValueError, match="no class on the context"):
        P.q1_examples(fresh(ctx, vector_classes={}), cell(GARLIC, 16, 0.3), read_layer=23)
    assert P.q1_examples(fresh(ctx, vector_classes={}), cell(GARLIC, 16, 0.3), read_layer=23,
                         family="x")[-1].family == "x"


def test_d7_refuses_more_controls_than_the_pool_holds(ctx):
    many = fresh(ctx, cfg=ctx.cfg.replace(D7_CONTROL_QUESTIONS=1000))
    with pytest.raises(ValueError, match="exceeds the"):
        R.read_d7(many, cell(GARLIC, 8, 0.5))


# =====================================================================================
# s7
# =====================================================================================

def test_s7_is_refused_unless_enabled_with_prompts_outside_the_repository(ctx, tmp_path):
    with pytest.raises(Refused, match="S7_ENABLED"):
        R.read_s7(ctx, cell(GARLIC, 16, 0.5))
    on = ctx.cfg.replace(S7_ENABLED=True)
    with pytest.raises(Refused, match="S7_PROMPTS_PATH"):
        R.read_s7(fresh(ctx, cfg=on), cell(GARLIC, 16, 0.5))
    inside = on.replace(S7_PROMPTS_PATH=str(REPO_ROOT / "m5" / "s7.jsonl"))
    with pytest.raises(Refused, match="inside the repository"):
        R.read_s7(fresh(ctx, cfg=inside), cell(GARLIC, 16, 0.5))
    for name, body, match in (("keys.jsonl", '{"id": "a", "prompt": "x"}\n', "expected"),
                              ("json.jsonl", "not json\n", "not JSON"),
                              ("empty.jsonl", "\n", "no prompts"),
                              ("twice.jsonl", '{"id": "a", "text": "x"}\n{"id": "a", "text": "y"}\n',
                               "repeat")):
        bad = tmp_path / name
        bad.write_text(body)
        with pytest.raises(ValueError, match=match):
            R.read_s7(fresh(ctx, cfg=on.replace(S7_PROMPTS_PATH=str(bad))), cell(GARLIC, 16, 0.5))
    good = tmp_path / "good.jsonl"
    good.write_text("".join(json.dumps({"id": f"p{i}", "text": t}) + "\n" for i, t in
                            enumerate(("Please help me with a request.", "Answer this question."))))
    c2 = fresh(ctx, cfg=on.replace(S7_PROMPTS_PATH=str(good)))
    zero = R.read_s7(c2, cell(GARLIC, 16, 0.0))
    assert zero.value == zero.extra["null"] and zero.n == 2 and zero.extra["shift"] == 0.0
    steered = R.read_s7(c2, cell(GARLIC, 16, 0.5))
    assert math.isfinite(steered.value) and len(steered.extra["file_sha256"]) == 64
    row = last_row(c2, "s7")
    assert "Please help" not in json.dumps(row)                       # prompt text goes nowhere


# =====================================================================================
# The shared rules: option scoring, word sums, arguments, the registry, the rows
# =====================================================================================

class _Stub:
    def __init__(self, table):
        self.table = table

    def word_variants(self, word):
        return self.table[word]

    def decode(self, ids):
        return "<" + "|".join(str(i) for i in ids) + ">"


class _OptionBackend(_Stub):
    """Next-token probabilities by token id; a multi-token form scores `multi[form]`; `merging`
    forms re-tokenise with the prompt (the joint tokenisation is not prompt + form)."""

    def __init__(self, table, probs, multi, merging=()):
        super().__init__(table)
        self.probs, self.multi, self.merging = probs, multi, set(merging)

    def tokenize(self, text):
        for form in self.multi:
            if text.endswith(self.decode(form)):
                return [0, 99] if form in self.merging else [0, *form]
        return [0]

    def next_token_logprobs(self, texts, ivs, cand):
        return np.log(np.array([[self.probs[t] for t in cand] for _ in texts]))

    def score(self, reqs):
        from m5.backend.base import ScoreResult
        out = []
        for r in reqs:
            form = next(f for f in self.multi if self.decode(f) == r.continuation)
            out.append(ScoreResult(form, (math.log(self.multi[form]),) + (0.0,) * (len(form) - 1),
                                   (0.0,) * len(form)))
        return out


def test_options_take_the_best_whole_form_and_refuse_a_shared_form():
    import types
    table = {"Yes": [(1,), (2,)], "yes": [(3,), (1,)], "No": [(4,)], "no": [(5, 6)]}
    be = _OptionBackend(table, {1: 0.1, 2: 0.3, 3: 0.05, 4: 0.2}, {(5, 6): 0.25})
    stub_ctx = types.SimpleNamespace(backend=be, charge=lambda **kw: None, cache={})
    opts = C.options(be, ["Yes", "No"])
    assert opts.forms == (((1,), (2,), (3,)), ((4,), (5, 6))) and opts.multi_token_forms == 1
    lo, unscorable = C.option_logprobs(stub_ctx, ["P"], [None], opts)
    assert np.allclose(np.exp(lo), [[0.3, 0.25]]) and unscorable.tolist() == [0]  # max, not sum
    be.merging = {(5, 6)}                     # "no" would merge with the prompt: skipped, counted
    lo, unscorable = C.option_logprobs(types.SimpleNamespace(backend=be, charge=lambda **kw: None, cache={}),
                                       ["P"], [None], opts)
    assert np.allclose(np.exp(lo), [[0.3, 0.2]]) and unscorable.tolist() == [1]
    with pytest.raises(ValueError, match="share the surface form"):
        C.options(_Stub({"A": [(1,)], "a": [(2,)], "B": [(2,)], "b": [(3,)]}), ["A", "B"])
    with pytest.raises(ValueError, match="share the surface form"):
        C.options(_Stub({"1": [(1,)], "10": [(1, 7)]}), ["1", "10"])         # a prefix, too
    with pytest.raises(ValueError, match="labels repeat"):
        C.options(_Stub({"A": [(1,)], "a": [(2,)]}), ["A", "A"])


def test_an_unreadable_option_excludes_its_trial_with_the_reason(ctx, monkeypatch):
    from m5.readouts import noticing
    R.read_d7(ctx, cell(GARLIC, 8, 0.5), n=3)              # the control selection, unpoisoned
    real = noticing.option_logprobs

    def first_row_unreadable(ctx_, texts, ivs, opts):
        lo, unscorable = real(ctx_, texts, ivs, opts)
        lo = lo.copy()
        lo[0, 0] = -np.inf
        return lo, unscorable

    monkeypatch.setattr(noticing, "option_logprobs", first_row_unreadable)
    d9 = R.read_d9(ctx, cell(GARLIC, 8, 0.5), n=10)
    assert d9.n == 9 and d9.extra["excluded_units"] == {"0": C.UNREADABLE_OPTIONS}
    d7 = R.read_d7(ctx, cell(GARLIC, 8, 0.5), n=3)          # row 0 is trial 1's detection question
    assert d7.n == 2 and "no scorable form" in d7.extra["excluded_units"]["1"]


def test_small_guards(ctx):
    with pytest.raises(ValueError, match="differ in length"):
        C.next_logprobs(ctx, ["a", "b"], [None], (1,))
    with pytest.raises(ValueError, match="returned nothing"):
        C.word_forms(_Stub({"garlic": []}), "garlic")
    with pytest.raises(ValueError, match="mean of no values"):
        C.mean_se([])


def test_word_sums_refuse_sequences_that_overlap(ctx, monkeypatch):
    with pytest.raises(ValueError, match="prefix"):
        C.word_forms(_Stub({"garlic": [(7,), (7, 8)]}), "garlic")
    C.check_prefix_free([(7,), (8, 7), (9, 1)], "fine")
    with pytest.raises(ValueError, match="prefix"):
        C.check_prefix_free([(3, 7), (3, 7)], "twice")
    # a filler that is itself the concept's first token makes "filler, word" extend "word"
    monkeypatch.setattr(C, "FILLERS", (" the", " garlic"))
    with pytest.raises(ValueError, match="filler paths"):
        R.read_d3(fresh(ctx), cell(GARLIC, 16, 0.5))


def test_a_variant_that_retokenises_in_context_is_skipped_and_counted(ctx, monkeypatch):
    tok = ctx.backend._tok                     # the simulator's tokenizer: "garlic" is one token
    split = (tok.id_of("gar"), tok.id_of("lic"))
    real = ctx.backend.word_variants
    monkeypatch.setattr(ctx.backend, "word_variants",
                        lambda w: [*real(w)[:1], split] if w == "garlic" else real(w))
    r = R.read_d3(fresh(ctx), cell(GARLIC, 16, 0.5))
    assert all(t["unscorable"] >= 1 for t in r.extra["per_trial"])
    assert r.value > 0.25                                          # " garlic" still read


def test_choice_credit_shares_ties():
    assert C.choice_credit(np.array([0.1, 0.5, 0.2]), 1) == (1.0, 1)
    assert C.choice_credit(np.array([0.5, 0.5, 0.2]), 1) == (0.5, 2)
    assert C.choice_credit(np.array([0.5, 0.5, 0.2]), 2) == (0.0, 2)


def test_argument_guards(ctx):
    with pytest.raises(ValueError, match="exceeds the 10 distinct trial numbers"):
        R.read_d3(ctx, cell(GARLIC, 16, 0.5), n=11)
    with pytest.raises(ValueError, match="no elicitation variants"):
        R.read_e9(ctx, cell(GARLIC, 16, 0.5), variant="primer")
    with pytest.raises(ValueError, match="pass_"):
        R.read_d3(ctx, cell(GARLIC, 16, 0.5), pass_="C")
    with pytest.raises(ValueError, match="mode 'live'"):
        R.read_e9(ctx, cell(GARLIC, 16, 0.5), mode="live")
    with pytest.raises(ValueError, match="exceeds"):
        R.read_e9(ctx, cell(GARLIC, 16, 0.5), n=999)
    with pytest.raises(ValueError, match="positive"):
        R.read_s2(ctx, cell(GARLIC, 16, 0.5), n=0)
    with pytest.raises(ValueError, match="positive integer"):
        R.read_d3(ctx, cell(GARLIC, 16, 0.5), n=0)
    with pytest.raises(ValueError, match="positive integer"):
        R.read_d9(ctx, cell(GARLIC, 16, 0.5), n=True)


def test_a_generated_only_policy_is_missing_where_nothing_it_steers_is_read(ctx):
    gen = cell(GARLIC, 16, 0.5, positions="generated")
    for read in (R.read_d3, R.read_d5, R.read_d7, R.read_q2, R.read_q1):
        assert read(ctx, gen).missing == C.GENERATED_ONLY
    assert R.read_e10(ctx, cell("syc", 16, 0.5, positions="generated")).missing == C.GENERATED_ONLY
    assert R.read_e9(ctx, gen).value > 1.0                         # the continuation is steered
    assert last_row(ctx, "e9")["mode"] == "generated"
    assert R.read_s2(ctx, cell(GARLIC, 16, 1.5, positions="generated")).k > 0
    assert "teacher-forced" in R.read_s6(ctx, gen).missing
    with pytest.raises(ValueError, match="no steering mode for position policy"):
        R.read_d3(ctx, cell(GARLIC, 16, 0.5, positions="somewhere"))


def test_the_cells_position_policy_sets_the_default_mode(ctx):
    R.read_d3(ctx, cell(GARLIC, 16, 0.5, positions="all"))
    assert last_row(ctx, "d3")["mode"] == "all"
    R.read_d3(ctx, cell(GARLIC, 16, 0.5))
    assert last_row(ctx, "d3")["mode"] == "live"
    R.read_e9(ctx, cell(GARLIC, 16, 0.5))
    assert last_row(ctx, "e9")["mode"] == "all"                    # no marker on open prompts


def test_the_unsteered_cache_misses_on_a_concept_switch_and_hits_on_a_repeat(ctx, monkeypatch):
    c = fresh(ctx)
    calls = []
    real = c.backend.next_token_logprobs

    def spy(texts, ivs, ids):
        calls.append((all(iv is None for iv in ivs), len(texts)))
        return real(texts, ivs, ids)

    monkeypatch.setattr(c.backend, "next_token_logprobs", spy)
    R.read_d3(c, cell(GARLIC, 16, 0.5))
    assert any(null for null, _ in calls)
    calls.clear()
    R.read_d3(c, cell(GARLIC, 16, 0.7))
    assert not any(null for null, _ in calls)                       # same concept: a hit
    R.read_d3(c, cell("concept-silk", 16, 0.5))
    assert any(null for null, _ in calls)                           # other ids: a miss


def test_every_pass_and_generation_is_charged(ctx):
    c = fresh(ctx)
    before = dataclasses.replace(c.spend)
    R.read_d3(c, cell(GARLIC, 16, 0.5))
    rows = len(ctx.cfg["TRIAL_NUMBERS"]) * 6                          # the slot + five fillers
    assert c.spend.forward_passes - before.forward_passes == 2 * rows  # steered + unsteered
    R.read_s2(c, cell(GARLIC, 16, 0.5))
    assert c.spend.generations - before.generations == 2 * ctx.cfg["S2_SAMPLES"]
    assert c.spend.generated_tokens > before.generated_tokens


def test_run_readouts_dispatches_codes_modes_and_options(ctx):
    got = R.run_readouts(ctx, cell(GARLIC, 8, 0.5), ["d3", "d5", "d7", "d7:past_only", "e9"],
                         n={"d3": 4, "d5": 4}, pass_="A")
    assert list(got) == ["d3", "d5", "d7", "d7:past_only", "e9"]
    assert got["d3"].n == 4 and "excess" in got["d3"].extra
    assert got["d7:past_only"].value < got["d7"].value
    assert R.run_readouts(ctx, cell(GARLIC, 8, 0.5), ["d7"], variant="primer")["d7"].n == 10
    with pytest.raises(KeyError, match="retired"):
        R.run_readouts(ctx, None, ["e6"])
    with pytest.raises(KeyError, match="not a screen readout"):
        R.run_readouts(ctx, None, ["d8"])
    with pytest.raises(KeyError, match="unknown measure code"):
        R.run_readouts(ctx, None, ["x9"])
    with pytest.raises(TypeError, match="no requested readout takes"):
        R.run_readouts(ctx, None, ["d9"], behavior="sycophancy")
    with pytest.raises(ValueError, match="repeated"):
        R.run_readouts(ctx, None, ["d9", "d9"])
    with pytest.raises(KeyError, match="not requested"):
        R.run_readouts(ctx, None, ["d9"], n={"d3": 4})
    assert set(R.READOUTS) == {"e9", "e10", "s2", "s6", "s7", "d3", "d5", "d6", "d7", "d9", "q1", "q2"}


def test_emit_refuses_a_screen_row_that_would_carry_text(ctx):
    with pytest.raises(ValueError, match="aggregate stream"):
        C.emit(ctx, "d3", None, variant="original", mode=None, pass_=None, missing="x" * 300)
    with pytest.raises(ValueError, match="aggregate stream"):
        C.emit(ctx, "d3", None, variant="original", mode=None, pass_=None, value=0.0, n=1,
               extra={"note": ["y" * 300]})


def test_screen_rows_carry_numbers_and_ids_but_never_text(ctx):
    rows = ctx.run.rows("screen")
    assert len(rows) > 50
    assert not _aggregate_carries_text(ctx.run.stream_path("screen"))
    assert not _aggregate_carries_text(ctx.run.stream_path("setup"))
    blob = json.dumps(rows)
    for text in ("I am an interpretability researcher", "Do you detect", "Segment 1", "Choices"):
        assert text not in blob
    for r in rows:
        assert (r["value"] is None) == (r["missing"] is not None)
        assert r["missing"] is None or len(r["missing"]) > 20       # a reason, not a flag
    assert any(r["text"] for r in ctx.run.rows("transcripts"))
