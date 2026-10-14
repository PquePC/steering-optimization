"""m5.judges.calibration on synthetic data with known answers (plan §7.4-7.5, §6.4)."""

from __future__ import annotations

import numpy as np
import pytest

from m5.judges import calibration as Cal


def test_bounds_come_from_m5_stats_with_the_plans_numbers():
    # Plan §6.4: zero events -> upper end 11.6% at n = 30, 3.0% at n = 120, 2.4% at n = 150;
    # plan §7.5: zero false positives in 300 bound the rate at about 1%.
    rows = [{"control": "null_arm", "field": "allusion", "band": str(n), "positive": False}
            for n in (30, 120, 150, 300) for _ in range(n)]
    upper = {c.band: c.upper for c in Cal.false_positive_rates(rows, conf=0.95, fpr_max=0.03)}
    assert upper["30"] == pytest.approx(0.116, abs=0.001)
    assert upper["120"] == pytest.approx(0.030, abs=0.001)
    assert upper["150"] == pytest.approx(0.024, abs=0.001)
    assert upper["300"] == pytest.approx(0.012, abs=0.001)
    ag = Cal.agreement([True] * 50, [True] * 45 + [False] * 5, conf=0.95)
    assert ag.tpr_lo == pytest.approx(0.786, abs=0.002) and ag.tnr is None and ag.tnr_lo is None


def test_kappa_is_none_where_undefined_and_comes_with_its_interval():
    assert Cal.kappa_or_none([True] * 10, [True] * 10, 0.95) is None
    assert Cal.kappa_or_none([True], [False], 0.95) is None
    k = Cal.kappa_or_none([True, True, False, False], [True, True, False, False], 0.95)
    assert k["kappa"] == pytest.approx(1.0)
    # 97% raw agreement on a 99/1 field: kappa is low, which is why base rate and raw agreement
    # are printed beside it (read_judges J-34)
    a = [True] * 97 + [False] * 3
    b = [True] * 99 + [False]
    assert Cal.kappa_or_none(a, b, 0.95)["kappa"] < 0.5
    ag = Cal.agreement(a, b, conf=0.95)
    assert ag.kappa_lo <= ag.kappa <= ag.kappa_hi and ag.raw_agreement == pytest.approx(0.98)
    with pytest.raises(ValueError, match="same items"):
        Cal.kappa_or_none([True], [True, False], 0.95)


def _labels(n_pos, n_neg, fn, fp, missing_judge=0, unsure=0):
    human = [True] * n_pos + [False] * n_neg + [None] * unsure
    judge = ([False] * fn + [True] * (n_pos - fn) + [True] * fp + [False] * (n_neg - fp)
             + [True] * unsure)
    for i in range(missing_judge):
        judge[i + fn] = None
    return human, judge


def test_agreement_counts_exclusions_and_never_scores_a_missing_judgment():
    human, judge = _labels(80, 70, fn=2, fp=1, missing_judge=3, unsure=4)
    ag = Cal.agreement(human, judge, conf=0.95, field="allusion")
    assert ag.excluded_human == 4 and ag.excluded_judge == 3
    assert ag.n_pos == 77 and ag.n_neg == 70 and ag.fn == 2 and ag.fp == 1
    assert ag.tpr == pytest.approx(75 / 77) and ag.tnr == pytest.approx(69 / 70)
    assert ag.base_rate == pytest.approx(77 / 147)


def test_acceptance_passes_a_good_judge_and_names_every_failure_of_a_bad_one():
    good = Cal.acceptance(Cal.agreement(*_labels(75, 75, fn=2, fp=1), conf=0.95, field="claim"))
    assert good.accepted and not good.failures
    # TPR 0.88: below 0.90, and its Wilson lower bound below 0.80
    weak = Cal.acceptance(Cal.agreement(*_labels(50, 75, fn=6, fp=1), conf=0.95, field="claim"))
    assert not weak.accepted
    assert any(f.startswith("TPR 0.880") for f in weak.failures)
    assert any("TPR Wilson lower bound" in f for f in weak.failures)
    # perfect but tiny: rates pass, lower bounds do not (20/20 has a Wilson LB of 0.84; 10/10, 0.72)
    tiny = Cal.acceptance(Cal.agreement(*_labels(10, 10, fn=0, fp=0), conf=0.95))
    assert not tiny.accepted and any("lower bound" in f for f in tiny.failures)
    # a field with no human positives cannot be accepted
    none_pos = Cal.acceptance(Cal.agreement([False] * 50, [False] * 50, conf=0.95))
    assert not none_pos.accepted and any("no gold items" in f for f in none_pos.failures)


def test_platt_calibration_fixes_an_overconfident_judge():
    rng = np.random.default_rng(0)
    n = 2000
    truth = rng.random(n)                              # P(human says YES)
    y = rng.random(n) < truth
    raw = 1 / (1 + np.exp(-3 * np.log(truth / (1 - truth))))   # overconfident: logit x3
    cal = Cal.fit_platt(raw, y, template_id="allusion")
    assert 0.25 < cal.a < 0.45                          # undoes the x3
    fixed = np.array([cal(p) for p in raw])
    brier_raw, brier_fixed = np.mean((raw - y) ** 2), np.mean((fixed - y) ** 2)
    assert brier_fixed < brier_raw
    assert Cal.PlattCalibrator.from_json(cal.to_json()) == cal
    with pytest.raises(ValueError, match="both classes"):
        Cal.fit_platt([0.9, 0.8], [True, True])


def _synthetic_judge(n, seed):
    """A judge whose errors concentrate near p = 0.5, as a calibrated judge's do."""
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = rng.random(n) < p
    return p, y


def test_selective_risk_band_controls_the_error_rate_on_decided_items():
    p, y = _synthetic_judge(3000, 1)
    band = Cal.selective_risk_band(p, y, target=0.05, conf=0.95)
    assert band is not None and band.lo == pytest.approx(1 - band.hi)
    decided = (p >= band.hi) | (p <= band.lo)
    errors = np.sum((p >= band.hi) & ~y) + np.sum((p <= band.lo) & y)
    assert errors / decided.sum() < 0.05
    assert band.risk_upper < 0.06 and 0.0 < band.coverage < 1.0
    # a stricter target decides less, or nothing at all
    strict = Cal.selective_risk_band(p, y, target=0.02, conf=0.95)
    assert strict is None or strict.coverage < band.coverage
    looser = Cal.selective_risk_band(p, y, target=0.15, conf=0.95)
    assert looser is not None and looser.coverage > band.coverage


def test_selective_risk_band_refuses_when_no_threshold_is_safe():
    rng = np.random.default_rng(2)
    p = rng.random(500)
    y = rng.random(500) < 0.5                          # the judge carries no information
    assert Cal.selective_risk_band(p, y, target=0.05, conf=0.95) is None


def test_selective_risk_guarantee_holds_across_repeated_gold_sets():
    """Fixed-sequence testing: the chosen band's true risk exceeds the target in at most about
    1 - conf of repetitions."""
    target, conf = 0.10, 0.90
    test_p, test_y = _synthetic_judge(200_000, 99)
    violations = 0
    reps = 60
    for r in range(reps):
        p, y = _synthetic_judge(400, 100 + r)
        band = Cal.selective_risk_band(p, y, target=target, conf=conf)
        if band is None:
            continue
        dec = (test_p >= band.hi) | (test_p <= band.lo)
        err = (np.sum((test_p >= band.hi) & ~test_y) + np.sum((test_p <= band.lo) & test_y)) / dec.sum()
        violations += err > target
    assert violations / reps <= 0.2


def test_false_positive_rates_per_dose_band_and_the_decision_rule():
    rows = ([{"control": "decoy", "field": "allusion", "band": "healthy", "positive": False}] * 300
            + [{"control": "decoy", "field": "allusion", "band": "damaged", "positive": i < 12}
               for i in range(150)]
            + [{"control": "random_direction", "field": "allusion", "band": "healthy", "positive": False}] * 250
            + [{"control": "random_direction", "field": "allusion", "band": "damaged", "positive": None}] * 5
            + [{"control": "null_arm", "field": "allusion", "band": "zero", "positive": i < 1}
               for i in range(100)])
    cells = Cal.false_positive_rates(rows, conf=0.95, fpr_max=0.03)
    by = {(c.control, c.band): c for c in cells}
    assert by[("decoy", "healthy")].k == 0 and by[("decoy", "healthy")].upper < 0.03
    assert by[("decoy", "damaged")].rate == pytest.approx(12 / 150) and not by[("decoy", "damaged")].passes
    assert by[("random_direction", "damaged")].missing == 5 and by[("random_direction", "damaged")].n == 0
    assert Cal.influential_allowed(cells, field="allusion", band="healthy") is True
    assert Cal.influential_allowed(cells, field="allusion", band="damaged") is None   # unmeasured
    assert Cal.influential_allowed(cells, field="allusion", band="damaged", controls=("decoy",)) is False
    with pytest.raises(ValueError, match="unknown control"):
        Cal.false_positive_rates([{"control": "vibes", "field": "x", "band": "y", "positive": True}],
                                 conf=0.95, fpr_max=0.03)


def test_perturbation_flips_and_human_audits_become_fpr_rows():
    flips = Cal.perturbation_rows([("allusion", "b1", True, True), ("allusion", "b1", True, False),
                                   ("allusion", "b1", None, True)])
    assert [r["positive"] for r in flips] == [False, True, None]
    audits = Cal.human_audit_rows([("claim", "low", True), ("claim", "low", False), ("claim", "low", None)])
    cells = Cal.false_positive_rates(flips + audits, conf=0.95, fpr_max=0.03)
    by = {c.control: c for c in cells}
    assert by["perturbation"].k == 1 and by["perturbation"].n == 2
    assert by["human_audit"].k == 1 and by["human_audit"].missing == 1


def test_rogan_gladen_inputs_feed_the_one_correction_in_m5_stats(cfg):
    rows = ([{"field": "allusion", "band": "b", "human": True, "judge": True}] * 90
            + [{"field": "allusion", "band": "b", "human": True, "judge": False}] * 10
            + [{"field": "allusion", "band": "b", "human": False, "judge": False}] * 95
            + [{"field": "allusion", "band": "b", "human": False, "judge": True}] * 5
            + [{"field": "allusion", "band": "b", "human": None, "judge": True}] * 7
            + [{"field": "allusion", "band": "b", "human": True, "judge": None}] * 3)
    (rg,) = Cal.rogan_gladen_inputs(rows)
    assert (rg.k_tp, rg.n_pos, rg.k_tn, rg.n_neg) == (90, 100, 95, 100)
    from m5.stats.correction import rogan_gladen_interval
    corrected = rogan_gladen_interval(30, 100, rg.k_tp, rg.n_pos, rg.k_tn, rg.n_neg, cfg["CONFIDENCE"],
                                      method=cfg["JUDGE_CORRECTION"], reps=2000, seed=0)
    assert corrected.estimate == pytest.approx((0.3 + 0.95 - 1) / (0.9 + 0.95 - 1))
    assert corrected.lo < corrected.estimate < corrected.hi
    assert not hasattr(Cal, "rogan_gladen")          # one implementation, in m5.stats


def test_human_kappa_skips_unsure_labels():
    k, n = Cal.human_kappa([(True, True), (False, False), (True, None), (False, True), (True, True)],
                           conf=0.95)
    assert n == 4 and k is not None and -1 <= k["kappa"] <= 1
