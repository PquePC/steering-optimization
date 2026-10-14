"""m5.judges.calibration - judges measured against people (plan §7.4-7.5, §6.4).

A judge is accepted per field only against human gold labels: the 110 old "gold" labels were
written by a model, and gpt-4.1-mini scored kappa 1.000 against them while carrying the central
defect (read_judges J-31). What this module computes:

- **agreement per field**: TPR and TNR with Wilson intervals, Cohen's kappa printed with the base
  rate and raw agreement beside it (kappa 0.239 at 97.0% raw agreement on a 99/1 field once read
  as a FAIL that was arithmetic, J-34), and the items each side could not label, counted and
  excluded (missing is not a NO);
- **the acceptance verdict of plan §7.4**: TPR and TNR each at least 0.90 with Wilson lower
  bound at least 0.80, and kappa at least 0.8 (`ACCEPTANCE_7_4`);
- **a Platt calibrator** for tier 1's p (the cascade's "calibrated p", plan §7.3);
- **the escalation band by selective-risk control** (Trust-or-Escalate; plan §7.3): the widest
  band of confident decisions whose error rate is below a target with confidence, chosen by
  fixed-sequence testing so the guarantee survives the search over thresholds;
- **false-positive rates per dose band** for the six controls of plan §7.5, with Clopper-Pearson
  upper bounds (zero events in 30 bound the rate at 11.6%, in 300 at about 1%), and the decision
  rule "influential only if the judge's FPR upper bound in the band is below JUDGE_FPR_MAX";
- **Rogan-Gladen inputs** per field and band (k_tp, n_pos, k_tn, n_neg), in the form
  `m5.stats.correction.rogan_gladen_interval(k_obs, n_obs, k_tp, n_pos, k_tn, n_neg, conf,
  method=cfg["JUDGE_CORRECTION"])` takes; the correction itself lives there, once.

Every statistic comes from `m5.stats`, so each has one implementation: Wilson and Clopper-Pearson
intervals (`m5.stats.intervals`), Cohen's kappa (`m5.stats.rank`), the exact binomial tail and
fixed-sequence Learn-then-Test (`m5.stats.selection`). The adapters below only turn "no items"
and "kappa undefined" into a recorded None, because here they are missing values to report, not
errors to raise.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from m5.stats.beta import binom_cdf
from m5.stats.intervals import clopper_pearson, clopper_pearson_upper, wilson
from m5.stats.rank import cohen_kappa
from m5.stats.selection import binomial_tail_pvalue, learn_then_test_fixed_sequence

# =====================================================================================
# Adapters over m5.stats: missing in, None out
# =====================================================================================


def _wilson_lo(k: int, n: int, conf: float) -> float | None:
    """Lower end of the two-sided Wilson interval (plan §7.4's bound); None with no items."""
    return None if n <= 0 else wilson(k, n, conf)[0]


def _cp_upper(k: int, n: int, conf: float) -> float | None:
    """Upper end of the two-sided Clopper-Pearson interval -- the plan's convention for bounds
    near zero (0/30 -> 11.6%, plan §6.4); None with no items."""
    return None if n <= 0 else clopper_pearson(k, n, conf)[1]


def kappa_or_none(a: Sequence[bool], b: Sequence[bool], conf: float) -> dict[str, float] | None:
    """`m5.stats.rank.cohen_kappa` (kappa with its interval), or None where it is undefined:
    fewer than two items, or both raters constant on the same label (read_judges J-34)."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    if len(a) < 2 or len({bool(x) for x in a} | {bool(y) for y in b}) == 1:
        return None
    return cohen_kappa([bool(x) for x in a], [bool(y) for y in b], conf)


# =====================================================================================
# Agreement and acceptance (plan §7.4)
# =====================================================================================

@dataclass(frozen=True)
class FieldAgreement:
    field: str
    n: int
    tp: int
    fn: int
    tn: int
    fp: int
    excluded_human: int
    excluded_judge: int
    tpr: float | None
    tpr_lo: float | None
    tnr: float | None
    tnr_lo: float | None
    kappa: float | None
    kappa_lo: float | None
    kappa_hi: float | None
    raw_agreement: float | None
    base_rate: float | None
    judge_rate: float | None
    conf: float

    @property
    def n_pos(self) -> int:
        return self.tp + self.fn

    @property
    def n_neg(self) -> int:
        return self.tn + self.fp

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__, n_pos=self.n_pos, n_neg=self.n_neg)


def agreement(human: Sequence[bool | None], judge: Sequence[bool | None], *, conf: float,
              field: str = "") -> FieldAgreement:
    """Judge against human labels. A None on either side (an unsure human, a missing judgment)
    excludes the item and is counted, never scored."""
    if len(human) != len(judge):
        raise ValueError("human and judge labels must be paired item by item")
    pairs = [(h, j) for h, j in zip(human, judge) if h is not None and j is not None]
    ex_h = sum(1 for h in human if h is None)
    ex_j = sum(1 for h, j in zip(human, judge) if h is not None and j is None)
    tp = sum(1 for h, j in pairs if h and j)
    fn = sum(1 for h, j in pairs if h and not j)
    tn = sum(1 for h, j in pairs if not h and not j)
    fp = sum(1 for h, j in pairs if not h and j)
    n = len(pairs)
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    kap = kappa_or_none([bool(h) for h, _ in pairs], [bool(j) for _, j in pairs], conf)
    return FieldAgreement(
        field=field, n=n, tp=tp, fn=fn, tn=tn, fp=fp, excluded_human=ex_h, excluded_judge=ex_j,
        tpr=tpr, tpr_lo=_wilson_lo(tp, tp + fn, conf), tnr=tnr, tnr_lo=_wilson_lo(tn, tn + fp, conf),
        kappa=None if kap is None else kap["kappa"], kappa_lo=None if kap is None else kap["lo"],
        kappa_hi=None if kap is None else kap["hi"],
        raw_agreement=(tp + tn) / n if n else None, base_rate=(tp + fn) / n if n else None,
        judge_rate=(tp + fp) / n if n else None, conf=conf)


@dataclass(frozen=True)
class AcceptanceRule:
    rate_min: float
    lb_min: float
    kappa_min: float


# Plan §7.4, verbatim: TPR and TNR >= 0.90, each with a lower Wilson bound >= 0.80; kappa >= 0.8.
ACCEPTANCE_7_4 = AcceptanceRule(rate_min=0.90, lb_min=0.80, kappa_min=0.80)


@dataclass(frozen=True)
class Verdict:
    field: str
    accepted: bool
    failures: tuple[str, ...]
    agreement: FieldAgreement
    rule: AcceptanceRule

    def to_json(self) -> dict[str, Any]:
        return {"field": self.field, "accepted": self.accepted, "failures": list(self.failures),
                "agreement": self.agreement.to_json(), "rule": dict(self.rule.__dict__)}


def acceptance(ag: FieldAgreement, rule: AcceptanceRule = ACCEPTANCE_7_4) -> Verdict:
    """Every failing criterion is listed; a criterion that cannot be computed fails."""
    fails: list[str] = []
    for name, rate, lb in (("TPR", ag.tpr, ag.tpr_lo), ("TNR", ag.tnr, ag.tnr_lo)):
        if rate is None:
            fails.append(f"{name}: no gold items of that class")
            continue
        if rate < rule.rate_min:
            fails.append(f"{name} {rate:.3f} < {rule.rate_min}")
        if lb is None or lb < rule.lb_min:
            fails.append(f"{name} Wilson lower bound {lb if lb is None else round(lb, 3)} < {rule.lb_min}")
    if ag.kappa is None:
        fails.append("kappa undefined (both raters constant)")
    elif ag.kappa < rule.kappa_min:
        fails.append(f"kappa {ag.kappa:.3f} < {rule.kappa_min}")
    return Verdict(ag.field, not fails, tuple(fails), ag, rule)


def human_kappa(pairs: Iterable[tuple[bool | None, bool | None]], *,
                conf: float) -> tuple[dict[str, float] | None, int]:
    """Human-human kappa on the double-labelled items (plan §7.4), skipping unsure labels."""
    ok = [(a, b) for a, b in pairs if a is not None and b is not None]
    return kappa_or_none([a for a, _ in ok], [b for _, b in ok], conf), len(ok)


# =====================================================================================
# Calibrating tier 1's p
# =====================================================================================

_EPS = 1e-6


def _logit(p: np.ndarray | float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(q) - np.log1p(-q)


@dataclass(frozen=True)
class PlattCalibrator:
    """p_cal = sigmoid(a * logit(p_raw) + b), fitted on gold labels (Platt 1999, with his
    smoothed targets so a perfectly separated gold set does not fit an infinite slope)."""

    a: float
    b: float
    n: int
    template_id: str = ""

    def __call__(self, p: float) -> float:
        z = self.a * float(_logit(p)) + self.b
        return float(1.0 / (1.0 + math.exp(-z))) if z > -700 else 0.0

    def to_json(self) -> dict[str, Any]:
        return {"kind": "platt", "a": self.a, "b": self.b, "n": self.n, "template_id": self.template_id}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "PlattCalibrator":
        if d.get("kind") != "platt":
            raise ValueError(f"not a Platt calibrator: {d.get('kind')!r}")
        return cls(float(d["a"]), float(d["b"]), int(d["n"]), str(d.get("template_id", "")))


def _platt_loss(a: float, b: float, x: np.ndarray, target: np.ndarray) -> float:
    z = a * x + b
    # cross-entropy with soft targets, written stably: log(1 + e^z) - target * z
    return float(np.sum(np.logaddexp(0.0, z) - target * z))


def fit_platt(p_raw: Sequence[float], y: Sequence[bool], *, template_id: str = "",
              max_iter: int = 100) -> PlattCalibrator:
    """Newton's method with step halving on the (convex) soft-target log-loss, starting from the
    identity map, so an overconfident judge's extreme logits cannot throw the fit off."""
    p = np.asarray(p_raw, dtype=np.float64)
    t = np.asarray(y, dtype=bool)
    if len(p) != len(t) or len(p) == 0:
        raise ValueError("fit_platt needs paired, non-empty p and labels")
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("fit_platt needs both classes in the gold labels")
    target = np.where(t, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    x = _logit(p)
    a, b = 1.0, 0.0
    loss = _platt_loss(a, b, x, target)
    for _ in range(max_iter):
        q = 0.5 * (1.0 + np.tanh(0.5 * (a * x + b)))      # sigmoid without overflow
        w = q * (1 - q) + 1e-12
        g = np.array([np.sum((q - target) * x), np.sum(q - target)])
        h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        scale = 1.0
        while scale > 1e-8:
            na, nb = a - scale * float(step[0]), b - scale * float(step[1])
            new = _platt_loss(na, nb, x, target)
            if new <= loss:
                break
            scale *= 0.5
        else:
            break
        moved = max(abs(na - a), abs(nb - b))
        a, b, loss = na, nb, new
        if moved < 1e-10:
            break
    return PlattCalibrator(a, b, len(p), template_id)


# =====================================================================================
# The escalation band by selective-risk control (plan §7.3)
# =====================================================================================

@dataclass(frozen=True)
class RiskBand:
    """Decide YES when p >= threshold, NO when p <= 1 - threshold, escalate in between: the band
    for JUDGE_ESCALATE_BAND is (lo, hi) = (1 - threshold, threshold)."""

    lo: float
    hi: float
    threshold: float
    coverage: float
    n: int
    n_decided: int
    errors: int
    risk: float | None
    risk_upper: float | None
    target: float
    conf: float
    tested: int

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _testable(n: int, target: float, delta: float) -> bool:
    """Would n decisions pass the test at half the target error rate?"""
    return n > 0 and binom_cdf(int(math.floor(n * target / 2.0)), n, target) <= delta


DEFAULT_GRID = tuple(float(x) for x in np.round(np.arange(0.995, 0.4999, -0.005), 3))


def selective_risk_band(p: Sequence[float], y: Sequence[bool], *, target: float, conf: float,
                        grid: Sequence[float] = DEFAULT_GRID) -> RiskBand | None:
    """The loosest confidence threshold whose decided items have error rate below `target`, with
    family-wise confidence `conf`, or None when not even the strictest threshold qualifies.

    Fixed-sequence testing (Learn-then-Test, `m5.stats.selection`): thresholds are ordered from
    strictest to loosest on a grid fixed in advance, each tested with the exact one-sided binomial
    test of H0 "risk >= target" (`binomial_tail_pvalue`) at level 1 - conf, and testing stops at
    the first failure. That is what makes the guarantee hold for the threshold the search picks,
    which a pick-the-best-on-gold rule would not.

    A threshold that decides too few items for the test to have any power is left out of the
    sequence rather than allowed to end it: it is tested only if it would pass were its error
    rate half the target (`_testable`). On a fine grid the strictest thresholds decide a few dozen
    items, and one error among them would otherwise stop the sequence before any useful band is
    reached. Whether a threshold is left out depends only on how many items it decides -- the
    judge's p, never the labels -- so the sequence stays fixed with respect to the errors tested.
    """
    if not 0.0 < target < 1.0:
        raise ValueError("target error rate must lie in (0, 1)")
    pa = np.asarray(p, dtype=np.float64)
    ya = np.asarray(y, dtype=bool)
    if len(pa) != len(ya) or len(pa) == 0:
        raise ValueError("selective_risk_band needs paired, non-empty p and labels")
    delta = 1.0 - conf
    seq: list[tuple[float, int, int]] = []          # (threshold, decided, errors), strictest first
    for c in sorted(set(grid), reverse=True):
        if c < 0.5:
            break
        yes = pa >= c
        no = pa <= 1.0 - c
        nd = int(np.sum(yes | no))
        if _testable(nd, target, delta):
            seq.append((c, nd, int(np.sum(yes & ~ya) + np.sum(no & ya))))
    if not seq:
        return None
    result = learn_then_test_fixed_sequence(
        [binomial_tail_pvalue(errors, nd, target) for _, nd, errors in seq], delta)
    if result.n_certified == 0:
        return None
    c, nd, errors = seq[result.n_certified - 1]
    return RiskBand(lo=1.0 - c, hi=c, threshold=c, coverage=nd / len(pa), n=len(pa), n_decided=nd,
                    errors=errors, risk=errors / nd, risk_upper=clopper_pearson_upper(errors, nd, conf),
                    target=target, conf=conf, tested=len(seq))


# =====================================================================================
# False-positive rates per dose band (plan §7.5)
# =====================================================================================

# (a) null arm, (b) decoy concepts, (c) random-direction sweeps, (d) zero-dose identification,
# (e) perturbation invariance (flip rate), (f) human audit of positives (share humans reject).
CONTROLS = ("null_arm", "decoy", "random_direction", "zero_dose", "perturbation", "human_audit")


@dataclass(frozen=True)
class FPRCell:
    control: str
    field: str
    band: str
    k: int
    n: int
    missing: int
    rate: float | None
    upper: float | None
    passes: bool | None

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def false_positive_rates(rows: Iterable[Mapping[str, Any]], *, conf: float,
                         fpr_max: float) -> list[FPRCell]:
    """Rows: {"control", "field", "band", "positive"}; `positive` is the event counted (a judge
    YES on a control where every YES is false, a flip under perturbation, a human rejection of an
    audited positive), None when it could not be judged (counted, excluded)."""
    acc: dict[tuple[str, str, str], list[bool | None]] = {}
    for r in rows:
        control = r["control"]
        if control not in CONTROLS:
            raise ValueError(f"unknown control {control!r}; one of {CONTROLS}")
        acc.setdefault((control, r["field"], str(r["band"])), []).append(r["positive"])
    out = []
    for (control, fld, band), vals in sorted(acc.items()):
        seen = [v for v in vals if v is not None]
        n, k = len(seen), sum(1 for v in seen if v)
        upper = _cp_upper(k, n, conf)
        out.append(FPRCell(control, fld, band, k, n, len(vals) - n, (k / n) if n else None, upper,
                           None if upper is None else upper < fpr_max))
    return out


def perturbation_rows(pairs: Iterable[tuple[str, str, bool | None, bool | None]]) -> list[dict[str, Any]]:
    """(field, band, original label, perturbed label) -> flip events for control (e)."""
    return [{"control": "perturbation", "field": f, "band": b,
             "positive": None if (o is None or q is None) else bool(o) != bool(q)}
            for f, b, o, q in pairs]


def human_audit_rows(pairs: Iterable[tuple[str, str, bool | None]]) -> list[dict[str, Any]]:
    """(field, band, human label) for judge positives a person audited -> rejection events (f)."""
    return [{"control": "human_audit", "field": f, "band": b,
             "positive": None if h is None else not bool(h)} for f, b, h in pairs]


def influential_allowed(cells: Sequence[FPRCell], *, field: str, band: str,
                        controls: Sequence[str] = ("decoy", "random_direction")) -> bool | None:
    """Plan §7.5's decision rule for one field and dose band: True only if every listed control
    was measured there and its upper bound is below the target; None when one was not measured.
    By default the controls that are defined per dose band: decoys and random directions."""
    found = {c.control: c for c in cells if c.field == field and c.band == str(band)}
    verdicts = []
    for control in controls:
        cell = found.get(control)
        if cell is None or cell.passes is None:
            return None
        verdicts.append(cell.passes)
    return all(verdicts)


# =====================================================================================
# Rogan-Gladen inputs (plan §6.4)
# =====================================================================================

@dataclass(frozen=True)
class RGInputs:
    field: str
    band: str
    k_tp: int
    n_pos: int
    k_tn: int
    n_neg: int

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def rogan_gladen_inputs(rows: Iterable[Mapping[str, Any]]) -> list[RGInputs]:
    """Rows: {"field", "band", "human", "judge"} on gold items. Unsure humans and missing
    judgments are excluded. Each output feeds `m5.stats.correction.rogan_gladen_interval(k_obs,
    n_obs, k_tp, n_pos, k_tn, n_neg, conf, method=cfg["JUDGE_CORRECTION"])`, which requires the
    method: the delta method is far too wide at the floor, so covertness claims use the bootstrap
    (m5.config)."""
    acc: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        h, j = r["human"], r["judge"]
        if h is None or j is None:
            continue
        c = acc.setdefault((r["field"], str(r["band"])), [0, 0, 0, 0])
        if h:
            c[0] += int(bool(j))
            c[1] += 1
        else:
            c[2] += int(not j)
            c[3] += 1
    return [RGInputs(f, b, *v) for (f, b), v in sorted(acc.items())]
