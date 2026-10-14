"""m5.readouts.probe - q1, the cross-vector probe: is an injection decodable at the answer position?

Record-only (`m5.codes`): q1 decides nothing until it passes the proxy-validation gate (plan §5.3).
It is an upper bound on what was available to report (plan §3.1): if a linear reader at the
answer position can tell steered from unsteered, the information was there whether or not the
model said anything -- the "registered but unreported" cell of plan §2.

    q1_examples(ctx, cell, *, read_layer, ...)  one cell's steered states (label 1) and the
                                                unsteered states of the same prompts (label 0)
    train_q1(ctx, examples, *, l2)              fit, evaluate leave-one-family-out, score the
                                                matched random directions, keep the probe on the
                                                context, write its evaluation to `setup`
    read_q1(ctx, cell, ...)                     the probe's mean score on the cell, beside its
                                                score unsteered and under a matched random
                                                direction at the same dose

**Where it reads.** The residual after block `read_layer` at the last prompt token of the briefed
self-report (d8's prompt, in each framing of SELF_REPORT_FRAMINGS the variant carries): the
position at which the model is about to say whether it noticed anything. Features have the
profile's nuisance coordinates zeroed -- Gemma-3's loud coordinate would otherwise be most of the
vector -- and are divided by s_L, then standardised on the training set. A cell that steers only
after the read layer cannot touch the state the probe reads, so it is refused (examples) or
missing (readout). The read layer has no config key yet (one is asked for in the report) and is
an explicit argument; so is the penalty `l2`.

**Across vectors, held out by family.** The probe is trained on many cells of many vectors, so
what it finds is a signal common to injections rather than one vector's direction. A family is a
vector class by default (concept, evaluation_awareness, sycophancy, ...). Each family is
evaluated by a probe trained on the other families only, and on prompts that probe never saw:
the prompts are split in two halves by a seeded shuffle, the probe trains on one half and is
scored (AUC with DeLong's interval) on the held-out family's steered states against the
unsteered states of the other half. The unsteered state of a prompt is the same for every cell,
so evaluating on prompts seen in training would score memorised negatives.

**The matched random-direction control** (plan §3.4). Examples of family "control" -- norm-matched
Gaussian vectors at the same doses, `m5.vectors.gaussian_recipe` -- are never trained on. They
are scored by the half-prompt probe exactly as a held-out family is: if random directions are
decoded as well as the real vectors, the probe is reading "something perturbed this", not the
vectors. read_q1 reports the same comparison per cell, by steering the cell's Gaussian control
`<vector_id>.gauss0` at the cell's dose when the store holds it. The whitened-norm control of the
plan needs the residual covariance, which nothing measures yet, and is not read.

**The fit.** L2-penalised logistic regression with the two classes weighted equally, solved by
Newton's method with conjugate gradients to convergence. The penalised loss is strictly convex,
so the converged solution does not depend on the solver's settings; failing to converge raises.

Probe weights are derived from activations: they stay in memory, and only their evaluation is
written (CLAUDE.md hard rule 3).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m5 import prompts
from m5.data import detection
from m5.intervention import Cell
from m5.readouts._common import (GENERATED_ONLY, NULL_CELL_ID, check_args, emit, mean_se, peek,
                                 put, resolve_mode, trial_numbers)
from m5.results import Readout
from m5.stats import auc
from m5.steer import intervention_for
from m5.vectors import gaussian_recipe

Q1_MODES = ("live", "past_only", "all", "generated")
CONTROL_FAMILY = "control"
UNSTEERED_FAMILY = "unsteered"
_PROBE_KEY = ("q1-probe",)

# Solver settings. They decide when Newton's method has converged, not what it converges to.
_NEWTON_MAX = 100
_NEWTON_TOL = 1e-9
_CG_TOL = 1e-12


@dataclass(frozen=True, eq=False)
class Q1Example:
    """One answer-position state: its features, steered (1) or not (0), and where it came from."""

    features: np.ndarray
    label: int
    family: str
    vector_id: str | None
    unit_id: str
    cell_id: str
    read_layer: int


@dataclass(frozen=True, eq=False)
class Q1Probe:
    read_layer: int
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    l2: float
    families: tuple[str, ...]
    n_pos: int
    n_neg: int
    newton_steps: int
    lofo: dict[str, Any] = field(default_factory=dict)
    control: dict[str, Any] = field(default_factory=dict)

    def score(self, features: np.ndarray) -> np.ndarray:
        """P(steered) for each row of features."""
        z = ((np.atleast_2d(features) - self.mean) / self.scale) @ self.weights + self.bias
        return _expit(z)

    def summary(self) -> dict[str, Any]:
        """Numbers and ids only (never the weights)."""
        aucs = [v["auc"] for v in self.lofo.values() if "auc" in v]
        return {"read_layer": self.read_layer, "l2": self.l2, "families": list(self.families),
                "n_pos": self.n_pos, "n_neg": self.n_neg, "newton_steps": self.newton_steps,
                "lofo": self.lofo, "lofo_mean_auc": float(np.mean(aucs)) if aucs else None,
                "control": self.control}


def _expit(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# =====================================================================================
# The fit
# =====================================================================================

def _cg(hv: Any, g: np.ndarray, max_iter: int) -> np.ndarray:
    """Solve H x = g by conjugate gradients, H positive definite, given x -> H x."""
    x = np.zeros_like(g)
    r = g.copy()
    p = r.copy()
    rs = float(r @ r)
    stop = _CG_TOL * max(1.0, math.sqrt(rs))
    for _ in range(max_iter):
        if math.sqrt(rs) <= stop:
            break
        hp = hv(p)
        alpha = rs / float(p @ hp)
        x += alpha * p
        r -= alpha * hp
        rs_new = float(r @ r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def _fit_logistic(Z: np.ndarray, y: np.ndarray, c: np.ndarray, l2: float) -> tuple[np.ndarray, float, int]:
    """Minimise sum_i c_i log(1 + exp(-(2y_i - 1) z_i)) + l2/2 ||w||^2, z = Z w + b (b unpenalised)."""
    n, d = Z.shape
    theta = np.zeros(d + 1)
    sign = 2.0 * y - 1.0

    def split(t: np.ndarray) -> tuple[np.ndarray, float]:
        return t[:d], float(t[d])

    def loss(t: np.ndarray) -> float:
        w, b = split(t)
        return float(np.sum(c * np.logaddexp(0.0, -sign * (Z @ w + b))) + 0.5 * l2 * (w @ w))

    for step in range(_NEWTON_MAX):
        w, b = split(theta)
        p = _expit(Z @ w + b)
        r = c * (p - y)
        g = np.concatenate([Z.T @ r + l2 * w, [r.sum()]])
        if math.sqrt(float(g @ g)) <= _NEWTON_TOL:
            return w, b, step
        s = c * p * (1.0 - p)

        def hv(v: np.ndarray) -> np.ndarray:
            vw, vb = v[:d], v[d]
            u = s * (Z @ vw + vb)
            return np.concatenate([Z.T @ u + l2 * vw, [u.sum()]])

        delta = _cg(hv, -g, max_iter=2 * (d + 1))
        f0, slope, t = loss(theta), float(g @ delta), 1.0
        while loss(theta + t * delta) > f0 + 1e-4 * t * slope and t > 1e-12:
            t *= 0.5
        theta = theta + t * delta
    raise RuntimeError(f"q1: the probe's Newton iterations did not converge in {_NEWTON_MAX} steps")


def _fit(pos: Sequence[np.ndarray], neg: Sequence[np.ndarray], l2: float) -> tuple[Any, ...]:
    X = np.vstack([*pos, *neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    c = np.concatenate([np.full(len(pos), 0.5 / len(pos)), np.full(len(neg), 0.5 / len(neg))])
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[~(scale > 0)] = 1.0
    w, b, steps = _fit_logistic((X - mean) / scale, y, c, l2)
    return w, b, mean, scale, steps


def _scorer(fit: tuple[Any, ...]) -> Any:
    w, b, mean, scale, _ = fit
    return lambda X: _expit(((np.atleast_2d(X) - mean) / scale) @ w + b)


# =====================================================================================
# Answer-position states
# =====================================================================================

def _specs(ctx: Any, n: int | None, variant: str) -> tuple[list[str], list[Any]]:
    trials = trial_numbers(ctx, "q1", n)
    carried = detection.variant(variant).framings
    framings = [f for f in ctx.cfg["SELF_REPORT_FRAMINGS"] if f in carried]
    if not framings:
        raise ValueError(f"q1: variant {variant!r} carries none of the framings {ctx.cfg['SELF_REPORT_FRAMINGS']}")
    units, specs = [], []
    for f in framings:
        for t in trials:
            units.append(f"{variant}:{f}:{t}")
            specs.append(prompts.self_report(ctx.backend, t, framing=f, variant=variant))
    return units, specs


def _check_layer(ctx: Any, layer: int) -> int:
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) \
            or not 0 <= int(layer) < ctx.backend.info.n_layers:
        raise ValueError(f"q1: read_layer must be a layer of the model (0..{ctx.backend.info.n_layers - 1}), "
                         f"got {layer!r}")
    return int(layer)


def _transform(ctx: Any, states: Sequence[np.ndarray], layer: int) -> np.ndarray:
    X = np.stack([np.asarray(h, dtype=np.float64)[0, 0, :] for h in states])
    X[:, list(ctx.profile.nuisance(layer))] = 0.0
    return X / ctx.profile.s(layer)


def _features(ctx: Any, texts: Sequence[str], ivs: Sequence[Any], layer: int) -> np.ndarray:
    ctx.charge(forward=len(texts))
    return _transform(ctx, ctx.backend.hidden(list(texts), list(ivs), [layer], [[-1]] * len(texts)), layer)


def _null_features(ctx: Any, texts: Sequence[str], layer: int) -> np.ndarray:
    todo = [t for t in dict.fromkeys(texts) if peek(ctx, ("q1-null", t, layer)) is None]
    if todo:
        X = _features(ctx, todo, [None] * len(todo), layer)
        for t, x in zip(todo, X):
            put(ctx, ("q1-null", t, layer), x)
    return np.stack([peek(ctx, ("q1-null", t, layer)) for t in texts])


def q1_examples(ctx: Any, cell: Cell | None, *, read_layer: int, n: int | None = None,
                variant: str = "original", mode: str | None = None,
                family: str | None = None) -> list[Q1Example]:
    """The unsteered states of the q1 prompts (label 0) and, for a cell, its steered states on the
    same prompts (label 1, family = the vector's class unless given). n trial numbers per framing."""
    layer = _check_layer(ctx, read_layer)
    mode = resolve_mode("q1", cell, mode, kind="marker", allowed=Q1_MODES)
    if mode == "generated":
        raise ValueError(f"q1: {GENERATED_ONLY}")
    units, specs = _specs(ctx, n, variant)
    texts = [s.text for s in specs]
    X_u = _null_features(ctx, texts, layer)
    out = [Q1Example(x, 0, UNSTEERED_FAMILY, None, u, NULL_CELL_ID, layer) for u, x in zip(units, X_u)]
    if cell is None:
        return out
    if min(cell.layers) > layer:
        raise ValueError(f"q1: {cell.cell_id()} steers from layer {min(cell.layers)}, after the read "
                         f"layer {layer}; it cannot change the state the probe reads")
    if family is None:
        if cell.vector_id not in ctx.vector_classes:
            raise ValueError(f"q1: vector {cell.vector_id!r} has no class on the context; pass family=")
        family = ctx.vector_classes[cell.vector_id]
    X_s = _features(ctx, texts, [intervention_for(ctx, cell, s, mode) for s in specs], layer)
    out += [Q1Example(x, 1, family, cell.vector_id, u, cell.cell_id(), layer) for u, x in zip(units, X_s)]
    return out


# =====================================================================================
# Training and evaluation
# =====================================================================================

def _auc(scores_pos: Sequence[float], scores_neg: Sequence[float], conf: float) -> dict[str, Any]:
    if len(scores_pos) < 2 or len(scores_neg) < 2:
        return {"missing": f"needs two steered and two unsteered held-out states, has "
                           f"{len(scores_pos)} and {len(scores_neg)}"}
    res = auc(np.asarray(scores_pos), np.asarray(scores_neg), conf)
    return {k: res[k] for k in ("auc", "lo", "hi", "se", "n_pos", "n_neg")} | {
        "mean_score": float(np.mean(scores_pos)), "null_mean_score": float(np.mean(scores_neg))}


def train_q1(ctx: Any, examples: Sequence[Q1Example], *, l2: float) -> Q1Probe:
    """Fit the cross-vector probe on every non-control steered example and every distinct
    unsteered prompt, evaluate it leave-one-family-out on held-out prompts, score the matched
    random-direction controls, keep it on the context for read_q1 and write its evaluation (no
    weights) to the `setup` stream."""
    if not examples:
        raise ValueError("q1: no examples")
    if isinstance(l2, bool) or not isinstance(l2, (int, float)) or not (math.isfinite(l2) and l2 > 0):
        raise ValueError(f"q1: l2 must be a positive number, got {l2!r}")
    layers = {e.read_layer for e in examples}
    if len(layers) != 1:
        raise ValueError(f"q1: examples read at several layers {sorted(layers)}; one probe reads one layer")
    shapes = {np.shape(e.features) for e in examples}
    if len(shapes) != 1:
        raise ValueError(f"q1: examples of several shapes {sorted(shapes)}")
    neg: dict[str, np.ndarray] = {}
    for e in examples:
        if e.label not in (0, 1):
            raise ValueError(f"q1: label must be 0 or 1, got {e.label!r}")
        if e.label == 0:
            seen = neg.setdefault(e.unit_id, np.asarray(e.features, dtype=np.float64))
            if not np.allclose(seen, e.features, rtol=1e-5, atol=1e-6):
                raise ValueError(f"q1: unsteered prompt {e.unit_id!r} has two different states; "
                                 "examples from different models or variants were mixed")
    pos = [e for e in examples if e.label == 1 and e.family != CONTROL_FAMILY]
    ctrl = [e for e in examples if e.label == 1 and e.family == CONTROL_FAMILY]
    if not pos:
        raise ValueError("q1: needs steered examples of at least one family other than 'control' "
                         "(controls are evaluated, never trained on)")
    if len(neg) < 4:
        raise ValueError(f"q1: needs unsteered states of at least four prompts (two per evaluation "
                         f"half), has {len(neg)}")
    l2 = float(l2)
    layer = layers.pop()
    conf = ctx.cfg["CONFIDENCE"]
    families = tuple(sorted({e.family for e in pos}))
    full = _fit([e.features for e in pos], list(neg.values()), l2)

    units = sorted(neg)
    random.Random(ctx.seed("q1-split", layer, len(units))).shuffle(units)
    train_units, test_units = set(units[:len(units) // 2]), set(units[len(units) // 2:])
    neg_train = [neg[u] for u in sorted(train_units)]
    neg_test = [neg[u] for u in sorted(test_units)]

    lofo: dict[str, Any] = {}
    for fam in families:
        tr = [e.features for e in pos if e.family != fam and e.unit_id in train_units]
        te = [e.features for e in pos if e.family == fam and e.unit_id in test_units]
        if len(families) < 2:
            lofo[fam] = {"missing": "leave-one-family-out needs steered examples of two families"}
        elif not tr:
            lofo[fam] = {"missing": "no other family has steered states on the training prompts"}
        else:
            score = _scorer(_fit(tr, neg_train, l2))
            lofo[fam] = _auc(score(np.vstack(te)).tolist() if te else [],
                             score(np.vstack(neg_test)).tolist(), conf)

    control: dict[str, Any]
    if not ctrl:
        control = {"missing": "no matched random-direction examples (family 'control') were given"}
    else:
        tr = [e.features for e in pos if e.unit_id in train_units]
        te = [e.features for e in ctrl if e.unit_id in test_units]
        if not tr:
            control = {"missing": "no steered states on the training prompts"}
        else:
            score = _scorer(_fit(tr, neg_train, l2))
            control = _auc(score(np.vstack(te)).tolist() if te else [],
                           score(np.vstack(neg_test)).tolist(), conf)

    w, b, mean, scale, steps = full
    probe = Q1Probe(layer, w, float(b), mean, scale, l2, families, len(pos), len(neg), steps,
                    lofo, control)
    put(ctx, _PROBE_KEY, probe)
    ctx.write("setup", {"quantity": "q1_probe", **probe.summary(),
                        "n_control": len(ctrl), "train_units": len(train_units),
                        "test_units": len(test_units)})
    return probe


def trained_probe(ctx: Any) -> Q1Probe | None:
    """The probe train_q1 last kept on this context, or None."""
    return peek(ctx, _PROBE_KEY)


# =====================================================================================
# The readout
# =====================================================================================

def read_q1(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str | None = None, pass_: str | None = None, probe: Q1Probe | None = None,
            control_vector_id: str | None = None) -> Readout:
    """q1 (mean probe score P(steered) at the answer position over the q1 prompts; record-only).
    The unsteered score is `null`; the score under the cell's matched Gaussian control at the
    same dose is `control`."""
    code = "q1"
    check_args(code, variant=variant, variant_aware=True, pass_=pass_)
    mode = resolve_mode(code, cell, mode, kind="marker", allowed=Q1_MODES)

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    if mode == "generated":
        return out(missing=GENERATED_ONLY)
    probe = probe if probe is not None else trained_probe(ctx)
    if probe is None:
        return out(missing="no q1 probe has been trained on this context "
                           "(m5.readouts.probe.train_q1)")
    layer = probe.read_layer
    units, specs = _specs(ctx, n, variant)
    texts = [s.text for s in specs]
    s_u = probe.score(_null_features(ctx, texts, layer))
    null, null_se = mean_se(s_u)
    base = {"null": null, "null_se": null_se, "probe": {"read_layer": layer, "l2": probe.l2,
                                                        "lofo_mean_auc": probe.summary()["lofo_mean_auc"]}}
    if cell is None:
        return out(value=null, n=len(units), extra={**base, "se": null_se})
    if min(cell.layers) > layer:
        return out(missing=f"the cell steers from layer {min(cell.layers)}, after the probe's read "
                           f"layer {layer}; it cannot change the state the probe reads")
    s_s = probe.score(_features(ctx, texts, [intervention_for(ctx, cell, s, mode) for s in specs], layer))
    value, se = mean_se(s_s)

    cid = control_vector_id
    if cid is None and ctx.store is not None and ctx.store.has(gaussian_recipe(cell.vector_id, 0).vector_id):
        cid = gaussian_recipe(cell.vector_id, 0).vector_id
    if cid is None:
        control: dict[str, Any] = {"missing": f"no matched Gaussian control stored "
                                              f"({gaussian_recipe(cell.vector_id, 0).vector_id}); "
                                              "pass control_vector_id="}
    else:
        ccell = Cell(cid, cell.schedule, cell.layers, cell.rho, cell.op, cell.positions, cell.tags)
        s_c = probe.score(_features(ctx, texts, [intervention_for(ctx, ccell, s, mode) for s in specs], layer))
        c_mean, c_se = mean_se(s_c)
        control = {"vector_id": cid, "score": c_mean, "se": c_se, "excess": value - c_mean}
    per = [{"unit": u, "score": float(a), "null": float(b)} for u, a, b in zip(units, s_s, s_u)]
    return out(value=value, n=len(units), extra={**base, "se": se, "lift": value - null,
                                                 "control": control, "per_prompt": per})
