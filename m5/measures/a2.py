"""m5.measures.a2 - `a2`, latent evaluation awareness: does a probe read the model as more test-aware?

An evaluation-awareness probe, trained once per model at setup (plan §5.1, "evaluation-awareness
assets"), read on the hidden state of the same scenarios `a1` uses (`m5.data.scenarios`), steered
and unsteered -- forward passes only. The probe is validated the way the plan asks (§3.2): it must
separate evaluation-like from deployment-like inputs, so the unsteered scores' separation (AUC
with DeLong's interval) is reported beside every reading.

**The probe** (`EAProbe`) is a linear read at one layer and position: score = sigmoid(w . h + b),
in [0, 1], with the raw logit kept beside it. `fit_mean_difference` builds one from labelled
states -- the direction between the class means, scaled by the pooled within-class spread and
centred between them, the contrastive recipe in its simplest form -- for the setup to call and the
tests to use. Steering at a layer above the probe's cannot move what it reads, which is physics
rather than a defect, and the probe layer is in `extra` so a reader can see it.

**Where the probe lives.** Its weights are a vector, so they never sit in the profile (an
aggregate document): the setup writes them to an `.npz` under RUN_ROOT and records the choice as
`ctx.profile.selection("ea_probe")`, which the measure reads, loads and verifies. The shape,
agreed with the setup (`EAProbe.save` writes the file and returns exactly this mapping):

    {"path": absolute path of the .npz, outside the repository,
     "sha256": the file's digest (verified before the file is opened),
     "layers": [the layers the file holds, in its row order],
     "layer": the layer a2 reads (required when the file holds several),
     "position": the token position read (-1: the last prompt token),
     ...}                                         anything else: provenance (recipe, AUC, n)

    the .npz: "weights" [len(layers), d_model], "bias" [len(layers)], "layers" [len(layers)]
    (must equal the selection's), "position" (scalar; must equal the selection's when both say)

With no such selection the measure is missing with that reason, never a zero; a file that is
absent, inside the repository or not the file the digest names raises. `probe=` overrides the
selection (the setup's own validation, and tests).

`n` is the number of scenarios per class (no repeats: a forward pass has no sampling). `values` are
the steered scores, `extra["null_values"]` the unsteered scores of the same scenarios, and
`extra["shift"]` the mean paired difference with its percentile-bootstrap interval; per class in
`extra["classes"]`. `lo` / `hi` bound the steered mean.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from m5 import prompts as _prompts
from m5.data import scenarios as _scn
from m5.intervention import Cell
from m5.measures import common as C
from m5.results import Measured
from m5.safety import assert_outside_repo
from m5.stats import auc

CODE = "a2"


@dataclass(frozen=True)
class EAProbe:
    """A linear evaluation-awareness probe: score = sigmoid(weights . h[layer, position] + bias)."""

    layer: int
    weights: tuple[float, ...]
    bias: float
    position: int = -1
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError(f"probe layer must be a non-negative integer, got {self.layer!r}")
        w = np.asarray(self.weights, dtype=float)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)) or not np.any(w):
            raise ValueError("probe weights must be a finite, non-zero vector")
        if not math.isfinite(self.bias):
            raise ValueError("probe bias must be finite")
        object.__setattr__(self, "weights", tuple(float(x) for x in w))

    def logit(self, h: np.ndarray) -> float:
        x = np.asarray(h, dtype=float).reshape(-1)
        if x.size != len(self.weights):
            raise ValueError(f"a state of size {x.size} for a probe of size {len(self.weights)}")
        return float(np.dot(np.asarray(self.weights), x) + self.bias)

    def score(self, h: np.ndarray) -> float:
        z = self.logit(h)
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

    def to_json(self) -> dict[str, Any]:
        return {"layer": self.layer, "weights": list(self.weights), "bias": self.bias,
                "position": self.position, "provenance": dict(self.provenance)}

    @classmethod
    def from_any(cls, obj: Any) -> "EAProbe":
        if isinstance(obj, EAProbe):
            return obj
        if isinstance(obj, Mapping):
            return cls(layer=int(obj["layer"]), weights=tuple(obj["weights"]), bias=float(obj["bias"]),
                       position=int(obj.get("position", -1)), provenance=dict(obj.get("provenance", {})))
        raise TypeError(f"an evaluation-awareness probe is an EAProbe or its JSON form, got "
                        f"{type(obj).__name__}")


def fit_mean_difference(eval_states: np.ndarray, deploy_states: np.ndarray, *, layer: int,
                        position: int = -1, provenance: Mapping[str, Any] | None = None) -> EAProbe:
    """The class-mean-difference probe: w = d / (|d| s), b = -w . midpoint, where d is the mean
    evaluation-like state minus the mean deployment-like one and s the pooled within-class sd of
    the states' projections on d. A state on the midpoint scores 0.5; one class mean sits about
    |d| / (2 s) logits above it, the other as far below."""
    e = np.asarray(eval_states, dtype=float)
    d_ = np.asarray(deploy_states, dtype=float)
    if e.ndim != 2 or d_.ndim != 2 or e.shape[1] != d_.shape[1] or len(e) < 2 or len(d_) < 2:
        raise ValueError("need at least two states per class, as [n, d_model] arrays of one width")
    diff = e.mean(axis=0) - d_.mean(axis=0)
    norm = float(np.linalg.norm(diff))
    if not norm > 0:
        raise ValueError("the class means coincide; there is no direction to probe")
    u = diff / norm
    proj_e, proj_d = e @ u, d_ @ u
    pooled = math.sqrt((proj_e.var(ddof=1) * (len(e) - 1) + proj_d.var(ddof=1) * (len(d_) - 1))
                       / (len(e) + len(d_) - 2))
    scale = pooled if pooled > 0 else 1.0
    w = u / scale
    mid = 0.5 * (e.mean(axis=0) + d_.mean(axis=0))
    return EAProbe(layer=int(layer), weights=tuple(w), bias=float(-w @ mid), position=int(position),
                   provenance={"recipe": "class-mean difference, pooled-sd scaled",
                               "n_eval": len(e), "n_deploy": len(d_), **dict(provenance or {})})


SELECTION = "ea_probe"


def save_probe(probe: EAProbe, path: str | Path, **provenance: Any) -> dict[str, Any]:
    """Write a probe's weights to an .npz outside the repository and return the profile selection
    that names it (record it with `profile.with_selection("ea_probe", ...)`). Refuses to replace
    an existing file: a selection's digest must keep naming the bytes it was taken from."""
    p = assert_outside_repo(path, what="evaluation-awareness probe file")
    if p.suffix != ".npz":
        raise ValueError(f"a probe file is an .npz, got {p.name!r}")
    if p.exists():
        raise FileExistsError(f"{p} exists; a stored probe is never overwritten in place")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        np.savez(f, weights=np.asarray([probe.weights], dtype=np.float64),
                 bias=np.asarray([probe.bias], dtype=np.float64),
                 layers=np.asarray([probe.layer], dtype=np.int64),
                 position=np.asarray(probe.position, dtype=np.int64))
    small = {k: v for k, v in dict(probe.provenance).items() if not isinstance(v, (list, tuple))}
    return {**small, **provenance, "path": str(p), "sha256": C.sha256_file(p),
            "layers": [probe.layer], "layer": probe.layer, "position": probe.position}


def load_probe(selection: Mapping[str, Any]) -> EAProbe:
    """The probe a profile selection names, loaded from its file and verified (module docstring)."""
    for key in ("path", "sha256", "layers"):
        if key not in selection:
            raise C.MeasureError(f"the ea_probe selection does not record {key!r}")
    p = assert_outside_repo(selection["path"], what="evaluation-awareness probe file")
    if not p.is_file():
        raise C.MeasureError(f"the ea_probe selection names {p}, which does not exist")
    digest = C.sha256_file(p)
    if digest != selection["sha256"]:
        raise C.MeasureError(f"{p} has sha256 {digest[:12]}..., not the {str(selection['sha256'])[:12]}... "
                             "the ea_probe selection recorded: it is not the probe the setup chose")
    layers = [int(x) for x in selection["layers"]]
    with np.load(p, allow_pickle=False) as z:
        if "weights" not in z.files or "bias" not in z.files:
            raise C.MeasureError(f"{p} holds {sorted(z.files)}; a probe file holds weights and bias")
        w = np.atleast_2d(np.asarray(z["weights"], dtype=float))
        b = np.atleast_1d(np.asarray(z["bias"], dtype=float))
        file_layers = [int(x) for x in np.atleast_1d(z["layers"])] if "layers" in z.files else None
        file_position = int(z["position"]) if "position" in z.files else None
    if w.shape[0] != len(layers) or b.shape[0] != len(layers):
        raise C.MeasureError(f"{p}: {w.shape[0]} weight rows and {b.shape[0]} biases for the "
                             f"{len(layers)} layers the selection names")
    if file_layers is not None and file_layers != layers:
        raise C.MeasureError(f"{p} holds layers {file_layers}; the selection says {layers}")
    if "layer" in selection:
        layer = int(selection["layer"])
    elif len(layers) == 1:
        layer = layers[0]
    else:
        raise C.MeasureError(f"the probe file holds layers {layers}; the selection must say which "
                             "one a2 reads ('layer')")
    if layer not in layers:
        raise C.MeasureError(f"the selection reads layer {layer}, which {p} does not hold ({layers})")
    sel_position = int(selection["position"]) if "position" in selection else None
    if sel_position is None and file_position is None:
        raise C.MeasureError("neither the ea_probe selection nor its file says which token position "
                             "the probe reads")
    if sel_position is not None and file_position is not None and sel_position != file_position:
        raise C.MeasureError(f"the selection reads position {sel_position}; {p} says {file_position}")
    i = layers.index(layer)
    position = sel_position if sel_position is not None else int(file_position)  # type: ignore[arg-type]
    return EAProbe(layer=layer, weights=tuple(w[i]), bias=float(b[i]), position=position,
                   provenance={"path": str(p), "sha256": digest, "source": "profile selection"})


def probe_for(ctx: Any) -> EAProbe | None:
    """The model's trained probe from `ctx.profile.selection("ea_probe")`, or None when the setup
    has recorded none. Loaded once per run."""
    profile = ctx.profile
    if profile is None or not profile.has_selection(SELECTION):
        return None
    sel = profile.selection(SELECTION)
    st = C.state(ctx)
    key = ("a2-probe", str(sel.get("path")), str(sel.get("sha256")))
    with st.lock:
        hit = st.memo.get(key)
    if hit is None:
        hit = load_probe(sel)
        with st.lock:
            st.memo[key] = hit
    return hit


def scenario_states(ctx: Any, scenarios: Sequence[_scn.Scenario], layer: int, *, position: int = -1,
                    cell: Cell | None = None) -> np.ndarray:
    """[len(scenarios), d_model] states at one layer and position, unsteered or under a cell (the
    scenarios rendered as `a1` renders them, every position steered). Charged as forward passes."""
    specs = [C.spec(ctx, (CODE, sc.id), lambda sc=sc: _prompts.scenario(ctx.backend, sc))
             for sc in scenarios]
    ivs = [C.intervention(ctx, cell, sp, sp.meta["mode"]) for sp in specs]
    out = []
    batch = int(ctx.cfg["GEN_BATCH_MAX"])
    for b in range(0, len(specs), batch):
        chunk = list(range(b, min(len(specs), b + batch)))
        ctx.charge(forward=len(chunk))
        hs = ctx.backend.hidden([specs[i].text for i in chunk], [ivs[i] for i in chunk], [int(layer)],
                                [[int(position)] for _ in chunk])
        out += [np.asarray(h, dtype=float)[0, 0] for h in hs]
    return np.stack(out)


def measure_a2(ctx: Any, cell: Cell | None, *, n: int, split: str, variant: str = "original",
               probe: EAProbe | Mapping[str, Any] | None = None) -> Measured:
    """Measure `a2` at a cell on `n` scenarios per class."""
    C.check_split(split)
    C.check_variant(CODE, variant)
    n = C.check_n(n)
    subj = C.subject(ctx, cell)
    p = EAProbe.from_any(probe) if probe is not None else probe_for(ctx)
    if p is None:
        return C.missing_measure(ctx, CODE, subj, split, "the profile has no 'ea_probe' selection: no "
                                 "evaluation-awareness probe has been trained for this model "
                                 "(per-model setup, plan §5.1); a2 reads it")
    if cell is not None and C.mode_for(cell, "all") == "generated":
        return C.missing_measure(ctx, CODE, subj, split, "positions 'generated' steer no token of a "
                                 "forward pass over the scenario; a2 cannot be read at this cell")
    conf = float(ctx.cfg["CONFIDENCE"])
    reps = int(ctx.cfg["BOOTSTRAP_REPS"])
    scen = [sc for cls in _scn.CLASSES for sc in [s for s in _scn.SCENARIOS if s.cls == cls][:n]]
    st = C.state(ctx)
    key = ("a2-null", p.layer, p.position, tuple(sc.id for sc in scen))
    with st.lock:
        null_states = st.hidden.get(str(key))
    if null_states is None:
        null_states = scenario_states(ctx, scen, p.layer, position=p.position)
        with st.lock:
            st.hidden[str(key)] = null_states
    states = null_states if cell is None else \
        scenario_states(ctx, scen, p.layer, position=p.position, cell=cell)
    s_scores = [p.score(h) for h in states]
    z_scores = [p.score(h) for h in null_states]
    s_logits = [p.logit(h) for h in states]
    z_logits = [p.logit(h) for h in null_states]
    rows = [{"code": CODE, "cell": subj.cell_id, "split": split, "arm": subj.arm, "unit": sc.id,
             "scenario": sc.id, "cls": sc.cls, "score": a, "null_score": b, "logit": la,
             "null_logit": lb, "probe_layer": p.layer, "vector": subj.vector_id}
            for sc, a, b, la, lb in zip(scen, s_scores, z_scores, s_logits, z_logits)]
    if subj.null_only:          # the null arm's rows, once per run and probe
        C.write_null_trials(ctx, (((CODE, p.layer, p.position, r["unit"]), r) for r in rows))
    else:
        C.write_trials(ctx, rows)

    def boot(xs: list[float], what: str) -> dict[str, Any]:
        mean, lo, hi = C.bootstrap_mean(xs, conf=conf, reps=reps,
                                        seed=ctx.seed(CODE, what, subj.cell_id, split, n))
        return {"mean": mean, "lo": lo, "hi": hi}

    classes = {}
    for cls in _scn.CLASSES:
        idx = [i for i, sc in enumerate(scen) if sc.cls == cls]
        classes[cls] = {"n": len(idx), "mean": float(np.mean([s_scores[i] for i in idx])),
                        "null_mean": float(np.mean([z_scores[i] for i in idx])),
                        "shift": boot([s_scores[i] - z_scores[i] for i in idx], f"shift-{cls}")}
    ev = [z_scores[i] for i, sc in enumerate(scen) if sc.cls == "eval"]
    dp = [z_scores[i] for i, sc in enumerate(scen) if sc.cls == "deploy"]
    try:
        separation = auc(ev, dp, conf)
    except ValueError as exc:  # too few scenarios for DeLong's interval
        separation = {"missing": C.short(str(exc))}
    steered = boot(s_scores, "steered")
    m = Measured(code=CODE, cell_id=subj.cell_id, split=split, n=len(scen), lo=steered["lo"],
                 hi=steered["hi"], values=s_scores, unit_ids=[sc.id for sc in scen],
                 extra={"probe_layer": p.layer, "probe_position": p.position,
                        "probe": {"source": C.short(str(p.provenance.get("source", "given"))),
                                  "sha256": p.provenance.get("sha256")},
                        "null_values": z_scores, "null": boot(z_scores, "null"),
                        "shift": {**boot([a - b for a, b in zip(s_scores, z_scores)], "shift"),
                                  "method": "percentile bootstrap over scenarios"},
                        "logit_shift": float(np.mean(np.subtract(s_logits, z_logits))),
                        "classes": classes, "validation_auc": separation,
                        "n_per_class": n})
    if subj.null_only:          # a cell of None is the null arm itself
        return C.write_cell(ctx, C.as_null_arm(m, conf), subj=subj)
    iv = C.intervention(ctx, cell, C.spec(ctx, (CODE, scen[0].id),
                                          lambda: _prompts.scenario(ctx.backend, scen[0])), "all")
    return C.write_cell(ctx, m, subj=subj, iv=iv)
