"""m5.stats.summaries - the continuous summaries that replace the 30% box.

Plan §6.3 (Mentor 2: the box was arbitrary). From posterior draws of each layer's three curves --
detection (`m5.stats.psychometric`), effect (`m5.stats.effect`) and the integrity threshold -- this
module computes:

    R*(q)     the largest effect reachable at any layer while noticing stays at most q above its
              floor and integrity holds:  R*(q) = max_l R_l(q),  R_l(q) = E_l(min{u_D,l(q), u_S,l})
    frontier  R*(q) over the q values FRONTIER_Q
    W*        u_D(0.5) - u_E(1/2), the log-dose gap between detection reaching floor + 0.5 and the
              effect reaching half its plateau: unit-free, unchanged by a judge recalibration that
              shifts both, and positive exactly when detection follows influence (O20)
    P_win     the posterior probability that at least one cell of a grid meets the old box
    A_win     the size of the conservative excursion set (Azzimonti et al. 2021): the largest
              Vorob'ev level set {cells with P(meets) >= rho} that lies wholly inside the box
              with joint posterior probability >= 0.95

**Input format: `LayerDraws`**, one per layer, holding n aligned joint draws (entry j of every
array belongs to draw j):

    mu, s, floor, lapse                 the detection curve P(yes | u) = floor + (1 - floor - lapse)
                                        F((u - mu)/s)  (`LayerPosterior.draws`, `LayerChain.joint_draws`)
    eff_bottom, eff_top, eff_u50, eff_s the effect curve E(u) = bottom + (top - bottom) F((u - u50)/s)
                                        (`bootstrap_hill` draws)
    u_s                                 the integrity threshold: the log-dose at which integrity
                                        fails (+inf = held over the whole range searched)
    u_max                               the highest log-dose measured at this layer (a scalar)

Draws from independent sources (detection posterior, effect bootstrap, integrity) are combined by
`combine_draws`, which resamples each to a common n. Statements about all layers at once (R*, the
argmax, P_win) need draw j to be a joint draw across layers: independent per-layer posteriors
satisfy that trivially; chain-coupled layers should use `LayerChain.joint_draws`.

**Nothing is extrapolated.** The operating dose of a layer is capped at its u_max: an effect at a
dose never measured is not reachable on the evidence. When that cap is what binds, the summary
carries the flag `censored` -- the window may extend beyond what was searched.

**Flags** (each the posterior probability, at the best layer of each draw, and raised at >= 0.5):

    S-limited   integrity fails before detection rises: u_S < u_D(q), within the range searched
    floor       detection never rises (by q) in the healthy range: u_D(q) >= min(u_S, u_max)
                (so S-limited implies floor; floor without S-limited means censored)
    no-effect   the effect never rises in the healthy range: E(min(u_S, u_max)) - E(-inf) < delta
    censored    the top of the range searched binds, not detection and not integrity
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from m5.stats._checks import as_float_array, check_conf, check_count, check_prob, check_real
from m5.stats.effect import expit
from m5.stats.psychometric import detection_curve, detection_threshold

_DETECTION_FIELDS = ("mu", "s", "floor", "lapse")
_EFFECT_FIELDS = ("eff_bottom", "eff_top", "eff_u50", "eff_s")


@dataclass(frozen=True, eq=False)
class LayerDraws:
    """n aligned joint posterior draws of one layer's detection, effect and integrity curves."""

    layer: int
    mu: np.ndarray
    s: np.ndarray
    floor: np.ndarray
    lapse: np.ndarray
    eff_bottom: np.ndarray
    eff_top: np.ndarray
    eff_u50: np.ndarray
    eff_s: np.ndarray
    u_s: np.ndarray
    u_max: float

    def __post_init__(self) -> None:
        n = None
        for f in fields(self):
            if f.name in ("layer", "u_max"):
                continue
            arr = np.asarray(getattr(self, f.name), dtype=float).ravel()
            if arr.size == 0:
                raise ValueError(f"layer {self.layer}: {f.name} has no draws")
            if n is None:
                n = arr.size
            elif arr.size != n:
                raise ValueError(f"layer {self.layer}: {f.name} has {arr.size} draws, expected {n} "
                                 "(draws must be aligned)")
            if f.name == "u_s":
                if np.any(np.isnan(arr)) or np.any(np.isneginf(arr)):
                    raise ValueError(f"layer {self.layer}: u_s must be a number or +inf")
            elif not np.all(np.isfinite(arr)):
                raise ValueError(f"layer {self.layer}: {f.name} contains a non-finite draw")
            object.__setattr__(self, f.name, arr)
        object.__setattr__(self, "u_max", check_real(self.u_max, "u_max"))
        if np.any(self.s <= 0) or np.any(self.eff_s <= 0):
            raise ValueError(f"layer {self.layer}: scales s and eff_s must be positive")
        if np.any(self.floor < 0) or np.any(self.lapse < 0) or np.any(self.floor + self.lapse >= 1):
            raise ValueError(f"layer {self.layer}: need floor, lapse >= 0 and floor + lapse < 1")
        if np.any(self.eff_top < self.eff_bottom):
            raise ValueError(f"layer {self.layer}: the effect curve must rise (eff_top >= eff_bottom)")

    @property
    def n(self) -> int:
        return int(self.mu.size)

    def effect(self, u: np.ndarray | float) -> np.ndarray:
        """E(u) per draw (u broadcasts against the draws; -inf gives the bottom)."""
        uu = np.asarray(u, dtype=float)
        z = np.where(np.isneginf(uu), -np.inf, (uu - self.eff_u50) / self.eff_s)
        return self.eff_bottom + (self.eff_top - self.eff_bottom) * expit(np.where(np.isfinite(z), z, np.sign(z) * 800.0))

    def detection(self, u: np.ndarray | float) -> np.ndarray:
        """P(yes | u) per draw."""
        return detection_curve(u, self.mu, self.s, self.floor, self.lapse)

    def u_d(self, q: float) -> np.ndarray:
        """u_D(q) per draw (+inf where the curve never rises by q)."""
        return detection_threshold(self.mu, self.s, self.floor, self.lapse, q)


def combine_draws(layer: int, detection: Mapping[str, np.ndarray], effect: Mapping[str, np.ndarray],
                  u_s: np.ndarray | float, u_max: float, *, n: int, seed: int) -> LayerDraws:
    """Assemble a `LayerDraws` from independent sources, each brought to n draws.

    `detection` has mu, s, floor, lapse (e.g. `LayerPosterior.draws`); `effect` has bottom, top,
    u50, s (`EffectBootstrap.draws`); `u_s` is an array of integrity-threshold draws or one value.
    Rows within a source stay together (a source's draw is joint). A source that already has
    exactly n draws is used in order -- so detection draws from `LayerChain.joint_draws`, aligned
    across layers, stay aligned -- and any other source is resampled with replacement, which
    pairs it at random with the rest: right for independent posteriors.
    """
    n = check_count(n, "n", minimum=1)
    rng = np.random.default_rng(seed)

    def pick(arrs: Mapping[str, np.ndarray], names: Sequence[str]) -> dict[str, np.ndarray]:
        cols = [np.asarray(arrs[k], dtype=float).ravel() for k in names]
        size = cols[0].size
        if size == 0 or any(c.size != size for c in cols):
            raise ValueError(f"draws for {names} must be non-empty and aligned")
        if size == n:
            return dict(zip(names, cols))
        idx = rng.integers(0, size, size=n)
        return {k: c[idx] for k, c in zip(names, cols)}

    det = pick(detection, _DETECTION_FIELDS)
    eff = pick(effect, ("bottom", "top", "u50", "s"))
    us = np.asarray(u_s, dtype=float).ravel()
    if us.size == 1:
        us = np.full(n, us[0])
    elif us.size != n:
        us = us[rng.integers(0, us.size, size=n)]
    return LayerDraws(layer=layer, mu=det["mu"], s=det["s"], floor=det["floor"], lapse=det["lapse"],
                      eff_bottom=eff["bottom"], eff_top=eff["top"], eff_u50=eff["u50"],
                      eff_s=eff["s"], u_s=us, u_max=u_max)


def _check_layers(layers: Sequence[LayerDraws]) -> int:
    if len(layers) == 0:
        raise ValueError("no layers")
    n = layers[0].n
    if any(d.n != n for d in layers):
        raise ValueError("every layer needs the same number of aligned draws")
    if len({d.layer for d in layers}) != len(layers):
        raise ValueError("duplicate layer")
    return n


def _summ(x: np.ndarray, conf: float) -> dict[str, float]:
    """Median and equal-tailed interval of draws; the inverse-CDF quantile (no interpolation)
    so that +inf draws (a curve that never rises) give +inf bounds, never NaN."""
    alpha = 1.0 - conf
    lo, med, hi = np.quantile(x, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0], method="inverted_cdf")
    return {"median": float(med), "lo": float(lo), "hi": float(hi)}


def _json_float(v: float) -> float | None:
    return v if math.isfinite(v) else None


def layer_terms(d: LayerDraws, q: float, delta: float) -> dict[str, np.ndarray]:
    """Per-draw R_l(q) and the per-draw conditions behind the flags, for one layer."""
    q = check_prob(q, "q", open_interval=True)
    ud = d.u_d(q)
    healthy_top = np.minimum(d.u_s, d.u_max)
    u_op = np.minimum(ud, healthy_top)
    r = d.effect(u_op)
    rise = d.effect(healthy_top) - d.eff_bottom
    return {
        "R": r,
        "u_op": u_op,
        "u_d": ud,
        "s_limited": (d.u_s < ud) & (d.u_s <= d.u_max),
        "floor": ud >= healthy_top,
        "no_effect": rise < delta,
        "censored": (d.u_max < ud) & (d.u_max < d.u_s),
        "detection_binds": ud < healthy_top,
    }


FLAG_NAMES = (("s_limited", "S-limited"), ("floor", "floor"), ("no_effect", "no-effect"),
              ("censored", "censored"))


def r_star(layers: Sequence[LayerDraws], q: float, conf: float, delta: float) -> dict[str, Any]:
    """R*(q) = max over layers of R_l(q), per draw, with its credible interval and flags.

    `delta` is the effect rise below which the effect is said not to rise (pass DELTA). The best
    layer differs between draws; `p_best` gives each layer's posterior probability of being it,
    and the flags are evaluated at each draw's own best layer. `per_layer` holds every layer's
    R_l(q) summary, the input to successive halving (`m5.stats.halving.keep_layers`).
    """
    conf = check_conf(conf)
    delta = check_real(delta, "delta")
    n = _check_layers(layers)
    terms = [layer_terms(d, q, delta) for d in layers]
    R = np.stack([t["R"] for t in terms])
    best = np.argmax(R, axis=0)
    cols = np.arange(n)
    rstar = R[best, cols]
    flag_p = {name: float(np.mean(np.stack([t[key] for t in terms])[best, cols]))
              for key, name in FLAG_NAMES}
    p_best = np.bincount(best, minlength=len(layers)) / n
    per_layer = []
    for d, t in zip(layers, terms):
        row = {"layer": d.layer, **_summ(t["R"], conf),
               "p_detection_binds": float(np.mean(t["detection_binds"]))}
        row.update({f"p_{name}": float(np.mean(t[key])) for key, name in FLAG_NAMES})
        per_layer.append(row)
    return {"q": q, "conf": conf, **_summ(rstar, conf), "mean": float(rstar.mean()),
            "best_layer": layers[int(np.argmax(p_best))].layer,
            "p_best": {d.layer: float(p) for d, p in zip(layers, p_best)},
            "flags": [name for name, p in flag_p.items() if p >= 0.5],
            "flag_probabilities": flag_p, "per_layer": per_layer, "n_draws": n}


def frontier(layers: Sequence[LayerDraws], qs: Sequence[float], conf: float, delta: float) -> list[dict[str, Any]]:
    """R*(q) at every q of FRONTIER_Q: how much more effect each extra point of noticing buys."""
    q_list = [check_prob(q, "q", open_interval=True) for q in qs]
    if any(b <= a for a, b in zip(q_list, q_list[1:])):
        raise ValueError("qs must be strictly increasing")
    return [r_star(layers, q, conf, delta) for q in q_list]


def w_star(layers: Sequence[LayerDraws], conf: float) -> dict[int, dict[str, Any]]:
    """W* = u_D(0.5) - u_E(1/2) per layer: the log-dose by which detection lags the effect.

    Positive: the effect reaches half its plateau before detection reaches floor + 0.5, i.e. there
    is room to steer before the model notices. +inf in a draw where detection never rises by 0.5;
    -inf in a draw whose effect curve is flat (it has no half-plateau dose, so nothing precedes
    detection). `p_positive` is the posterior probability that detection follows influence (O20);
    `p_never` and `p_flat_effect` give the shares of the two infinite cases.
    """
    conf = check_conf(conf)
    _check_layers(layers)
    out: dict[int, dict[str, Any]] = {}
    for d in layers:
        flat = (d.eff_top - d.eff_bottom) <= 1e-12
        ud = d.u_d(0.5)
        w = np.where(flat, -np.inf, ud - np.where(flat, 0.0, d.eff_u50))
        s = _summ(w, conf)
        out[d.layer] = {"median": _json_float(s["median"]), "lo": _json_float(s["lo"]),
                        "hi": _json_float(s["hi"]), "p_positive": float(np.mean(w > 0)),
                        "p_never": float(np.mean(np.isposinf(w))),
                        "p_flat_effect": float(np.mean(flat))}
    return out


def _cell_widths(u: np.ndarray) -> np.ndarray:
    mid = 0.5 * (u[1:] + u[:-1])
    edges = np.concatenate([[u[0] - (mid[0] - u[0])], mid, [u[-1] + (u[-1] - mid[-1])]])
    return np.diff(edges)


def p_win(layers: Sequence[LayerDraws], u_grid: Sequence[float] | np.ndarray, *, effect_min: float,
          detection_max: float, relative_to_floor: bool, level: float) -> dict[str, Any]:
    """P_win and A_win: the old box answer, with its uncertainty, for comparison with M3.

    A cell (layer, u) meets the box in a draw when E(u) >= effect_min, the detection rate (minus
    the floor if `relative_to_floor`) is <= detection_max, and integrity holds (u < u_S). Cells
    above a layer's u_max are not part of the grid (never extrapolated).

    P_win is the share of draws in which at least one cell meets the box. A_win follows Azzimonti,
    Ginsbourger, Chevalier, Bect & Richet (2021, Technometrics 63:13-26, "Adaptive design of
    experiments for conservative estimation of excursion sets"): among the Vorob'ev quantiles
    Q_rho = {cells with P(meets) >= rho}, the largest one whose cells *all* meet the box with joint
    posterior probability >= `level` (0.95). The joint requirement is what makes it conservative:
    the set of cells each meeting it with probability 0.95 is larger (`marginal_count`), and the
    chance that all of them meet it at once can be much lower. Area is in layers x log-dose.
    """
    level = check_prob(level, "level", open_interval=True)
    effect_min = check_real(effect_min, "effect_min")
    detection_max = check_real(detection_max, "detection_max")
    ug = as_float_array(u_grid, "u_grid", min_len=2)
    if np.any(np.diff(ug) <= 0):
        raise ValueError("u_grid must be strictly increasing")
    n = _check_layers(layers)
    widths = _cell_widths(ug)
    meets_rows = []
    cell_ids = []
    cell_area = []
    for d in layers:
        for j, u in enumerate(ug.tolist()):
            if u > d.u_max:
                continue
            det = d.detection(u)
            if relative_to_floor:
                det = det - d.floor
            meets_rows.append((d.effect(u) >= effect_min) & (det <= detection_max) & (u < d.u_s))
            cell_ids.append((d.layer, u))
            cell_area.append(widths[j])
    if not meets_rows:
        raise ValueError("no grid cell lies within any layer's searched range")
    meets = np.stack(meets_rows)                   # (cells, draws)
    p_cell = meets.mean(axis=1)
    p_any = float(np.mean(meets.any(axis=0)))
    order = np.argsort(-p_cell, kind="mergesort")
    chosen: list[int] = []
    running = np.ones(n, dtype=bool)
    i = 0
    while i < order.size and p_cell[order[i]] > 0:
        j = i
        while j < order.size and p_cell[order[j]] == p_cell[order[i]]:  # a level set takes ties whole
            j += 1
        group = order[i:j]
        trial = running & meets[group].all(axis=0)
        if trial.mean() < level:
            break
        running = trial
        chosen.extend(group.tolist())
        i = j
    marginal = [int(c) for c in np.flatnonzero(p_cell >= level)]
    area = np.asarray(cell_area)
    return {
        "P_win": p_any,
        "A_win_count": len(chosen),
        "A_win_area": float(area[chosen].sum()) if chosen else 0.0,
        "conservative_cells": [cell_ids[c] for c in chosen],
        "joint_probability": float(running.mean()) if chosen else 1.0,
        "marginal_count": len(marginal),
        "marginal_area": float(area[marginal].sum()) if marginal else 0.0,
        "n_cells": len(cell_ids),
        "level": level,
        "n_draws": n,
    }
