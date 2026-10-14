"""m5.backend.liveness - prove the hooks steer, on the paths a run uses, before anything trusts them.

Every run calls `check_liveness(backend)` after the model loads and before it spends anything. It
returns a report (log it, store it beside the run) and raises `LivenessError` -- carrying the same
report -- when any check fails. The run must stop there.

Why it exists. The 2026-08-04 run read exactly 0.000 at 30 of 30 cells for an hour because the
upstream hook returned decoder outputs unmodified whenever a start position was set (bug 26). The
check that was then added (R14) attested M2's own `injected` hook, while every measured generation
went through a different one (trap 14), with no padded batch, no decode step and no wrong-layer
control. Here every check runs through the backend's public methods -- the same batched,
left-padded, per-row code that measurement uses -- and the decode checks go through the generation
loop itself (`forced_logprobs(path="decode")` and `generate`).

The checks, each of which a test trips by breaking one thing in the backend:

    zero_dose            a zero dose equals no intervention, bit for bit within tolerance, on the
                         next-token read, the decode path and sampled generation (same seeds)
    dose_moves           at the steered layer the output moves by exactly the operation's delta at
                         the steered positions and not at all before them, and the next-token
                         distribution moves
    wrong_layer          hooked layers L and O read what HF itself reports as hidden_states[L + 1]
                         (an independent witness of the layer index); steering at L and at O give
                         different distributions; steering the later layer leaves the earlier
                         layer's output untouched, steering the earlier one moves the later
    batch_independence   rows of a mixed, left-padded batch (different prompt lengths, vectors,
                         layers, operations, positions, a zero dose and no intervention) equal the
                         same rows run alone: next-token distribution, hidden states, decode path
    decode_path          teacher forcing through the generation loop equals one prefill pass for
                         the from, span and generated policies; under 'generated' the first token
                         is unsteered and the later ones move (generated tokens are steered)
    lens_matches_logits  the logit lens at the last layer reproduces the model's own next-token
                         distribution, steered and not (the softcap and final-norm path)
    generate_path        greedy generate() under steering emits, step by step, the argmax of the
                         steered decode-path distribution: generation applies the steering that
                         the decode checks attest

By default the probe vector is a seeded random direction per layer, scaled to the median residual
norm there, at alpha 1: a dose no live hook can hide. A caller may probe a stored vector instead
(pass `vector_id` and `alpha`); a dose below MIN_DOSE_RATIO of the residual norm is refused,
because a weak probe cannot tell a live hook from a dead one.

No torch here (M5 rule 3): everything goes through the backend and numpy. The checks need the HF
backend's extensions `forced_logprobs`, `reference_hidden`, `vector_values` and
`unregister_vector`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from m5.backend.base import GenRequest, locate
from m5.intervention import Intervention, LayerTerm, Positions

LIVENESS_VECTOR_ID = "__liveness_random__"

# Two user questions of different lengths: a batch of both is left-padded.
PROBE_QUESTIONS = (
    "Tell me a fact related to water.",
    "Name three things you might find in a kitchen, and say what each is for.",
)
# Teacher-forced continuation for the decode checks (its first FORCED_TOKENS tokens).
CONTINUATION = " Water covers most of the Earth, and the oceans hold most of it."
FORCED_TOKENS = 6
GENERATED_TOKENS = 8

# The default probe layer, as a fraction of depth (Macar's reference depth; M2's R14 used it too),
# and the gap to the wrong layer, as a fraction of depth.
LIVENESS_DEPTH = 0.6
WRONG_LAYER_GAP = 0.125

# The weakest probe accepted: ||alpha v|| / median ||h|| at the probed layers.
MIN_DOSE_RATIO = 0.25

# How far apart two computations may be and still count as equal, by model dtype. `exact` is for
# computations that should be identical (zero dose, untouched positions; the slack only absorbs
# non-deterministic kernels); `edit` is relative to max(||alpha v||, ||h||); `batch_tv` is a total
# variation distance between next-token distributions; `batch_hidden` is relative per position;
# `paths` is an absolute log-prob difference between teacher forcing through the decode loop and
# through one prefill pass; `lens` is a total variation distance between the last-layer lens and
# the model's own logits, which run the same operations on the same values (0.0 observed in every
# dtype), so it stays tight. The `min_*` are the least a live probe must move things.
#
# The bfloat16 and float16 bounds come from tiny models on CPU, where the observed values sat 3-15x
# inside them, loosened for the noise a 62-layer model accumulates; every defect the tests inject
# still fails them by a wide margin at the liveness dose. They are to be confirmed against the
# report of the first pod run, which prints every observed value beside its bound.
TOLERANCES: dict[str, dict[str, float]] = {
    "float32": dict(exact=1e-5, edit=1e-4, batch_tv=1e-4, batch_hidden=1e-4, paths=1e-3, lens=1e-5,
                    min_tv=1e-3, min_lp=1e-2, min_rel=1e-3),
    "bfloat16": dict(exact=1e-3, edit=2e-2, batch_tv=5e-2, batch_hidden=0.1, paths=0.5, lens=2e-3,
                     min_tv=1e-3, min_lp=1e-2, min_rel=1e-3),
    "float16": dict(exact=1e-3, edit=5e-3, batch_tv=2e-2, batch_hidden=5e-2, paths=0.25, lens=1e-3,
                    min_tv=1e-3, min_lp=1e-2, min_rel=1e-3),
}

_NEEDS = ("forced_logprobs", "reference_hidden", "vector_values", "unregister_vector",
          "register_vector", "has_vector", "info")


class LivenessError(RuntimeError):
    """The hooks failed a liveness check. `.report` holds every check and its numbers."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        lines = [f"liveness FAILED at L{report['layer']} (wrong layer L{report['other_layer']}, "
                 f"alpha {report['alpha']:g}): {', '.join(report['failed'])}. Do not run anything "
                 "on this backend; every measure would describe a different intervention than the "
                 "one it names (bug 26)."]
        for name in report["failed"]:
            lines.append(f"  {name}: {report['checks'][name]}")
        super().__init__("\n".join(lines))


def _tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Total variation distance between rows of log-probabilities."""
    return 0.5 * np.abs(np.exp(a) - np.exp(b)).sum(axis=-1)


def _max(x: Any) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(arr.max()) if arr.size else 0.0


def check_liveness(backend: Any, *, layer: int | None = None, other_layer: int | None = None,
                   vector_id: str | None = None, alpha: float | None = None, seed: int = 0,
                   tolerances: dict[str, float] | None = None) -> dict[str, Any]:
    """Run every check; return the report, or raise LivenessError carrying it.

    `layer` defaults to LIVENESS_DEPTH of the depth (kept below the last layer, which has no
    independent reference); `other_layer` to WRONG_LAYER_GAP of the depth below it (above it when
    there is no room). `tolerances` overrides entries of TOLERANCES for the model's dtype.
    """
    missing = [name for name in _NEEDS if not hasattr(backend, name)]
    if missing:
        raise TypeError(f"liveness needs the HF backend's {', '.join(missing)}")
    info = backend.info
    n_layers = int(info.n_layers)
    if n_layers < 2:
        raise ValueError("liveness needs at least two layers (one to steer, one to be the wrong one)")
    L = min(int(LIVENESS_DEPTH * n_layers), n_layers - 2) if layer is None else int(layer)
    gap = max(1, int(WRONG_LAYER_GAP * n_layers))
    O = (L - gap if L - gap >= 0 else L + gap) if other_layer is None else int(other_layer)
    if not (0 <= L < n_layers and 0 <= O < n_layers) or L == O:
        raise ValueError(f"liveness needs two distinct layers in [0, {n_layers}), got {L} and {O}")
    if info.dtype not in TOLERANCES:
        raise ValueError(f"no liveness tolerances for dtype {info.dtype!r}")
    tol = dict(TOLERANCES[info.dtype])
    for key, value in (tolerances or {}).items():
        if key not in tol:
            raise KeyError(f"unknown liveness tolerance {key!r}")
        tol[key] = float(value)

    texts = [backend.render([{"role": "user", "content": q}]) for q in PROBE_QUESTIONS]
    starts = [locate(backend, t, q) for t, q in zip(texts, PROBE_QUESTIONS)]
    lengths = [len(backend.tokenize(t)) for t in texts]
    if lengths[0] == lengths[1]:
        raise RuntimeError("the probe prompts tokenise to equal lengths; a batch of them would not "
                           "be padded, and padding is half of what is being checked")
    cont = backend.tokenize(CONTINUATION)[:FORCED_TOKENS]
    if len(cont) < 3:
        raise RuntimeError("the forced continuation tokenises to fewer than three tokens")
    all_ids = list(range(int(info.vocab_size)))
    layers = sorted({L, O})
    stats = backend.residual_stats(texts, layers, skip_first=1)
    h_scale = {lay: float(np.median(stats[lay].norms)) for lay in layers}

    own = vector_id is None
    if own:
        vid = LIVENESS_VECTOR_ID
        rng = np.random.default_rng(seed)
        per_layer = {}
        for lay in layers:
            d = rng.standard_normal(int(info.d_model))
            per_layer[lay] = (d / np.linalg.norm(d) * h_scale[lay]).astype(np.float32)
        if backend.has_vector(vid):
            backend.unregister_vector(vid)
        backend.register_vector(vid, per_layer)
        alpha = 1.0 if alpha is None else float(alpha)
    else:
        if alpha is None:
            raise ValueError("probing a stored vector needs its dose: pass alpha")
        vid = vector_id
        alpha = float(alpha)

    report: dict[str, Any] = dict(check="liveness", layer=L, other_layer=O, alpha=alpha,
                                  vector="random (seeded)" if own else vid, seed=seed,
                                  dtype=info.dtype, model=info.hf_id, tolerances=tol)
    try:
        v = {lay: backend.vector_values(vid, lay) for lay in layers}
        ratio = {lay: alpha * float(np.linalg.norm(v[lay])) / h_scale[lay] for lay in layers}
        report["dose_ratio"] = ratio
        if min(ratio.values()) < MIN_DOSE_RATIO:
            raise ValueError(f"liveness dose too weak: ||alpha v|| / median ||h|| = {ratio}; below "
                             f"{MIN_DOSE_RATIO} a live hook and a dead one look alike")
        ctx = _Probe(backend, vid, alpha, L, O, texts, starts, lengths, cont, all_ids, v, h_scale,
                     tol, seed)
        checks = {
            "zero_dose": ctx.zero_dose(),
            "dose_moves": ctx.dose_moves(),
            "wrong_layer": ctx.wrong_layer(),
            "batch_independence": ctx.batch_independence(),
            "decode_path": ctx.decode_path(),
            "lens_matches_logits": ctx.lens_matches_logits(),
            "generate_path": ctx.generate_path(),
        }
    finally:
        if own and backend.has_vector(vid):
            backend.unregister_vector(vid)
    report["checks"] = checks
    report["failed"] = [name for name, c in checks.items() if not c["passed"]]
    report["passed"] = not report["failed"]
    if not report["passed"]:
        raise LivenessError(report)
    return report


class _Probe:
    """The probe prompts, vector and tolerances, and one method per check."""

    def __init__(self, be: Any, vid: str, alpha: float, L: int, O: int, texts: list[str],
                 starts: list[int], lengths: list[int], cont: list[int], all_ids: list[int],
                 v: dict[int, np.ndarray], h_scale: dict[int, float], tol: dict[str, float],
                 seed: int) -> None:
        self.be, self.vid, self.alpha, self.L, self.O = be, vid, alpha, L, O
        self.texts, self.starts, self.lengths = texts, starts, lengths
        self.cont, self.all_ids = cont, all_ids
        self.v, self.h_scale, self.tol, self.seed = v, h_scale, tol, seed
        self.live = [Positions("from", s) for s in starts]
        self.span = [Positions("span", s, s + 3) for s in starts]
        self.gen = [Positions("generated")] * len(texts)
        self.all_pos = [list(range(n)) for n in lengths]
        self._base: np.ndarray | None = None

    def iv(self, layer: int, alpha: float, pos: Positions, op: str = "add") -> Intervention:
        return Intervention(self.vid, op, (LayerTerm(layer, alpha),), pos)

    def next(self, ivs: list[Intervention | None], texts: list[str] | None = None) -> np.ndarray:
        return self.be.next_token_logprobs(texts or self.texts, ivs, self.all_ids)

    def base(self) -> np.ndarray:
        if self._base is None:
            self._base = self.next([None] * len(self.texts))
        return self._base

    def forced(self, ivs: list[Intervention | None], path: str, texts: list[str] | None = None,
               conts: list[list[int]] | None = None) -> list[Any]:
        texts = texts or self.texts
        return self.be.forced_logprobs(texts, ivs, conts or [self.cont] * len(texts), path=path)

    # ---------------------------------------------------------------- checks

    def zero_dose(self) -> dict[str, Any]:
        n = len(self.texts)
        zero = [self.iv(self.L, 0.0, p) for p in self.live]
        d_next = _max(np.abs(self.next(zero) - self.base()))
        un = self.forced([None] * n, "decode")
        z = self.forced(zero, "decode")
        d_decode = max(_max(np.abs(a.logprobs - b.logprobs)) for a, b in zip(un, z))
        reqs = [GenRequest(t, None, GENERATED_TOKENS, 1.0, top_p=1.0, top_k=0, seed=self.seed + i)
                for i, t in enumerate(self.texts)]
        g_none = self.be.generate(reqs)
        g_zero = self.be.generate([replace(r, intervention=iv) for r, iv in zip(reqs, zero)])
        same = all(a.token_ids == b.token_ids for a, b in zip(g_none, g_zero))
        ok = d_next <= self.tol["exact"] and d_decode <= self.tol["exact"] and same
        return dict(passed=ok, next_token_max_abs=d_next, decode_max_abs=d_decode,
                    generation_identical=same)

    def dose_moves(self) -> dict[str, Any]:
        n = len(self.texts)
        ivs = [self.iv(self.L, self.alpha, p) for p in self.live]
        hu = self.be.hidden(self.texts, [None] * n, [self.L], self.all_pos)
        hs = self.be.hidden(self.texts, ivs, [self.L], self.all_pos)
        want = self.alpha * self.v[self.L].astype(np.float64)
        edit_err, untouched = 0.0, 0.0
        for i, s in enumerate(self.starts):
            delta = hs[i][s:, 0, :].astype(np.float64) - hu[i][s:, 0, :]
            scale = np.maximum(np.linalg.norm(want), np.linalg.norm(hu[i][s:, 0, :], axis=-1))
            edit_err = max(edit_err, _max(np.linalg.norm(delta - want, axis=-1) / scale))
            untouched = max(untouched, _max(np.abs(hs[i][:s] - hu[i][:s])))
        tv = float(_tv(self.base(), self.next(ivs)).min())
        ok = edit_err <= self.tol["edit"] and untouched <= self.tol["exact"] and tv >= self.tol["min_tv"]
        return dict(passed=ok, edit_rel_error=edit_err, untouched_max_abs=untouched, next_token_tv_min=tv)

    def wrong_layer(self) -> dict[str, Any]:
        n = len(self.texts)
        lo, hi = sorted((self.L, self.O))
        last = self.be.info.n_layers - 1
        witnessed = [lay for lay in (lo, hi) if lay != last]
        index_err = 0.0
        if witnessed:
            for i, t in enumerate(self.texts):
                ref = self.be.reference_hidden(t, witnessed)
                mine = self.be.hidden([t], [None], witnessed, [self.all_pos[i]])[0]
                rel = np.linalg.norm(mine - ref, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-30)
                index_err = max(index_err, _max(rel))
        tv = float(_tv(self.next([self.iv(self.L, self.alpha, p) for p in self.live]),
                       self.next([self.iv(self.O, self.alpha, p) for p in self.live])).min())
        hu = self.be.hidden(self.texts, [None] * n, [lo, hi], self.all_pos)
        h_hi = self.be.hidden(self.texts, [self.iv(hi, self.alpha, p) for p in self.live], [lo],
                              self.all_pos)
        h_lo = self.be.hidden(self.texts, [self.iv(lo, self.alpha, p) for p in self.live], [hi],
                              self.all_pos)
        earlier = max(_max(np.abs(h_hi[i][:, 0] - hu[i][:, 0])) for i in range(n))
        later = min(_max(np.linalg.norm(h_lo[i][s:, 0] - hu[i][s:, 1], axis=-1)
                         / np.linalg.norm(hu[i][s:, 1], axis=-1)) for i, s in enumerate(self.starts))
        ok = (index_err <= self.tol["batch_hidden"] and tv >= self.tol["min_tv"]
              and earlier <= self.tol["exact"] and later >= self.tol["min_rel"])
        return dict(passed=ok, witnessed_layers=witnessed, index_rel_error=index_err,
                    layer_vs_other_tv_min=tv, earlier_untouched_max_abs=earlier, later_moved_rel_min=later)

    def batch_independence(self) -> dict[str, Any]:
        t0, t1 = self.texts
        L, O, a = self.L, self.O, self.alpha
        clamp_c = 0.5 * self.h_scale[L]
        rows: list[tuple[str, Intervention | None]] = [
            (t0, None),
            (t1, self.iv(L, a, self.live[1])),
            (t0, self.iv(O, a, Positions("all"))),
            (t1, self.iv(L, 0.0, self.live[1])),
            (t0, self.iv(L, a, self.span[0])),
            (t1, Intervention(self.vid, "project", (LayerTerm(L, 1.0),), self.live[1])),
            (t0, Intervention(self.vid, "sub", (LayerTerm(L, a), LayerTerm(O, a)), self.live[0])),
            (t1, Intervention(self.vid, "clamp", (LayerTerm(L, 0.0),), self.live[1], clamp_value=clamp_c)),
        ][:self.be.batch_size]
        if len(rows) < 2:
            return dict(passed=True, note="batch_size 1: rows never share a batch in this backend")
        prompts, ivs = [p for p, _ in rows], [iv for _, iv in rows]
        lens = [self.lengths[self.texts.index(p)] for p in prompts]
        pos = [list(range(n)) for n in lens]
        lp_batch = self.next(ivs, prompts)
        lp_alone = np.stack([self.next([iv], [p])[0] for p, iv in rows])
        tv = _max(_tv(lp_batch, lp_alone))
        h_batch = self.be.hidden(prompts, ivs, [L], pos)
        hid = 0.0
        for i, (p, iv) in enumerate(rows):
            alone = self.be.hidden([p], [iv], [L], [pos[i]])[0]
            rel = (np.linalg.norm(h_batch[i] - alone, axis=-1)
                   / np.maximum(np.linalg.norm(alone, axis=-1), 1e-30))
            hid = max(hid, _max(rel))
        dec_batch = self.forced(ivs, "decode", prompts, [self.cont] * len(rows))
        dec = 0.0
        for i, (p, iv) in enumerate(rows):
            alone = self.forced([iv], "decode", [p], [self.cont])[0]
            dec = max(dec, _max(np.abs(dec_batch[i].logprobs - alone.logprobs)))
        ok = tv <= self.tol["batch_tv"] and hid <= self.tol["batch_hidden"] and dec <= self.tol["paths"]
        return dict(passed=ok, rows=len(rows), next_token_tv_max=tv, hidden_rel_max=hid, decode_max_abs=dec)

    def decode_path(self) -> dict[str, Any]:
        n = len(self.texts)
        agree = {}
        decoded = {}
        for name, positions in (("from", self.live), ("span", self.span), ("generated", self.gen)):
            ivs = [self.iv(self.L, self.alpha, p) for p in positions]
            pre = self.forced(ivs, "prefill")
            dec = self.forced(ivs, "decode")
            decoded[name] = dec
            agree[name] = max(_max(np.abs(a.logprobs - b.logprobs)) for a, b in zip(pre, dec))
        un = self.forced([None] * n, "decode")
        first = max(abs(float(g.logprobs[0] - u.logprobs[0])) for g, u in zip(decoded["generated"], un))
        moved = min(_max(np.abs(g.logprobs[1:] - u.logprobs[1:])) for g, u in zip(decoded["generated"], un))
        ok = (max(agree.values()) <= self.tol["paths"] and first <= self.tol["exact"]
              and moved >= self.tol["min_lp"])
        return dict(passed=ok, decode_vs_prefill_max_abs=agree, generated_first_token_max_abs=first,
                    generated_later_tokens_moved_min=moved)

    def lens_matches_logits(self) -> dict[str, Any]:
        n = len(self.texts)
        last = self.be.info.n_layers - 1
        worst = 0.0
        for ivs in ([None] * n, [self.iv(self.L, self.alpha, p) for p in self.live]):
            nt = self.next(ivs)
            ln = self.be.lens(self.texts, ivs, [last], -1, self.all_ids)[:, 0, :]
            worst = max(worst, _max(_tv(nt, ln)))
        return dict(passed=worst <= self.tol["lens"], lens_vs_logits_tv_max=worst)

    def generate_path(self) -> dict[str, Any]:
        n = len(self.texts)
        ivs = [self.iv(self.L, self.alpha, p) for p in self.live]
        gens = self.be.generate([GenRequest(t, iv, GENERATED_TOKENS, 0.0, top_p=1.0, top_k=0,
                                            seed=self.seed) for t, iv in zip(self.texts, ivs)])
        tokens = [list(g.token_ids) for g in gens]
        steered = self.forced(ivs, "decode", self.texts, tokens)
        unsteered = self.forced([None] * n, "decode", self.texts, tokens)
        mismatched = [i for i in range(n) if not np.array_equal(steered[i].argmax, np.array(tokens[i]))]
        moved = min(_max(np.abs(s.logprobs - u.logprobs)) for s, u in zip(steered, unsteered))
        ok = not mismatched and moved >= self.tol["min_lp"]
        return dict(passed=ok, rows_not_argmax_of_steered=mismatched, steered_vs_unsteered_min=moved)
