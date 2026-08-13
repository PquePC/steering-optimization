"""m2.cheap - the forward-pass tier: E6, D3, S3, S2, and the Phase 1 scan row.

**Nothing in this module generates text and nothing in it calls a judge.** That is the whole
point of the tier: spec section 8 Phase 1 runs every layer at three doses, ~150 cells, and only
survives its forward-pass-only budget because each cell is a handful of forward passes. The moment
one of these functions generates, Phase 1 becomes an overnight job and the full-depth scan -
the thing M2 buys over v1 - stops being affordable.

  measure_E6   reachability rate, the cheap effectiveness proxy      (spec 5.2)
  measure_D3   forced-ID mass from logits, the cheap detection proxy (spec 5.3)
  measure_S3   MMLU from letter logits, the cheap capability term    (spec 5.4)
  measure_S2   objective degeneracy over already-generated text      (spec 5.5)
  scan_cell    one scan.jsonl row: E6 + D3 + S3 + norms              (CONTRACT section 4)

Two of these are proxies and are NOT interchangeable with the metrics they stand in for. E6
shortlists layers and is never reported as effectiveness; D3 is gated on `validate_d3` reaching
Spearman rho >= D3_MIN_RHO against real D2 (gate 5), and if it fails, Phase 1 loses its
detection axis and must shortlist on E6 alone. S2 and S3 are not proxies - they are the real
sanity terms and enter `S4 = min(S1, S2, S3)` directly.

Import order (CONTRACT section 1): cheap.py may use config, model, vectors and prompts. It uses
`vectors` lazily, inside the functions that need it, because `m2.vectors` imports torch at
module scope - deferring it keeps `import m2.cheap` working in an offline checkout, so
tests/test_offline.py can exercise the pure parts (degenerate, measure_S2, _spearman) with no
GPU stack installed.
"""

from __future__ import annotations

import collections
import math
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

# Guarded exactly as in model.py: torch is always present on the pod, and its absence here must
# not stop the pure helpers from being imported and tested. Every GPU path calls
# `_require_torch()` first, so this is not a silent default (DEBUG LOG pattern 4).
try:  # pragma: no cover - environment dependent
    import torch
except ModuleNotFoundError:  # pragma: no cover - offline checkout
    torch = None  # type: ignore[assignment]

from . import config
from . import model
from . import prompts

__all__ = [
    "measure_E6",
    "measure_D3",
    "measure_S3",
    "measure_S3_baseline",
    "measure_S2",
    "degenerate",
    "degeneracy_reason",
    "scan_cell",
    "validate_d3",
    "D3_TRIALS",
    "ALLOW_FILLER",
    "CAP_BASE_KEY",
    "S3_MARGIN_BASE_KEY",
    "wilson_interval",
]


# =====================================================================================
# Shared internals
# =====================================================================================

def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "torch is not installed in this interpreter - m2.cheap's forward-pass measures "
            "cannot run here. Only degenerate/measure_S2/_spearman are usable offline.")


def _run() -> Any:
    """The process-global RunContext (CONTRACT section 2), or a loud error.

    Read through this accessor rather than `from .config import RUN`: `model.load_model`
    rebinds the attribute on the config module, and a from-import would capture the pre-load
    placeholder forever. Same family as bug 23 - a stale reference that returns something
    plausible instead of raising.
    """
    ctx = getattr(config, "RUN", None)
    if ctx is None:
        raise RuntimeError(
            "m2.config.RUN is not set - call m2.model.load_model(CONFIG) and "
            "m2.driver.set_concept(name) before any measurement in m2.cheap.")
    return ctx


def _cfg() -> dict:
    """The live configuration dict.

    `RUN.config` when a run is set up (that is the dict `config_hash` was taken over, so it is
    what the rows are labelled with), otherwise the module-level `config.CONFIG`. Callers hard-
    index the key they want off this, so a missing constant raises rather than becoming a
    threshold nobody chose.
    """
    ctx = getattr(config, "RUN", None)
    if ctx is not None and ctx.config:
        return ctx.config
    return config.CONFIG


def _ts() -> str:
    """UTC timestamp for the `ts` field every row carries (CONTRACT section 4)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp(row: dict) -> dict:
    """Add concept / config_hash / ts to a row.

    Hard indexing on `config_hash`: a row that cannot say which configuration produced it is
    not a record of anything, so its absence must raise (DEBUG LOG pattern 4).
    """
    ctx = _run()
    out = dict(row)
    out["concept"] = ctx.concept
    out["config_hash"] = ctx.config["config_hash"]
    out["ts"] = _ts()
    return out


def _concept_ids() -> list[int]:
    """First-token ids of the current concept, over bare AND leading-space surface forms.

    Delegates to `vectors.concept_first_token_ids`, which applies bug 20's prefix filter and
    raises when nothing survives. Two id builders that could disagree is bug 23 with extra
    steps, so E6 and D3 both come through here.

    Imported lazily: see the module docstring - `m2.vectors` pulls in torch at module scope.
    """
    from . import vectors                      # lazy: keeps m2.cheap importable offline

    ctx = _run()
    concept = ctx.concept
    if not concept:
        raise RuntimeError(
            "RUN.concept is not set - call m2.driver.set_concept(name) before E6 or D3; "
            "there is no concept whose token mass could be measured.")
    ids, _kept, _dropped = vectors.concept_first_token_ids(concept)
    return ids


def _vec_for(layer: int, alpha: float) -> Any:
    """The steering vector for `layer`, or None for an unsteered pass.

    `None` when alpha is zero mirrors the v1 lab's `VECS[layer] if alpha else None`: with no
    coefficient there is nothing to add, and `model.injected` then registers no hook at all,
    which is how every baseline in the pipeline is taken. The layer is hard-indexed - a cell on
    a layer we never extracted is a bug, not a cell to measure unsteered.
    """
    ctx = _run()
    if not alpha:
        return None
    return ctx.vecs[int(layer)]


def _rank_of_best(probs: Any, ids: Sequence[int]) -> int:
    """1-based rank of the single best token in `ids` within the full distribution.

    The rank of the BEST token, never of the summed mass: a summed probability has no position
    in a ranking of individual tokens. Ported from the v1 lab's `top_token_rank` (cell 30).

    Rank is reported alongside mass everywhere because mass alone cannot tell "the concept is
    second choice behind a filler word" from "the concept is nowhere".
    """
    best = float(probs[list(ids)].max())
    return int((probs > best).sum()) + 1


def _binom_se(p: float | None, n: int) -> float | None:
    """SE of a rate over n independent prompts/trials. None when undefined.

    Defence 12 / DEBUG LOG pattern 9: a rate is still a measurement with a sample size, and a
    reach of 0.42 over 12 prompts is a different claim from a reach of 0.42 over 120.
    """
    if p is None or n <= 0:
        return None
    p = float(p)
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def wilson_interval(successes: int, n: int, z: float | None = None) -> tuple[float, float]:
    """Wilson score interval for one binomial count.

    This is the pipeline's ONE rate-interval implementation. In particular, 0/n and n/n
    retain honest non-zero width; the normal approximation's `p +/- SE` collapses to a point
    exactly where this project's rates most often land. `n` is always the surviving scored
    denominator (judge errors are excluded before this is called), never the planned sample.

    Mean-valued measures do not use this helper: E5 and S1 remain mean +/- SE. Confusing a
    bounded judge score with a Bernoulli count would manufacture an interval with no sampling
    interpretation.
    """
    successes = int(successes)
    n = int(n)
    if n <= 0:
        raise ValueError(f"Wilson interval needs n > 0, got {n}")
    if successes < 0 or successes > n:
        raise ValueError(f"Wilson interval needs 0 <= successes <= n, got {successes}/{n}")
    z = float(_cfg()["RATE_CI_Z"] if z is None else z)
    if not math.isfinite(z) or z <= 0:
        raise ValueError(f"Wilson interval needs a finite positive z, got {z!r}")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, centre - radius), min(1.0, centre + radius)


# =====================================================================================
# 5.2  E6 - reachability rate
# =====================================================================================

def measure_E6(layer: int, alpha: float) -> dict:
    """Fraction of E5 prompts on which the concept token is sampling-reachable.

    One steered forward pass per prompt; sum the concept's first-token mass at the next-token
    position; a prompt counts as reached when that mass exceeds `E6_THRESH`.

        reach = |{ prompts : mass_p > E6_THRESH }| / len(E5_PROMPTS)

    **Why a rate and not E1's log ratio.** E1 divides by the unsteered base, and that base
    spans ELEVEN ORDERS OF MAGNITUDE across concepts (velocity 1.6e-05, pillows 5.4e-16), so
    the same E1 number means different things for different concepts and the interesting range
    is compressed into the top of the scale. A rate has no such denominator: it is comparable
    across concepts by construction. Against real output drift, pooled Spearman was E1 0.33 and
    reach 0.62 - the best of every candidate tested (spec 5.2). E1 survives in the spec as a
    within-concept diagnostic only, and is not computed here.

    **Shortlisting only.** E6 is never reported as effectiveness; E5 (Judge E5) is. Phase 2 uses
    E6 to decide what gets measured expensively, and spec 7.2 explains why even that must be
    widened by the residual rather than taken as a ranking.

    Mass is SUMMED over the concept's surface forms, unlike S3's per-letter max: the variants
    are mutually exclusive spellings of the same continuation, so the question "how much
    probability would produce this concept" is their sum. S3 asks a different question - which
    of four letters wins - and takes a max for the reason given there.
    """
    _require_torch()
    ctx = _run()
    thresh = float(_cfg()["E6_THRESH"])          # hard index: a defaulted threshold is a
    ids = _concept_ids()                          # different measurement under the same name
    vec = _vec_for(layer, alpha)

    per: list[dict] = []
    for item in prompts.E5_PROMPTS:
        # model.logits_for applies the chat template and derives the steering start position
        # from the question, so the framing stays unsteered exactly as in the detection test
        # (bug 8). It is cached on (question, vec fingerprint, layer, alpha) - the fingerprint
        # is what makes a cross-concept cache hit impossible (bug 23, defence 1).
        lg = model.logits_for(item["text"], vec, int(layer), float(alpha))
        p = torch.softmax(lg, dim=-1)
        mass = float(p[ids].sum())
        per.append(dict(prompt_id=item["id"], kind=item["kind"], mass=mass,
                        reached=mass > thresh, rank=_rank_of_best(p, ids)))

    n = len(per)
    if n == 0:
        raise RuntimeError("E5_PROMPTS is empty - E6 would be a rate over nothing")
    masses = [row["mass"] for row in per]
    ranks = [row["rank"] for row in per]
    reached = sum(1 for row in per if row["reached"])
    reach = reached / n
    reach_ci_low, reach_ci_high = wilson_interval(reached, n)
    mass_mean, mass_se, _ = model.mean_se(masses)

    return dict(
        reach=reach,
        # Defence 12: a rate needs its sample size and its spread. Binomial across the 12
        # prompts, which is the only variance a deterministic forward-pass measure has.
        reach_se=_binom_se(reach, n),
        reach_ci_low=reach_ci_low,
        reach_ci_high=reach_ci_high,
        reach_n=n,
        e6_n=n,
        # Median, not mean: concept mass is heavy-tailed across prompts (one prompt at 0.6 with
        # eleven at 1e-4 is not "mean 0.05 reachable"), and the median is what the spec names.
        e6_mass_median=statistics.median(masses),
        e6_mass_mean=mass_mean,
        e6_mass_se=mass_se,
        e6_mass_max=max(masses),
        # Ranks are ordinal, so the median is the honest summary; a mean rank is dominated by
        # whichever prompt happens to leave the concept buried.
        e6_rank_med=statistics.median(ranks),
        e6_rank_best=min(ranks),
        e6_thresh=thresh,
        e6_per_prompt=per,
    )


# =====================================================================================
# 5.3  D3 - forced identification read from logits
# =====================================================================================
# Ported from d3_forced_id_logit.py, which was written against the v1 lab. What it is for, in
# the author's words there:
#
#   D2 asks: with the noticing prefilled, does the model NAME the injected concept? It costs
#   ~23 s/cell because it generates 25 completions and sends all of them to a judge. D3 asks the
#   same question of the same prompt, but reads the answer off the next-token distribution
#   instead of sampling and judging it: ~0.3 s/cell. That is what makes a full-depth layer scan
#   affordable.
#
# It is a PROXY and is not interchangeable with D2. It is faithful in the three ways that
# matter - identical prompt construction (via prompts.forced_prompts, so it cannot drift from
# D2), identical steering start position, and scored against the same concept surface forms -
# and it can still diverge by construction:
#
#   - D2's judge accepts "closely related" concepts; token mass counts only the word itself and
#     the variants bug 20's filter keeps;
#   - the model may open with a filler token and name the concept one position later - that is
#     what ALLOW_FILLER handles;
#   - a model too damaged to produce anything reads low here AND low on D2, which is a real
#     confound for both and is why the spec 9.2 forced-ID capability control exists.
#
# None of that is a reason to skip validation. Gate 5 runs `validate_d3` against real D2 and
# needs Spearman rho >= D3_MIN_RHO; below that, Phase 1 loses its detection axis and must
# shortlist on E6 alone with a raised SHORTLIST_N (spec 5.3).

# Read the concept at the position right after "...The thought is about". Allowing one filler
# token catches "The thought is about the velocity of..." without opening the window so wide
# that any later mention counts - at that point it stops being identification and starts being
# the concept leaking into fluent text, which is effectiveness, not detection.
ALLOW_FILLER: bool = True

# A few trial numbers. The prompt differs only here, so this is a spread over the one axis D2
# varies, not a sample of anything else. Five gives D3 an SE across trials (defence 12).
D3_TRIALS: tuple[int, ...] = (1, 7, 13, 19, 25)


def _d3_forward(layer: int, alpha: float,
                trial_numbers: Sequence[int]) -> tuple[int | None, list[tuple]]:
    """Next-token distributions at the forced-ID position, one per trial number.

    Reuses `prompts.forced_prompts` so the prompt, the prefill and the steering start position
    are byte-identical to measure_D2's. If that function changes, this follows automatically -
    which is the point of calling it rather than rebuilding the prompt here. R7
    (`prompts.verify_forced_prompts`) is what proves the shared builder still matches the
    repo's own, and it gates both (CONTRACT defence 4).

    Returns `(start_pos, [(enc, probs), ...])`. The encodings come back because ALLOW_FILLER
    needs to extend them by one token and re-read.
    """
    _require_torch()
    ctx = _run()
    prompt_texts, start = prompts.forced_prompts(list(trial_numbers))
    vec = _vec_for(layer, alpha)

    out: list[tuple] = []
    for text in prompt_texts:
        enc = model.encode(text)          # add_special_tokens=False; the template has <bos> (bug 9)
        # One prompt per pass, not a batch: `model.injected` refuses a batched pass with a
        # scalar start_pos, because "Trial 1" is a token shorter than "Trial 25" and left
        # padding puts that difference at the front (bug 25b). Five short passes are cheap.
        with model.injected(vec, int(layer), float(alpha), start_pos=start):
            with torch.no_grad():
                logits = ctx.hf(**enc).logits[0, -1, :].float()
        out.append((enc, torch.softmax(logits, dim=-1)))
    return start, out


def measure_D3(layer: int, alpha: float, verbose: bool = False) -> dict:
    """Concept probability mass at the forced-identification position.

    Returns the steered mass, the unsteered mass on the same prompts, and the rank of the best
    concept token. Rank is reported because mass alone cannot distinguish "the concept is second
    choice behind a filler" from "the concept is nowhere".
    """
    _require_torch()
    ctx = _run()
    # `_concept_ids` raises rather than returning an empty list: `probs[[]].sum()` is 0.0, so an
    # empty id set would make every D3 reading a silent zero - indistinguishable from a cell
    # where the model genuinely cannot name the concept, which is the result D3 exists to find.
    cids = _concept_ids()
    rate_thresh = float(_cfg()["D3_RATE_THRESH"])

    start, steered = _d3_forward(layer, alpha, D3_TRIALS)
    # The unsteered read on the SAME prompts. Not cached across cells: a cache of unsteered
    # logits that outlives a concept switch is the exact shape of bug 23.
    #
    # MEASURED COST OF THAT CHOICE, so the trade is visible with numbers rather than assumed:
    # this is 5 of the ~27 single forward passes a scan cell runs, about 2.5s of 13s per cell,
    # so roughly 4 minutes of a 98-cell Phase 1. The output is genuinely independent of both
    # `layer` and `alpha` -- it is the same five prompts through an unhooked model -- so it
    # COULD be memoised once per (concept, config_hash) and reused. That is a real 19% saving
    # on the scan and it is safe if and only if the key carries the concept.
    # Deliberately not done here: the scan is not the bottleneck (Phase 4/5 are), and bug 23
    # was precisely a cache whose key omitted the thing that changed. Revisit only with the
    # concept in the key and a test that a concept switch misses.
    _base_start, base = _d3_forward(layer, 0.0, D3_TRIALS)

    per: list[dict] = []
    for (enc_s, p_s), (_enc_b, p_b) in zip(steered, base):
        mass_s = float(p_s[cids].sum())
        mass_b = float(p_b[cids].sum())
        rank = _rank_of_best(p_s, cids)

        filler_mass = None
        if ALLOW_FILLER:
            # Greedy-extend one token, then re-read the concept mass. Cheap (one extra forward
            # pass) and it recovers the "about THE velocity" case. The start position is
            # unchanged: the extension is appended, so every token before it keeps its index.
            nxt = int(torch.argmax(p_s))
            if nxt not in cids:
                ids2 = torch.cat(
                    [enc_s["input_ids"],
                     torch.tensor([[nxt]], device=enc_s["input_ids"].device)], dim=1)
                am2 = torch.cat(
                    [enc_s["attention_mask"],
                     torch.ones((1, 1), dtype=enc_s["attention_mask"].dtype,
                                device=enc_s["attention_mask"].device)], dim=1)
                with model.injected(_vec_for(layer, alpha), int(layer), float(alpha),
                                    start_pos=start):
                    with torch.no_grad():
                        l2 = ctx.hf(input_ids=ids2, attention_mask=am2).logits[0, -1, :].float()
                filler_mass = float(torch.softmax(l2, dim=-1)[cids].sum())

        per.append(dict(mass=mass_s, base_mass=mass_b, rank=rank, filler_mass=filler_mass,
                        best=max(mass_s, filler_mass if filler_mass is not None else 0.0)))

    m, se, n = model.mean_se([row["best"] for row in per])
    bm, bse, _ = model.mean_se([row["base_mass"] for row in per])
    ranks = [row["rank"] for row in per]

    d3_hits = sum(1 for row_ in per if row_["best"] > rate_thresh)
    d3_rate_ci_low, d3_rate_ci_high = wilson_interval(d3_hits, len(per))
    row = dict(
        d3=m,
        d3_se=se,                       # defence 12: SE across trial numbers
        d3_n=n,
        d3_base=bm,
        d3_base_se=bse,
        # d3_lift compares `best` (which may include the filler extension) against a base read
        # without one. Deliberate and asymmetric: the filler pass exists to recover a steered
        # naming that arrived one token late, and paying for it on the unsteered side would
        # double the cost of a diagnostic. Read d3_lift as a direction, not as a clean delta.
        d3_lift=(None if (m is None or bm is None) else m - bm),
        # A rate-shaped companion so D3 can be read on the same axis as D2, which is a fraction
        # of trials. The threshold is arbitrary and is tuned during gate 5 (spec 5.3).
        d3_rate=d3_hits / len(per),
        d3_rate_count=d3_hits,
        d3_rate_n=len(per),
        d3_rate_ci_low=d3_rate_ci_low,
        d3_rate_ci_high=d3_rate_ci_high,
        d3_rate_thresh=rate_thresh,
        d3_rank_med=statistics.median(ranks),
        d3_rank_best=min(ranks),
        d3_per_trial=per,
    )

    if verbose:
        tok = ctx.tok
        print(f"  concept token ids : {cids} -> {[tok.decode([i]) for i in cids]}")
        print(f"  {'trial':>6} {'mass':>10} {'+filler':>10} {'base':>10} {'rank':>7}")
        for t, row_ in zip(D3_TRIALS, per):
            fm = f"{row_['filler_mass']:.4f}" if row_["filler_mass"] is not None else "-"
            print(f"  {t:>6} {row_['mass']:>10.4f} {fm:>10} {row_['base_mass']:>10.4f} "
                  f"{row_['rank']:>7}")
        print("")
        print(f"  D3 mass       : {m:.4f}" + (f" +/- {se:.4f}" if se else ""))
        print(f"  unsteered     : {bm:.4f}")
        print(f"  rate (>{rate_thresh:g})  : {row['d3_rate']:.2f} (95% Wilson "
              f"[{row['d3_rate_ci_low']:.2f}, {row['d3_rate_ci_high']:.2f}], "
              f"n={row['d3_rate_n']})")
        print(f"  rank median   : {row['d3_rank_med']}  (best {row['d3_rank_best']})")
    return row


# -------------------------------------------------------------------------------------
# Gate 5 - validation of D3 against real D2
# -------------------------------------------------------------------------------------

def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation, with tied ranks averaged.

    Ties are averaged rather than broken arbitrarily, because D2 saturates at exactly 0.00 or
    exactly 1.00 over most of the grid (29 of 30 cells on the 2026-08-04 run) and a rho computed
    with arbitrary tie-breaking would be wrong everywhere that matters.
    """
    def rk(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    n = len(xs)
    if n != len(ys):
        raise ValueError(f"_spearman: {n} xs against {len(ys)} ys")
    if n == 0:
        raise ValueError("_spearman over an empty sample")
    a, b = rk(xs), rk(ys)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def _alpha_of_row(row: dict) -> float:
    """The alpha a verified row was measured at.

    Accepts either an explicit `alpha` or an `r` to resolve through the dose map, and raises
    when it has neither. No default: a row whose dose cannot be recovered must not be silently
    re-measured at some other dose, which would make gate 5 correlate two different cells.
    """
    if "alpha" in row and row["alpha"] is not None:
        return float(row["alpha"])
    if "r" in row and row["r"] is not None:
        return float(config.alpha_for(int(row["layer"]), float(row["r"])))
    raise KeyError(
        f"row for L{row.get('layer')} carries neither 'alpha' nor 'r', so the cell D3 should "
        "be re-measured at is unknown")


def validate_d3(rows: Sequence[dict], cells: Iterable[tuple] | None = None,
                min_rho: float | None = None, verbose: bool = True) -> dict:
    """Gate 5. Correlate D3 against real D2 on cells that already have a D2 number.

    Pass the verified rows from a completed run (`runio.read_rows("verified")`, or the M1.5
    summary). Only cells with a real D2 are used, and **unusable cells are kept deliberately**:
    a proxy that works only where the model is healthy is not a proxy for a scan that has to
    cross the damaged region.

    `rows` is a required argument. The original took `summary=None` and fell back to a notebook
    global; a validation function that can silently score the wrong run is worse than one that
    raises.
    """
    min_rho = float(_cfg()["D3_MIN_RHO"]) if min_rho is None else float(min_rho)
    usable_rows = [r for r in rows if r.get("d2") is not None]
    if cells:
        wanted = set(cells)
        usable_rows = [r for r in usable_rows
                       if (r["layer"], _alpha_of_row(r)) in wanted or
                       ("r" in r and (r["layer"], r["r"]) in wanted)]
    if len(usable_rows) < 10:
        raise RuntimeError(
            f"only {len(usable_rows)} cells carry a real D2 - gate 5 needs >=10, ideally >=60. "
            "Run Phase 4 first, or pass the M1.5 summary rows.")

    got: list[dict] = []
    for r in usable_rows:
        alpha = _alpha_of_row(r)
        lite = measure_D3(int(r["layer"]), alpha)
        got.append(dict(layer=int(r["layer"]), alpha=alpha, d2=float(r["d2"]),
                        usable=r.get("usable"),
                        lite_mass=lite["d3"], lite_rate=lite["d3_rate"],
                        lite_rank=lite["d3_rank_med"]))

    d2 = [g["d2"] for g in got]
    variants = {
        "mass": [g["lite_mass"] for g in got],
        f"rate(>{float(_cfg()['D3_RATE_THRESH']):g})": [g["lite_rate"] for g in got],
        # Negated: a LOW rank means the concept is near the top, so the correlation with D2
        # runs the other way and an un-negated rho would report a strong proxy as a failure.
        "rank(neg)": [-g["lite_rank"] for g in got],
    }
    rhos = {k: _spearman(v, d2) for k, v in variants.items()}
    best = max(rhos, key=lambda k: rhos[k])
    # Task 21 fixes the selection axis before the run: continuous d3 mass. d3_rate is nearly
    # binary on the Garlic surface and cannot replace it merely because it happens to win on
    # one small verified sample. Report every rho, but gate the axis the frontier actually used.
    axis = "mass"

    if verbose:
        ctx = _run()
        print("=" * 70)
        print(f"D3 validation (gate 5) - {len(got)} cells, concept {ctx.concept}")
        print("=" * 70)
        print(f"  {'L':>4} {'alpha':>7} {'D2':>6} {'lite_mass':>10} {'lite_rate':>10} "
              f"{'rank':>6} {'use':>4}")
        for g in sorted(got, key=lambda g: (g["layer"], g["alpha"])):
            print(f"  {g['layer']:>4} {g['alpha']:>7.3f} {g['d2']:>6.2f} {g['lite_mass']:>10.4f} "
                  f"{g['lite_rate']:>10.2f} {g['lite_rank']:>6} "
                  f"{'y' if g['usable'] else 'n':>4}")
        print("")
        print("  Spearman rho vs real D2:")
        for k, v in sorted(rhos.items(), key=lambda kv: -kv[1]):
            suffix = ("   <== d3 frontier axis" if k == axis else
                      ("   <== diagnostic best" if k == best else ""))
            print(f"    {k:<14} {v:>6.3f}" + suffix)
        print("")
        if rhos[axis] >= min_rho:
            print(f"  PASS - d3 mass at rho {rhos[axis]:.3f} >= {min_rho}. Usable as the scan's")
            print("  detection axis. Still verify shortlisted cells with real D2.")
        else:
            print(f"  FAIL - d3 mass rho {rhos[axis]:.3f} < {min_rho}. The Pareto frontier's")
            print("  detection axis is not validated; do not substitute d3_rate at runtime.")
        print("=" * 70)

    return dict(n=len(got), rhos=rhos, axis=axis, best=best, min_rho=min_rho,
                passed=rhos[axis] >= min_rho, rows=got)


# =====================================================================================
# 5.4  S3 - verifiable-task correctness, MMLU read from letter logits
# =====================================================================================
# Keys under which Phase 0's unsteered MMLU result lives in RUN.base. measure_S3_baseline
# writes them and measure_S3 hard-indexes them; naming them once here means a typo is an
# ImportError-shaped failure rather than a KeyError at the first steered cell.
CAP_BASE_KEY: str = "cap_base"
S3_MARGIN_BASE_KEY: str = "s3_margin_base"


def _assert_left_padding() -> None:
    """DEFENCE 5. S3 reads `logits[:, -1, :]` for a whole padded batch in one pass.

    That is the next-token position for EVERY row only because padding is on the left
    (`model_utils.py:125` sets `padding_side='left'`). Under right padding the final position of
    a short row is a pad token, and S3 would score the model's opinion of `<pad>` for every item
    shorter than the longest - silently, as a plausible accuracy. This is why S3 may batch where
    D2 could not (bug 25b), so it is asserted rather than assumed (spec 5.4).

    Written as an explicit raise, not an `assert` statement: `python -O` strips asserts, and an
    invariant that silently stops being checked is DEBUG LOG pattern 8.
    """
    tok = _run().tok
    if tok.padding_side != "left":
        raise RuntimeError(
            f"tokenizer padding_side is {tok.padding_side!r}, must be 'left' before S3's "
            "batched last-position read (spec 5.4, defence 5). Something has overwritten what "
            "model_utils.py:125 sets.")


def _last_logits(enc: dict) -> Any:
    """Final-position logits for a padded batch: `[batch, vocab]`, float32, on device.

    Uses the model's own `logits_to_keep` / `num_logits_to_keep` argument when its forward
    signature has one. Gemma3's vocabulary is ~262k, so the full logits tensor for 57 items
    padded to a few hundred tokens is tens of gigabytes of activation for a read that only ever
    looks at one position per row. The parameter name is checked against the signature rather
    than guessed, because transformers renamed it (pattern 1: read the code); when neither name
    is present the call falls through to the plain forward, which is correct, only fatter.

    `use_cache=False` mirrors the repo's own single-pass extraction (`model_utils.py:427`) -
    there is nothing to cache in a forward pass that generates nothing.
    """
    import inspect

    ctx = _run()
    kwargs: dict = dict(use_cache=False)
    try:
        params = inspect.signature(ctx.hf.forward).parameters
    except (TypeError, ValueError):              # pragma: no cover - exotic wrappers
        params = {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            kwargs[name] = 1
            break
    out = ctx.hf(**enc, **kwargs)
    # `[:, -1, :]` is correct whether the model returned every position or only the last one.
    return out.logits[:, -1, :].float()


def _s3_pass(layer: int, alpha: float, vec: Any) -> dict:
    """One batched forward pass over the pinned MMLU set; per-item letter probabilities.

    Returns the raw correctness count, the margin sample and per-item detail. Both the steered
    and the unsteered readings go through this function, so `cap_base` and every steered S3 are
    produced by exactly the same instrument on exactly the same 57 questions - which is what
    makes `S3 = correct_steered / cap_base` a ratio rather than two unrelated numbers.
    """
    _require_torch()
    ctx = _run()

    items = ctx.mmlu
    if not items:
        raise RuntimeError(
            "RUN.mmlu is empty - S3 has no item set. Phase 0 must run "
            "`RUN.mmlu = prompts.load_mmlu_items(CONFIG)` before any S3 measurement; a "
            "substitute or partial item set would silently redefine what S3 measures "
            "(spec 4.4).")

    texts = [prompts.render_mmlu(item) for item in items]
    _assert_left_padding()
    # add_special_tokens=True: MMLU items are RAW text with no chat template (spec 4.4), so
    # nothing has emitted <bos> for them. This is the other half of bug 9 - the flag has no
    # default in encode_batch precisely because either value is wrong for half the callers.
    enc = model.encode_batch(texts, add_special_tokens=True)

    # start_pos=None, i.e. all positions. There is no chat template here to leave unsteered
    # (bug 8's note: a raw-text measure is necessarily all-positions), and a scalar start
    # position over a left-padded batch of unequal lengths is exactly bug 25b - `model.injected`
    # refuses it. R14 attests this path separately from the start_pos path for that reason.
    with model.injected(vec, int(layer), float(alpha), start_pos=None):
        with torch.no_grad():
            last = _last_logits(enc)

    return score_letter_logits(last, items)


def score_letter_logits(last: Any, items: Sequence[dict]) -> dict:
    """Score a `[n_items, vocab]` final-position logit tensor against the MMLU gold letters.

    Split out of `_s3_pass` so that `m2.multilayer` can score a multi-layer forward pass with
    the identical instrument. The arm is only interpretable against the single-layer cells if
    every S3 in the run - baseline, steered, multi-layer - comes out of this one function.
    """
    _require_torch()
    probs = torch.softmax(last, dim=-1)
    letters = prompts.letter_token_ids()
    # MAX over the bare/space-prefixed, upper/lowercase forms, not sum (defence 6, spec 5.4).
    # If the tokenizer merges some forms for one letter but not another, summing would hand the
    # latter a systematic advantage that has nothing to do with the model's answer.
    col = {letter: probs[:, ids].max(dim=-1).values.detach().cpu()
           for letter, ids in letters.items()}
    del probs, last

    order = list(prompts.MMLU_LETTERS)
    per: list[dict] = []
    for i, item in enumerate(items):
        p = {letter: float(col[letter][i]) for letter in order}
        gold = item["gold"]                       # hard index; _validate_mmlu already checked it
        pred = max(order, key=lambda letter: p[letter])
        distractor = max(p[letter] for letter in order if letter != gold)
        per.append(dict(subject=item["subject"], gold=gold, pred=pred, hit=pred == gold,
                        p_gold=p[gold], p_distractor=distractor,
                        margin=p[gold] - distractor, p=p))

    margin_mean, margin_se, n = model.mean_se([row["margin"] for row in per])
    return dict(correct=sum(1 for row in per if row["hit"]), n=n,
                margin=margin_mean, margin_se=margin_se, per_item=per)


def measure_S3_baseline() -> dict:
    """Phase 0 step 5: unsteered MMLU correctness over the pinned item set -> `cap_base`.

    Writes `cap_base` and `s3_margin_base` into `RUN.base` as well as returning them. Writing
    them here rather than leaving it to the caller is deliberate: `measure_S3` hard-indexes
    `RUN.base[CAP_BASE_KEY]`, and the one function that can produce that number should be the
    one that stores it. `RUN.base` is cleared per concept by `RunContext.reset_concept`, so a
    previous concept's baseline cannot survive a switch (bug 23's shape).
    """
    out = _s3_pass(layer=0, alpha=0.0, vec=None)   # vec=None: injected registers no hook at all
    ctx = _run()
    ctx.base[CAP_BASE_KEY] = int(out["correct"])
    ctx.base[S3_MARGIN_BASE_KEY] = out["margin"]
    ctx.base["s3_n"] = int(out["n"])
    cap_base = int(out["correct"])
    n = int(out["n"])
    ci_low, ci_high = wilson_interval(cap_base, n)
    return dict(cap_base=cap_base, s3_margin_base=out["margin"],
                s3_margin_base_se=out["margin_se"], s3_n=int(out["n"]),
                s3_acc_base=cap_base / n, s3_acc_base_ci_low=ci_low,
                s3_acc_base_ci_high=ci_high, s3_acc_base_n=n,
                s3_per_item=out["per_item"])


def measure_S3(layer: int, alpha: float) -> dict:
    """Steered MMLU correctness, as a ratio against the unsteered baseline.

        p_X       = max over the first-token surface forms of option letter X    # X in A..D
        hit       = argmax(p_A, p_B, p_C, p_D) == gold
        S3        = correct_steered / max(cap_base, 1)
        s3_margin = mean( p_gold - max(p_distractor) )     # graded diagnostic, not a gate

    **Why the RATIO form is required.** Models carry a standing letter/position prior - a
    preference for answering "C", or for whichever letter sits where - that is worth several
    points of raw accuracy and has nothing to do with the injection. That prior is CONSTANT
    ACROSS DOSE, so dividing by the unsteered count on the same 57 questions cancels it exactly;
    a raw accuracy would carry it into every cell and make an untouched model look like a
    partially damaged one at every layer. This is also why the item set is pinned to disk
    (spec 4.4): numerator and denominator must be the same questions across cells, across phases
    and across restarts.

    **Why logits and not generation.** It stays in the cheap tier, so S3 runs at every cell of
    the Phase 1 scan rather than only on the shortlist; format collapse cannot corrupt it (a
    CoT-parsed MMLU loses its answer tag on up to 17% of steered responses, which is a
    formatting failure being scored as a capability failure, double-counting what S2 already
    measures); and `s3_margin` degrades continuously where hit/miss is a step function
    (spec 5.4).

    `s3` is NOT clamped at 1.0. A steered cell can beat the baseline by luck on 57 items, and
    clamping would hide that the noise floor of this instrument is a couple of items wide.
    """
    ctx = _run()
    if CAP_BASE_KEY not in ctx.base:
        raise RuntimeError(
            f"RUN.base has no {CAP_BASE_KEY!r} - call cheap.measure_S3_baseline() in Phase 0 "
            "before any steered S3. S3 is a ratio against the unsteered count on the same "
            "items (spec 5.4); without it there is no S3, and defaulting the denominator to "
            "S3_N would silently rescale every sanity number in the run.")

    cap_base = int(ctx.base[CAP_BASE_KEY])
    out = _s3_pass(int(layer), float(alpha), _vec_for(layer, alpha))
    correct = int(out["correct"])
    n = int(out["n"])
    acc_ci_low, acc_ci_high = wilson_interval(correct, n)

    row = dict(
        s3_correct=correct,
        # max(cap_base, 1) per spec 5.4: a model that scored zero unsteered has no capability to
        # retain, and the ratio would be a division by zero rather than a sanity term.
        s3=correct / max(cap_base, 1),
        s3_margin=out["margin"],
        s3_margin_se=out["margin_se"],          # defence 12: SE across the 57 items
        s3_n=n,
        s3_acc=correct / n,
        s3_acc_ci_low=acc_ci_low,
        s3_acc_ci_high=acc_ci_high,
        s3_acc_n=n,
        s3_cap_base=cap_base,
        s3_per_item=out["per_item"],
    )
    # Membership test, not a defaulted .get: the delta is a diagnostic, so its absence is
    # acceptable, but it must be absent rather than zero.
    if S3_MARGIN_BASE_KEY in ctx.base and ctx.base[S3_MARGIN_BASE_KEY] is not None:
        row["s3_margin_base"] = ctx.base[S3_MARGIN_BASE_KEY]
        if out["margin"] is not None:
            row["s3_margin_delta"] = out["margin"] - float(ctx.base[S3_MARGIN_BASE_KEY])
    return row


# =====================================================================================
# 5.5  S2 - objective degeneracy
# =====================================================================================
# BUG 27 (2026-08-04). The judge's coherency rubric scored 23 of 25 responses at L46 alpha=3 as
# coherent when they were literally "## ## ## ##" repeated to the token limit, and a detection
# rate was then computed over that garbage and passed the usable gate. **A sanity score that a
# classifier can talk round is not a sanity score.** So S2 is mechanical, has no prompt, and is
# folded into S4 with min() rather than averaged with the judge's opinion (defence 9).
#
# It fired on 8 of 180 M1.5 cells and correctly failed 7 (karma L46 alpha=4 at 0.92 degenerate).
# It catches loop collapse, which the judge terms may not; the judge catches fluent fixation,
# which this cannot. Complementary, not redundant (spec 5.5).

_S2_NGRAM_N: int = 5              # spec 5.5: any 5-gram occurring >= 3 times
_S2_NGRAM_MAX_REPEAT: int = 3
_S2_DISTINCT_N: int = 3           # spec 5.5: distinct-3-gram ratio < 0.5
_S2_DISTINCT_MIN: float = 0.5

# v1's `len(w) < 5` guard (lab_cells.py cell 38), kept because the two spec rules are
# STRUCTURALLY BLIND below it: a text with fewer than 5 whitespace tokens has no 5-gram at all,
# and its handful of 3-grams are almost always distinct, so an empty or three-word response
# would score as perfectly healthy and S2 would read 1.0 for a cell that generated nothing.
# That is a wrong number rather than a missing one (DEBUG LOG pattern 4), which is the exact
# class this pipeline exists to be unable to reproduce.
#
# v1's other two rules (alphabetic-character fraction, single-word dominance) are deliberately
# NOT carried over: spec 5.5 replaced them, and both cases they caught - "## ## ## ##" and a
# one-word loop - are caught by the 5-gram rule anyway.
_S2_MIN_WORDS: int = 5


def _ngrams(words: Sequence[str], n: int) -> list[tuple]:
    """All n-grams of `words`, in order. Empty when the text is shorter than n."""
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def degeneracy_reason(text: str) -> str | None:
    """Which degeneracy rule fired, or None. `degenerate()` is this, as a bool.

    The returned string is `"<rule>: <detail>"`, and the rule half is a fixed label
    (`too_short`, `ngram_repeat`, `low_distinct`) so `measure_S2` can count rules without
    parsing prose that carries counts and ratios in it.

    Split on whitespace with no case-folding or punctuation stripping, as v1 did (lab_cells.py
    cell 38): normalising would merge "##" with "## " and with "##." and make the repetition
    detector weaker exactly where collapsed output lives.
    """
    words = text.split()
    if len(words) < _S2_MIN_WORDS:
        return f"too_short: under {_S2_MIN_WORDS} words ({len(words)})"

    grams = _ngrams(words, _S2_NGRAM_N)
    if grams:
        worst, count = collections.Counter(grams).most_common(1)[0]
        if count >= _S2_NGRAM_MAX_REPEAT:
            return (f"ngram_repeat: {_S2_NGRAM_N}-gram repeated {count} times: "
                    f"{' '.join(worst)[:40]!r}")

    tri = _ngrams(words, _S2_DISTINCT_N)
    if tri:
        ratio = len(set(tri)) / len(tri)
        if ratio < _S2_DISTINCT_MIN:
            return (f"low_distinct: distinct-{_S2_DISTINCT_N}-gram ratio {ratio:.2f} "
                    f"< {_S2_DISTINCT_MIN}")
    return None


def degenerate(text: str) -> bool:
    """True if the response has collapsed. Mechanical: no judge, nothing to talk round.

    Degenerate if EITHER any 5-gram occurs at least 3 times OR the distinct-3-gram ratio is
    below 0.5 (spec 5.5), plus the minimum-length guard documented above.
    """
    return degeneracy_reason(text) is not None


def measure_S2(responses: list[str]) -> dict:
    """`S2 = 1 - (degenerate responses / total responses)`.

    Pure: no model call, no judge call. Takes already-generated text, which is why the same
    function serves the cheap tier (folded into a scan row when generations happen to exist)
    and `expensive.verify_cell`, where the E5 generations are already in hand.

    An empty response list RAISES. `1 - 0/0` has no value, and returning 1.0 would report
    perfect sanity for a cell that produced nothing - which is precisely the failure mode S2
    was added to catch.
    """
    if not isinstance(responses, (list, tuple)):
        raise TypeError(f"measure_S2 expects a list of responses, got {type(responses).__name__}")
    if len(responses) == 0:
        raise ValueError(
            "measure_S2 over zero responses: S2 = 1 - degenerate/total is undefined, and "
            "returning 1.0 would score a cell that generated nothing as perfectly sane.")

    reasons = [degeneracy_reason(str(text)) for text in responses]
    bad = [r for r in reasons if r is not None]
    n = len(reasons)
    bad_n = len(bad)
    good_n = n - bad_n
    frac = bad_n / n
    bad_ci_low, bad_ci_high = wilson_interval(bad_n, n)
    good_ci_low, good_ci_high = wilson_interval(good_n, n)
    return dict(
        s2=1.0 - frac,
        s2_count=good_n,
        s2_ci_low=good_ci_low,
        s2_ci_high=good_ci_high,
        degenerate_frac=frac,
        degenerate_count=bad_n,
        degenerate_frac_ci_low=bad_ci_low,
        degenerate_frac_ci_high=bad_ci_high,
        s2_n=n,
        # Which rule fired, counted by its fixed label. A cell that is 100% degenerate because
        # every response is empty is a different failure from one that is 100% degenerate
        # because every response loops, and the D4 reading of spec 9.2 turns on that
        # distinction (`empty` and `degenerate` are separate failure modes there).
        s2_reasons=dict(collections.Counter(r.split(":", 1)[0] for r in bad)),
    )


# =====================================================================================
# The Phase 1 scan row
# =====================================================================================

def scan_cell(layer: int, r: float, responses: list[str] | None = None) -> dict:
    """One `scan.jsonl` row for `(layer, r)`: E6 + D3 + S3 + norms (CONTRACT section 4).

    Alpha is resolved through `config.alpha_for`, the pipeline's single dose map. An
    `Unreachable` cell is recorded with `reachable: false` and every metric null - **never
    clamped to ALPHA_CEIL**. A clamped alpha would be measured at one dose and labelled with
    another, so the row would be a wrong number rather than a missing one (spec 8 Phase 1).

    `s2` is None unless `responses` are supplied. Phase 1 generates nothing - this module is
    forbidden from generating - so at scan time there is no text for the n-gram detector to
    read. The key is present so downstream readers can index it uniformly, and its value is
    null rather than 1.0 because "no generations were scored" and "no generation was degenerate"
    are different claims and only one of them is true here. `expensive.verify_cell` computes the
    real S2 from the E5 generations.

    Null metrics on an unreachable row mean NOT MEASURED. Nothing downstream may read them as
    zero; Phase 2 filters on `reachable` first.
    """
    ctx = _run()
    layer = int(layer)
    r = float(r)
    t0 = time.time()

    # Norms are read before anything is measured, so an unreachable row still carries the ||v||
    # and ||h|| that made it unreachable. Hard index: a cell on an uncalibrated layer is a bug.
    norms = ctx.norms[layer]
    vec_norm = float(norms["vec_norm"])
    resid_norm = float(norms["resid_norm"])

    base = dict(phase="SCAN", layer=layer, r=r, vec_norm=vec_norm, resid_norm=resid_norm,
                vec_fingerprint=norms["vec_fingerprint"])

    try:
        alpha = float(config.alpha_for(layer, r))
    except config.Unreachable as exc:
        return _stamp(dict(
            base,
            alpha=None,
            reachable=False,
            # The alpha the cell WOULD have needed, so the dose map can be read back from the
            # scan file without recomputing it, and so "unreachable" is a number and not a mood.
            alpha_needed=exc.alpha,
            alpha_ceil=exc.ceiling,
            reach=None, reach_se=None, reach_ci_low=None, reach_ci_high=None, reach_n=None,
            e6_mass_median=None, e6_rank_med=None,
            d3=None, d3_se=None, d3_rate=None, d3_rate_ci_low=None,
            d3_rate_ci_high=None, d3_rate_n=None, d3_rank_med=None,
            s2=None, s3=None, s3_margin=None, s3_correct=None, s3_n=None,
            secs=round(time.time() - t0, 3),
        ))

    e6 = measure_E6(layer, alpha)
    d3 = measure_D3(layer, alpha)
    s3 = measure_S3(layer, alpha)
    s2 = measure_S2(responses) if responses else None

    return _stamp(dict(
        base,
        alpha=alpha,
        reachable=True,
        reach=e6["reach"],
        reach_se=e6["reach_se"],
        reach_ci_low=e6["reach_ci_low"],
        reach_ci_high=e6["reach_ci_high"],
        reach_n=e6["reach_n"],
        e6_mass_median=e6["e6_mass_median"],
        e6_rank_med=e6["e6_rank_med"],
        d3=d3["d3"],
        d3_se=d3["d3_se"],
        d3_rate=d3["d3_rate"],
        d3_rate_ci_low=d3["d3_rate_ci_low"],
        d3_rate_ci_high=d3["d3_rate_ci_high"],
        d3_rate_n=d3["d3_rate_n"],
        d3_rank_med=d3["d3_rank_med"],
        s2=None if s2 is None else s2["s2"],
        s2_n=None if s2 is None else s2["s2_n"],
        s2_ci_low=None if s2 is None else s2["s2_ci_low"],
        s2_ci_high=None if s2 is None else s2["s2_ci_high"],
        s3=s3["s3"],
        s3_margin=s3["s3_margin"],
        s3_correct=s3["s3_correct"],
        s3_n=s3["s3_n"],
        s3_acc=s3["s3_acc"],
        s3_acc_ci_low=s3["s3_acc_ci_low"],
        s3_acc_ci_high=s3["s3_acc_ci_high"],
        # Per-unit timing feeds monitor.RunStatus's rate model, which costs each phase
        # separately because a scan cell (~2 s) and a verification cell (~50 s) are three orders
        # of magnitude apart and one blended rate would give a badly wrong ETA (spec 14.5).
        secs=round(time.time() - t0, 3),
    ))
