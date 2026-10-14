"""m5.analysis.replay - the adaptive search replayed on M3's measured cells (Phase 0, item 4).

M5 gives up M3's "measure every cell" for an adaptive search over a modelled surface (plan §5.2,
decision 4). The mitigation the plan promises first is this replay: run the search offline on the
462 cells M3 measured in full, and check that it recovers each concept's R* and window. The
go/no-go of Phase 0 reads it: the replayed search recovers R* within its interval on all three
concepts, or Phase 2 keeps a fuller grid.

**`ArchiveMeasurer`** is the search's `Measurer` (`m5.search.measurer`: `writes_rows`,
`screen(cell, pass_)`, `gate(cell, n, split)`, `effect(cell, n, split)`, `integrity(cell, n,
split)`, `measure(code, cell, n, split, variant=)`, returning `m5.results` Readout / Measured),
answering from the archive by resampling the archived per-trial outcomes of the measured cell a
request maps to. Like the simulator's truth sampler it writes no rows itself (`writes_rows =
False`): the search records what it was told.

- **The dose axis is the archive's own.** M3 swept fractions of each layer's `dose_max` (its
  coherence boundary); M5's search moves in rho (dose over the random-vector integrity boundary).
  The two are not interconvertible without a model profile M3 never measured, so the replay does
  not pretend: in a replay a cell's `rho` *is* the M3 dose fraction (rho = 1 is the layer's
  dose_max, where M3 found its concept vector broke the model, as rho = 1 is where random vectors
  do in M5), and the search runs with RHO_GRID = `summary.dose_fractions` and a layer band and
  stride that cover exactly the measured layers (`replay_config`). A dose that is not one of
  those fractions raises `OffGrid`: nothing is interpolated between measured doses, nothing
  extrapolated beyond them. A grid point M3 did not measure (a layer without cells, a missing
  fraction) and the rho = 0 null are answered missing, with the reason.
- **What it answers, and from what.** `gate`: `d2` (forced naming, generated and judged; lower is
  better for an operating point) from the identify trials -- the judge's strict `matches` -- and,
  when GATE asks for noticing, `d8` (briefed self-report) from the unprefilled self_report trials
  (claims YES; one framing, three per cell). `effect`: `e8` (judged concept presence) from the
  effect channel's influence verdicts: k counts prompts with influence >= 1 (M3's rubric:
  present, at least faintly), `values` carry influence / 10; the influence judge already scored
  each reply against the model's own unsteered reply, so it is an excess by construction.
  `integrity`, with `integrity="m3_flags"`: `s1` (coherence) from the concept-blind coherence
  verdicts (>= 7 is "coherent" on M3's rubric) and `s2` (degeneracy) from the run's own
  mechanical flag, both over the same dealt effect and explain rows -- M5's integrity rules then
  act on M3's evidence. With `integrity="boundary"` both are missing: integrity is M3's own
  boundary search, below which every archived cell lies, and the search learns nothing more.
- **What it cannot answer** is returned missing with the reason, never filled: every screen-tier
  readout (M3 recorded no forward-pass readouts), `s5` (the null arm carries no capability
  check, so the paired panel cannot be formed), the null arm of every judged channel (M3 never
  judged it), and the confirm split (the archive has no held-out prompts).
- **Resampling.** `mode="without_replacement"` deals each cell's archived trials out in a seeded
  random order, each at most once; a request beyond what is left gets what is left, and an
  exhausted cell answers missing -- the search cannot buy more information than M3 paid for.
  `mode="with_replacement"` draws with replacement, so n is unbounded and repeated trials
  overstate the information; it separates the allocation question from the archive's sample
  sizes, and every row says which mode ran. A trial whose verdict failed is dealt like any other
  and counted in `excluded`, so a replayed visit loses judgments the way a real one would. Every
  request is recorded (`requests`).

**`replay_search`** imports `m5.search` lazily, runs `run_screen` then `AdaptiveSearch` with
GATE=naming against one measurer, and compares in two ways:

1. **The search's own model, against the full grid** (`search_vs_survey`): `m5.search.survey`
   measures every archived cell once at its full archived n through a fresh measurer of the same
   settings, and `m5.search.summarize.summaries` turns each into R*(Q). Same model, same
   measurer, same integrity evidence: the two differ only in which cells and trials each saw.
   This is the comparison the go/no-go reads.
2. **One estimator on both** (`refit_visited_vs_full`): the trials the search drew are fitted
   with exactly the estimator of `m5.analysis.refit` (chain-coupled d2, the presence rate on the
   excess scale, M3's boundary as integrity) and set against the same fit of every archived
   trial. Independent of the search's summary code; a second reading of the allocation.

Each reports R*(Q) of both sides, whether the search's median lies inside the full grid's
interval (`recovered`), the best layer of each, the windows (layers whose R_l upper bound reaches
the best layer's lower bound) and their overlap; and the trials the search spent against the
trials M3 spent. When `m5.search` is not importable, `replay_search` raises `SearchUnavailable`.
"""

from __future__ import annotations

import importlib
import math
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m5.analysis.m3_archive import ArchiveCell, ArchiveTrial, M3Archive
from m5.analysis.refit import (
    INFLUENCE_SCALE,
    PRESENCE_CUT,
    ConceptFit,
    EffectObs,
    LayerData,
    Point,
    RefitError,
    _u_s,
    fit_concept,
    fit_layers,
    null_floor_control,
)
from m5.backend.base import ModelInfo, derive_seed
from m5.intervention import Cell, canonical_dose
from m5.results import Measured, Readout
from m5.stats import wilson

MODES = ("without_replacement", "with_replacement")
INTEGRITY_SOURCES = ("m3_flags", "boundary")

# Screen readouts of each pass (plan §5.2; m5.search.measurer). M3 recorded none of them; each
# is answered missing.
SCREEN_PASS_CODES = {"A": ("e9", "s6", "s2", "d3", "d5", "q2"),
                     "B": ("e9", "s6", "s2", "d3", "d5", "q2", "d6", "d7", "d9")}
GATE_CODES = {"naming": ("d2",), "noticing": ("d8",), "both": ("d8", "d2")}

# M3's coherence rubric (m3/battery.py _FLUENCY_CUTS): 7-10 is "coherent". A definition of the
# archive's scale, not a tunable; presence (influence >= 1) is `refit.PRESENCE_CUT`.
M3_COHERENT = 7.0

NO_SCREEN = ("m3_archive: M3 recorded no forward-pass readouts; only generated, judged batteries "
             "are on disk")
NO_NULL = "m3_archive: the M3 null arm was never judged (spec §9), so this channel has no null"
NO_CONFIRM = "m3_archive: the archive has no held-out prompts; a confirm split cannot be replayed"
NO_S5 = ("m3_archive: the null arm carries no capability check, so the paired panel s5 cannot be "
         "formed")
BOUNDARY_INTEGRITY = ("m3_archive: integrity is M3's own boundary search; every archived cell lies "
                      "below the layer's first failing dose (integrity='boundary')")


class OffGrid(ValueError):
    """A request off the archive's dose grid: refused, never interpolated or extrapolated."""


class SearchUnavailable(ImportError):
    """`m5.search` (run_screen, AdaptiveSearch) is not importable in this checkout."""


class ReplayError(RuntimeError):
    """The replay could not run as configured (the message says what the search asked for)."""


# =====================================================================================
# The measurer
# =====================================================================================

@dataclass
class ReplayRequest:
    """One request the search made and what the archive answered (no text)."""

    index: int
    method: str
    code: str
    cell: str | None
    layer: int | None
    fraction: float | None
    split: str
    n_requested: int
    n_returned: int
    k: int | None
    excluded: int
    recycled: int
    missing: str | None
    transcript_ids: list[str]
    effect_values: list[tuple[str, float]] = field(default_factory=list)  # (prompt, influence/10)

    def to_row(self, mode: str) -> dict[str, Any]:
        return {"kind": "replay_request", "index": self.index, "method_called": self.method,
                "code": self.code, "cell": self.cell, "layer": self.layer,
                "fraction": self.fraction, "split": self.split, "n_requested": self.n_requested,
                "n_returned": self.n_returned, "k": self.k, "excluded": self.excluded,
                "recycled": self.recycled, "missing": self.missing, "mode": mode,
                "transcript_ids": self.transcript_ids}


class ArchiveMeasurer:
    """The search's Measurer, answering from an M3 archive (module docstring)."""

    writes_rows = False
    source = "m3_archive"

    def __init__(self, archive: M3Archive, cfg: Any, *, mode: str, integrity: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if integrity not in INTEGRITY_SOURCES:
            raise ValueError(f"integrity must be one of {INTEGRITY_SOURCES}, got {integrity!r}")
        if not archive.dose_fractions:
            raise OffGrid(f"{archive.run_id}: the archive has no dose grid (an explicit-CELLS run or "
                          "a run without boundaries); there is nothing to replay on")
        self.archive = archive
        self.cfg = cfg
        self.mode = mode
        self.integrity_source = integrity
        self.conf = float(cfg["CONFIDENCE"])
        self.gate_mode = cfg["GATE"]
        self.fractions = tuple(float(f) for f in archive.dose_fractions)
        self.grid: dict[tuple[int, float], ArchiveCell] = {
            (c.layer, c.fraction): c for c in archive.cells if c.fraction is not None}
        self.requests: list[ReplayRequest] = []
        self._order: dict[tuple, list[ArchiveTrial]] = {}
        self._next: dict[tuple, int] = {}
        self._base = int(cfg["SEED"])

    # ---------------------------------------------------------------- the grid

    def fraction_of(self, rho: float) -> float | None:
        r = canonical_dose(float(rho))
        for f in self.fractions:
            if abs(canonical_dose(f) - r) <= 1e-9 + 1e-6 * abs(f):
                return f
        return None

    def resolve(self, cell: Cell) -> ArchiveCell | str:
        """The measured cell a request maps to; or why the grid point holds no measurement (a
        string: answered missing); or `OffGrid` for a request no replay may answer."""
        a = self.archive
        if cell.vector_id != a.vector_id:
            raise OffGrid(f"vector {cell.vector_id!r} is not this archive's ({a.vector_id!r})")
        if cell.schedule != "single" or len(cell.layers) != 1:
            raise OffGrid(f"{cell.cell_id()}: M3 measured single-layer cells only")
        if cell.op != "add" or cell.positions != "from_marker":
            raise OffGrid(f"{cell.cell_id()}: M3 measured addition from the trial marker only")
        if cell.rho == 0:
            return NO_NULL
        f = self.fraction_of(cell.rho)
        if f is None:
            raise OffGrid(f"{cell.cell_id()}: rho {cell.rho} is not one of the archive's dose "
                          f"fractions {self.fractions}; in a replay rho is the M3 fraction of "
                          "dose_max, and nothing between or beyond them is interpolated")
        hit = self.grid.get((cell.layer, f))
        if hit is None:
            return f"m3_archive: M3 did not measure layer {cell.layer} at fraction {f}"
        return hit

    # ---------------------------------------------------------------- dealing trials

    def _pool(self, c: ArchiveCell, pool: str) -> list[ArchiveTrial]:
        if pool == "integrity":
            return [t for t in self.archive.trials(c) if t.channel in ("effect", "explain")]
        return self.archive.trials(c, pool)

    def pool_size(self, c: ArchiveCell, pool: str) -> int:
        return len(self._pool(c, pool))

    def _deal(self, c: ArchiveCell, pool: str, n: int) -> tuple[list[ArchiveTrial], int, str | None]:
        """(trials, recycled, missing reason) for a request of n units from one cell's pool."""
        trials = self._pool(c, pool)
        if not trials:
            return [], 0, f"m3_archive: no {pool} trials were recorded at {c.cell_id}"
        key = (c.key, pool)
        if self.mode == "with_replacement":
            rng = np.random.default_rng(derive_seed(self._base, "replay", self.archive.run_id, key,
                                                    len(self.requests)))
            idx = rng.integers(0, len(trials), size=max(0, int(n)))
            picked = [trials[i] for i in idx]
            return picked, max(0, len(idx) - len(set(int(i) for i in idx))), None
        if key not in self._order:
            rng = np.random.default_rng(derive_seed(self._base, "replay", self.archive.run_id, key))
            self._order[key] = [trials[i] for i in rng.permutation(len(trials))]
            self._next[key] = 0
        start = self._next[key]
        order = self._order[key]
        if start >= len(order):
            return [], 0, (f"m3_archive: all {len(order)} archived {pool} trials of {c.cell_id} "
                           "were already used (without replacement)")
        picked = order[start:start + max(0, int(n))]
        self._next[key] = start + len(picked)
        return picked, 0, None

    def _record(self, method: str, code: str, cell: Cell | None, c: ArchiveCell | None, split: str,
                n: int, m: Measured, ids: Sequence[str], recycled: int,
                effect_values: Sequence[tuple[str, float]] = ()) -> None:
        self.requests.append(ReplayRequest(
            index=len(self.requests), method=method, code=code,
            cell=None if cell is None else cell.cell_id(),
            layer=None if c is None else c.layer, fraction=None if c is None else c.fraction,
            split=split, n_requested=int(n), n_returned=int(m.n), k=m.k,
            excluded=int(m.excluded), recycled=recycled, missing=m.missing,
            transcript_ids=list(ids), effect_values=list(effect_values)))

    def _missing(self, method: str, code: str, cell: Cell | None, split: str, n: int,
                 reason: str) -> Measured:
        m = Measured(code, "null" if cell is None else cell.cell_id(), split, 0, missing=reason[:200])
        self._record(method, code, cell, None, split, n, m, (), 0)
        return m

    def _rate(self, method: str, code: str, cell: Cell, c: ArchiveCell, split: str, n: int,
              pool: str, outcome: Callable[[ArchiveTrial], bool | None], extra: dict[str, Any],
              dealt: tuple[list[ArchiveTrial], int, str | None] | None = None) -> Measured:
        """A rate over the trials dealt for this request (or over `dealt`, when one batch of
        generations is read several ways, as s1 and s2 are)."""
        trials, recycled, why = dealt if dealt is not None else self._deal(c, pool, n)
        valid = [(t, outcome(t)) for t in trials]
        valid = [(t, o) for t, o in valid if o is not None]
        excluded = len(trials) - len(valid)
        ids = [t.transcript_id for t, _ in valid]
        if not valid:
            reason = why or (f"m3_archive: none of the {len(trials)} trials dealt at {c.cell_id} "
                             "has a verdict")
            m = Measured(code, cell.cell_id(), split, 0, missing=reason, excluded=excluded,
                         extra=dict(extra, source="m3_archive", mode=self.mode))
        else:
            k = sum(1 for _, o in valid if o)
            lo, hi = wilson(k, len(valid), self.conf)
            m = Measured(code, cell.cell_id(), split, len(valid), k=k, lo=lo, hi=hi,
                         unit_ids=ids, excluded=excluded,
                         extra=dict(extra, source="m3_archive", mode=self.mode, recycled=recycled,
                                    short_by=max(0, int(n) - len(trials))))
        self._record(method, code, cell, c, split, n, m, ids, recycled)
        return m

    # ---------------------------------------------------------------- the protocol

    def screen(self, cell: Cell, pass_: Any) -> dict[str, Readout]:
        """Every screen readout of the pass, missing: M3 recorded none. Never raises -- a missing
        readout fabricates nothing, whatever the cell."""
        key = str(pass_).upper().replace("PASS", "").strip(" _")
        key = {"1": "A", "2": "B"}.get(key, key)
        codes = SCREEN_PASS_CODES.get(key, SCREEN_PASS_CODES["B"])
        return {code: Readout(code, cell.cell_id(), None, 0, missing=NO_SCREEN) for code in codes}

    def _gate_one(self, code: str, cell: Cell, n: int, split: str) -> Measured:
        if split == "confirm":
            return self._missing("gate", code, cell, split, n, NO_CONFIRM)
        c = self.resolve(cell)
        if isinstance(c, str):
            return self._missing("gate", code, cell, split, n, c)
        if code == "d2":
            return self._rate("gate", code, cell, c, split, n, "identify", lambda t: t.matches,
                              {"estimand": "M3 forced identification: judge strict matches"})
        return self._rate("gate", code, cell, c, split, n, "self_report",
                          lambda t: None if t.claims is None else t.claims == "YES",
                          {"estimand": "M3 unprefilled self_report: claims YES"})

    def gate(self, cell: Cell | None, n: int, split: str) -> dict[str, Measured]:
        """The gating channel(s) per GATE: d2 from identify, d8 from self_report."""
        codes = GATE_CODES[self.gate_mode]
        if cell is None:
            return {code: self._missing("gate", code, None, split, n, NO_NULL) for code in codes}
        return {code: self._gate_one(code, cell, n, split) for code in codes}

    def effect(self, cell: Cell | None, n: int, split: str) -> Measured:
        """e8 from the influence verdicts: k = prompts with influence >= 1, values = influence/10."""
        if cell is None:
            return self._missing("effect", "e8", None, split, n, NO_NULL)
        if split == "confirm":
            return self._missing("effect", "e8", cell, split, n, NO_CONFIRM)
        c = self.resolve(cell)
        if isinstance(c, str):
            return self._missing("effect", "e8", cell, split, n, c)
        trials, recycled, why = self._deal(c, "effect", n)
        valid = [t for t in trials if t.influence is not None and not t.empty]
        excluded = len(trials) - len(valid)
        extra = {"source": "m3_archive", "mode": self.mode,
                 "estimand": "M3 influence 0-10 against the unsteered reply; k: influence >= 1",
                 "presence_cut": PRESENCE_CUT, "scale": INFLUENCE_SCALE}
        if not valid:
            m = Measured("e8", cell.cell_id(), split, 0, excluded=excluded, extra=extra,
                         missing=why or f"m3_archive: no influence verdict among {len(trials)} trials")
            self._record("effect", "e8", cell, c, split, n, m, (), recycled)
            return m
        k = sum(1 for t in valid if t.influence >= PRESENCE_CUT)  # type: ignore[operator]
        lo, hi = wilson(k, len(valid), self.conf)
        values = [float(t.influence) / INFLUENCE_SCALE for t in valid]  # type: ignore[arg-type]
        m = Measured("e8", cell.cell_id(), split, len(valid), k=k, lo=lo, hi=hi, values=values,
                     unit_ids=[t.transcript_id for t in valid], excluded=excluded,
                     extra=dict(extra, recycled=recycled, short_by=max(0, int(n) - len(trials))))
        self._record("effect", "e8", cell, c, split, n, m, [t.transcript_id for t in valid],
                     recycled, [(t.unit, v) for t, v in zip(valid, values)])
        return m

    def integrity(self, cell: Cell | None, n: int, split: str) -> dict[str, Measured]:
        """s1 (coherence >= 7) and s2 (mechanical degeneracy), read off one dealt batch of effect
        and explain rows (integrity="m3_flags"), or missing (integrity="boundary"). No s5."""
        codes = ("s1", "s2")
        if cell is None:
            return {code: self._missing("integrity", code, None, split, n, NO_NULL) for code in codes}
        if split == "confirm":
            return {code: self._missing("integrity", code, cell, split, n, NO_CONFIRM)
                    for code in codes}
        c = self.resolve(cell)
        if isinstance(c, str):
            return {code: self._missing("integrity", code, cell, split, n, c) for code in codes}
        if self.integrity_source == "boundary":
            return {code: self._missing("integrity", code, cell, split, n, BOUNDARY_INTEGRITY)
                    for code in codes}
        dealt = self._deal(c, "integrity", n)   # one batch of generations, read two ways
        return {
            "s1": self._rate("integrity", "s1", cell, c, split, n, "integrity",
                             lambda t: None if t.coherence is None or t.empty
                             else t.coherence >= M3_COHERENT,
                             {"estimand": "M3 concept-blind coherence >= 7 (rubric 7-10 coherent)"},
                             dealt=dealt),
            "s2": self._rate("integrity", "s2", cell, c, split, n, "integrity",
                             lambda t: t.degenerate,
                             {"estimand": "the run's own M3 mechanical degeneracy flag"},
                             dealt=dealt),
        }

    def measure(self, code: str, cell: Cell | None, n: int, split: str, *,
                variant: str = "original") -> Measured:
        """Any one measure at a fixed n (the confirm tier and audit mode use this): d2 and d8
        from the gate, e8 from the effect, s1 and s2 from integrity; the rest missing."""
        if variant != "original":
            return self._missing("measure", code, cell, split, n,
                                 f"m3_archive: M3 ran the original prompts only, not {variant!r}")
        if code in ("d2", "d8"):
            if cell is None:
                return self._missing("measure", code, None, split, n, NO_NULL)
            return self._gate_one(code, cell, n, split)
        if code == "e8":
            return self.effect(cell, n, split)
        if code in ("s1", "s2"):
            return self.integrity(cell, n, split)[code]
        if code == "s5":
            return self._missing("measure", code, cell, split, n, NO_S5)
        return self._missing("measure", code, cell, split, n,
                             f"m3_archive: M3 has no measure behind {code}")

    # ---------------------------------------------------------------- what was drawn

    def visited_data(self) -> dict[int, LayerData]:
        """The trials the search drew, as refit data on the fraction axis (d2 points summed per
        cell; every influence verdict drawn)."""
        a = self.archive
        pts: dict[tuple[int, float], list[int]] = {}
        data: dict[int, LayerData] = {}
        for r in self.requests:
            if r.layer is None or r.fraction is None or r.missing is not None:
                continue
            ld = data.setdefault(r.layer, LayerData(r.layer, u_s=_u_s(a, r.layer, "fraction"),
                                                    dose_max=a.boundaries[r.layer].dose_max
                                                    if r.layer in a.boundaries else None))
            u = math.log(r.fraction)
            if r.code == "d2" and r.k is not None:
                acc = pts.setdefault((r.layer, r.fraction), [0, 0])
                acc[0] += r.k
                acc[1] += r.n_returned
            if r.code == "e8":
                for prompt, y in r.effect_values:
                    ld.effect.append(EffectObs(r.layer, u, y, prompt, r.cell or ""))
        for (layer, f), (k, n) in sorted(pts.items()):
            data[layer].points.append(Point(layer, math.log(f), k, n, f"L{layer}@f{f}"))
        return dict(sorted(data.items()))

    def spend(self) -> dict[str, Any]:
        d2 = [r for r in self.requests if r.code == "d2" and r.missing is None]
        e8 = [r for r in self.requests if r.code == "e8" and r.missing is None]
        archived_d2 = sum(c.ident_n + c.ident_failed for c in self.archive.cells
                          if c.fraction is not None)
        archived_e8 = sum(len(self.archive.trials(c, "effect")) for c in self.archive.cells
                          if c.fraction is not None)
        return {"d2_trials_drawn": sum(r.n_returned + r.excluded for r in d2),
                "d2_trials_archived": archived_d2,
                "e8_prompts_drawn": sum(r.n_returned + r.excluded for r in e8),
                "e8_prompts_archived": archived_e8,
                "cells_visited": len({r.cell for r in d2 + e8}),
                "cells_archived": len(self.grid),
                "requests": len(self.requests),
                "requests_missing": sum(1 for r in self.requests if r.missing is not None)}


# =====================================================================================
# The search, run on the archive
# =====================================================================================

@dataclass(frozen=True)
class SearchAPI:
    """The entry points of `m5.search` the replay uses. `survey` and `summaries` may be None (the
    search-vs-survey comparison is then skipped and says so)."""

    run_screen: Callable[..., Any]
    AdaptiveSearch: Callable[..., Any]
    source: str
    survey: Callable[..., Any] | None = None
    summaries: Callable[..., Any] | None = None
    layer_table: Callable[..., Any] | None = None


def load_search_api() -> SearchAPI:
    """`run_screen`, `AdaptiveSearch`, `survey` and `summarize.summaries` from `m5.search`,
    imported lazily. Raises `SearchUnavailable` when the first two are not there."""
    try:
        pkg = importlib.import_module("m5.search")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("m5.search"):
            raise SearchUnavailable(
                "the replay needs m5.search (run_screen and AdaptiveSearch, implementation guide "
                "§3.7), which is not installed in this checkout yet. The archive, the measurer "
                "and the full-grid refit work without it; re-run `--what replay` once m5.search "
                "exists.") from exc
        raise
    run_screen = getattr(pkg, "run_screen", None)
    adaptive = getattr(pkg, "AdaptiveSearch", None)
    if run_screen is None or adaptive is None:
        try:
            run_screen = run_screen or importlib.import_module("m5.search.screen").run_screen
            adaptive = adaptive or importlib.import_module("m5.search.adaptive").AdaptiveSearch
        except (ModuleNotFoundError, AttributeError) as exc:
            raise SearchUnavailable(f"m5.search exists but does not provide run_screen and "
                                    f"AdaptiveSearch where documented: {exc}") from exc
    # `survey` lives in its own module (`from m5.search.survey import survey`): the package does
    # not re-export it, and once the submodule is loaded `m5.search.survey` names the module.
    survey = _function_from("m5.search.survey", "survey")
    return SearchAPI(run_screen=run_screen, AdaptiveSearch=adaptive, source="m5.search",
                     survey=survey, summaries=_function_from("m5.search.summarize", "summaries"),
                     layer_table=_function_from("m5.search.summarize", "layer_table"))


def _function_from(module: str, name: str) -> Any:
    """`module.name` when it is a callable that is not itself a module, else None."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and module.startswith(exc.name):
            return None
        raise
    fn = getattr(mod, name, None)
    return fn if callable(fn) and not isinstance(fn, types.ModuleType) else None


def _stride(layers: Sequence[int]) -> int:
    diffs = [b - a for a, b in zip(layers, layers[1:]) if b > a]
    if not diffs:
        return 1
    return int(np.gcd.reduce(np.array(diffs)))


def _depth(archive: M3Archive) -> int:
    return int(archive.n_layers or (max(archive.layers) + 1 if archive.layers else 0))


def replay_config(archive: M3Archive, cfg: Any) -> Any:
    """The config a replay runs under: GATE=naming; RHO_GRID = the archive's fractions (rho IS
    the fraction here); LAYER_BAND and SCREEN_STRIDE covering exactly the measured layers, in the
    depth unit of `m5.dose.layer_depth` (L / (n_layers - 1)); no pass-B refinement onto layers M3
    did not measure."""
    layers = sorted({c.layer for c in archive.cells if c.fraction is not None})
    if not layers:
        raise ReplayError(f"{archive.run_id}: no measured cell lies on the fraction grid (an "
                          "explicit-CELLS run, or cells without boundary rows); nothing to replay")
    if len(archive.dose_fractions) < 2:
        raise ReplayError(f"{archive.run_id}: the archive's grid has {len(archive.dose_fractions)} "
                          "dose; a dose search needs at least two")
    depth = max(_depth(archive), max(layers) + 1, 2)
    lo = layers[0] / (depth - 1)
    hi = layers[-1] / (depth - 1)
    if not hi > lo:                       # one measured layer: a band around it alone
        hi = min(1.0, lo + 0.5 / (depth - 1))
        lo = max(0.0, hi - 1.0 / (depth - 1))
    grid = tuple(sorted(canonical_dose(f) for f in archive.dose_fractions))
    return cfg.replace(GATE="naming", RHO_GRID=grid, LAYER_BAND=(lo, hi),
                       SCREEN_STRIDE=_stride(layers), SCREEN_REFINE_RADIUS=0,
                       SCHEDULE="single", OPERATION="add", POSITIONS="from_marker")


class _ReplayBackend:
    """Stands in for the model in a replay: it has the archive's shape and nothing else. Any call
    that would need a model raises, naming what the search asked for."""

    def __init__(self, archive: M3Archive) -> None:
        depth = max(_depth(archive), 2)
        self.info = ModelInfo(name=str(archive.model or "m3_archive"), hf_id="m3_archive",
                              revision=None, family="other", n_layers=depth, d_model=0,
                              vocab_size=0, dtype="none", bos_token_id=None, eos_token_ids=(),
                              chat_template_hash="none")

    def _refuse(self, name: str) -> Any:
        raise ReplayError(f"the archive replay has no model: the search called backend.{name}, "
                          "but a replay reaches the archive only through the Measurer")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        def call(*_: Any, **__: Any) -> Any:
            return self._refuse(name)
        return call


def replay_context(archive: M3Archive, cfg: Any, run: Any = None) -> Any:
    from m5.context import Context
    vid = archive.vector_id
    return Context(cfg=cfg, backend=_ReplayBackend(archive), run=run, profile=None,
                   concepts={vid: archive.concept}, vector_classes={vid: "concept"})


# =====================================================================================
# Comparisons
# =====================================================================================

def _window(r: Mapping[str, Any]) -> set[int] | None:
    """Layers that remain contenders: R_l upper bound at least the best layer's lower bound
    (the successive-halving keep rule, plan §5.2). None when per-layer bounds are absent."""
    if "per_layer" not in r:
        return None
    per = {int(p["layer"]): p for p in r["per_layer"]}
    best = per.get(int(r["best_layer"]))
    if best is None:
        return set()
    return {L for L, p in per.items() if p["hi"] is not None and best["lo"] is not None
            and p["hi"] >= best["lo"]}


def compare_rstar(full: Mapping[str, Any] | None, part: Mapping[str, Any] | None) -> dict[str, Any]:
    """R*(Q) of the search side against the full grid: recovered (the search's median inside the
    full grid's interval), overlap, best layers, windows."""
    out: dict[str, Any] = {"full": None, "search": None, "recovered": None,
                           "intervals_overlap": None, "same_best_layer": None}
    keys = ("median", "lo", "hi", "best_layer", "flags")
    if full is not None:
        out["full"] = {k: full.get(k) for k in keys}
    if part is not None:
        out["search"] = {k: part.get(k) for k in keys}
    if full is None or part is None or None in (full.get("lo"), full.get("hi"), part.get("median")):
        return out
    out["recovered"] = bool(full["lo"] <= part["median"] <= full["hi"])
    out["intervals_overlap"] = bool(part["lo"] <= full["hi"] and full["lo"] <= part["hi"])
    out["same_best_layer"] = int(full["best_layer"]) == int(part["best_layer"])
    p_best = {int(k): v for k, v in (full.get("p_best") or {}).items()}
    out["full_p_best_of_search_best"] = p_best.get(int(part["best_layer"]))
    wf, wp = _window(full), _window(part)
    if wf is not None and wp is not None:
        out["window_full"] = sorted(wf)
        out["window_search"] = sorted(wp)
        out["window_jaccard"] = (len(wf & wp) / len(wf | wp)) if (wf | wp) else None
        out["search_best_in_full_window"] = int(part["best_layer"]) in wf
    return out


def _rstar_at(fit: ConceptFit, q: float) -> dict[str, Any] | None:
    for r in fit.rstar or ():
        if abs(r["q"] - q) < 1e-12:
            return r
    return None


def compare_fits(full: ConceptFit, visited: ConceptFit, cfg: Any) -> dict[str, Any]:
    """`compare_rstar` on the two refits at Q, plus the per-layer threshold differences."""
    q = float(cfg["Q"])
    out = {"q": q, **compare_rstar(_rstar_at(full, q), _rstar_at(visited, q))}
    diffs = []
    for L, lf in visited.layers.items():
        ff = full.layers.get(L)
        if lf.coupled is None or ff is None or ff.coupled is None:
            continue
        tv = next((t for t in lf.coupled["thresholds"] if t["q"] == q), None)
        tf = next((t for t in ff.coupled["thresholds"] if t["q"] == q), None)
        if tv and tf and tv["median"] is not None and tf["median"] is not None:
            diffs.append({"layer": L, "full_median": tf["median"], "search_median": tv["median"],
                          "diff": tv["median"] - tf["median"],
                          "inside_full_interval": (tf["lo"] is not None and tf["hi"] is not None
                                                   and tf["lo"] <= tv["median"] <= tf["hi"])})
    out["thresholds"] = diffs
    if diffs:
        d = np.abs([x["diff"] for x in diffs])
        out["threshold_abs_diff_median"] = float(np.median(d))
        out["threshold_abs_diff_max"] = float(np.max(d))
        out["threshold_inside_share"] = float(np.mean([x["inside_full_interval"] for x in diffs]))
    return out


def _numbers(v: Any) -> Any:
    """Numbers, booleans, None and identifier strings of a summary, NaN/inf as None."""
    if isinstance(v, Mapping):
        return {str(k): _numbers(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_numbers(x) for x in v]
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if hasattr(v, "item") and not hasattr(v, "__len__"):
        return _numbers(v.item())
    return v


@dataclass
class ReplayResult:
    concept: str
    rows: list[dict[str, Any]]
    comparison: dict[str, Any]        # refit_visited_vs_full
    search_vs_survey: dict[str, Any]
    spend: dict[str, Any]
    measurer: ArchiveMeasurer
    full: ConceptFit
    visited: ConceptFit | None
    search_result: Any
    survey_result: Any
    search_api: str


def replay_search(archive: M3Archive, cfg: Any, *, mode: str, integrity: str, floor_control: str,
                  run: Any = None, full: ConceptFit | None = None,
                  search_api: SearchAPI | None = None) -> ReplayResult:
    """Run the search on the archive and compare what it recovers with the full grid (module
    docstring). `full` may be passed when the caller has already fitted the full grid on the
    fraction axis under `replay_config(archive, cfg)`. `search_api` replaces the lazily imported
    `m5.search` (tests use a stand-in). The search writes its own `search_log` / `screen` /
    `cells` rows into `run`; the survey writes none (its numbers are in the summary row).
    Raises `SearchUnavailable` without m5.search, `OffGrid` when the search asks for a dose the
    archive does not hold.
    """
    api = search_api or load_search_api()
    rcfg = replay_config(archive, cfg)
    measurer = ArchiveMeasurer(archive, rcfg, mode=mode, integrity=integrity)
    ctx = replay_context(archive, rcfg, run)
    try:
        screen = api.run_screen(ctx, measurer, archive.vector_id)
        result = api.AdaptiveSearch(ctx, measurer, archive.vector_id, screen).run()
    except OffGrid as exc:
        raise OffGrid(f"{archive.run_id}: the search asked for a dose off the archive's grid "
                      f"({exc}); the replay refuses rather than interpolate. The search must take "
                      "its doses from RHO_GRID.") from exc
    base = int(rcfg["SEED"])
    q = float(rcfg["Q"])

    # 1. the search's own model against its own full-grid survey, through the same measurer
    survey_result = None
    svs: dict[str, Any] = {"q": q, "available": False}
    if api.survey is not None and api.summaries is not None and hasattr(result, "state"):
        full_m = ArchiveMeasurer(archive, rcfg, mode="without_replacement", integrity=integrity)
        n_gate = max(full_m.pool_size(c, "identify") for c in full_m.grid.values())
        n_eff = max(max(full_m.pool_size(c, "effect") for c in full_m.grid.values()), 1)
        on_grid = sorted({c.layer for c in full_m.grid.values()})
        survey_result = api.survey(replay_context(archive, rcfg, None), full_m, archive.vector_id,
                                   n_gate=max(n_gate, 1), n_effect=n_eff, layers=on_grid,
                                   rhos=list(rcfg["RHO_GRID"]))
        s_search = api.summaries(result.state, rcfg, seed=derive_seed(base, "replay", "search"))
        s_full = api.summaries(survey_result.state, rcfg, seed=derive_seed(base, "replay", "survey"))
        if s_search.get("available") and s_full.get("available"):
            r_search, r_full = dict(s_search.get("R_star") or {}), dict(s_full.get("R_star") or {})
            if api.layer_table is not None:   # per-layer R_l bounds, for the windows
                for r, state, tag in ((r_search, result.state, "search"),
                                      (r_full, survey_result.state, "survey")):
                    table = api.layer_table(state, rcfg, seed=derive_seed(base, "replay", tag))
                    r["per_layer"] = [{"layer": int(L), **(row.get("R") or {})}
                                      for L, row in table.items()]
            svs = {"q": q, "available": True, **compare_rstar(r_full, r_search),
                   "effect_scale": _numbers(s_search.get("effect_scale")),
                   "survey_n_gate": int(n_gate), "survey_n_effect": int(n_eff)}
        else:
            svs = {"q": q, "available": False,
                   "reason": "the search's summaries had nothing to compare (no gating visits)"}
    else:
        svs["reason"] = "m5.search provides no survey / summarize.summaries to compare against"

    # 2. one estimator (m5.analysis.refit) on the trials the search drew and on every trial
    if full is None:
        try:
            full = fit_concept(archive, rcfg, unit="fraction", floor_control=floor_control,
                               with_effect=True)
        except RefitError as exc:
            raise ReplayError(f"{archive.run_id}: the full grid cannot be fitted on the fraction "
                              f"axis, so there is nothing to compare the replay with: {exc}") from exc
    data = measurer.visited_data()
    visited = None
    if any(ld.points for ld in data.values()):
        control = null_floor_control(archive) if floor_control == "null_literal" else None
        visited = fit_layers(data, rcfg, concept=archive.concept, unit="fraction", control=control,
                             floor_control=floor_control if control is not None else "none",
                             seed_tag=("analysis", "replay", archive.run_id), with_effect=True)
    comparison = (compare_fits(full, visited, rcfg) if visited is not None
                  else {"q": q, "recovered": None, "reason": "the search drew no valid d2 trial"})
    spend = measurer.spend()
    rows: list[dict[str, Any]] = [r.to_row(mode) | {"concept": archive.concept}
                                  for r in measurer.requests]
    search_spend = _numbers(getattr(result, "spend", None)) if isinstance(
        getattr(result, "spend", None), Mapping) else None
    rows.append({"kind": "replay_summary", "concept": archive.concept, "mode": mode,
                 "integrity_source": integrity, "dose_unit": "fraction", "search_api": api.source,
                 "rho_grid": list(rcfg["RHO_GRID"]), "layer_band": list(rcfg["LAYER_BAND"]),
                 "screen_stride": int(rcfg["SCREEN_STRIDE"]), **spend,
                 "search_spend": search_spend,
                 "search_vs_survey": svs, "refit_visited_vs_full": comparison,
                 "assumptions": ["in a replay rho is the M3 fraction of dose_max",
                                 "the search takes its doses from RHO_GRID",
                                 "the search reaches the subject only through the Measurer",
                                 "a rho = 0 or None cell is the null arm"]})
    return ReplayResult(concept=archive.concept, rows=rows, comparison=comparison,
                        search_vs_survey=svs, spend=spend, measurer=measurer, full=full,
                        visited=visited, search_result=result, survey_result=survey_result,
                        search_api=api.source)
