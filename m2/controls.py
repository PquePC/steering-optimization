"""m2.controls - the two hard gates and the escalation ladder (spec section 9).

A cell with high E5 and low D2 is the result the pipeline exists to find, and it is also what
two artifacts look like. This module is the difference between the two.

    9.1  random direction   -- is the suppression about the DIRECTION we injected, or would any
                               perturbation of this magnitude do the same?
    9.2  forced-ID capability -- is THIS concept unidentifiable, or is NOTHING identifiable
                               because the identification pathway is broken?
    9.3  escalation ladder  -- does no operating point exist, or is the vector simply dead?

Both 9.1 and 9.2 are HARD GATES: a winner that fails either is an artifact, and reporting it as
a result would be the spectacular false positive velocity L37 alpha=3.0 nearly produced in v1
(D2 fell 0.44 -> 0.00 between alpha=2 and alpha=3 because the model was lobotomised, and the
old sanity metric passed the cell at 0.779).

Dependency position: this module sits between `phases` and `gates` in the CONTRACT order, so it
may not import `runio`. Rows are written with a local `_append_row` matching that module's shape,
exactly as `phases.py`, `vectors.py` and `expensive.py` already do.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from contextlib import contextmanager

from . import cheap
from . import config
from . import expensive
from . import prompts
from . import vectors


# Reported to two decimals everywhere so that a control row and the verified row it is compared
# against cannot disagree on which cell they describe (spec section 3: comparison happens in r).
_R_PLACES = 4

# The z used for "within noise" on a difference of two binomial rates. 1.96 is the same 95%
# figure the v1 rig check used for its Wilson intervals, kept so the two are readable together.
_Z = 1.96


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "torch is required for m2.controls: the random-direction control draws a vector in "
            "residual space. Run Setup 2 in the driver notebook first.") from exc
    return torch


def _run() -> Any:
    """The process-global RunContext, or a loud error.

    Read through the accessor rather than `from .config import RUN`: `model.load_model` rebinds
    the attribute on the config module and a from-import would capture the pre-load placeholder
    forever. Same family as bug 23 - a stale reference returning something plausible.
    """
    ctx = getattr(config, "RUN", None)
    if ctx is None:
        raise RuntimeError(
            "m2.config.RUN is not set - call m2.model.load_model(CONFIG) and "
            "m2.driver.set_concept(name) before running any control.")
    return ctx


def _cfg() -> dict:
    ctx = getattr(config, "RUN", None)
    if ctx is not None and ctx.config:
        return ctx.config
    return config.CONFIG


def _const(key: str) -> Any:
    """Hard index into the config. A missing constant must raise, never become a default.

    DEBUG LOG pattern 4: silent bugs cluster around anything with a default, and every value
    read here is a gate threshold - a defaulted one would decide whether a result stands.
    """
    cfg = _cfg()
    if key not in cfg:
        raise KeyError(f"{key} missing from CONFIG - m2.config.CONSTANTS defines it (spec 11)")
    return cfg[key]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp(row: dict) -> dict:
    ctx = _run()
    out = dict(row)
    out.setdefault("concept", ctx.concept)
    out.setdefault("config_hash", _cfg().get("config_hash"))
    out.setdefault("ts", _ts())
    return out


def _path(name: str) -> Path:
    ctx = _run()
    if ctx.run_dir is None:
        raise RuntimeError(
            "RUN.run_dir is not set - call m2.driver.set_concept(name) before any control.")
    return Path(ctx.run_dir) / name


def _append_row(name: str, row: dict) -> Path:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_stamp(row), ensure_ascii=False, default=str) + "\n")
    return path


def _read_rows(name: str) -> list[dict]:
    path = _path(name)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _diff_se(p1: float, n1: int, p2: float, n2: int) -> float:
    """Standard error of a difference of two independent binomial rates."""
    if n1 <= 0 or n2 <= 0:
        return float("inf")
    return math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)


# =====================================================================================
# 9.1 - random-direction control
# =====================================================================================

def random_unit_vector(layer: int, seed: int) -> Any:
    """A reproducible random direction in residual space, scaled to ||v_layer||.

    Scaled to the concept vector's own norm so that the dose is matched BY CONSTRUCTION rather
    than by a second calculation that could drift: with ||v_rand|| == ||v_L||, the alpha that
    delivers dose r is literally the same number. `random_direction_control` still recomputes
    alpha from the random vector's measured norm and asserts the two agree, because "matched on
    r, not alpha" (spec 9.1) is the whole validity of this control and an assertion is cheaper
    than a retraction.

    Seeded per draw so the control is reproducible: a gate whose verdict cannot be re-derived
    is not a gate.
    """
    torch = _require_torch()
    ctx = _run()
    if layer not in ctx.vecs:
        raise KeyError(f"no concept vector at layer {layer} - run phase0_calibrate first")
    ref = ctx.vecs[layer]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(ref.shape, generator=gen, dtype=torch.float32)
    unit = raw / (raw.norm() + 1e-12)
    scaled = unit * float(ref.float().norm())
    return scaled.to(device=ref.device, dtype=ref.dtype)


def random_direction_control(layer: int, r: float, seeds: Sequence[int] | None = None,
                             *, n_d2: int | None = None,
                             concept_row: dict | None = None) -> dict:
    """Spec 9.1. Inject a random direction at the SAME DOSE and re-run forced identification.

    The question, restated: a cell with high E5 and low D2 admits two readings.
      (a) the concept is present but not accessible to identification  -- the result
      (b) this cell degrades identification for ANY perturbation       -- an artifact
    E5 cannot tell them apart. This can.

    MANDATORY at relative depth >= 0.72. Macar places report-gate features at L45-61 and
    effectiveness peaks at L46, both inside that band, and carrier features there are documented
    to "detect perturbations monotonically along diverse directions" - so (b) is a mechanism
    someone has already observed, not a hypothetical.

    Two readouts, and the second is the one that discriminates:

      d2_random   Judge D2 scored against the REAL concept's word while a random direction is
                  injected. Expected near zero simply because the concept was never injected, so
                  a LOW value here is uninformative on its own - it is reported because a
                  non-zero value would mean Judge D2 accepts almost anything, which is a judge
                  false-positive rate and a finding about the instrument.

      d4_random   the failure-mode distribution under the random direction. THIS is the
                  discriminator. If the model still answers with a specific wrong concept
                  (`wrong_concept`) or hedges (`vague`), the identification pathway survives a
                  perturbation of this magnitude and the concept cell's suppression is about
                  direction. If it collapses into `degenerate` / `saturated` / `empty` at the
                  same rate the concept cell does, the magnitude is doing the work and the
                  result is reading (b).

    The spec's gate sentence - "if random-direction D2 suppression is within noise of the
    concept cell's, reject" - is evaluated literally as well, and both verdicts are returned.
    Where they disagree, `verdict` follows the damage-mode comparison and `notes` says so; a
    control that quietly picks one of two readings is worse than one that shows its working.
    """
    ctx = _run()
    seeds = list(seeds) if seeds is not None else list(range(int(_const("N_RANDOM_SEEDS"))))
    n_d2 = int(n_d2 if n_d2 is not None else _const("N_D2"))
    dose = round(float(r), _R_PLACES)

    alpha = config.alpha_for(layer, dose)

    per_seed: list[dict] = []
    for seed in seeds:
        vec = random_unit_vector(layer, seed)
        # "Matched on r, not alpha" (spec 9.1): a random vector has a different norm, so
        # matching alpha would compare different doses. Assert the match rather than trust it.
        rand_norm = float(vec.float().norm())
        resid_norm = float(ctx.norms[layer]["resid_norm"])
        alpha_check = vectors.alpha_from_norms(dose, rand_norm, resid_norm)
        if not math.isclose(alpha_check, alpha, rel_tol=1e-6, abs_tol=1e-9):
            raise AssertionError(
                f"dose mismatch at L{layer} r={dose}: concept alpha {alpha:.6f} vs random-vector "
                f"alpha {alpha_check:.6f}. The control would compare different doses, which is "
                f"exactly what spec 9.1 forbids.")

        with expensive.phase_scope("CONTROL_RANDOM"):
            row = expensive.measure_D2(layer, alpha, n_d2, r=dose, vec=vec)
        row = dict(row)
        row.update(seed=int(seed), vec_norm=rand_norm,
                   vec_fingerprint=vectors.vec_fingerprint(vec))
        per_seed.append(row)
        _append_row("controls.jsonl", dict(control="random_direction", layer=int(layer),
                                           r=dose, alpha=float(alpha), **row))

    # Aggregate the actual surviving counts, not the per-seed point estimates. Judge errors can
    # make the denominators differ, and an interval over a claimed 75 trials must really contain
    # 75 observed verdicts rather than three equally weighted rates of 24, 25 and 25.
    n_scored = sum(int(row["n_d2"]) for row in per_seed)
    n_identified = sum(int(row["d2_identified"]) for row in per_seed)
    d2_random = n_identified / n_scored if n_scored else None
    d2_random_ci = (cheap.wilson_interval(n_identified, n_scored)
                    if n_scored else (None, None))
    n_d4 = sum(int(row["d4_n"]) for row in per_seed)
    n_damage = sum(int(row["d4_damage_count"]) for row in per_seed)
    d4_damage_random = n_damage / n_d4 if n_d4 else None
    d4_damage_random_ci = (cheap.wilson_interval(n_damage, n_d4)
                           if n_d4 else (None, None))

    concept_row = concept_row or _concept_row_for(layer, dose)
    d2_concept = float(concept_row["d2"]) if concept_row and concept_row.get("d2") is not None else None
    n_concept = int(concept_row.get("n_d2", 0)) if concept_row else 0
    d4_damage_concept = (float(concept_row["d4_damage_frac"])
                         if concept_row and concept_row.get("d4_damage_frac") is not None else None)

    # Literal reading of the spec's gate sentence.
    literal_separated = None
    gap = None
    if d2_concept is not None and d2_random is not None and n_concept and n_scored:
        gap = d2_random - d2_concept
        se = _diff_se(d2_random, n_scored, d2_concept, n_concept)
        literal_separated = bool(gap > _Z * se)

    # The discriminating reading: does the random direction break identification the same way?
    damage_separated = None
    if d4_damage_concept is not None and d4_damage_random is not None:
        # The concept cell may legitimately show SOME damage; the artifact case is the random
        # direction showing as much or more. Tolerance is deliberately generous - this gate
        # rejects, so it should only fire when the two are genuinely comparable.
        damage_separated = bool(d4_damage_random < d4_damage_concept - 0.15
                                or d4_damage_random < 0.25)

    if damage_separated is None:
        verdict = "inconclusive"
    elif damage_separated:
        verdict = "pass"
    else:
        verdict = "reject"

    notes = []
    if literal_separated is not None and damage_separated is not None \
            and literal_separated != damage_separated:
        notes.append(
            "the literal D2 comparison and the damage-mode comparison disagree; verdict follows "
            "the damage-mode reading because D2 scored against a concept that was never "
            "injected is near zero by construction and cannot discriminate")
    depth = layer / max(int(getattr(_run(), "n_layers", 0)) or 1, 1)
    if depth >= 0.72:
        notes.append(f"d(L)={depth:.2f} >= 0.72: this control is MANDATORY here (spec 9.1)")

    out = dict(control="random_direction", layer=int(layer), r=dose, alpha=float(alpha),
               seeds=[int(s) for s in seeds], n_seeds=len(seeds), n_d2_per_seed=n_d2,
               d2_random=d2_random, d2_random_n=n_scored,
               d2_random_ci_low=d2_random_ci[0], d2_random_ci_high=d2_random_ci[1],
               d2_concept=d2_concept, d2_concept_n=n_concept,
               d2_concept_ci_low=(concept_row.get("d2_ci_low") if concept_row else None),
               d2_concept_ci_high=(concept_row.get("d2_ci_high") if concept_row else None),
               d2_gap=gap,
               d4_damage_random=d4_damage_random, d4_damage_random_n=n_d4,
               d4_damage_random_ci_low=d4_damage_random_ci[0],
               d4_damage_random_ci_high=d4_damage_random_ci[1],
               d4_damage_concept=d4_damage_concept,
               d4_damage_concept_n=(concept_row.get("d4_n") if concept_row else None),
               d4_damage_concept_ci_low=(concept_row.get("d4_damage_frac_ci_low")
                                         if concept_row else None),
               d4_damage_concept_ci_high=(concept_row.get("d4_damage_frac_ci_high")
                                          if concept_row else None),
               literal_separated=literal_separated, damage_separated=damage_separated,
               verdict=verdict, mandatory=bool(depth >= 0.72), depth=depth,
               notes=notes, per_seed=per_seed)
    _append_row("controls.jsonl", dict(control="random_direction_summary", **out))
    return out


def _concept_row_for(layer: int, r: float) -> dict | None:
    """The verified row for this cell, so the control compares against a measured D2."""
    dose = round(float(r), _R_PLACES)
    best = None
    for row in _read_rows("verified.jsonl"):
        if int(row.get("layer", -1)) != int(layer):
            continue
        if round(float(row.get("r", -1.0)), _R_PLACES) != dose:
            continue
        if row.get("d2") is None:
            continue
        best = row
    return best


# =====================================================================================
# 9.2 - forced-identification capability control
# =====================================================================================

@contextmanager
def _as_concept(name: str) -> Iterator[str]:
    """Temporarily point RUN.concept at `name`.

    Judge D2 asks whether the continuation names the CURRENT concept, so measuring whether a
    control concept is identifiable means the run context has to name it for the duration. The
    swap is scoped and restored in a finally, and every row produced inside carries the control
    concept's name, so nothing downstream has to guess which concept a transcript belongs to.

    The judge cache is namespaced by concept, so a control-concept call cannot collide with the
    target's cached rows.
    """
    ctx = _run()
    previous = ctx.concept
    ctx.concept = name
    try:
        yield name
    finally:
        ctx.concept = previous


def forced_id_capability_control(layer: int, r: float,
                                 *, control_concepts: Sequence[str] | None = None,
                                 n_d2: int | None = None,
                                 run_secondary: bool = True) -> dict:
    """Spec 9.2. D2 near zero admits two readings; this decides which one.

      (a) THIS concept is unidentifiable         -- the result
      (b) NOTHING is identifiable because the pathway is broken -- an artifact

    Velocity L37 alpha=3.0 is exactly (b): D2 fell 0.44 -> 0.00 between alpha=2 and alpha=3
    because the model was lobotomised, and the old sanity metric passed it at 0.779. Reading (b)
    as (a) produces a spectacular false result, which is why this gate is hard.

    PRIMARY METHOD - D4, at zero extra perturbation and zero extra judge calls. The failure-mode
    distribution already collected by Judge D2 IS the answer:

        wrong_concept / vague          -> (a) retrieval works, it just cannot find this concept
        degenerate / saturated / empty -> (b) damage

    SECONDARY METHOD - inject a control concept ALONE at the same layer and the same r,
    REPLACING the target vector. Same total perturbation, different direction. If control
    concepts are identifiable at these exact parameters and the target is not, the operating
    point preserves forced identification.

    NEVER stack. Injecting a control concept on top of the target doubles the perturbation and
    lobotomises by construction, so it would fail every time and mean nothing (spec 9.2). The
    `vec=` override below replaces; it does not add.

    ⚠️ THE SECONDARY METHOD IS FLAGGED FOR REVIEW (spec 9.2). `Steering Awareness` finds that
    concept vectors converge onto one shared detection direction as they propagate, with
    cos(delta_c, d_detect) rising monotonically and SD < 0.05 across 18 held-out concepts - so
    two concept vectors are close to interchangeable to whatever does the detecting, and this
    control may be guaranteed to reproduce the target's result.

    It is kept because the question here is not "is this direction special" (that is 9.1's job)
    but "is the identification pathway working at all", and for THAT question interchangeability
    is a feature: a control concept identified at matched r proves the pathway functions at these
    parameters. The control is weak in the conservative direction.

    `secondary_agrees_with_primary` is recorded on every row so the agreement rate can be read
    across concepts. If it never disagrees with the free D4 reading, it is spending ~75 judge
    calls per concept to restate a measurement already in hand and should be demoted to a
    diagnostic. Decide that on the benign arm, before the harmful one.
    """
    ctx = _run()
    dose = round(float(r), _R_PLACES)
    n_d2 = int(n_d2 if n_d2 is not None else _const("N_D2"))
    d2_max = float(_const("D2_MAX"))

    row = _concept_row_for(layer, dose)
    if row is None:
        raise RuntimeError(
            f"no verified row at L{layer} r={dose} - the primary 9.2 control reads D4 off "
            f"Judge D2's output, so the cell must be verified first (spec 5.9 requires "
            f"D2_transcripts.jsonl; gate 7 checks it landed).")

    d4 = row.get("d4")
    if d4 is None:
        raise KeyError(
            f"verified row at L{layer} r={dose} has no d4 distribution. measure_D2 must write "
            f"one - without it the primary control cannot run at all (spec 5.9).")

    primary_reading = row.get("d4_reading")
    damage_frac = float(row["d4_damage_frac"])
    retrieval_frac = float(row["d4_retrieval_frac"])
    primary_pass = primary_reading == "retrieval"

    secondary: list[dict] = []
    if run_secondary:
        names = list(control_concepts) if control_concepts is not None else list(prompts.CONTROL_CONCEPTS)
        for name in names:
            try:
                ctrl_vecs = vectors.extract_all_layers(name, [int(layer)])
            except Exception as exc:  # extraction failure must not be read as "not identifiable"
                secondary.append(dict(concept=name, error=type(exc).__name__,
                                      identifiable=None,
                                      note="extraction failed; this control is inconclusive, "
                                           "not a failure"))
                continue
            vec = ctrl_vecs[int(layer)]
            ctrl_norm = float(vec.float().norm())
            resid_norm = float(ctx.norms[layer]["resid_norm"])
            # Matched on r: the control concept's vector has its own norm, so its alpha differs
            # from the target's. Matching alpha would inject a different dose and the control
            # would answer a different question.
            ctrl_alpha = vectors.alpha_from_norms(dose, ctrl_norm, resid_norm)
            if ctrl_alpha > float(_const("ALPHA_CEIL")):
                secondary.append(dict(concept=name, identifiable=None,
                                      alpha=ctrl_alpha,
                                      note="alpha above ALPHA_CEIL at this dose; skipped rather "
                                           "than clamped"))
                continue
            with _as_concept(name):
                with expensive.phase_scope("CONTROL_CONCEPT"):
                    ctrl_row = expensive.measure_D2(layer, ctrl_alpha, n_d2, r=dose, vec=vec)
            entry = dict(concept=name, alpha=float(ctrl_alpha), vec_norm=ctrl_norm,
                         d2=ctrl_row.get("d2"), d2_se=ctrl_row.get("d2_se"),
                         n_d2=ctrl_row.get("n_d2"),
                         d2_ci_low=ctrl_row.get("d2_ci_low"),
                         d2_ci_high=ctrl_row.get("d2_ci_high"),
                         d4_damage_frac=ctrl_row.get("d4_damage_frac"),
                         d4_damage_frac_ci_low=ctrl_row.get("d4_damage_frac_ci_low"),
                         d4_damage_frac_ci_high=ctrl_row.get("d4_damage_frac_ci_high"),
                         d4_n=ctrl_row.get("d4_n"),
                         d4_reading=ctrl_row.get("d4_reading"),
                         vec_fingerprint=vectors.vec_fingerprint(vec))
            # "Identifiable" means it clears the same bar the target had to fail to be
            # interesting. A control that also sits under D2_MAX is not evidence of a preserved
            # pathway.
            entry["identifiable"] = (None if entry["d2"] is None
                                     else bool(float(entry["d2"]) > d2_max))
            secondary.append(entry)
            _append_row("controls.jsonl", dict(control="control_concept", layer=int(layer),
                                               r=dose, **entry))

    judged = [e for e in secondary if e.get("identifiable") is not None]
    n_identifiable = sum(1 for e in judged if e["identifiable"])
    identifiable_ci = (cheap.wilson_interval(n_identifiable, len(judged))
                       if judged else (None, None))
    secondary_pass = (None if not judged else bool(n_identifiable >= 1))

    if primary_pass and secondary_pass is not False:
        verdict = "pass"
    elif not primary_pass:
        verdict = "reject"
    elif secondary_pass is False:
        verdict = "reject"
    else:
        verdict = "inconclusive"

    detail = (f"D4 dominant={row.get('d4_dominant')} damage={damage_frac:.2f} "
              f"retrieval={retrieval_frac:.2f}")
    if judged:
        detail += (f"; {n_identifiable}/{len(judged)} control concepts identifiable at "
                   f"D2 > {d2_max}")

    out = dict(control="forced_id_capability", layer=int(layer), r=dose,
               d2_concept=row.get("d2"), d2_concept_n=row.get("n_d2"),
               d2_concept_ci_low=row.get("d2_ci_low"),
               d2_concept_ci_high=row.get("d2_ci_high"),
               d4=d4, d4_dominant=row.get("d4_dominant"),
               d4_damage_frac=damage_frac, d4_damage_n=row.get("d4_n"),
               d4_damage_ci_low=row.get("d4_damage_frac_ci_low"),
               d4_damage_ci_high=row.get("d4_damage_frac_ci_high"),
               d4_retrieval_frac=retrieval_frac,
               d4_retrieval_ci_low=row.get("d4_retrieval_frac_ci_low"),
               d4_retrieval_ci_high=row.get("d4_retrieval_frac_ci_high"),
               primary_reading=primary_reading, primary_pass=primary_pass,
               secondary=secondary, secondary_pass=secondary_pass,
               n_control_identifiable=n_identifiable, n_control_judged=len(judged),
               control_identifiable_rate=(n_identifiable / len(judged) if judged else None),
               control_identifiable_ci_low=identifiable_ci[0],
               control_identifiable_ci_high=identifiable_ci[1],
               d2_max=d2_max, verdict=verdict, detail=detail,
               # Spec 9.2 review flag: does the secondary method ever change a verdict the free
               # D4 reading did not already reach? Aggregate this across concepts before the
               # harmful arm and demote the method if the answer is never.
               secondary_agrees_with_primary=(None if secondary_pass is None
                                              else bool(secondary_pass == primary_pass)),
               secondary_under_review=True)
    _append_row("controls.jsonl", dict(control="forced_id_capability_summary", **out))
    return out


# =====================================================================================
# 9.3 - positive control / escalation ladder
# =====================================================================================

def escalation_ladder(ref_layer: int | None = None,
                      doses: Sequence[float] | None = None) -> dict:
    """Spec 9.3. If no layer cleared E6_FLOOR at any scan dose, does the vector work anywhere?

    Escalate r at the reference layer until reachability clears the floor or alpha would exceed
    ALPHA_CEIL. This distinguishes two very different run outcomes that otherwise look
    identical in the output:

        "no operating point exists for this concept"   -- a result
        "the vector is dead"                           -- a broken run

    Worth knowing what this can turn up: Macar reports Silk as the documented no-effect case -
    "the steering produces no discernible thematic effect" - but our Silk reached full drift at
    L46 alpha=4. Some "no effect" concepts may be artifacts of fixing the injection layer at
    L37. Confirming that would itself be a finding, so the ladder records every rung rather than
    only the one that cleared.
    """
    ctx = _run()
    if ref_layer is None:
        ref_layer = max(ctx.norms) if ctx.norms else None
    if ref_layer is None:
        raise RuntimeError("no calibrated layers - run phase0_calibrate before the ladder")

    floor = float(_const("E6_FLOOR"))
    ceil = float(_const("ALPHA_CEIL"))
    ladder = list(doses) if doses is not None else [0.15, 0.30, 0.45, 0.60, 0.80, 1.00]

    rungs: list[dict] = []
    cleared = None
    for dose in ladder:
        dose = round(float(dose), _R_PLACES)
        try:
            alpha = config.alpha_for(int(ref_layer), dose)
        except config.Unreachable as exc:
            rungs.append(dict(r=dose, alpha=None, reachable=False, reach=None,
                              reach_ci_low=None, reach_ci_high=None, reach_n=None,
                              note=f"alpha above ALPHA_CEIL ({ceil}): {exc}"))
            _append_row("controls.jsonl", dict(control="escalation", layer=int(ref_layer),
                                               **rungs[-1]))
            continue
        e6 = cheap.measure_E6(int(ref_layer), alpha)
        rung = dict(r=dose, alpha=float(alpha), reachable=True, reach=float(e6["reach"]),
                    reach_ci_low=e6.get("reach_ci_low"),
                    reach_ci_high=e6.get("reach_ci_high"), reach_n=e6.get("reach_n"),
                    e6_mass_median=e6.get("e6_mass_median"),
                    e6_rank_med=e6.get("e6_rank_med"),
                    cleared=bool(float(e6["reach"]) >= floor))
        rungs.append(rung)
        _append_row("controls.jsonl", dict(control="escalation", layer=int(ref_layer), **rung))
        if rung["cleared"]:
            cleared = rung
            break

    verdict = "vector_alive" if cleared else "vector_dead_or_unreachable"
    detail = (f"cleared E6_FLOOR={floor} at r={cleared['r']} (alpha={cleared['alpha']:.2f})"
              if cleared else
              f"never cleared E6_FLOOR={floor} up to r={ladder[-1]} at L{ref_layer}")

    out = dict(control="escalation_ladder", ref_layer=int(ref_layer), floor=floor,
               rungs=rungs, cleared=cleared, verdict=verdict, detail=detail)
    _append_row("controls.jsonl", dict(control="escalation_ladder_summary", **out))
    return out


# =====================================================================================
# the whole control battery for one winner
# =====================================================================================

def run_controls(winner: dict, *, n_d2: int | None = None,
                 run_escalation: bool = False) -> dict:
    """Every applicable control for the selected operating point, with one combined verdict.

    Spec 14.6 rule 7: both controls rejecting the winner is itself the finding, and the board
    should say so rather than reporting an operating point that does not survive its own checks.
    """
    layer = int(winner["layer"])
    dose = round(float(winner["r"]), _R_PLACES)
    ctx = _run()
    depth = layer / max(int(getattr(ctx, "n_layers", 0)) or 1, 1)

    random_ctl = random_direction_control(layer, dose, n_d2=n_d2)
    capability = forced_id_capability_control(layer, dose, n_d2=n_d2)
    ladder = escalation_ladder() if run_escalation else None

    rejects = [c["verdict"] == "reject" for c in (random_ctl, capability)]
    if all(rejects):
        verdict = "both_reject"
    elif any(rejects):
        verdict = "one_rejects"
    elif any(c["verdict"] == "inconclusive" for c in (random_ctl, capability)):
        verdict = "inconclusive"
    else:
        verdict = "pass"

    out = dict(layer=layer, r=dose, depth=depth,
               random_direction=random_ctl, forced_id_capability=capability,
               escalation=ladder, verdict=verdict,
               winner_survives=bool(verdict == "pass"))
    _append_row("controls.jsonl", dict(control="battery_summary",
                                       layer=layer, r=dose, verdict=verdict,
                                       random_verdict=random_ctl["verdict"],
                                       capability_verdict=capability["verdict"]))
    return out
