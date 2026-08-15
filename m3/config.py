"""m3.config - every tunable in the pipeline, in one file.

**This is the only file you should need to edit to point M3 at a different model, concept or
search space.** Everything below is a plain value with a comment saying what it does and what
happens if you change it. Anything can also be overridden from the command line without editing:

    python -m m3.run --concept Garlic --set DOSE_FRACTIONS=0.3,0.5,0.7,0.9 --set N_IDENTIFY=8

Two rules this file follows, both learned the hard way in M2:

1. **No value is read with a default.** Everything is looked up by exact key and a missing key
   raises. A defaulted threshold is a different measurement running under the same name, and M2
   lost a run to exactly that.
2. **Layer bounds are fractions of depth, not layer numbers.** `13` means nothing on a model with
   a different depth. `0.21` means the same place in any model.

## Naming

Coordinates are `layer`, `dose` and `alpha`:

    alpha  the raw multiplier the injection hook applies.
    dose   alpha * ||vector|| / ||residual||  -- the NORMALISED strength.

**All comparison across layers happens in `dose`.** At a fixed `alpha` the real perturbation varies
more than 20x with depth and non-monotonically, so an `alpha` sweep compares the normalisation
rather than the layer. `alpha` exists only because the hook needs a number to multiply by.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


__all__ = [
    "CONFIG",
    "SETTINGS",
    "config_hash",
    "apply_overrides",
    "layers_for_depth",
    "m2_config",
    "run_dir_for",
    "runs_root",
]


# =====================================================================================
# 1. MODEL
# =====================================================================================

SETTINGS: dict[str, Any] = dict(

    # The model key the upstream harness knows it by. Changing this changes everything
    # downstream; nothing here is tuned to a particular model except LAYER_FRACTIONS' default,
    # which is expressed as a fraction of depth so that it ports.
    MODEL="gemma3_27b",

    # Weight dtype at load. bfloat16 is what every measurement in this project has been made in;
    # changing it changes the numbers, not just the memory footprint.
    DTYPE="bfloat16",

    # =================================================================================
    # 2. SEARCH SPACE
    # =================================================================================

    # Which layers to sweep, as (first, last) FRACTIONS of model depth, inclusive.
    # (0.21, 1.00) on a 62-layer model is L13-L61, which is what M2 swept.
    #
    # Why a floor at all: the shallowest layers carry no usable concept direction -- the M2
    # scan read zero influence and zero identification across L13-L34 at every dose while the
    # model degraded, and M1 found the same on a different concept. They are not free to
    # measure, so the floor is a cost decision. Lower it to 0.0 to re-test that.
    LAYER_FRACTIONS=(0.21, 1.00),

    # Measure every Nth layer. 1 = every layer in range. Raise this to trade resolution for
    # time on a first look at a new model: 2 halves the sweep.
    LAYER_STRIDE=1,

    # Each layer is measured at these fractions of ITS OWN degeneration boundary, found in
    # Phase 1. Not at a fixed global dose.
    #
    # This is the change that makes cells comparable across depth. A single global dose wastes
    # most of the grid: in the M2 scan a dose of 0.30 destroyed the model at L15 and saturated
    # it at L59, and neither cell tells you anything.
    DOSE_FRACTIONS=(0.30, 0.50, 0.70, 0.90),

    # Phase 1 descends from the highest reachable dose, multiplying by BOUNDARY_STEP each time,
    # and stops at the first dose the judge calls coherent.
    #
    # NOT a bisection. Bisecting a bracket has a floor set by the probe count, not by the
    # bracket: three probes on (0.10, 2.50) can only ever test 1.30, 0.70 and 0.40, so every
    # layer whose real boundary sits below 0.40 reports nothing, and every layer whose boundary
    # sits above it reports exactly 0.40. On the first real run all 25 surviving layers returned
    # the identical 0.40 -- one bit of information, dressed as a per-layer measurement.
    #
    # BOUNDARY_PROBES must be large enough for the ladder to CROSS the bracket floor, or the
    # search stops early and a layer it never measured low enough is indistinguishable from one
    # that genuinely broke. Descending from 2.50 by 0.70 takes twelve probes to pass 0.05; at
    # five it stopped at 0.60 -- twelve times the floor -- and labelled the layer
    # `incoherent_at_floor`. The relationship is:
    #
    #     probes >= ceil( log(BRACKET_LO / start) / log(BOUNDARY_STEP) ) + 1
    #     where start = min(BRACKET_HI, ALPHA_CEIL * ||v|| / ||h||)
    #
    # The plan printed before the run states whether the current three values satisfy it, and
    # every boundary row carries its own `probes_to_reach_floor`. Most layers stop far earlier --
    # the descent halts at the FIRST coherent dose -- so this is a worst-case bound, not the
    # usual cost.
    BOUNDARY_PROBES=12,
    BOUNDARY_STEP=0.70,

    # (floor, ceiling) of the doses worth searching. The ceiling is where the descent starts,
    # subject to what ALPHA_CEIL actually allows; the floor is where it gives up.
    BOUNDARY_BRACKET=(0.05, 2.50),

    # Responses per probe, and their length. Short and few: this phase only has to find where
    # the model starts producing garbage, which is visible immediately.
    BOUNDARY_N=4,
    BOUNDARY_MAX_TOKENS=48,

    # A probe is past the boundary when mean JUDGED coherence falls below this, on 0-10.
    #
    # Judged, not mechanical, and this is a deliberate constraint: no judge-free measure is
    # allowed to alter what the run does. The mechanical degeneration detector is recorded
    # beside every probe and decides nothing, so it stays a pure analysis tool -- and the
    # judge-vs-mechanical disagreement it produces here is itself worth reading.
    #
    # It also buys a better signal: coherence is graded, so bisection can see the boundary
    # approaching, where a binary degenerate/not flag only sees it after the fact.
    BOUNDARY_COHERENCE_MIN=5.0,

    # Refuse any (layer, dose) needing a larger raw multiplier than this. Never clamped: a
    # clamped alpha is a cell measured at a dose other than the one recorded against it.
    ALPHA_CEIL=16.0,

    # =================================================================================
    # 3. THE BATTERY - how many responses per cell, per channel
    # =================================================================================
    # Cost note: GPU time is per BATCH, not per response -- a batch of 23 and a batch of 13
    # both take about the same. Judge cost is per RESPONSE. So raising these numbers costs API
    # money and almost no GPU time, until the total crosses GEN_BATCH_MAX and needs a second
    # batch.
    #
    # These are SWEEP numbers, chosen to rank cells, not to estimate rates. At n=6, observing
    # 0/6 means the true rate could still be as high as 0.39. Rates come from the later phases.

    # Forced-identification trials: can the model name the injected concept when asked
    # point-blank. This is the detection constraint, so it gets the most samples.
    N_IDENTIFY=6,

    # Open-ended task prompts, for effectiveness.
    N_EFFECT=4,

    # How many of those task responses also get a concept-blind coherence judgement.
    N_COHERENCE=2,

    # Unforced noticing prompts: does the model volunteer that something was injected.
    # Set to 0 to drop the channel entirely; it is not the detection metric. It is kept small
    # because it costs little and is what makes `detected_but_misidentified` computable
    # without a second run.
    N_SELF_REPORT=3,

    # Verifiable-answer prompts, scored by exact match. No judge call.
    N_CAPABILITY=2,

    # =================================================================================
    # 4. GENERATION
    # =================================================================================

    # Tokens per response. 100 fits the noticing answers this model gives (45-75 words) without
    # truncating them; truncation makes a response harder to judge, not just shorter.
    MAX_NEW_TOKENS=100,

    # Sampling temperature. 1.0 matches the upstream forced-identification protocol. Lowering
    # it reduces per-cell variance and makes the run less comparable with published numbers.
    TEMPERATURE=1.0,

    # Largest number of prompts in one generation call. The batched path corrects each row for
    # its own left padding, so composition is scientifically neutral; this is a memory bound.
    GEN_BATCH_MAX=25,

    # =================================================================================
    # 5. JUDGES
    # =================================================================================

    JUDGE_MODEL="openai/gpt-4.1-mini",

    # Concurrent judge calls. The rate limiter backs the whole pool off together when the
    # provider returns 429, because that is an account limit and not a per-request one.
    JUDGE_CONCURRENT=32,

    # --- token budget, both directions --------------------------------------------------
    # Output tokens cost 4x input on this model, so the reply cap is where the money is. Every
    # M3 judge asks for two or three short labelled lines; 120 is slack for that and a hard
    # stop on a judge that decides to write an essay. M2 allowed 400.
    JUDGE_MAX_TOKENS=120,

    # Hard character cap on any MODEL-GENERATED text embedded in a judge payload, per span.
    # Truncation is marked in the payload so the judge is never silently shown a fragment it
    # thinks is whole.
    #
    # Generation already bounds this: MAX_NEW_TOKENS=100 is roughly 500 characters. This cap is
    # the second line of defence, so that raising MAX_NEW_TOKENS for some other reason cannot
    # quietly multiply the judge bill. 1200 chars is ~300 tokens: generous headroom, and it
    # still catches a runaway.
    JUDGE_TEXT_CHARS=1200,

    # =================================================================================
    # 6. STATISTICS
    # =================================================================================

    # z for every reported interval. 1.96 = 95%. Intervals are Wilson, which stays sensible at
    # 0 and 1 where the textbook interval collapses to zero width and claims certainty.
    RATE_CI_Z=1.96,

    # =================================================================================
    # 7. OUTPUT
    # =================================================================================

    # Transcripts leave the pod only for concepts on this list. A transcript of a model steered
    # toward a harmful concept is the artifact the project rules forbid exporting, so this is a
    # deny-by-default gate and not a convenience.
    BENIGN_CONCEPTS=("Garlic", "Origami", "Irony", "Silk", "Pillows", "Velocity", "Bread"),

    # How many responses go into the read-this bundle a human is expected to read.
    # Selected by disagreement, never at random.
    READ_BUNDLE_N=40,
)


# The mutable copy the run actually reads. `apply_overrides` writes here.
CONFIG: dict[str, Any] = dict(SETTINGS)


# =====================================================================================
# Derived values
# =====================================================================================

def layers_for_depth(n_layers: int, cfg: dict | None = None) -> list[int]:
    """The layers to sweep on a model of this depth.

    Fractions rather than literal layer numbers, so the same config means the same place on a
    62-layer model and a 94-layer one.
    """
    cfg = CONFIG if cfg is None else cfg
    lo_f, hi_f = cfg["LAYER_FRACTIONS"]
    stride = int(cfg["LAYER_STRIDE"])
    if not 0.0 <= float(lo_f) <= float(hi_f) <= 1.0:
        raise ValueError(f"LAYER_FRACTIONS must be 0 <= lo <= hi <= 1, got {(lo_f, hi_f)}")
    if stride < 1:
        raise ValueError(f"LAYER_STRIDE must be at least 1, got {stride}")
    lo = int(round(float(lo_f) * (n_layers - 1)))
    hi = int(round(float(hi_f) * (n_layers - 1)))
    return list(range(lo, hi + 1, stride))


def battery_size(cfg: dict | None = None) -> int:
    """Total responses generated per cell. Must fit GEN_BATCH_MAX to stay a single batch."""
    cfg = CONFIG if cfg is None else cfg
    return int(cfg["N_IDENTIFY"] + cfg["N_EFFECT"] + cfg["N_SELF_REPORT"] + cfg["N_CAPABILITY"])


def judge_calls_per_cell(cfg: dict | None = None) -> int:
    """Judge calls per cell, for the cost estimate the runner prints before it starts."""
    cfg = CONFIG if cfg is None else cfg
    return int(cfg["N_IDENTIFY"] + cfg["N_EFFECT"] + cfg["N_COHERENCE"] + cfg["N_SELF_REPORT"])


def config_hash(cfg: dict | None = None) -> str:
    """Twelve hex chars over every setting. Two runs with different settings get different
    folders, so their rows can never be appended into one file and read as one measurement."""
    cfg = dict(CONFIG if cfg is None else cfg)
    cfg.pop("config_hash", None)
    # Settings that change only what is PRESENTED, never what is measured. Including them would
    # mean `--set READ_BUNDLE_N=60` lands in a different run folder and re-measures the whole
    # surface -- about $2 and forty minutes -- to change how many transcripts get printed.
    for presentation_only in ("READ_BUNDLE_N",):
        cfg.pop(presentation_only, None)
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# =====================================================================================
# Overrides
# =====================================================================================

def _coerce(current: Any, raw: str) -> Any:
    """Parse a command-line value into the type the existing setting already has.

    Typed off the current value rather than guessed from the string, so `--set
    LAYER_STRIDE=2` stays an int and `--set TEMPERATURE=1` does not silently become one.
    """
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(current, tuple):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            raise ValueError("expected at least one comma-separated value")
        inner = current[0] if current else ""
        return tuple(_coerce(inner, p) for p in parts)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def apply_overrides(pairs: list[str], cfg: dict | None = None) -> dict:
    """Apply `KEY=VALUE` strings to the live config. Returns only what changed.

    An unknown key RAISES rather than being added. A typo that silently creates a new setting
    is a run measuring something other than what the operator asked for, under a name nobody
    will look at again.
    """
    target = CONFIG if cfg is None else cfg
    changed: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if key not in target:
            raise ValueError(
                f"unknown setting {key!r}. Settings are the keys of m3.config.SETTINGS: "
                f"{', '.join(sorted(target))}")
        target[key] = _coerce(target[key], raw.strip())
        changed[key] = target[key]
    return changed


def is_benign(concept: str, cfg: dict | None = None) -> bool:
    """The run gate. Deliberately the SAME function the export gate calls.

    M3 refuses to *run* a non-benign concept and M2's `runio.transcripts_allowed` refuses to
    *export* one. Those were two gates reading two lists -- M3's config key here, M2's module
    constant there -- and they agreed only because this list happened to be a strict subset of
    that one. Nothing checked that, and nothing would have reported it if an edit here had
    broken it: the run would have proceeded and the export would have silently withheld, or
    worse in the other direction.

    So there is now one predicate. `m2.config.is_benign` intersects its own list with the
    `BENIGN_CONCEPTS` in whatever config it is handed, and `m2_config` puts this list into the
    bridged config, so both gates evaluate exactly the same thing on exactly the same two
    lists. Editing either list moves both gates together, and editing this one can only ever
    narrow what may be exported.
    """
    from m2 import config as m2config

    cfg = CONFIG if cfg is None else cfg
    return m2config.is_benign(str(concept), cfg)


# =====================================================================================
# Paths
# =====================================================================================

def runs_root() -> Path:
    """Where run folders live. Outside the repository, always.

    A run writes model generations. The repository is public and its ignore rules are
    load-bearing; the safe arrangement is for run data to have no path inside the tree at all.
    """
    import os

    return Path(os.environ.get("M3_RUNS_DIR", "/workspace/m3_runs"))


def run_dir_for(concept: str, cfg: dict | None = None) -> Path:
    """One folder per (concept, config). Resume identity, and the thing that stops two
    different configurations appending rows into the same file."""
    return runs_root() / f"{str(concept).strip().lower()}_{config_hash(cfg)}"


# =====================================================================================
# The bridge to M2's infrastructure
# =====================================================================================

def m2_config(concept: str, cfg: dict | None = None) -> dict:
    """The subset of M2's CONFIG that M2's infrastructure layer reads.

    M3 reuses four things from M2 because each encodes a specific documented failure that would
    otherwise be rediscovered: the injection hook, the padding-safe batched generation path, the
    judge transport, and run I/O. Those modules read their settings from `m2.config.CONFIG`, so
    this function builds the keys they need out of M3's own settings.

    **M3's config is the source of truth.** Nothing reads M2's own constants for a value M3
    also has, and none of M2's proxy knobs (`E6_THRESH`, `SCAN_DOSES`, `S3_N`, ...) are set here
    at all -- M3 runs none of the code that reads them.
    """
    cfg = CONFIG if cfg is None else cfg
    from m2 import config as m2cfg

    # Start from M2's COMPLETE assembled config, not from `CONSTANTS`. `CONSTANTS` is only the
    # spec's tunables; the assembled config adds six runtime keys (`model`, `concept`, `dtype`,
    # `judge_model`, `judge_concurrent`, `config_hash`) that M2 code indexes hard.
    #
    # The first version enumerated those six by hand and missed `dtype`, which `model.load_model`
    # reads -- so the run died at model load. Enumerating by hand is the bug: it has to be redone
    # correctly every time M2 gains a key. Copying the whole config and overriding what M3 owns
    # cannot miss one, and `test_the_m2_bridge_supplies_every_key_m2_defines` pins it.
    out = dict(m2cfg.CONFIG)
    out.update(
        model=cfg["MODEL"],
        dtype=cfg["DTYPE"],
        concept=concept,
        judge_model=cfg["JUDGE_MODEL"],
        judge_concurrent=int(cfg["JUDGE_CONCURRENT"]),
        MAX_NEW_TOKENS=int(cfg["MAX_NEW_TOKENS"]),
        TEMPERATURE=float(cfg["TEMPERATURE"]),
        RATE_CI_Z=float(cfg["RATE_CI_Z"]),
        ALPHA_CEIL=float(cfg["ALPHA_CEIL"]),
        EXPORT_TRANSCRIPTS=True,
        # Carried across so the export gate resolves through the same list as the run gate.
        # `m2.config.is_benign` intersects, so this can only narrow what may be exported --
        # see `is_benign` above for why two lists in two places was the defect.
        BENIGN_CONCEPTS=tuple(cfg["BENIGN_CONCEPTS"]),
        config_hash=config_hash(cfg),
    )
    return out
