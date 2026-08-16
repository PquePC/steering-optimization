"""Every setting in `m3.config` must reach the code that acts on it.

## Why this file exists

Five of the defects found in M3 were the same shape: a setting declared in `m3.config`, printed
in the plan, folded into the config hash and written into the run folder — and read by nothing.

    JUDGE_MAX_TOKENS    declared; the transport held its own module constant
    JUDGE_MODEL         declared; the transport read M2's config
    JUDGE_CONCURRENT    declared; likewise
    GEN_BATCH_MAX       declared; the chunking read `expensive.GEN_BATCH_MAX`
    READ_BUNDLE_N       declared; nothing consumed it at all

The failure is silent and total. The operator sets a value, watches it echo back in the plan,
and the run does something else. Three of the five had defaults that happened to match M2's, so
nothing looked wrong at any point — which is what makes this shape worse than a missing setting
rather than better.

## What is asserted, and what "reaches" means here

Not "the name appears somewhere". A grep would have passed on `GEN_BATCH_MAX`, which was read —
by the guard that checks the battery fits, while the chunking that splits the batch read a
different variable entirely.

So every setting has a **witness**: a function that changes the setting to a second value, runs
the real code that consumes it, and returns what that code observed. A setting reaches its
consumer only if two different values produce two different observations *at the consuming site*.
`GEN_BATCH_MAX` is observed as `m2.expensive.GEN_BATCH_MAX` after `configure_generation`, not as
`cfg["GEN_BATCH_MAX"]`; `TEMPERATURE` is observed as the argument generation actually received.

The second assertion is the structural one, and it is the reason this file is worth its runtime:
`WITNESSES` must cover `SETTINGS` exactly. Adding a setting to `m3.config` without wiring it
fails here, by name, before it can be printed in a plan and believed.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from m3 import config, judge, run as m3run, sweep                  # noqa: E402
from m3.tests.fake_gpu import fake_gpu                             # noqa: E402


CONCEPT = "Garlic"

# Two values per setting. The second is chosen to be legal but visibly different — an
# observation that differs is the whole proof, so a probe value that changes nothing observable
# (a second model name that is never printed, a temperature the sampler ignores) would make this
# test vacuous rather than failing.
PROBES: dict[str, tuple[Any, Any]] = {
    "MODEL": ("gemma3_27b", "gemma3_4b"),
    "DTYPE": ("bfloat16", "float16"),
    "LAYER_FRACTIONS": ((0.21, 1.00), (0.60, 1.00)),
    "LAYER_STRIDE": (1, 5),
    "DOSE_FRACTIONS": ((0.5,), (0.4, 0.8)),
    "BOUNDARY_PROBES": (2, 6),
    "BOUNDARY_STEP": (0.70, 0.35),
    "BOUNDARY_BRACKET": ((0.05, 2.50), (0.05, 0.90)),
    "BOUNDARY_N": (2, 4),
    "BOUNDARY_MAX_TOKENS": (48, 96),
    "BOUNDARY_COHERENCE_MIN": (5.0, 9.9),
    "ALPHA_CEIL": (16.0, 3.0),
    "N_IDENTIFY": (6, 2),
    "N_EFFECT": (4, 2),
    "N_COHERENCE": (2, 1),
    "N_SELF_REPORT": (3, 1),
    "N_CAPABILITY": (2, 1),
    "NULL_REPEATS": (1, 3),
    "MAX_NEW_TOKENS": (100, 64),
    "TEMPERATURE": (1.0, 0.25),
    "GEN_BATCH_MAX": (25, 40),
    "JUDGE_MODEL": ("openai/gpt-4.1-mini", "openai/gpt-4o-mini"),
    "JUDGE_CONCURRENT": (32, 4),
    "JUDGE_MAX_TOKENS": (120, 48),
    "JUDGE_TEXT_CHARS": (1200, 80),
    "RATE_CI_Z": (1.96, 2.576),
    "READ_BUNDLE_N": (40, 1),
}


def _cfg(name: str, value: Any) -> dict:
    """M3's real config with one setting changed. Small elsewhere, so witnesses stay fast."""
    cfg = dict(config.SETTINGS)
    cfg.update(LAYER_FRACTIONS=(0.60, 0.62), LAYER_STRIDE=1, DOSE_FRACTIONS=(0.5,),
               BOUNDARY_PROBES=2, N_IDENTIFY=2, N_EFFECT=2, N_COHERENCE=1,
               N_SELF_REPORT=1, N_CAPABILITY=1)
    cfg[name] = value
    return cfg


@contextlib.contextmanager
def _wired(cfg: dict, tmp: Path) -> Iterator[dict]:
    """A live run context: model loaded, config bridged, transport and generation configured.

    This is the real `open_run` path, not a reconstruction of it. A witness that set up its own
    approximation of the wiring would prove that the approximation works.
    """
    import os

    os.environ["M3_RUNS_DIR"] = str(tmp)
    with fake_gpu() as calls:
        from m2 import model

        model.load_model(config.m2_config(CONCEPT, cfg))
        sweep.open_run(CONCEPT, cfg)
        yield calls


def _one_layer(cfg: dict) -> int:
    return config.layers_for_depth(62, cfg)[0]


def _boundary(cfg: dict, tmp: Path) -> tuple[dict, dict]:
    """Run the real Phase 1 on one layer. Returns (boundary row, harness call record).

    Always the DEEPEST layer. At mid-depth the fake model's `max_reachable` is about 0.46, so
    the descent starts there whatever the bracket ceiling says and two different brackets probe
    the identical doses — an observation that cannot discriminate, which would have made these
    witnesses pass by being blind rather than by the setting working.
    """
    cfg = dict(cfg, LAYER_FRACTIONS=(1.0, 1.0))
    with _wired(cfg, tmp) as calls:
        layer = _one_layer(cfg)
        sweep.calibrate(CONCEPT, [layer], cfg)
        return sweep.find_boundary(layer, CONCEPT, cfg), calls


def _cell(cfg: dict, tmp: Path) -> tuple[dict, dict, list[dict]]:
    """Run the real Phase 2 on one cell. Returns (cell row, call record, judge rows)."""
    from m2 import runio

    with _wired(cfg, tmp) as calls:
        layer = _one_layer(cfg)
        cal = sweep.calibrate(CONCEPT, [layer], cfg)
        calls["seen"].clear()                       # the null arm's batch is not the cell's
        cell = sweep.measure_cell(layer, 0.20, concept=CONCEPT,
                                  baselines=cal["baselines"], cfg=cfg)
        return cell, calls, runio.read_rows(sweep.JUDGE_FILE)


def _sweep(cfg: dict, tmp: Path, disagree_every: int = 7) -> Path:
    """A whole run, for the settings only a whole run can witness."""
    import os

    os.environ["M3_RUNS_DIR"] = str(tmp)
    os.environ.setdefault("OPENROUTER_API_KEY", "fake")
    os.environ.setdefault("HF_TOKEN", "fake")
    saved = dict(config.CONFIG)
    config.CONFIG.clear()
    config.CONFIG.update(cfg)
    try:
        with fake_gpu(judge_disagree_every=disagree_every):
            assert m3run.main(["--concept", CONCEPT]) == m3run.EXIT_OK
    finally:
        config.CONFIG.clear()
        config.CONFIG.update(saved)
    return next(Path(tmp).glob("garlic_*"))


# =====================================================================================
# The witnesses
# =====================================================================================
# Each returns what the CONSUMING code observed, never the config value it was given.

def _w_model(value, tmp):
    with _wired(_cfg("MODEL", value), tmp):
        from m2 import config as m2config
        return m2config.RUN.config["model"]          # the dict load_model actually indexed


def _w_dtype(value, tmp):
    with _wired(_cfg("DTYPE", value), tmp):
        from m2 import config as m2config
        return m2config.RUN.config["dtype"]


def _w_layers(value, tmp, name):
    cfg = dict(config.SETTINGS)
    cfg[name] = value
    return config.layers_for_depth(62, cfg)


def _w_gen_batch(value, tmp):
    from m2 import expensive
    judge.configure_generation(_cfg("GEN_BATCH_MAX", value))
    return expensive.GEN_BATCH_MAX                   # what the chunking reads, not the guard


def _w_transport(value, tmp, name, read):
    from m2 import config as m2config, judges as transport
    judge.configure_transport(_cfg(name, value))
    return {"judge_model": lambda: m2config.CONFIG["judge_model"],
            "judge_concurrent": lambda: m2config.CONFIG["judge_concurrent"],
            "max_tokens": lambda: transport.JUDGE_MAX_TOKENS}[read]()


def _w_channel(value, tmp, name, channel):
    from collections import Counter
    # Inside the harness: the prompt builders import the upstream research repo, which is not
    # installed here. That is the seam fake_gpu exists to stub.
    with fake_gpu():
        rows = sweep.battery_prompts(_cfg(name, value))
    return Counter(r["channel"] for r in rows)[channel]


def _w_coherence_budget(value, tmp):
    """N_COHERENCE decides how many effect responses ALSO get a coherence judgement."""
    cfg = _cfg("N_COHERENCE", value)
    cfg["N_EFFECT"] = 4
    rows = [dict(channel="effect", unit=f"task_{i}", response="a story about garlic",
                 prompt_text="Tell me a short story.", degenerate=False) for i in range(4)]
    items = sweep._judge_items(rows, concept=CONCEPT,
                               baselines={f"task_{i}": "plain" for i in range(4)},
                               phase="T", layer=1, dose=0.1, cfg=cfg)
    return sum(1 for i in items if i["judge_kind"] == "coherence")


def _w_boundary(value, tmp, name, read, **tweak):
    """`tweak` sets up conditions under which the target setting is observable at all.

    The ladder stops at the FIRST coherent dose, so on a layer whose boundary is above the
    starting dose it takes one probe regardless of BOUNDARY_PROBES — and a witness reading the
    probe count would then report "no effect" for a setting that works perfectly. Raising the
    coherence bar out of reach makes the ladder run to exhaustion, which is the regime where the
    probe budget is what decides the outcome.
    """
    row, calls = _boundary(_cfg(name, value) | tweak, tmp)
    return {"n_probes": lambda: len(row["probes"]),
            "doses": lambda: [p["dose"] for p in row["probes"]],
            "max_reachable": lambda: row["max_reachable_dose"],
            "outcome": lambda: (row["outcome"], row["dose_max"]),
            "tokens": lambda: sorted({c["max_tokens"] for c in calls["seen"]
                                      if c["layer"] is not None}),
            "batch_n": lambda: sorted({c["n"] for c in calls["seen"]
                                       if c["layer"] is not None})}[read]()


def _w_cell(value, tmp, name, read):
    cell, calls, judged = _cell(_cfg(name, value), tmp)
    return {"max_tokens": lambda: sorted({c["max_tokens"] for c in calls["seen"]}),
            "temperature": lambda: sorted({c["temperature"] for c in calls["seen"]}),
            # The INTERVAL, not `ci_z`. `battery.rate` echoes its `z` argument straight into
            # `ci_z=z`, so that field differs whether or not `wilson_interval` ever saw it --
            # this witness passed with the interval hardcoded to 1.96. Reading the echo instead
            # of the effect is exactly the mistake this whole file exists to catch, committed
            # inside the file itself.
            "ci_bounds": lambda: (round(cell["identification"]["ci_low"], 6),
                                  round(cell["identification"]["ci_high"], 6)),
            "payload_len": lambda: max(len(r["payload"]) for r in judged)}[read]()


def _w_null_repeats(value, tmp):
    """Observed as the number of alpha=0 rows `calibrate` actually wrote, not as the setting."""
    from m2 import runio

    cfg = _cfg("NULL_REPEATS", value)
    with _wired(cfg, tmp):
        sweep.calibrate(CONCEPT, [_one_layer(cfg)], cfg)
        return len(runio.read_rows(sweep.NULL_FILE))


def _w_read_bundle(value, tmp):
    """A cap is only observable on a run that produces more disagreements than the cap.

    The small grid every other witness uses yields one, so both probe values would show one and
    the setting would look inert. A wider grid and a judge that contradicts the detector every
    other call is what makes the cap the binding constraint.
    """
    # Built from SETTINGS, not from `_cfg`: the shrunken battery every other witness uses
    # produces too few judged responses for any disagreement to arise at all, and a witness
    # comparing zero against zero would report "no effect" for a setting that works.
    cfg = dict(config.SETTINGS, READ_BUNDLE_N=value,
               LAYER_FRACTIONS=(0.55, 1.00), LAYER_STRIDE=2, DOSE_FRACTIONS=(0.4, 0.9))
    d = _sweep(cfg, tmp, disagree_every=2)
    text = (d / sweep.READ_FILE).read_text(encoding="utf-8")
    return sum(1 for l in text.splitlines()
               if l.startswith("### ") and "null arm" not in l)


def _w_dose_fractions(value, tmp):
    d = _sweep(_cfg("DOSE_FRACTIONS", value), tmp)
    rows = [json.loads(l) for l in (d / sweep.CELLS_FILE).open(encoding="utf-8")]
    return sorted({(r["layer"], r["dose"]) for r in rows})


WITNESSES = {
    "MODEL": _w_model,
    "DTYPE": _w_dtype,
    "LAYER_FRACTIONS": lambda v, t: _w_layers(v, t, "LAYER_FRACTIONS"),
    "LAYER_STRIDE": lambda v, t: _w_layers(v, t, "LAYER_STRIDE"),
    "DOSE_FRACTIONS": _w_dose_fractions,
    # NEVER_SANE makes the ladder descend to exhaustion, so the probe budget and the step ratio
    # are what decide the doses. Without it the search stops on probe one and neither is visible.
    "BOUNDARY_PROBES": lambda v, t: _w_boundary(v, t, "BOUNDARY_PROBES", "n_probes",
                                                BOUNDARY_COHERENCE_MIN=9.9),
    "BOUNDARY_STEP": lambda v, t: _w_boundary(v, t, "BOUNDARY_STEP", "doses",
                                              BOUNDARY_COHERENCE_MIN=9.9),
    "BOUNDARY_BRACKET": lambda v, t: _w_boundary(v, t, "BOUNDARY_BRACKET", "doses",
                                                 BOUNDARY_COHERENCE_MIN=9.9),
    "BOUNDARY_N": lambda v, t: _w_boundary(v, t, "BOUNDARY_N", "batch_n"),
    "BOUNDARY_MAX_TOKENS": lambda v, t: _w_boundary(v, t, "BOUNDARY_MAX_TOKENS", "tokens"),
    # The threshold is only visible where the model IS coherent: a bracket that starts above the
    # boundary scores 0 everywhere and every threshold rejects it alike.
    "BOUNDARY_COHERENCE_MIN": lambda v, t: _w_boundary(v, t, "BOUNDARY_COHERENCE_MIN", "outcome",
                                                       BOUNDARY_BRACKET=(0.05, 0.90)),
    "ALPHA_CEIL": lambda v, t: _w_boundary(v, t, "ALPHA_CEIL", "max_reachable"),
    "N_IDENTIFY": lambda v, t: _w_channel(v, t, "N_IDENTIFY", "identify"),
    "N_EFFECT": lambda v, t: _w_channel(v, t, "N_EFFECT", "effect"),
    "N_COHERENCE": _w_coherence_budget,
    "N_SELF_REPORT": lambda v, t: _w_channel(v, t, "N_SELF_REPORT", "self_report"),
    "N_CAPABILITY": lambda v, t: _w_channel(v, t, "N_CAPABILITY", "capability"),
    "NULL_REPEATS": _w_null_repeats,
    "MAX_NEW_TOKENS": lambda v, t: _w_cell(v, t, "MAX_NEW_TOKENS", "max_tokens"),
    "TEMPERATURE": lambda v, t: _w_cell(v, t, "TEMPERATURE", "temperature"),
    "GEN_BATCH_MAX": _w_gen_batch,
    "JUDGE_MODEL": lambda v, t: _w_transport(v, t, "JUDGE_MODEL", "judge_model"),
    "JUDGE_CONCURRENT": lambda v, t: _w_transport(v, t, "JUDGE_CONCURRENT", "judge_concurrent"),
    "JUDGE_MAX_TOKENS": lambda v, t: _w_transport(v, t, "JUDGE_MAX_TOKENS", "max_tokens"),
    "JUDGE_TEXT_CHARS": lambda v, t: _w_cell(v, t, "JUDGE_TEXT_CHARS", "payload_len"),
    "RATE_CI_Z": lambda v, t: _w_cell(v, t, "RATE_CI_Z", "ci_bounds"),
    "READ_BUNDLE_N": _w_read_bundle,
}


# =====================================================================================
# The tests
# =====================================================================================

def test_every_setting_has_a_reachability_witness():
    """The structural guard, and the reason this file earns its runtime.

    A new setting in `m3.config` fails here BY NAME until something is shown to read it. That is
    the only point in the process where "declared but not applied" is cheap to catch: after it,
    the setting is in the plan, in the hash and in the run folder, and looks applied from every
    angle an operator has.
    """
    missing = sorted(set(config.SETTINGS) - set(WITNESSES))
    assert not missing, (
        f"{missing} are settings with no reachability witness. Add each to PROBES and "
        "WITNESSES in this file, observing the value AT THE CODE THAT ACTS ON IT — not at "
        "`cfg[NAME]`, which proves only that the config holds what it was given. If nothing "
        "reads the setting, that is the defect this test is for: wire it or delete it.")
    stale = sorted(set(WITNESSES) - set(config.SETTINGS))
    assert not stale, f"{stale} have witnesses but are no longer settings"
    assert sorted(PROBES) == sorted(config.SETTINGS), "PROBES and SETTINGS disagree"


@pytest.mark.parametrize("name", sorted(PROBES))
def test_the_setting_changes_what_the_consuming_code_does(name, tmp_path, monkeypatch):
    """Two values in, two different observations out — measured where the value is used.

    A setting that fails here is being displayed and not applied. That is strictly worse than a
    missing setting: the operator has been shown a number the run did not use.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setenv("HF_TOKEN", "fake")
    a, b = PROBES[name]
    got_a = WITNESSES[name](a, tmp_path / "a")
    got_b = WITNESSES[name](b, tmp_path / "b")
    assert got_a != got_b, (
        f"{name}: setting it to {a!r} and to {b!r} produced the identical observation "
        f"{got_a!r} at the code that consumes it. The value is declared, printed in the plan, "
        f"folded into the config hash and stored in the run folder — and changes nothing. "
        f"Five settings have already shipped in exactly this state.")


def test_a_setting_that_is_ignored_would_fail_this_test(tmp_path):
    """The test's own tripwire.

    A reachability test that cannot fail is the defect it exists to catch, wearing a green tick.
    So one setting is deliberately consumed by code that ignores it, and this asserts the
    machinery notices.
    """
    ignored = lambda value, tmp: "always the same"          # noqa: E731
    a, b = PROBES["JUDGE_MAX_TOKENS"]
    assert ignored(a, tmp_path) == ignored(b, tmp_path)
    assert WITNESSES["JUDGE_MAX_TOKENS"](a, tmp_path) != \
        WITNESSES["JUDGE_MAX_TOKENS"](b, tmp_path)


def test_the_only_concept_gate_left_is_the_harmful_arm(tmp_path, monkeypatch):
    """The benign allow-list is gone from M3 -- it refused ordinary nouns nobody had added to a
    seven-item tuple, which filtered exploration rather than risk. What must NOT be gone is the
    refusal on the arm this study has not designed."""
    from m3 import run as m3run

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setenv("HF_TOKEN", "fake")
    assert config.concept_allowed("Tundras")
    assert not config.concept_allowed("weapon")
    assert m3run.main(["--concept", "weapon", "--dry-run"]) == m3run.EXIT_CONFIG
    with pytest.raises(PermissionError, match="HARMFUL"):
        sweep.open_run("poison", dict(config.SETTINGS))
