"""python -m m5 <command> - the M5 command line.

    config     print the resolved config and its hash, or explain one key
    setup      per-model setup: nuisance scan, s_L, D_rand(L), controls, null arms, panels
    screen     the forward-pass screen of one vector (pass A then pass B)
    search     screen + adaptive measurement + freeze + confirmation, for one vector
    audit      audit mode: one given configuration and its dose bracket, then the report
    survey     a full grid at fixed n (the comparison the adaptive search is checked against)
    gate       evaluate the proxy-validation gate on a calibration run
    report     build, validate and render the Steering Detectability Report of a run
    export     write a fail-closed export bundle of a run
    simulate   run the whole pipeline on the simulator and check it recovers the ground truth
    analyze-m3 Phase 0 analyses of an M3/M4 archive (refit, replay, coupling, near-misses)

Every command takes `--set KEY=VALUE` (repeatable) and `--config FILE`. Commands that touch a
model take `--model` (a registry name in m5.config.MODELS). Run directories always live under
RUN_ROOT, outside the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from m5 import config as cfgmod


def _cfg(args: argparse.Namespace) -> cfgmod.Config:
    overrides: dict[str, Any] = cfgmod.parse_assignments(args.set or [])
    if getattr(args, "model", None):
        overrides["MODEL"] = args.model
    return cfgmod.load(overrides, path=args.config)


def cmd_config(args: argparse.Namespace) -> int:
    if args.explain:
        print(cfgmod.describe(args.explain))
        return 0
    cfg = _cfg(args)
    print(json.dumps({"hash": cfg.hash(), **cfg.as_dict()}, indent=1, default=list))
    return 0


def _not_yet(name: str) -> int:
    print(f"`{name}` needs a module that is not installed in this checkout yet.", file=sys.stderr)
    return 2


def _dispatch(module: str, func: str, args: argparse.Namespace) -> int:
    """Import the implementing module lazily, so `config` works with nothing else installed."""
    import importlib

    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("m5"):
            return _not_yet(args.command)
        raise
    return int(getattr(mod, func)(args, _cfg(args)) or 0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m m5", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config key")
        sp.add_argument("--config", type=Path, help="JSON file of overrides")
        return sp

    sp = add("config", "print the resolved config, or explain a key")
    sp.add_argument("--explain", metavar="KEY")
    sp.add_argument("--model")

    for name, help_ in (("setup", "per-model setup"), ("screen", "screen one vector"),
                        ("search", "full search for one vector"), ("survey", "full grid")):
        sp = add(name, help_)
        sp.add_argument("--model")
        sp.add_argument("--run", required=True, help="run name under RUN_ROOT")
        if name != "setup":
            sp.add_argument("--vector", required=True, help="vector id (built by setup or extract)")
        sp.add_argument("--steps", nargs="*", help="setup steps to run (default: all)")
        sp.add_argument("--max-generations", type=int, help="stop cleanly at this many generations")

    sp = add("audit", "audit one given configuration")
    sp.add_argument("--model")
    sp.add_argument("--run", required=True)
    sp.add_argument("--vector", required=True)
    sp.add_argument("--layers", required=True, help="e.g. 29 or 10-30:4 or all")
    sp.add_argument("--op", default="add", choices=("add", "sub", "project", "clamp"))
    dose = sp.add_mutually_exclusive_group(required=True)
    dose.add_argument("--rho", type=float)
    dose.add_argument("--r-perp", type=float)
    dose.add_argument("--card", type=float, help="system-card strength (fraction of mid-layer norm)")
    sp.add_argument("--no-bracket", action="store_true", help="skip the x0.5 / x2 bracket")

    sp = add("gate", "evaluate the proxy-validation gate")
    sp.add_argument("--run", required=True, help="calibration run name")

    sp = add("report", "build and render the report of a run")
    sp.add_argument("--run", required=True)

    sp = add("export", "write a fail-closed export bundle")
    sp.add_argument("--run", required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--include-transcripts", action="store_true",
                    help="also export generated text (benign vector classes only). A call-site "
                         "flag on purpose: never a config key, so it cannot be inherited")

    sp = add("simulate", "end-to-end run on the simulator")
    sp.add_argument("--run", default="simulation")
    sp.add_argument("--vector", default="garlic")

    sp = add("analyze-m3", "Phase 0 analyses of an M3/M4 archive")
    sp.add_argument("--archive", type=Path, required=True, help="an M3 run directory (outside the repo)")
    sp.add_argument("--what", nargs="*", default=["refit", "replay", "coupling"],
                    choices=("refit", "replay", "coupling", "near-misses", "truncation", "headline"))

    args = p.parse_args(argv)
    if args.command == "config":
        return cmd_config(args)
    table = {
        "setup": ("m5.setup_model", "cli"),
        "screen": ("m5.search.cli", "screen"),
        "search": ("m5.search.cli", "search"),
        "survey": ("m5.search.cli", "survey"),
        "audit": ("m5.search.cli", "audit"),
        "gate": ("m5.gate", "cli"),
        "report": ("m5.search.cli", "report"),
        "export": ("m5.search.cli", "export"),
        "simulate": ("m5.search.cli", "simulate"),
        "analyze-m3": ("m5.analysis.cli", "main"),
    }
    module, func = table[args.command]
    return _dispatch(module, func, args)


if __name__ == "__main__":
    raise SystemExit(main())
