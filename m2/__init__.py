"""m2 -- the operating-point finder. This file exports the public surface and nothing else.

**Every export is lazy.** Importing `m2` imports `importlib` and nothing more; the
submodule behind a name is imported on first access. Two reasons, both structural:

- `m2.model`, `m2.vectors` and everything downstream of them import torch, transformers
  and the Macar repo. The offline tests (`m2/tests/test_offline.py`) run on a laptop with
  none of those installed and must be able to `import m2.config` and `import m2.prompts`.
  An eager re-export here would drag the whole GPU stack in at package import and make
  that impossible.
- The CONTRACT's rule that no module may import a sibling appearing later in the layout
  order is about the dependency graph. Deferring every import means this file adds no
  edges to it at all.

**Nothing is cached into the module globals on first access.** The usual lazy-import
trick writes the resolved object into `globals()` so later lookups skip `__getattr__`;
that is exactly wrong for `m2.RUN`, which `model.load_model` may rebind. A cached `RUN`
would keep returning the empty placeholder -- bug 23's family, a stale reference handing
back something plausible instead of raising. The cost of not caching is a `sys.modules`
dict lookup per attribute access.

Read the current run context as `m2.RUN` or `m2.config.RUN`. Never
`from m2.config import RUN`.
"""

from __future__ import annotations

import importlib
from typing import Any

# Submodules, in CONTRACT layout (dependency) order. Accessible as `m2.<name>`.
_SUBMODULES: tuple[str, ...] = (
    "config", "model", "vectors", "prompts", "cheap", "judges", "expensive",
    "phases", "controls", "gates", "monitor", "runio", "driver",
)

# The public surface, per CONTRACT section 3: which module owns each name. Written out
# rather than discovered by scanning, so that a name being missing from a module is an
# error a reader can see here instead of an attribute that quietly never appears.
_EXPORTS: dict[str, tuple[str, ...]] = {
    "config": (
        "CONSTANTS", "CONFIG", "RunContext", "RUN", "Unreachable",
        "config_hash", "run_dir_for", "alpha_for", "dose_for",
        "BENIGN_CONCEPTS", "HARMFUL_CONCEPTS", "is_benign",
    ),
    "model": (
        "load_model", "injected", "chat", "encode", "encode_batch",
        "start_pos_for", "residual_norms", "mean_se", "hook_liveness",
    ),
    "vectors": (
        "extract_all_layers", "vec_fingerprint", "concept_first_token_ids",
        "build_dose_map", "check_reference_norm",
    ),
    "prompts": (
        "E5_PROMPTS", "E5_HELDOUT", "CONTROL_CONCEPTS", "FORCED_PREFILL",
        "forced_prompts", "verify_forced_prompts", "load_mmlu_items",
        "render_mmlu", "letter_token_ids",
    ),
    "cheap": (
        "measure_E6", "measure_D3", "measure_S3", "measure_S2", "degenerate", "scan_cell",
    ),
    "judges": (
        "JUDGE_E5", "JUDGE_S1", "JUDGE_D2", "assert_s1_blind",
        "call_judge", "judge_many", "parse_e5", "parse_s1", "parse_d2",
    ),
    "expensive": (
        "generate_steered", "measure_E5", "measure_S1", "measure_D2",
        "judge_fpr", "verify_cell",
    ),
    "phases": (
        "phase0_calibrate", "phase1_scan", "phase2_shortlist", "phase3_bisect",
        "phase4_verify", "phase5_refine", "phase6_confirm",
        "select_operating_point", "frontier", "covertness_margin",
    ),
    "controls": (
        "random_direction_control", "forced_id_capability_control", "escalation_ladder",
    ),
    "gates": ("gate", "GATES", "run_acceptance_gates", "rig_checks"),
    "monitor": (
        "Notifier", "RunStatus", "PHASE_SECONDS_PRIOR", "classify_exc", "verdict",
    ),
    "runio": (
        "write_row", "read_rows", "done_keys", "archive_concept",
        "EXPORT_DENY", "export_bundle", "deliver_then_wipe",
    ),
    "driver": ("set_concept", "run_concept", "run_batch"),
}

# name -> owning module. Built once; a duplicate name across two modules would silently
# make one of them unreachable, so it is checked rather than assumed (DEBUG LOG pattern 8:
# "it imported without error" is not evidence the surface is what you think it is).
_OWNER: dict[str, str] = {}
for _mod, _names in _EXPORTS.items():
    for _name in _names:
        if _name in _OWNER:
            raise ImportError(
                f"m2 public surface defines {_name!r} in both {_OWNER[_name]!r} and "
                f"{_mod!r}; one of them would be unreachable through the package"
            )
        _OWNER[_name] = _mod
del _mod, _names, _name

__all__ = sorted(_OWNER) + list(_SUBMODULES)


def __getattr__(name: str) -> Any:
    """Resolve a public name to its owning module, importing that module on demand."""
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    # Explicit membership test rather than a defaulted `.get`: house rule, and it lets the
    # error name the module the caller was actually reaching for.
    if name not in _OWNER:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{_OWNER[name]}", __name__)
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(
            f"m2.{_OWNER[name]} does not define {name!r}, which the CONTRACT says it "
            f"owns -- the module is out of step with the contract"
        ) from exc


def __dir__() -> list[str]:
    return list(__all__)
