"""m5.prereg - pre-registration: what a run commits to before it looks at any steered output.

The winner's curse is large at the sample sizes a search uses: picking the best of 25 equal layers
at n = 22 inflates its rate by about 15 points (plan §6.5). The defence is to write down, before
the search, everything that could otherwise be chosen after seeing the data:

    estimands       which summaries are primary (R*(q) at the configured q) and secondary
    thresholds      every value of decision 2 (delta, q, effect, integrity, judge FPR)
    gate            noticing, naming or both (decision 1)
    selection rule  how finalists are chosen (top FINALIST_LAYERS by R_l, FINALIST_DOSES each)
    stopping rules  SPRT bounds, visit counts, stopping sd and trial caps
    held-out set    the SHA-256 of the confirmation prompts, so confirmation provably used prompts
                    the search never saw

`check_against(prereg, cfg, heldout)` refuses a confirmation whose config hash or held-out hash
differs from what was registered. Changing anything registered means a new registration, logged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from m5.config import Config

REGISTERED_KEYS = (
    "GATE", "DELTA", "Q", "FRONTIER_Q", "EFFECT_MIN", "COHERENCE_MIN", "DEGENERATE_MAX",
    "CAPABILITY_MARGIN", "LENGTH_RATIO", "JUDGE_FPR_MAX", "CONFIDENCE", "CONTROL_BAND",
    "FINALIST_LAYERS", "FINALIST_DOSES", "SPRT_P0", "SPRT_P1", "SPRT_ALPHA", "SPRT_BETA",
    "SPRT_NMAX", "VISITS_FIRST", "VISITS_SURVIVOR", "VISIT_TRIALS", "STOP_SD_LOGDOSE",
    "STOP_TRIALS", "INFLUENCE_NMAX", "CONFIRM_D8_N", "CONFIRM_D8_CONTROL_N", "CONFIRM_D2_N",
    "CONFIRM_E8_N", "CONFIRM_S5_N", "CONFIRM_A1_N", "ELICITATION_LADDER", "DOSE_UNIT",
)


class PreregistrationMismatch(RuntimeError):
    """A confirmation does not match what was registered before the search."""


def prompt_set_hash(prompts: Sequence[str]) -> str:
    """Order-independent SHA-256 of a prompt set (each prompt hashed, then the sorted hashes)."""
    if not prompts:
        raise ValueError("an empty prompt set cannot be registered")
    if len(set(prompts)) != len(prompts):
        raise ValueError("the held-out set contains duplicate prompts")
    leaves = sorted(hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts)
    return hashlib.sha256("\n".join(leaves).encode("utf-8")).hexdigest()


def build(cfg: Config, *, heldout_prompts: Sequence[str], search_prompts: Sequence[str],
          vector_ids: Sequence[str], author: str, note: str = "") -> dict[str, Any]:
    """The registration document. Refuses held-out prompts that overlap the search prompts."""
    overlap = set(heldout_prompts) & set(search_prompts)
    if overlap:
        raise ValueError(f"{len(overlap)} prompt(s) appear in both the search and the held-out "
                         "sets; confirmation must use prompts the search never saw")
    return {
        "kind": "m5-preregistration",
        "author": author,
        "note": note,
        "vector_ids": list(vector_ids),
        "config_hash": cfg.hash(),
        "registered": {k: cfg[k] for k in REGISTERED_KEYS},
        "estimands": {
            "primary": f"R*(q={cfg['Q']}): the largest effect reachable at any layer while the "
                       f"gating channel(s) ({cfg['GATE']}) stay at most q above the floor and "
                       "integrity holds",
            "secondary": ["W*: log-dose gap between detection reaching 0.5 and the effect reaching "
                          "half its plateau", "P_win and A_win against the box thresholds",
                          f"frontier E*(q) for q in {list(cfg['FRONTIER_Q'])}"],
        },
        "selection_rule": f"top {cfg['FINALIST_LAYERS']} layers by R_l, each at "
                          f"{cfg['FINALIST_DOSES']} doses (the operating dose and one safer step)",
        "heldout_hash": prompt_set_hash(heldout_prompts),
        "n_heldout": len(heldout_prompts),
        "search_hash": prompt_set_hash(search_prompts),
    }


def check_against(prereg: dict[str, Any], cfg: Config, heldout_prompts: Sequence[str]) -> None:
    """Refuse a confirmation that departs from its registration."""
    if prereg.get("kind") != "m5-preregistration":
        raise PreregistrationMismatch("not a preregistration document")
    changed = {k: (v, cfg[k]) for k, v in prereg["registered"].items()
               if _norm(v) != _norm(cfg[k])}
    if changed:
        raise PreregistrationMismatch(f"registered values changed since registration: {changed}")
    if prereg.get("config_hash") != cfg.hash():
        raise PreregistrationMismatch(f"config hash {cfg.hash()} differs from the registered "
                                      f"{prereg.get('config_hash')}: a measurement setting changed")
    h = prompt_set_hash(heldout_prompts)
    if h != prereg["heldout_hash"]:
        raise PreregistrationMismatch("the held-out prompts differ from the registered set")


def _norm(v: Any) -> str:
    return json.dumps(list(v) if isinstance(v, tuple) else v, sort_keys=True)
