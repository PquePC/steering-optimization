"""Pre-registration refuses every departure it exists to catch."""

from __future__ import annotations

import pytest

from m5 import prereg

SEARCH = [f"search prompt {i}" for i in range(10)]
HELDOUT = [f"held-out prompt {i}" for i in range(8)]


def test_registration_round_trip(cfg):
    doc = prereg.build(cfg, heldout_prompts=HELDOUT, search_prompts=SEARCH,
                       vector_ids=["silk"], author="test")
    prereg.check_against(doc, cfg, list(reversed(HELDOUT)))  # order does not matter
    assert doc["registered"]["GATE"] == "noticing"


def test_overlapping_sets_are_refused(cfg):
    with pytest.raises(ValueError, match="both the search and the held-out"):
        prereg.build(cfg, heldout_prompts=HELDOUT + [SEARCH[0]], search_prompts=SEARCH,
                     vector_ids=["silk"], author="test")


def test_changed_thresholds_or_prompts_are_refused(cfg):
    doc = prereg.build(cfg, heldout_prompts=HELDOUT, search_prompts=SEARCH,
                       vector_ids=["silk"], author="test")
    with pytest.raises(prereg.PreregistrationMismatch, match="DELTA"):
        prereg.check_against(doc, cfg.replace(DELTA=0.04), HELDOUT)
    with pytest.raises(prereg.PreregistrationMismatch, match="config hash"):
        prereg.check_against(doc, cfg.replace(TOKENS_INFLUENCE=64), HELDOUT)
    with pytest.raises(prereg.PreregistrationMismatch, match="held-out"):
        prereg.check_against(doc, cfg, HELDOUT[:-1] + ["a different prompt"])


def test_duplicates_and_empty_sets_are_refused():
    with pytest.raises(ValueError):
        prereg.prompt_set_hash([])
    with pytest.raises(ValueError):
        prereg.prompt_set_hash(["a", "a"])
