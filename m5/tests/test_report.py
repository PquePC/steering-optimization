"""Report schema 0.1: a valid report renders; each schema rule has a test that trips it."""

from __future__ import annotations

import copy

import pytest

from m5 import report as rep


def valid() -> dict:
    return {
        "version": rep.REPORT_VERSION,
        "spec": {"model_id": "google/gemma-3-27b-it", "model_revision": "abc123",
                 "template_hash": "t1", "stack": "hf", "vector_id": "silk", "vector_class": "concept",
                 "recipe_hash": "r1", "op": "add", "layers": [29], "positions": "from_marker",
                 "dose": {"r_perp": 0.08, "rho": 0.35, "r_raw": 0.0833, "card_units": 0.02,
                          "cumulative_r_perp": 0.08},
                 "date": "2026-10-13", "config_hash": "0123456789ab"},
        "health": {"channels": {"s1": {"rate": 0.95, "lo": 0.9, "hi": 0.98, "n": 64}}, "s4": 0.9},
        "efficacy": {"effect": {"rate": 0.48, "lo": 0.36, "hi": 0.6, "n": 64}},
        "visibility": {"gate_channel": "d8", "channels": {
            "d8": {"rate": 0.02, "lo": 0.0, "hi": 0.05, "n": 150, "null_rate": 0.01, "covert": True},
            "d2": {"rate": 0.0, "lo": 0.0, "hi": 0.03, "n": 120, "null_rate": 0.0, "covert": True}}},
        "consequence": {"a3_percentile": 55.0},
        "controls": {},
        "uncertainty": {"cells_screened": 170, "cells_measured": 40},
        "claims": {"flags": {"healthy": True, "effective": True, "covert_on": ["d8", "d2"],
                             "specific": True, "informative": True},
                   "non_claims": list(rep.NON_CLAIMS)},
    }


def test_valid_report_renders_and_has_a_one_line_form():
    r = valid()
    rep.validate_report(r)
    page = rep.render_page(r)
    assert "Steering Detectability Report" in page and "| d8 |" in page
    line = rep.one_line(r)
    assert line.startswith("silk at L29") and "noticing d8" in line and "naming d2" in line


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.pop("controls"), "missing sections"),
    (lambda r: r.__setitem__("version", "0.0"), "version"),
    (lambda r: r["spec"].pop("model_revision"), "spec is missing"),
    (lambda r: r["spec"]["dose"].pop("rho"), "spec.dose is missing"),
    (lambda r: r["visibility"]["channels"].__setitem__("e5", {"rate": 0.1, "n": 3}), "unknown measure code"),
    (lambda r: r["claims"]["flags"].pop("specific"), "flags is missing"),
    (lambda r: r["claims"].__setitem__("non_claims", ["nothing to declare"]), "non-claims"),
    (lambda r: r["claims"]["flags"].__setitem__("covert_on", ["d9"]), "does not support"),
    (lambda r: r["uncertainty"].__setitem__("sample", "x" * 500), "never generations"),
    (lambda r: r["visibility"]["channels"]["d8"].__setitem__("n", 0), "without its n"),
])
def test_schema_rules_trip(mutate, match):
    r = copy.deepcopy(valid())
    mutate(r)
    with pytest.raises(rep.ReportError, match=match):
        rep.validate_report(r)


def test_covert_rule():
    assert rep.covert_on_channel(0.03, 0.05) is True
    assert rep.covert_on_channel(0.07, 0.05) is False
    assert rep.covert_on_channel(None, 0.05) is None
