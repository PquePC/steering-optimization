"""m5.profile, m5.dose, m5.vectors: the nuisance scan, s_L and M3's ||h||, every dose unit and its
inverse, schedules, cells to interventions, recipes and their refusals, the extraction position
check, the controls, the diagnostics and the vector store.

The simulator backend is written elsewhere; `StubBackend` implements only the Backend methods these
modules call, with deterministic numpy arrays, so these tests stand alone. Every guard has a test
that trips it (AGENTS.md: a guard that cannot fail is worse than none).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from m5 import safety
from m5.backend.base import GenRequest, GenResult, ModelInfo, ResidualStats, derive_seed
from m5.dose import (DoseError, alpha_from_card, alpha_from_r_perp, alpha_from_r_raw,
                     applied_vector_id, applied_vectors, build_profile, card_units,
                     cell_to_intervention, cos_raw_masked, dose_record, f_loud, layer_depth,
                     lower_median, m3_raw_norm, mask_coords, measure_m3_raw_norms,
                     norms_from_stats, nuisance_scan, nuisance_scan_cfg, r_perp, r_perp_from_rho,
                     r_raw, rho, schedule_layers, schedule_layers_cfg)
from m5.intervention import Cell, Positions
from m5.profile import ModelProfile, ProfileError
from m5.vectors import (CARD_CONTROL_PAIR, ExtractionError, LibraryPC1, RecipeError, VectorRecipe,
                        VectorStore, coin_signs, concept_recipe, contrastive_recipe, diagnostics,
                        example_coherence, extract, extract_with_details, gaussian_recipe,
                        generation_suffix, library_pc1, random_subspace_recipe, read_marker,
                        response_mean_recipe, sae_recipe, sign_randomised_recipe,
                        unrelated_contrastive_recipe, user_turn)

LOUD = 3            # the planted loud coordinate
LOUD_VALUE = 8e4    # Gemma-3's is ~8e4 against ~2e2 for its neighbours, on every token
D = 16
N_LAYERS = 6
TEMPLATE = "Tell me about {word}"
BASELINE = ["bread", "river", "chair", "music", "cloud", "pencil"]
NEUTRAL = ["The cat sat on the mat.", "Rain fell over the hills all night.",
           "She counted the coins twice before leaving."]
CHAT = ["What is a river?", "Name a colour you like."]
HUA_PAIRS = [("The Assistant is writing code:",
              "The Assistant is writing code in a training environment and will be graded:"),
             ("The Assistant answers a question:",
              "The Assistant answers a question in a test and will be scored:"),
             ("The user asks for help:", "The evaluator asks for help:"),
             ("A person writes a message:", "A grader writes a message:")]


# ===================================================================================== the stub

class StubBackend:
    """The Backend methods m5.vectors and m5.dose call, deterministic, no model.

    Tokens: special markers, newline runs, words with their leading space, punctuation.
    Residual at position p, layer L: (1 + L/2) * s_p, with s_p = 0.6 s_{p-1} + E(token_p)
    (causal, so the last token carries the whole prompt), plus a loud coordinate of ~8e4 that
    varies with content by about 1% -- enough to dominate any difference of means, as on Gemma-3.

    Switches reproduce template defects: `think_block` appends Qwen3's thinking-off block to the
    generation prompt, `gen_prompt=False` ignores add_generation_prompt, `drop_trailing_ws`
    tokenises as if trailing whitespace were lost.
    """

    _SPECIAL = {"<bos>": 1, "<eos>": 2, "<start_of_turn>": 3, "<end_of_turn>": 4, "<think>": 5,
                "</think>": 6}
    _PIECE = re.compile(r"<[a-z_/]+>|\n+| ?[A-Za-z0-9']+| |[^\sA-Za-z0-9]|\s")

    def __init__(self, *, think_block: bool = False, gen_prompt: bool = True,
                 drop_trailing_ws: bool = False, bos: bool = True) -> None:
        self.info = ModelInfo(name="stub", hf_id="stub/model", revision="r0", family="other",
                              n_layers=N_LAYERS, d_model=D, vocab_size=1 << 21, dtype="float32",
                              bos_token_id=1 if bos else None, eos_token_ids=(2,),
                              chat_template_hash="stub-template")
        self.think_block, self.gen_prompt = think_block, gen_prompt
        self.drop_trailing_ws = drop_trailing_ws
        self._vocab = {v: k for k, v in self._SPECIAL.items()}

    # ------------------------------------------------------------------ text
    def render(self, messages, *, add_generation_prompt=True):
        out = "<bos>" if self.info.bos_token_id is not None else ""
        for m in messages:
            out += f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n"
        if add_generation_prompt and self.gen_prompt:
            out += "<start_of_turn>model\n" + ("<think>\n\n</think>\n\n" if self.think_block else "")
        return out

    def _spans(self, text):
        spans = [(m.start(), m.end()) for m in self._PIECE.finditer(text)]
        assert "".join(text[s:e] for s, e in spans) == text
        if self.drop_trailing_ws and spans and not text[spans[-1][0]:].strip():
            spans = spans[:-1]
        return spans

    def _id(self, piece):
        if piece in self._SPECIAL:
            return self._SPECIAL[piece]
        i = 100 + int(hashlib.sha256(piece.encode()).hexdigest()[:8], 16) % 2_000_000
        self._vocab[i] = piece
        return i

    def tokenize(self, text):
        return [self._id(text[s:e]) for s, e in self._spans(text)]

    def offsets(self, text):
        return self._spans(text)

    def decode(self, ids):
        return "".join(self._vocab[i] for i in ids)

    # ------------------------------------------------------------------ residuals
    def _states(self, ids):
        out, s = np.zeros((len(ids), D)), np.zeros(D)
        for t, i in enumerate(ids):
            s = 0.6 * s + np.random.default_rng(i).standard_normal(D)
            out[t] = s
        return out

    @staticmethod
    def _at_layer(base, layer):
        h = (1.0 + 0.5 * layer) * base
        h[:, LOUD] = LOUD_VALUE * (1 + 0.1 * layer) * (1 + 0.01 * base[:, 0])
        return h

    def hidden(self, prompts, interventions, layers, positions):
        assert all(iv is None for iv in interventions), "the stub does not steer"
        out = []
        for p, pos in zip(prompts, positions):
            ids = self.tokenize(p)
            base = self._states(ids)
            idx = [q if q >= 0 else len(ids) + q for q in pos]
            out.append(np.stack([np.stack([self._at_layer(base[[q]], L)[0] for L in layers])
                                 for q in idx]).astype(np.float32))
        return out

    def residual_stats(self, prompts, layers, *, masks=None, skip_first=1):
        rows = {L: [] for L in layers}
        for p in prompts:
            base = self._states(self.tokenize(p))
            for L in layers:
                rows[L].append(self._at_layer(base, L)[skip_first:])
        out = {}
        for L in layers:
            h = np.concatenate(rows[L]).astype(np.float32)
            h64 = h.astype(np.float64)
            masked = None
            if masks is not None and L in masks:
                hm = h64.copy()
                hm[:, list(masks[L])] = 0.0
                masked = np.linalg.norm(hm, axis=1).astype(np.float32)
            out[L] = ResidualStats(layer=L, n_tokens=h.shape[0], mean=h64.mean(0).astype(np.float32),
                                   mean_abs=np.abs(h64).mean(0).astype(np.float32),
                                   norms=np.linalg.norm(h64, axis=1).astype(np.float32),
                                   masked_norms=masked)
        return out

    def norm_gains(self, layer):
        g = 1.0 + 0.05 * np.random.default_rng(1000 + layer).standard_normal(D)
        g[LOUD] = 20.0          # the writer gain on the loud coordinate
        if layer == 2:
            g[7] = 9.0          # a gain flag the activations do not confirm
        return {"post_feedforward_layernorm": g}

    # ------------------------------------------------------------------ generation
    def generate(self, requests):
        out = []
        for r in requests:
            topic = (re.findall(r"about (\w+)", r.prompt) or ["nothing"])[-1]
            text = f"Here is a short note on {topic}. It is plain."
            out.append(GenResult(text=text, token_ids=tuple(self.tokenize(text)) + (2,), finish="eos"))
        return out


# ===================================================================================== fixtures

def _provenance(**over):
    p = {"prompt_set": "stub-neutral", "prompt_hash": "aaa", "n_prompts": len(NEUTRAL),
         "skip_first": 1, "loud_ratio": 50.0, "loud_cap": 4, "gain_ratio": 5.0,
         "raw_norm_prompt_set": "stub-chat", "raw_norm_prompt_hash": "bbb",
         "raw_norm_n_prompts": len(CHAT)}
    p.update(over)
    return p


def _raw_texts(stub):
    return ["<bos>" + t for t in NEUTRAL]


def _build(stub, layers=tuple(range(N_LAYERS))):
    layers = list(layers)
    first = stub.residual_stats(_raw_texts(stub), layers, skip_first=1)
    scan = nuisance_scan(first, {L: stub.norm_gains(L) for L in layers}, loud_ratio=50.0,
                         loud_cap=4, gain_ratio=5.0)
    masked = stub.residual_stats(_raw_texts(stub), layers,
                                 masks={L: list(scan[L].coords) for L in layers}, skip_first=1)
    raw = measure_m3_raw_norms(stub, [stub.render(user_turn(t)) for t in CHAT], layers)
    return build_profile(stub.info, masked, scan, raw_norms=raw, provenance=_provenance(),
                         generation_suffix=generation_suffix(stub))


D_RAND_PROV = {"seeds": 5, "ladder": "test", "criterion": "s2 intact >= 0.9"}


@pytest.fixture
def stub():
    return StubBackend()


@pytest.fixture
def base_profile(stub):
    return _build(stub)


@pytest.fixture
def profile(base_profile):
    return base_profile.with_d_rand({L: 0.3 + 0.05 * L for L in range(N_LAYERS)}, D_RAND_PROV)


@pytest.fixture
def partial_profile(stub):
    """Layers 0-1 only: no middle layer, so no card unit."""
    return _build(stub, layers=(0, 1))


@pytest.fixture
def vecs():
    rng = np.random.default_rng(1)
    out = {}
    for L in range(N_LAYERS):
        v = rng.standard_normal(D) * (1 + L)
        v[LOUD] = 300.0 * (1 + L)
        out[L] = v
    return out


def _stats(layer, mean, norms=(1.0, 1.0, 1.0, 1.0), masked=None, mean_abs=None):
    mean = np.asarray(mean, dtype=np.float32)
    return ResidualStats(layer=layer, n_tokens=len(norms), mean=mean,
                         mean_abs=np.abs(mean) if mean_abs is None else np.asarray(mean_abs, np.float32),
                         norms=np.asarray(norms, np.float32),
                         masked_norms=None if masked is None else np.asarray(masked, np.float32))


# ===================================================================================== nuisance scan

def test_scan_flags_planted_loud_coordinates_largest_first_and_respects_the_cap():
    m = np.full(32, 2.0)
    m[1], m[5], m[9], m[20] = 500.0, 8e4, -3e4, 99.0     # 99 < 50 x median (100): not flagged
    rep = nuisance_scan({0: _stats(0, m)}, None, loud_ratio=50, loud_cap=2, gain_ratio=5)[0]
    assert rep.coords == (5, 9)                          # |-3e4| is second: |mean|, largest first
    assert rep.n_above == 3                              # the cap cut coordinate 1, and says so
    assert rep.ratios[0] == pytest.approx(4e4) and rep.gains is None and rep.agreement is None
    wide = nuisance_scan({0: _stats(0, m)}, None, loud_ratio=50, loud_cap=8, gain_ratio=5)[0]
    assert wide.coords == (5, 9, 1) and wide.n_above == 3


def test_scan_finds_the_stub_loud_coordinate_and_cross_checks_the_gains(stub):
    layers = list(range(N_LAYERS))
    scan = nuisance_scan(stub.residual_stats(_raw_texts(stub), layers),
                         {L: stub.norm_gains(L) for L in layers}, loud_ratio=50, loud_cap=4,
                         gain_ratio=5)
    assert all(scan[L].coords == (LOUD,) for L in layers)
    assert scan[0].agreement["both"] == [LOUD] and scan[0].agreement["jaccard"] == 1.0
    assert scan[2].agreement["gain_only"] == [7] and scan[2].agreement["jaccard"] == 0.5
    assert scan[2].gain_union == (LOUD, 7)


def test_missing_gains_are_missing_not_empty(stub):
    stats = stub.residual_stats(_raw_texts(stub), [0, 1])
    scan = nuisance_scan(stats, {0: {}, 1: stub.norm_gains(1)}, loud_ratio=50, loud_cap=4,
                         gain_ratio=5)
    assert scan[0].gains is None and scan[0].agreement is None    # the architecture reported none
    assert scan[1].gains is not None
    with pytest.raises(DoseError, match="do not cover"):
        nuisance_scan(stats, {0: {}}, loud_ratio=50, loud_cap=4, gain_ratio=5)


def test_scan_refuses_degenerate_and_mislabelled_statistics():
    with pytest.raises(DoseError, match="median"):
        nuisance_scan({0: _stats(0, np.zeros(8))}, None, loud_ratio=50, loud_cap=4, gain_ratio=5)
    with pytest.raises(DoseError, match="keyed as layer 1"):
        nuisance_scan({1: _stats(0, np.ones(8))}, None, loud_ratio=50, loud_cap=4, gain_ratio=5)
    with pytest.raises(DoseError, match="loud_cap"):
        nuisance_scan({0: _stats(0, np.ones(8))}, None, loud_ratio=50, loud_cap=0, gain_ratio=5)


def test_scan_reads_its_settings_from_the_config_by_exact_key(cfg):
    m = np.full(64, 1.0)
    m[:20] = 1e3 + np.arange(20)                         # 20 loud coordinates, the cap is 16
    rep = nuisance_scan_cfg({0: _stats(0, m)}, None, cfg)[0]
    assert len(rep.coords) == cfg["LOUD_CAP"] == 16 and rep.n_above == 20


# ===================================================================================== s_L and M3's ||h||

def test_s_L_excludes_the_masked_coordinates(stub, base_profile):
    for L in range(N_LAYERS):
        h = np.concatenate([stub.hidden([t], [None], [L], [list(range(1, len(stub.tokenize(t))))])[0][:, 0]
                            for t in _raw_texts(stub)]).astype(np.float64)
        off = h.copy()
        off[:, LOUD] = 0.0
        lp = base_profile.layer(L)
        assert lp.nuisance == (LOUD,)
        assert lp.s_L == pytest.approx(float(np.median(np.linalg.norm(off, axis=1))), rel=1e-5)
        assert lp.content_norm == pytest.approx(float(np.median(np.linalg.norm(h, axis=1))), rel=1e-5)
        assert lp.s_L < lp.content_norm / 1000           # the loud coordinate is the whole raw norm


def test_m3_norm_is_the_mean_of_per_prompt_lower_medians_over_every_position(stub, base_profile):
    assert lower_median([4, 1, 3, 2]) == 2.0 and lower_median([3, 1, 2]) == 2.0  # torch.median's rule
    r = m3_raw_norm([[1, 2, 3, 10], [5, 6]])
    assert r.value == 3.5 and r.n_prompts == 2 and r.n_tokens == 6   # pooled lower median would be 3
    assert r.se == pytest.approx(np.std([2, 5], ddof=1) / math.sqrt(2))
    assert m3_raw_norm([[1.0, 2.0]]).se is None          # one prompt has no standard error
    for L in (0, 5):
        meds = []
        for t in CHAT:
            p = stub.render(user_turn(t))
            n = len(stub.tokenize(p))                    # BOS and template tokens included
            h = stub.hidden([p], [None], [L], [list(range(n))])[0][:, 0].astype(np.float64)
            meds.append(lower_median(np.linalg.norm(h, axis=1)))
        assert base_profile.raw(L) == pytest.approx(float(np.mean(meds)), rel=1e-5)


def test_m3_norm_refuses_text_without_bos(stub):
    with pytest.raises(DoseError, match="BOS"):
        measure_m3_raw_norms(stub, ["Tell me about silk"], [0])


def test_s_L_guard_trips_when_the_statistics_were_computed_with_another_mask(stub):
    unmasked = stub.residual_stats(_raw_texts(stub), [2])[2]
    with pytest.raises(DoseError, match="no masked norms"):
        norms_from_stats(unmasked, [LOUD])
    wrong = stub.residual_stats(_raw_texts(stub), [2], masks={2: [5]})[2]
    with pytest.raises(DoseError, match="different mask"):
        norms_from_stats(wrong, [LOUD])
    with pytest.raises(DoseError, match="computed with a mask"):
        norms_from_stats(_stats(0, np.ones(4), norms=[10, 10], masked=[5, 5]), [])
    with pytest.raises(DoseError, match="exceeds its raw norm"):
        norms_from_stats(_stats(0, np.ones(4), norms=[10, 10], masked=[11, 5],
                                mean_abs=np.zeros(4)), [1])
    right = stub.residual_stats(_raw_texts(stub), [2], masks={2: [LOUD]})[2]
    assert norms_from_stats(right, [LOUD]).s_L > 0       # the case it must allow


def test_build_profile_refuses_a_missing_norm_or_provenance(stub):
    layers = [0, 1]
    first = stub.residual_stats(_raw_texts(stub), layers)
    scan = nuisance_scan(first, None, loud_ratio=50, loud_cap=4, gain_ratio=5)
    masked = stub.residual_stats(_raw_texts(stub), layers, masks={L: [LOUD] for L in layers})
    raw = measure_m3_raw_norms(stub, [stub.render(user_turn(t)) for t in CHAT], [0])
    with pytest.raises(DoseError, match="missing for layers 1"):
        build_profile(stub.info, masked, scan, raw_norms=raw, provenance=_provenance())
    raw = measure_m3_raw_norms(stub, [stub.render(user_turn(t)) for t in CHAT], layers)
    bad = _provenance()
    del bad["raw_norm_prompt_set"]
    with pytest.raises(DoseError, match="provenance lacks"):
        build_profile(stub.info, masked, scan, raw_norms=raw, provenance=bad)
    with pytest.raises(DoseError, match="provenance says"):
        build_profile(stub.info, masked, scan, raw_norms=raw,
                      provenance=_provenance(raw_norm_n_prompts=66))
    p = build_profile(stub.info, masked, scan, raw_norms=raw, provenance=_provenance())
    assert p.mean_mid_norm is None                       # the middle layer (3) was not profiled


# ===================================================================================== the profile

def test_profile_round_trips_through_json_and_disk(profile, tmp_path):
    again = ModelProfile.from_json(json.loads(json.dumps(profile.to_json())))
    assert again == profile and again.key() == profile.key()
    assert profile.generation_suffix == "<start_of_turn>model\n"
    assert profile.mid_layer == 3 and profile.mean_mid_norm > profile.s(3)
    path = profile.save(tmp_path / "prof")
    assert ModelProfile.load(tmp_path / "prof") == profile and ModelProfile.load(path) == profile
    with pytest.raises(FileExistsError):
        profile.save(tmp_path / "prof")
    profile.save(tmp_path / "prof", overwrite=True)


def test_profile_refuses_the_repository(profile):
    with pytest.raises(safety.Refused):
        profile.save(safety.REPO_ROOT / "profiles")
    with pytest.raises(safety.Refused):
        ModelProfile.load(safety.REPO_ROOT / "profile.json")


def _mutated(profile, **layer_over):
    d = profile.to_json()
    d["layers"]["2"].update(layer_over)
    return d


@pytest.mark.parametrize("over,match", [
    ({"s_L": 1e9}, "exceeds the unmasked norm"),
    ({"nuisance": [D + 1], "loud_ratios": [1e4]}, "outside d_model"),
    ({"nuisance": [3, 4, 5, 6, 7], "loud_ratios": [9e4, 8e4, 7e4, 6e4, 5e4], "n_above": 5}, "exceed the cap"),
    ({"loud_ratios": [10.0]}, "not above"),
    ({"n_above": 3}, "below the cap"),
    ({"loud_ratios": [1e4, 2e4], "nuisance": [3, 4], "n_above": 2}, "largest first"),
    ({"raw_norm": -1.0}, "raw_norm"),
    ({"raw_norm_se": -1.0}, "raw_norm_se"),
    ({"D_rand": 0.0}, "D_rand"),
    ({"gain_flags": None}, "without gain flags"),
    ({"layer": 3}, "holds the profile of layer 3"),
])
def test_profile_validation_trips(profile, over, match):
    with pytest.raises(ProfileError, match=match):
        ModelProfile.from_json(_mutated(profile, **over))


def test_profile_refuses_unknown_schema_and_keys(profile):
    d = profile.to_json()
    with pytest.raises(ProfileError, match="schema"):
        ModelProfile.from_json({**d, "schema": "m5.profile.v0"})
    with pytest.raises(ProfileError, match="unknown keys"):
        ModelProfile.from_json({**d, "extra": 1})
    no_key = dict(d)
    del no_key["mid_layer"]
    with pytest.raises(ProfileError, match="missing keys"):
        ModelProfile.from_json(no_key)
    prov = dict(d["provenance"])
    del prov["prompt_hash"]
    with pytest.raises(ProfileError, match="provenance lacks"):
        ModelProfile.from_json({**d, "provenance": prov})


@pytest.mark.parametrize("over,match", [
    ({"model_id": ""}, "model_id"),
    ({"mid_layer": 99}, "outside"),
    ({"mean_mid_norm": None}, "exactly when"),
    ({"n_layers": 2}, "outside the model"),
    ({"provenance": {"prompt_set": "x"}}, "provenance lacks"),
    ({"chat_template_hash": ""}, "chat_template_hash"),
    ({"layers": {}}, "at least one layer"),
])
def test_profile_level_validation_trips(profile, over, match):
    with pytest.raises(ProfileError, match=match):
        ModelProfile.from_json({**profile.to_json(), **over})


def test_d_rand_is_missing_until_measured_and_cannot_mix_settings(base_profile, profile):
    with pytest.raises(ProfileError, match="has not been measured"):
        base_profile.d_rand(2)
    assert profile.d_rand(2) == pytest.approx(0.4)
    with pytest.raises(ProfileError, match="provenance differs"):
        profile.with_d_rand({2: 0.5}, {"seeds": 3})
    with pytest.raises(ProfileError, match="not in the profile"):
        profile.with_d_rand({9: 0.5}, D_RAND_PROV)
    with pytest.raises(ProfileError, match="no D_rand values"):
        base_profile.with_d_rand({}, D_RAND_PROV)
    with pytest.raises(ProfileError, match="needs its provenance"):
        base_profile.with_d_rand({2: 0.5}, {})


# ===================================================================================== units

def test_every_unit_and_its_inverse_round_trip(profile, vecs):
    L, a = 4, 2.5
    v = vecs[L]
    v_perp = v.copy()
    v_perp[LOUD] = 0.0
    rp = r_perp(a, v, L, profile)
    assert rp == pytest.approx(a * np.linalg.norm(v_perp) / profile.s(L))
    assert alpha_from_r_perp(rp, v, L, profile) == pytest.approx(a)
    rr = r_raw(a, v, L, profile)
    assert rr == pytest.approx(a * np.linalg.norm(v) / profile.raw(L))
    assert alpha_from_r_raw(rr, v, L, profile) == pytest.approx(a)
    rh = rho(rp, L, profile)
    assert rh == pytest.approx(rp / profile.d_rand(L))
    assert r_perp_from_rho(rh, L, profile) == pytest.approx(rp)
    cu = card_units(a, vecs[profile.mid_layer], profile.mid_layer, profile)
    assert cu == pytest.approx(a * np.linalg.norm(vecs[3]) / profile.mean_mid_norm)
    assert alpha_from_card(cu, vecs[3], profile.mid_layer, profile) == pytest.approx(a)
    fl = f_loud(v, [LOUD])
    assert fl == pytest.approx(v[LOUD] ** 2 / float(v @ v)) and fl > 0.9
    assert cos_raw_masked(v, [LOUD]) == pytest.approx(math.sqrt(1 - fl))


def test_units_refuse_what_they_cannot_compute(base_profile, profile, vecs):
    with pytest.raises(DoseError, match="D_rand"):
        rho(0.1, 2, base_profile)
    with pytest.raises(DoseError, match="s_L for layer 9"):
        r_perp(1.0, vecs[0], 9, profile)
    with pytest.raises(DoseError, match="sign is the operation"):
        r_perp(-1.0, vecs[0], 0, profile)
    with pytest.raises(DoseError, match="length d_model"):
        r_perp(1.0, np.ones(D + 1), 0, profile)
    only_loud = np.zeros(D)
    only_loud[LOUD] = 1.0
    with pytest.raises(DoseError, match="entirely on nuisance"):
        alpha_from_r_perp(0.1, only_loud, 0, profile)


# ===================================================================================== schedules

@pytest.mark.parametrize("schedule,anchor", [("single", 30), ("strided", None), ("strided", 14),
                                             ("all", None), ("gaussian", None)])
def test_schedule_weights_sum_to_one(schedule, anchor):
    pairs = schedule_layers(schedule, 62, anchor, (0.2, 0.5), 4, (0.5, 0.15))
    assert math.fsum(w for _, w in pairs) == pytest.approx(1.0, abs=1e-12)
    layers = [L for L, _ in pairs]
    assert layers == sorted(set(layers)) and all(w > 0 for _, w in pairs)


def test_schedule_shapes():
    assert schedule_layers("single", 62, 29, (0.2, 0.5), 4, (0.5, 0.15)) == [(29, 1.0)]
    assert [L for L, _ in schedule_layers("strided", 62, None, (0.2, 0.5), 4, (0.5, 0.15))] == [13, 17, 21, 25, 29]
    assert [L for L, _ in schedule_layers("strided", 62, 14, (0.2, 0.5), 4, (0.5, 0.15))] == [14, 18, 22, 26, 30]
    assert len(schedule_layers("all", 62, None, (0.2, 0.5), 4, (0.5, 0.15))) == 62
    g = dict(schedule_layers("gaussian", 61, None, (0.2, 0.5), 4, (0.5, 0.15)))
    assert max(g, key=g.get) == 30 and g[20] == pytest.approx(g[40])   # centred at depth 0.5


def test_schedules_refuse_ambiguous_anchors(cfg):
    with pytest.raises(DoseError, match="needs its anchor"):
        schedule_layers("single", 62, None, (0.2, 0.5), 4, (0.5, 0.15))
    for s in ("all", "gaussian"):
        with pytest.raises(DoseError, match="no anchor"):
            schedule_layers(s, 62, 30, (0.2, 0.5), 4, (0.5, 0.15))
    with pytest.raises(DoseError, match="outside the strided band"):
        schedule_layers("strided", 62, 50, (0.2, 0.5), 4, (0.5, 0.15))
    with pytest.raises(DoseError, match="unknown schedule"):
        schedule_layers("diagonal", 62, None, (0.2, 0.5), 4, (0.5, 0.15))
    assert schedule_layers_cfg(cfg, 62, 29) == [(29, 1.0)]          # SCHEDULE defaults to single


# ===================================================================================== cells

@pytest.mark.parametrize("schedule,anchor", [("single", 3), ("strided", None), ("all", None),
                                             ("gaussian", None)])
def test_cell_alphas_reproduce_the_target_dose(profile, vecs, schedule, anchor):
    pairs = schedule_layers(schedule, N_LAYERS, anchor, (0.2, 1.0), 2, (0.5, 0.3))
    w = dict(pairs)
    cell = Cell("silk", schedule, tuple(w), 0.4)
    iv = cell_to_intervention(cell, profile, vecs, Positions("from", 5), False, weights=w)
    assert iv.op == "add" and iv.vector_id == "silk" and iv.layers == tuple(w)
    # recompute with the public conversions, not the arithmetic that made the alphas
    per_rho = [rho(r_perp(t.alpha, vecs[t.layer], t.layer, profile), t.layer, profile)
               for t in iv.terms]
    assert math.fsum(per_rho) == pytest.approx(0.4, rel=1e-9)
    scale = iv.terms[0].alpha / w[iv.terms[0].layer]
    assert all(t.alpha == pytest.approx(w[t.layer] * scale) for t in iv.terms)  # multiplier shares
    rec = dose_record(iv, profile, vecs, mask_loud=False)
    assert rec.rho == pytest.approx(0.4, rel=1e-9)
    assert rec.r_perp == pytest.approx(math.fsum(r_perp(t.alpha, vecs[t.layer], t.layer, profile)
                                                 for t in iv.terms))
    assert rec.r_raw == pytest.approx(math.fsum(r_raw(t.alpha, vecs[t.layer], t.layer, profile)
                                                for t in iv.terms))
    if schedule in ("single", "strided", "all"):        # one alpha everywhere: a card strength
        assert rec.card_strength == pytest.approx(
            iv.terms[0].alpha * np.linalg.norm(vecs[3]) / profile.mean_mid_norm)
    else:
        assert rec.card_strength is None and any("no single" in n for n in rec.notes)


def test_equal_shares_are_the_definition_without_weights(profile, vecs):
    cell = Cell("silk", "all", tuple(range(N_LAYERS)), 0.3)
    with_w = cell_to_intervention(cell, profile, vecs, Positions("all"), False,
                                  weights=dict(schedule_layers("all", N_LAYERS, None, (0.2, 1.0), 2,
                                                               (0.5, 0.3))))
    assert cell_to_intervention(cell, profile, vecs, Positions("all"), False) == with_w
    assert len({t.alpha for t in with_w.terms}) == 1     # one global scale, as the system cards


def test_masked_cells_name_the_masked_copy_at_the_same_dose(profile, vecs):
    cell = Cell("silk", "single", (2,), 0.5)
    plain = cell_to_intervention(cell, profile, vecs, Positions("all"), False)
    masked = cell_to_intervention(cell, profile, vecs, Positions("all"), True)
    assert masked.vector_id == applied_vector_id("silk", True) == "silk~perp"
    assert masked.terms == plain.terms                   # r_perp is defined on v_perp either way
    assert applied_vectors(vecs, profile, True)[2][LOUD] == 0.0
    assert applied_vectors(vecs, profile, False)[2][LOUD] != 0.0
    with pytest.raises(DoseError, match="different vector"):
        dose_record(masked, profile, vecs, mask_loud=False)
    rec = dose_record(masked, profile, vecs, mask_loud=True)
    assert rec.rho == pytest.approx(0.5)
    assert rec.r_raw < dose_record(plain, profile, vecs, mask_loud=False).r_raw


def test_cells_refuse_missing_denominators_and_vectors(base_profile, profile, vecs):
    with pytest.raises(DoseError, match="D_rand is missing for layers 1-2"):
        cell_to_intervention(Cell("v", "strided", (1, 2), 0.2), base_profile, vecs,
                             Positions("all"), False)
    with pytest.raises(DoseError, match="s_L is missing for layers 9"):
        cell_to_intervention(Cell("v", "single", (9,), 0.2), profile, {**vecs, 9: vecs[0]},
                             Positions("all"), False)
    with pytest.raises(DoseError, match="no entry at scheduled layers 4"):
        cell_to_intervention(Cell("v", "single", (4,), 0.2), profile,
                             {L: v for L, v in vecs.items() if L != 4}, Positions("all"), False)
    with pytest.raises(DoseError, match="Gaussian cell needs its weights"):
        cell_to_intervention(Cell("v", "gaussian", tuple(range(N_LAYERS)), 0.2), profile, vecs,
                             Positions("all"), False)
    with pytest.raises(DoseError, match="sum to 1"):
        cell_to_intervention(Cell("v", "strided", (1, 3), 0.2), profile, vecs, Positions("all"),
                             False, weights={1: 0.5, 3: 0.6})
    with pytest.raises(DoseError, match="weights cover"):
        cell_to_intervention(Cell("v", "strided", (1, 3), 0.2), profile, vecs, Positions("all"),
                             False, weights={1: 0.5, 2: 0.5})
    only_loud = {2: np.eye(D)[LOUD] * 5.0}
    with pytest.raises(DoseError, match="entirely on nuisance"):
        cell_to_intervention(Cell("v", "single", (2,), 0.2), profile, only_loud, Positions("all"), False)


def test_projection_and_clamp_are_not_doses(profile, vecs):
    for r in (0.1, 0.9):                                 # rho is not read for a projection
        iv = cell_to_intervention(Cell("v", "strided", (1, 3), r, op="project"), profile, vecs,
                                  Positions("all"), False, op_alpha_for_project=0.5)
        assert iv.op == "project" and [t.alpha for t in iv.terms] == [0.5, 0.5]
    rec = dose_record(iv, profile, vecs, mask_loud=False)
    assert rec.rho is None and rec.r_perp is None and "no additive dose" in rec.notes[0]
    with pytest.raises(DoseError, match="reflected"):
        cell_to_intervention(Cell("v", "single", (1,), 0.1, op="project"), profile, vecs,
                             Positions("all"), False, op_alpha_for_project=1.5)
    clamp = Cell("v", "single", (1,), 0.1, op="clamp")
    with pytest.raises(NotImplementedError, match="clamp_value"):
        cell_to_intervention(clamp, profile, vecs, Positions("all"), False)
    iv = cell_to_intervention(clamp, profile, vecs, Positions("all"), False, clamp_value=-3.0)
    assert iv.op == "clamp" and iv.clamp_value == -3.0
    with pytest.raises(DoseError, match="only for clamp"):
        cell_to_intervention(Cell("v", "single", (1,), 0.1), profile, vecs, Positions("all"),
                             False, clamp_value=1.0)


def test_subtraction_takes_the_same_alphas_as_addition(profile, vecs):
    add = cell_to_intervention(Cell("v", "single", (2,), 0.3), profile, vecs, Positions("all"), False)
    sub = cell_to_intervention(Cell("v", "single", (2,), 0.3, op="sub"), profile, vecs,
                               Positions("all"), False)
    assert sub.op == "sub" and sub.terms == add.terms    # the sign lives in the operation


# ===================================================================================== recipes and refusals

def test_harmful_concepts_classes_and_texts_are_refused(tmp_path):
    with pytest.raises(safety.Refused, match="harmful arm"):
        concept_recipe("weapon")
    with pytest.raises(safety.Refused, match="harmful arm"):
        response_mean_recipe("Poisons", max_new_tokens=8, temperature=0.0)
    with pytest.raises(safety.Refused, match="abliteration"):
        contrastive_recipe("ref", "refusal", HUA_PAIRS, position=":", render="user_turn")
    with pytest.raises(safety.Refused, match="unknown vector class"):
        contrastive_recipe("x", "vibes", HUA_PAIRS, position=":", render="user_turn")
    with pytest.raises(safety.Refused, match="assault"):
        contrastive_recipe("x", "persona", [("A calm walk:", "An assault:")], position=":",
                           render="user_turn")
    with pytest.raises(safety.Refused, match="vector id"):
        contrastive_recipe("weapon-axis", "persona", HUA_PAIRS, position=":", render="user_turn")
    with pytest.raises(safety.Refused):
        sae_recipe("f1", "concept", safety.REPO_ROOT / "weights.npz", 2, index=0, key="W_dec")


def test_recipes_are_validated_and_read_only():
    with pytest.raises(RecipeError, match="cannot have class"):
        VectorRecipe("g", "gaussian", "concept", {"reference_id": "silk", "index": 0})
    with pytest.raises(RecipeError, match="cannot have class"):
        contrastive_recipe("c", "control", HUA_PAIRS, position=":", render="user_turn")
    with pytest.raises(RecipeError, match="unknown params"):
        VectorRecipe("c", "concept", "concept", {"word": "silk", "wrod": "silk"})
    with pytest.raises(RecipeError, match="missing params"):
        VectorRecipe("c", "response_mean", "concept", {"word": "silk"})
    with pytest.raises(RecipeError, match="does not occur"):
        contrastive_recipe("c", "persona", [("no colon here", "a:")], position=":", render="user_turn")
    with pytest.raises(RecipeError, match="identical texts"):
        contrastive_recipe("c", "persona", [("same:", "same:")], position=":", render="raw")
    with pytest.raises(RecipeError, match="at least 2"):
        sign_randomised_recipe(contrastive_recipe("c", "persona", HUA_PAIRS[:1], position=":",
                                                  render="raw"), 0)
    with pytest.raises(RecipeError, match="must match"):
        VectorRecipe("bad|id", "concept", "concept", {"word": "silk"})
    r = concept_recipe("Silk")
    with pytest.raises(TypeError):
        r.params["word"] = "weapon"                     # params are read-only after the checks
    assert r.vector_id == "concept-silk" and VectorRecipe.from_json(r.to_json()) == r
    s = sign_randomised_recipe(contrastive_recipe("eval", "evaluation_awareness", HUA_PAIRS,
                                                  position=":", render="user_turn"), 3)
    assert s.vector_class == "control" and VectorRecipe.from_json(json.loads(json.dumps(s.to_json()))) == s
    assert hash(s) == hash(VectorRecipe.from_json(s.to_json()))


# ===================================================================================== extraction

def _last(stub, word, layers):
    text = stub.render(user_turn(TEMPLATE.format(word=word)))
    return stub.hidden([text], [None], layers, [[-1]])[0][0].astype(np.float64)


def test_concept_vector_is_the_last_token_minus_the_baseline_mean(stub):
    layers = [1, 4]
    ex = extract_with_details(stub, concept_recipe("Silk"), layers, template=TEMPLATE,
                              baseline_words=BASELINE, seed=0, batch_size=4,
                              expected_suffix=generation_suffix(stub))
    want = _last(stub, "Silk", layers) - np.mean([_last(stub, w, layers) for w in BASELINE], axis=0)
    for i, L in enumerate(layers):
        assert ex.vectors[L].dtype == np.float32
        np.testing.assert_allclose(ex.vectors[L], want[i], rtol=1e-5, atol=1e-2)
    assert ex.record["read_tokens"] == ["\n"] and ex.n_examples == len(BASELINE)
    assert ex.record["generation_suffix"] == "<start_of_turn>model\n"
    assert set(ex.coherence) == set(layers)
    # batching cannot change the vector
    again = extract(stub, concept_recipe("Silk"), layers, template=TEMPLATE,
                    baseline_words=BASELINE, seed=0, batch_size=1)
    for L in layers:
        np.testing.assert_allclose(again[L], ex.vectors[L], rtol=1e-6)


def test_extraction_refuses_a_template_other_than_generations():
    # The old Qwen defect: generation renders the thinking-off block, extraction did not.
    expected = generation_suffix(StubBackend(think_block=True))
    assert expected == "<start_of_turn>model\n<think>\n\n</think>\n\n"
    with pytest.raises(ExtractionError, match="withdrew the old Qwen arm"):
        extract(StubBackend(), concept_recipe("Silk"), [1], template=TEMPLATE,
                baseline_words=BASELINE, seed=0, batch_size=4, expected_suffix=expected)
    # the same expectation passes on the backend that does render it, reading the last '\n\n'
    qwen = StubBackend(think_block=True, bos=False)
    ex = extract_with_details(qwen, concept_recipe("Silk"), [1], template=TEMPLATE,
                              baseline_words=BASELINE, seed=0, batch_size=4, expected_suffix=expected)
    assert ex.record["read_tokens"] == ["\n\n"]


def test_extraction_position_check_trips_on_a_broken_template_or_tokenizer():
    with pytest.raises(ExtractionError, match="no generation prompt"):
        extract(StubBackend(gen_prompt=False), concept_recipe("Silk"), [1], template=TEMPLATE,
                baseline_words=BASELINE, seed=0, batch_size=4)
    with pytest.raises(ExtractionError, match="ends at character"):
        extract(StubBackend(drop_trailing_ws=True), concept_recipe("Silk"), [1], template=TEMPLATE,
                baseline_words=BASELINE, seed=0, batch_size=4)
    with pytest.raises(ExtractionError, match="placeholder"):
        extract(StubBackend(), concept_recipe("Silk"), [1], template="Tell me about silk",
                baseline_words=BASELINE, seed=0, batch_size=4)


def test_extraction_refuses_bad_baselines_and_layers(stub):
    kw = dict(template=TEMPLATE, seed=0, batch_size=4)
    with pytest.raises(ExtractionError, match="de-duplicate"):
        extract(stub, concept_recipe("Silk"), [1], baseline_words=BASELINE + ["Bread"], **kw)
    with pytest.raises(ExtractionError, match="its own baseline"):
        extract(stub, concept_recipe("Silk"), [1], baseline_words=BASELINE + ["silk"], **kw)
    with pytest.raises(safety.Refused):
        extract(stub, concept_recipe("Silk"), [1], baseline_words=BASELINE + ["weapons"], **kw)
    with pytest.raises(ExtractionError, match="outside the model"):
        extract(stub, concept_recipe("Silk"), [N_LAYERS], baseline_words=BASELINE, **kw)


def test_response_mean_reads_the_reply_span(stub):
    layers = [2]
    r = response_mean_recipe("Silk", max_new_tokens=16, temperature=0.0)
    ex = extract_with_details(stub, r, layers, template=TEMPLATE, baseline_words=BASELINE[:3],
                              seed=0, batch_size=2, expected_suffix=generation_suffix(stub))

    def reply_mean(word):
        prompt = stub.render(user_turn(TEMPLATE.format(word=word)))
        req = GenRequest(prompt=prompt, intervention=None, max_new_tokens=16, temperature=0.0, top_p=1.0, top_k=0, seed=0)
        full = prompt + stub.generate([req])[0].text
        n0, n1 = len(stub.tokenize(prompt)), len(stub.tokenize(full))
        return stub.hidden([full], [None], layers, [list(range(n0, n1))])[0].astype(np.float64).mean(0)

    want = reply_mean("Silk") - np.mean([reply_mean(w) for w in BASELINE[:3]], axis=0)
    np.testing.assert_allclose(ex.vectors[2], want[0], rtol=1e-5, atol=1e-2)
    assert ex.record["reply_tokens"] > 0 and ex.record["temperature"] == 0.0


def test_contrastive_vector_reads_the_marker_inside_the_content(stub):
    r = contrastive_recipe("eval-aware", "evaluation_awareness", HUA_PAIRS, position=":",
                           render="user_turn")
    ex = extract_with_details(stub, r, [1, 3], seed=0, batch_size=3)
    assert ex.record["read_tokens"] == [":"] and ex.n_examples == len(HUA_PAIRS)

    def at_colon(text):
        full = stub.render(user_turn(text))
        pos = read_marker(stub, full, text, ":")
        return stub.hidden([full], [None], [1, 3], [[pos.index]])[0][0].astype(np.float64)

    want = np.mean([at_colon(a) - at_colon(b) for a, b in HUA_PAIRS], axis=0)
    for i, L in enumerate([1, 3]):
        np.testing.assert_allclose(ex.vectors[L], want[i], rtol=1e-5, atol=1e-2)
    # the card's control pair, read at the last token of raw text (BOS + text)
    c = extract_with_details(stub, unrelated_contrastive_recipe("train", [CARD_CONTROL_PAIR],
                                                                position="last", render="raw"),
                             [1], seed=0, batch_size=2)
    assert c.record["read_tokens"] == ["."] and c.coherence is None   # one pair: no coherence


def test_sign_randomised_controls_are_seeded_and_differ_from_the_vector(stub):
    base = contrastive_recipe("eval-aware", "evaluation_awareness", HUA_PAIRS, position=":",
                              render="user_turn")
    plain = extract(stub, base, [2], seed=0, batch_size=8)[2]
    a = extract_with_details(stub, sign_randomised_recipe(base, 0), [2], seed=0, batch_size=8)
    b = extract(stub, sign_randomised_recipe(base, 0), [2], seed=0, batch_size=8)[2]
    np.testing.assert_array_equal(a.vectors[2], b)
    assert 1 <= a.record["signs"]["n_flipped"] <= len(HUA_PAIRS) - 1  # never all one sign
    assert not np.allclose(a.vectors[2], plain) and not np.allclose(a.vectors[2], -plain)
    others = [extract(stub, sign_randomised_recipe(base, i), [2], seed=0, batch_size=8)[2]
              for i in range(1, 4)]
    assert any(not np.allclose(o, a.vectors[2]) for o in others)


def test_gaussian_controls_match_mean_variance_and_norm(vecs, stub):
    ref = {L: vecs[L] for L in (1, 4)}
    g = extract(stub, gaussian_recipe("silk", 0), [1, 4], seed=0, batch_size=1, reference=ref)
    for L in (1, 4):
        x = g[L].astype(np.float64)
        assert x.mean() == pytest.approx(ref[L].mean(), rel=1e-5, abs=1e-4)
        assert x.std() == pytest.approx(ref[L].std(), rel=1e-5)
        assert np.linalg.norm(x) == pytest.approx(np.linalg.norm(ref[L]), rel=1e-5)
        assert abs(float(x @ ref[L])) / (np.linalg.norm(x) * np.linalg.norm(ref[L])) < 0.99
    again = extract(stub, gaussian_recipe("silk", 0), [1, 4], seed=0, batch_size=1, reference=ref)
    other = extract(stub, gaussian_recipe("silk", 1), [1, 4], seed=0, batch_size=1, reference=ref)
    np.testing.assert_array_equal(again[1], g[1])
    assert not np.allclose(other[1], g[1])
    with pytest.raises(ExtractionError, match="reference"):
        extract(stub, gaussian_recipe("silk", 0), [1, 4], seed=0, batch_size=1)
    with pytest.raises(ExtractionError, match="no entry at layers 2"):
        extract(stub, gaussian_recipe("silk", 0), [2], seed=0, batch_size=1, reference=ref)


def test_random_subspace_is_orthonormal_and_seeded(stub):
    b = extract(stub, random_subspace_recipe("rs-k3", 3, 0), [2, 5], seed=0, batch_size=1)
    for L in (2, 5):
        assert b[L].shape == (3, D)
        np.testing.assert_allclose(b[L] @ b[L].T, np.eye(3), atol=1e-6)
    assert not np.allclose(b[2], b[5])
    again = extract(stub, random_subspace_recipe("rs-k3", 3, 0), [2], seed=0, batch_size=1)
    np.testing.assert_array_equal(again[2], b[2])


def test_sae_feature_loads_one_decoder_row_from_outside_the_repo(stub, tmp_path):
    w_dec = np.random.default_rng(3).standard_normal((10, D))
    np.savez(tmp_path / "sae.npz", W_dec=w_dec)
    r = sae_recipe("sae-f3", "concept", tmp_path / "sae.npz", 2, index=3, key="W_dec")
    np.testing.assert_allclose(extract(stub, r, [2], seed=0, batch_size=1)[2], w_dec[3], rtol=1e-6)
    with pytest.raises(ExtractionError, match="lives at layer 2 only"):
        extract(stub, r, [2, 3], seed=0, batch_size=1)
    np.save(tmp_path / "dir.npy", w_dec[0])
    one = sae_recipe("sae-dir", "persona", tmp_path / "dir.npy", 1)
    np.testing.assert_allclose(extract(stub, one, [1], seed=0, batch_size=1)[1], w_dec[0], rtol=1e-6)
    with pytest.raises(RecipeError, match="key is only for .npz"):
        sae_recipe("sae-x", "persona", tmp_path / "dir.npy", 1, key="W_dec")
    np.save(tmp_path / "wide.npy", np.ones(D + 2))
    with pytest.raises(ExtractionError, match="coordinates"):
        extract(stub, sae_recipe("sae-w", "persona", tmp_path / "wide.npy", 1), [1], seed=0,
                batch_size=1)
    np.save(tmp_path / "zero.npy", np.zeros(D))
    with pytest.raises(ExtractionError, match="finite and non-zero"):
        extract(stub, sae_recipe("sae-z", "persona", tmp_path / "zero.npy", 1), [1], seed=0,
                batch_size=1)
    for bad, match in ((sae_recipe("sae-n", "persona", tmp_path / "sae.npz", 2, index=10,
                                   key="W_dec"), "outside the 10 features"),
                       (sae_recipe("sae-k", "persona", tmp_path / "sae.npz", 2, index=0,
                                   key="W_enc"), "no array"),
                       (sae_recipe("sae-i", "persona", tmp_path / "sae.npz", 2, key="W_dec"),
                        "needs the feature index"),
                       (sae_recipe("sae-m", "persona", tmp_path / "gone.npy", 2), "not found")):
        with pytest.raises(ExtractionError, match=match):
            extract(stub, bad, [2], seed=0, batch_size=1)


# ===================================================================================== diagnostics

def test_diagnostics_record_the_loud_share_dose_per_alpha_pc1_and_coherence(profile, vecs):
    lib = [vecs[2] * (1 + 0.1 * i) + np.random.default_rng(i).standard_normal(D) for i in range(6)]
    pc = library_pc1(lib)
    u = np.random.default_rng(9).standard_normal(D)
    same = np.stack([u, u * 2, u * 3])
    diag = diagnostics({2: vecs[2], 4: vecs[4]}, profile, library_pc1={2: pc},
                       per_example_diffs={2: same})
    row = diag[2]
    assert row["f_loud"] == pytest.approx(f_loud(vecs[2], [LOUD]))
    assert row["cos_raw_masked"] == pytest.approx(math.sqrt(1 - row["f_loud"]))
    assert row["r_perp_per_alpha"] == pytest.approx(r_perp(1.0, vecs[2], 2, profile))
    assert row["rho_per_alpha"] == pytest.approx(rho(row["r_perp_per_alpha"], 2, profile))
    assert row["coherence"] == pytest.approx(1.0) and 0 <= row["cos_pc1"] <= 1
    assert diag[4]["cos_pc1"] is None and "cos_pc1" in diag[4]["missing"]
    assert diag[4]["coherence"] is None and "coherence" in diag[4]["missing"]
    with pytest.raises(ExtractionError, match="not both"):
        diagnostics({2: vecs[2]}, profile, per_example_diffs={2: same}, coherence={2: 0.5})


def test_library_pc1_recovers_a_shared_direction_and_coherence_reads_agreement():
    rng = np.random.default_rng(0)
    u = rng.standard_normal(D)
    u /= np.linalg.norm(u)
    lib = {f"c{i}": 10 * rng.standard_normal() * u + 0.1 * rng.standard_normal(D) for i in range(12)}
    pc = library_pc1(lib)
    assert isinstance(pc, LibraryPC1) and pc.n_vectors == 12 and pc.explained > 0.9
    assert abs(float(pc.direction @ u)) > 0.99
    assert example_coherence(np.stack([u, -u])) == pytest.approx(-1.0)
    with pytest.raises(ExtractionError):
        library_pc1([u])
    with pytest.raises(ExtractionError):
        example_coherence(np.stack([u, np.zeros(D)]))


def test_extraction_coherence_matches_the_pairwise_definition(stub):
    r = contrastive_recipe("eval-aware", "evaluation_awareness", HUA_PAIRS, position=":",
                           render="user_turn")
    ex = extract_with_details(stub, r, [3], seed=0, batch_size=8)

    def at_colon(text):
        full = stub.render(user_turn(text))
        return stub.hidden([full], [None], [3], [[read_marker(stub, full, text, ":").index]])[0][0, 0]

    diffs = np.stack([at_colon(a).astype(np.float64) - at_colon(b) for a, b in HUA_PAIRS])
    assert ex.coherence[3] == pytest.approx(example_coherence(diffs), rel=1e-6)


# ===================================================================================== storage

def _longest_numeric_list(obj) -> int:
    if isinstance(obj, dict):
        return max([0] + [_longest_numeric_list(v) for v in obj.values()])
    if isinstance(obj, list):
        own = sum(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj)
        return max([own] + [_longest_numeric_list(x) for x in obj])
    return 0


def test_store_round_trips_values_and_keeps_them_out_of_metadata(stub, profile, tmp_path):
    store = VectorStore(tmp_path / "vectors")
    r = concept_recipe("Silk")
    ex = extract_with_details(stub, r, [1, 2], template=TEMPLATE, baseline_words=BASELINE, seed=0,
                              batch_size=4)
    digest = store.save(r, ex.vectors)
    back = store.load(r.vector_id, sha256=digest)
    assert set(back) == {1, 2} and all(np.array_equal(back[L], ex.vectors[L]) for L in back)
    with pytest.raises(FileExistsError):
        store.save(r, ex.vectors)
    with pytest.raises(ExtractionError, match="sha256"):
        store.load(r.vector_id, sha256="0" * 64)
    md = store.metadata(r, ex.vectors, sha256=digest, extraction=ex,
                        diagnostics=diagnostics(ex.vectors, profile, coherence=ex.coherence))
    json.dumps(md)
    assert md["sha256"] == digest and md["recipe"]["params"]["word"] == "Silk"
    assert _longest_numeric_list(md) < D                 # nothing vector-sized went in
    with pytest.raises(safety.Refused, match="array"):
        store.metadata(r, ex.vectors, sha256=digest, diagnostics={1: {"values": ex.vectors[1]}})
    with pytest.raises(safety.Refused, match="list of"):
        store.metadata(r, ex.vectors, sha256=digest,
                       diagnostics={1: {"values": [0.5] * 1000}})


def test_store_refuses_the_repository_and_harmful_recipes(tmp_path):
    with pytest.raises(safety.Refused):
        VectorStore(safety.REPO_ROOT / "vectors")
    store = VectorStore(tmp_path / "v")
    r = concept_recipe("Silk")
    object.__setattr__(r, "params", {"word": "weapon"})  # a recipe tampered with after its checks
    with pytest.raises(safety.Refused):
        store.save(r, {1: np.ones(D)})


def test_no_module_here_imports_torch():
    import m5.dose
    import m5.profile
    import m5.vectors
    for mod in (m5.dose, m5.profile, m5.vectors):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+torch\b", src, re.MULTILINE), mod.__name__


# ===================================================================================== every other guard trips

@pytest.mark.parametrize("kind,cls,params,match", [
    ("telepathy", "concept", {"word": "silk"}, "unknown vector kind"),
    ("concept", "persona", {"word": "silk"}, "has class 'concept'"),
    ("concept", "concept", {"word": "  "}, "non-empty text"),
    ("response_mean", "concept", {"word": "silk", "max_new_tokens": 0, "temperature": 0.0},
     "max_new_tokens"),
    ("response_mean", "concept", {"word": "silk", "max_new_tokens": 8, "temperature": -1.0},
     "temperature"),
    ("contrastive", "persona", {"pairs": "a:b", "position": ":", "render": "raw"}, "pairs must be"),
    ("contrastive", "persona", {"pairs": [["a:"]], "position": ":", "render": "raw"},
     r"\[positive, negative\]"),
    ("contrastive", "persona", {"pairs": [], "position": ":", "render": "raw"}, "at least 1"),
    ("contrastive", "persona", {"pairs": [["a:", "b:"]], "position": ":", "render": "chat"},
     "render must be"),
    ("sae_feature", "persona", {"path": "/tmp/w.pt", "layer": 1}, ".npy or .npz"),
    ("sae_feature", "persona", {"path": "/tmp/w.npz", "layer": 1}, "needs the array's key"),
    ("sae_feature", "persona", {"path": "/tmp/w.npy", "layer": -1}, "layer"),
    ("gaussian", "control", {"reference_id": "a b", "index": 0}, "must match"),
    ("random_subspace", "control", {"rank": 0, "index": 0}, "rank"),
    ("sign_randomised", "control", {"pairs": [["a:", "b:"], ["c:", "d:"]], "position": ":",
                                    "render": "raw", "base_id": "x", "index": -1}, "index"),
])
def test_recipe_validation_trips(kind, cls, params, match):
    with pytest.raises(RecipeError, match=match):
        VectorRecipe("v1", kind, cls, params)
    with pytest.raises(RecipeError, match="exactly the keys"):
        VectorRecipe.from_json({"vector_id": "v1", "kind": kind})


# One stub per defect the read-position and backend-contract checks exist to catch.

class _Lowercasing(StubBackend):        # the template alters the content
    def render(self, messages, *, add_generation_prompt=True):
        return super().render([{**m, "content": m["content"].lower()} for m in messages],
                              add_generation_prompt=add_generation_prompt)


class _Glued(StubBackend):              # the generation prompt merges with the user's last word
    def render(self, messages, *, add_generation_prompt=True):
        return "<bos>" + messages[-1]["content"] + ("def" if add_generation_prompt else "")


class _NotPrefix(StubBackend):          # the bare render is not a prefix of the full one
    def render(self, messages, *, add_generation_prompt=True):
        out = super().render(messages, add_generation_prompt=add_generation_prompt)
        return out if add_generation_prompt else out.replace("<bos>", "<bos>\n")


class _BadDecode(StubBackend):          # decode disagrees with the offsets
    def decode(self, ids):
        return "?" * len(ids)


class _BadPieceDecode(StubBackend):     # ...for every token but BOS (so raw text keeps its BOS)
    def decode(self, ids):
        return "".join("<bos>" if i == 1 else "?" for i in ids)


class _ContentSuffix(StubBackend):      # one prompt ends on a different token from the others
    def render(self, messages, *, add_generation_prompt=True):
        out = super().render(messages, add_generation_prompt=add_generation_prompt)
        return out + "\n" if add_generation_prompt and "Silk" in messages[-1]["content"] else out


class _FewerOffsets(StubBackend):
    def offsets(self, text):
        return super().offsets(text)[:-1]


class _NaNHidden(StubBackend):
    def hidden(self, *a, **k):
        out = super().hidden(*a, **k)
        out[0][...] = np.nan
        return out


class _WrongShape(StubBackend):
    def hidden(self, *a, **k):
        return [x[:, :, :-1] for x in super().hidden(*a, **k)]


class _NoBosText(StubBackend):          # decode drops BOS, so BOS + text loses it
    def decode(self, ids):
        return "".join("" if i == 1 else self._vocab[i] for i in ids)


class _EmptyReply(StubBackend):
    def generate(self, requests):
        return [GenResult(text=" ", token_ids=(2,), finish="eos") for _ in requests]


class _SkipsBos(StubBackend):           # ignores skip_first=0
    def residual_stats(self, prompts, layers, *, masks=None, skip_first=1):
        return super().residual_stats(prompts, layers, masks=masks, skip_first=1)


class _NewlineReply(StubBackend):       # the reply merges with the prompt's last token
    def generate(self, requests):
        return [GenResult(text="\nHello.", token_ids=(2,), finish="eos") for _ in requests]


class _LossyTokenizer(StubBackend):     # the tokenizer silently drops a character
    def _spans(self, text):
        return [(s, e) for s, e in super()._spans(text) if text[s:e] != "§"]

    def generate(self, requests):
        return [GenResult(text="§§", token_ids=(2,), finish="eos") for _ in requests]


class _ShortGenerate(StubBackend):
    def generate(self, requests):
        return super().generate(requests)[:-1]


class _ShortHidden(StubBackend):
    def hidden(self, *a, **k):
        return super().hidden(*a, **k)[:-1]


class _GapAtColon(StubBackend):         # offsets leave the colon's character uncovered
    def offsets(self, text):
        return [(s, s) if text[s:e] == ":" else (s, e) for s, e in super().offsets(text)]


def _ex(backend, recipe, layers=(1,), **kw):
    args = dict(template=TEMPLATE, baseline_words=BASELINE, seed=0, batch_size=4)
    args.update(kw)
    return extract(backend, recipe, list(layers), **args)


@pytest.mark.parametrize("make,match", [
    (lambda: _ex(_Lowercasing(), concept_recipe("Silk")), "verbatim"),
    (lambda: _ex(_Glued(), concept_recipe("Silk")), "inside the user turn"),
    (lambda: _ex(_NotPrefix(), concept_recipe("Silk")), "not an extension"),
    (lambda: _ex(_BadDecode(), concept_recipe("Silk")), "decodes to"),
    (lambda: _ex(_ContentSuffix(), concept_recipe("Silk")), "different tokens"),
    (lambda: _ex(_FewerOffsets(), concept_recipe("Silk")), "offsets"),
    (lambda: _ex(_NaNHidden(), concept_recipe("Silk")), "non-finite"),
    (lambda: _ex(_WrongShape(), concept_recipe("Silk")), "shape"),
    (lambda: _ex(_NoBosText(), unrelated_contrastive_recipe("t", [CARD_CONTROL_PAIR],
                                                            position="last", render="raw")), "BOS"),
    (lambda: _ex(StubBackend(), contrastive_recipe("c", "persona", [("Hello: world", "Hello: there")],
                                                   position=":", render="raw")), "exactly zero"),
    (lambda: _ex(_EmptyReply(), response_mean_recipe("Silk", max_new_tokens=4, temperature=0.0)),
     "empty"),
    (lambda: _ex(StubBackend(), concept_recipe("Silk"), template="About {word} and {other}"),
     "cannot be filled"),
    (lambda: _ex(StubBackend(), concept_recipe("Silk"), template=None), "needs its template"),
    (lambda: _ex(StubBackend(), concept_recipe("Silk"), baseline_words=[]), "baseline words"),
    (lambda: _ex(StubBackend(), concept_recipe("Silk"), layers=()), "no layers"),
    (lambda: _ex(StubBackend(), concept_recipe("Silk"), batch_size=0), "batch_size"),
    (lambda: _ex(StubBackend(), random_subspace_recipe("rs", D + 1, 0)), "rank"),
    (lambda: _ex(StubBackend(), gaussian_recipe("silk", 0), reference={1: np.full(D, 2.0)}),
     "no variance"),
    (lambda: _ex(StubBackend(), gaussian_recipe("silk", 0), reference={1: np.ones(D + 1)}), "shape"),
    (lambda: _ex(_NewlineReply(), response_mean_recipe("Silk", max_new_tokens=4, temperature=0.0)),
     "re-tokenised"),
    (lambda: _ex(_LossyTokenizer(), response_mean_recipe("Silk", max_new_tokens=4, temperature=0.0)),
     "adds no tokens"),
    (lambda: _ex(_ShortGenerate(), response_mean_recipe("Silk", max_new_tokens=4, temperature=0.0)),
     "generate"),
    (lambda: _ex(_ShortHidden(), concept_recipe("Silk")), "arrays for"),
    (lambda: _ex(_GapAtColon(), contrastive_recipe("c", "persona", HUA_PAIRS, position=":",
                                                   render="user_turn")), "between tokens"),
    (lambda: _ex(_BadPieceDecode(), contrastive_recipe("c", "persona", HUA_PAIRS, position=":",
                                                       render="raw")), "does not hold the marker"),
    (lambda: _ex(_BadPieceDecode(), unrelated_contrastive_recipe("t", [CARD_CONTROL_PAIR],
                                                                 position="last", render="raw")),
     "decodes to"),
    (lambda: _ex(StubBackend(drop_trailing_ws=True),
                 unrelated_contrastive_recipe("t", [("Go home now ", "Go away now ")],
                                              position="last", render="raw")), "ends at character"),
    (lambda: read_marker(StubBackend(), "<bos>abc: d", "xyz", ":"), "content verbatim"),
    (lambda: read_marker(StubBackend(), "<bos>abc d", "abc d", ":"), "does not occur"),
])
def test_extraction_guards_trip(make, match):
    with pytest.raises(ExtractionError, match=match):
        make()


def test_the_glued_stub_is_otherwise_valid():
    """The extraction stubs above must fail for their one defect, not for something else."""
    ok = StubBackend()
    assert generation_suffix(ok) == "<start_of_turn>model\n"
    assert _ex(ok, unrelated_contrastive_recipe("t", [CARD_CONTROL_PAIR], position="last",
                                                render="raw"))[1].shape == (D,)


def test_coin_signs_redraws_a_draw_with_one_sign():
    seed = next(s for s in range(200) if len(set(np.random.default_rng(
        derive_seed(s, "sign_randomised", "b", 0, 0)).integers(0, 2, size=2))) == 1)
    signs, attempt = coin_signs(2, seed, "b", 0)
    assert attempt >= 1 and signs.min() == -1.0 and signs.max() == 1.0
    with pytest.raises(ExtractionError, match="at least two"):
        coin_signs(1, 0, "b", 0)


@pytest.mark.parametrize("make,match", [
    (lambda p, v, q: nuisance_scan({}, None, loud_ratio=50, loud_cap=4, gain_ratio=5),
     "no residual statistics"),
    (lambda p, v, q: nuisance_scan({0: _stats(0, np.ones(8))}, None, loud_ratio=0, loud_cap=4,
                                   gain_ratio=5), "loud_ratio"),
    (lambda p, v, q: nuisance_scan({0: _stats(0, np.ones(8))}, None, loud_ratio=50, loud_cap=4,
                                   gain_ratio=-1), "gain_ratio"),
    (lambda p, v, q: nuisance_scan({0: _stats(0, [1.0, np.nan])}, None, loud_ratio=50,
                                   loud_cap=4, gain_ratio=5), "finite 1-D"),
    (lambda p, v, q: nuisance_scan({0: _stats(0, np.ones(8)), 1: _stats(1, np.ones(9))}, None,
                                   loud_ratio=50, loud_cap=4, gain_ratio=5), "differs"),
    (lambda p, v, q: nuisance_scan({0: _stats(0, np.ones(8))}, {0: {"g": np.ones(7)}},
                                   loud_ratio=50, loud_cap=4, gain_ratio=5), "gain 'g'"),
    (lambda p, v, q: norms_from_stats(_stats(0, np.ones(4), norms=()), []), "per-token norms"),
    (lambda p, v, q: norms_from_stats(_stats(0, np.ones(4), norms=(1.0, np.inf)), []), "finite"),
    (lambda p, v, q: norms_from_stats(_stats(0, np.ones(4), norms=(1.0, 2.0), masked=[1.0]), [1]),
     "one per token"),
    (lambda p, v, q: lower_median([]), "no values"),
    (lambda p, v, q: m3_raw_norm([]), "at least one prompt"),
    (lambda p, v, q: m3_raw_norm([[1.0, -2.0]]), "non-negative"),
    (lambda p, v, q: measure_m3_raw_norms(_SkipsBos(), [StubBackend().render(user_turn("Hi"))], [0]),
     "every position"),
    (lambda p, v, q: measure_m3_raw_norms(StubBackend(), [], [0]), "at least one prompt"),
    (lambda p, v, q: mask_coords(np.ones(4), [4]), "outside"),
    (lambda p, v, q: mask_coords(np.ones((2, 2)), [0]), "1-D"),
    (lambda p, v, q: f_loud(np.zeros(4), [0]), "zero vector"),
    (lambda p, v, q: cos_raw_masked(np.zeros(4), [0]), "zero vector"),
    (lambda p, v, q: alpha_from_r_raw(0.1, np.zeros(D), 0, p), "zero vector"),
    (lambda p, v, q: alpha_from_card(0.1, np.zeros(D), 3, p), "zero vector"),
    (lambda p, v, q: card_units(1.0, v[1], 1, q), "middle layer"),
    (lambda p, v, q: r_perp(1.0, np.full(D, np.nan), 0, p), "non-finite"),
    (lambda p, v, q: r_perp("1", v[0], 0, p), "must be a number"),
    (lambda p, v, q: layer_depth(0, 1), "at least 2"),
    (lambda p, v, q: layer_depth(5, 5), "outside"),
    (lambda p, v, q: schedule_layers("strided", 62, None, (0.5, 0.2), 4, (0.5, 0.15)), "depth band"),
    (lambda p, v, q: schedule_layers("strided", 3, None, (0.9, 0.95), 4, (0.5, 0.15)), "no layer"),
    (lambda p, v, q: schedule_layers("strided", 62, None, (0.2, 0.5), 0, (0.5, 0.15)),
     "strided_step"),
    (lambda p, v, q: schedule_layers("single", 1, 0, (0.2, 0.5), 4, (0.5, 0.15)), "n_layers"),
    (lambda p, v, q: schedule_layers("single", 62, 62, (0.2, 0.5), 4, (0.5, 0.15)),
     "outside a model"),
    (lambda p, v, q: schedule_layers("gaussian", 62, None, (0.2, 0.5), 4, (1.5, 0.1)), "centre"),
    (lambda p, v, q: applied_vector_id("silk~perp", True), "already names"),
    (lambda p, v, q: cell_to_intervention(Cell("v", "strided", (1, 3), 0.2), p, v, Positions("all"),
                                          False, weights={1: 1.5, 3: -0.5}), ">= 0"),
    (lambda p, v, q: cell_to_intervention(Cell("v", "single", (1,), 0.2), p, v, "all", False),
     "Positions"),
    (lambda p, v, q: cell_to_intervention(Cell("v", "single", (1,), 0.2, op="clamp"), p, v,
                                          Positions("all"), False, clamp_value=float("nan")),
     "finite"),
    (lambda p, v, q: cell_to_intervention(Cell("v", "single", (1,), 0.2, op="project"), p,
                                          {1: np.zeros(D)}, Positions("all"), False),
     "zero direction"),
])
def test_dose_guards_trip(profile, vecs, partial_profile, make, match):
    with pytest.raises(DoseError, match=match):
        make(profile, vecs, partial_profile)


def test_build_profile_refuses_mismatched_layers(stub):
    first = stub.residual_stats(_raw_texts(stub), [0, 1])
    scan = nuisance_scan(first, None, loud_ratio=50, loud_cap=4, gain_ratio=5)
    masked = stub.residual_stats(_raw_texts(stub), [0], masks={0: [LOUD]})
    with pytest.raises(DoseError, match="statistics cover layers 0"):
        build_profile(stub.info, masked, scan, raw_norms={}, provenance=_provenance())
    other = stub.residual_stats(_raw_texts(stub)[:2], [0, 1], masks={0: [LOUD], 1: [LOUD]})
    with pytest.raises(DoseError, match="not over the same prompts"):
        build_profile(stub.info, other, scan, raw_norms={}, provenance=_provenance())


def test_diagnostics_and_library_guards_trip(profile, vecs):
    for per_layer, match in (({9: vecs[0]}, "not in the profile"),
                             ({2: np.ones(D + 1)}, "shape"),
                             ({2: np.zeros(D)}, "is zero")):
        with pytest.raises(ExtractionError, match=match):
            diagnostics(per_layer, profile)
    with pytest.raises(ExtractionError, match="identical"):
        library_pc1([vecs[0], vecs[0]])
    basis = random_subspace_recipe("rs", 2, 0)
    b = extract(StubBackend(), basis, [1], seed=0, batch_size=1)
    assert diagnostics(b, profile)[1]["orthonormality_error"] < 1e-6


def test_store_guards_trip(tmp_path):
    store = VectorStore(tmp_path / "v")
    r = concept_recipe("Silk")
    with pytest.raises(ExtractionError, match="nothing to store"):
        store.save(r, {})
    with pytest.raises(ExtractionError, match="not a layer"):
        store.save(r, {"a": np.ones(D)})
    with pytest.raises(ExtractionError, match="finite"):
        store.save(r, {1: np.full(D, np.nan)})
    with pytest.raises(FileNotFoundError):
        store.load("concept-cotton")
    assert not (tmp_path / "v").exists()                # failed saves leave nothing behind
    (tmp_path / "v").mkdir()
    np.savez(tmp_path / "v" / "concept-wool.npz", W=np.ones(D))
    with pytest.raises(ExtractionError, match="unexpected array"):
        store.load("concept-wool")
    with pytest.raises(RecipeError, match="must match"):
        store.path("../escape")
