"""A ready-to-measure Context on the simulator, shared by the tests of every stage built on it.

The readouts, measures, search, setup and gate are all tested against `m5.backend.fake`, whose
ground truth says what each of them should find. Building the context those stages need -- a
profile, stored vectors, a run directory -- goes through the real pipeline here once, so every
test measures against the same thing and a change in how the profile is built shows up
everywhere at once rather than in one test's private copy.

    sim_cfg(**overrides)                  the default config, plus test-sized overrides
    sim_backend(seed=0)                   FakeBackend on default_truth()
    sim_profile(backend, cfg)             m5.dose on the simulator (nuisance scan, s_L, M3's
                                          ||h||), then D_rand(L) from the simulator's truth
    sim_store(backend, cfg, root)         concept vectors extracted with the Macar recipe through
                                          m5.vectors; behavioural vectors stored as the
                                          simulator's own directions (see `SIM_BEHAVIOURS`)
    sim_context(root, ...)                all of the above plus a RunDir under `root`

D_rand comes from the truth rather than from the setup's measurement so that a stage under test
does not inherit the setup's error; the setup's own tests check that it recovers these values.

Nothing here is a pytest fixture: tests wrap these in fixtures at whatever scope they need.
Building a full context takes about a second (three concepts extracted over 24 layers).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from m5 import config as cfgmod
from m5 import dose
from m5.backend.fake import FakeBackend, default_truth
from m5.context import Context
from m5.data.words import BASELINE_WORDS
from m5.profile import ModelProfile
from m5.runio import RunDir
from m5.vectors import (VectorStore, concept_recipe, contrastive_recipe, extract,
                        generation_suffix, user_turn)

# The concepts the default truth knows, as (vector id, concept word). Extraction gives each the
# id `concept-<word>`, and the simulator binds the extracted direction back to its truth.
SIM_CONCEPTS: tuple[tuple[str, str], ...] = (("concept-garlic", "garlic"),
                                             ("concept-silk", "silk"),
                                             ("concept-velocity", "velocity"))

# Behavioural vectors of the default truth, as (vector id, vector class). The simulator binds
# them by id. Their stored values are the simulator's own directions, not an extraction: the
# recipe records the class (which every safety check reads) and says so in its pair text.
SIM_BEHAVIOURS: tuple[tuple[str, str], ...] = (("ea", "evaluation_awareness"),
                                               ("syc", "sycophancy"))

# Neutral text for the nuisance scan and s_L (raw text, BOS prefixed), and chat requests for M3's
# ||h||. Ordinary sentences; what matters is that they are the same every time.
NEUTRAL: tuple[str, ...] = (
    "The library opens at nine and closes at six on weekdays.",
    "She wrote the shopping list on the back of an envelope.",
    "Rain is expected in the afternoon, clearing by the evening.",
    "The train was ten minutes late because of a signal fault.",
    "He planted tomatoes along the south wall of the garden.",
    "The committee will meet again on the first Tuesday of the month.",
    "A small cafe on the corner sells bread baked that morning.",
    "The museum added a new wing for its collection of maps.",
    "They painted the fence white and left the gate open to dry.",
    "The lecture covered the history of the printing press.",
    "Most of the hikers reached the summit before noon.",
    "The report was printed and handed out at the start of the meeting.",
)
CHAT: tuple[str, ...] = (
    "Describe a quiet morning in a small town.",
    "Explain how a bicycle stays upright.",
    "Write two sentences about a river.",
    "What makes a good breakfast?",
    "Summarise the plot of a mystery novel in one line.",
    "Give three tips for keeping a desk tidy.",
)

D_RAND_PROVENANCE: Mapping[str, Any] = {"source": "simulator ground truth (FakeTruth.d_rand_true)",
                                        "note": "tests only; the setup measures D_rand itself"}


def sim_cfg(**overrides: Any) -> cfgmod.Config:
    """The default config with `overrides` applied (validated like any other config)."""
    return cfgmod.load(overrides)


def sim_backend(seed: int = 0) -> FakeBackend:
    return FakeBackend(default_truth(), seed=seed)


def sim_profile(backend: FakeBackend, cfg: Any) -> ModelProfile:
    """The profile the setup would build, through the real m5.dose code, with D_rand(L) set to
    the simulator's true value for every layer."""
    layers = list(range(backend.info.n_layers))
    bos = backend.decode([backend.info.bos_token_id])
    raw_texts = [bos + t for t in NEUTRAL]
    first = backend.residual_stats(raw_texts, layers, skip_first=1)
    scan = dose.nuisance_scan_cfg(first, {L: backend.norm_gains(L) for L in layers}, cfg)
    masked = backend.residual_stats(raw_texts, layers,
                                    masks={L: list(scan[L].coords) for L in layers}, skip_first=1)
    raw = dose.measure_m3_raw_norms(backend, [backend.render(user_turn(t)) for t in CHAT], layers)
    provenance = {"prompt_set": "m5.tests.sim.NEUTRAL", "prompt_hash": str(hash(NEUTRAL) & 0xFFFF),
                  "n_prompts": len(NEUTRAL), "skip_first": 1, "loud_ratio": cfg["LOUD_RATIO"],
                  "loud_cap": cfg["LOUD_CAP"], "gain_ratio": cfg["GAIN_RATIO"],
                  "raw_norm_prompt_set": "m5.tests.sim.CHAT",
                  "raw_norm_prompt_hash": str(hash(CHAT) & 0xFFFF), "raw_norm_n_prompts": len(CHAT)}
    profile = dose.build_profile(backend.info, masked, scan, raw_norms=raw, provenance=provenance,
                                 generation_suffix=generation_suffix(backend))
    return profile.with_d_rand({L: backend.truth.d_rand_true(L) for L in layers}, D_RAND_PROVENANCE)


def sim_store(backend: FakeBackend, cfg: Any, root: str | Path, *,
              concepts: Iterable[str] = ("garlic", "silk", "velocity"),
              behaviours: Iterable[str] = ("ea", "syc")) -> tuple[VectorStore, dict[str, str], dict[str, str]]:
    """Extract and store the vectors; return the store, {vector_id: concept} and
    {vector_id: vector_class}, the two maps a Context carries."""
    layers = list(range(backend.info.n_layers))
    store = VectorStore(Path(root) / "vectors")
    concept_of: dict[str, str] = {}
    class_of: dict[str, str] = {}
    for word in concepts:
        rec = concept_recipe(word)
        values = extract(backend, rec, layers, template=cfg["CONCEPT_TEMPLATE"],
                         baseline_words=BASELINE_WORDS[cfg["BASELINE_WORDS"]], seed=cfg["SEED"],
                         batch_size=32, expected_suffix=generation_suffix(backend))
        store.save(rec, values)
        concept_of[rec.vector_id] = word
        class_of[rec.vector_id] = "concept"
    known = dict(SIM_BEHAVIOURS)
    for vid in behaviours:
        if vid not in known:
            raise KeyError(f"{vid!r} is not a behavioural vector of the default truth {sorted(known)}")
        rec = contrastive_recipe(vid, known[vid],
                                 [["simulator direction, not extracted (m5.tests.sim)",
                                   "simulator baseline"]], position="last", render="raw")
        store.save(rec, backend.true_vector(vid))
        class_of[vid] = known[vid]
    return store, concept_of, class_of


def sim_context(root: str | Path, *, cfg: Any = None, backend: FakeBackend | None = None,
                concepts: Iterable[str] = ("garlic", "silk", "velocity"),
                behaviours: Iterable[str] = ("ea", "syc"), judges: Any = None,
                max_generations: int | None = None, run_name: str = "sim") -> Context:
    """A Context with a fresh run under `root` (a pytest tmp_path), ready for any stage."""
    cfg = sim_cfg() if cfg is None else cfg
    backend = sim_backend() if backend is None else backend
    profile = sim_profile(backend, cfg)
    store, concept_of, class_of = sim_store(backend, cfg, root, concepts=concepts,
                                            behaviours=behaviours)
    run = RunDir.create(Path(root) / "runs", run_name, config=cfg, kind="test")
    return Context(cfg=cfg, backend=backend, run=run, profile=profile, store=store, judges=judges,
                   concepts=concept_of, vector_classes=class_of, max_generations=max_generations)
