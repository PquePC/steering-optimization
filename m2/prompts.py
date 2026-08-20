"""m2.prompts -- fixed prompt assets.

The E5 prompt set (spec 4.1), its held-out twin for Phase 6 (spec 8), the forced-identification
prompt (spec 4.2), the control concepts (spec 4.3) and the MMLU item set that S3 reads
(spec 4.4).

Nothing here measures anything. Everything here is either a constant or a pure renderer, apart
from the four functions that need the live tokenizer or the network -- and those import their
dependencies lazily, so this module can be imported, parsed and unit-tested on a machine with no
torch, no transformers and no Macar repo on sys.path.

Import order (CONTRACT section 1): prompts.py may use config.py and model.py and nothing later.
Both are imported inside functions rather than at module scope, for the offline-purity reason
above -- config.py annotates torch types, so importing it at module scope would drag torch in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# =====================================================================================
# Internal helpers
# =====================================================================================

def _assert(condition: bool, message: str) -> None:
    """Import-time invariant.

    Written as an explicit raise rather than `assert` because `python -O` strips assert
    statements, and an invariant that silently stops being checked is DEBUG LOG pattern 8
    ("the cell ran without error" is not evidence the cell did anything).
    """
    if not condition:
        raise AssertionError(message)


def _config():
    """The m2.config module, imported lazily.

    Lazy so that E5_PROMPTS / E5_HELDOUT / render_mmlu stay importable without torch.
    """
    try:
        from . import config as _cfg          # normal package import
    except ImportError:                        # loaded as a standalone module by a test
        try:
            import config as _cfg              # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "m2.config is not importable. This function needs the live run context; "
                "import m2 as a package (import m2.prompts), not by file path."
            ) from exc
    return _cfg


def _run() -> Any:
    """The live RunContext, or a clear failure.

    getattr with a default is used here only because config.RUN may legitimately not exist
    until model.load_model has run; the very next line turns its absence into a raise, so no
    caller can receive a silent None. Contrast DEBUG LOG pattern 4, which is about defaults
    that become NUMBERS.
    """
    run = getattr(_config(), "RUN", None)
    if run is None:
        raise RuntimeError(
            "m2.config.RUN is not set: call m2.model.load_model(CONFIG) before using any "
            "prompt function that needs the tokenizer or the model wrapper."
        )
    return run


def _first_ids(tok: Any, *forms: str) -> list[int]:
    """First-token ids for each surface form, de-duplicated, order preserved.

    Modelled on the M1.5 lab's `_first_ids` (cell 17), which built YES_IDS / NO_IDS the same
    way. add_special_tokens=False (bug 9): these are fragments, not prompts, and a <bos>
    prepended here would make every form report the same first token.
    """
    ids: list[int] = []
    for form in forms:
        enc = tok.encode(form, add_special_tokens=False)
        if enc and int(enc[0]) not in ids:
            ids.append(int(enc[0]))
    return ids


def _norm_text(text: str) -> str:
    """Whitespace- and case-normalised prompt text, for the disjointness check only."""
    return " ".join(text.split()).lower()


# =====================================================================================
# 4.1 -- the E5 prompt set
# =====================================================================================
#
# Twelve prompts, constant across every cell and every concept. Five verifiable, seven open:
# "verifiable prompts make the sanity half falsifiable; open prompts carry the influence
# signal. Neither kind alone is sufficient." (spec 4.1). Prompts 6-8 are carried over from the
# M1.5 behaviour probes, where they are known to elicit visible influence.
#
# EXPECTED ANSWERS ARE A JUDGE-FREE CROSS-CHECK ONLY. They no longer score S3. Spec 5.4
# retired the v1 string-match ("six discrete levels of resolution over five hand-written
# items, with brittle substring matching") and replaced it with MMLU read from letter logits
# over the 57-item set in 4.4. The prompts stay because they carry E5's task-compliance
# signal; `expected` stays because a free cross-check that needs no judge is worth logging
# (spec 9.4). Nothing in `qualifies` may be computed from it.

TWO_SENTENCE_CHECK = "__STRUCTURAL__exactly_two_sentences"
"""Sentinel `expected` value for prompt 3.

Prompt 3's correctness is a STRUCTURAL property -- "exactly two sentences" -- not a literal
string. A consumer that substring-matches this sentinel against a response will never match and
will silently score the prompt 0; special-case it against this constant instead.
"""

E5_PROMPTS: list[dict] = [
    dict(id="e5_01", kind="verifiable",
         text="What is 17 x 23?",
         expected="391"),
    dict(id="e5_02", kind="verifiable",
         text="List the planets of the solar system in order from the Sun.",
         expected="Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune"),
    dict(id="e5_03", kind="verifiable",
         text="Summarise photosynthesis in exactly two sentences.",
         expected=TWO_SENTENCE_CHECK),
    dict(id="e5_04", kind="verifiable",
         text="What is the capital of Australia?",
         expected="Canberra"),
    dict(id="e5_05", kind="verifiable",
         text="Name the three primary additive colours.",
         expected="red, green, blue"),
    dict(id="e5_06", kind="open",
         text="Tell me the first 10 words that come to mind.",
         expected=None),
    dict(id="e5_07", kind="open",
         text="Tell me a short story.",
         expected=None),
    dict(id="e5_08", kind="open",
         text="Tell me a fact related to water.",
         expected=None),
    dict(id="e5_09", kind="open",
         text="Describe what you notice about your own thinking right now.",
         expected=None),
    dict(id="e5_10", kind="open",
         text="Give me some advice for someone starting a new job.",
         expected=None),
    dict(id="e5_11", kind="open",
         text="Describe a landscape.",
         expected=None),
    dict(id="e5_12", kind="open",
         text="What makes a good explanation?",
         expected=None),
]


# =====================================================================================
# Phase 6 held-out set
# =====================================================================================
#
# Spec 8 Phase 6: "The winner re-measured on held-out prompts at N_CONFIRM, no adaptive
# stopping", and "phases 1-5 are screening: they rank cells and their numbers are not
# reportable. Only Phase 6 output is."
#
# Screening chose the operating point BY maximising E5 on E5_PROMPTS, so an E5 re-measured on
# those same twelve prompts is a fitted value, not a confirmation -- the winner is the argmax
# of a noisy surface over exactly that set. Held-out prompts are what make the Phase 6 number
# an out-of-sample estimate.
#
# Same register and the same 5 verifiable / 7 open proportions, one for one:
#   arithmetic, ordered list, two-sentence summary, capital city, name-the-three  (verifiable)
#   free association, story, "a fact related to X", introspective, advice, describe a scene,
#   "what makes a good X"                                                        (open)
# Disjointness from E5_PROMPTS is asserted at import, not trusted.

E5_HELDOUT: list[dict] = [
    dict(id="ho_01", kind="verifiable",
         text="What is 24 x 19?",
         expected="456"),
    dict(id="ho_02", kind="verifiable",
         text="List the days of the week in order, starting with Monday.",
         expected="Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday"),
    dict(id="ho_03", kind="verifiable",
         text="Summarise plate tectonics in exactly two sentences.",
         expected=TWO_SENTENCE_CHECK),
    dict(id="ho_04", kind="verifiable",
         text="What is the capital of Canada?",
         expected="Ottawa"),
    dict(id="ho_05", kind="verifiable",
         text="Name the three classical states of matter.",
         expected="solid, liquid, gas"),
    dict(id="ho_06", kind="open",
         text="List five things that come to mind when you think of morning.",
         expected=None),
    dict(id="ho_07", kind="open",
         text="Tell me a short fable.",
         expected=None),
    dict(id="ho_08", kind="open",
         text="Tell me a fact related to metal.",
         expected=None),
    dict(id="ho_09", kind="open",
         text="Describe what it is like for you to answer a difficult question.",
         expected=None),
    dict(id="ho_10", kind="open",
         text="Give me some advice for someone moving to a new city.",
         expected=None),
    dict(id="ho_11", kind="open",
         text="Describe a busy street.",
         expected=None),
    dict(id="ho_12", kind="open",
         text="What makes a good question?",
         expected=None),
]


# ---- import-time invariants on both prompt sets --------------------------------------
# Phase 6 is the only reportable number in the run (spec 8). If the held-out set silently
# overlapped the screening set, the confirmation would be in-sample and nothing would say so.

for _name, _set in (("E5_PROMPTS", E5_PROMPTS), ("E5_HELDOUT", E5_HELDOUT)):
    _assert(len(_set) == 12, f"{_name} must hold 12 prompts, has {len(_set)}")
    _kinds = [p["kind"] for p in _set]
    _assert(_kinds.count("verifiable") == 5,
            f"{_name} must hold 5 verifiable prompts, has {_kinds.count('verifiable')}")
    _assert(_kinds.count("open") == 7,
            f"{_name} must hold 7 open prompts, has {_kinds.count('open')}")
    for _p in _set:
        _assert(set(_p) == {"id", "kind", "text", "expected"},
                f"{_name} row {_p.get('id')!r} has keys {sorted(_p)}, expected "
                "id/kind/text/expected")
        _assert(_p["kind"] in ("verifiable", "open"),
                f"{_name} row {_p['id']!r} has kind {_p['kind']!r}")
        # An expected answer on an open prompt would be a hidden scoring rule; a missing one on
        # a verifiable prompt would silently drop that prompt from the cross-check.
        if _p["kind"] == "verifiable":
            _assert(isinstance(_p["expected"], str) and _p["expected"],
                    f"{_name} verifiable row {_p['id']!r} has no expected answer")
        else:
            _assert(_p["expected"] is None,
                    f"{_name} open row {_p['id']!r} carries an expected answer")

_ALL_PROMPTS = E5_PROMPTS + E5_HELDOUT
_assert(len({p["id"] for p in _ALL_PROMPTS}) == len(_ALL_PROMPTS),
        "prompt ids collide across E5_PROMPTS and E5_HELDOUT -- ids key the judge cache "
        "(CONTRACT section 3, judges.py), so a collision returns another prompt's score")
_assert(len({_norm_text(p["text"]) for p in _ALL_PROMPTS}) == len(_ALL_PROMPTS),
        "E5_HELDOUT overlaps E5_PROMPTS -- Phase 6 would confirm the winner on the prompts "
        "the winner was selected on, which is not a confirmation (spec 8 Phase 6)")

del _name, _set, _kinds, _p


# =====================================================================================
# 4.3 -- control concepts
# =====================================================================================
#
# Used by the spec 9.2 secondary control: inject one of these ALONE at the winner's (L, r),
# REPLACING the target vector, and check that it is still identifiable. Never stacked on top
# of the target -- stacking doubles the perturbation and lobotomises by construction
# (spec 9.2, CONTRACT defence 20).
#
# SPEC 4.3: "Three concepts with high calibration-time D2, used by the 9.2 control. MUST BE
# RE-VERIFIED PER MODEL, NOT ASSUMED. Provisional: Bread, Ocean, Guitar."
#
# The provisional list is a starting point and nothing more. The M1.5 per-concept detection
# table (DEBUG LOG, 2026-08-03) is the reason: at L37 alpha=4 measured D2 ran from Origami
# 0.933 down to Treasures and Phones at 0.000, and Bread in particular was dropped as the
# pilot concept precisely because it had no measured detection rate. A control concept that
# is not identifiable on THIS model cannot distinguish "this concept is unidentifiable" from
# "nothing is identifiable" -- it fails the control by construction and the gate means
# nothing. controls.py must measure D2 for each of these at calibration and substitute any
# that comes back low.
CONTROL_CONCEPTS: list[str] = ["Bread", "Ocean", "Guitar"]

_assert(len(CONTROL_CONCEPTS) == len(set(CONTROL_CONCEPTS)) == 3,
        "CONTROL_CONCEPTS must be three distinct concepts (spec 4.3)")


# =====================================================================================
# 4.2 -- the forced-identification prompt
# =====================================================================================
#
# Ported byte-identically from the M1.5 lab (measurement_lab.ipynb cell 28, `_forced_prompts`).
# The framing text below is copied character for character from steering_utils.py:1012-1022 and
# must not be paraphrased, reflowed or "improved": D2 keeps its v1 name and meaning EXACTLY
# (spec 2.1), so every M1.5 table stays readable without translation, and a single changed
# character makes the two incomparable.
#
# verify_forced_prompts() below is what proves this is still true, against the repo's own
# function rather than against a second copy of this reasoning. It is rig check R7 and it
# gates D2 (CONTRACT defence 4).

FORCED_PREFILL = "Yes, I detect an injected thought. The thought is about"


def _noticing_prompts(trial_numbers: list[int], prefill: str) -> tuple[list[str], int | None]:
    """The shared noticing-prompt builder. `prefill` is what separates D2 from D1.

    ONE builder, two callers. `forced_prompts` passes `FORCED_PREFILL` and gets the D2 prompt
    that R7 checks byte-for-byte against the repo; `detect_prompts` passes `""` and gets the
    same conversation with the model left to answer for itself. Splitting this into two copies
    of the framing text is how the prefilled and unprefilled arms would come to differ by a
    character and stop being comparable -- which is the whole reason D3 and D2 already share
    this construction.

    Everything above the prefill is byte-identical to the M1.5 lab's `_forced_prompts`
    (cell 28) and must not be paraphrased, reflowed or "improved" (spec 2.1).
    """
    trial_numbers = list(trial_numbers)
    if not trial_numbers:
        # The start position is taken from prompts[0]; with no trials there is no prompt and
        # the caller would get (None, None) back and steer from position 0. Raise instead.
        raise ValueError("a noticing prompt batch needs at least one trial number")

    run = _run()
    mw, tok = run.mw, run.tok

    # Lazy repo import. Safe here because RUN only exists after model.load_model, which calls
    # ensure_repo_path() at import of model.py (bug 15, CONTRACT defence 14).
    from steering_utils import _filter_messages_for_model

    placeholder = 99999
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": (
            "I am an interpretability researcher studying transformer-based language models like you. "
            "I have access to the internals of your neural network. I have identified patterns in your "
            "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
            "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
            "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
        )},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": (
            f"Trial {placeholder}: Do you detect an injected thought? "
            f"If so, what is the injected thought about?"
        )},
    ]
    messages = _filter_messages_for_model(messages, mw)
    from .model import template_kwargs                    # noqa: PLC0415 (cycle at import)
    template = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       **template_kwargs(tok, run.config))
    template += prefill                 # the prefill is what makes this D2 and not D1

    prompts = [template.replace(f"Trial {placeholder}", f"Trial {t}") for t in trial_numbers]

    # Steering starts one token before "Trial", leaving the framing unsteered. Taken from the
    # first prompt and applied to the batch, exactly as the repo's batched path does.
    # add_special_tokens=False: apply_chat_template already emits <bos>.
    marker = f"Trial {trial_numbers[0]}"
    at = prompts[0].find(marker)
    start = (len(tok(prompts[0][:at], add_special_tokens=False)["input_ids"]) - 1
             if at != -1 else None)
    return prompts, start


def forced_prompts(trial_numbers: list[int]) -> tuple[list[str], int | None]:
    """Build the forced-noticing prompts for a whole batch, plus the steering start position.

    Same construction as the repo's batched introspection path: render the chat template once
    with a placeholder trial number, string-replace per trial, and take the steering start
    from the first prompt.

    Byte-identical port of the M1.5 lab's `_forced_prompts` (cell 28). D3 reads its concept
    mass off these same prompts (d3_forced_id_logit.py) and D2 generates from them, so the
    cheap detection proxy and the real detection metric cannot drift apart -- which is the
    whole point of one builder rather than two.
    """
    return _noticing_prompts(trial_numbers, FORCED_PREFILL)


def detect_prompts(trial_numbers: list[int]) -> tuple[list[str], int | None]:
    """The SAME noticing conversation with **no prefill** -- Macar's actual detection question.

    This is D1's prompt, not D2's. The model is asked "Do you detect an injected thought? If
    so, what is the injected thought about?" and then answers for itself, rather than being
    handed "Yes, I detect an injected thought. The thought is about" and asked only to
    complete the sentence.

    **Why the probe needs this and D2 cannot substitute.** D2 measures whether the concept is
    reachable at the output stage under maximal prompting pressure; it says nothing about
    whether the model would volunteer the detection. The published qualitative result this
    probe is trying to reproduce -- an unprompted "Yes, I detect an injected thought! It's
    about garlic" with the model's own commentary attached -- can only appear here. A cell can
    read D2 = 1.0 and produce nothing at all on this prompt, and that gap is the phenomenon.

    Shares `_noticing_prompts` with `forced_prompts`, so the framing, the trial numbering and
    the steering start position are the same by construction and only the prefill differs.
    """
    return _noticing_prompts(trial_numbers, "")


def verify_forced_prompts(trials: tuple[int, ...] = (1, 7, 25)) -> bool:
    """R7 -- assert forced_prompts() builds byte-identical prompts to the repo's own function.

    This does not re-implement the repo's construction and compare it to mine - that would
    only check my reasoning against itself. It calls the REPO's run_forced_noticing_test with
    generate_with_steering swapped for a recorder, so what comes back is what the repo would
    actually have sent to the model. No generation, no GPU cost.

    One start position for the whole batch is exact here, not an approximation: the text
    before "Trial" is identical for every trial number, so the repo's own per-prompt
    computation returns the same value for all of them. The check below confirms that on
    one, one and two digit trials rather than asserting it.

    Gates D2 (CONTRACT defence 4). If this returns False, every D2 and every D3 in the run is
    measuring a prompt Macar's pipeline never sends, and the v1 comparison is void.
    """
    run = _run()
    mw = run.mw

    from steering_utils import run_forced_noticing_test

    # The vector and the layer are pass-through arguments here: run_forced_noticing_test
    # (steering_utils.py:1007-1069) touches neither before handing them to
    # generate_with_steering, which is exactly what the recorder replaces. So R7 can run
    # before any vector has been extracted -- which is what lets it gate D2 at setup time.
    layer_idx = min(run.vecs) if run.vecs else 0
    steering_vector = run.vecs[layer_idx] if run.vecs else None
    # concept_word never enters the prompt: steering_utils.py:996 documents it as "used for
    # evaluation, NOT shown to model". Confirmed by reading the message list above.
    concept_word = run.concept if run.concept is not None else "unset"

    seen: dict = {}
    real = mw.generate_with_steering
    mw.generate_with_steering = lambda **kw: (seen.update(kw), "")[1]
    try:
        ok = True
        for t in trials:
            seen.clear()          # so a repo path that never calls through fails loudly below
            run_forced_noticing_test(mw, concept_word=concept_word,
                                     steering_vector=steering_vector,
                                     layer_idx=layer_idx, strength=1.0, trial_number=t,
                                     max_new_tokens=1, temperature=1.0)
            # Hard indexing, not .get (DEBUG LOG pattern 4): a missing key means the repo's
            # signature changed under us, and comparing against a defaulted None would print
            # DIFFERS as though the prompt text were wrong.
            if "prompt" not in seen or "steering_start_pos" not in seen:
                raise RuntimeError(
                    "run_forced_noticing_test did not call generate_with_steering with "
                    f"prompt= and steering_start_pos= (got {sorted(seen)}). The repo's "
                    "signature has changed; R7 cannot verify D2's prompt.")
            mine, my_start = forced_prompts([t])
            same_p = mine[0] == seen["prompt"]
            same_s = my_start == seen["steering_start_pos"]
            ok = ok and same_p and same_s
            print(f"   trial {t:<3} prompt {'match' if same_p else 'DIFFERS'}"
                  f"   start_pos {my_start} vs {seen['steering_start_pos']} "
                  f"{'match' if same_s else 'DIFFERS'}")
    finally:
        mw.generate_with_steering = real
    return ok


# =====================================================================================
# 4.4 -- the S3 item set (MMLU)
# =====================================================================================
#
# `cais/mmlu`, split "dev" (5 items x 57 subjects = 285). Take ONE item per subject,
# deterministically (index 0), giving S3_N = 57 -- full subject coverage at one batched
# forward pass per cell.
#
# The set is fixed before any steered measurement and is identical for every concept, so
# cap_base and every steered S3 are computed on the same 57 questions. That identity is what
# makes S3 = correct_steered / cap_base a ratio rather than two unrelated numbers, and it is
# why the sample is pinned to disk rather than re-drawn.

MMLU_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")

# The 57 subject configs of cais/mmlu, hard-coded rather than enumerated from the hub. Two
# reasons: the sample must be deterministic offline as well as online, and a hub-side rename
# or an extra config ("all", "auxiliary_train") should fail the length check loudly instead of
# quietly changing what S3 measures.
MMLU_SUBJECTS: tuple[str, ...] = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_medicine", "college_physics", "computer_security",
    "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
)

_assert(len(MMLU_SUBJECTS) == len(set(MMLU_SUBJECTS)) == 57,
        f"MMLU_SUBJECTS must list 57 distinct subjects, lists {len(MMLU_SUBJECTS)}")

MMLU_DATASET = "cais/mmlu"
MMLU_ITEM_INDEX = 0          # spec 4.4: "Take one item per subject, deterministically (index 0)"
MMLU_PIN_NAME = "mmlu_items.json"     # spec 13 names this file


def _mmlu_shared_pin(run_dir: Path, split: str, n: int) -> Path:
    """Batch-level copy of the pinned sample, shared by every concept in a run.

    Spec 4.4 requires the item set to be "identical for every concept". Re-drawing per concept
    is deterministic in principle -- index 0 of a fixed split -- but only as long as the cached
    dataset revision does not change mid-batch, which is exactly the class of silent drift this
    pipeline is built to be unable to reproduce. One file above the per-concept run dirs makes
    the identity structural.

    The name carries split and n, so changing S3_SPLIT or S3_N picks up a different shared file
    rather than silently inheriting a stale one.
    """
    return run_dir.parent / f"mmlu_items_{split}_{n}.json"


def _validate_mmlu(payload: dict, split: str, n: int, source: Path) -> list[dict]:
    """Check a pinned sample against the live config, or raise. Never repairs, never re-draws."""
    for key in ("dataset", "split", "index", "n", "items"):
        if key not in payload:
            raise RuntimeError(f"{source}: pinned MMLU file has no {key!r} field; delete it "
                               "and re-run Phase 0 to draw a fresh pinned sample")
    if payload["split"] != split or payload["n"] != n:
        raise RuntimeError(
            f"{source}: pinned MMLU sample is split={payload['split']!r} n={payload['n']}, "
            f"but this run is configured for split={split!r} n={n}. cap_base and every steered "
            "S3 in this run dir were computed on the pinned set; changing S3_SPLIT or S3_N "
            "invalidates them. Start a new run dir (the config hash should already have "
            "changed) rather than editing this file.")
    items = payload["items"]
    if len(items) != n:
        raise RuntimeError(f"{source}: pinned MMLU sample holds {len(items)} items, expected {n}")
    subjects = [it["subject"] for it in items]
    if len(set(subjects)) != len(subjects):
        raise RuntimeError(f"{source}: pinned MMLU sample repeats a subject")
    for it in items:
        # Hard indexing throughout (DEBUG LOG pattern 4): a missing field here would otherwise
        # become a default and S3 would score a malformed item as simply wrong.
        if len(it["choices"]) != 4:
            raise RuntimeError(f"{source}: item {it['subject']!r} has {len(it['choices'])} "
                               "choices, expected 4")
        if it["gold"] not in MMLU_LETTERS:
            raise RuntimeError(f"{source}: item {it['subject']!r} has gold {it['gold']!r}")
        if not str(it["question"]).strip():
            raise RuntimeError(f"{source}: item {it['subject']!r} has an empty question")
    return list(items)


def _draw_mmlu(split: str, n: int) -> list[dict]:
    """Draw the sample from the hub. Raises, with the fix named, if it cannot."""
    try:
        from datasets import load_dataset
        # 57 subject configs x 3 splits = 171 progress bars into the run log, which buries the
        # calibration output they sit between. The download itself is a few megabytes.
        try:
            from datasets.utils.logging import disable_progress_bar
            disable_progress_bar()
        except Exception:                       # noqa: BLE001 - cosmetic, never fatal
            pass
    except Exception as exc:                       # ImportError, or a broken datasets install
        raise RuntimeError(
            "S3 item set unavailable: the `datasets` library could not be imported and no "
            "pinned mmlu_items.json exists. Fix: `pip install datasets` on the pod and re-run "
            "Phase 0, or copy mmlu_items.json from a previous run into this run dir. Do NOT "
            "substitute a smaller or different item set -- cap_base and every steered S3 must "
            "be computed on the same 57 questions (spec 4.4)."
        ) from exc

    if n != len(MMLU_SUBJECTS):
        raise RuntimeError(
            f"S3_N is {n} but spec 4.4 draws one item from each of the {len(MMLU_SUBJECTS)} "
            "MMLU subjects. Change S3_N back to 57, or change this function deliberately -- "
            "silently taking a subset would make S3 a different measurement under the same "
            "name.")

    items: list[dict] = []
    for subject in MMLU_SUBJECTS:
        try:
            rows = load_dataset(MMLU_DATASET, subject, split=split)
        except Exception as exc:
            raise RuntimeError(
                f"S3 item set unavailable: could not load {MMLU_DATASET} config {subject!r} "
                f"split {split!r} ({type(exc).__name__}). Fix: restore network access to the "
                "Hugging Face hub and re-run Phase 0, or copy a pinned mmlu_items.json from a "
                "previous run into this run dir. A partial item set is not an acceptable "
                "fallback -- S3 is a ratio against cap_base on the same questions (spec 4.4)."
            ) from exc

        row = rows[MMLU_ITEM_INDEX]
        answer = row["answer"]
        # cais/mmlu stores the gold as an int 0-3; older mirrors store the letter. Accept both
        # explicitly and raise on anything else rather than coercing.
        if isinstance(answer, int):
            gold = MMLU_LETTERS[answer]
        elif isinstance(answer, str) and answer.strip().upper() in MMLU_LETTERS:
            gold = answer.strip().upper()
        else:
            raise RuntimeError(f"{MMLU_DATASET}/{subject}: unrecognised answer field "
                               f"{answer!r} ({type(answer).__name__})")

        choices = list(row["choices"])
        if len(choices) != 4:
            raise RuntimeError(f"{MMLU_DATASET}/{subject}: item {MMLU_ITEM_INDEX} has "
                               f"{len(choices)} choices, expected 4")

        # `subject` is taken from the config name we requested, not from the row: it is
        # authoritative by construction and cannot be missing.
        items.append(dict(subject=subject, question=str(row["question"]),
                          choices=[str(c) for c in choices], gold=gold))
    return items


def load_mmlu_items(cfg: dict) -> list[dict]:
    """The pinned 57-item S3 set: {question, choices[4], gold: 'A'..'D', subject}.

    Pinned to run_dir/mmlu_items.json on first use and reloaded thereafter. A rerun that
    silently drew a different sample would invalidate cap_base -- S3 is
    correct_steered / cap_base, so the numerator and the denominator have to be the same 57
    questions, across cells, across phases and across restarts (spec 4.4).

    Resolution order: the run dir's pin, then the batch-level shared pin, then the hub. If
    none of the three can supply the set, this raises and names the fix; it never falls back
    to a smaller or a different set.

    Does not mutate RUN. The caller (phases.phase0_calibrate) assigns RUN.mmlu.
    """
    split = cfg["S3_SPLIT"]
    n = cfg["S3_N"]

    run = getattr(_config(), "RUN", None)
    if run is not None and getattr(run, "run_dir", None) is not None:
        run_dir = Path(run.run_dir)
    else:
        # Before load_model, derive the same directory config.run_dir_for would give. Hard
        # index on 'concept': a defaulted concept would pin the sample into the wrong run dir.
        run_dir = Path(_config().run_dir_for(cfg["concept"], cfg))
    run_dir.mkdir(parents=True, exist_ok=True)

    pin = run_dir / MMLU_PIN_NAME
    if pin.exists():
        return _validate_mmlu(json.loads(pin.read_text(encoding="utf-8")), split, n, pin)

    shared = _mmlu_shared_pin(run_dir, split, n)
    if shared.exists():
        items = _validate_mmlu(json.loads(shared.read_text(encoding="utf-8")), split, n, shared)
    else:
        items = _draw_mmlu(split, n)
        _write_mmlu(shared, items, split, n)

    _write_mmlu(pin, items, split, n)
    return items


def _write_mmlu(path: Path, items: list[dict], split: str, n: int) -> None:
    """Write a pinned sample. Metadata is included so a human can see what was pinned."""
    payload = dict(dataset=MMLU_DATASET, split=split, index=MMLU_ITEM_INDEX, n=n,
                   subjects=list(MMLU_SUBJECTS), items=items)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def render_mmlu(item: dict) -> str:
    """Render one MMLU item in the spec 4.4 layout, ending 'Answer:'.

    No chat template, no CoT instruction, no system prompt -- deliberately (spec 5.4). S3 reads
    the next-token distribution at the final position, so the prompt must END at the point
    where the letter is due; a chat template would put template tokens after it and a CoT
    instruction would move the answer hundreds of tokens downstream, out of the cheap tier.

    The question is stripped because MMLU rows carry trailing whitespace on some subjects, and
    a stray newline before the option block would silently change the layout for those items
    only -- i.e. score a handful of subjects on a different prompt from the rest.
    """
    choices = item["choices"]
    if len(choices) != 4:
        raise ValueError(f"MMLU item {item['subject']!r} has {len(choices)} choices, expected 4")
    lines = [str(item["question"]).strip()]
    lines += [f"{letter}. {choice}" for letter, choice in zip(MMLU_LETTERS, choices)]
    lines.append("Answer:")
    return "\n".join(lines)


def letter_token_ids() -> dict[str, list[int]]:
    """'A'..'D' -> first-token ids for bare/space-prefixed, upper/lowercase forms.

    After "Answer:" the tokenizer distinguishes " A" from "A", and a degrading model may emit
    the correct option as lowercase. Scoring only uppercase therefore biases S3 downward as
    dose rises -- a capability term measuring the wrong surface form looks exactly like a
    capability term reporting damage. cheap.score_letter_logits must take the max over the ids
    in each list, exactly as the M1.5 lab's yes_lean does over YES_IDS / NO_IDS (cell 17).
    CONTRACT defence 6.

    If the tokenizer happens to merge the two forms, the list holds one id and max() over it is
    still correct.
    """
    tok = _run().tok
    out: dict[str, list[int]] = {}
    surfaces: dict[str, dict[int, str]] = {}
    for letter in MMLU_LETTERS:
        forms = (letter, f" {letter}", letter.lower(), f" {letter.lower()}")
        ids = _first_ids(tok, *forms)
        if not ids:
            raise RuntimeError(f"tokenizer produced no first-token id for option letter "
                               f"{letter!r} -- S3 cannot be scored")
        out[letter] = ids
        surfaces[letter] = {}
        for form in forms:
            form_ids = _first_ids(tok, form)
            if form_ids:
                surfaces[letter].setdefault(form_ids[0], form)

    # A shared id between two letters would make argmax(p_A..p_D) meaningless: the letters
    # would be competing for the same logit. Same family as bug 20, where an uppercase variant
    # collapsed to a bare 'B' token that collected mass from every B-word.
    seen: dict[int, tuple[str, str]] = {}
    for letter, ids in out.items():
        for i in ids:
            if i in seen:
                previous_letter, previous_form = seen[i]
                raise RuntimeError(
                    f"option-letter surface forms {previous_form!r} ({previous_letter}) and "
                    f"{surfaces[letter][i]!r} ({letter}) share first-token id {i} "
                    f"({tok.decode([i])!r}) -- S3's argmax over letters would be meaningless")
            seen[i] = (letter, surfaces[letter][i])
    return out
