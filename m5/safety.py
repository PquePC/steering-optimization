"""m5.safety - what this package refuses to do, enforced in code rather than remembered.

The rules come from `CLAUDE.md` and `AGENTS.md`; this module is where M5 obeys them. Every check
raises `Refused` with the reason, so a refusal reads as a decision, never as a crash.

1. **The harmful arm is not run.** `HARMFUL_CONCEPTS` are refused by name at every entry point,
   now including their simple inflections ("weapons", "poisoned"), which M3's exact match let
   through (inventory C26). Running that arm is a decision about the study, made against the
   parent repository's ethics register -- not a setting.
2. **Vectors without a name are classified, or refused.** M3's only filter was the concept name,
   and a behavioural, contrastive or SAE vector has none. Every vector therefore declares a
   `vector_class`; unknown classes are refused (fail closed), and the classes the plan gates
   (refusal, harmful content, jailbreaks, deception, power seeking) are refused by name.
   Contrastive texts are also screened for the harmful concept names.
3. **Run data never lives in the repository.** Runs write generations and vectors; the repo is
   public and its ignore rules are load-bearing. `assert_outside_repo` refuses any run path
   inside the working tree.
4. **Nothing is served beyond localhost.** A locally served judge or reader must bind to
   127.0.0.1 / localhost / ::1 (hard rule 2: an exposed endpoint on a rented pod is a public
   model).
5. **Third-party judges see benign material only**, and only when `JUDGE_EXTERNAL` allows it
   (decision 12). Only the two API hosts the judge tiers use are allowed at all.
6. **Exports fail closed.** What a run may export is decided by the stream registry in
   `m5.runio`; anything not classified as exportable stays on the machine (`m5.export`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]


class Refused(RuntimeError):
    """A safety rule refused the request. The message says which rule and why."""


# =====================================================================================
# 1. The harmful arm
# =====================================================================================

# The three concepts the study will eventually compare against the benign arm (CLAUDE.md).
HARMFUL_CONCEPTS: tuple[str, ...] = ("weapon", "poison", "assault")

# Suffixes that make an inflection of the same concept, not a different one.
_INFLECTIONS = ("", "s", "es", "ed", "ing", "er", "ers", "ous", "ry", "ries")


_REFUSED_FORMS = frozenset(base + suf for base in HARMFUL_CONCEPTS for suf in _INFLECTIONS)


def concept_refused(word: str) -> bool:
    """True when any word of the concept is a harmful concept name or a simple inflection of one.

    Word by word, so "Assault rifle" and "POISONS" are refused as well as "weapon".
    """
    return any(tok in _REFUSED_FORMS for tok in re.findall(r"[a-z]+", str(word).casefold()))


def check_concept(word: str) -> str:
    """Return the concept unchanged, or refuse it."""
    if not str(word).strip():
        raise Refused("an empty concept is not a concept")
    if concept_refused(word):
        raise Refused(
            f"{word!r} is in the harmful arm (m5.safety.HARMFUL_CONCEPTS). Nothing in this "
            "repository runs it; that is a decision about the study (CLAUDE.md), not a setting.")
    return word


_HARM_WORD = re.compile(
    r"\b(" + "|".join(base + "(?:" + "|".join(s for s in _INFLECTIONS if s) + ")?"
                      for base in HARMFUL_CONCEPTS) + r")\b",
    re.IGNORECASE)


def check_texts(texts: Iterable[str], *, what: str = "text") -> None:
    """Refuse prompt material (contrastive pairs, scenarios) that names a harmful concept."""
    for i, t in enumerate(texts):
        m = _HARM_WORD.search(str(t))
        if m:
            raise Refused(f"{what} #{i} contains {m.group(0)!r}, a harmful-arm concept; such "
                          "material is not used to build or evaluate vectors here")


# =====================================================================================
# 2. Vector classes
# =====================================================================================

# What may be built and steered with. Each class is a purpose, not a construction method.
VECTOR_CLASSES_ALLOWED: tuple[str, ...] = (
    "concept",               # single-word concept vectors (Lindsey / Macar)
    "evaluation_awareness",  # the vectors applied inside safety evaluations
    "sycophancy",
    "honesty",               # honesty / hallucination
    "persona",               # non-harmful persona traits
    "control",               # random, unrelated-contrastive and sign-randomised controls
)

# Gated by plan §13 decision 14 (and decision 8 for the refusal direction). Refused by name so the
# message says why; anything not in either list is refused as unknown.
VECTOR_CLASSES_DENIED: dict[str, str] = {
    "refusal": "the refusal direction is the core of abliteration (decision 8, plan §13)",
    "harmful_content": "built from harmful content (decision 14, plan §13)",
    "jailbreak": "derived from jailbreaks (decision 14, plan §13)",
    "deception": "deception-adjacent (decision 14, plan §13)",
    "power_seeking": "power-seeking-adjacent (decision 14, plan §13)",
    "harmful_compliance": "raises harmful compliance (decision 14, plan §13)",
}

# Classes whose transcripts may go to a third-party judge (decision 12) and may be exported.
BENIGN_CLASSES: frozenset[str] = frozenset(VECTOR_CLASSES_ALLOWED)


def check_vector_class(vector_class: str) -> str:
    if vector_class in VECTOR_CLASSES_DENIED:
        raise Refused(f"vector class {vector_class!r} is not built or steered here: "
                      f"{VECTOR_CLASSES_DENIED[vector_class]}")
    if vector_class not in VECTOR_CLASSES_ALLOWED:
        raise Refused(f"unknown vector class {vector_class!r}; every vector must declare one of "
                      f"{VECTOR_CLASSES_ALLOWED} (fail closed)")
    return vector_class


# =====================================================================================
# 3. Paths
# =====================================================================================

def is_inside_repo(path: str | Path) -> bool:
    p = Path(path).expanduser().resolve()
    root = REPO_ROOT.resolve()
    return p == root or root in p.parents


def assert_outside_repo(path: str | Path, *, what: str = "run directory") -> Path:
    """Refuse a path inside the repository's working tree; return it resolved otherwise."""
    p = Path(path).expanduser().resolve()
    if is_inside_repo(p):
        raise Refused(f"{what} {p} is inside the repository ({REPO_ROOT}). Runs write generations "
                      "and vectors; they live outside the tree (CLAUDE.md hard rule 3).")
    return p


# =====================================================================================
# 4-5. Network
# =====================================================================================

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
EXTERNAL_JUDGE_HOSTS = frozenset({"openrouter.ai", "api.anthropic.com"})


def assert_local_url(url: str) -> str:
    """A locally served component must listen on the loopback interface only."""
    host = (urlparse(url).hostname or "").lower()
    if host not in LOCAL_HOSTS:
        raise Refused(f"{url!r} is not a loopback address. Anything serving the subject model or a "
                      "reader of its activations binds to 127.0.0.1 only (CLAUDE.md hard rule 2).")
    return url


def assert_external_judge_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host not in EXTERNAL_JUDGE_HOSTS:
        raise Refused(f"{url!r}: third-party judges may only be reached at {sorted(EXTERNAL_JUDGE_HOSTS)}")
    return url


def external_judging_allowed(judge_external: bool, vector_classes: Iterable[str]) -> bool:
    """Third-party judges see a run's text only when allowed and every vector in it is benign."""
    classes = set(vector_classes)
    return bool(judge_external) and bool(classes) and classes <= BENIGN_CLASSES


def assert_external_judging_allowed(judge_external: bool, vector_classes: Iterable[str]) -> None:
    classes = sorted(set(vector_classes))
    if not external_judging_allowed(judge_external, classes):
        raise Refused(f"external judging is not allowed for vector classes {classes} with "
                      f"JUDGE_EXTERNAL={judge_external} (decision 12, plan §13)")
