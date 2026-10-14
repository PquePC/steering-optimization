"""m5.judges.matching - is what the model named the concept that was injected? Decided in code
wherever code can decide it (plan §7.2, decision 7 in §13).

The judge records *what was named*; code decides whether it matches. DeepSeek and Sonnet agreed at
kappa = 1.000 on 600 identification items, so the hard part is not reading the name but the rule
applied to it, and the rule has to be stated once and computed the same way every time:

- Wrists read 8.8% under a strict rule and 20.2% under a loose one, the difference being
  `wristwatches` and `hand` at a handful of cells (docs/RESULTS-WRISTS.md §3.1);
- the README described a root-matching rule that did not exist in code (plan §1);
- a judge scored the Qwen run's 大蒜 ("garlic" in Chinese) as "onion" (read_judges J-21).

Decision 7's three rules, all computed for every named item:

    matches   (a) primary: the target, a synonym, an inflection or an unambiguous other name.
              Exact, inflection and curated-synonym matches are scored here; every other name
              goes to the `same_concept` judge, which also handles translations.
    loose     (b) a related concept: matches, or shares a root with the target in code
              ("wristwatches" for "wrists"), or the `related` judge says so ("hand").
    semantic  (c) a continuous score, cosine similarity under an embedding function the caller
              passes in; None when there is none. Reported beside the rules, never a gate.

Names in another script are routed to the judge, never matched or rejected by spelling: script
detection is the "language ID first" step of plan §7.2 (item J8).

The synonym table is deliberately tiny: spelling variants and synonyms nobody would dispute. It
exists to save judge calls on obvious cases, not to encode a theory of relatedness; everything it
does not list is the judge's (`same_concept`), and the judge's calls are what the gold labels
check.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------------- scripts


def script_of(ch: str) -> str | None:
    """The Unicode script family of a letter ('LATIN', 'CJK', 'CYRILLIC', ...); None if not a
    letter. Read from the character name, so no language-ID dependency is needed."""
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    if not name:
        return "UNKNOWN"
    first = name.split()[0]
    return "CJK" if first in ("CJK", "HIRAGANA", "KATAKANA") else first


def script_shares(text: str) -> dict[str, float]:
    """Share of the letters of `text` in each script family."""
    counts: dict[str, int] = {}
    for ch in text:
        s = script_of(ch)
        if s is not None:
            counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())} if total else {}


def has_non_latin(text: str) -> bool:
    return any(s != "LATIN" for s in script_shares(text))


# ---------------------------------------------------------------------------------- normalising

_CURLY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_STRIP = " \t\r\n\"'.,;:!?()[]{}*_`<>«»"
# What a judge or a model puts in front of a name it is reporting.
_FRAME_PREFIXES = ("the word ", "the concept of ", "the idea of ", "the notion of ",
                   "the concept ", "the word", "word ", "concept ")
_ARTICLES = ("the ", "a ", "an ")


def normalise(word: str) -> str:
    """Case-folded, NFKC, curly quotes straightened, outer punctuation and a leading article or
    'the word'/'the concept of' frame removed, inner whitespace collapsed."""
    s = unicodedata.normalize("NFKC", str(word)).translate(_CURLY).casefold().strip(_STRIP)
    for p in _FRAME_PREFIXES:
        if s.startswith(p):
            s = s[len(p):].strip(_STRIP)
            break
    for a in _ARTICLES:
        if s.startswith(a):
            s = s[len(a):].strip(_STRIP)
            break
    return re.sub(r"\s+", " ", s).strip(_STRIP)


def fold_accents(s: str) -> str:
    """'café' -> 'cafe'. Used only to compare Latin-script words."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# ---------------------------------------------------------------------------------- inflection

_PLURAL_TO_SINGULAR = {
    "mice": "mouse", "geese": "goose", "teeth": "tooth", "feet": "foot", "children": "child",
    "people": "person", "men": "man", "women": "woman", "knives": "knife", "wives": "wife",
    "lives": "life", "wolves": "wolf", "leaves": "leaf", "loaves": "loaf", "halves": "half",
    "shelves": "shelf", "calves": "calf", "thieves": "thief", "oxen": "ox", "cacti": "cactus",
    "fungi": "fungus", "octopi": "octopus",
}
_SINGULAR_TO_PLURAL = {v: k for k, v in _PLURAL_TO_SINGULAR.items()}
_VOWELS = set("aeiou")
# Words ending in "s" that are not plurals, so "news" never matches "new".
_NOT_PLURAL = frozenset({"news", "series", "species", "means", "lens", "chaos", "bias", "atlas",
                         "canvas", "gas", "physics", "mathematics", "economics", "politics",
                         "ethics", "measles", "always", "perhaps", "yes", "plus", "bonus"})


def _strip_possessive(w: str) -> str:
    if w.endswith("'s"):
        return w[:-2]
    if w.endswith("s'"):
        return w[:-1]
    return w


def singular_candidates(word: str) -> set[str]:
    """The word and every simple singular it could be the plural of. Over-generating is safe
    here because a match needs both sides to share a candidate ("shoes" yields shoe and sho, and
    only "shoe" will ever meet anything)."""
    w = _strip_possessive(word.casefold())
    out = {w}
    if w in _PLURAL_TO_SINGULAR:
        out.add(_PLURAL_TO_SINGULAR[w])
    if w in _NOT_PLURAL:
        return out
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        out.add(w[:-1])
        if w.endswith("es"):
            out.add(w[:-2])
        if len(w) > 4 and w.endswith("ies") and w[-4] not in _VOWELS:
            out.add(w[:-3] + "y")
    return out


def lemma(word: str) -> str:
    """One canonical singular for a single word (display and pattern building)."""
    w = _strip_possessive(word.casefold())
    if w in _PLURAL_TO_SINGULAR:
        return _PLURAL_TO_SINGULAR[w]
    if w in _NOT_PLURAL:
        return w
    if len(w) > 4 and w.endswith("ies") and w[-4] not in _VOWELS:
        return w[:-3] + "y"
    if len(w) > 4 and re.search(r"(?:ss|x|z|ch|sh)es$", w):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def inflections(word: str) -> set[str]:
    """Surface forms of one word: itself, its singular, and the simple plurals of that."""
    w = word.casefold()
    base = lemma(w)
    out = {w, base, base + "s", base + "es"}
    if re.search(r"[^aeiou]y$", base):
        out.add(base[:-1] + "ies")
    if base in _SINGULAR_TO_PLURAL:
        out.add(_SINGULAR_TO_PLURAL[base])
    return out


def same_inflection(a: str, b: str) -> bool:
    """True when two phrases differ only by simple inflection of their words, word by word."""
    wa, wb = a.split(), b.split()
    if len(wa) != len(wb) or not wa:
        return False
    return all(x == y or bool(singular_candidates(x) & singular_candidates(y))
               for x, y in zip(wa, wb))


# ---------------------------------------------------------------------------------- synonyms

# Only cases nobody would dispute: spelling variants and plain synonyms. Everything else is the
# judge's (`same_concept`).
SYNONYM_SETS: tuple[frozenset[str], ...] = tuple(frozenset(s) for s in (
    {"car", "automobile"},
    {"sofa", "couch"},
    {"film", "movie"},
    {"trash", "garbage", "rubbish"},
    {"trousers", "pants"},
    {"telephone", "phone"},
    {"airship", "dirigible"},
    {"truck", "lorry"},
    {"amphitheater", "amphitheatre"},
    {"theater", "theatre"},
    {"color", "colour"},
    {"gray", "grey"},
    {"jewelry", "jewellery"},
    {"aluminum", "aluminium"},
    {"doughnut", "donut"},
))


def are_synonyms(a: str, b: str) -> bool:
    ca = {lemma(x) for x in singular_candidates(a)} | singular_candidates(a)
    cb = {lemma(x) for x in singular_candidates(b)} | singular_candidates(b)
    return any((ca & s) and (cb & s) for s in SYNONYM_SETS)


# ---------------------------------------------------------------------------------- rules

def code_match(named: str, target: str) -> str | None:
    """'exact', 'inflection' or 'synonym' when code can say the name is the target; None when it
    cannot (which is not a NO: the judge decides those)."""
    n, t = normalise(named), normalise(target)
    if not n or not t:
        return None
    if n == t:
        return "exact"
    if has_non_latin(n) or has_non_latin(t):
        return None
    fn, ft = fold_accents(n), fold_accents(t)
    if fn == ft:
        return "exact"
    if same_inflection(fn, ft):
        return "inflection"
    if are_synonyms(fn, ft):
        return "synonym"
    return None


ROOT_MIN_CHARS = 4  # a shorter shared prefix is noise ("car" in "carpet")


def shares_root(named: str, target: str) -> bool:
    """True when a word of one, in some singular form, starts a word of the other and is at least
    ROOT_MIN_CHARS long: "wrist" starts "wristwatches", "garlic" is a word of "garlic bread",
    "apple" does not start "pineapple"."""
    a = [fold_accents(w) for w in normalise(named).split()]
    b = [fold_accents(w) for w in normalise(target).split()]
    for x in a:
        for y in b:
            for cx in singular_candidates(x):
                for cy in singular_candidates(y):
                    short, long_ = sorted((cx, cy), key=len)
                    if len(short) >= ROOT_MIN_CHARS and long_.startswith(short):
                        return True
    return False


def cosine(u: Sequence[float], v: Sequence[float]) -> float | None:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0 or nv == 0:
        return None
    return float(sum(x * y for x, y in zip(u, v)) / (nu * nv))


@dataclass(frozen=True)
class Identification:
    """The three rules of decision 7 for one named item.

    `matches` and `loose` are None when a judgment they need has not been supplied; `needs`
    lists those judgments as (template id, payload) so the caller can make them. A None is
    missing, never a NO (implementation guide §1 rule 5).
    """

    named: str | None
    target: str
    matches: bool | None
    loose: bool | None
    semantic: float | None
    how: str
    needs: tuple[tuple[str, dict[str, str]], ...] = ()

    @property
    def decided(self) -> bool:
        return self.matches is not None and self.loose is not None

    def to_json(self) -> dict:
        return {"named": self.named, "target": self.target, "matches": self.matches,
                "loose": self.loose, "semantic": self.semantic, "how": self.how,
                "needs": [t for t, _ in self.needs]}


NONE_NAMES = frozenset({"", "none", "nothing", "n/a"})


def identify(named: str | None, target: str, *, same_concept: bool | None = None,
             related: bool | None = None,
             embed: Callable[[str], Sequence[float]] | None = None) -> Identification:
    """Apply decision 7's rules to what a reply named (the `named` judge's field, or None).

    `same_concept` and `related` are the judges' answers for this pair when code could not
    decide; pass None until they exist and read `needs`.
    """
    t = normalise(target)
    if not t:
        raise ValueError("identify needs a target concept")
    if named is None or normalise(named) in NONE_NAMES:
        return Identification(None, target, False, False, None, "none_named")
    n = normalise(named)
    semantic = None
    if embed is not None:
        semantic = cosine(embed(n), embed(t))
    code = code_match(n, t)
    if code is not None:
        return Identification(n, target, True, True, semantic, code)
    # The `same_concept` judgment comes first: if it is YES the `related` one is never needed.
    pair = {"word_a": str(named).strip(), "word_b": target}
    root = shares_root(n, t)
    matches = same_concept
    loose: bool | None
    if matches is None:
        needs = (("same_concept", pair),)
        loose = True if root else None
        how = "needs_judge:non_latin" if has_non_latin(n) else "needs_judge"
        return Identification(n, target, None, loose, semantic, how, needs)
    how = "judge_same" if matches else "judge_different"
    if matches or root:
        return Identification(n, target, bool(matches), True, semantic, how)
    if related is None:
        return Identification(n, target, False, None, semantic, how, (("related", pair),))
    return Identification(n, target, False, bool(related), semantic, how)
