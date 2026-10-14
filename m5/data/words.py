"""m5.data.words - the word lists: the baseline behind every concept vector, the calibration
concepts, and the decoy pool.

Three lists, three jobs:

    BASELINE_WORDS["macar"]  the 100 words whose mean "Tell me about {word}" residual is
                             subtracted from a concept's (`cfg["BASELINE_WORDS"]` names the list)
    CALIBRATION_CONCEPTS     the concepts the calibration set (plan §5.1) and the proxy-validation
                             gate (plan §5.3) are measured on: published detection rates at both
                             ends, concrete and abstract
    DECOY_POOL               ordinary nouns that stand in for "some other concept": the K-way list
                             of `d6`, the mismatch question of `d7`, the judge's decoy-concept
                             control

**The baseline list is copied, not chosen, and kept twice.** `MACAR_BASELINE_VERBATIM` is Macar
et al.'s list entry for entry and in order, *including its duplicate*: "Butterflies" appears twice
(entries 68 and 86), so Macar's 100-prompt mean -- and M2/M3's, which asserted upstream's
extraction defaults -- counts it twice. `BASELINE_WORDS["macar"]`, the list M5 extracts with, is
the same list de-duplicated in order (99 words): `m5.vectors` refuses a repeated baseline word,
because a repetition is a weighting nobody chose. The difference is a reweighting of one word's
contribution: Butterflies goes from 2/100 to 1/99 of the mean, every other word from 1/100 to 1/99.
Both lists are pinned by tests, so the departure from Macar's mean is documented, not silent.

**Decoys must be far from the concept, or the decoy list measures synonymy.** A list offering
Wool beside Silk can be "failed" by a model that perceived the injection perfectly. Every concept
and every decoy carries one of Macar et al.'s twenty semantic categories (Fig. 18); a draw excludes
the concept's own category and its neighbours, excludes anything lexically related (Wrists /
Wristwatches), and takes at most one decoy per category, so the list is spread across meaning.
A caller with a better distance -- cosine between concept vectors at a reference layer, as
Sridharan's mismatch design uses -- passes it as `distance` and the farthest half is kept first.

Every word here passes `m5.safety.check_concept`; the harmful arm is refused by name before any
of this runs.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from m5.backend.base import derive_seed
from m5.safety import check_concept

# =====================================================================================
# Baseline words
# =====================================================================================

# VERBATIM. Macar et al. 2026, vendored harness: upstream/introspection_mechanisms/src/
# vector_utils.py:384-405 (`DEFAULT_BASELINE_WORDS`, pinned commit 5d5d9b48); identical entry for
# entry and in order to Lindsey 2026's list (vault Papers/lindsey-2026-emergent-introspective-
# awareness.md:871-885). 100 entries, 99 distinct: "Butterflies" is entry 68 and entry 86.
# `get_baseline_words(n=100)` returns all of it; Macar's mean is over these 100 prompts
# (read_macar_protocol.md §3.3).
MACAR_BASELINE_VERBATIM: tuple[str, ...] = (
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence",
    "Bicycles", "Chairs", "Orchestras", "Sand", "Pottery",
    "Arrowheads", "Jewelry", "Daffodils", "Plateaus", "Estuaries",
    "Quilts", "Moments", "Bamboo", "Ravines", "Archives",
    "Hieroglyphs", "Stars", "Clay", "Fossils", "Wildlife",
    "Flour", "Traffic", "Bubbles", "Honey", "Geodes",
    "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes",
    "Anthills", "Galaxies", "Poverty", "Diamonds", "Universes",
    "Vinegar", "Nebulae", "Knowledge", "Marble", "Fog",
    "Rivers", "Scrolls", "Silhouettes", "Marbles", "Cakes",
    "Valleys", "Whispers", "Pendulums", "Towers", "Tables",
    "Glaciers", "Whirlpools", "Jungles", "Wool", "Anger",
    "Ramparts", "Flowers", "Research", "Hammers", "Clouds",
    "Justice", "Dogs", "Butterflies", "Needles", "Fortresses",
    "Bonfires", "Skyscrapers", "Caravans", "Patience", "Bacon",
    "Velocities", "Smoke", "Electricity", "Sunsets", "Anchors",
    "Parchments", "Courage", "Statues", "Oxygen", "Time",
    "Butterflies", "Fabric", "Pasta", "Snowflakes", "Mountains",
    "Echoes", "Pianos", "Sanctuaries", "Abysses", "Air",
    "Dewdrops", "Gardens", "Literature", "Rice", "Enigmas",
)

# The duplicate the verbatim list carries, so a test can tell it from a copying error.
MACAR_BASELINE_DUPLICATES: tuple[str, ...] = ("Butterflies",)


def _dedupe(words: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out = []
    for w in words:
        if w.casefold() not in seen:
            seen.add(w.casefold())
            out.append(w)
    return tuple(out)


# The lists M5 extracts with, by the name `cfg["BASELINE_WORDS"]` gives. "macar" is the verbatim
# list with its repeated "Butterflies" removed (its first occurrence kept): Macar's mean counts that
# word twice, M5's counts each word once (module docstring).
BASELINE_WORDS: dict[str, tuple[str, ...]] = {"macar": _dedupe(MACAR_BASELINE_VERBATIM)}


def baseline(name: str) -> tuple[str, ...]:
    """The baseline list `cfg["BASELINE_WORDS"]` names (de-duplicated); unknown names raise."""
    try:
        return BASELINE_WORDS[name]
    except KeyError:
        raise KeyError(f"unknown baseline word list {name!r}; one of {sorted(BASELINE_WORDS)}") from None


# =====================================================================================
# Semantic categories (Macar et al. 2026, Fig. 18; vault macar-2026-...md:555-558)
# =====================================================================================

CATEGORIES: tuple[str, ...] = (
    "foods", "animals", "vehicles", "sports", "weather", "science", "actions", "plants",
    "buildings", "music", "materials", "abstract", "professions", "body_parts", "social",
    "emotions", "geography", "everyday_objects", "colors", "time",
)

# Categories close enough that a decoy from one could pass for a concept in the other (garlic is
# a plant and a food; silk is a material in an everyday object). Written for M5; symmetric.
_NEIGHBOUR_PAIRS: tuple[tuple[str, str], ...] = (
    ("foods", "plants"), ("foods", "animals"), ("weather", "geography"), ("weather", "time"),
    ("materials", "everyday_objects"), ("materials", "colors"), ("buildings", "geography"),
    ("buildings", "everyday_objects"), ("body_parts", "actions"), ("sports", "actions"),
    ("sports", "vehicles"), ("music", "professions"), ("abstract", "emotions"),
    ("abstract", "social"), ("abstract", "science"), ("social", "emotions"),
    ("social", "professions"), ("science", "geography"),
)
NEIGHBOURS: dict[str, frozenset[str]] = {
    c: frozenset({b for a, b in _NEIGHBOUR_PAIRS if a == c} | {a for a, b in _NEIGHBOUR_PAIRS if b == c})
    for c in CATEGORIES
}


# =====================================================================================
# Calibration concepts
# =====================================================================================

@dataclass(frozen=True)
class Concept:
    """A calibration concept and what is published about it.

    `band` is the end of Macar et al.'s detection distribution it sits at (Gemma3-27B-it, L37,
    alpha 4, 100 trials): "strong" (named as high-detection) or "weak" (named at or near 0%).
    `macar_detection` is the per-concept rate when the paper states one, else None.
    """

    word: str
    category: str
    kind: str                       # "concrete" | "abstract"
    band: str                       # "strong" | "weak"
    macar_detection: float | None
    source: str


# Every entry is named in Macar et al. 2026 with its detection outcome, so the calibration set has
# a published answer at both ends. Wrists and Velocity are bottom-10 labels of Fig. 19 (read from
# the figure; carried over from m2/config.py:296-301, where M2/M3 ran them), with no rate in text.
CALIBRATION: tuple[Concept, ...] = (
    # strong: concrete, sensory-rich, distinctive (MAC:570)
    Concept("Garlic", "foods", "concrete", "strong", 1.00, "Macar 2026 B.5, vault macar-2026-...md:584 (100%)"),
    Concept("Chocolate", "foods", "concrete", "strong", 0.99, "Macar 2026 B.5, vault macar-2026-...md:576 (99%)"),
    Concept("Trees", "plants", "concrete", "strong", 0.97, "Macar 2026 B.5, vault macar-2026-...md:580 and :1058 (97%)"),
    Concept("Bread", "foods", "concrete", "strong", 0.91, "Macar 2026 App., vault macar-2026-...md:1058 (91%)"),
    Concept("Thunderstorms", "weather", "concrete", "strong", None, "Macar 2026 B.4, vault macar-2026-...md:570 (named high)"),
    Concept("Scorpions", "animals", "concrete", "strong", None, "Macar 2026 B.4, vault macar-2026-...md:570 (named high)"),
    # weak: abstract terms and concrete but generic nouns (MAC:570)
    Concept("Silk", "materials", "concrete", "weak", 0.0, "Macar 2026 B.5, vault macar-2026-...md:570, :590 (0%)"),
    Concept("Pillows", "everyday_objects", "concrete", "weak", 0.0, "Macar 2026 B.4, vault macar-2026-...md:570 (0%)"),
    Concept("Mirrors", "everyday_objects", "concrete", "weak", 0.0, "Macar 2026 B.5, vault macar-2026-...md:570, :596 (0%)"),
    Concept("Monuments", "buildings", "concrete", "weak", 0.0, "Macar 2026 App., vault macar-2026-...md:1058 (0%)"),
    Concept("Irony", "abstract", "abstract", "weak", 0.0, "Macar 2026 B.5, vault macar-2026-...md:570, :600 (0%)"),
    Concept("Karma", "abstract", "abstract", "weak", 0.0, "Macar 2026 B.4, vault macar-2026-...md:570 (0%)"),
    Concept("Skepticism", "abstract", "abstract", "weak", 0.0, "Macar 2026 B.4, vault macar-2026-...md:570 (0%)"),
    Concept("Wrists", "body_parts", "concrete", "weak", None, "Macar 2026 Fig. 19 bottom-10 label; m2/config.py:296-301"),
    Concept("Velocity", "science", "abstract", "weak", None, "Macar 2026 Fig. 19 bottom-10 label; m2/config.py:296-301"),
)

CALIBRATION_CONCEPTS: tuple[str, ...] = tuple(c.word for c in CALIBRATION)

# VERBATIM. Lindsey 2026's 50 concepts, the "Original (mixed)" column of Macar's 500 (25.3%
# detection as a group, Fig. 18): vault Papers/lindsey-2026-emergent-introspective-awareness.md:
# 892-898 (spaces after commas restored; read_macar_protocol.md §3.4(a)). Same words, lower-cased,
# as Sridharan's `CONCEPT_NOUNS` (agastyasridharan/introspection src/introspection/constants.py:1-52).
LINDSEY_CONCEPTS: tuple[str, ...] = (
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions", "Cameras", "Lightning",
    "Constellations", "Treasures", "Phones", "Trees", "Avalanches", "Mirrors", "Fountains",
    "Quarries", "Sadness", "Xylophones", "Secrecy", "Oceans", "Information", "Deserts",
    "Kaleidoscopes", "Sugar", "Vegetables", "Poetry", "Aquariums", "Bags", "Peace", "Caverns",
    "Memories", "Frosts", "Volcanoes", "Boulders", "Harmonies", "Masquerades", "Rubber", "Plastic",
    "Blood", "Amphitheaters", "Contraptions", "Youths", "Dynasties", "Snow", "Dirigibles",
    "Algorithms", "Denim", "Monoliths", "Milk", "Bread", "Silver",
)

# Category of each Lindsey concept. Written for M5 (Macar files all 50 under "Original (mixed)");
# used only to keep decoys away from them.
_LINDSEY_CATEGORIES: dict[str, str] = {
    "Dust": "materials", "Satellites": "science", "Trumpets": "music", "Origami": "actions",
    "Illusions": "abstract", "Cameras": "everyday_objects", "Lightning": "weather",
    "Constellations": "science", "Treasures": "everyday_objects", "Phones": "everyday_objects",
    "Trees": "plants", "Avalanches": "weather", "Mirrors": "everyday_objects",
    "Fountains": "buildings", "Quarries": "geography", "Sadness": "emotions",
    "Xylophones": "music", "Secrecy": "social", "Oceans": "geography", "Information": "abstract",
    "Deserts": "geography", "Kaleidoscopes": "everyday_objects", "Sugar": "foods",
    "Vegetables": "foods", "Poetry": "abstract", "Aquariums": "buildings",
    "Bags": "everyday_objects", "Peace": "social", "Caverns": "geography", "Memories": "abstract",
    "Frosts": "weather", "Volcanoes": "geography", "Boulders": "geography", "Harmonies": "music",
    "Masquerades": "social", "Rubber": "materials", "Plastic": "materials", "Blood": "body_parts",
    "Amphitheaters": "buildings", "Contraptions": "everyday_objects", "Youths": "social",
    "Dynasties": "social", "Snow": "weather", "Dirigibles": "vehicles", "Algorithms": "science",
    "Denim": "materials", "Monoliths": "buildings", "Milk": "foods", "Bread": "foods",
    "Silver": "materials",
}


# =====================================================================================
# Decoy pool (written for M5)
# =====================================================================================

# Eight ordinary nouns per category, capitalised and mixed singular/plural like the concept lists
# so that no option's form gives it away. None is a calibration concept, a Lindsey concept or a
# baseline word, or lexically related to one (checked at import): a decoy that is also a concept
# could be the injected concept on another trial, and one that is a baseline word sits inside every
# concept vector's subtraction.
DECOY_POOL: dict[str, tuple[str, ...]] = {
    "foods": ("Pancakes", "Lentils", "Oatmeal", "Pretzels", "Dumplings", "Yogurt", "Noodles", "Walnuts"),
    "animals": ("Giraffes", "Otters", "Penguins", "Hedgehogs", "Flamingos", "Badgers", "Pelicans", "Tortoises"),
    "vehicles": ("Tractors", "Canoes", "Scooters", "Ferries", "Trams", "Sailboats", "Helicopters", "Wagons"),
    "sports": ("Tennis", "Archery", "Rowing", "Badminton", "Gymnastics", "Volleyball", "Skiing", "Bowling"),
    "weather": ("Drizzle", "Hailstones", "Breezes", "Rainbows", "Humidity", "Monsoons", "Sleet", "Heatwaves"),
    "science": ("Molecules", "Gravity", "Photosynthesis", "Electrons", "Friction", "Isotopes", "Enzymes", "Wavelengths"),
    "actions": ("Juggling", "Whistling", "Knitting", "Climbing", "Hiking", "Sketching", "Swimming", "Sneezing"),
    "plants": ("Cacti", "Tulips", "Ivy", "Moss", "Sunflowers", "Orchids", "Clover", "Lavender"),
    "buildings": ("Lighthouses", "Cathedrals", "Barns", "Igloos", "Pagodas", "Windmills", "Greenhouses", "Bungalows"),
    "music": ("Violins", "Banjos", "Lullabies", "Choirs", "Accordions", "Tambourines", "Melodies", "Cellos"),
    "materials": ("Copper", "Leather", "Granite", "Cotton", "Porcelain", "Cardboard", "Linen", "Velvet"),
    "abstract": ("Freedom", "Curiosity", "Nostalgia", "Ambiguity", "Destiny", "Infinity", "Balance", "Simplicity"),
    "professions": ("Plumbers", "Librarians", "Carpenters", "Astronomers", "Pilots", "Firefighters", "Architects", "Dentists"),
    "body_parts": ("Elbows", "Eyebrows", "Knees", "Ankles", "Shoulders", "Fingernails", "Eyelashes", "Knuckles"),
    "social": ("Friendship", "Neighbourhoods", "Committees", "Traditions", "Festivals", "Elections", "Partnerships", "Etiquette"),
    "emotions": ("Joy", "Envy", "Relief", "Boredom", "Gratitude", "Pride", "Loneliness", "Excitement"),
    "geography": ("Peninsulas", "Lagoons", "Canyons", "Islands", "Meadows", "Fjords", "Prairies", "Marshes"),
    "everyday_objects": ("Umbrellas", "Teapots", "Staplers", "Keychains", "Doormats", "Envelopes", "Paperclips", "Lanterns"),
    "colors": ("Turquoise", "Crimson", "Beige", "Magenta", "Scarlet", "Teal", "Ochre", "Maroon"),
    "time": ("Mornings", "Weekends", "Centuries", "Midnight", "Holidays", "Afternoons", "Birthdays", "Anniversaries"),
}

DECOY_WORDS: tuple[str, ...] = tuple(w for c in CATEGORIES for w in DECOY_POOL[c])


# =====================================================================================
# Relatedness and categories
# =====================================================================================

def _forms(word: str) -> set[str]:
    """Lower-cased forms of one word with simple plural endings removed."""
    w = word.casefold().strip()
    out = {w}
    if w.endswith("ies") and len(w) > 4:
        out.add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        out.add(w[:-2])
    if w.endswith("s") and len(w) > 2:
        out.add(w[:-1])
    return out


def related(a: str, b: str) -> bool:
    """True when two words (or multi-word names) share a form or a stem of four letters or more.

    Lexical, and deliberately over-inclusive: "Irony" and "Iron" count as related, which only
    costs a decoy. Used to keep a concept's own inflections out of its decoys, mismatch
    questions and localisation segments (C3 in the lessons register: `Wrists` must catch
    `wristwatch`).
    """
    for x in str(a).split():
        for y in str(b).split():
            fx, fy = _forms(x), _forms(y)
            if fx & fy:
                return True
            for p in fx:
                for q in fy:
                    short, long_ = (p, q) if len(p) <= len(q) else (q, p)
                    if len(short) >= 4 and long_.startswith(short):
                        return True
    return False


def category_of(word: str) -> str | None:
    """The semantic category of a known concept or decoy, case-insensitively; None if unknown."""
    key = str(word).casefold()
    for c in CALIBRATION:
        if c.word.casefold() == key:
            return c.category
    for w, cat in _LINDSEY_CATEGORIES.items():
        if w.casefold() == key:
            return cat
    for cat, words in DECOY_POOL.items():
        if any(w.casefold() == key for w in words):
            return cat
    return None


def baseline_overlap(concept: str, list_name: str = "macar") -> list[str]:
    """Baseline words lexically related to `concept`.

    Velocity is the case that matters: "Velocities" is one of Macar's 100 baseline words, so 1/100
    of Velocity's own plural is subtracted from its vector. That is how Macar, M2 and M3 built it;
    this makes it visible rather than silent.
    """
    return [w for w in baseline(list_name) if related(w, concept)]


# =====================================================================================
# Drawing decoys
# =====================================================================================

def _shuffled(items: Sequence, rng: random.Random) -> list:
    """Fisher-Yates on randrange, whose stream is stable across Python versions."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = rng.randrange(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def draw_decoys(concept: str, k: int, seed: int, *, category: str | None = None,
                exclude: Iterable[str] = (), distance: Callable[[str], float] | None = None,
                pool: dict[str, Sequence[str]] | None = None) -> list[str]:
    """`k` decoys far from `concept`, deterministic in (`seed`, concept, k).

    Excluded: the concept's category and its neighbours (`category` overrides the lookup; a
    concept nobody has categorised gets lexical exclusion only, so pass a category or a
    `distance` for new concepts), every word lexically related to the concept or to anything in
    `exclude`. With `distance` (word -> larger is farther, e.g. 1 - cosine between concept
    vectors), only the farther half of the remaining candidates is drawn from. Decoys come from
    distinct categories while there are categories left.
    """
    check_concept(concept)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    pool = DECOY_POOL if pool is None else pool
    cat = category if category is not None else category_of(concept)
    if cat is not None and cat not in CATEGORIES:
        raise ValueError(f"unknown category {cat!r}; one of {CATEGORIES}")
    banned = {cat} | set(NEIGHBOURS.get(cat, ())) if cat else set()
    avoid = [concept, *exclude]
    candidates = [(c, w) for c in pool if c not in banned for w in pool[c]
                  if not any(related(w, a) for a in avoid)]
    if distance is not None:
        ranked = sorted(candidates, key=lambda cw: (-float(distance(cw[1])), cw[1]))
        candidates = ranked[: max(k, (len(ranked) + 1) // 2)]
    if len(candidates) < k:
        raise ValueError(f"only {len(candidates)} decoys are far enough from {concept!r}; asked for {k}")
    rng = random.Random(derive_seed(seed, "decoys", str(concept).casefold(), k))
    by_cat: dict[str, list[str]] = {}
    for c, w in candidates:
        by_cat.setdefault(c, []).append(w)
    order = _shuffled(sorted(by_cat), rng)
    queues = {c: _shuffled(by_cat[c], rng) for c in order}
    out: list[str] = []
    while len(out) < k:  # one per category per round, so the list spreads across meaning
        for c in order:
            if queues[c] and len(out) < k:
                out.append(queues[c].pop())
    return out


# =====================================================================================
# Import-time checks: a malformed list fails here, not halfway through a run
# =====================================================================================

def _check() -> None:
    if len(MACAR_BASELINE_VERBATIM) != 100:
        raise AssertionError("Macar's baseline list has 100 entries")
    dups = sorted({w for w in MACAR_BASELINE_VERBATIM if MACAR_BASELINE_VERBATIM.count(w) > 1})
    if tuple(dups) != MACAR_BASELINE_DUPLICATES:
        raise AssertionError(f"undocumented duplicate in the Macar baseline: {dups}")
    for name, words in BASELINE_WORDS.items():
        if len({w.casefold() for w in words}) != len(words):
            raise AssertionError(f"baseline list {name!r} repeats a word; m5.vectors refuses it")
    if len(BASELINE_WORDS["macar"]) != 99:
        raise AssertionError("the de-duplicated Macar list has 99 words")
    if len(LINDSEY_CONCEPTS) != 50 or set(LINDSEY_CONCEPTS) != set(_LINDSEY_CATEGORIES):
        raise AssertionError("Lindsey's list has 50 concepts, each categorised")
    for c in CALIBRATION:
        if c.category not in CATEGORIES or c.kind not in ("concrete", "abstract") \
                or c.band not in ("strong", "weak"):
            raise AssertionError(f"malformed calibration entry {c}")
    if set(DECOY_POOL) != set(CATEGORIES):
        raise AssertionError("the decoy pool has one entry per category")
    if len(set(w.casefold() for w in DECOY_WORDS)) != len(DECOY_WORDS):
        raise AssertionError("a decoy appears twice in the pool")
    for word in (*CALIBRATION_CONCEPTS, *LINDSEY_CONCEPTS, *DECOY_WORDS, *MACAR_BASELINE_VERBATIM):
        check_concept(word)
    protected = (*CALIBRATION_CONCEPTS, *LINDSEY_CONCEPTS, *MACAR_BASELINE_VERBATIM)
    for w in DECOY_WORDS:
        clash = [p for p in protected if related(w, p)]
        if clash:
            raise AssertionError(f"decoy {w!r} is related to {clash}")


_check()
