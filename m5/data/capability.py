"""m5.data.capability - the `s5` capability panel: auto-verifiable items, generated here or loaded
from public benchmark files the operator keeps outside the repository.

`s5` compares the steered and the unsteered model on the *same* items (a paired test), so what
matters is that every item can be scored mechanically and that the panel cannot saturate. Mentor
feedback was that hand-written capability questions saturate on stronger models; the plan's answer
(§3.2, §12 item 1) is a large candidate pool -- generated families plus MMLU-Pro, GPQA Diamond and
GSM8K -- from which the per-model setup keeps the items the unsteered model passes 50-90% of the
time (`cfg["S5_PASS_BAND"]`, `cfg["S5_CANDIDATES"]`, `cfg["S5_SAMPLES_PER_ITEM"]`). Harder items for
a stronger model, by selection rather than by editing.

**Every question ends with `markers.CAPABILITY_INSTRUCTION`** ("Give only the final answer after
'Answer:'."), and `CapabilityItem.check(reply)` reads only the text after the *last* "Answer:".
A reply with no "Answer:" fails: under steering, losing the format is losing the instruction, and
the unsteered arm is held to the same rule on the same item. `parse_answer` returns None for that
case so a scorer can count it separately from a wrong answer (missing is not zero).

**What the checkers forgive, and what they do not.** Case, surrounding quotes, markdown emphasis,
a trailing full stop and thousands separators are forgiven: they are formatting, and M3 scored a
lower-case option letter as wrong once (lessons C2). The value is not: an integer item compares the
first number after "Answer:", a date item the first YYYY-MM-DD, a choice item one standalone
letter.

Generated families (WRITTEN FOR M5), each at difficulty 1-3:

    arithmetic    multi-step expressions             int
    word_problem  GSM8K-style two- and three-step    int
    units         exact metric and time conversions  int / decimal
    strings       reversal, letter counts, sorting   text / int / list
    dates         weekdays, offsets, differences     text / date / int
    logic         orderings, sequences, syllogisms   text / int / yesno
    constraints   IFEval-style programmatic rules    constraint (`answer` is one reply that passes)

Nothing in any family mentions a calibration concept or a harmful-arm word (checked at generation).
"""

from __future__ import annotations

import datetime as _dt
import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from m5.backend.base import derive_seed
from m5.data import markers
from m5.safety import Refused, assert_outside_repo, check_texts

FAMILIES: tuple[str, ...] = ("arithmetic", "word_problem", "units", "strings", "dates", "logic",
                             "constraints")
KINDS: tuple[str, ...] = ("int", "decimal", "text", "list", "date", "choice", "yesno", "constraint")
CHOICE_LETTERS = "ABCDEFGHIJ"


# =====================================================================================
# The item and its checker
# =====================================================================================

_ANSWER_MARK = re.compile(r"answer[\s*_`]*[:：]", re.IGNORECASE)
_NUMBER = re.compile(r"[-−]?\d[\d,]*(?:\.\d+)?")
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def parse_answer(reply: str, *, multiline: bool = False) -> str | None:
    """The text after the last "Answer:" in `reply`, up to the first blank line; the first line
    only unless `multiline`. None when the reply has no "Answer:" or nothing after it."""
    marks = list(_ANSWER_MARK.finditer(reply or ""))
    if not marks:
        return None
    tail = reply[marks[-1].end():]
    tail = re.sub(r"^[\s*_`]+", "", tail)
    tail = tail.split("\n\n", 1)[0].strip()
    if not multiline:
        tail = tail.split("\n", 1)[0].strip()
    return tail or None


def _clean(text: str) -> str:
    """Markdown emphasis, code ticks, surrounding quotes and trailing punctuation removed."""
    t = re.sub(r"[*_`]+", "", text).strip()
    t = t.strip("\"'“”‘’ ")
    t = re.sub(r"[.!?;,:]+$", "", t).strip()
    return t.strip("\"'“”‘’ ")


def _norm(text: str) -> str:
    return " ".join(_clean(text).casefold().split())


def _first_number(text: str) -> Decimal | None:
    m = _NUMBER.search(text.replace("−", "-"))
    if not m:
        return None
    raw = m.group(0).replace("−", "-")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", raw):
        raw = raw.replace(",", "")
    elif "," in raw:
        raw = raw.split(",", 1)[0]
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _choice_letter(text: str) -> str | None:
    t = _clean(text)
    if re.fullmatch(r"\(?([A-Ja-j])\)?", t):
        return t.strip("()").upper()
    m = re.search(r"\(([A-Ja-j])\)", t)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", t)
    return m.group(1) if m else None


def words_of(text: str) -> list[str]:
    return _WORD.findall(text)


@dataclass(frozen=True)
class CapabilityItem:
    """One auto-verifiable item. `answer` is the correct answer (for a constraint item, one reply
    that satisfies the constraint); `params` holds what the checker needs, as (name, value) pairs so
    the item stays hashable and JSON-friendly."""

    id: str
    family: str
    question: str
    answer: str
    kind: str
    params: tuple[tuple[str, Any], ...] = ()
    source: str = "generated"
    difficulty: int = 0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: unknown kind {self.kind!r}")
        if not self.question.endswith(markers.CAPABILITY_INSTRUCTION):
            raise ValueError(f"{self.id}: the question must end with the capability instruction")

    def param(self, name: str) -> Any:
        for k, v in self.params:
            if k == name:
                return v
        raise KeyError(f"{self.id}: no parameter {name!r}")

    def check(self, reply: str) -> bool:
        """True when the text after the last "Answer:" in `reply` is correct."""
        got = parse_answer(reply, multiline=self.kind == "constraint")
        if got is None:
            return False
        k = self.kind
        if k == "int":
            v = _first_number(got)
            return v is not None and v == v.to_integral_value() and int(v) == int(self.answer)
        if k == "decimal":
            v = _first_number(got)
            return v is not None and abs(v - Decimal(self.answer)) <= Decimal(str(dict(self.params).get("tol", "0.005")))
        if k == "text":
            return _norm(got) == _norm(self.answer)
        if k == "list":
            return [_norm(x) for x in re.split(r",|\band\b", _clean(got)) if _norm(x)] == \
                   [_norm(x) for x in self.answer.split(",")]
        if k == "date":
            m = _DATE.search(got)
            return m is not None and m.group(0) == self.answer
        if k == "choice":
            return _choice_letter(got) == self.answer
        if k == "yesno":
            first = re.match(r"[a-z]*", _norm(got)).group(0)
            return first == self.answer
        return _CONSTRAINTS[self.param("rule")](got, dict(self.params))

    def to_json(self) -> dict:
        return {"id": self.id, "family": self.family, "question": self.question,
                "answer": self.answer, "kind": self.kind, "params": [list(p) for p in self.params],
                "source": self.source, "difficulty": self.difficulty}


# =====================================================================================
# IFEval-style constraint checkers (each reads the whole answer block)
# =====================================================================================

def _c_word_count(a: str, p: dict) -> bool:
    return len(words_of(a)) == p["n"]


def _c_lowercase(a: str, p: dict) -> bool:
    letters = [c for c in a if c.isalpha()]
    return len(letters) >= 3 and not any(c.isupper() for c in letters)


def _c_caps_list(a: str, p: dict) -> bool:
    items = [x.strip() for x in _clean(a).split(",")]
    return len(items) == p["n"] and all(re.fullmatch(r"[A-Z]+(?:[ '-][A-Z]+)*", x) for x in items)


def _c_starts_with(a: str, p: dict) -> bool:
    ws = [w.casefold() for w in words_of(a)]
    return len(ws) == p["n"] and len(set(ws)) == len(ws) and all(w.startswith(p["letter"]) for w in ws)


def _c_no_letter(a: str, p: dict) -> bool:
    return len(words_of(a)) >= p["n"] and p["letter"] not in a.casefold()


def _c_ends_with(a: str, p: dict) -> bool:
    ws = words_of(a)
    return bool(ws) and ws[-1].casefold() == p["word"]


def _c_keyword_twice(a: str, p: dict) -> bool:
    return len(re.findall(rf"\b{re.escape(p['word'])}\b", a, re.IGNORECASE)) == 2


def _c_numbered_list(a: str, p: dict) -> bool:
    lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
    nums = [re.match(r"(\d+)\.\s+\S", ln) for ln in lines]
    return all(nums) and [int(m.group(1)) for m in nums] == list(range(1, p["n"] + 1))


_CONSTRAINTS: dict[str, Callable[[str, dict], bool]] = {
    "word_count": _c_word_count, "lowercase": _c_lowercase, "caps_list": _c_caps_list,
    "starts_with": _c_starts_with, "no_letter": _c_no_letter, "ends_with": _c_ends_with,
    "keyword_twice": _c_keyword_twice, "numbered_list": _c_numbered_list,
}


# =====================================================================================
# Generators (WRITTEN FOR M5)
# =====================================================================================

_NAMES = ("Anna", "Ben", "Carla", "David", "Elena", "Farid", "Grace", "Hugo", "Ines", "Jonas",
          "Kofi", "Lena", "Mateo", "Nadia", "Omar", "Priya")
# Common nouns for the string tasks; none is a calibration concept or related to one.
_WORDS = ("lantern", "umbrella", "notebook", "calendar", "keyboard", "harbour", "festival",
          "envelope", "staircase", "telescope", "backpack", "triangle", "carousel", "pineapple",
          "dolphin", "elephant", "kangaroo", "compass", "library", "painter", "journey", "blanket",
          "balcony", "cabinet", "ladder", "pencil", "window", "garden", "rocket", "puzzle",
          "button", "candle", "bottle", "island", "meadow", "castle", "bridge", "planet",
          "mountain", "whistle", "ribbon", "anchor", "harvest", "pocket", "saddle", "kitten",
          "parrot", "tunnel", "wallet", "jacket", "teacher", "sailor", "picnic", "orange",
          "bicycle", "cottage", "highway", "quarter", "strawberry", "banana", "bookkeeper",
          "committee", "balloon", "coffee", "letter", "success", "address", "necessary",
          "assessment", "mississippi", "occurrence", "programming", "accommodation")
_TOPICS = ("the sea", "a library", "a quiet morning", "a city park", "a long journey", "the moon",
           "a market", "a museum", "a river", "a mountain path", "a train ride", "a classroom",
           "a harbour", "a village square", "a bicycle", "a bridge", "an old letter", "a lighthouse",
           "a football match", "a sailing boat", "a post office", "a quiet street", "a busy station",
           "a school play")
_FILLER = ("the", "quiet", "street", "was", "empty", "before", "morning", "light", "slowly",
           "filled", "every", "corner", "with", "soft", "colour", "and", "sound", "people", "began",
           "walking", "towards", "small", "shops", "near", "old", "bridge", "over", "river", "a",
           "cat", "sat", "on", "sunny", "wall", "all", "day", "long", "birds", "sang", "loudly",
           "in", "warm", "air", "kids", "ran", "up", "hill", "to", "town", "fast", "but", "it",
           "got", "dark", "soon")
_LETTER_WORDS = {
    "b": ("blue", "boat", "bring", "basket", "bright", "bell", "bake", "busy"),
    "c": ("cloud", "candle", "calm", "copper", "count", "cabin", "clear", "crisp"),
    "d": ("door", "drift", "daily", "dusk", "dance", "deep", "dial", "draw"),
    "m": ("moon", "map", "music", "mild", "mountain", "meadow", "mark", "mellow"),
    "p": ("paper", "pond", "pale", "piano", "pocket", "path", "plain", "pebble"),
    "s": ("stone", "simple", "street", "soft", "sand", "season", "slow", "sound"),
    "t": ("table", "tide", "tower", "tall", "tune", "track", "timber", "travel"),
    "f": ("field", "frame", "fresh", "fold", "ferry", "fair", "flag", "focus"),
    "g": ("gate", "green", "glass", "grain", "gentle", "guide", "garden", "gravel"),
    "h": ("harbour", "hill", "honest", "habit", "hollow", "handle", "harvest", "humble"),
    "l": ("lamp", "ladder", "lake", "lively", "label", "lantern", "leaf", "local"),
    "r": ("river", "road", "rapid", "ribbon", "rocket", "rain", "ready", "record"),
    "w": ("window", "wall", "wide", "wagon", "warm", "whistle", "winter", "wonder"),
}
_ENDINGS = ("today", "again", "outside", "slowly", "together", "tomorrow")
_KEYWORDS = ("light", "window", "river", "morning", "quiet", "road")
_NONSENSE = ("bloops", "razzies", "lazzies", "wumps", "florbs", "glims", "trobs", "zinks")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh")


def _pick(rng: random.Random, seq: Sequence):
    return seq[rng.randrange(len(seq))]


def _between(rng: random.Random, lo: int, hi: int) -> int:
    return lo + rng.randrange(hi - lo + 1)


def _sample(rng: random.Random, seq: Sequence, k: int) -> list:
    pool = list(seq)
    out = []
    for _ in range(k):
        out.append(pool.pop(rng.randrange(len(pool))))
    return out


def _date_text(d: _dt.date) -> str:
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def _rand_date(rng: random.Random, lo: int = 1950, hi: int = 2030) -> _dt.date:
    start = _dt.date(lo, 1, 1).toordinal()
    end = _dt.date(hi, 12, 31).toordinal()
    return _dt.date.fromordinal(_between(rng, start, end))


Gen = tuple[str, str, str, tuple[tuple[str, Any], ...]]  # body, answer, kind, params


def _arithmetic(rng: random.Random, d: int) -> Gen:
    if d == 1:
        a, b, c = _between(rng, 11, 29), _between(rng, 11, 29), _between(rng, 10, 99)
        return f"What is {a} × {b} + {c}?", str(a * b + c), "int", ()
    if d == 2:
        a, b, c, e = (_between(rng, 12, 49) for _ in range(4))
        return f"What is {a} × {b} - {c} × {e}?", str(a * b - c * e), "int", ()
    a, b = _between(rng, 100, 999), _between(rng, 100, 999)
    c, e, q = _between(rng, 3, 9), _between(rng, 3, 12), _between(rng, 10, 60)
    return (f"What is ({a} + {b}) × {c} - {e * q} ÷ {e}?", str((a + b) * c - q), "int", ())


def _word_problem(rng: random.Random, d: int) -> Gen:
    scale = (1, 3, 9)[d - 1]
    t = rng.randrange(5)
    name = _pick(rng, _NAMES)
    if t == 0:
        s, b = _between(rng, 4, 9), _between(rng, 12, 30) * scale
        lent, back = _between(rng, 10, 40) * scale, _between(rng, 5, 20) * scale
        return (f"A library has {s} shelves with {b} books on each shelf. It lends out {lent} books "
                f"and then receives {back} returned books. How many books are on the shelves now?",
                str(s * b - lent + back), "int", ())
    if t == 1:
        packs, per = _between(rng, 4, 9) * scale, _between(rng, 6, 12)
        k, g = _between(rng, 2, 6), _between(rng, 1, 3)          # g * k < packs * per
        return (f"{name} buys {packs} packs of pencils with {per} pencils in each pack and gives "
                f"{g} pencils to each of {k} classmates. How many pencils does {name} have left?",
                str(packs * per - g * k), "int", ())
    if t == 2:
        rate, m1, m2 = _between(rng, 8, 30), _between(rng, 10, 45) * scale, _between(rng, 10, 45)
        jam = _between(rng, 5, 40)
        return (f"A printer prints {rate} pages per minute. It runs for {m1} minutes in the morning "
                f"and {m2} minutes in the afternoon, and {jam} pages are thrown away after a paper "
                f"jam. How many usable pages were printed?", str(rate * (m1 + m2) - jam), "int", ())
    if t == 3:
        adult, child = _between(rng, 8, 25), _between(rng, 3, 12)
        na, nc = _between(rng, 2, 6) * scale, _between(rng, 2, 9) * scale
        total = adult * na + child * nc
        paid = (total // 50 + 1) * 50
        return (f"Museum tickets cost {adult} dollars for adults and {child} dollars for children. A "
                f"group of {na} adults and {nc} children pays with {paid} dollars. How much change "
                f"do they receive, in dollars?", str(paid - total), "int", ())
    weekly, weeks = _between(rng, 5, 25), _between(rng, 4, 20) * scale
    a, b = _between(rng, 10, 60), _between(rng, 5, 30)
    weeks = max(weeks, (a + b) // weekly + 2)                    # never a negative balance
    return (f"{name} saves {weekly} dollars a week for {weeks} weeks, then spends {a} dollars on a "
            f"concert ticket and {b} dollars on a book. How many dollars are left?",
            str(weekly * weeks - a - b), "int", ())


def _units(rng: random.Random, d: int) -> Gen:
    if d == 1:
        if rng.randrange(2):
            days = _between(rng, 2, 30)
            return f"How many minutes are there in {days} days?", str(days * 1440), "int", ()
        m, c = _between(rng, 2, 40), _between(rng, 1, 99)
        return f"How many centimetres are there in {m} metres and {c} centimetres?", str(m * 100 + c), "int", ()
    if d == 2:
        t = rng.randrange(3)
        if t == 0:
            h, m = _between(rng, 1, 9), _between(rng, 1, 59)
            return (f"How many seconds are there in {h} hours and {m} minutes?",
                    str((h * 60 + m) * 60), "int", ())
        if t == 1:
            k = _between(rng, 11, 999)
            return f"Convert {k / 10:.1f} kilometres to metres.", str(k * 100), "int", ()
        k = _between(rng, 101, 9999)
        return f"How many millilitres are there in {k / 100:.2f} litres?", str(k * 10), "int", ()
    if rng.randrange(2):
        c = 5 * _between(rng, -8, 20)
        return (f"Convert {c} degrees Celsius to degrees Fahrenheit.", str(c * 9 // 5 + 32), "int", ())
    g = _between(rng, 1001, 99999)
    kg = (Decimal(g) / Decimal(1000)).normalize()
    return (f"Convert {g} grams to kilograms.", format(kg, "f"), "decimal", (("tol", "0.0005"),))


def _strings(rng: random.Random, d: int) -> Gen:
    short = [w for w in _WORDS if len(w) <= 6]
    mid = [w for w in _WORDS if 7 <= len(w) <= 9]
    long_ = [w for w in _WORDS if len(w) >= 10]
    pool = (short, mid, long_)[d - 1]
    t = rng.randrange(5)
    w = _pick(rng, pool)
    if t == 0:
        return f"Write the word '{w}' backwards.", w[::-1], "text", ()
    if t == 1:
        letters = sorted(set(w))
        # harder levels ask about the letter that repeats most
        c = max(letters, key=lambda x: (w.count(x), x)) if d == 3 else _pick(rng, letters)
        return (f"How many times does the letter '{c}' appear in the word '{w}'?",
                str(w.count(c)), "int", ())
    if t == 2:
        i = rng.randrange(min(len(w), len(_ORDINALS)))
        return f"What is the {_ORDINALS[i]} letter of the word '{w}'?", w[i], "text", ()
    if t == 3:
        ws = _sample(rng, _WORDS, 2 + d)
        return (f"Sort these words into alphabetical order and list them separated by commas: "
                f"{', '.join(ws)}.", ", ".join(sorted(ws)), "list", ())
    ws = _sample(rng, _WORDS, 2 + d)
    return (f"Write the first letter of each of these words, in order, as one word in capital "
            f"letters: {' '.join(ws)}.", "".join(x[0] for x in ws).upper(), "text", ())


def _dates(rng: random.Random, d: int) -> Gen:
    day = _rand_date(rng)
    t = rng.randrange(3)
    if t == 0:
        return f"What day of the week was {_date_text(day)}?", _WEEKDAYS[day.weekday()], "text", ()
    span = (_between(rng, 2, 20), _between(rng, 21, 90), _between(rng, 91, 400))[d - 1]
    if t == 1:
        out = day + _dt.timedelta(days=span)
        return (f"What is the date {span} days after {_date_text(day)}? Write it as YYYY-MM-DD.",
                out.isoformat(), "date", ())
    later = day + _dt.timedelta(days=span)
    return f"How many days after {_date_text(day)} is {_date_text(later)}?", str(span), "int", ()


def _logic(rng: random.Random, d: int) -> Gen:
    t = rng.randrange(4)
    if t == 0:
        n = 2 + d
        order = _sample(rng, _NAMES, n)            # tallest first
        facts = [f"{order[i]} is taller than {order[i + 1]}." for i in range(n - 1)]
        facts = _sample(rng, facts, len(facts))
        q = rng.randrange(3 if d == 3 else 2)
        question, ans = (("Who is the tallest?", order[0]), ("Who is the shortest?", order[-1]),
                         ("Who is the second tallest?", order[1]))[q]
        return " ".join(facts) + " " + question, ans, "text", ()
    if t == 1:
        # six terms, five shown: constant step (d1), constant ratio (d2), steps growing by a
        # constant (d3)
        a = _between(rng, 2, 30)
        if d == 1:
            step = _between(rng, 2, 12)
            seq = [a + step * i for i in range(6)]
        elif d == 2:
            r = _between(rng, 2, 3)
            seq = [a * r ** i for i in range(6)]
        else:
            inc = _between(rng, 1, 4)
            seq = [a]
            for i in range(5):
                seq.append(seq[-1] + inc * (i + 1))
        return (f"What is the next number in the sequence {', '.join(map(str, seq[:5]))}?",
                str(seq[5]), "int", ())
    if t == 2:
        x, y, z, w = _sample(rng, _NONSENSE, 4)
        if d == 1:
            return (f"All {x} are {y}. All {y} are {z}. Does it follow that all {x} are {z}? Answer yes or no.",
                    "yes", "yesno", ())
        if d == 2:
            return (f"All {x} are {y}. Some {y} are {z}. Does it follow that all {x} are {z}? Answer yes or no.",
                    "no", "yesno", ())
        valid = rng.randrange(2) == 0
        third = f"All {z} are {w}." if valid else f"Some {z} are {w}."
        return (f"All {x} are {y}. All {y} are {z}. {third} Does it follow that all {x} are {w}? "
                f"Answer yes or no.", "yes" if valid else "no", "yesno", ())
    n = _between(rng, 6, 12) * d
    k = _between(rng, 2, n - 1)
    name = _pick(rng, _NAMES)
    return (f"In a row of {n} seats, {name} sits in seat {k} counting from the left. Which seat is "
            f"it counting from the right?", str(n - k + 1), "int", ())


def _constraints(rng: random.Random, d: int) -> Gen:
    topic = _pick(rng, _TOPICS)
    t = rng.randrange(8)
    if t == 0:
        n = (5, 8, 12)[d - 1] + rng.randrange(3)
        return (f"Write one sentence about {topic} that contains exactly {n} words.",
                " ".join(_FILLER[:n]).capitalize() + ".", "constraint", (("rule", "word_count"), ("n", n)))
    if t == 1:
        return (f"Write one sentence about {topic} using only lowercase letters.",
                "the quiet street was empty before the morning light.", "constraint",
                (("rule", "lowercase"),))
    if t == 2:
        n = (3, 5, 7)[d - 1] + rng.randrange(2)
        ex = ", ".join(w.upper() for w in _FILLER[1:1 + n])
        return (f"Write {n} single words related to {topic} in capital letters, separated by commas.",
                ex, "constraint", (("rule", "caps_list"), ("n", n)))
    if t == 3:
        letter = _pick(rng, sorted(_LETTER_WORDS))
        n = (3, 5, 7)[d - 1] + rng.randrange(2)
        return (f"Write {n} different words that all begin with the letter '{letter}', separated by spaces.",
                " ".join(_LETTER_WORDS[letter][:n]), "constraint",
                (("rule", "starts_with"), ("letter", letter), ("n", n)))
    if t == 4:
        letter = _pick(rng, ("zqxj", "kvwb", "aoe")[d - 1])
        n = (6, 8, 10)[d - 1] + rng.randrange(3)
        usable = [w for w in _FILLER if letter not in w]
        ex = " ".join((usable * n)[:n]).capitalize() + "."
        return (f"Write a sentence of at least {n} words about {topic} that does not use the letter '{letter}'.",
                ex, "constraint", (("rule", "no_letter"), ("letter", letter), ("n", n)))
    if t == 5:
        word = _pick(rng, _ENDINGS)
        return (f"Write a sentence about {topic} that ends with the word '{word}'.",
                f"We walked along the path {word}.", "constraint", (("rule", "ends_with"), ("word", word)))
    if t == 6:
        word = _pick(rng, _KEYWORDS)
        return (f"Write a sentence about {topic} that uses the word '{word}' exactly twice.",
                f"The {word} was near, and the {word} was clear.", "constraint",
                (("rule", "keyword_twice"), ("word", word)))
    n = (3, 5, 7)[d - 1] + rng.randrange(2)
    return (f"List exactly {n} things you might see at {topic}, one per line, each line starting "
            f"with its number and a full stop (1., 2., ...).",
            "\n".join(f"{i}. {w}" for i, w in enumerate(_FILLER[1:1 + n], start=1)), "constraint",
            (("rule", "numbered_list"), ("n", n)))


_GENERATORS: dict[str, Callable[[random.Random, int], Gen]] = {
    "arithmetic": _arithmetic, "word_problem": _word_problem, "units": _units,
    "strings": _strings, "dates": _dates, "logic": _logic, "constraints": _constraints,
}


def _question(body: str) -> str:
    sep = "\n\n" if "\n" in body else " "
    return f"{body}{sep}{markers.CAPABILITY_INSTRUCTION}"


def generate(n: int, seed: int, *, families: Sequence[str] = FAMILIES,
             difficulties: Sequence[int] = (1, 2, 3)) -> tuple[CapabilityItem, ...]:
    """`n` generated items, deterministic in (`seed`, n, families, difficulties).

    Families and difficulties are interleaved round-robin, so any prefix of the list is spread
    across both. Question texts are unique; a collision is redrawn with a fresh derived seed.
    """
    unknown = [f for f in families if f not in _GENERATORS]
    if unknown or not families:
        raise ValueError(f"unknown or empty families {unknown or families}; one of {FAMILIES}")
    if not difficulties or any(d not in (1, 2, 3) for d in difficulties):
        raise ValueError("difficulties are drawn from 1, 2, 3")
    out: list[CapabilityItem] = []
    seen: set[str] = set()
    for i in range(n):
        fam = families[i % len(families)]
        d = difficulties[(i // len(families)) % len(difficulties)]
        for attempt in range(100):
            rng = random.Random(derive_seed(seed, "s5", fam, d, i, attempt))
            body, answer, kind, params = _GENERATORS[fam](rng, d)
            q = _question(body)
            if q not in seen:
                break
        else:  # pragma: no cover - the families have far more than 100 distinct draws
            raise RuntimeError(f"could not draw a new {fam} item at difficulty {d}")
        seen.add(q)
        out.append(CapabilityItem(f"{fam}-d{d}-{i:04d}", fam, q, answer, kind, params, "generated", d))
    check_texts([it.question for it in out], what="capability item")
    return tuple(out)


# =====================================================================================
# Public benchmarks, read from files the operator keeps OUTSIDE the repository
# =====================================================================================

@dataclass(frozen=True)
class BenchmarkLoad:
    """What a loader read: the items, and the ids it skipped because their text names a
    harmful-arm concept (MMLU law and history questions say "assault" and "weapons"; such items
    are left out of the panel rather than allowed into a steered run)."""

    source: str
    path: str
    items: tuple[CapabilityItem, ...]
    skipped: tuple[str, ...]


def _read_jsonl(path: str | Path, what: str) -> list[tuple[int, dict]]:
    p = assert_outside_repo(path, what=what)
    rows = []
    with open(p, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{n}: not JSON ({exc.msg})") from None
            if not isinstance(obj, dict):
                raise ValueError(f"{p}:{n}: expected a JSON object per line")
            rows.append((n, obj))
    return rows


def _require(obj: dict, fields: Sequence[str], where: str) -> None:
    missing = [f for f in fields if f not in obj]
    if missing:
        raise ValueError(f"{where}: missing field(s) {missing}")


def _admit(item: CapabilityItem, items: list, skipped: list) -> None:
    try:
        check_texts([item.question], what=item.id)
    except Refused:
        skipped.append(item.id)
        return
    items.append(item)


def _choice_question(stem: str, options: Sequence[str]) -> str:
    lines = "\n".join(f"{CHOICE_LETTERS[i]}. {str(o).strip()}" for i, o in enumerate(options))
    return _question(f"{str(stem).strip()}\n\n{lines}\n\nAnswer with the letter of the correct option.")


def load_mmlu_pro(path: str | Path, *, limit: int | None = None,
                  categories: Sequence[str] | None = None) -> BenchmarkLoad:
    """MMLU-Pro (TIGER-Lab/MMLU-Pro) as JSONL, one question per line, with the dataset's fields:

        question_id   int or str     the item's id in the dataset
        question      str            the stem
        options       list[str]      2-10 options, lettered A-J in order
        answer        str            the correct letter, "A"-"J"
        answer_index  int, optional  the correct option's index; must agree with `answer`
        category      str, optional  used by `categories`
    """
    items: list[CapabilityItem] = []
    skipped: list[str] = []
    for n, row in _read_jsonl(path, "MMLU-Pro file"):
        where = f"{path}:{n}"
        _require(row, ("question_id", "question", "options", "answer"), where)
        opts = row["options"]
        if not isinstance(opts, list) or not 2 <= len(opts) <= len(CHOICE_LETTERS):
            raise ValueError(f"{where}: options must be a list of 2-10 strings")
        letter = str(row["answer"]).strip().upper()
        if letter not in CHOICE_LETTERS[: len(opts)]:
            raise ValueError(f"{where}: answer {row['answer']!r} is not one of the option letters")
        if "answer_index" in row and CHOICE_LETTERS[int(row["answer_index"])] != letter:
            raise ValueError(f"{where}: answer and answer_index disagree")
        if categories is not None and row.get("category") not in categories:
            continue
        _admit(CapabilityItem(f"mmlu_pro:{row['question_id']}", "mmlu_pro",
                              _choice_question(row["question"], opts), letter, "choice",
                              (("category", str(row.get("category", ""))),), "mmlu_pro"), items, skipped)
        if limit is not None and len(items) >= limit:
            break
    return BenchmarkLoad("mmlu_pro", str(path), tuple(items), tuple(skipped))


def load_gpqa_diamond(path: str | Path, *, seed: int, limit: int | None = None) -> BenchmarkLoad:
    """GPQA Diamond (Rein et al.) as JSONL, one row per question, with the column names of the
    released CSV:

        Question              str
        Correct Answer        str
        Incorrect Answer 1    str
        Incorrect Answer 2    str
        Incorrect Answer 3    str
        Record ID             str, optional (else the line number keys the item)

    The four options are shuffled per item with a seed derived from `seed` and the record id, so
    the correct letter is uniform over A-D and the same on every run. The dataset asks that its
    questions not be republished; this reads them from the operator's copy and stores none.
    """
    items: list[CapabilityItem] = []
    skipped: list[str] = []
    fields = ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
    for n, row in _read_jsonl(path, "GPQA file"):
        where = f"{path}:{n}"
        _require(row, fields, where)
        rid = str(row.get("Record ID", n))
        options = [str(row[f]).strip() for f in fields[1:]]
        rng = random.Random(derive_seed(seed, "gpqa", rid))
        order = _sample(rng, range(4), 4)
        shuffled = [options[j] for j in order]
        letter = CHOICE_LETTERS[order.index(0)]
        _admit(CapabilityItem(f"gpqa:{rid}", "gpqa_diamond", _choice_question(row["Question"], shuffled),
                              letter, "choice", (), "gpqa_diamond"), items, skipped)
        if limit is not None and len(items) >= limit:
            break
    return BenchmarkLoad("gpqa_diamond", str(path), tuple(items), tuple(skipped))


def load_gsm8k(path: str | Path, *, limit: int | None = None) -> BenchmarkLoad:
    """GSM8K (Cobbe et al.) as JSONL, the dataset's own format:

        question  str
        answer    str, the worked solution ending in "#### <number>"

    The number after "####" (thousands separators removed) is the answer.
    """
    items: list[CapabilityItem] = []
    skipped: list[str] = []
    for n, row in _read_jsonl(path, "GSM8K file"):
        where = f"{path}:{n}"
        _require(row, ("question", "answer"), where)
        m = re.search(r"####\s*([-\d,\.]+)\s*$", str(row["answer"]).strip())
        if not m:
            raise ValueError(f"{where}: answer does not end in '#### <number>'")
        value = Decimal(m.group(1).replace(",", ""))
        integral = value == value.to_integral_value()
        _admit(CapabilityItem(f"gsm8k:{n}", "gsm8k", _question(str(row["question"]).strip()),
                              str(int(value)) if integral else format(value.normalize(), "f"),
                              "int" if integral else "decimal", (), "gsm8k"), items, skipped)
        if limit is not None and len(items) >= limit:
            break
    return BenchmarkLoad("gsm8k", str(path), tuple(items), tuple(skipped))
