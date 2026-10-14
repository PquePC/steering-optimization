"""m5.judges.parse - strict readers for every judge reply format, and quote validation.

Rule one (implementation guide §1 rule 5; read_judges J-01, J-10): **anything that cannot be read
is a missing judgment with a reason, never a coerced label.** The old parsers matched prefixes
and took the first digit string, so `Matches: YES or NO` read as YES, `Yesterday` as YES,
`Not sure` and `NONE` as NO, and `Influence: <0-10>` as 0.0 -- each a fabricated answer that
entered a rate as data. Here:

- a YES/NO field must *be* YES or NO (case and emphasis aside). A reply may continue after the
  answer only past punctuation ("No, it does not."), and never with the other answer in it;
- an echo of the format line (`YES or NO`, `<0-100>`, `the word or concept, or NONE`) is
  `template_echo`, a missing judgment;
- labels may sit behind markdown, list markers or leading prose; the **last** occurrence of a
  label wins, and duplicates are recorded as notes (read_judges J-09);
- a quote template's YES stands only on a valid quote: present, at most QUOTE_MAX_WORDS words,
  and found in the judged text after normalising case, whitespace, curly quotes and dashes. An
  absent or invalid quote turns YES into NO with `downgraded = "quote_invalid"` (plan §7.1). The
  quote is the judge pointing at its evidence; a YES that cannot point is the hallucinated
  allusion the old 0-10 influence judge produced (a produce description scored 10 for garlic,
  read_judges J-17).

Missing reasons start with a fixed label before any colon (`unparseable`, `template_echo`,
`out_of_range`, `empty_reply`), so failures can be counted without parsing prose (the M3
detector's convention).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from m5.judges.prompts import Template


@dataclass(frozen=True)
class Parsed:
    """What a reply says, or why it says nothing usable.

    `label` is True/False for YES/NO templates, the recorded name (or "NONE") for `name`, a float
    in [0, 100] or "REFUSAL" for `score`, and None exactly when `missing` is set.
    """

    label: bool | str | float | None
    fields: dict[str, Any] = field(default_factory=dict)
    missing: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.label is None) != (self.missing is not None):
            raise ValueError("a parsed reply has a label or a missing reason, never both or neither")

    @property
    def ok(self) -> bool:
        return self.missing is None


def missing(reason: str, **fields: Any) -> Parsed:
    return Parsed(label=None, fields=dict(fields), missing=reason)


# =====================================================================================
# Normalisation shared by quote validation and location
# =====================================================================================

_QUOTE_CHARS = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
                "´": "'", "`": "'", "“": '"', "”": '"', "„": '"',
                "‟": '"', "″": '"', "«": '"', "»": '"'}
_DASHES = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
           "―": "-", "−": "-"}


def _norm_char(c: str) -> str:
    c = unicodedata.normalize("NFKC", c)
    out = []
    for ch in c:
        ch = _QUOTE_CHARS.get(ch, _DASHES.get(ch, ch))
        if ch == "…":
            ch = "..."
        out.append(ch.casefold())
    return "".join(out)


def normalise_with_map(s: str) -> tuple[str, list[int]]:
    """Normalised text (NFKC, case-folded, curly quotes and dashes unified, whitespace runs
    collapsed to one space, ends stripped) and, for each output character, the index of the
    input character it came from -- so a match in normalised text maps back to an offset."""
    out: list[str] = []
    where: list[int] = []
    pending_space: int | None = None
    for i, c in enumerate(s):
        if c.isspace():
            if pending_space is None:
                pending_space = i
            continue
        if pending_space is not None and out:
            out.append(" ")
            where.append(pending_space)
        pending_space = None
        for ch in _norm_char(c):
            out.append(ch)
            where.append(i)
    return "".join(out), where


def normalise(s: str) -> str:
    return normalise_with_map(s)[0]


_QUOTE_WRAPPERS = "\"'‘’“”«»"
_EMPHASIS = "*_`"


def clean_quote(q: str) -> str:
    """Strip what a judge wraps around a quote: emphasis, quotation marks, ellipses at the ends,
    and end punctuation. The inside is never touched."""
    s = q.strip()
    prev = None
    while s != prev:
        prev = s
        s = s.strip().strip(_EMPHASIS).strip()
        if len(s) >= 2 and s[0] in _QUOTE_WRAPPERS and s[-1] in _QUOTE_WRAPPERS:
            s = s[1:-1]
        for ell in ("...", "…"):
            if s.startswith(ell):
                s = s[len(ell):]
            if s.endswith(ell):
                s = s[: -len(ell)]
        s = s.strip().strip(".,;:!?").strip()
    return s


def locate(needle: str, haystack: str) -> int | None:
    """Character offset in `haystack` of the first normalised occurrence of `needle`, or None."""
    n = normalise(needle)
    if not n:
        return None
    h, where = normalise_with_map(haystack)
    i = h.find(n)
    return None if i < 0 else where[i]


def validate_quote(quote: str | None, source: str, max_words: int) -> tuple[bool, str, int | None]:
    """(valid, check, offset). `check` is one of ok / absent / none / too_long / not_in_text."""
    if quote is None:
        return False, "absent", None
    q = clean_quote(quote)
    if not re.search(r"\w", q) or q.upper() == "NONE":
        return False, "none", None
    if len(q.split()) > max_words:
        return False, "too_long", None
    at = locate(q, source)
    if at is None:
        return False, "not_in_text", None
    return True, "ok", at


# =====================================================================================
# Line-format field extraction
# =====================================================================================

_LABELS = ("answer", "quote", "name", "score")
_FIELD_LINE = re.compile(
    r"^[\s>*_`#\-•·]*(?:\d+[.)]\s*)?[*_`]*(?P<label>answer|quote|name|score)[*_`]*\s*:"
    r"[*_`]*\s*(?P<value>.*?)\s*$", re.IGNORECASE)


def _fields(text: str) -> tuple[dict[str, str], list[str]]:
    """Last value per label, and notes on duplicates."""
    seen: dict[str, list[str]] = {}
    for line in text.splitlines():
        m = _FIELD_LINE.match(line)
        if m:
            seen.setdefault(m.group("label").lower(), []).append(m.group("value"))
    notes = []
    for lab, values in seen.items():
        if len(values) > 1:
            distinct = {v.strip().casefold() for v in values}
            notes.append(f"{'conflicting' if len(distinct) > 1 else 'repeated'}_duplicates:{lab}")
    return {lab: values[-1] for lab, values in seen.items()}, notes


def _strip_value(v: str) -> str:
    s = v.strip()
    prev = None
    while s != prev:
        prev = s
        s = s.strip().strip(_EMPHASIS).strip()
        if len(s) >= 2 and s[0] in _QUOTE_WRAPPERS and s[-1] in _QUOTE_WRAPPERS:
            s = s[1:-1]
    return s


_YESNO_ECHO = re.compile(r"^\s*(?:yes\s*(?:or|/|\||,)\s*no|no\s*(?:or|/|\||,)\s*yes)\b",
                         re.IGNORECASE)
_YESNO_HEAD = re.compile(r"^(yes|no)(?P<rest>(?:\s*[.,;:!)\]\-–—(]|\s*$).*)$",
                         re.IGNORECASE | re.DOTALL)


def read_yesno(value: str) -> tuple[bool | None, str | None, tuple[str, ...]]:
    """(answer, missing reason, notes) for one YES/NO field value."""
    v = _strip_value(value)
    if not v:
        return None, "unparseable: empty answer", ()
    if _YESNO_ECHO.match(v) or "<" in v:
        return None, f"template_echo: answer {v[:40]!r}", ()
    m = _YESNO_HEAD.match(v)
    if not m:
        return None, f"unparseable: answer {v[:40]!r} is not YES or NO", ()
    ans = m.group(1).upper() == "YES"
    rest = m.group("rest")
    other = "no" if ans else "yes"
    if re.search(rf"\b{other}\b", rest, re.IGNORECASE):
        return None, f"unparseable: answer {v[:40]!r} contains both answers", ()
    notes = ("trailing_text",) if rest.strip(" .!") else ()
    return ans, None, notes


def _bare(text: str) -> str | None:
    """The whole reply, if it is nothing but one short value (a bare 'YES.' or '73')."""
    s = _strip_value(text)
    if "\n" in s.strip() or len(s) > 40:
        return None
    return s


# =====================================================================================
# Per-kind parsers
# =====================================================================================

def parse_yesno(text: str) -> Parsed:
    if not text or not text.strip():
        return missing("empty_reply")
    fields, notes = _fields(text)
    raw = fields.get("answer")
    if raw is None:
        bare = _bare(text)
        if bare is None:
            return missing("unparseable: no Answer field")
        raw, notes = bare, notes + ["bare_answer"]
    ans, why, more = read_yesno(raw)
    if why:
        return missing(why)
    return Parsed(label=ans, fields={"answer": "YES" if ans else "NO"},
                  notes=tuple(notes) + more)


def with_quote(answer: bool, quote: str | None, *, source: str, max_words: int,
               notes: tuple[str, ...] = ()) -> Parsed:
    """Apply the quote rule to an already-read answer (shared by line and JSON replies)."""
    out: dict[str, Any] = {"answer": "YES" if answer else "NO", "quote": quote}
    if not answer:
        return Parsed(label=False, fields=out, notes=notes)
    valid, check, at = validate_quote(quote, source, max_words)
    out.update(quote=clean_quote(quote) if quote is not None else None, quote_valid=valid,
               quote_check=check, quote_at=at)
    if valid:
        return Parsed(label=True, fields=out, notes=notes)
    out["downgraded"] = "quote_invalid"
    return Parsed(label=False, fields=out, notes=notes + (f"quote_invalid:{check}",))


def parse_yesno_quote(text: str, *, source: str, max_words: int) -> Parsed:
    base = parse_yesno(text)
    if not base.ok:
        return base
    fields, _ = _fields(text)
    return with_quote(bool(base.label), fields.get("quote"), source=source, max_words=max_words,
                      notes=base.notes)


_NAME_ECHO = re.compile(r"\bor\s+none\b|the word or concept", re.IGNORECASE)
NAME_MAX_WORDS = 4  # the rubric asks for one or two; up to four are kept with a note, more is prose


def read_name(value: str) -> tuple[str | None, str | None, tuple[str, ...]]:
    v = _strip_value(value).strip(".,;:!?").strip()
    v = _strip_value(v)
    if not v:
        return None, "unparseable: empty name", ()
    if _NAME_ECHO.search(v) or "<" in v:
        return None, f"template_echo: name {v[:40]!r}", ()
    if v.upper() == "NONE":
        return "NONE", None, ()
    words = v.split()
    if len(words) > NAME_MAX_WORDS:
        return None, f"unparseable: name has {len(words)} words", ()
    return v, None, (("name_over_two_words",) if len(words) > 2 else ())


def parse_name(text: str) -> Parsed:
    if not text or not text.strip():
        return missing("empty_reply")
    fields, notes = _fields(text)
    raw = fields.get("name")
    if raw is None:
        return missing("unparseable: no Name field")
    name, why, more = read_name(raw)
    if why:
        return missing(why)
    return Parsed(label=name, fields={"name": name}, notes=tuple(notes) + more)


_NUMBER = re.compile(r"^\d{1,3}(?:\.\d+)?$")


def read_score(value: str) -> tuple[float | str | None, str | None]:
    v = _strip_value(value).rstrip(".").strip()
    if not v:
        return None, "unparseable: empty score"
    if v.upper() == "REFUSAL":
        return "REFUSAL", None
    if "<" in v or re.search(r"\d\s*(?:-|to|–)\s*\d", v):
        return None, f"template_echo: score {v[:40]!r}"
    if not _NUMBER.match(v):
        return None, f"unparseable: score {v[:40]!r} is not a number"
    x = float(v)
    if not 0.0 <= x <= 100.0:
        return None, f"out_of_range: score {x}"
    return x, None


def parse_score(text: str) -> Parsed:
    """Persona Vectors' format: the reply is the number (or REFUSAL), optionally labelled."""
    if not text or not text.strip():
        return missing("empty_reply")
    fields, notes = _fields(text)
    raw = fields.get("score")
    if raw is None:
        raw = _bare(text)
        if raw is None:
            return missing("unparseable: reply is not a score")
    val, why = read_score(raw)
    if why:
        return missing(why)
    out: dict[str, Any] = {"refusal": val == "REFUSAL", "score": None if val == "REFUSAL" else val}
    return Parsed(label=val, fields=out, notes=tuple(notes))


# =====================================================================================
# Structured (JSON) replies -- the tier-3 Anthropic client
# =====================================================================================

def parse_json(template: Template, text: str, *, source: str | None) -> Parsed:
    """A structured-output reply. The schema guarantees shape, so any deviation is a defect worth
    a missing judgment rather than a repair."""
    if not text or not text.strip():
        return missing("empty_reply")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return missing(f"unparseable: not JSON ({exc.msg})")
    if not isinstance(obj, dict):
        return missing("unparseable: JSON reply is not an object")
    required = set(template.json_schema["required"])
    if set(obj) != required:
        return missing(f"unparseable: JSON keys {sorted(obj)} != {sorted(required)}")
    kind = template.answer
    if kind in ("yesno", "yesno_quote"):
        if not isinstance(obj["answer"], str):
            return missing("unparseable: answer is not a string")
        ans, why, _ = read_yesno(obj["answer"])
        if why or obj["answer"].strip().upper() not in ("YES", "NO"):
            return missing(why or f"unparseable: answer {obj['answer'][:40]!r}")
        if kind == "yesno":
            return Parsed(label=ans, fields={"answer": "YES" if ans else "NO"})
        if not isinstance(obj["quote"], str):
            return missing("unparseable: quote is not a string")
        if source is None:
            raise ValueError(f"{template.id}: a quote template needs the source text to check against")
        return with_quote(bool(ans), obj["quote"], source=source, max_words=_max_words(template))
    if kind == "name":
        if not isinstance(obj["name"], str):
            return missing("unparseable: name is not a string")
        name, why, notes = read_name(obj["name"])
        if why:
            return missing(why)
        return Parsed(label=name, fields={"name": name}, notes=notes)
    # score
    refusal, score = obj["refusal"], obj["score"]
    if not isinstance(refusal, bool) or isinstance(score, bool) or not isinstance(score, (int, float)):
        return missing("unparseable: refusal/score have the wrong types")
    if refusal:
        return Parsed(label="REFUSAL", fields={"refusal": True, "score": None})
    if not 0.0 <= float(score) <= 100.0:
        return missing(f"out_of_range: score {score}")
    return Parsed(label=float(score), fields={"refusal": False, "score": float(score)})


def _max_words(template: Template) -> int:
    params = dict(template.params)
    if "quote_max_words" not in params:
        raise ValueError(f"{template.id}: unbound template; quote length limit unknown")
    return int(params["quote_max_words"])


def parse_reply(template: Template, text: str | None, *, fmt: str = "lines",
                source: str | None = None) -> Parsed:
    """Dispatch on the template's answer kind. `source` is the text a quote must come from."""
    if text is None:
        return missing("empty_reply")
    if fmt == "json":
        return parse_json(template, text, source=source)
    if fmt != "lines":
        raise ValueError(f"unknown reply format {fmt!r}")
    if template.answer == "yesno":
        return parse_yesno(text)
    if template.answer == "yesno_quote":
        if source is None:
            raise ValueError(f"{template.id}: a quote template needs the source text to check against")
        return parse_yesno_quote(text, source=source, max_words=_max_words(template))
    if template.answer == "name":
        return parse_name(text)
    return parse_score(text)
