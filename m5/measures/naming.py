"""m5.measures.naming - is what the model named the injected concept? (decision 7, plan §7.2)

The `named` judge records what a reply names; code decides whether it matches
(`m5.judges.matching.identify`): exact, inflection and curated-synonym matches in code, the
`same_concept` judge only for the names code cannot decide, and the `related` judge for the looser
rule (b) where neither code nor `same_concept` settles it. Shared by `d8` (names and matches in a
self-report) and `d2` (forced naming).

A name whose judgment failed has no identification: missing, never a NO. A failed `related`
judgment leaves the primary rule decided and only the loose rule missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from m5.judges.matching import Identification, identify
from m5.judges.prompts import make_item, templates
from m5.measures import common as C


@dataclass(frozen=True)
class Ident:
    """One unit's identification: the matching result (None when undecided) and failure labels."""

    result: Identification | None
    missing: tuple[str, ...] = ()

    @property
    def matches(self) -> bool | None:
        return None if self.result is None else self.result.matches

    @property
    def loose(self) -> bool | None:
        return None if self.result is None else self.result.loose

    @property
    def how(self) -> str:
        return "undecided" if self.result is None else self.result.how


def resolve(ctx: Any, names: list[str | None], concept: str, *, item_ids: list[str],
            meta: dict[str, Any]) -> list[Ident]:
    """Identify each recorded name against `concept`. `names[i]` is the `named` judge's label
    ("NONE" for nothing named), or None when that judgment failed (then so does the unit's)."""
    T = templates(ctx.cfg)
    results: list[Identification | None] = [None if nm is None else identify(nm, concept) for nm in names]
    missing: list[list[str]] = [[] if nm is not None else ["named:failed"] for nm in names]
    for round_ in ("same_concept", "related"):
        need = [i for i, r in enumerate(results)
                if r is not None and r.needs and r.needs[0][0] == round_]
        if not need:
            continue
        items = [make_item(T[round_], results[i].needs[0][1], item_id=f"{item_ids[i]}:{round_}",
                           meta={**meta, "slot": round_})
                 for i in need]
        js = C.judge(ctx, items)
        for i, j in zip(need, js):
            if j.missing is not None:
                missing[i].append(C.reason(j))
                if round_ == "same_concept":
                    results[i] = None
                else:  # the primary rule stands; only the loose one is undecided
                    r = results[i]
                    results[i] = Identification(r.named, r.target, r.matches, None, r.semantic,
                                                r.how)
                continue
            if round_ == "same_concept":
                results[i] = identify(names[i], concept, same_concept=bool(j.label))
            else:
                results[i] = identify(names[i], concept, same_concept=bool(results[i].matches),
                                      related=bool(j.label))
    return [Ident(r, tuple(m)) for r, m in zip(results, missing)]
