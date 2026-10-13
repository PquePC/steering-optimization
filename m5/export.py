"""m5.export - what may leave the machine, decided by class, failing closed.

M3's export gate matched file *names*: anything it did not recognise was exported, so
`judge_calls.jsonl` and `read_this.md`, which carry generations, left the pod by default
(inventory H4; the fail-closed fix was written and reverted). Here the decision is made from the
export class every stream and document is registered with in `m5.runio`, and three rules hold:

1. **Unclassified means withheld.** A file that is not a registered stream or document -- a
   vector archive, a stray log, a subdirectory -- never goes into a bundle.
2. **Text needs two keys.** `transcript`-class files go only when the caller passes
   `allow_transcripts=True` at the call site -- never a config key, environment variable or
   module global, so the permission cannot be inherited by a later run (old invariant 142) --
   *and* every vector the run steered with is in a benign class (`m5.safety.BENIGN_CLASSES`).
   A harmful-arm concept withholds every transcript whatever the caller asked (defence in depth:
   export does not rely on the run having been refused earlier). `internal` files (judge calls,
   human labels) never go.
3. **Aggregates are checked, not trusted.** An `aggregate` stream is supposed to hold numbers. If
   any row carries a long string, the file is withheld as carrying text: that is how a transcript
   written to the wrong stream would otherwise leak.

Every bundle carries `EXPORT_MANIFEST.json` listing what was included and what was withheld, with
the reason, so a reader can see what is missing and why.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import os

from m5.runio import DOCUMENTS, STREAMS, RunDir, read_jsonl
from m5.safety import BENIGN_CLASSES, Refused, assert_outside_repo, concept_refused

# Longest string an aggregate row may carry (ids, codes, cell ids and short reasons fit easily).
MAX_AGGREGATE_STRING = 240


@dataclass
class ExportPlan:
    include: list[Path] = field(default_factory=list)
    withheld: list[tuple[Path, str]] = field(default_factory=list)

    def to_json(self, root: Path) -> dict:
        return {"included": [str(p.relative_to(root)) for p in self.include],
                "withheld": [{"file": str(p.relative_to(root)), "reason": r} for p, r in self.withheld]}


def _long_strings(obj: object, limit: int) -> bool:
    if isinstance(obj, str):
        return len(obj) > limit
    if isinstance(obj, dict):
        return any(_long_strings(v, limit) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_long_strings(v, limit) for v in obj)
    return False


def _aggregate_carries_text(path: Path) -> bool:
    rows = read_jsonl(path, schema=None).rows
    return any(_long_strings(r, MAX_AGGREGATE_STRING) for r in rows)


def plan_export(run: RunDir, *, allow_transcripts: bool, vector_classes: Iterable[str],
                concepts: Iterable[str] = ()) -> ExportPlan:
    """Classify every file in the run directory as included or withheld (with a reason)."""
    classes = set(vector_classes)
    harmful = sorted(c for c in concepts if concept_refused(c))
    benign = bool(classes) and classes <= BENIGN_CLASSES and not harmful
    plan = ExportPlan()
    by_name = {f"{name}.jsonl": cls for name, (cls, _) in STREAMS.items()}
    for path in sorted(run.path.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(run.path)
        name = rel.as_posix()
        if len(rel.parts) > 1:
            plan.withheld.append((path, "inside a subdirectory; unclassified (fail closed)"))
            continue
        if name in DOCUMENTS:
            cls = DOCUMENTS[name]
        elif name in by_name:
            cls = by_name[name]
        else:
            plan.withheld.append((path, "not a registered stream or document (fail closed)"))
            continue
        if cls == "internal":
            plan.withheld.append((path, "internal: judge calls and labels embed transcripts"))
        elif cls == "transcript":
            if not allow_transcripts:
                plan.withheld.append((path, "transcripts need ALLOW_TRANSCRIPT_EXPORT"))
            elif harmful:
                plan.withheld.append((path, f"harmful-arm concept(s) {harmful}: transcripts never leave"))
            elif not benign:
                plan.withheld.append((path, f"transcripts need every vector class benign; got "
                                            f"{sorted(classes) or 'none recorded'}"))
            else:
                plan.include.append(path)
        elif name.endswith(".jsonl") and _aggregate_carries_text(path):
            plan.withheld.append((path, f"aggregate stream carries a string longer than "
                                        f"{MAX_AGGREGATE_STRING} characters; treated as text"))
        else:
            plan.include.append(path)
    return plan


def write_export(run: RunDir, zip_path: str | Path, *, allow_transcripts: bool,
                 vector_classes: Iterable[str], concepts: Iterable[str] = ()) -> ExportPlan:
    """Write the bundle outside the repository and return the plan it followed.

    Written to `<zip>.part` and renamed, so a crash never leaves a bundle that looks complete;
    a bundle with nothing to include is refused rather than written empty.
    """
    out = assert_outside_repo(zip_path, what="export bundle")
    plan = plan_export(run, allow_transcripts=allow_transcripts, vector_classes=vector_classes,
                       concepts=concepts)
    if not plan.include:
        raise Refused(f"nothing in {run.path} may be exported; withheld: "
                      f"{[(p.name, r) for p, r in plan.withheld]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in plan.include:
            zf.write(p, arcname=p.relative_to(run.path).as_posix())
        zf.writestr("EXPORT_MANIFEST.json", json.dumps(plan.to_json(run.path), indent=1))
    os.replace(part, out)
    return plan
