"""tools.repair_jsonl - show, and optionally quarantine, unreadable lines in a run's artefacts.

    python -m tools.repair_jsonl /workspace/m3_runs/garlic_4f97c68d7b08          # report only
    python -m tools.repair_jsonl /workspace/m3_runs/garlic_4f97c68d7b08 --apply  # and repair

**Reporting is the default and repairing needs a flag, deliberately.** `runio.read_rows` refuses
to resume from a file with a bad line anywhere but the end, and that refusal is load-bearing: a
malformed line in the middle is not a torn write, it is corruption or two writers, and silently
editing it away destroys the evidence for whichever of those it was. `runio.heal_torn_tail`
handles the one shape that is safe to fix automatically -- an unreadable FINAL row, left by a
process killed mid-append -- and stops there on purpose. This tool is the operator's explicit
"yes, I have looked at the bytes, take it out".

## What it is for

A run crashed mid-append, was resumed under a version without `heal_torn_tail`, and the resumed
run's rows landed after the torn one. The bad line is now buried, every read raises, and the
directory holds thousands of good rows nobody can get at. That is not hypothetical: on
2026-08-24 a nine-cell Qwen sweep measured and wrote every cell and then reported FAILED,
because line 1 of its `cells.jsonl` would not parse.

## What it prints

For every bad line: the line number, its length, and the first 64 bytes as `repr`. Read those
bytes before passing `--apply` -- they say what happened, and nothing else does:

  * leading `\\x00` bytes      a filesystem that recorded a size and lost the data. Check `df`.
  * a fragment starting `{`   a torn write, the ordinary crash shape.
  * something starting `}`,   a partially overwritten file.
    `,` or `"`
  * readable prose            something outside this pipeline wrote to the path.

`write_row` emits `json.dumps(...)`, which always begins with `{`. Anything else did not come
from this code, and that is worth knowing before you delete it.

Quarantined lines go to `<name>.quarantine`, never to /dev/null.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def bad_lines(path: Path) -> list[tuple[int, bytes]]:
    """Every line that is not a JSON object, as (1-based line number, raw bytes).

    Splits on b"\\n" only, matching `runio.write_row`'s single record separator. `splitlines()`
    would also break on U+2028, U+2029 and U+0085, which appear inside model generations and are
    not record separators -- reporting one of those as corruption would send an operator
    hunting for a bug that is not there.
    """
    data = path.read_bytes()
    if not data:
        return []
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    out: list[tuple[int, bytes]] = []
    for index, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            if isinstance(json.loads(raw.decode("utf-8")), dict):
                continue
        except (ValueError, UnicodeDecodeError):
            pass
        out.append((index, raw))
    return out


def repair(path: Path, apply: bool) -> tuple[int, int]:
    """Report, and with `apply` remove, the unreadable lines. Returns (bad, good)."""
    bad = bad_lines(path)
    data = path.read_bytes()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    good = len([raw for raw in lines if raw.strip()]) - len(bad)

    if not bad:
        print(f"  {path.name:<34} {good:>7} rows, clean")
        return 0, good

    print(f"  {path.name:<34} {good:>7} rows, {len(bad)} UNREADABLE")
    for number, raw in bad:
        print(f"      line {number:<7} {len(raw):>7} bytes  {raw[:64]!r}")
    if not apply:
        return len(bad), good

    quarantine = path.with_suffix(path.suffix + ".quarantine")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(quarantine, "ab") as handle:
        for number, raw in bad:
            handle.write(f"# {path.name} line {number}, removed {stamp}\n".encode("utf-8"))
            handle.write(raw + b"\n")
    keep = [raw for raw in lines if raw.strip() and raw not in {r for _, r in bad}]
    path.write_bytes(b"\n".join(keep) + b"\n")
    print(f"      -> {len(bad)} lines moved to {quarantine.name}; {len(keep)} rows kept")
    return len(bad), good


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.repair_jsonl",
        description="Show, and with --apply quarantine, unreadable lines in a run's artefacts.")
    parser.add_argument("target", type=Path,
                        help="a run directory, or a single .jsonl file")
    parser.add_argument("--apply", action="store_true",
                        help="actually remove the bad lines. Read the printed bytes first.")
    args = parser.parse_args(argv)

    target = Path(args.target)
    if target.is_dir():
        files = sorted(target.glob("*.jsonl"))
    elif target.exists():
        files = [target]
    else:
        raise SystemExit(f"{target} does not exist")
    if not files:
        raise SystemExit(f"no .jsonl files in {target}")

    print(f"{target}\n")
    total_bad = 0
    for path in files:
        bad, _ = repair(path, args.apply)
        total_bad += bad

    print()
    if not total_bad:
        print("  Every artefact reads clean. If a run still refuses to resume, the problem is")
        print("  not the files -- check the config hash names the directory you think it does.")
    elif args.apply:
        print(f"  {total_bad} lines quarantined. The run can be resumed, and the removed bytes")
        print("  are in the .quarantine files next to each artefact.")
    else:
        print(f"  {total_bad} unreadable lines. Nothing was changed.")
        print("  Read the bytes above before repairing: leading \\x00 means the volume lost data")
        print("  and you should check `df -h` first, because repairing will not fix a full disk.")
        print("  Re-run with --apply to quarantine them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
