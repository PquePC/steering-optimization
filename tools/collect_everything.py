"""Collect everything a pod holds that a later analysis could need, into one archive.

Run this on the pod when the runs are done, BEFORE stopping it. It gathers:

  * every run folder under M3_RUNS_DIR -- including aborted ones, which are evidence
  * the console logs, which live outside the run folders and are otherwise lost
  * the exact code state: commit, branch, dirty flag, and the full diff if dirty
  * the environment: python, pip freeze, nvidia-smi, and the resolved model revisions
  * a MANIFEST.md saying what each file is and how many rows it holds

The manifest's row counts are the point of the self-check: a truncated archive looks
identical to a complete one until someone opens it a week later, on a machine that no
longer has the pod.

Secrets are never written. Environment variables are captured by NAME ONLY.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUNS = Path(os.environ.get("M3_RUNS_DIR", "/workspace/m3_runs"))
OUT = RUNS / "_pod_snapshot"
SECRET = ("TOKEN", "KEY", "SECRET", "PASSWORD")


def sh(*cmd: str, cwd: str | None = None) -> str:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:                                  # noqa: BLE001
        return f"({type(exc).__name__}: {exc})"


def main() -> int:
    if not RUNS.is_dir():
        print(f"no runs directory at {RUNS}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent

    # ---- console logs, into the run folder they belong to -------------------------------
    moved = []
    for log in sorted(glob.glob("/workspace/*.out")) + sorted(glob.glob("/workspace/*.log")):
        stem = Path(log).stem.replace("q_", "")
        hits = [d for d in glob.glob(str(RUNS / f"{stem}_*/"))]
        for d in hits:
            shutil.copy(log, Path(d) / "console.log")
            moved.append(f"{Path(log).name} -> {Path(d).name}/console.log")
        # keep a copy centrally too, so a log with no matching folder is never lost
        shutil.copy(log, OUT / Path(log).name)

    # ---- code state ----------------------------------------------------------------------
    (OUT / "git_state.txt").write_text("\n".join([
        "commit:  " + sh("git", "rev-parse", "HEAD", cwd=str(repo)),
        "branch:  " + sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=str(repo)),
        "",
        "=== git log -5 ===", sh("git", "log", "--oneline", "-5", cwd=str(repo)),
        "", "=== git status ===", sh("git", "status", "--short", cwd=str(repo)),
    ]), encoding="utf-8")
    diff = sh("git", "diff", "HEAD", cwd=str(repo))
    if diff:
        (OUT / "git_uncommitted.diff").write_text(diff, encoding="utf-8")

    # ---- environment ---------------------------------------------------------------------
    (OUT / "environment.txt").write_text("\n".join([
        "python:  " + sys.version.replace("\n", " "),
        "", "=== nvidia-smi ===", sh("nvidia-smi"),
        "", "=== pip freeze ===", sh(sys.executable, "-m", "pip", "freeze"),
        "", "=== env var NAMES only (values withheld) ===",
        "\n".join(sorted(k + ("  [redacted]" if any(s in k.upper() for s in SECRET) else f"={v}")
                         for k, v in os.environ.items())),
    ]), encoding="utf-8")

    # ---- which weights, exactly ----------------------------------------------------------
    # provenance records the model KEY ("qwen3_32b"), not the revision it resolved to. If the
    # repo on Hugging Face moves, nothing else here would say which weights produced these
    # numbers.
    revs = []
    for snap in sorted(glob.glob(os.path.expanduser(
            os.environ.get("HF_HOME", "/workspace/hf") + "/hub/models--*/snapshots/*"))):
        revs.append(f"{Path(snap).parts[-3]}  revision {Path(snap).name}")
    (OUT / "model_revisions.txt").write_text("\n".join(revs) or "(no HF cache found)",
                                             encoding="utf-8")

    # ---- manifest with row counts ---------------------------------------------------------
    lines = ["# Pod snapshot", "",
             "Everything from this pod that a later analysis could need. Row counts are here so",
             "a truncated archive is detectable without the pod.", ""]
    if moved:
        lines += ["## Console logs placed", ""] + [f"- {m}" for m in moved] + [""]
    total = 0
    for d in sorted(glob.glob(str(RUNS / "*/"))):
        name = Path(d).name
        if name == OUT.name:
            continue
        lines += [f"## {name}", ""]
        summary = Path(d) / "summary.json"
        if summary.is_file():
            try:
                s = json.loads(summary.read_text(encoding="utf-8"))
                lines.append(f"- concept `{s.get('concept')}` · config `{s.get('config_hash')}` · "
                             f"{s.get('n_cells_on_disk')}/{s.get('n_cells_planned')} cells · "
                             f"{len(s.get('layers') or [])} layers")
            except Exception:                                 # noqa: BLE001
                lines.append("- summary.json present but unreadable")
        for f in sorted(Path(d).rglob("*")):
            if not f.is_file():
                continue
            size = f.stat().st_size
            total += size
            rows = ""
            if f.suffix == ".jsonl":
                with f.open(encoding="utf-8", errors="replace") as fh:
                    rows = f"  {sum(1 for line in fh if line.strip())} rows"
            lines.append(f"  - `{f.relative_to(d)}`  {size/1024:.0f} KB{rows}")
        lines.append("")
    lines += ["## Snapshot files", "",
              "- `git_state.txt` — commit, branch, status",
              "- `git_uncommitted.diff` — present only if the tree was dirty",
              "- `environment.txt` — python, nvidia-smi, pip freeze, env var names",
              "- `model_revisions.txt` — the Hugging Face revision each model resolved to",
              "- `*.out` — console logs, also copied into their run folders", ""]
    (OUT / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"snapshot written to {OUT}")
    print(f"run data total: {total/1e6:.1f} MB")
    print("\nnow archive everything:")
    print(f"  cd {RUNS} && tar czf /workspace/pod_everything.tgz .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
