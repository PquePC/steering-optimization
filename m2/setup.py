"""m2.setup - find out what this pod already has, and set up whatever it does not.

    python -m m2.setup            # report only, changes nothing
    python -m m2.setup --repair   # fix what can be fixed, then re-report

Written because migrating the pod is routine, not exceptional. A network volume survives a
migration and the container does not, so after every move the same four things are true and the
same four are false, and working out which by hand wastes a GPU-hour's worth of attention:

    survives   /workspace: the repo, the harness clone, the HF cache, every run folder
    does not   pip packages, environment variables, ~/.bashrc, the shell you exported into

The expensive failure is not noticing. `HF_HOME` unset in a fresh shell sends 54 GB to container
disk, where it evaporates on the next stop - that has already happened once here. A half-written
JSONL from a hard termination reads as a complete phase until something parses it.

Every check reports one of:

    OK       present and usable
    FIX      absent or wrong, and `--repair` can do something about it
    BLOCKED  absent or wrong, and only you can fix it (credentials, GPU, the volume itself)

Nothing here imports torch at module scope, so it runs on a pod whose dependencies are not
installed yet - which is exactly when it is most needed.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

WORKSPACE = Path(os.environ.get("M2_WORKSPACE", "/workspace"))
REPO_DIR = WORKSPACE / "steering-optimization"
# The repo root IS the project root, since the 2026-08-11 split out of Emergent-Introspection.
# There is no longer a nested "Steering Optimization" directory, and no space in any pod path.
PROJECT_DIR = REPO_DIR
MAIN_BRANCH = "main"
HARNESS_URL = "https://github.com/safety-research/introspection-mechanisms"
HARNESS_DIRS = (
    Path(os.environ["M2_HARNESS_DIR"]) if os.environ.get("M2_HARNESS_DIR") else None,
    WORKSPACE / "introspection-mechanisms",
    WORKSPACE / "steering-opt" / "introspection-mechanisms",
    PROJECT_DIR / "introspection-mechanisms",
)
DEFAULT_HF_HOME = WORKSPACE / "hf"

# Gemma3-27B in bf16 is ~54 GB of weights. Anything much under that is a partial download.
MODEL_GB_MIN = 45.0
VRAM_GB_MIN = 70.0
MIN_VOLUME_FREE_GB = 20.0
RUNPOD_VOLUME_API = "https://rest.runpod.io/v1/networkvolumes/{volume_id}"

REQUIRED_ENV = ("HF_TOKEN", "OPENROUTER_API_KEY")
OPTIONAL_ENV = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "HEALTHCHECK_URL", "RUNPOD_API_KEY")
PIP_EXTRA = ("nest_asyncio", "datasets", "pytest")

OK, FIX, BLOCKED = "OK", "FIX", "BLOCKED"


@dataclass
class Check:
    name: str
    state: str
    detail: str = ""
    repair: Callable[[], str] | None = None
    hint: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *args: Any, **kw: Any) -> Check:
        c = Check(*args, **kw)
        self.checks.append(c)
        return c

    @property
    def blocked(self) -> list[Check]:
        return [c for c in self.checks if c.state == BLOCKED]

    @property
    def fixable(self) -> list[Check]:
        return [c for c in self.checks if c.state == FIX]

    def ready(self) -> bool:
        return not self.blocked and not self.fixable


def _sh(cmd: Sequence[str], cwd: Path | None = None, timeout: int = 900) -> tuple[int, str]:
    try:
        p = subprocess.run([str(c) for c in cmd], cwd=None if cwd is None else str(cwd),
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:                        # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}"


def _dir_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total / 1024 ** 3


def _tree_allocated_gb(path: Path) -> float:
    """Allocated bytes below `path`, without double-counting links to model blobs.

    RunPod network volumes expose the backing storage pool through `statvfs`, so
    `shutil.disk_usage` can report hundreds of petabytes free on a 150 GB allocation. The
    allocation API supplies the ceiling; this walk supplies the used side. `st_blocks` measures
    real allocated storage (including copied HuggingFace snapshots), while `lstat` and inode
    deduplication keep symlinks and hard links from counting the same bytes twice.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                info = os.lstat(Path(root) / name)
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            identity = (int(info.st_dev), int(info.st_ino))
            if identity in seen:
                continue
            seen.add(identity)
            blocks = getattr(info, "st_blocks", None)
            total += (int(blocks) * 512
                      if blocks is not None and int(blocks) > 0 else int(info.st_size))
    return total / 1024 ** 3


def _runpod_volume_size_gb(volume_id: str, api_key: str, *, timeout: float = 10.0) -> float:
    """Read the network-volume allocation, not the distributed pool behind its mount.

    RunPod provides `RUNPOD_VOLUME_ID` and a pod-scoped `RUNPOD_API_KEY`. Its documented
    network-volume endpoint returns `size` in GB. No filesystem syscall can recover this number
    on mounts whose `statvfs` describes the shared backing pool, which is exactly why the old
    under-20-GB guard could never fail.
    """
    request = urllib.request.Request(
        RUNPOD_VOLUME_API.format(volume_id=volume_id),
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    size = payload["size"]
    if isinstance(size, bool) or not isinstance(size, (int, float)) or float(size) <= 0:
        raise ValueError("RunPod network-volume response has no positive numeric size")
    return float(size)


# Probe by IMPORT, and treat `__version__` as optional. `nest_asyncio` is a single-module
# package that never defined `__version__`, so the old `print(pkg.__version__)` probe raised
# AttributeError on a perfectly good install and reported it missing forever: --repair would
# pip-install it, say "extras ok", then the re-check would call it missing again.
_PROBE = (
    "import importlib, sys\n"
    "name = sys.argv[1]\n"
    "try:\n"
    "    mod = importlib.import_module(name)\n"
    "except Exception:\n"
    "    sys.exit(1)\n"
    "v = getattr(mod, '__version__', None)\n"
    "if not v:\n"
    "    try:\n"
    "        from importlib.metadata import version\n"
    "        v = version(name.replace('_', '-'))\n"
    "    except Exception:\n"
    "        v = 'installed'\n"
    "print(v)\n"
)


def _version(pkg: str) -> str | None:
    """Version string if importable, else None. 'installed' when it carries no version."""
    code, out = _sh([sys.executable, "-c", _PROBE, pkg], timeout=120)
    return out.splitlines()[-1] if code == 0 and out else None


# =====================================================================================
# checks
# =====================================================================================

def check_volume(rep: Report) -> None:
    if not WORKSPACE.is_dir():
        rep.add("persistent volume", BLOCKED, f"{WORKSPACE} does not exist",
                hint="Deploy the pod with a network volume mounted at /workspace. Without one "
                     "every check below is moot: the 54 GB model and all run data would live on "
                     "container disk and vanish on the next stop.")
        return
    volume_id = os.environ.get("RUNPOD_VOLUME_ID")
    if volume_id:
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            rep.add(
                "persistent volume", BLOCKED,
                f"{WORKSPACE}, allocation unknown (RUNPOD_API_KEY is missing)",
                hint=("RUNPOD_VOLUME_ID is set, so filesystem free space describes the shared "
                      "backing pool rather than this volume's allocation. Restore RunPod's "
                      "pod-scoped RUNPOD_API_KEY; do not trust df/shutil.disk_usage here."))
            return
        try:
            allocated_gb = _runpod_volume_size_gb(volume_id, api_key)
            used_gb = _tree_allocated_gb(WORKSPACE)
        except Exception as exc:                    # noqa: BLE001 - a check reports, never aborts
            rep.add(
                "persistent volume", BLOCKED,
                f"{WORKSPACE}, allocation check failed ({type(exc).__name__})",
                hint=("Could not read the RunPod allocation. Retry setup with "
                      "RUNPOD_VOLUME_ID and the pod-scoped RUNPOD_API_KEY present; the "
                      "filesystem's backing-pool free count is not a safe fallback."))
            return
        free_gb = max(0.0, allocated_gb - used_gb)
        detail = (f"{WORKSPACE}, {free_gb:.0f} GB free "
                  f"({used_gb:.0f}/{allocated_gb:.0f} GB allocation used)")
    else:
        # Local disks expose their actual filesystem capacity. The special API path above is
        # only for RunPod network volumes, whose mount reports the distributed backing pool.
        free_gb = shutil.disk_usage(WORKSPACE).free / 1024 ** 3
        detail = f"{WORKSPACE}, {free_gb:.0f} GB free (filesystem)"
    state = OK if free_gb > MIN_VOLUME_FREE_GB else BLOCKED
    rep.add("persistent volume", state, detail,
            hint="" if state == OK else
            f"Under {MIN_VOLUME_FREE_GB:g} GB free. The model alone needs ~54 GB.")


def check_hf_home(rep: Report) -> None:
    current = os.environ.get("HF_HOME")
    on_volume = bool(current) and str(current).startswith(str(WORKSPACE))

    if on_volume:
        rep.add("HF_HOME", OK, current)
    else:
        def _fix() -> str:
            os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
            DEFAULT_HF_HOME.mkdir(parents=True, exist_ok=True)
            return f"set HF_HOME={DEFAULT_HF_HOME} for THIS process"

        rep.add("HF_HOME", FIX,
                f"{current or 'unset'} - not under {WORKSPACE}", repair=_fix,
                hint=f"export HF_HOME={DEFAULT_HF_HOME}\n"
                     "      Put this in every shell. ~/.bashrc is on container disk and does "
                     "not survive a migration, which is how 54 GB ends up downloaded twice.")

    # Is the model actually there, wherever HF_HOME points?
    root = Path(os.environ.get("HF_HOME") or DEFAULT_HF_HOME)
    hits = list(root.glob("**/models--google--gemma-3-27b-it")) if root.exists() else []
    stray = [p for p in (WORKSPACE / ".cache", Path.home() / ".cache")
             if p.exists() and list(p.glob("**/models--google--gemma-3-27b-it"))]
    if hits:
        gb = _dir_gb(hits[0])
        state = OK if gb >= MODEL_GB_MIN else FIX
        rep.add("model cache", state, f"{gb:.0f} GB at {hits[0].parent}",
                hint="" if state == OK else
                     "Partial download. Re-run the preflight; it resumes.")
    elif stray:
        rep.add("model cache", FIX, f"found OUTSIDE HF_HOME at {stray[0]}",
                hint=f"The model downloaded before HF_HOME was set. Either move it:\n"
                     f"        mkdir -p {DEFAULT_HF_HOME} && mv {stray[0]}/huggingface/* "
                     f"{DEFAULT_HF_HOME}/\n"
                     "      or accept a re-download on the next pod.")
    else:
        # OK, not FIX. An absent cache on a fresh volume is the expected state, and the
        # preflight resolves it by design - `--repair` has nothing to do here and would report
        # a permanently unfixable item, which makes READY unreachable and trains you to ignore
        # the summary line. A PARTIAL or STRANDED cache is a real defect and is FIX above.
        rep.add("model cache", OK,
                "absent - the preflight downloads ~54 GB (15-25 min), which is expected on a "
                "fresh volume")


def check_repo(rep: Report) -> None:
    if not (REPO_DIR / ".git").is_dir():
        rep.add("project repo", BLOCKED, f"{REPO_DIR} is not a git clone",
                hint="git clone https://x-access-token:$GH@github.com/PquePC/"
                     "steering-optimization.git " + str(REPO_DIR))
        return
    _, branch = _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_DIR)
    _, head = _sh(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    code, _ = _sh(["git", "fetch", "--quiet"], cwd=REPO_DIR, timeout=120)
    behind = "0"
    if code == 0:
        _, behind = _sh(["git", "rev-list", "--count", "HEAD..@{u}"], cwd=REPO_DIR)
    behind_n = int(behind) if behind.isdigit() else 0

    # `main` since the 2026-08-11 split: this repo's whole history IS the M2 work, so there is
    # no longer an M2 branch to be on and checking for one would report a healthy clone broken.
    if branch != MAIN_BRANCH:
        rep.add("project repo", FIX, f"on branch {branch!r}, expected {MAIN_BRANCH!r}",
                repair=lambda: _sh(["git", "checkout", MAIN_BRANCH], cwd=REPO_DIR)[1]
                or f"checked out {MAIN_BRANCH}",
                hint=f"git -C {REPO_DIR} checkout {MAIN_BRANCH}")
    elif behind_n:
        rep.add("project repo", FIX,
                f"{MAIN_BRANCH}, {behind_n} commit(s) behind origin | {head}",
                repair=lambda: _sh(["git", "pull", "--ff-only"], cwd=REPO_DIR)[1] or "pulled",
                hint="git -C %s pull" % REPO_DIR)
    else:
        rep.add("project repo", OK, f"{MAIN_BRANCH}, up to date | {head}")


def check_harness(rep: Report) -> None:
    found = next((d for d in HARNESS_DIRS
                  if d is not None and (d / "src" / "model_utils.py").is_file()), None)
    if found:
        rep.add("upstream harness", OK, str(found))
        return
    target = WORKSPACE / "introspection-mechanisms"

    def _fix() -> str:
        if target.exists() and not (target / ".git").is_dir():
            shutil.rmtree(target, ignore_errors=True)
        code, out = _sh(["git", "clone", "--depth", "1", HARNESS_URL, str(target)])
        return "cloned" if code == 0 else f"clone failed: {out[:120]}"

    rep.add("upstream harness", FIX, "not found on any searched path", repair=_fix,
            hint=f"git clone --depth 1 {HARNESS_URL} {target}")


def check_deps(rep: Report) -> None:
    need = {"torch": None, "transformers": None, "datasets": None,
            "pytest": None, "nest_asyncio": None, "numpy": None}
    for pkg in list(need):
        need[pkg] = _version(pkg)
    missing = [p for p, v in need.items() if v is None]

    if not missing:
        rep.add("python packages", OK,
                ", ".join(f"{p} {v}" for p, v in need.items() if p in ("torch", "transformers")))
        return

    harness = next((d for d in HARNESS_DIRS
                    if d is not None and (d / "requirements.txt").is_file()), None)

    def _fix() -> str:
        steps = []
        if harness is not None:
            code, _ = _sh([sys.executable, "-m", "pip", "install", "-q", "-r",
                           str(harness / "requirements.txt")], timeout=1800)
            steps.append(f"requirements.txt {'ok' if code == 0 else 'FAILED'}")
        code, _ = _sh([sys.executable, "-m", "pip", "install", "-q", *PIP_EXTRA], timeout=900)
        steps.append(f"extras {'ok' if code == 0 else 'FAILED'}")
        return "; ".join(steps)

    rep.add("python packages", FIX, f"missing: {', '.join(missing)}", repair=_fix,
            hint="pip lives on CONTAINER disk, so it is gone after every migration. "
                 "This is normal, not damage.")


def check_gpu(rep: Report) -> None:
    code, out = _sh([sys.executable, "-c",
                     "import torch;"
                     "print(torch.cuda.is_available());"
                     "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '');"
                     "print(torch.cuda.get_device_properties(0).total_memory/1024**3 "
                     "if torch.cuda.is_available() else 0)"], timeout=300)
    if code != 0:
        rep.add("GPU", FIX, "torch not importable yet", hint="install packages first")
        return
    lines = out.splitlines()
    if len(lines) < 3 or lines[0].strip() != "True":
        rep.add("GPU", BLOCKED, "no CUDA device visible",
                hint="Redeploy on a GPU pod. Gemma3-27B in bf16 needs ~54 GB resident.")
        return
    name, gb = lines[1].strip(), float(lines[2])
    state = OK if gb >= VRAM_GB_MIN else BLOCKED
    rep.add("GPU", state, f"{name}, {gb:.0f} GB",
            hint="" if state == OK else
                 f"Under {VRAM_GB_MIN:.0f} GB. The model alone is ~54 GB; do not split it "
                 "across two cards - the injection hook assumes one device.")


def check_credentials(rep: Report) -> None:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        rep.add("credentials (required)", BLOCKED, f"missing: {', '.join(missing)}",
                hint="Environment variables only - under nohup there is no TTY to prompt on.\n"
                     "       export HF_TOKEN=...  OPENROUTER_API_KEY=...   (leading space)")
    else:
        rep.add("credentials (required)", OK, ", ".join(REQUIRED_ENV))

    absent = [k for k in OPTIONAL_ENV if not os.environ.get(k)]
    if absent:
        note = ("no dead man's switch - a pod that dies outright will not report it"
                if "HEALTHCHECK_URL" in absent else "reduced alerting / no auto-stop")
        rep.add("credentials (optional)", OK, f"unset: {', '.join(absent)} ({note})")
    else:
        rep.add("credentials (optional)", OK, "all set")


def _jsonl_rows(path: Path) -> tuple[int, bool]:
    """(complete rows, saw a truncated final line). A hard pod kill can leave a partial line."""
    if not path.is_file():
        return 0, False
    n, truncated = 0, False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                n += 1
            except json.JSONDecodeError:
                truncated = True
    return n, truncated


def check_run_data(rep: Report) -> None:
    """What is already measured, per concept. The whole point after a migration."""
    runs = Path(os.environ.get("M2_RUNS_DIR") or (WORKSPACE / "m2_runs"))
    if not runs.is_dir():
        rep.add("run data", OK, "no previous runs - starting clean")
        return

    folders = sorted(p for p in runs.iterdir() if p.is_dir())
    archives = sorted(p.name for p in runs.glob("*.zip"))
    if not folders and not archives:
        rep.add("run data", OK, f"{runs} is empty - starting clean")
        return

    lines, damaged = [], []
    for folder in folders:
        parts = []
        for label, name in (("scan", "scan.jsonl"), ("bisect", "bisect.jsonl"),
                            ("verified", "verified.jsonl"), ("confirm", "confirm.jsonl")):
            n, trunc = _jsonl_rows(folder / name)
            if trunc:
                damaged.append(f"{folder.name}/{name}")
            if n:
                parts.append(f"{label} {n}")
        vecs = folder / "vectors"
        if vecs.is_dir() and any(vecs.iterdir()):
            parts.insert(0, "vectors")
        if (folder / "operating_point.json").is_file():
            parts.append("DONE")
        lines.append(f"{folder.name}: {', '.join(parts) if parts else 'empty'}")

    detail = "; ".join(lines) + (f" | archives: {len(archives)}" if archives else "")
    if damaged:
        def _fix() -> str:
            for rel in damaged:
                p = runs / rel
                text = p.read_text(encoding="utf-8").splitlines()
                good = [ln for ln in text if ln.strip() and _is_json(ln)]
                p.write_text("\n".join(good) + "\n", encoding="utf-8")
            return f"trimmed {len(damaged)} truncated file(s)"

        rep.add("run data", FIX, detail + f" | TRUNCATED: {', '.join(damaged)}", repair=_fix,
                hint="A hard pod kill can leave a half-written final line. It would parse as a "
                     "missing row and silently re-measure, or crash the reader.")
    else:
        rep.add("run data", OK, detail)


def _is_json(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def check_tests(rep: Report) -> None:
    if not (PROJECT_DIR / "m2" / "tests" / "test_offline.py").is_file():
        rep.add("offline tests", FIX, "not found (repo missing or wrong branch)")
        return
    if _version("pytest") is None:
        rep.add("offline tests", FIX, "pytest not installed yet")
        return
    code, out = _sh([sys.executable, "-m", "pytest", "m2/tests/test_offline.py", "-q"],
                    cwd=PROJECT_DIR, timeout=300)
    last = out.splitlines()[-1] if out else ""
    rep.add("offline tests", OK if code == 0 else BLOCKED, last,
            hint="" if code == 0 else "The package is not intact. Do not spend GPU time.")


CHECKS: tuple[Callable[[Report], None], ...] = (
    check_volume, check_hf_home, check_repo, check_harness, check_deps,
    check_gpu, check_credentials, check_run_data, check_tests,
)


# =====================================================================================
# driving
# =====================================================================================

def diagnose() -> Report:
    rep = Report()
    for fn in CHECKS:
        try:
            fn(rep)
        except Exception as exc:                    # noqa: BLE001 - a check must never abort
            rep.add(fn.__name__, BLOCKED, f"check raised {type(exc).__name__}: {exc}")
    return rep


def render(rep: Report) -> None:
    mark = {OK: "  ok  ", FIX: " FIX  ", BLOCKED: "BLOCK "}
    print("=" * 78)
    print("M2 SETUP")
    print("=" * 78)
    for c in rep.checks:
        print(f"[{mark[c.state]}] {c.name:<24} {c.detail}")
        if c.state != OK and c.hint:
            for line in c.hint.splitlines():
                print(f"          {line}")
    print("=" * 78)
    if rep.ready():
        print("READY. Next:")
        print('  cd "%s"' % PROJECT_DIR)
        print("  python -m m2.run --concepts Garlic --preflight")
    elif rep.blocked:
        print(f"BLOCKED on {len(rep.blocked)}: " +
              ", ".join(c.name for c in rep.blocked) + " - only you can fix these.")
        if rep.fixable:
            print(f"{len(rep.fixable)} other item(s) fixable with --repair.")
    else:
        print(f"{len(rep.fixable)} item(s) fixable. Re-run with --repair.")
    print("=" * 78)


def repair(rep: Report) -> Report:
    todo = [c for c in rep.fixable if c.repair is not None]
    if not todo:
        print("nothing to repair automatically")
        return rep
    print("=" * 78)
    print(f"REPAIRING {len(todo)} item(s)")
    print("=" * 78)
    for c in todo:
        print(f"--- {c.name}: {c.detail}")
        try:
            print(f"    {c.repair()}")
        except Exception as exc:                    # noqa: BLE001
            print(f"    FAILED: {type(exc).__name__}: {exc}")
    print("\nre-checking\n")
    return diagnose()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m m2.setup",
        description="Report what this pod already has, and optionally set up what it does not.")
    ap.add_argument("--repair", action="store_true",
                    help="clone the harness, install packages, pull the repo, set HF_HOME for "
                         "this process, trim truncated JSONL. Never deletes measured rows.")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    args = ap.parse_args(argv)

    rep = diagnose()
    if args.repair:
        rep = repair(rep)
    if args.json:
        print(json.dumps([dict(name=c.name, state=c.state, detail=c.detail)
                          for c in rep.checks], indent=2))
    else:
        render(rep)
    return 0 if rep.ready() else (2 if rep.blocked else 1)


if __name__ == "__main__":
    sys.exit(main())
