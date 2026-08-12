#!/usr/bin/env bash
# =====================================================================================
# M2 - one-shot pod setup. Idempotent: safe to re-run after a pod restart.
#
#   bash "/workspace/steering-optimization/m2/setup_pod.sh"
#
# Clones the upstream harness, installs dependencies, and runs the offline tests. Does not
# touch credentials and does not load the model - see QUICKSTART.md steps 4 and 5.
#
# PREFER `python -m m2.setup --repair`. It does everything this does AND diagnoses what is
# already present - the model cache, the run data, the GPU, the credentials - which is what
# you actually want after a pod migration. This script stays as the fallback for a pod so
# bare that python is not usable yet.
# =====================================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/steering-optimization}"
HARNESS_DIR="${HARNESS_DIR:-/workspace/introspection-mechanisms}"
# The upstream harness. Taken from the v1 notebook's Setup 2 and from the working clone's
# own remote - NOT guessed from the author's username, which is a different account and
# gives a GitHub auth prompt rather than an honest 404.
HARNESS_URL="https://github.com/safety-research/introspection-mechanisms"
PROJECT_DIR="$REPO_DIR"   # the repo root IS the project root since the split

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "M2 pod setup"

# ---- 0. sanity ----------------------------------------------------------------------
[ -d /workspace ] || die "/workspace does not exist. This script is for a RunPod pod with a
   persistent volume mounted at /workspace. Without one, everything below is lost when the
   pod stops - including the 54 GB model download."

[ -d "$REPO_DIR" ] || die "$REPO_DIR not found. Clone this repository first:

  GH=YOUR_GITHUB_TOKEN
  git clone https://x-access-token:\$GH@github.com/PquePC/steering-optimization.git $REPO_DIR
"

[ -d "$PROJECT_DIR/m2" ] || die "$PROJECT_DIR/m2 not found. Is the clone complete?
"

# ---- 1. HF cache on the volume ------------------------------------------------------
# The single most expensive mistake available here: the default cache is on container disk,
# which does NOT survive a pod stop, so the 54 GB model re-downloads every restart.
if [ -z "${HF_HOME:-}" ]; then
  export HF_HOME=/workspace/hf
  say "HF_HOME was unset - defaulting to /workspace/hf (on the volume)"
fi
mkdir -p "$HF_HOME"
case "$HF_HOME" in
  /workspace/*) : ;;
  *) printf '\n\033[33mWARNING: HF_HOME=%s is not under /workspace. The 54 GB model download
   will not survive a pod stop. Set HF_HOME=/workspace/hf.\033[0m\n' "$HF_HOME" ;;
esac

# Persist it so a new shell after a restart inherits it.
if ! grep -qs 'HF_HOME' ~/.bashrc 2>/dev/null; then
  echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
fi

# ---- 2. upstream harness ------------------------------------------------------------
say "Upstream harness"
if [ -d "$HARNESS_DIR/.git" ]; then
  echo "already cloned at $HARNESS_DIR"
else
  git clone --depth 1 "$HARNESS_URL" "$HARNESS_DIR"
fi
[ -f "$HARNESS_DIR/requirements.txt" ] || die "$HARNESS_DIR/requirements.txt missing - the
   clone did not produce the expected tree. Check $HARNESS_URL is reachable."

# ---- 3. dependencies ----------------------------------------------------------------
# Install BEFORE any version check. Bug 14: the v1 environment check ran first and failed on
# a numpy pin that the install itself fixes.
say "Dependencies (a few minutes on a fresh pod)"
python -m pip install -q --upgrade pip
python -m pip install -q -r "$HARNESS_DIR/requirements.txt"
python -m pip install -q nest_asyncio datasets pytest

# Read versions through a subprocess so nothing is imported into this shell's interpreter -
# no kernel restart is ever needed (bug 14 again).
say "Versions"
for pkg in numpy torch transformers datasets pytest; do
  ver=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "NOT INSTALLED")
  printf '  %-14s %s\n' "$pkg" "$ver"
done

python -c "import torch; print('  cuda available', torch.cuda.is_available(),
  '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU visible')" \
  2>/dev/null || echo "  cuda check failed"

# ---- 4. offline tests ---------------------------------------------------------------
say "Offline tests (no GPU, no keys, no network)"
cd "$PROJECT_DIR"
python -m pytest m2/tests/test_offline.py -q

# ---- 5. next ------------------------------------------------------------------------
say "Setup complete"
cat <<EOF

  Next, in this same terminal:

  1. Export your credentials. Prefix each line with a SPACE so it stays out of
     ~/.bash_history:

       export HF_TOKEN=...
       export OPENROUTER_API_KEY=...
       export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...
       export HEALTHCHECK_URL=...     RUNPOD_API_KEY=...

  2. Preflight - loads the model, runs the rig checks, spends no judge calls:

       cd "$PROJECT_DIR"
       python -m m2.run --concepts Garlic --preflight

  3. Start the watchdog in a SECOND terminal, then run:

       nohup python -m m2.run --concepts Garlic,Origami > /workspace/m2.out 2>&1 &
       tail -f /workspace/m2.out

  Full runbook: $PROJECT_DIR/m2/QUICKSTART.md

EOF
