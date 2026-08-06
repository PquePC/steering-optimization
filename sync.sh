#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# sync.sh - update this repo on the pod to exactly match GitHub, before a run or after a
# code change. Replaces "delete the old notebook and re-upload it".
#
# It FORCE-matches the remote (git fetch + reset --hard): running the notebook writes cell
# outputs back into the .ipynb, so a normal `git pull` would conflict. Those local outputs
# are disposable, so we discard them and take the remote's version. Your own machine is where
# you edit; the pod is a mirror.
#
# AUTH. Private repo, so a fetch needs credentials. In order of preference it uses:
#   1. $GITHUB_TOKEN, or a token in /workspace/.gh_token  -> HTTPS fetch (recommended)
#   2. otherwise `git fetch origin` -> works if the pod has an SSH key / is public
# The token is used transiently for the fetch and is never written into .git/config.
#
# FIRST TIME (repo not cloned yet) - run ONE of these in a terminal, then use sync.sh after:
#   HTTPS+token:
#     GH=YOUR_TOKEN; git clone https://x-access-token:$GH@github.com/PquePC/Emergent-Introspection.git \
#       /workspace/Emergent-Introspection \
#       && git -C /workspace/Emergent-Introspection remote set-url origin \
#            https://github.com/PquePC/Emergent-Introspection.git \
#       && printf '%s' "$GH" > /workspace/.gh_token && chmod 600 /workspace/.gh_token
#   SSH (needs an SSH key on the pod):
#     git clone git@github.com:PquePC/Emergent-Introspection.git /workspace/Emergent-Introspection
#
# THEN, before each run / after any code change:
#   bash "/workspace/Emergent-Introspection/Steering Optimization/sync.sh"
#   ...and in Jupyter: File -> Reload Notebook from Disk (or re-open it), then re-run.
# ---------------------------------------------------------------------------------------
set -u

# Locate the repo this script lives in (works wherever you cloned it).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${REPO_DIR:-}" ]; then
  echo "[sync] not inside a git repo. Clone it first (see the header of this script)."; exit 1
fi
BRANCH="${SYNC_BRANCH:-main}"
echo "[sync] repo   : $REPO_DIR"
echo "[sync] branch : $BRANCH"

# Resolve a token (env wins, else the persisted file).
TOKEN="${GITHUB_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -f /workspace/.gh_token ]; then TOKEN="$(cat /workspace/.gh_token)"; fi

# owner/repo from the origin URL (handles git@... and https://...).
SLUG="$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null | sed -E 's#.*github.com[:/]##; s#\.git$##')"

before="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo none)"

if [ -n "$TOKEN" ] && [ -n "$SLUG" ]; then
  echo "[sync] fetching via HTTPS token"
  git -C "$REPO_DIR" fetch --quiet "https://x-access-token:${TOKEN}@github.com/${SLUG}.git" "$BRANCH" || {
    echo "[sync] token fetch failed - check the token / its graphql-unrelated 'repo' read scope"; exit 1; }
else
  echo "[sync] no token found - fetching via configured remote (SSH / public)"
  git -C "$REPO_DIR" fetch --quiet origin "$BRANCH" || {
    echo "[sync] fetch failed - no SSH key and no token. See this script's header for setup."; exit 1; }
fi

# Force the working tree to match the fetched commit; local notebook outputs are discarded.
git -C "$REPO_DIR" reset --hard --quiet FETCH_HEAD
after="$(git -C "$REPO_DIR" rev-parse --short HEAD)"

# Post-steps: shell scripts need LF + execute bit on the pod.
find "$REPO_DIR" -name '*.sh' -print0 2>/dev/null | while IFS= read -r -d '' f; do
  sed -i 's/\r$//' "$f" 2>/dev/null; chmod +x "$f" 2>/dev/null
done

echo ""
if [ "$before" = "$after" ]; then
  echo "[sync] already up to date at $after (local outputs reset)"
else
  echo "[sync] updated: $before -> $after"
  echo "[sync] new commits:"
  git -C "$REPO_DIR" --no-pager log --oneline "${before}..${after}" 2>/dev/null | sed 's/^/    /'
fi
NB="$REPO_DIR/Steering Optimization/measurement_lab.ipynb"
echo ""
echo "[sync] notebook: $NB"
echo "[sync] NEXT: in Jupyter -> File -> Reload Notebook from Disk (or re-open), then re-run"
echo "[sync]       the CONTROL PANEL cell and Run All."
