#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# pod_watchdog.sh - stop a hung RunPod pod from OUTSIDE the Jupyter kernel.
#
# WHY THIS EXISTS. A kernel hang (the "Kernel appears to have died / is unresponsive" popup)
# cannot be caught from inside the notebook: a stuck main thread holds the GIL, so even the
# notebook's own heartbeat thread stops running. Detection has to live in a SEPARATE process.
# RunPod also has no native "auto-stop an idle POD" (that idle timeout is a serverless
# feature, not a pod feature), so nothing stops it for you.
#
# WHAT IT DOES. Polls the GPU. During a batch the GPU should be busy; if it sits at ~0%
# utilisation WHILE the model is still resident in VRAM (so a run is genuinely in progress,
# not just an idle pod between jobs) for IDLE_MINUTES straight, that is a hang - and it STOPS
# the pod (STOP, not terminate: the volume and every zip are preserved, restartable later).
#
# WHY GPU + VRAM, not GPU alone. The judge runs over the network (OpenRouter), during which
# the GPU is legitimately ~0% for a while. Requiring the model to still be loaded, plus a
# generous window, keeps normal judge waits from tripping it. Tune IDLE_MINUTES ABOVE your
# longest real GPU-idle stretch (a slow judge batch); 20 min is safe for this pipeline.
#
# RUN IT in a second terminal / tmux window, NOT in the kernel:
#     nohup bash pod_watchdog.sh > /workspace/watchdog.log 2>&1 &
# Needs RUNPOD_POD_ID (auto-set on RunPod) and RUNPOD_API_KEY (or an authenticated runpodctl).
# ---------------------------------------------------------------------------------------
set -u

IDLE_MINUTES="${IDLE_MINUTES:-20}"      # sustained GPU-idle-while-loaded before stopping
POLL_SECONDS="${POLL_SECONDS:-30}"      # how often to sample
UTIL_MAX="${UTIL_MAX:-5}"               # GPU util <= this counts as idle (percent)
MEM_MIN_MB="${MEM_MIN_MB:-10000}"       # VRAM used >= this means the model is loaded (a run is live)
POD_ID="${RUNPOD_POD_ID:-}"
API_KEY="${RUNPOD_API_KEY:-}"

need=$(( IDLE_MINUTES * 60 / POLL_SECONDS ))   # consecutive idle samples to trip
idle=0

echo "[watchdog] start: stop pod after ${IDLE_MINUTES}m of GPU<=${UTIL_MAX}% while VRAM>=${MEM_MIN_MB}MB"
echo "[watchdog] pod=${POD_ID:-<unset>} poll=${POLL_SECONDS}s need=${need} samples"
if [ -z "$POD_ID" ]; then echo "[watchdog] WARNING: RUNPOD_POD_ID unset - cannot stop the pod"; fi

stop_pod() {
  echo "[watchdog] $(date -u +%FT%TZ) STOPPING pod ${POD_ID} (hang: ${IDLE_MINUTES}m idle-while-loaded)"
  if command -v runpodctl >/dev/null 2>&1 && [ -n "$POD_ID" ]; then
    runpodctl stop pod "$POD_ID" && { echo "[watchdog] stopped via runpodctl"; return 0; }
  fi
  if [ -n "$POD_ID" ] && [ -n "$API_KEY" ]; then
    curl -s -X POST "https://api.runpod.io/graphql" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${API_KEY}" \
      -d "{\"query\":\"mutation { podStop(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus } }\"}" \
      && { echo "[watchdog] stop issued via API"; return 0; }
  fi
  echo "[watchdog] STOP FAILED - stop the pod manually in the RunPod console"
  return 1
}

while true; do
  # "util, mem_used" for GPU 0; if nvidia-smi fails, skip this sample rather than false-trip
  read -r util mem < <(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                        --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' | tr ',' ' ')
  if [ -z "${util:-}" ]; then
    echo "[watchdog] $(date -u +%FT%TZ) nvidia-smi unavailable - skipping sample"
    sleep "$POLL_SECONDS"; continue
  fi
  if [ "$util" -le "$UTIL_MAX" ] && [ "$mem" -ge "$MEM_MIN_MB" ]; then
    idle=$(( idle + 1 ))
    echo "[watchdog] $(date -u +%FT%TZ) idle sample ${idle}/${need} (util=${util}% mem=${mem}MB)"
    if [ "$idle" -ge "$need" ]; then stop_pod; exit 0; fi
  else
    if [ "$idle" -ne 0 ]; then
      echo "[watchdog] $(date -u +%FT%TZ) active again (util=${util}% mem=${mem}MB) - counter reset"
    fi
    idle=0
  fi
  sleep "$POLL_SECONDS"
done
