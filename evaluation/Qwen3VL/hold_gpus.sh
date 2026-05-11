#!/usr/bin/env bash
# Launch one full copy of Qwen3-VL-8B per GPU (0-7). Ctrl+C to release all.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/hold_gpus.py"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
LOG_DIR="${SCRIPT_DIR}/hold_gpus_logs"
mkdir -p "$LOG_DIR"

pids=()
cleanup() {
  echo; echo "Releasing GPUs..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

for g in $GPUS; do
  echo "Launching model on GPU ${g}..."
  CUDA_VISIBLE_DEVICES="$g" python "$PY" > "${LOG_DIR}/gpu_${g}.log" 2>&1 &
  pids+=($!)
done

echo "All processes launched. PIDs: ${pids[*]}"
echo "Logs: ${LOG_DIR}/gpu_*.log"
echo "Press Ctrl+C to release all GPUs."
wait
