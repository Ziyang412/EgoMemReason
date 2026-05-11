#!/usr/bin/env bash
# InternVL3.5 eval on final_benchmark_500_apr22.json at 256 frames.
# Runs 8B first (3 parallel workers, one GPU each), then 38B (single worker
# sharded across 3 GPUs), both on GPUs 1-3. After both sweeps finish,
# each GPU is held by loading InternVL3.5-8B and idling until Ctrl+C.
#
# Usage:  bash run_internvl_final_benchmark_apr22_256f.sh
#   SKIP_8B=1 bash run_internvl_final_benchmark_apr22_256f.sh   (skip 8B step)
#   GPU_GROUPS_8B="1 2 3"   (space-separated, one spec per worker)
#   GPU_GROUPS_38B="1,2,3"  (comma-separated, single worker shards)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSON="${INPUT_JSON:-/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results/final_benchmark_500_apr22}"
MAX_FRAMES="${MAX_FRAMES:-256}"

# 8B: one worker per GPU; 38B: one worker sharded across all GPUs.
GPU_GROUPS_8B="${GPU_GROUPS_8B:-1 2 3}"
NUM_WORKERS_8B="${NUM_WORKERS_8B:-3}"
GPU_GROUPS_38B="${GPU_GROUPS_38B:-1,2,3}"
NUM_WORKERS_38B="${NUM_WORKERS_38B:-1}"
SKIP_8B="${SKIP_8B:-0}"
SKIP_38B="${SKIP_38B:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -f "$INPUT_JSON" ]; then
    echo "Input JSON not found: $INPUT_JSON" >&2
    exit 1
fi

mkdir -p "$RESULTS_ROOT"

if [ "${SKIP_8B}" = "1" ] || [ "${SKIP_8B,,}" = "true" ]; then
    echo "=== SKIP 8B (SKIP_8B=${SKIP_8B}) ==="
else
    echo ""
    echo "======================================================="
    echo "  [1/2] InternVL3.5-8B  (256f, ${NUM_WORKERS_8B} workers on GPUs ${GPU_GROUPS_8B})"
    echo "  Input:  ${INPUT_JSON}"
    echo "  Output: ${RESULTS_ROOT}"
    echo "======================================================="
    INPUT_JSON="${INPUT_JSON}" \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    MODEL_NAME="OpenGVLab/InternVL3_5-8B" \
    MODEL_TAG="internvl3_5_8b" \
    MAX_FRAMES="${MAX_FRAMES}" \
    GPU_GROUPS="${GPU_GROUPS_8B}" \
    NUM_WORKERS="${NUM_WORKERS_8B}" \
    bash "${SCRIPT_DIR}/run_all_task_types_v2_internvl3_5_8b_2jobs_256f.sh"
fi

if [ "${SKIP_38B}" = "1" ] || [ "${SKIP_38B,,}" = "true" ]; then
    echo "=== SKIP 38B (SKIP_38B=${SKIP_38B}) ==="
else
    echo ""
    echo "======================================================="
    echo "  [2/2] InternVL3.5-38B (256f, ${NUM_WORKERS_38B} worker on GPUs ${GPU_GROUPS_38B})"
    echo "  Input:  ${INPUT_JSON}"
    echo "  Output: ${RESULTS_ROOT}"
    echo "======================================================="
    INPUT_JSON="${INPUT_JSON}" \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    MODEL_NAME="OpenGVLab/InternVL3_5-38B" \
    MODEL_TAG="internvl3_5_38b" \
    MAX_FRAMES="${MAX_FRAMES}" \
    GPU_GROUPS="${GPU_GROUPS_38B}" \
    NUM_WORKERS="${NUM_WORKERS_38B}" \
    bash "${SCRIPT_DIR}/run_all_task_types_v2_internvl3_5_8b_2jobs_256f.sh"
fi

echo ""
echo "======================================================="
echo "  Sweeps complete. Holding GPUs ${GPU_GROUPS_8B} with InternVL3.5-8B."
echo "  Press Ctrl+C to release."
echo "======================================================="

HOLD_LOG_DIR="${RESULTS_ROOT}/hold_gpus_logs"
mkdir -p "$HOLD_LOG_DIR"

HOLD_PY="${SCRIPT_DIR}/hold_gpus_internvl.py"
if [ ! -f "$HOLD_PY" ]; then
    cat > "$HOLD_PY" <<'PYEOF'
"""Load one full copy of InternVL3.5-8B on a single GPU and idle."""
import os
import time
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ.get("INTERNVL_MODEL_ID", "OpenGVLab/InternVL3_5-8B")
DTYPE_STR = os.environ.get("INTERNVL_DTYPE", "bf16")
gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
dtype = dtype_map.get(DTYPE_STR, torch.bfloat16)

print(f"[GPU {gpu_tag}] visible devices: {torch.cuda.device_count()}", flush=True)
print(f"[GPU {gpu_tag}] loading {MODEL_NAME} (dtype={DTYPE_STR})...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="cuda:0",
).eval()

print(f"[GPU {gpu_tag}] model loaded. Idling. Ctrl+C to release.", flush=True)
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print(f"[GPU {gpu_tag}] releasing.", flush=True)
PYEOF
fi

# Fan out one holder per individual GPU (convert spec to single-GPU list).
IFS=',' read -r -a HOLD_GPUS <<< "${GPU_GROUPS_8B// /,}"

HOLD_PIDS=()
cleanup() {
    echo
    echo "Releasing GPUs..."
    for pid in "${HOLD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

for GPU_ID in "${HOLD_GPUS[@]}"; do
    [ -z "$GPU_ID" ] && continue
    echo "Holding GPU ${GPU_ID}..."
    INTERNVL_MODEL_ID="OpenGVLab/InternVL3_5-8B" INTERNVL_DTYPE="bf16" \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON_BIN} "${HOLD_PY}" \
        > "${HOLD_LOG_DIR}/gpu_${GPU_ID}.log" 2>&1 &
    HOLD_PIDS+=($!)
done

echo "Hold PIDs: ${HOLD_PIDS[*]}"
echo "Logs: ${HOLD_LOG_DIR}/gpu_*.log"
echo "Press Ctrl+C to release all GPUs."
wait
