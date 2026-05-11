#!/usr/bin/env bash
# Molmo2-8B eval on final_benchmark_500_apr22.json across 16/32/64/128/256 frames.
# Uses GPUs 0-3 in parallel (one chunk per GPU). After all sweeps complete, each
# GPU is held by loading Molmo2-8B and idling until Ctrl+C.
#
# Usage:  bash run_molmo2_final_benchmark_apr22.sh
#   GPU_GROUPS="0 1 2 3"  bash run_molmo2_final_benchmark_apr22.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/nas-ssd2/ziyang/pip_dir/hub}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INPUT_JSON="${INPUT_JSON:-/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json}"
FRAME_INDEX="${FRAME_INDEX:-/nas-ssd2/video_datasets/EgoLife/egolife_frames_index.json}"
EVAL_PY="${SCRIPT_DIR}/eval_molmo2_video.py"
MODEL_ID="${MODEL_ID:-allenai/Molmo2-8B}"
DTYPE="${DTYPE:-bf16}"
GPU_GROUPS="${GPU_GROUPS:-0 1 2 3}"
FRAMES_LIST="${FRAMES_LIST:-16 32 64 128 256}"
RESULTS_ROOT="${SCRIPT_DIR}/results/final_benchmark_500_apr22"

read -r -a GPU_SPECS <<< "$GPU_GROUPS"
NUM_WORKERS="${#GPU_SPECS[@]}"
INPUT_NAME="$(basename "${INPUT_JSON}" .json)"

run_frames() {
    local NF="$1"
    local EXP_DIR="${RESULTS_ROOT}/frames_${NF}f_direct"
    local MERGED="${EXP_DIR}/results_${INPUT_NAME}_merged.json"

    if [ -f "$MERGED" ]; then
        echo "=== SKIP frames_${NF}f_direct: ${MERGED} already exists ==="
        return 0
    fi

    echo ""
    echo "======================================================="
    echo "  Experiment: frames_${NF}f_direct"
    echo "  Input: ${INPUT_JSON}"
    echo "  GPUs: ${GPU_GROUPS} (${NUM_WORKERS} workers)"
    echo "======================================================="

    mkdir -p "${EXP_DIR}/chunks" "${EXP_DIR}/logs"

    ${PYTHON_BIN} -c "
import json, math
data = json.load(open('${INPUT_JSON}'))
samples = data.get('samples', data) if isinstance(data, dict) else data
n = ${NUM_WORKERS}
chunk_size = math.ceil(len(samples) / n)
for i in range(n):
    chunk = samples[i*chunk_size:(i+1)*chunk_size]
    out = '${EXP_DIR}/chunks/chunk_%02d.json' % i
    json.dump(chunk, open(out, 'w'), indent=2)
    print(f'  Chunk {i}: {len(chunk)} samples -> {out}')
"

    local -a PIDS=()
    for ((w=0; w<NUM_WORKERS; w++)); do
        local GPU_ID="${GPU_SPECS[$w]}"
        local CHUNK_FILE="${EXP_DIR}/chunks/chunk_$(printf '%02d' $w).json"
        local OUT_FILE="${EXP_DIR}/results_${INPUT_NAME}_chunk_$(printf '%02d' $w).json"
        local LOG_FILE="${EXP_DIR}/logs/chunk_$(printf '%02d' $w).log"

        if [ ! -f "$CHUNK_FILE" ]; then
            continue
        fi

        echo "  Worker $w on GPU $GPU_ID -> ${OUT_FILE}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON_BIN} "${EVAL_PY}" \
            --dataset "${CHUNK_FILE}" \
            --frame_index "${FRAME_INDEX}" \
            --output "${OUT_FILE}" \
            --model_id "${MODEL_ID}" \
            --dtype "${DTYPE}" \
            --max_frames "${NF}" \
            --print_each \
            --save_every 1 \
            > "${LOG_FILE}" 2>&1 &
        PIDS+=($!)
        sleep 2
    done

    local FAILED=0
    for ((w=0; w<${#PIDS[@]}; w++)); do
        if ! wait "${PIDS[$w]}"; then
            echo "  Worker $w FAILED (see ${EXP_DIR}/logs/chunk_$(printf '%02d' $w).log)"
            FAILED=1
        else
            echo "  Worker $w done"
        fi
    done

    if [ "$FAILED" -eq 1 ]; then
        echo "ERROR: Some workers failed for frames_${NF}f_direct"
        return 1
    fi

    ${PYTHON_BIN} -c "
import json, glob
files = sorted(glob.glob('${EXP_DIR}/results_${INPUT_NAME}_chunk_*.json'))
merged = []
for f in files:
    merged.extend(json.load(open(f)))
json.dump(merged, open('${MERGED}', 'w'), indent=2)
correct = sum(1 for r in merged if r.get('correct') is True)
valid = sum(1 for r in merged if r.get('pred') is not None and r.get('answer') is not None)
acc = correct / valid if valid else 0
print(f'  Merged: {len(merged)} samples, acc={acc:.4f} ({correct}/{valid})')
"
    echo "  Saved: ${MERGED}"
}

for NF in $FRAMES_LIST; do
    run_frames "$NF"
done

echo ""
echo "======================================================="
echo "  All frame sweeps complete. Holding GPUs ${GPU_GROUPS}"
echo "  (loading ${MODEL_ID} per GPU; Ctrl+C to release)"
echo "======================================================="

HOLD_LOG_DIR="${RESULTS_ROOT}/hold_gpus_logs"
mkdir -p "$HOLD_LOG_DIR"

HOLD_PY="${SCRIPT_DIR}/hold_gpus.py"
if [ ! -f "$HOLD_PY" ]; then
    cat > "$HOLD_PY" <<'PYEOF'
"""Load one full copy of Molmo2-8B on a single GPU and idle."""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_NAME = os.environ.get("MOLMO2_MODEL_ID", "allenai/Molmo2-8B")
DTYPE_STR = os.environ.get("MOLMO2_DTYPE", "bf16")
gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "auto": "auto"}
dtype = dtype_map.get(DTYPE_STR, torch.bfloat16)

print(f"[GPU {gpu_tag}] visible devices: {torch.cuda.device_count()}", flush=True)
print(f"[GPU {gpu_tag}] loading {MODEL_NAME} (dtype={DTYPE_STR})...", flush=True)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="cuda:0",
)

print(f"[GPU {gpu_tag}] model loaded. Idling. Ctrl+C to release.", flush=True)
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print(f"[GPU {gpu_tag}] releasing.", flush=True)
PYEOF
fi

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

for GPU_ID in "${GPU_SPECS[@]}"; do
    echo "Holding GPU ${GPU_ID}..."
    MOLMO2_MODEL_ID="${MODEL_ID}" MOLMO2_DTYPE="${DTYPE}" \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON_BIN} "${HOLD_PY}" \
        > "${HOLD_LOG_DIR}/gpu_${GPU_ID}.log" 2>&1 &
    HOLD_PIDS+=($!)
done

echo "Hold PIDs: ${HOLD_PIDS[*]}"
echo "Logs: ${HOLD_LOG_DIR}/gpu_*.log"
echo "Press Ctrl+C to release all GPUs."
wait
