#!/bin/bash
# Evaluate VideoLLaMA3-7B on final_benchmark_500_apr22.json using GPUs 4-7 (4 workers).
# Usage: bash run_final_benchmark_500_apr22.sh [MAX_FRAMES]
#   MAX_FRAMES: number of frames to sample (default: 256)

set -e

MAX_FRAMES=${1:-256}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/run_egolife_videollama3_image.py"
MODEL_PATH="DAMO-NLP-SG/VideoLLaMA3-7B"

DATASET="/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json"
FRAMES_INDEX="/nas-ssd2/video_datasets/EgoLife/egolife_frames_index.json"

GPU_LIST="${GPU_LIST:-4 5 6 7}"
read -r -a GPUS <<< "$GPU_LIST"
NUM_CHUNKS="${#GPUS[@]}"

OUTPUT_DIR="${SCRIPT_DIR}/results/final_benchmark_500_apr22/mf${MAX_FRAMES}"
mkdir -p "${OUTPUT_DIR}"

echo "=== VideoLLaMA3-7B on final_benchmark_500_apr22 (max_frames=${MAX_FRAMES}) ==="
echo "  GPUs: ${GPU_LIST} (${NUM_CHUNKS} workers)"

PIDS=()
for i in "${!GPUS[@]}"; do
    CHUNK_OUTPUT="${OUTPUT_DIR}/chunk_${i}.json"
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python "${EVAL_PY}" \
        --dataset "${DATASET}" \
        --output "${CHUNK_OUTPUT}" \
        --model_path "${MODEL_PATH}" \
        --frames_index "${FRAMES_INDEX}" \
        --max_frames ${MAX_FRAMES} \
        --max_new_tokens 64 \
        --print_each \
        --save_every 10 \
        --device cuda:0 \
        --chunk ${i} \
        --num_chunks ${NUM_CHUNKS} \
        > "${OUTPUT_DIR}/log_chunk${i}.txt" 2>&1 &
    PIDS+=($!)
    echo "  Worker ${i} on GPU ${GPUS[$i]} (PID $!) -> ${CHUNK_OUTPUT}"
done

echo "Waiting for all workers to finish..."
FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: Worker ${i} (PID ${PIDS[$i]}) failed. Check ${OUTPUT_DIR}/log_chunk${i}.txt"
        FAIL=1
    else
        echo "  Worker ${i} done."
    fi
done

if [ "${FAIL}" -eq 1 ]; then
    echo "Some workers failed. Check logs above."
    exit 1
fi

MERGED="${OUTPUT_DIR}/merged.json"
echo "Merging chunk results -> ${MERGED}"
python -c "
import json
from collections import defaultdict

chunks = []
for i in range(${NUM_CHUNKS}):
    path = '${OUTPUT_DIR}/chunk_{}.json'.format(i)
    with open(path) as f:
        chunks.extend(json.load(f))

with open('${MERGED}', 'w') as f:
    json.dump(chunks, f, indent=2)

valid = [r for r in chunks if r.get('pred') is not None and r.get('correct_answer') is not None]
correct = [r for r in valid if r.get('correct') is True]
acc = len(correct) / len(valid) if valid else 0
print(f'Total: {len(chunks)} samples, Valid: {len(valid)}, Correct: {len(correct)}, Accuracy: {acc:.4f}')

by_type = defaultdict(list)
for r in chunks:
    by_type[r.get('query_type', 'unknown')].append(r)
print('--- Per query_type accuracy ---')
for qt in sorted(by_type):
    items = by_type[qt]
    v = [r for r in items if r.get('pred') is not None and r.get('correct_answer') is not None]
    c = [r for r in v if r.get('correct') is True]
    a = len(c) / len(v) if v else 0
    print(f'  {qt}: {a:.4f} ({len(c)}/{len(v)})')
"

echo "Done. Merged results: ${MERGED}"
