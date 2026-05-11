#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/eval_gemini_frames.py"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_temporal_ordering_eval_shards.py"
DATASET="/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json"
FRAMES_INDEX="/nas-ssd2/video_datasets/EgoLife/egolife_frames_index.json"
BASE_OUT="${SCRIPT_DIR}/results/updated_runs"
RUN_NAME="final_benchmark_500_apr22_gemini_3_1_pro_frames_512"
RUN_DIR="${BASE_OUT}/${RUN_NAME}"
JOB_DIR="${RUN_DIR}/jobs"
OUT_PREFIX="${JOB_DIR}/${RUN_NAME}"
MODEL="${MODEL:-gemini-3.1-pro-preview}"
MAX_FRAMES=512

JOBS="${1:-10}"
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "Usage: $0 [num_jobs>=1]"
  exit 1
fi
if [[ $# -ge 1 ]]; then shift; fi

mkdir -p "$RUN_DIR" "$JOB_DIR"

PIDS=()
for ((job_id=0; job_id<JOBS; job_id++)); do
  shard_out="${OUT_PREFIX}_job${job_id}of${JOBS}.json"
  echo "Starting job ${job_id}/${JOBS} -> ${shard_out}"
  python "$SCRIPT" \
    --dataset "$DATASET" \
    --frames_index "$FRAMES_INDEX" \
    --output "$shard_out" \
    --model "$MODEL" \
    --max_frames "$MAX_FRAMES" \
    --num_jobs "$JOBS" \
    --job_id "$job_id" \
    "$@" &
  PIDS+=("$!")
done

set +e
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid"
  status=$?
  if [[ "$status" -ne 0 ]]; then FAIL=1; fi
done
set -e

if [[ "$FAIL" -ne 0 ]]; then
  echo "One or more shard jobs failed."
  exit 1
fi

MERGED_OUT="${RUN_DIR}/${RUN_NAME}_merged_${JOBS}jobs.json"
python "$MERGE_SCRIPT" \
  --inputs_glob "${OUT_PREFIX}_job*of${JOBS}.json" \
  --output "$MERGED_OUT"

echo "Done. Merged output: $MERGED_OUT"
