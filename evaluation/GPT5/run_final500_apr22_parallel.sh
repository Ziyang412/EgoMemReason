#!/usr/bin/env bash
set -euo pipefail

# GPT-5 frames eval on final_benchmark_500_apr22.json
# Usage: bash run_final500_apr22_parallel.sh [num_jobs] [extra args]

SCRIPT="/nas-ssd2/ziyang/Memory_project/COLM/evaluation/GPT5/eval_gpt5_frames.py"
MERGE_SCRIPT="/nas-ssd2/ziyang/Memory_project/COLM/evaluation/GPT5/merge_temporal_ordering_eval_shards.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json"
FRAMES_INDEX="/nas-ssd2/video_datasets/EgoLife/egolife_frames_index.json"
OUT_DIR="${OUT_DIR:-/nas-ssd2/ziyang/Memory_project/COLM/evaluation/GPT5/results/final500_apr22}"
OUT_BASENAME="${OUT_BASENAME:-final500_apr22_gpt5_chat_frames_50}"
DEPLOYMENT="${DEPLOYMENT:-gpt-5-chat}"
AZURE_ENDPOINT="${AZURE_ENDPOINT:-https://YOUR-RESOURCE.cognitiveservices.azure.com/}"
API_VERSION="${API_VERSION:-2024-12-01-preview}"
API_KEY="${API_KEY:-${AZURE_OPENAI_API_KEY:-}}"
MAX_FRAMES="${MAX_FRAMES:-50}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2048}"

JOBS="${1:-10}"
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "Usage: $0 [num_jobs>=1] [extra python args...]"
  exit 1
fi
if [[ $# -ge 1 ]]; then
  shift
fi

DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry_run" ]]; then
    DRY_RUN=1
  fi
  case "$arg" in
    --output|--num_jobs|--job_id)
      echo "Do not pass ${arg} to this parallel script; it is managed internally."
      exit 1
      ;;
  esac
done

if [[ -z "${API_KEY}" && "$DRY_RUN" -ne 1 ]]; then
  echo "Missing API key. Set API_KEY or AZURE_OPENAI_API_KEY."
  exit 1
fi

DEPLOYMENT_SAFE="${DEPLOYMENT//[^a-zA-Z0-9._-]/_}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)_${DEPLOYMENT_SAFE}_${JOBS}jobs}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"
OUT_PREFIX="${RUN_DIR}/${OUT_BASENAME}"
LOG_DIR="${RUN_DIR}/logs"
EVENT_DIR="${RUN_DIR}/events"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$EVENT_DIR"
echo "Dataset: $DATASET"
echo "Output root: $OUT_DIR"
echo "Run subfolder: $RUN_DIR"
echo "Output prefix: $OUT_PREFIX"
echo "Job logs: $LOG_DIR"

PIDS=()
for ((job_id=0; job_id<JOBS; job_id++)); do
  shard_out="${OUT_PREFIX}_job${job_id}of${JOBS}.json"
  job_log="${LOG_DIR}/job${job_id}of${JOBS}.log"
  job_events="${EVENT_DIR}/job${job_id}of${JOBS}.jsonl"
  echo "Starting job ${job_id}/${JOBS} -> ${shard_out}"
  "$PYTHON_BIN" "$SCRIPT" \
    --dataset "$DATASET" \
    --frames_index "$FRAMES_INDEX" \
    --output "$shard_out" \
    --deployment "$DEPLOYMENT" \
    --azure_endpoint "$AZURE_ENDPOINT" \
    --api_key "$API_KEY" \
    --api_version "$API_VERSION" \
    --max_frames "$MAX_FRAMES" \
    --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
    --num_jobs "$JOBS" \
    --job_id "$job_id" \
    --debug_print_missing_pred \
    --debug_print_error_response \
    --debug_max_chars 800 \
    --events_jsonl "$job_events" \
    "$@" >"$job_log" 2>&1 &
  PIDS+=("$!")
done

set +e
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid"
  status=$?
  if [[ "$status" -ne 0 ]]; then
    FAIL=1
  fi
done
set -e

if [[ "$FAIL" -ne 0 ]]; then
  echo "One or more shard jobs failed."
  exit 1
fi

MERGED_OUT="${OUT_PREFIX}_merged_${JOBS}jobs.json"
"$PYTHON_BIN" "$MERGE_SCRIPT" \
  --inputs_glob "${OUT_PREFIX}_job*of${JOBS}.json" \
  --output "$MERGED_OUT"

echo ""
echo "Run summary:"
"$PYTHON_BIN" - "$MERGED_OUT" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    obj = json.load(f)
summary = obj.get("summary", {})
overall = summary.get("overall", {})
print(f"Overall: {overall.get('correct', 0)}/{overall.get('total', 0)} = {overall.get('accuracy', 0.0):.4f}")
rows = obj.get("results", [])
n_error = sum(1 for r in rows if bool(r.get("error")))
print(f"Error rows: {n_error}")
print("Per task type:")
for k, s in sorted(summary.get("by_task_type", {}).items()):
    print(f"  - {k}: {s.get('correct', 0)}/{s.get('total', 0)} = {s.get('accuracy', 0.0):.4f}")
PY

echo ""
echo "Done. Merged output: $MERGED_OUT"
