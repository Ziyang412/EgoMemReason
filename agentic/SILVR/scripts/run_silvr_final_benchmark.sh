#!/usr/bin/env bash
set -euo pipefail

# SILVR eval on benchmark v3 with GPT-5 captions
# Usage: bash scripts/run_v3_gpt5cap.sh [num_jobs]

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

API_KEY="${AZURE_OPENAI_API_KEY:?Set AZURE_OPENAI_API_KEY}"
ANNO_PATH="/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json"
CAPTION_PATH="/nas-ssd2/ziyang/Memory_project/openai/egolife/gpt5_caption/results/updated_clip_caption/merged_caption_only"
OUTPUT_BASE="output/final_benchmark_500/gpt5_cap_uniform"

JOBS="${1:-10}"

echo "============================================"
echo "SILVR Eval | Benchmark v3 + GPT-5 Captions"
echo "  Jobs: $JOBS"
echo "  Output: ${SCRIPT_DIR}/${OUTPUT_BASE}"
echo "============================================"

LOG_DIR="${SCRIPT_DIR}/${OUTPUT_BASE}/job_logs"
mkdir -p "$LOG_DIR"

PIDS=()
for ((job_id=0; job_id<JOBS; job_id++)); do
    job_log="${LOG_DIR}/job_${job_id}_of_${JOBS}.log"
    echo "Starting job ${job_id}/${JOBS} -> ${job_log}"
    cd "$SCRIPT_DIR" && "$PYTHON_BIN" main.py \
        --dataset worldmm \
        --anno_path "$ANNO_PATH" \
        --caption_path "$CAPTION_PATH" \
        --caption_type gpt5 \
        --subtitle_path "" \
        --max_caption_chars 450000 \
        --model gpt-5-chat \
        --azure_endpoint "https://YOUR-RESOURCE.cognitiveservices.azure.com/" \
        --api_key "$API_KEY" \
        --azure_api_version "2024-12-01-preview" \
        --azure_deployment "gpt-5-chat" \
        --max_completion_tokens 2048 \
        --prompt_type worldmm \
        --output_base_path "$OUTPUT_BASE" \
        --num_workers 1 \
        --single_process \
        --time_sleep 2 \
        --num_jobs "$JOBS" \
        --job_id "$job_id" \
        --disable_eval \
        >"$job_log" 2>&1 &
    PIDS+=("$!")
done

echo ""
echo "All $JOBS jobs launched. Waiting for completion..."
echo "  Monitor: tail -f ${LOG_DIR}/job_*_of_${JOBS}.log"
echo ""

set +e
FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    status=$?
    if [[ "$status" -ne 0 ]]; then
        echo "  [FAIL] Job $i exited with status $status"
        FAIL=1
    else
        echo "  [OK]   Job $i completed"
    fi
done
set -e

if [[ "$FAIL" -ne 0 ]]; then
    echo ""
    echo "WARNING: Some jobs failed. Check logs in ${LOG_DIR}"
    echo "Continuing with evaluation on completed results..."
fi

echo ""
echo "Running evaluation..."
cd "$SCRIPT_DIR" && "$PYTHON_BIN" main.py \
    --dataset worldmm \
    --anno_path "$ANNO_PATH" \
    --caption_path "$CAPTION_PATH" \
    --caption_type gpt5 \
    --subtitle_path "" \
    --model gpt-5-chat \
    --azure_endpoint "https://YOUR-RESOURCE.cognitiveservices.azure.com/" \
    --api_key unused \
    --azure_deployment gpt-5-chat \
    --prompt_type worldmm \
    --output_base_path "$OUTPUT_BASE" \
    --disable_infer

echo ""
echo "============================================"
echo "Done! Results:"
cat "${SCRIPT_DIR}/${OUTPUT_BASE}/results.json"
echo ""
echo "============================================"
