#!/bin/bash
# Evaluate WorldMM on final_benchmark_500_apr22.json
# Usage: bash script/6_eval_benchmark_final500_apr22.sh [--retriever-model gpt-5-chat] [--respond-model gpt-5-chat]

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

BENCHMARK_JSON="/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json"
CAPTION_DIR="/nas-ssd2/ziyang/Memory_project/WorldMM_caption"
METADATA_DIR="output/metadata"
RET_MODEL="gpt-5-chat"
RESP_MODEL="gpt-5-chat"
MAX_ROUNDS=5
MAX_ERRORS=5
EPISODIC_TOP_K=3
SEMANTIC_TOP_K=10
VISUAL_TOP_K=3
OUTPUT_DIR="output"

while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-json) BENCHMARK_JSON="$2"; shift 2 ;;
        --caption-dir) CAPTION_DIR="$2"; shift 2 ;;
        --metadata-dir) METADATA_DIR="$2"; shift 2 ;;
        --retriever-model) RET_MODEL="$2"; shift 2 ;;
        --respond-model) RESP_MODEL="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --max-errors) MAX_ERRORS="$2"; shift 2 ;;
        --episodic-top-k) EPISODIC_TOP_K="$2"; shift 2 ;;
        --semantic-top-k) SEMANTIC_TOP_K="$2"; shift 2 ;;
        --visual-top-k) VISUAL_TOP_K="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate worldmm

# Azure OpenAI configuration
export AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.cognitiveservices.azure.com/"
export AZURE_OPENAI_KEY=""
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

python eval/eval_benchmark.py \
    --benchmark-json "$BENCHMARK_JSON" \
    --caption-dir "$CAPTION_DIR" \
    --metadata-dir "$METADATA_DIR" \
    --retriever-model "$RET_MODEL" \
    --respond-model "$RESP_MODEL" \
    --max-rounds "$MAX_ROUNDS" \
    --max-errors "$MAX_ERRORS" \
    --episodic-top-k "$EPISODIC_TOP_K" \
    --semantic-top-k "$SEMANTIC_TOP_K" \
    --visual-top-k "$VISUAL_TOP_K" \
    --output-dir "$OUTPUT_DIR"
