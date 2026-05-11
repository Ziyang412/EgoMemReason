#!/bin/bash

export VIDEO_LLM_URL=http://127.0.0.1:8060/video_llm
export VLM_URL=http://127.0.0.1:8080/vlm
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}"
export ENDPOINT_URL="${ENDPOINT_URL:-https://YOUR-RESOURCE.cognitiveservices.azure.com/}"
export DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-gpt-5-chat}"

datetime=$(date +%Y%m%d_%H%M%S)
max_turns=12
mkdir -p infer_logs

model_name_or_path=Ego-R1/Ego-R1-Agent-3B
benchmark_json=/nas-ssd2/ziyang/Memory_project/COLM/benchmark/filtered_batch_1/all_task_types_v2.json

PYTHONPATH=. python eval/infer_egolife_bench.py \
    --model_name_or_path ${model_name_or_path} \
    --benchmark_json ${benchmark_json} \
    --max_turns ${max_turns} \
    --data_start 0 \
    --data_end -1 \
    --result_dir results/egolife_bench \
    --vllm_base_url http://localhost:23333/v1 \
    | tee infer_logs/log_egolife_bench_mt${max_turns}_${datetime}.log 2>&1
