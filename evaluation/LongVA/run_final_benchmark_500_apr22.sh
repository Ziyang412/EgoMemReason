#!/usr/bin/env bash
# Run LongVA-7B-DPO on final_benchmark_500_apr22.json using GPUs 4-7 (4 workers).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSON=/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json \
GPU_GROUPS="4 5 6 7" \
NUM_WORKERS=4 \
MAX_FRAMES=256 \
bash "${SCRIPT_DIR}/run_all_task_types_v2_longva_7b_4jobs_256f.sh"
