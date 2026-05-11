#!/usr/bin/env bash
# Run InternVideo2.5-Chat-8B on final_benchmark_500_apr22.json using GPUs 0-3 (4 workers).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSON=/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json \
RESULTS_ROOT=/nas-ssd2/ziyang/Memory_project/COLM/evaluation/Internvideo/results/final_benchmark_500_apr22 \
GPU_GROUPS="0 1 2 3" \
NUM_WORKERS=4 \
MAX_FRAMES=256 \
bash "${SCRIPT_DIR}/run_all_task_types_v2_internvideo2_5_8b_2jobs_256f.sh"

echo ""
echo "=========================================="
echo "Evaluation finished. Holding GPUs 0-3 by re-loading InternVideo2.5 on each..."
echo "Press Ctrl+C to release."
echo "=========================================="

HOLD_MODEL="${HOLD_MODEL:-OpenGVLab/InternVideo2_5_Chat_8B}"
HOLD_PIDS=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$g" python3 -u - <<PY &
import time, torch
from transformers import AutoModel, AutoTokenizer
mp = "${HOLD_MODEL}"
print(f"[hold gpu=${g}] loading {mp} ...", flush=True)
tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(mp, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
print(f"[hold gpu=${g}] model loaded, holding forever", flush=True)
while True:
    time.sleep(3600)
PY
  HOLD_PIDS+=($!)
done

trap 'echo "Releasing holds..."; for pid in "${HOLD_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; exit 0' INT TERM
wait
