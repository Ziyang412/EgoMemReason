#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/nas-ssd2/ziyang/Memory_project/COLM/evaluation/Gemini/eval_text_only_gemini.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_JSON="/nas-ssd2/ziyang/Memory_project/COLM/benchmark/filtered_batch_1/all_task_types_v2.json"
OUT_DIR="/nas-ssd2/ziyang/Memory_project/COLM/evaluation/Gemini/results/filtered_batch_1"
OUT_PREFIX="${OUT_DIR}/all_task_types_v2_text_only_gemini"
MODEL="gemini-3-flash-preview"

JOBS="${1:-4}"
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "Usage: $0 [num_jobs>=1] [extra python args...]"
  exit 1
fi
if [[ $# -ge 1 ]]; then
  shift
fi

mkdir -p "$OUT_DIR"

PIDS=()
for ((job_id=0; job_id<JOBS; job_id++)); do
  shard_out="${OUT_PREFIX}_job${job_id}of${JOBS}.json"
  echo "Starting job ${job_id}/${JOBS} -> ${shard_out}"
  "$PYTHON_BIN" "$SCRIPT" \
    --input_json "$INPUT_JSON" \
    --output_json "$shard_out" \
    --metrics_output_json "${OUT_PREFIX}_job${job_id}of${JOBS}_metrics.json" \
    --model "$MODEL" \
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
  if [[ "$status" -ne 0 ]]; then
    FAIL=1
  fi
done
set -e

if [[ "$FAIL" -ne 0 ]]; then
  echo "One or more shard jobs failed."
  exit 1
fi

MERGED_OUT="${OUT_PREFIX}.json"
METRICS_OUT="${OUT_PREFIX}_metrics.json"
"$PYTHON_BIN" - "$OUT_PREFIX" "$JOBS" "$INPUT_JSON" "$MODEL" "$MERGED_OUT" "$METRICS_OUT" << 'PY'
import json
import os
import sys
from collections import defaultdict

prefix, jobs_str, input_json, model, merged_out, metrics_out = sys.argv[1:7]
jobs = int(jobs_str)

shard_paths = [f"{prefix}_job{i}of{jobs}.json" for i in range(jobs)]
payloads = []
for path in shard_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing shard output: {path}")
    with open(path, "r") as f:
        payloads.append(json.load(f))

results = []
for p in payloads:
    results.extend(p.get("results", []))

results.sort(
    key=lambda row: (
        10**12 if row.get("sample_index") is None else int(row.get("sample_index")),
        str(row.get("example_id") or ""),
    )
)

correct = sum(1 for row in results if bool(row.get("is_correct")))
evaluated = len(results)
accuracy = (correct / evaluated) if evaluated else 0.0

query_type_to_memory_type = {
    "temporal_ordering": "episodic",
    "temporal_reasoning": "episodic",
    "state_tracking": "episodic",
    "spatial_tracking": "episodic",
    "multi_entity": "episodic",
    "semantic_event": "semantic",
}

def group_accuracy(rows, key):
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        value = row.get(key)
        group = str(value).strip() if value is not None and str(value).strip() else "unknown"
        if key == "memory_type" and group == "unknown":
            query_type = str(row.get("query_type") or "").strip().lower()
            if query_type in query_type_to_memory_type:
                group = query_type_to_memory_type[query_type]
            elif query_type:
                group = "episodic"
        stats[group]["total"] += 1
        if bool(row.get("is_correct")):
            stats[group]["correct"] += 1
    out = {}
    for group in sorted(stats.keys()):
        total = stats[group]["total"]
        c = stats[group]["correct"]
        out[group] = {"total": total, "correct": c, "accuracy": round((c / total) if total else 0.0, 4)}
    return out

task_type_accuracy = group_accuracy(results, "query_type")
memory_type_accuracy = group_accuracy(results, "memory_type")
summary = {
    "overall": {"total": evaluated, "correct": correct, "accuracy": round(accuracy, 4)},
    "task_type_accuracy": task_type_accuracy,
    "memory_type_accuracy": memory_type_accuracy,
}

for row in results:
    row.pop("sample_index", None)

first = payloads[0] if payloads else {}
merged = {
    "task_type": first.get("task_type"),
    "task_description": first.get("task_description"),
    "input_file": input_json,
    "model": model,
    "num_shards": jobs,
    "shard_files": shard_paths,
    "evaluated_examples": evaluated,
    "correct": correct,
    "accuracy": round(accuracy, 4),
    "task_type_accuracy": task_type_accuracy,
    "memory_type_accuracy": memory_type_accuracy,
    "results": results,
}

os.makedirs(os.path.dirname(merged_out), exist_ok=True)
with open(merged_out, "w") as f:
    json.dump(merged, f, indent=2)

metrics_payload = {
    "input_file": input_json,
    "output_file": merged_out,
    "model": model,
    "num_shards": jobs,
    "summary": summary,
}
os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
with open(metrics_out, "w") as f:
    json.dump(metrics_payload, f, indent=2)

print(f"Merged {jobs} shards -> {merged_out}")
print(f"Saved metrics -> {metrics_out}")
print(f"Overall Accuracy: {correct}/{evaluated} = {accuracy:.4f}")
print("Task Type Accuracy:")
for k, v in task_type_accuracy.items():
    print(f"  - {k}: {v['correct']}/{v['total']} = {v['accuracy']:.4f}")
print("Memory Type Accuracy:")
for k, v in memory_type_accuracy.items():
    print(f"  - {k}: {v['correct']}/{v['total']} = {v['accuracy']:.4f}")
PY

echo "Done. Merged output: ${MERGED_OUT}"
echo "Done. Metrics output: ${METRICS_OUT}"
