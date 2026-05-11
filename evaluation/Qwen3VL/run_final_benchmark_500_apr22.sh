#!/usr/bin/env bash
# Pipeline on final_benchmark_500_apr22.json:
#   Stage 1: Qwen3-VL-8B  1024f, 1 job on GPUs 0-7
#   Stage 2: Qwen3-VL-32B 256f,  1 job on GPUs 0-7
#   Stage 3: hold GPUs (one 8B copy per GPU) until Ctrl+C
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/run_egolife_qwen3vl.py"
HOLD_SH="${SCRIPT_DIR}/hold_gpus.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"
INDEX_DIR="/nas-ssd2/video_datasets/EgoLife/egolife_frames_index_per_identity"
MAX_PIXELS="${MAX_PIXELS:-151200}"
SAVE_EVERY="${SAVE_EVERY:-1}"
INPUT_JSON="/nas-ssd2/ziyang/Memory_project/COLM/final_benchmark/final_benchmark_500_apr22.json"
RESULTS_ROOT="${SCRIPT_DIR}/results/final_benchmark_500_apr22"

chunk_data() {
  local input_json="$1" chunk_dir="$2" num_workers="$3"
  python - "$input_json" "$chunk_dir" "$num_workers" <<'PY'
import json, os, re, sys
def parse_qt(qt):
    if isinstance(qt, dict):
        d=str(qt.get("date") or "").strip(); t=str(qt.get("time") or "").strip()
        if d and t:
            if re.fullmatch(r"\d{8}",t): return {"date":d,"time":t}
            if re.fullmatch(r"\d{6}",t): return {"date":d,"time":f"{t}00"}
        return None
    if not isinstance(qt,str): return None
    m=re.match(r"^DAY(\d+)\s*,\s*(\d{1,2}):(\d{2}):(\d{2})$",qt.strip(),re.IGNORECASE)
    if m: return {"date":f"DAY{int(m.group(1))}","time":f"{int(m.group(2)):02d}{int(m.group(3)):02d}{int(m.group(4)):02d}00"}
    return None
def norm_opts(opts):
    if isinstance(opts,list):
        return [{"id":str((v.get("id") or v.get("label") or chr(65+i)) if isinstance(v,dict) else chr(65+i)),"text":str((v.get("text") or v.get("value") or "") if isinstance(v,dict) else v)} for i,v in enumerate(opts)]
    if isinstance(opts,dict): return [{"id":str(k),"text":str(v)} for k,v in opts.items()]
    return []
ij,cd,nw=sys.argv[1],sys.argv[2],int(sys.argv[3])
with open(ij) as f: payload=json.load(f)
data=payload.get("samples") if isinstance(payload,dict) else payload
if data is None:
    for k in ("examples","items","data"):
        if isinstance(payload.get(k),list): data=payload[k]; break
rows=[]
for raw in (data or []):
    if not isinstance(raw,dict): continue
    item=dict(raw); item.pop("evidence_timestamps",None)
    if not item.get("question"): item["question"]=item.get("question_text") or item.get("query") or ""
    item["options"]=norm_opts(item.get("options") or item.get("choices") or item.get("candidates"))
    if not item.get("answer"): item["answer"]=item.get("correct_answer") or item.get("correct_choice") or ""
    if not item.get("target_time"):
        p=parse_qt(item.get("query_time"))
        if p: item["target_time"]=p
    if not item.get("video_id") and item.get("identity"): item["video_id"]=item["identity"]
    rows.append(item)
cs=(len(rows)+nw-1)//nw
for i in range(nw):
    chunk=rows[i*cs:min((i+1)*cs,len(rows))]
    path=os.path.join(cd,f"chunk_{i:02d}.json")
    with open(path,"w") as f: json.dump(chunk,f,indent=2,ensure_ascii=False)
    print(f"  {path}: {len(chunk)} items")
PY
}

merge_results() {
  local out_dir="$1" input_base="$2" num_workers="$3"
  python - "$out_dir" "$input_base" "$num_workers" <<'PY'
import json, sys
od,base,nw=sys.argv[1],sys.argv[2],int(sys.argv[3])
merged=[]
for i in range(nw):
    p=f"{od}/results_{base}_chunk_{i:02d}.json"
    try:
        with open(p) as f: merged.extend(json.load(f))
    except FileNotFoundError:
        continue
ans=[x for x in merged if x.get("correct") is not None]
cor=[x for x in ans if x.get("correct")]
acc=len(cor)/len(ans) if ans else 0
with open(f"{od}/results_{base}_merged.json","w") as f: json.dump(merged,f,indent=2,ensure_ascii=False)
print(f"  Merged {len(merged)} -> Acc: {len(cor)}/{len(ans)} = {acc:.4f}")
PY
}

run_stage() {
  local exp_name="$1" model_name="$2" max_frames="$3" gpu_groups="$4"
  local output_dir="${RESULTS_ROOT}/${exp_name}"
  local chunk_dir="${output_dir}/chunks" log_dir="${output_dir}/logs"
  local input_base; input_base="$(basename "$INPUT_JSON" .json)"
  read -r -a gpu_specs <<< "$gpu_groups"
  local num_workers="${#gpu_specs[@]}"

  echo ""; echo "==== [$(date -Iseconds)] ${exp_name} | ${model_name} | ${max_frames}f | ${num_workers} worker(s) | GPUs ${gpu_groups} ===="
  mkdir -p "$chunk_dir" "$log_dir"
  echo "Chunking..."; chunk_data "$INPUT_JSON" "$chunk_dir" "$num_workers"

  local pids=()
  for w in $(seq 0 $((num_workers - 1))); do
    local cid; cid="$(printf '%02d' "$w")"
    echo "  Worker ${w} on GPUs ${gpu_specs[$w]}"
    CUDA_VISIBLE_DEVICES="${gpu_specs[$w]}" $PYTHON_BIN "$EVAL_PY" \
      --dataset "${chunk_dir}/chunk_${cid}.json" \
      --egolife_frame_index_dir "${INDEX_DIR}" \
      --max_pixels "${MAX_PIXELS}" --model_name "${model_name}" \
      --save_every "${SAVE_EVERY}" \
      --output "${output_dir}/results_${input_base}_chunk_${cid}.json" \
      --print_each --max_frames "${max_frames}" \
      2>&1 | tee "${log_dir}/chunk_${cid}.log" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do wait "$pid" || echo "Worker $pid failed!"; done
  echo "Merging..."; merge_results "$output_dir" "$input_base" "$num_workers"
  echo "[$(date -Iseconds)] Done: ${exp_name}"
}

# Stage 1: 8B, 1024 frames, 1 job on all 8 GPUs
run_stage "apr22_8b_1024f" "Qwen/Qwen3-VL-8B-Instruct" 1024 "0,1,2,3,4,5,6,7"

# Stage 2: 32B, 256 frames, 1 job on all 8 GPUs
run_stage "apr22_32b_256f"  "Qwen/Qwen3-VL-32B-Instruct" 256  "0,1,2,3,4,5,6,7"

# Stage 3: hold GPUs indefinitely
echo ""; echo "==== [$(date -Iseconds)] Holding GPUs 0-7. Ctrl+C to release. ===="
GPUS="0 1 2 3 4 5 6 7" exec bash "$HOLD_SH"
