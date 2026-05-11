# Qwen3-VL

Evaluation for **Qwen-3-VL** at 8B / 30B-A3B (MoE) / 32B.

## Install

```bash
pip install transformers accelerate qwen-vl-utils
# HF weights:
#   Qwen/Qwen3-VL-8B-Instruct
#   Qwen/Qwen3-VL-30B-A3B-Instruct
#   Qwen/Qwen3-VL-32B-Instruct
```

## Run

### Main table (8B → 32B sequential pipeline)

```bash
# Stage 1: Qwen3-VL-8B at 1024 frames on GPUs 0–7
# Stage 2: Qwen3-VL-32B at 256 frames on GPUs 0–7
INPUT_JSON=$EGOMEM_DATA bash run_final_benchmark_500_apr22.sh
```

The script chunks the 500 questions across 8 GPUs, runs each stage, and merges chunked outputs. It then calls `hold_gpus.sh` to pin GPU memory; Ctrl+C to release.

### Frame ablation (Figure 6 in the paper)

```bash
INPUT_JSON=$EGOMEM_DATA bash run_frame_ablation.sh
```

### Per-split prompt ablations

```bash
bash run_per_split_instructions.sh   # task-specific system prompts
bash run_per_split_icl.sh            # task-specific ICL examples
bash run_per_split_short.sh          # short-form prompts
```

See `PER_SPLIT_INSTRUCTIONS.md` for the prompt content used in each variant.

## Files

| File | Purpose |
|---|---|
| `run_egolife_qwen3vl.py` | Main inference (8B / 32B) |
| `run_egolife_qwen3vl_per_split_instructions.py` | Per-split prompt variant |
| `run_egolife_qwen3vl_moe.py` | 30B-A3B MoE variant |
| `eval_temporal_ordering_qwen3vl_text_only.py` | Text-only ablation |
| `context_retrieval.py` | Frame retrieval / windowing helpers |
| `hold_gpus.py`, `hold_gpus.sh` | GPU memory holders between sweeps |
| `score_ablation.py` | Aggregates frame-ablation results |
| `run_final_benchmark_500_apr22.sh` | Canonical 8B+32B run |
| `run_frame_ablation.sh` | Frame-budget sweep |
| `run_per_split_*.sh` | Per-split prompt ablations |
| `PER_SPLIT_INSTRUCTIONS.md` | Documentation of per-split prompts |
