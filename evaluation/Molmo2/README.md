# Molmo2

Evaluation for **Molmo2-8B**.

## Install

```bash
pip install transformers accelerate
# HF weights are pulled at runtime from `allenai/Molmo2-8B`.
```

## Run

```bash
INPUT_JSON=$EGOMEM_DATA FRAME_INDEX=$EGOLIFE_FRAMES_INDEX \
GPU_GROUPS="0 1 2 3" FRAMES_LIST="16 32 64 128 256" \
  bash run_molmo2_final_benchmark_apr22.sh
```

The canonical run sweeps 16/32/64/128/256 frames across 4 GPUs in parallel — this is the data behind the frame-ablation curve in the paper. To match the main-table number only, use `FRAMES_LIST="256"`.

After sweeps complete, the script holds GPU memory via `hold_gpus.py` until you Ctrl+C.

## Files

| File | Purpose |
|---|---|
| `eval_molmo2_video.py` | Main inference script |
| `context_retrieval.py` | Frame retrieval / windowing helpers |
| `hold_gpus.py` | Holds GPU memory between sweeps |
| `score_ablation.py` | Aggregates per-frame-count results into the ablation table |
| `run_molmo2_final_benchmark_apr22.sh` | Canonical run (16-256f sweep) |
