# InternVL3.5

Evaluation for **InternVL3.5-8B** and **InternVL3.5-38B**.

## Install

Follow the upstream InternVL setup:
- Repo: https://github.com/OpenGVLab/InternVL
- HF weights: `OpenGVLab/InternVL3_5-8B`, `OpenGVLab/InternVL3_5-38B`

```bash
# In a fresh conda env per InternVL's instructions
pip install transformers timm accelerate flash-attn
```

## Run

```bash
INPUT_JSON=$EGOMEM_DATA FRAME_INDEX=$EGOLIFE_FRAMES_INDEX \
  bash run_internvl_final_benchmark_apr22_256f.sh
```

The script sweeps both 8B and 38B at 256 frames (matches paper Table 1). Override:

- `MODEL_ID=OpenGVLab/InternVL3_5-8B` to run only 8B
- `GPU_GROUPS="0 1 2 3"` to control GPU assignment
- `hold_gpus_internvl.py` is launched at the end to pin GPU memory between sweeps (kill it manually when done)

## Files

| File | Purpose |
|---|---|
| `run_egolife_internvl3_5_image.py` | Main inference script |
| `eval_temporal_ordering_internvl_text_only.py` | Text-only ablation (captions/transcripts) |
| `hold_gpus_internvl.py` | Holds GPU memory between sweeps |
| `run_internvl_final_benchmark_apr22_256f.sh` | Canonical run (8B + 38B, 256f) |
