# InternVideo2.5

Evaluation for **InternVideo2.5-8B**.

## Install

Follow the upstream setup:
- Repo: https://github.com/OpenGVLab/InternVideo
- HF weights: `OpenGVLab/InternVideo2_5_Chat_8B`

## Run

```bash
INPUT_JSON=$EGOMEM_DATA FRAME_INDEX=$EGOLIFE_FRAMES_INDEX \
  bash run_final_benchmark_500_apr22.sh
```

## Files

| File | Purpose |
|---|---|
| `run_egolife_internvideo2_5_image.py` | Main inference script |
| `run_final_benchmark_500_apr22.sh` | Canonical run |
