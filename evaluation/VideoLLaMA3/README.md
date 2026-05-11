# VideoLLaMA3

Evaluation for **VideoLLaMA3-8B**.

## Install

```bash
git clone https://github.com/DAMO-NLP-SG/VideoLLaMA3
cd VideoLLaMA3 && pip install -e .
```

HF weights: `DAMO-NLP-SG/VideoLLaMA3-8B`.

## Run

```bash
INPUT_JSON=$EGOMEM_DATA FRAME_INDEX=$EGOLIFE_FRAMES_INDEX \
  bash run_final_benchmark_500_apr22.sh
```

## Files

| File | Purpose |
|---|---|
| `run_egolife_videollama3_image.py` | Main inference script |
| `run_final_benchmark_500_apr22.sh` | Canonical run |
