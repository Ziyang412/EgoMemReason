# LongVA

Evaluation for **LongVA-7B**.

## Install

```bash
git clone https://github.com/EvolvingLMMs-Lab/LongVA
cd LongVA && pip install -e ".[train]"
# Then point PYTHONPATH at LongVA/ when running our script.
```

HF weights: `lmms-lab/LongVA-7B-DPO`.

## Run

```bash
# From inside the cloned LongVA repo (so `from longva import ...` works)
INPUT_JSON=$EGOMEM_DATA FRAME_INDEX=$EGOLIFE_FRAMES_INDEX \
  bash /abs/path/to/run_final_benchmark_500_apr22.sh
```

## Files

| File | Purpose |
|---|---|
| `run_egolife_longva_image.py` | Main inference script (drop into the LongVA repo or symlink) |
| `run_final_benchmark_500_apr22.sh` | Canonical run |
