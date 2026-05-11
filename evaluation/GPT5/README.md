# GPT-5

Frame-based evaluation for **GPT-5** via Azure OpenAI.

## Install

```bash
pip install openai
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-5
```

## Run

```bash
INPUT_JSON=$EGOMEM_DATA FRAMES_INDEX=$EGOLIFE_FRAMES_INDEX \
  bash run_final500_apr22_parallel.sh 16   # 16 shards
```

Shards merge via `merge_temporal_ordering_eval_shards.py`.

## Files

| File | Purpose |
|---|---|
| `eval_gpt5_frames.py` | Main inference script |
| `merge_temporal_ordering_eval_shards.py` | Merge sharded outputs |
| `run_final500_apr22_parallel.sh` | Canonical parallel run |
