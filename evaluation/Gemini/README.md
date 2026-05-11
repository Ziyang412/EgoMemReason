# Gemini

Frame-based evaluation for **Gemini-3-Flash** and **Gemini-3.1-Pro**.

## Install

```bash
pip install google-genai
export GOOGLE_API_KEY=...   # or GEMINI_API_KEY
```

## Run

```bash
# Gemini-3-Flash (paper main: 39.6 overall)
INPUT_JSON=$EGOMEM_DATA FRAMES_INDEX=$EGOLIFE_FRAMES_INDEX \
  bash run_final_benchmark_500_apr22_gemini_flash_512.sh 20      # 20 parallel API jobs

# Gemini-3.1-Pro (paper: 37.4 overall)
INPUT_JSON=$EGOMEM_DATA FRAMES_INDEX=$EGOLIFE_FRAMES_INDEX \
  bash run_final_benchmark_500_apr22_gemini_pro_512.sh 20

# Text-only ablation (caption / transcript only — Table 3 in the paper)
bash run_text_only_gemini_parallel.sh 8
```

Each shell script shards the 500 questions across `N` workers, then merges with `merge_temporal_ordering_eval_shards.py`. Final accuracy prints at the end.

## Files

| File | Purpose |
|---|---|
| `eval_gemini_frames.py` | Main per-question inference: samples frames around `query_time`, sends to Gemini, parses MCQ answer |
| `eval_text_only_gemini.py` | Text-only ablation (no frames, captions/transcripts only) |
| `eval_temporal_ordering_frames_gemini.py` | Specialized scorer for the Event Ordering split |
| `merge_temporal_ordering_eval_shards.py` | Merges sharded prediction JSONs |
| `run_final_benchmark_500_apr22_gemini_flash_512.sh` | Canonical Flash run |
| `run_final_benchmark_500_apr22_gemini_pro_512.sh` | Canonical Pro run |
| `run_text_only_gemini_parallel.sh` | Text-only ablation run |
