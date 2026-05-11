# Evaluation Scripts

One folder per system in the paper's main table. Each folder ships only the **inference + run scripts** — model weights and upstream code must be installed separately.

| Folder | Paper row | Backbone(s) | Notes |
|---|---|---|---|
| `Gemini/` | Gemini-3-Flash, Gemini-3.1-Pro | `gemini-3-flash-preview`, `gemini-3.1-pro` | Frame inputs; text-only ablation included |
| `GPT5/` | GPT-5 | Azure OpenAI | Frame inputs; sharded parallel runs |
| `InternVL/` | InternVL3.5-8B / 38B | HF: `OpenGVLab/InternVL3_5-*` | 256-frame default |
| `InternVideo/` | InternVideo2.5-8B | HF: `OpenGVLab/InternVideo2_5_Chat_8B` | |
| `LongVA/` | LongVA-7B | [LongVA upstream](https://github.com/EvolvingLMMs-Lab/LongVA) | Install LongVA in its own env first |
| `Molmo2/` | Molmo2-8B | HF: `allenai/Molmo2-8B` | Pixel-grounded; sweep 16/32/64/128/256 frames |
| `Qwen3VL/` | Qwen-3-VL 8B / 30B-A3B / 32B | HF: `Qwen/Qwen3-VL-*-Instruct` | Multi-stage 8B→32B run; per-split + frame ablations |
| `VideoLLaMA3/` | VideoLLaMA3-8B | HF: `DAMO-NLP-SG/VideoLLaMA3-8B` | |

## Common paths

Every run script expects these (set them once in your shell):

```bash
export INPUT_JSON=/abs/path/to/final_benchmark_500_apr22.json
export FRAME_INDEX=/abs/path/to/egolife_frames_index.json
```

The exact env-var names vary slightly per script (some use `INPUT_JSON`, others `DATASET`) — check the top of the `.sh` you're running.

## Output format

All scripts write a JSON list of records:

```json
{
  "id": "ENT_TRK_017",
  "task_type": "cumulative_state_tracking",
  "memory_type": "entity",
  "predicted_answer": "B",
  "answer": "B",
  "correct": true,
  "raw_response": "..."
}
```

A line at the end of each run prints per-split + overall accuracy. For sharded runs (Gemini, GPT-5, Qwen3VL, Molmo2), use the `merge_*` helper in the same folder to combine shards into a single file before scoring.
