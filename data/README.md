# Dataset

EgoMemReason is built on top of the [EgoLife](https://egolife-ai.github.io/) week-long egocentric videos. We release **500 multiple-choice questions** (public, no answer keys) plus a frame index that maps `(identity, day, time)` to extracted frames.

## Where to get it

The public benchmark lives on Hugging Face:

**→ https://huggingface.co/datasets/Ted412/EgoMemReason**

```python
from datasets import load_dataset
ds = load_dataset("Ted412/EgoMemReason")["test"]   # 500 questions, no answer keys
```

or download the raw file:

```bash
hf download Ted412/EgoMemReason annotations_public.jsonl --repo-type dataset --local-dir ./data
```

Then point the eval scripts at it:

```bash
export EGOMEM_DATA=/abs/path/to/annotations_public.jsonl       # or the JSON used in the run scripts
export EGOLIFE_FRAMES_INDEX=/abs/path/to/egolife_frames_index.json
```

> **Note on answer keys.** The HF dataset is the *public* split — questions and options only. The held-out answer key is not distributed; submit your predictions to the **leaderboard Space** (<https://huggingface.co/spaces/Ted412/EgoMemReason>), which scores them against the private answer key.

## Video frames

The frames themselves come from **EgoLife** and are *not* redistributed here (tens of TB). Accept the EgoLife data agreement at <https://egolife-ai.github.io/>, then build an `egolife_frames_index.json` that maps each frame on disk to `(identity, day, time)` so the eval scripts can sample frames around each question's `query_time`. The Qwen3VL scripts use a per-identity sharded index (`egolife_frames_index_per_identity/`); the rest use the single-file index.

## Question schema

```jsonc
{
  "example_id": 1,                              // unique int, 1-500
  "p_id": "A1_JAKE_DAY7_19_00_00_q001",         // unique per-question key (identity_daytime_qNNN)
  "identity": "A1_JAKE",                         // EgoLife participant
  "query_time": "DAY7, 19:00:00",                // when the question is asked
  "question": "What do I most often eat for breakfast?",
  "options": {                                   // 4-10 options, letters A-J
    "A": "Pancake",
    "B": "Rice",
    "C": "Burger",
    "D": "Dumplings"
  },
  "query_type": "Activity Pattern"               // one of the 6 capability tags
  // "correct_answer": "A"  — present only in the private split, not the public release
}
```

`options` is a dict; the valid answer letters for a question are exactly its keys. Event Ordering questions tend to have the most options (up to J).

## Splits used in the paper

The six `query_type` values map onto three memory types:

| Memory type | `query_type` | # Qs |
|---|---|---:|
| Entity   | `Cumulative State Tracking` | 100 |
| Entity   | `Temporal Counting`         | 100 |
| Event    | `Event Ordering`            | 100 |
| Event    | `Event Linking`             | 100 |
| Behavior | `Spatial Preference`        |  50 |
| Behavior | `Activity Pattern`          |  50 |
| **Total** | | **500** |

The main results table (`README.md` of the parent dir) reports accuracy per `query_type` plus overall. Submissions on the leaderboard Space are scored the same way.
