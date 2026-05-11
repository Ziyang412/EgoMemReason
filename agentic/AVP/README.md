# AVP — Agentic Video Pipeline (ours)

Multi-round retrieval-and-reason agent over the week-long video, with a global frame budget.

**Backbone:** Gemini-3-Flash. **Paper row:** 34.0 overall (best agentic system).

## How it works

1. The agent gets the question + a coarse frame index (timestamps only, no pixels).
2. Round 1: it picks one or more time windows it thinks are relevant.
3. We retrieve dense frames inside those windows (up to `MAX_FRAMES`) and pass them back.
4. Round 2-N: the agent either commits to an answer or requests new windows, subject to `TOTAL_FRAME_BUDGET`.
5. Final answer is parsed from the last MCQ output.

See `agent_loop.py` for the loop, `prompts.py` for the system prompt, `frame_window.py` for window expansion.

## Install

```bash
pip install google-genai
export GOOGLE_API_KEY=...
```

## Run

```bash
INPUT_JSON=$EGOMEM_DATA FRAMES_INDEX=$EGOLIFE_FRAMES_INDEX \
  bash run_avp_final_benchmark.sh 20      # 20 parallel jobs
```

Tunables (env vars):

| Var | Default | Purpose |
|---|---|---|
| `MODEL` | `gemini-3-flash-preview` | Gemini variant |
| `MAX_FRAMES` | 512 | Frames per single API call |
| `MAX_ROUNDS` | 3 | Agent rounds before forced answer |
| `TOTAL_FRAME_BUDGET` | 1024 | Total frames the agent may retrieve across rounds |

## Files

| File | Purpose |
|---|---|
| `eval_agentic_gemini.py` | Per-question driver — calls the agent loop and writes results |
| `agent_loop.py` | Multi-round retrieval + reason loop |
| `prompts.py` | Agent system prompt + tool descriptions |
| `frame_window.py` | Time-window expansion under a frame budget |
| `run_avp_final_benchmark.sh` | Canonical sharded run |
