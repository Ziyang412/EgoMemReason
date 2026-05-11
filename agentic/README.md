# Agentic Video Frameworks

Methods that perform multi-round retrieval + reasoning over the week-long video, rather than a single forward pass through an MLLM.

| Folder | Paper row | Upstream | Notes |
|---|---|---|---|
| `AVP/` | **AVP (ours)** | This repo | Agentic Video Pipeline — Gemini-3-Flash backbone, 3 retrieval rounds, 1024-frame budget |
| `EgoR1/` | Ego-R1 | https://github.com/egolife-ntu/Ego-R1 | Ships the inference scripts; install the agent + LLaMA-Factory backbone separately |
| `SILVR/` | SiLVR | https://github.com/CeeZh/SILVR | Caption-and-reason agent |
| `WorldMM/` | WorldMM | https://github.com/Lyumumumu/WorldMM | Hippo-RAG retrieval + LLM reasoning |

## AVP (ours)

`AVP/` is our agentic baseline. It maintains a **frame budget** across multiple Gemini rounds: each round, the model decides which time windows to expand, retrieves dense frames in those windows, and updates its hypothesis until the budget is exhausted or it commits to an answer.

```bash
cd agentic/AVP
export GOOGLE_API_KEY=...
INPUT_JSON=$EGOMEM_DATA \
FRAMES_INDEX=$EGOLIFE_FRAMES_INDEX \
bash run_avp_final_benchmark.sh 20    # 20 parallel jobs
```

Key knobs (from `run_avp_final_benchmark.sh`):

- `MODEL` — Gemini variant (`gemini-3-flash-preview` by default)
- `MAX_FRAMES` — frames per single API call (512)
- `MAX_ROUNDS` — agent rounds before forced answer (3)
- `TOTAL_FRAME_BUDGET` — total frames the agent may retrieve across rounds (1024)

## Third-party agentic methods

For Ego-R1, SiLVR, WorldMM we ship **only the wiring needed to run them on EgoMemReason** — the agent / model / retrieval-DB code lives in their respective upstreams. Each subfolder's README has the install snippet and the env vars our run script expects.
