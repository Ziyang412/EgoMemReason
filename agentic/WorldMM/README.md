# WorldMM

Hippo-RAG retrieval over a memory graph + LLM reasoning.

Upstream: https://github.com/Lyumumumu/WorldMM (also depends on **HippoRAG** — install separately).

## Install

```bash
# Upstream WorldMM
git clone https://github.com/Lyumumumu/WorldMM
cd WorldMM && pip install -e .

# HippoRAG dependency (vendored upstream — not shipped here)
pip install hipporag
```

## Build the memory graph

```bash
python preprocess/build_memory.py \
  --frames_root /path/to/egolife/frames \
  --out /path/to/h-rag_database/egolife
```

This produces a per-identity memory database that the eval script reads at retrieval time. Building takes hours; cache it.

## Run

```bash
INPUT_JSON=$EGOMEM_DATA \
WORLDMM_DB=/path/to/h-rag_database/egolife \
  bash script/run_worldmm_final_benchmark.sh
```

## Files

| File / folder | Purpose |
|---|---|
| `eval/eval_benchmark.py` | Generic benchmark scorer |
| `eval/eval_egolife.py` | EgoMemReason-specific eval |
| `script/run_worldmm_final_benchmark.sh` | Canonical run |
| `src/worldmm/` | WorldMM retrieval + reasoning code (your local additions) |
| `preprocess/build_memory.py` | Builds the per-identity memory graph |
| `pyproject.toml` | Package metadata |
