# SiLVR

Caption-then-reason agent. Generates dense captions over the week-long video, then runs an LLM over the captions to answer the question.

Upstream: https://github.com/CeeZh/SILVR

## Install

```bash
git clone https://github.com/CeeZh/SILVR
cd SILVR && pip install -r requirements.txt   # we ship a copy of requirements.txt here
```

You'll also need an API key for the captioner (e.g., GPT-5 in our setup) and the reasoning LLM.

## Run

```bash
# We use GPT-5 captions in our paper run
export OPENAI_API_KEY=...
INPUT_JSON=$EGOMEM_DATA \
  bash scripts/run_silvr_final_benchmark.sh
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Top-level driver |
| `dataset.py` | EgoMemReason dataset adapter |
| `model.py` | Captioner + reasoner wiring |
| `prompts.py` | Caption + reasoning prompts |
| `utils.py` | Helpers |
| `requirements.txt` | Python deps (mirrors upstream) |
| `eval/egolife.py` | EgoMemReason scorer |
| `scripts/run_silvr_final_benchmark.sh` | Canonical run (GPT-5 captions) |
