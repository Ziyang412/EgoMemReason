# Ego-R1

Inference wiring for the **Ego-R1** reasoning agent on EgoMemReason.

We do **not** ship the agent / training code — install the upstream:

```bash
git clone https://github.com/egolife-ntu/Ego-R1
# Follow Ego-R1's README to set up the agent + LLaMA-Factory backbone.
```

Then drop our `eval/` and `cott_gen/` folders into the cloned repo (or set `PYTHONPATH` to include them).

## Run

```bash
cd /path/to/Ego-R1/Ego-R1-Agent
INPUT_JSON=$EGOMEM_DATA bash <this-folder>/eval/infer_egolife_bench.sh
```

## Files

| Folder / file | Purpose |
|---|---|
| `eval/infer_egolife_bench.py` | EgoMemReason inference driver |
| `eval/infer_egolife_bench.sh` | Canonical run |
| `eval/infer.py` | Generic inference helper |
| `cott_gen/` | Chain-of-tool-thought generation utilities (used by Ego-R1's reasoning loop) |
