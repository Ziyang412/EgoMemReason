"""Load one full copy of InternVL3.5-8B on a single GPU and idle."""
import os
import time
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ.get("INTERNVL_MODEL_ID", "OpenGVLab/InternVL3_5-8B")
DTYPE_STR = os.environ.get("INTERNVL_DTYPE", "bf16")
gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
dtype = dtype_map.get(DTYPE_STR, torch.bfloat16)

print(f"[GPU {gpu_tag}] visible devices: {torch.cuda.device_count()}", flush=True)
print(f"[GPU {gpu_tag}] loading {MODEL_NAME} (dtype={DTYPE_STR})...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="cuda:0",
).eval()

print(f"[GPU {gpu_tag}] model loaded. Idling. Ctrl+C to release.", flush=True)
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print(f"[GPU {gpu_tag}] releasing.", flush=True)
