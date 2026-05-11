"""Load one full copy of Molmo2-8B on a single GPU and idle."""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_NAME = os.environ.get("MOLMO2_MODEL_ID", "allenai/Molmo2-8B")
DTYPE_STR = os.environ.get("MOLMO2_DTYPE", "bf16")
gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "auto": "auto"}
dtype = dtype_map.get(DTYPE_STR, torch.bfloat16)

print(f"[GPU {gpu_tag}] visible devices: {torch.cuda.device_count()}", flush=True)
print(f"[GPU {gpu_tag}] loading {MODEL_NAME} (dtype={DTYPE_STR})...", flush=True)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="cuda:0",
)

print(f"[GPU {gpu_tag}] model loaded. Idling. Ctrl+C to release.", flush=True)
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print(f"[GPU {gpu_tag}] releasing.", flush=True)
