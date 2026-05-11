#!/usr/bin/env python3
"""Load one full copy of Qwen3-VL-8B on a single GPU and idle.

Launched per-GPU by hold_gpus.sh via CUDA_VISIBLE_DEVICES=<n>.
"""

import os
import time
import torch
import transformers
from transformers import AutoProcessor

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

print(f"[GPU {gpu_tag}] visible devices: {torch.cuda.device_count()}")

model_class = None
for cls_name in ("Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration",
                 "Qwen2VLForConditionalGeneration", "AutoModelForImageTextToText"):
    model_class = getattr(transformers, cls_name, None)
    if model_class is not None:
        break
if model_class is None:
    raise RuntimeError("No suitable model class found")

print(f"[GPU {gpu_tag}] loading {MODEL_NAME} via {model_class.__name__}...")
model = model_class.from_pretrained(MODEL_NAME, dtype="auto", device_map="cuda:0")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print(f"[GPU {gpu_tag}] model loaded. Idling. Ctrl+C to release.")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print(f"[GPU {gpu_tag}] releasing.")
