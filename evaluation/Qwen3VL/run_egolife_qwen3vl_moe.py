#!/usr/bin/env python3
"""Qwen3-VL MoE evaluation (30B-A3B / 235B-A22B).

Identical to run_egolife_qwen3vl_ablation.py except for the model class:
uses Qwen3VLMoeForConditionalGeneration. All prompt building, vision input
handling (via qwen_vl_utils.process_vision_info), and parsing logic are
reused verbatim.
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

# Reduce fragmentation BEFORE importing torch / loading CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm

sys.path.insert(0, "/nas-ssd2/ziyang/Memory_project/openai/eval")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from gemini_pure_image import (
    build_query_from_item,
    get_egolife_frames_before_target_from_index,
    get_image_list_for_video,
    normalize_item,
)
from context_retrieval import get_context_text


def env_get(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else value


def env_get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


IDENTITY_FILENAME_PATTERN = re.compile(r"^[A-Z]\d+_[A-Za-z]+$")


# ---------------------------------------------------------------------------
# Answer extraction (same as ablation script)
# ---------------------------------------------------------------------------

def parse_choice(raw: str, options: dict[str, str]) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    if text in options:
        return text

    matches = re.findall(r"(?:option|answer|choice)\s*[:\-]?\s*\(?\s*([A-H])\s*\)?", text, flags=re.IGNORECASE)
    for letter in matches:
        letter = letter.upper()
        if letter in options:
            return letter

    matches = re.findall(r"\b([A-H])\b", text, flags=re.IGNORECASE)
    for letter in matches:
        letter = letter.upper()
        if letter in options:
            return letter

    normalized = re.sub(r"[\s\.\,\!\?\:\;\(\)\[\]\{\}\"']", "", text).lower()
    for key, value in options.items():
        value_normalized = re.sub(r"[\s\.\,\!\?\:\;\(\)\[\]\{\}\"']", "", str(value)).lower()
        if value_normalized and value_normalized in normalized:
            return key

    return None


def normalize_options(item: dict[str, Any]) -> dict[str, str]:
    if "options" in item and isinstance(item["options"], list):
        return {
            str(opt["id"]): str(opt["text"])
            for opt in item["options"]
            if isinstance(opt, dict) and "id" in opt and "text" in opt
        }
    if "choices" in item and isinstance(item["choices"], list):
        return {
            str(opt["label"]): str(opt["text"])
            for opt in item["choices"]
            if isinstance(opt, dict) and "label" in opt and "text" in opt
        }
    options: dict[str, str] = {}
    for letter, key in [("A", "choice_a"), ("B", "choice_b"), ("C", "choice_c"), ("D", "choice_d"), ("E", "choice_e")]:
        if item.get(key):
            options[letter] = str(item[key])
    return options


def extract_eval_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if item.get("choices") is not None:
        metadata["choices"] = item.get("choices")
    elif item.get("options") is not None:
        metadata["options"] = item.get("options")
    for key in (
        "query_type", "difficulty_score", "difficulty_tier", "split",
        "object_name", "combined_sample_id",
    ):
        if item.get(key) is not None:
            metadata[key] = item.get(key)
    return metadata


# ---------------------------------------------------------------------------
# Prompt (same as ablation script: direct -> Answer + explanation)
# ---------------------------------------------------------------------------

def build_prompt_direct(item: dict[str, Any], options: dict[str, str], context_text: str) -> str:
    query = build_query_from_item(item)
    parts = []
    if options:
        parts.append(
            "You are reviewing a week-long video log, presented as an ordered sequence of image frames. "
            "Review the frames carefully and answer the multiple-choice question.\n"
            "First state your answer as: Answer: [LETTER]\n"
            "Then provide a brief explanation of your reasoning on the next lines."
        )
    else:
        parts.append(
            "You are reviewing a week-long video log, presented as an ordered sequence of image frames. "
            "Review the frames carefully and answer the question. Provide the answer first, then a brief explanation."
        )
    if context_text:
        parts.append("")
        parts.append(context_text)
    parts.append("")
    parts.append(query)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Inference helpers (same as ablation script: process_vision_info + processor)
# ---------------------------------------------------------------------------

def infer_one(
    model, processor, image_paths: list[str], prompt_text: str,
    max_new_tokens: int, greedy: bool, temperature: float,
    top_p: float, top_k: int, repetition_penalty: float, max_pixels: int,
) -> str:
    if not image_paths:
        raise ValueError("image_paths must be non-empty")
    content: list[dict[str, Any]] = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        content.append({
            "type": "image",
            "image": f"file://{os.path.abspath(path)}",
            "max_pixels": int(max_pixels),
        })
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
    }
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k)})

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0] if output_text else ""


def infer_text_only(
    model, processor, prompt_text: str,
    max_new_tokens: int, greedy: bool, temperature: float,
    top_p: float, top_k: int, repetition_penalty: float,
) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
    }
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k)})

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0] if output_text else ""


def compute_accuracy(results):
    valid = [row for row in results if row.get("pred") is not None and row.get("answer") is not None]
    correct = [row for row in valid if row.get("correct") is True]
    accuracy = (len(correct) / len(valid)) if valid else None
    return len(correct), len(valid), accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL MoE evaluation.")
    parser.add_argument("--dataset", type=str, required=True, help="Flat JSON list")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")

    parser.add_argument("--egolife_frame_index_dir", type=str, default=None)
    parser.add_argument("--image_list_dir", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_hours_before_target", type=float, default=None)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)
    parser.add_argument("--text_only", action="store_true")

    parser.add_argument("--caption_root", type=str, default=None)
    parser.add_argument("--caption_duration", type=str, default="3min",
        choices=["30sec", "3min", "10min", "1h"])
    parser.add_argument("--transcript_root", type=str, default=None)
    parser.add_argument("--max_context_chars", type=int, default=4000)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--print_each", action="store_true")
    parser.add_argument("--flash_attn2", action="store_true")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--greedy", type=str, default=env_get("greedy", "true"))
    parser.add_argument("--top_p", type=float, default=env_get_float("top_p", 0.9))
    parser.add_argument("--top_k", type=int, default=env_get_int("top_k", 50))
    parser.add_argument("--temperature", type=float, default=env_get_float("temperature", 0.0))
    parser.add_argument("--repetition_penalty", type=float, default=env_get_float("repetition_penalty", 1.0))
    parser.add_argument("--out_seq_length", type=int, default=env_get_int("out_seq_length", 512))
    parser.add_argument("--device_map", type=str, default="auto",
        help="HF device_map: 'auto', 'balanced', 'balanced_low_0', or 'sequential'.")
    parser.add_argument("--max_gpu_mem", type=str, default=None,
        help="Per-GPU weight cap, e.g. '55GiB'. Leaves headroom for activations.")
    args = parser.parse_args()

    if not args.text_only and not (args.image_list_dir or args.egolife_frame_index_dir):
        parser.error("Provide --image_list_dir, --egolife_frame_index_dir, or --text_only")

    with open(args.dataset, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list")

    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(data)
    if args.limit is not None:
        data = data[: args.limit]

    dataset_base = os.path.basename(args.dataset)
    default_video_id = None
    match = re.search(r"EgoLifeQA_(.+)\.json$", dataset_base)
    if match:
        default_video_id = match.group(1)
    else:
        match = re.search(r"^(.+)\.json$", dataset_base)
        if match and IDENTITY_FILENAME_PATTERN.match(match.group(1)):
            default_video_id = match.group(1)

    greedy_bool = str(args.greedy).strip().lower() in {"1", "true", "yes", "y", "t"}
    dtype_map: dict[str, Any] = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    # --- ONLY change from the ablation script: load the MoE class directly. ---
    load_kwargs: dict[str, Any] = {
        "dtype": dtype_map[args.dtype],
        "device_map": args.device_map,
    }
    if args.flash_attn2:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    if args.max_gpu_mem:
        n_visible = torch.cuda.device_count()
        load_kwargs["max_memory"] = {i: args.max_gpu_mem for i in range(n_visible)}
        print(f"[moe] Capping each of {n_visible} GPUs to {args.max_gpu_mem} for weights.")

    print(f"[moe] Loading {args.model_name} via Qwen3VLMoeForConditionalGeneration...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(args.model_name, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_name)

    print(f"[moe] Config: text_only={args.text_only}, max_frames={args.max_frames}, "
          f"caption_root={args.caption_root}, transcript_root={args.transcript_root}, "
          f"out_seq_length={args.out_seq_length}")

    results: list[dict[str, Any]] = []
    image_paths_log: list[dict[str, Any]] = []
    base, _ = os.path.splitext(args.output)
    log_image_paths_path = f"{base}_image_paths.json"
    index_cache: dict[str, list] = {}

    for i, raw_item in enumerate(tqdm(data, desc="Evaluating", total=len(data))):
        item = normalize_item(raw_item, default_video_id=default_video_id)
        video_id = item.get("video_id") or item.get("identity") or default_video_id
        if not video_id:
            results.append({
                "sample_id": item.get("sample_id") or item.get("combined_sample_id"),
                "video_id": None, "question": item.get("question"),
                "answer": item.get("answer"), "pred": None, "correct": None,
                "raw_output": None, "error": "no video_id or identity",
            })
            continue

        image_paths: list[str] = []
        if not args.text_only:
            if args.egolife_frame_index_dir and item.get("target_time"):
                image_paths = get_egolife_frames_before_target_from_index(
                    args.egolife_frame_index_dir, video_id, item["target_time"],
                    max_frames=args.max_frames, index_cache=index_cache,
                    max_hours_before_target=args.max_hours_before_target,
                )
            if not image_paths and args.image_list_dir:
                image_paths = get_image_list_for_video(args.image_list_dir, video_id)

        if not args.text_only:
            image_paths_log.append({
                "index": i,
                "sample_id": item.get("sample_id") or item.get("combined_sample_id"),
                "video_id": video_id,
                "target_time": item.get("target_time"),
                "num_frames": len(image_paths),
            })
            if (i + 1) <= 5 or (i + 1) % 5 == 0:
                with open(log_image_paths_path, "w") as f:
                    json.dump(image_paths_log, f, indent=2)

        context_text = get_context_text(
            raw_item,
            caption_root=args.caption_root,
            caption_duration=args.caption_duration,
            transcript_root=args.transcript_root,
            max_context_chars=args.max_context_chars,
        )

        options = normalize_options(item)
        prompt = build_prompt_direct(item, options, context_text)

        out_item: dict[str, Any] = {
            "sample_id": item.get("sample_id") or item.get("combined_sample_id"),
            "video_id": video_id,
            "question": item.get("question"),
            "answer": item.get("answer"),
            "answer_text": item.get("answer_text"),
            "pred": None,
            "correct": None,
            "raw_output": None,
            "error": None,
            "latency_s": None,
        }
        out_item.update(extract_eval_metadata(item))

        try:
            t_start = time.time()
            if args.text_only or not image_paths:
                raw_output = infer_text_only(
                    model=model, processor=processor, prompt_text=prompt,
                    max_new_tokens=args.out_seq_length, greedy=greedy_bool,
                    temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                )
            else:
                raw_output = infer_one(
                    model=model, processor=processor, image_paths=image_paths,
                    prompt_text=prompt, max_new_tokens=args.out_seq_length,
                    greedy=greedy_bool, temperature=args.temperature,
                    top_p=args.top_p, top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    max_pixels=args.max_pixels,
                )

            out_item["latency_s"] = time.time() - t_start
            out_item["raw_output"] = raw_output

            if options:
                pred = parse_choice(raw_output, options)
                out_item["pred"] = pred
                if pred is not None and item.get("answer") is not None:
                    out_item["correct"] = (pred == item.get("answer"))
            else:
                out_item["pred"] = raw_output.strip()

            if args.print_each:
                lat = out_item["latency_s"]
                lat_str = f"{lat:.2f}s" if isinstance(lat, (int, float)) else "NA"
                print(f"[{i+1}/{len(data)}] latency={lat_str} pred={out_item['pred']} "
                      f"gt={out_item['answer']} correct={out_item['correct']}")

        except Exception as exc:
            out_item["error"] = repr(exc)
            if args.print_each:
                print(f"[{i+1}/{len(data)}] ERROR {video_id}: {out_item['error']}")

        results.append(out_item)

        # Free cached activations between items.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            correct, valid, accuracy = compute_accuracy(results)
            print(f"[{i+1}/{len(data)}] acc_valid={accuracy} (correct={correct}, valid={valid})")

        if args.save_every and args.save_every > 0 and (i + 1) % args.save_every == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    correct, valid, accuracy = compute_accuracy(results)
    print(f"\n[moe] Final: {correct}/{valid} = {accuracy}")


if __name__ == "__main__":
    main()
