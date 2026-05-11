#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


# ---------------------------------------------------------------------------
# Query time parsing
# ---------------------------------------------------------------------------

def parse_query_time(query_time_str: str) -> tuple[str, int] | None:
    """Parse 'DAY6, 18:30:00' -> ('DAY6', 18300000) i.e. (day_str, 8-digit HHMMSSHH)."""
    if not query_time_str or "," not in query_time_str:
        return None
    parts = query_time_str.split(",", 1)
    day = parts[0].strip()
    time_part = parts[1].strip()
    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})", time_part)
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    time_int = h * 1000000 + mi * 10000 + s * 100
    return day, time_int


def _parse_day_str(s: str) -> int | None:
    """Parse 'DAY1' / 'day1' -> 1."""
    if not s:
        return None
    s = str(s).strip().upper().replace("DAY", "")
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Frames index loading & filtering
# ---------------------------------------------------------------------------

def load_frames_index(index_path: str, cache: dict) -> dict:
    """Load single-file frames index (identity -> list of entries). Cached after first load."""
    if "data" not in cache:
        with open(index_path, "r") as f:
            cache["data"] = json.load(f)
    return cache["data"]


def get_frames_before_target(
    index_data: dict,
    identity: str,
    day_str: str,
    target_time_int: int,
    max_frames: int,
) -> list[str]:
    """Filter frames <= target time for an identity, then uniform sample up to max_frames."""
    entries = index_data.get(identity)
    if not entries:
        # Try case variations
        for key in index_data:
            if key.upper() == identity.upper():
                entries = index_data[key]
                break
    if not entries:
        return []

    target_day_num = _parse_day_str(day_str)
    if target_day_num is None:
        return []

    before = []
    for ent in entries:
        frame_day_num = _parse_day_str(ent.get("day", ""))
        if frame_day_num is None:
            continue
        frame_time = ent.get("time")
        if frame_time is None:
            continue
        frame_time = int(frame_time)

        if frame_day_num < target_day_num:
            before.append(ent["path"])
        elif frame_day_num == target_day_num and frame_time <= target_time_int:
            before.append(ent["path"])

    if not before:
        return []
    if len(before) <= max_frames:
        return before
    indices = np.linspace(0, len(before) - 1, max_frames, dtype=int)
    return [before[i] for i in indices]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(question: str, options: dict[str, str]) -> str:
    """Build MC prompt from question + dict options {A: text, B: text, ...}."""
    option_letters = sorted(options.keys())
    option_lines = [f"{k}. {options[k]}" for k in option_letters]
    letter_list = ", ".join(option_letters)
    return (
        "You are an image QA system. Review the ordered image frames carefully "
        "and answer the multiple-choice question.\n"
        f"Return ONLY the option letter (for example: {letter_list}) with no explanation.\n\n"
        f"{question}\n\n"
        + "\n".join(option_lines)
    )


# ---------------------------------------------------------------------------
# Choice parsing
# ---------------------------------------------------------------------------

def parse_choice(raw: str, options: dict[str, str]) -> str | None:
    """Extract an option letter (A-H) from model output."""
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # Exact match
    if text in options:
        return text

    # Look for standalone letters A-H
    matches = re.findall(r"\b([A-H])\b", text, flags=re.IGNORECASE)
    for letter in reversed(matches):
        letter = letter.upper()
        if letter in options:
            return letter

    # "option/answer: X" pattern
    matches = re.findall(
        r"(?:option|answer)\s*[:\-]?\s*\(?\s*([A-H])\s*\)?",
        text, flags=re.IGNORECASE,
    )
    for letter in reversed(matches):
        letter = letter.upper()
        if letter in options:
            return letter

    # Fuzzy value match
    normalized = re.sub(r"[\s\.\,\!\?\:\;\(\)\[\]\{\}\"']", "", text).lower()
    for key, value in options.items():
        value_normalized = re.sub(
            r"[\s\.\,\!\?\:\;\(\)\[\]\{\}\"']", "", str(value)
        ).lower()
        if value_normalized and value_normalized in normalized:
            return key

    return None


# ---------------------------------------------------------------------------
# VideoLLaMA3 inference
# ---------------------------------------------------------------------------

def infer_one(
    model,
    processor,
    image_paths: list[str],
    prompt_text: str,
    max_new_tokens: int,
    device: str,
) -> str:
    """Run VideoLLaMA3 inference with multiple images + text prompt."""
    content: list[dict[str, Any]] = []
    for path in image_paths:
        content.append({"type": "image", "image": {"image_path": path}})
    content.append({"type": "text", "text": prompt_text})

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(results: list[dict]) -> tuple[int, int, float | None]:
    valid = [r for r in results if r.get("pred") is not None and r.get("correct_answer") is not None]
    correct = [r for r in valid if r.get("correct") is True]
    acc = (len(correct) / len(valid)) if valid else None
    return len(correct), len(valid), acc


def print_per_type_accuracy(results: list[dict]) -> None:
    """Print accuracy breakdown per query_type."""
    from collections import defaultdict
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        qt = r.get("query_type", "unknown")
        by_type[qt].append(r)

    print("\n--- Per query_type accuracy ---")
    for qt in sorted(by_type.keys()):
        items = by_type[qt]
        c, v, a = compute_accuracy(items)
        acc_str = f"{a:.4f}" if a is not None else "N/A"
        print(f"  {qt}: {acc_str} ({c}/{v})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VideoLLaMA3 on EgoLife benchmark with pre-extracted frames."
    )
    parser.add_argument(
        "--dataset", type=str,
        default="/nas-ssd2/ziyang/Memory_project/COLM/benchmark/filtered_batch_1/all_task_types_v2.json",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--model_path", type=str, default="DAMO-NLP-SG/VideoLLaMA3-7B",
    )
    parser.add_argument(
        "--frames_index", type=str,
        default="/nas-ssd2/video_datasets/EgoLife/egolife_frames_index.json",
    )
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None, help="Chunk index (0-based) for parallel runs")
    parser.add_argument("--num_chunks", type=int, default=None, help="Total number of chunks for parallel runs")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--print_each", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Load benchmark
    with open(args.dataset, "r") as f:
        data = json.load(f)

    # Unwrap if needed
    if isinstance(data, dict):
        if "samples" in data:
            data = data["samples"]
        elif "data" in data:
            data = data["data"]
        elif "items" in data:
            data = data["items"]
        else:
            raise ValueError(f"Cannot find sample list in dataset keys: {list(data.keys())}")

    if not isinstance(data, list):
        raise ValueError("Dataset samples must be a list")

    if args.limit is not None:
        data = data[: args.limit]

    # Chunk splitting for parallel runs
    if args.chunk is not None and args.num_chunks is not None:
        total = len(data)
        chunk_size = (total + args.num_chunks - 1) // args.num_chunks
        start = args.chunk * chunk_size
        end = min(start + chunk_size, total)
        data = data[start:end]
        print(f"Chunk {args.chunk}/{args.num_chunks}: samples {start}-{end-1} ({len(data)} items)")

    print(f"Loaded {len(data)} samples from {args.dataset}")

    # Load model
    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map={"": args.device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print("Model loaded.")

    # Frames index cache
    index_cache: dict = {}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results: list[dict[str, Any]] = []

    for i, item in enumerate(tqdm(data, desc="Evaluating")):
        identity = item.get("identity", "")
        query_time_str = item.get("query_time", "")
        question = item.get("question", "")
        options = item.get("options", {})
        correct_answer = item.get("correct_answer")
        example_id = item.get("example_id", "")
        query_type = item.get("query_type", "")

        out = {
            "example_id": example_id,
            "identity": identity,
            "query_type": query_type,
            "question": question,
            "correct_answer": correct_answer,
            "pred": None,
            "correct": None,
            "raw_output": None,
            "error": None,
            "latency_s": None,
            "num_frames": 0,
        }

        try:
            # Parse query time
            parsed = parse_query_time(query_time_str)
            if parsed is None:
                raise ValueError(f"Cannot parse query_time: {query_time_str!r}")
            day_str, target_time_int = parsed

            # Get frames
            index_data = load_frames_index(args.frames_index, index_cache)
            image_paths = get_frames_before_target(
                index_data, identity, day_str, target_time_int, args.max_frames,
            )
            out["num_frames"] = len(image_paths)

            if not image_paths:
                raise ValueError(f"No frames found for {identity} before {query_time_str}")

            # Build prompt
            prompt = build_prompt(question, options)

            # Inference
            t0 = time.time()
            raw_output = infer_one(
                model, processor, image_paths, prompt, args.max_new_tokens, args.device,
            )
            out["latency_s"] = time.time() - t0
            out["raw_output"] = raw_output

            # Parse choice
            pred = parse_choice(raw_output, options)
            out["pred"] = pred
            if pred is not None and correct_answer is not None:
                out["correct"] = pred == correct_answer

            if args.print_each:
                lat = f"{out['latency_s']:.2f}s" if out["latency_s"] else "NA"
                print(
                    f"[{i+1}/{len(data)}] frames={out['num_frames']} "
                    f"latency={lat} pred={pred} gt={correct_answer} "
                    f"correct={out['correct']}"
                )

        except Exception as exc:
            out["error"] = repr(exc)
            if args.print_each:
                print(f"[{i+1}/{len(data)}] ERROR: {out['error']}")

        results.append(out)

        # Running accuracy
        if (i + 1) % 10 == 0:
            c, v, a = compute_accuracy(results)
            acc_str = f"{a:.4f}" if a is not None else "N/A"
            print(f"[{i+1}/{len(data)}] running acc={acc_str} ({c}/{v})")

        # Save checkpoint
        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    c, v, a = compute_accuracy(results)
    acc_str = f"{a:.4f}" if a is not None else "N/A"
    print(f"\nDone. {len(results)} samples -> {args.output}")
    print(f"Overall accuracy: {acc_str} ({c}/{v})")
    print_per_type_accuracy(results)


if __name__ == "__main__":
    main()
