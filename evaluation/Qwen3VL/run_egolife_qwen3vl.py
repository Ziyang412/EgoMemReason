#!/usr/bin/env python3
"""Qwen3-VL evaluation with ablation support: prompt strategies, captions, transcripts, text-only.

Extended from run_egolife_qwen3vl8b_image.py with additional flags:
  --prompt_strategy {direct,cot,icl}
  --caption_root        Path to WorldMM_caption dir
  --caption_duration    Caption granularity: 30sec, 3min, 10min, 1h
  --transcript_root     Path to EgoLifeCap/Transcript dir
  --max_context_chars   Max chars for injected caption+transcript text
  --text_only           Skip frames, text-only evaluation
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

# Add parent paths for imports
sys.path.insert(0, "/nas-ssd2/ziyang/Memory_project/openai/eval")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

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
# Answer extraction
# ---------------------------------------------------------------------------

def parse_choice(raw: str, options: dict[str, str]) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    if text in options:
        return text

    # Prefer explicit "Answer: X" / "option: X" / "choice: X" (first occurrence wins).
    matches = re.findall(r"(?:option|answer|choice)\s*[:\-]?\s*\(?\s*([A-L])\s*\)?", text, flags=re.IGNORECASE)
    for letter in matches:
        letter = letter.upper()
        if letter in options:
            return letter

    # Fallback: first standalone letter token in the text.
    matches = re.findall(r"\b([A-L])\b", text, flags=re.IGNORECASE)
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
# Prompt strategies
# ---------------------------------------------------------------------------

ICL_EXAMPLES = """Example 1:
Question: What color is the mug on the kitchen counter?
A. Red
B. Blue
C. White
D. Green
Answer: C

Example 2:
Question: How many people are seated at the dining table?
A. 2
B. 3
C. 4
D. 5
Answer: B

Now answer the following:
"""


def build_prompt_direct(item: dict[str, Any], options: dict[str, str], context_text: str) -> str:
    """Baseline prompt: answer with the option letter followed by an explanation."""
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


def build_prompt_cot(item: dict[str, Any], options: dict[str, str], context_text: str) -> str:
    """Chain-of-thought prompt: reason first, then answer."""
    query = build_query_from_item(item)
    parts = []
    if options:
        parts.append(
            "You are an image QA system. Review the ordered image frames carefully "
            "and answer the multiple-choice question.\n"
            "Let's think step by step. First, describe what you observe in the frames "
            "relevant to the question. Then reason about the answer. "
            "Finally, state your answer as: Answer: [LETTER]"
        )
    else:
        parts.append(
            "You are an image QA system. Review the ordered image frames carefully.\n"
            "Let's think step by step. Reason about the answer, then give a concise final answer."
        )
    if context_text:
        parts.append("")
        parts.append(context_text)
    parts.append("")
    parts.append(query)
    return "\n".join(parts)


def build_prompt_icl(item: dict[str, Any], options: dict[str, str], context_text: str) -> str:
    """In-context learning prompt: 2 examples before the question."""
    query = build_query_from_item(item)
    parts = []
    if options:
        parts.append(
            "You are an image QA system. Review the ordered image frames carefully "
            "and answer the multiple-choice question.\n"
            "Return ONLY the option letter with no explanation."
        )
    else:
        parts.append(
            "You are an image QA system. Review the ordered image frames carefully "
            "and answer the question concisely."
        )
    if context_text:
        parts.append("")
        parts.append(context_text)
    parts.append("")
    parts.append(ICL_EXAMPLES)
    parts.append(query)
    return "\n".join(parts)


def build_prompt_icl_cot(item: dict[str, Any], options: dict[str, str], context_text: str) -> str:
    """In-context examples + chain-of-thought: examples first, then reason step by step."""
    query = build_query_from_item(item)
    parts = []
    if options:
        parts.append(
            "You are an image QA system. Review the ordered image frames carefully "
            "and answer the multiple-choice question.\n"
            "Let's think step by step. First, describe what you observe in the frames "
            "relevant to the question. Then reason about the answer. "
            "Finally, state your answer as: Answer: [LETTER]"
        )
    else:
        parts.append(
            "You are an image QA system. Review the ordered image frames carefully.\n"
            "Let's think step by step. Reason about the answer, then give a concise final answer."
        )
    if context_text:
        parts.append("")
        parts.append(context_text)
    parts.append("")
    parts.append(ICL_EXAMPLES)
    parts.append(query)
    return "\n".join(parts)


def build_prompt_text_only(item: dict[str, Any], options: dict[str, str], context_text: str) -> str:
    """Text-only prompt (no frames) with metadata and context."""
    lines: list[str] = []
    if options:
        labels = ", ".join(options.keys())
        lines.append("You are solving a multiple-choice benchmark query.")
        lines.append(f"Return only one option label from: {labels}. Do not output anything else.")
    else:
        lines.append("You are solving a benchmark query.")
        lines.append("Answer concisely and directly.")
    lines.append("")

    query_type = item.get("query_type") or item.get("question_type") or item.get("type")
    question = item.get("question") or item.get("question_text") or item.get("query") or ""
    if query_type:
        lines.append(f"Query type: {query_type}")
    if question:
        lines.append("Question:")
        lines.append(question)
        lines.append("")

    # Context metadata
    context_lines: list[str] = []
    for key in ("query_time", "target_time", "identity", "video_id", "p_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            context_lines.append(f"- {key}: {value}")
    if context_lines:
        lines.append("Context:")
        lines.extend(context_lines)
        lines.append("")

    # Injected captions/transcripts
    if context_text:
        lines.append(context_text)
        lines.append("")

    # Events
    events = item.get("events")
    if isinstance(events, dict) and events:
        lines.append("Events:")
        for idx, (k, v) in enumerate(sorted(events.items()), 1):
            lines.append(f"{idx}. {v}")
        lines.append("")

    # Options
    if options:
        lines.append("Options:")
        for label, text in options.items():
            lines.append(f"{label}. {text}")
        lines.append("")

    return "\n".join(lines).strip()


PROMPT_BUILDERS = {
    "direct": build_prompt_direct,
    "cot": build_prompt_cot,
    "icl": build_prompt_icl,
    "icl_cot": build_prompt_icl_cot,
}


def build_chunk_summary_prompt(question: str) -> str:
    return (
        "You are reviewing an ordered subset of image frames from a longer egocentric sequence.\n"
        "Summarize only the evidence relevant to answering the question below.\n"
        "Focus on object state, location, interaction, count changes, and temporal ordering.\n"
        "Keep the summary concise and factual.\n\n"
        f"Question:\n{question}"
    )


def build_final_prompt_from_summaries(item: dict[str, Any], options: dict[str, str], summaries: list[str]) -> str:
    summary_blocks = []
    for idx, text in enumerate(summaries, start=1):
        summary_blocks.append(f"Chunk {idx} summary:\n{text}")
    joined = "\n\n".join(summary_blocks)

    if options:
        query = build_query_from_item(item)
        return (
            "You are answering a multiple-choice question using chunk summaries from a long sequence of image frames.\n"
            "Use the summaries as evidence. Return ONLY the option letter with no explanation.\n\n"
            f"{joined}\n\nQuestion and options:\n{query}"
        )
    return (
        "You are answering a question using chunk summaries from a long sequence of image frames.\n"
        "Use the summaries as evidence and answer concisely.\n\n"
        f"{joined}\n\nQuestion:\n{item['question']}"
    )


# ---------------------------------------------------------------------------
# Inference helpers
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


def chunk_list(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def compute_accuracy(results: list[dict[str, Any]]) -> tuple[int, int, float | None]:
    valid = [row for row in results if row.get("pred") is not None and row.get("answer") is not None]
    correct = [row for row in valid if row.get("correct") is True]
    accuracy = (len(correct) / len(valid)) if valid else None
    return len(correct), len(valid), accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL ablation evaluation.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to QA JSON")
    parser.add_argument("--output", type=str, required=True, help="Output path for result JSON list")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")

    # Frame options
    parser.add_argument("--egolife_frame_index_dir", type=str, default=None)
    parser.add_argument("--image_list_dir", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_hours_before_target", type=float, default=None)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)
    parser.add_argument("--vision_batch_size", type=int, default=0,
        help="If >0 and frame count exceeds this, split into batches and summarize.")
    parser.add_argument("--text_only", action="store_true",
        help="Text-only evaluation, skip frames entirely.")

    # Prompt strategy
    parser.add_argument("--prompt_strategy", type=str, default="direct",
        choices=["direct", "cot", "icl", "icl_cot"],
        help="Prompt strategy: direct (baseline), cot (chain-of-thought), icl (in-context examples)")

    # Context injection
    parser.add_argument("--caption_root", type=str, default=None,
        help="Path to WorldMM_caption root dir for caption injection")
    parser.add_argument("--caption_duration", type=str, default="3min",
        choices=["30sec", "3min", "10min", "1h"],
        help="Caption granularity to use")
    parser.add_argument("--transcript_root", type=str, default=None,
        help="Path to EgoLifeCap/Transcript root dir for transcript injection")
    parser.add_argument("--max_context_chars", type=int, default=4000,
        help="Max chars for injected caption+transcript text")

    # General
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
    parser.add_argument("--out_seq_length", type=int, default=env_get_int("out_seq_length", 64))
    args = parser.parse_args()

    # Validate args
    if not args.text_only and not (args.image_list_dir or args.egolife_frame_index_dir):
        parser.error("One of --image_list_dir, --egolife_frame_index_dir, or --text_only is required")

    # Auto-bump out_seq_length for CoT
    if args.prompt_strategy in ("cot", "icl_cot") and args.out_seq_length < 8192:
        print(f"[ablation] CoT mode: bumping out_seq_length from {args.out_seq_length} to 8192")
        args.out_seq_length = 8192

    # Load dataset
    with open(args.dataset, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list")

    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(data)
    if args.limit is not None:
        data = data[:args.limit]

    dataset_base = os.path.basename(args.dataset)
    default_video_id = None
    match = re.search(r"EgoLifeQA_(.+)\.json$", dataset_base)
    if match:
        default_video_id = match.group(1)
    else:
        match = re.search(r"^(.+)\.json$", dataset_base)
        if match and IDENTITY_FILENAME_PATTERN.match(match.group(1)):
            default_video_id = match.group(1)

    # Load model
    greedy_bool = str(args.greedy).strip().lower() in {"1", "true", "yes", "y", "t"}
    dtype_map: dict[str, Any] = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    # Auto-detect model class
    import transformers
    model_class = None
    for class_name in (
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            break
    if model_class is None:
        raise RuntimeError("No suitable model class found in transformers")

    load_kwargs: dict[str, Any] = {
        "dtype": dtype_map[args.dtype],
        "device_map": "auto",
    }
    if args.flash_attn2:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"[ablation] Loading {args.model_name} via {model_class.__name__}...")
    model = model_class.from_pretrained(args.model_name, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_name)

    print(f"[ablation] Config: prompt_strategy={args.prompt_strategy}, text_only={args.text_only}, "
          f"max_frames={args.max_frames}, caption_root={args.caption_root}, "
          f"transcript_root={args.transcript_root}, out_seq_length={args.out_seq_length}")

    # Evaluate
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

        # Get image paths (skip if text_only)
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

        # Build context text from captions/transcripts
        context_text = get_context_text(
            raw_item,
            caption_root=args.caption_root,
            caption_duration=args.caption_duration,
            transcript_root=args.transcript_root,
            max_context_chars=args.max_context_chars,
        )

        options = normalize_options(item)

        # Build prompt based on strategy
        if args.text_only:
            prompt = build_prompt_text_only(item, options, context_text)
        else:
            prompt_builder = PROMPT_BUILDERS[args.prompt_strategy]
            prompt = prompt_builder(item, options, context_text)

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
                # Text-only inference
                raw_output = infer_text_only(
                    model=model, processor=processor, prompt_text=prompt,
                    max_new_tokens=args.out_seq_length, greedy=greedy_bool,
                    temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                )
            elif args.vision_batch_size and len(image_paths) > args.vision_batch_size:
                # Chunked vision + summarize
                summaries: list[str] = []
                for batch_paths in chunk_list(image_paths, args.vision_batch_size):
                    summary_prompt = build_chunk_summary_prompt(item["question"])
                    summary_text = infer_one(
                        model=model, processor=processor, image_paths=batch_paths,
                        prompt_text=summary_prompt, max_new_tokens=args.out_seq_length,
                        greedy=greedy_bool, temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        max_pixels=args.max_pixels,
                    )
                    summaries.append(summary_text.strip())
                final_prompt = build_final_prompt_from_summaries(item, options, summaries)
                raw_output = infer_text_only(
                    model=model, processor=processor, prompt_text=final_prompt,
                    max_new_tokens=args.out_seq_length, greedy=greedy_bool,
                    temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                )
                out_item["chunk_summary_count"] = len(summaries)
            else:
                # Standard vision inference
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

        if (i + 1) % 10 == 0:
            correct, valid, accuracy = compute_accuracy(results)
            print(f"[{i+1}/{len(data)}] acc_valid={accuracy} (correct={correct}, valid={valid})")

        if args.save_every and args.save_every > 0 and (i + 1) % args.save_every == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    correct, valid, accuracy = compute_accuracy(results)
    print(f"Done. Wrote {len(results)} rows to {args.output}. "
          f"acc_valid={accuracy} (correct={correct}, valid={valid})")


if __name__ == "__main__":
    main()
