#!/usr/bin/env python3
"""
Evaluate Molmo2-8B on EgoLife-style videoQA benchmarks.

Loads pre-extracted 1FPS frames from a single JSON index file,
uniformly samples frames before (inclusive) the query time,
passes them as video input to Molmo2, and parses multiple-choice answers.

Ablation support: prompt strategies, captions, transcripts, text-only.
  --prompt_strategy {direct,cot,icl,icl_cot}
  --caption_root        Path to WorldMM_caption dir
  --caption_duration    Caption granularity: 30sec, 3min, 10min, 1h
  --transcript_root     Path to EgoLifeCap/Transcript dir
  --text_only           Text-only evaluation, skip frames entirely
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

# Import context retrieval (model-agnostic)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context_retrieval import get_context_text


# ---------------------------------------------------------------------------
# Time parsing helpers (from gemini_pure_image.py)
# ---------------------------------------------------------------------------

def _parse_day_str(s: str) -> int | None:
    if not s:
        return None
    s = str(s).strip().upper().replace("DAY", "")
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_time_str(s: str) -> int | None:
    s = str(s).strip()
    try:
        return int(s)
    except ValueError:
        pass
    match = re.search(r"\d{8}(?!\d)", s)
    return int(match.group(0)) if match else None


def parse_query_time_to_target(query_time: object) -> dict | None:
    """Parse query_time to {"date": "DAYx", "time": "HHMMSSHH"}."""
    if isinstance(query_time, dict):
        date = str(query_time.get("date") or "").strip()
        time_str = str(query_time.get("time") or "").strip()
        if date and time_str:
            if re.fullmatch(r"\d{8}", time_str):
                return {"date": date, "time": time_str}
            if re.fullmatch(r"\d{6}", time_str):
                return {"date": date, "time": f"{time_str}00"}
        return None

    if not isinstance(query_time, str):
        return None
    text = query_time.strip()

    # "DAY6, 18:30:00"
    m = re.match(r"^DAY(\d+)\s*,\s*(\d{1,2}):(\d{2}):(\d{2})$", text, flags=re.IGNORECASE)
    if m:
        day = int(m.group(1))
        hh = int(m.group(2))
        mm = int(m.group(3))
        ss = int(m.group(4))
        return {"date": f"DAY{day}", "time": f"{hh:02d}{mm:02d}{ss:02d}00"}

    # "DAY6_18300000"
    m = re.match(r"^DAY(\d+)[_ ](\d{6,8})$", text, flags=re.IGNORECASE)
    if m:
        day = int(m.group(1))
        raw = m.group(2)
        if len(raw) == 6:
            raw = f"{raw}00"
        return {"date": f"DAY{day}", "time": raw}

    return None


# ---------------------------------------------------------------------------
# Frame sampling from single JSON index
# ---------------------------------------------------------------------------

def get_frames_before_target(
    frame_index: dict[str, list],
    identity: str,
    target_time: dict,
    max_frames: int = 256,
) -> list[str]:
    """Get frame paths before (inclusive) target_time for identity, uniformly sampled."""
    entries = frame_index.get(identity)
    if not entries:
        return []

    target_date = (target_time.get("date") or "").strip()
    target_time_str = str(target_time.get("time") or "0").strip()
    target_day_num = _parse_day_str(target_date)
    target_time_int = _parse_time_str(target_time_str)
    if target_day_num is None:
        return []

    before: list[str] = []
    for ent in entries:
        day_str = ent.get("day") or ""
        day_num = _parse_day_str(day_str)
        if day_num is None:
            continue
        t = ent.get("time")
        if t is None:
            continue
        frame_time_int = int(t) if isinstance(t, int) else _parse_time_str(str(t))
        if frame_time_int is None:
            continue

        if day_num < target_day_num:
            before.append(ent["path"])
        elif day_num == target_day_num:
            if target_time_int is None or frame_time_int <= target_time_int:
                before.append(ent["path"])

    if not before:
        return []
    if len(before) <= max_frames:
        return before
    indices = np.linspace(0, len(before) - 1, max_frames, dtype=int)
    return [before[i] for i in indices]


# ---------------------------------------------------------------------------
# Benchmark loading & normalization
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> list[dict]:
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        data = payload
    elif isinstance(payload, dict):
        for key in ("samples", "examples", "items", "queries", "data"):
            if isinstance(payload.get(key), list):
                data = payload[key]
                break
        else:
            raise ValueError("Unsupported JSON dict schema")
    else:
        raise ValueError("Unsupported JSON format")

    return [item for item in data if isinstance(item, dict)]


def normalize_options(raw_options: object) -> list[dict[str, str]]:
    if isinstance(raw_options, list):
        out = []
        for i, value in enumerate(raw_options):
            if isinstance(value, dict):
                label = value.get("id") or value.get("label") or chr(ord("A") + i)
                text = value.get("text") or value.get("value") or ""
                out.append({"id": str(label), "text": str(text)})
            else:
                out.append({"id": chr(ord("A") + i), "text": str(value)})
        return out

    if isinstance(raw_options, dict):
        return [{"id": str(k), "text": str(v)} for k, v in raw_options.items()]

    return []


def normalize_item(item: dict) -> dict:
    """Normalize a benchmark item to a common structure."""
    out = dict(item)
    out.pop("evidence_timestamps", None)

    if not out.get("question"):
        out["question"] = (
            item.get("question_text") or item.get("query") or item.get("prompt") or ""
        )

    out["options"] = normalize_options(
        item.get("options") or item.get("choices") or item.get("candidates")
    )

    if not out.get("answer"):
        out["answer"] = (
            item.get("correct_answer")
            or item.get("correct_choice")
            or item.get("answer_label")
            or item.get("answer_key")
            or ""
        )

    if not out.get("target_time"):
        parsed = parse_query_time_to_target(item.get("query_time"))
        if parsed:
            out["target_time"] = parsed

    if not out.get("video_id") and item.get("identity"):
        out["video_id"] = item["identity"]

    return out


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_query_from_item(item: dict) -> str:
    parts = [item["question"], ""]
    for opt in item.get("options", []):
        parts.append(f"{opt['id']}. {opt['text']}")
    return "\n".join(parts)


def build_prompt(item: dict, options: list[dict[str, str]]) -> str:
    if options:
        query = build_query_from_item(item)
        return (
            "You are a video QA system. Review the ordered video frames carefully "
            "and answer the multiple-choice question.\n"
            "Return ONLY the option letter with no explanation.\n\n"
            f"{query}"
        )
    return (
        "You are a video QA system. Review the ordered video frames carefully "
        "and answer the question concisely.\n\n"
        f"{item['question']}"
    )


# ---------------------------------------------------------------------------
# Ablation prompt strategies
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


def build_prompt_direct(item: dict, options: dict[str, str], context_text: str) -> str:
    query = build_query_from_item(item)
    parts = []
    if options:
        parts.append(
            "You are a video QA system. Review the ordered video frames carefully "
            "and answer the multiple-choice question.\n"
            "Return ONLY the option letter with no explanation."
        )
    else:
        parts.append(
            "You are a video QA system. Review the ordered video frames carefully "
            "and answer the question concisely."
        )
    if context_text:
        parts.append("")
        parts.append(context_text)
    parts.append("")
    parts.append(query)
    return "\n".join(parts)


def build_prompt_cot(item: dict, options: dict[str, str], context_text: str) -> str:
    query = build_query_from_item(item)
    parts = []
    if options:
        parts.append(
            "You are a video QA system. Review the ordered video frames carefully "
            "and answer the multiple-choice question.\n"
            "Let's think step by step. First, describe what you observe in the frames "
            "relevant to the question. Then reason about the answer. "
            "Finally, state your answer as: Answer: [LETTER]"
        )
    else:
        parts.append(
            "You are a video QA system. Review the ordered video frames carefully.\n"
            "Let's think step by step. Reason about the answer, then give a concise final answer."
        )
    if context_text:
        parts.append("")
        parts.append(context_text)
    parts.append("")
    parts.append(query)
    return "\n".join(parts)


def build_prompt_icl(item: dict, options: dict[str, str], context_text: str) -> str:
    query = build_query_from_item(item)
    parts = []
    if options:
        parts.append(
            "You are a video QA system. Review the ordered video frames carefully "
            "and answer the multiple-choice question.\n"
            "Return ONLY the option letter with no explanation."
        )
    else:
        parts.append(
            "You are a video QA system. Review the ordered video frames carefully "
            "and answer the question concisely."
        )
    if context_text:
        parts.append("")
        parts.append(context_text)
    parts.append("")
    parts.append(ICL_EXAMPLES)
    parts.append(query)
    return "\n".join(parts)


def build_prompt_icl_cot(item: dict, options: dict[str, str], context_text: str) -> str:
    query = build_query_from_item(item)
    parts = []
    if options:
        parts.append(
            "You are a video QA system. Review the ordered video frames carefully "
            "and answer the multiple-choice question.\n"
            "Let's think step by step. First, describe what you observe in the frames "
            "relevant to the question. Then reason about the answer. "
            "Finally, state your answer as: Answer: [LETTER]"
        )
    else:
        parts.append(
            "You are a video QA system. Review the ordered video frames carefully.\n"
            "Let's think step by step. Reason about the answer, then give a concise final answer."
        )
    if context_text:
        parts.append("")
        parts.append(context_text)
    parts.append("")
    parts.append(ICL_EXAMPLES)
    parts.append(query)
    return "\n".join(parts)


def build_prompt_text_only(item: dict, options: dict[str, str], context_text: str) -> str:
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
    question = item.get("question") or ""
    if query_type:
        lines.append(f"Query type: {query_type}")
    if question:
        lines.append("Question:")
        lines.append(question)
    if options:
        lines.append("")
        for k, v in options.items():
            lines.append(f"{k}. {v}")
    if context_text:
        lines.append("")
        lines.append(context_text)
    return "\n".join(lines)


PROMPT_BUILDERS = {
    "direct": build_prompt_direct,
    "cot": build_prompt_cot,
    "icl": build_prompt_icl,
    "icl_cot": build_prompt_icl_cot,
}


# ---------------------------------------------------------------------------
# Choice parsing (from run_egolife_qwen3vl8b_image.py)
# ---------------------------------------------------------------------------

def parse_choice(raw: str, options: dict[str, str]) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    if text in options:
        return text

    matches = re.findall(r"\b([A-H])\b", text, flags=re.IGNORECASE)
    for letter in reversed(matches):
        letter = letter.upper()
        if letter in options:
            return letter

    matches = re.findall(
        r"(?:option|answer)\s*[:\-]?\s*\(?\s*([A-H])\s*\)?",
        text,
        flags=re.IGNORECASE,
    )
    for letter in reversed(matches):
        letter = letter.upper()
        if letter in options:
            return letter

    normalized = re.sub(r"[\s\.\,\!\?\:\;\(\)\[\]\{\}\"']", "", text).lower()
    for key, value in options.items():
        value_normalized = re.sub(
            r"[\s\.\,\!\?\:\;\(\)\[\]\{\}\"']", "", str(value)
        ).lower()
        if value_normalized and value_normalized in normalized:
            return key

    return None


# ---------------------------------------------------------------------------
# Model loading (from caption_molmo2_from_segmented_folder.py)
# ---------------------------------------------------------------------------

def get_real_model_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    raise RuntimeError("All model parameters are on 'meta'")


def load_molmo2(
    model_id: str,
    dtype: str = "bf16",
    device_map: str = "auto",
    offload_folder: str = "./offload_molmo2",
):
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, padding_side="left",
    )

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and device_map == "auto":
        print(f"  Multi-GPU: {num_gpus} GPUs (no max_memory cap)")

    kwargs = dict(
        trust_remote_code=True,
        device_map=device_map,
        dtype="auto",
    )
    if device_map == "auto":
        kwargs["offload_folder"] = offload_folder

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    _ = get_real_model_device(model)
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_video(
    model,
    processor,
    frame_paths: list[str],
    prompt_text: str,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    if not frame_paths:
        raise ValueError("frame_paths must be non-empty")

    # Load frames, resize to uniform 378x378, stack into numpy array
    frames_np = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB").resize((378, 378), Image.BILINEAR)
        frames_np.append(np.array(img, dtype=np.uint8))
    video_array = np.stack(frames_np)  # (T, 378, 378, 3)

    # Build VideoMetadata so the processor knows FPS and timestamps
    # timestamps is a read-only property = frames_indices / fps
    T = len(frames_np)
    metadata = VideoMetadata(
        total_num_frames=T,
        fps=1.0,
        duration=float(T),
        frames_indices=np.arange(T, dtype=np.float64),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "video", "video": video_array},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        videos_kwargs={
            "do_sample_frames": False,
            "video_metadata": [metadata],
        },
    )

    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
    }
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
            }
        )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_ids = model.generate(**inputs, **gen_kwargs)
    gen = out_ids[0, inputs["input_ids"].size(1) :]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.inference_mode()
def infer_text_only(
    model,
    processor,
    prompt_text: str,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    """Run inference with text only (no video/image frames)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
    }
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature),
                           "top_p": float(top_p), "top_k": int(top_k)})

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_ids = model.generate(**inputs, **gen_kwargs)
    gen = out_ids[0, inputs["input_ids"].size(1):]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(results: list[dict]) -> tuple[int, int, float | None]:
    valid = [r for r in results if r.get("pred") is not None and r.get("answer") is not None]
    correct = [r for r in valid if r.get("correct") is True]
    accuracy = (len(correct) / len(valid)) if valid else None
    return len(correct), len(valid), accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Molmo2-8B on EgoLife-style videoQA benchmarks."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to QA JSON")
    parser.add_argument("--output", type=str, required=True, help="Output result JSON")
    parser.add_argument("--model_id", type=str, default="allenai/Molmo2-8B")
    parser.add_argument(
        "--frame_index", type=str, required=True,
        help="Path to egolife_frames_index.json (single JSON keyed by identity)",
    )
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "auto"])
    parser.add_argument("--offload_folder", type=str, default="./offload_molmo2")
    parser.add_argument("--greedy", type=str, default="true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print_each", action="store_true")
    parser.add_argument("--save_every", type=int, default=0)
    # Ablation flags
    parser.add_argument("--text_only", action="store_true",
                        help="Text-only evaluation, skip frames entirely.")
    parser.add_argument("--prompt_strategy", type=str, default="direct",
                        choices=["direct", "cot", "icl", "icl_cot"],
                        help="Prompt strategy: direct, cot, icl, icl_cot")
    parser.add_argument("--caption_root", type=str, default=None,
                        help="Path to WorldMM_caption dir for context injection")
    parser.add_argument("--caption_duration", type=str, default="3min",
                        help="Caption granularity: 30sec, 3min, 10min, 1h")
    parser.add_argument("--transcript_root", type=str, default=None,
                        help="Path to EgoLifeCap/Transcript dir for context injection")
    parser.add_argument("--max_context_chars", type=int, default=4000,
                        help="Max characters for context injection")
    args = parser.parse_args()

    # Auto-bump max_new_tokens for CoT
    if args.prompt_strategy in ("cot", "icl_cot") and args.max_new_tokens < 512:
        print(f"[ablation] CoT mode: bumping max_new_tokens from {args.max_new_tokens} to 512")
        args.max_new_tokens = 512

    greedy_bool = str(args.greedy).strip().lower() in {"1", "true", "yes", "y", "t"}

    # Load frame index
    print(f"Loading frame index from {args.frame_index} ...")
    with open(args.frame_index, "r") as f:
        frame_index = json.load(f)
    print(f"Frame index loaded: {len(frame_index)} identities")

    # Load benchmark
    data = load_benchmark(args.dataset)
    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(data)
    if args.limit is not None:
        data = data[: args.limit]
    print(f"Loaded {len(data)} examples from {args.dataset}")

    # Load model
    print(f"Loading model {args.model_id} ...")
    model, processor = load_molmo2(
        model_id=args.model_id,
        dtype=args.dtype,
        offload_folder=args.offload_folder,
    )
    dev = get_real_model_device(model)
    print(f"Model loaded on device: {dev}")

    print(f"[ablation] Config: prompt_strategy={args.prompt_strategy}, text_only={args.text_only}, "
          f"max_frames={args.max_frames}, caption_root={args.caption_root}, "
          f"transcript_root={args.transcript_root}, max_new_tokens={args.max_new_tokens}")

    # Evaluate
    results: list[dict[str, Any]] = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    for i, raw_item in enumerate(tqdm(data, desc="Evaluating", total=len(data))):
        item = normalize_item(raw_item)
        video_id = item.get("video_id") or item.get("identity")
        options_list = item.get("options", [])
        options_dict = {opt["id"]: opt["text"] for opt in options_list}

        out_item: dict[str, Any] = {
            "sample_id": item.get("example_id") or item.get("sample_id") or item.get("global_sample_id"),
            "video_id": video_id,
            "question": item.get("question"),
            "answer": item.get("answer"),
            "query_type": item.get("query_type"),
            "pred": None,
            "correct": None,
            "raw_output": None,
            "error": None,
            "latency_s": None,
            "num_frames": 0,
        }

        try:
            if not video_id:
                raise ValueError("no video_id or identity")

            target_time = item.get("target_time")
            if not target_time:
                raise ValueError("no target_time")

            # Get context text (captions/transcripts) if configured
            context_text = ""
            if args.caption_root or args.transcript_root:
                context_text = get_context_text(
                    item, caption_root=args.caption_root,
                    caption_duration=args.caption_duration,
                    transcript_root=args.transcript_root,
                    max_context_chars=args.max_context_chars,
                )

            if args.text_only:
                # Text-only mode: no frames
                prompt = build_prompt_text_only(item, options_dict, context_text)
                t_start = time.time()
                raw_output = infer_text_only(
                    model=model, processor=processor,
                    prompt_text=prompt,
                    max_new_tokens=args.max_new_tokens,
                    greedy=greedy_bool,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )
            else:
                # Get frames
                frame_paths = get_frames_before_target(
                    frame_index, video_id, target_time, max_frames=args.max_frames
                )
                if not frame_paths:
                    raise ValueError(f"no frames found for {video_id} before {target_time}")
                out_item["num_frames"] = len(frame_paths)

                # Build prompt with strategy
                prompt_builder = PROMPT_BUILDERS[args.prompt_strategy]
                prompt = prompt_builder(item, options_dict, context_text)

                t_start = time.time()
                raw_output = infer_video(
                    model=model, processor=processor,
                    frame_paths=frame_paths,
                    prompt_text=prompt,
                    max_new_tokens=args.max_new_tokens,
                    greedy=greedy_bool,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )

            out_item["latency_s"] = round(time.time() - t_start, 3)
            out_item["raw_output"] = raw_output

            if options_dict:
                pred = parse_choice(raw_output, options_dict)
                out_item["pred"] = pred
                if pred is not None and item.get("answer"):
                    out_item["correct"] = pred == item["answer"]
            else:
                out_item["pred"] = raw_output.strip()

        except Exception as exc:
            out_item["error"] = repr(exc)
            if args.print_each:
                print(f"[{i+1}/{len(data)}] ERROR {video_id}: {out_item['error']}")

        if args.print_each and out_item["error"] is None:
            print(
                f"[{i+1}/{len(data)}] latency={out_item['latency_s']}s "
                f"frames={out_item['num_frames']} pred={out_item['pred']} "
                f"gt={out_item['answer']} correct={out_item['correct']}"
            )

        results.append(out_item)

        if (i + 1) % 10 == 0:
            correct, valid, accuracy = compute_accuracy(results)
            print(
                f"[{i+1}/{len(data)}] acc={accuracy} "
                f"(correct={correct}, valid={valid})"
            )

        if args.save_every and args.save_every > 0 and (i + 1) % args.save_every == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    correct, valid, accuracy = compute_accuracy(results)
    print(
        f"Done. Wrote {len(results)} rows to {args.output}. "
        f"acc={accuracy} (correct={correct}, valid={valid})"
    )


if __name__ == "__main__":
    main()
