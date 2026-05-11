#!/usr/bin/env python3

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
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_GEMINI_UTILS_DIR = Path("/nas-ssd2/ziyang/Memory_project/openai/eval")
if str(_GEMINI_UTILS_DIR) not in sys.path:
    sys.path.append(str(_GEMINI_UTILS_DIR))

from gemini_pure_image import (  # noqa: E402
    build_query_from_item,
    get_egolife_frames_before_target,
    get_egolife_frames_before_target_from_index,
    get_image_list_for_video,
    normalize_item,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IDENTITY_FILENAME_PATTERN = re.compile(r"^[A-Z]\d+_[A-Za-z]+$")


def env_get(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else value


def env_get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def build_transform(input_size: int) -> T.Compose:
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    sorted_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, sorted_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


def _model_device(model: Any) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_choice(raw: str, options: dict[str, str]) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    if text in options:
        return text

    matches = re.findall(r"\b([A-F])\b", text, flags=re.IGNORECASE)
    for letter in reversed(matches):
        letter = letter.upper()
        if letter in options:
            return letter

    matches = re.findall(r"(?:option|answer)\s*[:\-]?\s*\(?\s*([A-F])\s*\)?", text, flags=re.IGNORECASE)
    for letter in reversed(matches):
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

    if item.get("correct_choice") and item.get("choices"):
        metadata["distractors"] = [
            choice for choice in item["choices"] if isinstance(choice, dict) and choice.get("label") != item.get("correct_choice")
        ]

    if item.get("evidence_times") is not None:
        metadata["evidence_times"] = item.get("evidence_times")

    for key in (
        "answer_support_clip_id",
        "answer_support_frame_path",
        "support_clip_id",
        "support_frame_path",
        "template_id",
        "query_type",
        "difficulty_score",
        "difficulty_tier",
        "split",
        "object_name",
        "combined_sample_id",
    ):
        if item.get(key) is not None:
            metadata[key] = item.get(key)

    return metadata


def build_prompt(item: dict[str, Any], options: dict[str, str]) -> str:
    if options:
        query = build_query_from_item(item)
        return (
            "You are an image QA system. Review the ordered image frames carefully and answer the multiple-choice question.\n"
            "Return ONLY the option letter with no explanation.\n\n"
            f"{query}"
        )

    return (
        "You are an image QA system. Review the ordered image frames carefully and answer the question concisely.\n\n"
        f"{item['question']}"
    )


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
    joined_summaries = "\n\n".join(summary_blocks)

    if options:
        query = build_query_from_item(item)
        return (
            "You are answering a multiple-choice question using chunk summaries from a long sequence of image frames.\n"
            "Use the summaries as evidence. Return ONLY the option letter with no explanation.\n\n"
            f"{joined_summaries}\n\nQuestion and options:\n{query}"
        )

    return (
        "You are answering a question using chunk summaries from a long sequence of image frames.\n"
        "Use the summaries as evidence and answer concisely.\n\n"
        f"{joined_summaries}\n\nQuestion:\n{item['question']}"
    )


def _build_generation_config(
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
    }
    if greedy:
        cfg["do_sample"] = False
    else:
        cfg.update(
            {
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
            }
        )
    return cfg


def _load_frame_tensor(
    image_paths: list[str],
    input_size: int,
    frame_max_tiles: int,
    model: Any,
    tensor_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    if not image_paths:
        raise ValueError("image_paths must be non-empty")

    pixel_values_list: list[torch.Tensor] = []
    num_patches_list: list[int] = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        frame_tensor = load_image(path, input_size=input_size, max_num=frame_max_tiles)
        pixel_values_list.append(frame_tensor)
        num_patches_list.append(int(frame_tensor.shape[0]))

    pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=tensor_dtype)
    pixel_values = pixel_values.to(device=_model_device(model))
    return pixel_values, num_patches_list


def infer_one(
    model: Any,
    tokenizer: Any,
    image_paths: list[str],
    prompt_text: str,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    input_size: int,
    frame_max_tiles: int,
    tensor_dtype: torch.dtype,
) -> str:
    pixel_values, num_patches_list = _load_frame_tensor(
        image_paths=image_paths,
        input_size=input_size,
        frame_max_tiles=frame_max_tiles,
        model=model,
        tensor_dtype=tensor_dtype,
    )

    frame_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
    question = frame_prefix + prompt_text
    generation_config = _build_generation_config(
        max_new_tokens=max_new_tokens,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    with torch.inference_mode():
        output = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )

    if isinstance(output, tuple) and output:
        output = output[0]
    return str(output)


def infer_text_only(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    generation_config = _build_generation_config(
        max_new_tokens=max_new_tokens,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    with torch.inference_mode():
        output = model.chat(
            tokenizer,
            None,
            prompt_text,
            generation_config,
            history=None,
            return_history=True,
        )

    if isinstance(output, tuple) and output:
        output = output[0]
    return str(output)


def chunk_list(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def compute_accuracy(results: list[dict[str, Any]]) -> tuple[int, int, float | None]:
    valid = [row for row in results if row.get("pred") is not None and row.get("answer") is not None]
    correct = [row for row in valid if row.get("correct") is True]
    accuracy = (len(correct) / len(valid)) if valid else None
    return len(correct), len(valid), accuracy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run InternVideo2.5-Chat-8B on EgoLife-style frame QA datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to QA JSON")
    parser.add_argument("--output", type=str, required=True, help="Output path for result JSON list")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVideo2_5_Chat_8B")
    parser.add_argument("--image_list_dir", type=str, default=None)
    parser.add_argument("--egolife_frames", action="store_true")
    parser.add_argument("--egolife_frame_index_dir", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_hours_before_target", type=float, default=None)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--frame_max_tiles", type=int, default=1)
    parser.add_argument(
        "--vision_batch_size",
        type=int,
        default=0,
        help="If >0 and frame count exceeds this size, summarize each frame chunk first, then answer from summaries.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--print_each", action="store_true")
    parser.add_argument("--use_flash_attn", dest="use_flash_attn", action="store_true")
    parser.add_argument("--no_flash_attn", dest="use_flash_attn", action="store_false")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--greedy", type=str, default=env_get("greedy", "true"))
    parser.add_argument("--top_p", type=float, default=env_get_float("top_p", 0.9))
    parser.add_argument("--top_k", type=int, default=env_get_int("top_k", 50))
    parser.add_argument("--temperature", type=float, default=env_get_float("temperature", 0.0))
    parser.add_argument("--repetition_penalty", type=float, default=env_get_float("repetition_penalty", 1.0))
    parser.add_argument("--out_seq_length", type=int, default=env_get_int("out_seq_length", 64))
    parser.set_defaults(use_flash_attn=True)
    args = parser.parse_args()
    print(f"[startup] script={__file__}")
    try:
        print(f"[startup] script_mtime={os.path.getmtime(__file__)}")
    except Exception:
        pass

    if not (args.image_list_dir or args.egolife_frame_index_dir):
        parser.error("One of --image_list_dir or --egolife_frame_index_dir is required")

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

    greedy_bool = _to_bool(args.greedy)
    dtype_map: dict[str, Any] = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    resolved_dtype = dtype_map[args.dtype]
    if resolved_dtype == "auto":
        resolved_dtype = torch.bfloat16

    # Normalize device_map textual "none"/"null" into None.
    device_map_value: Any = args.device_map
    if isinstance(device_map_value, str):
        lowered = device_map_value.strip().lower()
        if lowered in {"none", "null", ""}:
            device_map_value = None

    # Candidate compatible with your previous working style:
    # AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16, trust_remote_code=True)
    simple_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "trust_remote_code": True,
    }
    if args.use_flash_attn:
        simple_kwargs["use_flash_attn"] = True

    load_candidates: list[dict[str, Any]] = []

    # Primary candidates with explicit device-map handling. Keep the first one
    # aligned with the official multi-GPU pattern:
    # AutoModel.from_pretrained(..., torch_dtype=..., low_cpu_mem_usage=True,
    # use_flash_attn=True, trust_remote_code=True, device_map="auto")
    base_load_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "device_map": device_map_value,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.use_flash_attn:
        base_load_kwargs["use_flash_attn"] = True
    load_candidates.append(dict(base_load_kwargs))
    if "use_flash_attn" in base_load_kwargs:
        no_flash = dict(base_load_kwargs)
        no_flash.pop("use_flash_attn", None)
        load_candidates.append(no_flash)

    dtype_kwargs = dict(base_load_kwargs)
    dtype_kwargs["dtype"] = dtype_kwargs.pop("torch_dtype")
    load_candidates.append(dtype_kwargs)
    if "use_flash_attn" in dtype_kwargs:
        dtype_no_flash = dict(dtype_kwargs)
        dtype_no_flash.pop("use_flash_attn", None)
        load_candidates.append(dtype_no_flash)

    if device_map_value is not None:
        cuda_single = dict(base_load_kwargs)
        cuda_single["device_map"] = "cuda"
        cuda_single["low_cpu_mem_usage"] = False
        load_candidates.append(cuda_single)
        if "use_flash_attn" in cuda_single:
            cuda_single_no_flash = dict(cuda_single)
            cuda_single_no_flash.pop("use_flash_attn", None)
            load_candidates.append(cuda_single_no_flash)

    cpu_then_move = dict(base_load_kwargs)
    cpu_then_move["device_map"] = None
    cpu_then_move["low_cpu_mem_usage"] = False
    load_candidates.append(cpu_then_move)
    if "use_flash_attn" in cpu_then_move:
        cpu_then_move_no_flash = dict(cpu_then_move)
        cpu_then_move_no_flash.pop("use_flash_attn", None)
        load_candidates.append(cpu_then_move_no_flash)

    # Only try no-device-map loading as a last fallback when the user did not
    # request an explicit device map.
    if device_map_value is None:
        load_candidates.append(dict(simple_kwargs))
        if "use_flash_attn" in simple_kwargs:
            simple_no_flash = dict(simple_kwargs)
            simple_no_flash.pop("use_flash_attn", None)
            load_candidates.append(simple_no_flash)

    for kw in list(load_candidates):
        if kw.get("low_cpu_mem_usage") is True:
            slower = dict(kw)
            slower["low_cpu_mem_usage"] = False
            load_candidates.append(slower)

    model = None
    last_exc: Exception | None = None
    loaded_kwargs: dict[str, Any] | None = None
    seen_signatures: set[tuple[tuple[str, str], ...]] = set()
    print(f"[loader] trying {len(load_candidates)} model-load candidates for {args.model_name}")
    for load_kwargs in load_candidates:
        signature = tuple(sorted((k, repr(v)) for k, v in load_kwargs.items()))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        try:
            print(f"[loader] trying kwargs: {load_kwargs}")
            model = AutoModel.from_pretrained(args.model_name, **load_kwargs).eval()
            loaded_kwargs = dict(load_kwargs)
            print(f"[loader] success with kwargs: {loaded_kwargs}")
            break
        except TypeError as exc:
            last_exc = exc
            print(f"[loader] TypeError with kwargs {load_kwargs}: {exc}")
            continue
        except RuntimeError as exc:
            # Some OpenGVLab remote-code models can fail under meta init path.
            if "meta tensors" in str(exc).lower() or "tensor.item() cannot be called on meta tensors" in str(exc):
                last_exc = exc
                print(f"[loader] meta-tensor RuntimeError with kwargs {load_kwargs}: {exc}")
                continue
            raise
        except Exception as exc:
            last_exc = exc
            print(f"[loader] {type(exc).__name__} with kwargs {load_kwargs}: {exc}")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load model {args.model_name!r} after compatibility retries") from last_exc

    # If loaded without a device_map, place the model on the current CUDA device.
    if loaded_kwargs is not None and "device_map" not in loaded_kwargs and torch.cuda.is_available():
        model = model.to("cuda").eval()
    if loaded_kwargs is not None and loaded_kwargs.get("device_map") is None and torch.cuda.is_available():
        model = model.to("cuda").eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    if not hasattr(model, "chat"):
        raise AttributeError(f"Loaded model {args.model_name!r} does not expose a .chat(...) API")

    # Guardrail: cap frames to avoid context overflow from visual tokens.
    tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
    llm_max_len = None
    try:
        llm_cfg = getattr(model.config, "llm_config", None)
        llm_max_len = getattr(llm_cfg, "max_position_embeddings", None)
    except Exception:
        llm_max_len = None

    context_candidates: list[int] = []
    for value in (tokenizer_max_len, llm_max_len):
        if isinstance(value, int) and 0 < value < 1_000_000:
            context_candidates.append(value)
    max_context_len = min(context_candidates) if context_candidates else None
    image_tokens_per_patch = int(getattr(model, "num_image_token", 0) or 0)
    max_tiles_per_frame = args.frame_max_tiles if args.frame_max_tiles <= 1 else args.frame_max_tiles + 1
    image_tokens_per_frame = image_tokens_per_patch * max(1, max_tiles_per_frame)
    reserved_text_tokens = 1024
    if max_context_len is not None and image_tokens_per_frame > 0:
        safe_frame_limit = max(1, (max_context_len - reserved_text_tokens) // image_tokens_per_frame)
        print(
            f"[context] tokenizer_max_len={tokenizer_max_len}, "
            f"llm_max_len={llm_max_len}, using_max_len={max_context_len}, "
            f"image_tokens_per_frame~{image_tokens_per_frame}, requested_max_frames={args.max_frames}"
        )
        if args.max_frames > safe_frame_limit:
            print(
                f"[context] requested max_frames={args.max_frames} exceeds estimated safe limit "
                f"{safe_frame_limit}; capping to {safe_frame_limit} to avoid context overflow."
            )
            args.max_frames = safe_frame_limit

    tensor_dtype = dtype_map[args.dtype]
    if tensor_dtype == "auto":
        tensor_dtype = torch.bfloat16

    results: list[dict[str, Any]] = []
    image_paths_log: list[dict[str, Any]] = []
    base, _ = os.path.splitext(args.output)
    log_image_paths_path = f"{base}_image_paths.json"
    index_cache: dict[str, list] = {}

    for i, raw_item in enumerate(tqdm(data, desc="Evaluating", total=len(data))):
        item = normalize_item(raw_item, default_video_id=default_video_id)
        video_id = item.get("video_id") or item.get("identity") or default_video_id
        if not video_id:
            results.append(
                {
                    "sample_id": item.get("sample_id") or item.get("combined_sample_id"),
                    "video_id": None,
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "pred": None,
                    "correct": None,
                    "raw_output": None,
                    "error": "no video_id or identity",
                }
            )
            continue

        image_paths: list[str] = []
        if args.egolife_frame_index_dir and item.get("target_time"):
            image_paths = get_egolife_frames_before_target_from_index(
                args.egolife_frame_index_dir,
                video_id,
                item["target_time"],
                max_frames=args.max_frames,
                index_cache=index_cache,
                max_hours_before_target=args.max_hours_before_target,
            )
        if not image_paths and args.egolife_frames and item.get("target_time") and args.image_list_dir:
            image_paths = get_egolife_frames_before_target(
                args.image_list_dir,
                video_id,
                item["target_time"],
                max_frames=args.max_frames,
                max_hours_before_target=args.max_hours_before_target,
            )
        if not image_paths and args.image_list_dir:
            image_paths = get_image_list_for_video(args.image_list_dir, video_id)

        image_paths_log.append(
            {
                "index": i,
                "sample_id": item.get("sample_id") or item.get("combined_sample_id"),
                "video_id": video_id,
                "question": item.get("question"),
                "target_time": item.get("target_time"),
                "image_paths": image_paths,
            }
        )
        example_idx = i + 1
        if example_idx <= 5 or example_idx % 5 == 0:
            with open(log_image_paths_path, "w") as f:
                json.dump(image_paths_log, f, indent=2)

        options = normalize_options(item)
        prompt = build_prompt(item, options)

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
        if item.get("correct_choice"):
            out_item["correct_choice"] = item.get("correct_choice")
        out_item.update(extract_eval_metadata(item))

        try:
            if not image_paths:
                raise ValueError("no image list")

            t_start = time.time()
            if args.vision_batch_size and len(image_paths) > args.vision_batch_size:
                summaries: list[str] = []
                for batch_paths in chunk_list(image_paths, args.vision_batch_size):
                    summary_prompt = build_chunk_summary_prompt(item["question"])
                    summary_text = infer_one(
                        model=model,
                        tokenizer=tokenizer,
                        image_paths=batch_paths,
                        prompt_text=summary_prompt,
                        max_new_tokens=args.out_seq_length,
                        greedy=greedy_bool,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        input_size=args.input_size,
                        frame_max_tiles=args.frame_max_tiles,
                        tensor_dtype=tensor_dtype,
                    )
                    summaries.append(summary_text.strip())

                final_prompt = build_final_prompt_from_summaries(item, options, summaries)
                raw_output = infer_text_only(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=final_prompt,
                    max_new_tokens=args.out_seq_length,
                    greedy=greedy_bool,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )
                out_item["chunk_summary_count"] = len(summaries)
            else:
                raw_output = infer_one(
                    model=model,
                    tokenizer=tokenizer,
                    image_paths=image_paths,
                    prompt_text=prompt,
                    max_new_tokens=args.out_seq_length,
                    greedy=greedy_bool,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    input_size=args.input_size,
                    frame_max_tiles=args.frame_max_tiles,
                    tensor_dtype=tensor_dtype,
                )
            out_item["latency_s"] = time.time() - t_start
            out_item["raw_output"] = raw_output

            if options:
                pred = parse_choice(raw_output, options)
                out_item["pred"] = pred
                if pred is not None and item.get("answer") is not None:
                    out_item["correct"] = pred == item.get("answer")
            else:
                out_item["pred"] = raw_output.strip()

            if args.print_each:
                latency = out_item["latency_s"]
                latency_str = f"{latency:.2f}s" if isinstance(latency, (int, float)) else "NA"
                print(
                    f"[{i + 1}/{len(data)}] latency={latency_str} pred={out_item['pred']} "
                    f"gt={out_item['answer']} correct={out_item['correct']}"
                )
        except Exception as exc:
            out_item["error"] = repr(exc)
            if args.print_each:
                print(f"[{i + 1}/{len(data)}] ERROR {video_id}: {out_item['error']}")

        results.append(out_item)

        if (i + 1) % 10 == 0:
            correct, valid, accuracy = compute_accuracy(results)
            print(
                f"[{i + 1}/{len(data)}] acc_valid={accuracy} "
                f"(correct={correct}, valid={valid})"
            )

        if args.save_every and args.save_every > 0 and (i + 1) % args.save_every == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    correct, valid, accuracy = compute_accuracy(results)
    print(
        f"Done. Wrote {len(results)} rows to {args.output}. "
        f"acc_valid={accuracy} (correct={correct}, valid={valid})"
    )


if __name__ == "__main__":
    main()
