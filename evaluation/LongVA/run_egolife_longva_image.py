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

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_GEMINI_UTILS_CANDIDATES = [
    Path(env) for env in [os.environ.get("GEMINI_UTILS_DIR")] if env
]
_GEMINI_UTILS_CANDIDATES.append(Path("/nas-ssd2/ziyang/Memory_project/openai/eval"))
if len(_SCRIPT_DIR.parents) >= 5:
    _GEMINI_UTILS_CANDIDATES.append(_SCRIPT_DIR.parents[4] / "openai" / "eval")
for _candidate in _GEMINI_UTILS_CANDIDATES:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.append(str(_candidate))
        break

from gemini_pure_image import (  # noqa: E402
    build_query_from_item,
    get_egolife_frames_before_target,
    get_egolife_frames_before_target_from_index,
    get_image_list_for_video,
    normalize_item,
)

IMAGE_TOKEN_INDEX: int | None = None
process_images = None
tokenizer_image_token = None
load_pretrained_model = None


def _ensure_longva_imports() -> None:
    global IMAGE_TOKEN_INDEX, process_images, tokenizer_image_token, load_pretrained_model
    if IMAGE_TOKEN_INDEX is not None and process_images is not None and tokenizer_image_token is not None and load_pretrained_model is not None:
        return
    try:
        from longva.constants import IMAGE_TOKEN_INDEX as _IMAGE_TOKEN_INDEX  # noqa: WPS433
        from longva.mm_utils import process_images as _process_images, tokenizer_image_token as _tokenizer_image_token  # noqa: WPS433
        from longva.model.builder import load_pretrained_model as _load_pretrained_model  # noqa: WPS433
    except Exception as exc:
        raise ImportError(
            "Failed to import LongVA modules. Install/import `longva` in this environment first."
        ) from exc
    IMAGE_TOKEN_INDEX = _IMAGE_TOKEN_INDEX
    process_images = _process_images
    tokenizer_image_token = _tokenizer_image_token
    load_pretrained_model = _load_pretrained_model

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


def _model_compute_dtype(model: Any) -> torch.dtype:
    for param in model.parameters():
        try:
            if param.is_floating_point() and getattr(param.device, "type", None) != "meta":
                return param.dtype
        except Exception:
            continue
    return torch.float16


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

    option_keys = {str(key).strip().upper() for key in options.keys()}

    matches = re.findall(r"\b([A-Z])\b", text, flags=re.IGNORECASE)
    for letter in reversed(matches):
        letter = letter.upper()
        if letter in option_keys:
            return letter

    matches = re.findall(r"(?:option|answer)\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?", text, flags=re.IGNORECASE)
    for letter in reversed(matches):
        letter = letter.upper()
        if letter in option_keys:
            return letter

    normalized = re.sub(r"[\s\.\,\!\?\:\;\(\)\[\]\{\}\"']", "", text).lower()
    for key, value in options.items():
        value_normalized = re.sub(r"[\s\.\,\!\?\:\;\(\)\[\]\{\}\"']", "", str(value)).lower()
        if value_normalized and value_normalized in normalized:
            return key

    return None


def _decode_generation(
    tokenizer: Any,
    output_ids: torch.Tensor,
    input_ids: torch.Tensor,
) -> str:
    tail_text = ""
    if output_ids.shape[1] > input_ids.shape[1]:
        generated_ids = output_ids[:, input_ids.shape[1] :]
        tail_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if tail_text:
        return tail_text

    full_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if full_text:
        return full_text

    # Debug fallback: keep special tokens if needed to inspect why text is empty.
    return tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()


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
        "num_beams": 1,
        "use_cache": True,
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


def _build_longva_chat_prompt(user_text: str, with_image_token: bool) -> str:
    if with_image_token:
        user_block = f"<image>\n{user_text}"
    else:
        user_block = user_text
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_block}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _load_video_tensor_from_frames(
    image_paths: list[str],
    image_processor: Any,
    model: Any,
    tensor_dtype: torch.dtype,
) -> torch.Tensor:
    if not image_paths:
        raise ValueError("image_paths must be non-empty")

    frames: list[np.ndarray] = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        frames.append(np.array(Image.open(path).convert("RGB")))

    frame_array = np.stack(frames, axis=0)
    video_tensor = image_processor.preprocess(frame_array, return_tensors="pt")["pixel_values"]
    return video_tensor.to(device=_model_device(model), dtype=tensor_dtype)


def infer_one(
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    image_paths: list[str],
    prompt_text: str,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    tensor_dtype: torch.dtype,
) -> str:
    prompt = _build_longva_chat_prompt(prompt_text, with_image_token=True)
    device = _model_device(model)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    generation_config = _build_generation_config(
        max_new_tokens=max_new_tokens,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    with torch.inference_mode():
        if len(image_paths) == 1:
            image = Image.open(image_paths[0]).convert("RGB")
            images_tensor = process_images([image], image_processor, model.config).to(device=device, dtype=tensor_dtype)
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                modalities=["image"],
                **generation_config,
            )
        else:
            video_tensor = _load_video_tensor_from_frames(
                image_paths=image_paths,
                image_processor=image_processor,
                model=model,
                tensor_dtype=tensor_dtype,
            )
            output_ids = model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                **generation_config,
            )

    return _decode_generation(tokenizer, output_ids, input_ids)


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
    prompt = _build_longva_chat_prompt(prompt_text, with_image_token=False)
    device = _model_device(model)
    model_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generation_config = _build_generation_config(
        max_new_tokens=max_new_tokens,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
    return _decode_generation(tokenizer, output_ids, input_ids)


def chunk_list(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def compute_accuracy(results: list[dict[str, Any]]) -> tuple[int, int, float | None]:
    valid = [row for row in results if row.get("pred") is not None and row.get("answer") is not None]
    correct = [row for row in valid if row.get("correct") is True]
    accuracy = (len(correct) / len(valid)) if valid else None
    return len(correct), len(valid), accuracy


def _load_dataset_rows(dataset_path: str) -> list[dict[str, Any]]:
    with open(dataset_path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            rows = payload["samples"]
        elif isinstance(payload.get("examples"), list):
            rows = payload["examples"]
        elif isinstance(payload.get("items"), list):
            rows = payload["items"]
        elif isinstance(payload.get("queries"), list):
            rows = payload["queries"]
        elif isinstance(payload.get("data"), list):
            rows = payload["data"]
        elif isinstance(payload.get("identities"), dict):
            rows = []
            for identity, items in payload["identities"].items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    normalized = dict(item)
                    if not normalized.get("identity"):
                        normalized["identity"] = identity
                    rows.append(normalized)
        else:
            raise ValueError(
                "Unsupported JSON dict schema. Expected one of: "
                "samples/examples/items/queries/data/identities."
            )
    else:
        raise ValueError("Dataset JSON must be a list or dict")

    filtered_rows = [row for row in rows if isinstance(row, dict)]
    if not filtered_rows:
        raise ValueError("No usable rows found in dataset JSON")
    return filtered_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LongVA on EgoLife-style frame QA datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to QA JSON")
    parser.add_argument("--output", type=str, required=True, help="Output path for result JSON list")
    parser.add_argument("--model_name", type=str, default="lmms-lab/LongVA-7B-DPO")
    parser.add_argument("--model_arch", type=str, default="llava_qwen")
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
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
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

    data = _load_dataset_rows(args.dataset)

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
    requested_tensor_dtype = dtype_map[args.dtype]

    device_map_value: Any = args.device_map
    if isinstance(device_map_value, str):
        lowered = device_map_value.strip().lower()
        if lowered in {"none", "null", ""}:
            device_map_value = None

    _ensure_longva_imports()
    print(
        "[loader] loading LongVA with "
        f"model_name={args.model_name}, model_arch={args.model_arch}, device_map={device_map_value}"
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_name,
        None,
        args.model_arch,
        device_map=device_map_value,
    )
    model.eval()
    if not hasattr(model, "generate"):
        raise AttributeError(f"Loaded model {args.model_name!r} does not expose .generate(...)")

    model_dtype = _model_compute_dtype(model)
    if requested_tensor_dtype == "auto":
        tensor_dtype = model_dtype
    else:
        tensor_dtype = requested_tensor_dtype
        if tensor_dtype != model_dtype:
            print(
                f"[dtype] requested input dtype={tensor_dtype}, model dtype={model_dtype}; "
                "overriding inputs to model dtype."
            )
            tensor_dtype = model_dtype

    tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max_len, int) and tokenizer_max_len > 0:
        print(f"[context] tokenizer.model_max_length={tokenizer_max_len}")
    if args.vision_batch_size and args.vision_batch_size > 0:
        print(
            f"[context] chunk summarization enabled: max_frames={args.max_frames}, "
            f"vision_batch_size={args.vision_batch_size}"
        )

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
                        image_processor=image_processor,
                        image_paths=batch_paths,
                        prompt_text=summary_prompt,
                        max_new_tokens=args.out_seq_length,
                        greedy=greedy_bool,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        tensor_dtype=tensor_dtype,
                    )
                    summaries.append(summary_text.strip())

                final_prompt = build_final_prompt_from_summaries(item, options, summaries)
                # LongVA backends can fail on pure text-only generate() in this path.
                # Use one anchor frame plus summary text for the final decision step.
                anchor_frame = [image_paths[-1]]
                raw_output = infer_one(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image_paths=anchor_frame,
                    prompt_text=final_prompt,
                    max_new_tokens=args.out_seq_length,
                    greedy=greedy_bool,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    tensor_dtype=tensor_dtype,
                )
                out_item["chunk_summary_count"] = len(summaries)
                out_item["final_stage_mode"] = "summary_plus_anchor_frame"
            else:
                raw_output = infer_one(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image_paths=image_paths,
                    prompt_text=prompt,
                    max_new_tokens=args.out_seq_length,
                    greedy=greedy_bool,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
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
                print(f"[{i + 1}/{len(data)}] raw_output={out_item['raw_output']!r}")
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
