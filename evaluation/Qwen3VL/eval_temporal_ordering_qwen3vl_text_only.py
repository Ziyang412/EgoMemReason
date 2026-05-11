#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
import transformers


def load_examples(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {"type": "generic_queries", "description": "chunked examples"}, [dict(row) for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("samples"), list):
            return data, [dict(row) for row in data["samples"] if isinstance(row, dict)]

        if isinstance(data.get("identities"), dict):
            examples: list[dict[str, Any]] = []
            for identity, items in data["identities"].items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    row = dict(item)
                    row["identity"] = row.get("identity", identity)
                    examples.append(row)
            return data, examples

        for key in ("examples", "items", "queries", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return data, [dict(row) for row in value if isinstance(row, dict)]

    raise ValueError("Input JSON must be a list, or a dict with one of: samples / identities / examples / items / queries / data")


def _sorted_event_items(events: dict[str, Any]) -> list[tuple[str, Any]]:
    def key_fn(kv: tuple[str, Any]) -> tuple[int, str]:
        key = str(kv[0])
        if key.isdigit():
            return (0, f"{int(key):06d}")
        return (1, key)

    return sorted(((str(k), v) for k, v in events.items()), key=key_fn)


def _normalize_text(text: Any) -> str:
    return re.sub(r"[\W_]+", "", str(text).lower())


def _first_nonempty_string(example: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten_values(item))
        return out
    return [value]


def _is_primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if _is_primitive(value):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def normalize_options(example: dict[str, Any]) -> dict[str, str]:
    options_raw = example.get("options")
    if options_raw is None:
        options_raw = example.get("choices")
    if options_raw is None:
        options_raw = example.get("candidates")

    options: dict[str, str] = {}
    if isinstance(options_raw, dict):
        for key, value in options_raw.items():
            label = str(key).strip().upper()
            if not label:
                continue
            if isinstance(value, dict):
                text = value.get("text") or value.get("option") or value.get("value") or json.dumps(value, ensure_ascii=False)
            else:
                text = value
            options[label] = str(text)
        if options:
            return options

    if isinstance(options_raw, list):
        for idx, value in enumerate(options_raw):
            fallback_label = chr(ord("A") + idx)
            label = fallback_label
            text: Any = value
            if isinstance(value, dict):
                raw_label = value.get("label") or value.get("id") or value.get("key")
                if raw_label is not None and str(raw_label).strip():
                    label = str(raw_label).strip().upper()
                text = value.get("text") or value.get("option") or value.get("value") or json.dumps(value, ensure_ascii=False)
            options[label] = str(text)
        if options:
            return options

    fallback_option_keys = [
        ("A", "choice_a"),
        ("B", "choice_b"),
        ("C", "choice_c"),
        ("D", "choice_d"),
        ("E", "choice_e"),
        ("F", "choice_f"),
        ("G", "choice_g"),
        ("H", "choice_h"),
    ]
    for label, key in fallback_option_keys:
        value = example.get(key)
        if value is not None:
            options[label] = _to_string(value)
    return options


def build_prompt(example: dict[str, Any], options: dict[str, str]) -> str:
    lines: list[str] = []

    if options:
        labels = ", ".join(options.keys())
        lines.append("You are solving a multiple-choice benchmark query.")
        lines.append(f"Return only one option label from: {labels}. Do not output anything else.")
    else:
        lines.append("You are solving a benchmark query.")
        lines.append("Answer concisely and directly.")
    lines.append("")

    query_type = _first_nonempty_string(example, ("query_type", "question_type", "type"))
    question = _first_nonempty_string(
        example,
        ("question", "question_text", "query", "query_text", "prompt", "instruction", "task"),
    )
    if not question and isinstance(example.get("events"), dict):
        question = "Answer the question using the listed events."
    if query_type:
        lines.append(f"Query type: {query_type}")
    if question:
        lines.append("Question:")
        lines.append(question)
        lines.append("")

    context_lines: list[str] = []
    for key in ("query_time", "target_time", "identity", "video_id", "p_id", "question_source"):
        value = example.get(key)
        if value is not None and str(value).strip():
            context_lines.append(f"- {key}: {value}")

    skip_context_keys = {
        "events",
        "event_times",
        "evidence_timestamps",
        "options",
        "choices",
        "candidates",
        "question",
        "question_text",
        "query",
        "query_text",
        "prompt",
        "instruction",
        "task",
        "correct_answer",
        "answer",
        "answer_text",
        "correct_choice",
        "ground_truth",
        "gt",
        "label",
        "target",
        "answer_key",
        "answer_label",
        "answer_idx",
        "answer_id",
        "correct_option",
        "correct_option_id",
        "correct_order",
    }
    for key, value in example.items():
        if key in skip_context_keys:
            continue
        if key in {"query_time", "target_time", "identity", "video_id", "p_id", "question_source"}:
            continue
        if _is_primitive(value):
            value_str = _to_string(value).strip()
            if value_str and len(value_str) <= 300:
                context_lines.append(f"- {key}: {value_str}")
        elif isinstance(value, list) and value:
            if all(_is_primitive(x) for x in value):
                joined = " | ".join(_to_string(x).strip() for x in value[:8] if _to_string(x).strip())
                if joined:
                    context_lines.append(f"- {key}: {joined}")
        elif isinstance(value, dict) and value:
            if all(_is_primitive(v) for v in value.values()):
                dict_lines = ", ".join(f"{k}={_to_string(v)}" for k, v in list(value.items())[:8])
                if dict_lines and len(dict_lines) <= 300:
                    context_lines.append(f"- {key}: {dict_lines}")

    context_lines = _unique_preserve_order(context_lines)
    if context_lines:
        lines.append("Context:")
        lines.extend(context_lines)
        lines.append("")

    events = example.get("events")
    if isinstance(events, dict) and events:
        lines.append("Events:")
        lines.extend([f"{idx}. {text}" for idx, text in _sorted_event_items(events)])
        lines.append("")

    # Never expose evidence_timestamps to the model.
    for time_key in ("event_times",):
        ts = example.get(time_key)
        if isinstance(ts, dict) and ts:
            lines.append(f"{time_key}:")
            lines.extend([f"{idx}: {text}" for idx, text in _sorted_event_items(ts)])
            lines.append("")

    if options:
        lines.append("Options:")
        for label, text in options.items():
            lines.append(f"{label}. {text}")
        lines.append("")

    return "\n".join(lines).strip()


def _extract_label(text: str, labels: list[str]) -> str:
    if not text:
        return ""

    raw = str(text).strip().upper()
    label_set = {label.upper() for label in labels}
    if raw in label_set:
        return raw

    found: list[str] = []
    for label in labels:
        lu = label.upper()
        if re.search(rf"(?<!\w){re.escape(lu)}(?!\w)", raw):
            found.append(lu)
    if found:
        return found[-1]

    for label in labels:
        lu = label.upper()
        if re.search(rf"(?:ANSWER|OPTION|CHOICE)\s*[:\-]?\s*\(?\s*{re.escape(lu)}\s*\)?", raw):
            return lu
    return ""


def parse_choice(raw_output: str, options: dict[str, str]) -> str:
    labels = list(options.keys())
    by_label = _extract_label(raw_output, labels)
    if by_label:
        return by_label

    normalized_raw = _normalize_text(raw_output)
    if not normalized_raw:
        return ""

    option_items = [(label, _normalize_text(text)) for label, text in options.items()]
    option_items.sort(key=lambda kv: len(kv[1]), reverse=True)
    for label, normalized_option_text in option_items:
        if normalized_option_text and normalized_option_text in normalized_raw:
            return label.upper()

    return ""


def _extract_ground_truth_candidates(example: dict[str, Any], options: dict[str, str]) -> tuple[list[str], list[str]]:
    candidate_keys = (
        "correct_answer",
        "answer",
        "answer_text",
        "correct_choice",
        "ground_truth",
        "gt",
        "label",
        "target",
        "answer_key",
        "answer_label",
        "answer_idx",
        "answer_index",
        "answer_number",
        "answer_id",
        "correct_option",
        "correct_option_id",
        "reference_answer",
        "expected_answer",
        "gold_answer",
    )
    raw_entries: list[tuple[str, Any]] = []
    for key in candidate_keys:
        value = example.get(key)
        if value is not None and _to_string(value).strip():
            for item in _flatten_values(value):
                raw_entries.append((key, item))

    if not raw_entries:
        return [], []

    labels = list(options.keys())
    label_set_upper = {label.upper() for label in labels}
    gt_labels: list[str] = []
    gt_texts: list[str] = []

    def maybe_add_label_from_int(num: int, source_key: str) -> None:
        if not labels:
            return
        key_l = source_key.lower()
        if "idx" in key_l or "index" in key_l:
            if 0 <= num < len(labels):
                gt_labels.append(labels[num].upper())
            return
        if "number" in key_l or "rank" in key_l or "position" in key_l or "order" in key_l:
            if 1 <= num <= len(labels):
                gt_labels.append(labels[num - 1].upper())
            return

        if 0 <= num < len(labels):
            gt_labels.append(labels[num].upper())
            return
        if 1 <= num <= len(labels):
            gt_labels.append(labels[num - 1].upper())

    idx = 0
    while idx < len(raw_entries):
        source_key, raw = raw_entries[idx]
        idx += 1
        if raw is None:
            continue

        if isinstance(raw, dict):
            for dict_key in ("label", "choice", "option", "id", "key"):
                dv = raw.get(dict_key)
                if dv is not None:
                    raw_entries.append((f"{source_key}.{dict_key}", dv))
            for dict_key in ("text", "option_text", "value", "answer"):
                dv = raw.get(dict_key)
                if dv is not None:
                    raw_entries.append((f"{source_key}.{dict_key}", dv))
            continue

        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            maybe_add_label_from_int(int(raw), source_key=source_key)
            gt_texts.append(str(raw))
            continue

        text = str(raw).strip()
        if not text:
            continue

        upper = text.upper()
        if upper in label_set_upper:
            gt_labels.append(upper)
            gt_texts.append(text)
            continue

        label_from_text = _extract_label(text, labels)
        if label_from_text:
            gt_labels.append(label_from_text)
            gt_texts.append(text)
            continue

        if options:
            normalized = _normalize_text(text)
            for label, option_text in options.items():
                if _normalize_text(option_text) == normalized:
                    gt_labels.append(label.upper())
                    break
        gt_texts.append(text)

    gt_labels = _unique_preserve_order([x.upper() for x in gt_labels if x.upper() in label_set_upper or not options])
    gt_texts = _unique_preserve_order([x for x in gt_texts if x.strip()])
    return gt_labels, gt_texts


def _normalized_match(lhs: str, rhs: str) -> bool:
    return _normalize_text(lhs) == _normalize_text(rhs)


def _compute_is_correct(
    pred_text: str,
    pred_label: str | None,
    gt_labels: list[str],
    gt_texts: list[str],
    options: dict[str, str],
) -> bool | None:
    if options:
        if gt_labels:
            return bool(pred_label) and pred_label in {x.upper() for x in gt_labels}
        if gt_texts:
            if pred_label:
                pred_option_text = options.get(pred_label, "")
                for gt in gt_texts:
                    if _normalized_match(pred_option_text, gt):
                        return True
            for gt in gt_texts:
                if _normalized_match(pred_text, gt):
                    return True
            return False
        return None

    if gt_texts:
        for gt in gt_texts:
            if _normalized_match(pred_text, gt):
                return True
        return False
    return None


def infer_text_only(
    model: Any,
    processor: Any,
    prompt_text: str,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
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
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
            }
        )

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL text-only evaluation for generic benchmark query JSON"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="/nas-ssd2/ziyang/Memory_project/COLM/benchmark/raw_files_batch1/event_order/temporal_ordering_60_current_format.json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/nas-ssd2/ziyang/Memory_project/COLM/evaluation/Qwen3VL/results/temporal_ordering/temporal_ordering_60_current_format_text_only_qwen3vl.json",
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--flash_attn2", action="store_true")
    parser.add_argument("--greedy", type=str, default=os.environ.get("GREEDY", "true"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    parser.add_argument("--top_p", type=float, default=float(os.environ.get("TOP_P", "0.9")))
    parser.add_argument("--top_k", type=int, default=int(os.environ.get("TOP_K", "50")))
    parser.add_argument(
        "--repetition_penalty", type=float, default=float(os.environ.get("REPETITION_PENALTY", "1.0"))
    )
    parser.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "16")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print_each", action="store_true")
    args = parser.parse_args()

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
        raise ImportError(
            "No compatible Qwen-VL model class found in transformers. "
            f"Current version: {transformers.__version__}. "
            "Please upgrade transformers to a version that supports Qwen-VL."
        )

    auto_processor = getattr(transformers, "AutoProcessor", None)
    if auto_processor is None:
        raise ImportError("transformers.AutoProcessor is not available in this environment.")

    meta, examples = load_examples(args.input_json)
    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(examples)
    if args.limit is not None and args.limit > 0:
        examples = examples[:args.limit]

    greedy_bool = str(args.greedy).strip().lower() in {"1", "true", "yes", "y", "t"}

    dtype_map: dict[str, Any] = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[args.dtype],
        "device_map": "auto",
    }
    if args.flash_attn2:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = model_class.from_pretrained(args.model_name, **load_kwargs)
    processor = auto_processor.from_pretrained(args.model_name)

    results: list[dict[str, Any]] = []
    correct = 0

    for idx, ex in enumerate(examples, start=1):
        options = normalize_options(ex)
        prompt = build_prompt(ex, options)
        t0 = time.time()
        error = None
        raw_response = ""
        pred = ""
        pred_label: str | None = None
        gt_labels, gt_texts = _extract_ground_truth_candidates(ex, options)
        is_correct: bool | None = None

        try:
            raw_response = infer_text_only(
                model=model,
                processor=processor,
                prompt_text=prompt,
                max_new_tokens=args.max_new_tokens,
                greedy=greedy_bool,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            ).strip()
            if options:
                pred = parse_choice(raw_response, options)
                pred_label = pred if pred else None
            else:
                pred = raw_response

            is_correct = _compute_is_correct(
                pred_text=pred,
                pred_label=pred_label,
                gt_labels=gt_labels,
                gt_texts=gt_texts,
                options=options,
            )
        except Exception as exc:
            error = repr(exc)
            is_correct = None

        if is_correct is True:
            correct += 1

        row = dict(ex)
        row["normalized_options"] = options or None
        row["gt_labels"] = gt_labels
        row["gt_texts"] = gt_texts
        row["pred"] = pred
        row["pred_label"] = pred_label
        row["raw_response"] = raw_response
        row["is_correct"] = is_correct
        row["error"] = error
        row["latency_sec"] = round(time.time() - t0, 3)
        results.append(row)

        if args.print_each:
            example_id = row.get("example_id") or row.get("sample_id") or row.get("id") or idx
            gt_display = ",".join(gt_labels) if gt_labels else (" | ".join(gt_texts) if gt_texts else "N/A")
            pred_display = pred_label or pred or "N/A"
            print(
                f"[{idx}/{len(examples)}] {example_id} "
                f"gt={gt_display} pred={pred_display} ok={is_correct}"
                + (f" error={error}" if error else "")
            )

    evaluated = len(results)
    scored = sum(1 for row in results if row.get("is_correct") is not None)
    accuracy = (correct / scored) if scored else 0.0
    payload = {
        "task_type": meta.get("type"),
        "task_description": meta.get("description"),
        "input_file": args.input_json,
        "model_name": args.model_name,
        "evaluated_examples": evaluated,
        "scored_examples": scored,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "results": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to: {out_path}")
    print(f"Accuracy: {correct}/{scored} = {accuracy:.4f} (scored/evaluated: {scored}/{evaluated})")


if __name__ == "__main__":
    main()
