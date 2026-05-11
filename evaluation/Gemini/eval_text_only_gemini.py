import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types


# Keep compatibility with the existing Gemini examples in this workspace.
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # set GEMINI_API_KEY or pass --api_key
DEFAULT_MODEL = "gemini-3-flash-preview"
QUERY_TYPE_TO_MEMORY_TYPE = {
    "temporal_ordering": "episodic",
    "temporal_reasoning": "episodic",
    "state_tracking": "episodic",
    "spatial_tracking": "episodic",
    "multi_entity": "episodic",
    "semantic_event": "semantic",
}


def load_examples(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        meta = {
            "type": "generic_queries",
            "description": "flat list input",
            "total_examples": len(data),
        }
        examples = [dict(row) for row in data if isinstance(row, dict)]
        return meta, examples

    if not isinstance(data, dict):
        raise ValueError(
            "Input JSON must be a list, or a dict with one of: samples / identities / examples / items / queries / data"
        )

    if isinstance(data.get("samples"), list):
        examples = [dict(row) for row in data["samples"] if isinstance(row, dict)]
        return data, examples

    if isinstance(data.get("identities"), dict):
        examples = []
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
            examples = [dict(row) for row in value if isinstance(row, dict)]
            return data, examples

    raise ValueError(
        "Input JSON must be a list, or a dict with one of: samples / identities / examples / items / queries / data"
    )


def _sorted_kv_lines(values: Dict[str, Any]) -> List[str]:
    def key_fn(item: Tuple[str, Any]) -> Tuple[int, str]:
        k = str(item[0]).strip()
        if k.isdigit():
            return (0, f"{int(k):06d}")
        return (1, k)

    return [f"{str(k)}: {v}" for k, v in sorted(values.items(), key=key_fn)]


def _first_nonempty_string(example: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_text(text: Any) -> str:
    return re.sub(r"[\W_]+", "", str(text).lower())


def normalize_options(example: Dict[str, Any]) -> Dict[str, str]:
    options_raw = example.get("options")
    if options_raw is None:
        options_raw = example.get("choices")
    if options_raw is None:
        options_raw = example.get("candidates")

    options: Dict[str, str] = {}

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
            label = chr(ord("A") + idx)
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
            options[label] = str(value)
    return options


def build_prompt(question: Optional[str], query_time: Optional[str], options: Dict[str, str]) -> str:
    lines: List[str] = []

    if options:
        labels = ", ".join(options.keys())
        lines.append("You are solving a multiple-choice benchmark query.")
        lines.append(f"Return only one option label from: {labels}. Do not output anything else.")
    else:
        lines.append("You are solving a benchmark query.")
        lines.append("Answer concisely and directly.")
    lines.append("")

    if question:
        lines.append("Question:")
        lines.append(question)
        lines.append("")

    if query_time is not None and str(query_time).strip():
        lines.append("Context:")
        lines.append(f"- query_time: {query_time}")
        lines.append("")

    if options:
        lines.append("Options:")
        for label, text in options.items():
            lines.append(f"{label}. {text}")
        lines.append("")

    return "\n".join(lines).strip()


def _extract_label(text: str, labels: List[str]) -> str:
    if not text or not labels:
        return ""

    normalized_labels: List[str] = []
    seen = set()
    for label in labels:
        upper_label = str(label).strip().upper()
        if upper_label and upper_label not in seen:
            seen.add(upper_label)
            normalized_labels.append(upper_label)
    if not normalized_labels:
        return ""

    raw = str(text).strip().upper()
    # Remove markdown wrappers (for example: **C**).
    raw = re.sub(r"[`*_]", "", raw)
    label_set = set(normalized_labels)
    if raw in label_set:
        return raw

    label_alt = "|".join(re.escape(label) for label in normalized_labels)
    lines = [line.strip().upper() for line in raw.splitlines() if line.strip()]

    # First scan last lines: model usually prints final answer at the end.
    for line in reversed(lines[-8:]):
        if line in label_set:
            return line

        m = re.match(rf"^\(?\s*({label_alt})\s*\)?[.)]?$", line)
        if m:
            return m.group(1)

        # Support common compact format like: "D. 9" / "D: ...".
        m = re.match(rf"^\(?\s*({label_alt})\s*\)?\s*[.):\-]\s*.+$", line)
        if m:
            return m.group(1)

        m = re.match(
            rf"^(?:FINAL\s*ANSWER|ANSWER|OPTION|CHOICE|SELECTED\s*OPTION|CORRECT\s*OPTION)\s*[:\-]?\s*\(?\s*({label_alt})\s*\)?\.?$",
            line,
        )
        if m:
            return m.group(1)

        m = re.match(
            rf"^(?:THEREFORE|THUS|HENCE|SO|IN\s+CONCLUSION).{{0,80}}(?:OPTION\s*)?\(?\s*({label_alt})\s*\)?\.?$",
            line,
        )
        if m:
            return m.group(1)

    # Then scan explicit answer markers near the end of output.
    tail = raw[-1500:]
    explicit_patterns = [
        rf"(?:FINAL\s*ANSWER|ANSWER|OPTION|CHOICE|SELECTED\s*OPTION|CORRECT\s*OPTION)\s*[:\-]?\s*\(?\s*({label_alt})\s*\)?",
        rf"(?:CORRESPONDS\s+TO|MATCHES|IS)\s+(?:OPTION\s*)?\(?\s*({label_alt})\s*\)?\.?\s*$",
    ]
    for pattern in explicit_patterns:
        matches = list(re.finditer(pattern, tail, re.MULTILINE))
        if matches:
            return matches[-1].group(1)

    return ""


def parse_choice(raw_output: str, options: Dict[str, str]) -> str:
    labels = list(options.keys())
    by_label = _extract_label(raw_output, labels)
    if by_label:
        return by_label

    normalized_raw = _normalize_text(raw_output)
    if not normalized_raw:
        return ""

    exact_matches: List[str] = []
    contains_matches: List[str] = []
    for label, text in options.items():
        normalized_option_text = _normalize_text(text)
        if not normalized_option_text:
            continue
        if normalized_option_text == normalized_raw:
            exact_matches.append(label.upper())
        if normalized_option_text in normalized_raw:
            contains_matches.append(label.upper())

    exact_unique = sorted(set(exact_matches))
    if len(exact_unique) == 1:
        return exact_unique[0]

    contains_unique = sorted(set(contains_matches))
    if len(contains_unique) == 1:
        return contains_unique[0]

    return ""


def extract_ground_truth(example: Dict[str, Any], options: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    gt_raw: Any = None
    for key in (
        "correct_answer",
        "answer",
        "correct_choice",
        "ground_truth",
        "gt",
        "label",
        "target",
    ):
        value = example.get(key)
        if value is not None and str(value).strip():
            gt_raw = value
            break

    if gt_raw is None:
        return None, None

    if not options:
        return None, str(gt_raw).strip()

    labels = list(options.keys())
    label_set = {label.upper() for label in labels}

    if isinstance(gt_raw, int):
        if 0 <= gt_raw < len(labels):
            return labels[gt_raw].upper(), str(gt_raw)
        if 1 <= gt_raw <= len(labels):
            return labels[gt_raw - 1].upper(), str(gt_raw)

    gt_text = str(gt_raw).strip()
    gt_upper = gt_text.upper()
    if gt_upper in label_set:
        return gt_upper, gt_text

    label_from_text = _extract_label(gt_text, labels)
    if label_from_text:
        return label_from_text, gt_text

    normalized_gt = _normalize_text(gt_text)
    for label, option_text in options.items():
        if _normalize_text(option_text) == normalized_gt:
            return label.upper(), gt_text

    return None, gt_text


def shard_examples(examples: List[Dict[str, Any]], num_jobs: int, job_id: int) -> List[Dict[str, Any]]:
    if num_jobs <= 1:
        return examples
    total = len(examples)
    if total == 0:
        return []
    chunk_size = (total + num_jobs - 1) // num_jobs
    start = job_id * chunk_size
    end = min((job_id + 1) * chunk_size, total)
    return examples[start:end]


def _normalize_group_value(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def infer_query_type(example: Dict[str, Any]) -> str:
    value = _first_nonempty_string(example, ("query_type", "task_type", "type"))
    return _normalize_group_value(value)


def infer_memory_type(example: Dict[str, Any], query_type: str) -> str:
    explicit = _first_nonempty_string(example, ("memory_type", "memory_category"))
    if explicit:
        return explicit

    key = _normalize_group_value(query_type).lower()
    if key in QUERY_TYPE_TO_MEMORY_TYPE:
        return QUERY_TYPE_TO_MEMORY_TYPE[key]

    source_dataset = _normalize_group_value(example.get("source_dataset"), default="")
    if source_dataset and "semantic" in source_dataset.lower():
        return "semantic"

    if key != "unknown":
        # Default for known benchmark question families when explicit memory type is absent.
        return "episodic"
    return "unknown"


def _compute_group_accuracy(rows: List[Dict[str, Any]], group_key: str) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        group = _normalize_group_value(row.get(group_key))
        stats[group]["total"] += 1
        if bool(row.get("is_correct")):
            stats[group]["correct"] += 1

    output: Dict[str, Dict[str, Any]] = {}
    for group in sorted(stats.keys()):
        total = stats[group]["total"]
        correct = stats[group]["correct"]
        output[group] = {
            "total": total,
            "correct": correct,
            "accuracy": round((correct / total) if total else 0.0, 4),
        }
    return output


def build_accuracy_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("is_correct")))
    overall_acc = (correct / total) if total else 0.0
    return {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": round(overall_acc, 4),
        },
        "task_type_accuracy": _compute_group_accuracy(rows, "query_type"),
        "memory_type_accuracy": _compute_group_accuracy(rows, "memory_type"),
    }


def default_metrics_output_path(output_json: str) -> str:
    if output_json.endswith(".json"):
        return f"{output_json[:-5]}_metrics.json"
    return f"{output_json}_metrics.json"


def print_accuracy_summary(summary: Dict[str, Any]) -> None:
    overall = summary.get("overall", {})
    print(
        f"Overall Accuracy: {overall.get('correct', 0)}/{overall.get('total', 0)}"
        f" = {overall.get('accuracy', 0.0):.4f}"
    )

    print("Task Type Accuracy:")
    for key, value in summary.get("task_type_accuracy", {}).items():
        print(f"  - {key}: {value.get('correct', 0)}/{value.get('total', 0)} = {value.get('accuracy', 0.0):.4f}")

    print("Memory Type Accuracy:")
    for key, value in summary.get("memory_type_accuracy", {}).items():
        print(f"  - {key}: {value.get('correct', 0)}/{value.get('total', 0)} = {value.get('accuracy', 0.0):.4f}")


def run_eval(
    input_json: str,
    output_json: str,
    metrics_output_json: Optional[str],
    model: str,
    api_key: str,
    limit: Optional[int],
    sleep_sec: float,
    num_jobs: int,
    job_id: int,
) -> None:
    meta, examples = load_examples(input_json)
    if limit is not None and limit > 0:
        examples = examples[:limit]

    for idx, ex in enumerate(examples):
        ex["_sample_index"] = idx
    total_after_limit = len(examples)
    examples = shard_examples(examples, num_jobs=num_jobs, job_id=job_id)

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    results = []
    correct = 0

    for i, ex in enumerate(examples, start=1):
        options = normalize_options(ex)
        question = _first_nonempty_string(ex, ("question", "query", "prompt", "instruction"))
        query_time = ex.get("query_time")
        query_type = infer_query_type(ex)
        memory_type = infer_memory_type(ex, query_type=query_type)
        prompt = build_prompt(question=question, query_time=query_time, options=options)
        t0 = time.time()
        error = None
        response_text = ""
        pred = ""

        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            response_text = (response.text or "").strip()
            pred = parse_choice(response_text, options) if options else response_text
        except Exception as e:
            error = str(e)

        gt_label, gt_raw = extract_ground_truth(ex, options)
        if options and gt_label:
            is_correct = bool(pred) and (pred.upper() == gt_label.upper())
        elif gt_raw is not None:
            is_correct = _normalize_text(pred) == _normalize_text(gt_raw)
        else:
            is_correct = False

        if is_correct:
            correct += 1

        row = {
            "sample_index": ex.get("_sample_index"),
            "p_id": ex.get("p_id"),
            "example_id": ex.get("example_id"),
            "identity": ex.get("identity"),
            "perspective": ex.get("perspective"),
            "query_time": ex.get("query_time"),
            "query_type": query_type,
            "memory_type": memory_type,
            "source_dataset": ex.get("source_dataset"),
            "question": ex.get("question"),
            "events": ex.get("events"),
            "evidence_timestamps": ex.get("evidence_timestamps"),
            "options": options,
            "correct_answer_raw": gt_raw,
            "correct_answer_label": gt_label,
            "pred": pred,
            "raw_response": response_text,
            "is_correct": is_correct,
            "error": error,
            "latency_sec": round(time.time() - t0, 3),
        }
        results.append(row)

        print(
            f"[job {job_id + 1}/{num_jobs}] [{i}/{len(examples)}] {ex.get('example_id')} "
            f"gt={gt_label or (gt_raw or 'N/A')} pred={pred or 'N/A'} ok={is_correct}"
            + (f" error={error}" if error else "")
        )
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    evaluated = len(results)
    accuracy = (correct / evaluated) if evaluated else 0.0
    summary = build_accuracy_summary(results)

    payload = {
        "task_type": meta.get("type"),
        "task_description": meta.get("description"),
        "input_file": input_json,
        "model": model,
        "num_jobs": num_jobs,
        "job_id": job_id,
        "total_after_limit_before_shard": total_after_limit,
        "evaluated_examples": evaluated,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "task_type_accuracy": summary.get("task_type_accuracy", {}),
        "memory_type_accuracy": summary.get("memory_type_accuracy", {}),
        "results": results,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    metrics_path = metrics_output_json or default_metrics_output_path(output_json)
    metrics_payload = {
        "input_file": input_json,
        "output_file": output_json,
        "model": model,
        "num_jobs": num_jobs,
        "job_id": job_id,
        "total_after_limit_before_shard": total_after_limit,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved results to: {output_json}")
    print(f"Saved metrics to: {metrics_path}")
    print_accuracy_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini text-only evaluation for generic benchmark query JSON."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="/nas-ssd2/ziyang/Memory_project/COLM/benchmark/filtered_batch_1/all_task_types_v2.json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/nas-ssd2/ziyang/Memory_project/COLM/evaluation/Gemini/results/filtered_batch_1/all_task_types_v2_text_only_gemini.json",
    )
    parser.add_argument(
        "--metrics_output_json",
        type=str,
        default=None,
        help="Optional path for metrics summary JSON. Defaults to <output_json>_metrics.json.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY", DEFAULT_API_KEY),
        help="Gemini API key. Defaults to GEMINI_API_KEY env var, then local fallback key.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N examples for quick checks.",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=0.0,
        help="Optional delay between requests to avoid rate limits.",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=1,
        help="Split examples into this many shards. Use with --job_id for parallel runs.",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Shard id in [0, num_jobs-1].",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Set --api_key or GEMINI_API_KEY.")
    if args.num_jobs < 1:
        raise ValueError("--num_jobs must be >= 1")
    if args.job_id < 0 or args.job_id >= args.num_jobs:
        raise ValueError("--job_id must satisfy 0 <= job_id < num_jobs")

    run_eval(
        input_json=args.input_json,
        output_json=args.output_json,
        metrics_output_json=args.metrics_output_json,
        model=args.model,
        api_key=args.api_key,
        limit=args.limit,
        sleep_sec=args.sleep_sec,
        num_jobs=args.num_jobs,
        job_id=args.job_id,
    )


if __name__ == "__main__":
    main()
