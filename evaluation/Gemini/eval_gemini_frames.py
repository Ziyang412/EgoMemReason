import argparse
import io
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from tqdm import tqdm


DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # set GEMINI_API_KEY or pass --api_key
DEFAULT_MODEL = "gemini-3-pro-preview"

QUERY_TIME_PATTERN = re.compile(r"^\s*DAY(\d+)\s*,\s*(\d{1,2}):(\d{2}):(\d{2})\s*$", re.IGNORECASE)


def parse_day_num(day_str: str) -> Optional[int]:
    if not day_str:
        return None
    s = str(day_str).strip().upper()
    if not s.startswith("DAY"):
        return None
    try:
        return int(s[3:])
    except ValueError:
        return None


def parse_query_time(query_time: str) -> Tuple[int, int]:
    """Parse 'DAY6, 13:11:38' -> (6, 13113800)."""
    m = QUERY_TIME_PATTERN.match(str(query_time))
    if not m:
        raise ValueError(f"Invalid query_time format: {query_time!r}")
    day_num = int(m.group(1))
    h = int(m.group(2))
    mi = int(m.group(3))
    sec = int(m.group(4))
    time_int = h * 1000000 + mi * 10000 + sec * 100
    return day_num, time_int


def time_int_to_seconds(time_int: int) -> float:
    s = f"{int(time_int):08d}"
    h = int(s[0:2])
    mi = int(s[2:4])
    sec = int(s[4:6])
    cs = int(s[6:8])
    return h * 3600 + mi * 60 + sec + cs / 100.0


def absolute_seconds(day_num: int, time_int: int) -> float:
    return (day_num - 1) * 24 * 3600 + time_int_to_seconds(time_int)


def uniform_sample(items: List[str], k: int) -> List[str]:
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return items
    if k == 1:
        return [items[0]]
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in idx]


def shard_examples(examples: List[Dict], num_jobs: int, job_id: int) -> List[Dict]:
    if num_jobs <= 1:
        return examples
    return [ex for i, ex in enumerate(examples) if i % num_jobs == job_id]


def atomic_json_dump(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def load_benchmark(
    path: str,
    default_query_type: Optional[str] = None,
) -> Tuple[Dict, List[Dict]]:
    with open(path, "r") as f:
        payload = json.load(f)

    examples: List[Dict] = []

    if isinstance(payload, list):
        # Current flat format: root is a list of samples.
        examples = [dict(x) for x in payload]
        meta = {
            "type": "TemporalOrdering",
            "description": "",
            "total_examples": len(examples),
        }
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        # Current flat format: {"samples": [...]} with optional metadata.
        examples = [dict(x) for x in payload["samples"]]
        meta = {
            "type": payload.get("type"),
            "description": payload.get("description", ""),
            "total_examples": payload.get("total_examples", len(examples)),
        }
    elif isinstance(payload, dict):
        # Legacy format with nested identities.
        task_description = (payload.get("description") or "").strip()
        for identity, rows in payload.get("identities", {}).items():
            for row in rows:
                item = dict(row)
                item["identity"] = item.get("identity", identity)
                query_time = str(item.get("query_time") or "").strip()
                if not item.get("p_id"):
                    item["p_id"] = build_pid(item["identity"], query_time)
                if not item.get("question"):
                    merged_question = merge_description_and_events(
                        task_description=task_description,
                        events=item.get("events"),
                    )
                    if merged_question:
                        item["question"] = merged_question
                        item["question_source"] = "merged_description_events"
                if "evidence_timestamps" not in item and item.get("event_times") is not None:
                    item["evidence_timestamps"] = item.get("event_times")
                examples.append(item)
        meta = {
            "type": payload.get("type"),
            "description": payload.get("description", ""),
            "total_examples": payload.get("total_examples", len(examples)),
        }
    else:
        raise ValueError("Unsupported dataset JSON format.")

    # Shared normalization to keep evaluator robust across schemas.
    for item in examples:
        if not item.get("identity"):
            raise ValueError(f"Missing identity in sample: {item}")
        query_time = str(item.get("query_time") or "").strip()
        if not item.get("p_id"):
            item["p_id"] = build_pid(item["identity"], query_time)
        if "evidence_timestamps" not in item and item.get("event_times") is not None:
            item["evidence_timestamps"] = item.get("event_times")
        # Datasets that use "answer" instead of "correct_answer" (e.g.
        # event_ordering_v2.json). Alias so downstream scoring works without
        # silently marking everything wrong.
        if not item.get("correct_answer") and item.get("answer"):
            item["correct_answer"] = item["answer"]
        # Fill in query_type when the dataset omits it. Lets the agentic planner
        # and the per-split hint allowlist function for single-task datasets.
        if default_query_type and not item.get("query_type"):
            item["query_type"] = default_query_type

    return meta, examples


def merge_description_and_events(task_description: str, events: object) -> str:
    lines: List[str] = []
    if task_description:
        lines.append(task_description)

    if isinstance(events, dict) and events:
        lines.append("Events:")
        for key in sorted(events.keys(), key=lambda x: str(x)):
            lines.append(f"{key}. {events[key]}")

    return "\n".join(lines).strip()


def normalize_query_time_for_pid(query_time: str) -> str:
    s = str(query_time or "").strip().upper()
    s = s.replace(",", "")
    s = s.replace(":", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "UNKNOWN_TIME"


def build_pid(identity: str, query_time: str) -> str:
    return f"{identity}_{normalize_query_time_for_pid(query_time)}"


def load_frames_index(index_path: str) -> Dict[str, List[Dict]]:
    with open(index_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("frames index must be a dict keyed by identity")
    return data


def select_frames_before_query(
    frames_by_identity: Dict[str, List[Dict]],
    identity: str,
    query_day_num: int,
    query_time_int: int,
    max_frames: int,
    max_hours_before_query: Optional[float],
) -> List[str]:
    entries = frames_by_identity.get(identity, [])
    if not entries:
        return []

    target_seconds = absolute_seconds(query_day_num, query_time_int)
    selected: List[Tuple[int, int, str]] = []

    for ent in entries:
        day_num = parse_day_num(ent.get("day", ""))
        if day_num is None:
            continue
        t = ent.get("time")
        if t is None:
            continue
        try:
            time_int = int(t)
        except (TypeError, ValueError):
            continue

        # Keep frames up to and including query time.
        if day_num > query_day_num:
            continue
        if day_num == query_day_num and time_int > query_time_int:
            continue

        if max_hours_before_query is not None and max_hours_before_query > 0:
            frame_seconds = absolute_seconds(day_num, time_int)
            if target_seconds - frame_seconds > max_hours_before_query * 3600:
                continue

        p = ent.get("path")
        if isinstance(p, str) and p:
            selected.append((day_num, time_int, p))

    if not selected:
        return []

    # Ensure strict chronological order even if the index has occasional local disorder.
    selected.sort(key=lambda x: (x[0], x[1]))
    ordered_paths = [p for _, _, p in selected]

    if max_frames is None or max_frames <= 0:
        sampled = ordered_paths
    else:
        sampled = uniform_sample(ordered_paths, max_frames)
    # Keep only valid existing files.
    return [p for p in sampled if os.path.exists(p)]


def _collect_option_lines(example: Dict) -> List[str]:
    """Support multiple option schemas: dict(A->text), list[{id,text}], list[str], choice_a..choice_f."""
    lines: List[str] = []

    options = example.get("options")
    if isinstance(options, dict):
        for key in ["A", "B", "C", "D", "E", "F"]:
            if key in options:
                lines.append(f"{key}. {options[key]}")
    elif isinstance(options, list):
        for i, opt in enumerate(options):
            if isinstance(opt, dict):
                opt_id = opt.get("id")
                opt_text = opt.get("text")
                if opt_id and opt_text:
                    lines.append(f"{opt_id}. {opt_text}")
            elif isinstance(opt, str):
                lines.append(opt)

    for letter in ["A", "B", "C", "D", "E", "F"]:
        k = f"choice_{letter.lower()}"
        v = example.get(k)
        if v:
            lines.append(f"{letter}. {v}")

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for x in lines:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped


def sanitize_example_for_prompt(example: Dict) -> Dict:
    """Allowlist-only prompt payload to avoid leaking supervision fields."""
    safe: Dict = {
        "question": example.get("question"),
        "query": example.get("query"),
        "query_time": example.get("query_time"),
        "options": example.get("options"),
    }
    for letter in ["A", "B", "C", "D", "E", "F"]:
        k = f"choice_{letter.lower()}"
        if k in example:
            safe[k] = example.get(k)
    return safe


EPISODIC_QUERY_TYPES = {
    "temporal_ordering",
    "temporal_reasoning",
    "state_tracking",
    "multi_entity",
}


def infer_memory_type(example: Dict) -> str:
    for k in ["memory_type", "memory", "memory_category"]:
        v = example.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()

    query_type = str(example.get("query_type") or "").strip().lower()
    if query_type == "semantic_event":
        return "semantic"
    if query_type in EPISODIC_QUERY_TYPES:
        return "episodic"
    return "unknown"


def _compute_group_accuracy(results: List[Dict], key: str) -> Dict[str, Dict]:
    grouped: Dict[str, Dict[str, float]] = {}
    for row in results:
        g = str(row.get(key) or "unknown").strip() or "unknown"
        bucket = grouped.setdefault(g, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if bool(row.get("is_correct")):
            bucket["correct"] += 1

    out: Dict[str, Dict] = {}
    for g in sorted(grouped.keys()):
        total = int(grouped[g]["total"])
        correct = int(grouped[g]["correct"])
        out[g] = {
            "total": total,
            "correct": correct,
            "accuracy": round((correct / total), 4) if total else 0.0,
        }
    return out


def build_accuracy_summary(results: List[Dict]) -> Dict:
    total = len(results)
    correct = sum(1 for r in results if bool(r.get("is_correct")))
    return {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": round((correct / total), 4) if total else 0.0,
        },
        "by_task_type": _compute_group_accuracy(results, "query_type"),
        "by_memory_type": _compute_group_accuracy(results, "memory_type"),
    }


def summary_output_path(output_path: str) -> str:
    if output_path.lower().endswith(".json"):
        return output_path[:-5] + "_summary.json"
    return f"{output_path}_summary.json"


def build_summary_payload(base_payload: Dict) -> Dict:
    return {
        "dataset_path": base_payload.get("dataset_path"),
        "model": base_payload.get("model"),
        "max_frames": base_payload.get("max_frames"),
        "num_jobs": base_payload.get("num_jobs"),
        "job_id": base_payload.get("job_id"),
        "is_complete": base_payload.get("is_complete"),
        "summary": base_payload.get("summary", {}),
    }


def build_prompt(example: Dict, num_frames: int) -> str:
    question = (example.get("question") or example.get("query") or "").strip()
    query_time = (example.get("query_time") or "").strip()

    context_lines: List[str] = []
    if query_time:
        context_lines.append(f"Query time reference: {query_time}")

    if not question:
        question = "Please answer the query using the provided visual evidence."

    option_lines = _collect_option_lines(example)
    answer_instruction = (
        "This is a multiple-choice question. Return exactly one option letter from the listed choices "
        "(for example: A). Do not output any explanation, words, punctuation, or extra characters."
    )

    sections = [
        "You are assisting with EgoLife egocentric video QA.",
        "The frames come from one person's first-person daily life record and are in chronological order.",
        "The video starts from Day 1 at around 11:00:00.",
        f"I am giving you {num_frames} uniformly sampled video frames.",
        "",
        f"User query: {question}",
    ]

    if context_lines:
        sections.extend(["", "Additional context:", *context_lines])
    sections.extend(["", "Options:", *option_lines])

    sections.extend(["", answer_instruction])
    return "\n".join(sections)


def mime_for_path(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    if ext == ".bmp":
        return "image/bmp"
    return "image/jpeg"


def _normalize_text(text: object) -> str:
    return re.sub(r"[\W_]+", "", str(text).lower())


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


def extract_option_letter(text: str, options: object = None) -> str:
    labels: List[str] = []
    if isinstance(options, dict):
        labels = [str(k).strip().upper() for k in options.keys() if str(k).strip()]
    if not labels:
        labels = ["A", "B", "C", "D", "E", "F"]

    by_label = _extract_label(text, labels)
    if by_label:
        return by_label

    # Fallback for non-compliant outputs that return option content instead of label.
    if isinstance(options, dict) and options:
        normalized_raw = _normalize_text(text)
        if not normalized_raw:
            return ""

        exact_matches: List[str] = []
        contains_matches: List[str] = []
        for label, option_text in options.items():
            upper_label = str(label).strip().upper()
            if not upper_label:
                continue
            normalized_option = _normalize_text(option_text)
            if not normalized_option:
                continue
            if normalized_option == normalized_raw:
                exact_matches.append(upper_label)
            if normalized_option in normalized_raw:
                contains_matches.append(upper_label)

        exact_unique = sorted(set(exact_matches))
        if len(exact_unique) == 1:
            return exact_unique[0]

        contains_unique = sorted(set(contains_matches))
        if len(contains_unique) == 1:
            return contains_unique[0]

    return ""


def ask_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    frame_paths: List[str],
    frame_size: int,
) -> str:
    parts: List[types.Part] = [types.Part(text=prompt)]

    def _frame_bytes_for_model(path: str) -> Tuple[bytes, str]:
        if frame_size is None or frame_size <= 0:
            with open(path, "rb") as f:
                return f.read(), mime_for_path(path)

        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "Pillow is required for frame resizing. Install with `pip install Pillow`."
            ) from e

        with Image.open(path) as img:
            img = img.convert("RGB")
            if hasattr(Image, "Resampling"):
                resample = Image.Resampling.BILINEAR
            else:
                resample = Image.BILINEAR
            img = img.resize((frame_size, frame_size), resample)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue(), "image/jpeg"

    for path in frame_paths:
        data, mime_type = _frame_bytes_for_model(path)
        parts.append(
            types.Part.from_bytes(
                data=data,
                mime_type=mime_type,
            )
        )

    config = types.GenerateContentConfig(
        temperature=0.0,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    )
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=config,
    )
    return (response.text or "").strip()


def evaluate(
    dataset_path: str,
    frames_index_path: str,
    output_path: str,
    model: str,
    api_key: str,
    max_frames: int,
    frame_size: int,
    max_hours_before_query: Optional[float],
    limit: Optional[int],
    num_jobs: int,
    job_id: int,
    save_every: int,
    sleep_sec: float,
    save_frame_paths: bool,
    default_query_type: Optional[str] = None,
) -> None:
    meta, examples = load_benchmark(dataset_path, default_query_type=default_query_type)
    if limit is not None and limit > 0:
        examples = examples[:limit]
    total_after_limit = len(examples)
    examples = shard_examples(examples, num_jobs=num_jobs, job_id=job_id)

    frames_by_identity = load_frames_index(frames_index_path)
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    results: List[Dict] = []
    n_correct = 0

    def make_payload(is_complete: bool) -> Dict:
        total = len(results)
        acc = (n_correct / total) if total else 0.0
        payload = {
            "task_type": meta.get("type"),
            "task_description": meta.get("description"),
            "dataset_path": dataset_path,
            "frames_index_path": frames_index_path,
            "model": model,
            "max_frames": max_frames,
            "frame_size": frame_size,
            "max_hours_before_query": max_hours_before_query,
            "num_jobs": num_jobs,
            "job_id": job_id,
            "total_after_limit_before_shard": total_after_limit,
            "save_every": save_every,
            "is_complete": is_complete,
            "total": total,
            "correct": n_correct,
            "accuracy": round(acc, 4),
            "results": results,
        }
        payload["summary"] = build_accuracy_summary(results)
        return payload

    desc = f"Evaluating job {job_id + 1}/{num_jobs}" if num_jobs > 1 else "Evaluating"
    iterator = tqdm(
        enumerate(examples, start=1),
        total=len(examples),
        desc=desc,
        position=job_id if num_jobs > 1 else 0,
        dynamic_ncols=True,
        leave=True,
    )
    for idx, ex in iterator:
        sample_start = time.time()
        ex_id = ex.get("example_id")
        identity = ex.get("identity")
        gt = ex.get("correct_answer", "")
        pred = ""
        raw_response = ""
        error = None
        api_latency_sec = None

        try:
            query_day_num, query_time_int = parse_query_time(ex.get("query_time", ""))
            frame_paths = select_frames_before_query(
                frames_by_identity=frames_by_identity,
                identity=identity,
                query_day_num=query_day_num,
                query_time_int=query_time_int,
                max_frames=max_frames,
                max_hours_before_query=max_hours_before_query,
            )
            if not frame_paths:
                raise ValueError(
                    f"No frames found for identity={identity} up to query_time={ex.get('query_time')}"
                )

            prompt_example = sanitize_example_for_prompt(ex)
            prompt = build_prompt(prompt_example, num_frames=len(frame_paths))
            api_start = time.time()
            raw_response = ask_gemini(
                client=client,
                model=model,
                prompt=prompt,
                frame_paths=frame_paths,
                frame_size=frame_size,
            )
            api_latency_sec = round(time.time() - api_start, 3)
            pred = extract_option_letter(raw_response, ex.get("options"))
        except Exception as e:
            error = str(e)
            frame_paths = []

        response_time_sec = round(time.time() - sample_start, 3)
        is_correct = bool(pred) and pred == gt
        if is_correct:
            n_correct += 1

        row = {
            "p_id": ex.get("p_id"),
            "example_id": ex_id,
            "identity": identity,
            "query_time": ex.get("query_time"),
            "query_type": ex.get("query_type"),
            "memory_type": infer_memory_type(ex),
            "question": ex.get("question"),
            "question_events": ex.get("events"),
            "evidence_timestamps": ex.get("evidence_timestamps"),
            "options": ex.get("options"),
            "correct_answer": gt,
            "pred": pred,
            "is_correct": is_correct,
            "raw_response": raw_response,
            "n_input_frames": len(frame_paths),
            "response_time_sec": response_time_sec,
            "api_latency_sec": api_latency_sec,
            "error": error,
        }
        if save_frame_paths:
            row["frame_paths"] = frame_paths
        results.append(row)

        if save_every > 0 and len(results) % save_every == 0:
            partial_payload = make_payload(is_complete=False)
            atomic_json_dump(output_path, partial_payload)
            atomic_json_dump(summary_output_path(output_path), build_summary_payload(partial_payload))
            tqdm.write(
                f"[job {job_id + 1}/{num_jobs}] autosaved {len(results)} examples -> {output_path}"
            )

        tqdm.write(
            f"[job {job_id + 1}/{num_jobs}] [{idx}/{len(examples)}] "
            f"{ex_id} gt={gt} pred={pred or 'N/A'} ok={is_correct} "
            f"response_time={response_time_sec}s api_time={api_latency_sec if api_latency_sec is not None else 'N/A'}s"
        )
        iterator.set_postfix_str(f"acc={n_correct}/{idx}")
        if error:
            tqdm.write(
                f"[job {job_id + 1}/{num_jobs}] ERROR {ex_id} "
                f"frames={len(frame_paths)} err={error}"
            )
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    total = len(results)
    acc = (n_correct / total) if total else 0.0

    final_payload = make_payload(is_complete=True)
    atomic_json_dump(output_path, final_payload)
    summary_path = summary_output_path(output_path)
    atomic_json_dump(summary_path, build_summary_payload(final_payload))

    print(f"\nSaved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(
        f"Job {job_id + 1}/{num_jobs} Accuracy: {n_correct}/{total} = {acc:.4f} "
        f"(evaluated {total} shard examples from {total_after_limit} total)"
    )
    summary = final_payload["summary"]
    print(f"Overall accuracy: {summary['overall']['correct']}/{summary['overall']['total']} = {summary['overall']['accuracy']:.4f}")
    print("Task type accuracy:")
    for task_name, stats in summary["by_task_type"].items():
        print(
            f"  - {task_name}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}"
        )
    print("Memory type accuracy:")
    for mem_name, stats in summary["by_memory_type"].items():
        print(
            f"  - {mem_name}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini frame-based VideoQA using EgoLife frame index."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/nas-ssd2/ziyang/Memory_project/COLM/benchmark/raw_files_batch1/multi_entity/hard_multi_entity_Dylan_merged.json",
    )
    parser.add_argument(
        "--frames_index",
        type=str,
        default="/nas-ssd2/video_datasets/EgoLife/egolife_frames_index.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/nas-ssd2/ziyang/Memory_project/COLM/evaluation/Gemini/results/hard_multi_entity_Dylan_merged_gemini_frames.json",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY", DEFAULT_API_KEY),
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1024,
        help="Maximum number of frames sent to Gemini per example. <=0 uses all frames before query_time.",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        default=384,
        help="Resize each frame to frame_size x frame_size before API upload. <=0 disables resizing.",
    )
    parser.add_argument(
        "--max_hours_before_query",
        type=float,
        default=None,
        help="If set, only keep frames within this many hours before query_time.",
    )
    parser.add_argument("--limit", type=int, default=None)
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
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Autosave output JSON every N processed examples per job. 0 disables periodic autosave.",
    )
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument(
        "--save_frame_paths",
        action="store_true",
        help="Include selected frame paths in output JSON.",
    )
    parser.add_argument(
        "--default_query_type",
        type=str,
        default=None,
        help=(
            "Fill in query_type when the dataset omits it (e.g. event_ordering_v2.json "
            "where every sample has query_type=null). Set to 'Event Ordering' for that file."
        ),
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Set GEMINI_API_KEY or pass --api_key.")
    if args.num_jobs < 1:
        raise ValueError("--num_jobs must be >= 1")
    if args.job_id < 0 or args.job_id >= args.num_jobs:
        raise ValueError("--job_id must satisfy 0 <= job_id < num_jobs")
    if args.save_every < 0:
        raise ValueError("--save_every must be >= 0")

    evaluate(
        dataset_path=args.dataset,
        frames_index_path=args.frames_index,
        output_path=args.output,
        model=args.model,
        api_key=args.api_key,
        max_frames=args.max_frames,
        frame_size=args.frame_size,
        max_hours_before_query=args.max_hours_before_query,
        limit=args.limit,
        num_jobs=args.num_jobs,
        job_id=args.job_id,
        save_every=args.save_every,
        sleep_sec=args.sleep_sec,
        save_frame_paths=args.save_frame_paths,
        default_query_type=args.default_query_type,
    )


if __name__ == "__main__":
    main()
