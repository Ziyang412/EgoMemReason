import argparse
import base64
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


DEFAULT_AZURE_ENDPOINT = "https://YOUR-RESOURCE.cognitiveservices.azure.com/"
DEFAULT_DEPLOYMENT = "gpt-5.2-chat"
DEFAULT_API_VERSION = "2024-12-01-preview"
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


def append_jsonl(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        json.dump(payload, f)
        f.write("\n")


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


def load_benchmark(path: str) -> Tuple[Dict, List[Dict]]:
    with open(path, "r") as f:
        payload = json.load(f)

    examples: List[Dict] = []

    if isinstance(payload, list):
        examples = [dict(x) for x in payload]
        meta = {
            "type": "TemporalOrdering",
            "description": "",
            "total_examples": len(examples),
        }
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        examples = [dict(x) for x in payload["samples"]]
        meta = {
            "type": payload.get("type"),
            "description": payload.get("description", ""),
            "total_examples": payload.get("total_examples", len(examples)),
        }
    elif isinstance(payload, dict):
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

    for item in examples:
        if not item.get("identity"):
            raise ValueError(f"Missing identity in sample: {item}")
        query_time = str(item.get("query_time") or "").strip()
        if not item.get("p_id"):
            item["p_id"] = build_pid(item["identity"], query_time)
        if "evidence_timestamps" not in item and item.get("event_times") is not None:
            item["evidence_timestamps"] = item.get("event_times")

    return meta, examples


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
    selected: List[str] = []

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

        if day_num > query_day_num:
            continue
        if day_num == query_day_num and time_int >= query_time_int:
            continue

        if max_hours_before_query is not None and max_hours_before_query > 0:
            frame_seconds = absolute_seconds(day_num, time_int)
            if target_seconds - frame_seconds > max_hours_before_query * 3600:
                continue

        p = ent.get("path")
        if isinstance(p, str) and p:
            selected.append(p)

    if not selected:
        return []
    sampled = uniform_sample(selected, max_frames)
    return [p for p in sampled if os.path.exists(p)]


def _collect_option_lines(example: Dict) -> List[str]:
    lines: List[str] = []

    options = example.get("options")
    if isinstance(options, dict):
        for key in sorted(options.keys(), key=lambda x: str(x)):
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

    for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        k = f"choice_{letter.lower()}"
        v = example.get(k)
        if v:
            lines.append(f"{letter}. {v}")

    seen = set()
    deduped = []
    for x in lines:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped


def sanitize_example_for_prompt(example: Dict) -> Dict:
    safe: Dict = {
        "question": example.get("question"),
        "query": example.get("query"),
        "query_time": example.get("query_time"),
        "options": example.get("options"),
    }
    for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
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
    if option_lines:
        answer_instruction = "If options are provided, output only the option letter (for example: A)."
    else:
        answer_instruction = "If no options are provided, answer concisely in 1-2 sentences."

    sections = [
        "You are assisting with EgoLife egocentric video QA.",
        "The images come from one person's first-person daily life record and are in chronological order.",
        f"I am giving you {num_frames} uniformly sampled images.",
        "Use only the provided images and do not assume unseen events.",
        "",
        f"User query: {question}",
    ]

    if context_lines:
        sections.extend(["", "Additional context:", *context_lines])
    if option_lines:
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


def encode_image_data_url(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_for_path(path)};base64,{b64}"


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
    raw = re.sub(r"[`*_]", "", raw)
    label_set = set(normalized_labels)
    if raw in label_set:
        return raw

    label_alt = "|".join(re.escape(label) for label in normalized_labels)
    lines = [line.strip().upper() for line in raw.splitlines() if line.strip()]

    for line in reversed(lines[-8:]):
        if line in label_set:
            return line

        m = re.match(rf"^\(?\s*({label_alt})\s*\)?[.)]?$", line)
        if m:
            return m.group(1)

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
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

    by_label = _extract_label(text, labels)
    if by_label:
        return by_label

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


def make_client(api_key: str, azure_endpoint: str, api_version: str) -> Dict[str, Any]:
    if AzureOpenAI is not None:
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return {"kind": "v1", "client": client}

    import openai

    openai.api_type = "azure"
    openai.api_base = azure_endpoint
    openai.api_version = api_version
    openai.api_key = api_key
    return {"kind": "legacy", "client": openai}


def extract_text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value is not None and str(text_value).strip():
                    parts.append(str(text_value).strip())
                elif item.get("content") is not None and str(item.get("content")).strip():
                    parts.append(str(item.get("content")).strip())
                continue
            text_attr = getattr(item, "text", None)
            if text_attr is not None and str(text_attr).strip():
                parts.append(str(text_attr).strip())
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        if content.get("text") is not None:
            return str(content.get("text")).strip()
        if content.get("content") is not None:
            return str(content.get("content")).strip()

    return ""


def _clip_for_log(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    s = str(text or "").replace("\n", "\\n").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "...<truncated>"


def ask_gpt5(
    client_bundle: Dict[str, Any],
    deployment: str,
    prompt: str,
    frame_paths: List[str],
    max_completion_tokens: int,
    image_detail: str,
) -> Tuple[str, Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in frame_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_data_url(path),
                    "detail": image_detail,
                },
            }
        )

    messages = [
        {
            "role": "system",
            "content": "You are a careful visual QA assistant for multiple-choice benchmark evaluation.",
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    kind = client_bundle.get("kind")
    if kind == "v1":
        client = client_bundle["client"]
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        meta: Dict[str, Any] = {}
        meta["response_id"] = getattr(response, "id", None)
        meta["response_model"] = getattr(response, "model", None)
        if response.choices and response.choices[0].message:
            choice0 = response.choices[0]
            meta["finish_reason"] = getattr(choice0, "finish_reason", None)
            message = choice0.message
            meta["refusal"] = getattr(message, "refusal", None) if message is not None else None
            text = extract_text_from_content(message.content if message is not None else None)
            if not text:
                try:
                    payload = response.model_dump()
                    c0 = (payload.get("choices") or [{}])[0]
                    meta["content_filter_results"] = c0.get("content_filter_results")
                    meta["prompt_filter_results"] = payload.get("prompt_filter_results")
                except Exception:
                    pass
            return text, meta
        meta["no_choices"] = True
        return "", meta

    if kind == "legacy":
        openai = client_bundle["client"]
        response = openai.ChatCompletion.create(
            engine=deployment,
            messages=messages,
            max_tokens=max_completion_tokens,
        )
        try:
            meta = {
                "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            }
            return extract_text_from_content(response["choices"][0]["message"]["content"]), meta
        except Exception:
            return "", {"legacy_parse_error": True}

    raise RuntimeError(f"Unsupported client kind: {kind}")


def evaluate(
    dataset_path: str,
    frames_index_path: str,
    output_path: str,
    deployment: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    max_frames: int,
    max_hours_before_query: Optional[float],
    max_completion_tokens: int,
    image_detail: str,
    limit: Optional[int],
    num_jobs: int,
    job_id: int,
    save_every: int,
    sleep_sec: float,
    save_frame_paths: bool,
    dry_run: bool,
    debug_print_missing_pred: bool,
    debug_print_error_response: bool,
    debug_max_chars: int,
    events_jsonl: Optional[str],
) -> None:
    meta, examples = load_benchmark(dataset_path)
    if limit is not None and limit > 0:
        examples = examples[:limit]
    total_after_limit = len(examples)
    examples = shard_examples(examples, num_jobs=num_jobs, job_id=job_id)

    frames_by_identity = load_frames_index(frames_index_path)
    client_bundle = None if dry_run else make_client(api_key, azure_endpoint, api_version)

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
            "model": deployment,
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
            "max_frames": max_frames,
            "max_hours_before_query": max_hours_before_query,
            "max_completion_tokens": max_completion_tokens,
            "image_detail": image_detail,
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

    append_jsonl(
        events_jsonl,
        {
            "event_type": "run_start",
            "ts": datetime.utcnow().isoformat() + "Z",
            "dataset_path": dataset_path,
            "frames_index_path": frames_index_path,
            "model": deployment,
            "num_jobs": num_jobs,
            "job_id": job_id,
            "max_frames": max_frames,
            "limit": limit,
            "dry_run": dry_run,
        },
    )

    for idx, ex in iterator:
        sample_start = time.time()
        ex_id = ex.get("example_id")
        identity = ex.get("identity")
        gt = ex.get("correct_answer", "")
        pred = ""
        raw_response = ""
        response_meta: Dict[str, Any] = {}
        error = None
        api_latency_sec = None
        frame_paths: List[str] = []

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
                    f"No frames found for identity={identity} before query_time={ex.get('query_time')}"
                )

            prompt_example = sanitize_example_for_prompt(ex)
            prompt = build_prompt(prompt_example, num_frames=len(frame_paths))
            api_start = time.time()
            if dry_run:
                labels = []
                options = ex.get("options")
                if isinstance(options, dict):
                    labels = [str(k).strip().upper() for k in options.keys() if str(k).strip()]
                raw_response = labels[0] if labels else "A"
                response_meta = {"dry_run": True}
            else:
                raw_response, response_meta = ask_gpt5(
                    client_bundle=client_bundle,
                    deployment=deployment,
                    prompt=prompt,
                    frame_paths=frame_paths,
                    max_completion_tokens=max_completion_tokens,
                    image_detail=image_detail,
                )
            api_latency_sec = round(time.time() - api_start, 3)
            pred = extract_option_letter(raw_response, ex.get("options"))
            if not raw_response and not pred:
                finish_reason = response_meta.get("finish_reason")
                if finish_reason is not None:
                    error = f"empty_model_output (finish_reason={finish_reason})"
                else:
                    error = "empty_model_output"
        except Exception as e:
            error = str(e)

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
            "response_meta": response_meta,
            "n_input_frames": len(frame_paths),
            "response_time_sec": response_time_sec,
            "api_latency_sec": api_latency_sec,
            "error": error,
            "dry_run": dry_run,
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
            append_jsonl(
                events_jsonl,
                {
                    "event_type": "error",
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "job_id": job_id,
                    "num_jobs": num_jobs,
                    "sample_index_in_shard": idx,
                    "example_id": ex_id,
                    "p_id": ex.get("p_id"),
                    "identity": identity,
                    "query_type": ex.get("query_type"),
                    "memory_type": infer_memory_type(ex),
                    "query_time": ex.get("query_time"),
                    "n_input_frames": len(frame_paths),
                    "correct_answer": gt,
                    "pred": pred,
                    "response_time_sec": response_time_sec,
                    "api_latency_sec": api_latency_sec,
                    "error": error,
                    "raw_response_snippet": _clip_for_log(raw_response, debug_max_chars),
                    "response_meta": response_meta,
                },
            )
            if debug_print_error_response:
                snippet = _clip_for_log(raw_response, debug_max_chars)
                if snippet:
                    tqdm.write(
                        f"[job {job_id + 1}/{num_jobs}] ERROR_RAW {ex_id} raw_response={snippet}"
                    )
        elif not pred and debug_print_missing_pred:
            snippet = _clip_for_log(raw_response, debug_max_chars)
            tqdm.write(
                f"[job {job_id + 1}/{num_jobs}] MISSING_PRED {ex_id} "
                f"frames={len(frame_paths)} raw_response={snippet or '<empty>'}"
            )
            append_jsonl(
                events_jsonl,
                {
                    "event_type": "missing_pred",
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "job_id": job_id,
                    "num_jobs": num_jobs,
                    "sample_index_in_shard": idx,
                    "example_id": ex_id,
                    "p_id": ex.get("p_id"),
                    "identity": identity,
                    "query_type": ex.get("query_type"),
                    "memory_type": infer_memory_type(ex),
                    "query_time": ex.get("query_time"),
                    "n_input_frames": len(frame_paths),
                    "correct_answer": gt,
                    "pred": pred,
                    "response_time_sec": response_time_sec,
                    "api_latency_sec": api_latency_sec,
                    "raw_response_snippet": snippet,
                    "response_meta": response_meta,
                },
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
    print(
        "Overall accuracy: "
        f"{summary['overall']['correct']}/{summary['overall']['total']} = {summary['overall']['accuracy']:.4f}"
    )
    print("Task type accuracy:")
    for task_name, stats in summary["by_task_type"].items():
        print(f"  - {task_name}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")
    print("Memory type accuracy:")
    for mem_name, stats in summary["by_memory_type"].items():
        print(f"  - {mem_name}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

    append_jsonl(
        events_jsonl,
        {
            "event_type": "run_end",
            "ts": datetime.utcnow().isoformat() + "Z",
            "job_id": job_id,
            "num_jobs": num_jobs,
            "evaluated_examples": total,
            "correct": n_correct,
            "accuracy": round(acc, 4),
            "output_path": output_path,
            "summary_path": summary_path,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-5 frame-based VideoQA using EgoLife frame index."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/nas-ssd2/ziyang/Memory_project/COLM/benchmark/filtered_batch_1/all_task_types_v2.json",
    )
    parser.add_argument(
        "--frames_index",
        type=str,
        default="/nas-ssd2/video_datasets/EgoLife/egolife_frames_index.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/nas-ssd2/ziyang/Memory_project/COLM/evaluation/GPT5/results/all_v2/all_v2_gpt5_frames_50_single.json",
    )
    parser.add_argument(
        "--deployment",
        "--model",
        dest="deployment",
        type=str,
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", DEFAULT_DEPLOYMENT),
        help="Azure deployment name for GPT-5 chat model.",
    )
    parser.add_argument(
        "--azure_endpoint",
        type=str,
        default=os.environ.get("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=50,
        help="Maximum number of images sent to GPT-5 per example.",
    )
    parser.add_argument(
        "--max_hours_before_query",
        type=float,
        default=None,
        help="If set, only keep frames within this many hours before query_time.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=2048,
        help="Max completion tokens for GPT-5 response.",
    )
    parser.add_argument(
        "--image_detail",
        type=str,
        default="low",
        choices=["low", "high", "auto"],
        help="Vision detail setting for each image_url.",
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
        "--dry_run",
        action="store_true",
        help="Do not call Azure OpenAI. Emit deterministic placeholder predictions for pipeline testing.",
    )
    parser.add_argument(
        "--debug_print_missing_pred",
        action="store_true",
        help="Print sample id and raw model output snippet when prediction label cannot be parsed.",
    )
    parser.add_argument(
        "--debug_print_error_response",
        action="store_true",
        help="Print raw model output snippet when request returns an error.",
    )
    parser.add_argument(
        "--debug_max_chars",
        type=int,
        default=600,
        help="Max raw output characters shown in debug print lines.",
    )
    parser.add_argument(
        "--events_jsonl",
        type=str,
        default=None,
        help="If set, append structured per-sample error/missing_pred/run events to this JSONL file.",
    )
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        raise ValueError("No API key provided. Set AZURE_OPENAI_API_KEY or pass --api_key.")
    if args.num_jobs < 1:
        raise ValueError("--num_jobs must be >= 1")
    if args.job_id < 0 or args.job_id >= args.num_jobs:
        raise ValueError("--job_id must satisfy 0 <= job_id < num_jobs")
    if args.save_every < 0:
        raise ValueError("--save_every must be >= 0")
    if args.max_frames < 1:
        raise ValueError("--max_frames must be >= 1")
    if args.max_completion_tokens < 1:
        raise ValueError("--max_completion_tokens must be >= 1")
    if args.debug_max_chars < 0:
        raise ValueError("--debug_max_chars must be >= 0")

    evaluate(
        dataset_path=args.dataset,
        frames_index_path=args.frames_index,
        output_path=args.output,
        deployment=args.deployment,
        api_key=args.api_key,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        max_frames=args.max_frames,
        max_hours_before_query=args.max_hours_before_query,
        max_completion_tokens=args.max_completion_tokens,
        image_detail=args.image_detail,
        limit=args.limit,
        num_jobs=args.num_jobs,
        job_id=args.job_id,
        save_every=args.save_every,
        sleep_sec=args.sleep_sec,
        save_frame_paths=args.save_frame_paths,
        dry_run=args.dry_run,
        debug_print_missing_pred=args.debug_print_missing_pred,
        debug_print_error_response=args.debug_print_error_response,
        debug_max_chars=args.debug_max_chars,
        events_jsonl=args.events_jsonl,
    )


if __name__ == "__main__":
    main()
