"""Top-level evaluator for the agentic Gemini pipeline.

Mirrors evaluation/Gemini/eval_gemini_frames.py argument surface so that the
existing run-script template and merge_temporal_ordering_eval_shards.py work
unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

from google import genai
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GEMINI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "Gemini"))
if _GEMINI_DIR not in sys.path:
    sys.path.insert(0, _GEMINI_DIR)

from eval_gemini_frames import (  # noqa: E402
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    atomic_json_dump,
    build_accuracy_summary,
    build_summary_payload,
    extract_option_letter,
    infer_memory_type,
    load_benchmark,
    shard_examples,
    summary_output_path,
)

from agent_loop import AgenticEvaluator  # noqa: E402

import json  # noqa: E402


def load_frames_index(index_path: str) -> Dict[str, List[Dict]]:
    with open(index_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("frames index must be a dict keyed by identity")
    return data


def evaluate(
    *,
    dataset_path: str,
    frames_index_path: str,
    output_path: str,
    model: str,
    planner_model: Optional[str],
    observer_model: Optional[str],
    reflector_model: Optional[str],
    synthesizer_model: Optional[str],
    api_key: str,
    max_frames: int,
    frame_size: int,
    max_rounds: int,
    total_frame_budget: int,
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

    agent = AgenticEvaluator(
        client=client,
        planner_model=planner_model or model,
        observer_model=observer_model or model,
        reflector_model=reflector_model or model,
        synthesizer_model=synthesizer_model or model,
        max_rounds=max_rounds,
        default_max_frames=max_frames,
        default_frame_size=frame_size,
        total_frame_budget=total_frame_budget,
    )

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
            "planner_model": agent.planner_model,
            "observer_model": agent.observer_model,
            "reflector_model": agent.reflector_model,
            "synthesizer_model": agent.synthesizer_model,
            "max_frames": max_frames,
            "frame_size": frame_size,
            "max_rounds": max_rounds,
            "total_frame_budget": total_frame_budget,
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
        agent_result: Dict = {}
        error = None

        try:
            agent_result = agent.run(example=ex, frames_by_identity=frames_by_identity)
            pred = agent_result.get("pred") or ""
            if not pred:
                # Fall back to the existing letter-extraction parser on the raw
                # synthesizer output, in case Gemini ignored the JSON contract.
                pred = extract_option_letter(
                    agent_result.get("synthesizer_raw", ""), ex.get("options")
                )
        except Exception as e:
            error = str(e)

        response_time_sec = round(time.time() - sample_start, 3)
        is_correct = bool(pred) and pred == gt
        if is_correct:
            n_correct += 1

        trace = agent_result.get("agent_trace", [])
        # Strip the largest raw fields from the persisted trace if save_frame_paths
        # is off, to keep output JSON manageable on the full 500-example run.
        compact_trace: List[Dict] = []
        for entry in trace:
            compact_entry = {
                "round": entry["round"],
                "plan": {
                    "load_mode": entry["plan"]["load_mode"],
                    "regions": entry["plan"].get("raw_regions") or entry["plan"]["regions"],
                    "max_frames": entry["plan"]["max_frames"],
                    "max_frames_requested": entry["plan"].get("max_frames_requested"),
                    "budget_clamped": entry["plan"].get("budget_clamped", False),
                    "frame_size": entry["plan"]["frame_size"],
                    "focus": entry["plan"]["focus"],
                    "reasoning": entry["plan"].get("reasoning"),
                },
                "n_frames_used": entry["n_frames_used"],
                "frames_used_total_after_round": entry.get("frames_used_total_after_round"),
                "remaining_budget_after_round": entry.get("remaining_budget_after_round"),
                "observation_summary": entry["observation"].get("summary"),
                "observation_key_evidence": entry["observation"].get("key_evidence"),
                "reflection": entry["reflection"],
                "plan_latency_sec": entry["plan_latency_sec"],
                "observation_latency_sec": entry["observation_latency_sec"],
                "reflection_latency_sec": entry["reflection_latency_sec"],
            }
            if save_frame_paths:
                compact_entry["plan_raw"] = entry.get("plan_raw")
                compact_entry["observation_raw"] = entry.get("observation_raw")
                compact_entry["reflection_raw"] = entry.get("reflection_raw")
            compact_trace.append(compact_entry)

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
            "rounds_used": agent_result.get("rounds_used"),
            "frames_used_total": agent_result.get("frames_used_total"),
            "total_frame_budget": agent_result.get("total_frame_budget"),
            "total_frames_available": agent_result.get("total_frames_available"),
            "days_spanned": agent_result.get("days_spanned"),
            "agent_trace": compact_trace,
            "synthesizer_raw": agent_result.get("synthesizer_raw"),
            "synthesizer_latency_sec": agent_result.get("synthesizer_latency_sec"),
            "response_time_sec": response_time_sec,
            "error": error,
        }
        results.append(row)

        if save_every > 0 and len(results) % save_every == 0:
            partial_payload = make_payload(is_complete=False)
            atomic_json_dump(output_path, partial_payload)
            atomic_json_dump(summary_output_path(output_path), build_summary_payload(partial_payload))
            tqdm.write(
                f"[job {job_id + 1}/{num_jobs}] autosaved {len(results)} examples -> {output_path}"
            )

        rounds_used = agent_result.get("rounds_used", "?")
        tqdm.write(
            f"[job {job_id + 1}/{num_jobs}] [{idx}/{len(examples)}] {ex_id} "
            f"gt={gt} pred={pred or 'N/A'} ok={is_correct} "
            f"rounds={rounds_used} response_time={response_time_sec}s"
        )
        iterator.set_postfix_str(f"acc={n_correct}/{idx}")
        if error:
            tqdm.write(f"[job {job_id + 1}/{num_jobs}] ERROR {ex_id} err={error}")
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    final_payload = make_payload(is_complete=True)
    atomic_json_dump(output_path, final_payload)
    summary_path = summary_output_path(output_path)
    atomic_json_dump(summary_path, build_summary_payload(final_payload))

    total = len(results)
    acc = (n_correct / total) if total else 0.0
    print(f"\nSaved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(
        f"Job {job_id + 1}/{num_jobs} Accuracy: {n_correct}/{total} = {acc:.4f} "
        f"(evaluated {total} shard examples from {total_after_limit} total)"
    )
    summary = final_payload["summary"]
    print(
        f"Overall accuracy: {summary['overall']['correct']}/{summary['overall']['total']} "
        f"= {summary['overall']['accuracy']:.4f}"
    )
    print("Task type accuracy:")
    for task_name, stats in summary["by_task_type"].items():
        print(f"  - {task_name}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")
    print("Memory type accuracy:")
    for mem_name, stats in summary["by_memory_type"].items():
        print(f"  - {mem_name}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic Gemini frame-based VideoQA on EgoLife (plan/observe/reflect)."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--frames_index",
        type=str,
        default="/nas-ssd2/video_datasets/EgoLife/egolife_frames_index.json",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--planner_model", type=str, default=None)
    parser.add_argument("--observer_model", type=str, default=None)
    parser.add_argument("--reflector_model", type=str, default=None)
    parser.add_argument("--synthesizer_model", type=str, default=None)
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY", DEFAULT_API_KEY),
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=512,
        help="Default per-observation frame budget. Planner may pick less.",
    )
    parser.add_argument("--frame_size", type=int, default=384)
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="Maximum plan/observe/reflect rounds before forced synthesis.",
    )
    parser.add_argument(
        "--total_frame_budget",
        type=int,
        default=1024,
        help=(
            "Hard cap on total frames sent across all observation rounds. "
            "Planner is told the remaining budget each round and may sample very sparsely."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument("--save_frame_paths", action="store_true")
    parser.add_argument(
        "--default_query_type",
        type=str,
        default=None,
        help="Fill missing query_type (e.g. 'Event Ordering' for event_ordering_v2.json).",
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
    if args.max_rounds < 1:
        raise ValueError("--max_rounds must be >= 1")
    if args.total_frame_budget < 4:
        raise ValueError("--total_frame_budget must be >= 4")

    evaluate(
        dataset_path=args.dataset,
        frames_index_path=args.frames_index,
        output_path=args.output,
        model=args.model,
        planner_model=args.planner_model,
        observer_model=args.observer_model,
        reflector_model=args.reflector_model,
        synthesizer_model=args.synthesizer_model,
        api_key=args.api_key,
        max_frames=args.max_frames,
        frame_size=args.frame_size,
        max_rounds=args.max_rounds,
        total_frame_budget=args.total_frame_budget,
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
