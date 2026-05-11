import argparse
import glob
import json
import os
from typing import Dict, List


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


EPISODIC_QUERY_TYPES = {
    "temporal_ordering",
    "temporal_reasoning",
    "state_tracking",
    "multi_entity",
}


def infer_memory_type(row: Dict) -> str:
    for k in ["memory_type", "memory", "memory_category"]:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()

    query_type = str(row.get("query_type") or "").strip().lower()
    if query_type == "semantic_event":
        return "semantic"
    if query_type in EPISODIC_QUERY_TYPES:
        return "episodic"
    return "unknown"


def compute_group_accuracy(results: List[Dict], key: str) -> Dict[str, Dict]:
    grouped: Dict[str, Dict[str, int]] = {}
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
        "by_task_type": compute_group_accuracy(results, "query_type"),
        "by_memory_type": compute_group_accuracy(results, "memory_type"),
    }


def summary_output_path(output_path: str) -> str:
    if output_path.lower().endswith(".json"):
        return output_path[:-5] + "_summary.json"
    return f"{output_path}_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sharded GPT-5 frame evaluation JSON files."
    )
    parser.add_argument(
        "--inputs_glob",
        type=str,
        required=True,
        help="Glob pattern for shard JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for merged JSON output.",
    )
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.inputs_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No files matched --inputs_glob: {args.inputs_glob}")

    shard_payloads: List[Dict] = [load_json(p) for p in shard_paths]

    merged_results = []
    for p in shard_payloads:
        merged_results.extend(p.get("results", []))

    for row in merged_results:
        if not row.get("memory_type"):
            row["memory_type"] = infer_memory_type(row)

    total = len(merged_results)
    correct = sum(1 for r in merged_results if bool(r.get("is_correct")))
    accuracy = (correct / total) if total else 0.0
    summary = build_accuracy_summary(merged_results)

    first = shard_payloads[0]
    merged = {
        "task_type": first.get("task_type"),
        "task_description": first.get("task_description"),
        "dataset_path": first.get("dataset_path"),
        "frames_index_path": first.get("frames_index_path"),
        "model": first.get("model"),
        "azure_endpoint": first.get("azure_endpoint"),
        "api_version": first.get("api_version"),
        "max_frames": first.get("max_frames"),
        "max_hours_before_query": first.get("max_hours_before_query"),
        "max_completion_tokens": first.get("max_completion_tokens"),
        "image_detail": first.get("image_detail"),
        "num_shards": len(shard_paths),
        "shard_files": shard_paths,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "summary": summary,
        "results": merged_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    summary_path = summary_output_path(args.output)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset_path": merged.get("dataset_path"),
                "model": merged.get("model"),
                "max_frames": merged.get("max_frames"),
                "num_shards": merged.get("num_shards"),
                "summary": summary,
            },
            f,
            indent=2,
        )

    print(f"Merged {len(shard_paths)} shards into: {args.output}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"Summary saved to: {summary_path}")
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


if __name__ == "__main__":
    main()
