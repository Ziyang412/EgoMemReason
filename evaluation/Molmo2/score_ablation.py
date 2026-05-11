#!/usr/bin/env python3
"""Score and compare ablation study results across experiments.

Usage:
  python score_ablation.py --ablation_dir results/ablation/ \
    --extra "frames_64f_direct=results/batch_v2/.../results_all_task_types_v2_merged.json" \
    --extra "frames_256f_direct=results/batch_v2/.../results_all_task_types_v2_merged.json" \
    --extra "frames_1024f_direct=results/batch_v2/.../results_all_task_types_v2_merged.json"
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# Desired display order for experiments
EXPERIMENT_ORDER = [
    "frames_0f_direct",
    "frames_64f_direct",
    "frames_256f_direct",
    "frames_1024f_direct",
    "frames_2048f_direct",
    "frames_256f_cot",
    "frames_256f_icl",
    "frames_256f_icl_cot",
    "frames_256f_captions",
    "frames_256f_transcripts",
    "frames_256f_cap_trans",
]


def score_results(results: list[dict]) -> dict:
    """Compute overall and per-query-type accuracy."""
    by_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    overall = {"total": 0, "correct": 0}

    for row in results:
        pred = row.get("pred")
        answer = row.get("answer")
        correct = row.get("correct")

        if pred is None or answer is None:
            continue

        is_correct = correct is True
        overall["total"] += 1
        if is_correct:
            overall["correct"] += 1

        qt = row.get("query_type") or "unknown"
        by_type[qt]["total"] += 1
        if is_correct:
            by_type[qt]["correct"] += 1

    overall["accuracy"] = overall["correct"] / overall["total"] if overall["total"] else 0.0
    for qt_data in by_type.values():
        qt_data["accuracy"] = qt_data["correct"] / qt_data["total"] if qt_data["total"] else 0.0

    return {"overall": overall, "by_type": dict(by_type)}


def load_experiment(path: str) -> list[dict] | None:
    """Load results JSON (list format)."""
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Score ablation study results.")
    parser.add_argument("--ablation_dir", type=str, required=True,
        help="Directory containing ablation experiment subdirs")
    parser.add_argument("--extra", type=str, action="append", default=[],
        help="Extra results: 'name=path' to include existing results")
    parser.add_argument("--output_json", type=str, default=None,
        help="Output path for JSON metrics (default: ablation_dir/ablation_scores.json)")
    args = parser.parse_args()

    experiments: dict[str, dict] = {}

    # Load ablation results from subdirectories
    if os.path.isdir(args.ablation_dir):
        for entry in sorted(os.listdir(args.ablation_dir)):
            exp_dir = os.path.join(args.ablation_dir, entry)
            if not os.path.isdir(exp_dir):
                continue
            # Find merged result file
            for fname in os.listdir(exp_dir):
                if fname.endswith("_merged.json"):
                    results = load_experiment(os.path.join(exp_dir, fname))
                    if results is not None:
                        experiments[entry] = score_results(results)
                    break

    # Load extra results
    for extra in args.extra:
        if "=" not in extra:
            print(f"WARNING: Skipping malformed --extra '{extra}' (expected 'name=path')")
            continue
        name, path = extra.split("=", 1)
        results = load_experiment(path)
        if results is not None:
            experiments[name] = score_results(results)
        else:
            print(f"WARNING: Could not load '{name}' from '{path}'")

    if not experiments:
        print("No experiments found!")
        sys.exit(1)

    # Collect all query types across experiments
    all_types: set[str] = set()
    for scores in experiments.values():
        all_types.update(scores["by_type"].keys())
    type_cols = sorted(all_types)

    # Sort experiments by desired order, then alphabetically for unknown
    def sort_key(name):
        if name in EXPERIMENT_ORDER:
            return (0, EXPERIMENT_ORDER.index(name))
        return (1, name)

    sorted_names = sorted(experiments.keys(), key=sort_key)

    # Print table
    col_width = 20
    name_width = 35
    header = f"{'Experiment':<{name_width}} | {'Overall':>{col_width}}"
    for t in type_cols:
        header += f" | {t:>{col_width}}"
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for name in sorted_names:
        scores = experiments[name]
        ov = scores["overall"]
        row = f"{name:<{name_width}} | {ov['correct']:>3}/{ov['total']:<3} = {ov['accuracy']:.4f}  "
        for t in type_cols:
            td = scores["by_type"].get(t, {"correct": 0, "total": 0, "accuracy": 0.0})
            row += f" | {td['correct']:>3}/{td['total']:<3} = {td['accuracy']:.4f}  "
        print(row)

    print(sep)
    print()

    # Save JSON
    output_path = args.output_json or os.path.join(args.ablation_dir, "ablation_scores.json")
    json_out = {}
    for name in sorted_names:
        json_out[name] = experiments[name]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Scores written to {output_path}")


if __name__ == "__main__":
    main()
