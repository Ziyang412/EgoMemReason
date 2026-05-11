#!/usr/bin/env python3
"""
Benchmark evaluation script using WorldMM unified memory system.
Supports variable number of options (A-H), multiple actors, and multiple task types.
Supports resume: re-run to retry errors and process unfinished examples.
"""

import os
import json
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    """Extracts A, B, C... from a prediction like (C), B. Bryan, etc."""
    match = re.match(r"\(?([A-Za-z])[\.\)]?\s*", text.strip())
    return match.group(1).upper() if match else None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    """Evaluate if prediction matches the gold answer."""
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])

    if pred_norm == gold_candidate:
        return True

    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True

    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}")
    ]
    if pred_norm in full_patterns:
        return True

    return False


def parse_query_time(query_time_str: str) -> int:
    """
    Convert query_time string to integer timestamp.

    'DAY6, 18:30:00' -> 618300000 (day + HHMMSS zero-padded to 8 chars)
    """
    match = re.match(r'(?:DAY|Day\s*)(\d+),?\s*(\d{1,2}):(\d{2}):(\d{2})', query_time_str)
    if not match:
        raise ValueError(f"Cannot parse query_time: {query_time_str}")
    day = match.group(1)
    hh = match.group(2).zfill(2)
    mm = match.group(3).zfill(2)
    ss = match.group(4).zfill(2)
    return int(f"{day}{hh}{mm}{ss}00")


def load_existing_results(output_path: str) -> Dict[str, Dict]:
    """Load existing results and return a dict keyed by example_id.
    Only keeps results with a valid (non-error) response."""
    if not os.path.exists(output_path):
        return {}
    try:
        data = load_json(output_path)
        results = data.get("results", [])
        completed = {}
        retry_responses = {"Error", "Unable to generate answer"}
        for r in results:
            if r.get("response") and r["response"] not in retry_responses:
                completed[r["example_id"]] = r
        n_retry = len(results) - len(completed)
        logger.info(f"Loaded {len(completed)} completed results from {output_path} "
                     f"({n_retry} to retry)")
        return completed
    except Exception as e:
        logger.warning(f"Failed to load existing results: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation with WorldMM")
    parser.add_argument("--benchmark-json", type=str, required=True,
                        help="Path to benchmark JSON (all_task_types_v2.json)")
    parser.add_argument("--caption-dir", type=str, required=True,
                        help="Root caption directory (contains {ACTOR}/ subdirs)")
    parser.add_argument("--metadata-dir", type=str, default="output/metadata",
                        help="Root metadata directory for memories")
    parser.add_argument("--retriever-model", type=str, default="gpt-5-mini",
                        help="LLM model for retrieval (NER, OpenIE)")
    parser.add_argument("--respond-model", type=str, default="gpt-5",
                        help="LLM model for iterative reasoning and generating answers")
    parser.add_argument("--max-rounds", type=int, default=5,
                        help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5,
                        help="Maximum errors before forcing answer")
    parser.add_argument("--episodic-top-k", type=int, default=3,
                        help="Top-k for episodic retrieval")
    parser.add_argument("--semantic-top-k", type=int, default=10,
                        help="Top-k for semantic retrieval")
    parser.add_argument("--visual-top-k", type=int, default=3,
                        help="Top-k for visual retrieval")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory")
    args = parser.parse_args()

    # Determine output path (use benchmark filename to avoid collisions)
    bench_name = os.path.splitext(os.path.basename(args.benchmark_json))[0]
    output_path = os.path.join(
        args.output_dir,
        f"{args.retriever_model.replace('-', '_')}_{args.respond_model.replace('-', '_')}",
        f"{bench_name}_eval.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing completed results for resume
    completed = load_existing_results(output_path)

    # Load benchmark
    logger.info(f"Loading benchmark from {args.benchmark_json}")
    benchmark = load_json(args.benchmark_json)
    samples = benchmark if isinstance(benchmark, list) else benchmark["samples"]

    todo_ids = set(s["example_id"] for s in samples) - set(completed.keys())
    logger.info(f"Total: {len(samples)}, Already done: {len(completed)}, To run: {len(todo_ids)}")

    if not todo_ids:
        logger.info("All examples already completed. Nothing to do.")
        return

    # Group samples by actor
    samples_by_actor = defaultdict(list)
    for s in samples:
        samples_by_actor[s["identity"]].append(s)

    # Check which actors have pending work
    actors_with_work = set()
    for s in samples:
        if s["example_id"] in todo_ids:
            actors_with_work.add(s["identity"])
    logger.info(f"Actors with pending work: {sorted(actors_with_work)}")

    # Initialize models
    logger.info("Initializing models...")
    embedding_model = EmbeddingModel()
    retriever_llm_model = LLMModel(model_name=args.retriever_model)
    respond_llm_model = LLMModel(model_name=args.respond_model, fps=1)
    prompt_template_manager = PromptTemplateManager()

    # Initialize WorldMemory with benchmark QA template
    logger.info("Initializing WorldMemory...")
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        qa_template_name="qa_benchmark",
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
    )

    world_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    # Evaluation loop: process one actor at a time (only actors with pending work)
    new_results = {}

    for actor in sorted(actors_with_work):
        actor_samples = samples_by_actor[actor]
        actor_todo = [s for s in actor_samples if s["example_id"] in todo_ids]

        logger.info(f"\n{'='*50}")
        logger.info(f"Processing actor: {actor} ({len(actor_todo)}/{len(actor_samples)} pending)")
        logger.info(f"{'='*50}")

        # Reset memory for new actor and set per-actor episodic cache
        world_memory.reset()
        actor_cache_root = f".cache/benchmark/{actor}/episodic_memory"
        world_memory.episodic_memory.save_dir_root = actor_cache_root
        world_memory.episodic_memory.hipporag = {}  # clear stale HippoRAG instances

        # Load episodic captions
        granularities = ["30sec", "3min", "10min", "1h"]
        caption_files = {
            g: os.path.join(args.caption_dir, actor, f"{actor}_{g}.json")
            for g in granularities
        }

        missing = [g for g, p in caption_files.items() if not os.path.exists(p)]
        if missing:
            logger.warning(f"Missing caption files for {actor}: {missing}")
            continue

        world_memory.load_episodic_captions(caption_files=caption_files)

        # Load semantic triples
        semantic_path = os.path.join(
            args.metadata_dir, "semantic_memory", actor,
            f"semantic_consolidation_results_{args.retriever_model}.json"
        )
        if os.path.exists(semantic_path):
            semantic_data = load_json(semantic_path)
            world_memory.load_semantic_triples(data=semantic_data)
        else:
            logger.warning(f"Semantic memory not found for {actor}: {semantic_path}")

        # Load visual embeddings
        visual_path = os.path.join(
            args.metadata_dir, "visual_memory", actor, "visual_embeddings.pkl"
        )
        captions_30s = load_json(caption_files["30sec"])
        if os.path.exists(visual_path):
            world_memory.load_visual_clips(
                embeddings_path=visual_path, clips_data=captions_30s
            )
        else:
            logger.warning(f"Visual memory not found for {actor}: {visual_path}")

        # Sort pending samples by query_time for efficient monotonic indexing
        actor_todo.sort(key=lambda s: parse_query_time(s["query_time"]))

        for sample in tqdm(actor_todo, desc=f"Eval {actor}"):
            example_id = sample["example_id"]
            question = sample["question"]
            choices = sample["options"]
            gold_answer = sample.get("correct_answer") or sample.get("answer")
            query_type = sample.get("query_type", "event_ordering")

            query_time = parse_query_time(sample["query_time"])

            logger.info(f"Processing {example_id}: {question[:50]}...")

            qa_result: Optional[QAResult] = None
            try:
                qa_result = world_memory.answer(
                    query=question,
                    choices=choices,
                    until_time=query_time,
                )
                response = qa_result.answer
            except Exception as e:
                logger.error(f"Error processing {example_id}: {e}")
                response = "Error"

            # Evaluate
            correct = evaluate_prediction(response, gold_answer, choices)

            result_entry = {
                "example_id": example_id,
                "identity": actor,
                "query_type": query_type,
                "question": question,
                "choices": choices,
                "correct_answer": gold_answer,
                "response": response,
                "round_history": qa_result.round_history if qa_result else [],
                "num_rounds": qa_result.num_rounds if qa_result else 0,
                "evaluate": correct,
                "query_time": sample["query_time"],
                "query_time_int": query_time,
            }
            new_results[example_id] = result_entry

            logger.info(
                f"{example_id} Answer: {response}, Gold: {gold_answer}, Correct: {correct}"
            )

            # Save incrementally after each sample (merge completed + new so far)
            _save_results(output_path, samples, completed, new_results, args)

    # Final save
    _save_results(output_path, samples, completed, new_results, args)

    # Print summary
    merged = {**completed, **new_results}
    all_results = [merged[s["example_id"]] for s in samples if s["example_id"] in merged]
    total_correct = sum(1 for r in all_results if r["evaluate"])

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Complete")
    logger.info(f"Total: {len(all_results)}, Correct: {total_correct}, "
                f"Accuracy: {total_correct/len(all_results):.4f}")

    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    actor_correct_map = defaultdict(int)
    actor_total_map = defaultdict(int)
    for r in all_results:
        type_total[r["query_type"]] += 1
        actor_total_map[r["identity"]] += 1
        if r["evaluate"]:
            type_correct[r["query_type"]] += 1
            actor_correct_map[r["identity"]] += 1

    logger.info(f"\nPer-type accuracy:")
    for t in sorted(type_total.keys()):
        acc = type_correct[t] / type_total[t] if type_total[t] > 0 else 0
        logger.info(f"  {t}: {type_correct[t]}/{type_total[t]} = {acc:.4f}")
    logger.info(f"\nPer-actor accuracy:")
    for a in sorted(actor_total_map.keys()):
        acc = actor_correct_map[a] / actor_total_map[a] if actor_total_map[a] > 0 else 0
        logger.info(f"  {a}: {actor_correct_map[a]}/{actor_total_map[a]} = {acc:.4f}")
    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"{'='*60}")

    # Cleanup
    world_memory.cleanup()


def _save_results(output_path, samples, completed, new_results, args):
    """Merge completed + new results, recompute stats, and save."""
    merged = {**completed, **new_results}
    # Preserve benchmark sample order
    all_results = [merged[s["example_id"]] for s in samples if s["example_id"] in merged]

    total_correct = sum(1 for r in all_results if r["evaluate"])

    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    actor_correct_map = defaultdict(int)
    actor_total_map = defaultdict(int)
    for r in all_results:
        type_total[r["query_type"]] += 1
        actor_total_map[r["identity"]] += 1
        if r["evaluate"]:
            type_correct[r["query_type"]] += 1
            actor_correct_map[r["identity"]] += 1

    output_data = {
        "benchmark": args.benchmark_json,
        "retriever_model": args.retriever_model,
        "respond_model": args.respond_model,
        "total": len(all_results),
        "correct": total_correct,
        "accuracy": total_correct / len(all_results) if all_results else 0,
        "accuracy_by_type": {
            t: {"correct": type_correct[t], "total": type_total[t],
                "accuracy": type_correct[t] / type_total[t] if type_total[t] > 0 else 0}
            for t in sorted(type_total.keys())
        },
        "accuracy_by_actor": {
            a: {"correct": actor_correct_map[a], "total": actor_total_map[a],
                "accuracy": actor_correct_map[a] / actor_total_map[a] if actor_total_map[a] > 0 else 0}
            for a in sorted(actor_total_map.keys())
        },
        "results": all_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
