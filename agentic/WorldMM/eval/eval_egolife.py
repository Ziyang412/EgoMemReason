#!/usr/bin/env python3
"""
EgoLifeQA evaluation script using WorldMM unified memory system.
"""

import os
import json
import re
import argparse
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult, transform_timestamp


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
    """
    Evaluate if prediction matches the gold answer.
    
    Args:
        prediction: Model's prediction
        gold_letter: Correct answer letter (e.g., 'A', 'B', 'C', 'D')
        choices: Dict of answer choices
        
    Returns:
        True if prediction is correct
    """
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


def find_30s_segment(target_timestamp: int, segments_30s: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Find the 30s segment that contains the target timestamp.
    
    Args:
        target_timestamp: Target timestamp as integer (format: day + time.zfill(8))
        segments_30s: List of 30s segments
    
    Returns:
        Tuple of (start_time, end_time) for the matching segment, or (0, 0) if not found
    """
    for segment in segments_30s:
        date = segment.get('date', '')
        start_time_raw = segment.get('start_time', 0)
        end_time_raw = segment.get('end_time', 0)
        
        day = date.replace('DAY', '').replace('Day', '') if isinstance(date, str) else str(date)
        
        # Format times
        if isinstance(start_time_raw, str):
            start_time = int(day + start_time_raw.zfill(8))
        elif isinstance(start_time_raw, int):
            start_time = int(day + str(start_time_raw).zfill(8))
        else:
            continue
        
        if isinstance(end_time_raw, str):
            end_time = int(day + end_time_raw.zfill(8))
        elif isinstance(end_time_raw, int):
            end_time = int(day + str(end_time_raw).zfill(8))
        else:
            continue
        
        # Check if target timestamp falls within this segment
        if start_time <= target_timestamp <= end_time:
            return (start_time, end_time)
    
    return (0, 0)


def parse_target_time(row: Dict[str, Any], segments_30s: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    Parse target time from row data.
    
    Args:
        row: QA row data
        segments_30s: List of 30s segments for finding time ranges
        
    Returns:
        List of (start_time, end_time) tuples
    """
    target_time_list = []
    
    if "time" in row['target_time'] and row['target_time']["time"]:
        time_str = row['target_time']["time"]
        time_str_upper = time_str.upper()
        
        if "DAY" in time_str_upper:
            # Parse range format: "11153417DAY1_11181201"
            parts = re.split(r'DAY|Day', time_str, maxsplit=1)
            if len(parts) == 2:
                start_time_str = parts[0]
                day_and_end = parts[1].split("_")
                if len(day_and_end) == 2:
                    end_day = day_and_end[0]
                    end_time_str = day_and_end[1]
                    start_day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
                    
                    start_time = int(start_day + start_time_str.zfill(8))
                    end_time = int(end_day + end_time_str.zfill(8))
                    target_time_list.append((start_time, end_time))
        else:
            # Single timestamp - find its 30s segment
            day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)
    
    elif "time_list" in row['target_time'] and row['target_time']["time_list"]:
        # Multiple timestamps
        day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
        for time_str in row['target_time']["time_list"]:
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)
    
    return target_time_list


def main():
    parser = argparse.ArgumentParser(description="EgoLifeQA Evaluation with WorldMM")
    parser.add_argument("--subject", type=str, default="A1_JAKE", help="Subject ID")
    parser.add_argument("--retriever-model", type=str, default="gpt-5-mini", help="LLM model for retrieval (NER, OpenIE)")
    parser.add_argument("--respond-model", type=str, default="gpt-5", help="LLM model for iterative reasoning and generating answers")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum errors before forcing answer")
    parser.add_argument("--episodic-top-k", type=int, default=3, help="Top-k for episodic retrieval")
    parser.add_argument("--semantic-top-k", type=int, default=10, help="Top-k for semantic retrieval")
    parser.add_argument("--visual-top-k", type=int, default=3, help="Top-k for visual retrieval")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data/EgoLife", help="Data directory")
    args = parser.parse_args()

    # Initialize models
    logger.info("Initializing models...")
    embedding_model = EmbeddingModel()
    retriever_llm_model = LLMModel(
        model_name=args.retriever_model,
    )
    respond_llm_model = LLMModel(
        model_name=args.respond_model,
        fps=1,
    )
    prompt_template_manager = PromptTemplateManager()

    # Initialize WorldMemory
    logger.info("Initializing WorldMemory...")
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
    )
    
    # Set retrieval top-k
    world_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    # Load data
    logger.info("Loading data...")
    subject = args.subject
    data_dir = args.data_dir
    
    eval_data_path = os.path.join(data_dir, f"EgoLifeQA/EgoLifeQA_{subject}.json")
    eval_data = load_json(eval_data_path)
    
    # Load episodic captions for all granularities (multiscale memory)
    episodic_caption_dir = os.path.join(data_dir, f"EgoLifeCap/{subject}")
    granularities = ["30sec", "3min", "10min", "1h"]
    episodic_caption_files = {
        g: os.path.join(episodic_caption_dir, f"{subject}_{g}.json")
        for g in granularities
    }
    # Load 30sec captions separately for target time parsing
    episodic_captions_30sec = load_json(episodic_caption_files["30sec"])
    
    # Load semantic results
    semantic_path = os.path.join(f"output/metadata/semantic_memory/{subject}/semantic_consolidation_results_gpt-5-mini.json")
    semantic_results = load_json(semantic_path)
    
    # Load visual embeddings
    visual_path = os.path.join(f"output/metadata/visual_memory/{subject}/visual_embeddings.pkl")
    
    # Load data into WorldMemory
    logger.info("Loading data into WorldMemory...")
    
    # Load episodic captions for all granularities
    world_memory.load_episodic_captions(caption_files=episodic_caption_files)
    
    # Load semantic triples
    world_memory.load_semantic_triples(data=semantic_results)
    
    # Load visual embeddings
    world_memory.load_visual_clips(embeddings_path=visual_path, clips_data=episodic_captions_30sec)

    # Evaluation loop
    logger.info(f"Starting evaluation on {len(eval_data)} samples...")
    results = []
    evaluate_true = 0

    for row in tqdm(eval_data):
        ID = row['ID']
        query_type = row['type']
        question = row['question']
        answer = row['answer']

        # Parse choices
        choices = {}
        for key, label in [('choice_a', 'A'), ('choice_b', 'B'), ('choice_c', 'C'), ('choice_d', 'D')]:
            if key in row and row[key]:
                choices[label] = row[key]

        # Parse query time
        query_time = int(row['query_time']["date"][-1] + row['query_time']["time"].zfill(8))
        
        # Parse target time (use 30sec captions for segment lookup)
        target_time_list = parse_target_time(row, episodic_captions_30sec)

        logger.info(f"Processing ID {ID}: {question[:50]}...")

        qa_result: Optional[QAResult] = None
        try:            
            # Answer the question
            qa_result = world_memory.answer(
                query=question,
                choices=choices,
                until_time=query_time,
            )
            
            response = qa_result.answer
            
        except Exception as e:
            logger.error(f"Error processing ID {ID}: {e}")
            response = "Error"

        # Evaluate
        evaluate = evaluate_prediction(response, answer, choices)
        evaluate_true += int(evaluate)

        # Build result entry
        result_entry = {
            "ID": ID,
            "type": query_type,
            "question": question,
            "choices": choices,
            "answer": answer,
            "response": response,
            "round_history": qa_result.round_history if qa_result else [],
            "num_rounds": qa_result.num_rounds if qa_result else 0,
            "evaluate": evaluate,
            "query_time": query_time,
            # "query_time_str": transform_timestamp(str(query_time)),
            "target_time": target_time_list,
            # "target_time_str": [
            #     (transform_timestamp(str(start)), transform_timestamp(str(end))) 
            #     for start, end in target_time_list
            # ],
        }
        results.append(result_entry)

        logger.info(
            f"ID {ID} Answer: {response}, Gold: {answer}, Correct: {evaluate} "
            f"// Accuracy: {evaluate_true}/{len(results)} = {evaluate_true/len(results):.4f}"
        )

    # Save results
    output_path = os.path.join(
        args.output_dir, 
        f"{args.retriever_model.replace('-', '_')}_{args.respond_model.replace('-', '_')}",
        f"egolife_eval_{subject}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    final_accuracy = evaluate_true / len(results) if results else 0
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Complete")
    logger.info(f"Subject: {subject}")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Correct: {evaluate_true}")
    logger.info(f"Accuracy: {final_accuracy:.4f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*50}")

    # Cleanup
    world_memory.cleanup()


if __name__ == "__main__":
    main()
