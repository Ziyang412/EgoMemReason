import transformers
import json
import os
import re
import time
import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional, Dict, List, Any, Tuple
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from openai import OpenAI, AzureOpenAI
import requests

import utils.process as process
import utils.constants as constants

# Azure OpenAI setup
endpoint = os.getenv("ENDPOINT_URL", "https://YOUR-RESOURCE.cognitiveservices.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5-chat")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")

summ_client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-12-01-preview",
)

QUERY_TYPE_TO_MEMORY_TYPE = {
    "temporal_ordering": "episodic",
    "temporal_reasoning": "episodic",
    "state_tracking": "episodic",
    "spatial_tracking": "episodic",
    "multi_entity": "episodic",
    "semantic_event": "semantic",
}


@dataclass
class ScriptArgs:
    model_name_or_path: Optional[str] = field(
        default='Ego-R1/Ego-R1-Agent-3B',
        metadata={"help": "Model name or path"}
    )
    benchmark_json: Optional[str] = field(
        default='/nas-ssd2/ziyang/Memory_project/COLM/benchmark/filtered_batch_1/all_task_types_v2.json',
        metadata={"help": "Path to benchmark JSON file"}
    )
    data_start: Optional[int] = field(
        default=0,
        metadata={"help": "Data start index"}
    )
    data_end: Optional[int] = field(
        default=-1,
        metadata={"help": "Data end index (-1 for all)"}
    )
    temperature: Optional[float] = field(
        default=0.6,
        metadata={"help": "Temperature for the model"}
    )
    top_p: Optional[float] = field(
        default=0.95,
        metadata={"help": "Top-p for the model"}
    )
    max_turns: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of turns"}
    )
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens"}
    )
    result_dir: Optional[str] = field(
        default='results/egolife_bench',
        metadata={"help": "Result directory"}
    )
    vllm_base_url: Optional[str] = field(
        default='http://localhost:23333/v1',
        metadata={"help": "vLLM server OpenAI-compatible API base URL"}
    )


def load_benchmark(path: str) -> Tuple[Dict, List[Dict]]:
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {"type": "flat_list", "total_examples": len(data)}, data

    if isinstance(data, dict):
        if isinstance(data.get("samples"), list):
            return data, data["samples"]
        for key in ("examples", "items", "queries", "data"):
            if isinstance(data.get(key), list):
                return data, data[key]

    raise ValueError(f"Cannot parse benchmark JSON at {path}")


def format_question(sample: Dict) -> str:
    identity = sample.get("identity", "")
    query_time = sample.get("query_time", "")
    question = sample.get("question", "")
    options = sample.get("options", {})

    lines = []
    if identity:
        lines.append(f"You are {identity}.")
    if query_time:
        lines.append(f"The current time is {query_time}.")
    if lines:
        lines.append("")

    lines.append(f"Question: {question}")
    lines.append("")

    if options:
        lines.append("Options:")
        for label in sorted(options.keys()):
            lines.append(f"{label}. {options[label]}")

    return "\n".join(lines)


def build_summarization_prompt(options: Dict) -> str:
    labels = sorted(options.keys())
    format_str = "|".join(labels)
    last_label = labels[-1] if labels else "A"
    return (
        f"You are given some information and a chain-of-thought reasoning process, "
        f"with actions made and observations from the environment. Given these information, "
        f"try to answer the MCQ question in {format_str} format. You should only answer one option. "
        f"For example, if the answer to this question is {last_label}, you should only return "
        f"```<answer>{last_label}</answer>```."
    )


def summarize_with_azure(chat_history: List[Dict], question_text: str, options: Dict) -> str:
    client_prompt = build_summarization_prompt(options)
    ch_prompt = f"The chat history is as follows:\n{chat_history}\n"
    messages = [
        {"role": "system", "content": client_prompt},
        {"role": "user", "content": ch_prompt},
        {"role": "user", "content": f"Question: {question_text}"},
    ]

    output_message = "<answer>A</answer>"  # fallback
    for attempt in range(3):
        try:
            output_message = summ_client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=800,
                temperature=1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False,
            ).choices[0].message.content
            assert '<answer>' in output_message and '</answer>' in output_message
            break
        except Exception as e:
            print(f'Summarization error (attempt {attempt+1}): {e}')
            if attempt == 2:
                if '<answer>' not in output_message:
                    print("Max retries reached, using fallback answer")
                    output_message = "<answer>A</answer>"

    return output_message


logger = logging.getLogger("egolife_bench")


def logged_tool_call(query_info, identity):
    """Wrapper around process.tool_call with detailed logging."""
    if query_info is None:
        logger.warning("Tool call skipped: query_info is None")
        return ""

    tool_name = query_info.get("name", "unknown")
    tool_args = query_info.get("arguments", {})
    logger.info(f"Tool call: name={tool_name}, identity={identity}, args={json.dumps(tool_args, default=str)}")

    # Log the target URL
    if 'rag' in tool_name.lower():
        url = constants.RAG_URL.get(identity, "UNKNOWN")
        logger.info(f"  RAG URL: {url}")
    elif 'video_llm' in tool_name.lower():
        logger.info(f"  Video LLM URL: {constants.VIDEO_LLM_URL}")
    elif 'vlm' in tool_name.lower():
        logger.info(f"  VLM URL: {constants.VLM_URL}")

    try:
        result = process.tool_call(query_info, identity)
        result_preview = str(result)[:500] if result else "(empty)"
        logger.info(f"  Tool result: {result_preview}")
        return result
    except Exception as e:
        logger.error(f"  Tool call EXCEPTION: {type(e).__name__}: {e}", exc_info=True)
        return f"Tool call error: {e}"


def load_completed_ids(jsonl_path: str) -> set:
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                if "example_id" in result:
                    completed.add(result["example_id"])
            except json.JSONDecodeError:
                continue
    return completed


def save_result_jsonl(jsonl_path: str, result: Dict):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_all_results_jsonl(jsonl_path: str) -> List[Dict]:
    results = []
    if not os.path.exists(jsonl_path):
        return results
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def compute_group_accuracy(results: List[Dict], group_key: str) -> Dict[str, Dict]:
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in results:
        group = str(row.get(group_key, "unknown")).strip() or "unknown"
        stats[group]["total"] += 1
        if row.get("is_correct"):
            stats[group]["correct"] += 1

    output = {}
    for group in sorted(stats.keys()):
        total = stats[group]["total"]
        correct = stats[group]["correct"]
        output[group] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }
    return output


def build_accuracy_summary(results: List[Dict]) -> Dict:
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    return {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
        },
        "task_type_accuracy": compute_group_accuracy(results, "query_type"),
        "memory_type_accuracy": compute_group_accuracy(results, "memory_type"),
        "identity_accuracy": compute_group_accuracy(results, "identity"),
    }


def print_accuracy_summary(summary: Dict):
    overall = summary.get("overall", {})
    print(f"\nOverall Accuracy: {overall.get('correct', 0)}/{overall.get('total', 0)} = {overall.get('accuracy', 0.0):.4f}")

    print("\nTask Type Accuracy:")
    for key, value in summary.get("task_type_accuracy", {}).items():
        print(f"  {key}: {value['correct']}/{value['total']} = {value['accuracy']:.4f}")

    print("\nMemory Type Accuracy:")
    for key, value in summary.get("memory_type_accuracy", {}).items():
        print(f"  {key}: {value['correct']}/{value['total']} = {value['accuracy']:.4f}")

    print("\nIdentity Accuracy:")
    for key, value in summary.get("identity_accuracy", {}).items():
        print(f"  {key}: {value['correct']}/{value['total']} = {value['accuracy']:.4f}")


def main():
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

    # Setup logging
    model_tag = os.path.basename(args.model_name_or_path)
    log_dir = "infer_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"egolife_bench_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger.info(f"Logging to {log_file}")

    # Log tool server URLs for debugging
    logger.info(f"RAG URLs: {constants.RAG_URL}")
    logger.info(f"VIDEO_LLM_URL: {constants.VIDEO_LLM_URL}")
    logger.info(f"VLM_URL: {constants.VLM_URL}")
    logger.info(f"Azure endpoint: {endpoint}, deployment: {deployment}")

    # Load benchmark
    meta, samples = load_benchmark(args.benchmark_json)
    print(f"Loaded {len(samples)} samples from {args.benchmark_json}")

    # Slice data range
    if args.data_end > 0:
        samples = samples[args.data_start:args.data_end]
    elif args.data_start > 0:
        samples = samples[args.data_start:]
    print(f"Evaluating samples [{args.data_start}:{args.data_end}] → {len(samples)} samples")

    # Initialize model client (OpenAI-compatible vLLM server) + tokenizer for chat template
    system_prompt = constants.prompt + constants.format_prompt
    vllm_client = OpenAI(
        api_key="EMPTY",
        base_url=args.vllm_base_url,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    logger.info(f"Using vLLM server at {args.vllm_base_url}, model={args.model_name_or_path}")

    # Setup result directory
    run_id = f"{model_tag}_mt{args.max_turns}_s{args.data_start}_e{args.data_end}"
    save_dir = os.path.join(args.result_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    jsonl_path = os.path.join(save_dir, "results.jsonl")
    completed_ids = load_completed_ids(jsonl_path)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} samples already completed, skipping them")

    acc = 0
    evaluated = 0

    for i, sample in enumerate(tqdm(samples, desc="Evaluating...")):
        example_id = sample.get("example_id", f"sample_{i + args.data_start}")
        if example_id in completed_ids:
            continue

        question_text = format_question(sample)
        identity = sample["identity"]
        gt = sample["correct_answer"]
        options = sample.get("options", {})
        query_type = sample.get("query_type", "unknown")
        memory_type = QUERY_TYPE_TO_MEMORY_TYPE.get(query_type, "unknown")

        print(f'\n\n################# [{example_id}] Start Reasoning + Tool Calling ##################\n')
        print('=' * 20, 'system', '=' * 20)
        print(system_prompt)

        chat_history = [{"role": "system", "content": system_prompt}]
        user_input = question_text
        turn_cnt = 0
        output_text = ""

        while True:
            if turn_cnt >= args.max_turns:
                print(f"Turns exceed maximum: {args.max_turns}")
                break

            chat_history.append({"role": "user", "content": user_input})
            print('=' * 20, 'user_input', '=' * 20)
            print(user_input)

            if turn_cnt == args.max_turns - 1:
                # Last turn: use Azure OpenAI summarization
                output_message = summarize_with_azure(chat_history, question_text, options)
            else:
                # Regular turn: apply chat template and use completions endpoint
                client_input = tokenizer.apply_chat_template(
                    chat_history, add_generation_prompt=True, tokenize=False
                )
                try:
                    response = vllm_client.completions.create(
                        model=args.model_name_or_path,
                        prompt=client_input,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        n=1,
                        timeout=300,
                    )
                    output_message = response.choices[0].text
                except Exception as e:
                    logger.error(f"vLLM generation error: {e}")
                    output_message = summarize_with_azure(chat_history, question_text, options)
                    chat_history.append({"role": "assistant", "content": output_message})
                    output_text = output_message
                    break

            chat_history.append({"role": "assistant", "content": output_message})
            output_text = output_message

            print('=' * 20, 'output_text', '=' * 20)
            print(output_text)

            if '<answer>' in output_message and '</answer>' in output_message:
                # Model answered — re-summarize for consistency if not on last turn
                if turn_cnt != args.max_turns - 1:
                    revised = summarize_with_azure(chat_history[:-1], question_text, options)
                    output_text = revised
                    chat_history.append({
                        "role": "assistant",
                        "content": f"## Revised Summarized Answer: {revised}",
                    })
                    print('=' * 20, 'summarized_output_message', '=' * 20)
                    print(revised)
                break

            # Tool call processing
            tmp_query = process.get_query(output_text)
            tmp_query = process.parse_tool_call(tmp_query)
            if tmp_query:
                search_results = logged_tool_call(tmp_query, identity)
            else:
                logger.warning(f"No tool call parsed from output (turn {turn_cnt})")
                search_results = ""

            user_input = f"<information>{search_results}</information>"
            turn_cnt += 1

        # Score
        score = process.compute_score(output_text, gt)
        is_correct = score == 1.0
        if is_correct:
            acc += 1
        evaluated += 1

        # Build result
        result = {
            "sample_index": i + args.data_start,
            "example_id": example_id,
            "p_id": sample.get("p_id"),
            "identity": identity,
            "query_time": sample.get("query_time"),
            "query_type": query_type,
            "memory_type": memory_type,
            "question": sample.get("question"),
            "options": options,
            "correct_answer": gt,
            "predicted_answer": process.parse_answer(output_text),
            "raw_output": output_text,
            "is_correct": is_correct,
            "num_turns": turn_cnt + 1,
            "chat_history": chat_history,
        }

        # Save incrementally
        save_result_jsonl(jsonl_path, result)

        print(f"GT: {gt} | Pred: {result['predicted_answer']} | Correct: {is_correct}")
        print(f"Running accuracy: {acc}/{evaluated} = {acc / evaluated:.4f}")

    # Final metrics
    all_results = load_all_results_jsonl(jsonl_path)
    summary = build_accuracy_summary(all_results)
    print_accuracy_summary(summary)

    # Save final results JSON
    final_payload = {
        "model": args.model_name_or_path,
        "benchmark": args.benchmark_json,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": all_results,
    }
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    # Save metrics JSON
    metrics_path = os.path.join(save_dir, "metrics.json")
    metrics_payload = {
        "model": args.model_name_or_path,
        "benchmark": args.benchmark_json,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
