import json
import os
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

from .utils import SemanticRawOutput, SemanticOutput
from ...llm import LLMModel, PromptTemplateManager

logger = logging.getLogger(__name__)

class SemanticExtraction:
    def __init__(self, llm_model: LLMModel):
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model

    def semantic_extraction(self, chunk_key: str, episodic_triples: List[List[str]]) -> SemanticOutput:
        # PREPROCESSING
        formatted_triples = "\n".join(f"{i}. {triple}" for i, triple in enumerate(episodic_triples))
        messages = self.prompt_template_manager.render(name='semantic_extraction', episodic_triples=formatted_triples)

        try:
            # LLM INFERENCE (entire try-block is retried by the decorator)
            response = self.llm_model.generate(messages, text_format=SemanticRawOutput)

        except Exception as e:
            logger.warning(e)
            return SemanticOutput(
                chunk_id=chunk_key,
                semantic_triples=[],
                episodic_evidence=[]
            )

        return SemanticOutput(
            chunk_id=chunk_key,
            semantic_triples=response.semantic_triples,
            episodic_evidence=response.episodic_evidence
        )
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "."):
        """
        Save extraction results to a JSON file.
        
        Args:
            results: The results dictionary to save
            output_dir: Output directory path.
        """

        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if hasattr(value, '__dict__'):
                json_results[key] = value.__dict__
            else:
                json_results[key] = value
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"semantic_extraction_results_{self.llm_model.model_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

    def batch_semantic_extraction(self, episodic_triples_batch: Dict[str, List[List[str]]], output_dir: str = ".") -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[int]]]]:
        """
        Conduct batch semantic extraction synchronously using multi-threading.

        Args:
            episodic_triples_batch: A dictionary mapping chunk IDs to lists of episodic triples.
            output_dir (str): Directory to save output file.

        Returns:
            Tuple[Dict[str, List[List[str]]], Dict[str, List[List[int]]]]
                - A dict with keys as the chunk ids (mdhash) and values as the semantic triples
                - A dict with keys as the chunk ids (mdhash) and values as the episodic evidence indices
        """
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.semantic_extraction, chunk_key, episodic_triples): episodic_triples
                for chunk_key, episodic_triples in episodic_triples_batch.items()
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Extracting semantic triples")
            for future in pbar:
                result = future.result()
                results.append(result)

        semantic_triples_map = {res.chunk_id: res.semantic_triples for res in results}
        episodic_evidence_map = {res.chunk_id: res.episodic_evidence for res in results}

        chunk_keys = list(episodic_triples_batch.keys())

        ordered_semantic_triples = {key: semantic_triples_map.get(key, []) for key in chunk_keys}
        ordered_episodic_evidence = {key: episodic_evidence_map.get(key, []) for key in chunk_keys}

        combined_results = {
            "semantic_triples": ordered_semantic_triples,
            # "episodic_evidence": ordered_episodic_evidence,
        }
        self.save_results(combined_results, output_dir)

        return ordered_semantic_triples, ordered_episodic_evidence
