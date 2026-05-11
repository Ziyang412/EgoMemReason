from dataclasses import dataclass
from pydantic import BaseModel, field_validator
from typing import List

class SemanticRawOutput(BaseModel):
    semantic_triples: List[List[str]]
    episodic_evidence: List[List[int]]

    @field_validator("semantic_triples")
    def validate_semantic_triples(cls, v):
        if not all(len(triple) == 3 for triple in v):
            raise ValueError("Each semantic triple must contain exactly 3 elements.", v)
        return v
    
    # @field_validator("episodic_evidence")
    # def validate_evidence_length(cls, v, info):
    #     semantic_triples = info.data.get("semantic_triples", [])
    #     if len(v) != len(semantic_triples):
    #         raise ValueError("Length of semantic_triples and episodic_evidence must be the same.", v)
    #     return v


class ConsolidationRawOutput(BaseModel):
    updated_triple: List[str]
    triples_to_remove: List[int]

    @field_validator("updated_triple")
    def validate_updated_triple(cls, v):
        if len(v) != 3:
            raise ValueError("Updated triple must contain exactly 3 elements.", v)
        return v

    @field_validator("triples_to_remove")
    def validate_triples_to_remove(cls, v):
        if not all(isinstance(i, int) for i in v):
            raise ValueError("All indices in triples_to_remove must be integers.", v)
        return v


@dataclass
class SemanticOutput:
    chunk_id: str
    semantic_triples: List[List[str]]
    episodic_evidence: List[List[int]]