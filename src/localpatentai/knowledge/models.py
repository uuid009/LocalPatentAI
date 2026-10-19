"""知识与 RAG 数据模型（严格使用 Pydantic v2 Field(default_factory=...)）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceType = Literal[
    "direct_fact",
    "experimental_observation",
    "derived_result",
    "inference",
    "external_prior_art",
]

SupportStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "needs_confirmation",
]


class ParsedSectionBlock(BaseModel):
    block_type: Literal[
        "heading",
        "paragraph",
        "table",
        "claim",
        "experiment_step",
        "formula",
        "figure_caption",
    ] = "paragraph"
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    paragraph_id: str | None = None
    table_id: str | None = None
    figure_id: str | None = None
    quality_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    knowledge_space_id: str
    group_id: str | None = None
    project_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    paragraph_ids: list[str] = Field(default_factory=list)
    content: str
    content_hash: str
    token_count: int = 0
    parser_version: str = "v1.0"
    embedding_model_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str = ""
    document_type: str = "general"
    knowledge_space_id: str
    group_id: str | None = None
    project_id: str | None = None
    content: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    paragraph_ids: list[str] = Field(default_factory=list)
    source_hash: str
    index_version: str = "v1"
    parser_version: str = "v1.0"
    retrieval_method: list[str] = Field(default_factory=list)
    bm25_score: float | None = None
    dense_score: float | None = None
    fusion_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedChunk(RetrievedChunk):
    reranker_score: float | None = None


class EvidenceItem(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    document_name: str = ""
    knowledge_space_id: str
    quote: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    paragraph_ids: list[str] = Field(default_factory=list)
    evidence_type: EvidenceType = "direct_fact"
    support_status: SupportStatus = "supported"
    retrieval_method: list[str] = Field(default_factory=list)
    retrieval_score: float | None = None
    reranker_score: float | None = None
    source_hash: str
    index_version: str = "v1"


class EvidencePack(BaseModel):
    query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    authorized_space_ids: list[str] = Field(default_factory=list)
    items: list[EvidenceItem] = Field(default_factory=list)
    index_version: str = "v1"
    retrieval_latency_ms: float = 0.0
    insufficient_evidence: bool = False
    missing_aspects: list[str] = Field(default_factory=list)
