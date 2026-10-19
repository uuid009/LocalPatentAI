"""Evidence Pack 组装器（规格 5.7 节）：分类证据类型并区分直接实验记载与外部现有技术。"""

from __future__ import annotations

import re

from localpatentai.knowledge.models import (
    EvidenceItem,
    EvidencePack,
    EvidenceType,
    RankedChunk,
    SupportStatus,
)

EXPERIMENT_RE = re.compile(
    r"(?:实验|测试|测量|表\s*\d+|对照组|实施例|准确率|耗时|温度|压力|MPa|ms|℃|%|\d+\.\d+)"
)
PRIOR_ART_RE = re.compile(r"(?:现有技术|背景技术|对比文件|CN\d{6,}|US\d{6,}|专利号|公开号)")
DERIVED_RE = re.compile(r"(?:计算得|由此推导|相比.*提升|降低.*%|公式)")


def infer_evidence_type(chunk: RankedChunk, current_project_id: str | None) -> EvidenceType:
    text = chunk.content
    # 若来自通用文献空间或明确包含对比文件特征，标为 external_prior_art
    if (
        chunk.project_id is None
        and chunk.group_id is None
        or PRIOR_ART_RE.search(text)
    ):
        if current_project_id and chunk.project_id == str(current_project_id):
            if PRIOR_ART_RE.search(text):
                return "external_prior_art"
        elif PRIOR_ART_RE.search(text) or chunk.project_id is None:
            return "external_prior_art"

    if EXPERIMENT_RE.search(text) or chunk.metadata.get("block_type") in {
        "table",
        "experiment_step",
    }:
        return "experimental_observation"
    if DERIVED_RE.search(text) or chunk.metadata.get("block_type") == "formula":
        return "derived_result"
    return "direct_fact"


def determine_support_status(chunk: RankedChunk) -> SupportStatus:
    flags = chunk.metadata.get("quality_flags") or []
    if "low_ocr_confidence" in flags or "prompt_injection_neutralized" in flags:
        return "needs_confirmation"
    score = chunk.reranker_score or chunk.fusion_score or 0.0
    if score <= 0.005:
        return "partially_supported"
    return "supported"


def build_evidence_pack(
    *,
    query: str,
    ranked_chunks: list[RankedChunk],
    authorized_space_ids: list[str],
    project_id: str | None = None,
    index_version: str = "v1",
    latency_ms: float = 0.0,
    rewritten_queries: list[str] | None = None,
) -> EvidencePack:
    items: list[EvidenceItem] = []
    for idx, chunk in enumerate(ranked_chunks, start=1):
        ev_type = infer_evidence_type(chunk, project_id)
        support = determine_support_status(chunk)
        items.append(
            EvidenceItem(
                evidence_id=f"E{idx}",
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                knowledge_space_id=chunk.knowledge_space_id,
                quote=chunk.content,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=list(chunk.section_path),
                paragraph_ids=list(chunk.paragraph_ids),
                evidence_type=ev_type,
                support_status=support,
                retrieval_method=list(chunk.retrieval_method),
                retrieval_score=chunk.fusion_score or chunk.bm25_score or chunk.dense_score,
                reranker_score=chunk.reranker_score,
                source_hash=chunk.source_hash,
                index_version=chunk.index_version or index_version,
            )
        )

    insufficient = len(items) == 0
    missing: list[str] = []
    if insufficient:
        missing.append("未在已授权知识空间内检索到与查询直接匹配的技术证据，相关技术主张需人工确认")

    return EvidencePack(
        query=query,
        rewritten_queries=rewritten_queries or [query],
        authorized_space_ids=list(authorized_space_ids),
        items=items,
        index_version=index_version,
        retrieval_latency_ms=round(latency_ms, 2),
        insufficient_evidence=insufficient,
        missing_aspects=missing,
    )
