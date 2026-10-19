"""混合检索编排器（支持 bm25_only、dense_only、hybrid_rrf、hybrid_rrf_rerank 四种模式以便离线消融评测）。"""

from __future__ import annotations

import time
from typing import Literal

from sqlalchemy.orm import Session

from localpatentai.core.config import Settings, get_settings
from localpatentai.core.errors import AppError, AuthorizationError
from localpatentai.knowledge.bm25 import search_bm25_in_authorized_spaces
from localpatentai.knowledge.embedding import EmbeddingProvider
from localpatentai.knowledge.evidence import build_evidence_pack
from localpatentai.knowledge.fusion import fuse_retrieved_chunks
from localpatentai.knowledge.models import EvidencePack, RankedChunk, RetrievedChunk
from localpatentai.knowledge.reranker import Reranker
from localpatentai.knowledge.vector_store import search_vectors_in_authorized_spaces

RetrievalMode = Literal["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rrf_rerank"]


def execute_authorized_retrieval(
    db: Session,
    *,
    query: str,
    authorized_space_ids: list[str],
    project_id: str | None = None,
    top_k: int | None = None,
    mode: RetrievalMode = "hybrid_rrf_rerank",
    document_types: list[str] | None = None,
    document_ids: list[str] | None = None,
    settings: Settings | None = None,
) -> EvidencePack:
    runtime = settings or get_settings()
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise AppError(400, "empty_query", "检索查询不能为空")
    if not authorized_space_ids:
        raise AuthorizationError(
            code="no_authorized_spaces",
            message="未提供服务端推导的已授权知识空间列表，拒绝检索",
        )

    t0 = time.perf_counter()
    final_top_n = top_k if top_k is not None else runtime.reranker_top_n
    final_top_n = max(1, min(final_top_n, runtime.max_retrieval_candidates))

    bm25_hits: list[RetrievedChunk] = []
    dense_hits: list[RetrievedChunk] = []

    if mode in {"bm25_only", "hybrid_rrf", "hybrid_rrf_rerank"}:
        bm25_hits = search_bm25_in_authorized_spaces(
            db,
            query=cleaned_query,
            authorized_space_ids=authorized_space_ids,
            top_k=runtime.bm25_top_k,
            document_types=document_types,
            document_ids=document_ids,
        )

    if mode in {"dense_only", "hybrid_rrf", "hybrid_rrf_rerank"}:
        embedder = EmbeddingProvider(runtime)
        q_vec = embedder.embed_query(cleaned_query)
        dense_hits = search_vectors_in_authorized_spaces(
            db,
            query_vector=q_vec,
            authorized_space_ids=authorized_space_ids,
            top_k=runtime.dense_top_k,
            document_types=document_types,
            document_ids=document_ids,
        )

    if mode == "bm25_only":
        candidates = bm25_hits[:final_top_n]
        ranked = [
            RankedChunk(
                **c.model_dump(),
                reranker_score=c.bm25_score,
            )
            for c in candidates
        ]
    elif mode == "dense_only":
        candidates = dense_hits[:final_top_n]
        ranked = [
            RankedChunk(
                **c.model_dump(),
                reranker_score=c.dense_score,
            )
            for c in candidates
        ]
    elif mode == "hybrid_rrf":
        fused = fuse_retrieved_chunks(
            bm25_hits,
            dense_hits,
            rrf_k=runtime.rrf_k,
            max_candidates=runtime.max_retrieval_candidates,
        )
        ranked = [
            RankedChunk(
                **c.model_dump(),
                reranker_score=c.fusion_score,
            )
            for c in fused[:final_top_n]
        ]
    else:
        fused = fuse_retrieved_chunks(
            bm25_hits,
            dense_hits,
            rrf_k=runtime.rrf_k,
            max_candidates=runtime.max_retrieval_candidates,
        )
        reranker = Reranker(runtime)
        ranked = reranker.rerank(cleaned_query, fused, top_n=final_top_n)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return build_evidence_pack(
        query=cleaned_query,
        ranked_chunks=ranked,
        authorized_space_ids=authorized_space_ids,
        project_id=project_id,
        index_version=runtime.index_version,
        latency_ms=elapsed_ms,
    )
