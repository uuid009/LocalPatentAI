"""RRF（Reciprocal Rank Fusion）排序融合算法（规格 5.5 节）。"""

from __future__ import annotations

from collections.abc import Sequence

from localpatentai.knowledge.models import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    RRF(d) = sum_{r in R} 1 / (k + rank_r(d))
    rank_r(d) 从 1 开始计数。
    """
    if k <= 0:
        raise ValueError("RRF 平滑常数 k 必须为正整数")
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item_id in enumerate(ranked, start=1):
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def fuse_retrieved_chunks(
    bm25_chunks: list[RetrievedChunk],
    dense_chunks: list[RetrievedChunk],
    *,
    rrf_k: int = 60,
    max_candidates: int = 50,
) -> list[RetrievedChunk]:
    bm25_ids = [c.chunk_id for c in bm25_chunks]
    dense_ids = [c.chunk_id for c in dense_chunks]
    fused_scores = reciprocal_rank_fusion([bm25_ids, dense_ids], k=rrf_k)

    by_id: dict[str, RetrievedChunk] = {}
    for item in bm25_chunks:
        by_id[item.chunk_id] = item.model_copy(deep=True)
    for item in dense_chunks:
        if item.chunk_id in by_id:
            existing = by_id[item.chunk_id]
            methods = list(existing.retrieval_method)
            for m in item.retrieval_method:
                if m not in methods:
                    methods.append(m)
            existing.retrieval_method = methods
            existing.dense_score = item.dense_score
        else:
            by_id[item.chunk_id] = item.model_copy(deep=True)

    merged: list[RetrievedChunk] = []
    for chunk_id, score in fused_scores[:max_candidates]:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        chunk.fusion_score = round(score, 6)
        merged.append(chunk)
    return merged
