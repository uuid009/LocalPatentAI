"""本地 BGE Reranker 客户端（规格 4.4 节）。仅对已授权过滤后的候选执行精排。"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from localpatentai.core.config import Settings, get_settings
from localpatentai.core.errors import AppError
from localpatentai.knowledge.models import RankedChunk, RetrievedChunk

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_.\-]+")


class Reranker:
    """统一 Reranker 接口（规格 4.4 节）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RankedChunk]:
        if not candidates:
            return []
        limit_n = top_n if top_n is not None else self.settings.reranker_top_n
        limit_n = max(1, min(limit_n, len(candidates)))

        # 限制输入给 Reranker 的最大候选数，防止超时或内存耗尽
        capped = candidates[: self.settings.max_retrieval_candidates]

        if not self.settings.reranker_enabled or self.settings.reranker_mock:
            return _lexical_assisted_rerank(query, capped, limit_n)

        scores = _score_with_local_cross_encoder(query, capped, self.settings)
        ranked_pairs = sorted(
            zip(capped, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        results: list[RankedChunk] = []
        for chunk, score in ranked_pairs[:limit_n]:
            methods = list(chunk.retrieval_method)
            if "reranker" not in methods:
                methods.append("reranker")
            results.append(
                RankedChunk(
                    **chunk.model_dump(exclude={"retrieval_method"}),
                    retrieval_method=methods,
                    reranker_score=float(score),
                )
            )
        return results


def _lexical_assisted_rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int
) -> list[RankedChunk]:
    """当本地 CrossEncoder 未启用或处于 mock 模式时，基于融合分与精确术语覆盖率计算精排分。"""
    q_terms = set(TOKEN_RE.findall(query.lower()))
    for ch in query:
        if "\u4e00" <= ch <= "\u9fff":
            q_terms.add(ch)

    scored: list[tuple[RetrievedChunk, float]] = []
    for cand in candidates:
        base = float(cand.fusion_score or cand.bm25_score or cand.dense_score or 0.0)
        text_lower = cand.content.lower()
        hits = sum(1 for term in q_terms if term and term in text_lower)
        overlap_bonus = (hits / max(len(q_terms), 1)) * 0.15
        scored.append((cand, round(base + overlap_bonus, 6)))

    scored.sort(key=lambda item: item[1], reverse=True)
    results: list[RankedChunk] = []
    for cand, score in scored[:top_n]:
        results.append(
            RankedChunk(
                **cand.model_dump(),
                reranker_score=score,
            )
        )
    return results


@lru_cache(maxsize=2)
def _load_local_cross_encoder(model_path_str: str, device: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise AppError(
            500,
            "sentence_transformers_missing",
            "本地加载 BGE Reranker 需要安装 sentence-transformers",
        ) from exc

    settings = get_settings()
    resolved = settings.resolve_path(model_path_str)
    if not resolved.exists():
        raise AppError(
            500,
            "reranker_model_not_found",
            f"本地 BGE Reranker 模型目录不存在: {resolved}",
        )
    logger.info("loading_local_bge_reranker path=%s device=%s", resolved, device)
    return CrossEncoder(str(resolved), device=device)


def _score_with_local_cross_encoder(
    query: str,
    candidates: list[RetrievedChunk],
    settings: Settings,
    max_passage_chars: int = 1024,
) -> list[float]:
    model = _load_local_cross_encoder(settings.reranker_model_path, settings.reranker_device)
    pairs = [(query[:512], cand.content[:max_passage_chars]) for cand in candidates]
    try:
        raw_scores = model.predict(
            pairs,
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(s) for s in raw_scores]
    except Exception as exc:
        raise AppError(503, "reranker_inference_failed", "本地 Reranker 推理失败") from exc
