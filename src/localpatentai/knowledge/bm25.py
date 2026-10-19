"""BM25 词法检索引擎（规格 5.5 & 5.6）。先限定已授权知识空间与 indexed 状态文档，再计算 BM25 Okapi 得分。"""

from __future__ import annotations

import math
import re
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from localpatentai.core.errors import AuthorizationError
from localpatentai.db.models import Chunk, Document, DocumentVersion, KnowledgeSpace
from localpatentai.knowledge.models import RetrievedChunk

CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
ALNUM_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*", re.IGNORECASE)


def tokenize_chinese_and_english(text: str) -> list[str]:
    """针对中文技术与专利语料的轻量级分词器：英文/数字词元 + 中文单字与二元词（bigrams）。"""
    lowered = text.lower()
    tokens: list[str] = []
    tokens.extend(ALNUM_TOKEN_RE.findall(lowered))
    for run in CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(list(run))
            for idx in range(len(run) - 1):
                tokens.append(run[idx : idx + 2])
    return tokens


def compute_bm25_scores(
    query_tokens: list[str],
    corpus_tokens: list[list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not query_tokens or not corpus_tokens:
        return [0.0] * len(corpus_tokens)

    n_docs = len(corpus_tokens)
    doc_lens = [len(doc) for doc in corpus_tokens]
    avgdl = sum(doc_lens) / max(n_docs, 1) or 1.0

    # 统计每个查询词的文档频率 df
    unique_q = list(dict.fromkeys(query_tokens))
    df: dict[str, int] = {t: 0 for t in unique_q}
    doc_term_freqs: list[dict[str, int]] = []

    for doc in corpus_tokens:
        tf_map: dict[str, int] = {}
        for tok in doc:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        doc_term_freqs.append(tf_map)
        for t in unique_q:
            if t in tf_map:
                df[t] += 1

    idf: dict[str, float] = {}
    for t, freq in df.items():
        idf[t] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    scores: list[float] = []
    for idx, tf_map in enumerate(doc_term_freqs):
        dl = doc_lens[idx]
        score = 0.0
        for t in query_tokens:
            tf = tf_map.get(t, 0)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf.get(t, 0.0) * (numerator / denominator)
        scores.append(score)
    return scores


def search_bm25_in_authorized_spaces(
    db: Session,
    *,
    query: str,
    authorized_space_ids: list[str],
    top_k: int = 30,
    document_types: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    严格遵守规格 5.6：
    先在 SQL 层强制限定 `knowledge_space_id IN authorized_space_ids` 且 `Document.status == 'indexed'`
    且 `DocumentVersion.status == 'active'`，绝不加载任何未授权文档到内存。
    """
    if not authorized_space_ids:
        raise AuthorizationError(
            code="empty_authorized_spaces",
            message="未提供有效的已授权知识空间列表，拒绝执行 BM25 检索",
        )

    space_uuids = [uuid.UUID(str(sid)) for sid in authorized_space_ids]
    conditions = [
        Chunk.knowledge_space_id.in_(space_uuids),
        KnowledgeSpace.status == "active",
        Document.status == "indexed",
        DocumentVersion.status == "active",
    ]
    if document_types:
        conditions.append(Document.document_type.in_(document_types))
    if document_ids:
        doc_uuids = [uuid.UUID(str(did)) for did in document_ids]
        conditions.append(Document.id.in_(doc_uuids))

    stmt = (
        select(Chunk, Document, DocumentVersion)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .join(Document, Document.id == Chunk.document_id)
        .join(KnowledgeSpace, KnowledgeSpace.id == Chunk.knowledge_space_id)
        .where(*conditions)
    )
    rows = db.execute(stmt).all()
    if not rows:
        return []

    q_tokens = tokenize_chinese_and_english(query)
    if not q_tokens:
        return []

    corpus_tokens = [tokenize_chinese_and_english(chunk.content) for chunk, _, _ in rows]
    scores = compute_bm25_scores(q_tokens, corpus_tokens)

    scored_rows = [
        (chunk, doc, ver, score)
        for (chunk, doc, ver), score in zip(rows, scores, strict=True)
        if score > 0.0
    ]
    scored_rows.sort(key=lambda item: item[3], reverse=True)

    results: list[RetrievedChunk] = []
    for chunk, doc, ver, score in scored_rows[:top_k]:
        results.append(
            RetrievedChunk(
                chunk_id=str(chunk.id),
                document_id=str(doc.id),
                document_name=doc.original_filename,
                document_type=doc.document_type,
                knowledge_space_id=str(chunk.knowledge_space_id),
                group_id=str(chunk.group_id) if chunk.group_id else None,
                project_id=str(chunk.project_id) if chunk.project_id else None,
                content=chunk.content,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=list(chunk.section_path_json or []),
                paragraph_ids=list(chunk.paragraph_ids_json or []),
                source_hash=doc.sha256,
                index_version=ver.index_version,
                parser_version=chunk.parser_version,
                retrieval_method=["bm25"],
                bm25_score=round(float(score), 6),
                metadata=dict(chunk.metadata_json or {}),
            )
        )
    return results
