"""向量存储与余弦相似度检索引擎（规格 5.5 & 5.6）。在索引查询阶段强制执行授权空间过滤。"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from localpatentai.core.errors import AuthorizationError
from localpatentai.db.models import Chunk, Document, DocumentVersion, KnowledgeSpace
from localpatentai.knowledge.models import RetrievedChunk


def search_vectors_in_authorized_spaces(
    db: Session,
    *,
    query_vector: list[float],
    authorized_space_ids: list[str],
    top_k: int = 30,
    document_types: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    在向量检索查询阶段直接应用 `Chunk.knowledge_space_id IN authorized_space_ids`
    以及 `Document.status == 'indexed'` 过滤，绝不先全库召回再过滤。
    """
    if not authorized_space_ids:
        raise AuthorizationError(
            code="empty_authorized_spaces",
            message="未提供有效的已授权知识空间列表，拒绝执行向量检索",
        )

    space_uuids = [uuid.UUID(str(sid)) for sid in authorized_space_ids]
    conditions = [
        Chunk.knowledge_space_id.in_(space_uuids),
        KnowledgeSpace.status == "active",
        Document.status == "indexed",
        DocumentVersion.status == "active",
        Chunk.embedding.is_not(None),
    ]
    if document_types:
        conditions.append(Document.document_type.in_(document_types))
    if document_ids:
        doc_uuids = [uuid.UUID(str(did)) for did in document_ids]
        conditions.append(Document.id.in_(doc_uuids))

    distance_expr = Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(Chunk, Document, DocumentVersion, distance_expr)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .join(Document, Document.id == Chunk.document_id)
        .join(KnowledgeSpace, KnowledgeSpace.id == Chunk.knowledge_space_id)
        .where(*conditions)
        .order_by(distance_expr.asc())
        .limit(top_k)
    )
    rows = db.execute(stmt).all()

    results: list[RetrievedChunk] = []
    for chunk, doc, ver, dist in rows:
        similarity = max(0.0, 1.0 - float(dist))
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
                retrieval_method=["dense"],
                dense_score=round(similarity, 6),
                metadata=dict(chunk.metadata_json or {}),
            )
        )
    return results
