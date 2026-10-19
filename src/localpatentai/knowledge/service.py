"""RAG / Knowledge 服务门面（规格 3.2 & 第 5 章）。统一执行权限推导、检索与知识空间重建。"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from localpatentai.audit import record_audit_event
from localpatentai.auth.service import AuthorizationService
from localpatentai.core.config import Settings, get_settings
from localpatentai.db.models import Document
from localpatentai.documents.ingestion import index_document_version
from localpatentai.knowledge.models import EvidencePack
from localpatentai.knowledge.retrieval import RetrievalMode, execute_authorized_retrieval


class KnowledgeService:
    """
    受控的 RAG 服务层：
    - LangGraph 节点与 API 统一通过此服务执行检索；
    - 强制从服务端 AuthorizationService.resolve_search_scopes 推导可访问空间，忽略客户端未验证的越权空间。
    """

    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.authz = AuthorizationService(db)

    def search_for_project(
        self,
        *,
        user_id: str | uuid.UUID,
        project_id: str | uuid.UUID,
        query: str,
        top_k: int | None = None,
        mode: RetrievalMode = "hybrid_rrf_rerank",
        document_types: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> EvidencePack:
        authorized_spaces = self.authz.resolve_search_scopes(user_id, project_id)
        pack = execute_authorized_retrieval(
            self.db,
            query=query,
            authorized_space_ids=authorized_spaces,
            project_id=str(project_id),
            top_k=top_k,
            mode=mode,
            document_types=document_types,
            document_ids=document_ids,
            settings=self.settings,
        )
        record_audit_event(
            self.db,
            actor_user_id=user_id,
            project_id=project_id,
            action="knowledge.search",
            resource_type="project",
            resource_id=project_id,
            result="success",
            metadata={
                "mode": mode,
                "top_k": top_k or self.settings.reranker_top_n,
                "authorized_space_count": len(authorized_spaces),
                "evidence_ids": [it.evidence_id for it in pack.items],
                "chunk_ids": [it.chunk_id for it in pack.items],
                "latency_ms": pack.retrieval_latency_ms,
                "index_version": pack.index_version,
            },
            commit=True,
        )
        return pack

    def reindex_knowledge_space(
        self,
        *,
        user_id: str | uuid.UUID,
        knowledge_space_id: str | uuid.UUID,
    ) -> dict:
        space = self.authz.assert_knowledge_space_access(user_id, knowledge_space_id, "write")
        docs = self.db.scalars(
            select(Document).where(
                Document.knowledge_space_id == space.id,
                Document.status != "deleted",
            )
        ).all()
        reindexed_count = 0
        for doc in docs:
            index_document_version(
                self.db,
                user_id=user_id,
                document_id=doc.id,
                settings=self.settings,
            )
            reindexed_count += 1
        return {
            "knowledge_space_id": str(space.id),
            "reindexed_documents": reindexed_count,
            "index_version": self.settings.index_version,
        }
