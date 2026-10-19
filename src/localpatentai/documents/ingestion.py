"""文档导入、解析、结构化分块、索引构建、更新、删除与重建服务（规格 5.2 & 5.8）。"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from localpatentai.audit import record_audit_event
from localpatentai.auth.service import AuthorizationService
from localpatentai.core.config import Settings, get_settings
from localpatentai.core.errors import AppError, AuthorizationError, ResourceNotFoundError
from localpatentai.core.security import normalize_filename, resolve_safe_storage_path
from localpatentai.db.models import Chunk, Document, DocumentVersion, KnowledgeSpace
from localpatentai.documents.chunking import build_structure_aware_chunks
from localpatentai.documents.normalization import validate_file_type_and_magic
from localpatentai.documents.parser import parse_document_bytes
from localpatentai.knowledge.embedding import EmbeddingProvider
from localpatentai.projects.service import get_project_private_space


def _as_uuid(val: str | uuid.UUID, err_cls: type[AppError] = ResourceNotFoundError) -> uuid.UUID:
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except (ValueError, TypeError) as exc:
        raise err_cls() from exc


def ingest_project_document(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    project_id: str | uuid.UUID,
    filename: str,
    content: bytes,
    document_type: str = "technical_report",
    knowledge_space_id: str | uuid.UUID | None = None,
    auto_index: bool = True,
    settings: Settings | None = None,
) -> Document:
    """
    上传并导入文档（规格 5.2）：
    1. 校验用户对项目及目标知识空间的写权限（默认写入当前项目 project_private 空间，禁止隐式提升共享范围）。
    2. 校验文件大小、扩展名、文件头（Magic Bytes）与文件名规范化。
    3. 保存原始文件至服务端生成的随机 storage_key 并计算 SHA-256。
    4. 创建 Document 记录并（当 auto_index=True 时）执行解析、分块、Embedding 与索引入库。
    """
    runtime = settings or get_settings()
    authz = AuthorizationService(db)
    project = authz.assert_project_access(user_id, project_id, "upload_document")

    if len(content) > runtime.max_upload_bytes:
        raise AppError(
            413,
            "file_too_large",
            f"上传文件超过最大限制 ({runtime.max_upload_size_mb} MB)",
        )

    safe_name = normalize_filename(filename)
    ext_type, mime_type = validate_file_type_and_magic(safe_name, content)

    if knowledge_space_id is not None:
        space = authz.assert_knowledge_space_access(user_id, knowledge_space_id, "write")
        # 若指定了 project_private 空间，必须属于当前项目
        if space.scope_type == "project_private" and space.project_id != project.id:
            raise AuthorizationError(
                code="cross_project_space_forbidden",
                message="不能将文档写入其他项目的私有知识空间",
            )
        if space.scope_type == "group_shared" and space.group_id != project.group_id:
            raise AuthorizationError(
                code="cross_group_space_forbidden",
                message="不能将文档写入其他小组的共享知识空间",
            )
    else:
        space = get_project_private_space(db, project.id)

    sha256_hex = hashlib.sha256(content).hexdigest()
    suffix = Path(safe_name).suffix.lower() or f".{ext_type}"
    storage_key = f"{project.id.hex}/{uuid.uuid4().hex}{suffix}"
    upload_root = runtime.resolve_path(runtime.upload_dir)
    target_path = resolve_safe_storage_path(upload_root, storage_key)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)

    uid = _as_uuid(user_id)
    doc = Document(
        knowledge_space_id=space.id,
        project_id=project.id,
        uploaded_by=uid,
        original_filename=safe_name,
        document_type=document_type.strip() or "general",
        storage_key=storage_key,
        mime_type=mime_type,
        file_size=len(content),
        sha256=sha256_hex,
        status="uploaded",
        parser_version=runtime.parser_version,
    )
    db.add(doc)
    db.flush()

    record_audit_event(
        db,
        actor_user_id=uid,
        group_id=project.group_id,
        project_id=project.id,
        action="document.upload",
        resource_type="document",
        resource_id=doc.id,
        result="success",
        metadata={
            "knowledge_space_id": str(space.id),
            "scope_type": space.scope_type,
            "file_size": len(content),
            "sha256": sha256_hex,
        },
        commit=True,
    )

    if auto_index:
        index_document_version(db, user_id=uid, document_id=doc.id, settings=runtime)
    db.refresh(doc)
    return doc


def index_document_version(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    document_id: str | uuid.UUID,
    settings: Settings | None = None,
) -> DocumentVersion:
    """
    执行文档解析、结构化分块、BGE Embedding 计算与索引写入（规格 5.2 & 5.8）：
    - 若存在旧版本，先将旧版本标记为 superseded，避免新旧版本混用。
    - 失败时将 Document.status 设为 failed 并记录 error_code，绝不静默标为成功。
    """
    runtime = settings or get_settings()
    authz = AuthorizationService(db)
    doc_uuid = _as_uuid(document_id)
    doc = db.get(Document, doc_uuid)
    if doc is None or doc.status == "deleted":
        raise ResourceNotFoundError()

    if doc.project_id:
        authz.assert_project_access(user_id, doc.project_id, "reindex_document")
    else:
        authz.assert_knowledge_space_access(user_id, doc.knowledge_space_id, "write")

    space = db.get(KnowledgeSpace, doc.knowledge_space_id)
    if space is None or space.status != "active":
        raise ResourceNotFoundError()

    upload_root = runtime.resolve_path(runtime.upload_dir)
    file_path = resolve_safe_storage_path(upload_root, doc.storage_key)
    if not file_path.is_file():
        doc.status = "failed"
        doc.error_code = "source_file_missing"
        db.commit()
        raise AppError(404, "source_file_missing", "原始文档文件不存在，无法建立索引")

    content = file_path.read_bytes()
    ext_type, _ = validate_file_type_and_magic(doc.original_filename, content)

    try:
        doc.status = "parsing"
        db.flush()
        blocks = parse_document_bytes(
            file_ext_type=ext_type,
            content=content,
            filename=doc.original_filename,
            parser_version=runtime.parser_version,
        )
        doc.status = "parsed"
        db.flush()

        embedder = EmbeddingProvider(runtime)
        chunk_records = build_structure_aware_chunks(
            blocks,
            document_id=str(doc.id),
            knowledge_space_id=str(doc.knowledge_space_id),
            group_id=str(space.group_id) if space.group_id else None,
            project_id=str(doc.project_id) if doc.project_id else None,
            parser_version=runtime.parser_version,
            embedding_model_id=embedder.model_id,
            target_tokens=runtime.chunk_size,
            overlap_tokens=runtime.chunk_overlap,
        )
        if not chunk_records:
            raise AppError(422, "empty_chunks", "文档切分后未产生有效片段")

        doc.status = "indexing"
        db.flush()
        vectors = embedder.embed_documents([c.content for c in chunk_records])

        # 计算新版本号，并将旧版本置为 superseded 并清理旧版本关联的活跃检索块
        existing_versions = db.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == doc.id)
            .order_by(DocumentVersion.version.desc())
        ).all()
        next_ver_num = (existing_versions[0].version + 1) if existing_versions else 1
        if existing_versions:
            old_ver_ids = [v.id for v in existing_versions]
            db.execute(
                update(DocumentVersion)
                .where(DocumentVersion.id.in_(old_ver_ids))
                .values(status="superseded")
            )
            db.execute(delete(Chunk).where(Chunk.document_version_id.in_(old_ver_ids)))

        new_version = DocumentVersion(
            document_id=doc.id,
            version=next_ver_num,
            sha256=doc.sha256,
            parser_version=runtime.parser_version,
            embedding_model_id=embedder.model_id,
            index_version=runtime.index_version,
            status="active",
        )
        db.add(new_version)
        db.flush()

        for idx, (rec, vec) in enumerate(zip(chunk_records, vectors, strict=True)):
            db.add(
                Chunk(
                    id=_as_uuid(rec.chunk_id),
                    document_version_id=new_version.id,
                    document_id=doc.id,
                    knowledge_space_id=doc.knowledge_space_id,
                    group_id=space.group_id,
                    project_id=doc.project_id,
                    chunk_index=idx,
                    page_start=rec.page_start,
                    page_end=rec.page_end,
                    section_path_json=rec.section_path,
                    paragraph_ids_json=rec.paragraph_ids,
                    content=rec.content,
                    content_hash=rec.content_hash,
                    token_count=rec.token_count,
                    parser_version=rec.parser_version,
                    embedding_model_id=rec.embedding_model_id,
                    metadata_json=rec.metadata,
                    embedding=vec,
                )
            )

        doc.status = "indexed"
        doc.parser_version = runtime.parser_version
        doc.error_code = None

        record_audit_event(
            db,
            actor_user_id=user_id,
            group_id=space.group_id,
            project_id=doc.project_id,
            action="document.index",
            resource_type="document",
            resource_id=doc.id,
            result="success",
            metadata={
                "version": next_ver_num,
                "chunk_count": len(chunk_records),
                "index_version": runtime.index_version,
                "embedding_model_id": embedder.model_id,
            },
            commit=True,
        )
        db.refresh(new_version)
        return new_version
    except AppError as exc:
        doc.status = "failed"
        doc.error_code = exc.code
        db.commit()
        raise
    except Exception as exc:
        doc.status = "failed"
        doc.error_code = type(exc).__name__
        db.commit()
        raise AppError(500, "indexing_failed", "文档解析或索引构建失败") from exc


def delete_document_completely(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    document_id: str | uuid.UUID,
    settings: Settings | None = None,
) -> Document:
    """
    彻底删除文档（规格 5.8）：
    清理原始文件、Chunk 记录（含 BM25 tsvector 与向量点）、将版本置为 deleted，并记录审计事件。
    """
    runtime = settings or get_settings()
    authz = AuthorizationService(db)
    doc_uuid = _as_uuid(document_id)
    doc = db.get(Document, doc_uuid)
    if doc is None or doc.status == "deleted":
        raise ResourceNotFoundError()

    if doc.project_id:
        project = authz.assert_project_access(user_id, doc.project_id, "delete_document")
        group_id = project.group_id
    else:
        space = authz.assert_knowledge_space_access(user_id, doc.knowledge_space_id, "write")
        group_id = space.group_id

    doc.status = "deleting"
    db.flush()

    # 删除所有关联 Chunk（同时从向量列与全文索引移除）
    db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    db.execute(
        update(DocumentVersion)
        .where(DocumentVersion.document_id == doc.id)
        .values(status="deleted")
    )

    upload_root = runtime.resolve_path(runtime.upload_dir)
    try:
        file_path = resolve_safe_storage_path(upload_root, doc.storage_key)
        if file_path.is_file():
            file_path.unlink()
    except Exception:
        pass

    doc.status = "deleted"
    record_audit_event(
        db,
        actor_user_id=user_id,
        group_id=group_id,
        project_id=doc.project_id,
        action="document.delete",
        resource_type="document",
        resource_id=doc.id,
        result="success",
        commit=True,
    )
    db.refresh(doc)
    return doc


def list_project_documents(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    project_id: str | uuid.UUID,
) -> list[Document]:
    authz = AuthorizationService(db)
    project = authz.assert_project_access(user_id, project_id, "view")
    return list(
        db.scalars(
            select(Document)
            .where(
                Document.project_id == project.id,
                Document.status != "deleted",
            )
            .order_by(Document.created_at.desc())
        ).all()
    )


def get_authorized_document(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    document_id: str | uuid.UUID,
) -> Document:
    authz = AuthorizationService(db)
    doc_uuid = _as_uuid(document_id)
    doc = db.get(Document, doc_uuid)
    if doc is None or doc.status == "deleted":
        raise ResourceNotFoundError()
    if doc.project_id:
        authz.assert_project_access(user_id, doc.project_id, "view")
    else:
        authz.assert_knowledge_space_access(user_id, doc.knowledge_space_id, "read")
    return doc
