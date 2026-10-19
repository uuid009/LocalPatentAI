"""API 路由实现（认证、项目、文档、检索、工作流与产物下载，规格第 10 章）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from localpatentai.api.dependencies import get_current_user
from localpatentai.api.schemas import (
    ArtifactResponse,
    AuthResponse,
    DemoLoginRequest,
    DocumentResponse,
    HumanInputRequest,
    KnowledgeSpaceResponse,
    LoginRequest,
    PatentTaskCreateRequest,
    ProjectCreateRequest,
    ProjectMemberAddRequest,
    ProjectResponse,
    ProjectUpdateRequest,
    RegisterRequest,
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
    TaskStatusResponse,
)
from localpatentai.auth.service import (
    AuthorizationService,
    ensure_demo_group_user,
    login_with_password,
    register_user_in_group,
)
from localpatentai.core.errors import AuthorizationError
from localpatentai.db.models import Document, Group, GroupMember, User
from localpatentai.db.session import get_db
from localpatentai.documents.ingestion import (
    delete_document_completely,
    get_authorized_document,
    index_document_version,
    ingest_project_document,
    list_project_documents,
)
from localpatentai.knowledge.service import KnowledgeService
from localpatentai.projects.service import (
    add_project_member,
    create_project,
    get_project_for_user,
    get_project_private_space,
    list_accessible_projects,
    list_project_spaces,
    remove_project_member,
    update_project_metadata,
)
from localpatentai.tasks.service import (
    create_and_start_patent_task,
    get_authorized_artifact_for_download,
    get_authorized_task,
    list_task_artifacts,
    resume_recoverable_task,
    submit_human_input_or_approval,
)

router = APIRouter(prefix="/api/v1")


def _to_project_response(db: Session, proj) -> ProjectResponse:
    priv_id = None
    try:
        priv = get_project_private_space(db, proj.id)
        priv_id = str(priv.id)
    except Exception:
        priv_id = None
    return ProjectResponse(
        id=str(proj.id),
        name=proj.name,
        description=proj.description,
        owner_id=str(proj.owner_id),
        group_id=str(proj.group_id),
        status=proj.status,
        private_space_id=priv_id,
        created_at=proj.created_at,
        updated_at=proj.updated_at,
    )


def _to_document_response(doc: Document) -> DocumentResponse:
    return DocumentResponse(
        id=str(doc.id),
        knowledge_space_id=str(doc.knowledge_space_id),
        project_id=str(doc.project_id) if doc.project_id else None,
        uploaded_by=str(doc.uploaded_by),
        original_filename=doc.original_filename,
        document_type=doc.document_type,
        mime_type=doc.mime_type,
        file_size=doc.file_size,
        sha256=doc.sha256,
        status=doc.status,
        parser_version=doc.parser_version,
        error_code=doc.error_code,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def _to_task_response(task) -> TaskStatusResponse:
    snap = task.state_snapshot_json or {}
    return TaskStatusResponse(
        id=str(task.id),
        project_id=str(task.project_id),
        created_by=str(task.created_by),
        title=task.title,
        status=task.status,
        current_stage=task.current_stage,
        workflow_version=task.workflow_version,
        revision_count=task.revision_count,
        llm_calls_used=task.llm_calls_used,
        human_approval_status=task.human_approval_status,
        error_code=task.error_code,
        technical_analysis=snap.get("technical_analysis"),
        evidence_research=snap.get("evidence_research"),
        evidence_gaps=snap.get("evidence_gaps") or [],
        draft=snap.get("draft"),
        review_result=snap.get("review_result"),
        artifact_id=snap.get("artifact_id"),
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


# ==================== 10.0 认证接口 ====================


@router.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def api_register(payload: RegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user, group, token = register_user_in_group(
        db,
        username=payload.username,
        password=payload.password,
        group_code=payload.group_code,
        display_name=payload.display_name,
        group_role=payload.group_role,
    )
    return AuthResponse(
        user_id=str(user.id),
        username=user.username,
        display_name=user.display_name,
        group_id=str(group.id),
        group_code=group.code,
        group_name=group.name,
        token=token,
    )


@router.post("/auth/login", response_model=AuthResponse)
def api_login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user, token = login_with_password(db, payload.username, payload.password)
    row = db.execute(
        select(Group)
        .join(GroupMember, GroupMember.group_id == Group.id)
        .where(GroupMember.user_id == user.id)
    ).first()
    group = row[0] if row else None
    return AuthResponse(
        user_id=str(user.id),
        username=user.username,
        display_name=user.display_name,
        group_id=str(group.id) if group else "",
        group_code=group.code if group else "",
        group_name=group.name if group else "",
        token=token,
    )


@router.post("/auth/demo-login", response_model=AuthResponse)
def api_demo_login(payload: DemoLoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user, group, token = ensure_demo_group_user(db, payload.group_code)
    return AuthResponse(
        user_id=str(user.id),
        username=user.username,
        display_name=user.display_name,
        group_id=str(group.id),
        group_code=group.code,
        group_name=group.name,
        token=token,
    )


# ==================== 10.1 项目接口 ====================


@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def api_create_project(
    payload: ProjectCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProjectResponse:
    proj, priv_space = create_project(
        db,
        user_id=current_user.id,
        name=payload.name,
        description=payload.description,
        group_code=payload.group_code,
    )
    return ProjectResponse(
        id=str(proj.id),
        name=proj.name,
        description=proj.description,
        owner_id=str(proj.owner_id),
        group_id=str(proj.group_id),
        status=proj.status,
        private_space_id=str(priv_space.id),
        created_at=proj.created_at,
        updated_at=proj.updated_at,
    )


@router.get("/projects", response_model=list[ProjectResponse])
def api_list_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ProjectResponse]:
    projects = list_accessible_projects(db, current_user.id)
    return [_to_project_response(db, p) for p in projects]


@router.get("/projects/{project_id}", response_model=ProjectResponse)
def api_get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProjectResponse:
    proj = get_project_for_user(db, current_user.id, project_id)
    return _to_project_response(db, proj)


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
def api_update_project(
    project_id: str,
    payload: ProjectUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProjectResponse:
    proj = update_project_metadata(
        db,
        user_id=current_user.id,
        project_id=project_id,
        name=payload.name,
        description=payload.description,
        status=payload.status,
    )
    return _to_project_response(db, proj)


@router.post("/projects/{project_id}/members", status_code=status.HTTP_201_CREATED)
def api_add_member(
    project_id: str,
    payload: ProjectMemberAddRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    member = add_project_member(
        db,
        actor_user_id=current_user.id,
        project_id=project_id,
        target_user_id=payload.user_id,
        role=payload.role,
    )
    return {
        "project_id": str(member.project_id),
        "user_id": str(member.user_id),
        "role": member.role,
    }


@router.delete("/projects/{project_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def api_remove_member(
    project_id: str,
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    remove_project_member(
        db,
        actor_user_id=current_user.id,
        project_id=project_id,
        target_user_id=user_id,
    )


@router.get("/projects/{project_id}/knowledge-spaces", response_model=list[KnowledgeSpaceResponse])
def api_list_project_spaces(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[KnowledgeSpaceResponse]:
    spaces = list_project_spaces(db, current_user.id, project_id)
    return [
        KnowledgeSpaceResponse(
            id=str(s.id),
            name=s.name,
            scope_type=s.scope_type,
            group_id=str(s.group_id) if s.group_id else None,
            project_id=str(s.project_id) if s.project_id else None,
            status=s.status,
        )
        for s in spaces
    ]


# ==================== 10.2 文档接口 ====================


@router.post(
    "/projects/{project_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def api_upload_document(
    project_id: str,
    file: UploadFile = File(...),
    document_type: str = Form(default="technical_report"),
    knowledge_space_id: str | None = Form(default=None),
    auto_index: bool = Form(default=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    content = await file.read()
    doc = ingest_project_document(
        db,
        user_id=current_user.id,
        project_id=project_id,
        filename=file.filename or "upload.txt",
        content=content,
        document_type=document_type,
        knowledge_space_id=knowledge_space_id,
        auto_index=auto_index,
    )
    return _to_document_response(doc)


@router.get("/projects/{project_id}/documents", response_model=list[DocumentResponse])
def api_list_documents(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[DocumentResponse]:
    docs = list_project_documents(db, user_id=current_user.id, project_id=project_id)
    return [_to_document_response(d) for d in docs]


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def api_get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    doc = get_authorized_document(db, user_id=current_user.id, document_id=document_id)
    return _to_document_response(doc)


@router.post("/documents/{document_id}/reindex", response_model=DocumentResponse)
def api_reindex_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    index_document_version(db, user_id=current_user.id, document_id=document_id)
    doc = get_authorized_document(db, user_id=current_user.id, document_id=document_id)
    return _to_document_response(doc)


@router.delete("/documents/{document_id}", response_model=DocumentResponse)
def api_delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    doc = delete_document_completely(db, user_id=current_user.id, document_id=document_id)
    return _to_document_response(doc)


# ==================== 10.3 检索接口 ====================


@router.post("/projects/{project_id}/search", response_model=SearchResponse)
def api_search_project_knowledge(
    project_id: str,
    payload: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SearchResponse:
    # 规格 2.2 第 5 条 & 10.3：严禁客户端通过传入未经验证的 authorized_space_ids 扩大检索范围
    authz = AuthorizationService(db)
    server_spaces = set(authz.resolve_search_scopes(current_user.id, project_id))
    client_spaces = payload.authorized_space_ids or payload.authorized_spaces
    if client_spaces:
        for cs in client_spaces:
            if str(cs) not in server_spaces:
                raise AuthorizationError(
                    code="unauthorized_space_injection",
                    message="拒绝客户端注入未授权的知识空间 ID",
                )

    ks = KnowledgeService(db)
    doc_types = payload.filters.document_type if payload.filters else None
    doc_ids = payload.filters.document_ids if payload.filters else None

    pack = ks.search_for_project(
        user_id=current_user.id,
        project_id=project_id,
        query=payload.query,
        top_k=payload.top_k,
        mode=payload.mode,
        document_types=doc_types,
        document_ids=doc_ids,
    )
    hits = [
        SearchHitResponse(
            evidence_id=it.evidence_id,
            chunk_id=it.chunk_id,
            document_id=it.document_id,
            document_name=it.document_name,
            page_start=it.page_start,
            page_end=it.page_end,
            section_path=it.section_path,
            quote=it.quote,
            evidence_type=it.evidence_type,
            support_status=it.support_status,
            retrieval_score=it.retrieval_score,
            reranker_score=it.reranker_score,
            retrieval_method=it.retrieval_method,
            internal_ref_id=f"{it.document_id}#p{it.page_start or 1}:{it.chunk_id}",
        )
        for it in pack.items
    ]
    return SearchResponse(
        query=pack.query,
        index_version=pack.index_version,
        latency_ms=pack.retrieval_latency_ms,
        insufficient_evidence=pack.insufficient_evidence,
        hits=hits,
    )


# ==================== 10.4 工作流与产物接口 ====================


@router.post(
    "/projects/{project_id}/patent-tasks",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_201_CREATED,
)
def api_create_patent_task(
    project_id: str,
    payload: PatentTaskCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TaskStatusResponse:
    task = create_and_start_patent_task(
        db,
        user_id=current_user.id,
        project_id=project_id,
        title=payload.title,
        user_prompt=payload.user_prompt,
        input_document_ids=payload.input_document_ids,
        require_gap_confirmation=payload.require_gap_confirmation,
    )
    return _to_task_response(task)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def api_get_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TaskStatusResponse:
    task = get_authorized_task(db, user_id=current_user.id, task_id=task_id, action="view")
    return _to_task_response(task)


@router.post("/tasks/{task_id}/resume", response_model=TaskStatusResponse)
def api_resume_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TaskStatusResponse:
    task = resume_recoverable_task(db, user_id=current_user.id, task_id=task_id)
    return _to_task_response(task)


@router.post("/tasks/{task_id}/human-input", response_model=TaskStatusResponse)
def api_submit_human_input(
    task_id: str,
    payload: HumanInputRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TaskStatusResponse:
    task = submit_human_input_or_approval(
        db,
        user_id=current_user.id,
        task_id=task_id,
        action=payload.action,
        comment=payload.comment,
        supplement=payload.supplement,
    )
    return _to_task_response(task)


@router.get("/tasks/{task_id}/artifacts", response_model=list[ArtifactResponse])
def api_list_task_artifacts(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ArtifactResponse]:
    artifacts = list_task_artifacts(db, user_id=current_user.id, task_id=task_id)
    return [
        ArtifactResponse(
            id=str(a.id),
            task_id=str(a.task_id),
            project_id=str(a.project_id),
            artifact_type=a.artifact_type,
            version=a.version,
            sha256=a.sha256,
            metadata=a.metadata_json or {},
            created_at=a.created_at,
        )
        for a in artifacts
    ]


@router.get("/artifacts/{artifact_id}/download")
def api_download_artifact(
    artifact_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    artifact, file_path = get_authorized_artifact_for_download(
        db, user_id=current_user.id, artifact_id=artifact_id
    )
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=f"patent_disclosure_v{artifact.version}.pdf",
    )
