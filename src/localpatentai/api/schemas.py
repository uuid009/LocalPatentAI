"""API 请求与响应 Pydantic v2 Schemas（规格第 10 章）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    group_code: Literal["algorithm", "framework", "cloud"]
    display_name: str | None = None
    group_role: Literal["member", "group_admin"] = "member"


class LoginRequest(BaseModel):
    username: str
    password: str


class DemoLoginRequest(BaseModel):
    group_code: Literal["algorithm", "framework", "cloud"]


class AuthResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    group_id: str
    group_code: str
    group_name: str
    token: str


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    group_code: Literal["algorithm", "framework", "cloud"] | None = None


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    status: Literal["active", "archived", "closed"] | None = None


class ProjectMemberAddRequest(BaseModel):
    user_id: str
    role: Literal["project_owner", "project_editor", "project_viewer", "reviewer"] = (
        "project_viewer"
    )


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str
    group_id: str
    status: str
    private_space_id: str | None = None
    created_at: datetime
    updated_at: datetime


class KnowledgeSpaceResponse(BaseModel):
    id: str
    name: str
    scope_type: str
    group_id: str | None = None
    project_id: str | None = None
    status: str


class DocumentResponse(BaseModel):
    id: str
    knowledge_space_id: str
    project_id: str | None = None
    uploaded_by: str
    original_filename: str
    document_type: str
    mime_type: str
    file_size: int
    sha256: str
    status: str
    parser_version: str
    error_code: str | None = None
    created_at: datetime
    updated_at: datetime


class SearchFilters(BaseModel):
    document_type: list[str] | None = None
    document_ids: list[str] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=8, ge=1, le=30)
    mode: Literal["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rrf_rerank"] = (
        "hybrid_rrf_rerank"
    )
    filters: SearchFilters | None = None
    # 若恶意客户端试图注入 authorized_space_ids，API 将明确拒绝或忽略
    authorized_space_ids: list[str] | None = None
    authorized_spaces: list[str] | None = None


class SearchHitResponse(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    quote: str
    evidence_type: str
    support_status: str
    retrieval_score: float | None = None
    reranker_score: float | None = None
    retrieval_method: list[str] = Field(default_factory=list)
    internal_ref_id: str


class SearchResponse(BaseModel):
    query: str
    index_version: str
    latency_ms: float
    insufficient_evidence: bool
    hits: list[SearchHitResponse] = Field(default_factory=list)


class PatentTaskCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    user_prompt: str = ""
    input_document_ids: list[str] = Field(default_factory=list)
    require_gap_confirmation: bool = False


class HumanInputRequest(BaseModel):
    action: Literal["approve", "approved", "revise", "rejected", "supplement"] = "approve"
    comment: str = ""
    supplement: str = ""


class TaskStatusResponse(BaseModel):
    id: str
    project_id: str
    created_by: str
    title: str
    status: str
    current_stage: str
    workflow_version: str
    revision_count: int
    llm_calls_used: int
    human_approval_status: str
    error_code: str | None = None
    technical_analysis: dict[str, Any] | None = None
    evidence_research: dict[str, Any] | None = None
    evidence_gaps: list[dict[str, Any]] = Field(default_factory=list)
    draft: dict[str, Any] | None = None
    review_result: dict[str, Any] | None = None
    artifact_id: str | None = None
    created_at: datetime
    updated_at: datetime


class ArtifactResponse(BaseModel):
    id: str
    task_id: str
    project_id: str
    artifact_type: str
    version: int
    sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
