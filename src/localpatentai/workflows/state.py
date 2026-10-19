"""LangGraph 状态 Schema 定义（严格遵循规格 7.2 节 PatentWorkflowState）。"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class PatentWorkflowState(TypedDict):
    task_id: str
    project_id: str
    user_id: str
    workflow_version: str
    current_stage: str
    input_document_ids: list[str]
    authorized_space_ids: list[str]
    task_title: NotRequired[str]
    user_prompt: NotRequired[str]
    technical_analysis: NotRequired[dict]
    evidence_ids: NotRequired[list[str]]
    evidence_items: NotRequired[list[dict]]
    evidence_research: NotRequired[dict]
    evidence_gaps: NotRequired[list[dict]]
    draft: NotRequired[dict]
    review_result: NotRequired[dict]
    review_issues: NotRequired[list[dict]]
    revision_count: int
    llm_calls_used: NotRequired[int]
    human_approval: NotRequired[Literal["pending", "approved", "rejected"]]
    human_feedback: NotRequired[str]
    require_gap_confirmation: NotRequired[bool]
    gap_confirmed: NotRequired[bool]
    pdf_bytes_b64: NotRequired[str]
    pdf_sha256: NotRequired[str]
    pdf_validation: NotRequired[dict]
    artifact_id: NotRequired[str]
    trace_events: NotRequired[list[dict]]
    error_code: NotRequired[str]
