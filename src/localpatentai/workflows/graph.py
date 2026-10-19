"""LangGraph 图构建与 Checkpointer 管理（严格遵循规格 7.3 图结构建议）。"""

from __future__ import annotations

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from localpatentai.core.config import get_settings
from localpatentai.db.session import to_psycopg_url
from localpatentai.workflows.nodes import (
    analyze_technical_solution,
    assess_evidence_gaps,
    draft_sections,
    human_approval,
    parse_or_load_documents,
    render_pdf,
    request_human_input,
    retrieve_evidence,
    review_draft,
    save_artifact_and_audit,
    targeted_revision,
    validate_draft_schema,
    validate_pdf,
    validate_task_and_authorize,
)
from localpatentai.workflows.state import PatentWorkflowState

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_checkpointer: BaseCheckpointSaver | None = None
_compiled_graph = None
_schema_initialized: bool = False


def init_workflow_checkpointer(database_url: str | None = None) -> BaseCheckpointSaver:
    global _pool, _checkpointer, _schema_initialized
    if _checkpointer is not None:
        return _checkpointer
    raw_url = database_url or get_settings().database_url
    conninfo = to_psycopg_url(raw_url)
    try:
        _pool = ConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=6,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=True,
        )
        saver = PostgresSaver(_pool)
        if not _schema_initialized:
            with _pool.connection() as conn:
                has_table = conn.execute(
                    "SELECT to_regclass('public.checkpoints') AS reg"
                ).fetchone()
                if not has_table or not has_table.get("reg"):
                    saver.setup()
            _schema_initialized = True
        _checkpointer = saver
        logger.info("workflow_postgres_checkpointer_ready")
        return _checkpointer
    except Exception as exc:
        logger.warning(
            "postgres_checkpointer_unavailable_fallback_memory error=%s",
            type(exc).__name__,
        )
        _checkpointer = MemorySaver()
        return _checkpointer


def close_workflow_checkpointer() -> None:
    global _pool, _checkpointer, _compiled_graph
    if _pool is not None:
        _pool.close()
    _pool = None
    _checkpointer = None
    _compiled_graph = None


def get_checkpointer_health() -> dict:
    if _checkpointer is None:
        return {"ready": False, "backend": "uninitialized"}
    if isinstance(_checkpointer, MemorySaver):
        return {"ready": True, "backend": "memory"}
    return {"ready": True, "backend": "postgres"}


def _route_after_gap_assessment(state: PatentWorkflowState) -> str:
    # 若显式要求在缺失关键证据时暂停等待人工确认且尚未确认，则路由到 request_human_input
    if state.get("require_gap_confirmation") and not state.get("gap_confirmed"):
        gaps = state.get("evidence_gaps") or []
        if any(g.get("severity") == "blocking" for g in gaps):
            return "request_human_input"
    return "draft_sections"


def _route_after_review(state: PatentWorkflowState) -> str:
    settings = get_settings()
    review = state.get("review_result") or {}
    passed = bool(review.get("passed", True))
    rev_count = int(state.get("revision_count") or 0)

    if not passed and rev_count < settings.max_revision_rounds:
        return "targeted_revision"
    return "human_approval"


def _route_after_human_approval(state: PatentWorkflowState) -> str:
    approval = state.get("human_approval")
    if approval == "approved":
        return "render_pdf"
    if approval == "pending" and state.get("human_feedback"):
        settings = get_settings()
        if int(state.get("revision_count") or 0) < settings.max_revision_rounds:
            return "targeted_revision"
    return END


def build_patent_workflow_graph(checkpointer: BaseCheckpointSaver | None = None):
    saver = checkpointer or init_workflow_checkpointer()
    builder = StateGraph(PatentWorkflowState)

    builder.add_node("validate_task_and_authorize", validate_task_and_authorize)
    builder.add_node("parse_or_load_documents", parse_or_load_documents)
    builder.add_node("retrieve_evidence", retrieve_evidence)
    builder.add_node("analyze_technical_solution", analyze_technical_solution)
    builder.add_node("assess_evidence_gaps", assess_evidence_gaps)
    builder.add_node("request_human_input", request_human_input)
    builder.add_node("draft_sections", draft_sections)
    builder.add_node("validate_draft_schema", validate_draft_schema)
    builder.add_node("review_draft", review_draft)
    builder.add_node("targeted_revision", targeted_revision)
    builder.add_node("human_approval", human_approval)
    builder.add_node("render_pdf", render_pdf)
    builder.add_node("validate_pdf", validate_pdf)
    builder.add_node("save_artifact_and_audit", save_artifact_and_audit)

    builder.add_edge(START, "validate_task_and_authorize")
    builder.add_edge("validate_task_and_authorize", "parse_or_load_documents")
    builder.add_edge("parse_or_load_documents", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "analyze_technical_solution")
    builder.add_edge("analyze_technical_solution", "assess_evidence_gaps")
    builder.add_conditional_edges(
        "assess_evidence_gaps",
        _route_after_gap_assessment,
        {
            "request_human_input": "request_human_input",
            "draft_sections": "draft_sections",
        },
    )
    builder.add_edge("request_human_input", "retrieve_evidence")
    builder.add_edge("draft_sections", "validate_draft_schema")
    builder.add_edge("validate_draft_schema", "review_draft")
    builder.add_conditional_edges(
        "review_draft",
        _route_after_review,
        {
            "targeted_revision": "targeted_revision",
            "human_approval": "human_approval",
        },
    )
    builder.add_edge("targeted_revision", "validate_draft_schema")
    builder.add_conditional_edges(
        "human_approval",
        _route_after_human_approval,
        {
            "render_pdf": "render_pdf",
            "targeted_revision": "targeted_revision",
            END: END,
        },
    )
    builder.add_edge("render_pdf", "validate_pdf")
    builder.add_edge("validate_pdf", "save_artifact_and_audit")
    builder.add_edge("save_artifact_and_audit", END)

    return builder.compile(checkpointer=saver)


def get_patent_workflow_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_patent_workflow_graph()
    return _compiled_graph
