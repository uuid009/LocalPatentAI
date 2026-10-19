"""LangGraph 各节点实现（规格 7.1 & 7.3 节）。"""

from __future__ import annotations

import base64
import contextvars
import time
import uuid

from langgraph.types import interrupt
from sqlalchemy import select
from sqlalchemy.orm import Session

from localpatentai.audit import record_audit_event
from localpatentai.auth.service import AuthorizationService
from localpatentai.core.config import get_settings
from localpatentai.core.errors import AppError
from localpatentai.core.security import resolve_safe_storage_path
from localpatentai.db.models import Artifact, Document
from localpatentai.drafting.schemas import (
    PatentDisclosureDraft,
    QualityReviewResult,
    TechnicalSolutionAnalysis,
)
from localpatentai.drafting.service import (
    apply_targeted_revision,
    build_technical_analysis,
    draft_patent_disclosure,
    evaluate_evidence_research,
)
from localpatentai.harness.budget import BudgetController
from localpatentai.harness.tracing import trace_node_execution
from localpatentai.knowledge.models import EvidenceItem
from localpatentai.knowledge.service import KnowledgeService
from localpatentai.pdf.renderer import render_pdf_bytes_from_draft
from localpatentai.pdf.validator import validate_generated_pdf
from localpatentai.workflows.prompts import PROMPT_VERSION, WORKFLOW_VERSION
from localpatentai.workflows.state import PatentWorkflowState
from localpatentai.workflows.validators import (
    assert_ready_for_pdf_generation,
    validate_draft_output,
)

_workflow_db_ctx: contextvars.ContextVar[Session | None] = contextvars.ContextVar(
    "workflow_db_ctx", default=None
)


def bind_workflow_db(db: Session) -> None:
    _workflow_db_ctx.set(db)


def unbind_workflow_db() -> None:
    _workflow_db_ctx.set(None)


def require_workflow_db() -> Session:
    db = _workflow_db_ctx.get()
    if db is None:
        raise RuntimeError("工作流节点缺少绑定的数据库会话")
    return db


def _append_trace(
    state: PatentWorkflowState,
    node_name: str,
    start_perf: float,
    *,
    schema_valid: bool = True,
    error_code: str | None = None,
    evidence_ids: list[str] | None = None,
) -> list[dict]:
    settings = get_settings()
    existing = list(state.get("trace_events") or [])
    existing.append(
        trace_node_execution(
            task_id=state["task_id"],
            node_name=node_name,
            start_perf=start_perf,
            model_version="mock" if settings.llm_mock else settings.llm_model,
            prompt_version=PROMPT_VERSION,
            retrieval_version=settings.index_version,
            schema_valid=schema_valid,
            error_code=error_code,
            evidence_ids=evidence_ids,
        )
    )
    return existing


def validate_task_and_authorize(state: PatentWorkflowState) -> dict:
    """
    节点 1：验证用户对项目的执行权限，并由后端 AuthorizationService 生成 authorized_space_ids，
    绝不信任外部直接传入的 authorized_space_ids。
    """
    t0 = time.perf_counter()
    db = require_workflow_db()
    authz = AuthorizationService(db)
    authz.assert_project_access(state["user_id"], state["project_id"], "run_workflow")
    resolved_spaces = authz.resolve_search_scopes(state["user_id"], state["project_id"])
    return {
        "current_stage": "validate_task_and_authorize",
        "authorized_space_ids": resolved_spaces,
        "trace_events": _append_trace(state, "validate_task_and_authorize", t0),
    }


def parse_or_load_documents(state: PatentWorkflowState) -> dict:
    """节点 2：校验或加载项目内已索引的授权文档列表。"""
    t0 = time.perf_counter()
    db = require_workflow_db()
    authz = AuthorizationService(db)
    project = authz.assert_project_access(state["user_id"], state["project_id"], "view")

    input_doc_ids = list(state.get("input_document_ids") or [])
    if input_doc_ids:
        verified_ids: list[str] = []
        for did in input_doc_ids:
            doc = db.get(Document, uuid.UUID(str(did)))
            if (
                doc is not None
                and doc.status == "indexed"
                and str(doc.knowledge_space_id) in set(state["authorized_space_ids"])
            ):
                verified_ids.append(str(doc.id))
        doc_ids = verified_ids
    else:
        docs = db.scalars(
            select(Document).where(
                Document.project_id == project.id,
                Document.status == "indexed",
            )
        ).all()
        doc_ids = [str(d.id) for d in docs]

    return {
        "current_stage": "parse_or_load_documents",
        "input_document_ids": doc_ids,
        "trace_events": _append_trace(state, "parse_or_load_documents", t0),
    }


def retrieve_evidence(state: PatentWorkflowState) -> dict:
    """节点 3 (Node B 检索阶段)：通过 KnowledgeService 在授权空间内执行混合检索 + RRF + Reranker。"""
    t0 = time.perf_counter()
    db = require_workflow_db()
    settings = get_settings()
    ks = KnowledgeService(db, settings)

    query = f"{state.get('task_title') or ''} {state.get('user_prompt') or ''}".strip()
    if not query:
        query = "专利核心技术方案与实验效果"

    doc_filter = state.get("input_document_ids") or None
    pack = ks.search_for_project(
        user_id=state["user_id"],
        project_id=state["project_id"],
        query=query,
        top_k=settings.reranker_top_n,
        document_ids=doc_filter if doc_filter else None,
    )
    # 若仅限指定文档未召回，则放宽到全部已授权知识空间
    if not pack.items and doc_filter:
        pack = ks.search_for_project(
            user_id=state["user_id"],
            project_id=state["project_id"],
            query=query,
            top_k=settings.reranker_top_n,
            document_ids=None,
        )

    ev_ids = [it.evidence_id for it in pack.items]
    ev_dicts = [it.model_dump() for it in pack.items]
    return {
        "current_stage": "retrieve_evidence",
        "authorized_space_ids": pack.authorized_space_ids,
        "evidence_ids": ev_ids,
        "evidence_items": ev_dicts,
        "trace_events": _append_trace(
            state, "retrieve_evidence", t0, evidence_ids=ev_ids
        ),
    }


def analyze_technical_solution(state: PatentWorkflowState) -> dict:
    """节点 4 (Node A)：技术方案解析，提取技术领域、核心技术特征、实施步骤、实验数据与技术效果。"""
    t0 = time.perf_counter()
    settings = get_settings()
    budget = BudgetController(settings)
    llm_used = budget.consume_llm_call(int(state.get("llm_calls_used") or 0), 1)

    ev_items = [EvidenceItem.model_validate(d) for d in (state.get("evidence_items") or [])]
    analysis = build_technical_analysis(
        task_title=state.get("task_title") or "专利技术方案",
        user_prompt=state.get("user_prompt") or "",
        evidence_items=ev_items,
        settings=settings,
    )
    return {
        "current_stage": "analyze_technical_solution",
        "technical_analysis": analysis.model_dump(),
        "llm_calls_used": llm_used,
        "trace_events": _append_trace(
            state,
            "analyze_technical_solution",
            t0,
            evidence_ids=state.get("evidence_ids") or [],
        ),
    }


def assess_evidence_gaps(state: PatentWorkflowState) -> dict:
    """节点 5 (Node B 评估阶段)：评估证据充分性、缺失数据清单与相似现有技术线索。"""
    t0 = time.perf_counter()
    analysis = TechnicalSolutionAnalysis.model_validate(state.get("technical_analysis") or {})
    ev_items = [EvidenceItem.model_validate(d) for d in (state.get("evidence_items") or [])]
    report = evaluate_evidence_research(analysis, ev_items)
    gaps_dump = [g.model_dump() for g in report.missing_data_checklist]
    return {
        "current_stage": "assess_evidence_gaps",
        "evidence_research": report.model_dump(),
        "evidence_gaps": gaps_dump,
        "trace_events": _append_trace(
            state,
            "assess_evidence_gaps",
            t0,
            evidence_ids=state.get("evidence_ids") or [],
        ),
    }


def request_human_input(state: PatentWorkflowState) -> dict:
    """人工补充或确认节点（规格 7.1 Node E & 7.3）：当关键证据缺失或审查需要人工补充事实时暂停。"""
    t0 = time.perf_counter()
    payload = interrupt(
        {
            "checkpoint_type": "human_input_required",
            "task_id": state["task_id"],
            "current_stage": "request_human_input",
            "evidence_gaps": state.get("evidence_gaps") or [],
            "technical_analysis": state.get("technical_analysis") or {},
        }
    )
    supplement = ""
    if isinstance(payload, dict):
        supplement = str(
            payload.get("supplement") or payload.get("comment") or payload.get("notes") or ""
        ).strip()
    elif payload is not None:
        supplement = str(payload).strip()

    db = require_workflow_db()
    # 恢复时重新校验权限（规格 2.2 第 10 条 & 7.2）
    AuthorizationService(db).assert_project_access(
        state["user_id"], state["project_id"], "resume_workflow"
    )

    updated_prompt = state.get("user_prompt") or ""
    if supplement:
        updated_prompt = f"{updated_prompt}\n【人工确认补充说明】：{supplement}".strip()

    return {
        "current_stage": "request_human_input",
        "user_prompt": updated_prompt,
        "human_feedback": supplement,
        "gap_confirmed": True,
        "trace_events": _append_trace(state, "request_human_input", t0),
    }


def draft_sections(state: PatentWorkflowState) -> dict:
    """节点 6 (Node C)：专利草拟节点，生成完整章节与权利要求书草案，无依据内容严格标记待确认。"""
    t0 = time.perf_counter()
    settings = get_settings()
    budget = BudgetController(settings)
    llm_used = budget.consume_llm_call(int(state.get("llm_calls_used") or 0), 1)

    analysis = TechnicalSolutionAnalysis.model_validate(state.get("technical_analysis") or {})
    ev_items = [EvidenceItem.model_validate(d) for d in (state.get("evidence_items") or [])]

    draft = draft_patent_disclosure(
        task_title=state.get("task_title") or "专利技术方案",
        user_prompt=state.get("user_prompt") or "",
        analysis=analysis,
        evidence_items=ev_items,
        version=1,
        settings=settings,
    )
    return {
        "current_stage": "draft_sections",
        "draft": draft.model_dump(),
        "llm_calls_used": llm_used,
        "trace_events": _append_trace(
            state,
            "draft_sections",
            t0,
            evidence_ids=state.get("evidence_ids") or [],
        ),
    }


def validate_draft_schema(state: PatentWorkflowState) -> dict:
    """节点 7：Output Validator 校验草案 Pydantic Schema 与证据归属。"""
    t0 = time.perf_counter()
    ev_items = [EvidenceItem.model_validate(d) for d in (state.get("evidence_items") or [])]
    draft, _ = validate_draft_output(
        state.get("draft") or {},
        ev_items,
        state.get("authorized_space_ids") or [],
    )
    return {
        "current_stage": "validate_draft_schema",
        "draft": draft.model_dump(),
        "trace_events": _append_trace(state, "validate_draft_schema", t0),
    }


def review_draft(state: PatentWorkflowState) -> dict:
    """节点 8 (Node D)：自动质量审查（确定性规则 + 结构检查）。"""
    t0 = time.perf_counter()
    ev_items = [EvidenceItem.model_validate(d) for d in (state.get("evidence_items") or [])]
    _, review = validate_draft_output(
        state.get("draft") or {},
        ev_items,
        state.get("authorized_space_ids") or [],
    )
    return {
        "current_stage": "review_draft",
        "review_result": review.model_dump(),
        "review_issues": [iss.model_dump() for iss in review.issues],
        "trace_events": _append_trace(
            state,
            "review_draft",
            t0,
            schema_valid=review.passed,
        ),
    }


def targeted_revision(state: PatentWorkflowState) -> dict:
    """节点 9：定向修订，修复审查发现的阻断问题或应用人工修订意见，受 max_revision_rounds 限制。"""
    t0 = time.perf_counter()
    settings = get_settings()
    budget = BudgetController(settings)
    llm_used = budget.consume_llm_call(int(state.get("llm_calls_used") or 0), 1)

    draft = PatentDisclosureDraft.model_validate(state.get("draft") or {})
    review = QualityReviewResult.model_validate(state.get("review_result") or {"passed": False})
    ev_items = [EvidenceItem.model_validate(d) for d in (state.get("evidence_items") or [])]

    revised = apply_targeted_revision(
        draft,
        review,
        ev_items,
        human_feedback=state.get("human_feedback") or "",
    )
    next_rev = int(state.get("revision_count") or 0) + 1
    return {
        "current_stage": "targeted_revision",
        "draft": revised.model_dump(),
        "revision_count": next_rev,
        "llm_calls_used": llm_used,
        "human_approval": "pending",
        "trace_events": _append_trace(state, "targeted_revision", t0),
    }


def human_approval(state: PatentWorkflowState) -> dict:
    """
    节点 10 (Node E)：人工审核确认点（草案审核与最终 PDF 导出确认）。
    恢复时重新校验用户对项目的访问权限。
    """
    t0 = time.perf_counter()
    decision_payload = interrupt(
        {
            "checkpoint_type": "human_approval",
            "task_id": state["task_id"],
            "current_stage": "human_approval",
            "draft_version": (state.get("draft") or {}).get("version", 1),
            "review_passed": (state.get("review_result") or {}).get("passed", True),
            "blocking_issues": (state.get("review_result") or {}).get("blocking_count", 0),
        }
    )
    db = require_workflow_db()
    authz = AuthorizationService(db)
    # 恢复时重新校验权限（规格 2.2 第 10 条 & 12.1 第 11 条）
    authz.assert_project_access(state["user_id"], state["project_id"], "review_draft")

    action = "approved"
    comment = ""
    if isinstance(decision_payload, dict):
        raw_action = str(decision_payload.get("action") or decision_payload.get("decision") or "approved").lower()
        if raw_action in {"approve", "approved"}:
            action = "approved"
        elif raw_action in {"revise", "revision"}:
            action = "revise"
        else:
            action = "rejected"
        comment = str(decision_payload.get("comment") or decision_payload.get("feedback") or "").strip()

    record_audit_event(
        db,
        actor_user_id=state["user_id"],
        project_id=state["project_id"],
        action=f"workflow.human_approval.{action}",
        resource_type="task",
        resource_id=state["task_id"],
        result="success",
        metadata={"draft_version": (state.get("draft") or {}).get("version", 1)},
    )

    if action == "revise":
        return {
            "current_stage": "human_approval",
            "human_approval": "pending",
            "human_feedback": comment,
            "trace_events": _append_trace(state, "human_approval", t0),
        }

    return {
        "current_stage": "human_approval",
        "human_approval": "approved" if action == "approved" else "rejected",
        "human_feedback": comment,
        "trace_events": _append_trace(state, "human_approval", t0),
    }


def render_pdf(state: PatentWorkflowState) -> dict:
    """节点 11 (Node F 渲染阶段)：仅在 Schema 校验通过且人工批准后渲染 PDF。"""
    t0 = time.perf_counter()
    draft = PatentDisclosureDraft.model_validate(state.get("draft") or {})
    review = QualityReviewResult.model_validate(state.get("review_result") or {"passed": True})
    ev_items = [EvidenceItem.model_validate(d) for d in (state.get("evidence_items") or [])]

    assert_ready_for_pdf_generation(
        draft,
        review,
        state.get("human_approval") or "pending",
    )

    pdf_bytes, sha256_hex = render_pdf_bytes_from_draft(
        draft,
        evidence_items=ev_items,
        review_result=review,
        workflow_version=state.get("workflow_version") or WORKFLOW_VERSION,
    )
    b64_data = base64.b64encode(pdf_bytes).decode("ascii")
    return {
        "current_stage": "render_pdf",
        "pdf_bytes_b64": b64_data,
        "pdf_sha256": sha256_hex,
        "trace_events": _append_trace(state, "render_pdf", t0),
    }


def validate_pdf(state: PatentWorkflowState) -> dict:
    """节点 12 (Node F 后检验阶段)：重新打开 PDF 并验证页数、章节、引用与占位符。"""
    t0 = time.perf_counter()
    b64_data = state.get("pdf_bytes_b64") or ""
    if not b64_data:
        raise AppError(500, "pdf_bytes_missing", "未找到待验证的 PDF 数据")
    pdf_bytes = base64.b64decode(b64_data.encode("ascii"))
    report = validate_generated_pdf(
        pdf_bytes,
        expected_sha256=state.get("pdf_sha256") or "",
        expected_evidence_ids=state.get("evidence_ids") or [],
    )
    return {
        "current_stage": "validate_pdf",
        "pdf_validation": report.model_dump(),
        "trace_events": _append_trace(state, "validate_pdf", t0),
    }


def save_artifact_and_audit(state: PatentWorkflowState) -> dict:
    """节点 13：将验证通过的 PDF 存储在项目受控目录，创建 Artifact 记录并写入审计日志。"""
    t0 = time.perf_counter()
    db = require_workflow_db()
    settings = get_settings()
    authz = AuthorizationService(db)
    project = authz.assert_project_access(
        state["user_id"], state["project_id"], "run_workflow"
    )

    b64_data = state.get("pdf_bytes_b64") or ""
    pdf_bytes = base64.b64decode(b64_data.encode("ascii"))
    sha256_hex = state.get("pdf_sha256") or ""
    draft_ver = int((state.get("draft") or {}).get("version") or 1)

    storage_key = f"{project.id.hex}/{state['task_id']}_v{draft_ver}.pdf"
    artifact_root = settings.resolve_path(settings.artifact_dir)
    target_path = resolve_safe_storage_path(artifact_root, storage_key)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(pdf_bytes)

    # 校验落盘后的文件可读性
    validate_generated_pdf(
        pdf_bytes,
        expected_sha256=sha256_hex,
        expected_evidence_ids=state.get("evidence_ids") or [],
        storage_root=artifact_root,
        storage_key=storage_key,
    )

    artifact = Artifact(
        task_id=uuid.UUID(str(state["task_id"])),
        project_id=project.id,
        artifact_type="patent_disclosure_pdf",
        storage_key=storage_key,
        version=draft_ver,
        sha256=sha256_hex,
        metadata_json={
            "page_count": (state.get("pdf_validation") or {}).get("page_count", 1),
            "workflow_version": state.get("workflow_version") or WORKFLOW_VERSION,
            "evidence_ids": state.get("evidence_ids") or [],
        },
    )
    db.add(artifact)
    db.flush()

    record_audit_event(
        db,
        actor_user_id=state["user_id"],
        group_id=project.group_id,
        project_id=project.id,
        action="artifact.pdf_generated",
        resource_type="artifact",
        resource_id=artifact.id,
        result="success",
        metadata={
            "task_id": state["task_id"],
            "sha256": sha256_hex,
            "version": draft_ver,
            "page_count": (state.get("pdf_validation") or {}).get("page_count", 1),
        },
    )

    return {
        "current_stage": "completed",
        "artifact_id": str(artifact.id),
        "pdf_bytes_b64": "",  # 落盘后清空状态中的大体积字节流，避免膨胀 checkpoint
        "trace_events": _append_trace(state, "save_artifact_and_audit", t0),
    }
