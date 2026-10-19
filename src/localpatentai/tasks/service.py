"""专利任务生命周期服务：创建任务、执行/中断/恢复 LangGraph、人工审核提交、产物读取与安全下载（规格 7.3 & 10.4）。"""

from __future__ import annotations

import secrets
import uuid
from pathlib import Path
from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.orm import Session

from localpatentai.audit import record_audit_event
from localpatentai.auth.service import AuthorizationService
from localpatentai.core.config import get_settings
from localpatentai.core.errors import AppError, ResourceNotFoundError
from localpatentai.core.security import resolve_safe_storage_path
from localpatentai.db.models import Artifact, Task
from localpatentai.workflows.graph import get_patent_workflow_graph
from localpatentai.workflows.nodes import bind_workflow_db, unbind_workflow_db
from localpatentai.workflows.prompts import WORKFLOW_VERSION


def _as_uuid(val: str | uuid.UUID) -> uuid.UUID:
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError() from exc


def create_and_start_patent_task(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    project_id: str | uuid.UUID,
    title: str,
    user_prompt: str = "",
    input_document_ids: list[str] | None = None,
    require_gap_confirmation: bool = False,
) -> Task:
    authz = AuthorizationService(db)
    project = authz.assert_project_access(user_id, project_id, "run_workflow")
    uid = _as_uuid(user_id)

    cleaned_title = title.strip() or f"{project.name}-专利技术交底书"
    task = Task(
        id=uuid.uuid4(),
        project_id=project.id,
        created_by=uid,
        thread_id=secrets.token_hex(16),
        title=cleaned_title,
        user_prompt=user_prompt.strip(),
        status="running",
        current_stage="init",
        input_document_ids_json=list(input_document_ids or []),
        workflow_version=WORKFLOW_VERSION,
        state_snapshot_json={},
        llm_calls_used=0,
        revision_count=0,
        human_approval_status="pending",
    )
    db.add(task)
    db.flush()

    record_audit_event(
        db,
        actor_user_id=uid,
        group_id=project.group_id,
        project_id=project.id,
        action="task.create",
        resource_type="task",
        resource_id=task.id,
        result="success",
        metadata={"workflow_version": WORKFLOW_VERSION},
        commit=True,
    )

    initial_state: dict[str, Any] = {
        "task_id": str(task.id),
        "project_id": str(project.id),
        "user_id": str(uid),
        "workflow_version": WORKFLOW_VERSION,
        "current_stage": "init",
        "input_document_ids": list(input_document_ids or []),
        "authorized_space_ids": [],  # 由 validate_task_and_authorize 节点从后端生成
        "task_title": cleaned_title,
        "user_prompt": user_prompt.strip(),
        "revision_count": 0,
        "llm_calls_used": 0,
        "human_approval": "pending",
        "require_gap_confirmation": require_gap_confirmation,
        "gap_confirmed": False,
        "trace_events": [],
    }

    _execute_graph_step(db, task, initial_state)
    return task


def get_authorized_task(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    task_id: str | uuid.UUID,
    action: str = "view",
) -> Task:
    """
    每次访问 task_id 都必须重新校验当前用户对关联项目的访问权限（规格 10.4）。
    """
    tid = _as_uuid(task_id)
    task = db.get(Task, tid)
    if task is None:
        raise ResourceNotFoundError()
    AuthorizationService(db).assert_project_access(user_id, task.project_id, action)
    return task


def submit_human_input_or_approval(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    task_id: str | uuid.UUID,
    action: str = "approve",
    comment: str = "",
    supplement: str = "",
) -> Task:
    """
    提交人工确认（技术特征/缺失证据确认、草案审核通过/退回修订/拒绝、PDF 导出确认）。
    恢复前强制校验当前用户是否仍具备项目权限。
    """
    auth_action = "approve_export" if action in {"approve", "approved"} else "resume_workflow"
    task = get_authorized_task(db, user_id=user_id, task_id=task_id, action=auth_action)

    graph = get_patent_workflow_graph()
    config = {"configurable": {"thread_id": task.thread_id}}
    snapshot = graph.get_state(config)
    next_nodes = set(snapshot.next or ())

    if not next_nodes:
        raise AppError(409, "task_not_awaiting_human", "当前任务未处于等待人工确认的暂停点")

    resume_payload = {
        "action": action,
        "decision": action,
        "comment": comment,
        "supplement": supplement or comment,
        "actor_user_id": str(user_id),
    }
    _execute_graph_step(db, task, Command(resume=resume_payload))
    return task


def resume_recoverable_task(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    task_id: str | uuid.UUID,
) -> Task:
    """
    恢复因可恢复错误中断或暂停的长任务（规格 7.2 & 10.4）。
    恢复前强制通过 AuthorizationService 重新校验用户对关联项目的权限及最新可访问空间。
    """
    task = get_authorized_task(db, user_id=user_id, task_id=task_id, action="resume_workflow")
    authz = AuthorizationService(db)
    fresh_spaces = authz.resolve_search_scopes(user_id, task.project_id)

    graph = get_patent_workflow_graph()
    config = {"configurable": {"thread_id": task.thread_id}}
    snapshot = graph.get_state(config)
    next_nodes = set(snapshot.next or ())

    if "human_approval" in next_nodes or "request_human_input" in next_nodes:
        # 若停在人工节点，保持 awaiting_human 状态并更新最新授权空间
        graph.update_state(config, {"authorized_space_ids": fresh_spaces})
        _sync_task_from_snapshot(db, task, graph.get_state(config))
        return task

    if task.status == "failed":
        # 从保存的状态快照重新发起可恢复阶段
        saved = dict(task.state_snapshot_json or {})
        saved["user_id"] = str(user_id)
        saved["authorized_space_ids"] = fresh_spaces
        task.error_code = None
        task.status = "running"
        db.flush()
        _execute_graph_step(db, task, saved)
        return task

    _execute_graph_step(db, task, None)
    return task


def list_task_artifacts(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    task_id: str | uuid.UUID,
) -> list[Artifact]:
    task = get_authorized_task(db, user_id=user_id, task_id=task_id, action="view")
    return list(
        db.scalars(
            select(Artifact)
            .where(Artifact.task_id == task.id)
            .order_by(Artifact.created_at.desc())
        ).all()
    )


def get_authorized_artifact_for_download(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    artifact_id: str | uuid.UUID,
) -> tuple[Artifact, Path]:
    """
    下载 PDF 产物前重新校验当前用户对关联项目的访问权限（规格 10.4）。
    """
    aid = _as_uuid(artifact_id)
    artifact = db.get(Artifact, aid)
    if artifact is None:
        raise ResourceNotFoundError()

    project = AuthorizationService(db).assert_project_access(
        user_id, artifact.project_id, "download_artifact"
    )
    settings = get_settings()
    artifact_root = settings.resolve_path(settings.artifact_dir)
    file_path = resolve_safe_storage_path(artifact_root, artifact.storage_key)
    if not file_path.is_file():
        raise AppError(404, "artifact_file_missing", "PDF 产物文件不存在")

    record_audit_event(
        db,
        actor_user_id=user_id,
        group_id=project.group_id,
        project_id=project.id,
        action="artifact.download",
        resource_type="artifact",
        resource_id=artifact.id,
        result="success",
        metadata={"sha256": artifact.sha256},
        commit=True,
    )
    return artifact, file_path


def _execute_graph_step(db: Session, task: Task, input_or_command: Any) -> None:
    graph = get_patent_workflow_graph()
    config = {"configurable": {"thread_id": task.thread_id}}
    bind_workflow_db(db)
    try:
        graph.invoke(input_or_command, config)
    except GraphInterrupt:
        pass
    except AppError as exc:
        task.status = "failed"
        task.error_code = exc.code
        db.commit()
        raise
    except Exception as exc:
        task.status = "failed"
        task.error_code = type(exc).__name__
        db.commit()
        raise AppError(
            500,
            "workflow_execution_error",
            f"工作流执行失败: {type(exc).__name__}",
        ) from exc
    finally:
        unbind_workflow_db()

    snapshot = graph.get_state(config)
    _sync_task_from_snapshot(db, task, snapshot)


def _sync_task_from_snapshot(db: Session, task: Task, snapshot: Any) -> None:
    values: dict[str, Any] = dict(snapshot.values or {})
    next_nodes = set(snapshot.next or ())

    # 不在数据库的 state_snapshot_json 中保存大体积临时 base64 PDF
    clean_values = {k: v for k, v in values.items() if k != "pdf_bytes_b64"}
    task.state_snapshot_json = clean_values
    task.current_stage = str(values.get("current_stage") or task.current_stage)
    task.revision_count = int(values.get("revision_count") or 0)
    task.llm_calls_used = int(values.get("llm_calls_used") or 0)
    task.human_approval_status = str(values.get("human_approval") or "pending")

    if "human_approval" in next_nodes:
        task.status = "awaiting_approval"
        task.current_stage = "human_approval"
    elif "request_human_input" in next_nodes:
        task.status = "awaiting_human_input"
        task.current_stage = "request_human_input"
    elif task.human_approval_status == "rejected":
        task.status = "rejected"
    elif values.get("artifact_id"):
        task.status = "completed"
        task.current_stage = "completed"
    elif not next_nodes:
        task.status = "completed"

    db.commit()
    db.refresh(task)
