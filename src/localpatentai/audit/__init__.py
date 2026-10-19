"""审计服务模块。默认记录最小必要元数据，严禁记录文档全文、完整 Prompt 或敏感路径。"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from localpatentai.core.logging import sanitize_log_dict
from localpatentai.db.models import AuditEvent

logger = logging.getLogger(__name__)


def _to_uuid(val: str | uuid.UUID | None) -> uuid.UUID | None:
    if val is None:
        return None
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except ValueError:
        return None


def record_audit_event(
    db: Session,
    *,
    action: str,
    resource_type: str,
    resource_id: str | uuid.UUID,
    result: str,
    actor_user_id: str | uuid.UUID | None = None,
    group_id: str | uuid.UUID | None = None,
    project_id: str | uuid.UUID | None = None,
    metadata: dict[str, Any] | None = None,
    commit: bool = False,
) -> AuditEvent:
    safe_meta = sanitize_log_dict(metadata or {})
    event = AuditEvent(
        actor_user_id=_to_uuid(actor_user_id),
        group_id=_to_uuid(group_id),
        project_id=_to_uuid(project_id),
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id),
        result=result,
        metadata_json=safe_meta,
    )
    db.add(event)
    if commit:
        db.commit()
        db.refresh(event)
    else:
        db.flush()
    logger.info(
        "audit_event action=%s resource_type=%s resource_id=%s result=%s actor=%s",
        action,
        resource_type,
        resource_id,
        result,
        actor_user_id,
    )
    return event


def list_project_audit_events(
    db: Session, project_id: str | uuid.UUID, limit: int = 100
) -> list[AuditEvent]:
    pid = _to_uuid(project_id)
    if pid is None:
        return []
    return list(
        db.scalars(
            select(AuditEvent)
            .where(AuditEvent.project_id == pid)
            .order_by(AuditEvent.timestamp.desc())
            .limit(limit)
        )
    )
