"""专利项目、项目成员与知识空间管理服务（强制三组隔离与默认创建项目私有知识空间）。"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from localpatentai.audit import record_audit_event
from localpatentai.auth.models import PROJECT_ROLES
from localpatentai.auth.service import AuthorizationService
from localpatentai.core.errors import AppError, AuthorizationError, ResourceNotFoundError
from localpatentai.db.models import (
    Group,
    GroupMember,
    KnowledgeSpace,
    Project,
    ProjectMember,
    User,
)


def _as_uuid(val: str | uuid.UUID, err_cls: type[AppError] = ResourceNotFoundError) -> uuid.UUID:
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except (ValueError, TypeError) as exc:
        raise err_cls() from exc


def create_project(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    name: str,
    description: str = "",
    group_code: str | None = None,
) -> tuple[Project, KnowledgeSpace]:
    """
    创建专利项目：
    1. 确定用户所属小组；若指定 group_code，必须在用户已加入的组内，严禁跨组创建项目。
    2. 自动将创建人加入 project_members (role=project_owner)。
    3. 自动为该项目创建唯一的 project_private 私有知识空间。
    """
    authz = AuthorizationService(db)
    ctx = authz.get_user_context(user_id)
    if not ctx.groups:
        raise AuthorizationError(
            code="user_has_no_group",
            message="当前用户未加入任何科研小组，无法创建专利项目",
        )

    cleaned_name = name.strip()
    if not cleaned_name:
        raise AppError(422, "invalid_project_name", "项目名称不能为空")

    target_group_info = None
    if group_code:
        for g in ctx.groups:
            if g.group_code == group_code:
                target_group_info = g
                break
        if target_group_info is None:
            record_audit_event(
                db,
                actor_user_id=ctx.user_id,
                action="project.create",
                resource_type="project",
                resource_id="new",
                result="denied",
                metadata={"requested_group_code": group_code},
                commit=True,
            )
            raise AuthorizationError(
                code="cross_group_project_forbidden",
                message="无权在非本人所属的小组下创建专利项目",
            )
    else:
        target_group_info = ctx.groups[0]

    uid = _as_uuid(ctx.user_id)
    gid = _as_uuid(target_group_info.group_id)

    project = Project(
        name=cleaned_name,
        description=description.strip(),
        owner_id=uid,
        group_id=gid,
        status="active",
    )
    db.add(project)
    db.flush()

    member = ProjectMember(
        project_id=project.id,
        user_id=uid,
        role="project_owner",
    )
    db.add(member)

    private_space = KnowledgeSpace(
        name=f"{cleaned_name}-项目私有空间",
        scope_type="project_private",
        group_id=gid,
        project_id=project.id,
        created_by=uid,
        status="active",
    )
    db.add(private_space)
    db.flush()

    record_audit_event(
        db,
        actor_user_id=uid,
        group_id=gid,
        project_id=project.id,
        action="project.create",
        resource_type="project",
        resource_id=project.id,
        result="success",
        metadata={"knowledge_space_id": str(private_space.id)},
        commit=True,
    )
    db.refresh(project)
    db.refresh(private_space)
    return project, private_space


def list_accessible_projects(db: Session, user_id: str | uuid.UUID) -> list[Project]:
    """仅列出当前用户在服务端有明确成员关系且所属组匹配的项目。"""
    authz = AuthorizationService(db)
    ctx = authz.get_user_context(user_id)
    if not ctx.group_ids:
        return []
    uid = _as_uuid(ctx.user_id)
    allowed_gids = [_as_uuid(gid) for gid in ctx.group_ids]
    stmt = (
        select(Project)
        .join(ProjectMember, ProjectMember.project_id == Project.id)
        .where(
            ProjectMember.user_id == uid,
            Project.group_id.in_(allowed_gids),
        )
        .order_by(Project.created_at.desc())
    )
    return list(db.scalars(stmt).all())


def get_project_for_user(
    db: Session, user_id: str | uuid.UUID, project_id: str | uuid.UUID
) -> Project:
    authz = AuthorizationService(db)
    return authz.assert_project_access(user_id, project_id, "view")


def update_project_metadata(
    db: Session,
    *,
    user_id: str | uuid.UUID,
    project_id: str | uuid.UUID,
    name: str | None = None,
    description: str | None = None,
    status: str | None = None,
) -> Project:
    authz = AuthorizationService(db)
    # 即使项目已归档，project_owner 也应能查看，但修改前先校验权限
    ctx = authz.get_user_context(user_id)
    pid = _as_uuid(project_id)
    project = db.get(Project, pid)
    if project is None or str(project.group_id) not in ctx.group_ids:
        raise ResourceNotFoundError()
    role = ctx.project_role(str(pid))
    if role not in {"project_owner", "project_editor"}:
        raise AuthorizationError(code="insufficient_project_role", message="无权修改项目元数据")

    if name is not None:
        cleaned = name.strip()
        if not cleaned:
            raise AppError(422, "invalid_project_name", "项目名称不能为空")
        project.name = cleaned
    if description is not None:
        project.description = description.strip()
    if status is not None:
        if status not in {"active", "archived", "closed"}:
            raise AppError(422, "invalid_project_status", "项目状态必须是 active、archived 或 closed")
        project.status = status

    record_audit_event(
        db,
        actor_user_id=ctx.user_id,
        group_id=project.group_id,
        project_id=project.id,
        action="project.update",
        resource_type="project",
        resource_id=project.id,
        result="success",
        metadata={"status": project.status},
        commit=True,
    )
    db.refresh(project)
    return project


def add_project_member(
    db: Session,
    *,
    actor_user_id: str | uuid.UUID,
    project_id: str | uuid.UUID,
    target_user_id: str | uuid.UUID,
    role: str = "project_viewer",
) -> ProjectMember:
    authz = AuthorizationService(db)
    project = authz.assert_project_access(actor_user_id, project_id, "manage_members")
    if role not in PROJECT_ROLES:
        raise AppError(422, "invalid_project_role", f"项目角色必须是 {PROJECT_ROLES} 之一")

    target_uid = _as_uuid(target_user_id)
    target_user = db.get(User, target_uid)
    if target_user is None or not target_user.is_active:
        raise ResourceNotFoundError(code="user_not_found", message="目标用户不存在")

    # 严格三组隔离：不允许将其他小组的用户直接加进本组项目（跨组协作默认关闭）
    same_group = db.scalar(
        select(GroupMember).where(
            GroupMember.user_id == target_uid,
            GroupMember.group_id == project.group_id,
        )
    )
    if same_group is None:
        record_audit_event(
            db,
            actor_user_id=actor_user_id,
            group_id=project.group_id,
            project_id=project.id,
            action="project.add_member",
            resource_type="project_member",
            resource_id=target_uid,
            result="denied",
            metadata={"reason": "cross_group_member_forbidden"},
            commit=True,
        )
        raise AuthorizationError(
            code="cross_group_member_forbidden",
            message="禁止跨小组添加项目成员（不同小组严格隔离）",
        )

    existing = db.scalar(
        select(ProjectMember).where(
            ProjectMember.project_id == project.id,
            ProjectMember.user_id == target_uid,
        )
    )
    if existing is not None:
        existing.role = role
        member = existing
    else:
        member = ProjectMember(project_id=project.id, user_id=target_uid, role=role)
        db.add(member)

    record_audit_event(
        db,
        actor_user_id=actor_user_id,
        group_id=project.group_id,
        project_id=project.id,
        action="project.add_member",
        resource_type="project_member",
        resource_id=target_uid,
        result="success",
        metadata={"role": role},
        commit=True,
    )
    db.refresh(member)
    return member


def remove_project_member(
    db: Session,
    *,
    actor_user_id: str | uuid.UUID,
    project_id: str | uuid.UUID,
    target_user_id: str | uuid.UUID,
) -> None:
    authz = AuthorizationService(db)
    project = authz.assert_project_access(actor_user_id, project_id, "manage_members")
    target_uid = _as_uuid(target_user_id)
    if target_uid == project.owner_id:
        raise AppError(409, "cannot_remove_owner", "不能移除项目创建者/所有者")

    member = db.scalar(
        select(ProjectMember).where(
            ProjectMember.project_id == project.id,
            ProjectMember.user_id == target_uid,
        )
    )
    if member is None:
        raise ResourceNotFoundError(code="member_not_found", message="该项目成员不存在")

    db.delete(member)
    record_audit_event(
        db,
        actor_user_id=actor_user_id,
        group_id=project.group_id,
        project_id=project.id,
        action="project.remove_member",
        resource_type="project_member",
        resource_id=target_uid,
        result="success",
        commit=True,
    )


def get_project_private_space(db: Session, project_id: uuid.UUID) -> KnowledgeSpace:
    space = db.scalar(
        select(KnowledgeSpace).where(
            KnowledgeSpace.project_id == project_id,
            KnowledgeSpace.scope_type == "project_private",
            KnowledgeSpace.status == "active",
        )
    )
    if space is None:
        raise ResourceNotFoundError(
            code="project_space_missing", message="未找到项目的私有知识空间"
        )
    return space


def list_project_spaces(
    db: Session, user_id: str | uuid.UUID, project_id: str | uuid.UUID
) -> list[KnowledgeSpace]:
    authz = AuthorizationService(db)
    space_ids = authz.resolve_search_scopes(user_id, project_id)
    uuids = [_as_uuid(sid) for sid in space_ids]
    return list(
        db.scalars(
            select(KnowledgeSpace)
            .where(KnowledgeSpace.id.in_(uuids))
            .order_by(KnowledgeSpace.scope_type.asc())
        ).all()
    )


def get_group_by_id(db: Session, group_id: uuid.UUID) -> Group | None:
    return db.get(Group, group_id)
