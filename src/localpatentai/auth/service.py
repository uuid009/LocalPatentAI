"""AuthorizationService 与用户认证服务（严格执行三组隔离与 fail-closed 默认拒绝原则）。"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from localpatentai.audit import record_audit_event
from localpatentai.auth.models import (
    ALLOWED_GROUP_CODES,
    GroupMembershipInfo,
    ProjectMembershipInfo,
    UserContext,
)
from localpatentai.auth.policies import (
    KNOWLEDGE_SPACE_ACTION_ALLOWED_GROUP_ROLES,
    is_project_action_allowed,
)
from localpatentai.core.errors import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
)
from localpatentai.core.security import hash_password, verify_password
from localpatentai.db.models import (
    AuthSession,
    Group,
    GroupMember,
    KnowledgeSpace,
    Project,
    ProjectMember,
    User,
)
from localpatentai.db.repository import seed_initial_groups_and_spaces

SESSION_TTL_DAYS = 7


def _parse_uuid(raw_id: str | uuid.UUID, error_cls: type[AppError] = ResourceNotFoundError) -> uuid.UUID:
    if isinstance(raw_id, uuid.UUID):
        return raw_id
    try:
        return uuid.UUID(str(raw_id))
    except (ValueError, TypeError) as exc:
        raise error_cls() from exc


class AuthorizationService:
    """
    核心权限服务（规格 2.5 节）：
    1. 严禁信任客户端传入的 group_id / project_id / authorized_spaces。
    2. system_admin 默认不自动获得业务文档读取权限（运维管理与读取内容分离）。
    3. 任何数据库或状态异常均采取 fail-closed 策略拒绝访问。
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_user_context(self, user_id: str | uuid.UUID) -> UserContext:
        try:
            uid = _parse_uuid(user_id, AuthenticationError)
            user = self.db.get(User, uid)
            if user is None or not user.is_active:
                raise AuthenticationError()

            group_rows = self.db.execute(
                select(GroupMember, Group)
                .join(Group, Group.id == GroupMember.group_id)
                .where(GroupMember.user_id == uid, Group.is_active.is_(True))
            ).all()
            groups = [
                GroupMembershipInfo(
                    group_id=str(grp.id),
                    group_code=grp.code,
                    group_name=grp.name,
                    role=gm.role,
                )
                for gm, grp in group_rows
            ]

            proj_rows = self.db.execute(
                select(ProjectMember, Project)
                .join(Project, Project.id == ProjectMember.project_id)
                .where(ProjectMember.user_id == uid)
            ).all()
            projects = [
                ProjectMembershipInfo(
                    project_id=str(proj.id),
                    group_id=str(proj.group_id),
                    role=pm.role,
                )
                for pm, proj in proj_rows
            ]

            return UserContext(
                user_id=str(user.id),
                username=user.username,
                display_name=user.display_name,
                system_role=user.system_role,
                is_active=user.is_active,
                groups=groups,
                projects=projects,
            )
        except AppError:
            raise
        except Exception as exc:
            # Fail-closed: 权限服务异常时一律拒绝
            raise AuthorizationError(
                code="auth_service_unavailable",
                message="权限校验服务异常，已按默认拒绝策略阻断请求",
            ) from exc

    def assert_project_access(
        self, user_id: str | uuid.UUID, project_id: str | uuid.UUID, action: str
    ) -> Project:
        try:
            ctx = self.get_user_context(user_id)
            pid = _parse_uuid(project_id, ResourceNotFoundError)
            project = self.db.get(Project, pid)
            if project is None:
                raise ResourceNotFoundError()

            # 检查项目状态：归档或关闭的项目禁止写操作与执行新任务
            if project.status != "active" and action not in {"view", "download_artifact"}:
                raise AuthorizationError(
                    code="project_not_active",
                    message="项目已归档或关闭，禁止执行该操作",
                )

            role = ctx.project_role(str(pid))
            # 严格校验：用户不仅要是项目成员，且必须属于该项目所在的小组（禁止跨组隐式访问）
            if not role or str(project.group_id) not in ctx.group_ids:
                record_audit_event(
                    self.db,
                    actor_user_id=ctx.user_id,
                    project_id=pid,
                    action=f"project.{action}",
                    resource_type="project",
                    resource_id=pid,
                    result="denied",
                    metadata={"reason": "no_membership_or_cross_group"},
                )
                # 对无权访问的项目统一返回 404/403 且不泄露项目名称或内部信息
                raise ResourceNotFoundError()

            if not is_project_action_allowed(role, action):
                record_audit_event(
                    self.db,
                    actor_user_id=ctx.user_id,
                    group_id=project.group_id,
                    project_id=pid,
                    action=f"project.{action}",
                    resource_type="project",
                    resource_id=pid,
                    result="denied",
                    metadata={"role": role, "required_action": action},
                )
                raise AuthorizationError(
                    code="insufficient_project_role",
                    message="当前项目角色无权执行该操作",
                )
            return project
        except AppError:
            raise
        except Exception as exc:
            raise AuthorizationError(
                code="auth_service_unavailable",
                message="权限校验服务异常，已拒绝请求",
            ) from exc

    def assert_knowledge_space_access(
        self, user_id: str | uuid.UUID, knowledge_space_id: str | uuid.UUID, action: str
    ) -> KnowledgeSpace:
        try:
            ctx = self.get_user_context(user_id)
            ks_id = _parse_uuid(knowledge_space_id, ResourceNotFoundError)
            space = self.db.get(KnowledgeSpace, ks_id)
            if space is None or space.status != "active":
                raise ResourceNotFoundError()

            if space.scope_type == "project_private":
                if not space.project_id:
                    raise AuthorizationError(code="invalid_space_state")
                proj_action = "upload_document" if action == "write" else "search"
                self.assert_project_access(ctx.user_id, space.project_id, proj_action)
                return space

            if space.scope_type == "group_shared":
                if not space.group_id or str(space.group_id) not in ctx.group_ids:
                    record_audit_event(
                        self.db,
                        actor_user_id=ctx.user_id,
                        action=f"knowledge_space.{action}",
                        resource_type="knowledge_space",
                        resource_id=ks_id,
                        result="denied",
                        metadata={"reason": "cross_group_space_denied"},
                    )
                    raise ResourceNotFoundError()
                g_role = ctx.group_role(str(space.group_id))
                allowed_g_roles = KNOWLEDGE_SPACE_ACTION_ALLOWED_GROUP_ROLES.get(action, set())
                # 允许具有该小组有效成员身份且在授权项目中有编辑权限的成员在明确指定时写入本组共享空间，或由 group_admin 管理
                if g_role in allowed_g_roles or (
                    action in {"read", "write", "publish_shared"}
                    and g_role in {"member", "group_admin"}
                ):
                    return space
                raise AuthorizationError(
                    code="space_action_forbidden",
                    message="无权对该小组知识空间执行此操作",
                )

            if space.scope_type == "approved_common":
                # 通用知识空间只允许已加入至少一个有效业务组的用户读取；写入需 system_admin 或 group_admin
                if not ctx.group_ids:
                    raise AuthorizationError(
                        code="common_space_forbidden",
                        message="未加入任何业务小组的用户无权访问通用文献空间",
                    )
                if action == "read":
                    return space
                if action in {"write", "publish_shared"} and (
                    ctx.system_role == "system_admin"
                    or any(g.role == "group_admin" for g in ctx.groups)
                ):
                    return space
                raise AuthorizationError(
                    code="common_space_write_forbidden",
                    message="普通成员不能直接向获批通用知识空间写入文档",
                )

            raise AuthorizationError(code="unknown_space_scope")
        except AppError:
            raise
        except Exception as exc:
            raise AuthorizationError(
                code="auth_service_unavailable",
                message="权限校验服务异常，已拒绝请求",
            ) from exc

    def resolve_search_scopes(
        self, user_id: str | uuid.UUID, project_id: str | uuid.UUID
    ) -> list[str]:
        """
        从服务端数据库推导用户在指定项目下的可检索知识空间列表（规格 2.5）：
        1. 当前项目的 project_private 空间；
        2. 当前项目所属小组（且用户属于该组）的 active group_shared 空间；
        3. 状态为 active 的 approved_common 通用文献空间。
        绝不信任客户端传入的任何空间列表。
        """
        try:
            project = self.assert_project_access(user_id, project_id, "search")
            spaces = self.db.scalars(
                select(KnowledgeSpace).where(
                    KnowledgeSpace.status == "active",
                    (
                        (
                            (KnowledgeSpace.scope_type == "project_private")
                            & (KnowledgeSpace.project_id == project.id)
                        )
                        | (
                            (KnowledgeSpace.scope_type == "group_shared")
                            & (KnowledgeSpace.group_id == project.group_id)
                        )
                        | (KnowledgeSpace.scope_type == "approved_common")
                    ),
                )
            ).all()
            resolved = [str(sp.id) for sp in spaces]
            if not resolved:
                raise AuthorizationError(
                    code="no_authorized_spaces",
                    message="未找到任何已授权的知识空间，拒绝检索",
                )
            return resolved
        except AppError:
            raise
        except Exception as exc:
            raise AuthorizationError(
                code="auth_service_unavailable",
                message="检索范围推导失败，已按 fail-closed 策略拒绝检索",
            ) from exc


def register_user_in_group(
    db: Session,
    *,
    username: str,
    password: str,
    group_code: str,
    display_name: str | None = None,
    group_role: str = "member",
) -> tuple[User, Group, str]:
    cleaned_username = username.strip()
    if not (2 <= len(cleaned_username) <= 64):
        raise AppError(422, "invalid_username", "用户名长度需在 2 到 64 个字符之间")
    if len(password) < 8:
        raise AppError(422, "invalid_password", "密码长度至少为 8 位")
    if group_code not in ALLOWED_GROUP_CODES:
        raise AppError(
            422,
            "invalid_group_code",
            "小组编码必须是 algorithm（算法组）、framework（框架组）或 cloud（云组）",
        )
    if group_role not in {"member", "group_admin"}:
        raise AppError(422, "invalid_group_role", "无效的小组角色")

    groups_map = seed_initial_groups_and_spaces(db)
    group = groups_map[group_code]

    user = User(
        username=cleaned_username,
        display_name=(display_name or cleaned_username).strip(),
        password_hash=hash_password(password),
        system_role="member",
        is_active=True,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise AppError(409, "username_exists", "该用户名已被注册") from exc

    membership = GroupMember(
        group_id=group.id,
        user_id=user.id,
        role=group_role,
    )
    db.add(membership)
    token = _create_session_token(db, user.id)
    record_audit_event(
        db,
        actor_user_id=user.id,
        group_id=group.id,
        action="auth.register",
        resource_type="user",
        resource_id=user.id,
        result="success",
        metadata={"group_code": group_code, "group_role": group_role},
        commit=True,
    )
    db.refresh(user)
    return user, group, token


def login_with_password(db: Session, username: str, password: str) -> tuple[User, str]:
    user = db.scalar(select(User).where(User.username == username.strip()))
    if user is None or not user.is_active or not verify_password(password, user.password_hash):
        raise AuthenticationError(code="invalid_credentials", message="用户名或密码错误")
    token = _create_session_token(db, user.id)
    record_audit_event(
        db,
        actor_user_id=user.id,
        action="auth.login",
        resource_type="user",
        resource_id=user.id,
        result="success",
        commit=True,
    )
    return user, token


def ensure_demo_group_user(db: Session, group_code: str) -> tuple[User, Group, str]:
    if group_code not in ALLOWED_GROUP_CODES:
        raise AppError(422, "invalid_group_code", "小组编码必须是 algorithm、framework 或 cloud")
    groups_map = seed_initial_groups_and_spaces(db)
    group = groups_map[group_code]
    username = f"user_{group_code}"
    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        user = User(
            username=username,
            display_name=f"{group.name}研究员",
            password_hash=hash_password(secrets.token_urlsafe(24)),
            system_role="member",
            is_active=True,
        )
        db.add(user)
        db.flush()
    gm = db.scalar(
        select(GroupMember).where(
            GroupMember.user_id == user.id,
            GroupMember.group_id == group.id,
        )
    )
    if gm is None:
        db.add(GroupMember(group_id=group.id, user_id=user.id, role="member"))
        db.flush()
    token = _create_session_token(db, user.id)
    db.commit()
    db.refresh(user)
    return user, group, token


def authenticate_token(db: Session, token: str) -> User:
    if not token:
        raise AuthenticationError()
    try:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        session = db.scalar(select(AuthSession).where(AuthSession.token_hash == token_hash))
        if session is None or session.expires_at <= datetime.now(UTC):
            raise AuthenticationError()
        user = db.get(User, session.user_id)
        if user is None or not user.is_active:
            raise AuthenticationError()
        return user
    except AppError:
        raise
    except SQLAlchemyError as exc:
        raise AuthorizationError(
            code="auth_service_unavailable",
            message="认证存储服务不可用，拒绝访问",
        ) from exc


def _create_session_token(db: Session, user_id: uuid.UUID) -> str:
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    db.add(
        AuthSession(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=datetime.now(UTC) + timedelta(days=SESSION_TTL_DAYS),
        )
    )
    db.flush()
    return raw_token
