"""基础数据仓储与初始化辅助（三组初始化、公共知识空间初始化等）。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from localpatentai.core.security import hash_password
from localpatentai.db.models import Group, KnowledgeSpace, User

INITIAL_GROUPS: tuple[tuple[str, str], ...] = (
    ("algorithm", "算法组"),
    ("framework", "框架组"),
    ("cloud", "云组"),
)


def seed_initial_groups_and_spaces(db: Session) -> dict[str, Group]:
    """幂等创建系统初始三个隔离小组（algorithm / framework / cloud）及各自共享空间与通用文献空间。"""
    sys_admin = db.scalar(select(User).where(User.username == "system_bootstrap"))
    if sys_admin is None:
        sys_admin = User(
            username="system_bootstrap",
            display_name="系统初始化账号",
            password_hash=hash_password("bootstrap-disabled-direct-login"),
            system_role="system_admin",
            is_active=True,
        )
        db.add(sys_admin)
        db.flush()

    groups_by_code: dict[str, Group] = {}
    for code, name in INITIAL_GROUPS:
        group = db.scalar(select(Group).where(Group.code == code))
        if group is None:
            group = Group(code=code, name=name, is_active=True)
            db.add(group)
            db.flush()
        groups_by_code[code] = group

        shared_space = db.scalar(
            select(KnowledgeSpace).where(
                KnowledgeSpace.scope_type == "group_shared",
                KnowledgeSpace.group_id == group.id,
            )
        )
        if shared_space is None:
            shared_space = KnowledgeSpace(
                name=f"{name}共享知识空间",
                scope_type="group_shared",
                group_id=group.id,
                project_id=None,
                created_by=sys_admin.id,
                status="active",
            )
            db.add(shared_space)
            db.flush()

    common_space = db.scalar(
        select(KnowledgeSpace).where(KnowledgeSpace.scope_type == "approved_common")
    )
    if common_space is None:
        common_space = KnowledgeSpace(
            name="获批通用文献知识空间",
            scope_type="approved_common",
            group_id=None,
            project_id=None,
            created_by=sys_admin.id,
            status="active",
        )
        db.add(common_space)
        db.flush()

    db.commit()
    return groups_by_code
