"""RBAC + 资源范围策略表（结合角色与组/项目/知识空间判断，默认拒绝）。"""

from __future__ import annotations

PROJECT_ACTION_ALLOWED_ROLES: dict[str, set[str]] = {
    "view": {"project_owner", "project_editor", "project_viewer", "reviewer"},
    "search": {"project_owner", "project_editor", "project_viewer", "reviewer"},
    "upload_document": {"project_owner", "project_editor"},
    "reindex_document": {"project_owner", "project_editor"},
    "delete_document": {"project_owner", "project_editor"},
    "run_workflow": {"project_owner", "project_editor"},
    "resume_workflow": {"project_owner", "project_editor", "reviewer"},
    "review_draft": {"project_owner", "project_editor", "reviewer"},
    "approve_export": {"project_owner", "reviewer"},
    "download_artifact": {"project_owner", "project_editor", "project_viewer", "reviewer"},
    "update_project": {"project_owner", "project_editor"},
    "manage_members": {"project_owner"},
}

KNOWLEDGE_SPACE_ACTION_ALLOWED_GROUP_ROLES: dict[str, set[str]] = {
    "read": {"member", "group_admin"},
    "write": {"group_admin"},
    "publish_shared": {"group_admin"},
}


def is_project_action_allowed(project_role: str | None, action: str) -> bool:
    if not project_role:
        return False
    allowed_roles = PROJECT_ACTION_ALLOWED_ROLES.get(action)
    if not allowed_roles:
        return False
    return project_role in allowed_roles
