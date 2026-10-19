"""认证与权限上下文 Pydantic 模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field

ALLOWED_GROUP_CODES = ("algorithm", "framework", "cloud")
PROJECT_ROLES = ("project_owner", "project_editor", "project_viewer", "reviewer")
GROUP_ROLES = ("member", "group_admin")
SYSTEM_ROLES = ("member", "system_admin")


class GroupMembershipInfo(BaseModel):
    group_id: str
    group_code: str
    group_name: str
    role: str


class ProjectMembershipInfo(BaseModel):
    project_id: str
    group_id: str
    role: str


class UserContext(BaseModel):
    user_id: str
    username: str
    display_name: str
    system_role: str = "member"
    is_active: bool = True
    groups: list[GroupMembershipInfo] = Field(default_factory=list)
    projects: list[ProjectMembershipInfo] = Field(default_factory=list)

    @property
    def group_ids(self) -> set[str]:
        return {g.group_id for g in self.groups}

    @property
    def group_codes(self) -> set[str]:
        return {g.group_code for g in self.groups}

    def project_role(self, project_id: str) -> str | None:
        for pm in self.projects:
            if pm.project_id == str(project_id):
                return pm.role
        return None

    def group_role(self, group_id: str) -> str | None:
        for gm in self.groups:
            if gm.group_id == str(group_id):
                return gm.role
        return None
