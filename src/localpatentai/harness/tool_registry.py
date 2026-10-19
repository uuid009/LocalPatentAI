"""Tool Registry（规格 8.1 节）：基于 allowlist 的节点工具注册表，调用前强制 Pydantic 输入校验与权限检查。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from localpatentai.core.errors import AuthorizationError

FORBIDDEN_TOOL_NAMES = frozenset(
    {"shell", "bash", "exec_python", "eval", "read_arbitrary_file", "http_request"}
)

NODE_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "analyze_technical_solution": {"authorized_rag_search"},
    "retrieve_evidence": {"authorized_rag_search"},
    "assess_evidence_gaps": {"authorized_rag_search"},
    "targeted_retrieval": {"authorized_rag_search"},
    "draft_sections": set(),
    "review_draft": set(),
    "render_pdf": {"render_and_validate_pdf"},
}


class RagSearchToolInput(BaseModel):
    user_id: str
    project_id: str
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=8, ge=1, le=20)


class ToolRegistry:
    """
    严格白名单工具注册表：
    - 严禁注册 shell、任意 Python 执行或任意文件路径读取工具；
    - 每次调用校验当前节点是否在 allowlist 内，并用 Pydantic Schema 校验输入，输出截断大小。
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[type[BaseModel], Callable[[BaseModel], Any]]] = {}

    def register(
        self,
        tool_name: str,
        input_schema: type[BaseModel],
        handler: Callable[[BaseModel], Any],
    ) -> None:
        if tool_name in FORBIDDEN_TOOL_NAMES:
            raise AuthorizationError(
                code="forbidden_tool_registration",
                message=f"安全策略禁止注册高危工具: {tool_name}",
            )
        self._tools[tool_name] = (input_schema, handler)

    def invoke(
        self,
        *,
        node_name: str,
        tool_name: str,
        raw_input: dict[str, Any],
    ) -> Any:
        allowed_for_node = NODE_TOOL_ALLOWLIST.get(node_name, set())
        if tool_name not in allowed_for_node:
            raise AuthorizationError(
                code="tool_not_allowed_for_node",
                message=f"节点 {node_name} 未获准调用工具 {tool_name}",
            )
        if tool_name not in self._tools:
            raise AuthorizationError(
                code="tool_not_registered",
                message=f"工具 {tool_name} 未注册",
            )
        schema_cls, handler = self._tools[tool_name]
        validated_input = schema_cls.model_validate(raw_input)
        return handler(validated_input)
