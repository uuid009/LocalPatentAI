"""Budget Controller（规格 8.1 节）：管理单任务 LLM 调用次数预算、修订轮次上限与超时控制。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from localpatentai.core.config import Settings, get_settings
from localpatentai.core.errors import AppError


class TaskBudgetState(BaseModel):
    max_llm_calls_per_task: int = Field(default=30, ge=1)
    max_revision_rounds: int = Field(default=2, ge=0)
    node_timeout_seconds: int = Field(default=120, ge=1)
    task_timeout_seconds: int = Field(default=1800, ge=10)
    max_context_tokens: int = Field(default=4096, ge=256)
    max_retrieval_candidates: int = Field(default=50, ge=1)
    max_upload_size_mb: int = Field(default=100, ge=1)

    llm_calls_used: int = 0
    revisions_used: int = 0


class BudgetController:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def create_initial_budget(self) -> TaskBudgetState:
        return TaskBudgetState(
            max_llm_calls_per_task=self.settings.max_llm_calls_per_task,
            max_revision_rounds=self.settings.max_revision_rounds,
            node_timeout_seconds=self.settings.node_timeout_seconds,
            task_timeout_seconds=self.settings.task_timeout_seconds,
            max_context_tokens=self.settings.max_context_tokens,
            max_retrieval_candidates=self.settings.max_retrieval_candidates,
            max_upload_size_mb=self.settings.max_upload_size_mb,
        )

    def consume_llm_call(self, current_used: int, count: int = 1) -> int:
        next_val = current_used + count
        if next_val > self.settings.max_llm_calls_per_task:
            raise AppError(
                429,
                "llm_budget_exceeded",
                f"任务 LLM 调用次数已达上限 ({self.settings.max_llm_calls_per_task})",
            )
        return next_val

    def can_revise(self, revision_count: int) -> bool:
        return revision_count < self.settings.max_revision_rounds
