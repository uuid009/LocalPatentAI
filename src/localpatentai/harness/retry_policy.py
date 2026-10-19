"""Retry Policy（规格 8.1 节）：区分权限错误、JSON 格式错误、服务不可用与 PDF 渲染错误的重试策略。"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from localpatentai.core.errors import AppError, AuthenticationError, AuthorizationError

T = TypeVar("T")

NON_RETRYABLE_CODES = frozenset(
    {
        "forbidden",
        "unauthenticated",
        "not_found",
        "insufficient_project_role",
        "project_not_active",
        "cross_group_space_forbidden",
        "unsupported_file_type",
        "llm_budget_exceeded",
    }
)


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (AuthenticationError, AuthorizationError)):
        return False
    if isinstance(exc, AppError):
        return exc.code not in NON_RETRYABLE_CODES and exc.status_code in {408, 429, 502, 503, 504}
    return isinstance(exc, TimeoutError)


def execute_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 2,
    backoff_seconds: float = 0.05,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not is_retryable_error(exc) or attempt >= max_retries:
                raise
            time.sleep(backoff_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc
