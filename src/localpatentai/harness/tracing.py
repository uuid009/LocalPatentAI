"""Trace & Audit（规格 8.1 节）：记录工作流节点级执行轨迹，脱敏且不记录完整原文或完整 Prompt。"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class NodeTraceEntry(BaseModel):
    task_id: str
    node_name: str
    started_at: str
    finished_at: str
    duration_ms: float
    model_version: str = "Qwen3.8-27B-FP8"
    prompt_version: str = "v1.0"
    retrieval_version: str = "v1"
    schema_valid: bool = True
    error_code: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def trace_node_execution(
    *,
    task_id: str,
    node_name: str,
    start_perf: float,
    model_version: str = "Qwen3.8-27B-FP8",
    prompt_version: str = "v1.0",
    retrieval_version: str = "v1",
    schema_valid: bool = True,
    error_code: str | None = None,
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    now_iso = datetime.now(UTC).isoformat()
    duration_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
    entry = NodeTraceEntry(
        task_id=task_id,
        node_name=node_name,
        started_at=now_iso,
        finished_at=now_iso,
        duration_ms=duration_ms,
        model_version=model_version,
        prompt_version=prompt_version,
        retrieval_version=retrieval_version,
        schema_valid=schema_valid,
        error_code=error_code,
        evidence_ids=list(evidence_ids or []),
    )
    return entry.model_dump()
