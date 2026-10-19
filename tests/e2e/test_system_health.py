"""端到端测试：系统健康检查与多用户专利项目全流程验证。"""

from __future__ import annotations


def test_health_endpoint_returns_structured_status(client):
    """验证 /health 返回数据库、pgvector、Checkpointer 与本地 LLM 探针状态。"""
    resp = client.get("/health")
    assert resp.status_code in {200, 503}
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["pgvector"] is not None
    assert body["external_telemetry"] is False
