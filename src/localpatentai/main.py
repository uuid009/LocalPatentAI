"""FastAPI 应用入口与健康检查（规格 3.3 & 14 Sprint 1）。"""

from __future__ import annotations

from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from localpatentai import __version__
from localpatentai.api.routes import router as api_v1_router
from localpatentai.core.config import get_settings
from localpatentai.core.errors import register_exception_handlers
from localpatentai.core.logging import configure_logging
from localpatentai.db.repository import seed_initial_groups_and_spaces
from localpatentai.db.session import get_session_factory, to_psycopg_url
from localpatentai.llm.qwen_client import probe_vllm_service
from localpatentai.workflows.graph import (
    close_workflow_checkpointer,
    get_checkpointer_health,
    init_workflow_checkpointer,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.resolve_path(settings.data_root).mkdir(parents=True, exist_ok=True)
    settings.resolve_path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    settings.resolve_path(settings.artifact_dir).mkdir(parents=True, exist_ok=True)
    try:
        with get_session_factory()() as db:
            seed_initial_groups_and_spaces(db)
    except Exception:
        pass
    init_workflow_checkpointer(settings.database_url)
    yield
    close_workflow_checkpointer()


app = FastAPI(
    title="LocalPatentAI",
    description="面向科研成果转化的本地化多智能体专利辅助撰写系统",
    version=__version__,
    lifespan=lifespan,
)
register_exception_handlers(app)
app.include_router(api_v1_router)


@app.get("/health")
def health_check():
    settings = get_settings()
    db_status = "error"
    pgvector_version = None
    db_error = None
    try:
        conninfo = to_psycopg_url(settings.database_url)
        with (
            psycopg.connect(conninfo, connect_timeout=4) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            pgvector_version = row[0] if row else None
        db_status = "ok"
    except Exception as exc:
        db_error = type(exc).__name__

    models_health = probe_vllm_service(settings)
    payload = {
        "status": "ok" if db_status == "ok" else "degraded",
        "version": __version__,
        "app_env": settings.app_env,
        "database": db_status,
        "pgvector": pgvector_version,
        "checkpointer": get_checkpointer_health(),
        "models": models_health,
        "external_telemetry": settings.enable_external_telemetry,
    }
    if db_error:
        payload["database_error"] = db_error
        return JSONResponse(status_code=503, content=payload)
    return payload
