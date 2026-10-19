"""配置读取与强校验。确保所有推理、检索和存储严格指向本地获准环境。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_EMBEDDING_DIMENSION = 1024

LLMProviderType = Literal["openai_compatible", "ollama", "mock"]
EmbeddingBackendType = Literal["local", "openai_compatible", "ollama", "mock"]
RerankerBackendType = Literal["local", "mock"]
VectorStoreBackendType = Literal["pgvector", "qdrant", "memory"]

BLOCKED_EXTERNAL_HOSTS = frozenset(
    {
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "api.groq.com",
        "api.deepseek.com",
        "openrouter.ai",
        "api.mistral.ai",
        "api.together.xyz",
        "api.fireworks.ai",
        "api.x.ai",
        "api.cohere.ai",
        "api.jina.ai",
        "dashscope.aliyuncs.com",
    }
)


def load_yaml_config(name: str) -> dict:
    path = PROJECT_ROOT / "configs" / name
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8080, ge=1, le=65535)
    log_level: str = "INFO"

    database_url: str
    data_root: str = "./data/workspaces"
    upload_dir: str = "./data/uploads"
    artifact_dir: str = "./data/artifacts"

    llm_provider: LLMProviderType = "openai_compatible"
    llm_base_url: str = "http://127.0.0.1:8001/v1"
    llm_model: str = "Qwen3.8-27B-FP8"
    llm_api_key: str = "local-only-placeholder"
    llm_timeout_seconds: int = Field(default=120, ge=1, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    embedding_backend: EmbeddingBackendType = Field(
        default="local",
        validation_alias=AliasChoices("EMBEDDING_BACKEND", "EMBEDDING_PROVIDER"),
    )
    embedding_model_path: str = Field(
        default="models/bge-large-zh-v1.5",
        validation_alias=AliasChoices("EMBEDDING_MODEL_PATH", "EMBEDDING_MODEL"),
    )
    embedding_dimension: int = 1024
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_device: str = "cpu"
    embedding_query_instruction: str = "为这个句子生成表示以用于检索："

    reranker_backend: RerankerBackendType = "local"
    reranker_enabled: bool = False
    reranker_model_path: str = Field(
        default="models/bge-reranker-v2-m3",
        validation_alias=AliasChoices("RERANKER_MODEL_PATH", "RERANKER_MODEL"),
    )
    reranker_device: str = "cpu"

    vector_store_backend: VectorStoreBackendType = "pgvector"
    qdrant_url: str = "http://127.0.0.1:6333"
    bm25_backend: str = "local"

    dense_top_k: int = Field(
        default=30,
        ge=1,
        le=200,
        validation_alias=AliasChoices("DENSE_TOP_K", "RETRIEVAL_TOP_K"),
    )
    bm25_top_k: int = Field(default=30, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1, le=500)
    reranker_top_n: int = Field(
        default=8,
        ge=1,
        le=100,
        validation_alias=AliasChoices("RERANKER_TOP_N", "RERANK_TOP_N"),
    )
    max_retrieval_candidates: int = Field(default=50, ge=1, le=200)

    chunk_size: int = Field(default=500, ge=50, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)
    parser_version: str = "v1.0"
    index_version: str = "v1"

    max_revision_rounds: int = Field(default=2, ge=0, le=5)
    max_llm_calls_per_task: int = Field(default=30, ge=1, le=200)
    node_timeout_seconds: int = Field(default=120, ge=1, le=600)
    task_timeout_seconds: int = Field(default=1800, ge=10, le=7200)
    max_context_tokens: int = Field(default=4096, ge=256, le=32768)
    max_upload_size_mb: int = Field(default=100, ge=1, le=500)

    enable_external_telemetry: bool = False

    @property
    def llm_mock(self) -> bool:
        return self.llm_provider == "mock"

    @property
    def embedding_mock(self) -> bool:
        return self.embedding_backend == "mock"

    @property
    def reranker_mock(self) -> bool:
        return self.reranker_backend == "mock" or self.embedding_mock

    @property
    def mock_mode(self) -> bool:
        return self.llm_mock or self.embedding_mock

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path

    @model_validator(mode="after")
    def validate_local_constraints(self) -> Settings:
        if self.enable_external_telemetry:
            raise ValueError("ENABLE_EXTERNAL_TELEMETRY 必须为 false，严禁启用外部遥测")
        if self.embedding_dimension != SCHEMA_EMBEDDING_DIMENSION:
            raise ValueError(
                f"EMBEDDING_DIMENSION ({self.embedding_dimension}) 与向量维度 {SCHEMA_EMBEDDING_DIMENSION} 不一致"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        _assert_local_endpoint("LLM_BASE_URL", self.llm_base_url, self.llm_provider)
        if self.embedding_backend in {"openai_compatible", "ollama"}:
            _assert_local_endpoint(
                "EMBEDDING_BASE_URL", self.embedding_base_url, self.embedding_backend
            )
        if self.vector_store_backend == "qdrant":
            _assert_local_endpoint("QDRANT_URL", self.qdrant_url, "qdrant")
        return self


def _assert_local_endpoint(field_name: str, url: str, backend: str) -> None:
    if backend in {"mock", "local", "memory", "pgvector"}:
        return
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError(f"{field_name} 必须是合法的 http(s) 地址")
    if (
        host in BLOCKED_EXTERNAL_HOSTS
        or host.endswith((".openai.azure.com", ".aliyuncs.com"))
    ):
        raise ValueError(f"{field_name} 指向外部云端服务 ({host})，系统严禁连接外部 AI API")


@lru_cache
def get_settings() -> Settings:
    return Settings()
