"""本地 BGE Embedding 客户端。支持本地路径加载（sentence-transformers）与确定性测试 mock，严禁回退外部云 API。"""

from __future__ import annotations

import hashlib
import logging
import math
from functools import lru_cache
from pathlib import Path

import httpx

from localpatentai.core.config import Settings, get_settings
from localpatentai.core.errors import AppError

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """统一 Embedding 接口（规格 4.3 节）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def model_id(self) -> str:
        if self.settings.embedding_mock:
            return "mock-bge-1024"
        path = Path(self.settings.embedding_model_path)
        return path.name or "bge-large-zh-v1.5"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [self._sanitize_input(t) for t in texts]
        if self.settings.embedding_backend == "mock":
            return [
                _deterministic_mock_vector(text, self.settings.embedding_dimension)
                for text in cleaned
            ]
        if self.settings.embedding_backend == "local":
            return _encode_with_local_bge(cleaned, self.settings, is_query=False)
        if self.settings.embedding_backend in {"openai_compatible", "ollama"}:
            return _encode_with_local_http(cleaned, self.settings)
        raise AppError(
            500,
            "unsupported_embedding_backend",
            f"不支持的 EMBEDDING_BACKEND: {self.settings.embedding_backend}",
        )

    def embed_query(self, query: str) -> list[float]:
        cleaned = self._sanitize_input(query)
        if not cleaned:
            raise AppError(400, "empty_query", "查询文本不能为空")
        if self.settings.embedding_backend == "mock":
            return _deterministic_mock_vector(cleaned, self.settings.embedding_dimension)
        if self.settings.embedding_backend == "local":
            return _encode_with_local_bge([cleaned], self.settings, is_query=True)[0]
        return self.embed_documents([cleaned])[0]

    @staticmethod
    def _sanitize_input(text: str, max_chars: int = 4096) -> str:
        stripped = (text or "").replace("\x00", "").strip()
        if not stripped:
            return " "
        return stripped[:max_chars]


def _deterministic_mock_vector(text: str, dimension: int) -> list[float]:
    """
    基于字符与二元词特征的确定性哈希投影向量（用于离线单元测试与快速回归测试），
    使具有相同中文术语的查询与片段在余弦空间中也具备稳定的相似度。
    """
    vec = [0.0] * dimension
    cleaned = text.strip()
    tokens: list[str] = [cleaned]
    for i in range(len(cleaned)):
        ch = cleaned[i]
        if not ch.isspace():
            tokens.append(ch)
        if i + 1 < len(cleaned):
            bi = cleaned[i : i + 2].strip()
            if len(bi) == 2:
                tokens.append(bi)

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        weight = 2.0 if len(token) >= 2 else 1.0
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


@lru_cache(maxsize=2)
def _load_local_sentence_transformer(model_path_str: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise AppError(
            500,
            "sentence_transformers_missing",
            "本地加载 BGE Embedding 需要安装 sentence-transformers",
        ) from exc

    settings = get_settings()
    resolved = settings.resolve_path(model_path_str)
    if not resolved.exists():
        raise AppError(
            500,
            "embedding_model_not_found",
            f"本地 BGE Embedding 模型目录不存在: {resolved}",
        )
    logger.info("loading_local_bge_embedding path=%s device=%s", resolved, device)
    return SentenceTransformer(str(resolved), device=device)


def _encode_with_local_bge(
    texts: list[str], settings: Settings, *, is_query: bool
) -> list[list[float]]:
    model = _load_local_sentence_transformer(
        settings.embedding_model_path, settings.embedding_device
    )
    inputs = texts
    if is_query and settings.embedding_query_instruction:
        inputs = [f"{settings.embedding_query_instruction}{t}" for t in texts]
    raw_vectors = model.encode(
        inputs,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    results: list[list[float]] = []
    for vec in raw_vectors:
        float_vec = [float(x) for x in vec.tolist()]
        if len(float_vec) != settings.embedding_dimension:
            raise AppError(
                500,
                "embedding_dimension_mismatch",
                f"模型输出维度 {len(float_vec)} 与配置维度 {settings.embedding_dimension} 不一致",
            )
        results.append(float_vec)
    return results


def _encode_with_local_http(texts: list[str], settings: Settings) -> list[list[float]]:
    base = settings.embedding_base_url.rstrip("/")
    if not base.endswith("/v1") and settings.embedding_backend == "openai_compatible":
        base = f"{base}/v1"
    endpoint = (
        f"{base}/api/embed"
        if settings.embedding_backend == "ollama"
        else f"{base}/embeddings"
    )
    try:
        resp = httpx.post(
            endpoint,
            json={"model": settings.embedding_model_path, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise AppError(503, "embedding_service_unavailable", "本地 Embedding 服务不可用") from exc

    if settings.embedding_backend == "ollama":
        vectors = data.get("embeddings") or []
    else:
        items = sorted(data.get("data") or [], key=lambda x: x.get("index", 0))
        vectors = [it.get("embedding") or [] for it in items]

    if len(vectors) != len(texts):
        raise AppError(502, "embedding_count_mismatch", "Embedding 返回向量数与请求文本数不一致")

    results: list[list[float]] = []
    for vec in vectors:
        float_vec = [float(x) for x in vec]
        if len(float_vec) != settings.embedding_dimension:
            raise AppError(
                500,
                "embedding_dimension_mismatch",
                "Embedding 向量维度与配置不一致",
            )
        results.append(float_vec)
    return results
