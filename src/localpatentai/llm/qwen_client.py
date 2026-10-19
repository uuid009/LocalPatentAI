"""本地 Qwen + vLLM 推理客户端（规格 4.2 节）。只连接配置指定的本地 endpoint，禁止外部回退。"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from localpatentai.core.config import Settings, get_settings
from localpatentai.core.errors import AppError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def normalize_openai_base_url(url: str) -> str:
    parsed = urlparse(url)
    base = url.rstrip("/")
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1"):
        return base
    return f"{base}/v1"


def probe_vllm_service(settings: Settings | None = None) -> dict:
    runtime = settings or get_settings()
    bge_emb_path = runtime.resolve_path(runtime.embedding_model_path)
    bge_rerank_path = runtime.resolve_path(runtime.reranker_model_path)

    status = {
        "llm_provider": runtime.llm_provider,
        "llm_model": runtime.llm_model,
        "llm_ready": False,
        "embedding_backend": runtime.embedding_backend,
        "embedding_model_exists": bge_emb_path.exists(),
        "reranker_backend": runtime.reranker_backend,
        "reranker_model_exists": bge_rerank_path.exists(),
        "detail": "not_checked",
    }
    if runtime.llm_mock:
        status["detail"] = "mock_mode"
        return status

    models_url = (
        f"{runtime.llm_base_url.rstrip('/')}/api/tags"
        if runtime.llm_provider == "ollama"
        else f"{normalize_openai_base_url(runtime.llm_base_url)}/models"
    )
    try:
        headers = {"Authorization": f"Bearer {runtime.llm_api_key}"}
        resp = httpx.get(models_url, headers=headers, timeout=1.5)
        resp.raise_for_status()
        status["llm_ready"] = True
        status["detail"] = "ok"
    except Exception as exc:
        status["llm_ready"] = False
        status["detail"] = type(exc).__name__
    return status


class QwenClient:
    """面向本地 vLLM + Qwen 的受控推理客户端。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> str:
        if self.settings.llm_mock:
            return '{"status": "mock", "note": "Mock LLM response"}'

        url = f"{normalize_openai_base_url(self.settings.llm_base_url)}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        payload: dict = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.settings.llm_provider == "openai_compatible":
            payload["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking,
                "preserve_thinking": False,
            }

        last_exc: Exception | None = None
        for _attempt in range(self.settings.llm_max_retries + 1):
            try:
                resp = httpx.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=float(self.settings.llm_timeout_seconds),
                )
                resp.raise_for_status()
                body = resp.json()
                choices = body.get("choices") or []
                if not choices:
                    raise AppError(502, "llm_empty_choices", "本地 vLLM 返回空 choices")
                content = (choices[0].get("message") or {}).get("content") or ""
                if not str(content).strip():
                    raise AppError(502, "llm_empty_content", "本地 vLLM 返回空文本内容")
                return str(content)
            except AppError:
                raise
            except Exception as exc:
                last_exc = exc

        raise AppError(
            503,
            "llm_service_unavailable",
            "本地 Qwen/vLLM 推理服务不可用或超时，已拒绝回退外部 API",
            {"error_type": type(last_exc).__name__ if last_exc else "UnknownError"},
        )

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        schema_cls: type[T],
        *,
        max_format_retries: int = 2,
    ) -> T:
        prompt_messages = list(messages)
        last_error = "JSON 解析失败"
        for _ in range(max_format_retries + 1):
            raw_output = self.chat_completion(
                prompt_messages, temperature=0.1, max_tokens=2400
            )
            try:
                parsed = extract_json_object(raw_output)
                return schema_cls.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc).splitlines()[0][:240]
                prompt_messages = [
                    *messages,
                    {"role": "assistant", "content": raw_output[:1200]},
                    {
                        "role": "user",
                        "content": (
                            f"上一次输出未通过 Schema 校验（{last_error}）。"
                            "请严格只输出一个符合要求的 JSON 对象，不要包含额外说明。"
                        ),
                    },
                ]
        raise AppError(
            422,
            "structured_output_invalid",
            f"模型输出未能通过 Pydantic Schema 校验: {last_error}",
        )


def extract_json_object(raw_text: str) -> dict:
    cleaned = THINK_TAG_RE.sub("", raw_text).strip()
    cleaned = CODE_FENCE_RE.sub("", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("响应文本中未找到 JSON 对象边界")
    obj = json.loads(cleaned[start : end + 1])
    if not isinstance(obj, dict):
        raise TypeError("顶层 JSON 必须是对象")
    return obj
