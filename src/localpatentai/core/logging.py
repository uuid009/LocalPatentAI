"""日志配置与敏感内容脱敏过滤器。确保日志不泄露文档全文、Prompt、口令或敏感路径。"""

from __future__ import annotations

import logging
import re
from typing import Any

SENSITIVE_PATTERNS = [
    (re.compile(r"(postgresql(?:\+psycopg)?://[^:]+:)([^@]+)(@)"), r"\1[REDACTED]\3"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+"), r"\1[REDACTED]"),
    (re.compile(r"(password\s*=\s*)[^\s,;]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(token\s*=\s*)[^\s,;]+", re.IGNORECASE), r"\1[REDACTED]"),
]

FORBIDDEN_LOG_KEYS = frozenset(
    {
        "content",
        "raw_content",
        "document_text",
        "prompt",
        "full_prompt",
        "messages",
        "model_output",
        "password",
        "password_hash",
        "token",
        "database_url",
    }
)


def mask_sensitive_text(text: str, max_length: int = 500) -> str:
    masked = text
    for pattern, repl in SENSITIVE_PATTERNS:
        masked = pattern.sub(repl, masked)
    if len(masked) > max_length:
        return masked[:max_length] + "...[TRUNCATED]"
    return masked


def sanitize_log_dict(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in FORBIDDEN_LOG_KEYS:
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_dict(value)
        elif isinstance(value, str):
            sanitized[key] = mask_sensitive_text(value)
        else:
            sanitized[key] = value
    return sanitized


class SanitizingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_sensitive_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = sanitize_log_dict(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    mask_sensitive_text(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
        return True


def configure_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger("localpatentai")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not any(isinstance(f, SanitizingFilter) for f in root_logger.filters):
        root_logger.addFilter(SanitizingFilter())
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(SanitizingFilter())
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
