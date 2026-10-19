"""统一异常定义与安全错误响应（避免在 API 错误中泄漏跨组资源名、敏感路径或数据库凭证）。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


class AuthorizationError(AppError):
    def __init__(
        self,
        code: str = "forbidden",
        message: str = "无权访问该资源或资源不存在",
        status_code: int = 403,
    ) -> None:
        super().__init__(status_code=status_code, code=code, message=message)


class AuthenticationError(AppError):
    def __init__(
        self,
        code: str = "unauthenticated",
        message: str = "未认证或会话已失效，请重新登录",
    ) -> None:
        super().__init__(status_code=401, code=code, message=message)


class ResourceNotFoundError(AppError):
    def __init__(
        self,
        code: str = "not_found",
        message: str = "请求的资源不存在或无权访问",
    ) -> None:
        super().__init__(status_code=404, code=code, message=message)


def error_response_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": _sanitize_details(details or {}),
        }
    }


def _sanitize_details(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            key_str = str(k)
            if any(
                secret in key_str.lower()
                for secret in ("password", "secret", "token", "storage_key", "raw_text", "prompt")
            ):
                cleaned[key_str] = "[REDACTED]"
            else:
                cleaned[key_str] = _sanitize_details(v)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_sanitize_details(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response_body(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "请求无法完成"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response_body("http_error", detail),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_response_body(
                "validation_error",
                "请求参数校验失败",
                {"errors": _sanitize_details(exc.errors())},
            ),
        )
