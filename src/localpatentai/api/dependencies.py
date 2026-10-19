"""FastAPI 依赖项注入（从 Bearer Token 或 X-User-Id 认证头识别可信用户身份）。"""

from __future__ import annotations

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from localpatentai.auth.service import authenticate_token
from localpatentai.core.errors import AuthenticationError
from localpatentai.db.models import User
from localpatentai.db.session import get_db


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization:
        raise AuthenticationError()
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthenticationError(
            code="invalid_auth_header",
            message="Authorization 头格式必须为 Bearer <token>",
        )
    token = parts[1].strip()
    return authenticate_token(db, token)
