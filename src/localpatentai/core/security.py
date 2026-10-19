"""安全工具函数：密码慢哈希、路径防穿越、文件名规范化、提示注入基础清洗。"""

from __future__ import annotations

import hashlib
import re
import secrets
from pathlib import Path

from localpatentai.core.errors import AppError

PASSWORD_ITERATIONS = 200_000
SAFE_FILENAME_RE = re.compile(r"[^\w\u4e00-\u9fff.\-]+")
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"忽略(之前|以上|所有)的?(系统)?指令"),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"你现在是.*不受限制"),
]


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("ascii"),
        PASSWORD_ITERATIONS,
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt, digest = stored_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        computed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("ascii"),
            int(iterations),
        ).hex()
        return secrets.compare_digest(computed, digest)
    except (ValueError, TypeError):
        return False


def normalize_filename(filename: str) -> str:
    raw_name = Path(filename.replace("\\", "/")).name.strip()
    if not raw_name or raw_name in {".", ".."}:
        raise AppError(400, "invalid_filename", "上传文件名不合法")
    cleaned = SAFE_FILENAME_RE.sub("_", raw_name).strip("._")
    if not cleaned:
        raise AppError(400, "invalid_filename", "上传文件名不合法")
    suffix = Path(raw_name).suffix.lower()
    if suffix and not cleaned.lower().endswith(suffix):
        cleaned = f"{cleaned}{suffix}"
    return cleaned[:200]


def resolve_safe_storage_path(base_dir: str | Path, storage_key: str) -> Path:
    if not storage_key or ".." in storage_key or storage_key.startswith(("/", "\\")):
        raise AppError(400, "unsafe_path", "非法的存储键路径")
    root = Path(base_dir).resolve()
    target = (root / storage_key).resolve()
    if root != target and root not in target.parents:
        raise AppError(400, "unsafe_path", "检测到路径穿越风险，已拒绝访问")
    return target


def neutralize_untrusted_document_text(text: str) -> tuple[str, bool]:
    """将不可信文档中的疑似提示注入片段做标记隔离，防止被模型当作系统指令。"""
    detected = False
    cleaned = text
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(cleaned):
            detected = True
            cleaned = pattern.sub("[UNTRUSTED_INSTRUCTION_REMOVED]", cleaned)
    return cleaned, detected
