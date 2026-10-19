"""文本规范化、中文分词估算与格式识别辅助。"""

from __future__ import annotations

import re
from pathlib import Path

from localpatentai.core.errors import AppError
from localpatentai.core.security import neutralize_untrusted_document_text

ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

MULTI_BLANK_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t\f\v]+")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def validate_file_type_and_magic(filename: str, content: bytes) -> tuple[str, str]:
    """校验文件扩展名与实际文件头（Magic Bytes），返回 (ext_type, mime_type)。"""
    if not content:
        raise AppError(400, "empty_file", "上传文件内容不能为空")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise AppError(
            400,
            "unsupported_file_type",
            "仅支持上传 PDF、DOCX、TXT 或 Markdown 格式的文档",
        )
    mime = ALLOWED_EXTENSIONS[suffix]
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise AppError(400, "invalid_pdf_magic", "文件内容与 PDF 格式签名不一致")
        return "pdf", mime
    if suffix == ".docx":
        if not content.startswith(b"PK\x03\x04"):
            raise AppError(400, "invalid_docx_magic", "文件内容与 DOCX 格式签名不一致")
        return "docx", mime
    # TXT / Markdown 校验 UTF-8 / GB18030 可解码且不含二进制空字节
    if b"\x00" in content[:4096]:
        raise AppError(400, "binary_in_text_file", "文本文件中检测到非法二进制字节")
    ext_type = "md" if suffix in {".md", ".markdown"} else "txt"
    return ext_type, mime


def decode_text_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AppError(422, "text_decode_failed", "文本文件编码无法解析（需 UTF-8 或 GB18030）")


def normalize_text(raw: str) -> tuple[str, list[str]]:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_BLANK_RE.sub("\n\n", text).strip()
    cleaned, injection_detected = neutralize_untrusted_document_text(text)
    flags: list[str] = []
    if injection_detected:
        flags.append("prompt_injection_neutralized")
    return cleaned, flags


def estimate_tokens(text: str) -> int:
    """估算中英文混合文本的 token 数量（1 个汉字约 1 token，1 个英文词约 1.3 token）。"""
    cjk_count = len(CJK_CHAR_RE.findall(text))
    word_count = len(WORD_TOKEN_RE.findall(text))
    other_chars = max(0, len(text) - cjk_count - sum(len(w) for w in WORD_TOKEN_RE.findall(text)))
    return max(1, int(cjk_count + word_count * 1.3 + other_chars * 0.3))
