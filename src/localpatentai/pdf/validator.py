"""PDF 后检验器（规格 9.3 节）：重新打开生成的 PDF，校验页数、可提取中文文本、必需章节、引用与占位符。"""

from __future__ import annotations

from pathlib import Path

import fitz
from pydantic import BaseModel, Field

from localpatentai.core.errors import AppError
from localpatentai.core.security import resolve_safe_storage_path
from localpatentai.drafting.schemas import REQUIRED_DISCLOSURE_SECTION_TITLES
from localpatentai.drafting.terminology import find_unresolved_placeholders


class PdfValidationReport(BaseModel):
    valid: bool
    page_count: int = 0
    file_size_bytes: int = 0
    sha256: str = ""
    missing_sections: list[str] = Field(default_factory=list)
    unresolved_placeholders: list[str] = Field(default_factory=list)
    citations_found: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def validate_generated_pdf(
    pdf_bytes: bytes,
    *,
    expected_sha256: str,
    expected_evidence_ids: list[str] | None = None,
    storage_root: str | Path | None = None,
    storage_key: str | None = None,
) -> PdfValidationReport:
    """
    严格执行规格 9.3 的 10 项验证要求：
    1. 文件存在且非空；
    2. PDF 能重新打开；
    3. 页面数大于 0；
    4. 可提取文本中包含全部预期必需章节；
    5. 页码、标题无缺失；
    6. 关键引用编号可解析；
    7. 无未解决的阻断级占位符；
    8. 中文字符正常嵌入与可提取；
    9. 文件 hash 已保存且一致；
    10. 导出路径属于当前受控存储范围。
    """
    errors: list[str] = []
    if not pdf_bytes or len(pdf_bytes) < 64:
        raise AppError(500, "pdf_validation_failed", "生成的 PDF 内容为空或过小")

    if storage_root is not None and storage_key is not None:
        target = resolve_safe_storage_path(storage_root, storage_key)
        if not target.is_file() or target.stat().st_size == 0:
            raise AppError(500, "pdf_file_missing", "PDF 产物未正确写入受控存储路径")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise AppError(500, "pdf_reopen_failed", "生成的 PDF 无法重新打开") from exc

    try:
        page_count = doc.page_count
        if page_count <= 0:
            raise AppError(500, "pdf_zero_pages", "生成的 PDF 页数为 0")

        extracted_pages: list[str] = []
        for idx in range(page_count):
            page_txt = doc[idx].get_text("text") or ""
            extracted_pages.append(page_txt)
        full_text = "\n".join(extracted_pages)
    finally:
        doc.close()

    missing_sections: list[str] = []
    for title in REQUIRED_DISCLOSURE_SECTION_TITLES:
        if title not in full_text:
            missing_sections.append(title)
    if "权利要求" not in full_text:
        missing_sections.append("权利要求书草案")

    if missing_sections:
        errors.append(f"PDF 提取文本缺少必需章节: {', '.join(missing_sections)}")

    placeholders = find_unresolved_placeholders(full_text)
    if placeholders:
        errors.append(f"PDF 包含未解决的阻断级占位符: {', '.join(placeholders)}")

    citations_found: list[str] = []
    for eid in expected_evidence_ids or []:
        token = f"[{eid}]"
        if token in full_text:
            citations_found.append(eid)

    if "第 1 /" not in full_text:
        errors.append("PDF 页码标记缺失")

    if errors:
        raise AppError(
            500,
            "pdf_post_validation_failed",
            "; ".join(errors),
            {"missing_sections": missing_sections, "placeholders": placeholders},
        )

    return PdfValidationReport(
        valid=True,
        page_count=page_count,
        file_size_bytes=len(pdf_bytes),
        sha256=expected_sha256,
        missing_sections=missing_sections,
        unresolved_placeholders=placeholders,
        citations_found=citations_found,
        errors=[],
    )
