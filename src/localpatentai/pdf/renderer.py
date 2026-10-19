"""基于 Jinja2 + PyMuPDF Story/DocumentWriter 的本地中文 PDF 渲染器（规格第 9 章）。"""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path

import fitz
from jinja2 import Environment, FileSystemLoader, select_autoescape

from localpatentai.core.errors import AppError
from localpatentai.drafting.schemas import PatentDisclosureDraft, QualityReviewResult
from localpatentai.knowledge.models import EvidenceItem

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

DEFAULT_PDF_CSS = """
body {
    font-family: sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
}
h1 {
    font-size: 18pt;
    color: #1f4e79;
    margin-bottom: 6pt;
}
h2 {
    font-size: 13.5pt;
    color: #1f4e79;
    margin-top: 12pt;
    margin-bottom: 4pt;
}
h3 {
    font-size: 11.5pt;
    color: #2b579a;
    margin-top: 8pt;
    margin-bottom: 3pt;
}
p {
    margin-top: 3pt;
    margin-bottom: 5pt;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 6pt;
    margin-bottom: 8pt;
    font-size: 9pt;
}
th {
    background-color: #e8eef5;
    font-weight: bold;
}
"""


def render_disclosure_html(
    draft: PatentDisclosureDraft,
    *,
    evidence_items: list[EvidenceItem],
    review_result: QualityReviewResult | None = None,
    workflow_version: str = "v1.0",
    include_evidence_appendix: bool = True,
    include_review_appendix: bool = True,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("patent_disclosure.html.j2")
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return template.render(
        draft=draft,
        evidence_items=evidence_items,
        review_result=review_result,
        workflow_version=workflow_version,
        generated_at=generated_at,
        include_evidence_appendix=include_evidence_appendix,
        include_review_appendix=include_review_appendix,
    )


def render_pdf_bytes_from_draft(
    draft: PatentDisclosureDraft,
    *,
    evidence_items: list[EvidenceItem],
    review_result: QualityReviewResult | None = None,
    workflow_version: str = "v1.0",
    include_evidence_appendix: bool = True,
    include_review_appendix: bool = True,
) -> tuple[bytes, str]:
    """
    将已通过 Schema 与引用校验的 PatentDisclosureDraft 渲染为多页 A4 中文 PDF，
    并在每页页脚写入页码，返回 (pdf_bytes, sha256_hex)。
    """
    html_str = render_disclosure_html(
        draft,
        evidence_items=evidence_items,
        review_result=review_result,
        workflow_version=workflow_version,
        include_evidence_appendix=include_evidence_appendix,
        include_review_appendix=include_review_appendix,
    )

    try:
        story = fitz.Story(html=html_str, user_css=DEFAULT_PDF_CSS)
        buffer = io.BytesIO()
        writer = fitz.DocumentWriter(buffer)
        mediabox = fitz.paper_rect("a4")
        content_rect = mediabox + (48, 48, -48, -54)
        more = 1
        page_guard = 0
        while more:
            page_guard += 1
            if page_guard > 200:
                break
            device = writer.begin_page(mediabox)
            more, _ = story.place(content_rect)
            story.draw(device)
            writer.end_page()
        writer.close()

        raw_pdf = buffer.getvalue()
        doc = fitz.open(stream=raw_pdf, filetype="pdf")
        total_pages = doc.page_count
        for idx in range(total_pages):
            page = doc[idx]
            footer_text = f"LocalPatentAI 专利技术交底书草案 | 第 {idx + 1} / {total_pages} 页"
            page.insert_text(
                (48, mediabox.height - 24),
                footer_text,
                fontname="china-s",
                fontsize=8.5,
            )
        final_bytes = doc.tobytes()
        doc.close()
    except Exception as exc:
        raise AppError(500, "pdf_render_failed", f"PDF 渲染失败: {type(exc).__name__}") from exc

    sha256_hex = hashlib.sha256(final_bytes).hexdigest()
    return final_bytes, sha256_hex
