"""Output Validator（规格 7.3 & 8.1）：校验草案 JSON Schema、必需章节、授权证据引用与 PDF 导出前置条件。"""

from __future__ import annotations

from localpatentai.core.errors import AppError
from localpatentai.drafting.schemas import (
    REQUIRED_DISCLOSURE_SECTION_TITLES,
    PatentDisclosureDraft,
    QualityReviewResult,
)
from localpatentai.drafting.service import review_patent_draft
from localpatentai.knowledge.models import EvidenceItem


def validate_draft_output(
    draft_dict: dict,
    authorized_evidence: list[EvidenceItem],
    authorized_space_ids: list[str],
) -> tuple[PatentDisclosureDraft, QualityReviewResult]:
    """
    1. 验证 Pydantic Schema；
    2. 验证引用的证据 ID 不仅存在于当前 Evidence Pack，且其 knowledge_space_id 属于当前 authorized_space_ids；
    3. 执行确定性质量审查。
    """
    draft = PatentDisclosureDraft.model_validate(draft_dict)
    allowed_spaces = set(authorized_space_ids)
    filtered_evidence = [
        ev for ev in authorized_evidence if ev.knowledge_space_id in allowed_spaces
    ]
    review = review_patent_draft(draft, filtered_evidence)
    return draft, review


def assert_ready_for_pdf_generation(
    draft: PatentDisclosureDraft,
    review: QualityReviewResult,
    human_approval: str,
) -> None:
    if human_approval != "approved":
        raise AppError(
            409,
            "human_approval_required",
            "草案尚未通过人工审批，禁止生成正式 PDF 产物",
        )
    section_titles = {s.title.strip() for s in draft.sections}
    missing = [t for t in REQUIRED_DISCLOSURE_SECTION_TITLES if t not in section_titles]
    if missing:
        raise AppError(
            422,
            "missing_required_sections",
            f"草案缺少必需章节：{', '.join(missing)}，拒绝渲染 PDF",
        )
    if review.blocking_count > 0:
        raise AppError(
            422,
            "blocking_review_issues_unresolved",
            f"草案仍存在 {review.blocking_count} 项阻断级质量问题，拒绝渲染 PDF",
        )
