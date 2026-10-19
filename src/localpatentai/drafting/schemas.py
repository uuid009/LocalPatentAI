"""结构化专利技术交底书、权利要求书草案、技术分析与质量审查 Pydantic Schema（规格第 7、8、9 章）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

REQUIRED_DISCLOSURE_SECTION_TITLES: tuple[str, ...] = (
    "技术领域",
    "背景技术",
    "待解决的技术问题",
    "技术方案",
    "有益效果",
    "附图说明",
    "具体实施方式",
)


class FactWithEvidence(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    support_status: Literal[
        "supported", "partially_supported", "unsupported", "needs_confirmation"
    ] = "supported"
    is_experimental: bool = False


class TechnicalSolutionAnalysis(BaseModel):
    technical_field: str
    prior_art_problems: list[str] = Field(default_factory=list)
    core_technical_features: list[FactWithEvidence] = Field(default_factory=list)
    implementation_steps: list[FactWithEvidence] = Field(default_factory=list)
    experimental_data: list[FactWithEvidence] = Field(default_factory=list)
    technical_effects: list[FactWithEvidence] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class EvidenceGapItem(BaseModel):
    gap_id: str
    aspect: str
    description: str
    severity: Literal["blocking", "warning", "info"] = "warning"
    requires_human_confirmation: bool = True


class EvidenceResearchReport(BaseModel):
    supported_feature_count: int = 0
    unsupported_feature_count: int = 0
    prior_art_clues: list[str] = Field(default_factory=list)
    missing_data_checklist: list[EvidenceGapItem] = Field(default_factory=list)
    human_confirmation_questions: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "本证据研究报告仅基于已获准访问的项目私有与共享文献做技术线索辅助比对，"
        "不构成正式专利新颖性、创造性或授权前景的法律审查意见。"
    )


class PatentSectionDraft(BaseModel):
    section_key: str
    title: str
    content: str
    evidence_ids: list[str] = Field(default_factory=list)
    support_status: Literal[
        "supported", "partially_supported", "unsupported", "needs_confirmation"
    ] = "supported"
    needs_confirmation: bool = False
    confirmation_notes: list[str] = Field(default_factory=list)
    version: int = 1


class ClaimDraftItem(BaseModel):
    claim_number: int = Field(ge=1)
    claim_type: Literal["independent", "dependent"] = "independent"
    depends_on: list[int] = Field(default_factory=list)
    preamble: str
    characterizing_portion: str
    full_text: str
    evidence_ids: list[str] = Field(default_factory=list)
    needs_confirmation: bool = False


class PatentDisclosureDraft(BaseModel):
    title: str
    version: int = 1
    generation_mode: Literal["local_llm", "mock"] = "local_llm"
    sections: list[PatentSectionDraft] = Field(default_factory=list)
    claims: list[ClaimDraftItem] = Field(default_factory=list)
    abstract_text: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)
    terminology_glossary: dict[str, str] = Field(default_factory=dict)
    disclaimer: str = (
        "本文件为辅助生成的专利技术交底书及权利要求书草案，仅供参考与内部评审，"
        "不可未经代理师/发明人复核直接向专利主管机关提交。"
    )


class ReviewIssue(BaseModel):
    issue_id: str
    category: Literal[
        "missing_section",
        "terminology_inconsistency",
        "unit_inconsistency",
        "unsupported_effect",
        "invalid_citation",
        "claim_structure_error",
        "unresolved_placeholder",
        "hallucinated_data",
    ]
    severity: Literal["blocking", "warning", "info"] = "warning"
    location: str
    description: str
    auto_fixable: bool = True


class QualityReviewResult(BaseModel):
    passed: bool
    blocking_count: int = 0
    warning_count: int = 0
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str = ""
    deterministic_checks_passed: bool = True
    llm_review_notes: str = ""
