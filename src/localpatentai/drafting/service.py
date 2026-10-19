"""专利交底书结构化生成与确定性规则修订辅助服务。"""

from __future__ import annotations

import re

from localpatentai.core.config import Settings, get_settings
from localpatentai.drafting.schemas import (
     REQUIRED_DISCLOSURE_SECTION_TITLES,
    ClaimDraftItem,
    EvidenceGapItem,
    EvidenceResearchReport,
    FactWithEvidence,
    PatentDisclosureDraft,
    PatentSectionDraft,
    QualityReviewResult,
    ReviewIssue,
    TechnicalSolutionAnalysis,
)
from localpatentai.drafting.terminology import (
    UNVERIFIED_NUMBER_RE,
    check_terminology_consistency,
    check_unit_consistency,
    find_unresolved_placeholders,
)
from localpatentai.knowledge.models import EvidenceItem
from localpatentai.llm.qwen_client import QwenClient

CLAIM_NUM_IN_TEXT_RE = re.compile(r"权利要求\s*(\d+)")


def build_technical_analysis(
    *,
    task_title: str,
    user_prompt: str,
    evidence_items: list[EvidenceItem],
    settings: Settings | None = None,
) -> TechnicalSolutionAnalysis:
    runtime = settings or get_settings()
    internal_evidence = [it for it in evidence_items if it.evidence_type != "external_prior_art"]
    prior_art_evidence = [it for it in evidence_items if it.evidence_type == "external_prior_art"]

    if not runtime.llm_mock:
        client = QwenClient(runtime)
        ev_summary = "\n".join(
            f"[{it.evidence_id}] ({it.evidence_type}, p.{it.page_start}): {it.quote[:300]}"
            for it in evidence_items[:10]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是专利技术交底书技术分析节点。严格根据给定的证据片段提取技术领域、现有技术问题、"
                    "核心技术特征、实施步骤、实验数据与技术效果。严禁捏造证据中没有的实验数值；"
                    "缺少证据支撑的内容必须将 support_status 设为 needs_confirmation。"
                    "只输出符合 TechnicalSolutionAnalysis 的 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": f"专利主题：{task_title}\n任务说明：{user_prompt}\n已授权证据：\n{ev_summary}",
            },
        ]
        try:
            return client.generate_structured(messages, TechnicalSolutionAnalysis)
        except Exception:
            pass

    # 确定性提取（用于本地 mock 模式或当真实模型未启动时的受控基于证据构建，绝不捏造无证据参数）
    features: list[FactWithEvidence] = []
    steps: list[FactWithEvidence] = []
    experiments: list[FactWithEvidence] = []
    effects: list[FactWithEvidence] = []
    unresolved: list[str] = []

    for it in internal_evidence:
        snippet = it.quote.strip().replace("\n", " ")[:180]
        if it.evidence_type == "experimental_observation":
            experiments.append(
                FactWithEvidence(
                    statement=f"根据实验记录记载：{snippet}",
                    evidence_ids=[it.evidence_id],
                    support_status=it.support_status,
                    is_experimental=True,
                )
            )
            effects.append(
                FactWithEvidence(
                    statement=f"实验验证的技术效果：{snippet}",
                    evidence_ids=[it.evidence_id],
                    support_status=it.support_status,
                    is_experimental=True,
                )
            )
        else:
            features.append(
                FactWithEvidence(
                    statement=f"核心技术特征：{snippet}",
                    evidence_ids=[it.evidence_id],
                    support_status=it.support_status,
                    is_experimental=False,
                )
            )
            steps.append(
                FactWithEvidence(
                    statement=f"实施步骤要点：{snippet}",
                    evidence_ids=[it.evidence_id],
                    support_status=it.support_status,
                    is_experimental=False,
                )
            )

    if not features:
        features.append(
            FactWithEvidence(
                statement=f"【待确认】围绕“{task_title}”的具体结构连接与控制逻辑参数尚未提供原文证据，需发明人补充确认。",
                evidence_ids=[],
                support_status="needs_confirmation",
                is_experimental=False,
            )
        )
        unresolved.append("缺少核心技术特征的详细参数与结构说明文档")

    if not experiments:
        experiments.append(
            FactWithEvidence(
                statement="【待确认】当前项目材料中未包含完整实验对比数据表，禁止自动补造实验数值，请发明人补充实测记录。",
                evidence_ids=[],
                support_status="needs_confirmation",
                is_experimental=True,
            )
        )
        unresolved.append("缺少具体实施例的实测数据或对照组记录")

    if not effects:
        effects.append(
            FactWithEvidence(
                statement="【待确认】有益效果的量化指标待根据补充的实验记录进一步核实。",
                evidence_ids=features[0].evidence_ids,
                support_status="needs_confirmation",
                is_experimental=False,
            )
        )

    prior_problems = [
        f"现有技术文献（[{it.evidence_id}]）指出的局限：{it.quote[:120]}"
        for it in prior_art_evidence
    ] or [f"针对“{task_title}”领域的现有方案在稳定性、效率或工程适应性方面存在不足。"]

    return TechnicalSolutionAnalysis(
        technical_field=f"本发明属于{task_title}相关技术领域，具体涉及：{user_prompt[:100] or task_title}。",
        prior_art_problems=prior_problems,
        core_technical_features=features,
        implementation_steps=steps or features,
        experimental_data=experiments,
        technical_effects=effects,
        unresolved_questions=unresolved,
    )


def evaluate_evidence_research(
    analysis: TechnicalSolutionAnalysis,
    evidence_items: list[EvidenceItem],
) -> EvidenceResearchReport:
    all_facts = (
        analysis.core_technical_features
        + analysis.implementation_steps
        + analysis.experimental_data
        + analysis.technical_effects
    )
    supported_count = sum(
        1 for f in all_facts if f.evidence_ids and f.support_status == "supported"
    )
    unsupported_count = sum(
        1 for f in all_facts if not f.evidence_ids or f.support_status != "supported"
    )

    gaps: list[EvidenceGapItem] = []
    questions: list[str] = list(analysis.unresolved_questions)

    if not evidence_items:
        gaps.append(
            EvidenceGapItem(
                gap_id="GAP-1",
                aspect="全量项目证据缺失",
                description="当前已授权知识空间内未检索到任何匹配证据片段，所有核心技术主张均处于待确认状态。",
                severity="blocking",
                requires_human_confirmation=True,
            )
        )
        questions.append("请上传或确认本专利的核心技术方案说明或实验记录。")

    has_exp = any(it.evidence_type == "experimental_observation" for it in evidence_items)
    if not has_exp:
        gaps.append(
            EvidenceGapItem(
                gap_id=f"GAP-{len(gaps) + 1}",
                aspect="实验与量化效果数据",
                description="未检索到实验观测数据（experimental_observation），有益效果章节将标记为 needs_confirmation，不虚构量化指标。",
                severity="warning",
                requires_human_confirmation=True,
            )
        )
        questions.append("是否补充具体实验测试数据以支撑有益效果？")

    prior_clues = [
        f"[{it.evidence_id}] {it.document_name}: {it.quote[:120]}"
        for it in evidence_items
        if it.evidence_type == "external_prior_art"
    ]

    return EvidenceResearchReport(
        supported_feature_count=supported_count,
        unsupported_feature_count=unsupported_count,
        prior_art_clues=prior_clues,
        missing_data_checklist=gaps,
        human_confirmation_questions=list(dict.fromkeys(questions)),
    )


def draft_patent_disclosure(
    *,
    task_title: str,
    user_prompt: str,
    analysis: TechnicalSolutionAnalysis,
    evidence_items: list[EvidenceItem],
    version: int = 1,
    settings: Settings | None = None,
) -> PatentDisclosureDraft:
    runtime = settings or get_settings()
    mode = "mock" if runtime.llm_mock else "local_llm"

    valid_ev_ids = [it.evidence_id for it in evidence_items]
    internal_ev_ids = [
        it.evidence_id for it in evidence_items if it.evidence_type != "external_prior_art"
    ]
    prior_ev_ids = [
        it.evidence_id for it in evidence_items if it.evidence_type == "external_prior_art"
    ]
    primary_ev_ids = internal_ev_ids or valid_ev_ids

    has_sufficient_ev = len(primary_ev_ids) > 0
    default_status = "supported" if has_sufficient_ev else "needs_confirmation"

    feat_lines = "\n".join(
        f"（{i}）{f.statement}"
        + (f" [{' '.join(f.evidence_ids)}]" if f.evidence_ids else " 【待确认】")
        for i, f in enumerate(analysis.core_technical_features, start=1)
    )
    step_lines = "\n".join(
        f"步骤 S{i}：{s.statement}"
        + (f" [{' '.join(s.evidence_ids)}]" if s.evidence_ids else " 【待确认】")
        for i, s in enumerate(analysis.implementation_steps, start=1)
    )
    exp_lines = "\n".join(
        f"- {e.statement}"
        + (f" [{' '.join(e.evidence_ids)}]" if e.evidence_ids else " 【待确认】")
        for e in analysis.experimental_data
    )
    eff_lines = "\n".join(
        f"（{i}）{ef.statement}"
        + (f" [{' '.join(ef.evidence_ids)}]" if ef.evidence_ids else " 【待确认】")
        for i, ef in enumerate(analysis.technical_effects, start=1)
    )

    exp_needs_confirm = any(
        e.support_status != "supported" or not e.evidence_ids for e in analysis.experimental_data
    )

    sections: list[PatentSectionDraft] = [
        PatentSectionDraft(
            section_key="technical_field",
            title="技术领域",
            content=analysis.technical_field,
            evidence_ids=primary_ev_ids[:2],
            support_status=default_status,  # type: ignore[arg-type]
            needs_confirmation=not has_sufficient_ev,
            version=version,
        ),
        PatentSectionDraft(
            section_key="background_art",
            title="背景技术",
            content=(
                "在现有技术中，相关方案存在以下技术局限：\n"
                + "\n".join(f"- {p}" for p in analysis.prior_art_problems)
                + "\n上述外部现有技术或背景描述仅作为技术背景说明，不作为本发明的创新特征主张。"
            ),
            evidence_ids=prior_ev_ids or primary_ev_ids[:1],
            support_status=default_status,  # type: ignore[arg-type]
            needs_confirmation=not has_sufficient_ev,
            version=version,
        ),
        PatentSectionDraft(
            section_key="technical_problem",
            title="待解决的技术问题",
            content=(
                f"本发明旨在针对上述现有技术缺陷，提供一种{task_title}，"
                f"以解决：{'; '.join(analysis.prior_art_problems)}。"
            ),
            evidence_ids=primary_ev_ids[:2],
            support_status=default_status,  # type: ignore[arg-type]
            needs_confirmation=not has_sufficient_ev,
            version=version,
        ),
        PatentSectionDraft(
            section_key="technical_solution",
            title="技术方案",
            content=(
                f"为实现上述发明目的，本发明提供的“{task_title}”主要包括以下核心技术特征与执行步骤：\n"
                f"一、核心技术特征：\n{feat_lines}\n"
                f"二、处理流程：\n{step_lines}"
            ),
            evidence_ids=primary_ev_ids[:4],
            support_status=default_status,  # type: ignore[arg-type]
            needs_confirmation=not has_sufficient_ev,
            version=version,
        ),
        PatentSectionDraft(
            section_key="beneficial_effects",
            title="有益效果",
            content=(
                "与现有技术相比，本发明技术方案带来的有益效果如下：\n"
                f"{eff_lines}\n"
                "（注：所有量化技术效果严格基于已授权实验记录提取；若标记为【待确认】，需发明人核实后方可定稿。）"
            ),
            evidence_ids=primary_ev_ids[:3],
            support_status="needs_confirmation" if exp_needs_confirm else "supported",
            needs_confirmation=exp_needs_confirm,
            confirmation_notes=["请核实有益效果中的实验条件与对比基准是否完整"]
            if exp_needs_confirm
            else [],
            version=version,
        ),
        PatentSectionDraft(
            section_key="brief_description_of_drawings",
            title="附图说明",
            content=(
                f"图 1 为本发明实施例提供的“{task_title}”整体流程示意图；\n"
                f"图 2 为本发明实施例提供的核心模块结构框图。"
            ),
            evidence_ids=primary_ev_ids[:1],
            support_status=default_status,  # type: ignore[arg-type]
            needs_confirmation=False,
            version=version,
        ),
        PatentSectionDraft(
            section_key="detailed_description",
            title="具体实施方式",
            content=(
                "下面结合附图和具体实施例对本发明的技术方案作进一步详细说明：\n"
                f"实施例 1：\n{step_lines}\n"
                f"实验与观测记录：\n{exp_lines}"
            ),
            evidence_ids=primary_ev_ids[:5],
            support_status="needs_confirmation" if exp_needs_confirm else "supported",
            needs_confirmation=exp_needs_confirm,
            version=version,
        ),
    ]

    claims: list[ClaimDraftItem] = [
        ClaimDraftItem(
            claim_number=1,
            claim_type="independent",
            depends_on=[],
            preamble=f"一种{task_title}，其特征在于，包括：",
            characterizing_portion="; ".join(
                f.statement for f in analysis.core_technical_features[:3]
            ),
            full_text=(
                f"1. 一种{task_title}，其特征在于，包括："
                + "；".join(f.statement for f in analysis.core_technical_features[:3])
                + "。"
            ),
            evidence_ids=primary_ev_ids[:3],
            needs_confirmation=not has_sufficient_ev,
        ),
        ClaimDraftItem(
            claim_number=2,
            claim_type="dependent",
            depends_on=[1],
            preamble=f"根据权利要求 1 所述的{task_title}，其特征在于：",
            characterizing_portion="; ".join(
                s.statement for s in analysis.implementation_steps[:2]
            ),
            full_text=(
                f"2. 根据权利要求 1 所述的{task_title}，其特征在于，还包括以下执行步骤："
                + "；".join(s.statement for s in analysis.implementation_steps[:2])
                + "。"
            ),
            evidence_ids=primary_ev_ids[:2],
            needs_confirmation=not has_sufficient_ev,
        ),
        ClaimDraftItem(
            claim_number=3,
            claim_type="dependent",
            depends_on=[1, 2],
            preamble=f"根据权利要求 1 或 2 所述的{task_title}，其特征在于：",
            characterizing_portion=analysis.experimental_data[0].statement
            if analysis.experimental_data
            else "参数范围待发明人确认",
            full_text=(
                f"3. 根据权利要求 1 或 2 所述的{task_title}，其特征在于："
                + (
                    analysis.experimental_data[0].statement
                    if analysis.experimental_data
                    else "【待确认】具体运行参数范围待补充"
                )
                + "。"
            ),
            evidence_ids=primary_ev_ids[:2],
            needs_confirmation=exp_needs_confirm,
        ),
    ]

    abstract_text = (
        f"本发明公开了一种{task_title}，属于技术转化研发领域。"
        f"该方案通过{analysis.core_technical_features[0].statement[:80]}，"
        f"有效解决了{analysis.prior_art_problems[0][:60]}。"
    )

    return PatentDisclosureDraft(
        title=task_title,
        version=version,
        generation_mode=mode,  # type: ignore[arg-type]
        sections=sections,
        claims=claims,
        abstract_text=abstract_text,
        unresolved_questions=list(analysis.unresolved_questions),
        terminology_glossary={task_title: "本发明核心保护主题"},
    )


def review_patent_draft(
    draft: PatentDisclosureDraft,
    evidence_items: list[EvidenceItem],
) -> QualityReviewResult:
    """
    质量审查节点（规格 7.1 Node D）：
    采用“确定性规则校验 + 结构化审查”：
    1. 必需章节是否完整；
    2. 技术术语与参数单位是否一致；
    3. 引用证据 ID 是否真实存在且属于当前授权证据包；
    4. 权利要求编号是否连续、从属权利要求引用关系是否合法（不得自引用或引用后序权利要求）；
    5. 是否存在未解决的阻断级占位符（TODO/TBD/{{...}}）；
    6. 是否有未标注 needs_confirmation 且无证据支撑的量化技术效果。
    """
    issues: list[ReviewIssue] = []
    valid_ev_map = {it.evidence_id: it for it in evidence_items}

    # 1. 检查必需章节
    existing_titles = {sec.title.strip() for sec in draft.sections}
    for req_title in REQUIRED_DISCLOSURE_SECTION_TITLES:
        if req_title not in existing_titles:
            issues.append(
                ReviewIssue(
                    issue_id=f"ISS-{len(issues) + 1}",
                    category="missing_section",
                    severity="blocking",
                    location=f"sections.{req_title}",
                    description=f"缺少专利交底书必需章节：{req_title}",
                    auto_fixable=True,
                )
            )

    if not draft.claims:
        issues.append(
            ReviewIssue(
                issue_id=f"ISS-{len(issues) + 1}",
                category="missing_section",
                severity="blocking",
                location="claims",
                description="缺少权利要求书草案（至少需包含 1 项独立权利要求）",
                auto_fixable=True,
            )
        )

    # 2. 检查章节证据 ID、占位符与无依据量化指标
    full_sections_text = "\n".join(sec.content for sec in draft.sections)
    full_claims_text = "\n".join(c.full_text for c in draft.claims)
    combined_text = f"{full_sections_text}\n{full_claims_text}"

    for sec in draft.sections:
        for eid in sec.evidence_ids:
            if eid not in valid_ev_map:
                issues.append(
                    ReviewIssue(
                        issue_id=f"ISS-{len(issues) + 1}",
                        category="invalid_citation",
                        severity="blocking",
                        location=f"section:{sec.title}",
                        description=f"章节《{sec.title}》引用了不存在或未授权的证据编号 [{eid}]",
                        auto_fixable=True,
                    )
                )
        placeholders = find_unresolved_placeholders(sec.content)
        if placeholders:
            issues.append(
                ReviewIssue(
                    issue_id=f"ISS-{len(issues) + 1}",
                    category="unresolved_placeholder",
                    severity="blocking",
                    location=f"section:{sec.title}",
                    description=f"章节《{sec.title}》包含未解决的模板占位符：{', '.join(placeholders)}",
                    auto_fixable=True,
                )
            )
        # 检查有益效果是否无证据却写死百分比提升
        if (
            sec.section_key == "beneficial_effects"
            and UNVERIFIED_NUMBER_RE.search(sec.content)
            and not sec.evidence_ids
            and not sec.needs_confirmation
        ):
            issues.append(
                ReviewIssue(
                    issue_id=f"ISS-{len(issues) + 1}",
                    category="hallucinated_data",
                    severity="blocking",
                    location=f"section:{sec.title}",
                    description="《有益效果》包含量化百分比断言但未关联任何证据且未标记待确认",
                    auto_fixable=True,
                )
            )

    # 3. 检查权利要求书编号与引用关系
    expected_num = 1
    for claim in draft.claims:
        if claim.claim_number != expected_num:
            issues.append(
                ReviewIssue(
                    issue_id=f"ISS-{len(issues) + 1}",
                    category="claim_structure_error",
                    severity="blocking",
                    location=f"claim:{claim.claim_number}",
                    description=f"权利要求编号不连续，期望 {expected_num}，实际为 {claim.claim_number}",
                    auto_fixable=True,
                )
            )
        expected_num += 1

        if claim.claim_type == "independent" and claim.depends_on:
            issues.append(
                ReviewIssue(
                    issue_id=f"ISS-{len(issues) + 1}",
                    category="claim_structure_error",
                    severity="blocking",
                    location=f"claim:{claim.claim_number}",
                    description=f"独立权利要求 {claim.claim_number} 不应包含从属引用 depends_on",
                    auto_fixable=True,
                )
            )
        if claim.claim_type == "dependent":
            if not claim.depends_on:
                issues.append(
                    ReviewIssue(
                        issue_id=f"ISS-{len(issues) + 1}",
                        category="claim_structure_error",
                        severity="blocking",
                        location=f"claim:{claim.claim_number}",
                        description=f"从属权利要求 {claim.claim_number} 未声明所引用的在前权利要求编号",
                        auto_fixable=True,
                    )
                )
            for dep in claim.depends_on:
                if dep >= claim.claim_number or dep < 1:
                    issues.append(
                        ReviewIssue(
                            issue_id=f"ISS-{len(issues) + 1}",
                            category="claim_structure_error",
                            severity="blocking",
                            location=f"claim:{claim.claim_number}",
                            description=f"从属权利要求 {claim.claim_number} 非法引用了权利要求 {dep}（只能引用在前的权利要求）",
                            auto_fixable=True,
                        )
                    )
        for eid in claim.evidence_ids:
            if eid not in valid_ev_map:
                issues.append(
                    ReviewIssue(
                        issue_id=f"ISS-{len(issues) + 1}",
                        category="invalid_citation",
                        severity="blocking",
                        location=f"claim:{claim.claim_number}",
                        description=f"权利要求 {claim.claim_number} 引用了不存在的证据编号 [{eid}]",
                        auto_fixable=True,
                    )
                )

    # 4. 单位一致性与术语表一致性检查
    for unit_warn in check_unit_consistency(combined_text):
        issues.append(
            ReviewIssue(
                issue_id=f"ISS-{len(issues) + 1}",
                category="unit_inconsistency",
                severity="warning",
                location="document",
                description=unit_warn,
                auto_fixable=True,
            )
        )

    for term_warn in check_terminology_consistency(
        full_sections_text, full_claims_text, draft.terminology_glossary
    ):
        issues.append(
            ReviewIssue(
                issue_id=f"ISS-{len(issues) + 1}",
                category="terminology_inconsistency",
                severity="warning",
                location="terminology_glossary",
                description=term_warn,
                auto_fixable=True,
            )
        )

    blocking_count = sum(1 for it in issues if it.severity == "blocking")
    warning_count = sum(1 for it in issues if it.severity == "warning")
    passed = blocking_count == 0

    summary = (
        f"自动质量审查通过（阻断问题 0 项，提示项 {warning_count} 项）。"
        if passed
        else f"自动质量审查发现 {blocking_count} 项阻断问题、{warning_count} 项警告，需触发定向修订。"
    )
    return QualityReviewResult(
        passed=passed,
        blocking_count=blocking_count,
        warning_count=warning_count,
        issues=issues,
        summary=summary,
        deterministic_checks_passed=passed,
    )


def apply_targeted_revision(
    draft: PatentDisclosureDraft,
    review: QualityReviewResult,
    evidence_items: list[EvidenceItem],
    human_feedback: str = "",
) -> PatentDisclosureDraft:
    """
    针对审查发现的问题或人工退回意见执行定向修复（规格 7.3 targeted_revision）：
    - 移除无效证据 ID；
    - 清理未解决占位符并标为【待确认】；
    - 补齐缺失必需章节；
    - 修正权利要求编号与非法从属引用；
    - 融入人工修改意见并递增版本号。
    """
    valid_ev_ids = {it.evidence_id for it in evidence_items}
    next_version = draft.version + 1

    updated_sections: list[PatentSectionDraft] = []
    existing_by_title: dict[str, PatentSectionDraft] = {}

    for sec in draft.sections:
        cleaned_content = sec.content
        for pattern in (r"\b(?:TODO|TBD|FIXME|XXX)\b", r"\{\{[^}]+\}\}", r"\[待填写[^\]]*\]"):
            cleaned_content = re.sub(pattern, "【待确认】", cleaned_content, flags=re.IGNORECASE)

        filtered_ev_ids = [eid for eid in sec.evidence_ids if eid in valid_ev_ids]
        needs_conf = sec.needs_confirmation
        support_stat = sec.support_status
        if sec.section_key == "beneficial_effects" and not filtered_ev_ids:
            needs_conf = True
            support_stat = "needs_confirmation"

        if human_feedback and sec.section_key == "technical_solution":
            cleaned_content = f"{cleaned_content}\n\n【结合评审补充说明】：{human_feedback.strip()}"

        new_sec = sec.model_copy(
            update={
                "content": cleaned_content,
                "evidence_ids": filtered_ev_ids,
                "needs_confirmation": needs_conf,
                "support_status": support_stat,
                "version": next_version,
            }
        )
        updated_sections.append(new_sec)
        existing_by_title[new_sec.title.strip()] = new_sec

    # 补齐任何缺失的必需章节
    for req_title in REQUIRED_DISCLOSURE_SECTION_TITLES:
        if req_title not in existing_by_title:
            updated_sections.append(
                PatentSectionDraft(
                    section_key=f"auto_filled_{len(updated_sections) + 1}",
                    title=req_title,
                    content=f"【待确认】《{req_title}》章节待发明人补充具体技术细节。",
                    evidence_ids=[],
                    support_status="needs_confirmation",
                    needs_confirmation=True,
                    version=next_version,
                )
            )

    # 修复权利要求编号与从属关系
    updated_claims: list[ClaimDraftItem] = []
    for idx, claim in enumerate(draft.claims, start=1):
        filtered_ev = [eid for eid in claim.evidence_ids if eid in valid_ev_ids]
        if idx == 1 or claim.claim_type == "independent":
            c_type = "independent" if idx == 1 else claim.claim_type
            deps: list[int] = [] if c_type == "independent" else [1]
        else:
            c_type = "dependent"
            deps = [d for d in claim.depends_on if 1 <= d < idx] or [1]
        updated_claims.append(
            claim.model_copy(
                update={
                    "claim_number": idx,
                    "claim_type": c_type,
                    "depends_on": deps,
                    "evidence_ids": filtered_ev,
                }
            )
        )

    if not updated_claims:
        updated_claims.append(
            ClaimDraftItem(
                claim_number=1,
                claim_type="independent",
                depends_on=[],
                preamble=f"一种{draft.title}，其特征在于，包括：",
                characterizing_portion="【待确认】核心技术特征组合",
                full_text=f"1. 一种{draft.title}，其特征在于，包括：【待确认】核心技术特征组合。",
                evidence_ids=[],
                needs_confirmation=True,
            )
        )

    return draft.model_copy(
        update={
            "version": next_version,
            "sections": updated_sections,
            "claims": updated_claims,
        }
    )
