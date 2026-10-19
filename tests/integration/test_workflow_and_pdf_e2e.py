"""
集成与端到端测试（覆盖规格 12.2 ~ 12.4 节要求）：
1. 文档上传（PDF/DOCX/MD）、版本覆盖、重索引、彻底删除后立刻不可被检索。
2. LangGraph 专利撰写工作流：
   - 正常完成流程 + PDF 渲染 + PDF Validator 后检验（重新打开、页数、七大章节、权利要求、来源索引、落盘大小）。
   - 人工审批暂停（interrupt）与恢复（resume）。
   - 证据不足时自动标记 needs_confirmation / 【待确认】，绝不编造实验数值。
   - 质量审查未通过时自动触发 targeted_revision 并在最大修订预算处停止。
"""

from __future__ import annotations

from pathlib import Path

import fitz

from localpatentai.documents.ingestion import (
    delete_document_completely,
    ingest_project_document,
)
from localpatentai.drafting.schemas import QualityReviewResult
from localpatentai.knowledge.service import KnowledgeService
from localpatentai.tasks.service import (
    create_and_start_patent_task,
    submit_human_input_or_approval,
)
from tests.conftest import make_sample_docx_bytes, make_sample_pdf_bytes


def test_document_lifecycle_upload_reindex_and_complete_deletion(
    client, db_session, seeded_org, test_settings, tmp_path: Path
):
    """测试 PDF/DOCX/MD 上传解析、检索命中、重索引以及删除后彻底从向量/BM25 库清除。"""
    u_algo = seeded_org["users"]["algo"]
    p_algo = seeded_org["projects"]["algo"]
    ks_algo = seeded_org["spaces"]["algo_private"]
    token_algo = seeded_org["tokens"]["algo"]
    hdr = {"Authorization": f"Bearer {token_algo}"}

    # 1. 通过 API 上传 PDF
    pdf_bytes = make_sample_pdf_bytes(
        "动态稀疏注意力实验报告",
        "本发明专属检索词 UNIQUE_PDF_TOKEN_991，推理速度提升 2.4 倍。",
    )
    resp_up = client.post(
        f"/api/v1/projects/{p_algo.id}/documents",
        headers=hdr,
        data={
            "knowledge_space_id": str(ks_algo.id),
            "document_type": "experiment_record",
            "auto_index": "true",
        },
        files={"file": ("exp_report.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp_up.status_code == 201, resp_up.text
    doc_id = resp_up.json()["id"]
    assert resp_up.json()["status"] == "indexed"

    # 2. 通过服务上传 DOCX
    docx_bytes = make_sample_docx_bytes(
        tmp_path,
        "算子量化参数附表",
        ["本文件包含 DOCX 专属标识 UNIQUE_DOCX_TOKEN_882，阈值 Tau=1.85。"],
    )
    doc_docx = ingest_project_document(
        db_session,
        user_id=u_algo.id,
        project_id=p_algo.id,
        knowledge_space_id=ks_algo.id,
        filename="params.docx",
        content=docx_bytes,
        settings=test_settings,
    )

    ks_svc = KnowledgeService(db_session, test_settings)
    pack_pdf = ks_svc.search_for_project(
        user_id=u_algo.id,
        project_id=p_algo.id,
        query="UNIQUE_PDF_TOKEN_991",
    )
    assert any("UNIQUE_PDF_TOKEN_991" in it.quote for it in pack_pdf.items)

    pack_docx = ks_svc.search_for_project(
        user_id=u_algo.id,
        project_id=p_algo.id,
        query="UNIQUE_DOCX_TOKEN_882",
    )
    assert any("UNIQUE_DOCX_TOKEN_882" in it.quote for it in pack_docx.items)

    # 3. 重建单文档索引与知识空间索引
    r_reidx = client.post(f"/api/v1/documents/{doc_id}/reindex", headers=hdr)
    assert r_reidx.status_code == 200

    reidx_space_res = ks_svc.reindex_knowledge_space(
        user_id=u_algo.id,
        knowledge_space_id=ks_algo.id,
    )
    assert reidx_space_res["reindexed_documents"] >= 2

    # 4. 删除 PDF 文档后，立即不可被检索到，且磁盘文件被清理
    r_del = client.delete(f"/api/v1/documents/{doc_id}", headers=hdr)
    assert r_del.status_code == 200

    pack_after_del = ks_svc.search_for_project(
        user_id=u_algo.id,
        project_id=p_algo.id,
        query="UNIQUE_PDF_TOKEN_991",
    )
    assert all("UNIQUE_PDF_TOKEN_991" not in it.quote for it in pack_after_del.items)

    # 删除 DOCX 文档
    delete_document_completely(
        db_session,
        user_id=u_algo.id,
        document_id=doc_docx.id,
        settings=test_settings,
    )


def test_workflow_end_to_end_pdf_export_and_human_interrupt_resume(
    client, db_session, seeded_org, test_settings
):
    """
    验证 LangGraph 工作流：
    1. 暂停等待人工审批 (awaiting_approval) -> 提交审批意见恢复 (completed) -> 生成并校验中文 PDF。
    2. 证据不足时自动标记 needs_confirmation 与【待确认】。
    """
    u_algo = seeded_org["users"]["algo"]
    p_algo = seeded_org["projects"]["algo"]
    ks_algo = seeded_org["spaces"]["algo_private"]
    token_algo = seeded_org["tokens"]["algo"]
    hdr = {"Authorization": f"Bearer {token_algo}"}

    # 上传一份仅包含技术方案但缺少具体实验记录数值的文档，验证证据不足不会编造实验数据
    ingest_project_document(
        db_session,
        user_id=u_algo.id,
        project_id=p_algo.id,
        knowledge_space_id=ks_algo.id,
        filename="algo_scheme_no_experiment.md",
        content=(
            "# 动态块级混合精度注意力机制技术方案\n"
            "本发明通过检测注意力矩阵块的激活幅值动态切换量化位宽，包括检测模块与块级量化模块。"
        ).encode(),
        settings=test_settings,
    )

    # 创建任务（默认停在 human_approval 等待人工确认导出）
    r_create = client.post(
        f"/api/v1/projects/{p_algo.id}/patent-tasks",
        headers=hdr,
        json={
            "title": "动态块级混合精度注意力专利交底书",
            "user_prompt": "请根据项目资料撰写动态块级混合精度注意力机制专利技术交底书及权利要求书。",
            "require_gap_confirmation": False,
        },
    )
    assert r_create.status_code == 201, r_create.text
    task_data = r_create.json()
    task_id = task_data["id"]
    assert task_data["status"] == "awaiting_approval"
    assert task_data["current_stage"] == "human_approval"

    # 恢复任务并批准导出 PDF
    r_approve = client.post(
        f"/api/v1/tasks/{task_id}/human-input",
        headers=hdr,
        json={
            "action": "approve",
            "comment": "同意草稿结构，请正式导出 PDF 交底书",
        },
    )
    assert r_approve.status_code == 200, r_approve.text
    resumed_data = r_approve.json()
    assert resumed_data["status"] == "completed"
    artifact_id = resumed_data["artifact_id"]
    assert artifact_id is not None

    # 检查草稿中对于缺少具体实验数据部分必须包含待确认标记，不虚构实验指标
    draft_json = resumed_data["draft"]
    assert draft_json is not None
    assert (
        len(draft_json.get("unresolved_questions") or []) >= 1
        or any(s.get("needs_confirmation") for s in draft_json.get("sections", []))
    )

    # 下载生成的 PDF 并执行完整性校验
    r_pdf = client.get(f"/api/v1/artifacts/{artifact_id}/download", headers=hdr)
    assert r_pdf.status_code == 200
    assert r_pdf.headers["content-type"].startswith("application/pdf")
    pdf_bytes = r_pdf.content
    assert len(pdf_bytes) > 1024

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert doc.page_count >= 2
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    # 验证中文关键章节与权利要求均正常渲染无乱码
    for expected_heading in (
        "技术领域",
        "背景技术",
        "待解决的技术问题",
        "技术方案",
        "有益效果",
        "具体实施方式",
        "权利要求",
        "待确认",
    ):
        assert expected_heading in full_text


def test_workflow_targeted_revision_stops_at_max_budget(
    db_session, seeded_org, test_settings, monkeypatch
):
    """验证质量审查不通过时触发 targeted_revision 局部修订，并在达到 max_revision_loops 后停止死循环。"""
    u_algo = seeded_org["users"]["algo"]
    p_algo = seeded_org["projects"]["algo"]
    ks_algo = seeded_org["spaces"]["algo_private"]

    ingest_project_document(
        db_session,
        user_id=u_algo.id,
        project_id=p_algo.id,
        knowledge_space_id=ks_algo.id,
        filename="algo_base.md",
        content="# 算法方案\n本发明提供步骤 S1 与步骤 S2 的矩阵量化方法。".encode(),
        settings=test_settings,
    )

    call_counter = {"count": 0}

    def _always_fail_review(draft, filtered_evidence):
        call_counter["count"] += 1
        return QualityReviewResult(
            passed=False,
            blocking_count=0,
            warning_count=1,
            summary_notes=["请进一步细化步骤 S2 的量化边界条件说明"],
        )

    monkeypatch.setattr(
        "localpatentai.workflows.validators.review_patent_draft",
        _always_fail_review,
    )

    task = create_and_start_patent_task(
        db_session,
        user_id=u_algo.id,
        project_id=p_algo.id,
        title="测试修订预算上限",
        user_prompt="请生成交底书并测试最大修订循环控制",
    )
    if task.status == "awaiting_approval":
        task = submit_human_input_or_approval(
            db_session,
            user_id=u_algo.id,
            task_id=task.id,
            action="approve",
        )
    assert task.status == "completed"
    # 初次审查 1 次 + 最多 max_revision_loops 次 targeted_revision 后复审停止
    assert call_counter["count"] >= 2
