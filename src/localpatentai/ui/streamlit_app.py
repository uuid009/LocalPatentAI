"""
LocalPatentAI 本地 Streamlit 演示界面（可选 Web UI）。
展示三组（algorithm / framework / cloud）严格权限隔离、项目文档上传管理、受控混合 RAG 检索与 LangGraph 专利撰写 PDF 导出。
"""

from __future__ import annotations

import streamlit as st
from sqlalchemy import select

from localpatentai.auth.service import AuthorizationService, ensure_demo_group_user
from localpatentai.db.models import Group, User
from localpatentai.db.session import get_session_factory
from localpatentai.documents.ingestion import ingest_project_document, list_project_documents
from localpatentai.knowledge.service import KnowledgeService
from localpatentai.projects.service import create_project, list_accessible_projects
from localpatentai.tasks.service import (
    create_and_start_patent_task,
    get_authorized_artifact_for_download,
    list_task_artifacts,
    submit_human_input_or_approval,
)

st.set_page_config(page_title="LocalPatentAI 私有化科研专利辅助撰写系统", layout="wide")
st.title("LocalPatentAI — 私有化多智能体科研专利撰写与三组隔离平台")


def _ensure_demo_environment(db) -> None:
    """若当前数据库尚无演示用户与项目，自动初始化三组演示研究员与默认项目。"""
    existing_users = db.scalars(select(User).where(User.is_active.is_(True))).all()
    if existing_users:
        return

    demo_specs = [
        (
            "algorithm",
            "算法组混合精度稀疏注意力专利项目",
            "面向端侧大模型推理的动态块级 FP8/INT4 混合精度稀疏注意力方案",
            "algo_spec.md",
            ("# 算法组核心技术交底\n本发明提出一种动态块级 FP8/INT4 混合精度稀疏注意力计算方法。"
            "当注意力权重矩阵块的最大激活幅值超过阈值 Tau=1.85 时保留 FP8 精度，否则压缩为 INT4 分组量化（group_size=64），"
            "推理吞吐量提升 2.4 倍，显存占用降低 58.3%。\n实验步骤 S1：采集 8192 序列长度激活张量；实验结果：首字延迟 TTFT 为 59.2ms。"),
        ),
        (
            "framework",
            "框架组异构计算图算子融合专利项目",
            "深度学习编译器的静态内存规划与跨流算子融合调度器",
            "framework_spec.md",
            ("# 框架组核心技术交底\n本发明提供一种深度学习编译器的静态内存规划与跨流算子融合调度器。"
            "将通信张量切分为 4MB 环形缓冲区分片，实现计算与通信重叠率达 87.5%，消除中间张量 HBM 重复读写。"),
        ),
        (
            "cloud",
            "云组算力容器拓扑感知热迁移专利项目",
            "基于芯片互联拓扑亲和度评分的弹性容器热迁移方法",
            "cloud_spec.md",
            ("# 云组核心技术交底\n本发明公开一种基于芯片互联拓扑亲和度评分的弹性容器热迁移方法。"
            "当节点温度超过 82℃ 或触发 ECC 预警时，调度器在 350ms 内完成训练检查点增量快照并迁移至同交换机备用节点，稳态 PUE 降至 1.12。"),
        ),
    ]
    for group_code, proj_name, proj_desc, doc_name, doc_text in demo_specs:
        user, _, _ = ensure_demo_group_user(db, group_code)
        proj, priv_space = create_project(
            db,
            user_id=user.id,
            name=proj_name,
            description=proj_desc,
            group_code=group_code,
        )
        ingest_project_document(
            db,
            user_id=user.id,
            project_id=proj.id,
            knowledge_space_id=priv_space.id,
            filename=doc_name,
            content=doc_text.encode("utf-8"),
            document_type="technical_report",
            auto_index=True,
        )


session_factory = get_session_factory()
with session_factory() as db:
    _ensure_demo_environment(db)

    users = list(db.scalars(select(User).where(User.is_active.is_(True))).all())
    groups = {str(g.id): g.code for g in db.scalars(select(Group)).all()}

    user_map = {f"{u.display_name} ({u.username})": u for u in users}
    selected_label = st.sidebar.selectbox(
        "切换当前登录用户（演示三组严格隔离）", list(user_map.keys())
    )
    current_user = user_map[selected_label]

    authz = AuthorizationService(db)
    ctx = authz.get_user_context(current_user.id)
    st.sidebar.markdown(f"**当前用户**：`{ctx.username}`")
    st.sidebar.markdown(f"**所属科研小组**：`{', '.join(sorted(ctx.group_codes)) or '无业务组'}`")

    with st.sidebar.expander("新建本组专利项目"):
        new_proj_name = st.text_input("新项目名称", value="")
        new_proj_desc = st.text_area("新项目简介", value="")
        if st.button("创建项目") and new_proj_name.strip():
            create_project(
                db,
                user_id=current_user.id,
                name=new_proj_name.strip(),
                description=new_proj_desc.strip(),
            )
            st.rerun()

    projects = list_accessible_projects(db, current_user.id)
    if not projects:
        st.warning("当前用户没有任何可访问的业务项目。")
    else:
        proj_map = {f"{p.name} [{groups.get(str(p.group_id), '')}]": p for p in projects}
        proj_label = st.selectbox("选择当前工作项目（仅显示本组授权项目）", list(proj_map.keys()))
        current_proj = proj_map[proj_label]

        tab_docs, tab_search, tab_draft = st.tabs(
            ["1. 项目文档与索引管理", "2. 受控混合 RAG 检索", "3. 多智能体交底书生成与 PDF 导出"]
        )

        with tab_docs:
            st.subheader("已上传项目文档")
            docs = list_project_documents(db, user_id=current_user.id, project_id=current_proj.id)
            if not docs:
                st.info("当前项目暂无文档，请在下方上传。")
            else:
                for d in docs:
                    st.markdown(
                        f"- **{d.original_filename}** | 类型：`{d.document_type}` | "
                        f"状态：`{d.status}` | 大小：`{d.file_size} B`"
                    )

            st.divider()
            uploaded = st.file_uploader(
                "上传新科研材料（支持 PDF / DOCX / Markdown / TXT）",
                type=["pdf", "docx", "md", "txt"],
            )
            doc_type = st.selectbox(
                "文档类型",
                ["technical_report", "experiment_record", "prior_art", "code_doc"],
            )
            if uploaded is not None and st.button("上传并构建索引"):
                raw_bytes = uploaded.read()
                ingest_project_document(
                    db,
                    user_id=current_user.id,
                    project_id=current_proj.id,
                    filename=uploaded.name,
                    content=raw_bytes,
                    document_type=doc_type,
                    auto_index=True,
                )
                st.success(f"已成功上传并索引：{uploaded.name}")
                st.rerun()

        with tab_search:
            q = st.text_input("输入技术检索词（仅检索当前项目授权空间）", value="混合精度 阈值 缓冲区 热迁移")
            mode = st.selectbox(
                "检索模式",
                ["hybrid_rrf_rerank", "hybrid_rrf", "bm25_only", "dense_only"],
            )
            if st.button("执行受控检索"):
                ks_svc = KnowledgeService(db)
                pack = ks_svc.search_for_project(
                    user_id=current_user.id,
                    project_id=current_proj.id,
                    query=q,
                    mode=mode,  # type: ignore[arg-type]
                )
                st.write(
                    f"命中授权证据条数：`{len(pack.items)}` | "
                    f"耗时：`{pack.retrieval_latency_ms:.2f} ms`"
                )
                for item in pack.items:
                    score_val = item.reranker_score or item.retrieval_score
                    st.markdown(
                        f"- **[{item.evidence_id}] {item.document_name}** "
                        f"(页码: `{item.page_start or 1}`, 得分: `{score_val:.4f}`, "
                        f"状态: `{item.support_status}`)\n  > {item.quote}"
                    )

        with tab_draft:
            title = st.text_input(
                "专利交底书标题", value=f"{current_proj.name} - 专利技术交底书草案"
            )
            prompt = st.text_area(
                "发明核心要点与撰写指令",
                value="请基于项目内已上传的实验记录与技术报告，生成完整的专利技术交底书与权利要求书草案。",
            )
            if st.button("一键启动 LangGraph 工作流并导出中文 PDF"):
                with st.spinner("正在执行多智能体技术方案解析、受控证据检索、章节草拟、质量审查与中文 PDF 渲染..."):
                    task = create_and_start_patent_task(
                        db,
                        user_id=current_user.id,
                        project_id=current_proj.id,
                        title=title,
                        user_prompt=prompt,
                        require_gap_confirmation=False,
                    )
                    if task.status in {"awaiting_approval", "awaiting_human_input"}:
                        task = submit_human_input_or_approval(
                            db,
                            user_id=current_user.id,
                            task_id=task.id,
                            action="approve",
                            comment="界面一键审批通过并生成正式 PDF",
                        )

                st.success(f"任务执行完成！当前状态：`{task.status}`")
                snap = task.state_snapshot_json or {}
                draft_data = snap.get("draft") or {}
                if draft_data:
                    with st.expander("查看结构化交底书草案与权利要求书预览", expanded=True):
                        for sec in draft_data.get("sections", []):
                            st.markdown(f"#### {sec.get('title')}")
                            st.write(sec.get("content"))
                        st.markdown("#### 权利要求书草案")
                        for cl in draft_data.get("claims", []):
                            st.markdown(f"**权利要求 {cl.get('claim_number')}**：{cl.get('full_text')}")

                artifacts = list_task_artifacts(db, user_id=current_user.id, task_id=task.id)
                if artifacts:
                    art, pdf_path = get_authorized_artifact_for_download(
                        db, user_id=current_user.id, artifact_id=artifacts[0].id
                    )
                    st.download_button(
                        label="下载生成的中文专利技术交底书 PDF",
                        data=pdf_path.read_bytes(),
                        file_name=f"patent_disclosure_v{art.version}.pdf",
                        mime="application/pdf",
                    )
