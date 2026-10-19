# LocalPatentAI — 面向科研成果转化的本地化多智能体专利辅助撰写系统

`LocalPatentAI` 是一套面向科研机构与研发团队的**完全私有化部署**多智能体专利技术交底书与权利要求书草案辅助撰写系统。系统将 **算法组 (`algorithm`)**、**框架组 (`framework`)**、**云组 (`cloud`)** 三个科研小组的未公开技术方案、实验记录、代码说明与项目文档严格隔离在本地受控知识空间中，结合混合 RAG（BM25 + 稠密向量检索 + RRF 融合 + BGE Reranker 重排）、Agent Harness 受控执行治理与 LangGraph 持久化工作流，输出具备原文证据追溯链、待确认清单与自动质量审查报告的规范中文 PDF 专利交底书草案。

---

## 1. 核心架构与安全边界

### 1.1 系统分层架构
- **客户端与接入层**：提供统一 RESTful API (`/api/v1/*`) 与可选本地 Streamlit 演示台 (`src/localpatentai/ui/streamlit_app.py`)。
- **应用网关与权限隔离层 (`auth/` & `audit/`)**：
  - 全部请求强制经由服务端 `AuthorizationService` 推导用户上下文 (`UserContext`)；
  - 绝不信任客户端请求体、Query 参数或 Prompt 中传入的 `group_id`、`project_id` 或 `authorized_space_ids`；
  - **三组严格物理/逻辑隔离**：`algorithm`（算法组）、`framework`（框架组）、`cloud`（云组）跨组默认不可见、不可检索、不可通过 Prompt 注入或任务恢复越权访问；
  - **最小权限与 Fail-Closed**：`system_admin` 仅负责系统运维，默认不自动拥有业务项目文档读取权；当权限服务异常或检索作用域为空时，检索层强制拒绝执行（Fail-Closed），绝不回退为全库检索。
- **结构化文档解析与受控混合 RAG 层 (`documents/` & `knowledge/`)**：
  - 支持 `PDF`、`DOCX`、`TXT`、`Markdown` 解析，保留页码、章节路径、段落、表格、公式、附图说明与权利要求结构块；
  - 检索先在数据库层执行 `knowledge_space_id IN authorized_space_ids AND document.status = 'indexed'` 预过滤，再执行 **BM25 + 1024 维向量 (`bge-large-zh-v1.5`) 检索 -> RRF 融合 (`k=60`) -> 本地 `bge-reranker-v2-m3` 重排 -> Evidence Pack 组装**；
  - 文档删除同步清理磁盘原文件、版本记录、Chunk 文本与向量索引，删除后立即不可被检索。
- **Agent Harness 与 LangGraph 编排层 (`harness/` & `workflows/` & `drafting/`)**：
  - 包含 `ContextBuilder`（按节点预算裁剪上下文与不可信文档隔离）、`ToolRegistry`（节点级工具白名单）、`OutputValidator`（Pydantic Schema 强校验）、`RetryPolicy`、`BudgetController`（限制单任务最大 LLM 调用次数与最大修订轮数 `max_revision_rounds`）与 `Tracing`；
  - LangGraph 11 节点状态机：`validate_task_and_authorize -> parse_or_load_documents -> analyze_technical_solution -> retrieve_evidence -> assess_evidence_gaps -> [request_human_input] -> draft_sections -> validate_draft_schema -> review_draft -> [targeted_revision] -> human_approval -> render_pdf -> validate_pdf -> save_artifact_and_audit`；
  - 基于 PostgreSQL `PostgresSaver` 实现断点持久化、人工确认暂停 (`interrupt`) 与恢复时权限重校验；
  - **反幻觉约束**：对缺乏内部实验数据支撑的技术效果或参数，强制标记为 `needs_confirmation` 与 `【待确认】`，严禁虚构实验数值或现有技术对比结论。
- **中文 PDF 渲染与后检验层 (`pdf/`)**：
  - 基于 Jinja2 HTML/CSS 模板与 PyMuPDF `fitz.Story` + 内置简体中文字体 (`china-s`) 生成多页中文 PDF（涵盖封面、七大标准交底书章节、独立/从属权利要求书草案、待确认事项清单、检索证据来源索引与质量审查附录）；
  - `validate_generated_pdf` 自动重新打开生成的 PDF，校验页数、章节标题、权利要求、无未替换模板占位符及文件落盘大小，通过后方才登记为正式 `Artifact`。

---

## 2. 目录结构

```text
LocalPatentAI/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── alembic.ini
├── configs/
│   ├── app.yaml
│   ├── models.yaml
│   ├── retrieval.yaml
│   └── pdf.yaml
├── migrations/
│   ├── env.py
│   └── versions/0001_initial_schema.py
├── scripts/
│   ├── init_db.py
│   ├── seed_groups.py
│   ├── start_vllm.sh
│   ├── reindex_knowledge_space.py
│   └── evaluate_rag.py
├── src/localpatentai/
│   ├── main.py
│   ├── core/          # 配置强校验、统一脱敏错误、日志脱敏过滤器、安全路径与密码哈希
│   ├── auth/          # AuthorizationService、三组权限矩阵、会话认证
│   ├── projects/      # 项目与项目私有知识空间管理、同组成员校验
│   ├── documents/     # PDF/DOCX/TXT/MD 解析、结构化分块、版本化索引与彻底删除
│   ├── knowledge/     # BM25、pgvector 向量检索、RRF 融合、Reranker、Evidence Pack
│   ├── llm/           # 本地 Qwen vLLM 客户端、BGE Embedding 与 Reranker 封装
│   ├── harness/       # BudgetController、ContextBuilder、ToolRegistry、Retry、Tracing
│   ├── workflows/     # LangGraph 状态定义、节点实现、Checkpointer 与 Output Validator
│   ├── drafting/      # 交底书/权利要求书 Schema、撰写服务、术语与质量审查规则
│   ├── pdf/           # Jinja2 HTML 模板、中文 PDF 渲染器与 PDF Validator 后检验器
│   ├── tasks/         # 任务创建、中断恢复、人工审批与安全产物下载
│   ├── audit/         # 安全审计事件记录
│   ├── api/           # FastAPI 路由与请求/响应 Schema
│   └── ui/            # 本地 Streamlit 演示界面
└── tests/
    ├── conftest.py
    ├── unit/          # 核心配置、脱敏、解析分块、RRF、Harness、审查规则单元测试
    ├── security/      # P0 三组严格权限隔离与越权攻击防御测试（12 项核心验收）
    ├── integration/   # 文档全生命周期、LangGraph 工作流中断恢复与 PDF 校验集成测试
    └── e2e/           # 系统健康检查与端到端冒烟测试
```

---

## 3. 环境准备与快速启动

### 3.1 Python 环境准备
当前机器可直接复用已验证的 `/data/djh/algo_infer` 环境，或克隆为独立环境 `localpatentai`：

```bash
conda create -n localpatentai --clone algo_infer
conda activate localpatentai
```

### 3.2 数据库迁移与三组初始化
确保本地 PostgreSQL + `pgvector` 服务运行（默认端口 `6001`），复制 `.env.example` 为 `.env` 后执行：

```bash
# 运行 Alembic 迁移并初始化 algorithm / framework / cloud 三组与公共空间
/data/djh/algo_infer/bin/python scripts/init_db.py
/data/djh/algo_infer/bin/python scripts/seed_groups.py
```

### 3.3 启动本地 Qwen vLLM 推理服务（可选）
生产或联调模式下，通过本机 `qwen3vl-sft` 环境启动本地 OpenAI-compatible 推理服务（默认监听 `127.0.0.1:8001`）：

```bash
bash scripts/start_vllm.sh
```

在离线测试或无 GPU 环境下，可将环境变量设为 `LLM_PROVIDER=mock EMBEDDING_BACKEND=mock RERANKER_BACKEND=mock`。

### 3.4 启动 FastAPI 后端与 Streamlit 界面

```bash
# 启动 FastAPI 服务
/data/djh/algo_infer/bin/uvicorn localpatentai.main:app --app-dir src --host 127.0.0.1 --port 8080

# （可选）启动本地 Streamlit 演示台
PYTHONPATH=src /data/djh/algo_infer/bin/streamlit run src/localpatentai/ui/streamlit_app.py --server.port 8501
```

---

## 4. RAG 离线消融评测结果

使用本机真实 `models/bge-large-zh-v1.5` 与 `models/bge-reranker-v2-m3` 模型运行 100 道结构化专利评测集（涵盖技术参数精确检索、技术效果、实验表格、公式、同义术语、现有技术对比、页码定位与 10% 无答案拒答测试）：

```bash
/data/djh/algo_infer/bin/python scripts/evaluate_rag.py --real-bge
```

评测报告保存在 `data/rag_eval_report.json`，实测结果如下：

| 检索模式 (`mode`) | Top-K | Recall@5 | MRR@5 | nDCG@5 | Citation Precision | Citation Coverage | 平均延迟 (ms) | 无答案正确拒答率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense_only` | 5 | 1.0000 | 0.9352 | 0.9517 | 0.8889 | 1.0000 | 135.621 | 1.0000 |
| `bm25_only` | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.269 | 1.0000 |
| `hybrid_rrf` | 5 | 1.0000 | 0.9722 | 0.9795 | 0.9444 | 1.0000 | 123.711 | 1.0000 |
| `hybrid_rrf_rerank` | 5 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 120.131 | **1.0000** |

---

## 5. 运行测试与代码质量检查

```bash
# 运行 Ruff 代码风格与静态检查
/data/djh/algo_infer/bin/ruff check .

# 运行全量自动化测试（含 P0 三组权限隔离、单元测试、LangGraph 工作流与 PDF 端到端测试）
/data/djh/algo_infer/bin/pytest -v
```
