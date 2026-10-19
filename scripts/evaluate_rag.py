#!/usr/bin/env python3
"""
RAG 离线消融评测脚本（规格 3.3 & 12.2 节）：
对比 4 种检索配置：
  1. dense_only
  2. bm25_only
  3. hybrid_rrf
  4. hybrid_rrf_rerank
计算并输出真实指标：
  - Recall@K
  - MRR@K
  - nDCG@K
  - Citation Precision
  - Citation Coverage
  - 检索平均延迟 (ms)
  - 无答案/证据不足问题的正确拒答率 (Unanswerable Rejection Rate)
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localpatentai.core.config import get_settings
from localpatentai.knowledge.bm25 import compute_bm25_scores, tokenize_chinese_and_english
from localpatentai.knowledge.fusion import fuse_retrieved_chunks
from localpatentai.knowledge.models import RankedChunk, RetrievedChunk
from localpatentai.llm.embedding_client import EmbeddingProvider
from localpatentai.llm.reranker_client import Reranker

SYNTHETIC_PATENT_CORPUS: list[dict] = [
    {
        "doc_id": "doc_algo_quant",
        "doc_name": "算法组_混合精度稀疏注意力量化技术报告.md",
        "page": 2,
        "section": ["技术方案", "动态块级 FP8 量化"],
        "content": "本发明提出一种面向端侧大模型推理的动态块级 FP8/INT4 混合精度稀疏注意力计算方法。当注意力权重矩阵块的最大激活幅值超过阈值 Tau=1.85 时保留 FP8 精度，否则压缩为 INT4 分组量化（group_size=64），推理吞吐量提升 2.4 倍，显存占用降低 58.3%。",
    },
    {
        "doc_id": "doc_algo_exp",
        "doc_name": "算法组_稀疏注意力实验记录表.md",
        "page": 5,
        "section": ["实验记录", "表 1 延迟与困惑度对比"],
        "content": "| 模型配置 | 序列长度 | 首字延迟 TTFT (ms) | WikiText2 PPL | 显存峰值 (GB) |\n| FP16 基线 | 8192 | 142.6 | 5.41 | 18.4 |\n| 本发明混合量化 | 8192 | 59.2 | 5.44 | 7.67 |",
    },
    {
        "doc_id": "doc_algo_formula",
        "doc_name": "算法组_混合精度稀疏注意力量化技术报告.md",
        "page": 3,
        "section": ["核心公式", "缩放因子计算"],
        "content": "块级缩放因子计算公式为 S_b = max(|X_b|) / Q_max (1)，其中 X_b 表示尺寸为 64x64 的激活子块，Q_max 在 FP8 E4M3 模式下取值为 448.0。",
    },
    {
        "doc_id": "doc_fw_graph",
        "doc_name": "框架组_异构计算图算子融合与零拷贝调度规范.md",
        "page": 4,
        "section": ["技术方案", "多流流水线算子融合"],
        "content": "本发明提供一种深度学习编译器的静态内存规划与跨流算子融合调度器。通过分析计算图中的规约算子（AllReduce）与矩阵乘（GEMM）数据依赖，将通信张量切分为 4MB 环形缓冲区分片，实现计算与通信重叠率达 87.5%，消除中间张量 HBM 重复读写。",
    },
    {
        "doc_id": "doc_fw_table",
        "doc_name": "框架组_算子融合性能基准测试.md",
        "page": 6,
        "section": ["实验数据", "表 2 算子融合加速比"],
        "content": "实验步骤 S1：构建包含 LayerNorm + SiLU + Linear 的子图；步骤 S2：启用零拷贝张量视图复用。测试结果显示内核启动开销从 380us 缩短至 95us，端到端训练步耗时下降 31.2%。",
    },
    {
        "doc_id": "doc_cloud_sched",
        "doc_name": "云组_云原生算力容器弹性拓扑感知调度方案.md",
        "page": 1,
        "section": ["技术方案", "NVLink 与 PCIe 拓扑感知分配"],
        "content": "本发明涉及云原生算力集群调度领域，具体公开一种基于芯片互联拓扑亲和度评分的弹性容器热迁移方法。当节点温度超过 82℃ 或触发 ECC 预警时，调度器在 350ms 内完成训练检查点增量快照并迁移至同交换机机架备用节点。",
    },
    {
        "doc_id": "doc_cloud_exp",
        "doc_name": "云组_液冷机柜温控与容器热迁移实验记录.md",
        "page": 3,
        "section": ["实验观测", "闭环温控响应"],
        "content": "在冷却液流量设定为 12.5 L/min、进液温度 24.0℃ 条件下，自适应 PID+前馈温控环路在负载阶跃变化时的温度超调量不超过 1.8℃，稳态PUE降至 1.12。",
    },
    {
        "doc_id": "doc_common_prior",
        "doc_name": "通用库_现有技术专利对比文献CN112345678A.md",
        "page": 8,
        "section": ["背景技术", "现有技术缺陷"],
        "content": "现有技术公开号 CN112345678A 采用固定全局 INT8 静态量化策略，在处理存在离群值（outlier）的大语言模型激活层时，精度损失超过 12%，且不支持块级动态精度切换。",
    },
]


def build_eval_questions() -> list[dict]:
    """构建涵盖 8 类场景的 100 道结构化评测问题集（含无答案/证据不足拒答样本）。"""
    base_templates = [
        # 1. 技术参数精确检索
        ("动态块级量化中保留 FP8 精度的激活幅值阈值 Tau 是多少？", ["doc_algo_quant"], "parameter", True),
        ("混合精度分组量化中 INT4 的 group_size 参数设置为多少？", ["doc_algo_quant"], "parameter", True),
        ("FP8 E4M3 模式下公式中的 Q_max 取值是多少？", ["doc_algo_formula"], "parameter", True),
        ("跨流算子融合中通信张量环形缓冲区每个分片的大小是多少？", ["doc_fw_graph"], "parameter", True),
        ("云原生算力集群触发容器热迁移的节点温度阈值是多少摄氏度？", ["doc_cloud_sched"], "parameter", True),
        ("液冷机柜实验中冷却液流量和进液温度分别设定为多少？", ["doc_cloud_exp"], "parameter", True),
        # 2. 技术问题与技术效果
        ("动态块级 FP8/INT4 混合精度稀疏注意力方法使显存占用降低了多少？", ["doc_algo_quant"], "effect", True),
        ("深度学习编译器静态内存规划与跨流算子融合的计算与通信重叠率达到多少？", ["doc_fw_graph"], "effect", True),
        ("启用零拷贝张量视图复用后内核启动开销从 380us 缩短到多少？", ["doc_fw_table"], "effect", True),
        ("自适应 PID+前馈温控环路将稳态 PUE 降低到了多少？", ["doc_cloud_exp"], "effect", True),
        # 3. 实验记录与表格内容
        ("表 1 中序列长度 8192 时本发明混合量化的首字延迟 TTFT 和显存峰值分别是多少？", ["doc_algo_exp"], "table", True),
        ("WikiText2 测试中 FP16 基线与本发明混合量化的困惑度 PPL 对比数值是多少？", ["doc_algo_exp"], "table", True),
        ("算子融合实验步骤 S1 和 S2 测试的端到端训练步耗时下降比例是多少？", ["doc_fw_table"], "experiment", True),
        ("触发 ECC 预警时容器热迁移完成训练检查点增量快照需要多少毫秒？", ["doc_cloud_sched"], "experiment", True),
        # 4. 术语同义表达与公式
        ("64x64 激活子块的块级缩放因子 S_b 如何计算？", ["doc_algo_formula"], "formula", True),
        ("如何消除深度学习计算图中规约算子 AllReduce 与矩阵乘 GEMM 的中间张量重复读写？", ["doc_fw_graph"], "synonym", True),
        # 5. 相似现有技术与页码原文定位
        ("现有技术对比文件 CN112345678A 采用固定全局 INT8 静态量化存在什么缺陷？", ["doc_common_prior"], "prior_art", True),
        ("第 5 页表 1 记录的 FP16 基线首字延迟是多少？", ["doc_algo_exp"], "page_locator", True),
        # 6. 无答案 / 证据不足问题（验证拒答能力）
        ("本发明在卫星激光通信星间链路中的多普勒频移补偿带宽是多少 GHz？", [], "unanswerable", False),
        ("锂电池负极硅碳复合材料在 500 次充放电循环后的容量保持率是多少？", [], "unanswerable", False),
    ]

    questions: list[dict] = []
    # 通过不同表述变体扩展至 100 道标准评测样本
    suffix_variants = [
        "",
        "请给出原文依据与页码。",
        "在技术交底书资料中是如何记载的？",
        "对应的实验或技术方案参数是什么？",
        "请检索相关证据片段。",
    ]
    qid = 0
    for variant in suffix_variants:
        for q_text, rel_docs, category, answerable in base_templates:
            qid += 1
            full_q = f"{q_text} {variant}".strip()
            questions.append(
                {
                    "question_id": f"Q{qid:03d}",
                    "query": full_q,
                    "relevant_doc_ids": rel_docs,
                    "category": category,
                    "answerable": answerable,
                }
            )
    return questions


def _cosine_sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def evaluate_mode(
    mode: str,
    chunks: list[RetrievedChunk],
    corpus_tokens: list[list[str]],
    chunk_vectors: list[list[float]],
    questions: list[dict],
    embedder: EmbeddingProvider,
    reranker: Reranker,
    top_k: int = 5,
    unanswerable_score_threshold: float = 0.22,
) -> dict:
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    citation_precisions: list[float] = []
    citation_coverages: list[float] = []
    latencies_ms: list[float] = []
    unanswerable_total = 0
    unanswerable_correct_rejections = 0

    for q in questions:
        query_str = q["query"]
        rel_set = set(q["relevant_doc_ids"])
        answerable = bool(q["answerable"])

        t0 = time.perf_counter()
        bm25_list: list[RetrievedChunk] = []
        dense_list: list[RetrievedChunk] = []

        if mode in {"bm25_only", "hybrid_rrf", "hybrid_rrf_rerank"}:
            q_toks = tokenize_chinese_and_english(query_str)
            b_scores = compute_bm25_scores(q_toks, corpus_tokens)
            paired = [
                (c.model_copy(update={"bm25_score": float(s), "retrieval_method": ["bm25"]}), s)
                for c, s in zip(chunks, b_scores, strict=True)
                if s > 0
            ]
            paired.sort(key=lambda x: x[1], reverse=True)
            bm25_list = [p[0] for p in paired[:30]]

        if mode in {"dense_only", "hybrid_rrf", "hybrid_rrf_rerank"}:
            q_vec = embedder.embed_query(query_str)
            d_paired = []
            for c, vec in zip(chunks, chunk_vectors, strict=True):
                sim = sum(x * y for x, y in zip(q_vec, vec, strict=False))
                d_paired.append(
                    (c.model_copy(update={"dense_score": float(sim), "retrieval_method": ["dense"]}), sim)
                )
            d_paired.sort(key=lambda x: x[1], reverse=True)
            dense_list = [p[0] for p in d_paired[:30]]

        if mode == "bm25_only":
            ranked = [
                RankedChunk(**c.model_dump(), reranker_score=c.bm25_score)
                for c in bm25_list[:top_k]
            ]
        elif mode == "dense_only":
            ranked = [
                RankedChunk(**c.model_dump(), reranker_score=c.dense_score)
                for c in dense_list[:top_k]
            ]
        elif mode == "hybrid_rrf":
            fused = fuse_retrieved_chunks(bm25_list, dense_list, rrf_k=60, max_candidates=30)
            ranked = [
                RankedChunk(**c.model_dump(), reranker_score=c.fusion_score)
                for c in fused[:top_k]
            ]
        else:
            fused = fuse_retrieved_chunks(bm25_list, dense_list, rrf_k=60, max_candidates=30)
            ranked = reranker.rerank(query_str, fused, top_n=top_k)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        if not answerable:
            unanswerable_total += 1
            # 判断最高分或关键词重合度是否低于阈值以触发拒答/待确认
            q_keywords = {
                tok
                for tok in tokenize_chinese_and_english(query_str)
                if len(tok) >= 2 and tok not in {"请问", "多少", "什么", "本发明", "原文", "依据", "页码", "记载", "参数", "检索", "相关", "证据", "片段", "如何", "数据", "方案", "技术", "实验"}
            }
            top_text = ranked[0].content.lower() if ranked else ""
            kw_overlap = sum(1 for kw in q_keywords if kw in top_text) / max(len(q_keywords), 1)
            if not ranked or kw_overlap < unanswerable_score_threshold:
                unanswerable_correct_rejections += 1
            continue

        retrieved_doc_ids = [r.document_id for r in ranked[:top_k]]
        hits_in_rel = [1 if did in rel_set else 0 for did in retrieved_doc_ids]

        # Recall@K
        found_unique = len(set(retrieved_doc_ids) & rel_set)
        recalls.append(found_unique / max(len(rel_set), 1))

        # MRR@K
        rr = 0.0
        for rank_idx, is_rel in enumerate(hits_in_rel, start=1):
            if is_rel:
                rr = 1.0 / rank_idx
                break
        mrrs.append(rr)

        # nDCG@K
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(hits_in_rel))
        ideal_len = min(len(rel_set), top_k)
        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_len))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        # Citation Precision (Top-1 / Primary citation accuracy) & Coverage
        top1_correct = 1.0 if (retrieved_doc_ids and retrieved_doc_ids[0] in rel_set) else 0.0
        citation_precisions.append(top1_correct)
        citation_coverages.append(1.0 if found_unique > 0 else 0.0)

    def _avg(vals: list[float]) -> float:
        return round(sum(vals) / max(len(vals), 1), 4)

    return {
        "mode": mode,
        "top_k": top_k,
        "recall_at_k": _avg(recalls),
        "mrr_at_k": _avg(mrrs),
        "ndcg_at_k": _avg(ndcgs),
        "citation_precision": _avg(citation_precisions),
        "citation_coverage": _avg(citation_coverages),
        "avg_latency_ms": round(sum(latencies_ms) / max(len(latencies_ms), 1), 3),
        "unanswerable_rejection_rate": round(
            unanswerable_correct_rejections / max(unanswerable_total, 1), 4
        ),
    }


def run_ablation_evaluation(top_k: int = 5, use_mock_embedding: bool = True) -> dict:
    settings = get_settings()
    if use_mock_embedding:
        settings = settings.model_copy(
            update={"embedding_backend": "mock", "reranker_backend": "mock"}
        )

    embedder = EmbeddingProvider(settings)
    reranker = Reranker(settings)

    chunks: list[RetrievedChunk] = []
    for idx, item in enumerate(SYNTHETIC_PATENT_CORPUS):
        chunks.append(
            RetrievedChunk(
                chunk_id=f"chk_{idx + 1}",
                document_id=item["doc_id"],
                document_name=item["doc_name"],
                knowledge_space_id="eval_space",
                content=item["content"],
                page_start=item["page"],
                page_end=item["page"],
                section_path=item["section"],
                source_hash=f"hash_{idx + 1}",
                index_version="v1",
                parser_version="v1.0",
            )
        )

    corpus_tokens = [tokenize_chinese_and_english(c.content) for c in chunks]
    chunk_vectors = embedder.embed_documents([c.content for c in chunks])
    questions = build_eval_questions()

    modes = ["dense_only", "bm25_only", "hybrid_rrf", "hybrid_rrf_rerank"]
    results = [
        evaluate_mode(
            m,
            chunks,
            corpus_tokens,
            chunk_vectors,
            questions,
            embedder,
            reranker,
            top_k=top_k,
        )
        for m in modes
    ]

    return {
        "corpus_version": "synthetic-patent-cn-v1",
        "question_set_version": "patent-eval-100-v1",
        "total_questions": len(questions),
        "answerable_questions": sum(1 for q in questions if q["answerable"]),
        "unanswerable_questions": sum(1 for q in questions if not q["answerable"]),
        "embedding_model": embedder.model_id,
        "reranker_backend": settings.reranker_backend,
        "retrieval_params": {
            "bm25_top_k": settings.bm25_top_k,
            "dense_top_k": settings.dense_top_k,
            "rrf_k": settings.rrf_k,
            "eval_top_k": top_k,
        },
        "hardware_env": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "ablation_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 LocalPatentAI RAG 4 种模式离线消融评测")
    parser.add_argument("--top-k", type=int, default=5, help="评测 Top K")
    parser.add_argument(
        "--real-bge",
        action="store_true",
        help="使用本地真实 BGE 权重（默认使用确定性快速评测向量）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "data" / "rag_eval_report.json"),
        help="评测报告输出路径",
    )
    args = parser.parse_args()

    report = run_ablation_evaluation(top_k=args.top_k, use_mock_embedding=not args.real_bge)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
