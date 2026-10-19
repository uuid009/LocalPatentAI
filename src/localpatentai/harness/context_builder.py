"""Context Builder（规格 8.1 节）：按节点职责与 token 预算裁剪最少且充分的上下文，保留证据 ID、页码和状态。"""

from __future__ import annotations

from localpatentai.documents.normalization import estimate_tokens
from localpatentai.knowledge.models import EvidenceItem


class ContextBuilder:
    def __init__(self, max_context_tokens: int = 4096) -> None:
        self.max_context_tokens = max_context_tokens

    def build_evidence_context(
        self,
        evidence_items: list[EvidenceItem],
        *,
        section_focus: str | None = None,
        token_budget: int | None = None,
    ) -> tuple[str, list[str]]:
        budget = token_budget or int(self.max_context_tokens * 0.75)
        lines: list[str] = []
        included_ids: list[str] = []
        used_tokens = 0

        # 优先选择与当前章节主题匹配或内部实验/直接事实证据
        sorted_items = sorted(
            evidence_items,
            key=lambda it: (
                0 if (section_focus and section_focus in it.quote) else 1,
                0 if it.evidence_type != "external_prior_art" else 1,
                -(it.reranker_score or it.retrieval_score or 0.0),
            ),
        )

        for item in sorted_items:
            entry = (
                f"[{item.evidence_id}] 来源={item.document_name} "
                f"页码={item.page_start or 1} 类型={item.evidence_type} "
                f"支持状态={item.support_status}\n"
                f"摘录：{item.quote.strip()}"
            )
            entry_tokens = estimate_tokens(entry)
            if used_tokens + entry_tokens > budget and lines:
                break
            lines.append(entry)
            included_ids.append(item.evidence_id)
            used_tokens += entry_tokens

        return "\n\n".join(lines), included_ids
