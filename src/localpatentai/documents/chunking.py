"""结构感知切分器（规格 5.4）：按普通正文、表格、权利要求、实验步骤、公式和图注分别处理。"""

from __future__ import annotations

import hashlib
import uuid

from localpatentai.documents.normalization import estimate_tokens
from localpatentai.knowledge.models import ChunkRecord, ParsedSectionBlock


def build_structure_aware_chunks(
    blocks: list[ParsedSectionBlock],
    *,
    document_id: str,
    knowledge_space_id: str,
    group_id: str | None = None,
    project_id: str | None = None,
    parser_version: str = "v1.0",
    embedding_model_id: str | None = None,
    target_tokens: int = 450,
    overlap_tokens: int = 80,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    buffer_texts: list[str] = []
    buffer_para_ids: list[str] = []
    buffer_section: list[str] = []
    buffer_page_start: int | None = None
    buffer_page_end: int | None = None
    buffer_flags: set[str] = set()

    def emit_chunk(
        text: str,
        section_path: list[str],
        page_start: int | None,
        page_end: int | None,
        para_ids: list[str],
        extra_meta: dict,
    ) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        for piece in _split_by_token_budget(cleaned, target_tokens, overlap_tokens):
            content_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            records.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    document_id=str(document_id),
                    knowledge_space_id=str(knowledge_space_id),
                    group_id=str(group_id) if group_id else None,
                    project_id=str(project_id) if project_id else None,
                    section_path=list(section_path),
                    page_start=page_start,
                    page_end=page_end,
                    paragraph_ids=list(para_ids),
                    content=piece,
                    content_hash=content_hash,
                    token_count=estimate_tokens(piece),
                    parser_version=parser_version,
                    embedding_model_id=embedding_model_id,
                    metadata=dict(extra_meta),
                )
            )

    def flush_buffer() -> None:
        nonlocal buffer_texts, buffer_para_ids, buffer_section
        nonlocal buffer_page_start, buffer_page_end, buffer_flags
        if not buffer_texts:
            return
        combined = "\n".join(buffer_texts)
        emit_chunk(
            combined,
            buffer_section,
            buffer_page_start,
            buffer_page_end,
            buffer_para_ids,
            {"block_type": "paragraph", "quality_flags": sorted(buffer_flags)},
        )
        buffer_texts = []
        buffer_para_ids = []
        buffer_section = []
        buffer_page_start = None
        buffer_page_end = None
        buffer_flags = set()

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue

        # 特殊结构块（表格、权利要求、实验步骤、公式、图注）保持语义独立
        if block.block_type in {"table", "claim", "experiment_step", "formula", "figure_caption"}:
            flush_buffer()
            meta = {
                "block_type": block.block_type,
                "table_id": block.table_id,
                "figure_id": block.figure_id,
                "quality_flags": list(block.quality_flags),
                **block.metadata,
            }
            emit_chunk(
                text,
                block.section_path,
                block.page_start,
                block.page_end,
                [block.paragraph_id] if block.paragraph_id else [],
                meta,
            )
            continue

        if block.block_type == "heading":
            flush_buffer()
            buffer_section = list(block.section_path)
            buffer_texts.append(f"【{text}】")
            if block.paragraph_id:
                buffer_para_ids.append(block.paragraph_id)
            buffer_page_start = block.page_start
            buffer_page_end = block.page_end
            buffer_flags.update(block.quality_flags)
            continue

        # 普通段落：若跨章节或超出 token 预算则先 flush
        current_tokens = estimate_tokens("\n".join(buffer_texts + [text]))
        if buffer_texts and (
            current_tokens > target_tokens
            or (buffer_section and block.section_path and buffer_section != block.section_path)
        ):
            flush_buffer()

        if not buffer_texts:
            buffer_section = list(block.section_path)
            buffer_page_start = block.page_start
        buffer_texts.append(text)
        if block.paragraph_id:
            buffer_para_ids.append(block.paragraph_id)
        if block.page_end is not None:
            buffer_page_end = block.page_end
        buffer_flags.update(block.quality_flags)

    flush_buffer()
    return records


def _split_by_token_budget(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]
    # 近似按字符窗口切分并保留重叠
    char_window = max(100, int(max_tokens * 1.2))
    char_overlap = max(20, min(int(overlap_tokens * 1.2), char_window // 2))
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + char_window)
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - char_overlap)
    return [p for p in pieces if p]
