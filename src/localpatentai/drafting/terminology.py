"""专利术语一致性、计量单位一致性与未解决占位符检测规则（规格 7.1 Node D & 8.1）。"""

from __future__ import annotations

import re

PLACEHOLDER_PATTERNS = [
    re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b", re.IGNORECASE),
    re.compile(r"\{\{[^}]+\}\}"),
    re.compile(r"\[待填写[^\]]*\]"),
    re.compile(r"<待补充[^>]*>"),
]

# 常见易混技术单位对（若同一草案混用全称/非标简写则提示）
UNIT_CONFLICT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("温度单位", ("摄氏度", "℃", "°C")),
    ("压力单位", ("MPa", "兆帕", "bar")),
    ("时间单位_毫秒", ("ms", "毫秒")),
)

# 禁用绝对化或捏造性断言词汇（当无证据支撑时）
UNVERIFIED_NUMBER_RE = re.compile(r"(?:提升|降低|缩短|提高)了?\s*(\d+(?:\.\d+)?\s*%)")


def find_unresolved_placeholders(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in PLACEHOLDER_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(0)
            if val not in hits:
                hits.append(val)
    return hits


def check_unit_consistency(full_text: str) -> list[str]:
    warnings: list[str] = []
    for group_name, variants in UNIT_CONFLICT_GROUPS:
        present = [v for v in variants if v in full_text]
        if len(present) > 1:
            warnings.append(f"{group_name}存在多种表述混用：{', '.join(present)}，建议统一为标准国际单位符号")
    return warnings


def check_terminology_consistency(
    sections_text: str, claims_text: str, glossary: dict[str, str]
) -> list[str]:
    issues: list[str] = []
    for standard_term in glossary:
        if standard_term and standard_term not in sections_text and standard_term not in claims_text:
            issues.append(f"核心术语“{standard_term}”在术语表中定义但未在正文或权利要求书中出现")
    return issues
