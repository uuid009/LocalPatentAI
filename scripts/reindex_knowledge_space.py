#!/usr/bin/env python3
"""按指定知识空间 ID 重建索引（规格 3.3 & 5.8）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localpatentai.db.session import get_session_factory
from localpatentai.knowledge.service import KnowledgeService


def main() -> None:
    parser = argparse.ArgumentParser(description="重建指定知识空间内全部文档的检索索引")
    parser.add_argument("--user-id", required=True, help="执行操作的用户 UUID")
    parser.add_argument("--knowledge-space-id", required=True, help="目标知识空间 UUID")
    args = parser.parse_args()

    with get_session_factory()() as db:
        service = KnowledgeService(db)
        result = service.reindex_knowledge_space(
            user_id=args.user_id,
            knowledge_space_id=args.knowledge_space_id,
        )
        print("Reindex completed:", result)


if __name__ == "__main__":
    main()
