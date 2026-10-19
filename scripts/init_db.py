#!/usr/bin/env python3
"""执行 Alembic 数据库迁移并初始化三小组与知识空间。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alembic import command
from alembic.config import Config

from localpatentai.db.repository import seed_initial_groups_and_spaces
from localpatentai.db.session import get_session_factory


def main() -> None:
    alembic_cfg = Config(str(ROOT / "alembic.ini"))
    command.upgrade(alembic_cfg, "head")
    with get_session_factory()() as session:
        groups = seed_initial_groups_and_spaces(session)
        print("Database migrated and initial groups seeded:", list(groups.keys()))


if __name__ == "__main__":
    main()
