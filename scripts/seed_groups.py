#!/usr/bin/env python3
"""幂等初始化算法组、框架组、云组三个隔离小组及默认知识空间。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localpatentai.db.repository import seed_initial_groups_and_spaces
from localpatentai.db.session import get_session_factory


def main() -> None:
    with get_session_factory()() as session:
        groups = seed_initial_groups_and_spaces(session)
        for code, grp in groups.items():
            print(f"Group ready: code={code} name={grp.name} id={grp.id}")


if __name__ == "__main__":
    main()
