"""快速登录获取 session。

用法：
    python scripts/login.py [--backend java|go|python] [--user admin]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import auth


def main() -> None:
    parser = argparse.ArgumentParser(description="登录获取 session_id")
    parser.add_argument("--backend", "-b", default="java", choices=["java", "go", "python"])
    parser.add_argument("--user", "-u", default="admin")
    parser.add_argument("--password", "-p", default="Dehaze2026")
    args = parser.parse_args()

    session_id = auth.login(username=args.user, password=args.password, backend=args.backend)
    print(f"backend: {args.backend}")
    print(f"user:    {args.user}")
    print(f"session: {session_id}")


if __name__ == "__main__":
    main()
