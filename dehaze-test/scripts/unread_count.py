"""查询未读消息数（API + DB 双重验证）。

用法：
    python scripts/unread_count.py [--user admin] [--backend java]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import auth, api, mysql, config


def main() -> None:
    parser = argparse.ArgumentParser(description="查未读消息数（API + DB 双验证）")
    parser.add_argument("--backend", "-b", default="java", choices=["java", "go", "python"])
    parser.add_argument("--user", "-u", default="admin")
    args = parser.parse_args()

    # 1. 登录
    session_id = auth.login(username=args.user, backend=args.backend)
    user_id = auth.get_user_id(args.user)
    print(f"user: {args.user} (id={user_id})")
    print(f"backend: {args.backend}")
    print(f"session: {session_id}")

    # 2. 从 API 查
    try:
        resp = api.get("/api/v1/messages/unread-count", backend=args.backend)
        api_count = resp["data"]["count"]
        print(f"\nAPI 未读数: {api_count}")
    except Exception as e:
        print(f"\nAPI 查询失败: {e}")
        api_count = None

    # 3. 从 DB 查
    try:
        db_count = mysql.get_unread_message_count(user_id)
        print(f"DB  未读数: {db_count}")
    except Exception as e:
        print(f"DB 查询失败: {e}")
        db_count = None

    # 4. 一致性检查
    if api_count is not None and db_count is not None:
        if api_count == db_count:
            print(f"\n✓ API 与 DB 一致（{api_count}）")
        else:
            print(f"\n✗ API 与 DB 不一致！API={api_count}, DB={db_count}")


if __name__ == "__main__":
    main()
