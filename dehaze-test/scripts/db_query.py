"""交互式 SQL 查询工具（类似 mysql 命令行）。

用法：
    python scripts/db_query.py "SELECT COUNT(*) FROM sys_message"
    python scripts/db_query.py --database dehaze_test "SHOW TABLES"
    python scripts/db_query.py  # 进入交互模式
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import mysql, config


def run_query(sql: str, database: str | None = None) -> None:
    """执行 SQL 并打印结果。"""
    sql_stripped = sql.strip().rstrip(";").strip()
    if not sql_stripped:
        return

    print(f"→ {sql_stripped}")
    try:
        if sql_stripped.upper().startswith("SELECT") or sql_stripped.upper().startswith("SHOW") \
                or sql_stripped.upper().startswith("DESC"):
            rows = mysql.query(sql_stripped, database=database)
            if not rows:
                print("  (empty)")
                return
            # 打印表头
            cols = list(rows[0].keys())
            print("  " + " | ".join(cols))
            print("  " + "-" * (len(" | ".join(cols))))
            for row in rows:
                print("  " + " | ".join(str(row[c]) for c in cols))
            print(f"\n  {len(rows)} row(s) in set")
        else:
            n = mysql.execute(sql_stripped, database=database)
            print(f"  Query OK, {n} row(s) affected")
    except Exception as e:
        print(f"  ERROR: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL 查询工具")
    parser.add_argument("sql", nargs="?", help="SQL 语句（不提供则进入交互模式）")
    parser.add_argument("--database", "-d", default=None, help="数据库（默认 dehaze）")
    args = parser.parse_args()

    db_info = args.database or config.MYSQL_DATABASE
    print(f"-- dehaze-test SQL shell (database: {db_info})")

    if args.sql:
        run_query(args.sql, args.database)
    else:
        print("-- 输入 SQL 语句，以 ; 结束；输入 exit 退出")
        buffer = ""
        while True:
            try:
                line = input("> " if not buffer else "... ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip().lower() in ("exit", "quit", "\\q"):
                break
            buffer += " " + line
            if line.rstrip().endswith(";"):
                run_query(buffer, args.database)
                buffer = ""


if __name__ == "__main__":
    main()
