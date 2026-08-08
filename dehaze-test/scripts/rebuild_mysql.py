"""重建 MySQL 数据库 + 清理 Redis 缓存。

迁移自 scripts/rebuild_mysql.sh，去除 docker 依赖：
- 用 pymysql 直连远程 MySQL
- 用 redis-py 直连远程 Redis

用法：
    python scripts/rebuild_mysql.py [--skip-redis] [--skip-mysql] [--only dehaze|dehaze_test]

注意：会清空 dehaze / dehaze_test 两个数据库的全部数据！
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本不依赖 PYTHONPATH 也能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymysql

from utils import config, redis as redis_utils, cleanup


def _connect_admin(database: str | None = None) -> pymysql.Connection:
    """以 root 连接 MySQL（不指定 database 时用于创建库）。"""
    return pymysql.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        database=database,
        autocommit=True,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,  # 重建库时用普通 cursor
        connect_timeout=5,
        read_timeout=60,
        write_timeout=60,
    )


def _exec_sql_file(conn: pymysql.Connection, sql_path: Path) -> int:
    """执行单个 SQL 文件，返回执行语句数。

    SQL 文件中每条语句以 ; 结尾。简化处理：按 ; 分割，跳过空语句和注释。
    """
    content = sql_path.read_text(encoding="utf-8")
    # 去除 -- 行注释
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(line)
    content = "\n".join(lines)

    # 按 ; 分割语句（不处理字符串内的 ;，因为 schema/data 文件没有这种情况）
    statements = [s.strip() for s in content.split(";") if s.strip()]
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    return len(statements)


def rebuild_mysql(targets: list[str] | None = None) -> None:
    """重建 dehaze / dehaze_test 数据库，导入 schema + data。"""
    targets = targets or [config.MYSQL_DATABASE, config.MYSQL_DATABASE_TEST]

    # 1. 用 admin 连接（不指定 database）创建/重建数据库
    print("=== 重建数据库 ===")
    with _connect_admin() as conn:
        with conn.cursor() as cur:
            for db in targets:
                cur.execute(f"DROP DATABASE IF EXISTS `{db}`")
                cur.execute(
                    f"CREATE DATABASE `{db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
                )
                print(f"  ✓ 重建 {db}")

    # 2. 逐个导入 schema
    schema_files = sorted(config.SQL_SCHEMA_DIR.glob("sys_*.sql"))
    print(f"\n=== 导入 Schema（{len(schema_files)} 个文件）===")
    for db in targets:
        print(f"--- {db} ---")
        with _connect_admin(db) as conn:
            for f in schema_files:
                n = _exec_sql_file(conn, f)
                print(f"  ✓ {f.name} ({n} stmts)")

    # 3. 逐个导入 data
    data_files = sorted(config.SQL_DATA_DIR.glob("sys_*.sql"))
    print(f"\n=== 导入 Data（{len(data_files)} 个文件）===")
    for db in targets:
        print(f"--- {db} ---")
        with _connect_admin(db) as conn:
            for f in data_files:
                n = _exec_sql_file(conn, f)
                print(f"  ✓ {f.name} ({n} stmts)")


def rebuild_redis() -> None:
    """清理 Redis 业务缓存（消息未读数、session、角色权限等）。"""
    print("\n=== 清理 Redis 缓存 ===")
    result = cleanup.clear_all()
    for k, v in result.items():
        print(f"  ✓ {k}: 删除 {v} 个 key")


def main() -> None:
    parser = argparse.ArgumentParser(description="重建 MySQL + 清理 Redis 缓存")
    parser.add_argument("--skip-redis", action="store_true", help="跳过 Redis 清理")
    parser.add_argument("--skip-mysql", action="store_true", help="跳过 MySQL 重建")
    parser.add_argument(
        "--only",
        choices=["dehaze", "dehaze_test"],
        help="只重建指定数据库（默认两个都重建）",
    )
    args = parser.parse_args()

    if args.skip_mysql and args.skip_redis:
        print("Nothing to do.")
        return

    if not args.skip_mysql:
        targets = [args.only] if args.only else None
        rebuild_mysql(targets)

    if not args.skip_redis:
        rebuild_redis()

    print("\n✓ Done")


if __name__ == "__main__":
    main()
