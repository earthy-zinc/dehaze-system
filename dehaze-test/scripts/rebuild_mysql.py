"""重建 MySQL 数据库 + 清理 Redis 缓存。

迁移自 scripts/rebuild_mysql.sh，去除 docker 依赖：
- 用 pymysql 直连远程 MySQL
- 用 redis-py 直连远程 Redis

用法：
    python scripts/rebuild_mysql.py [--skip-redis] [--skip-mysql] [--only dehaze|dehaze_test]
    python scripts/rebuild_mysql.py --import sys_ai_provider.sql sys_menu.sql

缓存清理：后端进程存活时走后端统一失效入口（Redis + L1 进程缓存），
未运行时裸删 Redis。

注意：会清空 dehaze / dehaze_test 两个数据库的全部数据！
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本不依赖 PYTHONPATH 也能 import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pymysql

from utils import config, cleanup


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


def _split_statements(content: str) -> list[str]:
    """按 ; 切分 SQL 语句，跳过 -- 行注释与单引号字符串内的分号（如 COMMENT '…a;b…'）。"""
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    content = "\n".join(lines)

    statements: list[str] = []
    buf: list[str] = []
    in_string = False
    i = 0
    n = len(content)
    while i < n:
        ch = content[i]
        if ch == "'":
            buf.append(ch)
            if in_string and i + 1 < n and content[i + 1] == "'":
                buf.append(content[i + 1])  # MySQL 用 '' 转义字符串内的单引号
                i += 1
            else:
                in_string = not in_string
        elif ch == ";" and not in_string:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _exec_sql_file(conn: pymysql.Connection, sql_path: Path) -> int:
    """执行单个 SQL 文件，返回执行语句数。"""
    content = sql_path.read_text(encoding="utf-8")
    statements = _split_statements(content)
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


def import_data_files(targets: list[str], file_names: list[str]) -> None:
    """增量同步指定 data SQL 文件：先清空对应表再导入（种子数据全量同步，不重建库、不影响其他表）。

    适用于菜单/字典/角色等由 SQL 文件全量管理的种子数据；导入前清表避免主键冲突。
    """
    print("\n=== 增量导入 data SQL（先清表再导入，不重建库）===")
    for db in targets:
        print(f"--- {db} ---")
        with _connect_admin(db) as conn:
            for name in file_names:
                table = Path(name).stem
                path = config.SQL_DATA_DIR / name
                if not path.exists():
                    raise FileNotFoundError(f"data SQL 文件不存在: {path}")
                with conn.cursor() as cur:
                    cur.execute("SET FOREIGN_KEY_CHECKS = 0")
                    cur.execute(f"DELETE FROM `{table}`")
                    cur.execute("SET FOREIGN_KEY_CHECKS = 1")
                n = _exec_sql_file(conn, path)
                print(f"  ✓ {name} ({n} stmts)，已清空 `{table}` 后导入")


CACHE_CLEAR_PATH = "/api/v1/cache/clear"


def _cache_backend(backend: str, targets: list[str] | None) -> str | None:
    """仅 dev 库有常驻后端进程（持有 L1 缓存）时返回后端名；测试库无服务直连，返回 None。"""
    if targets is not None and config.MYSQL_DATABASE not in targets:
        return None
    return backend


def _backend_alive(backend: str) -> bool:
    """探测后端进程是否存活（liveness 探针）。

    后端可能正忙（模型加载、批量任务），超时给到 10s，避免误判为未运行而回退裸删。
    """
    base = config.get_backend(backend).base_url
    try:
        with httpx.Client(base_url=base, timeout=10) as client:
            return client.get("/health").status_code < 400
    except Exception:
        return False


def clear_backend_cache(backend: str) -> None:
    """经后端统一失效入口清缓存：除 Redis 外，同步失效各实例的 L1 进程缓存。

    不带 key/pattern：后端按内置业务缓存前缀清单逐项失效。
    不可传 pattern=*，那会连同 session/限流/验证码等基础设施 key 一并扫删。
    """
    from utils import api, auth

    auth.login(backend=backend)  # 接口需管理员会话（X-Session-Id），登录后由 api 自动注入
    result = api.post(CACHE_CLEAR_PATH, backend=backend)
    deleted = sum(item.get("deleted", 0) for item in result.get("data") or [])
    print(f"    失效业务缓存 {deleted} 个 key")


def rebuild_redis(backend: str | None) -> None:
    """清理业务缓存（消息未读数、session、角色权限等）。

    backend 非空且进程存活时走后端统一失效入口（Redis + L1 进程缓存一并失效）；
    否则裸删 Redis，此时 L1 进程缓存只能靠重启后端清除。
    """
    print("\n=== 清理 Redis 缓存 ===")
    if backend and _backend_alive(backend):
        try:
            clear_backend_cache(backend)
            print(f"  ✓ 经后端 {backend} {CACHE_CLEAR_PATH} 统一失效（Redis + L1 进程缓存）")
            return
        except Exception as e:
            print(f"  ! 后端 {backend} 缓存失效失败，回退裸删 Redis: {e}")
    else:
        print("  ! 后端未运行，仅裸删 Redis；L1 进程缓存需重启后端清除")

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
    parser.add_argument(
        "--import",
        dest="import_files",
        nargs="+",
        metavar="FILE",
        help="增量同步 config/sql/data/ 下指定的种子数据 SQL 文件（先清对应表再导入，不重建库不清其他数据），"
        "如：--import sys_menu.sql sys_role_menu.sql",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(config.BACKENDS),
        default="python",
        help="缓存失效调用的后端（dev 库 dehaze 由 dehaze-python 承载，默认 python）",
    )
    args = parser.parse_args()

    if args.import_files:
        targets = [args.only] if args.only else [config.MYSQL_DATABASE]
        import_data_files(targets, args.import_files)
        if not args.skip_redis:
            rebuild_redis(_cache_backend(args.backend, targets))
        print("\n✓ Done")
        return

    if args.skip_mysql and args.skip_redis:
        print("Nothing to do.")
        return

    targets: list[str] | None = None
    if not args.skip_mysql:
        targets = [args.only] if args.only else None
        rebuild_mysql(targets)

    if not args.skip_redis:
        rebuild_redis(_cache_backend(args.backend, targets))

    print("\n✓ Done")


if __name__ == "__main__":
    main()
