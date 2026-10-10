#!/usr/bin/env python3
"""开发库 schema 漂移检测（只读，只报告不自动改库）。

背景：集成测试连的是开发库（根 `.env` 的 `MYSQL_DATABASE`），schema 变更须手工 ALTER 同步；
库落后于 `config/sql/schema/*.sql` 时，缺列会让整套件在 beforeAll 报 500 并大面积 skip，
错误只能**事后**从 `1054 Unknown column` 里反推。本脚本把这件事提前成一条命令。

比对口径：
- 基线 = `config/sql/schema/sys_*.sql`（每文件一张表）。调度中心独立库 `xxl_job.sql` 已移出该目录
  （现位于 `config/sql/schema-external/`），`sys_*.sql` 通配天然不命中、不参与比对。
- 实际 = 目标库 `information_schema.COLUMNS` 的「表 → 列集合」。
- 漂移 = ①DDL 有、库内无的表（缺表）；②DDL 有、库内无的列（缺列）；③库内有、DDL 无的列（多余列）；
  ④库内存在但 DDL 未覆盖的 `sys_*` 表（多余表）。仅比较**列名集合**，不比较类型/默认值/索引。

适用范围：脚本只读。目标库取自根 `.env` 的 `MYSQL_DATABASE`（当前 dev 为 `dehaze_dev`），
脚本只发 SELECT（`information_schema`），不执行任何 DDL/DML；改库仍须人工按
`config/sql/schema/*.sql` 或 `config/sql/migration/*.sql` 处理。

用法：
    dehaze-python/.venv/bin/python scripts/check_schema_drift.py

连接参数读仓库根 `.env` 的 `MYSQL_*`（环境变量优先，与仓库既有约定一致）。
退出码：0 = 无漂移；1 = 存在漂移（打印差异清单）；2 = 环境/用法错误（缺 .env、缺 pymysql、连不上）。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "config" / "sql" / "schema"
ENV_FILE = REPO_ROOT / ".env"
MANAGED_TABLE_PREFIX = "sys_"

CREATE_RE = re.compile(r"CREATE\s+TABLE\s+`(?P<table>[^`]+)`", re.IGNORECASE)
COLUMN_RE = re.compile(r"^\s*`(?P<column>[^`]+)`\s")
ENV_KEYS = ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_USERNAME", "MYSQL_PASSWORD")


class CheckError(Exception):
    """环境/用法错误（退出码 2）。"""


def display_path(path: Path) -> str:
    """展示用路径：仓库内用相对路径，仓库外（自证/临时夹具）用绝对路径。"""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise CheckError(f"找不到 {path}（连接参数从这里读 MYSQL_*）")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_settings() -> dict[str, str]:
    """`.env` 为底、环境变量覆盖（仓库既有约定：环境变量优先于 .env）。"""
    file_values = load_env_file(ENV_FILE)
    settings: dict[str, str] = {}
    for key in ENV_KEYS:
        value = os.environ.get(key) or file_values.get(key)
        if not value:
            raise CheckError(f"缺少连接参数 {key}（{ENV_FILE} 未定义且环境变量为空）")
        settings[key] = value
    return settings


def parse_ddl() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """解析 `config/sql/schema/sys_*.sql`。

    返回 ({表名: {列名: '文件:行号'}}, {表名: '文件:行号'}（建表语句位置，用于缺表定位）)。
    只用 CREATE TABLE 后、首个顶格 `)` 之前的行；顶格 `)` 之后的 PRIMARY KEY / UNIQUE KEY / KEY
    等索引行不以反引号开头，天然被列正则排除。
    """
    tables: dict[str, dict[str, str]] = {}
    table_sources: dict[str, str] = {}
    files = sorted(SCHEMA_DIR.glob("sys_*.sql"))
    if not files:
        raise CheckError(f"{SCHEMA_DIR} 下没有 sys_*.sql，无法取得 schema 基线")
    for path in files:
        text = path.read_text(encoding="utf-8")
        match = CREATE_RE.search(text)
        if not match:
            raise CheckError(f"{path.name}: 未找到 `CREATE TABLE \\`<表名>\\``，schema 基线不可信")
        table = match.group("table")
        if table in tables:
            raise CheckError(f"{path.name}: 表 {table} 已在其它 schema 文件中定义，基线重复")
        base_line = text[: match.end()].count("\n") + 1
        columns: dict[str, str] = {}
        for offset, line in enumerate(text[match.end() :].splitlines(), start=1):
            if line.startswith(")"):
                break
            column = COLUMN_RE.match(line)
            if column:
                columns[column.group("column")] = f"{display_path(path)}:{base_line + offset}"
        if not columns:
            raise CheckError(f"{path.name}: 未解析出任何列，schema 基线不可信")
        tables[table] = columns
        table_sources[table] = f"{display_path(path)}:{base_line}"
    return tables, table_sources


def fetch_db_columns(settings: dict[str, str]) -> dict[str, set[str]]:
    try:
        import pymysql  # 延迟导入：缺依赖时给出明确指引而非 ImportError 堆栈
    except ImportError as exc:
        raise CheckError(
            "缺少 pymysql：请用 dehaze-python 的虚拟环境运行，"
            "如 `dehaze-python/.venv/bin/python scripts/check_schema_drift.py`"
        ) from exc

    database = settings["MYSQL_DATABASE"]
    try:
        conn = pymysql.connect(
            host=settings["MYSQL_HOST"],
            port=int(settings["MYSQL_PORT"]),
            user=settings["MYSQL_USERNAME"],
            password=settings["MYSQL_PASSWORD"],
            database=database,
            connect_timeout=10,
            read_timeout=60,
            charset="utf8mb4",
        )
    except Exception as exc:  # pymysql 各错误类型不一，统一转成可读提示
        raise CheckError(
            f"连接 {settings['MYSQL_HOST']}:{settings['MYSQL_PORT']}/{database} 失败：{exc}"
        ) from exc

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = %s",
                (database,),
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    db_tables: dict[str, set[str]] = {}
    for table, column in rows:
        db_tables.setdefault(table, set()).add(column)
    if not db_tables:
        raise CheckError(f"库 {database} 内没有任何表，拒绝给出「无漂移」结论")
    return db_tables


def main() -> int:
    try:
        settings = resolve_settings()
        ddl, ddl_sources = parse_ddl()
        db_tables = fetch_db_columns(settings)
    except CheckError as exc:
        print(f"环境/用法错误：{exc}", file=sys.stderr)
        return 2

    database = settings["MYSQL_DATABASE"]
    ddl_tables = set(ddl)
    db_names = set(db_tables)
    ddl_column_count = sum(len(columns) for columns in ddl.values())
    db_column_count = sum(len(columns) for columns in db_tables.values())

    print(
        f"目标库 : {settings['MYSQL_HOST']}:{settings['MYSQL_PORT']}/{database}"
        f"（连接参数来自 {ENV_FILE.relative_to(REPO_ROOT)}，环境变量优先；只读）"
    )
    print(f"DDL 基线: config/sql/schema/sys_*.sql —— {len(ddl_tables)} 表 / {ddl_column_count} 列")
    print(f"实际库  : {database} —— {len(db_names)} 表 / {db_column_count} 列")

    missing_tables = sorted(ddl_tables - db_names)
    missing_columns: dict[str, list[str]] = {}
    extra_columns: dict[str, list[str]] = {}
    extra_tables = sorted(
        name for name in db_names - ddl_tables if name.startswith(MANAGED_TABLE_PREFIX)
    )

    for table in sorted(ddl_tables & db_names):
        absent = sorted(ddl[table].keys() - db_tables[table])
        surplus = sorted(db_tables[table] - ddl[table].keys())
        if absent:
            missing_columns[table] = absent
        if surplus:
            extra_columns[table] = surplus

    other_tables = sorted(
        name for name in db_names - ddl_tables if not name.startswith(MANAGED_TABLE_PREFIX)
    )
    drift_count = (
        len(missing_tables)
        + len(extra_tables)
        + sum(len(columns) for columns in missing_columns.values())
        + sum(len(columns) for columns in extra_columns.values())
    )

    if other_tables:
        print(
            f"\n提示（不判为漂移，非 sys_* 命名空间）：库内还有 {len(other_tables)} 张表 —— "
            + ", ".join(other_tables)
        )

    if drift_count == 0:
        print("\n\u2705 无漂移：DDL 与库的表/列集合一致（本脚本只读，未改库）")
        return 0

    print(f"\n\u274c 检测到 {drift_count} 处漂移（只报告不自动改库）：")
    for table in missing_tables:
        print(
            f"  [缺表] {table} —— DDL 建表于 {ddl_sources[table]}；库内不存在（整套建库用例会失败）"
        )
    for table in extra_tables:
        print(f"  [多余表] {table} —— 库内存在、DDL 未定义（DDL 已删表但库未同步？）")
    for table, columns in missing_columns.items():
        for column in columns:
            print(
                f"  [缺列] {table}.{column} —— DDL 定义于 {ddl[table][column]}；"
                "库内缺失（相关查询会 1054 Unknown column，整套件会大面积 500/skip）"
            )
    for table, columns in extra_columns.items():
        for column in columns:
            print(f"  [多余列] {table}.{column} —— 库内存在、DDL 未定义（DDL 已删列但库未同步？）")

    print(
        "\n修法（人工执行，脚本不代劳）：以 `config/sql/schema/*.sql` 为准补 ALTER，"
        "改 DDL 后须同步目标库；测试库由 schema+data 全量重建、无需手工同步。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
