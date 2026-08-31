"""
pytest 配置和共享 fixtures

基于 FastAPI + pytest-asyncio 的测试框架。

数据库策略（2026-08-23 决策，替代原 SQLite 内存库方案）：
真实 MySQL 测试库 `dehaze_test`（与开发同实例同版本，零方言漂移——SQLite 方案
因 DECIMAL 精度/外键/锁语义漂移被否决，见 README §5.1）：
- schema/种子数据由 `config/sql/`（schema+data）全量脚本重建，与开发库构建方式
  完全同源（Alembic 迁移链从未在任何库执行过、不完整且在 MySQL 8.4 上存在坏
  迁移，不能用作基线；待其基线化改造后再评估切换）
- session 级 DROP + CREATE + 导入脚本 = 每次运行全量重置
- 测试级数据隔离用外部事务 + SAVEPOINT 回滚：被测代码内部 commit 只释放
  SAVEPOINT，测试结束整体 ROLLBACK，种子数据零污染

Redis 桩见 tests/stubs.py（fakeredis，真实协议实现）；
Mongo 桩见 mongo_db fixture（mongomock-motor）；
router 级测试客户端由各测试文件自建（ai_client 等）。
"""

import os
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置测试环境变量（必须在 import app 之前：pydantic-settings 首次 import 时读取并缓存为单例）
# 注意：不覆盖 MYSQL_PASSWORD——db fixture 需连接真实 MySQL，凭证来自根目录 .env
os.environ["APP_ENV"] = "testing"
# TestingSettings 的类默认值会被根目录 .env 覆盖（dotenv 优先级高于类默认值），
# 此处显式重设以强制测试库名与 Redis 兜底端口（6390 无人监听）
os.environ["MYSQL_DATABASE"] = "dehaze_test"
os.environ["REDIS_PORT"] = "6390"

import fakeredis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import app.database as database_module
import app.dependencies.redis as redis_module
from app.config import PROJECT_ROOT, settings
from app.main import app as fastapi_app

# config/sql 位于仓库根（dehaze-python 的上级目录）
_SQL_SCHEMA_DIR = PROJECT_ROOT / "config" / "sql" / "schema"
_SQL_DATA_DIR = PROJECT_ROOT / "config" / "sql" / "data"


def _split_statements(content: str) -> list[str]:
    """按 ; 分割 SQL 语句，跳过单引号字符串与反引号标识符内的分号。

    （schema 脚本的 COMMENT 字符串含分号，如 sys_ai_agent 的 agent_code 注释，
    朴素按 ; 分割会截断语句。）
    """
    statements: list[str] = []
    buf: list[str] = []
    in_string = False
    in_backtick = False
    i = 0
    while i < len(content):
        ch = content[i]
        if in_string:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(content):
                buf.append(content[i + 1])
                i += 1
            elif ch == "'":
                if i + 1 < len(content) and content[i + 1] == "'":
                    buf.append("'")
                    i += 1
                else:
                    in_string = False
        elif in_backtick:
            buf.append(ch)
            if ch == "`":
                in_backtick = False
        elif ch == "'":
            in_string = True
            buf.append(ch)
        elif ch == "`":
            in_backtick = True
            buf.append(ch)
        elif ch == ";":
            statements.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    statements.append("".join(buf))
    return [s.strip() for s in statements if s.strip()]


def _exec_sql_file(conn, sql_path: Path) -> None:
    """执行单个 SQL 脚本（跳过 -- 行注释；语句分割见 _split_statements）。"""
    content = sql_path.read_text(encoding="utf-8")
    lines = [l for l in content.splitlines() if not l.strip().startswith("--")]
    statements = _split_statements("\n".join(lines))
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)


@pytest.fixture(scope="session")
def _mysql_schema() -> None:
    """session 级全量重建测试库（与开发库同源：config/sql schema+data 脚本）。

    DROP + CREATE DATABASE dehaze_test → 导入全部 schema 与 data 脚本，
    即"每次运行全量重置"；MySQL 不可达或脚本失败 → 直接 fail（fail-fast）。
    """
    import pymysql

    try:
        conn = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USERNAME,
            password=settings.MYSQL_PASSWORD,
            autocommit=True,
            charset="utf8mb4",
            connect_timeout=5,
        )
    except Exception as e:
        pytest.fail(
            f"MySQL 不可达（{settings.MYSQL_USERNAME}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}）：{e}\n"
            f"测试需连接真实 MySQL 测试库 `{settings.MYSQL_DATABASE}`（与开发同实例，"
            f"凭证见根目录 .env 的 MYSQL_HOST/MYSQL_PASSWORD）"
        )

    try:
        # 并发保护：多 pytest 进程同时重建 dehaze_test 会 DROP/CREATE/导入竞态
        # （偶发 Duplicate entry 主键冲突 / 表缺失）。用 MySQL 会话级 GET_LOCK 串行化
        # 重建（同一连接上获取与释放，autocommit 下锁立即生效、不随 DDL 释放）。
        with conn.cursor() as cur:
            cur.execute("SELECT GET_LOCK('dehaze_test_rebuild', 300)")
            if cur.fetchone()[0] != 1:
                pytest.fail("获取测试库重建锁超时（其他 pytest 进程正在重建 dehaze_test）")
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{settings.MYSQL_DATABASE}`")
                cur.execute(
                    f"CREATE DATABASE `{settings.MYSQL_DATABASE}` CHARACTER SET utf8mb4 "
                    "COLLATE utf8mb4_0900_ai_ci"
                )
            conn.select_db(settings.MYSQL_DATABASE)
            for sql_dir in (_SQL_SCHEMA_DIR, _SQL_DATA_DIR):
                for sql_file in sorted(sql_dir.glob("sys_*.sql")):
                    _exec_sql_file(conn, sql_file)
        finally:
            with conn.cursor() as cur:
                cur.execute("SELECT RELEASE_LOCK('dehaze_test_rebuild')")
    except Exception as e:
        pytest.fail(f"测试库 `{settings.MYSQL_DATABASE}` 重建失败：{e}")
    finally:
        conn.close()


@pytest.fixture
async def db(_mysql_schema, monkeypatch: pytest.MonkeyPatch):
    """真实 MySQL 会话（每测试独立事务，结束整体回滚，种子数据零污染）。

    - 外部事务 + SAVEPOINT（SQLAlchemy 2.0 标准测试事务模式）：被测代码内
      `async with get_db_session()` 的 commit 只释放 SAVEPOINT，不真正落库
    - monkeypatch async_session_factory：get_db_session 产出的 session 与本
      fixture 同源（同一连接、同一事务），fixture 预置的数据对被测代码可见
    """
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    connection = await engine.connect()
    transaction = await connection.begin()
    factory = async_sessionmaker(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    monkeypatch.setattr(database_module, "async_session_factory", factory)
    async with factory() as session:
        yield session
    await transaction.rollback()
    await connection.close()
    await engine.dispose()


@pytest.fixture
def mongo_db(monkeypatch: pytest.MonkeyPatch):
    """mongomock-motor 内存 Mongo（motor 异步兼容层）。

    patch `app.dependencies.mongo.get_mongo_client` 模块属性，单点覆盖全部
    使用方（mongo 仓储经模块属性延迟引用；勿 patch `_mongo_client` 私有单例）。
    """
    import mongomock_motor

    import app.dependencies.mongo as mongo_module

    client = mongomock_motor.AsyncMongoMockClient()
    monkeypatch.setattr(mongo_module, "get_mongo_client", lambda: client)
    return client[settings.MONGODB_DATABASE]


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch: pytest.MonkeyPatch) -> fakeredis.FakeAsyncRedis:
    """fakeredis 客户端（真实协议实现），autouse 全局接管 Redis。

    测试环境一律不触达真实 Redis，三层防线：
    1. 本 fixture（autouse 自动生效）：patch 中心入口 `app.dependencies.redis`
       的 get_redis_client/get_redis（覆盖函数内延迟导入的调用方），并动态扫描
       sys.modules 替换所有顶层直接持有这两个函数引用的模块——顶层
       `from app.dependencies.redis import get_redis_client` 持有独立引用，
       仅 patch 中心入口影响不到它们，动态扫描可覆盖后续新增模块。
    2. 需要操作 Redis 数据的测试显式请求本 fixture（fixture 缓存同一实例），
       或自行 `_bind_fake_redis` 绑定独立实例。
    3. `TestingSettings.REDIS_PORT=6390` 指向无人监听端口兜底（防函数别名导入
       等前两层覆盖不到的路径）。勿改用端口 0：redis-py 会把 0 当默认 6379，
       实测会直接连上真实 Redis。
    """
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)

    async def _override():
        return redis

    original_client = redis_module.get_redis_client
    original_get = redis_module.get_redis
    monkeypatch.setattr(redis_module, "get_redis_client", _override)
    monkeypatch.setattr(redis_module, "get_redis", _override)
    for module in tuple(sys.modules.values()):
        if getattr(module, "get_redis_client", None) is original_client:
            monkeypatch.setattr(module, "get_redis_client", _override)
        if getattr(module, "get_redis", None) is original_get:
            monkeypatch.setattr(module, "get_redis", _override)
    return redis


@pytest.fixture
def app():
    """FastAPI 应用实例"""
    return fastapi_app
