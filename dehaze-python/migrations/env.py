"""
Alembic 迁移环境配置

纯 Alembic CLI 模式，适配 FastAPI + SQLAlchemy 2.0 异步引擎
用法：
  alembic revision --autogenerate -m "描述"
  alembic upgrade head
  alembic downgrade -1
"""

import asyncio
import logging
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# 导入所有 Model 以便 autogenerate 能发现表结构变更
# noinspection PyUnresolvedReferences
import app.models  # noqa: F401
from app.database import Base

target_metadata = Base.metadata


def get_database_url() -> str:
    """从应用配置获取同步数据库 URL（Alembic 迁移使用同步连接）"""
    from app.config import settings

    # 将异步驱动替换为同步驱动：aiomysql → pymysql
    url = settings.DATABASE_URL.replace("+aiomysql", "+pymysql")
    return url


def run_migrations_offline() -> None:
    """离线模式：仅生成 SQL 脚本，不连接数据库"""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在已有连接上执行迁移"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """异步在线模式：创建异步引擎并执行迁移"""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """在线模式：连接数据库并执行迁移"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
