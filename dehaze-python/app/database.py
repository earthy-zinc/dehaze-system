import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger(__name__)

# 异步引擎
# pool_pre_ping: 连接借出前先 ping 一次，避免 MySQL 重启或空闲断开后报错
# pool_timeout: 从池中获取连接的最大等待秒数，避免无限阻塞
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_recycle=settings.DATABASE_POOL_RECYCLE,
    pool_pre_ping=True,
    pool_timeout=10,
    echo=settings.DATABASE_ECHO,
)

# 异步 Session 工厂
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类"""

    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入：获取数据库 Session（自动事务管理）

    事务边界 = 请求边界：
    - 请求正常完成 → 自动 commit
    - 请求抛出异常 → 自动 rollback
    - Service/Repository 层无需手动 commit，只需 flush() 获取 ID
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """上下文管理器：获取数据库 Session
    用于非 FastAPI 场景，如后台任务、消息消费者

    事务边界 = with 块边界：
    - with 块正常退出 → 自动 commit
    - with 块抛出异常 → 自动 rollback
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """初始化数据库连接池"""
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            logger.info("数据库连接成功")
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise RuntimeError(f"数据库不可用，服务启动失败: {e}") from e


async def close_db():
    """关闭数据库连接池"""
    await engine.dispose()
