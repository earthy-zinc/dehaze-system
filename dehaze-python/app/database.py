import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Request
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger(__name__)

# 记录每个连接开始执行 SQL 的时间戳，用于计算耗时
_sql_exec_timers: dict[int, float] = {}


def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany) -> None:
    """SQL 执行后输出结构化审计日志。

    正常执行输出 INFO 级（message=SQL），超过阈值则额外输出 WARNING 级
    （message=SLOW_SQL）。请求上下文字段由 JsonFormatter 自动注入。
    """
    conn_id = id(conn)
    start = _sql_exec_timers.pop(conn_id, None)
    duration_ms = round((time.perf_counter() - start) * 1000, 2) if start else 0.0
    sql_logger = logging.getLogger("sql")
    fields = {
        "sql": statement,
        "duration_ms": duration_ms,
        "rows": cursor.rowcount,
    }
    if duration_ms >= settings.SQL_SLOW_THRESHOLD_MS:
        sql_logger.warning(
            "SLOW_SQL", extra={**fields, "threshold_ms": settings.SQL_SLOW_THRESHOLD_MS}
        )
    else:
        sql_logger.info("SQL", extra=fields)


def _register_sql_logging() -> None:
    """注册 SQLAlchemy 事件监听器，以结构化 JSON 输出 SQL 审计日志。"""
    event.listen(engine.sync_engine, "before_cursor_execute",
                 lambda conn, cursor, statement, parameters, context, executemany:
                 _sql_exec_timers.__setitem__(id(conn), time.perf_counter()))
    event.listen(engine.sync_engine, "after_cursor_execute", _after_cursor_execute)


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
)
_register_sql_logging()

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


async def get_db(request: Request) -> AsyncSession:
    """FastAPI 依赖注入：获取数据库 Session

    事务边界由 DBSessionMiddleware 管理（响应发送前 commit/rollback），
    Service/Repository 层无需手动 commit，只需 flush() 获取 ID。
    """
    return request.state.db


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
        logger.error("数据库连接失败: %s", e)
        raise RuntimeError(f"数据库不可用，服务启动失败: {e}") from e


async def close_db():
    """关闭数据库连接池"""
    await engine.dispose()
