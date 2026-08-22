import logging
from collections.abc import AsyncGenerator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "get_mongo_db",
    "get_mongo_client",
    "close_mongo",
    "init_mongo_indexes",
]

_mongo_client: AsyncIOMotorClient | None = None


def get_mongo_client() -> AsyncIOMotorClient:
    """获取全局 Motor 客户端（单例）"""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(settings.MONGO_URI)
        logger.info("创建 MongoDB 客户端: db=%s", settings.MONGO_DB_NAME)
    return _mongo_client


async def get_mongo_db() -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """FastAPI 依赖注入：获取 MongoDB database"""
    yield get_mongo_client()[settings.MONGO_DB_NAME]


async def init_mongo_indexes() -> None:
    """创建 login_log / audit_log / ai_api_call_log 索引"""
    db = get_mongo_client()[settings.MONGO_DB_NAME]
    await db.login_log.create_index([("user_id", 1), ("create_time", -1)])
    await db.login_log.create_index([("create_time", -1)])
    await db.login_log.create_index([("status", 1)])
    await db.audit_log.create_index([("operator_id", 1), ("create_time", -1)])
    await db.audit_log.create_index([("target_type", 1), ("target_id", 1), ("create_time", -1)])
    await db.audit_log.create_index([("module", 1), ("create_time", -1)])
    # ai_api_call_log：复合索引支撑按 Key/用户/模型筛选对账；
    # TTL 索引按 create_time 30 天自动过期，无需定时清理
    await db.ai_api_call_log.create_index([("key_id", 1), ("create_time", -1)])
    await db.ai_api_call_log.create_index([("user_id", 1), ("create_time", -1)])
    await db.ai_api_call_log.create_index([("model", 1), ("create_time", -1)])
    await db.ai_api_call_log.create_index([("create_time", 1)], expireAfterSeconds=30 * 24 * 3600)
    logger.info("MongoDB 索引创建完成")


async def close_mongo() -> None:
    """关闭 MongoDB 连接"""
    global _mongo_client
    if _mongo_client is not None:
        logger.info("关闭 MongoDB 连接")
        _mongo_client.close()
        _mongo_client = None
