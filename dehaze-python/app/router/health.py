from app.config import settings
from app.database import engine
from app.dependencies.redis import check_redis_health
from fastapi import APIRouter
from sqlalchemy import text

router = APIRouter(prefix="/health", tags=["健康检查"])


@router.get("")
async def health_check():
    """
    基础健康检查

    用于负载均衡、K8s 探针等场景
    """
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@router.get("/db")
async def health_db():
    """
    数据库连接检查
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))

    return {"status": "healthy", "database": "connected"}


@router.get("/redis")
async def health_redis():
    """
    Redis 连接检查

    返回 Redis 连接状态和延迟信息
    """
    health_status = await check_redis_health()

    if health_status.healthy:
        return {
            "status": "healthy",
            "redis": "connected",
            "latency_ms": health_status.latency_ms,
        }
    else:
        return {
            "status": "unhealthy",
            "redis": "disconnected",
            "message": health_status.message,
        }
