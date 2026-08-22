from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import settings
from app.database import engine
from app.dependencies.redis import check_redis_health
from app.infrastructure.mq.connection import get_consumer, get_publisher

router = APIRouter(prefix="/health", tags=["健康检查"])
ready_router = APIRouter(tags=["健康检查"])


@router.get("")
async def health_check():
    """
    Liveness 探针 - 进程存活检查

    始终返回 200，仅表示进程正在运行。
    用于负载均衡、K8s liveness 探针等场景。
    """
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@ready_router.get("/ready")
async def readiness_check():
    """
    Readiness 探针 - 就绪检查

    检查 DB/Redis/RabbitMQ 依赖，任一不可用返回 503。
    用于 K8s readiness 探针，不可用时从负载均衡摘除。
    """
    components = {}
    all_healthy = True

    # DB check
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        components["db"] = "UP"
    except Exception:
        components["db"] = "DOWN"
        all_healthy = False

    # Redis check
    try:
        health_status = await check_redis_health()
        if health_status.healthy:
            components["redis"] = "UP"
        else:
            components["redis"] = "DOWN"
            all_healthy = False
    except Exception:
        components["redis"] = "DOWN"
        all_healthy = False

    # RabbitMQ check（仅当启用时检查）
    if settings.RABBITMQ_ENABLED:
        publisher = get_publisher()
        consumer = get_consumer()
        pub_ok = publisher is not None and publisher.is_connected
        con_ok = consumer is not None and consumer.is_connected
        if not pub_ok or not con_ok:
            components["rabbitmq"] = "DOWN"
            all_healthy = False
        else:
            components["rabbitmq"] = "UP"

    status = "UP" if all_healthy else "DOWN"
    status_code = 200 if all_healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": status, "components": components},
    )
