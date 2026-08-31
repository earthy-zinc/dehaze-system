import asyncio

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import settings
from app.database import engine
from app.dependencies.mongo import get_mongo_client
from app.dependencies.redis import check_redis_health
from app.infrastructure.es.es_client import es_client
from app.infrastructure.mq.connection import get_consumer, get_publisher
from app.infrastructure.storage.minio_client import get_minio_client
from app.service.storage.executor import storage_executor

router = APIRouter(prefix="/health", tags=["健康检查"])
ready_router = APIRouter(tags=["健康检查"])


async def _es_ready() -> bool:
    """ES 连通性检查：ping 成功即为就绪"""
    client = await es_client.get_client()
    if client is None:
        return False
    return bool(await client.ping())


async def _mongo_ready() -> bool:
    """MongoDB 连通性检查：ping 命令返回 ok"""
    result = await get_mongo_client().admin.command("ping")
    return result.get("ok") == 1.0


async def _minio_ready() -> bool:
    """MinIO 连通性检查：同步 SDK 走存储线程池，避免阻塞事件循环"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        storage_executor, get_minio_client().bucket_exists, settings.MINIO_BUCKET
    )


async def _probe(name: str, checker) -> bool:
    """带超时地探测某个基础设施，失败返回 False（不抛出，避免拖垮探针）"""
    try:
        return bool(await asyncio.wait_for(checker(), timeout=5))
    except Exception:
        return False


def _llm_http_healthy() -> bool:
    """只读探测本地 LLM 子进程 /health，要求服务就绪且模型已加载"""
    try:
        resp = httpx.get(
            f"http://{settings.LOCAL_LLM_HOST}:{settings.LOCAL_LLM_PORT}/health", timeout=3
        )
        return resp.status_code == 200 and resp.json().get("loaded") is True
    except Exception:
        return False


async def _local_llm_ready() -> bool:
    """本地 LLM 子进程就绪检查：先只读探测，未就绪则尝试拉起（核心依赖）"""
    if _llm_http_healthy():
        return True
    try:
        from app.infrastructure.llm.local.local_llm_manager import ensure_running

        # ensure_running 为同步阻塞（可能拉起子进程/等模型加载），放线程池并限时
        await asyncio.wait_for(asyncio.to_thread(ensure_running), timeout=4)
    except Exception:
        return False
    return _llm_http_healthy()


def _voice_engine_status() -> dict:
    """ASR/TTS 引擎只读状态（进程内引擎，模型懒加载；不触发加载，不阻断就绪）"""
    from app.infrastructure.voice.funasr_engine import engine_status as funasr_status
    from app.infrastructure.voice.piper_tts_engine import engine_status as piper_status

    asr = funasr_status()
    tts = piper_status()
    return {
        "voice_asr": asr["engine_status"],
        "voice_tts": tts["engine_status"],
    }


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

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        components["db"] = "UP"
    except Exception:
        components["db"] = "DOWN"
        all_healthy = False

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

    # Elasticsearch / MongoDB（必选基础设施）
    for name, checker in (("elasticsearch", _es_ready), ("mongodb", _mongo_ready)):
        ok = await _probe(name, checker)
        components[name] = "UP" if ok else "DOWN"
        if not ok:
            all_healthy = False

    # MinIO（仅当默认存储后端为 minio 时检查）
    if settings.FILE_STORAGE_TYPE == "minio":
        ok = await _probe("minio", _minio_ready)
        components["minio"] = "UP" if ok else "DOWN"
        if not ok:
            all_healthy = False

    # 本地 LLM 子进程（核心依赖，默认启用；纯云端 LLM 部署可置 LOCAL_LLM_ENABLED=False 跳过）
    if settings.LOCAL_LLM_ENABLED:
        ok = await _probe("local_llm", _local_llm_ready)
        components["local_llm"] = "UP" if ok else "DOWN"
        if not ok:
            all_healthy = False

    # ASR/TTS 引擎（进程内、懒加载）：附带只读状态，不阻断整体就绪
    for name, status_val in _voice_engine_status().items():
        components[name] = status_val

    status = "UP" if all_healthy else "DOWN"
    status_code = 200 if all_healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": status, "components": components},
    )
