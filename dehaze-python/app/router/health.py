import asyncio
import logging

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

logger = logging.getLogger(__name__)

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


async def _probe(name: str, checker, timeout: float = 5) -> bool:
    """带超时地探测某个基础设施，失败返回 False（不抛出，避免拖垮探针）。

    探针本身不因单个依赖故障而中断，但失败原因必须留痕——否则就绪探针只报
    DOWN 而无从定位是哪个依赖、因何失败。
    """
    try:
        return bool(await asyncio.wait_for(checker(), timeout=timeout))
    except Exception as exc:
        logger.warning("就绪探针 %s 失败: %s", name, exc, exc_info=True)
        return False


def _llm_ready_sync() -> bool:
    """只读探测 dehaze-llm 的 /ready（对话与向量模型预热完成才 200）。

    禁止在此触发模型下载或进程拉起：模型由该服务自身预热，探测只反映事实。
    """
    try:
        resp = httpx.get(
            f"{settings.LOCAL_LLM_BASE_URL.rstrip('/')}/ready",
            headers={"Authorization": f"Bearer {settings.INTERNAL_SERVICE_KEY}"},
            timeout=2,
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.warning("dehaze-llm /ready 探测失败: %s", exc, exc_info=True)
        return False


async def _local_llm_ready() -> bool:
    return await asyncio.to_thread(_llm_ready_sync)


async def _voice_engine_status() -> dict:
    """ASR/TTS 默认引擎只读状态（内置引擎探测 dehaze-voice，第三方引擎按其厂商协议探活）。

    引擎侧模型懒加载，此处不触发加载，且不参与整体就绪判定；未配置默认引擎或引擎
    不可达时按 offline 上报（状态聚合内部已吞异常，不拖垮探针）。
    """
    from app.service.voice.voice_service_status import voice_service_status

    status: dict[str, str] = {}
    for engine_type in ("asr", "tts"):
        engine = await voice_service_status.engine_status(engine_type)
        status[f"voice_{engine_type}"] = engine.get("engine_status", "offline")
    return status


async def _modal_engine_status() -> dict:
    """图像推理引擎只读状态（外置引擎服务 dehaze-service/dehaze-modal）。

    只探测服务就绪（不触发权重下载），不参与整体就绪判定（设计 §3.3）。
    """
    from app.infrastructure.modal import engine_client

    status = "online" if await engine_client.ready() else "offline"
    return {"modal_inference": status}


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
    except Exception as exc:
        logger.warning("就绪检查：数据库不可用: %s", exc, exc_info=True)
        components["db"] = "DOWN"
        all_healthy = False

    try:
        health_status = await check_redis_health()
        if health_status.healthy:
            components["redis"] = "UP"
        else:
            components["redis"] = "DOWN"
            all_healthy = False
    except Exception as exc:
        logger.warning("就绪检查：Redis 健康检查异常: %s", exc, exc_info=True)
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

    # 内置本地推理服务 dehaze-llm（非必选依赖）：始终探测并展示状态；
    # LOCAL_LLM_REQUIRED=True（本地推理为必选依赖的部署）时不可用才阻断整体就绪
    ok = await _probe("local_llm", _local_llm_ready, timeout=2)
    components["local_llm"] = "UP" if ok else "DOWN"
    if not ok and settings.LOCAL_LLM_REQUIRED:
        all_healthy = False

    # ASR/TTS 与图像推理引擎（外置引擎服务、模型懒加载）：附带只读状态，不阻断整体就绪
    components = {
        **components,
        **await _voice_engine_status(),
        **await _modal_engine_status(),
    }

    status = "UP" if all_healthy else "DOWN"
    status_code = 200 if all_healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": status, "components": components},
    )
