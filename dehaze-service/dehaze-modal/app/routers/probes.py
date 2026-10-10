"""探针端点：/health 存活、/ready 就绪。

只读判定：不触发任何下载或模型加载。
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.errors import ModalError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["probes"])


@router.get("/health")
async def health() -> dict:
    """liveness：进程存活恒 200"""
    return {"status": "ok", "service": settings.APP_NAME}


@router.get("/ready")
async def ready() -> dict:
    """readiness：设备与权重来源可用返回 200，否则 503"""
    device = _device()
    cache_dir = Path(settings.MODEL_CACHE_DIR)
    if not cache_dir.is_dir() or not os.access(cache_dir, os.W_OK):
        raise ModalError("SERVICE_UNAVAILABLE", f"权重缓存目录不可写: {cache_dir}")
    return {
        "status": "ready",
        "service": settings.APP_NAME,
        "device": device,
        "weight_source": {"cache_dir": str(cache_dir), "base_url": settings.MODEL_BASE_URL},
    }


def _device() -> str:
    try:
        from algorithm.config import Config
    except ImportError as e:
        raise ModalError("SERVICE_UNAVAILABLE", f"推理依赖不可用: {e}") from None
    return Config.DEVICE
