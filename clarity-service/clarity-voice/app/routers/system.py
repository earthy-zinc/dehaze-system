"""系统端点：能力发现、引擎状态、探针与指标。

鉴权由全局 BearerAuthMiddleware 承担，本模块不含免鉴权例外。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette.responses import Response

from app import catalog
from app.engines import asr as asr_engine
from app.engines import tts as tts_engine
from app.metrics import metrics_response

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    """liveness：进程存活即 200，不触发任何模型加载或下载"""
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> Response:
    """readiness：引擎依赖可加载即就绪；模型懒加载不阻塞就绪"""
    problems = [
        error for error in (asr_engine.dependency_error(), tts_engine.dependency_error()) if error
    ]
    if problems:
        return JSONResponse({"code": "NOT_READY", "message": "; ".join(problems)}, status_code=503)
    return JSONResponse({"status": "ready"})


@router.get("/v1/models")
async def models() -> dict[str, list[dict[str, Any]]]:
    """能力发现：引擎与已加载/已缓存模型、音色、格式与采样率、服务端目录"""
    return {"data": [asr_engine.capability(), tts_engine.capability(), _catalog_capability()]}


def _catalog_capability() -> dict[str, Any]:
    """模型/音色目录（事实源在服务侧，编排侧不再下发运行时工件）"""
    snapshot = catalog.catalog_snapshot()
    return {"id": "catalog", "type": "catalog", "catalog": snapshot}


@router.get("/v1/status")
async def status() -> dict[str, Any]:
    """引擎状态：模型是否加载、当前模型/音色与推理设备"""
    return {
        "asr": {
            "loaded": bool(asr_engine.loaded_models()),
            "model_id": asr_engine.loaded_model_id(),
            "device": asr_engine.device(),
        },
        "tts": {
            "loaded": tts_engine.loaded_voice_id() is not None,
            "voice_id": tts_engine.loaded_voice_id(),
            "device": tts_engine.device(),
        },
    }


@router.get("/metrics")
async def metrics() -> Response:
    return metrics_response()
