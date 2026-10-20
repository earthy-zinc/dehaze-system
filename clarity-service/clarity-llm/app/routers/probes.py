"""探针与指标端点：/health 存活、/ready 就绪、/metrics 抓取。

只读判定，不触发任何下载或模型加载；鉴权由全局 BearerAuthMiddleware 承担。
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from app import inference
from app.metrics import metrics_response
from app.model_store import CHAT_MODEL_ID

router = APIRouter(tags=["probes"])


@router.get("/health")
async def health() -> dict:
    """liveness：进程存活恒 200；纯只读，不触发任何模型加载"""
    return {
        "status": "ok",
        "model": CHAT_MODEL_ID,
        "loaded": inference.model_status()["chat"],
    }


@router.get("/ready")
async def ready() -> JSONResponse:
    """readiness：对话与向量模型预热完成（可立即服务）返回 200，否则 503。

    只读判定，禁止触发下载/加载；重排属可选能力，不参与就绪判定。
    """
    status = inference.model_status()
    components = {
        "chat": "UP" if status["chat"] else "DOWN",
        "embedding": "UP" if status["embedding"] else "DOWN",
        "rerank": "UP" if status["rerank"] else "DOWN",
    }
    ok = components["chat"] == "UP" and components["embedding"] == "UP"
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not_ready", "components": components},
    )


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus 抓取端点；显式路由而非 ASGI 子应用挂载，避免 /metrics/ 307 重定向"""
    return metrics_response()
