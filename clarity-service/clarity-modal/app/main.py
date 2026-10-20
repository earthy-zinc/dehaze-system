"""clarity-modal 服务入口（`python -m app.main`，端口 8994）。

零 MySQL/Redis/ES/MinIO 依赖，无业务状态，可无状态重启与水平扩容。
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

from app.config import settings
from app.errors import register_exception_handlers
from app.logging import setup_logging
from app.metrics import GPUMetricsCollector, metrics_response
from app.middleware import TraceContextMiddleware
from app.routers import discovery, images, metric, probes
from app.security import BearerAuthMiddleware

# 不用 __name__：`python -m app.main` 下会得到 __main__，三服务统一固定为 app.main
logger = logging.getLogger("app.main")

_gpu_collector: GPUMetricsCollector | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _gpu_collector
    # 日志在被服务的那份 app 实例上初始化（`python -m app.main` 会同时以 __main__
    # 与 app.main 执行一次本模块），放在 lifespan 可避免重复初始化
    setup_logging()
    Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    _gpu_collector = GPUMetricsCollector(collect_interval=settings.GPU_METRICS_INTERVAL_SECONDS)
    await _gpu_collector.start()
    logger.info(
        "clarity-modal 启动 host=%s port=%s cache_dir=%s model_base_url=%s",
        settings.HOST,
        settings.PORT,
        settings.MODEL_CACHE_DIR,
        settings.MODEL_BASE_URL,
    )
    yield
    if _gpu_collector is not None:
        await _gpu_collector.stop()
    logger.info("clarity-modal 停止")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    # 中间件顺序：后添加者在外层 —— trace 上下文先于鉴权建立，401 响应同样带 trace_id 与访问日志
    app.add_middleware(BearerAuthMiddleware)
    app.add_middleware(TraceContextMiddleware)
    register_exception_handlers(app)
    app.include_router(images.router)
    app.include_router(metric.router)
    app.include_router(discovery.router)
    app.include_router(probes.router)

    # 不用 prometheus_client 的 ASGI 子应用挂载：Mount 会把 /metrics 307 重定向到 /metrics/，
    # 徒增一次抓取往返；显式路由同时天然被鉴权中间件覆盖
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return metrics_response()

    return app


app = create_app()


if __name__ == "__main__":
    # 单进程部署：CUDA 上下文与线程池每进程一份，扩容靠起多个服务实例而非 uvicorn 多 worker
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="warning", access_log=False)
