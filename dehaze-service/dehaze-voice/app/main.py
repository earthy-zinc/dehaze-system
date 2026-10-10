"""dehaze-voice 服务入口：python -m app.main（默认端口 8993）。

语音推理引擎服务只做"引擎参数进、识别/合成结果出"：不连接 MySQL/Redis/ES/MinIO，
不感知用户身份、并发上限、计费与会话态（编排侧职责）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.config import settings
from app.errors import register_exception_handlers
from app.logging import setup_logging
from app.middleware import TraceContextMiddleware
from app.routers import asr, audio, system
from app.security import BearerAuthMiddleware

# 固定 logger 名：`python -m app.main` 下模块 __name__ 为 __main__，检索启动/停止日志需稳定名
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    # 日志在被服务的那份 app 实例上初始化（`python -m app.main` 会同时以 __main__
    # 与 app.main 执行一次本模块，放在 lifespan 可避免重复初始化）
    setup_logging()
    logger.info("语音推理引擎服务启动 host=%s port=%s", settings.HOST, settings.PORT)
    yield
    logger.info("语音推理引擎服务停止")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        # 内网服务：不开放 Swagger/OpenAPI 文档面（对齐 dehaze-llm）
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    # 鉴权在 ASGI 层短路（含探针与 /metrics）；TraceContext 在外层，鉴权失败的请求同样被记账
    app.add_middleware(BearerAuthMiddleware)
    app.add_middleware(TraceContextMiddleware)
    register_exception_handlers(app)
    app.include_router(system.router)
    app.include_router(asr.router)
    app.include_router(audio.router)
    return app


app = create_app()


if __name__ == "__main__":
    # 单进程部署：引擎模型每进程一份，多实例扩容靠起多个服务实例而非 uvicorn 多 worker，
    # 否则又回到"每 Worker 一份模型"的拆分前状态。ACCESS 日志由中间件自记，故关闭 uvicorn 访问日志
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="warning", access_log=False)
