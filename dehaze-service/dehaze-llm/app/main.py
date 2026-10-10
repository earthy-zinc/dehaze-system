"""dehaze-llm 服务入口（`python -m app.main`，与三端后端一致的 app 包启动约定）。

组装应用：鉴权与请求日志中间件、统一错误信封、routers 下的端点
（/v1/* 能力端点、/health、/ready、/metrics）。全部端点校验 Authorization: Bearer。
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app import inference
from app.config import settings
from app.errors import register_error_handlers
from app.logging import setup_logging
from app.middleware import TraceContextMiddleware
from app.routers import openai_compat, probes
from app.security import BearerAuthMiddleware

# 不用 __name__：`python -m app.main` 下会得到 __main__，三服务统一固定为 app.main
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    # 日志在被服务的那份 app 实例上初始化（`python -m app.main` 会同时以 __main__
    # 与 app.main 执行一次本模块），放在 lifespan 可避免重复初始化
    setup_logging()
    logger.info("启动 %s（%s:%s）", settings.APP_NAME, settings.HOST, settings.PORT)
    # 预热不阻塞进程启动：模型加载与首次下载在后台完成，预热完成前 /ready 返回 503
    warmup = asyncio.create_task(inference.prewarm())
    try:
        yield
    finally:
        warmup.cancel()
        logger.info("dehaze-llm 停止")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    # 中间件顺序：后添加者在外层 —— trace 上下文先于鉴权建立，鉴权失败的 401 同样被记账
    app.add_middleware(BearerAuthMiddleware)
    app.add_middleware(TraceContextMiddleware)
    register_error_handlers(app)
    app.include_router(openai_compat.router)
    app.include_router(probes.router)
    return app


app = create_app()


if __name__ == "__main__":
    # 单进程部署：三个 GGUF 每进程各一份，扩容靠起多个服务实例而非 uvicorn 多 worker；
    # ACCESS 日志由中间件自记，故关闭 uvicorn 访问日志
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="warning", access_log=False)
