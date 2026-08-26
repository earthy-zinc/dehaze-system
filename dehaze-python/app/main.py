from fastapi import FastAPI

from app.config import settings
from app.core.exceptions import register_exception_handlers
from app.lifecycle import lifespan
from app.middleware import init_middlewares
from app.middleware.non_null_response import NonNullJSONResponse
from app.router import init_routes
from app.router.websocket import router as ws_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="图像去雾系统 API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    default_response_class=NonNullJSONResponse,
)

register_exception_handlers(app)

init_middlewares(app, debug=settings.DEBUG, prometheus_enabled=settings.PROMETHEUS_ENABLED)

init_routes(app, prometheus_enabled=settings.PROMETHEUS_ENABLED)

# 注册 WebSocket 路由（消息通知；流式 ASR 已并入语音模块 voice router）
app.include_router(ws_router)


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Dehaze Python 算法服务")
    parser.add_argument("--host", default=settings.SERVER_HOST, help="绑定主机")
    parser.add_argument("--port", type=int, default=settings.SERVER_PORT, help="绑定端口")
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.SERVER_WORKERS,
        help="Worker 进程数（仅生产环境生效）",
    )
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else args.workers,
    )
