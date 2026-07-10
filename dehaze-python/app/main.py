from fastapi import FastAPI

from app.config import settings
from app.core.exceptions import register_exception_handlers
from app.lifecycle import lifespan
from app.middleware import init_middlewares
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
)

# 注册全局异常处理器
register_exception_handlers(app)

# 注册中间件
init_middlewares(app,
                 debug=settings.DEBUG,
                 prometheus_enabled=settings.PROMETHEUS_ENABLED)

# 注册路由
init_routes(app, prometheus_enabled=settings.PROMETHEUS_ENABLED)

# 注册 WebSocket 路由
app.include_router(ws_router)
