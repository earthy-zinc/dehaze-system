"""
中间件层
"""
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.config import settings
from app.middleware.ip_blacklist import IPBlacklistMiddleware
from app.middleware.operation_log import OperationLogMiddleware
from app.middleware.trace import TraceMiddleware


def init_middlewares(app: FastAPI, debug: bool = False, prometheus_enabled: bool = False):
    """
    注册所有中间件

    注意：中间件注册顺序影响执行顺序（后注册的先执行，即更靠近应用层）
    执行顺序（请求进入时）：CORS → Prometheus → IPBlacklist → OperationLog → Trace → DBSession → 业务逻辑

    Args:
        app: FastAPI 应用实例
        debug: 是否为调试模式（影响 CORS 配置）
        prometheus_enabled: 是否启用 Prometheus 指标采集
    """
    # 数据库事务中间件（最内层，响应发送前 commit/rollback）
    from app.middleware.db import DBSessionMiddleware
    app.add_middleware(DBSessionMiddleware)

    # TraceID 中间件（确保日志和业务代码能获取 trace_id）
    app.add_middleware(TraceMiddleware)

    # 操作日志中间件（纯 ASGI，需在 TraceID 之内以便获取 trace_id）
    app.add_middleware(OperationLogMiddleware)

    # IP 黑名单中间件（在日志之外，黑名单拦截不记录操作日志）
    app.add_middleware(IPBlacklistMiddleware)

    # Prometheus 指标采集中间件（如果启用）
    if prometheus_enabled:
        from starlette_exporter import PrometheusMiddleware
        app.add_middleware(
            PrometheusMiddleware,
            app_name="dehaze-python",
            prefix="dehaze",
            group_paths=True,
            skip_paths=["/health", "/ready",
                        "/metrics", "/docs", "/redoc", "/openapi.json"],
        )

    # CORS 中间件（最外层，从配置读取 Origin 白名单，禁止 "*" + credentials 组合）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
        expose_headers=["X-Trace-Id"],
    )


__all__ = [
    'OperationLogMiddleware',
    'IPBlacklistMiddleware',
    'TraceMiddleware',
    'init_middlewares',
]
