"""
中间件层
"""
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.middleware.operation_log import OperationLogMiddleware
from app.middleware.trace import TraceMiddleware


def init_middlewares(app: FastAPI, debug: bool = False, prometheus_enabled: bool = False):
    """
    注册所有中间件

    注意：中间件注册顺序影响执行顺序（后注册的先执行）
    当前顺序：CORS -> OperationLog -> Trace -> [Prometheus] -> 业务逻辑

    Args:
        app: FastAPI 应用实例
        debug: 是否为调试模式（影响 CORS 配置）
        prometheus_enabled: 是否启用 Prometheus 指标采集
    """
    # TraceID 中间件（必须在操作日志中间件之前，确保日志中包含 trace_id）
    app.add_middleware(TraceMiddleware)

    # 操作日志中间件
    app.add_middleware(OperationLogMiddleware)

    # Prometheus 指标采集中间件（如果启用）
    if prometheus_enabled:
        from starlette_exporter import PrometheusMiddleware
        app.add_middleware(
            PrometheusMiddleware,
            app_name="dehaze-python",
            prefix="dehaze",
            group_paths=True,
            skip_paths=["/health", "/health/db", "/health/redis",
                        "/metrics", "/docs", "/redoc", "/openapi.json"],
        )

    # CORS 中间件（开发环境也需要限制来源，防止 CSRF）
    _cors_origins = [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite 默认端口
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ] if debug else [
        "http://localhost:3000",
        "http://localhost:8080",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    )


__all__ = [
    'OperationLogMiddleware',
    'TraceMiddleware',
    'init_middlewares',
]
