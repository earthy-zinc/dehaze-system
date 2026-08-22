"""
中间件层
"""

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.config import settings
from app.middleware.anti_repeat import AntiRepeatMiddleware
from app.middleware.ip_blacklist import IPBlacklistMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.trace import TraceMiddleware


def init_middlewares(app: FastAPI, debug: bool = False, prometheus_enabled: bool = False):
    """
    注册所有中间件

    注意：Starlette 的 add_middleware 将新中间件插到链首，故**后注册的越靠外层（越先执行）**。
    执行顺序（请求进入时）：
    CORS → Prometheus → RateLimit → AntiRepeat → IPBlacklist → Trace → RequestLog →
    DBSession → ApiKeyAuth → 业务逻辑

    Trace 必须在 RequestLog 之外执行，先注入请求上下文（trace_id/method/path），
    否则访问日志拿不到 trace_id（RequestLog 须在 Trace 之内才能读到上下文）。

    Args:
        app: FastAPI 应用实例
        debug: 是否为调试模式（影响 CORS 配置）
        prometheus_enabled: 是否启用 Prometheus 指标采集
    """
    # API Key 认证中间件（先注册使其在 DBSession 之后执行，才能读到 request.state.db）
    from app.middleware.api_key_auth import ApiKeyAuthMiddleware

    app.add_middleware(ApiKeyAuthMiddleware)

    # 数据库事务中间件（最内层，响应发送前 commit/rollback）
    from app.middleware.db import DBSessionMiddleware

    app.add_middleware(DBSessionMiddleware)

    # 请求级访问日志中间件（须在 TraceMiddleware 之内执行，才能拿到请求上下文）
    from app.middleware.request_log import RequestLogMiddleware

    app.add_middleware(RequestLogMiddleware)

    # TraceID 中间件（在访问日志之外执行，先注入 trace_id/method/path）
    app.add_middleware(TraceMiddleware)

    # IP 黑名单中间件
    app.add_middleware(IPBlacklistMiddleware)

    # 防重复提交中间件（仅对 POST/PUT/DELETE 生效）
    app.add_middleware(AntiRepeatMiddleware)

    # 限流中间件（基于 IP + 路径的固定窗口限流）
    app.add_middleware(RateLimitMiddleware)

    # Prometheus 指标采集中间件（如果启用）
    # prefix="http" 使指标名为 http_requests_total / http_request_duration_seconds，
    # 与 Grafana 面板及三端监控规范对齐（starlette_exporter 默认前缀为 starlette）
    if prometheus_enabled:
        from starlette_exporter import PrometheusMiddleware

        app.add_middleware(
            PrometheusMiddleware,
            app_name="dehaze-python",
            prefix="http",
            group_paths=True,
            skip_paths=["/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"],
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
    "AntiRepeatMiddleware",
    "IPBlacklistMiddleware",
    "RateLimitMiddleware",
    "TraceMiddleware",
    "init_middlewares",
]
