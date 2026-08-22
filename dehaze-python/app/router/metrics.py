import os

from fastapi import APIRouter, Request
from fastapi.responses import Response
from starlette_exporter import handle_metrics

router = APIRouter(tags=["监控"])


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    """
    Prometheus 指标采集端点（内网免鉴权，通过网络层隔离保障安全）

    返回 Prometheus 格式的指标数据，包括：
    - HTTP 请求量（http_requests_total）
    - HTTP 请求延迟（http_request_duration_seconds）
    - 自定义业务指标（GPU 利用率、模型推理耗时等）

    多 Worker 模式下（设置了 PROMETHEUS_MULTIPROC_DIR），
    通过 MultiProcessCollector 聚合所有 Worker 的指标。
    """
    # 多进程模式：使用 MultiProcessCollector 聚合所有 Worker 的指标
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            generate_latest,
            multiprocess,
        )

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    # 单进程模式：直接使用 starlette_exporter 的默认实现
    return handle_metrics(request)
