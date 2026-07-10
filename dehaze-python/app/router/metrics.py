from fastapi import APIRouter, Request
from starlette_exporter import handle_metrics

router = APIRouter(tags=["监控"])


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    """
    Prometheus 指标采集端点

    返回 Prometheus 格式的指标数据，包括：
    - HTTP 请求量（http_requests_total）
    - HTTP 请求延迟（http_request_duration_seconds）
    - 自定义业务指标（GPU 利用率、模型推理耗时等）
    """
    return handle_metrics(request)
