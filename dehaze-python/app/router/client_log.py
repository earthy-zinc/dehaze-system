"""前端日志接收路由。

接收前端 SDK 批量上报的日志，复用全局 RateLimitMiddleware（按 path+ip 限流），
已登录用户从会话解析 user_id 注入，未登录允许匿名上报（仅 ERROR 且必须带 trace_id），
以 NDJSON 形式写入 logs/{yyyy-MM-dd}/client.log 供 filebeat 采集。

字段规范见 dehaze-doc/docs/02-系统架构/07-日志架构设计.md §3.5。
"""

import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, Field

from app.core.code import ResultCode
from app.core.result import error, success
from app.dependencies.auth import UserContext, get_current_user_optional
from app.infrastructure.logging import get_client_logger

router = APIRouter(prefix="/api/v1/logs", tags=["前端日志"])

MAX_BATCH_SIZE = 50
MAX_MESSAGE_LENGTH = 2000
MAX_ERROR_STACK_LENGTH = 8000

logger = logging.getLogger(__name__)


class ClientLogEntry(BaseModel):
    """前端单条日志条目（见 07-日志架构设计.md §3.5.2）。"""

    timestamp: Optional[str] = None
    level: Optional[str] = None
    message: Optional[str] = None
    app: Optional[str] = None
    app_version: Optional[str] = None
    url: Optional[str] = None
    user_agent: Optional[str] = None
    trace_id: Optional[str] = None
    error_type: Optional[str] = None
    error_source: Optional[str] = None
    error_stack: Optional[str] = None
    method: Optional[str] = None
    path: Optional[str] = None
    status: Optional[int] = None
    duration: Optional[float] = None
    code: Optional[str] = None
    type: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    navigation_type: Optional[str] = None
    resource_url: Optional[str] = None


class ClientLogBatch(BaseModel):
    """前端日志批量上报请求体（单次最多 50 条）。"""

    logs: list[ClientLogEntry] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


def _truncate(value: Optional[str], max_length: int) -> Optional[str]:
    if value is None:
        return None
    return value if len(value) <= max_length else value[:max_length]


def _is_error(level: Optional[str]) -> bool:
    return (level or "").upper() == "ERROR"


def _write_entry(entry: ClientLogEntry, user_id: Optional[int]) -> None:
    trace_id = (entry.trace_id or "").strip()

    # 匿名仅允许上报 ERROR 且必须携带 trace_id，否则丢弃该条，避免被滥用刷日志
    if user_id is None and (not _is_error(entry.level) or not trace_id):
        return

    fields = {
        key: value
        for key, value in {
            # 不注入前端 timestamp：ClientLogFormatter 已输出服务端接收时间的 timestamp，避免同键冲突
            "app": entry.app,
            "app_version": entry.app_version,
            "url": entry.url,
            "user_agent": entry.user_agent,
            "error_type": entry.error_type,
            "error_source": entry.error_source,
            "error_stack": _truncate(entry.error_stack, MAX_ERROR_STACK_LENGTH),
            "method": entry.method,
            "path": entry.path,
            "code": entry.code,
            "type": entry.type,
            "metric_name": entry.metric_name,
            "navigation_type": entry.navigation_type,
            "resource_url": entry.resource_url,
            "trace_id": trace_id,
        }.items()
        # 与 Java isNotBlank / Go TrimSpace 对齐：过滤 None 和纯空白字符串
        if value is not None and (not isinstance(value, str) or value.strip() != "")
    }
    if entry.status is not None:
        fields["status"] = entry.status
    if entry.duration is not None:
        fields["duration"] = entry.duration
    if entry.metric_value is not None:
        fields["metric_value"] = entry.metric_value
    if user_id is not None:
        fields["user_id"] = user_id

    client_logger = get_client_logger()
    message = _truncate(entry.message, MAX_MESSAGE_LENGTH) or ""
    level = (entry.level or "INFO").upper()
    if level == "ERROR":
        client_logger.error(message, extra={"client_fields": fields})
    elif level == "WARN":
        client_logger.warning(message, extra={"client_fields": fields})
    else:
        client_logger.info(message, extra={"client_fields": fields})


@router.post("/client", summary="前端日志批量上报")
async def collect_client_logs(
    body: ClientLogBatch = Body(...),
    user: Optional[UserContext] = Depends(get_current_user_optional),
):
    try:
        user_id = user.id if user else None
        for entry in body.logs:
            _write_entry(entry, user_id)
    except Exception:  # noqa: BLE001 - 落盘失败不应影响主流程，仅记录后返回失败
        logger.exception("前端日志落盘失败")
        return error("日志写入失败", ResultCode.SYSTEM_EXECUTION_ERROR.code)
    return success()
