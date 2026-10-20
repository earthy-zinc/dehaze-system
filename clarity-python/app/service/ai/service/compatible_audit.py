"""AI 兼容 API 调用审计（F-M08-010 §2.3.1）

- record_call：同步签名，内部 fire-and-forget 异步写入 MongoDB `ai_api_call_log`。
  审计与业务零耦合：任何异常不外抛，写入失败不影响主流程。
- list_calls：调用日志查询（时间参数解析 + 用户隔离查询），供审计查询端点调用。
"""

from datetime import datetime
from uuid import uuid4

from app.repository.mongo_ai_call_log_repository import mongo_ai_call_log_repository


async def list_calls(
    *,
    user_id: int,
    key_id: int | None = None,
    model: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    page: int = 1,
    size: int = 20,
) -> tuple[list[dict], int]:
    """查询当前用户的兼容端点调用日志（分页）。

    时间参数支持 `%Y-%m-%d %H:%M:%S` / ISO / `%Y-%m-%d` 三种格式，非法格式按无过滤处理。
    """
    return await mongo_ai_call_log_repository.query(
        user_id=user_id,
        key_id=key_id,
        model=model,
        start_time=_parse_datetime(start_time),
        end_time=_parse_datetime(end_time),
        page=page,
        size=size,
    )


def _parse_datetime(value: str | None) -> datetime | None:
    """解析查询参数时间字符串，非法格式返回 None（按无过滤处理）"""
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def record_call(
    *,
    user_id: int | None,
    key_id: int | None,
    key_prefix: str,
    conversation_id: int | None,
    model: str | None,
    endpoint: str,
    protocol: str,
    is_stream: bool,
    status_code: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    credits: float | None = None,
    error_msg: str | None = None,
    request_id: str | None = None,
    client_ip: str = "",
    duration_ms: int = 0,
) -> None:
    """记录一次兼容 API 调用（含 401/403/429/402 被拒调用）。

    key_prefix 传完整前缀（如 dhak_ab3x），不存完整 Key；request_id 缺省自动生成 uuid4。
    """
    mongo_ai_call_log_repository.insert_async(
        user_id=user_id,
        key_id=key_id,
        key_prefix=key_prefix,
        conversation_id=conversation_id,
        model=model,
        endpoint=endpoint,
        protocol=protocol,
        is_stream=is_stream,
        status_code=status_code,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        credits=credits,
        error_msg=error_msg,
        request_id=request_id or str(uuid4()),
        client_ip=client_ip,
        duration_ms=duration_ms,
    )
