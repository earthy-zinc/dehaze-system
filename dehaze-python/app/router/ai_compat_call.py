"""AI 兼容 API 调用审计查询（内部 API）

供登录用户查询自己的兼容端点调用日志，支撑对账与异常排查。
"""


from fastapi import APIRouter, Depends, Query, Request

from app.core.result import success
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.common import PageResult
from app.service.ai.service.compatible_audit import list_calls as list_audit_calls

router = APIRouter(
    prefix="/api/v1/ai/compat",
    tags=["AI兼容调用审计"],
    dependencies=[Depends(get_current_user)],
)


@router.get("/calls", summary="兼容调用审计查询（分页，当前用户）")
async def list_calls(
    request: Request,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    keyId: int | None = Query(default=None),
    model: str | None = Query(default=None),
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
):
    records, total = await list_audit_calls(
        user_id=user.id,
        key_id=keyId,
        model=model,
        start_time=startTime,
        end_time=endTime,
        page=page,
        size=size,
    )
    return success(
        PageResult(
            list=[
                {
                    "id": r.get("_id"),
                    "keyId": r.get("key_id"),
                    "keyPrefix": r.get("key_prefix"),
                    "conversationId": r.get("conversation_id"),
                    "model": r.get("model"),
                    "endpoint": r.get("endpoint"),
                    "protocol": r.get("protocol"),
                    "isStream": r.get("is_stream"),
                    "inputTokens": r.get("input_tokens"),
                    "outputTokens": r.get("output_tokens"),
                    "credits": r.get("credits"),
                    "statusCode": r.get("status_code"),
                    "durationMs": r.get("duration_ms"),
                    "clientIp": r.get("client_ip"),
                    "requestId": r.get("request_id"),
                    "errorMsg": r.get("error_msg"),
                    "createTime": r.get("create_time"),
                }
                for r in records
            ],
            total=total,
        )
    )

