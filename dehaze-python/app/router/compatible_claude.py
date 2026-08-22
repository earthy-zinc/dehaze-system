"""Claude 兼容 API 路由（F-M08-010）

端点：
- POST /api/v1/messages  消息对话（流式/非流式）

认证：x-api-key（由 ApiKeyAuthMiddleware 识别 Anthropic 协议并解析为 userId）
治理与审计：统一由 _run_compatible_call 处理（Key 级预检 + 全路径审计埋点）。

说明：GET /api/v1/models 与 OpenAI 兼容路由共用同一路径，为避免路由冲突，
统一在 compatible_openai.list_models 中按认证方式区分格式（携带 x-api-key 返回 Claude 格式）。
"""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.service.ai.compatible_api_service import (
    CompatibleApiService,
    _run_compatible_call,
)

router = APIRouter(prefix="/api/v1", tags=["Claude兼容API"])


@router.post("/messages", summary="消息对话（Claude 兼容）")
async def messages(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return await _run_compatible_call(
        request,
        db,
        user,
        protocol="claude",
        endpoint="messages",
        handler=CompatibleApiService.handle_claude_messages,
    )
