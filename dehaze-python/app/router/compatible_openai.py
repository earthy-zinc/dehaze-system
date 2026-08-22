"""OpenAI 兼容 API 路由（F-M08-010）

端点：
- POST /api/v1/chat/completions  对话补全（流式/非流式）
- GET  /api/v1/models            模型列表（OpenAI 格式；携带 x-api-key 时返回 Claude 格式）

认证：Authorization: Bearer dhak_xxx（由 ApiKeyAuthMiddleware 解析为 userId）
治理与审计：对话端点统一由 _run_compatible_call 处理（Key 级预检 + 全路径审计埋点）；
模型列表端点按 Key 白名单过滤。
"""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.service.ai.compatible_api_service import (
    CompatibleApiService,
    _run_compatible_call,
    _run_models_call,
)

router = APIRouter(prefix="/api/v1", tags=["OpenAI兼容API"])


@router.post("/chat/completions", summary="对话补全（OpenAI 兼容）")
async def chat_completions(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return await _run_compatible_call(
        request,
        db,
        user,
        protocol="openai",
        endpoint="chat/completions",
        handler=CompatibleApiService.handle_openai_chat,
    )


@router.get("/models", summary="模型列表（OpenAI/Claude 格式，按认证方式区分）")
async def list_models(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    # 中间件已统一完成 API Key 认证并注入 user_context，这里按认证方式区分返回格式；
    # 统一走 _run_models_call：Key 级配额/RPM 预检 + 白名单过滤 + 审计（§2.3.1 含 models）
    if request.headers.get("x-api-key"):
        return await _run_models_call(
            request,
            db,
            user,
            protocol="claude",
            handler=CompatibleApiService.list_models_claude,
        )
    return await _run_models_call(
        request,
        db,
        user,
        protocol="openai",
        handler=CompatibleApiService.list_models_openai,
    )
