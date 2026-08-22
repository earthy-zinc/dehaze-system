"""外部 A2A 端点管理路由（AI对话 - 智能体管理）

端点前缀 /api/v1/ai/a2a/endpoints：
- 外部 A2A Agent 端点 CRUD（注册/更新/删除/分页查询）
- Agent Card 手动刷新

权限：ai:agent:manage
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.ai_agent import EndpointCreate, EndpointResult, EndpointUpdate
from app.models.schema.common import PageResult
from app.service.ai_agent_endpoint_service import ai_agent_endpoint_service

router = APIRouter(prefix="/api/v1/ai/a2a/endpoints", tags=["AI对话-A2A端点管理"])


@router.post("", response_model=Result[EndpointResult], summary="注册外部A2A端点")
@require_permission("ai:agent:manage")
async def create_endpoint(
    form: EndpointCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_agent_endpoint_service.create_endpoint(db, form)
    return success(EndpointResult.model_validate(result))


@router.patch("/{endpoint_id}", response_model=Result[EndpointResult], summary="更新端点")
@require_permission("ai:agent:manage")
async def update_endpoint(
    endpoint_id: int,
    form: EndpointUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_agent_endpoint_service.update_endpoint(db, endpoint_id, form)
    return success(EndpointResult.model_validate(result))


@router.delete("/{endpoint_id}", response_model=Result[None], summary="删除端点")
@require_permission("ai:agent:manage")
async def delete_endpoint(
    endpoint_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_agent_endpoint_service.delete_endpoint(db, endpoint_id)
    return success(msg="一切ok")


@router.get("", response_model=Result[PageResult[EndpointResult]], summary="端点分页列表")
@require_permission("ai:agent:manage")
async def list_endpoints(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    keyword: str | None = Query(default=None),
    status: int | None = Query(default=None, ge=0, le=1),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result, total = await ai_agent_endpoint_service.list_endpoints(
        db, pageNum, pageSize, keyword=keyword, status=status
    )
    return success(PageResult(list=[EndpointResult.model_validate(e) for e in result], total=total))


@router.post(
    "/{endpoint_id}/refresh-card", response_model=Result[dict], summary="刷新端点Agent Card"
)
@require_permission("ai:agent:manage")
async def refresh_agent_card(
    endpoint_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    card = await ai_agent_endpoint_service.refresh_agent_card(db, endpoint_id)
    return success(card)
