from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.ai_conversation import (
    AiModelCreate,
    AiModelPageQuery,
    AiModelResult,
    AiModelUpdate,
)
from app.models.schema.common import PageResult
from app.service.ai_model_service import AiModelService

router = APIRouter(prefix="/api/v1/ai/models", tags=["AI对话"])


@router.get("", response_model=Result[PageResult[AiModelResult]], summary="模型分页列表")
@require_permission("ai:model:manage")
async def list_models(
    query: AiModelPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await AiModelService.list_models(db, query.pageNum, query.pageSize, query.keyword)
    return success(result)


@router.get("/enabled", response_model=Result[list[AiModelResult]], summary="启用模型列表")
async def list_enabled_models(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await AiModelService.list_enabled_models(db, redis, user.id)
    return success(result)


@router.post("", response_model=Result[AiModelResult], summary="新增模型")
@require_permission("ai:model:manage")
async def create_model(
    form: AiModelCreate,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await AiModelService.create_model(db, redis, form)
    return success(result)


@router.put("/{model_id}", response_model=Result[AiModelResult], summary="更新模型")
@require_permission("ai:model:manage")
async def update_model(
    model_id: str,
    form: AiModelUpdate,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await AiModelService.update_model(db, redis, model_id, form)
    return success(result)


@router.delete("/{model_id}", response_model=Result[None], summary="删除模型")
@require_permission("ai:model:manage")
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await AiModelService.delete_model(db, redis, model_id)
    return success(msg="一切ok")
