from fastapi import APIRouter, Depends, Query
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
from app.models.schema.ai_model_price import (
    ModelPriceCreateRequest,
    ModelPriceQuery,
    ModelPriceResult,
    ModelPriceUpdateRequest,
)
from app.models.schema.common import PageResult
from app.service.ai_model_price_service import ai_model_price_service
from app.service.ai_model_service import ai_model_service

router = APIRouter(prefix="/api/v1/ai/models", tags=["AI对话"])


@router.get("", response_model=Result[PageResult[AiModelResult]], summary="模型分页列表")
@require_permission("ai:model:manage")
async def list_models(
    query: AiModelPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_service.list_models(
        db, query.pageNum, query.pageSize, query.keyword, query.model_type
    )
    return success(result)


@router.get("/enabled", response_model=Result[list[AiModelResult]], summary="启用模型列表")
async def list_enabled_models(
    model_type: str | None = Query(default=None, alias="modelType", description="模型类型筛选(chat/embedding/rerank)"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_service.list_enabled_models(db, redis, user.id, model_type)
    return success(result)


@router.post("", response_model=Result[AiModelResult], summary="新增模型")
@require_permission("ai:model:manage")
async def create_model(
    form: AiModelCreate,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_service.create_model(db, redis, form)
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
    result = await ai_model_service.update_model(db, redis, model_id, form)
    return success(result)


@router.delete("/{model_id}", response_model=Result[None], summary="删除模型")
@require_permission("ai:model:manage")
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await ai_model_service.delete_model(db, redis, model_id)
    return success(msg="一切ok")


@router.get("/{model_id}/prices", response_model=Result[PageResult[ModelPriceResult]], summary="模型用户售价版本分页列表")
@require_permission("ai:model:manage")
async def list_model_prices(
    model_id: str,
    query: ModelPriceQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    query.model_id = model_id
    result = await ai_model_price_service.list_prices(db, query)
    return success(result)


@router.post("/{model_id}/prices", response_model=Result[ModelPriceResult], summary="新增模型用户售价版本")
@require_permission("ai:model:manage")
async def create_model_price(
    model_id: str,
    form: ModelPriceCreateRequest,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_price_service.create_price(db, form)
    return success(result)


@router.put("/{model_id}/prices/{price_id}", response_model=Result[ModelPriceResult], summary="更新模型用户售价版本")
@require_permission("ai:model:manage")
async def update_model_price(
    model_id: str,
    price_id: int,
    form: ModelPriceUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_price_service.update_price(
        db, price_id, form.model_dump(exclude_unset=True)
    )
    return success(result)


@router.delete("/{model_id}/prices/{price_id}", response_model=Result[None], summary="删除模型用户售价版本")
@require_permission("ai:model:manage")
async def delete_model_price(
    model_id: str,
    price_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_model_price_service.delete_price(db, price_id)
    return success(msg="一切ok")
