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
from app.service.ai.service.model_test_service import test_model_by_model_id
from app.service.ai_model_price_service import ai_model_price_service
from app.service.ai_model_service import ai_model_service

# model_id 是含 `/` 的厂商业务键（如 Qwen/Qwen3-8B），不能作路径段（会被当路径分隔符），
# 故除无 id 的集合端点外一律以 query 参数 modelId 传递。
router = APIRouter(prefix="/api/v1/ai/models", tags=["AI对话"])


@router.get("", response_model=Result[PageResult[AiModelResult]], summary="模型分页列表")
@require_permission("ai:model:manage")
async def list_models(
    query: AiModelPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_service.list_models(
        db, query.pageNum, query.pageSize, query.keyword, query.model_type, query.task_tags
    )
    return success(result)


@router.get("/enabled", response_model=Result[list[AiModelResult]], summary="启用模型列表")
async def list_enabled_models(
    model_type: str | None = Query(
        default=None, alias="modelType", description="模型类型筛选(见模型类型白名单)"
    ),
    task_tags: str | None = Query(
        default=None, alias="taskTags", description="任务标签筛选(逗号分隔,含任一即命中)"
    ),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_service.list_enabled_models(db, redis, user.id, model_type, task_tags)
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


@router.put("", response_model=Result[AiModelResult], summary="更新模型")
@require_permission("ai:model:manage")
async def update_model(
    form: AiModelUpdate,
    model_id: str = Query(..., alias="modelId", description="模型标识"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_service.update_model(db, redis, model_id, form)
    return success(result)


@router.delete("", response_model=Result[None], summary="删除模型")
@require_permission("ai:model:manage")
async def delete_model(
    model_id: str = Query(..., alias="modelId", description="模型标识"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await ai_model_service.delete_model(db, redis, model_id)
    return success(msg="一切ok")


@router.post("/test", response_model=Result[dict], summary="模型可用性测试")
@require_permission("ai:model:manage")
async def test_model(
    model_id: str = Query(..., alias="modelId", description="模型标识"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """按模型类型发最小真实推理请求验证可用性，结果落库（last_test_*）并返回。"""
    result = await test_model_by_model_id(db, redis, model_id)
    return success(result)


@router.get(
    "/prices",
    response_model=Result[PageResult[ModelPriceResult]],
    summary="模型用户售价版本分页列表",
)
@require_permission("ai:model:manage")
async def list_model_prices(
    model_id: str = Query(..., alias="modelId", description="模型标识"),
    query: ModelPriceQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    query.model_id = model_id
    result = await ai_model_price_service.list_prices(db, query)
    return success(result)


@router.post("/prices", response_model=Result[ModelPriceResult], summary="新增模型用户售价版本")
@require_permission("ai:model:manage")
async def create_model_price(
    form: ModelPriceCreateRequest,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_price_service.create_price(db, form)
    return success(result)


@router.put(
    "/prices/{price_id}",
    response_model=Result[ModelPriceResult],
    summary="更新模型用户售价版本",
)
@require_permission("ai:model:manage")
async def update_model_price(
    price_id: int,
    form: ModelPriceUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_model_price_service.update_price(
        db, price_id, form.model_dump(exclude_unset=True)
    )
    return success(result)


@router.delete("/prices/{price_id}", response_model=Result[None], summary="删除模型用户售价版本")
@require_permission("ai:model:manage")
async def delete_model_price(
    price_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_model_price_service.delete_price(db, price_id)
    return success(msg="一切ok")
