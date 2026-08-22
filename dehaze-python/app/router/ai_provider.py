from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import Field
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.ai_provider import (
    ProviderCreate,
    ProviderKeyCreate,
    ProviderKeyResult,
    ProviderKeyUpdate,
    ProviderResult,
    ProviderUpdate,
)
from app.models.schema.common import BasePageQuery, PageResult
from app.service.ai.provider_connectivity_service import test_connection
from app.infrastructure.llm.provider_health_service import provider_health_service
from app.service.ai_provider_key_service import ai_provider_key_service
from app.service.ai_provider_service import ai_provider_service

router = APIRouter(prefix="/api/v1/ai", tags=["AI对话"])


class ProviderPageQuery(BasePageQuery):
    keyword: str | None = Field(default=None, description="关键字(按显示名称/供应商编码模糊搜索)")


async def _run_connection_test(provider_id: int) -> None:
    """后台执行供应商连通性测试（结果仅提示不阻断，失败不抛出）。"""
    import logging

    from app.dependencies.redis import get_redis_client

    logger = logging.getLogger("ai_provider")
    try:
        from app.database import get_db_session

        redis = await get_redis_client()
        async with get_db_session() as db:
            await test_connection(db, redis, provider_id)
    except Exception as exc:  # noqa: BLE001 - 后台连通性测试失败不影响保存流程
        logger.warning("供应商 %s 连通性测试后台执行失败: %s", provider_id, exc)


@router.get(
    "/providers", response_model=Result[PageResult[ProviderResult]], summary="供应商分页列表"
)
@require_permission("ai:model:manage")
async def list_providers(
    query: ProviderPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_provider_service.list_providers(
        db, redis, query.pageNum, query.pageSize, query.keyword
    )
    return success(result)


@router.get(
    "/providers/enabled", response_model=Result[list[ProviderResult]], summary="启用供应商列表"
)
async def list_enabled_providers(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_provider_service.list_enabled(db, redis)
    return success(result)


@router.post("/providers", response_model=Result[ProviderResult], summary="新增供应商")
@require_permission("ai:model:manage")
async def create_provider(
    form: ProviderCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_provider_service.create_provider(db, redis, form)
    # 保存后异步触发连通性测试（结果仅提示不阻断保存流程）
    background_tasks.add_task(_run_connection_test, result.id)
    return success(result)


@router.put("/providers/{provider_id}", response_model=Result[ProviderResult], summary="更新供应商")
@require_permission("ai:model:manage")
async def update_provider(
    provider_id: int,
    form: ProviderUpdate,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_provider_service.update_provider(db, redis, provider_id, form)
    return success(result)


@router.delete("/providers/{provider_id}", response_model=Result[None], summary="删除供应商")
@require_permission("ai:model:manage")
async def delete_provider(
    provider_id: int,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await ai_provider_service.delete_provider(db, redis, provider_id)
    return success(msg="一切ok")


@router.get(
    "/providers/{provider_id}/keys",
    response_model=Result[list[ProviderKeyResult]],
    summary="供应商API Key列表",
)
@require_permission("ai:model:manage")
async def list_keys(
    provider_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_provider_key_service.list_keys(db, provider_id)
    return success(result)


@router.post(
    "/providers/{provider_id}/keys", response_model=Result[ProviderKeyResult], summary="新增API Key"
)
@require_permission("ai:model:manage")
async def create_key(
    provider_id: int,
    form: ProviderKeyCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_provider_key_service.create_key(db, provider_id, form)
    return success(result)


@router.put(
    "/providers/{provider_id}/keys/{key_id}",
    response_model=Result[ProviderKeyResult],
    summary="更新API Key",
)
@require_permission("ai:model:manage")
async def update_key(
    provider_id: int,
    key_id: int,
    form: ProviderKeyUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_provider_key_service.update_key(db, provider_id, key_id, form)
    return success(result)


@router.delete(
    "/providers/{provider_id}/keys/{key_id}", response_model=Result[None], summary="删除API Key"
)
@require_permission("ai:model:manage")
async def delete_key(
    provider_id: int,
    key_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_provider_key_service.delete_key(db, provider_id, key_id)
    return success(msg="一切ok")


@router.post(
    "/providers/{provider_id}/test-connection", response_model=Result[dict], summary="连通性测试"
)
@require_permission("ai:model:manage")
async def test_provider_connection(
    provider_id: int,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await test_connection(db, redis, provider_id)
    return success(result)


@router.post(
    "/providers/{provider_id}/circuit/close",
    response_model=Result[None],
    summary="手动解除供应商熔断",
)
@require_permission("ai:model:manage")
async def close_provider_circuit(
    provider_id: int,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await provider_health_service.close_circuit(redis, provider_id)
    return success(msg="一切ok")
