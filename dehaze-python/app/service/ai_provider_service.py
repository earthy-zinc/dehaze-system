"""AI 模型供应商管理服务"""

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService
from app.models.entity.sys_ai_provider import SysAiProvider
from app.models.schema.ai_provider import ProviderCreate, ProviderResult, ProviderUpdate
from app.models.schema.common import PageResult
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.ai.provider_health_service import (
    clear_provider_health,
    set_health_check_enabled,
)

# 启用供应商列表缓存（供应商配置低频变更，缓存降低模型选择时的 DB 压力）
PROVIDER_LIST_CACHE_KEY = "ai:provider:list"
PROVIDER_LIST_CACHE_TTL = CACHE_TTL_HOUR


async def _clear_provider_cache(redis: Redis) -> None:
    await CacheService(redis).delete(PROVIDER_LIST_CACHE_KEY)


class AiProviderService:
    async def list_providers(
        self,
        db: AsyncSession,
        redis: Redis,
        page: int,
        size: int,
        keyword: str | None = None,
    ) -> PageResult[ProviderResult]:
        from app.service.ai.provider_health_service import provider_health_service

        providers, total = await ai_provider_repository.paginate_providers(db, page, size, keyword)
        items = []
        for p in providers:
            item = ProviderResult.model_validate(p)
            snapshot = await provider_health_service.get_health_snapshot(redis, p.id)
            item.health = snapshot["status"]
            items.append(item)
        return PageResult(list=items, total=total)

    async def list_enabled(self, db: AsyncSession, redis: Redis) -> list[ProviderResult]:
        cache = CacheService(redis)
        cached = await cache.get_json(PROVIDER_LIST_CACHE_KEY)
        if cached is None:
            providers = await ai_provider_repository.list_enabled(db)
            cached = [
                ProviderResult.model_validate(p).model_dump(mode="json") for p in providers
            ]
            await cache.set_json(PROVIDER_LIST_CACHE_KEY, cached, PROVIDER_LIST_CACHE_TTL)
        return [ProviderResult.model_validate(item) for item in cached]

    async def create_provider(
        self,
        db: AsyncSession,
        redis: Redis,
        form: ProviderCreate,
    ) -> ProviderResult:
        existing = await ai_provider_repository.get_by_provider_code(
            db, form.provider_code, include_deleted=True
        )
        if existing:
            if existing.deleted:
                raise BusinessException(
                    ResultCode.DATA_EXISTS, "供应商编码已被历史记录占用，不可复用"
                )
            raise BusinessException(ResultCode.DATA_EXISTS, "供应商编码已存在")
        provider = SysAiProvider(
            provider_code=form.provider_code,
            display_name=form.display_name,
            api_base_url=form.api_base_url,
            protocol_type=form.protocol_type,
            auth_type=form.auth_type,
            default_headers=form.default_headers,
            sort_order=form.sort_order,
            health_check_enabled=form.health_check_enabled,
            remark=form.remark,
            status=form.status,
        )
        provider = await ai_provider_repository.create(db, provider)
        await _clear_provider_cache(redis)
        await set_health_check_enabled(redis, provider.id, provider.health_check_enabled == 1)
        return ProviderResult.model_validate(provider)

    async def update_provider(
        self,
        db: AsyncSession,
        redis: Redis,
        provider_id: int,
        form: ProviderUpdate,
    ) -> ProviderResult:
        provider = await ai_provider_repository.get_by_id(db, provider_id)
        if not provider:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "供应商不存在")
        data = form.model_dump(exclude_unset=True)
        for key, value in data.items():
            setattr(provider, key, value)
        await db.flush()
        await db.refresh(provider)
        await _clear_provider_cache(redis)
        await set_health_check_enabled(redis, provider.id, provider.health_check_enabled == 1)
        return ProviderResult.model_validate(provider)

    async def delete_provider(
        self,
        db: AsyncSession,
        redis: Redis,
        provider_id: int,
    ) -> None:
        provider = await ai_provider_repository.get_by_id(db, provider_id)
        if not provider:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "供应商不存在")
        active = await ai_provider_repository.count_enabled_models(db, provider_id)
        if active > 0:
            raise BusinessException(
                ResultCode.DATA_BIND_EXISTS,
                "存在启用模型引用该供应商，请先禁用或删除关联模型",
            )
        await ai_provider_repository.soft_delete_by_ids(db, [provider_id])
        await _clear_provider_cache(redis)
        await clear_provider_health(redis, provider_id)


ai_provider_service = AiProviderService()
