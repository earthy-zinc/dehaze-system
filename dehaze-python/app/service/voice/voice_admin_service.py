"""语音引擎注册表管理服务（Provider / Key / Model 管理端 CRUD）"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.crypto.aes_cipher import encrypt, hash_key, mask_key
from app.infrastructure.voice.provider.registry import voice_engine_registry
from app.models.entity.sys_voice_model import SysVoiceModel
from app.models.entity.sys_voice_provider import SysVoiceProvider
from app.models.entity.sys_voice_provider_key import SysVoiceProviderKey
from app.models.schema.common import PageResult
from app.models.schema.voice_admin import (
    VoiceModelCreate,
    VoiceModelResult,
    VoiceModelUpdate,
    VoiceProviderCreate,
    VoiceProviderKeyCreate,
    VoiceProviderKeyResult,
    VoiceProviderKeyUpdate,
    VoiceProviderResult,
    VoiceProviderUpdate,
)
from app.repository.voice_model_repository import voice_model_repository
from app.repository.voice_provider_key_repository import voice_provider_key_repository
from app.repository.voice_provider_repository import voice_provider_repository


async def _clear_defaults(db: AsyncSession, provider: SysVoiceProvider) -> None:
    """将该引擎同 engine_type 下其他引擎的 is_default 清除，保证每能力维度仅一条 default"""
    stmt = (
        select(SysVoiceProvider)
        .where(
            SysVoiceProvider.engine_type == provider.engine_type,
            SysVoiceProvider.is_default == 1,
            SysVoiceProvider.id != provider.id,
        )
        .execution_options(include_deleted=True)
    )
    for other in (await db.execute(stmt)).scalars().all():
        other.is_default = 0


class VoiceAdminService:
    def __init__(self, engine_registry=voice_engine_registry) -> None:
        # is_default/status 变更经注册表失效默认引擎 Redis 缓存（后端实现 §2.4 切换即时生效）
        self._engine_registry = engine_registry

    # ==================== Provider ====================

    async def list_providers(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
        engine_type: str | None = None,
    ) -> PageResult[VoiceProviderResult]:
        providers, total = await voice_provider_repository.paginate_providers(
            db, page, size, keyword, engine_type
        )
        return PageResult(
            list=[VoiceProviderResult.model_validate(p) for p in providers], total=total
        )

    async def list_enabled(self, db: AsyncSession, engine_type: str) -> list[VoiceProviderResult]:
        providers = await voice_provider_repository.list_enabled(db, engine_type)
        return [VoiceProviderResult.model_validate(p) for p in providers]

    async def create_provider(
        self, db: AsyncSession, form: VoiceProviderCreate
    ) -> VoiceProviderResult:
        existing = await voice_provider_repository.get_by_provider_and_engine(
            db, form.provider_code, form.engine_type, include_deleted=True
        )
        if existing:
            if existing.deleted:
                raise BusinessException(
                    ResultCode.DATA_EXISTS, "引擎编码已被历史记录占用，不可复用"
                )
            raise BusinessException(ResultCode.DATA_EXISTS, "引擎编码已存在")
        provider = SysVoiceProvider(
            provider_code=form.provider_code,
            engine_type=form.engine_type,
            display_name=form.display_name,
            api_base_url=form.api_base_url,
            auth_type=form.auth_type,
            default_headers=form.default_headers,
            is_default=form.is_default,
            sort_order=form.sort_order,
            health_check_enabled=form.health_check_enabled,
            remark=form.remark,
            status=form.status,
        )
        provider = await voice_provider_repository.create(db, provider)
        if provider.is_default == 1:
            await _clear_defaults(db, provider)
            await self._engine_registry.invalidate_default_cache(provider.engine_type)
        return VoiceProviderResult.model_validate(provider)

    async def update_provider(
        self, db: AsyncSession, provider_id: int, form: VoiceProviderUpdate
    ) -> VoiceProviderResult:
        provider = await voice_provider_repository.get_by_id(db, provider_id)
        if not provider:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "引擎不存在")
        changed = set(form.model_dump(exclude_unset=True))
        for key, value in form.model_dump(exclude_unset=True).items():
            setattr(provider, key, value)
        if provider.is_default == 1:
            await _clear_defaults(db, provider)
        await db.flush()
        await db.refresh(provider)
        if {"is_default", "status"} & changed:
            await self._engine_registry.invalidate_default_cache(provider.engine_type)
        return VoiceProviderResult.model_validate(provider)

    async def delete_provider(self, db: AsyncSession, provider_id: int) -> None:
        provider = await voice_provider_repository.get_by_id(db, provider_id)
        if not provider:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "引擎不存在")
        stmt = (
            select(SysVoiceModel)
            .where(SysVoiceModel.provider_id == provider_id, SysVoiceModel.status == 1)
            .limit(1)
        )
        if (await db.execute(stmt)).scalar_one_or_none():
            raise BusinessException(
                ResultCode.DATA_BIND_EXISTS, "存在启用模型引用该引擎，请先禁用或删除关联模型"
            )
        await voice_provider_repository.soft_delete_by_ids(db, [provider_id])
        # 被删引擎可能是当前默认：失效缓存使下次路由立即回源重新解析
        await self._engine_registry.invalidate_default_cache(provider.engine_type)

    async def test_connection(self, db: AsyncSession, provider_id: int) -> dict:
        provider = await voice_provider_repository.get_by_id(db, provider_id)
        if not provider:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "引擎不存在")
        # local 引擎走进程内 FunASR/Piper，无需连通性测试
        if provider.provider_code == "local":
            return {"result": "本地引擎", "connected": True}
        # 云端厂商协议未接入（CloudBase 仅有通用框架），真实连通性探测依赖厂商
        # API 规格；DNS/TCP 层探测无法验证鉴权与端点有效性，只会产生误导性
        # 误报，故显式跳过（API 契约：结果仅提示不阻断保存）
        return {"result": "云端厂商协议未接入，跳过连通性测试", "connected": None}

    # ==================== Key（物理删除，对齐 ai_provider_key） ====================

    async def list_keys(
        self, db: AsyncSession, provider_id: int
    ) -> list[VoiceProviderKeyResult]:
        await self._get_provider_or_raise(db, provider_id)
        stmt = (
            select(SysVoiceProviderKey)
            .where(SysVoiceProviderKey.provider_id == provider_id)
            .order_by(SysVoiceProviderKey.priority, SysVoiceProviderKey.id)
        )
        keys = (await db.execute(stmt)).scalars().all()
        return [VoiceProviderKeyResult.model_validate(k) for k in keys]

    async def create_key(
        self, db: AsyncSession, provider_id: int, form: VoiceProviderKeyCreate
    ) -> VoiceProviderKeyResult:
        await self._get_provider_or_raise(db, provider_id)
        key_hash = hash_key(form.key)
        if await voice_provider_key_repository.get_by_hash(db, key_hash):
            raise BusinessException(ResultCode.DATA_EXISTS, "该 API Key 已存在")
        key = SysVoiceProviderKey(
            provider_id=provider_id,
            name=form.name,
            key_hash=key_hash,
            key_prefix=mask_key(form.key),
            key_cipher=encrypt(form.key),
            status=form.status,
            priority=form.priority,
            weight=form.weight,
            daily_quota=form.daily_quota,
            rpm_limit=form.rpm_limit,
            expires_at=form.expires_at,
        )
        key = await voice_provider_key_repository.create(db, key)
        return VoiceProviderKeyResult.model_validate(key)

    async def update_key(
        self, db: AsyncSession, provider_id: int, key_id: int, form: VoiceProviderKeyUpdate
    ) -> VoiceProviderKeyResult:
        key = await self._get_key_or_raise(db, provider_id, key_id)
        for field, value in form.model_dump(exclude_unset=True).items():
            setattr(key, field, value)
        await db.flush()
        await db.refresh(key)
        return VoiceProviderKeyResult.model_validate(key)

    async def delete_key(self, db: AsyncSession, provider_id: int, key_id: int) -> None:
        await self._get_key_or_raise(db, provider_id, key_id)
        await voice_provider_key_repository.delete_by_ids(db, [key_id])

    # ==================== Model ====================

    async def list_models(
        self, db: AsyncSession, engine_type: str | None = None
    ) -> list[VoiceModelResult]:
        """模型/音色列表（管理端展示，可选按 engine_type 筛选）"""
        models = await voice_model_repository.list_by_engine_type(db, engine_type)
        return [VoiceModelResult.model_validate(m) for m in models]

    async def create_model(self, db: AsyncSession, form: VoiceModelCreate) -> VoiceModelResult:
        await self._get_provider_or_raise(db, form.provider_id)
        existing = await voice_model_repository.get_by_model_and_provider(
            db, form.model_id, form.provider_id
        )
        if existing:
            if existing.deleted:
                raise BusinessException(
                    ResultCode.DATA_EXISTS, "该引擎+模型组合已被历史记录占用，不可复用"
                )
            raise BusinessException(ResultCode.DATA_EXISTS, "该引擎+模型组合已存在")
        model = SysVoiceModel(
            provider_id=form.provider_id,
            model_id=form.model_id,
            engine_type=form.engine_type,
            model_type=form.model_type,
            display_name=form.display_name,
            params=form.params,
            status=form.status,
        )
        model = await voice_model_repository.create(db, model)
        return VoiceModelResult.model_validate(model)

    async def update_model(
        self, db: AsyncSession, model_id: int, form: VoiceModelUpdate
    ) -> VoiceModelResult:
        model = await voice_model_repository.get_by_id(db, model_id)
        if not model:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模型不存在")
        for key, value in form.model_dump(exclude_unset=True).items():
            setattr(model, key, value)
        await db.flush()
        await db.refresh(model)
        return VoiceModelResult.model_validate(model)

    async def delete_model(self, db: AsyncSession, model_id: int) -> None:
        model = await voice_model_repository.get_by_id(db, model_id)
        if not model:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模型不存在")
        await voice_model_repository.soft_delete_by_ids(db, [model_id])

    # ==================== 内部 ====================

    async def _get_provider_or_raise(self, db: AsyncSession, provider_id: int) -> None:
        if not await voice_provider_repository.get_by_id(db, provider_id):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "引擎不存在")

    async def _get_key_or_raise(
        self, db: AsyncSession, provider_id: int, key_id: int
    ) -> SysVoiceProviderKey:
        key = await voice_provider_key_repository.get_by_id(db, key_id)
        if not key or key.provider_id != provider_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "API Key 不存在")
        return key


voice_admin_service = VoiceAdminService()
