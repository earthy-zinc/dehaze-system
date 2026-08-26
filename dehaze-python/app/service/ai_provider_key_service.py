"""AI 供应商 API Key 管理服务"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.crypto.aes_cipher import encrypt, hash_key, mask_key
from app.models.entity.sys_ai_provider_key import SysAiProviderKey
from app.models.schema.ai_provider import (
    ProviderKeyCreate,
    ProviderKeyResult,
    ProviderKeyUpdate,
)
from app.repository.ai_provider_key_repository import ai_provider_key_repository
from app.repository.ai_provider_repository import ai_provider_repository


async def _get_provider_or_raise(db: AsyncSession, provider_id: int) -> None:
    provider = await ai_provider_repository.get_by_id(db, provider_id)
    if not provider:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "供应商不存在")


async def _get_key_or_raise(
    db: AsyncSession,
    provider_id: int,
    key_id: int,
) -> SysAiProviderKey:
    key = await ai_provider_key_repository.get_by_id(db, key_id)
    if not key or key.provider_id != provider_id:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "API Key 不存在")
    return key


class AiProviderKeyService:
    async def list_keys(self, db: AsyncSession, provider_id: int) -> list[ProviderKeyResult]:
        await _get_provider_or_raise(db, provider_id)
        keys = await ai_provider_key_repository.list_by_provider(db, provider_id)
        return [ProviderKeyResult.model_validate(k) for k in keys]

    async def create_key(
        self,
        db: AsyncSession,
        provider_id: int,
        form: ProviderKeyCreate,
    ) -> ProviderKeyResult:
        await _get_provider_or_raise(db, provider_id)
        key_hash = hash_key(form.key)
        if await ai_provider_key_repository.get_by_hash(db, key_hash):
            raise BusinessException(ResultCode.DATA_EXISTS, "该 API Key 已存在")
        key = SysAiProviderKey(
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
        key = await ai_provider_key_repository.create(db, key)
        return ProviderKeyResult.model_validate(key)

    async def update_key(
        self,
        db: AsyncSession,
        provider_id: int,
        key_id: int,
        form: ProviderKeyUpdate,
    ) -> ProviderKeyResult:
        key = await _get_key_or_raise(db, provider_id, key_id)
        data = form.model_dump(exclude_unset=True)
        for field, value in data.items():
            setattr(key, field, value)
        await db.flush()
        await db.refresh(key)
        return ProviderKeyResult.model_validate(key)

    async def delete_key(
        self,
        db: AsyncSession,
        provider_id: int,
        key_id: int,
    ) -> None:
        key = await _get_key_or_raise(db, provider_id, key_id)
        if key.status == 1:
            enabled = await ai_provider_key_repository.count_enabled_by_provider(db, provider_id)
            if enabled <= 1:
                raise BusinessException(
                    ResultCode.OPERATION_NOT_ALLOW,
                    "该供应商唯一启用 Key，不可删除，请先新增其他 Key 或禁用后再删除",
                )
        await ai_provider_key_repository.delete_by_ids(db, [key_id])

ai_provider_key_service = AiProviderKeyService()
