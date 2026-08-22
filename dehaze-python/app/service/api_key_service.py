import hashlib
import secrets
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.api_key import SysApiKey
from app.repository.ai_model_repository import ai_model_repository
from app.repository.api_key_repository import api_key_repository


def _to_result(entity: SysApiKey, raw_key: str | None = None) -> dict:
    return {
        "id": entity.id,
        "name": entity.name,
        "apiKey": raw_key,
        "keyPrefix": entity.key_prefix,
        "status": entity.status,
        "expiresAt": entity.expires_at,
        "lastUsedAt": entity.last_used_at,
        "createTime": entity.create_time,
        "dailyQuota": entity.daily_quota,
        "monthlyQuota": entity.monthly_quota,
        "rpmLimit": entity.rpm_limit,
        "modelWhitelist": entity.model_whitelist,
    }


async def _validate_whitelist(db: AsyncSession, whitelist: list[str] | None) -> None:
    """校验白名单模型 ID 均存在于启用模型表，非法模型 ID 抛参数异常。"""
    if not whitelist:
        return
    for model_id in whitelist:
        enabled = await ai_model_repository.list_enabled_by_model_id(db, model_id)
        if not enabled:
            raise BusinessException(ResultCode.PARAM_ERROR, f"模型 {model_id} 不存在或未启用")


class ApiKeyService:
    @staticmethod
    async def create_api_key(
        db: AsyncSession,
        user_id: int,
        name: str,
        expires_at: datetime | None = None,
        daily_quota: int | None = None,
        monthly_quota: int | None = None,
        rpm_limit: int | None = None,
        model_whitelist: list[str] | None = None,
    ) -> dict:
        await _validate_whitelist(db, model_whitelist)
        raw_key = "dhak_" + secrets.token_urlsafe(36)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        entity = SysApiKey(
            user_id=user_id,
            name=name,
            key_prefix=raw_key[:12],
            key_hash=key_hash,
            status=1,
            expires_at=expires_at,
            daily_quota=daily_quota or None,
            monthly_quota=monthly_quota or None,
            rpm_limit=rpm_limit or None,
            model_whitelist=model_whitelist or None,
        )
        entity = await api_key_repository.create(db, entity)
        return _to_result(entity, raw_key=raw_key)

    @staticmethod
    async def list_api_keys(db: AsyncSession, user_id: int) -> list[dict]:
        # 列表只展示未吊销的 key（revoked_at IS NULL），与 Java/Go 一致
        items = await api_key_repository.list_active_by_user(db, user_id)
        return [_to_result(item) for item in items]

    @staticmethod
    async def delete_api_key(db: AsyncSession, user_id: int, key_id: int) -> bool:
        """吊销 API 密钥：设置 revoked_at，永久保留 hash 以拒绝已泄露的旧密钥。"""
        entity = await api_key_repository.get_active_by_id_and_user(db, key_id, user_id)
        if not entity:
            return False
        await api_key_repository.revoke(db, entity)
        return True

