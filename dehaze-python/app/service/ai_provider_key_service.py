"""AI 供应商 API Key 管理服务"""

import json
import random
from datetime import datetime, timedelta

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.crypto.aes_cipher import decrypt, encrypt, hash_key, mask_key
from app.models.base import get_current_user_id
from app.models.entity.sys_ai_provider_key import SysAiProviderKey
from app.models.schema.ai_provider import (
    ProviderKeyCreate,
    ProviderKeyResult,
    ProviderKeyUpdate,
)
from app.repository.ai_provider_key_repository import ai_provider_key_repository
from app.repository.ai_provider_repository import ai_provider_repository

# Key 临时不可用标记（调用失败 401/429 时写入，冷却期结束自动参与选取）
KEY_UNAVAILABLE_PREFIX = "ai:provider_key:{}:unavailable"
# Key 连续失败计数（成功调用即清零，TTL 1h）
KEY_FAIL_STREAK_PREFIX = "ai:provider_key:{}:fail_streak"
KEY_FAIL_STREAK_TTL = 3600
# Key 日调用计数（Redis 计数，不落库）
KEY_DAILY_PREFIX = "ai:provider_key:{}:daily:{}"
# Key 最后使用信息缓冲（Redis 缓冲 + 定时批量刷库，避免高并发频繁写库）
KEY_LAST_USED_PREFIX = "ai:provider_key:{}:last_used"

# 冷却升级梯度（连续失败次数 -> 冷却时长秒）。1 次 5 分钟、≥3 次 15 分钟、≥5 次 30 分钟（上限）
_KEY_COOLDOWN_STEPS = [(5, 1800), (3, 900), (1, 300)]


def _cooldown_seconds(fail_streak: int) -> int:
    """按连续失败次数返回冷却时长（命中最大档后封顶）。"""
    for threshold, seconds in _KEY_COOLDOWN_STEPS:
        if fail_streak >= threshold:
            return seconds
    return _KEY_COOLDOWN_STEPS[-1][1]


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
    @staticmethod
    async def list_keys(db: AsyncSession, provider_id: int) -> list[ProviderKeyResult]:
        await _get_provider_or_raise(db, provider_id)
        keys = await ai_provider_key_repository.list_by_provider(db, provider_id)
        return [ProviderKeyResult.model_validate(k) for k in keys]

    @staticmethod
    async def create_key(
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
            expires_at=form.expires_at,
        )
        key = await ai_provider_key_repository.create(db, key)
        return ProviderKeyResult.model_validate(key)

    @staticmethod
    async def update_key(
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

    @staticmethod
    async def delete_key(
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

    @staticmethod
    async def list_usable_keys(
        db: AsyncSession,
        redis: Redis,
        provider_id: int,
    ) -> list[SysAiProviderKey]:
        """返回该供应商全部可用 Key（启用+未过期+非冷却+未超日额度），
        按优先级升序、同优先级 weight 降序排列的实体列表。
        select_key 与 llm_client 逐 Key 重试共用此资格过滤，Key 规则单一信息源。
        """
        keys = await ai_provider_key_repository.list_enabled_by_provider(db, provider_id)
        if not keys:
            return []

        today = datetime.now().strftime("%Y%m%d")
        usable = []
        for key in keys:
            if await redis.get(KEY_UNAVAILABLE_PREFIX.format(key.id)):
                continue
            if key.daily_quota:
                used = await redis.get(KEY_DAILY_PREFIX.format(key.id, today))
                if used and int(used) >= key.daily_quota:
                    continue
            usable.append(key)
        usable.sort(key=lambda k: (k.priority, -k.weight))
        return usable

    @staticmethod
    async def select_key(
        db: AsyncSession,
        redis: Redis,
        provider_id: int,
    ) -> str | None:
        """Key 选取策略：priority 优先 -> 同优先级 weight 加权随机 -> 解密返回明文"""
        candidates = await AiProviderKeyService.list_usable_keys(db, redis, provider_id)
        if not candidates:
            return None

        # 取最高优先级组（priority 数字越小越优先）
        min_priority = min(k.priority for k in candidates)
        top = [k for k in candidates if k.priority == min_priority]
        selected = random.choices(top, weights=[k.weight for k in top], k=1)[0]

        # 异步缓冲最后使用信息（Redis 缓冲，定时批量刷库）
        try:
            user_id = get_current_user_id()
        except LookupError:
            user_id = None
        await redis.set(
            KEY_LAST_USED_PREFIX.format(selected.id),
            json.dumps(
                {
                    "last_used_at": datetime.now().isoformat(),
                    "last_used_by": user_id,
                }
            ),
        )
        return decrypt(selected.key_cipher)

    @staticmethod
    async def mark_call_failed(redis: Redis, key_id: int, error_code: str | None = None) -> None:
        """Key 调用失败：连续失败计数 + 按失败次数升级冷却时长。

        冷却升级：第 1 次 5 分钟、连续 ≥3 次 15 分钟、≥5 次 30 分钟（上限）；
        401/403 认证失败意味着 Key 本身无效，直接顶格冷却。
        冷却期结束后 Key 自动参与选取。
        """
        streak = await redis.incr(KEY_FAIL_STREAK_PREFIX.format(key_id))
        await redis.expire(KEY_FAIL_STREAK_PREFIX.format(key_id), KEY_FAIL_STREAK_TTL)
        if error_code in ("401", "403"):
            cooldown = _KEY_COOLDOWN_STEPS[0][1]
        else:
            cooldown = _cooldown_seconds(streak)
        await redis.set(KEY_UNAVAILABLE_PREFIX.format(key_id), 1, ex=cooldown)

    @staticmethod
    async def mark_call_success(redis: Redis, key_id: int, used_by: int | None = None) -> None:
        """Key 调用成功：清零连续失败计数 + 日计数 INCR + 缓冲最后使用信息（定时刷库）。

        日计数（当日 TTL，次日自然过期）与 list_usable_keys 的日额度过滤同源。
        """
        await redis.delete(KEY_FAIL_STREAK_PREFIX.format(key_id))
        await redis.set(
            KEY_LAST_USED_PREFIX.format(key_id),
            json.dumps(
                {
                    "last_used_at": datetime.now().isoformat(),
                    "last_used_by": used_by,
                }
            ),
        )
        today = datetime.now().strftime("%Y%m%d")
        daily = KEY_DAILY_PREFIX.format(key_id, today)
        count = await redis.incr(daily)
        if count == 1:
            midnight = (datetime.now() + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            await redis.expire(daily, int((midnight - datetime.now()).total_seconds()))
