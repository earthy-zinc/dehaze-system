"""AI 供应商 API Key 运行时选择与健康管理

Key 资格过滤（list_usable_keys）、额度原子预留（acquire_key / select_key）、
失败冷却与成功标记（mark_call_failed / mark_call_success）是所有模态
（LLM/Embedding/TTS）共用的基础设施能力：LlmClient 逐 Key 重试与
Embedding / 连通性测试的取 Key 共用同一实现。
"""

import json
import random
from collections.abc import Awaitable
from datetime import datetime, timedelta
from typing import cast

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.crypto.aes_cipher import decrypt
from app.models.base import get_current_user_id
from app.models.entity.sys_ai_provider_key import SysAiProviderKey
from app.repository.ai_provider_key_repository import ai_provider_key_repository

# Key 临时不可用标记（调用失败 401/429 时写入，冷却期结束自动参与选取）
KEY_UNAVAILABLE_PREFIX = "ai:provider_key:{}:unavailable"
# Key 连续失败计数（成功调用即清零，TTL 1h）
KEY_FAIL_STREAK_PREFIX = "ai:provider_key:{}:fail_streak"
KEY_FAIL_STREAK_TTL = 3600
# Key 日调用计数（Redis 计数，不落库）
KEY_DAILY_PREFIX = "ai:provider_key:{}:daily:{}"
# Key 分钟调用计数（rpm_limit 限流，当日分钟键自然过期）
KEY_MINUTE_PREFIX = "ai:provider_key:{}:minute:{}"
# Key 最后使用信息缓冲（Redis 缓冲 + 定时批量刷库，避免高并发频繁写库）
KEY_LAST_USED_PREFIX = "ai:provider_key:{}:last_used"

# 冷却升级梯度（连续失败次数 -> 冷却时长秒）。1 次 5 分钟、≥3 次 15 分钟、≥5 次 30 分钟（上限）
_KEY_COOLDOWN_STEPS = [(5, 1800), (3, 900), (1, 300)]

# 日/分钟额度原子「检查+预留」：GET 判断与 INCR 计数分离时，并发请求会同时通过
# 检查而击穿额度；Lua 内自增后判定，超限回滚本次预留
_RESERVE_QUOTA_LUA = """
local quota = tonumber(ARGV[1])
local rpm = tonumber(ARGV[2])
if quota > 0 then
    local used = redis.call('INCR', KEYS[1])
    if used == 1 then redis.call('EXPIRE', KEYS[1], ARGV[3]) end
    if used > quota then
        redis.call('DECR', KEYS[1])
        return 0
    end
end
if rpm > 0 then
    local used = redis.call('INCR', KEYS[2])
    if used == 1 then redis.call('EXPIRE', KEYS[2], ARGV[4]) end
    if used > rpm then
        if quota > 0 then redis.call('DECR', KEYS[1]) end
        redis.call('DECR', KEYS[2])
        return 0
    end
end
return 1
"""


def _cooldown_seconds(fail_streak: int) -> int:
    """按连续失败次数返回冷却时长（命中最大档后封顶）。"""
    for threshold, seconds in _KEY_COOLDOWN_STEPS:
        if fail_streak >= threshold:
            return seconds
    return _KEY_COOLDOWN_STEPS[-1][1]


def _seconds_to_midnight(now: datetime) -> int:
    """当日剩余秒数（日额度计数键的自然过期时间）"""
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - now).total_seconds()))


class ProviderKeySelector:
    """供应商 API Key 选取器：资格过滤 + 额度原子预留 + 加权随机选取 + 失败冷却"""

    async def list_usable_keys(
        self,
        db: AsyncSession,
        redis: Redis,
        provider_id: int,
    ) -> list[SysAiProviderKey]:
        """返回该供应商全部可用 Key（启用+未过期+非冷却+未超日额度），
        按优先级升序、同优先级 weight 降序排列的实体列表。

        纯过滤不占额度：acquire_key / select_key 与 llm_client 逐 Key 重试共用
        此资格过滤，Key 规则单一信息源。
        """
        keys = await ai_provider_key_repository.list_enabled_by_provider(db, provider_id)
        if not keys:
            return []

        today = datetime.now().strftime("%Y%m%d")
        minute = datetime.now().strftime("%Y%m%d%H%M")
        usable = []
        for key in keys:
            if await redis.get(KEY_UNAVAILABLE_PREFIX.format(key.id)):
                continue
            if key.daily_quota:
                used = await redis.get(KEY_DAILY_PREFIX.format(key.id, today))
                if used and int(used) >= key.daily_quota:
                    continue
            if key.rpm_limit:
                minute_used = await redis.get(KEY_MINUTE_PREFIX.format(key.id, minute))
                if minute_used and int(minute_used) >= key.rpm_limit:
                    continue
            usable.append(key)
        usable.sort(key=lambda k: (k.priority, -k.weight))
        return usable

    async def reserve_key(self, redis: Redis, key: SysAiProviderKey) -> bool:
        """原子检查并预留该 Key 的日/分钟额度（Lua 内自增后判定，超限回滚）。

        取 Key 与计数必须为一次原子操作：分离成 GET 判断 + INCR 计数时，并发
        请求会同时通过判断而击穿额度。预留成功才可取用该 Key。
        """
        now = datetime.now()
        # daily_quota / rpm_limit 为可空的可选限额，语义均为"未配置=不限"：schema 以
        # ge=1（daily_quota）/ ge=0（rpm_limit，0=不限）约束，管理端校验通过；Lua 以
        # `quota > 0 / rpm > 0` 判定是否启用限流，故此处显式将 None 归一为 0 → 跳过该
        # 维度限流（不会误封），与 list_usable_keys 的 `if key.daily_quota / if key.rpm_limit`
        # 过滤口径一致。统一以字符串入参（与 redis 编码后的字节一致，Lua 内 tonumber 解析）。
        reserved = redis.eval(
            _RESERVE_QUOTA_LUA,
            2,
            KEY_DAILY_PREFIX.format(key.id, now.strftime("%Y%m%d")),
            KEY_MINUTE_PREFIX.format(key.id, now.strftime("%Y%m%d%H%M")),
            str(key.daily_quota or 0),
            str(key.rpm_limit or 0),
            str(_seconds_to_midnight(now)),
            "60",
        )
        # redis-py 同步/异步客户端共用 eval 签名（Union[Awaitable[str], str]）：
        # 此处为异步客户端，运行时恒返回协程，await 正确，cast 仅收窄类型（非消音）。
        return bool(await cast(Awaitable[str], reserved))

    async def select_key(
        self,
        db: AsyncSession,
        redis: Redis,
        provider_id: int,
    ) -> str | None:
        """Key 选取策略：priority 优先 -> 同优先级 weight 加权随机 -> 解密返回明文

        选中的 Key 先原子预留日/分钟额度，预留失败（并发抢占致超限）换下一个候选。
        """
        candidates = await self.list_usable_keys(db, redis, provider_id)
        if not candidates:
            return None

        # 取最高优先级组（priority 数字越小越优先）
        min_priority = min(k.priority for k in candidates)
        pool = [k for k in candidates if k.priority == min_priority]
        selected = None
        while pool:
            candidate = random.choices(pool, weights=[k.weight for k in pool], k=1)[0]
            if await self.reserve_key(redis, candidate):
                selected = candidate
                break
            pool.remove(candidate)
        if selected is None:
            return None

        # 异步缓冲最后使用信息（Redis 缓冲，定时批量刷库）。
        # get_current_user_id 的 contextvar 以 None 为默认值，无请求上下文（定时/后台
        # 任务）时返回 None 而非抛异常，故无需 try/except；last_used_by=None 即"系统触发"
        user_id = get_current_user_id()
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

    async def mark_call_failed(
        self, redis: Redis, key_id: int, error_code: str | None = None
    ) -> None:
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

    async def mark_call_success(
        self, redis: Redis, key_id: int, used_by: int | None = None
    ) -> None:
        """Key 调用成功：清零连续失败计数 + 缓冲最后使用信息（定时刷库）。

        日/分钟额度在取 Key 时已由 _reserve_quota 原子预留，此处不再重复计数。
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


provider_key_selector = ProviderKeySelector()
