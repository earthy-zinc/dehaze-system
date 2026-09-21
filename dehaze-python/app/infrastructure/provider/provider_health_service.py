"""供应商健康与熔断服务

健康指标由调用链路实时聚合到 Redis（滑动窗口近 24h），熔断标记走
Redis circuit_open（TTL 冷却期），不新增库表。阈值从 sys_dict 读取
（ai_provider_health 前缀），缺省值仅在种子 SQL 中。

Redis Key 约定：
- ai:provider:{id}:circuit_open      熔断标记（存在即熔断，TTL=冷却时长）
- ai:provider:{id}:circuit_recovery  熔断恢复周期标记（TTL=冷却+探测期）
- ai:provider:{id}:half_open_probe   半开探测租约（SETNX 获取，探测期只放行一个请求）
- ai:provider:{id}:fail_streak       连续失败计数（成功清零）
- ai:provider:{id}:window            24h 调用窗口计数（小时分桶 HASH，字段="{bucket}:t|f|l"）
- ai:provider:{id}:latency           延迟列表（LPUSH，成员=延迟毫秒）
- ai:provider:{id}:health            聚合快照缓存（status/success_rate/limit_rate/p95 等）
- ai:provider:{id}:health_enabled    健康检查开关缓存（由供应商 CRUD 写入）
- ai:provider:health:thresholds      熔断阈值缓存（sys_dict ai_provider_health）
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import cast

from redis.asyncio import Redis

from app.database import get_db_session
from app.infrastructure.cache.cache import CacheService
from app.repository.dict_repository import dict_repository

logger = logging.getLogger(__name__)

# 阈值字典类型（值存于 sys_dict）
_HEALTH_DICT = "ai_provider_health"

# 延迟滑动窗口保留条数（仅看板聚合读取，不在调用热路径）
_LATENCY_WINDOW = 500

# 24h 调用窗口：按小时分桶计数，聚合只读最近 24 个桶字段（开销与调用量无关）
_BUCKET_SECONDS = 3600
_WINDOW_BUCKETS = 24
_WINDOW_TTL = _BUCKET_SECONDS * (_WINDOW_BUCKETS + 1)

# 阈值缓存 TTL（秒）：Redis 跨进程共享 + 进程内热路径缓存
_THRESHOLDS_TTL = 300
_THRESHOLDS_PROCESS_TTL = 60

# 健康检查开关缓存前缀
HEALTH_ENABLED_PREFIX = "ai:provider:{}:health_enabled"

# 熔断 / 恢复周期 / 半开探测租约 / 连续失败 / 调用窗口 / 延迟 / 快照 Key 前缀
_CIRCUIT_KEY = "ai:provider:{}:circuit_open"
# 熔断恢复周期标记（覆盖冷却 + 半开探测全程）：冷却结束后据此判定是否处于半开
_RECOVERY_KEY = "ai:provider:{}:circuit_recovery"
_PROBE_KEY = "ai:provider:{}:half_open_probe"
_STREAK_KEY = "ai:provider:{}:fail_streak"
_WINDOW_KEY = "ai:provider:{}:window"
_LATENCY_KEY = "ai:provider:{}:latency"
_SNAPSHOT_KEY = "ai:provider:{}:health"
_THRESHOLDS_KEY = "ai:provider:health:thresholds"


class _Thresholds:
    """健康熔断阈值（从 sys_dict 读取，缺省回落到种子同源默认值）。"""

    def __init__(self, data: dict):
        self.error_rate_warn = float(data.get("error_rate_warn", 0.10))
        self.error_rate_open = float(data.get("error_rate_open", 0.30))
        self.min_window_calls = int(data.get("min_window_calls", 20))
        self.consecutive_failures = int(data.get("consecutive_failures", 5))
        self.circuit_cooldown = int(data.get("circuit_cooldown", 60))


_SEED_THRESHOLDS = {
    "error_rate_warn": 0.10,
    "error_rate_open": 0.30,
    "min_window_calls": 20,
    "consecutive_failures": 5,
    "circuit_cooldown": 60,
}


# 阈值进程内缓存（失效时刻 monotonic, _Thresholds）；失败判定每失败一次都要读阈值
_thresholds_cache: tuple[float, _Thresholds] | None = None


async def _load_thresholds(redis: Redis) -> _Thresholds:
    """读取熔断阈值（进程内缓存 → Redis 缓存 → sys_dict）。

    阈值来源唯一为 sys_dict（ai_provider_health），供应商健康阈值变更低频：
    进程内缓存 1 分钟（失败路径每次判定都要读阈值，Redis 缓存仍是一次网络
    往返），Redis 缓存 5 分钟跨进程共享；缺省键回落种子默认（与
    config/sql/data/sys_dict.sql 的 ai_provider_health 同源）。
    """
    global _thresholds_cache
    now = time.monotonic()
    if _thresholds_cache is not None and now < _thresholds_cache[0]:
        return _thresholds_cache[1]

    cache = CacheService(redis)
    data: dict[str, float | int | str] | None = await cache.get_json(_THRESHOLDS_KEY)
    if data is None:
        data = dict(_SEED_THRESHOLDS)
        try:
            async with get_db_session() as db:
                items = await dict_repository.list_enabled_by_type_code(db, _HEALTH_DICT)
                for item in items:
                    data[item.name] = _coerce_scalar(item.value)
        except Exception as exc:
            logger.warning("读取供应商健康阈值失败，使用种子默认: %s", exc)
        await cache.set_json(_THRESHOLDS_KEY, data, _THRESHOLDS_TTL)

    thresholds = _Thresholds(data)
    _thresholds_cache = (now + _THRESHOLDS_PROCESS_TTL, thresholds)
    return thresholds


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _coerce_scalar(raw: str) -> float | int | str:
    """将 sys_dict 字符串值转换为数值（int/float），无法转换时保留原字符串。"""
    try:
        return int(str(raw))
    except (ValueError, TypeError):
        try:
            return float(str(raw))
        except (ValueError, TypeError):
            # 保留原字符串是既有契约（非数值 dict 值按原样透传），但需可见以暴露配置错误
            logger.warning("sys_dict 值无法转换为数值，保留原字符串: value=%r", raw)
            return raw


async def get_health_check_enabled(redis: Redis, provider_id: int) -> bool:
    """读取健康检查开关（供应商 CRUD 写入缓存；缺省视为开启）。"""
    val = await redis.get(HEALTH_ENABLED_PREFIX.format(provider_id))
    if val is None:
        return True
    return str(val) != "0"


async def set_health_check_enabled(redis: Redis, provider_id: int, enabled: bool) -> None:
    """供应商 CRUD 时写入健康检查开关缓存。"""
    await redis.set(HEALTH_ENABLED_PREFIX.format(provider_id), 1 if enabled else 0)


async def clear_provider_health(redis: Redis, provider_id: int) -> None:
    """删除供应商健康相关 Key（删除供应商时清理，避免残留）。"""
    await redis.delete(
        _CIRCUIT_KEY.format(provider_id),
        _RECOVERY_KEY.format(provider_id),
        _PROBE_KEY.format(provider_id),
        _STREAK_KEY.format(provider_id),
        _WINDOW_KEY.format(provider_id),
        _LATENCY_KEY.format(provider_id),
        _SNAPSHOT_KEY.format(provider_id),
        HEALTH_ENABLED_PREFIX.format(provider_id),
    )


def _p95(values: list[int]) -> int:
    """计算列表 P95（升序后取 95% 位置），空列表返回 0。"""
    if not values:
        return 0
    values = sorted(values)
    idx = max(0, int(len(values) * 0.95) - 1)
    return values[idx]


class ProviderHealthService:
    """供应商健康与熔断（一期最小闭环）。"""

    async def get_status(self, redis: Redis, provider_id: int) -> str:
        """返回供应商健康状态：healthy | suspicious | half_open | open。

        高频调用链路的快速路径：仅读健康开关与熔断/探测标记，不做聚合。
        可疑判定聚合结果取自缓存的健康快照。

        half_open：熔断冷却期结束（circuit_open 键过期）后先放行一个探测请求
        （half_open 租约 SETNX 占用），探测成功（record_call）才全量恢复，
        探测失败重新熔断，避免冷却一过全量流量打向未恢复的供应商。
        """
        if not await get_health_check_enabled(redis, provider_id):
            return "healthy"
        if await redis.get(_CIRCUIT_KEY.format(provider_id)):
            return "open"
        if not await redis.get(_RECOVERY_KEY.format(provider_id)):
            snapshot = await CacheService(redis).get_json(_SNAPSHOT_KEY.format(provider_id))
            if snapshot and snapshot.get("status") in ("healthy", "suspicious"):
                return snapshot["status"]
            return "healthy"
        # 冷却已过、恢复周期内：抢占探测租约放行单个请求（未抢到继续阻断）
        cooldown = (await _load_thresholds(redis)).circuit_cooldown
        if await redis.set(_PROBE_KEY.format(provider_id), 1, nx=True, ex=cooldown):
            return "half_open"
        return "open"

    async def record_call(
        self,
        redis: Redis,
        provider_id: int,
        success: bool,
        error_code: str | None,
        latency_ms: int,
    ) -> None:
        """记录一次供应商调用并内联判定是否熔断。

        健康检查关闭的供应商不参与聚合与判定。成功调用清零连续失败计数并解除
        熔断（含半开探测成功 → 全量恢复）；失败调用推进窗口判定（错误率≥阈值且
        窗口≥最小调用数，或连续失败≥阈值）。
        """
        if not await get_health_check_enabled(redis, provider_id):
            return

        epoch_ms = _now_ms()

        # 延迟窗口（保留最近 N 条）。
        # redis-py 的 list 命令（lpush/ltrim/lrange）签名标注为 Union[Awaitable, T]
        # （同步/异步客户端共用签名），await 触发 Pylance 误报，属库的类型标注缺陷
        latency_key = _LATENCY_KEY.format(provider_id)
        await redis.lpush(latency_key, latency_ms)  # type: ignore
        await redis.ltrim(latency_key, 0, _LATENCY_WINDOW - 1)  # type: ignore

        # 调用窗口按小时分桶累加（O(1)），与熔断解除/快照失效同一管线提交
        bucket = epoch_ms // 1000 // _BUCKET_SECONDS
        window_key = _WINDOW_KEY.format(provider_id)
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hincrby(window_key, f"{bucket}:t", 1)
            if not success:
                pipe.hincrby(window_key, f"{bucket}:f", 1)
                if error_code == "429":
                    pipe.hincrby(window_key, f"{bucket}:l", 1)
            pipe.expire(window_key, _WINDOW_TTL)
            pipe.delete(
                _CIRCUIT_KEY.format(provider_id),
                _RECOVERY_KEY.format(provider_id),
                _PROBE_KEY.format(provider_id),
            )
            results = await pipe.execute()
        circuit_cleared = bool(results[-1])

        if success:
            await redis.delete(_STREAK_KEY.format(provider_id))
            if circuit_cleared:
                await CacheService(redis).delete(_SNAPSHOT_KEY.format(provider_id))
            return

        streak = await redis.incr(_STREAK_KEY.format(provider_id))
        await redis.expire(_STREAK_KEY.format(provider_id), 3600)

        # 仅失败调用参与熔断判定（成功调用只会降低错误率，无需开断）
        thresholds = await _load_thresholds(redis)
        circuit = streak >= thresholds.consecutive_failures

        if not circuit:
            total, failed = await self._window_counts(redis, provider_id, epoch_ms)
            if total >= thresholds.min_window_calls:
                error_rate = failed / total
                circuit = error_rate >= thresholds.error_rate_open

        if circuit:
            await self._open_circuit(redis, provider_id, thresholds.circuit_cooldown)

        # 失效快照缓存，下次看板/列表读取时重建（不在调用热路径同步聚合）
        await CacheService(redis).delete(_SNAPSHOT_KEY.format(provider_id))

    @staticmethod
    async def _open_circuit(redis: Redis, provider_id: int, cooldown: int) -> None:
        """开启/重新开启熔断：冷却期内全阻断，恢复周期标记覆盖冷却+探测全程。"""
        await redis.set(_CIRCUIT_KEY.format(provider_id), 1, ex=cooldown)
        await redis.set(_RECOVERY_KEY.format(provider_id), 1, ex=cooldown * 2)

    async def get_health_snapshot(self, redis: Redis, provider_id: int) -> dict:
        """返回供应商健康快照（看板用）：状态、成功率、429 率、P95、调用量等。

        优先读缓存（60s），miss 时按滑动窗口聚合并回填——模型列表/看板逐供应商
        调用，避免每次全量 lrange 聚合。
        """
        cache = CacheService(redis)
        cached = await cache.get_json(_SNAPSHOT_KEY.format(provider_id))
        if cached is not None:
            return cached
        thresholds = await _load_thresholds(redis)
        now_ms = _now_ms()
        total, failed, limit = await self._window_counts(
            redis, provider_id, now_ms, return_detail=True
        )
        raw_latency = await redis.lrange(_LATENCY_KEY.format(provider_id), 0, -1)  # type: ignore
        latency = [int(x) for x in raw_latency]
        p95 = _p95(latency)
        circuit_open = bool(await redis.get(_CIRCUIT_KEY.format(provider_id)))

        if not await get_health_check_enabled(redis, provider_id):
            status = "healthy"
        elif circuit_open:
            status = "open"
        elif total >= thresholds.min_window_calls:
            error_rate = failed / total
            if error_rate >= thresholds.error_rate_open:
                status = "open"
            elif error_rate >= thresholds.error_rate_warn:
                status = "suspicious"
            else:
                status = "healthy"
        else:
            status = "healthy"

        snapshot = {
            "status": status,
            "circuit_open": circuit_open,
            "total_calls_24h": total,
            "success_rate": round((total - failed) / total, 4) if total else 1.0,
            "error_rate": round(failed / total, 4) if total else 0.0,
            "limit_rate": round(limit / total, 4) if total else 0.0,
            "p95_latency_ms": p95,
        }
        await CacheService(redis).set_json(_SNAPSHOT_KEY.format(provider_id), snapshot, 60)
        return snapshot

    async def close_circuit(self, redis: Redis, provider_id: int) -> None:
        """管理员手动解除熔断：清除熔断标记、恢复周期与探测租约、连续失败计数
        与快照缓存（立即反映解除）。"""
        await redis.delete(
            _CIRCUIT_KEY.format(provider_id),
            _RECOVERY_KEY.format(provider_id),
            _PROBE_KEY.format(provider_id),
            _STREAK_KEY.format(provider_id),
        )
        await CacheService(redis).delete(_SNAPSHOT_KEY.format(provider_id))

    async def _window_counts(
        self,
        redis: Redis,
        provider_id: int,
        now_ms: int,
        return_detail: bool = False,
    ) -> tuple:
        """统计近 24h 调用记录总数/失败数/限流数。

        小时分桶计数：一次 HMGET 读最近 24 个桶的 t/f/l 字段，开销与调用量无关
        （旧实现 lrange 全量调用记录后逐条过滤，失败路径上随调用量线性放大）。
        """
        bucket = now_ms // 1000 // _BUCKET_SECONDS
        fields = [
            f"{bucket - offset}:{suffix}"
            for offset in range(_WINDOW_BUCKETS)
            for suffix in ("t", "f", "l")
        ]
        # redis-py 同步/异步客户端共用 hmget 签名（Union[Awaitable[list], list]）：
        # 此处为异步客户端，运行时恒返回协程，await 正确，cast 仅收窄类型（非消音）。
        raw = await cast(Awaitable[list], redis.hmget(_WINDOW_KEY.format(provider_id), fields))
        total = failed = limit = 0
        for i in range(_WINDOW_BUCKETS):
            total += int(raw[i * 3] or 0)
            failed += int(raw[i * 3 + 1] or 0)
            limit += int(raw[i * 3 + 2] or 0)
        return (total, failed, limit) if return_detail else (total, failed)


provider_health_service = ProviderHealthService()
