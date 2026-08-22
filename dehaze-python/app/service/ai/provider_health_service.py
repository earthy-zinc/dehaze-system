"""供应商健康与熔断服务

健康指标由调用链路实时聚合到 Redis（滑动窗口近 24h），熔断标记走
Redis circuit_open（TTL 冷却期），不新增库表。阈值从 sys_dict 读取
（ai_provider_health 前缀），缺省值仅在种子 SQL 中。

Redis Key 约定：
- ai:provider:{id}:circuit_open      熔断标记（存在即熔断，TTL=冷却时长）
- ai:provider:{id}:fail_streak       连续失败计数（成功清零）
- ai:provider:{id}:calls             调用记录列表（LPUSH，成员="{epoch_ms}:{ok|limit|fail}"）
- ai:provider:{id}:latency           延迟列表（LPUSH，成员=延迟毫秒）
- ai:provider:{id}:health            聚合快照缓存（status/success_rate/limit_rate/p95 等）
- ai:provider:{id}:health_enabled    健康检查开关缓存（由供应商 CRUD 写入）
- ai:provider:health:thresholds      熔断阈值缓存（sys_dict ai_provider_health）
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from redis.asyncio import Redis

from app.database import get_db_session
from app.infrastructure.cache.cache import CacheService
from app.repository.dict_repository import dict_repository

logger = logging.getLogger(__name__)

# 阈值字典类型（值存于 sys_dict）
_HEALTH_DICT = "ai_provider_health"

# 调用记录 / 延迟滑动窗口保留条数（简化分桶：保留最近 N 条取分位/成功率）
_CALLS_WINDOW = 5000
_LATENCY_WINDOW = 500

# 阈值缓存 TTL（秒）
_THRESHOLDS_TTL = 300

# 健康检查开关缓存前缀
HEALTH_ENABLED_PREFIX = "ai:provider:{}:health_enabled"

# 熔断标记 / 连续失败 / 调用记录 / 延迟 / 快照 Key 前缀
_CIRCUIT_KEY = "ai:provider:{}:circuit_open"
_STREAK_KEY = "ai:provider:{}:fail_streak"
_CALLS_KEY = "ai:provider:{}:calls"
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


async def _load_thresholds(redis: Redis) -> _Thresholds:
    """读取熔断阈值（Redis 缓存，TTL 防高频查库）。

    阈值来源唯一为 sys_dict（ai_provider_health），供应商健康阈值变更低频，
    缓存 5 分钟；缓存未命中时读取 sys_dict 并回填缓存，缺省键回落种子默认
    （与 config/sql/data/sys_dict.sql 的 ai_provider_health 同源）。
    """
    cache = CacheService(redis)
    cached = await cache.get_json(_THRESHOLDS_KEY)
    if cached is not None:
        return _Thresholds(cached)
    data = dict(_SEED_THRESHOLDS)
    try:
        async with get_db_session() as db:
            items = await dict_repository.list_enabled_by_type_code(db, _HEALTH_DICT)
            for item in items:
                data[item.name] = _coerce_scalar(item.value)
    except Exception as exc:  # noqa: BLE001 - 阈值读取失败回落种子默认，不影响健康判定
        logger.warning("读取供应商健康阈值失败，使用种子默认: %s", exc)
    await cache.set_json(_THRESHOLDS_KEY, data, _THRESHOLDS_TTL)
    return _Thresholds(data)


def _coerce_scalar(raw: str) -> float | int:
    """将 sys_dict 字符串值转换为数值（int/float）。"""
    try:
        return int(str(raw))
    except (ValueError, TypeError):
        try:
            return float(str(raw))
        except (ValueError, TypeError):
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
        _STREAK_KEY.format(provider_id),
        _CALLS_KEY.format(provider_id),
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
        """返回供应商健康状态：healthy | suspicious | open。

        高频调用链路的快速路径：仅读健康开关与熔断标记，不做聚合。
        可疑判定聚合结果取自缓存的健康快照。
        """
        if not await get_health_check_enabled(redis, provider_id):
            return "healthy"
        if await redis.get(_CIRCUIT_KEY.format(provider_id)):
            return "open"
        snapshot = await CacheService(redis).get_json(_SNAPSHOT_KEY.format(provider_id))
        if snapshot and snapshot.get("status") in ("healthy", "suspicious"):
            return snapshot["status"]
        return "healthy"

    async def record_call(
        self,
        redis: Redis,
        provider_id: int,
        success: bool,
        error_code: str | None,
        latency_ms: int,
    ) -> None:
        """记录一次供应商调用并内联判定是否熔断。

        健康检查关闭的供应商不参与聚合与判定。成功调用清零连续失败计数；
        失败调用推进窗口判定（错误率≥阈值且窗口≥最小调用数，或连续失败≥阈值）。
        """
        if not await get_health_check_enabled(redis, provider_id):
            return

        epoch_ms = int(datetime.now(UTC).timestamp() * 1000)

        # 延迟窗口（保留最近 N 条）
        await redis.lpush(_LATENCY_KEY.format(provider_id), latency_ms)
        await redis.ltrim(_LATENCY_KEY.format(provider_id), 0, _LATENCY_WINDOW - 1)

        if success:
            await redis.delete(_STREAK_KEY.format(provider_id))
            await redis.lpush(_CALLS_KEY.format(provider_id), f"{epoch_ms}:ok")
        else:
            streak = await redis.incr(_STREAK_KEY.format(provider_id))
            await redis.expire(_STREAK_KEY.format(provider_id), 3600)
            outcome = "limit" if error_code == "429" else "fail"
            await redis.lpush(_CALLS_KEY.format(provider_id), f"{epoch_ms}:{outcome}")

        await redis.ltrim(_CALLS_KEY.format(provider_id), 0, _CALLS_WINDOW - 1)

        # 仅失败调用参与熔断判定（成功调用只会降低错误率，无需开断）
        if success:
            return

        thresholds = await _load_thresholds(redis)
        circuit = streak >= thresholds.consecutive_failures

        if not circuit:
            total, failed = await self._window_counts(redis, provider_id, epoch_ms)
            if total >= thresholds.min_window_calls:
                error_rate = failed / total
                circuit = error_rate >= thresholds.error_rate_open

        if circuit:
            await redis.set(_CIRCUIT_KEY.format(provider_id), 1, ex=thresholds.circuit_cooldown)

        # 失效快照缓存，下次看板/列表读取时重建（不在调用热路径同步聚合）
        await CacheService(redis).delete(_SNAPSHOT_KEY.format(provider_id))

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
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        total, failed, limit = await self._window_counts(
            redis, provider_id, now_ms, return_detail=True
        )
        latency = [int(x) for x in await redis.lrange(_LATENCY_KEY.format(provider_id), 0, -1)]
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
        """管理员手动解除熔断：清除熔断标记、连续失败计数与快照缓存（立即反映解除）。"""
        await redis.delete(
            _CIRCUIT_KEY.format(provider_id),
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

        滑动窗口：读取最近保留的调用记录，过滤 24h 内按结果归类。
        """
        cutoff = now_ms - 86_400_000
        raw = await redis.lrange(_CALLS_KEY.format(provider_id), 0, -1)
        total = failed = limit = 0
        for item in raw:
            try:
                ts_str, _, outcome = item.partition(":")
                if int(ts_str) < cutoff:
                    continue
                total += 1
                if outcome == "fail":
                    failed += 1
                elif outcome == "limit":
                    failed += 1
                    limit += 1
            except ValueError:
                continue
        return (total, failed, limit) if return_detail else (total, failed)


provider_health_service = ProviderHealthService()
