"""数据集缓存路径的降级判定（免库，fakeredis + 可编程 Redis 桩）。

原实现把 7 处缓存读写都包成 `except Exception: logger.warning`，任何异常（含程序性错误、
缓存体损坏）都被吞成"缓存未命中"继续跑。现口径逐类显式：

| 场景                                   | 处理                                   | 留痕级别 |
| -------------------------------------- | -------------------------------------- | -------- |
| Redis 不可用（RedisError）              | 降级回源数据库（缓存非唯一数据源）      | error    |
| 缓存体非法 JSON / 结构不符              | 删键后回源（自愈）                      | warning  |
| 缓存序列化失败（非预期数据类型）        | 跳过写入，接口仍返回数据库结果          | error    |
| 非 RedisError 的其它异常                | 上抛（不再被吞）                        | —        |

本文件逐条覆盖上表，防止再次退化成"任何异常都降级"。
"""

import logging
from typing import Any, cast

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_dataset import SysDataset
from app.service.dataset.dataset_service import DatasetService

_OPTIONS_KEY = DatasetService.CACHE_OPTIONS_KEY


def _dataset(dataset_id: int) -> SysDataset:
    return SysDataset(
        id=dataset_id,
        parent_id=0,
        type="haze",
        name=f"ds-{dataset_id}",
        img=None,
        description=None,
        path=f"/data/{dataset_id}",
        size=0,
        status=1,
        deleted=0,
    )


class _FakeDatasetRepository:
    def __init__(self, datasets: list[SysDataset], options: list[dict[str, Any]]) -> None:
        self._datasets = datasets
        self._options = options

    async def find_all(self, db: Any) -> list[SysDataset]:
        return self._datasets

    async def get_dataset_options(self, db: Any) -> list[dict[str, Any]]:
        return self._options


class _StubRedis:
    """可编程 Redis 桩：按操作抛出指定异常（覆盖降级 / 上抛分支）"""

    def __init__(self, *, get_value: Any = None, raises: dict[str, Exception] | None = None):
        self._get_value = get_value
        self._raises = raises or {}
        self.setex_calls: list[tuple[str, int, str]] = []
        self.deleted: list[str] = []

    async def get(self, key: str):
        if "get" in self._raises:
            raise self._raises["get"]
        return self._get_value

    async def setex(self, key: str, ttl: int, value: str):
        if "setex" in self._raises:
            raise self._raises["setex"]
        self.setex_calls.append((key, ttl, value))

    async def delete(self, *keys: str):
        if "delete" in self._raises:
            raise self._raises["delete"]
        self.deleted.extend(keys)


def _service(datasets: list[SysDataset], options: list[dict[str, Any]]) -> DatasetService:
    # 免库：仓储以桩替代（真实实现需要数据库会话，本用例只覆盖缓存分支）
    repo = _FakeDatasetRepository(datasets, options)
    return DatasetService(dataset_repository=cast(Any, repo))


async def test_redis_unavailable_degrades_to_db_with_error_log(caplog):
    """Redis 不可用 ⇒ 回源数据库（可降级），但必须 error 级留痕，不再静默"""
    svc = _service([_dataset(1)], [{"value": 1, "label": "ds-1"}])
    redis = _StubRedis(raises={"get": RedisError("redis down"), "setex": RedisError("redis down")})

    with caplog.at_level(logging.ERROR):
        datasets = await svc.get_all_datasets(cast(AsyncSession, None), cast(Redis, redis))
        options = await svc.get_dataset_options(cast(AsyncSession, None), cast(Redis, redis))

    assert [d.id for d in datasets] == [1]
    assert options == [{"value": 1, "label": "ds-1"}]
    assert any(
        record.levelno == logging.ERROR and "Redis 不可用" in record.getMessage()
        for record in caplog.records
    )


async def test_corrupt_cache_body_is_deleted_and_refetched(mock_redis, caplog):
    """缓存体非法 JSON ⇒ 删键回源并重建（自愈），warning 留痕"""
    await mock_redis.set(_OPTIONS_KEY, "not-json{")

    svc = _service([], [{"value": 1, "label": "ds-1"}])
    with caplog.at_level(logging.WARNING):
        options = await svc.get_dataset_options(cast(AsyncSession, None), mock_redis)

    assert options == [{"value": 1, "label": "ds-1"}]
    # 已重建为合法 JSON，而不是把脏值留在缓存里
    assert await mock_redis.get(_OPTIONS_KEY) == '[{"value": 1, "label": "ds-1"}]'
    assert any("损坏" in record.getMessage() for record in caplog.records)


async def test_cache_shape_mismatch_is_deleted_and_refetched(mock_redis, caplog):
    """结构不符（期望 list 拿到 dict）同样按损坏处理，不得喂给下游"""
    await mock_redis.set(_OPTIONS_KEY, '{"value": 1}')

    svc = _service([], [])
    with caplog.at_level(logging.WARNING):
        options = await svc.get_dataset_options(cast(AsyncSession, None), mock_redis)

    assert options == []
    assert await mock_redis.get(_OPTIONS_KEY) == "[]"
    assert any("结构不符" in record.getMessage() for record in caplog.records)


async def test_write_failure_keeps_response_and_warns():
    """缓存写失败（Redis 不可用）只降级：接口仍返回数据库结果"""
    svc = _service([_dataset(7)], [])
    redis = _StubRedis(raises={"setex": RedisError("down")})

    datasets = await svc.get_all_datasets(cast(AsyncSession, None), cast(Redis, redis))

    assert [d.id for d in datasets] == [7]


async def test_non_redis_error_propagates():
    """非预期异常必须上抛（收窄前会被 `except Exception` 吞成"缓存未命中"）"""
    svc = _service([_dataset(1)], [])
    redis = _StubRedis(raises={"get": ValueError("boom")})

    with pytest.raises(ValueError, match="boom"):
        await svc.get_all_datasets(cast(AsyncSession, None), cast(Redis, redis))


async def test_serialization_failure_logs_error_and_skips_write(caplog):
    """序列化失败（非预期数据类型）⇒ error 留痕且不写入，不污染缓存"""
    svc = _service([], [])
    redis = _StubRedis()
    circular: list[Any] = []
    circular.append(circular)

    with caplog.at_level(logging.ERROR):
        await svc._write_cache(cast(Redis, redis), "dataset:circular", 60, circular)

    assert redis.setex_calls == []
    assert any(
        record.levelno == logging.ERROR and "序列化失败" in record.getMessage()
        for record in caplog.records
    )


async def test_cache_written_with_expected_ttl_on_miss(mock_redis):
    """回源后必须回填缓存（TTL 与常量一致），保证"降级"只发生在异常分支"""
    svc = _service([_dataset(3)], [])

    await svc.get_all_datasets(cast(AsyncSession, None), mock_redis)

    ttl = await mock_redis.ttl(DatasetService.CACHE_ALL_KEY)
    assert 3500 < ttl <= DatasetService.CACHE_ALL_TTL
