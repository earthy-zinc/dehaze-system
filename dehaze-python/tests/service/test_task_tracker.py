"""TaskTracker 全局任务视图：键扫描精确性 + 异常收窄回归（免库）。

背景：全局视图用 `task:running:*` 扫描，会连带命中同前缀的非追踪键
`task:running:cache:{id}` / `task:running:progress:{id}`（均为 string 类型）：

- `get_global_running_count` 直接按扫描命中数计数 ⇒ 计数虚高；
- `get_global_running_tasks` 对 string key 调 hgetall 抛 WRONGTYPE（RedisError），
  被外层 `except Exception` 吞掉后静默回退本 Worker 视图 ⇒ 真相被掩盖；
  `split(":", 2)[-1]` 还会把 `cache:{id}` 解析成假 task_id。

现口径：按精确键形态显式排除非追踪命名空间；外层只捕 RedisError（Redis 不可用时的
已声明降级），其余异常一律上抛（不再有"任何异常都回退"的兜底）。
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator

import pytest
from redis.exceptions import RedisError

from app.core.constants import TASK_REDIS_KEY_PREFIX
from app.service.task_tracker import TaskTracker, TrackedTask

_TRACKER = f"{TASK_REDIS_KEY_PREFIX}:"


class _FailingRedis:
    """伪客户端：scan_iter 按需抛出指定异常（模拟 Redis 故障 / 非预期错误）"""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def scan_iter(self, match: str = "*", count: int = 10) -> AsyncIterator[str]:
        raise self._exc


@pytest.fixture
async def tracker(mock_redis) -> AsyncIterator[TaskTracker]:
    tracker = TaskTracker()
    await tracker.start(mock_redis)
    yield tracker
    await tracker.stop()


@contextlib.asynccontextmanager
async def _local_running_task(tracker: TaskTracker, task_id: str) -> AsyncIterator[None]:
    """在追踪器内挂一个本 Worker 运行中的任务，用于识别"静默回退本端视图"。

    （回退一旦发生，全局视图会吐出这个只存在于本进程的任务。）
    """
    task = asyncio.create_task(asyncio.sleep(3600))
    tracker._running_tasks[task_id] = TrackedTask(task_id=task_id, task=task, task_type="dummy")
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        tracker._running_tasks.pop(task_id, None)


async def _seed_tracker_task(mock_redis, task_id: str, metadata: str = "{}") -> None:
    await mock_redis.hset(
        f"{_TRACKER}{task_id}",
        mapping={
            "task_type": "prediction",
            "worker": "1",
            "started_at": "1.0",
            "metadata": metadata,
        },
    )


async def _seed_non_tracker_keys(mock_redis) -> None:
    """同前缀的非追踪键：任务缓存 / 任务进度（string 类型，hgetall 会 WRONGTYPE）"""
    await mock_redis.set(f"{_TRACKER}cache:abc", json.dumps({"task_id": "abc"}))
    await mock_redis.set(f"{_TRACKER}progress:abc", json.dumps({"progress": 50}))


def _tracker_with(monkeypatch: pytest.MonkeyPatch, redis_like: object) -> TaskTracker:
    tracker = TaskTracker()
    monkeypatch.setattr(tracker, "_redis", redis_like)
    return tracker


async def test_global_view_excludes_non_tracker_namespaces(tracker, mock_redis):
    await _seed_tracker_task(mock_redis, "pred:7")
    await _seed_tracker_task(mock_redis, "eval:9")
    await _seed_non_tracker_keys(mock_redis)

    tasks = await tracker.get_global_running_tasks()

    assert {t["task_id"] for t in tasks} == {"pred:7", "eval:9"}
    assert all("cache:" not in t["task_id"] for t in tasks)
    assert all("progress:" not in t["task_id"] for t in tasks)
    # 计数同样只算追踪键（旧实现会把 cache/progress 两条一起算进去）
    assert await tracker.get_global_running_count() == 2


async def test_global_view_does_not_fall_back_on_non_tracker_keys(tracker, mock_redis):
    """只有非追踪键时：全局视图必须为空，不得回退成"本 Worker 视图"。

    （旧实现 hgetall 抛 WRONGTYPE 后静默回退 ⇒ 返回下面这个只存在于本进程的任务。）
    """
    async with _local_running_task(tracker, "local-only"):
        await _seed_non_tracker_keys(mock_redis)

        assert await tracker.get_global_running_tasks() == []
        assert await tracker.get_global_running_count() == 0


async def test_broken_metadata_is_logged_and_skipped(tracker, mock_redis, caplog):
    """单条 metadata 损坏：留 error 日志并跳过该条，不影响其余任务（不打整表回退）"""
    await _seed_tracker_task(mock_redis, "pred:7", metadata="{not json")
    await _seed_tracker_task(mock_redis, "eval:9")

    with caplog.at_level(logging.ERROR):
        tasks = await tracker.get_global_running_tasks()

    assert [t["task_id"] for t in tasks] == ["eval:9"]
    assert any(
        "非法 JSON" in record.getMessage() and "pred:7" in record.getMessage()
        for record in caplog.records
    )


async def test_redis_error_falls_back_to_local_view(monkeypatch, caplog):
    """预期异常（Redis 不可用）：保留已声明的降级，但必须以 error 级留痕"""
    tracker = _tracker_with(monkeypatch, _FailingRedis(RedisError("redis down")))

    async with _local_running_task(tracker, "local-only"):
        with caplog.at_level(logging.ERROR):
            tasks = await tracker.get_global_running_tasks()

        assert [t["task_id"] for t in tasks] == ["local-only"]
        assert await tracker.get_global_running_count() == 1

    assert any(
        record.levelno == logging.ERROR and "Redis 不可用" in record.getMessage()
        for record in caplog.records
    )


async def test_unexpected_error_propagates(monkeypatch):
    """非预期异常必须上抛（收窄前 `except Exception` 会把它吞成静默回退）"""
    tracker = _tracker_with(monkeypatch, _FailingRedis(ValueError("boom")))

    with pytest.raises(ValueError, match="boom"):
        await tracker.get_global_running_tasks()
    with pytest.raises(ValueError, match="boom"):
        await tracker.get_global_running_count()


def test_tracker_namespaces_are_derived_from_constants():
    """防回归锚点：非追踪命名空间必须仍精确指向 cache/progress 前缀"""
    from app.service.task_tracker import _NON_TRACKER_KEY_PREFIXES

    assert tuple(_NON_TRACKER_KEY_PREFIXES) == ("task:running:cache:", "task:running:progress:")
