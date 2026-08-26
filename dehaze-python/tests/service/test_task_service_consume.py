import json
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from app.core.constants import TASK_CACHE_PREFIX

pytestmark = pytest.mark.requires_db
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import TaskStatus
from app.service.task.task_consumer import consume_dlq_message, consume_export_message


def _make_task(status: int) -> SysTask:
    task = SysTask(
        task_id="t1",
        task_type="user_export",
        status=status,
        progress=0,
        total_files=0,
        processed_files=0,
        params="{}",
        result=None,
        error_message=None,
        retry_count=0,
        create_by=1,
        create_time=datetime.now(),
    )
    task.id = 1
    return task


def _patch_consumption(monkeypatch, task: SysTask):
    from app.repository.task_repository import task_repository as repo

    monkeypatch.setattr(repo, "get_by_id", AsyncMock(return_value=task))
    execute = AsyncMock()
    monkeypatch.setattr("app.service.task.task_consumer.execute_task_background", execute)
    push = AsyncMock()
    monkeypatch.setattr("app.service.task.task_consumer.push_task_ws_message", push)
    return execute, push


async def test_consume_export_skips_terminal_state(monkeypatch, db, mock_redis):
    task = _make_task(TaskStatus.COMPLETED.value)
    execute, push = _patch_consumption(monkeypatch, task)

    await consume_export_message(
        {"db_task_id": 1, "task_id": "t1", "task_type": "user_export"}, {}
    )

    execute.assert_not_awaited()
    push.assert_not_awaited()


async def test_consume_export_executes_pending_task(monkeypatch, db, mock_redis):
    task = _make_task(TaskStatus.PENDING.value)
    execute, _ = _patch_consumption(monkeypatch, task)

    await consume_export_message(
        {"db_task_id": 1, "task_id": "t1", "task_type": "user_export"}, {}
    )

    execute.assert_awaited_once_with(1, "t1", "user_export", "{}")


async def test_consume_export_updates_retry_count(monkeypatch, db, mock_redis):
    task = _make_task(TaskStatus.PENDING.value)
    execute, _ = _patch_consumption(monkeypatch, task)

    from app.repository.task_repository import task_repository as repo

    update_retry = AsyncMock(return_value=1)
    monkeypatch.setattr(repo, "update_retry_count", update_retry)

    await consume_export_message(
        {"db_task_id": 1, "task_id": "t1", "task_type": "user_export"},
        {"x-retry-count": "2"},
    )

    update_retry.assert_awaited_once()
    execute.assert_awaited_once()


async def test_consume_dlq_skips_terminal_state(monkeypatch, db, mock_redis):
    task = _make_task(TaskStatus.COMPLETED.value)
    _, push = _patch_consumption(monkeypatch, task)

    await consume_dlq_message(
        {"db_task_id": 1, "task_id": "t1", "task_type": "user_export"}, {}
    )

    assert task.status == TaskStatus.COMPLETED.value
    push.assert_not_awaited()
    assert await mock_redis.get(TASK_CACHE_PREFIX + "t1") is None


async def test_consume_dlq_marks_failed_and_refreshes_cache(monkeypatch, db, mock_redis):
    task = _make_task(TaskStatus.PENDING.value)
    _, push = _patch_consumption(monkeypatch, task)

    await consume_dlq_message(
        {"db_task_id": 1, "task_id": "t1", "task_type": "user_export"},
        {"x-retry-count": "3"},
    )

    assert task.status == TaskStatus.FAILED.value
    assert task.retry_count == 3
    assert task.completed_at is not None

    cached = json.loads(await mock_redis.get(TASK_CACHE_PREFIX + "t1"))
    assert cached["status"] == TaskStatus.FAILED.value
    assert cached["error_message"] == "消息重试耗尽进入死信队列（重试次数: 3）"
    push.assert_awaited_once()
