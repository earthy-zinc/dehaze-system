"""任务生命周期域测试：创建/幂等/取消/重试/状态查询/导出下载前置校验。

覆盖状态机非法流转（参数化）、幂等键按用户隔离、跨用户越权、
对抗性脏 task_id、列表分页与状态计数不变量。
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import TaskStatus
from app.repository.task_repository import task_repository
from app.service.task import task_service
from app.service.task.task_state import is_task_cancelled

pytestmark = pytest.mark.requires_db

USER_A = 1
USER_B = 2


def _insert_task(
    db,
    task_id: str,
    user_id: int,
    status: int,
    *,
    task_type: str = "user_export",
    result=None,
    error_message: str | None = None,
    retry_count: int = 0,
    created_at: datetime | None = None,
) -> SysTask:
    now = created_at or datetime.now()
    row = SysTask(
        task_id=task_id,
        task_type=task_type,
        status=status,
        progress=100 if status == TaskStatus.COMPLETED.value else 0,
        total_files=0,
        processed_files=0,
        params="{}",
        result=result,
        error_message=error_message,
        retry_count=retry_count,
        create_by=user_id,
        create_time=now,
        expires_at=now + timedelta(hours=24),
    )
    db.add(row)
    return row


@pytest.fixture(autouse=True)
def _no_dispatch(monkeypatch):
    """屏蔽任务分发（MQ/本地执行），单测只验证生命周期逻辑"""
    monkeypatch.setattr(task_service, "_dispatch_task", AsyncMock())


async def test_create_task_rejects_invalid_type(db, mock_redis):
    with pytest.raises(BusinessException) as exc_info:
        await task_service.create_task(db, mock_redis, "invalid_type", "{}", USER_A)
    assert exc_info.value.code == ResultCode.TASK_TYPE_UNSUPPORTED


async def test_create_task_idempotent_same_user(db, mock_redis):
    first = await task_service.create_task(
        db, mock_redis, "user_export", '{"format":"excel"}', USER_A, idempotency_key="key-a"
    )
    second = await task_service.create_task(
        db, mock_redis, "user_export", '{"format":"excel"}', USER_A, idempotency_key="key-a"
    )
    assert first["task_id"] == second["task_id"]
    assert second["status"] == TaskStatus.PENDING.value


async def test_create_task_idempotency_key_isolated_by_user(db, mock_redis):
    """幂等键按用户隔离：用户 B 复用用户 A 的键返回参数错误而非 A 的任务"""
    await task_service.create_task(
        db, mock_redis, "user_export", "{}", USER_A, idempotency_key="key-shared"
    )
    with pytest.raises(BusinessException) as exc_info:
        await task_service.create_task(
            db, mock_redis, "user_export", "{}", USER_B, idempotency_key="key-shared"
        )
    assert exc_info.value.code == ResultCode.TASK_PARAM_ERROR


@pytest.mark.parametrize(
    "task_id",
    [
        "task-🚀-emoji",
        "task\u200bzero\u200bwidth",
        "task\r\ncrlf",
        "task　full-width-space",
        "t" * 300,
        "task\x00null",
    ],
)
async def test_get_status_with_adversarial_task_ids(db, mock_redis, task_id):
    """脏 task_id（emoji/零宽/CRLF/全半角/超长/空字节）不触发异常，按不存在处理"""
    assert (await task_service.get_task_status(db, mock_redis, task_id, user_id=USER_A)) is None


async def test_get_status_ownership(db, mock_redis):
    _insert_task(db, "t-own-1", USER_A, TaskStatus.PROCESSING.value)
    await db.flush()

    data = await task_service.get_task_status(db, mock_redis, "t-own-1", user_id=USER_A)
    assert data is not None
    assert data["task_id"] == "t-own-1"

    with pytest.raises(BusinessException) as exc_info:
        await task_service.get_task_status(db, mock_redis, "t-own-1", user_id=USER_B)
    assert exc_info.value.code == ResultCode.TASK_UNAUTHORIZED


async def test_cancel_pending_task(db, mock_redis):
    _insert_task(db, "t-cancel-1", USER_A, TaskStatus.PENDING.value)
    await db.flush()

    await task_service.cancel_task(db, mock_redis, "t-cancel-1", user_id=USER_A)

    row = await task_repository.get_by_task_id(db, "t-cancel-1")
    assert row is not None
    assert row.status == TaskStatus.CANCELLED.value
    assert row.completed_at is not None
    assert await is_task_cancelled(mock_redis, "t-cancel-1") is True


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (TaskStatus.COMPLETED.value, ResultCode.TASK_STATUS_INVALID),
        (TaskStatus.FAILED.value, ResultCode.TASK_STATUS_INVALID),
        (TaskStatus.CANCELLED.value, ResultCode.TASK_CANCELLED),
    ],
)
async def test_cancel_rejects_terminal_states(db, mock_redis, status, expected_code):
    """状态机非法流转：终态任务不可取消（含重复取消返回 B0306）"""
    _insert_task(db, f"t-cancel-{status}", USER_A, status)
    await db.flush()

    with pytest.raises(BusinessException) as exc_info:
        await task_service.cancel_task(db, mock_redis, f"t-cancel-{status}", user_id=USER_A)
    assert exc_info.value.code == expected_code


async def test_cancel_other_user_task_rejected(db, mock_redis):
    _insert_task(db, "t-cancel-other", USER_A, TaskStatus.PROCESSING.value)
    await db.flush()

    with pytest.raises(BusinessException) as exc_info:
        await task_service.cancel_task(db, mock_redis, "t-cancel-other", user_id=USER_B)
    assert exc_info.value.code == ResultCode.TASK_UNAUTHORIZED


async def test_retry_failed_task(db, mock_redis):
    _insert_task(
        db,
        "t-retry-1",
        USER_A,
        TaskStatus.FAILED.value,
        error_message="boom",
        retry_count=0,
    )
    await db.flush()

    data = await task_service.retry_task(db, mock_redis, "t-retry-1", user_id=USER_A)

    assert data["status"] == TaskStatus.PENDING.value
    assert data["retry_count"] == 1
    # 绕过 identity map 直查 DB，验证重置字段确实落库
    await db.flush()
    stored = (
        await db.execute(select(SysTask.error_message).where(SysTask.task_id == "t-retry-1"))
    ).scalar_one()
    assert stored is None


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.PENDING.value,
        TaskStatus.PROCESSING.value,
        TaskStatus.COMPLETED.value,
        TaskStatus.CANCELLED.value,
    ],
)
async def test_retry_rejects_non_failed_states(db, mock_redis, status):
    """状态机非法流转：仅失败任务可重试"""
    _insert_task(db, f"t-retry-{status}", USER_A, status)
    await db.flush()

    with pytest.raises(BusinessException) as exc_info:
        await task_service.retry_task(db, mock_redis, f"t-retry-{status}", user_id=USER_A)
    assert exc_info.value.code == ResultCode.TASK_STATUS_INVALID


async def test_retry_other_user_task_rejected(db, mock_redis):
    _insert_task(db, "t-retry-other", USER_A, TaskStatus.FAILED.value)
    await db.flush()

    with pytest.raises(BusinessException) as exc_info:
        await task_service.retry_task(db, mock_redis, "t-retry-other", user_id=USER_B)
    assert exc_info.value.code == ResultCode.TASK_UNAUTHORIZED


async def test_get_export_object_name(db, mock_redis):
    _insert_task(
        db, "t-dl-ok", USER_A, TaskStatus.COMPLETED.value, result="exports/t-dl-ok/user_export.xlsx"
    )
    object_name = await task_service.get_export_object_name(
        db, mock_redis, "t-dl-ok", user_id=USER_A
    )
    assert object_name == "exports/t-dl-ok/user_export.xlsx"


async def test_get_export_object_name_import_task_rejected(db, mock_redis):
    """导入任务 result 为 JSON 对象，无下载文件（B0303）"""
    _insert_task(
        db,
        "t-dl-import",
        USER_A,
        TaskStatus.COMPLETED.value,
        result={"totalRows": 2, "successCount": 2},
        task_type="user_import",
    )
    with pytest.raises(BusinessException) as exc_info:
        await task_service.get_export_object_name(db, mock_redis, "t-dl-import", user_id=USER_A)
    assert exc_info.value.code == ResultCode.TASK_STATUS_INVALID


@pytest.mark.parametrize(
    ("status", "result", "message_part"),
    [
        (TaskStatus.PENDING.value, "exports/x.zip", "任务未完成"),
        (TaskStatus.FAILED.value, "exports/x.zip", "任务未完成"),
    ],
)
async def test_get_export_object_name_rejects_unfinished(
    db, mock_redis, status, result, message_part
):
    _insert_task(db, f"t-dl-{status}", USER_A, status, result=result)
    with pytest.raises(BusinessException) as exc_info:
        await task_service.get_export_object_name(db, mock_redis, f"t-dl-{status}", user_id=USER_A)
    assert exc_info.value.code == ResultCode.TASK_STATUS_INVALID
    assert message_part in exc_info.value.message


async def test_get_export_object_name_expired(db, mock_redis):
    row = _insert_task(
        db, "t-dl-expired", USER_A, TaskStatus.COMPLETED.value, result="exports/x.zip"
    )
    row.expires_at = datetime.now() - timedelta(hours=1)
    await db.flush()

    with pytest.raises(BusinessException) as exc_info:
        await task_service.get_export_object_name(db, mock_redis, "t-dl-expired", user_id=USER_A)
    assert exc_info.value.code == ResultCode.TASK_STATUS_INVALID


async def test_get_export_object_name_ownership(db, mock_redis):
    _insert_task(db, "t-dl-other", USER_A, TaskStatus.COMPLETED.value, result="exports/x.zip")
    with pytest.raises(BusinessException) as exc_info:
        await task_service.get_export_object_name(db, mock_redis, "t-dl-other", user_id=USER_B)
    assert exc_info.value.code == ResultCode.TASK_UNAUTHORIZED


async def test_list_tasks_invariant_and_isolation(db, mock_redis):
    """不变量：用户列表 total 与该用户任务总数一致；他人任务不可见"""
    statuses = [
        TaskStatus.PENDING.value,
        TaskStatus.PROCESSING.value,
        TaskStatus.COMPLETED.value,
        TaskStatus.FAILED.value,
        TaskStatus.CANCELLED.value,
    ]
    for i, status in enumerate(statuses):
        _insert_task(db, f"t-list-{i}", USER_A, status)
    _insert_task(db, "t-list-other", USER_B, TaskStatus.COMPLETED.value)
    await db.flush()

    data = await task_service.list_tasks(db, USER_A, page=1, size=10)
    assert data["total"] == len(statuses)
    assert {t["status"] for t in data["list"]} == set(statuses)
    assert all(t["create_by"] == USER_A for t in data["list"])

    # 状态计数不变量：各状态筛选之和等于总数
    sum_by_status = 0
    for status in statuses:
        filtered = await task_service.list_tasks(db, USER_A, status=status, page=1, size=10)
        assert all(t["status"] == status for t in filtered["list"])
        sum_by_status += filtered["total"]
    assert sum_by_status == len(statuses)

    # 类别筛选：导入类别不应包含导出任务
    imports = await task_service.list_tasks(db, USER_A, task_category="import", page=1, size=10)
    assert all(t["task_type"].endswith("_import") for t in imports["list"])


async def test_list_tasks_pagination_boundary(db, mock_redis):
    base = datetime.now()
    for i in range(5):
        _insert_task(
            db,
            f"t-page-{i}",
            USER_A,
            TaskStatus.COMPLETED.value,
            created_at=base - timedelta(minutes=i),
        )
    await db.flush()

    page1 = await task_service.list_tasks(db, USER_A, page=1, size=2)
    page2 = await task_service.list_tasks(db, USER_A, page=2, size=2)
    page3 = await task_service.list_tasks(db, USER_A, page=3, size=2)
    page4 = await task_service.list_tasks(db, USER_A, page=4, size=2)

    assert len(page1["list"]) == 2
    assert len(page2["list"]) == 2
    assert len(page3["list"]) == 1
    assert page4["list"] == []
    # 全量 task_id 不重不漏
    ids = {t["task_id"] for p in (page1, page2, page3) for t in p["list"]}
    assert ids == {f"t-page-{i}" for i in range(5)}
