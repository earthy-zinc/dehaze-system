"""批量删除数据集失败的异常口径（免库，桩替换仓储与会话）。

原实现 `except Exception as e:` 只把 `str(e)` 塞进响应体的逐项失败结果，日志里不留任何
痕迹（连 warning 都没有）⇒ 数据层故障/程序性错误被完全吞没，线上只能看到"删除失败"。
现口径：**保留**宽捕（跨端契约是逐项 results 返回，不改成 500），但必须 `logger.exception`
带全栈留痕（日志规范 §6 exc_info），且整体回滚、有效数据集全部标记失败（不"部分成功继续跑"）。
本用例锁定"口径未被吞回去"：异常上抛被翻译成逐项失败 + error 日志带 traceback。
"""

import logging
from typing import Any, cast

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_dataset import SysDataset
from app.service.dataset.dataset_service import DatasetService

_TARGET_ID = 42


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


class _FailingDeleteRepository:
    """只读步骤正常、`delete_by_ids`（物理删除）故障的仓储桩"""

    def __init__(self) -> None:
        self.rolled_back = False

    async def get_by_ids(self, db: Any, ids: list[int]) -> list[SysDataset]:
        return [_dataset(i) for i in ids]

    async def get_all_descendant_ids_batch(self, db: Any, ids: list[int]) -> dict[int, list[int]]:
        return {i: [i] for i in ids}

    async def get_datasets_by_ids(self, db: Any, ids: list[int]) -> list[SysDataset]:
        return [_dataset(i) for i in ids]

    async def get_item_ids_by_dataset_ids(self, db: Any, ids: list[int]) -> list[int]:
        return []

    async def delete_by_ids(self, db: Any, ids: list[int]) -> None:
        raise RuntimeError("db down")


class _StubSession:
    async def rollback(self) -> None:  # pragma: no cover - 仅断言被调用
        pass


class _StubRedis:
    async def delete(self, *keys: str) -> int:
        return 0


class _StubAuditRepository:
    def create_audit_async(self, **kwargs: Any) -> None:
        pass


async def test_batch_delete_failure_logs_traceback_and_marks_all_failed(caplog):
    repo = _FailingDeleteRepository()
    svc = DatasetService(
        dataset_repository=cast(Any, repo),
        mongo_audit_log_repository=cast(Any, _StubAuditRepository()),
    )

    with caplog.at_level(logging.ERROR):
        result = await svc.delete_datasets(
            cast(AsyncSession, _StubSession()),
            cast(Redis, _StubRedis()),
            [_TARGET_ID],
        )

    # 跨端契约不变：逐项失败结果（errorCode=SYSTEM_ERROR），不变成 500/部分成功
    assert result["succeeded"] == 0
    assert result["failed"] == 1
    assert result["results"] == [
        {
            "id": _TARGET_ID,
            "status": "failed",
            "message": "db down",
            "errorCode": "SYSTEM_ERROR",
        }
    ]

    # 关键：异常不再被吞没——error 级日志带定位上下文与完整 traceback
    failures = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR and "批量删除数据集失败" in record.getMessage()
    ]
    assert len(failures) == 1, "批量删除失败必须留 error 日志"
    assert str(_TARGET_ID) in failures[0].getMessage()
    assert failures[0].exc_info is not None, "必须带 traceback（exc_info），不得只记 str(e)"
