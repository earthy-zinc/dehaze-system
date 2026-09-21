import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import run_after_commit_callbacks
from app.models.base import set_current_user_id
from app.models.entity.sys_ai_memory import SysAiMemory
from app.service.ai_memory_service import ai_memory_service
from tests.stubs.factories import make_orm_mem
from tests.stubs.fakes import StubAsyncSession


def _stub_db() -> AsyncSession:
    """StubAsyncSession 的 cast 收口：tests/stubs 结构型 DB 桩（实现 add/flush/info）。"""
    return cast(AsyncSession, StubAsyncSession())  # 替身：StubAsyncSession 非 AsyncSession 子类


@pytest.fixture(autouse=True)
def record_audit(monkeypatch):
    audits = []
    monkeypatch.setattr(
        "app.service.ai_memory_service.mongo_audit_log_repository",
        SimpleNamespace(create_audit_async=lambda **kw: audits.append(kw)),
    )
    return audits


class TestBatchClear:
    async def test_requires_confirm(self, monkeypatch):
        async def fake_batch_clear(db, user_id, memory_type, start, end):
            return 5

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.batch_clear", fake_batch_clear
        )
        with pytest.raises(BusinessException):
            await ai_memory_service.batch_clear(_stub_db(), 1, confirm=False)

    @pytest.mark.parametrize(
        ("kwargs", "count", "expected"),
        [
            ({}, 5, {"memory_type": None, "start": None, "end": None}),
            (
                {"memory_type": "semantic"},
                3,
                {"memory_type": "semantic", "start": None, "end": None},
            ),
            (
                {"start": datetime(2026, 1, 1), "end": datetime(2026, 1, 31)},
                2,
                {
                    "memory_type": None,
                    "start": datetime(2026, 1, 1),
                    "end": datetime(2026, 1, 31),
                },
            ),
        ],
    )
    async def test_clear_by_granularity(self, monkeypatch, kwargs, count, expected):
        captured = {}

        async def fake_batch_clear(db, user_id, memory_type, start, end):
            captured.update(memory_type=memory_type, start=start, end=end)
            return count

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.batch_clear", fake_batch_clear
        )
        result = await ai_memory_service.batch_clear(_stub_db(), 1, confirm=True, **kwargs)
        assert result == count
        assert captured == expected

    async def test_clear_writes_audit_after_commit(self, monkeypatch, record_audit):
        async def fake_batch_clear(db, user_id, memory_type, start, end):
            return 4

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.batch_clear", fake_batch_clear
        )
        set_current_user_id(6)
        try:
            db = _stub_db()
            await ai_memory_service.batch_clear(db, 42, confirm=True)
            assert record_audit == []
            await run_after_commit_callbacks(db)
        finally:
            set_current_user_id(None)

        assert len(record_audit) == 1
        audit = record_audit[0]
        assert audit["operator_id"] == 6
        assert audit["target_type"] == "ai_memory"
        assert audit["target_id"] == 42
        assert audit["action"] == "clear"
        assert audit["after_value"]["count"] == 4


class TestDeleteSyncEs:
    async def test_delete_memory_calls_es_delete(self, monkeypatch):
        deleted_es = []
        memory = make_orm_mem(10, "semantic", "内容")

        async def fake_get(db, memory_id, user_id):
            return memory

        async def fake_soft(db, ids):
            return 1

        async def fake_delete_doc(memory_id):
            deleted_es.append(memory_id)
            return True

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.get_by_id_and_user", fake_get
        )
        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.soft_delete_with_time", fake_soft
        )
        monkeypatch.setattr("app.service.ai_memory_service.delete_memory_doc", fake_delete_doc)

        await ai_memory_service.delete_memory(AsyncMock(spec=AsyncSession), 10, 1)
        assert deleted_es == [10]


class TestListAndExport:
    async def test_list_passes_source(self, monkeypatch):
        captured = {}

        async def fake_list(db, user_id, memory_type, source, page, size):
            captured.update(source=source, memory_type=memory_type)
            return [], 0

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.list_by_user", fake_list
        )
        result = await ai_memory_service.list_memories(
            AsyncMock(spec=AsyncSession), 1, 1, 10, source="feedback"
        )
        assert result.total == 0
        assert captured["source"] == "feedback"

    async def test_export_json_structure(self, monkeypatch):
        memories = [
            make_orm_mem(
                1, "semantic", "偏好", source="conversation", importance=80, access_count=0
            ),
            make_orm_mem(2, "procedural", "习惯", source="manual", importance=60, access_count=0),
        ]

        async def fake_active(db, user_id, limit):
            return memories

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.get_active_by_user", fake_active
        )

        content_type, content = await ai_memory_service.export_memories(
            AsyncMock(spec=AsyncSession), 1, "json"
        )
        assert "json" in content_type
        data = json.loads(content)
        assert data["user_id"] == 1
        assert len(data["memories"]) == 2
        first = data["memories"][0]
        assert {"memory_type", "content", "source", "importance"} <= set(first.keys())

    async def test_export_markdown_structure(self, monkeypatch):
        memories = [make_orm_mem(1, "semantic", "偏好", source="conversation", importance=80)]

        async def fake_active(db, user_id, limit):
            return memories

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.get_active_by_user", fake_active
        )

        content_type, content = await ai_memory_service.export_memories(
            AsyncMock(spec=AsyncSession), 1, "markdown"
        )
        assert "text/markdown" in content_type
        assert "# 长期记忆导出" in content
        assert "semantic" in content
        assert "偏好" in content

    async def test_export_writes_audit(self, monkeypatch, record_audit):
        """记忆属敏感数据，批量导出需留痕。"""
        memories = [make_orm_mem(1, "semantic", "偏好", source="conversation", access_count=0)]

        async def fake_active(db, user_id, limit):
            return memories

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.get_active_by_user", fake_active
        )
        set_current_user_id(8)
        try:
            await ai_memory_service.export_memories(AsyncMock(spec=AsyncSession), 42, "json")
        finally:
            set_current_user_id(None)

        assert len(record_audit) == 1
        audit = record_audit[0]
        assert audit["operator_id"] == 8
        assert audit["target_id"] == 42
        assert audit["action"] == "export"
        assert audit["after_value"] == {"format": "json", "count": 1}


class TestUnarchiveMemory:
    """取消归档：恢复注入并重置衰减计时器。

    遗忘曲线按 last_accessed_at 计算优先级，不重置则该记忆在次日归档任务中按旧时点
    立刻再次归档，用户操作形同无效。
    """

    def _archived_memory(self, **overrides) -> SysAiMemory:
        """构造真实 SysAiMemory 实体（脱离 session 实例化），满足服务入参与序列化契约。"""
        fields: dict[str, Any] = {
            "id": 3,
            "user_id": 1,
            "memory_type": "semantic",
            "content": "偏好",
            "importance": 50,
            "access_count": 0,
            "source": "manual",
            "status": 1,
            "archived": 1,
            "create_time": datetime(2026, 1, 1, 8, 0, 0),
            "update_time": None,
            "last_accessed_at": datetime(2026, 1, 1, 8, 0, 0),
        }
        fields.update(overrides)
        return SysAiMemory(**fields)

    @staticmethod
    def _stub_get(monkeypatch, memory):
        async def fake_get(db, memory_id, user_id):
            return memory

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.get_by_id_and_user", fake_get
        )

    async def test_unarchive_restores_and_resets_decay_timer(self, monkeypatch):
        memory = self._archived_memory()
        self._stub_get(monkeypatch, memory)

        started_at = datetime.now()
        result = await ai_memory_service.unarchive_memory(_stub_db(), 3, 1)

        assert result.archived == 0
        assert memory.last_accessed_at is not None
        assert memory.last_accessed_at >= started_at

    async def test_unarchive_non_archived_rejected(self, monkeypatch):
        memory = self._archived_memory(archived=0)
        self._stub_get(monkeypatch, memory)

        with pytest.raises(BusinessException) as exc:
            await ai_memory_service.unarchive_memory(_stub_db(), 3, 1)
        assert exc.value.code == ResultCode.DATA_STATE_NOT_ALLOW
        assert memory.archived == 0

    async def test_unarchive_foreign_memory_not_found(self, monkeypatch):
        self._stub_get(monkeypatch, None)

        with pytest.raises(BusinessException) as exc:
            await ai_memory_service.unarchive_memory(_stub_db(), 3, 999)
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND
