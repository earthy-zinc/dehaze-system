import json
from datetime import datetime

import pytest

from app.core.exceptions import BusinessException
from app.service.ai_memory_service import AiMemoryService
from tests.stubs import make_orm_mem


class TestBatchClear:
    async def test_requires_confirm(self, monkeypatch):
        async def fake_batch_clear(db, user_id, memory_type, start, end):
            return 5

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.batch_clear", fake_batch_clear
        )
        with pytest.raises(BusinessException):
            await AiMemoryService.batch_clear(object(), 1, confirm=False)

    @pytest.mark.parametrize(
        "kwargs,count,expected",
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
        result = await AiMemoryService.batch_clear(object(), 1, confirm=True, **kwargs)
        assert result == count
        assert captured == expected


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

        await AiMemoryService.delete_memory(object(), 10, 1)
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
        result = await AiMemoryService.list_memories(object(), 1, 1, 10, source="feedback")
        assert result.total == 0
        assert captured["source"] == "feedback"

    async def test_export_json_structure(self, monkeypatch):
        memories = [
            make_orm_mem(1, "semantic", "偏好", source="conversation", importance=80, access_count=0),
            make_orm_mem(2, "procedural", "习惯", source="manual", importance=60, access_count=0),
        ]

        async def fake_active(db, user_id, limit):
            return memories

        monkeypatch.setattr(
            "app.service.ai_memory_service.ai_memory_repository.get_active_by_user", fake_active
        )

        content_type, content = await AiMemoryService.export_memories(object(), 1, "json")
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

        content_type, content = await AiMemoryService.export_memories(object(), 1, "markdown")
        assert "text/markdown" in content_type
        assert "# 长期记忆导出" in content
        assert "semantic" in content
        assert "偏好" in content
