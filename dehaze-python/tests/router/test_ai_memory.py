"""AI 长期记忆路由测试：列表 / 归档 / CRUD / 搜索 / 批量清空 / 恢复 / 导出

覆盖重点：路由注册、参数校验（A0400）、confirm 二次确认（A0400）、camelCase 序列化、导出流式。
"""
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.api

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_memory import MemoryResult
from app.models.schema.common import PageResult
from app.repository.ai_memory_repository import ai_memory_repository
from app.service.ai_memory_service import ai_memory_service


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=()):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


def _memory(**overrides) -> MemoryResult:
    base = {
        "id": 20,
        "user_id": 8,
        "memory_type": "semantic",
        "content": "用户偏好简洁回复",
        "importance": 80,
        "access_count": 3,
        "source": "manual",
        "status": 1,
        "archived": 0,
        "create_time": datetime(2026, 8, 29, 10, 0, 0),
    }
    base.update(overrides)
    return MemoryResult.model_validate(base)


@pytest.fixture
async def memory_client():
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser(id=8)}

    async def _override_user():
        return current_user["user"]

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client, current_user
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


def test_memory_paths_registered(app):
    schema = app.openapi()
    for path in (
        "/api/v1/ai/memories",
        "/api/v1/ai/memories/archived",
        "/api/v1/ai/memories/search",
        "/api/v1/ai/memories/clear",
        "/api/v1/ai/memories/restore",
        "/api/v1/ai/memories/export",
        "/api/v1/ai/memories/{memory_id}",
    ):
        assert path in schema["paths"], f"缺少路径 {path}"


class TestList:
    async def test_list_forwards_filters(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}

        async def _fake_list(db, user_id, page, size, memory_type=None, source=None):
            captured.update(
                user_id=user_id, page=page, size=size, memory_type=memory_type, source=source
            )
            return PageResult(list=[_memory()], total=1)

        monkeypatch.setattr(ai_memory_service, "list_memories", _fake_list)
        resp = await client.get(
            "/api/v1/ai/memories",
            params={"memoryType": "semantic", "source": "feedback", "pageNum": 2, "pageSize": 5},
        )
        assert resp.status_code == 200
        assert captured == {
            "user_id": 8,
            "page": 2,
            "size": 5,
            "memory_type": "semantic",
            "source": "feedback",
        }
        item = resp.json()["data"]["list"][0]
        assert item["memoryType"] == "semantic"
        assert item["accessCount"] == 3
        assert item["createTime"].startswith("2026-08-29")

    async def test_archived_list_forwards_filters(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}

        async def _fake_archived(db, user_id, page, size, memory_type=None):
            captured.update(user_id=user_id, page=page, size=size, memory_type=memory_type)
            return PageResult(list=[_memory(archived=1)], total=1)

        monkeypatch.setattr(ai_memory_service, "list_archived", _fake_archived)
        resp = await client.get(
            "/api/v1/ai/memories/archived", params={"memoryType": "episodic", "pageNum": 1}
        )
        assert resp.status_code == 200
        assert captured == {
            "user_id": 8,
            "page": 1,
            "size": 10,
            "memory_type": "episodic",
        }
        assert resp.json()["data"]["list"][0]["archived"] == 1

    async def test_list_invalid_page_num_rejected(self, memory_client):
        client, _ = memory_client
        resp = await client.get("/api/v1/ai/memories", params={"pageNum": 0})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestCreateUpdateDelete:
    async def test_create_forwards_form(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}

        async def _fake_create(db, user_id, form):
            captured.update(user_id=user_id, memory_type=form.memoryType, content=form.content)
            return _memory(id=21, memory_type=form.memoryType, content=form.content)

        monkeypatch.setattr(ai_memory_service, "create_memory", _fake_create)
        resp = await client.post(
            "/api/v1/ai/memories",
            json={"memoryType": "procedural", "content": "先做摘要再回答", "importance": 70},
        )
        assert resp.status_code == 200
        assert captured == {
            "user_id": 8,
            "memory_type": "procedural",
            "content": "先做摘要再回答",
        }
        assert resp.json()["data"]["id"] == 21

    async def test_create_requires_memory_type(self, memory_client):
        client, _ = memory_client
        resp = await client.post("/api/v1/ai/memories", json={"content": "x"})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_create_content_too_long_rejected(self, memory_client):
        client, _ = memory_client
        resp = await client.post(
            "/api/v1/ai/memories", json={"memoryType": "semantic", "content": "x" * 2001}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_create_importance_out_of_range_rejected(self, memory_client):
        client, _ = memory_client
        resp = await client.post(
            "/api/v1/ai/memories",
            json={"memoryType": "semantic", "content": "x", "importance": 101},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_update_forwards_form(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}

        async def _fake_update(db, memory_id, user_id, form):
            captured.update(memory_id=memory_id, user_id=user_id, content=form.content)
            return _memory(id=memory_id, content=form.content)

        monkeypatch.setattr(ai_memory_service, "update_memory", _fake_update)
        resp = await client.put("/api/v1/ai/memories/20", json={"content": "改后内容"})
        assert resp.status_code == 200
        assert captured == {"memory_id": 20, "user_id": 8, "content": "改后内容"}

    async def test_update_status_out_of_range_rejected(self, memory_client):
        client, _ = memory_client
        resp = await client.put("/api/v1/ai/memories/20", json={"status": 2})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_update_not_found_maps_a0401(self, memory_client, monkeypatch):
        client, _ = memory_client

        async def _fake_update(db, memory_id, user_id, form):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "记忆不存在")

        monkeypatch.setattr(ai_memory_service, "update_memory", _fake_update)
        resp = await client.put("/api/v1/ai/memories/20", json={"content": "x"})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"

    async def test_delete(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}

        async def _fake_delete(db, memory_id, user_id):
            captured.update(memory_id=memory_id, user_id=user_id)

        monkeypatch.setattr(ai_memory_service, "delete_memory", _fake_delete)
        resp = await client.delete("/api/v1/ai/memories/20")
        assert resp.status_code == 200
        assert captured == {"memory_id": 20, "user_id": 8}


class TestSearch:
    async def test_search_forwards_keyword_and_limit(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}

        async def _fake_search(db, user_id, keyword, limit=5):
            captured.update(user_id=user_id, keyword=keyword, limit=limit)
            return [_memory()]

        monkeypatch.setattr(ai_memory_service, "search_memories", _fake_search)
        resp = await client.get(
            "/api/v1/ai/memories/search", params={"keyword": "简洁", "limit": 3}
        )
        assert resp.status_code == 200
        assert captured == {"user_id": 8, "keyword": "简洁", "limit": 3}
        assert resp.json()["data"][0]["content"] == "用户偏好简洁回复"

    async def test_search_requires_keyword(self, memory_client):
        client, _ = memory_client
        resp = await client.get("/api/v1/ai/memories/search")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestClearAndRestore:
    """confirm 二次确认走真实 service 校验（仓储 mock）。"""

    async def test_clear_requires_confirm(self, memory_client, monkeypatch):
        client, _ = memory_client

        async def _fake_repo_clear(db, user_id, memory_type, start, end):
            raise AssertionError("未二次确认不应执行清空")

        monkeypatch.setattr(ai_memory_repository, "batch_clear", _fake_repo_clear)
        resp = await client.post("/api/v1/ai/memories/clear")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_clear_with_confirm_returns_count(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}

        async def _fake_repo_clear(db, user_id, memory_type, start, end):
            captured.update(user_id=user_id, memory_type=memory_type, start=start, end=end)
            return 4

        monkeypatch.setattr(ai_memory_repository, "batch_clear", _fake_repo_clear)
        resp = await client.post(
            "/api/v1/ai/memories/clear",
            params={"confirm": "true", "memoryType": "semantic"},
        )
        assert resp.status_code == 200
        assert captured["user_id"] == 8
        assert captured["memory_type"] == "semantic"
        assert resp.json()["data"] == 4
        assert "4" in resp.json()["msg"]

    async def test_restore_requires_confirm(self, memory_client, monkeypatch):
        client, _ = memory_client

        async def _fake_list_deleted(db, user_id, memory_type, start, end):
            raise AssertionError("未二次确认不应查询可恢复记忆")

        monkeypatch.setattr(
            ai_memory_repository, "list_deleted_for_restore", _fake_list_deleted
        )
        resp = await client.post("/api/v1/ai/memories/restore")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_restore_with_confirm_returns_count(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}
        now = datetime(2026, 8, 29, 10, 0, 0)

        async def _fake_list_deleted(db, user_id, memory_type, start, end):
            captured.update(start=start, end=end)
            return [_memory(id=1), _memory(id=2)]

        async def _fake_restore(db, ids):
            captured["ids"] = ids
            return len(ids)

        monkeypatch.setattr(
            ai_memory_repository, "list_deleted_for_restore", _fake_list_deleted
        )
        monkeypatch.setattr(ai_memory_repository, "restore_deleted", _fake_restore)
        resp = await client.post(
            "/api/v1/ai/memories/restore",
            params={
                "confirm": "true",
                "start": "2026-08-01T00:00:00",
                "end": "2026-08-29T00:00:00",
            },
        )
        assert resp.status_code == 200
        assert captured["ids"] == [1, 2]
        assert captured["start"] == datetime(2026, 8, 1)
        assert resp.json()["data"] == 2


class TestExport:
    async def test_export_json_stream(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}

        async def _fake_export(db, user_id, fmt):
            captured.update(user_id=user_id, fmt=fmt)
            return "application/json; charset=utf-8", '{"memories": []}'

        monkeypatch.setattr(ai_memory_service, "export_memories", _fake_export)
        resp = await client.get("/api/v1/ai/memories/export")
        assert resp.status_code == 200
        assert captured == {"user_id": 8, "fmt": "json"}
        assert resp.headers["content-type"].startswith("application/json")
        assert 'filename="memories.json"' in resp.headers["content-disposition"]
        assert resp.json()["memories"] == []

    async def test_export_markdown_stream(self, memory_client, monkeypatch):
        client, _ = memory_client
        captured: dict = {}

        async def _fake_export(db, user_id, fmt):
            captured["fmt"] = fmt
            return "text/markdown; charset=utf-8", "# 长期记忆导出\n"

        monkeypatch.setattr(ai_memory_service, "export_memories", _fake_export)
        resp = await client.get("/api/v1/ai/memories/export", params={"fmt": "markdown"})
        assert resp.status_code == 200
        assert captured["fmt"] == "markdown"
        assert resp.headers["content-type"].startswith("text/markdown")
        assert 'filename="memories.md"' in resp.headers["content-disposition"]
        assert "# 长期记忆导出" in resp.text
