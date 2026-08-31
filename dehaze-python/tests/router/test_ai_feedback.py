"""AI 消息反馈路由测试：提交 / 查询 / 撤销

提交走真实 service 校验（消息仓储与反馈仓储 mock），覆盖：
- 点赞/点踩标签白名单校验（A0400）
- 仅助手消息可反馈、30 天反馈时效（A0502）
- 消息不存在（A0401）、撤销反馈不存在（A0543）
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.api

import app.service.ai_feedback_service as feedback_module
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.repository.ai_message_feedback_repository import ai_message_feedback_repository
from app.repository.ai_message_repository import ai_message_repository


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=()):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


def _feedback(**overrides):
    base = {
        "id": 30,
        "message_id": 55,
        "user_id": 8,
        "rating": 1,
        "tags": ["accurate"],
        "create_time": datetime(2026, 8, 29, 10, 0, 0),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _message(**overrides):
    base = {
        "id": 55,
        "conversation_id": 3,
        "role": "assistant",
        "model": "qwen3-0.6b",
        "create_time": datetime.now(),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
async def feedback_client(monkeypatch):
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser(id=8)}

    async def _override_user():
        return current_user["user"]

    # 点踩反馈的偏好记忆沉淀为独立后台任务（真实写库），测试内禁用
    monkeypatch.setattr(
        feedback_module, "_spawn_feedback_memory_extraction", lambda *args, **kwargs: None
    )
    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client, current_user
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


def test_feedback_paths_registered(app):
    schema = app.openapi()
    path = "/api/v1/ai/messages/{message_id}/feedback"
    assert path in schema["paths"], f"缺少路径 {path}"
    for method in ("post", "get", "delete"):
        assert method in schema["paths"][path], f"缺少方法 {method}"


class TestSubmit:
    async def test_submit_like(self, feedback_client, monkeypatch):
        client, _ = feedback_client
        captured: dict = {}

        async def _fake_msg(db, message_id, user_id):
            return _message()

        async def _fake_upsert(
            db, message_id, user_id, rating, tags, comment, conversation_id=None,
            model=None, source=None
        ):
            captured.update(
                message_id=message_id,
                user_id=user_id,
                rating=rating,
                tags=tags,
                comment=comment,
                conversation_id=conversation_id,
                source=source,
            )
            return _feedback(rating=rating, tags=tags, comment=comment)

        monkeypatch.setattr(ai_message_repository, "get_by_id_and_user", _fake_msg)
        monkeypatch.setattr(
            ai_message_feedback_repository, "upsert_feedback", _fake_upsert
        )
        resp = await client.post(
            "/api/v1/ai/messages/55/feedback",
            json={"rating": 1, "tags": ["accurate", "concise"], "comment": "很有帮助"},
        )
        assert resp.status_code == 200
        assert captured == {
            "message_id": 55,
            "user_id": 8,
            "rating": 1,
            "tags": ["accurate", "concise"],
            "comment": "很有帮助",
            "conversation_id": 3,
            "source": "internal",
        }
        data = resp.json()["data"]
        assert data["messageId"] == 55
        assert data["userId"] == 8
        assert data["rating"] == 1

    async def test_submit_like_with_unknown_tag_rejected(self, feedback_client, monkeypatch):
        client, _ = feedback_client

        async def _fake_msg(db, message_id, user_id):
            return _message()

        monkeypatch.setattr(ai_message_repository, "get_by_id_and_user", _fake_msg)
        resp = await client.post(
            "/api/v1/ai/messages/55/feedback", json={"rating": 1, "tags": ["wrong_tag"]}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_submit_dislike_requires_tags(self, feedback_client, monkeypatch):
        client, _ = feedback_client

        async def _fake_msg(db, message_id, user_id):
            return _message()

        monkeypatch.setattr(ai_message_repository, "get_by_id_and_user", _fake_msg)
        resp = await client.post("/api/v1/ai/messages/55/feedback", json={"rating": -1})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_submit_dislike_with_unknown_tag_rejected(self, feedback_client, monkeypatch):
        client, _ = feedback_client

        async def _fake_msg(db, message_id, user_id):
            return _message()

        monkeypatch.setattr(ai_message_repository, "get_by_id_and_user", _fake_msg)
        resp = await client.post(
            "/api/v1/ai/messages/55/feedback", json={"rating": -1, "tags": ["accurate"]}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_submit_dislike_with_valid_tag(self, feedback_client, monkeypatch):
        client, _ = feedback_client
        captured: dict = {}

        async def _fake_msg(db, message_id, user_id):
            return _message()

        async def _fake_upsert(
            db, message_id, user_id, rating, tags, comment, conversation_id=None,
            model=None, source=None
        ):
            captured.update(rating=rating, tags=tags)
            return _feedback(rating=rating, tags=tags)

        monkeypatch.setattr(ai_message_repository, "get_by_id_and_user", _fake_msg)
        monkeypatch.setattr(
            ai_message_feedback_repository, "upsert_feedback", _fake_upsert
        )
        resp = await client.post(
            "/api/v1/ai/messages/55/feedback",
            json={"rating": -1, "tags": ["incomplete", "too_long"], "comment": "内容不完整"},
        )
        assert resp.status_code == 200
        assert captured == {"rating": -1, "tags": ["incomplete", "too_long"]}

    async def test_submit_on_user_message_maps_a0502(self, feedback_client, monkeypatch):
        client, _ = feedback_client

        async def _fake_msg(db, message_id, user_id):
            return _message(role="user")

        monkeypatch.setattr(ai_message_repository, "get_by_id_and_user", _fake_msg)
        resp = await client.post("/api/v1/ai/messages/54/feedback", json={"rating": 1})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0502"

    async def test_submit_expired_message_maps_a0502(self, feedback_client, monkeypatch):
        client, _ = feedback_client

        async def _fake_msg(db, message_id, user_id):
            return _message(create_time=datetime.now() - timedelta(days=31))

        monkeypatch.setattr(ai_message_repository, "get_by_id_and_user", _fake_msg)
        resp = await client.post("/api/v1/ai/messages/55/feedback", json={"rating": 1})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0502"

    async def test_submit_missing_message_maps_a0401(self, feedback_client, monkeypatch):
        client, _ = feedback_client

        async def _fake_msg(db, message_id, user_id):
            return None

        monkeypatch.setattr(ai_message_repository, "get_by_id_and_user", _fake_msg)
        resp = await client.post("/api/v1/ai/messages/999/feedback", json={"rating": 1})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"

    async def test_submit_rating_out_of_range_rejected(self, feedback_client):
        client, _ = feedback_client
        resp = await client.post("/api/v1/ai/messages/55/feedback", json={"rating": 2})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_submit_comment_too_long_rejected(self, feedback_client):
        client, _ = feedback_client
        resp = await client.post(
            "/api/v1/ai/messages/55/feedback", json={"rating": 1, "comment": "x" * 2001}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestQueryAndRevoke:
    async def test_get_feedback(self, feedback_client, monkeypatch):
        client, _ = feedback_client

        async def _fake_get(db, message_id, user_id):
            return _feedback()

        monkeypatch.setattr(
            ai_message_feedback_repository, "get_by_user_and_message", _fake_get
        )
        resp = await client.get("/api/v1/ai/messages/55/feedback")
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == 30
        assert resp.json()["data"]["rating"] == 1

    async def test_get_feedback_returns_null_when_absent(self, feedback_client, monkeypatch):
        client, _ = feedback_client

        async def _fake_get(db, message_id, user_id):
            return None

        monkeypatch.setattr(
            ai_message_feedback_repository, "get_by_user_and_message", _fake_get
        )
        resp = await client.get("/api/v1/ai/messages/55/feedback")
        assert resp.status_code == 200
        # 无反馈时 Result 序列化排除 null（exclude_none），data 字段缺省
        assert resp.json().get("data") is None

    async def test_revoke_feedback(self, feedback_client, monkeypatch):
        client, _ = feedback_client
        captured: dict = {}

        async def _fake_get(db, message_id, user_id):
            return _feedback()

        async def _fake_soft_delete(db, message_id, user_id):
            captured.update(message_id=message_id, user_id=user_id)

        monkeypatch.setattr(
            ai_message_feedback_repository, "get_by_user_and_message", _fake_get
        )
        monkeypatch.setattr(
            ai_message_feedback_repository, "soft_delete", _fake_soft_delete
        )
        resp = await client.delete("/api/v1/ai/messages/55/feedback")
        assert resp.status_code == 200
        assert captured == {"message_id": 55, "user_id": 8}

    async def test_revoke_missing_feedback_maps_a0543(self, feedback_client, monkeypatch):
        client, _ = feedback_client

        async def _fake_get(db, message_id, user_id):
            return None

        monkeypatch.setattr(
            ai_message_feedback_repository, "get_by_user_and_message", _fake_get
        )
        resp = await client.delete("/api/v1/ai/messages/55/feedback")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0543"
