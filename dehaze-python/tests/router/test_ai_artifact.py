"""AI 中间产物路由测试：会话产物列表 / 消息关联产物 / by-ref 反查 / 详情

覆盖重点：路由注册、查询参数校验（A0400）、归属校验错误码（A0401）、camelCase 序列化。
"""
import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.api

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_artifact import ArtifactResult
from app.models.schema.common import PageResult
from app.service.ai_artifact_service import ai_artifact_service


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=()):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


def _artifact(**overrides) -> ArtifactResult:
    base = {
        "id": 10,
        "conversation_id": 3,
        "message_id": 55,
        "type": "image",
        "ref_type": "sys_file",
        "ref_id": 77,
        "is_invalid": 0,
    }
    base.update(overrides)
    return ArtifactResult.model_validate(base)


@pytest.fixture
async def artifact_client():
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


def test_artifact_paths_registered(app):
    schema = app.openapi()
    for path in (
        "/api/v1/ai/conversations/{conv_id}/artifacts",
        "/api/v1/ai/messages/{msg_id}/artifacts",
        "/api/v1/ai/artifacts/by-ref",
        "/api/v1/ai/artifacts/{artifact_id}/detail",
    ):
        assert path in schema["paths"], f"缺少路径 {path}"


class TestListByConversation:
    async def test_list_forwards_paging(self, artifact_client, monkeypatch):
        client, _ = artifact_client
        captured: dict = {}

        async def _fake_list(db, conv_id, user_id, page, size):
            captured.update(conv_id=conv_id, user_id=user_id, page=page, size=size)
            return PageResult(list=[_artifact()], total=1)

        monkeypatch.setattr(ai_artifact_service, "list_by_conversation", _fake_list)
        resp = await client.get(
            "/api/v1/ai/conversations/3/artifacts", params={"pageNum": 2, "pageSize": 5}
        )
        assert resp.status_code == 200
        assert captured == {"conv_id": 3, "user_id": 8, "page": 2, "size": 5}
        item = resp.json()["data"]["list"][0]
        assert item["conversationId"] == 3
        assert item["messageId"] == 55
        assert item["refType"] == "sys_file"
        assert item["refId"] == 77
        assert item["isInvalid"] == 0

    async def test_list_conversation_not_found_maps_a0401(self, artifact_client, monkeypatch):
        client, _ = artifact_client

        async def _fake_list(db, conv_id, user_id, page, size):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")

        monkeypatch.setattr(ai_artifact_service, "list_by_conversation", _fake_list)
        resp = await client.get("/api/v1/ai/conversations/404/artifacts")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"

    async def test_list_invalid_page_size_rejected(self, artifact_client):
        client, _ = artifact_client
        resp = await client.get(
            "/api/v1/ai/conversations/3/artifacts", params={"pageSize": 101}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestListByMessage:
    async def test_list_by_message(self, artifact_client, monkeypatch):
        client, _ = artifact_client
        captured: dict = {}

        async def _fake_list(db, msg_id, user_id):
            captured.update(msg_id=msg_id, user_id=user_id)
            return [_artifact(id=10), _artifact(id=11, type="file")]

        monkeypatch.setattr(ai_artifact_service, "list_by_message", _fake_list)
        resp = await client.get("/api/v1/ai/messages/55/artifacts")
        assert resp.status_code == 200
        assert captured == {"msg_id": 55, "user_id": 8}
        assert [item["id"] for item in resp.json()["data"]] == [10, 11]

    async def test_list_by_message_not_found_maps_a0401(self, artifact_client, monkeypatch):
        client, _ = artifact_client

        async def _fake_list(db, msg_id, user_id):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")

        monkeypatch.setattr(ai_artifact_service, "list_by_message", _fake_list)
        resp = await client.get("/api/v1/ai/messages/55/artifacts")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"


class TestListByRef:
    async def test_list_by_ref_forwards_params(self, artifact_client, monkeypatch):
        client, _ = artifact_client
        captured: dict = {}

        async def _fake_list(db, ref_type, ref_id, user_id):
            captured.update(ref_type=ref_type, ref_id=ref_id, user_id=user_id)
            return [_artifact()]

        monkeypatch.setattr(ai_artifact_service, "list_by_ref", _fake_list)
        resp = await client.get(
            "/api/v1/ai/artifacts/by-ref", params={"refType": "sys_pred_log", "refId": 12}
        )
        assert resp.status_code == 200
        assert captured == {"ref_type": "sys_pred_log", "ref_id": 12, "user_id": 8}
        assert resp.json()["data"][0]["refType"] == "sys_file"

    async def test_list_by_ref_requires_ref_type(self, artifact_client):
        client, _ = artifact_client
        resp = await client.get("/api/v1/ai/artifacts/by-ref", params={"refId": 12})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_list_by_ref_invalid_ref_id_rejected(self, artifact_client):
        client, _ = artifact_client
        resp = await client.get(
            "/api/v1/ai/artifacts/by-ref", params={"refType": "sys_file", "refId": "abc"}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"


class TestDetail:
    async def test_detail_returns_runtime_image_url(self, artifact_client, monkeypatch):
        client, _ = artifact_client
        captured: dict = {}

        async def _fake_detail(db, artifact_id, user_id):
            captured.update(artifact_id=artifact_id, user_id=user_id)
            return {
                "artifact": _artifact(id=artifact_id),
                "imageUrl": "http://static/obj.png",
            }

        monkeypatch.setattr(ai_artifact_service, "get_detail", _fake_detail)
        resp = await client.get("/api/v1/ai/artifacts/10/detail")
        assert resp.status_code == 200
        assert captured == {"artifact_id": 10, "user_id": 8}
        data = resp.json()["data"]
        assert data["imageUrl"] == "http://static/obj.png"
        assert data["artifact"]["isInvalid"] == 0

    async def test_detail_invalid_artifact_maps_a0401(self, artifact_client, monkeypatch):
        client, _ = artifact_client

        async def _fake_detail(db, artifact_id, user_id):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "产物不存在或已失效")

        monkeypatch.setattr(ai_artifact_service, "get_detail", _fake_detail)
        resp = await client.get("/api/v1/ai/artifacts/10/detail")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"

    async def test_detail_cross_user_conversation_maps_a0401(self, artifact_client, monkeypatch):
        client, _ = artifact_client

        async def _fake_detail(db, artifact_id, user_id):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "产物所属会话不存在")

        monkeypatch.setattr(ai_artifact_service, "get_detail", _fake_detail)
        resp = await client.get("/api/v1/ai/artifacts/10/detail")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"
