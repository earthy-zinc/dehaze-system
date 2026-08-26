"""AI 知识库路由测试（管理端：view=admin / index-stats / test-sets / low-quality）。

遵循 05-python-test-rules：marker=api、构造注入（dependency_overrides + monkeypatch service）、
只断言业务结果（code/data），不测装饰器内部细节。
权限口径：管理端接口需 kb:audit（ROOT 放行），普通用户（仅 kb:manage）统一 A0301。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user

pytestmark = pytest.mark.api

from app.main import app as fastapi_app
from app.router import kb
from tests.stubs.factories import make_user_context


def _admin_ctx():
    return make_user_context(
        1, username="admin", roles=["ADMIN"], permissions=["kb:manage", "kb:audit"]
    )


def _user_ctx():
    # 普通用户持有 kb:manage（用户端管理自己的库），但无 kb:audit → 管理端接口 403 A0301
    return make_user_context(5, username="user", roles=[], permissions=["kb:manage"])


@pytest.fixture
async def kb_client():
    async def _override_db():
        return object()

    async def _override_user():
        return _admin_ctx()

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def _as_user(client: AsyncClient):
    """切换为普通用户（用户端权限校验路径）"""
    async def _override_user():
        return _user_ctx()
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    return client


async def _as_admin(client: AsyncClient):
    async def _override_user():
        return _admin_ctx()
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    return client


def test_index_stats_route_registered(app):
    schema = app.openapi()
    paths = schema["paths"]
    assert "/api/v1/kb/{kb_id}/index-stats" in paths
    assert "/api/v1/kb/{kb_id}/retrieve/test-sets" in paths
    assert "/api/v1/kb/{kb_id}/retrieve/test-sets/{test_set_id}/run" in paths
    assert "/api/v1/kb/{kb_id}/chunks/low-quality" in paths


# ===== view=admin 列表 =====


async def test_list_view_admin_returns_full_page(kb_client, monkeypatch):
    async def fake_get_page(db, redis, user_id, keyword, page, size, view=None):
        assert view == "admin"
        return {"list": [{"id": 1, "name": "私有库", "visibility": "private"}], "total": 1}

    monkeypatch.setattr(kb.knowledge_base_service, "get_page", fake_get_page)
    resp = await kb_client.get("/api/v1/kb", params={"view": "admin"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] == 1
    assert data["list"][0]["id"] == 1


async def test_list_view_admin_ordinary_user_forbidden(kb_client):
    await _as_user(kb_client)
    resp = await kb_client.get("/api/v1/kb", params={"view": "admin"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_list_normal_view_does_not_require_admin(kb_client, monkeypatch):
    async def fake_get_page(db, redis, user_id, keyword, page, size, view=None):
        assert view is None
        return {"list": [], "total": 0}

    monkeypatch.setattr(kb.knowledge_base_service, "get_page", fake_get_page)
    await _as_user(kb_client)
    resp = await kb_client.get("/api/v1/kb")
    assert resp.status_code == 200


# ===== index-stats =====


async def test_index_stats_admin_returns_camel_case(kb_client, monkeypatch):
    async def fake_stats(db, kb_id):
        return {"index_size": 100, "index_doc_count": 5, "threshold_warning": False}

    monkeypatch.setattr(kb.knowledge_base_service, "get_index_stats", fake_stats)
    resp = await kb_client.get("/api/v1/kb/1/index-stats")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data == {"indexSize": 100, "indexDocCount": 5, "thresholdWarning": False}


async def test_index_stats_ordinary_user_forbidden(kb_client):
    await _as_user(kb_client)
    resp = await kb_client.get("/api/v1/kb/1/index-stats")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 召回测试集 =====


async def test_create_test_set(kb_client, monkeypatch):
    async def fake_create(db, kb_id, question, expected_chunk_ids):
        assert kb_id == 1
        return {
            "id": 7,
            "knowledgeBaseId": 1,
            "question": question,
            "expectedChunkIds": expected_chunk_ids,
        }

    monkeypatch.setattr(kb.test_set_service, "create_test_set", fake_create)
    resp = await kb_client.post(
        "/api/v1/kb/1/retrieve/test-sets",
        json={"question": "RIDCP 是什么算法？", "expectedChunkIds": [1, 2]},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["id"] == 7
    assert data["expectedChunkIds"] == [1, 2]


async def test_list_test_sets_returns_array(kb_client, monkeypatch):
    async def fake_list(db, kb_id, page, size):
        return {"list": [{"id": 7, "question": "q", "expected_chunk_ids": [1]}], "total": 1}

    monkeypatch.setattr(kb.test_set_service, "list_test_sets", fake_list)
    resp = await kb_client.get("/api/v1/kb/1/retrieve/test-sets")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] == 1
    assert isinstance(data["list"], list)
    assert data["list"][0]["id"] == 7


async def test_run_test_set_defaults_top_k(kb_client, monkeypatch):
    async def fake_run(db, redis, user_id, kb_id, test_set_id, top_k):
        assert top_k == 5
        return {
            "testSetId": test_set_id,
            "recallAtK": 1.0,
            "hitRate": 1.0,
            "totalCases": 1,
            "hitCases": 1,
        }

    monkeypatch.setattr(kb.test_set_service, "run_test_set", fake_run)
    resp = await kb_client.post("/api/v1/kb/1/retrieve/test-sets/7/run")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["recallAtK"] == 1.0


async def test_test_sets_ordinary_user_forbidden(kb_client):
    await _as_user(kb_client)
    resp = await kb_client.get("/api/v1/kb/1/retrieve/test-sets")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_create_test_set_ordinary_user_forbidden(kb_client):
    await _as_user(kb_client)
    resp = await kb_client.post(
        "/api/v1/kb/1/retrieve/test-sets",
        json={"question": "RIDCP 是什么算法？", "expectedChunkIds": [1, 2]},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_run_test_set_ordinary_user_forbidden(kb_client):
    await _as_user(kb_client)
    resp = await kb_client.post("/api/v1/kb/1/retrieve/test-sets/7/run")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 用户端 kb:manage 权限（普通用户无该权限 → 403）=====


def _user_no_kb_perms_ctx():
    # 仅持 kb:audit（管理端审计），无用户端管理权限 kb:manage → 创建库 403
    return make_user_context(6, username="auditor", roles=[], permissions=["kb:audit"])


async def test_create_knowledge_base_without_manage_forbidden(kb_client):
    async def _override_user():
        return _user_no_kb_perms_ctx()

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    resp = await kb_client.post(
        "/api/v1/kb",
        json={
            "name": "无权限库",
            "visibility": "private",
            "embeddingModel": "bge-m3",
            "chunkingStrategy": "fixed",
        },
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def test_update_knowledge_base_without_manage_forbidden(kb_client):
    async def _override_user():
        return _user_no_kb_perms_ctx()

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    resp = await kb_client.put(
        "/api/v1/kb/1",
        json={"name": "改名"},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def test_create_knowledge_base_manage_allowed(kb_client, monkeypatch):
    async def fake_create(db, redis, body, user):
        return 9

    monkeypatch.setattr(kb.knowledge_base_service, "create", fake_create)
    await _as_user(kb_client)  # 持有 kb:manage
    resp = await kb_client.post(
        "/api/v1/kb",
        json={
            "name": "用户自建库",
            "visibility": "private",
            "embeddingModel": "bge-m3",
            "chunkingStrategy": "fixed",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == 9


# ===== 低质量片段 =====


async def test_list_low_quality(kb_client, monkeypatch):
    async def fake_list(db, kb_id, page, size):
        return {
            "list": [{"chunkId": 10, "content": "c", "documentId": 3, "thumbsDownCount": 2}],
            "total": 1,
        }

    monkeypatch.setattr(kb.low_quality_service, "list_low_quality_chunks", fake_list)
    resp = await kb_client.get("/api/v1/kb/1/chunks/low-quality")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["list"][0]["chunkId"] == 10
    assert data["total"] == 1


async def test_low_quality_ordinary_user_forbidden(kb_client):
    await _as_user(kb_client)
    resp = await kb_client.get("/api/v1/kb/1/chunks/low-quality")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"
