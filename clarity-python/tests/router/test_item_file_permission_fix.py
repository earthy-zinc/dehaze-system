"""图片文件路由权限回归测试（对齐 API接口.md §2.3 权限标识）。

历史实现仅要求登录态：任意登录用户可对任意数据项挂图/改图/删图。
修复后：上传/修改需 sys:dataset:edit，删除/批量删除需 sys:dataset:delete
（admin/root 通配放行，普通用户 403 A0301）。
"""

import io

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.router import item_file
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api


def _ctx(perms: list[str], user_id: int = 2):
    return make_user_context(user_id, username="editor", roles=["ADMIN"], permissions=perms)


def _plain_ctx():
    """普通用户：登录但无任何 sys:dataset:* 权限"""
    return make_user_context(5, username="user", roles=[], permissions=[])


@pytest.fixture
async def item_file_client():
    async def _override_db():
        return object()

    async def _override_user():
        return _ctx(["sys:dataset:edit"])

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def _as(client: AsyncClient, ctx):
    async def _override_user():
        return ctx

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    return client


def _png_part(content: bytes = b"png"):
    return {"file": ("test.png", io.BytesIO(content), "image/png")}


# ===== 无权限用户：全部写接口 403 A0301 =====


async def test_upload_item_file_without_permission_forbidden(item_file_client):
    await _as(item_file_client, _plain_ctx())
    resp = await item_file_client.post(
        "/api/v1/item-files",
        data={"itemId": "1", "type": "hazy"},
        files=_png_part(),
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_update_item_file_without_permission_forbidden(item_file_client):
    await _as(item_file_client, _plain_ctx())
    resp = await item_file_client.put("/api/v1/item-files/1", json={"description": "x"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_delete_item_file_without_permission_forbidden(item_file_client):
    await _as(item_file_client, _plain_ctx())
    resp = await item_file_client.delete("/api/v1/item-files/1")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_batch_delete_item_files_without_permission_forbidden(item_file_client):
    await _as(item_file_client, _plain_ctx())
    resp = await item_file_client.request("DELETE", "/api/v1/item-files/batch", json={"ids": [1]})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 仅持 edit 权限：删除接口 403（对齐文档 sys:dataset:delete 口径）=====


async def test_delete_item_file_requires_delete_permission(item_file_client):
    await _as(item_file_client, _ctx(["sys:dataset:edit"]))
    resp = await item_file_client.delete("/api/v1/item-files/1")
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


async def test_batch_delete_item_files_requires_delete_permission(item_file_client):
    await _as(item_file_client, _ctx(["sys:dataset:edit"]))
    resp = await item_file_client.request("DELETE", "/api/v1/item-files/batch", json={"ids": [1]})
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


# ===== 持权用户可正常进入业务逻辑（权限放行后调用 service）=====


async def test_upload_allowed_with_dataset_edit_permission(item_file_client, monkeypatch):
    called = {}

    async def _fake_upload(**kwargs):
        called["item_id"] = kwargs["item_id"]
        return {"id": 1, "itemId": 7, "type": "hazy"}

    monkeypatch.setattr(item_file.item_file_service, "upload_item_file", _fake_upload)
    resp = await item_file_client.post(
        "/api/v1/item-files",
        data={"itemId": "7", "type": "hazy"},
        files=_png_part(),
    )
    assert resp.status_code == 200
    assert called["item_id"] == 7


async def test_update_allowed_with_dataset_edit_permission(item_file_client, monkeypatch):
    called = {}

    async def _fake_update(db, redis, file_id, data):
        called["file_id"] = file_id

    monkeypatch.setattr(item_file.item_file_service, "update_item_file", _fake_update)
    resp = await item_file_client.put("/api/v1/item-files/9", json={"description": "x"})
    assert resp.status_code == 200
    assert called["file_id"] == 9


async def test_delete_allowed_with_dataset_delete_permission(item_file_client, monkeypatch):
    await _as(item_file_client, _ctx(["sys:dataset:delete"]))
    called = {}

    async def _fake_delete(db, redis, file_id):
        called["file_id"] = file_id

    monkeypatch.setattr(item_file.item_file_service, "delete_item_file", _fake_delete)
    resp = await item_file_client.delete("/api/v1/item-files/9")
    assert resp.status_code == 200
    assert called["file_id"] == 9


async def test_batch_delete_allowed_with_dataset_delete_permission(item_file_client, monkeypatch):
    await _as(item_file_client, _ctx(["sys:dataset:delete"]))
    called = {}

    async def _fake_batch(db, redis, ids):
        called["ids"] = ids
        return {"successCount": 1, "failedCount": 0, "message": "ok"}

    monkeypatch.setattr(item_file.item_file_service, "batch_delete_item_files", _fake_batch)
    resp = await item_file_client.request("DELETE", "/api/v1/item-files/batch", json={"ids": [9]})
    assert resp.status_code == 200
    assert called["ids"] == [9]
