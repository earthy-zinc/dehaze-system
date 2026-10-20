"""认证中心路由层测试（/me 平铺结构含 avatar、个人改密端点）。

遵循 05-python-test-rules：marker=api、构造注入（dependency_overrides + monkeypatch
repo）、只断言业务结果（code/data）。Redis 全部走 conftest autouse 的 fakeredis。
个人改密契约：旧密码错误 → A0210；新密码复杂度不合规 → A0400；
成功后踢出本人全部在线会话并失效其角色权限缓存。
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis
from app.main import app as fastapi_app
from app.repository.user_repository import user_repository
from app.utils.password import hash_password_async
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api


# conftest mock_redis 的动态扫描会把本模块顶层 get_redis 名字替换为桩函数，
# 依赖覆盖必须以导入期捕获的原始函数对象为键才能命中路由依赖
_ORIGINAL_GET_REDIS = get_redis


def _ctx(user_id=5, username="user"):
    return make_user_context(user_id, username=username, roles=["GUEST"], permissions=[])


@pytest.fixture
async def auth_client(mock_redis):
    async def _override_db():
        return object()

    async def _override_user():
        return _ctx()

    async def _override_redis():
        return mock_redis

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    fastapi_app.dependency_overrides[_ORIGINAL_GET_REDIS] = _override_redis
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)
    fastapi_app.dependency_overrides.pop(_ORIGINAL_GET_REDIS, None)


def _stub_user(password: str):
    return SimpleNamespace(id=5, username="user", password=password)


# ===== /me 平铺结构 =====


async def test_me_returns_avatar(auth_client, monkeypatch):
    monkeypatch.setattr(
        user_repository,
        "get_by_id",
        AsyncMock(return_value=SimpleNamespace(avatar="http://cdn/a.png")),
    )
    resp = await auth_client.get("/api/v1/auth/me")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["userId"] == 5
    assert data["avatar"] == "http://cdn/a.png"
    assert data["roles"] == ["GUEST"]


async def test_me_returns_null_avatar_when_user_missing(auth_client, monkeypatch):
    """用户记录缺失时 avatar 为空（全局 NonNullJSONResponse 剔除 null 字段，缺省即无）"""
    monkeypatch.setattr(user_repository, "get_by_id", AsyncMock(return_value=None))
    resp = await auth_client.get("/api/v1/auth/me")
    assert resp.status_code == 200
    assert resp.json()["data"].get("avatar") is None


# ===== 个人改密 PATCH /api/v1/auth/password =====


async def test_change_password_wrong_old_password_rejected(auth_client, monkeypatch, mock_redis):
    """旧密码 bcrypt 校验失败 → A0210"""
    hashed = await hash_password_async("OldPass123")
    monkeypatch.setattr(user_repository, "get_by_id", AsyncMock(return_value=_stub_user(hashed)))
    monkeypatch.setattr(user_repository, "get_user_role_codes", AsyncMock(return_value=[]))
    resp = await auth_client.patch(
        "/api/v1/auth/password",
        json={"oldPassword": "WrongOld1", "newPassword": "NewPass456"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0210"


@pytest.mark.parametrize(
    "new_password",
    [
        "Aa1",  # 过短（schema 长度校验）
        "abcdefgh",  # 纯字母
        "12345678",  # 纯数字
    ],
)
async def test_change_password_weak_new_password_rejected(
    auth_client, monkeypatch, mock_redis, new_password
):
    """新密码复杂度不合规 → A0400（schema 校验前置，服务层校验兜底）"""
    hashed = await hash_password_async("OldPass123")
    monkeypatch.setattr(user_repository, "get_by_id", AsyncMock(return_value=_stub_user(hashed)))
    monkeypatch.setattr(user_repository, "get_user_role_codes", AsyncMock(return_value=[]))
    resp = await auth_client.patch(
        "/api/v1/auth/password",
        json={"oldPassword": "OldPass123", "newPassword": new_password},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "A0400"


async def test_change_password_success_kicks_all_sessions(auth_client, monkeypatch, mock_redis):
    """成功改密：新密码生效 + 踢出本人全部在线会话 + 失效角色权限缓存"""
    hashed = await hash_password_async("OldPass123")
    monkeypatch.setattr(user_repository, "get_by_id", AsyncMock(return_value=_stub_user(hashed)))
    monkeypatch.setattr(user_repository, "get_user_role_codes", AsyncMock(return_value=["GUEST"]))
    await mock_redis.set(
        "session:s1", json.dumps({"userId": 5, "username": "op", "authorities": []})
    )
    await mock_redis.set(
        "session:s2", json.dumps({"userId": 5, "username": "op", "authorities": []})
    )
    await mock_redis.zadd("session:user:5", {"s1": 1_700_000_000, "s2": 1_700_000_001})
    await mock_redis.set("role:perms:GUEST", '["x"]')

    resp = await auth_client.patch(
        "/api/v1/auth/password",
        json={"oldPassword": "OldPass123", "newPassword": "NewPass456"},
    )
    assert resp.status_code == 200
    assert resp.json()["code"] == "00000"
    assert await mock_redis.get("session:s1") is None
    assert await mock_redis.get("session:s2") is None
    assert await mock_redis.get("session:user:5") is None
    assert await mock_redis.get("role:perms:GUEST") is None
