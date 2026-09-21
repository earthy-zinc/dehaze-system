"""Session 依赖解析与会话续期测试（对照测试用例.md §4.9，T-AM-080 ~ T-AM-083）。

Redis 走 conftest autouse 的 fakeredis（get_current_user 经 Depends(get_redis_client)
取到同一实例），会话数据由测试预置。
"""

import json

import pytest
from starlette.requests import Request

from app.core.code import ResultCode
from app.dependencies import auth as auth_module
from app.dependencies.auth import SESSION_PREFIX, SESSION_TTL, get_current_user


def _request_with_header(session_id: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-session-id", session_id.encode())],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def _session_payload(user_id: int = 5, username: str = "user") -> str:
    return json.dumps(
        {
            "userId": user_id,
            "username": username,
            "nickname": username,
            "deptId": None,
            "dataScope": 2,
            "authorities": ["ROLE_GUEST", "sys:auth:log:list"],
        }
    )


async def test_renews_session_ttl_below_threshold(mock_redis):
    """T-AM-080：TTL < 1 天的会话在请求后自动续期到 7 天。"""
    await mock_redis.setex(SESSION_PREFIX + "s-renew", SESSION_TTL - 1, _session_payload())
    ctx = await get_current_user(_request_with_header("s-renew"), None, mock_redis)
    assert ctx.id == 5
    ttl = await mock_redis.ttl(SESSION_PREFIX + "s-renew")
    # 续期与断言间可能流逝 1 秒，容忍微小漂移
    assert SESSION_TTL - 2 <= ttl <= SESSION_TTL


async def test_does_not_renew_session_ttl_above_threshold(mock_redis):
    """T-AM-081：TTL ≥ 1 天的会话不续期（避免活跃会话无限续命）。"""
    await mock_redis.setex(SESSION_PREFIX + "s-keep", SESSION_TTL + 3600, _session_payload())
    await get_current_user(_request_with_header("s-keep"), None, mock_redis)
    ttl = await mock_redis.ttl(SESSION_PREFIX + "s-keep")
    assert SESSION_TTL < ttl <= SESSION_TTL + 3600


async def test_missing_session_header_is_unauthorized(mock_redis):
    """T-AM-016：无 X-Session-Id 请求返回 401。"""
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""}
    with pytest.raises(auth_module.HTTPException) as exc:
        await get_current_user(Request(scope), None, mock_redis)
    assert exc.value.status_code == 401


async def test_expired_session_is_rejected(mock_redis):
    """T-AM-082：不存在的会话 ID（已过期/伪造）返回 401 TOKEN_INVALID。"""
    with pytest.raises(auth_module.HTTPException) as exc:
        await get_current_user(_request_with_header("nonexistent"), None, mock_redis)
    assert exc.value.status_code == 401
    assert exc.value.detail == ResultCode.TOKEN_INVALID.msg


async def test_tampered_session_payload_without_user_id_is_rejected(mock_redis):
    """会话数据被篡改（缺失 userId）时拒绝，不产生匿名上下文。"""
    await mock_redis.set(SESSION_PREFIX + "s-tampered", json.dumps({"username": "attacker"}))
    with pytest.raises(auth_module.HTTPException):
        await get_current_user(_request_with_header("s-tampered"), None, mock_redis)


async def test_authorities_split_into_roles_and_perms(mock_redis):
    """会话 authorities 正确拆分：ROLE_ 前缀入 roles，其余入 permissions。"""
    await mock_redis.setex(SESSION_PREFIX + "s-split", 3600, _session_payload())
    ctx = await get_current_user(_request_with_header("s-split"), None, mock_redis)
    assert ctx.roles == ["GUEST"]
    assert ctx.permissions == ["sys:auth:log:list"]
