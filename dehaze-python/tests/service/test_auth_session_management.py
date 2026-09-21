"""在线会话管理（F-AM-011）与同时在线设备数上限单元测试。

Redis 走 conftest autouse 的 fakeredis；数据库走 conftest 的真实 MySQL 测试库
（种子已含 level_0/level_1/level_2/level_3 四种会员与 root/admin 账号）。
会话数据结构对齐 auth_service._authenticate 写入的真实 payload。

设备数上限用例的判别力：新语义「不区分设备类型、按在线总数踢最早」——
旧语义（按 deviceType 互踢）下，跨设备类型登录不会踢任何会话，故下列断言必失败。
"""

import json

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.auth import get_current_user
from app.service.auth_service import SESSION_PREFIX, SESSION_TTL, SESSION_USER_PREFIX, auth_service

# 种子数据（config/sql/data）：用户与其会员等级 / 管理员角色
LEVEL_0_USER = 5  # user，level_0 → 1 台
LEVEL_1_USER = 6  # vip1，level_1 → 3 台
LEVEL_2_USER = 7  # vip2，level_2 → 5 台
LEVEL_3_USER = 8  # svip，level_3 → 10 台
NO_MEMBER_USER = 3  # test，无 sys_member 行 → 按 level_0
ADMIN_USER = 2  # admin，等级 level_0 但具 ADMIN 角色
ROOT_USER = 1  # root，无 sys_member 行

BASE_SCORE = 1_700_000_000  # 早于当前时间的固定登录时间戳，保证被踢的是预置会话


def _session_payload(
    user_id: int = LEVEL_0_USER,
    username: str = "user",
    device_type: str = "web",
    authorities: list[str] | None = None,
    login_time: str = "2026-09-01 10:00:00",
) -> str:
    return json.dumps(
        {
            "userId": user_id,
            "username": username,
            "nickname": username,
            "deptId": None,
            "dataScope": 0,
            "authorities": authorities if authorities is not None else ["ROLE_GUEST"],
            "deviceType": device_type,
            "loginIp": "1.2.3.4",
            "loginTime": login_time,
            "lastAccessTime": login_time,
        }
    )


async def _seed_session(
    mock_redis, session_id: str, payload: str, index_score: int | None = None
) -> None:
    """写入会话键；index_score 非空时同步写入设备索引 ZSet（score=登录 epoch 秒）。"""
    await mock_redis.setex(SESSION_PREFIX + session_id, SESSION_TTL, payload)
    if index_score is not None:
        user_id = json.loads(payload)["userId"]
        await mock_redis.zadd(f"{SESSION_USER_PREFIX}{user_id}", {session_id: index_score})


def _request_with_session(session_id: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [(b"x-session-id", session_id.encode())],
        }
    )


# ---------- list_sessions ----------


async def test_list_sessions_filters_by_username_excludes_index_keys(mock_redis):
    await _seed_session(
        mock_redis, "s-a-web", _session_payload(username="alice"), index_score=BASE_SCORE
    )
    await _seed_session(
        mock_redis,
        "s-a-android",
        _session_payload(username="alice", device_type="android"),
        index_score=BASE_SCORE + 1,
    )
    await _seed_session(mock_redis, "s-b", _session_payload(username="bob"), index_score=BASE_SCORE)
    # 设备索引键（session:user:*）不得混入会话列表
    await mock_redis.zadd(f"{SESSION_USER_PREFIX}1", {"s-a-web": BASE_SCORE})

    sessions = await auth_service.list_sessions(mock_redis, "alice")

    assert {s["sessionId"] for s in sessions} == {"s-a-web", "s-a-android"}
    web = next(s for s in sessions if s["sessionId"] == "s-a-web")
    assert web["deviceType"] == "web"
    assert web["ip"] == "1.2.3.4"
    assert web["loginTime"] == "2026-09-01 10:00:00"
    assert web["lastAccessTime"] == "2026-09-01 10:00:00"


async def test_list_sessions_empty_for_unknown_user(mock_redis):
    sessions = await auth_service.list_sessions(mock_redis, "nonexistent_user_xyz")
    assert sessions == []


async def test_list_sessions_sorted_by_login_time_desc(mock_redis):
    await _seed_session(
        mock_redis, "s-old", _session_payload(username="alice", login_time="2026-09-01 08:00:00")
    )
    await _seed_session(
        mock_redis, "s-new", _session_payload(username="alice", login_time="2026-09-02 09:00:00")
    )
    sessions = await auth_service.list_sessions(mock_redis, "alice")
    assert [s["sessionId"] for s in sessions] == ["s-new", "s-old"]


# ---------- kick_session ----------


async def test_kick_session_deletes_session_and_index_member(mock_redis):
    await _seed_session(mock_redis, "s-kick", _session_payload(), index_score=BASE_SCORE)

    await auth_service.kick_session(mock_redis, "s-kick")

    assert await mock_redis.get(SESSION_PREFIX + "s-kick") is None
    assert await mock_redis.zrange(f"{SESSION_USER_PREFIX}{LEVEL_0_USER}", 0, -1) == []


async def test_kick_session_keeps_index_of_other_session(mock_redis):
    """索引中其他会话元素不被误删。"""
    await _seed_session(mock_redis, "s-old", _session_payload())
    await mock_redis.zadd(f"{SESSION_USER_PREFIX}{LEVEL_0_USER}", {"s-new": BASE_SCORE})

    await auth_service.kick_session(mock_redis, "s-old")

    assert await mock_redis.zrange(f"{SESSION_USER_PREFIX}{LEVEL_0_USER}", 0, -1) == ["s-new"]


async def test_kick_session_rejects_root(mock_redis):
    await _seed_session(
        mock_redis,
        "s-root",
        _session_payload(
            user_id=ROOT_USER, username="root", authorities=["ROLE_ROOT", "sys:auth:session:kick"]
        ),
    )
    with pytest.raises(BusinessException) as exc:
        await auth_service.kick_session(mock_redis, "s-root")
    assert exc.value.code == ResultCode.OPERATION_NOT_ALLOW
    # 超管会话必须原样保留
    assert await mock_redis.get(SESSION_PREFIX + "s-root") is not None


async def test_kick_session_not_found(mock_redis):
    with pytest.raises(BusinessException) as exc:
        await auth_service.kick_session(mock_redis, "nonexistent-session")
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


# ---------- 同时在线设备数上限（F-AM-011） ----------


async def test_device_limit_level_0_allows_single_session(db, mock_redis):
    """level_0 上限 1 台：第 2 次登录踢掉唯一旧会话。"""
    await _seed_session(
        mock_redis, "s-0-old", _session_payload(device_type="android"), index_score=BASE_SCORE
    )

    await auth_service._enforce_device_limit(db, mock_redis, "s-0-new", LEVEL_0_USER, ["GUEST"])

    assert await mock_redis.get(SESSION_PREFIX + "s-0-old") is None
    assert await mock_redis.zrange(f"{SESSION_USER_PREFIX}{LEVEL_0_USER}", 0, -1) == ["s-0-new"]
    assert await mock_redis.zcard(f"{SESSION_USER_PREFIX}{LEVEL_0_USER}") == 1


async def test_device_limit_kicks_earliest_by_login_time_across_device_types(db, mock_redis):
    """level_1 上限 3 台：第 4 次登录踢最早登录者，与设备类型无关（旧语义跨端不互踢）。"""
    for idx, device in enumerate(("web", "android", "flutter")):
        await _seed_session(
            mock_redis,
            f"s-1-{device}",
            _session_payload(user_id=LEVEL_1_USER, device_type=device),
            index_score=BASE_SCORE + idx,
        )

    await auth_service._enforce_device_limit(db, mock_redis, "s-1-new", LEVEL_1_USER, ["GUEST"])

    # 最早登录的 web 端被踢，android/flutter 与新会话保留
    assert await mock_redis.get(SESSION_PREFIX + "s-1-web") is None
    assert await mock_redis.get(SESSION_PREFIX + "s-1-android") is not None
    assert await mock_redis.get(SESSION_PREFIX + "s-1-flutter") is not None
    index = await mock_redis.zrange(f"{SESSION_USER_PREFIX}{LEVEL_1_USER}", 0, -1)
    assert index == ["s-1-android", "s-1-flutter", "s-1-new"]


async def test_device_limit_not_exceeded_keeps_all_sessions(db, mock_redis):
    """level_3 上限 10 台：第 10 台登录仍在限内，不踢任何会话。"""
    for idx in range(9):
        await _seed_session(
            mock_redis,
            f"s-3-{idx}",
            _session_payload(user_id=LEVEL_3_USER),
            index_score=BASE_SCORE + idx,
        )

    await auth_service._enforce_device_limit(db, mock_redis, "s-3-new", LEVEL_3_USER, ["GUEST"])

    assert await mock_redis.zcard(f"{SESSION_USER_PREFIX}{LEVEL_3_USER}") == 10
    for idx in range(9):
        assert await mock_redis.get(SESSION_PREFIX + f"s-3-{idx}") is not None


async def test_device_limit_multi_excess_evicts_exactly_overflow(db, mock_redis):
    """一次性超出多台时，只踢超出部分（最早登录的若干台）。"""
    for idx in range(7):
        await _seed_session(
            mock_redis,
            f"s-2-{idx}",
            _session_payload(user_id=LEVEL_2_USER),
            index_score=BASE_SCORE + idx,
        )

    await auth_service._enforce_device_limit(db, mock_redis, "s-2-new", LEVEL_2_USER, ["GUEST"])

    # level_2 = 5 台：8 台在线踢掉最早的 3 台
    assert await mock_redis.zcard(f"{SESSION_USER_PREFIX}{LEVEL_2_USER}") == 5
    for idx in range(3):
        assert await mock_redis.get(SESSION_PREFIX + f"s-2-{idx}") is None
    assert await mock_redis.get(SESSION_PREFIX + "s-2-3") is not None
    assert await mock_redis.get(SESSION_PREFIX + "s-2-6") is not None


async def test_device_limit_admin_fixed_10_regardless_of_level(db, mock_redis):
    """管理员不受等级权益约束：admin 账号本身是 level_0，仍保留 10 台。"""
    for idx in range(10):
        await _seed_session(
            mock_redis,
            f"s-a-{idx}",
            _session_payload(user_id=ADMIN_USER, username="admin"),
            index_score=BASE_SCORE + idx,
        )

    await auth_service._enforce_device_limit(db, mock_redis, "s-a-new", ADMIN_USER, ["ADMIN"])

    assert await mock_redis.zcard(f"{SESSION_USER_PREFIX}{ADMIN_USER}") == 10
    assert await mock_redis.get(SESSION_PREFIX + "s-a-0") is None  # 第 11 台登录踢最早
    assert await mock_redis.get(SESSION_PREFIX + "s-a-1") is not None

    # 同一账号去掉管理员角色后按 level_0 收紧到 1 台（反证上面是"管理员豁免"而非等级权益生效）
    await auth_service._enforce_device_limit(db, mock_redis, "s-a-user", ADMIN_USER, ["GUEST"])
    assert await mock_redis.zcard(f"{SESSION_USER_PREFIX}{ADMIN_USER}") == 1
    assert await mock_redis.zrange(f"{SESSION_USER_PREFIX}{ADMIN_USER}", 0, -1) == ["s-a-user"]


async def test_device_limit_root_and_member_missing_defaults(db, mock_redis):
    """root 固定 10 台（无 sys_member 行）；无会员记录普通用户按 level_0=1 台。"""
    for idx in range(10):
        await _seed_session(
            mock_redis,
            f"s-r-{idx}",
            _session_payload(user_id=ROOT_USER, username="root"),
            index_score=BASE_SCORE + idx,
        )
    await auth_service._enforce_device_limit(db, mock_redis, "s-r-new", ROOT_USER, ["ROOT"])
    assert await mock_redis.zcard(f"{SESSION_USER_PREFIX}{ROOT_USER}") == 10

    await _seed_session(
        mock_redis,
        "s-nm-old",
        _session_payload(user_id=NO_MEMBER_USER, username="test"),
        index_score=BASE_SCORE,
    )
    await auth_service._enforce_device_limit(db, mock_redis, "s-nm-new", NO_MEMBER_USER, ["GUEST"])
    assert await mock_redis.zrange(f"{SESSION_USER_PREFIX}{NO_MEMBER_USER}", 0, -1) == ["s-nm-new"]


async def test_evicted_session_request_rejected_as_unauthorized(db, mock_redis):
    """被踢会话的下一次请求返回 401（会话键已删除，索引同步清理）。"""
    await _seed_session(mock_redis, "s-evictee", _session_payload(), index_score=BASE_SCORE)
    # 新会话的会话键由 _authenticate 在强制上限之后写入（索引先写，用于限额判定）
    await mock_redis.setex(SESSION_PREFIX + "s-survivor", SESSION_TTL, _session_payload())

    await auth_service._enforce_device_limit(db, mock_redis, "s-survivor", LEVEL_0_USER, ["GUEST"])

    with pytest.raises(HTTPException) as exc:
        await get_current_user(_request_with_session("s-evictee"), None, mock_redis)
    assert exc.value.status_code == 401
    # 新会话仍可正常鉴权
    ctx = await get_current_user(_request_with_session("s-survivor"), None, mock_redis)
    assert ctx.id == LEVEL_0_USER
