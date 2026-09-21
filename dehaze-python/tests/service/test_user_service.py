"""用户管理服务层集成测试（真实 MySQL dehaze_test，事务回滚零污染）。

覆盖文档测试用例.md：用户名唯一性（含大小写变体）、默认密码加密存储、
密码复杂度边界（8-20 位含字母数字）、删除保护（自己/超级管理员）、
状态管理保护（超级管理员/自己）、对抗性脏语料。
"""

import json
from datetime import datetime
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_member import SysMember
from app.models.schema.user import UserForm
from app.service.user_service import UserService, validate_password_complexity
from app.utils.password import check_password_async
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.requires_db


# 种子角色：ROOT(id=1)，用于角色关联
_ROOT_ROLE_ID = 1


def _svc() -> UserService:
    return UserService(audit_repo=Mock())


async def _create_user(db, username: str, nickname: str = "测试昵称", **extra) -> None:
    data = {
        "username": username,
        "nickname": nickname,
        "deptId": 1,
        "roleIds": [_ROOT_ROLE_ID],
        **extra,
    }
    await _svc().create_user_with_roles(db, data)


# ===== 用户新增 =====


async def test_create_user_stores_default_password_hashed(db):
    """默认密码 bcrypt 加密存储，非明文且可通过校验（文档 T-UM-038）"""
    await _create_user(db, "tum_create_hash")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_create_hash")
    assert user is not None
    assert user.password != settings.DEFAULT_PASSWORD
    assert user.password is not None
    assert await check_password_async(settings.DEFAULT_PASSWORD, user.password) is True


async def test_create_user_links_roles(db):
    await _create_user(db, "tum_create_roles")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_create_roles")
    assert user is not None
    role_ids = await UserService().repo.get_user_role_ids(db, user.id)
    assert role_ids == [_ROOT_ROLE_ID]


async def test_create_user_duplicate_username_rejected(db):
    """与种子用户 test 冲突 → A0501（三端统一业务码，文档 T-UM-012）"""
    with pytest.raises(BusinessException) as ei:
        await _create_user(db, "test")
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_create_username_case_insensitive_unique(db):
    """大小写变体视为同一用户名（utf8mb4_0900_ai_ci，防 Admin/admin 混淆登录）"""
    await _create_user(db, "tum_case_user")
    with pytest.raises(BusinessException) as ei:
        await _create_user(db, "TUM_CASE_USER")
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_create_username_deleted_still_unique(db, mock_redis):
    """软删用户名不可复用（注册/改名查重含软删行）"""
    await _create_user(db, "tum_deleted_user")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_deleted_user")
    assert user is not None
    ctx = make_user_context(id=user.id + 100, username="operator")
    await _svc().delete_users(db, mock_redis, str(user.id), current_user=ctx)
    with pytest.raises(BusinessException) as ei:
        await _create_user(db, "tum_deleted_user")
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_create_user_empty_username_rejected(db):
    with pytest.raises(BusinessException):
        await _svc().create_user_with_roles(db, {"username": "", "deptId": 1, "roleIds": [1]})


async def test_create_nickname_with_dirty_corpus_roundtrip(db):
    """对抗性脏语料：emoji + 全半角混杂昵称可创建且读回一致（不含换行符）"""
    dirty = "测试Γ用户·Ｆｕｌｌ-ｗｉｄｔｈ🎮中文ab_123"
    await _create_user(db, "tum_dirty_nick", nickname=dirty)
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_dirty_nick")
    assert user is not None
    assert user.nickname == dirty


def test_userform_nickname_exceeds_max_length_rejected():
    with pytest.raises(ValidationError):
        UserForm(
            username="tum_long_nick",
            nickname="n" * 65,
            deptId=1,
            roleIds=[1],
        )


def test_userform_username_exceeds_max_length_rejected():
    """username 列宽 varchar(64)，schema 必须同步拦截，避免落库 500"""
    with pytest.raises(ValidationError):
        UserForm(
            username="u" * 65,
            nickname="昵称",
            deptId=1,
            roleIds=[1],
        )


# ===== 密码管理 =====


async def test_update_password_rehashes_and_invalidates_old(db, mock_redis):
    """重置密码后新密码生效、旧密码失效（文档 T-UM-034）"""
    await _create_user(db, "tum_pwd_user")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_pwd_user")
    assert user is not None
    old_hash = user.password
    new_pwd = "NewPwd_tum_2026"
    await _svc().update_password(db, mock_redis, user.id, new_pwd)
    assert user.password is not None
    assert await check_password_async(new_pwd, user.password) is True
    # 旧哈希不再匹配新密码（bcrypt 加盐，每次哈希不同）
    assert user.password != old_hash
    assert await check_password_async(settings.DEFAULT_PASSWORD, user.password) is False


@pytest.mark.parametrize(
    "weak_pwd",
    [
        "Aa1",  # 过短
        "a" * 21,  # 超过 20 位上限（文档 T-UM-035：8-20 位）
        "abcdefgh",  # 纯字母
        "12345678",  # 纯数字
    ],
)
async def test_update_password_weak_rejected(db, mock_redis, weak_pwd):
    """弱密码/超限密码拒绝 → A0400（文档 T-UM-035/T-UM-036）"""
    await _create_user(db, "tum_weak_pwd")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_weak_pwd")
    assert user is not None
    with pytest.raises(BusinessException) as ei:
        await _svc().update_password(db, mock_redis, user.id, weak_pwd)
    assert ei.value.code == ResultCode.PARAM_ERROR


async def test_update_password_boundary_lengths_accepted(db, mock_redis):
    """边界：8 位（最短）与 20 位（最长）合法密码均通过"""
    await _create_user(db, "tum_pwd_boundary")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_pwd_boundary")
    assert user is not None
    await _svc().update_password(db, mock_redis, user.id, "Ab1aaaaa")  # 8 位
    await _svc().update_password(db, mock_redis, user.id, "A1b" + "c" * 17)  # 20 位
    assert user.password is not None
    assert await check_password_async("A1b" + "c" * 17, user.password) is True


async def test_update_password_user_not_found(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await _svc().update_password(db, mock_redis, 99999999, "Abcd1234")
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 用户删除 =====


async def test_delete_users_empty_ids_rejected(db, mock_redis):
    with pytest.raises(BusinessException):
        await _svc().delete_users(db, mock_redis, "  ", current_user=make_user_context(id=2))


async def test_delete_users_self_rejected(db, mock_redis):
    """不可删除自己（文档 T-UM-030）"""
    ctx = make_user_context(id=6, username="operator")
    with pytest.raises(BusinessException) as ei:
        await _svc().delete_users(db, mock_redis, "6", current_user=ctx)
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW


async def test_delete_users_root_protected(db, mock_redis):
    """种子 root(id=1) 不可删除（文档 T-UM-029）"""
    ctx = make_user_context(id=2, username="admin")
    with pytest.raises(BusinessException) as ei:
        await _svc().delete_users(db, mock_redis, "1", current_user=ctx)
    assert ei.value.code == ResultCode.ROOT_USER_PROTECTED


async def test_delete_users_batch_soft_delete_and_hidden_from_list(db, mock_redis):
    """批量删除：软删生效、get_by_id 不可见（文档 T-UM-028/T-UM-033a）"""
    await _create_user(db, "tum_del_batch1")
    await _create_user(db, "tum_del_batch2")
    repo = UserService().repo
    u1 = await repo.get_by_username_include_deleted(db, "tum_del_batch1")
    assert u1 is not None
    u2 = await repo.get_by_username_include_deleted(db, "tum_del_batch2")
    assert u2 is not None
    ctx = make_user_context(id=u2.id + 100, username="operator")
    result = await _svc().delete_users(db, mock_redis, f"{u1.id},{u2.id}", current_user=ctx)
    assert result == {"deleted_count": 2}
    assert await repo.get_by_id(db, u1.id) is None
    assert await repo.get_by_id(db, u2.id) is None


# ===== 状态管理 =====


async def test_update_status_root_disable_rejected(db, mock_redis):
    """种子 root(id=1) 不可禁用（文档 T-UM-041）"""
    with pytest.raises(BusinessException) as ei:
        await _svc().update_user_status(db, mock_redis, 1, 0, current_user=make_user_context(id=2))
    assert ei.value.code == ResultCode.ROOT_USER_PROTECTED


async def test_update_status_root_enable_allowed(db, mock_redis):
    """启用超级管理员不受限（保护仅针对禁用，防自锁场景）"""
    await _svc().update_user_status(db, mock_redis, 1, 1, current_user=make_user_context(id=2))


async def test_update_status_disable_self_rejected(db, mock_redis):
    """不可禁用自己（文档 T-UM-042）"""
    await _create_user(db, "tum_status_self")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_status_self")
    assert user is not None
    assert user.username is not None
    ctx = make_user_context(id=user.id, username=user.username)
    with pytest.raises(BusinessException) as ei:
        await _svc().update_user_status(db, mock_redis, user.id, 0, current_user=ctx)
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW


async def test_update_status_toggle_normal_user(db, mock_redis):
    """禁用→启用往返（文档 T-UM-039/T-UM-040）"""
    await _create_user(db, "tum_status_toggle")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_status_toggle")
    assert user is not None
    svc = _svc()
    await svc.update_user_status(db, mock_redis, user.id, 0, current_user=make_user_context(id=2))
    assert user.status == 0
    await svc.update_user_status(db, mock_redis, user.id, 1, current_user=make_user_context(id=2))
    assert user.status == 1


# ===== 密码复杂度纯函数边界 =====


@pytest.mark.parametrize(
    ("pwd", "valid"),
    [
        ("Aa1aaaaa", True),  # 8 位下界
        ("A1" + "a" * 18, True),  # 20 位上界
        ("Aa1aaaa", False),  # 7 位
        ("A" * 21 + "1", False),  # 21 位
        ("12345678", False),  # 无字母
        ("abcdefgh", False),  # 无数字
    ],
)
def test_validate_password_complexity_boundaries(pwd, valid):
    assert validate_password_complexity(pwd)[0] is valid


# ===== 用户类型 user_type =====


async def test_create_user_default_user_type_personal(db):
    """userType 缺省落库 personal（列默认值语义一致）"""
    await _create_user(db, "tum_user_type_default")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_user_type_default")
    assert user is not None
    assert user.user_type == "personal"


async def test_create_user_enterprise_type_roundtrip(db):
    await _create_user(db, "tum_user_type_ent", userType="enterprise")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_user_type_ent")
    assert user is not None
    assert user.user_type == "enterprise"


async def test_update_user_user_type(db):
    await _create_user(db, "tum_user_type_upd")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_user_type_upd")
    assert user is not None
    await _svc().update_user_with_roles(db, user.id, {"userType": "enterprise", "roleIds": [1]})
    assert user.user_type == "enterprise"


# ===== 用户列表会员字段聚合（memberLevel/memberExpireTime/quotaUsage） =====


async def _prefill_member(db, user_id: int, **extra) -> SysMember:
    member = SysMember(user_id=user_id, level_code="level_2", growth_value=100, **extra)
    db.add(member)
    await db.flush()
    return member


async def test_get_user_list_member_fields_aggregated(db):
    """有会员记录：memberLevel/memberExpireTime/quotaUsage 取自 sys_member；
    无会员记录：memberLevel/memberExpireTime 为 null、quotaUsage 为 "0/0"（§3.1.3）"""
    await _create_user(db, "tum_member_agg1")
    await _create_user(db, "tum_member_agg2")
    repo = UserService().repo
    u1 = await repo.get_by_username_include_deleted(db, "tum_member_agg1")
    assert u1 is not None
    u2 = await repo.get_by_username_include_deleted(db, "tum_member_agg2")
    assert u2 is not None
    await _prefill_member(
        db,
        u1.id,
        expire_time=datetime(2026, 12, 31, 23, 59, 59),
        monthly_dehaze_quota=50,
        monthly_dehaze_used=30,
        monthly_evaluate_quota=50,
        monthly_evaluate_used=0,
    )

    users, _total = await _svc().get_user_list(db, page=1, page_size=10)
    by_id = {u["id"]: u for u in users}

    m1 = by_id[u1.id]
    assert m1["user_type"] == "personal"
    assert m1["memberLevel"] == "level_2"
    assert m1["memberExpireTime"] == "2026-12-31 23:59:59"
    # 8 类任务 used/quota 求和（此处仅 2 类非零）
    assert m1["quotaUsage"] == "30/100"

    m2 = by_id[u2.id]
    assert m2["memberLevel"] is None
    assert m2["memberExpireTime"] is None
    assert m2["quotaUsage"] == "0/0"


async def test_get_user_list_soft_deleted_member_treated_as_none(db):
    """会员记录软删后视同无会员（memberLevel 为 null，不展示历史等级）"""
    await _create_user(db, "tum_member_softdel")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_member_softdel")
    assert user is not None
    member = await _prefill_member(db, user.id, monthly_dehaze_quota=100, monthly_dehaze_used=10)
    member.deleted = 1
    await db.flush()

    users, _ = await _svc().get_user_list(db, page=1, page_size=10)
    row = next(u for u in users if u["id"] == user.id)
    assert row["memberLevel"] is None
    assert row["quotaUsage"] == "0/0"


# ===== 会话踢出联动（禁用/删除/重置密码 → 踢出目标用户全部在线会话） =====


async def test_update_password_kicks_target_user_sessions_and_perm_cache(db, mock_redis):
    await _create_user(db, "tum_kick_pwd")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_kick_pwd")
    assert user is not None
    await mock_redis.set(
        "session:abc",
        json.dumps({"userId": user.id, "username": "tum_kick_pwd", "authorities": []}),
    )
    await mock_redis.zadd(f"session:user:{user.id}", {"abc": 1_700_000_000})
    await mock_redis.set("role:perms:ROOT", '["x"]')  # 新用户关联种子角色 ROOT(id=1)

    await _svc().update_password(db, mock_redis, user.id, "NewPwd_2026x")

    assert await mock_redis.get("session:abc") is None
    assert await mock_redis.get(f"session:user:{user.id}") is None
    assert await mock_redis.get("role:perms:ROOT") is None


async def test_disable_user_kicks_sessions_enable_does_not(db, mock_redis):
    await _create_user(db, "tum_kick_status")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_kick_status")
    assert user is not None
    svc = _svc()
    session_payload = json.dumps(
        {"userId": user.id, "username": "tum_kick_status", "authorities": []}
    )
    await mock_redis.set("session:def", session_payload)
    await mock_redis.zadd(f"session:user:{user.id}", {"def": 1_700_000_000})

    await svc.update_user_status(db, mock_redis, user.id, 0, current_user=make_user_context(id=2))
    assert await mock_redis.get("session:def") is None

    # 启用不踢会话（保护仅针对禁用）
    await mock_redis.set("session:ghi", session_payload)
    await mock_redis.zadd(f"session:user:{user.id}", {"ghi": 1_700_000_000})
    await svc.update_user_status(db, mock_redis, user.id, 1, current_user=make_user_context(id=2))
    assert await mock_redis.get("session:ghi") is not None
    assert await mock_redis.exists(f"session:user:{user.id}")


async def test_delete_users_kicks_sessions(db, mock_redis):
    await _create_user(db, "tum_kick_del")
    user = await UserService().repo.get_by_username_include_deleted(db, "tum_kick_del")
    assert user is not None
    await mock_redis.set(
        "session:xyz",
        json.dumps({"userId": user.id, "username": "tum_kick_del", "authorities": []}),
    )
    await mock_redis.zadd(f"session:user:{user.id}", {"xyz": 1_700_000_000})

    ctx = make_user_context(id=user.id + 100, username="operator")
    await _svc().delete_users(db, mock_redis, str(user.id), current_user=ctx)

    assert await mock_redis.get("session:xyz") is None
    assert await mock_redis.get(f"session:user:{user.id}") is None
