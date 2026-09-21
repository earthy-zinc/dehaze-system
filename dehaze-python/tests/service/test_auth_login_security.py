"""认证管理安全链路测试（pytest 层，对照测试用例.md §4.7 登录失败锁定）。

覆盖：T-AM-050（5 次失败锁定）、T-AM-051（锁定期正确密码也拒绝）、
T-AM-053（成功登录复位计数）、T-AM-005（密码错误脱敏）、T-AM-006（禁用用户）。

用户数据经 monkeypatch 桩掉仓储，Redis 用 fakeredis，不依赖真实库。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service import auth_service as m
from app.service.auth_service import auth_service
from tests.stubs.factories import fake_redis

# 测试替身：登录仓储/菜单权限均被 monkeypatch，db 仅传参占位
_DB: AsyncSession = AsyncMock(spec=AsyncSession)


def _make_user(username: str, status: int = 1):
    return SimpleNamespace(
        id=1,
        username=username,
        nickname=username,
        password="hashed",
        dept_id=None,
        status=status,
    )


def _patch_user_lookup(monkeypatch, username: str, status: int = 1):
    user = _make_user(username, status)
    monkeypatch.setattr(m.user_repository, "get_by_username", AsyncMock(return_value=user))
    monkeypatch.setattr(m.user_repository, "get_user_role_codes", AsyncMock(return_value=[]))
    monkeypatch.setattr(m, "check_password_async", AsyncMock(return_value=True))
    from app.repository.role_repository import role_repository
    from app.service.menu_service import menu_service

    monkeypatch.setattr(role_repository, "get_maximum_data_scope", AsyncMock(return_value=0))
    monkeypatch.setattr(menu_service, "list_role_perms", AsyncMock(return_value=set()))
    return user


def _patch_wrong_password(monkeypatch, username: str):
    """用户存在但密码错误（check_password 返回 False）。"""
    user = _make_user(username)
    monkeypatch.setattr(m.user_repository, "get_by_username", AsyncMock(return_value=user))
    monkeypatch.setattr(m, "check_password_async", AsyncMock(return_value=False))


class TestLoginLockout:
    async def test_five_failures_lock_account(self, monkeypatch):
        """T-AM-050：连续 5 次错误密码后，第 6 次返回锁定错误。"""
        redis = await fake_redis({f"{m.settings.CAPTCHA_KEY_PREFIX}k1": "abcd"})
        _patch_wrong_password(monkeypatch, "admin")
        for _ in range(m.settings.LOGIN_FAIL_MAX_ATTEMPTS):
            with pytest.raises(BusinessException):
                await auth_service._authenticate(_DB, redis, "admin", "pw", "1.1.1.1", "k1", "abcd")
        with pytest.raises(BusinessException) as exc:
            await auth_service._authenticate(_DB, redis, "admin", "pw", "1.1.1.1", "k1", "abcd")
        assert exc.value.code == ResultCode.PASSWORD_ENTER_EXCEED_LIMIT

    async def test_locked_account_rejects_correct_password(self, monkeypatch):
        """T-AM-051：锁定期内即使密码正确也被拒绝（锁前置校验先于凭证校验）。"""
        redis = await fake_redis(
            {
                m.LOGIN_FAIL_PREFIX + "admin": str(m.settings.LOGIN_FAIL_MAX_ATTEMPTS),
                f"{m.settings.CAPTCHA_KEY_PREFIX}k1": "abcd",
            }
        )
        _patch_user_lookup(monkeypatch, "admin")
        with pytest.raises(BusinessException) as exc:
            await auth_service._authenticate(_DB, redis, "admin", "pw", "1.1.1.1", "k1", "abcd")
        assert exc.value.code == ResultCode.PASSWORD_ENTER_EXCEED_LIMIT

    async def test_successful_login_resets_fail_counter(self, monkeypatch):
        """T-AM-053：失败 3 次后正确密码登录成功，计数清零。"""
        redis = await fake_redis(
            {
                m.LOGIN_FAIL_PREFIX + "admin": "3",
                f"{m.settings.CAPTCHA_KEY_PREFIX}k1": "abcd",
            }
        )
        _patch_user_lookup(monkeypatch, "admin")
        result = await auth_service._authenticate(
            _DB, redis, "admin", "pw", "1.1.1.1", "k1", "abcd"
        )
        assert result["sessionId"]
        assert await redis.get(m.LOGIN_FAIL_PREFIX + "admin") is None

    async def test_ip_failures_block_login(self, monkeypatch):
        """IP 维度锁定：同 IP 失败次数达上限后拒绝该 IP 的一切登录。"""
        redis = await fake_redis(
            {
                m.LOGIN_FAIL_IP_PREFIX + "2.2.2.2": str(m.settings.LOGIN_FAIL_MAX_ATTEMPTS),
                f"{m.settings.CAPTCHA_KEY_PREFIX}k1": "abcd",
            }
        )
        _patch_user_lookup(monkeypatch, "admin")
        with pytest.raises(BusinessException) as exc:
            await auth_service._authenticate(_DB, redis, "admin", "pw", "2.2.2.2", "k1", "abcd")
        assert exc.value.code == ResultCode.PASSWORD_ENTER_EXCEED_LIMIT


class TestLoginFailureMessages:
    async def test_user_not_found_uses_masked_message(self, monkeypatch):
        """T-AM-004：用户名不存在返回 A0210 脱敏提示，不泄露用户是否存在。"""
        redis = await fake_redis({f"{m.settings.CAPTCHA_KEY_PREFIX}k1": "abcd"})
        monkeypatch.setattr(m.user_repository, "get_by_username", AsyncMock(return_value=None))
        with pytest.raises(BusinessException) as exc:
            await auth_service._authenticate(_DB, redis, "ghost", "pw", "1.1.1.1", "k1", "abcd")
        assert exc.value.code == ResultCode.USERNAME_OR_PASSWORD_ERROR

    async def test_wrong_password_uses_masked_message(self, monkeypatch):
        """T-AM-005：密码错误与用户名不存在返回同一错误码（脱敏一致）。"""
        redis = await fake_redis({f"{m.settings.CAPTCHA_KEY_PREFIX}k1": "abcd"})
        _patch_wrong_password(monkeypatch, "admin")
        with pytest.raises(BusinessException) as exc:
            await auth_service._authenticate(_DB, redis, "admin", "bad", "1.1.1.1", "k1", "abcd")
        assert exc.value.code == ResultCode.USERNAME_OR_PASSWORD_ERROR
        assert exc.value.message.startswith("用户名或密码错误")

    async def test_disabled_user_rejected(self, monkeypatch):
        """T-AM-006：禁用用户凭正确凭证登录被拒。

        注：文档错误码表将禁用用户记为 A0211，但代码实际抛
        USER_ACCOUNT_LOCKED（A0202 用户账户被冻结）——文档与实现不一致，
        已上报待决策，此处按当前实现断言。
        """
        redis = await fake_redis({f"{m.settings.CAPTCHA_KEY_PREFIX}k1": "abcd"})
        _patch_user_lookup(monkeypatch, "admin", status=0)
        monkeypatch.setattr(m, "check_password_async", AsyncMock(return_value=True))
        with pytest.raises(BusinessException) as exc:
            await auth_service._authenticate(_DB, redis, "admin", "pw", "1.1.1.1", "k1", "abcd")
        assert exc.value.code == ResultCode.USER_ACCOUNT_LOCKED


class TestLoginLogAudit:
    async def test_failed_login_writes_audit_log(self, monkeypatch):
        """登录失败也写入审计日志（auth_service.login 外层 try/except 分支）。"""
        import json as _json
        from unittest.mock import MagicMock

        redis = await fake_redis({f"{m.settings.CAPTCHA_KEY_PREFIX}k1": "abcd"})
        _patch_wrong_password(monkeypatch, "admin")
        repo_mock = MagicMock()
        repo_mock.create_log = AsyncMock()
        monkeypatch.setattr(
            m.login_log_repository, "create_log", repo_mock.create_log, raising=False
        )
        with pytest.raises(BusinessException):
            await auth_service.login(_DB, redis, "admin", "bad", "3.3.3.3", "k1", "abcd", "UA")
        assert repo_mock.create_log.await_count == 1
        args = repo_mock.create_log.await_args.args
        # (user_id, username, ip, status=0 失败, message, browser, os)
        assert args[0] is None
        assert args[3] == 0

        # 对照组：成功登录写 user_id + status=1（首次失败已消费验证码，重新预置）
        from app.repository.member_repository import member_repository

        monkeypatch.setattr(member_repository, "get_or_init_member", AsyncMock())
        await redis.set(f"{m.settings.CAPTCHA_KEY_PREFIX}k2", "abcd")
        _patch_user_lookup(monkeypatch, "admin")
        await auth_service.login(_DB, redis, "admin", "pw", "3.3.3.3", "k2", "abcd", "UA")
        assert repo_mock.create_log.await_count == 2
        args2 = repo_mock.create_log.await_args.args
        assert args2[0] == 1
        assert args2[3] == 1
        assert _json.dumps(args2)  # 参数均为可序列化普通值


class TestLoginMemberProfile:
    async def test_login_seeds_member_profile(self, db, monkeypatch):
        """种子/后台创建的用户不走注册流程，登录成功后应兜底建立 sys_member 行，
        否则计费配额校验 fail-closed 误报"配额不足"。"""
        from app.models.entity.sys_user import SysUser
        from app.repository.member_repository import member_repository
        from app.utils.password import hash_password_async

        username = "seed_login_member_u1"
        user = SysUser(
            username=username,
            nickname=username,
            password=await hash_password_async("pw123456"),
            gender=1,
            status=1,
            deleted=0,
        )
        db.add(user)
        await db.flush()
        assert await member_repository.get_by_user_id(db, user.id) is None

        redis = await fake_redis(
            {
                f"{m.settings.CAPTCHA_KEY_PREFIX}k3": "abcd",
                f"{m.settings.CAPTCHA_KEY_PREFIX}k4": "abcd",
            }
        )
        await auth_service.login(db, redis, username, "pw123456", "4.4.4.4", "k3", "abcd", "UA")

        member = await member_repository.get_by_user_id(db, user.id)
        assert member is not None
        assert member.level_code == "level_0"
        assert member.status == 1
        # 幂等：已存在档案的活跃用户再次登录不重建
        await auth_service.login(db, redis, username, "pw123456", "4.4.4.4", "k4", "abcd", "UA")
        again = await member_repository.get_by_user_id(db, user.id)
        assert again is not None
        assert again.id == member.id
