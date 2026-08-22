from types import SimpleNamespace

import pytest

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service import auth_service as m
from app.service.auth_service import auth_service
from tests.stubs import async_ret, fake_redis


def _patch_auth_success(monkeypatch):
    user = SimpleNamespace(
        id=1, username="admin", nickname="管理员", password="hashed",
        dept_id=None, status=1,
    )
    monkeypatch.setattr(m.user_repository, "get_by_username", async_ret(user))
    monkeypatch.setattr(m.user_repository, "get_user_role_codes", async_ret([]))
    monkeypatch.setattr(m, "check_password_async", async_ret(True))
    from app.repository.role_repository import role_repository
    from app.service.menu_service import menu_service
    monkeypatch.setattr(role_repository, "get_maximum_data_scope", async_ret(0))
    monkeypatch.setattr(menu_service, "list_role_perms", async_ret(set()))


class TestVerifyCaptchaStatus:
    async def test_expired_when_key_missing(self):
        redis = await fake_redis()
        ok, expired = await auth_service.verify_captcha_status(redis, "no-such-key", "ABCD")
        assert ok is False and expired is True

    async def test_wrong_code_not_expired(self):
        redis = await fake_redis({f"{settings.CAPTCHA_KEY_PREFIX}k1": "abcd"})
        ok, expired = await auth_service.verify_captcha_status(redis, "k1", "wrong")
        assert ok is False and expired is False

    async def test_match_returns_ok(self):
        redis = await fake_redis({f"{settings.CAPTCHA_KEY_PREFIX}k1": "AbCd"})
        ok, expired = await auth_service.verify_captcha_status(redis, "k1", "abcd")
        assert ok is True and expired is False
        assert await redis.get(f"{settings.CAPTCHA_KEY_PREFIX}k1") is None


class TestLoginCaptchaErrorCode:
    async def test_wrong_captcha_returns_a0214(self, monkeypatch):
        redis = await fake_redis({f"{settings.CAPTCHA_KEY_PREFIX}k1": "abcd"})
        _patch_auth_success(monkeypatch)
        with pytest.raises(BusinessException) as exc:
            await auth_service._authenticate(None, redis, "admin", "pw", "ip", "k1", "WRONG")
        assert exc.value.code == ResultCode.VERIFY_CODE_ERROR

    async def test_expired_captcha_returns_a0213(self, monkeypatch):
        redis = await fake_redis()
        _patch_auth_success(monkeypatch)
        with pytest.raises(BusinessException) as exc:
            await auth_service._authenticate(None, redis, "admin", "pw", "ip", "no-key", "ABCD")
        assert exc.value.code == ResultCode.VERIFY_CODE_TIMEOUT

    async def test_captcha_failure_counts_toward_lockout(self, monkeypatch):
        redis = await fake_redis({f"{settings.CAPTCHA_KEY_PREFIX}k1": "abcd"})
        _patch_auth_success(monkeypatch)
        fail_key = m.LOGIN_FAIL_PREFIX + "admin"
        for _ in range(3):
            with pytest.raises(BusinessException):
                await auth_service._authenticate(None, redis, "admin", "pw", "ip", "k1", "WRONG")
        assert int(await redis.get(fail_key)) == 3
