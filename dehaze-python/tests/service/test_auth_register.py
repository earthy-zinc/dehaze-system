"""用户注册流程测试（对照测试用例.md §4.2，T-AM-031 ~ T-AM-044）。

db fixture 为真实 MySQL 测试库 + 测试级事务回滚；Redis 用独立 fakeredis
（register 由测试显式传入）。覆盖：正常注册、GUEST 默认角色、密码哈希存储、
重复用户名、对抗性脏语料（emoji/零宽/CRLF 昵称）、验证码区分。
"""

import pytest

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.auth_service import auth_service
from tests.stubs.factories import fake_redis


async def _captcha_ok(redis, key: str = "k1", code: str = "abcd"):
    await redis.set(f"{settings.CAPTCHA_KEY_PREFIX}{key}", code)
    return key, code


class TestRegisterSuccess:
    async def test_register_creates_user_with_guest_role_and_hashed_password(self, db):
        """T-AM-031/040/041/042：注册成功，默认 GUEST 角色，密码哈希存储。"""
        redis = await fake_redis()
        key, code = await _captcha_ok(redis)
        result = await auth_service.register(
            db, redis, "reguser01", "Str0ngPass!x", "注册用户", key, code
        )
        assert result["sessionId"]
        assert result["user"]["username"] == "reguser01"

        from app.repository.user_repository import user_repository

        user = await user_repository.get_by_username(db, "reguser01")
        assert user is not None
        assert user.password != "Str0ngPass!x"  # 非明文
        from app.utils.password import check_password_async

        assert user.password is not None
        assert await check_password_async("Str0ngPass!x", user.password)

        role_codes = await user_repository.get_user_role_codes(db, user.id)
        assert "GUEST" in role_codes

        # 注册成功后验证码已消费（一次性）
        assert await redis.get(f"{settings.CAPTCHA_KEY_PREFIX}{key}") is None

    async def test_username_normalized_to_lowercase(self, db):
        """T-AM-032 依据：注册用户名统一小写化，"Admin" 与 "admin" 冲突。"""
        redis = await fake_redis()
        key, code = await _captcha_ok(redis)
        result = await auth_service.register(
            db, redis, "  MiXeDCase  ", "Str0ngPass!x", "大小写", key, code
        )
        assert result["user"]["username"] == "mixedcase"


class TestRegisterRejected:
    async def test_duplicate_username_case_insensitive(self, db):
        """T-AM-032/033：与种子用户 admin 冲突（含大小写变体）返回 A0501。"""
        redis = await fake_redis()
        key, code = await _captcha_ok(redis)
        with pytest.raises(BusinessException) as exc:
            await auth_service.register(db, redis, "ADMIN", "Str0ngPass!x", "重复", key, code)
        assert exc.value.code == ResultCode.DATA_EXISTS

    async def test_captcha_expired_and_wrong_are_distinguished(self, db):
        """T-AM-034/035：验证码过期 A0213 与验证码错误 A0214 区分。"""
        redis = await fake_redis()
        with pytest.raises(BusinessException) as exc:
            await auth_service.register(db, redis, "regx01", "Str0ngPass!x", "x", "nokey", "0000")
        assert exc.value.code == ResultCode.VERIFY_CODE_TIMEOUT

        key, _ = await _captcha_ok(redis)
        with pytest.raises(BusinessException) as exc:
            await auth_service.register(db, redis, "regx01", "Str0ngPass!x", "x", key, "WRONG")
        assert exc.value.code == ResultCode.VERIFY_CODE_ERROR


class TestRegisterDirtyInput:
    async def test_dirty_nickname_stored_as_is(self, db):
        """对抗性脏语料：emoji、零宽字符、CRLF 昵称按 utf8mb4 原样入库（不崩、不 500）。"""
        redis = await fake_redis()
        dirty_nicknames = [
            "用户😀emoji",
            "zero​width",
            "crlf\r\nline",
        ]
        for i, nickname in enumerate(dirty_nicknames):
            key, code = await _captcha_ok(redis)
            result = await auth_service.register(
                db, redis, f"dirty{i:02d}", "Str0ngPass!x", nickname, key, code
            )
            assert result["user"]["nickname"] == nickname

    async def test_bom_username_and_empty_code_rejected(self, db):
        """BOM 用户名走正常注册路径（不崩）；验证码空值返回验证码错误而非异常。"""
        redis = await fake_redis()
        key, code = await _captcha_ok(redis)
        result = await auth_service.register(db, redis, "﻿bomuser", "Str0ngPass!x", "BOM", key, code)
        assert result["user"]["username"] == "﻿bomuser"

        key2, _ = await _captcha_ok(redis, "k2", "abcd")
        with pytest.raises(BusinessException) as exc:
            await auth_service.register(db, redis, "regx02", "Str0ngPass!x", "x", key2, "")
        assert exc.value.code == ResultCode.VERIFY_CODE_ERROR
