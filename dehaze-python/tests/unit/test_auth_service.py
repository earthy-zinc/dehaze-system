"""
认证服务测试

测试 AuthService 的核心功能：
- 用户登录
- 验证码生成与验证
- Token 刷新

使用 Mock 进行单元测试，不依赖真实数据库
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from jose import jwt

from app.config import settings
from app.service.auth_service import AuthService
from tests.conftest import MockRedis


@pytest.mark.unit
class TestAuthServiceLogin:
    """登录功能测试（使用 Mock）"""

    @pytest.mark.asyncio
    async def test_login_success(self, mock_redis: MockRedis):
        """测试登录成功"""
        # Mock 用户数据
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = "testuser"
        mock_user.nickname = "Test User"
        mock_user.status = 1
        mock_user.password = "$2b$12$test_hashed_password"  # 模拟的哈希密码

        # Mock repository
        with patch("app.service.auth_service.user_repository") as mock_repo:
            mock_repo.get_by_username = AsyncMock(return_value=mock_user)
            mock_repo.get_user_role_codes = AsyncMock(return_value=["USER"])
            mock_repo.get_user_permissions = AsyncMock(return_value=["system:user:list"])

            # Mock 密码验证
            with patch("app.service.auth_service.check_password_async", AsyncMock(return_value=True)):
                result = await AuthService.login(
                    db=AsyncMock(),
                    username="testuser",
                    password="password123",
                )

        assert result["tokenType"] == "Bearer"
        assert "accessToken" in result
        assert result["user"]["username"] == "testuser"

    @pytest.mark.asyncio
    async def test_login_invalid_username(self, mock_redis: MockRedis):
        """测试用户名不存在"""
        with patch("app.service.auth_service.user_repository") as mock_repo:
            mock_repo.get_by_username = AsyncMock(return_value=None)

            with pytest.raises(ValueError, match="用户名或密码错误"):
                await AuthService.login(
                    db=AsyncMock(),
                    username="nonexistent",
                    password="anypassword",
                )

    @pytest.mark.asyncio
    async def test_login_invalid_password(self, mock_redis: MockRedis):
        """测试密码错误"""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.status = 1
        mock_user.password = "$2b$12$test_hashed_password"

        with patch("app.service.auth_service.user_repository") as mock_repo:
            mock_repo.get_by_username = AsyncMock(return_value=mock_user)

            with patch("app.service.auth_service.check_password_async", AsyncMock(return_value=False)):
                with pytest.raises(ValueError, match="用户名或密码错误"):
                    await AuthService.login(
                        db=AsyncMock(),
                        username="testuser",
                        password="wrongpassword",
                    )

    @pytest.mark.asyncio
    async def test_login_disabled_user(self, mock_redis: MockRedis):
        """测试用户被禁用"""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.status = 0  # 禁用状态
        mock_user.password = "$2b$12$test_hashed_password"

        with patch("app.service.auth_service.user_repository") as mock_repo:
            mock_repo.get_by_username = AsyncMock(return_value=mock_user)

            with patch("app.service.auth_service.check_password_async", AsyncMock(return_value=True)):
                with pytest.raises(ValueError, match="用户已被禁用"):
                    await AuthService.login(
                        db=AsyncMock(),
                        username="testuser",
                        password="password123",
                    )


@pytest.mark.unit
class TestAuthServiceCaptcha:
    """验证码功能测试"""

    @pytest.mark.asyncio
    async def test_get_captcha(self, mock_redis: MockRedis):
        """测试获取验证码"""
        result = await AuthService.get_captcha(mock_redis)

        assert "captchaKey" in result
        assert "captchaBase64" in result
        assert result["captchaBase64"].startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_verify_captcha_success(self, mock_redis: MockRedis):
        """测试验证码验证成功"""
        # 先获取验证码
        captcha_result = await AuthService.get_captcha(mock_redis)
        captcha_key = captcha_result["captchaKey"]

        # 从 MockRedis 中获取存储的验证码
        stored = await mock_redis.get(f"captcha:{captcha_key}")
        assert stored is not None
        captcha_code = stored.decode()

        # 验证
        result = await AuthService.verify_captcha(mock_redis, captcha_key, captcha_code)
        assert result is True

        # 验证后应删除
        stored_after = await mock_redis.get(f"captcha:{captcha_key}")
        assert stored_after is None

    @pytest.mark.asyncio
    async def test_verify_captcha_wrong_code(self, mock_redis: MockRedis):
        """测试验证码错误"""
        # 先获取验证码
        captcha_result = await AuthService.get_captcha(mock_redis)
        captcha_key = captcha_result["captchaKey"]

        # 使用错误的验证码
        result = await AuthService.verify_captcha(mock_redis, captcha_key, "WRONG")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_captcha_expired(self, mock_redis: MockRedis):
        """测试验证码过期"""
        # 使用不存在的 key
        result = await AuthService.verify_captcha(mock_redis, "nonexistent_key", "ABCD")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_captcha_case_insensitive(self, mock_redis: MockRedis):
        """测试验证码不区分大小写"""
        # 手动设置验证码
        await mock_redis.set("captcha:test_key", "ABCD")

        # 使用小写验证
        result = await AuthService.verify_captcha(mock_redis, "test_key", "abcd")
        assert result is True


@pytest.mark.unit
class TestAuthServiceRefreshToken:
    """Token 刷新功能测试"""

    @pytest.mark.asyncio
    async def test_refresh_token_success(self, mock_redis: MockRedis):
        """测试刷新 Token 成功"""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = "testuser"
        mock_user.nickname = "Test User"
        mock_user.status = 1

        with patch("app.service.auth_service.user_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_user)
            mock_repo.get_user_role_codes = AsyncMock(return_value=["USER"])
            mock_repo.get_user_permissions = AsyncMock(return_value=[])

            result = await AuthService.refresh_token(
                db=AsyncMock(),
                user_id=1,
                redis=mock_redis,
            )

        assert result["tokenType"] == "Bearer"
        assert "accessToken" in result
        assert result["user"]["id"] == 1

    @pytest.mark.asyncio
    async def test_refresh_token_user_not_found(self, mock_redis: MockRedis):
        """测试用户不存在"""
        with patch("app.service.auth_service.user_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(ValueError, match="用户不存在"):
                await AuthService.refresh_token(
                    db=AsyncMock(),
                    user_id=99999,
                    redis=mock_redis,
                )

    @pytest.mark.asyncio
    async def test_refresh_token_disabled_user(self, mock_redis: MockRedis):
        """测试用户被禁用"""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.status = 0  # 禁用状态

        with patch("app.service.auth_service.user_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_user)

            with pytest.raises(ValueError, match="用户已被禁用"):
                await AuthService.refresh_token(
                    db=AsyncMock(),
                    user_id=1,
                    redis=mock_redis,
                )


@pytest.mark.unit
@pytest.mark.api
class TestAuthAPI:
    """认证 API 接口测试"""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """测试健康检查 API"""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


@pytest.mark.unit
class TestJWTToken:
    """JWT Token 相关测试"""

    def test_token_generation_and_verification(self):
        """测试 Token 生成和验证"""
        user_id = 1
        username = "testuser"

        # 生成 Token
        jti = str(uuid4())
        payload = {
            "jti": jti,
            "sub": str(user_id),
            "user_id": user_id,
            "username": username,
            "nickname": "Test User",
            "roles": "USER",
            "permissions": "",
            "exp": datetime.now(timezone.utc) + timedelta(seconds=3600),
            "iat": datetime.now(timezone.utc),
        }

        token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")

        # 验证 Token
        decoded = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
        assert decoded["user_id"] == user_id
        assert decoded["username"] == username
        assert decoded["jti"] == jti

    def test_token_expiration(self):
        """测试 Token 过期"""
        # 生成已过期的 Token
        payload = {
            "jti": str(uuid4()),
            "sub": "1",
            "user_id": 1,
            "username": "testuser",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1),  # 已过期
            "iat": datetime.now(timezone.utc) - timedelta(seconds=3600),
        }

        token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")

        # 验证过期
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
