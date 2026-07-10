"""
测试配置

验证测试框架配置是否正确
"""

import os

import pytest


class TestConfig:
    """测试配置验证"""

    def test_environment_is_testing(self):
        """验证测试环境变量"""
        assert os.getenv("APP_ENV") == "testing"

    def test_secret_key_set(self):
        """验证密钥已设置"""
        assert os.getenv("SECRET_KEY") is not None
        assert len(os.getenv("SECRET_KEY", "")) >= 32

    def test_jwt_secret_key_set(self):
        """验证 JWT 密钥已设置"""
        assert os.getenv("JWT_SECRET_KEY") is not None
        assert len(os.getenv("JWT_SECRET_KEY", "")) >= 32


class TestSettings:
    """测试配置类"""

    def test_settings_instance(self):
        """验证配置实例"""
        from app.config import settings

        assert settings.APP_NAME == "Dehaze API"
        assert settings.APP_VERSION == "1.0.0"

    def test_jwt_config(self):
        """验证 JWT 配置"""
        from app.config import settings

        assert settings.JWT_ACCESS_TOKEN_EXPIRES > 0
        assert settings.JWT_REFRESH_TOKEN_EXPIRES > 0

    def test_captcha_config(self):
        """验证验证码配置"""
        from app.config import settings

        assert settings.CAPTCHA_LENGTH >= 4
        assert settings.CAPTCHA_WIDTH > 0
        assert settings.CAPTCHA_HEIGHT > 0
        assert settings.CAPTCHA_EXPIRES > 0
