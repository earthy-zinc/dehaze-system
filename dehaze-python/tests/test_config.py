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


class TestSettings:
    """测试配置类"""

    def test_settings_instance(self):
        """验证配置实例"""
        from app.config import settings

        assert settings.APP_NAME == "Dehaze API"
        assert settings.APP_VERSION == "1.0.0"

    def test_captcha_config(self):
        """验证验证码配置"""
        from app.config import settings

        assert settings.CAPTCHA_LENGTH >= 4
        assert settings.CAPTCHA_WIDTH > 0
        assert settings.CAPTCHA_HEIGHT > 0
        assert settings.CAPTCHA_EXPIRES > 0
