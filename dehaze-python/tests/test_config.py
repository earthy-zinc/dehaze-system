import os


class TestConfig:

    def test_environment_is_testing(self):
        assert os.getenv("APP_ENV") == "testing"


class TestSettings:

    def test_settings_instance(self):
        from app.config import settings

        assert settings.APP_NAME == "Dehaze API"
        assert settings.APP_VERSION == "1.0.0"

    def test_captcha_config(self):
        from app.config import settings

        assert settings.CAPTCHA_LENGTH >= 4
        assert settings.CAPTCHA_WIDTH > 0
        assert settings.CAPTCHA_HEIGHT > 0
        assert settings.CAPTCHA_EXPIRES > 0
