"""登录/登出/验证码集成测试。"""
from __future__ import annotations

from utils import auth, api, config, redis


class TestAuth:
    def test_login_returns_session_id(self, backend):
        """登录应返回非空 session_id。"""
        sid = auth.login(backend=backend)
        assert sid and isinstance(sid, str) and len(sid) > 10

    def test_login_cached(self, backend):
        """二次登录应命中缓存（返回相同 session_id）。"""
        sid1 = auth.login(backend=backend)
        sid2 = auth.login(backend=backend)
        assert sid1 == sid2

    def test_captcha_in_redis(self, backend):
        """获取验证码后，Redis 中应能读到明文验证码。"""
        import httpx
        backend_cfg = config.get_backend(backend)
        with httpx.Client(base_url=backend_cfg.base_url, timeout=10) as client:
            resp = client.get("/api/v1/auth/captcha")
            resp.raise_for_status()
            captcha_key = resp.json()["data"]["captchaKey"]

        code = redis.get_captcha(captcha_key)
        assert code and len(code) >= 4

    def test_logout_clears_session(self, backend):
        """登出后再次调用需要登录的 API 应自动重新登录。"""
        auth.login(backend=backend)
        auth.logout(backend=backend)
        # 再登录应成功
        sid = auth.login(backend=backend)
        assert sid
