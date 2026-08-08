"""三端 API 响应对比测试。"""
from __future__ import annotations

import httpx

from utils import config, api


class TestCompareBackends:
    def test_captcha_all_backends(self):
        """三端验证码接口都应返回成功 + captchaKey。"""
        for name in config.BACKENDS:
            resp = api.get("/api/v1/auth/captcha", backend=name)
            assert resp["code"] == config.SUCCESS_CODE
            assert resp["data"]["captchaKey"]

    def test_health_all_backends(self):
        """三端健康检查都应返回 HTTP 200。

        三端响应格式不一致（Java: {"status":"UP"}, Go: {"code":"00000"}, Python: {"status":"healthy"}），
        只验证 HTTP 状态码。
        """
        health_paths = {"java": "/actuator/health", "go": "/health", "python": "/health"}
        for name in config.BACKENDS:
            backend_cfg = config.get_backend(name)
            with httpx.Client(base_url=backend_cfg.base_url, timeout=10) as client:
                resp = client.get(health_paths[name])
                assert resp.status_code == 200, f"{name} health check failed: {resp.status_code}"
