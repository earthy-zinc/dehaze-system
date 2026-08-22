"""
pytest 配置和共享 fixtures

基于 FastAPI + pytest-asyncio 的测试框架。

Redis 桩见 tests/stubs.py（fakeredis，真实协议实现）；
router 级测试客户端由各测试文件自建（ai_client 等）。
"""

import os
import sys

import pytest

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置测试环境变量
os.environ["APP_ENV"] = "testing"
os.environ["DEHAZE_PASSWORD"] = "test_password"

from fakeredis import FakeAsyncRedis

from app.main import app as fastapi_app


@pytest.fixture
def mock_redis(monkeypatch: pytest.MonkeyPatch) -> FakeAsyncRedis:
    """fakeredis 客户端（真实协议实现），统一覆盖 redis 获取入口，
    避免测试触达真实 Redis。

    - app.dependencies.redis.get_redis_client/get_redis：覆盖函数内延迟导入路径
      （如 reasoning_service.run）。
    - interrupt_handler.get_redis_client：覆盖模块顶层导入路径（如 send_message 的
      挂起检查调用 interrupt_handler.get_interrupt）。
    """
    redis = FakeAsyncRedis(decode_responses=True)

    async def _override():
        return redis

    monkeypatch.setattr("app.dependencies.redis.get_redis_client", _override)
    monkeypatch.setattr("app.dependencies.redis.get_redis", _override)
    monkeypatch.setattr("app.service.ai.interrupt_handler.get_redis_client", _override)
    monkeypatch.setattr("app.service.ai.async_resume.get_redis_client", _override)
    return redis


@pytest.fixture
def app():
    """FastAPI 应用实例"""
    return fastapi_app
