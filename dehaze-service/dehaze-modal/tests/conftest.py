"""测试夹具：凭据、图像构造、配置隔离。"""

import io

import PIL.Image
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """缓存目录落临时目录、权重下载源指向本机：测试不读写仓库根 models/，也不发出真实外网请求。"""
    monkeypatch.setattr(settings, "MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(settings, "NGINX_STATIC_HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "NGINX_STATIC_PORT", 9000)


@pytest.fixture
def client():
    """不进入 lifespan：端点测试无需拉起 GPU 采集器。

    raise_server_exceptions=False：未包装异常由全局处理器转为统一错误响应后仍会被
    ServerErrorMiddleware 重抛，交由断言校验响应体而非依赖异常穿透。
    """
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {settings.INTERNAL_SERVICE_KEY}"}


@pytest.fixture
def png_bytes():
    def _png(size=(32, 32), color=(120, 130, 140)) -> bytes:
        buffer = io.BytesIO()
        PIL.Image.new("RGB", size, color).save(buffer, format="PNG")
        return buffer.getvalue()

    return _png
