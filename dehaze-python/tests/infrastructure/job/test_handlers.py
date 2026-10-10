"""modelHealthCheck 定时任务回归：推理服务可用性 + GPU 显存阈值告警。

编排侧不再持有 CUDA 上下文，GPU 指标改由 dehaze-modal 采集暴露（设计 §6.10）；
引擎 HTTP 经 respx 接管，DB 用真实测试库，Redis 由 autouse mock_redis 接管。
"""

import httpx
import pytest
import respx

from app.config import settings
from app.infrastructure.job import handlers

pytestmark = pytest.mark.requires_db

_BASE = settings.MODAL_ENGINE_BASE_URL.rstrip("/")
_GPU_METRICS_TEXT = """# TYPE dehaze_gpu_memory_used_bytes gauge
dehaze_gpu_memory_used_bytes{device_id="0",device_name="RTX 4090"} 9.65e10
# TYPE dehaze_gpu_memory_total_bytes gauge
dehaze_gpu_memory_total_bytes{device_id="0",device_name="RTX 4090"} 1.0e11
"""


def _mock_engine(router, *, models: list[dict], metrics_text: str = "") -> None:
    router.get(f"{_BASE}/v1/models").mock(return_value=httpx.Response(200, json={"data": models}))
    router.get(f"{_BASE}/ready").mock(return_value=httpx.Response(200, json={"status": "ready"}))
    router.get(f"{_BASE}/metrics").mock(return_value=httpx.Response(200, text=metrics_text))


async def test_model_health_check_warns_on_gpu_memory_overuse(db):
    """显存使用率 >95% → 告警（阈值与拆分前一致）"""
    with respx.mock(assert_all_mocked=True) as router:
        _mock_engine(
            router,
            models=[{"id": "aecrnet", "available": True}],
            metrics_text=_GPU_METRICS_TEXT,
        )
        msg = await handlers.model_health_check()

    assert "GPU:0 显存使用率过高: 96.5%" in msg


async def test_model_health_check_flags_unavailable_and_missing_gpu(db):
    """不可服务项与无 GPU 指标均进告警"""
    with respx.mock(assert_all_mocked=True) as router:
        _mock_engine(router, models=[{"id": "psnr", "available": False}])
        msg = await handlers.model_health_check()

    assert "不可服务的模型/指标集(1): psnr" in msg
    assert "CUDA 不可用" in msg


async def test_model_health_check_reports_engine_down(db):
    """推理服务不可达 → 检查异常进告警，不影响 DB/Redis 检查继续执行"""
    with respx.mock(assert_all_mocked=True) as router:
        router.get(f"{_BASE}/v1/models").mock(side_effect=httpx.ConnectError("connection refused"))
        msg = await handlers.model_health_check()

    assert "推理服务检查异常" in msg
