"""外置图像推理引擎客户端 engine_client 单元测试。

engine_client 仅承载领域扩展与非模型 API：目录发现（`/v1/models`）、指标计算
（`/v1/metrics`）、健康与显存（`/ready`、`/metrics`）；图像推理已改由统一适配器
`openai_image_client` 调用，本模块不再承担推理。
覆盖：请求形状、引擎错误码 → ResultCode 集中映射、网络与超时失败路径、
能力发现与 GPU 指标解析。HTTP 经 respx 接管，不触达真实引擎服务。
"""

from io import BytesIO

import httpx
import pytest
import respx

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.modal import engine_client

_BASE = settings.MODAL_ENGINE_BASE_URL.rstrip("/")
_METRICS_URL = f"{_BASE}/v1/metrics"


# ── 请求形状 ────────────────────────────────────────────────────────────────


async def test_compute_metrics_posts_names_and_reference():
    """指标请求：下发指标名清单与参考图，返回引擎载荷"""
    with respx.mock(assert_all_mocked=True) as router:
        route = router.post(_METRICS_URL).mock(
            return_value=httpx.Response(
                200, json={"metrics": {"psnr": 30.5}, "failed": ["niqe"], "meta": {"image_size": 4}}
            )
        )
        payload = await engine_client.compute_metrics(
            "pyiqa", ["psnr", "niqe"], BytesIO(b"pred-image"), BytesIO(b"gt-image"), timeout=120
        )

    body = route.calls.last.request.read().decode("utf-8", "replace")
    assert payload == {"metrics": {"psnr": 30.5}, "failed": ["niqe"], "meta": {"image_size": 4}}
    assert '"psnr"' in body
    assert '"niqe"' in body
    assert "pred-image" in body
    assert "gt-image" in body


async def test_compute_metrics_omits_reference_when_absent():
    """无参考图时不带 reference 字段（无参考指标集）"""
    with respx.mock(assert_all_mocked=True) as router:
        route = router.post(_METRICS_URL).mock(
            return_value=httpx.Response(200, json={"metrics": {}, "failed": [], "meta": {}})
        )
        await engine_client.compute_metrics("pyiqa", ["niqe"], BytesIO(b"pred"), None, timeout=120)

    assert "reference" not in route.calls.last.request.read().decode("utf-8", "replace")


async def test_describe_model_keeps_weight_path_segments():
    """模型标识含斜杠：路由为 {model_id:path}，不得把斜杠编码掉"""
    weight_path = "AECR-Net/NH_train.pk"
    with respx.mock(assert_all_mocked=True) as router:
        route = router.get(f"{_BASE}/v1/models/{weight_path}").mock(
            return_value=httpx.Response(
                200, json={"id": weight_path, "available": True, "size_bytes": 1024}
            )
        )
        info = await engine_client.describe_model(weight_path)

    assert info["size_bytes"] == 1024
    assert route.calls.last.request.url.path == f"/v1/models/{weight_path}"


# ── 错误映射 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (400, "PARAM_ERROR", ResultCode.PARAM_ERROR),
        (422, "INVALID_IMAGE", ResultCode.PARAM_ERROR),
        (404, "NOT_FOUND", ResultCode.BUSINESS_ERROR),
        (503, "MODEL_UNAVAILABLE", ResultCode.BUSINESS_ERROR),
        (504, "INFERENCE_TIMEOUT", ResultCode.SYSTEM_EXECUTION_ERROR),
        (500, "INFERENCE_FAILED", ResultCode.SYSTEM_EXECUTION_ERROR),
        (500, "INTERNAL_ERROR", ResultCode.SYSTEM_EXECUTION_ERROR),
        # 未知错误码按 HTTP 状态兜底（编排侧不消费引擎错误码字面量）
        (400, "UNKNOWN_CODE", ResultCode.PARAM_ERROR),
        (500, "UNKNOWN_CODE", ResultCode.SYSTEM_EXECUTION_ERROR),
    ],
)
async def test_engine_error_codes_mapped(status, code, expected):
    """引擎错误码集中映射为业务异常，消息透出引擎可读信息"""
    with respx.mock(assert_all_mocked=True) as router:
        router.post(_METRICS_URL).mock(
            return_value=httpx.Response(status, json={"code": code, "message": "引擎失败详情"})
        )
        with pytest.raises(BusinessException) as ei:
            await engine_client.compute_metrics("pyiqa", ["psnr"], BytesIO(b"x"), None, timeout=120)

    assert ei.value.code == expected
    assert "引擎失败详情" in ei.value.message


async def test_timeout_raises_execution_error():
    """引擎未在超时内响应 → 按执行超时失败（与引擎 504 同口径）"""
    with respx.mock(assert_all_mocked=True) as router:
        router.post(_METRICS_URL).mock(side_effect=httpx.ReadTimeout("read timeout"))
        with pytest.raises(BusinessException) as ei:
            await engine_client.compute_metrics("pyiqa", ["psnr"], BytesIO(b"x"), None, timeout=120)

    assert ei.value.code == ResultCode.SYSTEM_EXECUTION_ERROR
    assert "超时" in ei.value.message


async def test_unreachable_service_raises_business_error():
    """引擎不可达 → 业务异常（不重试，编排侧按失败路径处理）"""
    with respx.mock(assert_all_mocked=True) as router:
        router.post(_METRICS_URL).mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(BusinessException) as ei:
            await engine_client.compute_metrics("pyiqa", ["psnr"], BytesIO(b"x"), None, timeout=120)

    assert ei.value.code == ResultCode.BUSINESS_ERROR
    assert "不可用" in ei.value.message


async def test_ready_false_when_unreachable():
    """就绪探测失败返回 False（探针不抛出）"""
    with respx.mock(assert_all_mocked=True) as router:
        router.get(f"{_BASE}/ready").mock(side_effect=httpx.ConnectError("connection refused"))
        assert await engine_client.ready() is False


# ── GPU 指标解析 ─────────────────────────────────────────────────────────────

_GPU_METRICS_TEXT = """# HELP dehaze_gpu_memory_used_bytes GPU memory used in bytes
# TYPE dehaze_gpu_memory_used_bytes gauge
dehaze_gpu_memory_used_bytes{device_id="0",device_name="RTX 4090"} 9.65e10
# HELP dehaze_gpu_memory_total_bytes GPU total memory in bytes
# TYPE dehaze_gpu_memory_total_bytes gauge
dehaze_gpu_memory_total_bytes{device_id="0",device_name="RTX 4090"} 1.0e11
"""


async def test_gpu_memory_usage_computes_percent_per_device():
    """显存使用率按设备计算（总显存与已用配套解析）"""
    with respx.mock(assert_all_mocked=True) as router:
        router.get(f"{_BASE}/metrics").mock(
            return_value=httpx.Response(200, text=_GPU_METRICS_TEXT)
        )
        usages = await engine_client.gpu_memory_usage()

    assert usages == [("0", pytest.approx(96.5))]


async def test_gpu_memory_usage_empty_without_gpu_series():
    """引擎未上报 GPU 显存指标（无 GPU 环境）→ 空列表"""
    with respx.mock(assert_all_mocked=True) as router:
        router.get(f"{_BASE}/metrics").mock(
            return_value=httpx.Response(
                200, text="# TYPE dehaze_inference_requests_total counter\n"
            )
        )
        assert await engine_client.gpu_memory_usage() == []
