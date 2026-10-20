"""外置图像推理引擎客户端（clarity-service/clarity-modal，默认 http://127.0.0.1:8994）。

仅承载**领域扩展与非模型 API**：目录发现（`/v1/models`）、指标计算（`/v1/metrics`）、
健康与显存（`/ready`、`/metrics`）。图像模型推理（`/v1/images/edits`）已改由统一适配器
`app.infrastructure.llm.client.openai_image_client` 调用（设计 §2.4/§2.5），本模块不再承担推理。
引擎返回统一错误模型 {"code","message"}，本模块集中映射为 ResultCode——引擎错误码不透传到
编排层，调用方只处理业务异常。
"""

from __future__ import annotations

import json
import logging
from io import BytesIO
from urllib.parse import quote

import httpx
from prometheus_client.parser import text_string_to_metric_families

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException

logger = logging.getLogger(__name__)

# 能力发现与探测：超时短、失败即视为不可用（不重试）
_DISCOVERY_TIMEOUT = 15.0
_READY_TIMEOUT = 2.0

# 引擎 code → 编排侧 ResultCode（集中映射，服务层只处理业务异常）
_CODE_TO_RESULT = {
    "PARAM_ERROR": ResultCode.PARAM_ERROR,
    "INVALID_IMAGE": ResultCode.PARAM_ERROR,
    "UNAUTHORIZED": ResultCode.SYSTEM_EXECUTION_ERROR,
    "NOT_FOUND": ResultCode.BUSINESS_ERROR,
    "MODEL_UNAVAILABLE": ResultCode.BUSINESS_ERROR,
    "SERVICE_UNAVAILABLE": ResultCode.BUSINESS_ERROR,
    "INFERENCE_TIMEOUT": ResultCode.SYSTEM_EXECUTION_ERROR,
    "INFERENCE_FAILED": ResultCode.SYSTEM_EXECUTION_ERROR,
    "INTERNAL_ERROR": ResultCode.SYSTEM_EXECUTION_ERROR,
}


def _new_client(timeout: httpx.Timeout | float) -> httpx.AsyncClient:
    """HTTP 客户端工厂（测试经它注入 MockTransport）"""
    return httpx.AsyncClient(timeout=timeout)


def _base_url() -> str:
    return settings.MODAL_ENGINE_BASE_URL.rstrip("/")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.INTERNAL_SERVICE_KEY}"}


def _raise_engine_error(status_code: int, body: bytes) -> None:
    code, message = "HTTP_ERROR", ""
    try:
        payload = json.loads(body)
        code = payload.get("code", code)
        message = payload.get("message", "")
    except (json.JSONDecodeError, AttributeError):
        message = body[:200].decode("utf-8", "replace")
    logger.warning("图像推理引擎请求失败 status=%s code=%s message=%s", status_code, code, message)
    fallback = ResultCode.PARAM_ERROR if status_code < 500 else ResultCode.SYSTEM_EXECUTION_ERROR
    raise BusinessException(
        _CODE_TO_RESULT.get(code, fallback), f"图像推理服务错误：{message or code}"
    )


async def _request(
    method: str, path: str, timeout: httpx.Timeout | float, **kwargs
) -> httpx.Response:
    """向引擎发起一次 HTTP 调用；网络异常、超时与错误响应都转成业务异常"""
    try:
        async with _new_client(timeout) as client:
            resp = await client.request(
                method, f"{_base_url()}{path}", headers=_headers(), **kwargs
            )
    except httpx.TimeoutException as exc:
        # 超时与「引擎返回 504」同口径：都按执行超时失败（不重试，终态由状态机裁决）
        logger.error("调用图像推理服务超时 %s %s: %s", method, path, exc)
        raise BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "图像推理服务响应超时") from exc
    except httpx.HTTPError as exc:
        logger.error("调用图像推理服务失败 %s %s: %s", method, path, exc)
        raise BusinessException(ResultCode.BUSINESS_ERROR, "图像推理服务不可用") from exc
    if resp.status_code >= 400:
        _raise_engine_error(resp.status_code, resp.content)
    return resp


async def ready(timeout: float = _READY_TIMEOUT) -> bool:
    """只读探测引擎就绪（设备与权重来源可用即 200；不触发模型加载）"""
    try:
        async with _new_client(timeout) as client:
            resp = await client.get(f"{_base_url()}/ready", headers=_headers())
        return resp.status_code == 200
    except httpx.HTTPError as exc:
        logger.warning("图像推理服务 /ready 探测失败: %s", exc)
        return False


async def describe_models() -> list[dict]:
    """能力发现：算法与指标集清单（形状见设计 §3.6）"""
    resp = await _request("GET", "/v1/models", timeout=_DISCOVERY_TIMEOUT)
    return resp.json()["data"]


async def describe_model(model_id: str) -> dict:
    """单项详情：算法标识，或权重相对路径（返回可用性与字节数）"""
    resp = await _request(
        "GET", f"/v1/models/{quote(model_id, safe='/')}", timeout=_DISCOVERY_TIMEOUT
    )
    return resp.json()


async def compute_metrics(
    model_id: str,
    names: list[str],
    image: BytesIO,
    reference: BytesIO | None,
    timeout: httpx.Timeout | float,
) -> dict:
    """计算图像质量指标：返回 {metrics, failed, meta}，单项失败进 failed 不整体失败"""
    image.seek(0)
    files = {"image": ("image.png", image, "image/png")}
    if reference is not None:
        reference.seek(0)
        files["reference"] = ("reference.png", reference, "image/png")
    resp = await _request(
        "POST",
        "/v1/metrics",
        timeout=timeout,
        files=files,
        data={"model": model_id, "metrics": json.dumps(names, ensure_ascii=False)},
    )
    return resp.json()


async def gpu_memory_usage() -> list[tuple[str, float]]:
    """各 GPU 设备的显存使用率（%）：由引擎 /metrics 的 dehaze_gpu_memory_* 指标推导。

    返回空列表表示引擎未上报任何 GPU 显存指标（无 GPU 环境）。
    """
    text = (await _request("GET", "/metrics", timeout=_DISCOVERY_TIMEOUT)).text
    used: dict[str, float] = {}
    total: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        target = {
            "dehaze_gpu_memory_used_bytes": used,
            "dehaze_gpu_memory_total_bytes": total,
        }.get(family.name)
        if target is None:
            continue
        for sample in family.samples:
            target[sample.labels.get("device_id", "0")] = sample.value
    return [
        (device_id, used[device_id] / total_bytes * 100)
        for device_id, total_bytes in total.items()
        if total_bytes > 0 and device_id in used
    ]
