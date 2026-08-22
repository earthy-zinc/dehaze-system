"""供应商连通性测试服务

按 protocol_type + auth_type + default_headers 组装最小探测请求
（GET 模型列表接口），不携带用户上下文、不产生业务计费。
"""

from __future__ import annotations

import logging
import time

import httpx

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_provider import SysAiProvider
from app.repository.ai_provider_repository import ai_provider_repository
from app.infrastructure.llm.model_client import build_auth_headers
from app.infrastructure.llm.provider_key_selector import provider_key_selector

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 5.0


def _probe_url(provider: SysAiProvider) -> str:
    """按协议类型组装最小探测 URL（模型列表接口）。"""
    base = provider.api_base_url.rstrip("/")
    if provider.protocol_type == "anthropic":
        return f"{base}/v1/models"
    return f"{base}/models"


async def test_connection(
    db,
    redis,
    provider_id: int,
) -> dict:
    """执行供应商连通性测试，返回 {connected, status_code, latency_ms, error}。"""
    provider = await ai_provider_repository.get_by_id(db, provider_id)
    if not provider:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "供应商不存在")

    api_key = await provider_key_selector.select_key(db, redis, provider_id)
    if not api_key:
        raise BusinessException(
            ResultCode.OPERATION_NOT_ALLOW,
            "该供应商没有可用的启用 API Key，无法进行连通性测试",
        )

    url = _probe_url(provider)
    headers = build_auth_headers(provider, api_key)

    result = {"connected": False, "status_code": None, "latency_ms": None, "error": None}
    try:
        async with httpx.AsyncClient(timeout=_CONNECT_TIMEOUT) as client:
            start = time.monotonic()
            resp = await client.get(url, headers=headers)
            result["latency_ms"] = int((time.monotonic() - start) * 1000)
            result["status_code"] = resp.status_code
            result["connected"] = resp.status_code < 400
            if not result["connected"]:
                result["error"] = f"HTTP {resp.status_code}"
    except httpx.TimeoutException:
        result["error"] = "连接超时（5s）"
    except httpx.HTTPError as exc:
        result["error"] = f"请求失败: {type(exc).__name__}"
    except Exception as exc:  # noqa: BLE001 - 连通性测试失败不阻断管理流程
        result["error"] = f"连接失败: {exc}"
        logger.warning("供应商连通性测试异常 provider_id=%s: %s", provider_id, exc)

    return result
