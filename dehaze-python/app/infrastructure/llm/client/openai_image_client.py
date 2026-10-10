"""OpenAI Images 兼容统一图像适配器（设计 §2.4/§2.5 阶段 3）

图像模型（image_gen/image_edit）经统一模型路由解析为 `sys_ai_model` + `sys_ai_provider`
后，由本模块按供应商协议调用引擎/云厂商：

- `edit`：`POST /v1/images/edits`（multipart：image/model/params）→
  `{created, data:[{b64_json}]}`，解码 b64 返回结果图字节。

密钥经统一 `sys_ai_provider_key`（ProviderKeySelector 单实例：按优先级/权重取候选，
Key 级失败冷却、成功回执），引擎错误码映射为业务异常。
"""

from __future__ import annotations

import base64
import binascii
import json
import logging

import httpx
from redis.asyncio import Redis

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.crypto.aes_cipher import decrypt
from app.infrastructure.provider.provider_key_selector import provider_key_selector

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 60.0
_READY_TIMEOUT = 3.0
# Key 级失败需重试/冷却的引擎状态码（限流/鉴权失败按 Key 归因，5xx 属引擎侧不冷却）
_KEY_UNAVAILABLE_STATUS = {401, 403, 429}
_RETRY_STATUS = _KEY_UNAVAILABLE_STATUS
# 引擎错误码 → 业务码映射
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


def _new_client(timeout: httpx.Timeout | float = _HTTP_TIMEOUT) -> httpx.AsyncClient:
    """HTTP 客户端工厂（测试经它注入 MockTransport）"""
    return httpx.AsyncClient(timeout=timeout)


# 约定：provider.api_base_url 已含 /v1（对齐 OpenAI 厂商口径），下方路径均相对 /v1 拼接
def _base_url(provider) -> str:
    if not provider.api_base_url:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR,
            f"供应商 {provider.provider_code} 未配置 API 地址，请检查 sys_ai_provider",
        )
    return provider.api_base_url.rstrip("/")


def _auth_headers(provider, key: str | None) -> dict[str, str]:
    """按 provider.auth_type 组装认证请求头（custom 时从 default_headers.auth_header 取头名）"""
    headers = dict(provider.default_headers or {})
    header_name = headers.pop("auth_header", None)
    if not key:
        return headers
    auth = provider.auth_type or "bearer"
    if auth == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif auth == "x-api-key":
        headers["X-API-Key"] = key
    elif auth == "custom" and header_name:
        headers[header_name] = key
    return headers


def _key_plaintext(key) -> str:
    """Key 实体 → 明文（AES 密文运行时解密，明文不落日志）"""
    return decrypt(key.key_cipher)


async def _usable_keys(provider) -> list:
    """该供应商当前可用 Key（启用、未过期、未冷却、未超额度），按优先级与权重排序"""
    redis: Redis = await get_redis_client()
    async with get_db_session() as db:
        return await provider_key_selector.list_usable_keys(db, redis, provider.id)


def _no_key_error(provider) -> BusinessException:
    return BusinessException(
        ResultCode.BUSINESS_ERROR,
        f"供应商 {provider.provider_code} 无可用 API Key，请检查 sys_ai_provider_key",
    )


def _raise_engine_error(status_code: int, body: bytes) -> None:
    code, message = "HTTP_ERROR", ""
    try:
        payload = json.loads(body)
        code = payload.get("code", code)
        message = payload.get("message", "") or payload.get("detail", "")
    except (json.JSONDecodeError, AttributeError):
        message = body[:200].decode("utf-8", "replace")
    # 完整原文只进日志：url=/error= 会带上内网地址与 httpx 异常原文，不外发到响应体
    logger.error("图像引擎请求失败 status=%s code=%s message=%s", status_code, code, message)
    cut = [
        i for i in (message.find(", url="), message.find(", error="), message.find("\n")) if i >= 0
    ]
    if cut:
        message = message[: min(cut)].rstrip(" ,")
    fallback = ResultCode.PARAM_ERROR if status_code < 500 else ResultCode.BUSINESS_ERROR
    raise BusinessException(
        _CODE_TO_RESULT.get(code, fallback), f"图像引擎服务错误：{message or code}"
    )


async def _request(
    method: str, path: str, provider, timeout: httpx.Timeout | float = _HTTP_TIMEOUT, **kwargs
) -> httpx.Response:
    """业务调用：按 Key 逐候选尝试，全部失败按引擎错误/服务不可用上报"""
    redis = await get_redis_client()
    candidates = await _usable_keys(provider)
    if not candidates:
        raise _no_key_error(provider)

    url = f"{_base_url(provider)}{path}"
    last_response: httpx.Response | None = None
    for key in candidates:
        if not await provider_key_selector.reserve_key(redis, key):
            continue
        try:
            async with _new_client(timeout) as client:
                resp = await client.request(
                    method,
                    url,
                    headers=_auth_headers(provider, _key_plaintext(key)),
                    **kwargs,
                )
        except httpx.TimeoutException as exc:
            logger.error("调用图像引擎超时 %s %s: %s", method, path, exc)
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR, "图像引擎服务响应超时"
            ) from exc
        except httpx.HTTPError as exc:
            # 连接层失败属引擎侧问题：不冷却该 Key，换下一个候选继续试
            logger.error("调用图像引擎失败 %s %s: %s", method, path, exc)
            last_response = None
            continue
        if resp.status_code < 400:
            await provider_key_selector.mark_call_success(redis, key.id)
            return resp
        if resp.status_code not in _RETRY_STATUS:
            # 参数/业务类错误对任何 Key 都一样，直接按引擎错误上报
            _raise_engine_error(resp.status_code, resp.content)
        last_response = resp
        if resp.status_code in _KEY_UNAVAILABLE_STATUS:
            await provider_key_selector.mark_call_failed(redis, key.id, str(resp.status_code))

    if last_response is not None:
        _raise_engine_error(last_response.status_code, last_response.content)
    raise BusinessException(ResultCode.BUSINESS_ERROR, "图像引擎服务不可用")


async def edit(
    provider,
    model_id: str,
    image: bytes,
    params: dict | None,
    timeout: httpx.Timeout | float = _HTTP_TIMEOUT,
) -> bytes:
    """图像编辑/去雾推理 → 结果图字节（multipart 对齐 OpenAI images/edits，b64 响应解码）"""
    data: dict[str, str] = {"model": model_id}
    if params is not None:
        data["params"] = json.dumps(params, ensure_ascii=False)
    resp = await _request(
        "POST",
        "/images/edits",
        provider,
        timeout=timeout,
        files={"image": ("image.png", image, "image/png")},
        data=data,
    )
    try:
        b64 = resp.json()["data"][0]["b64_json"]
        return base64.b64decode(b64)
    except (KeyError, IndexError, TypeError, ValueError, binascii.Error) as exc:
        logger.error("图像引擎响应缺少 b64_json 数据: %s", exc)
        raise BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "图像引擎响应格式错误") from exc
