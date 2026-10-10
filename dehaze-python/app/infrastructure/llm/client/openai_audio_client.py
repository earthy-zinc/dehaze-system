"""OpenAI Audio 兼容统一语音适配器（设计 §2.4/§3.1 阶段 2）

语音模型（asr/tts）经统一模型路由解析为 `sys_ai_model` + `sys_ai_provider` 后，
由本模块按供应商协议调用引擎服务：
- `transcribe`：`POST /v1/audio/transcriptions`（multipart：file/model/hotwords）→ {text}
- `synthesize`：`POST /v1/audio/speech`（JSON：model/input/voice，扩展 speed/sample_rate）→ 音频字节
- `open_stream`：`WS /v1/asr/stream?model_id=...`（流式 ASR 为引擎私有扩展，OpenAI 无标准
  流式 ASR 协议；握手头为供应商认证头，首条控制消息下发会话级热词与幂等键，协议不变）
- `status`：`GET /v1/status`（引擎健康/并发，供语音服务状态聚合）

密钥经统一 `sys_ai_provider_key`（ProviderKeySelector 单实例：按优先级/权重取候选，
Key 级失败冷却、成功回执、last_used 缓冲），引擎错误码原样上映射业务异常。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

import httpx
from redis.asyncio import Redis
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

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
# 引擎错误码 → 业务码映射（对齐原语音引擎口径，供上层按码分支）
_CODE_TO_RESULT = {
    "PARAM_ERROR": ResultCode.PARAM_ERROR,
    "MODEL_NOT_FOUND": ResultCode.BUSINESS_ERROR,
    "ENGINE_BUSY": ResultCode.BUSINESS_ERROR,
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


def _ws_url(provider, model_id: str) -> str:
    base = _base_url(provider)
    scheme = "wss" if base.startswith("https://") else "ws"
    return f"{scheme}://{base.split('://', 1)[-1]}/asr/stream?model_id={quote(model_id, safe='')}"


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
    logger.error("语音引擎请求失败 status=%s code=%s message=%s", status_code, code, message)
    cut = [
        i for i in (message.find(", url="), message.find(", error="), message.find("\n")) if i >= 0
    ]
    if cut:
        message = message[: min(cut)].rstrip(" ,")
    fallback = ResultCode.PARAM_ERROR if status_code < 500 else ResultCode.BUSINESS_ERROR
    raise BusinessException(
        _CODE_TO_RESULT.get(code, fallback), f"语音引擎服务错误：{message or code}"
    )


async def _request(
    method: str,
    path: str,
    provider,
    timeout: float = _HTTP_TIMEOUT,
    tolerate_not_found: bool = False,
    **kwargs,
) -> httpx.Response:
    """业务调用：按 Key 逐候选尝试，全部失败按引擎错误/服务不可用上报。

    tolerate_not_found=True 时，404 原样返回（供状态探测识别"供应商未提供该端点"），
    不按引擎错误上抛。
    """
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
        except httpx.HTTPError as exc:
            # 连接层失败属引擎侧问题：不冷却该 Key，换下一个候选继续试
            logger.error("调用语音引擎失败 %s %s: %s", method, path, exc)
            last_response = None
            continue
        if resp.status_code < 400:
            await provider_key_selector.mark_call_success(redis, key.id)
            return resp
        if tolerate_not_found and resp.status_code == 404:
            return resp
        if resp.status_code not in _RETRY_STATUS:
            # 参数/业务类错误对任何 Key 都一样，直接按引擎错误上报
            _raise_engine_error(resp.status_code, resp.content)
        last_response = resp
        if resp.status_code in _KEY_UNAVAILABLE_STATUS:
            await provider_key_selector.mark_call_failed(redis, key.id, str(resp.status_code))

    if last_response is not None:
        _raise_engine_error(last_response.status_code, last_response.content)
    raise BusinessException(ResultCode.BUSINESS_ERROR, "语音引擎服务不可用")


async def transcribe(provider, model_id: str, audio: bytes, hotwords: list[str]) -> str:
    """离线识别整段音频 → 转写文本（multipart 对齐 OpenAI：file/model，hotwords 为顶层扩展）"""
    resp = await _request(
        "POST",
        "/audio/transcriptions",
        provider,
        files={"file": ("audio.wav", audio, "audio/wav")},
        data={"model": model_id, **({"hotwords": ",".join(hotwords)} if hotwords else {})},
    )
    return resp.json().get("text", "")


async def synthesize(
    provider,
    model_id: str,
    text: str,
    voice: str,
    speed: float,
    response_format: str,
    sample_rate: int,
) -> bytes:
    """文本合成语音 → 音频字节（JSON 对齐 OpenAI：model/input/voice，扩展字段放顶层）"""
    resp = await _request(
        "POST",
        "/audio/speech",
        provider,
        json={
            "model": model_id,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": response_format,
            "sample_rate": sample_rate,
        },
    )
    return resp.content


async def status(provider) -> dict[str, Any]:
    """引擎状态 → 语音服务状态契约字段（探测成功即 online，模型加载态按引擎上报）

    引擎返回 {asr:{loaded,model_id,device}, tts:{loaded,voice_id,device}}，与状态接口的
    engineStatus/streamModelLoaded/offlineModelLoaded/voiceModelLoaded 不同构，故在此映射。
    """
    # 状态探测独立短超时：引擎挂起时监控接口最多阻塞 3s，不跟随业务调用的 60s
    resp = await _request(
        "GET", "/status", provider, timeout=_READY_TIMEOUT, tolerate_not_found=True
    )
    if resp.status_code == 404:
        # 供应商未提供 /status 端点（非内置引擎的 OpenAI 兼容语音服务，无法探测加载态）：
        # 按在线 + 加载态未知容错上报，不因"无探针"把可用引擎误报为 offline
        return {
            "engine_status": "online",
            "stream_model_loaded": False,
            "offline_model_loaded": False,
            "voice_model_loaded": False,
        }
    payload = resp.json()
    asr = payload.get("asr") or {}
    tts = payload.get("tts") or {}
    return {
        "engine_status": "online",
        # 流式与离线 ASR 共用引擎侧同一个已加载模型，故两者同源自 asr.loaded
        "stream_model_loaded": bool(asr.get("loaded")),
        "offline_model_loaded": bool(asr.get("loaded")),
        "voice_model_loaded": bool(tts.get("loaded")),
    }


class EngineStreamClosed(Exception):
    """引擎侧流式会话断开（音频未送达）"""


class AudioStreamSession:
    """引擎侧流式识别会话（两跳的第二跳）"""

    def __init__(self, ws) -> None:
        self._ws = ws

    async def send_audio(self, chunk: bytes) -> None:
        """推送 PCM 音频块：websockets 在高水位挂起，天然形成上行反压"""
        try:
            await self._ws.send(chunk)
        except ConnectionClosed as exc:
            raise EngineStreamClosed("引擎侧流式会话已断开，音频未送达") from exc

    async def send_eos(self) -> None:
        """结束信号：引擎转写剩余音频并给出最终结果；发送失败静默（收尾路径不因 EOS 失败而中断）"""
        try:
            await self._ws.send("EOS")
        except Exception as exc:  # 连接已断/状态异常都不影响编排侧按已收字节收尾
            logger.info("EOS 未送达引擎（会话已断开?）: %s", exc)

    async def recv_messages(self) -> AsyncIterator[str]:
        """迭代引擎下行结果 JSON；连接关闭即结束，错误帧转为业务异常"""
        try:
            async for raw in self._ws:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
                payload = json.loads(text)
                if "code" in payload:
                    _raise_engine_error(0, text.encode("utf-8"))
                yield text
        except ConnectionClosed:
            logger.info("引擎侧流式会话连接关闭")

    async def aclose(self) -> None:
        """关闭引擎侧会话（前端断开/会话收尾时调用），使引擎即时释放会话与幂等键"""
        try:
            await self._ws.close()
        except Exception as exc:  # 连接已断或关闭失败不影响编排侧收尾
            logger.debug("关闭引擎侧流式会话失败: %s", exc)


async def open_stream(
    provider, model_id: str, hotwords: list[str], session_id: str | None = None
) -> AudioStreamSession:
    """建立引擎侧流式会话：握手带供应商认证头，首条控制消息下发热词与幂等键。

    握手/控制消息阶段按 Key 逐候选重试（Key 级失败冷却该 Key），握手成功即返回会话。
    """
    redis: Redis = await get_redis_client()
    candidates = await _usable_keys(provider)
    if not candidates:
        raise _no_key_error(provider)

    config: dict = {"type": "config", "hotwords": hotwords}
    if session_id:
        config["session_id"] = session_id
    payload = json.dumps(config, ensure_ascii=False)

    rejected_status: int | None = None
    for key in candidates:
        if not await provider_key_selector.reserve_key(redis, key):
            continue
        auth = _auth_headers(provider, _key_plaintext(key))
        try:
            ws = await ws_connect(
                _ws_url(provider, model_id),
                additional_headers=auth,
                open_timeout=10,
                max_size=None,
            )
        except InvalidStatus as exc:
            rejected_status = exc.response.status_code
            logger.error("语音引擎流式握手失败 status=%s", rejected_status)
            if rejected_status in _KEY_UNAVAILABLE_STATUS:
                await provider_key_selector.mark_call_failed(redis, key.id, str(rejected_status))
            continue
        except Exception as exc:
            logger.error("连接语音引擎流式会话失败 model_id=%s: %s", model_id, exc)
            continue
        try:
            await ws.send(payload)
        except Exception as exc:
            logger.error("下发流式控制消息失败 model_id=%s: %s", model_id, exc)
            await ws.close()
            continue
        await provider_key_selector.mark_call_success(redis, key.id)
        return AudioStreamSession(ws)

    if rejected_status is not None:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR, f"语音引擎流式握手被拒绝: {rejected_status}"
        )
    raise BusinessException(ResultCode.BUSINESS_ERROR, "语音引擎流式会话不可用")
