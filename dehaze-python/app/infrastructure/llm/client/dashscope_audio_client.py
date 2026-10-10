"""阿里云百炼（DashScope）语音协议适配器（统一模型阶段 2）

语音模型（asr/tts）经统一模型路由解析为 `sys_ai_model` + `sys_ai_provider` 后，
本模块按百炼"指令框架 + 二进制音频帧"实时协议调用引擎：
- 统一 WS 地址：由 `provider.api_base_url` 取 scheme+host 拼 `/api-ws/v1/inference`
  （http→ws / https→wss；api_base_url 已含该路径则原样使用）
- 指令框架（ASR/TTS 共用）：run-task（须等 task-started 后方可发数据）→ 上行数据 →
  finish-task（复用同一 task_id）
- ASR：上行 PCM 16k 二进制帧；下行 result-generated 事件（sentence.text /
  sentence_end / heartbeat），归一化为统一 JSON `{"text", "is_final"}` 供 asr_service 消费
- TTS：continue-task 下发全文，下行二进制音频帧累积，收到 task-finished 返回聚合音频
- transcribe：离线整段识别复用流式会话（百炼无独立离线 ASR 端点），hotwords 暂不传
- status：百炼无引擎状态端点，配置存在即视为 online（字段与内置引擎完全一致）

密钥经统一 `sys_ai_provider_key`（ProviderKeySelector 单实例）逐候选尝试，握手阶段
401/403/429 冷却该 Key（连接层失败不冷却）；Key 密文运行时解密，明文不落日志。

易错点（务必保持）：
1. run-task 后必须等 task-started 事件，之后才能发音频 / continue-task；
2. finish-task 的 task_id 必须与 run-task 同一 uuid；
3. 上行音频与 TTS 下行音频都是**二进制帧**，与 JSON 指令帧共用一个 WS，收发按帧类型
   （isinstance str/bytes）分派。
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from redis.asyncio import Redis
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.crypto.aes_cipher import decrypt
from app.infrastructure.llm.client.openai_audio_client import EngineStreamClosed
from app.infrastructure.llm.common import build_auth_headers
from app.infrastructure.provider.provider_key_selector import provider_key_selector

logger = logging.getLogger(__name__)

# 百炼统一实时语音 WS 路径（由 api_base_url 派生；用于拼接与"是否已含"判断）
_WS_PATH = "/api-ws/v1/inference"
# WS 握手默认超时（供应商未配置 request_timeout_seconds 时使用）
_WS_OPEN_TIMEOUT = 10.0
# Key 级失败需冷却的握手状态码（鉴权失败/限流按 Key 归因）
_KEY_UNAVAILABLE_STATUS = {401, 403, 429}
# 引擎错误码 → 业务码映射（对齐现有语音引擎口径，供上层按码分支；未知码回落 BUSINESS_ERROR）
_CODE_TO_RESULT = {
    "InvalidParameter": ResultCode.PARAM_ERROR,
    "InvalidApiKey": ResultCode.BUSINESS_ERROR,
    "Throttling": ResultCode.BUSINESS_ERROR,
}


def _base_url(provider) -> str:
    if not provider.api_base_url:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR,
            f"供应商 {provider.provider_code} 未配置 API 地址，请检查 sys_ai_provider",
        )
    return provider.api_base_url.rstrip("/")


def _ws_url(provider) -> str:
    """由 api_base_url 派生统一实时语音 WS 地址（http→ws / https→wss）。

    api_base_url 已含 `/api-ws/v1/inference` 时原样使用，否则取 scheme+host 拼该路径。
    """
    parts = urlsplit(_base_url(provider))
    scheme = "wss" if parts.scheme == "https" else "ws"
    path = parts.path.rstrip("/")
    if not path.endswith(_WS_PATH):
        path = _WS_PATH
    return urlunsplit((scheme, parts.netloc, path, "", ""))


def _auth_headers(provider, key: str) -> dict[str, str]:
    """认证头：复用统一组装（bearer → Authorization），并合并 default_headers"""
    return build_auth_headers(provider, key)


def _open_timeout(provider) -> float:
    """WS 握手超时：优先供应商 request_timeout_seconds，缺省 10s"""
    return float(provider.request_timeout_seconds or _WS_OPEN_TIMEOUT)


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


# --- 指令帧构造（纯函数，便于单测断言协议结构）-----------------------------------


def _run_task(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """run-task 指令帧（duplex 双工；task_id 由本次会话生成）"""
    return {
        "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
        "payload": payload,
    }


def _finish_task(task_id: str) -> dict[str, Any]:
    """finish-task 指令帧（task_id 必须与 run-task 相同）"""
    return {
        "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {"input": {}},
    }


def _continue_task(task_id: str, text: str) -> dict[str, Any]:
    """TTS continue-task 指令帧（下发待合成文本）"""
    return {
        "header": {"action": "continue-task", "task_id": task_id},
        "payload": {"input": {"text": text}},
    }


def _asr_payload(model_id: str) -> dict[str, Any]:
    """ASR run-task payload（PCM 16k；关掉顺滑以对齐逐句结果的边界语义）"""
    return {
        "task_group": "audio",
        "task": "asr",
        "function": "recognition",
        "model": model_id,
        "parameters": {
            "format": "pcm",
            "sample_rate": 16000,
            "disfluency_removal_enabled": False,
        },
        "input": {},
    }


def _tts_payload(
    model_id: str,
    voice: str,
    speed: float,
    response_format: str,
    sample_rate: int,
) -> dict[str, Any]:
    """TTS run-task payload（参数经 parameters 透传；格式取值 pcm/wav/mp3）"""
    return {
        "task_group": "audio",
        "task": "tts",
        "function": "SpeechSynthesizer",
        "model": model_id,
        "parameters": {
            "text_type": "PlainText",
            "voice": voice,
            "format": response_format,
            "sample_rate": sample_rate,
            "rate": speed,
        },
        "input": {},
    }


def _raise_task_failed(header: dict[str, Any]) -> None:
    """task-failed 事件 → 业务异常（错误原文只进日志，对外前缀与内置引擎口径一致）"""
    code = header.get("error_code") or "TASK_FAILED"
    message = header.get("error_message") or ""
    logger.error("百炼语音任务失败 code=%s message=%s", code, message)
    raise BusinessException(
        _CODE_TO_RESULT.get(code, ResultCode.BUSINESS_ERROR),
        f"语音引擎服务错误：{message or code}",
    )


def _normalize_asr_event(event: dict[str, Any]) -> str | None:
    """ASR 下行事件 → 统一 JSON 文本；心跳 / 非结果事件返回 None（跳过）。

    归一化形状 `{"text": str, "is_final": bool}` 供 asr_service 直接消费；
    heartbeat=true 的结果不携带新文本，跳过。
    """
    header = event.get("header") or {}
    event_type = header.get("event")
    if event_type == "task-failed":
        _raise_task_failed(header)
    if event_type != "result-generated":
        return None
    sentence = ((event.get("payload") or {}).get("output") or {}).get("sentence") or {}
    if sentence.get("heartbeat"):
        return None
    return json.dumps(
        {"text": sentence.get("text", ""), "is_final": bool(sentence.get("sentence_end"))},
        ensure_ascii=False,
    )


async def _recv_started(ws) -> None:
    """等待 task-started（run-task 必须先起步才能发后续数据）；期间 task-failed 直接抛异常"""
    while True:
        raw = await ws.recv()
        if not isinstance(raw, str):
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_type = (event.get("header") or {}).get("event")
        if event_type == "task-failed":
            _raise_task_failed(event.get("header") or {})
        if event_type == "task-started":
            return


async def _safe_close(ws) -> None:
    try:
        await ws.close()
    except Exception as exc:  # 连接已断或关闭失败不影响编排侧收尾
        logger.debug("关闭百炼语音会话失败: %s", exc)


class AsrStreamSession:
    """百炼流式 ASR 会话（run-task 指令框架 + PCM 二进制上行）"""

    def __init__(self, ws, model_id: str) -> None:
        self._ws = ws
        self._model_id = model_id
        # 同一会话内 run-task / finish-task 复用同一 task_id
        self._task_id = str(uuid.uuid4())

    async def _send_json(self, payload: dict[str, Any]) -> None:
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def start(self) -> None:
        """下发 run-task 并等待 task-started（此前禁止发送音频帧）"""
        await self._send_json(_run_task(self._task_id, _asr_payload(self._model_id)))
        await _recv_started(self._ws)

    async def send_audio(self, chunk: bytes) -> None:
        """推送 PCM 16k 单声道音频块（二进制帧；websockets 高水位天然形成上行反压）"""
        try:
            await self._ws.send(chunk)
        except ConnectionClosed as exc:
            raise EngineStreamClosed("百炼流式会话已断开，音频未送达") from exc

    async def send_eos(self) -> None:
        """结束信号：发送 finish-task 令引擎给出最终结果；失败静默（收尾路径不中断）"""
        try:
            await self._send_json(_finish_task(self._task_id))
        except Exception as exc:  # 连接已断/状态异常都不影响编排侧按已收字节收尾
            logger.info("finish-task 未送达百炼引擎（会话已断开?）: %s", exc)

    async def recv_messages(self) -> AsyncIterator[str]:
        """迭代归一化后的下行结果 JSON；task-finished 或连接关闭即结束，错误帧转业务异常"""
        try:
            async for raw in self._ws:
                if not isinstance(raw, str):
                    # ASR 结果均为 JSON 文本事件；意外二进制帧跳过
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("百炼下行非 JSON 帧: %s", raw[:100])
                    continue
                if (event.get("header") or {}).get("event") == "task-finished":
                    return
                normalized = _normalize_asr_event(event)
                if normalized is not None:
                    yield normalized
        except ConnectionClosed:
            logger.info("百炼流式会话连接关闭")

    async def aclose(self) -> None:
        """关闭引擎侧会话（前端断开/会话收尾时调用），使引擎即时释放会话"""
        await _safe_close(self._ws)


async def open_stream(
    provider, model_id: str, hotwords: list[str], session_id: str | None = None
) -> AsrStreamSession:
    """建立百炼流式 ASR 会话：按 Key 逐候选握手并等 task-started（就绪后才返回）。

    hotwords 暂不传：百炼热词走 `vocabulary_id`（需外部创建词表后在 run-task 参数携带），
    当前未接入词表管理，故忽略入参以保持统一契约签名。
    握手阶段 401/403/429 冷却该 Key 并试下一个候选；连接层失败不冷却。
    """
    redis: Redis = await get_redis_client()
    candidates = await _usable_keys(provider)
    if not candidates:
        raise _no_key_error(provider)

    url = _ws_url(provider)
    open_timeout = _open_timeout(provider)
    rejected_status: int | None = None
    for key in candidates:
        if not await provider_key_selector.reserve_key(redis, key):
            continue
        try:
            ws = await ws_connect(
                url,
                additional_headers=_auth_headers(provider, _key_plaintext(key)),
                open_timeout=open_timeout,
                max_size=None,
            )
        except InvalidStatus as exc:
            rejected_status = exc.response.status_code
            logger.error("百炼语音流式握手失败 status=%s", rejected_status)
            if rejected_status in _KEY_UNAVAILABLE_STATUS:
                await provider_key_selector.mark_call_failed(redis, key.id, str(rejected_status))
            continue
        except Exception as exc:
            logger.error("连接百炼语音流式会话失败 model_id=%s: %s", model_id, exc)
            continue

        session = AsrStreamSession(ws, model_id)
        try:
            await session.start()
        except BusinessException:
            await _safe_close(ws)
            raise
        except Exception as exc:
            logger.error("百炼 ASR run-task 握手失败 model_id=%s: %s", model_id, exc)
            await _safe_close(ws)
            continue
        await provider_key_selector.mark_call_success(redis, key.id)
        return session

    if rejected_status is not None:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR, f"语音引擎流式握手被拒绝: {rejected_status}"
        )
    raise BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "百炼语音服务不可用")


async def _synthesize_once(
    ws,
    model_id: str,
    text: str,
    voice: str,
    speed: float,
    response_format: str,
    sample_rate: int,
) -> bytes:
    """单次 TTS：run-task → 等 task-started → continue-task(全文) → finish-task → 聚合音频帧"""
    task_id = str(uuid.uuid4())
    await ws.send(
        json.dumps(
            _run_task(task_id, _tts_payload(model_id, voice, speed, response_format, sample_rate)),
            ensure_ascii=False,
        )
    )
    await _recv_started(ws)
    await ws.send(json.dumps(_continue_task(task_id, text), ensure_ascii=False))
    await ws.send(json.dumps(_finish_task(task_id), ensure_ascii=False))

    chunks: list[bytes] = []
    while True:
        raw = await ws.recv()
        if isinstance(raw, (bytes, bytearray)):
            chunks.append(bytes(raw))
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        header = event.get("header") or {}
        event_type = header.get("event")
        if event_type == "task-failed":
            _raise_task_failed(header)
        if event_type == "task-finished":
            break
    return b"".join(chunks)


async def synthesize(
    provider,
    model_id: str,
    text: str,
    voice: str,
    speed: float,
    response_format: str,
    sample_rate: int,
) -> bytes:
    """文本合成语音 → 音频字节（WS 指令框架；按 Key 逐候选握手，连接层失败换 Key）"""
    redis: Redis = await get_redis_client()
    candidates = await _usable_keys(provider)
    if not candidates:
        raise _no_key_error(provider)

    url = _ws_url(provider)
    open_timeout = _open_timeout(provider)
    rejected_status: int | None = None
    for key in candidates:
        if not await provider_key_selector.reserve_key(redis, key):
            continue
        try:
            ws = await ws_connect(
                url,
                additional_headers=_auth_headers(provider, _key_plaintext(key)),
                open_timeout=open_timeout,
                max_size=None,
            )
        except InvalidStatus as exc:
            rejected_status = exc.response.status_code
            logger.error("百炼 TTS 握手失败 status=%s", rejected_status)
            if rejected_status in _KEY_UNAVAILABLE_STATUS:
                await provider_key_selector.mark_call_failed(redis, key.id, str(rejected_status))
            continue
        except Exception as exc:
            logger.error("连接百炼 TTS 会话失败 model_id=%s: %s", model_id, exc)
            continue

        try:
            audio = await _synthesize_once(
                ws, model_id, text, voice, speed, response_format, sample_rate
            )
        except BusinessException:
            # 引擎返回的业务错误（如参数非法/模型不存在）对任何 Key 都一样，不再换 Key 重试
            raise
        except Exception as exc:
            logger.error("百炼 TTS 合成失败 model_id=%s: %s", model_id, exc)
            continue
        finally:
            await _safe_close(ws)
        await provider_key_selector.mark_call_success(redis, key.id)
        return audio

    if rejected_status is not None:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR, f"语音引擎流式握手被拒绝: {rejected_status}"
        )
    raise BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "百炼语音合成服务不可用")


async def transcribe(provider, model_id: str, audio: bytes, hotwords: list[str]) -> str:
    """离线识别整段音频 → 文本。

    百炼无独立离线 ASR 端点，故复用流式协议实现：run-task → task-started → 发完整
    音频 → finish-task → 聚合结果（优先取 is_final=true 的最后一条，否则拼接全部）。
    hotwords 暂不传（百炼热词需外部创建 vocabulary_id，见 open_stream 说明）。
    """
    session = await open_stream(provider, model_id, hotwords)
    try:
        await session.send_audio(audio)
        await session.send_eos()
        texts: list[str] = []
        final_text: str | None = None
        async for raw in session.recv_messages():
            msg = json.loads(raw)
            texts.append(msg.get("text", ""))
            if msg.get("is_final"):
                final_text = msg.get("text", "")
        return final_text if final_text is not None else "".join(texts)
    finally:
        await session.aclose()


async def status(provider) -> dict[str, Any]:
    """引擎状态 → 语音服务状态契约字段。

    百炼无引擎状态端点，配置存在即可用（不做探测），字段名与内置引擎完全一致。
    """
    return {
        "engine_status": "online",
        "stream_model_loaded": False,
        "offline_model_loaded": False,
        "voice_model_loaded": False,
    }
