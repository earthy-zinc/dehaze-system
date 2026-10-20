"""火山引擎（豆包大模型语音）协议适配器。

火山 WS 协议与本地/OpenAI 兼容实现不同：传输的是「4 字节位域头 + 可选 sequence +
payload_size + (gzip) payload」的二进制帧，本模块把它翻译为统一契约
（audio_dispatch.VoiceAudioClient），调用方不感知协议差异：

- transcribe：复用流式会话（整段音频一次性下发 + EOS），聚合最终文本
- synthesize：TTS ws_binary 单请求帧 → 聚合音频帧为裸字节
- open_stream：流式 ASR 双向会话（send_audio / send_eos / recv_messages / aclose）
- status：固定上报在线（火山为托管云服务，编排侧无本地模型加载态可探测）

鉴权两套口径，均由 `provider.default_headers.auth_header` 驱动，无 provider_code 分支：
- **经典口径**（未配 auth_header）：ASR 头 `X-Api-App-Key`(APP ID，必填) + `X-Api-Access-Key`
  (Key 明文) + `X-Api-Resource-Id`；TTS 头 `Authorization: Bearer; {key}`（**分号分隔**，
  不是标准 Bearer 空格），走旧版 v1 端点、报文含 `app{appid,token,cluster}`。
- **单密钥口径**（新版控制台，配 `auth_header`，如 `X-Api-Key`）：Key 明文写入该头名；
  ASR 的 APP ID 变可选；TTS 走 v3 单向流式端点，另需 `X-Api-Resource-Id` 与 `X-Api-Request-Id`，
  报文只含 `req_params`。

配置项取自 `provider.default_headers`（缺失即报错并提示配置）：`asr_resource_id` /
`tts_resource_id` 按能力分列（同一供应商行服务 asr/tts 两条链路、资源 id 取值不同）；
`X-Api-App-Key`(APP ID) 与 `cluster`(经典 TTS 集群名) 仅经典口径使用。
密钥经统一 sys_ai_provider_key（ProviderKeySelector）选取、解密与失败冷却，明文不落日志。
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
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
from app.infrastructure.provider.provider_key_selector import provider_key_selector

logger = logging.getLogger(__name__)

# 端点路径（host 取自 provider.api_base_url，忽略其自带路径）
_ASR_PATH = "/api/v3/sauc/bigmodel"
_TTS_V1_PATH = "/api/v1/tts/ws_binary"  # 经典口径：旧版 ws_binary 单请求帧
_TTS_V3_PATH = "/api/v3/tts/unidirectional/stream"  # 单密钥口径：v3 单向流式

# WS 建连超时固定 10s；会话/合成整体超时取 provider.request_timeout_seconds，缺省 60s
_WS_OPEN_TIMEOUT = 10.0
_DEFAULT_TIMEOUT = 60.0

# ASR 协议固定的模型名（model_id 已在平台侧路由到本供应商，请求体 model_name 仅作声明）
_DEFAULT_ASR_MODEL = "bigmodel"
# TTS 集群名缺省值（default_headers.cluster 未配置时使用）
_DEFAULT_TTS_CLUSTER = "volcano_tts"

# ASR/TTS 结果成功码（火山统一口径，非 1000 即视为错误）
_SUCCESS_CODE = 1000

# Key 级失败需冷却的握手状态码（限流/鉴权失败按 Key 归因）
_KEY_UNAVAILABLE_STATUS = {401, 403, 429}

# ===== 二进制帧位域常量（官方 4 字节头）=====
_PROTOCOL_VERSION = 0b0001  # byte0 高 4 位：协议版本 1
_HEADER_SIZE_UNIT = 0b0001  # byte0 低 4 位：头长度单位数（1 × 4 字节 = 4 字节）

MSG_FULL_CLIENT_REQUEST = 0b0001  # byte1 高 4 位：携带 JSON 控制信息的请求帧
MSG_AUDIO_ONLY_REQUEST = 0b0010  # 纯音频请求帧
MSG_FULL_SERVER_RESPONSE = 0b1001  # 带 JSON 结果的响应帧
MSG_AUDIO_ONLY_SERVER_RESPONSE = 0b1011  # 纯音频响应帧（TTS）
MSG_ERROR = 0b1111  # 错误帧

FLAG_NO_SEQUENCE = 0b0000  # byte1 低 4 位：不带 sequence
FLAG_POSITIVE_SEQUENCE = 0b0001  # 带正 sequence（帧体含 4 字节序号）
FLAG_LAST_NO_SEQUENCE = 0b0010  # 最后一包，不带 sequence
FLAG_NEGATIVE_SEQUENCE = 0b0011  # 最后一包，带负 sequence（结束标志）

SERIALIZATION_NONE = 0b0000  # byte2 高 4 位：无序列化（音频帧）
SERIALIZATION_JSON = 0b0001  # JSON 序列化

COMPRESSION_NONE = 0b0000  # byte2 低 4 位：不压缩
COMPRESSION_GZIP = 0b0001  # gzip 压缩

# ===== v3 单向流式 TTS 事件码（帧内 event 字段；帧布局与经典帧同构，语义按 event 分发）=====
FLAG_WITH_EVENT = 0b0100  # byte1 低 4 位：帧体含 event(int32)

EVENT_CONNECTION_FAILED = 51  # 建连失败
EVENT_SESSION_FINISHED = 152  # 会话结束（正常终帧）
EVENT_SESSION_FAILED = 153  # 会话失败
EVENT_TTS_RESPONSE = 352  # 音频帧（message_type=AudioOnlyServer）


@dataclass(frozen=True)
class VolcFrame:
    """解码后的火山二进制帧（header 已剥离，payload 已按压缩位解压）"""

    message_type: int
    flags: int
    serialization: int
    compression: int
    sequence: int | None
    payload: bytes
    header_size: int


def encode_frame(
    message_type: int,
    flags: int,
    payload: bytes = b"",
    *,
    sequence: int | None = None,
    serialization: int = SERIALIZATION_NONE,
    compression: int = COMPRESSION_GZIP,
) -> bytes:
    """构造火山二进制帧（客户端 → 服务端）。

    字节布局：header(4B) | [sequence(4B, int32 大端)] | payload_size(4B, uint32 大端) | payload
    - 4 字节头：byte0=版本|头长度、byte1=消息类型|flags、byte2=序列化|压缩、byte3=保留 0
    - sequence 仅在 flags 含 0b0001（正/负序号）时写入；末帧序号取负值
    - payload 按 compression 位应用 gzip（默认 gzip，与协议 JSON/音频帧一致）

    易错点：sequence 为**有符号** int32（末帧为负数）；payload_size 为**无符号** uint32；均大端序。
    """
    header = bytes(
        [
            (_PROTOCOL_VERSION << 4) | _HEADER_SIZE_UNIT,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0,
        ]
    )
    body = b""
    if flags & FLAG_POSITIVE_SEQUENCE:
        body += int(sequence or 0).to_bytes(4, "big", signed=True)
    if compression == COMPRESSION_GZIP:
        payload = gzip.compress(payload)
    body += len(payload).to_bytes(4, "big", signed=False)
    return header + body + payload


def _gunzip(data: bytes) -> bytes:
    """按压缩位解压：非 gzip 魔数时原样返回。

    服务端部分音频帧压缩位标了 gzip 但实际下发裸字节，硬解压会抛 BadGzipFile，
    故先校验 gzip 魔数（1f 8b）再解压，避免误伤正常音频。
    """
    if len(data) < 2 or data[:2] != b"\x1f\x8b":
        return data
    return gzip.decompress(data)


def decode_frame(raw: bytes) -> VolcFrame:
    """解析火山二进制帧（服务端 → 客户端），与 encode_frame 对称。

    header 长度取 byte0 低 4 位 × 4（本协议恒为 4 字节，兼容带扩展头的情况）。
    """
    if len(raw) < 4:
        raise ValueError("火山帧长度不足 4 字节")
    header_size = (raw[0] & 0x0F) * 4
    message_type = raw[1] >> 4
    flags = raw[1] & 0x0F
    serialization = raw[2] >> 4
    compression = raw[2] & 0x0F

    pos = header_size
    sequence: int | None = None
    if flags & FLAG_POSITIVE_SEQUENCE:
        if len(raw) < pos + 4:
            raise ValueError("火山帧缺少 sequence 字段")
        sequence = int.from_bytes(raw[pos : pos + 4], "big", signed=True)
        pos += 4
    if len(raw) < pos + 4:
        raise ValueError("火山帧缺少 payload_size 字段")
    payload_size = int.from_bytes(raw[pos : pos + 4], "big", signed=False)
    pos += 4
    payload = raw[pos : pos + payload_size]
    if compression == COMPRESSION_GZIP:
        payload = _gunzip(payload)
    return VolcFrame(
        message_type=message_type,
        flags=flags,
        serialization=serialization,
        compression=compression,
        sequence=sequence,
        payload=payload,
        header_size=header_size,
    )


def parse_error_payload(payload: bytes) -> tuple[int, str]:
    """错误帧净载荷：int32 大端错误码 + uint32 大端消息长度 + UTF-8 消息"""
    if len(payload) < 8:
        return -1, payload.decode("utf-8", "replace")
    code = int.from_bytes(payload[:4], "big", signed=True)
    size = int.from_bytes(payload[4:8], "big", signed=False)
    message = payload[8 : 8 + size].decode("utf-8", "replace") if size else ""
    return code, message


def _raise_volc_error(code: int, message: str) -> None:
    """火山错误 → 业务异常（错误原文只进日志，不外发到响应体）"""
    logger.error("火山语音引擎返回错误 code=%s message=%s", code, message)
    raise BusinessException(ResultCode.BUSINESS_ERROR, f"语音引擎服务错误：{message or code}")


def normalize_asr_payload(payload: bytes) -> str:
    """把火山 ASR 结果帧（full server response）归一化为统一 JSON 文本。

    结果帧形如：
    {"code":1000,"result":{"text":"...","utterances":[{"text":"...","definite":true}]}}
    - result.text 为空时用 utterances 逐条拼接
    - is_final 取最后一条 utterance 的 definite（服务端未给 utterances 时恒为 False）
    - code != 1000 视为错误
    """
    obj = json.loads(payload.decode("utf-8"))
    code = obj.get("code")
    if code is not None and str(code) != str(_SUCCESS_CODE):
        message = obj.get("message") or obj.get("msg") or str(code)
        logger.error("火山 ASR 结果错误 code=%s message=%s", code, message)
        raise BusinessException(ResultCode.BUSINESS_ERROR, f"语音引擎服务错误：{message}")
    result = obj.get("result") or {}
    text = result.get("text") or ""
    utterances = result.get("utterances") or []
    if not text and utterances:
        text = "".join(u.get("text", "") for u in utterances)
    is_final = bool(utterances[-1].get("definite")) if utterances else False
    return json.dumps({"text": text, "is_final": is_final}, ensure_ascii=False)


# ===== Provider / Key 公共辅助（对齐 openai_audio_client 模式）=====


def _base_url(provider) -> str:
    if not provider.api_base_url:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR,
            f"供应商 {provider.provider_code} 未配置 API 地址，请检查 sys_ai_provider",
        )
    return provider.api_base_url.rstrip("/")


def _ws_url(provider, path: str) -> str:
    """取 api_base_url 的 scheme+host（忽略其自带路径），拼火山固定端点路径"""
    parts = urlsplit(_base_url(provider))
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


def _default_headers(provider) -> dict[str, str]:
    return {str(k): str(v) for k, v in dict(provider.default_headers or {}).items()}


def _require_header(provider, name: str) -> str:
    """从 default_headers 读取供应商特有配置（appid/resource-id 等），缺失即报错"""
    value = _default_headers(provider).get(name)
    if not value:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR,
            f"供应商 {provider.provider_code} 缺少 default_headers.{name} 配置，"
            "请检查 sys_ai_provider",
        )
    return value


def _key_auth_header(provider) -> str | None:
    """Key 明文承载的头名（default_headers.auth_header，如新版控制台单密钥 `X-Api-Key`）。

    未配置返回 None，由各链路走火山经典口径（ASR X-Api-Access-Key / TTS Authorization: Bearer;）。
    """
    return _default_headers(provider).get("auth_header")


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


def _timeout(provider) -> float:
    return float(provider.request_timeout_seconds or _DEFAULT_TIMEOUT)


def _uid(session_id: str | None = None) -> str:
    """火山 user.uid：无强语义的用户标识，用于服务端配额/日志归因"""
    return session_id or uuid.uuid4().hex


# ===== 流式 ASR =====


def _asr_headers(provider, key: str) -> dict[str, str]:
    """ASR 鉴权头：resource-id 取自 default_headers；Key 明文按鉴权口径落位。

    经典口径：APP ID(X-Api-App-Key) 必填、Key 写 X-Api-Access-Key；
    单密钥口径（配置了 default_headers.auth_header，如新版控制台 X-Api-Key）：APP ID 可选，
    Key 写该头名。
    """
    headers = {"X-Api-Resource-Id": _require_header(provider, "asr_resource_id")}
    auth_header = _key_auth_header(provider)
    if auth_header:
        headers[auth_header] = key
        app_id = _default_headers(provider).get("X-Api-App-Key")
        if app_id:
            headers["X-Api-App-Key"] = app_id
        return headers
    headers["X-Api-App-Key"] = _require_header(provider, "X-Api-App-Key")
    headers["X-Api-Access-Key"] = key
    return headers


def _asr_first_frame(model_id: str, uid: str) -> bytes:
    """ASR 首帧（full client request，gzip JSON）。

    首帧选用 flags=0b0000（不带 sequence）：协议允许，且音频帧使用独立计数从 1 递增，
    避免首帧序号与音频序号共用计数产生歧义。
    """
    request = {
        "user": {"uid": uid},
        "audio": {"format": "pcm", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1},
        "request": {
            "model_name": model_id or _DEFAULT_ASR_MODEL,
            "enable_itn": True,
            "enable_punc": True,
            "show_utterances": True,
        },
    }
    payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
    return encode_frame(
        MSG_FULL_CLIENT_REQUEST,
        FLAG_NO_SEQUENCE,
        payload,
        serialization=SERIALIZATION_JSON,
        compression=COMPRESSION_GZIP,
    )


class AsrStreamSession:
    """火山流式识别会话：把统一 send_audio/send_eos 翻译为二进制音频帧"""

    def __init__(self, ws) -> None:
        self._ws = ws
        self._seq = 0

    async def send_audio(self, chunk: bytes) -> None:
        """推送 PCM 音频块：每块一帧，正 sequence 从 1 递增，payload 为 gzip(音频块)"""
        self._seq += 1
        frame = encode_frame(
            MSG_AUDIO_ONLY_REQUEST,
            FLAG_POSITIVE_SEQUENCE,
            chunk,
            sequence=self._seq,
            compression=COMPRESSION_GZIP,
        )
        try:
            await self._ws.send(frame)
        except ConnectionClosed as exc:
            raise EngineStreamClosed("引擎侧流式会话已断开，音频未送达") from exc

    async def send_eos(self) -> None:
        """结束信号：末帧 flags=0b0011 + 负 sequence（取最后音频序号取负），payload 空"""
        end_seq = -(self._seq if self._seq > 0 else 1)
        frame = encode_frame(
            MSG_AUDIO_ONLY_REQUEST,
            FLAG_NEGATIVE_SEQUENCE,
            b"",
            sequence=end_seq,
            compression=COMPRESSION_GZIP,
        )
        try:
            await self._ws.send(frame)
        except Exception as exc:  # 连接已断/状态异常都不影响编排侧按已收字节收尾
            logger.info("火山 EOS 未送达引擎（会话已断开?）: %s", exc)

    async def recv_messages(self) -> AsyncIterator[str]:
        """迭代火山下行帧，归一化为统一 JSON 文本；负 sequence 即服务端结束"""
        try:
            async for raw in self._ws:
                data = raw if isinstance(raw, bytes) else raw.encode("utf-8")
                frame = decode_frame(data)
                if frame.message_type == MSG_ERROR:
                    code, message = parse_error_payload(frame.payload)
                    _raise_volc_error(code, message)
                if frame.message_type == MSG_FULL_SERVER_RESPONSE:
                    try:
                        yield normalize_asr_payload(frame.payload)
                    except json.JSONDecodeError:
                        logger.warning("火山 ASR 结果帧非 JSON: %r", frame.payload[:100])
                if frame.sequence is not None and frame.sequence < 0:
                    logger.info("火山流式会话服务端结束")
                    break
        except ConnectionClosed:
            logger.info("火山流式会话连接关闭")

    async def aclose(self) -> None:
        """关闭引擎侧会话（前端断开/会话收尾时调用）"""
        try:
            await self._ws.close()
        except Exception as exc:  # 连接已断或关闭失败不影响编排侧收尾
            logger.debug("关闭火山流式会话失败: %s", exc)


async def open_stream(
    provider, model_id: str, hotwords: list[str], session_id: str | None = None
) -> AsrStreamSession:
    """建立火山流式 ASR 会话：握手按 Key 逐候选重试，成功后下发首帧。

    hotwords 暂不支持：火山热词需在控制台预置热词表（boosting 词表），非请求参数，
    故仅记录不随帧下发（与协议能力一致）。
    """
    redis: Redis = await get_redis_client()
    candidates = await _usable_keys(provider)
    if not candidates:
        raise _no_key_error(provider)
    if hotwords:
        logger.info("火山流式 ASR 不支持请求级热词（需控制台热词表），已忽略 %d 个", len(hotwords))

    url = _ws_url(provider, _ASR_PATH)
    first_frame = _asr_first_frame(model_id, _uid(session_id))

    rejected_status: int | None = None
    for key in candidates:
        if not await provider_key_selector.reserve_key(redis, key):
            continue
        headers = _asr_headers(provider, _key_plaintext(key))
        try:
            ws = await ws_connect(
                url,
                additional_headers=headers,
                open_timeout=_WS_OPEN_TIMEOUT,
                max_size=None,
            )
        except InvalidStatus as exc:
            rejected_status = exc.response.status_code
            logger.error("火山流式握手失败 status=%s", rejected_status)
            if rejected_status in _KEY_UNAVAILABLE_STATUS:
                await provider_key_selector.mark_call_failed(redis, key.id, str(rejected_status))
            continue
        except Exception as exc:
            logger.error("连接火山流式会话失败 model_id=%s: %s", model_id, exc)
            continue
        try:
            await ws.send(first_frame)
        except Exception as exc:
            logger.error("下发火山首帧失败 model_id=%s: %s", model_id, exc)
            await _safe_close(ws)
            continue
        await provider_key_selector.mark_call_success(redis, key.id)
        return AsrStreamSession(ws)

    if rejected_status is not None:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR, f"语音引擎流式握手被拒绝: {rejected_status}"
        )
    raise BusinessException(ResultCode.BUSINESS_ERROR, "语音引擎流式会话不可用")


async def transcribe(provider, model_id: str, audio: bytes, hotwords: list[str]) -> str:
    """离线识别：复用流式会话（整段音频一次性下发 + EOS），聚合最终文本返回。

    hotwords 暂不支持（火山热词走控制台热词表，非请求参数）。
    """
    session = await open_stream(provider, model_id, hotwords)
    try:
        await session.send_audio(audio)
        await session.send_eos()
        text = ""
        async for raw in session.recv_messages():
            message = json.loads(raw)
            if message.get("text"):
                text = message["text"]
        return text
    finally:
        await session.aclose()


# ===== TTS =====


def _tts_headers(provider, key: str) -> dict[str, str]:
    """TTS 鉴权头。

    经典口径：`Authorization: Bearer; {key}`（**分号分隔**，不是标准 Bearer 空格）；
    单密钥口径（配置了 default_headers.auth_header）：Key 写该头名，并补 v3 端点必需的
    `X-Api-Resource-Id`（default_headers.tts_resource_id）与 `X-Api-Request-Id`。
    """
    auth_header = _key_auth_header(provider)
    if auth_header:
        return {
            auth_header: key,
            "X-Api-Resource-Id": _require_header(provider, "tts_resource_id"),
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
    return {"Authorization": f"Bearer; {key}"}


def _tts_v1_request(
    provider, key: str, text: str, voice: str, speed: float, response_format: str, sample_rate: int
) -> bytes:
    """经典口径 TTS 请求帧（v1 ws_binary：full client request + 正 sequence，gzip JSON）"""
    appid = _require_header(provider, "X-Api-App-Key")
    cluster = _default_headers(provider).get("cluster") or _DEFAULT_TTS_CLUSTER
    request = {
        "app": {"appid": appid, "token": key, "cluster": cluster},
        "user": {"uid": _uid()},
        "audio": {
            "voice_type": voice,
            "encoding": response_format,
            "speed_ratio": speed,
            "sample_rate": sample_rate,
        },
        "request": {
            "reqid": uuid.uuid4().hex,
            "text": text,
            "text_type": "plain",
            "operation": "query",
        },
    }
    payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
    return encode_frame(
        MSG_FULL_CLIENT_REQUEST,
        FLAG_POSITIVE_SEQUENCE,
        payload,
        sequence=1,
        serialization=SERIALIZATION_JSON,
        compression=COMPRESSION_GZIP,
    )


def _speech_rate(speed: float) -> int:
    """OpenAI 风格倍速（1.0 正常）→ 火山 speech_rate 百分比（[-50,100]，100 即 2.0 倍速）"""
    return max(-50, min(100, round((speed - 1.0) * 100)))


def _tts_v3_request(
    text: str, voice: str, speed: float, response_format: str, sample_rate: int
) -> bytes:
    """单密钥口径 TTS 请求帧（v3 单向流式）。

    帧头为 `FullClientRequest + 无 event/sequence`（[0x11,0x10,0x10,0x00]，JSON 不压缩）+
    payload_size + JSON；正文只含 `req_params`（经典口径的 `app{appid,token,cluster}` 已弃用）。
    """
    body = {
        "req_params": {
            "text": text,
            "speaker": voice,
            "audio_params": {
                "format": response_format,
                "sample_rate": sample_rate,
                "speech_rate": _speech_rate(speed),
            },
        },
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    return encode_frame(
        MSG_FULL_CLIENT_REQUEST,
        FLAG_NO_SEQUENCE,
        payload,
        serialization=SERIALIZATION_JSON,
        compression=COMPRESSION_NONE,
    )


async def _collect_tts_audio(ws) -> bytes:
    """读取经典口径 TTS 下行并聚合音频：0b1011 payload 为裸音频（非 base64），负 sequence 为末帧，
    0b1001 JSON 元信息帧忽略，0b1111 错误帧抛业务异常。"""
    chunks: list[bytes] = []
    async for raw in ws:
        data = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        frame = decode_frame(data)
        if frame.message_type == MSG_ERROR:
            code, message = parse_error_payload(frame.payload)
            _raise_volc_error(code, message)
        if frame.message_type == MSG_AUDIO_ONLY_SERVER_RESPONSE:
            if frame.payload:
                chunks.append(frame.payload)
            if frame.sequence is not None and frame.sequence < 0:
                break
    return b"".join(chunks)


@dataclass(frozen=True)
class VolcV3Frame:
    """解码后的 v3 TTS 下行帧（header/session_id 已剥离）"""

    message_type: int
    event: int
    payload: bytes


def _skip_length_prefixed(raw: bytes, pos: int) -> int:
    """跳过「int32 长度前缀 + 内容」字段，返回内容之后的偏移。

    v3 帧的 session_id / connect_id 均为该结构（仅需跳过取值）。
    """
    size = int.from_bytes(raw[pos : pos + 4], "big", signed=False)
    return pos + 4 + size


def decode_v3_frame(raw: bytes) -> VolcV3Frame:
    """解析 v3 单向流式 TTS 下行帧。

    帧体（flags=WithEvent 时）：event(int32) + session_id|connect_id(长度前缀) + payload_size
    + payload；不带 event 的帧（如错误帧）直接接 payload_size + payload。
    """
    if len(raw) < 4:
        raise ValueError("火山帧长度不足 4 字节")
    header_size = (raw[0] & 0x0F) * 4
    message_type = raw[1] >> 4
    flags = raw[1] & 0x0F
    pos = header_size
    event = 0
    if flags == FLAG_WITH_EVENT:
        if len(raw) < pos + 4:
            raise ValueError("火山 v3 帧缺少 event 字段")
        event = int.from_bytes(raw[pos : pos + 4], "big", signed=True)
        pos += 4
        pos = _skip_length_prefixed(raw, pos)
    if len(raw) < pos + 4:
        raise ValueError("火山 v3 帧缺少 payload_size 字段")
    size = int.from_bytes(raw[pos : pos + 4], "big", signed=False)
    pos += 4
    return VolcV3Frame(message_type=message_type, event=event, payload=raw[pos : pos + size])


async def _collect_v3_tts_audio(ws) -> bytes:
    """读取单密钥口径 TTS（v3 单向流式）下行并聚合音频。

    音频帧（message_type=AudioOnlyServer，event=TTSResponse）payload 为裸音频；
    event=SessionFinished 的正常终帧结束读取；错误帧与建连/会话失败事件抛业务异常。
    """
    chunks: list[bytes] = []
    async for raw in ws:
        data = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        frame = decode_v3_frame(data)
        if frame.message_type == MSG_ERROR:
            code, message = parse_error_payload(frame.payload)
            _raise_volc_error(code, message)
        if frame.event in (EVENT_CONNECTION_FAILED, EVENT_SESSION_FAILED):
            _raise_volc_error(-1, frame.payload.decode("utf-8", "replace"))
        if frame.message_type == MSG_AUDIO_ONLY_SERVER_RESPONSE and frame.payload:
            chunks.append(frame.payload)
        if frame.event == EVENT_SESSION_FINISHED:
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
    """文本合成语音 → 音频字节。

    火山 TTS 以 speaker（=voice）选音色，model_id 不参与请求体，此处仅作路由标识。
    鉴权口径决定端点与报文：单密钥走 v3 单向流式，经典走 v1 ws_binary。
    握手/单请求帧按 Key 逐候选重试，聚合音频帧返回；整体受 provider 超时约束。
    """
    redis: Redis = await get_redis_client()
    candidates = await _usable_keys(provider)
    if not candidates:
        raise _no_key_error(provider)

    single_key = _key_auth_header(provider) is not None
    url = _ws_url(provider, _TTS_V3_PATH if single_key else _TTS_V1_PATH)
    timeout = _timeout(provider)

    rejected_status: int | None = None
    for key in candidates:
        if not await provider_key_selector.reserve_key(redis, key):
            continue
        plaintext = _key_plaintext(key)
        try:
            ws = await ws_connect(
                url,
                additional_headers=_tts_headers(provider, plaintext),
                open_timeout=_WS_OPEN_TIMEOUT,
                max_size=None,
            )
        except InvalidStatus as exc:
            rejected_status = exc.response.status_code
            logger.error("火山 TTS 握手失败 status=%s", rejected_status)
            if rejected_status in _KEY_UNAVAILABLE_STATUS:
                await provider_key_selector.mark_call_failed(redis, key.id, str(rejected_status))
            continue
        except Exception as exc:
            logger.error("连接火山 TTS 会话失败: %s", exc)
            continue
        try:
            if single_key:
                await ws.send(_tts_v3_request(text, voice, speed, response_format, sample_rate))
                audio = await asyncio.wait_for(_collect_v3_tts_audio(ws), timeout=timeout)
            else:
                await ws.send(
                    _tts_v1_request(
                        provider, plaintext, text, voice, speed, response_format, sample_rate
                    )
                )
                audio = await asyncio.wait_for(_collect_tts_audio(ws), timeout=timeout)
        except TimeoutError:
            logger.error("火山 TTS 合成超时（%.1fs）", timeout)
            await _safe_close(ws)
            continue
        except BusinessException:
            await _safe_close(ws)
            raise
        except Exception as exc:
            logger.error("火山 TTS 合成失败: %s", exc)
            await _safe_close(ws)
            continue
        await provider_key_selector.mark_call_success(redis, key.id)
        await _safe_close(ws)
        return audio

    if rejected_status is not None:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR, f"语音引擎 TTS 握手被拒绝: {rejected_status}"
        )
    raise BusinessException(ResultCode.BUSINESS_ERROR, "语音引擎 TTS 会话不可用")


async def _safe_close(ws) -> None:
    """关闭 WS（已断/关闭失败不影响编排侧收尾）"""
    try:
        await ws.close()
    except Exception as exc:
        logger.debug("关闭火山 WS 失败: %s", exc)


async def status(provider) -> dict[str, Any]:
    """引擎状态 → 语音服务状态契约字段。

    火山为托管云服务，编排侧无本地模型加载态可探测，故固定上报在线、加载态为未知（False），
    字段名与 openai_audio_client.status 完全一致。
    """
    return {
        "engine_status": "online",
        "stream_model_loaded": False,
        "offline_model_loaded": False,
        "voice_model_loaded": False,
    }
