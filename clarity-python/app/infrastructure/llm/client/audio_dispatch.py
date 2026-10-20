"""语音协议分派：按 provider.protocol_type 选择 ASR/TTS 协议适配器（调用方不感知协议差异）

统一契约（与 openai_audio_client 同构，四个能力签名固定）：
- transcribe(provider, model_id, audio, hotwords) -> str          离线识别整段音频 → 文本
- synthesize(provider, model_id, text, voice, speed, response_format, sample_rate) -> bytes
                                                                 文本合成语音 → 音频字节
- open_stream(provider, model_id, hotwords, session_id=None) -> AudioStreamSession
                                                                 双向流式识别会话
- status(provider) -> dict                                       引擎状态（语音服务状态契约字段）

流式会话统一语义（AudioStreamSession，各适配器内部负责协议翻译）：
- send_audio(chunk: bytes)：PCM 16k 单声道音频块
- send_eos()：结束输入，等待引擎输出最终结果
- recv_messages() -> AsyncIterator[str]：下行统一 JSON 文本 {"text": str, "is_final": bool}
  （asr_service 直接消费该形状；各协议事件在此归一化）
- aclose()：关闭引擎侧会话

协议实现：
- openai_audio    内置语音引擎（clarity-voice 私有 WS 扩展），见 openai_audio_client
- dashscope_realtime 阿里云百炼（run-task/continue-task/finish-task 指令框架 + 二进制音频帧），
                  见 dashscope_audio_client
- volc_realtime   火山引擎（4 字节位域头 + gzip payload 的二进制帧协议），见 volc_audio_client

未知协议回落 openai_audio（与 create_chat_client 的"未知协议回落 OpenAI 兼容"口径一致）。
新增协议族须同步三处：config/sql/schema/sys_ai_provider.sql 的 protocol_type 列注释（值域声明处）、
管理端「协议类型」下拉（clarity-front-vue ProviderDrawer）、本文件的分派与 VOICE_PROTOCOL_TYPES。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

PROTOCOL_OPENAI_AUDIO = "openai_audio"
PROTOCOL_DASHSCOPE_REALTIME = "dashscope_realtime"
PROTOCOL_VOLC_REALTIME = "volc_realtime"

# 语音协议族值域（asr/tts 共用同一协议族；新增协议须在此登记并补 get_audio_client 分派）
VOICE_PROTOCOL_TYPES = frozenset(
    {PROTOCOL_OPENAI_AUDIO, PROTOCOL_DASHSCOPE_REALTIME, PROTOCOL_VOLC_REALTIME}
)

# 实时语音协议族：WS 私有帧协议，无 OpenAI 兼容 HTTP 端点（故不可用 HTTP 形状的探针探测）
REALTIME_VOICE_PROTOCOLS = frozenset({PROTOCOL_DASHSCOPE_REALTIME, PROTOCOL_VOLC_REALTIME})


class AudioStreamSession(Protocol):
    """流式识别会话统一接口（各协议适配器实现；语义见模块 docstring）"""

    async def send_audio(self, chunk: bytes) -> None: ...

    async def send_eos(self) -> None: ...

    def recv_messages(self) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


class VoiceAudioClient(Protocol):
    """语音协议适配器统一接口"""

    async def transcribe(
        self, provider: Any, model_id: str, audio: bytes, hotwords: list[str]
    ) -> str: ...

    async def synthesize(
        self,
        provider: Any,
        model_id: str,
        text: str,
        voice: str,
        speed: float,
        response_format: str,
        sample_rate: int,
    ) -> bytes: ...

    async def open_stream(
        self, provider: Any, model_id: str, hotwords: list[str], session_id: str | None = None
    ) -> AudioStreamSession: ...

    async def status(self, provider: Any) -> dict[str, Any]: ...


def get_audio_client(protocol_type: str) -> VoiceAudioClient:
    """按协议类型返回语音客户端（未知协议回落内置引擎的 OpenAI Audio 兼容实现）"""
    if protocol_type == PROTOCOL_DASHSCOPE_REALTIME:
        from app.infrastructure.llm.client import dashscope_audio_client

        return dashscope_audio_client
    if protocol_type == PROTOCOL_VOLC_REALTIME:
        from app.infrastructure.llm.client import volc_audio_client

        return volc_audio_client
    from app.infrastructure.llm.client import openai_audio_client

    return openai_audio_client
