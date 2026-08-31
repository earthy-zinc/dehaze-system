"""语音引擎 Provider 抽象接口

语音交互模块以「语音引擎注册表」统一本地（FunASR/Piper）与云端（阿里云/腾讯云/
讯飞 ASR、Azure/阿里云 TTS）引擎：asr/tts 各一个默认引擎，应用侧经 Provider 抽象
透明消费，不感知具体厂商。本地 Provider 封装进程内引擎（funasr_client /
piper_tts_engine），云端 Provider 为占位（厂商适配待实现）。
"""

import abc
from typing import AsyncIterator, ClassVar


class ASRStreamSession(abc.ABC):
    """流式识别会话：上行推送 PCM 音频块，下行迭代接收识别结果 JSON

    协议对齐现有 funasr_client 的流式会话：send_audio 分段增量识别，
    send_eos 结束并输出最终结果，recv_messages 迭代返回 {"text","is_final"}。
    """

    @abc.abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """推送一段 PCM 音频块（16kHz/16bit/mono）"""

    @abc.abstractmethod
    async def send_eos(self) -> None:
        """结束信号：转写剩余音频并输出最终结果"""

    @abc.abstractmethod
    def recv_messages(self) -> AsyncIterator[str]:
        """迭代接收识别结果 JSON 文本（收到最终结果后结束）"""


class ASRProvider(abc.ABC):
    """语音识别（ASR）Provider 抽象"""

    engine_type: ClassVar[str] = "asr"

    @abc.abstractmethod
    async def recognize_stream(self) -> ASRStreamSession:
        """建立流式识别会话（依赖缺失/模型加载失败时抛 Provider 异常）"""

    @abc.abstractmethod
    async def recognize_offline(self, audio_bytes: bytes) -> str:
        """离线转写完整音频（WAV/PCM，16kHz/16bit/mono），返回识别文本"""

    @abc.abstractmethod
    async def engine_status(self) -> dict:
        """查询引擎健康状态（不抛异常，返回 engine_status/模型加载等指标）"""

    @abc.abstractmethod
    async def register_hotwords(self, words: list[str]) -> None:
        """注册热词，替换式更新，即时生效"""


class TTSProvider(abc.ABC):
    """语音合成（TTS）Provider 抽象"""

    engine_type: ClassVar[str] = "tts"

    @abc.abstractmethod
    async def synthesize(
        self, text: str, voice_id: str | None, speed: float, format_: str, sample_rate: int
    ) -> bytes:
        """合成文本为音频字节（format_: mp3/wav/pcm; sample_rate: 采样率Hz），返回编码后的音频"""

    @abc.abstractmethod
    async def engine_status(self) -> dict:
        """查询引擎健康状态（不抛异常，返回 engine_status/模型加载等指标）"""
