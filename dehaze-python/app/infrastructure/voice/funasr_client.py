"""FunASR 语音识别客户端（进程内引擎调用）

引擎部署在 dehaze-python 进程内（见 funasr_engine），本模块对 asr_service
保持既有接口不变（stream_session / send_audio / send_eos / recv_messages /
offline / register_hotwords），内部切换为本地引擎调用：
- 流式：分段增量识别（伪流式，每累计 1 秒音频输出一次当前累计文本）
- 离线：完整音频一次性转写
- 热词：注册到引擎进程内热词表，即时生效
"""

import asyncio
import json
import logging

from app.config import settings
from app.infrastructure.voice import funasr_engine
from app.infrastructure.voice.funasr_engine import FunASREngineError

logger = logging.getLogger(__name__)

# 分段增量识别触发阈值（字节）：16kHz/16bit/mono 下 32000 字节 = 1 秒音频
_PARTIAL_BYTES = 16000 * 2


class FunASRClientError(Exception):
    """FunASR 调用失败（依赖缺失/模型加载失败/推理失败/音频格式不合规格）"""


class FunASRStreamSession:
    """FunASR 流式识别会话（进程内引擎，分段增量识别）"""

    def __init__(self, model: str | None) -> None:
        self._model = model
        self._buffer = bytearray()
        self._transcribed = 0
        self._text_parts: list[str] = []
        self._queue: asyncio.Queue = asyncio.Queue()

    async def send_audio(self, chunk: bytes) -> None:
        """推送一段 PCM 音频块（16kHz/16bit/mono），累计 1 秒触发分段增量识别"""
        self._buffer.extend(chunk)
        if len(self._buffer) - self._transcribed >= _PARTIAL_BYTES:
            await self._transcribe_segment()

    async def send_eos(self) -> None:
        """结束信号：转写剩余音频并输出最终结果"""
        await self._transcribe_segment(final=True)

    async def _transcribe_segment(self, *, final: bool = False) -> None:
        segment = bytes(self._buffer[self._transcribed :])
        self._transcribed = len(self._buffer)
        try:
            text = await funasr_engine.run_in_executor(
                funasr_engine.transcribe, segment, self._model, settings.VOICE_ASR_STREAM_MODEL
            )
        except FunASREngineError as e:
            raise FunASRClientError(str(e)) from e
        if text:
            self._text_parts.append(text)
        await self._queue.put({"text": "".join(self._text_parts), "is_final": final})

    async def recv_messages(self) -> "asyncio.AsyncIterator[str]":
        """迭代接收识别结果 JSON（收到最终结果后结束）"""
        while True:
            msg = await self._queue.get()
            yield json.dumps(msg, ensure_ascii=False)
            if msg["is_final"]:
                return


class FunASRClient:
    """FunASR 引擎客户端：流式会话 / 离线转写 / 热词注册"""

    async def stream_session(self, *, model: str | None = None) -> FunASRStreamSession:
        """建立流式识别会话（预加载模型，依赖缺失/加载失败时抛 FunASRClientError）"""
        try:
            await funasr_engine.run_in_executor(
                funasr_engine.ensure_model, model, settings.VOICE_ASR_STREAM_MODEL
            )
        except FunASREngineError as e:
            raise FunASRClientError(str(e)) from e
        return FunASRStreamSession(model)

    async def offline(self, audio: bytes, *, model: str | None = None) -> str:
        """离线转写完整音频（WAV/PCM，16kHz/16bit/mono），返回完整识别文本"""
        try:
            return await funasr_engine.run_in_executor(
                funasr_engine.transcribe, audio, model, settings.VOICE_ASR_OFFLINE_MODEL
            )
        except FunASREngineError as e:
            raise FunASRClientError(str(e)) from e

    async def register_hotwords(self, words: list[str]) -> None:
        """注册热词到引擎热词表，替换式更新，即时生效"""
        funasr_engine.register_hotwords(words)


funasr_client = FunASRClient()
