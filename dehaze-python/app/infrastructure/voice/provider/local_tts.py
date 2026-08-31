"""本地 TTS Provider：封装进程内 Piper 引擎（piper_tts_engine）

音色（huayan）从 sys_voice_model（engine_type=tts）解析，默认输出 mp3 /
24000Hz（对齐现有 VOICE_CATALOG 采样率档位）。延迟导入 piper_tts_engine 以
控制启动成本；engine_status 透传 piper_tts_engine.engine_status()。
"""

from app.config import settings
from app.infrastructure.voice.provider.base import TTSProvider


class LocalTtsProvider(TTSProvider):
    """本地 Piper 语音合成 Provider"""

    def __init__(self, provider) -> None:
        self._provider = provider
        self._default_voice: str | None = None

    async def _resolve_default_voice(self) -> str:
        """懒加载默认音色：从 sys_voice_model 解析音色并注入引擎注册表，缺省回退配置"""
        if self._default_voice is None:
            from app.database import get_db_session
            from app.infrastructure.voice import piper_tts_engine
            from app.repository.voice_model_repository import voice_model_repository

            async with get_db_session() as db:
                models = await voice_model_repository.list_enabled(db, "tts")
            voices = [m for m in models if m.model_type == "voice"]
            # 注册表化：音色 onnx/大小/下载URL 由 sys_voice_model.params 决定，注入引擎替代硬编码 _VOICE_MODEL_FILES
            piper_tts_engine.configure_voices(
                {m.model_id: (m.params or {}) for m in voices}
            )
            self._default_voice = next(
                (m.model_id for m in voices), settings.VOICE_TTS_VOICE_ID
            )
        return self._default_voice

    async def synthesize(
        self, text: str, voice_id: str | None, speed: float, format_: str, sample_rate: int
    ) -> bytes:
        from app.infrastructure.voice import piper_tts_engine

        voice = voice_id or await self._resolve_default_voice()
        return await piper_tts_engine.run_in_executor(
            piper_tts_engine.synthesize, text, voice, speed, format_, sample_rate
        )

    async def engine_status(self) -> dict:
        from app.infrastructure.voice import piper_tts_engine

        return piper_tts_engine.engine_status()
