"""本地 ASR Provider：封装进程内 FunASR 引擎（funasr_client）

模型（sensevoice/paraformer）从 sys_voice_model（engine_type=asr）解析，
映射到 funasr 逻辑模型名直接透传给 funasr_client（内部经 funasr_engine
resolve_model_id 映射为模型 ID）。延迟导入 funasr_client / funasr_engine 以
控制启动成本；engine_status 透传 funasr_engine.engine_status()。
"""

from typing import AsyncIterator

from app.infrastructure.voice.provider.base import ASRProvider, ASRStreamSession


class _LocalAsrStreamSession(ASRStreamSession):
    """包装 FunASR 流式会话为统一 ASRStreamSession 协议"""

    def __init__(self, funasr_session) -> None:
        self._session = funasr_session

    async def send_audio(self, chunk: bytes) -> None:
        await self._session.send_audio(chunk)

    async def send_eos(self) -> None:
        await self._session.send_eos()

    def recv_messages(self) -> AsyncIterator[str]:
        return self._session.recv_messages()


class LocalAsrProvider(ASRProvider):
    """本地 FunASR 语音识别 Provider"""

    def __init__(self, provider) -> None:
        self._provider = provider
        self._models: dict[str, str] | None = None  # {model_type: 逻辑模型名}

    async def _resolve_models(self) -> dict[str, str]:
        """懒加载并缓存本 Provider 启用的 ASR 模型：{stream/offline: 逻辑模型名}，并注入引擎模型注册表"""
        if self._models is None:
            from app.database import get_db_session
            from app.infrastructure.voice import funasr_engine
            from app.repository.voice_model_repository import voice_model_repository

            async with get_db_session() as db:
                models = await voice_model_repository.list_enabled(db, "asr")
            # 注册表化：逻辑模型名 → ModelScope ID 由 sys_voice_model 决定，注入引擎替代硬编码 _MODEL_IDS
            funasr_engine.configure_models(
                {m.model_id: (m.params or {}).get("model_id") for m in models}
            )
            self._models = {m.model_type: m.model_id for m in models}
        return self._models

    async def recognize_stream(self) -> ASRStreamSession:
        from app.infrastructure.voice.funasr_client import funasr_client

        models = await self._resolve_models()
        session = await funasr_client.stream_session(model=models.get("stream"))
        return _LocalAsrStreamSession(session)

    async def recognize_offline(self, audio_bytes: bytes) -> str:
        from app.infrastructure.voice.funasr_client import funasr_client

        models = await self._resolve_models()
        return await funasr_client.offline(audio_bytes, model=models.get("offline"))

    async def register_hotwords(self, words: list[str]) -> None:
        from app.infrastructure.voice.funasr_client import funasr_client

        await funasr_client.register_hotwords(words)

    async def engine_status(self) -> dict:
        from app.infrastructure.voice import funasr_engine

        return funasr_engine.engine_status()
