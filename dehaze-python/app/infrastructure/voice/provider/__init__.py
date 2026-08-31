"""语音引擎 Provider：本地（FunASR/Piper）与云端（占位）统一抽象"""

from app.infrastructure.voice.provider.base import ASRProvider, ASRStreamSession, TTSProvider
from app.infrastructure.voice.provider.cloud_asr import CloudAsrProvider
from app.infrastructure.voice.provider.cloud_tts import CloudTtsProvider
from app.infrastructure.voice.provider.local_asr import LocalAsrProvider
from app.infrastructure.voice.provider.local_tts import LocalTtsProvider

__all__ = [
    "ASRProvider",
    "ASRStreamSession",
    "TTSProvider",
    "LocalAsrProvider",
    "LocalTtsProvider",
    "CloudAsrProvider",
    "CloudTtsProvider",
]
