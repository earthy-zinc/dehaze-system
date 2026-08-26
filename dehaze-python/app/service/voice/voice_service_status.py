"""语音服务状态聚合服务：ASR/TTS 引擎状态 + ASR 并发会话数

对齐《后端实现.md》§2.3 与 API 契约 GET /api/v1/voice/service/status。
引擎状态由语音基础设施层（FunASR/Piper）的 engine_status() 查询，
并发会话数复用 ASR 服务的 Redis 有序集合 voice:asr:sessions（剪枝后 ZCARD）。
"""

import time
from typing import Any

from redis.asyncio import Redis

from app.config import settings
from app.dependencies.redis import get_redis_client
from app.infrastructure.voice.funasr_engine import engine_status as funasr_status
from app.infrastructure.voice.piper_tts_engine import engine_status as piper_status

# 与 ASR 服务并发计数共用同一 Redis 键与 TTL（后端实现 §10）
_CONCURRENT_KEY = "voice:asr:sessions"
_SESSION_TTL = 30 * 60


class VoiceServiceStatusService:
    """语音服务状态聚合服务"""

    async def get_status(self, redis: Redis) -> dict[str, Any]:
        """聚合语音服务状态（不抛异常）。

        引擎不可用时由引擎层返回 offline，本方法仅做字段映射与并发计数，
        不对引擎可用性做二次判定，确保 T-VS-067（引擎不可用接口正常返回）。
        """
        now = time.time()
        # 剪枝过期会话后计数，与 ASR 服务 _check_concurrency 保持一致
        await redis.zremrangebyscore(_CONCURRENT_KEY, 0, now - _SESSION_TTL)
        concurrent_sessions = await redis.zcard(_CONCURRENT_KEY)

        asr = funasr_status()
        tts = piper_status()

        return {
            "asr": {
                "engineStatus": asr["engine_status"],
                "concurrentSessions": concurrent_sessions,
                "maxConcurrentSessions": settings.VOICE_ASR_MAX_CONCURRENT_SESSIONS,
                "streamModelLoaded": asr["stream_model_loaded"],
                "offlineModelLoaded": asr["offline_model_loaded"],
            },
            "tts": {
                "engineStatus": tts["engine_status"],
                "voiceModelLoaded": tts["voice_model_loaded"],
            },
        }


# 单例（对齐 hotword_service / tts_service 风格）
voice_service_status = VoiceServiceStatusService()
