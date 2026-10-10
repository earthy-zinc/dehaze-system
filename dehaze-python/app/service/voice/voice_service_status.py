"""语音服务状态聚合服务：默认引擎健康 + ASR 并发会话数

对齐《后端实现.md》§2.3 与 API 契约 GET /api/v1/voice/service/status。
阶段 2 起引擎经统一模型路由（sys_ai_model + sys_ai_provider）解析，探测走
OpenAI Audio 适配器的 `GET /v1/status`；模型未配置/供应商不可用/引擎不可达时
按 offline 上报，状态接口始终可用（T-VS-067）。
并发会话数复用 ASR 服务的 Redis 有序集合 voice:asr:sessions（剪枝后 ZCARD）。
"""

import logging
import time
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db_session
from app.infrastructure.llm.client.audio_dispatch import get_audio_client
from app.service.voice.voice_model_route import resolve_voice_model

logger = logging.getLogger(__name__)

# 与 ASR 服务并发计数共用同一 Redis 键与 TTL（后端实现 §10）
_CONCURRENT_KEY = "voice:asr:sessions"
_SESSION_TTL = 30 * 60


class VoiceServiceStatusService:
    """语音服务状态聚合服务"""

    async def get_status(self, redis: Redis) -> dict[str, Any]:
        """聚合语音服务状态（不抛异常）"""
        now = time.time()
        # 剪枝过期会话后计数，与 ASR 服务 _check_concurrency 保持一致
        await redis.zremrangebyscore(_CONCURRENT_KEY, 0, now - _SESSION_TTL)
        concurrent_sessions = await redis.zcard(_CONCURRENT_KEY)

        asr = await self.engine_status("asr")
        tts = await self.engine_status("tts")

        return {
            "asr": {
                "engineStatus": asr.get("engine_status", "offline"),
                "concurrentSessions": concurrent_sessions,
                "maxConcurrentSessions": settings.VOICE_ASR_MAX_CONCURRENT_SESSIONS,
                "streamModelLoaded": asr.get("stream_model_loaded", False),
                "offlineModelLoaded": asr.get("offline_model_loaded", False),
            },
            "tts": {
                "engineStatus": tts.get("engine_status", "offline"),
                "voiceModelLoaded": tts.get("voice_model_loaded", False),
            },
        }

    async def engine_status(self, engine_type: str) -> dict[str, Any]:
        """平台默认语音模型的引擎运行时状态（不抛异常）。

        未配置默认模型、路由解析失败或引擎不可达时返回空 dict，由调用方按 offline
        上报——状态类接口不因引擎异常而不可用（T-VS-067）。
        """
        try:
            async with get_db_session() as db:
                route = await self._resolve(db, engine_type)
            if route is None:
                return {}
            return await get_audio_client(route.provider.protocol_type).status(route.provider)
        except Exception:
            logger.warning("查询默认%s引擎状态失败，按 offline 上报", engine_type, exc_info=True)
            return {}

    async def _resolve(self, db: AsyncSession, engine_type: str):
        """按 engine_type 独立解析平台默认模型路由；未配置/不可用返回 None。

        不做 asr/tts 互借探测：两端可以是不同供应商（如 ASR 阿里云 + TTS 火山），
        借另一侧解析会把一侧的引擎状态错报给另一侧。
        """
        try:
            return await resolve_voice_model(db, engine_type)
        except Exception:
            return None


# 单例（对齐 hotword_service / tts_service 风格）
voice_service_status = VoiceServiceStatusService()
