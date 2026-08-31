"""语音服务状态聚合服务：默认引擎健康 + ASR 并发会话数

对齐《后端实现.md》§2.3/§2.4 与 API 契约 GET /api/v1/voice/service/status。
状态反映注册表当前默认引擎：
- local：进程内引擎（FunASR/Piper）engine_status() 上报
- cloud：真实探活依赖厂商协议（未接入），按健康开关 + 熔断标记
  （Redis voice:provider:{id}:circuit_open，对齐 provider_health_service）上报；
  健康检查关闭视为 online（对齐 provider_health_service：不参与判定即视为健康）
并发会话数复用 ASR 服务的 Redis 有序集合 voice:asr:sessions（剪枝后 ZCARD）。
"""

import logging
import time
from typing import Any

from redis.asyncio import Redis

from app.config import settings
from app.infrastructure.voice.provider.registry import voice_engine_registry
from app.infrastructure.voice.funasr_engine import engine_status as funasr_status
from app.infrastructure.voice.piper_tts_engine import engine_status as piper_status

logger = logging.getLogger(__name__)

# 与 ASR 服务并发计数共用同一 Redis 键与 TTL（后端实现 §10）
_CONCURRENT_KEY = "voice:asr:sessions"
_SESSION_TTL = 30 * 60

# 语音引擎熔断标记（后端实现 §2.4，对齐 provider_health_service 的 circuit_open 机制）
_CIRCUIT_KEY = "voice:provider:{}:circuit_open"


class VoiceServiceStatusService:
    """语音服务状态聚合服务"""

    def __init__(self, engine_registry=voice_engine_registry) -> None:
        self.engine_registry = engine_registry

    async def get_status(self, redis: Redis) -> dict[str, Any]:
        """聚合语音服务状态（不抛异常）。

        默认引擎解析失败/未配置时按 offline 上报（T-VS-067：引擎不可用
        接口正常返回），确保引擎异常不影响状态接口可用性。
        """
        now = time.time()
        # 剪枝过期会话后计数，与 ASR 服务 _check_concurrency 保持一致
        await redis.zremrangebyscore(_CONCURRENT_KEY, 0, now - _SESSION_TTL)
        concurrent_sessions = await redis.zcard(_CONCURRENT_KEY)

        asr = await self._engine_status("asr", redis)
        tts = await self._engine_status("tts", redis)

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

    async def _engine_status(self, engine_type: str, redis: Redis) -> dict[str, Any]:
        try:
            row = await self.engine_registry.resolve_default_engine(engine_type)
        except Exception:  # noqa: BLE001 - 状态聚合不抛异常，解析失败视为引擎不可用
            logger.warning("解析默认%s引擎失败，按 offline 上报", engine_type, exc_info=True)
            return {}
        if row is None:
            return {}
        if row.provider_code == "local":
            return funasr_status() if engine_type == "asr" else piper_status()
        # 云端引擎：健康检查关闭 → 不参与判定视为 online（对齐 provider_health_service）
        if not row.health_check_enabled:
            return {"engine_status": "online", "engine": row.provider_code}
        # 健康检查开启：熔断标记命中 → offline；未熔断 → 厂商协议未接入无法
        # 真实探活，同样按 offline 上报（引擎当前确实不能提供云端服务）。
        # 熔断分支在厂商协议接入、引擎可真实探活后承担 online→offline 降级
        if await redis.exists(_CIRCUIT_KEY.format(row.id)):
            return {"engine_status": "offline", "engine": row.provider_code}
        return {"engine_status": "offline", "engine": row.provider_code}


# 单例（对齐 hotword_service / tts_service 风格）
voice_service_status = VoiceServiceStatusService()
