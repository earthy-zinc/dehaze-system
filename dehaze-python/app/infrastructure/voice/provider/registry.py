"""语音引擎注册表：按 engine_type 路由到默认 Provider

asr/tts 各配置一个默认 Provider（sys_voice_provider.is_default=1），应用侧经
get_asr_provider / get_tts_provider 透明获取。默认引擎解析结果缓存于 Redis
（voice:engine:{engine_type}，短 TTL），管理端修改 is_default/status 后由
VoiceAdminService 失效缓存即时生效，无需重启。Provider 实例按 engine_type
保留内存缓存，以 Redis 缓存中的 provider_id 判断是否需重建。default 为 local
但本地引擎依赖未安装（纯云端部署误配）时抛业务异常 A0500，不降级。
"""

import json
import logging
from typing import Any, Awaitable, Callable

from redis.asyncio import Redis

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.infrastructure.voice.provider.base import ASRProvider, TTSProvider
from app.infrastructure.voice.provider.cloud_asr import CloudAsrProvider
from app.infrastructure.voice.provider.cloud_tts import CloudTtsProvider
from app.infrastructure.voice.provider.local_asr import LocalAsrProvider
from app.infrastructure.voice.provider.local_tts import LocalTtsProvider
# 单例：显式注入真实依赖（无兜底路径；测试经 VoiceEngineRegistry(桩) 独立构造）
from app.repository.voice_provider_repository import voice_provider_repository

logger = logging.getLogger(__name__)

# 本地引擎依赖探测：ASR=funasr / TTS=piper（未安装即视为本地引擎不可用）
_LOCAL_ENGINE_DEP = {"asr": "funasr", "tts": "piper"}

# 默认引擎缓存（后端实现 §2.4）：管理端切换后失效即时生效，短 TTL 保证
# 失效机制异常（如 Redis 抖动丢 DEL）时也快速自愈
_ENGINE_CACHE_KEY = "voice:engine:{}"
_ENGINE_CACHE_TTL = 60


def _local_engine_available(engine_type: str) -> bool:
    try:
        __import__(_LOCAL_ENGINE_DEP[engine_type])
        return True
    except ImportError:
        return False


async def _default_redis_factory() -> Redis:
    # 延迟导入走 get_redis_client 中心入口：测试 mock_redis patch 中心入口即可覆盖
    from app.dependencies.redis import get_redis_client

    return await get_redis_client()


class VoiceEngineRegistry:
    """语音引擎注册表（单例）：按 engine_type 解析并缓存默认 Provider"""

    def __init__(
        self,
        repository,
        session_factory,
        engine_available: Callable[[str], bool],
        redis_factory: Callable[[], Awaitable[Redis]],
    ) -> None:
        # 依赖构造时显式注入（单例在模块末尾注入真实依赖，测试注入桩），
        # 不提供默认兜底，避免运行时多路径导致排查困难
        self._repository = repository
        self._session_factory = session_factory
        self._engine_available = engine_available
        self._redis_factory = redis_factory
        self._providers: dict[str, ASRProvider | TTSProvider] = {}
        # 各 engine_type 当前内存实例对应的默认引擎 id（与 Redis 缓存比对判断是否需重建）
        self._default_ids: dict[str, int] = {}

    async def get_asr_provider(self) -> ASRProvider:
        """获取默认 ASR Provider（default=local 但本地引擎不可用时抛 A0500）"""
        return await self._resolve("asr")

    async def get_tts_provider(self) -> TTSProvider:
        """获取默认 TTS Provider（default=local 但本地引擎不可用时抛 A0500）"""
        return await self._resolve("tts")

    async def resolve_default_engine(self, engine_type: str) -> Any | None:
        """解析默认引擎配置行（不实例化 Provider、不校验本地依赖）。

        供服务状态聚合上报引擎健康使用；未配置默认引擎返回 None，
        状态上报需降级为 offline 而非抛错（T-VS-067）。
        """
        return await self._load_default(engine_type)

    async def invalidate_default_cache(self, engine_type: str) -> None:
        """管理端 is_default/status 变更后失效默认引擎缓存（即时生效）。

        Redis 异常不阻断管理端操作：短 TTL 保证缓存最终自愈。
        """
        try:
            redis = await self._redis_factory()
            await redis.delete(_ENGINE_CACHE_KEY.format(engine_type))
        except Exception as exc:  # noqa: BLE001 - 缓存失效失败仅降级为等待 TTL 过期
            logger.warning("失效语音引擎默认缓存失败(engine_type=%s): %s", engine_type, exc)

    async def _resolve(self, engine_type: str) -> ASRProvider | TTSProvider:
        cached = await self._cache_get(engine_type)
        if (
            engine_type in self._providers
            and self._default_ids.get(engine_type) == (cached or {}).get("provider_id")
        ):
            # Redis 缓存与内存实例一致（含 Redis 不可用缓存为 None 的情况）：不查库直接复用
            return self._providers[engine_type]
        row = await self._load_default(engine_type, cached)
        if row is None:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"未配置默认{engine_type}语音引擎，请检查 sys_voice_provider"
            )
        if self._default_ids.get(engine_type) == row.id and engine_type in self._providers:
            return self._providers[engine_type]
        if row.provider_code == "local" and not self._engine_available(engine_type):
            raise BusinessException(
                ResultCode.BUSINESS_ERROR,
                f"默认{engine_type}语音引擎为本地引擎，但本地依赖未安装（{_LOCAL_ENGINE_DEP[engine_type]}），"
                "纯云端部署请配置云端默认引擎",
            )
        self._providers[engine_type] = self._instantiate(engine_type, row)
        self._default_ids[engine_type] = row.id
        return self._providers[engine_type]

    async def _load_default(self, engine_type: str, cached: dict | None = None) -> Any | None:
        """解析默认引擎配置行：Redis 缓存命中按 provider_id 取行，未命中查库回填。

        缓存指向的引擎已被禁用/取消默认/删除时视为未命中，回源 get_default
        重写缓存；Redis 读写异常静默降级为直查库——路由是语音调用主链路，
        Redis 不可用时不允许阻断，短 TTL 缓存在此场景下本就无意义。
        """
        if cached is None:
            cached = await self._cache_get(engine_type)
        if cached:
            async with self._session_factory() as db:
                row = await self._repository.get_by_id(db, cached["provider_id"])
            if row and row.status == 1 and row.is_default == 1:
                return row
        async with self._session_factory() as db:
            row = await self._repository.get_default(db, engine_type)
        if row:
            await self._cache_put(engine_type, row)
        return row

    async def _cache_get(self, engine_type: str) -> dict | None:
        try:
            redis = await self._redis_factory()
            raw = await redis.get(_ENGINE_CACHE_KEY.format(engine_type))
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取语音引擎默认缓存失败(engine_type=%s): %s", engine_type, exc)
            return None

    async def _cache_put(self, engine_type: str, row: Any) -> None:
        try:
            redis = await self._redis_factory()
            payload = json.dumps({"provider_id": row.id, "provider_code": row.provider_code})
            await redis.set(_ENGINE_CACHE_KEY.format(engine_type), payload, ex=_ENGINE_CACHE_TTL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入语音引擎默认缓存失败(engine_type=%s): %s", engine_type, exc)

    @staticmethod
    def _instantiate(engine_type: str, provider: Any) -> ASRProvider | TTSProvider:
        """按 provider_code 实例化 Provider：'local' → 本地，其余 → 云端占位"""
        if provider.provider_code == "local":
            return LocalAsrProvider(provider) if engine_type == "asr" else LocalTtsProvider(provider)
        return CloudAsrProvider(provider) if engine_type == "asr" else CloudTtsProvider(provider)

voice_engine_registry = VoiceEngineRegistry(
    repository=voice_provider_repository,
    session_factory=get_db_session,
    engine_available=_local_engine_available,
    redis_factory=_default_redis_factory,
)
