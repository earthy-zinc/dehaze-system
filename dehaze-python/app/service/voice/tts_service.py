"""TTS 语音合成服务：参数校验 → 缓存 → 余额预校验 → 本地合成 → 加密存储 → 实扣。

缓存策略（后端实现 §4.3/§6.4）：
- 缓存 Key = SHA256(text + voiceId + speed)，索引存 Redis：
  hash `voice:tts:cache:{userId}` → {cacheKey: {fileId, format}}，TTL=VOICE_TTS_CACHE_TTL
  zset `voice:tts:cache:lru:{userId}` 记录创建时间，超 MAX_CACHE_PER_USER 时 LRU 淘汰最旧
- 音频内容 AES-256-GCM 加密后经 FileService 存储（密钥 VOICE_TTS_CACHE_ENCRYPTION_KEY 派生）
- 缓存命中不合成、不扣费

TtsCacheManager 职责并入本服务，不额外拆分。
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.voice import piper_tts_engine
from app.infrastructure.voice.piper_tts_engine import LocalTtsError
from app.models.entity.sys_file import SysFile
from app.models.schema.voice_tts import FORMAT_VALUES, SAMPLE_RATE_VALUES, VOICE_CATALOG
from app.service.file_service import file_service
from app.service.storage.factory import get_storage_by_name
from app.service.voice.voice_billing_service import voice_billing_service

logger = logging.getLogger(__name__)

# 缓存 hash / zset key 前缀
_CACHE_HASH_PREFIX = "voice:tts:cache:{}"
_CACHE_LRU_PREFIX = "voice:tts:cache:lru:{}"
# AES-GCM 随机 nonce 长度（字节）
_GCM_NONCE_LEN = 12
# 存储对象读取线程池（存储 SDK 为同步调用，避免阻塞事件循环）
_storage_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tts-storage")


def _cache_key(text: str, voice: str, speed: float) -> str:
    """缓存 Key = SHA256(text + voiceId + speed)"""
    return hashlib.sha256(f"{text}{voice}{speed}".encode()).hexdigest()


def _derive_key() -> bytes:
    """由配置密钥派生 32 字节 AES-GCM 密钥"""
    return hashlib.sha256(settings.VOICE_TTS_CACHE_ENCRYPTION_KEY.encode()).digest()


def encrypt_audio(raw: bytes) -> bytes:
    """AES-256-GCM 加密，返回 nonce + ciphertext + tag"""
    key = _derive_key()
    nonce = os.urandom(_GCM_NONCE_LEN)
    ct = AESGCM(key).encrypt(nonce, raw, None)
    return nonce + ct


def decrypt_audio(blob: bytes) -> bytes:
    """AES-256-GCM 解密（blob = nonce + ciphertext + tag）"""
    key = _derive_key()
    nonce, ct = blob[:_GCM_NONCE_LEN], blob[_GCM_NONCE_LEN:]
    return AESGCM(key).decrypt(nonce, ct, None)


class TtsService:
    """TTS 合成与缓存管理"""

    def __init__(
        self,
        file_service=file_service,
        voice_billing_service=voice_billing_service,
        piper_tts_engine=piper_tts_engine,
    ):
        self.file_service = file_service
        self.voice_billing_service = voice_billing_service
        self.piper_tts_engine = piper_tts_engine

    # ==================== 合成 ====================

    async def synthesize(self, 
        db: AsyncSession,
        redis: Redis,
        user_id: int,
        text: str,
        voice: str,
        speed: float,
        format_: str,
        sample_rate: int,
    ) -> dict[str, str]:
        """文本转语音：命中缓存直接返回，否则合成 + 加密存储 + 实扣。

        返回 {"audioUrl", "format"} 契约字段。
        """
        voice = voice or settings.VOICE_TTS_VOICE_ID
        self._validate_params(text, voice, format_, sample_rate)

        cache_key = _cache_key(text, voice, speed)
        cache_hit = await self._get_cached(redis, user_id, cache_key)
        if cache_hit:
            return cache_hit

        # 余额预校验（预估 = 字符数 × 单价）
        estimated = math.ceil(len(text) * settings.VOICE_TTS_CREDITS_PER_CHAR)
        await self.voice_billing_service.ensure_balance(db, user_id, estimated)

        # 本地 Piper 引擎合成
        try:
            audio = await self.piper_tts_engine.run_in_executor(
                self.piper_tts_engine.synthesize, text, voice, speed, format_, sample_rate
            )
        except LocalTtsError as exc:
            logger.error("本地 TTS 合成失败 user_id=%s: %s", user_id, exc)
            raise BusinessException(ResultCode.BUSINESS_ERROR, f"语音合成失败: {exc}") from exc

        # 加密存储 + 写缓存索引
        audio_url = await self._store_and_cache(
            db, redis, user_id, cache_key, audio, format_
        )
        await self.voice_billing_service.charge_tts(db, user_id, len(text))
        return {"audioUrl": audio_url, "format": format_}

    def _validate_params(self, text: str, voice: str, format_: str, sample_rate: int) -> None:
        """参数校验：空文本/超长/音色/格式/采样率非法 → A0400。"""
        if not text or not text.strip():
            raise BusinessException(ResultCode.PARAM_ERROR, "待合成文本不能为空")
        if len(text) > settings.VOICE_TTS_MAX_TEXT_LENGTH:
            raise BusinessException(
                ResultCode.PARAM_ERROR,
                f"待合成文本不能超过 {settings.VOICE_TTS_MAX_TEXT_LENGTH} 字符",
            )
        if format_ not in FORMAT_VALUES:
            raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的音频格式: {format_}")
        if sample_rate not in SAMPLE_RATE_VALUES:
            raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的采样率: {sample_rate}")
        if voice not in {v["id"] for v in VOICE_CATALOG}:
            raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的音色: {voice}")

    # ==================== 缓存 ====================

    async def _get_cached(self, 
        redis: Redis, user_id: int, cache_key: str
    ) -> dict[str, str] | None:
        """命中缓存返回 audioUrl + format，否则 None。"""
        entry = await redis.hget(_CACHE_HASH_PREFIX.format(user_id), cache_key)
        if not entry:
            return None
        try:
            meta = json.loads(entry)
        except json.JSONDecodeError:
            return None
        audio_url = f"/api/v1/voice/tts/audio/{cache_key}"
        return {"audioUrl": audio_url, "format": meta.get("format", "mp3")}

    async def _store_and_cache(self, 
        db: AsyncSession,
        redis: Redis,
        user_id: int,
        cache_key: str,
        audio: bytes,
        format_: str,
    ) -> str:
        """加密存储音频 + 写入缓存索引（含 LRU 淘汰），返回 audioUrl。"""
        sys_file = await file_service.upload_file(
            db,
            filename=f"tts_{cache_key[:8]}.{format_}",
            content=encrypt_audio(audio),
            content_type=f"audio/{format_}",
        )

        hash_key = _CACHE_HASH_PREFIX.format(user_id)
        lru_key = _CACHE_LRU_PREFIX.format(user_id)
        meta = json.dumps({"fileId": sys_file.id, "format": format_})

        await redis.hset(hash_key, cache_key, meta)
        await redis.zadd(lru_key, {cache_key: time.time()})
        # 索引 TTL 与缓存过期时间一致（hash 与 LRU zset 同步过期，避免 zset 无限增长）
        await redis.expire(hash_key, settings.VOICE_TTS_CACHE_TTL)
        await redis.expire(lru_key, settings.VOICE_TTS_CACHE_TTL)

        # LRU 淘汰：超上限删除最旧
        count = await redis.zcard(lru_key)
        if count > settings.VOICE_TTS_MAX_CACHE_PER_USER:
            overflow = count - settings.VOICE_TTS_MAX_CACHE_PER_USER
            oldest = await redis.zrange(lru_key, 0, overflow - 1)
            if oldest:
                await redis.hdel(hash_key, *oldest)
                await redis.zrem(lru_key, *oldest)

        return f"/api/v1/voice/tts/audio/{cache_key}"

    # ==================== 缓存音频下载 ====================

    async def load_cached_audio(self, 
        db: AsyncSession, redis: Redis, user_id: int, cache_key: str
    ) -> tuple[bytes, str] | None:
        """按 cacheKey 读取并解密缓存音频（校验归属：仅本人缓存可访问）。

        返回 (解密后音频字节, 音频格式)；缓存不存在或非本人返回 None。
        """
        entry = await redis.hget(_CACHE_HASH_PREFIX.format(user_id), cache_key)
        if not entry:
            return None
        try:
            meta = json.loads(entry)
        except json.JSONDecodeError:
            return None

        file_info: SysFile | None = await self.file_service.get_file_by_id(db, meta.get("fileId"))
        if not file_info:
            return None

        storage = get_storage_by_name(file_info.storage)
        encrypted = await _read_object_bytes(
            storage, settings.MINIO_BUCKET, file_info.object_name
        )
        try:
            audio = decrypt_audio(encrypted)
        except Exception:
            logger.error("TTS 缓存音频解密失败 cache_key=%s", cache_key, exc_info=True)
            return None
        return audio, meta.get("format", "mp3")


async def _read_object_bytes(storage: Any, bucket: str, object_name: str) -> bytes:
    """在工作线程中读取存储对象完整字节（存储 SDK 为同步调用）。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_storage_executor, storage.download, bucket, object_name)


tts_service = TtsService()
