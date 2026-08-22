"""语音交互模块 - TTS 语音合成路由

对齐《API接口.md》§2.2：文本转语音、音色列表、缓存音频下载。
TTS 为基础能力，登录用户均可使用（无特殊权限标识）。
"""

from fastapi import APIRouter, Body, Depends, Path
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.voice_tts import VOICE_CATALOG, TtsForm
from app.service.voice.tts_service import tts_service

router = APIRouter(
    prefix="/api/v1/voice/tts",
    tags=["语音交互-TTS"],
    dependencies=[Depends(get_current_user)],
)


@router.post("", summary="文本转语音")
async def synthesize_tts(
    body: TtsForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await tts_service.synthesize(
        db,
        redis,
        user.id,
        body.text,
        body.voice,
        body.speed,
        body.format,
        body.sampleRate,
    )
    return success(result)


@router.get("/voices", summary="可用音色列表")
async def list_voices(
    user: UserContext = Depends(get_current_user),
):
    return success(VOICE_CATALOG)


@router.get("/audio/{cache_key}", summary="缓存音频下载（解密后流式返回）")
async def download_cached_audio(
    cache_key: str = Path(..., description="缓存 Key（SHA256）"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """按 cacheKey 下载缓存音频，仅本人可访问（他人 cacheKey 在本用户缓存中不存在 → 404）。"""
    audio = await tts_service.load_cached_audio(db, redis, user.id, cache_key)
    if audio is None:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "音频缓存不存在或已过期")
    audio_bytes, format_ = audio
    media_type = f"audio/{format_}"
    return StreamingResponse(
        iter([audio_bytes]),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="tts.{format_}"'},
    )
