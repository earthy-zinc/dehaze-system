"""语音交互模块路由（ASR / TTS / 热词 / 服务状态 / WebSocket）

单一 router 顺序注册，API 路径不变（对齐《API接口.md》）：
- /api/v1/voice/asr/stream-session|offline|result/{sessionId}（F-VS-001 §2.1）
- /api/v1/voice/tts|voices|audio/{cacheKey}（F-VS-002 §2.2）
- /api/v1/voice/hotwords 用户级 + /global 全局（F-VS-004 §2.3）
- /api/v1/voice/service/status（服务状态监控 §2.4，voice:service:monitor）
- /ws/asr（流式 WebSocket 识别，§2.1）
"""

from fastapi import APIRouter, Body, Depends, File, Form, Path, Query, Request, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import success
from app.database import get_db
from app.decorators.permission import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.voice import HotwordForm
from app.models.schema.voice_asr import StreamAsrSessionForm
from app.models.schema.voice_tts import VOICE_CATALOG, TtsForm
from app.service.voice.asr_service import asr_service
from app.service.voice.hotword_service import hotword_service
from app.service.voice.tts_service import tts_service
from app.service.voice.voice_service_status import voice_service_status

router = APIRouter(tags=["语音交互"])


# ==================== ASR（F-VS-001 §2.1） ====================


def _build_ws_url(request: Request, session_id: str) -> str:
    """将 HTTP base_url 推导为 WebSocket 地址（http→ws，https→wss）"""
    scheme = "wss" if request.url.scheme == "https" else "ws"
    host = request.url.netloc
    return f"{scheme}://{host}/ws/asr?sessionId={session_id}"


@router.post("/api/v1/voice/asr/stream-session", summary="创建流式 ASR 会话")
async def create_stream_session(
    body: StreamAsrSessionForm,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    session_id = await asr_service.create_stream_session(redis, db, user.id, body.model)
    return success({"sessionId": session_id, "wsUrl": _build_ws_url(request, session_id)})


@router.get("/api/v1/voice/asr/result/{session_id}", summary="查询流式 ASR 会话最终识别结果")
async def get_asr_result(
    session_id: str = Path(..., description="流式 ASR 会话ID"),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await asr_service.get_result(redis, session_id, user.id)
    return success(result)


@router.post("/api/v1/voice/asr/offline", summary="离线 ASR 识别（multipart 直传音频文件）")
async def offline_asr(
    file: UploadFile = File(..., description="音频文件(WAV/PCM，16kHz/16bit/mono)"),
    model: str | None = Form(default=None, max_length=64, description="ASR 模型(默认 paraformer)"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    audio = await file.read()
    result = await asr_service.offline_asr(redis, db, user.id, audio, model)
    return success(result)


@router.websocket("/ws/asr")
async def ws_asr_endpoint(
    websocket: WebSocket,
    sessionId: str = Query(..., description="语音会话ID（stream-session 创建）"),
):
    """流式 ASR WebSocket：鉴权/上行 PCM/下行增量结果编排见 asr_service.handle_stream_websocket"""
    await asr_service.handle_stream_websocket(websocket, sessionId)


# ==================== TTS（F-VS-002 §2.2） ====================


@router.post("/api/v1/voice/tts", summary="文本转语音")
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


@router.get("/api/v1/voice/tts/voices", summary="可用音色列表")
async def list_voices(
    user: UserContext = Depends(get_current_user),
):
    return success(VOICE_CATALOG)


@router.get("/api/v1/voice/tts/audio/{cache_key}", summary="缓存音频下载（解密后流式返回）")
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


# ==================== 热词（F-VS-004 §2.3） ====================


def _check_admin(user: UserContext) -> None:
    """管理员身份校验：非管理员抛出 A0301"""
    if not user.is_admin:
        raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "仅管理员可操作")


@router.get("/api/v1/voice/hotwords", summary="查询用户热词列表")
async def list_user_hotwords(
    db: AsyncSession = Depends(get_db),
    ctx: UserContext = Depends(get_current_user),
):
    return success(await hotword_service.list_user_hotwords(db, ctx.id))


@router.post("/api/v1/voice/hotwords", summary="新增用户热词")
@require_permission("voice:hotword:edit")
async def add_user_hotword(
    form: HotwordForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(await hotword_service.add_user_hotword(db, user.id, form))


@router.delete("/api/v1/voice/hotwords/{hotword_id}", summary="删除用户热词")
@require_permission("voice:hotword:edit")
async def delete_user_hotword(
    hotword_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await hotword_service.delete_user_hotword(db, hotword_id, user.id)
    return success()


@router.get("/api/v1/voice/hotwords/global", summary="查询全局热词列表")
async def list_global_hotwords(
    db: AsyncSession = Depends(get_db),
    ctx: UserContext = Depends(get_current_user),
):
    return success(await hotword_service.list_global_hotwords(db))


@router.post("/api/v1/voice/hotwords/global", summary="新增全局热词（仅管理员）")
@require_permission("voice:hotword:edit")
async def add_global_hotword(
    form: HotwordForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    _check_admin(user)
    return success(await hotword_service.add_global_hotword(db, form))


@router.delete("/api/v1/voice/hotwords/global/{hotword_id}", summary="删除全局热词（仅管理员）")
@require_permission("voice:hotword:edit")
async def delete_global_hotword(
    hotword_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    _check_admin(user)
    await hotword_service.delete_global_hotword(db, hotword_id)
    return success()


# ==================== 服务状态（§2.4，voice:service:monitor） ====================


@router.get("/api/v1/voice/service/status", summary="查询语音服务状态")
@require_permission("voice:service:monitor")
async def get_voice_service_status(
    user: UserContext = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """查询语音服务状态（ASR/TTS 引擎状态 + ASR 并发会话数）

    引擎不可用时正常返回 offline，不抛异常（T-VS-067）。
    """
    return success(await voice_service_status.get_status(redis))
