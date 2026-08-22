"""语音交互 ASR 路由

基础路径: /api/v1/voice/asr
对齐《API接口.md》§2.1：
- POST /asr/stream-session 创建流式 ASR 会话（获取 WebSocket 连接信息）
- GET  /asr/result/{sessionId} 查询流式 ASR 会话最终识别结果
- POST /asr/offline 离线 ASR 识别（multipart 直传音频文件）

wsUrl 从 request.base_url 推导（http→ws），形如 ws://{host}/ws/asr?sessionId={sessionId}。
"""

from fastapi import APIRouter, Depends, File, Form, Path, Request, UploadFile
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.voice_asr import StreamAsrSessionForm
from app.service.voice.asr_service import AsrService

router = APIRouter(
    prefix="/api/v1/voice/asr",
    tags=["语音交互-ASR"],
    dependencies=[Depends(get_current_user)],
)


def _build_ws_url(request: Request, session_id: str) -> str:
    """将 HTTP base_url 推导为 WebSocket 地址（http→ws，https→wss）"""
    scheme = "wss" if request.url.scheme == "https" else "ws"
    host = request.url.netloc
    return f"{scheme}://{host}/ws/asr?sessionId={session_id}"


@router.post("/stream-session", summary="创建流式 ASR 会话")
async def create_stream_session(
    body: StreamAsrSessionForm,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    session_id = await AsrService.create_stream_session(redis, db, user.id, body.model)
    return success({"sessionId": session_id, "wsUrl": _build_ws_url(request, session_id)})


@router.get("/result/{session_id}", summary="查询流式 ASR 会话最终识别结果")
async def get_asr_result(
    session_id: str = Path(..., description="流式 ASR 会话ID"),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await AsrService.get_result(redis, session_id, user.id)
    return success(result)


@router.post("/offline", summary="离线 ASR 识别（multipart 直传音频文件）")
async def offline_asr(
    file: UploadFile = File(..., description="音频文件(WAV/PCM，16kHz/16bit/mono)"),
    model: str | None = Form(default=None, max_length=64, description="ASR 模型(默认 paraformer)"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    audio = await file.read()
    result = await AsrService.offline_asr(redis, db, user.id, audio, model)
    return success(result)
