"""语音交互流式 ASR WebSocket 路由

端点: /ws/asr?sessionId={voiceSessionId}
- 鉴权：sessionId 必须命中 Redis 中的语音会话（stream-session 创建时写入并绑定 user_id），
  否则 accept 后发 {"type":"error","message":"..."} 并以 4001 关闭
- 上行：二进制 PCM 音频块（16kHz/16bit/mono）直传 FunASR；文本 "EOS" 结束
- 下行：JSON {"text": 增量, "isFinal": bool}

协议编排与资源回收由 AsrService.handle_stream_websocket 负责。
"""

from fastapi import APIRouter, Query, WebSocket

from app.service.voice.asr_service import AsrService

router = APIRouter(tags=["语音交互-ASR"])


@router.websocket("/ws/asr")
async def ws_asr_endpoint(
    websocket: WebSocket,
    sessionId: str = Query(..., description="语音会话ID（stream-session 创建）"),
):
    await AsrService.handle_stream_websocket(websocket, sessionId)
