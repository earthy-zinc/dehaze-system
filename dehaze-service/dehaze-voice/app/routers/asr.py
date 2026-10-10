"""流式 ASR 端点：WS `/v1/asr/stream` 保留现有自定义帧协议。

离线识别已迁至 `/v1/audio/transcriptions`（`app/routers/audio.py`）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import catalog
from app.config import settings
from app.engines import asr as asr_engine
from app.errors import VoiceError
from app.logging import set_request_context
from app.metrics import INFERENCE_DURATION
from app.security import authorize_websocket

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/asr", tags=["asr"])

# WS 关闭码：4000 段为应用自定义，与编排侧对前端的 4001 契约区分
_CLOSE_CODES = {
    "PARAM_ERROR": 4400,
    "UNAUTHORIZED": 4401,
    "SESSION_CONFLICT": 4409,
    "INFERENCE_FAILED": 4500,
    "MODEL_UNAVAILABLE": 4503,
}

# 活跃会话（sessionId 幂等键）：仅本实例内生效，编排侧保证同一会话路由到同一实例
_active_sessions: set[str] = set()


@router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    """流式识别（两跳的第二跳）。

    - 握手头携带 Authorization: Bearer（query 仅 model_id）；未认证直接拒绝握手
    - 首条控制消息 {"type":"config","hotwords":[...],"session_id":"可选幂等键"}
    - 上行二进制 PCM（16kHz/16bit/mono）或文本 EOS；下行 {"text","is_final"}
    - 超时只有"无输入自动收尾"兜底，空闲 60s / 最长 60s / 下行收尾 10s 归编排侧
    """
    # 先 accept 再鉴权：pre-accept close 会被 uvicorn 升级为 HTTP 403 拒绝握手，
    # 编排侧把 403 归因为 Key 失效而触发冷却级联；accept 后按 4401 关闭同样是
    # "拒绝未认证连接"，且不向其发送任何业务帧（协议细节不外泄语义不变）
    await websocket.accept()
    if not authorize_websocket(websocket.headers.get("authorization")):
        await websocket.close(code=4401)
        return

    model_id = websocket.query_params.get("model_id", "").strip()
    set_request_context(method="WS", path="/v1/asr/stream")
    if not model_id:
        await _reject(websocket, "PARAM_ERROR", "缺少 model_id 查询参数")
        return
    # model_id 取值口径为统一模型表标识（如 sensevoice），真实 ModelScope 标识由服务端目录解析
    try:
        model_id = catalog.resolve_asr_model(model_id)
    except VoiceError as e:
        await _reject(websocket, "PARAM_ERROR", e.message)
        return

    session_id: str | None = None
    session: asr_engine.StreamSession | None = None
    started = time.perf_counter()
    try:
        config = await _read_config(websocket)
        if config is None:
            return
        session_id = config["session_id"]
        if session_id:
            if session_id in _active_sessions:
                await _reject(websocket, "SESSION_CONFLICT", f"会话已在进行中: {session_id}")
                return
            _active_sessions.add(session_id)

        session = asr_engine.StreamSession(model_id, config["hotwords"])
        await asr_engine.preload(model_id)
        logger.info(
            "流式 ASR 会话开始",
            extra={
                "model_id": model_id,
                "session_id": session_id,
                "hotwords": len(config["hotwords"]),
            },
        )
        await _pump(websocket, session, session_id)
    except WebSocketDisconnect:
        logger.info("编排侧断开流式 ASR 会话", extra={"session_id": session_id})
    except VoiceError as e:
        logger.error(
            "流式 ASR 会话失败: %s", e.message, extra={"code": e.code, "session_id": session_id}
        )
        await _reject(websocket, e.code, e.message)
    except Exception as e:
        logger.error("流式 ASR 会话异常 session_id=%s error=%s", session_id, e, exc_info=True)
        await _reject(websocket, "INFERENCE_FAILED", str(e))
    finally:
        if session_id:
            _active_sessions.discard(session_id)
        INFERENCE_DURATION.labels("asr_stream").observe(time.perf_counter() - started)
        if session is not None:
            logger.info(
                "流式 ASR 会话结束",
                extra={
                    "model_id": model_id,
                    "session_id": session_id,
                    "audio_bytes": session.audio_bytes,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                },
            )


async def _read_config(websocket: WebSocket) -> dict | None:
    """读取首条控制消息；非法时发错误帧并关闭，返回 None"""
    try:
        message = await asyncio.wait_for(
            websocket.receive(), timeout=settings.ASR_STREAM_IDLE_TIMEOUT
        )
    except TimeoutError:
        await _reject(websocket, "PARAM_ERROR", "未在超时内收到 config 控制消息")
        return None
    if message["type"] == "websocket.disconnect":
        return None
    raw = message.get("text")
    if raw is None:
        await _reject(websocket, "PARAM_ERROR", "首条消息必须是 config 控制消息")
        return None
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        await _reject(websocket, "PARAM_ERROR", f"config 控制消息不是合法 JSON: {e}")
        return None
    if not isinstance(config, dict) or config.get("type") != "config":
        await _reject(websocket, "PARAM_ERROR", '首条消息必须是 {"type":"config"} 控制消息')
        return None
    hotwords = config.get("hotwords") or []
    if not isinstance(hotwords, list) or not all(isinstance(word, str) for word in hotwords):
        await _reject(websocket, "PARAM_ERROR", "config.hotwords 必须为字符串数组")
        return None
    session_id = config.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        await _reject(websocket, "PARAM_ERROR", "config.session_id 必须为字符串")
        return None
    return {"hotwords": hotwords, "session_id": session_id or None}


async def _pump(
    websocket: WebSocket, session: asr_engine.StreamSession, session_id: str | None
) -> None:
    """上行 PCM/EOS → 分段增量转写 → 下行累计文本；无输入超时兜底收尾"""
    while True:
        try:
            message = await asyncio.wait_for(
                websocket.receive(), timeout=settings.ASR_STREAM_IDLE_TIMEOUT
            )
        except TimeoutError:
            logger.info("流式 ASR 无输入超时自动收尾", extra={"session_id": session_id})
            break
        if message["type"] == "websocket.disconnect":
            logger.info("编排侧断开流式 ASR 会话", extra={"session_id": session_id})
            return  # 断连不追求最终结果：编排侧按已收字节自行收尾计费
        chunk = message.get("bytes")
        if chunk is not None:
            if session.feed(chunk):
                text = await session.transcribe_pending()
                await websocket.send_json({"text": text, "is_final": False})
            continue
        if (message.get("text") or "").strip() == "EOS":
            break
        logger.warning("忽略非 EOS 文本上行", extra={"session_id": session_id})
    await websocket.send_json({"text": await session.transcribe_pending(), "is_final": True})
    await websocket.close()  # 收尾后正常关闭，编排侧据最终结果与会话关闭收尾计费


async def _reject(websocket: WebSocket, code: str, message: str) -> None:
    """已 accept 的连接：发错误帧后按语义码关闭（错误模型与 HTTP 侧一致）"""
    try:
        await websocket.send_json({"code": code, "message": message})
        await websocket.close(code=_CLOSE_CODES.get(code, 4500))
    except Exception as e:  # 对端已断开时发送/关闭失败不影响收尾
        logger.warning("关闭引擎侧流式会话失败 code=%s error=%s", code, e)
