"""语音交互 ASR 服务：流式/离线识别编排

对齐《后端实现.md》§3 与 API 契约：
- 流式：创建会话（并发校验/计费预校验/热词注册）→ WebSocket 上行 PCM 下行增量 JSON
- 离线：multipart 直传 WAV/PCM → FunASR 识别 → 返回文本（音频处理完即弃不落盘）
- 会话状态存 Redis（voice:asr:{sessionId}），结束保留 30 分钟用于结果查询；
  并发计数用有序集合 voice:asr:sessions（按时间戳剪枝后 ZCARD）
- 计费：创建时 ensure_balance 预校验（按 10 秒起步），结束/识别完成时 charge_asr 按秒实扣

本模块用 get_db_session 手动管理事务（WebSocket/离线场景无请求级事务）。
"""

import asyncio
import json
import logging
import math
import time
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.voice.provider.registry import voice_engine_registry
from app.service.voice.hotword_service import hotword_service
from app.service.voice.voice_billing_service import voice_billing_service

logger = logging.getLogger(__name__)

# ===== Redis 键与常量 =====
_SESSION_KEY = "voice:asr:{session_id}"
_CONCURRENT_KEY = "voice:asr:sessions"
# 会话状态保留时长（秒）：用于结果查询（后端实现 §10 记录 30 分钟）
_SESSION_TTL = 30 * 60
# 16kHz × 16bit × mono = 32000 字节/秒，用于按字节折算音频时长
_BYTES_PER_SECOND = 16000 * 2
# 单块音频最大字节（10MB），超限拒绝
_MAX_BLOCK_BYTES = 10 * 1024 * 1024
# 计费预校验预估秒数（流式音频时长不可预知，按起步秒预估）
_ESTIMATE_SECONDS = 10


class AsrService:
    """ASR 编排服务"""

    def __init__(
        self,
        hotword_service=hotword_service,
        voice_billing_service=voice_billing_service,
        engine_registry=voice_engine_registry,
    ):
        self.hotword_service = hotword_service
        self.voice_billing_service = voice_billing_service
        self.engine_registry = engine_registry

    # ==================== 会话状态与并发（Redis） ====================

    def _session_key(self, session_id: str) -> str:
        return _SESSION_KEY.format(session_id=session_id)

    async def _load_session(self, redis: Redis, session_id: str) -> dict[str, Any] | None:
        raw = await redis.get(self._session_key(session_id))
        if not raw:
            return None
        return json.loads(raw)

    async def _save_session(self, 
        redis: Redis, session_id: str, data: dict[str, Any], *, ttl: int | None = _SESSION_TTL
    ) -> None:
        await redis.set(
            self._session_key(session_id),
            json.dumps(data, ensure_ascii=False),
            ex=ttl,
        )

    async def _check_concurrency(self, redis: Redis) -> None:
        """校验并发会话数不超过上限（有序集合按时间戳剪枝后计数）"""
        now = time.time()
        # 剪枝过期成员（保留窗口 = 会话 TTL），防止死会话长期占用并发名额
        await redis.zremrangebyscore(_CONCURRENT_KEY, 0, now - _SESSION_TTL)
        count = await redis.zcard(_CONCURRENT_KEY)
        if count >= settings.VOICE_ASR_MAX_CONCURRENT_SESSIONS:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "ASR 并发会话数已达上限")

    # ==================== 创建流式会话 ====================

    async def create_stream_session(self, 
        redis: Redis, db: AsyncSession, user_id: int, model: str | None
    ) -> str:
        """创建流式 ASR 会话：并发校验、计费预校验、注册热词，返回 sessionId

        连接 FunASR 延迟到 WebSocket 建立时（stream-session 仅创建会话元数据），
        wsUrl 由路由层基于请求地址构建。
        """
        await self._check_concurrency(redis)
        # 计费预校验（预估按 10 秒起步），不足直接拒绝
        await self.voice_billing_service.ensure_balance(
            db, user_id, math.ceil(_ESTIMATE_SECONDS * settings.VOICE_ASR_CREDITS_PER_SECOND)
        )

        session_id = uuid.uuid4().hex
        now = time.time()
        await self._save_session(
            redis,
            session_id,
            {
                "user_id": user_id,
                "status": "processing",
                "text": "",
                "model": model or "",
                "create_time": now,
            },
        )
        # 加入并发有序集合（score=创建时间戳）
        await redis.zadd(_CONCURRENT_KEY, {session_id: now})
        # 清理历史遗留的并发计数（避免上一步 ZADD 后集合膨胀）
        await redis.zremrangebyscore(_CONCURRENT_KEY, 0, now - _SESSION_TTL)

        # 注册热词（合并全局+用户级），失败仅告警不阻断会话创建
        provider = await self.engine_registry.get_asr_provider()
        await self._register_hotwords(db, user_id, provider)

        return session_id

    async def _register_hotwords(self, db: AsyncSession, user_id: int, provider) -> None:
        """合并全局+用户级热词注册到 ASR Provider；调用失败仅告警。"""
        try:
            words = await self.hotword_service.get_effective_words(db, user_id)
            if words:
                await provider.register_hotwords(words)
        except Exception as e:  # 热词注册失败不阻断会话
            logger.warning("注册热词失败(不影响会话) user_id=%s error=%s", user_id, e)

    # ==================== 查询结果 ====================

    async def get_result(self, redis: Redis, session_id: str, user_id: int) -> dict[str, str]:
        """查询流式 ASR 会话最终识别结果（校验会话归属，避免跨用户访问）"""
        session = await self._load_session(redis, session_id)
        if not session:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "ASR 会话不存在")
        if int(session["user_id"]) != user_id:
            raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权访问该 ASR 会话")
        return {
            "sessionId": session_id,
            "text": session.get("text", ""),
            "status": session.get("status", "processing"),
        }

    # ==================== 离线 ASR ====================

    async def offline_asr(self, 
        redis: Redis, db: AsyncSession, user_id: int, audio: bytes, model: str | None
    ) -> dict[str, str]:
        """离线识别完整音频（multipart 直传），处理完即弃不落盘，完成时按秒实扣。"""
        self._validate_audio(audio)
        # 计费预校验（预估按 10 秒起步）
        await self.voice_billing_service.ensure_balance(
            db, user_id, math.ceil(_ESTIMATE_SECONDS * settings.VOICE_ASR_CREDITS_PER_SECOND)
        )
        # 注册热词（离线识别也应用领域热词），失败仅告警
        provider = await self.engine_registry.get_asr_provider()
        await self._register_hotwords(db, user_id, provider)

        try:
            text = await provider.recognize_offline(audio)
        except Exception as e:
            raise BusinessException(ResultCode.BUSINESS_ERROR, f"离线识别失败: {e}") from e

        # 按音频时长（秒）实扣，音频字节数折算为秒向上取整
        audio_seconds = math.ceil(len(audio) / _BYTES_PER_SECOND)
        await self._charge(db, user_id, audio_seconds)

        session_id = uuid.uuid4().hex
        await self._save_session(
            redis,
            session_id,
            {
                "user_id": user_id,
                "status": "completed",
                "text": text,
                "model": "",
                "create_time": time.time(),
            },
        )
        return {"sessionId": session_id, "text": text}

    def _validate_audio(self, audio: bytes) -> None:
        """校验离线音频：仅接受 WAV（RIFF 魔数）/PCM；空文件或超限抛参数错误。"""
        if not audio:
            raise BusinessException(ResultCode.PARAM_ERROR, "音频文件为空")
        if len(audio) > settings.MAX_UPLOAD_SIZE:
            raise BusinessException(ResultCode.PARAM_ERROR, "音频文件大小超限")
        # WAV 必须以 RIFF 开头；PCM 为纯裸流（无头），通过字节数 >0 兜底
        if audio[:4] != b"RIFF" and len(audio) < _BYTES_PER_SECOND // 10:
            raise BusinessException(ResultCode.PARAM_ERROR, "仅支持 WAV/PCM 音频格式")

    # ==================== 流式 WebSocket 会话 ====================

    async def handle_stream_websocket(self, websocket: WebSocket, session_id: str) -> None:
        """流式 ASR WebSocket 会话编排：鉴权、双向代理、超时/时长控制、计费与状态落库。

        协议（前端 ↔ 业务后端）：
        - 上行：二进制 PCM（16kHz/16bit/mono）；文本 "EOS" 结束
        - 下行：JSON {"text": 增量, "isFinal": bool}
        """
        try:
            redis = await get_redis_client()
        except Exception as e:  # noqa: BLE001
            logger.error("获取 Redis 失败: %s", e)
            await self._reject(websocket, "服务不可用，请稍后重试")
            return

        session = await self._load_session(redis, session_id)
        if not session:
            await self._reject(websocket, "ASR 会话不存在或已过期")
            return

        user_id = int(session["user_id"])

        # 经注册表获取 ASR Provider 并建立流式会话，失败则标记会话 failed 并关闭前端
        try:
            provider = await self.engine_registry.get_asr_provider()
            funasr = await provider.recognize_stream()
        except Exception as e:
            logger.error("连接 ASR 失败 session=%s error=%s", session_id, e)
            await self._fail_session(redis, session_id)
            await self._reject(websocket, "识别服务不可用，请稍后重试")
            return

        try:
            await self._run_stream(websocket, redis, session_id, user_id, funasr)
        except WebSocketDisconnect:
            logger.info("客户端断开流式 ASR 会话 session=%s", session_id)
        except asyncio.CancelledError:
            logger.info("流式 ASR 会话被取消 session=%s", session_id)
            raise
        except Exception as e:  # noqa: BLE001 - 异常需兜底回收资源并标记失败
            logger.error("流式 ASR 会话异常 session=%s error=%s", session_id, e, exc_info=True)
            await self._fail_session(redis, session_id)
        finally:
            await redis.zrem(_CONCURRENT_KEY, session_id)
            logger.info("流式 ASR 会话结束 session=%s", session_id)

    async def _reject(self, websocket: WebSocket, message: str) -> None:
        """鉴权失败/会话无效：accept 后发 error 并以 4001 关闭。"""
        try:
            await websocket.accept()
            await websocket.send_json({"type": "error", "message": message})
            await websocket.close(code=4001)
        except Exception as e:  # noqa: BLE001
            logger.warning("拒绝 WebSocket 连接异常: %s", e)

    async def _run_stream(self, 
        websocket: WebSocket,
        redis: Redis,
        session_id: str,
        user_id: int,
        funasr,
    ) -> None:
        """执行双向流式代理，返回后由调用方完成计费与状态落库。

        客户端断开/达最大时长/空闲超时/收到 EOS 均走正常结束路径：
        按已接收音频时长计费并将会话置为 completed（断开时保留已识别的部分文本）。
        """
        await websocket.accept()

        # 下行：FunASR 增量文本 → 前端；收到最终结果后结束
        downlink_done = asyncio.Event()
        final_holder: dict[str, str] = {}

        async def pump_downlink():
            try:
                async for raw in funasr.recv_messages():
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("FunASR 返回非 JSON: %s session=%s", raw[:100], session_id)
                        continue
                    text = msg.get("text", "")
                    is_final = bool(msg.get("is_final", msg.get("isFinal", False)))
                    try:
                        await websocket.send_json({"text": text, "isFinal": is_final})
                    except Exception:  # 客户端已断开，停止转发
                        break
                    if is_final:
                        final_holder["text"] = text
                        break
            finally:
                downlink_done.set()

        downlink_task = asyncio.create_task(pump_downlink())
        total_bytes = 0
        try:
            # 上行：前端音频/结束信号 → FunASR，并受空闲/时长/单块上限约束
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive(), timeout=settings.VOICE_ASR_WS_IDLE_TIMEOUT
                    )
                except TimeoutError:
                    logger.info("流式 ASR 空闲超时关闭 session=%s", session_id)
                    break
                except WebSocketDisconnect:
                    logger.info("客户端断开流式 ASR 会话 session=%s", session_id)
                    break

                if "bytes" in data and data["bytes"] is not None:
                    chunk = data["bytes"]
                    if len(chunk) > _MAX_BLOCK_BYTES:
                        logger.warning("音频块超限拒绝 session=%s", session_id)
                        continue
                    total_bytes += len(chunk)
                    await funasr.send_audio(chunk)
                    # 累计音频达最大时长：发送 EOS 结束，FunASR 输出最终结果
                    if total_bytes >= _BYTES_PER_SECOND * settings.VOICE_ASR_MAX_DURATION:
                        logger.info("流式 ASR 达最大时长自动结束 session=%s", session_id)
                        await funasr.send_eos()
                        break
                elif "text" in data and data["text"]:
                    if data["text"].strip() == "EOS":
                        await funasr.send_eos()
                        break
        finally:
            # 等待下行把最终结果推送完（或超时取消），并取回下行任务异常避免告警
            try:
                await asyncio.wait_for(downlink_done.wait(), timeout=10)
            except TimeoutError:
                downlink_task.cancel()
            if downlink_task.done() and not downlink_task.cancelled():
                exc = downlink_task.exception()
                if exc:
                    logger.warning("FunASR 下行异常 session=%s error=%s", session_id, exc)

        final_text = final_holder.get("text", "")
        audio_seconds = math.ceil(total_bytes / _BYTES_PER_SECOND)
        await self._complete_session(redis, session_id, user_id, final_text, audio_seconds)

    async def _complete_session(self, 
        redis: Redis, session_id: str, user_id: int, text: str, audio_seconds: int
    ) -> None:
        """会话正常结束：落库状态（completed）+ 按秒实扣。"""
        session = await self._load_session(redis, session_id) or {}
        session["status"] = "completed"
        session["text"] = text
        await self._save_session(redis, session_id, session)
        if audio_seconds > 0:
            await self._charge(None, user_id, audio_seconds)

    async def _fail_session(self, redis: Redis, session_id: str) -> None:
        """会话失败：更新状态为 failed。"""
        session = await self._load_session(redis, session_id) or {}
        session["status"] = "failed"
        await self._save_session(redis, session_id, session)

    async def _charge(self, db: AsyncSession | None, user_id: int, audio_seconds: int) -> None:
        """按音频时长实扣（WebSocket 场景 db=None 用 get_db_session 手动事务）。"""
        try:
            if db is None:
                async with get_db_session() as s:
                    await self.voice_billing_service.charge_asr(s, user_id, audio_seconds)
            else:
                await self.voice_billing_service.charge_asr(db, user_id, audio_seconds)
        except Exception as e:  # noqa: BLE001 - 计费失败不影响识别结果返回
            logger.error("ASR 计费失败 user_id=%s seconds=%s error=%s", user_id, audio_seconds, e)


asr_service = AsrService()
