"""语音交互 ASR 服务：流式/离线识别编排

对齐《后端实现.md》§3 与 API 契约：
- 流式：创建会话（并发校验/计费预校验）→ WebSocket 上行 PCM 下行增量 JSON
- 离线：multipart 直传 WAV/PCM → 引擎识别 → 返回文本（音频处理完即弃不落盘）
- 会话状态存 Redis（voice:asr:{sessionId}），结束保留 30 分钟用于结果查询；
  并发计数用有序集合 voice:asr:sessions（按时间戳剪枝后 ZCARD）
- 计费：创建时 ensure_balance 预校验（按 10 秒起步），结束/识别完成时 charge_asr 按秒实扣

识别经 Provider 转发到外置引擎服务（clarity-service/clarity-voice）：推理参数（ModelScope
model_id）与热词随请求/会话下发，编排侧保留并发、计费、会话态与超时控制。
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
from app.dependencies.auth import SESSION_PREFIX
from app.dependencies.redis import get_redis_client
from app.infrastructure.llm.client.audio_dispatch import AudioStreamSession, get_audio_client
from app.infrastructure.llm.client.openai_audio_client import EngineStreamClosed
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.voice.hotword_service import hotword_service
from app.service.voice.voice_billing_service import voice_billing_service
from app.service.voice.voice_model_route import VoiceRoute, resolve_voice_model

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

# 常见非 PCM 二进制格式魔数（伪装扩展名文件拒绝：图片/文档/压缩包/可执行/MP3/OGG/FLAC）
_BINARY_MAGICS = (
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF8",
    b"%PDF",
    b"PK\x03\x04",
    b"MZ",
    b"\x1f\x8b",
    b"\x7fELF",
    b"ID3",
    b"OggS",
    b"fLaC",
)


class AsrService:
    """ASR 编排服务"""

    def __init__(
        self,
        hotword_service=hotword_service,
        voice_billing_service=voice_billing_service,
    ):
        self.hotword_service = hotword_service
        self.voice_billing_service = voice_billing_service

    # ==================== 会话状态与并发（Redis） ====================

    def _session_key(self, session_id: str) -> str:
        return _SESSION_KEY.format(session_id=session_id)

    async def _load_session(self, redis: Redis, session_id: str) -> dict[str, Any] | None:
        raw = await redis.get(self._session_key(session_id))
        if not raw:
            return None
        return json.loads(raw)

    async def _save_session(
        self, redis: Redis, session_id: str, data: dict[str, Any], *, ttl: int | None = _SESSION_TTL
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

    async def create_stream_session(
        self, redis: Redis, db: AsyncSession, user_id: int, model: str | None
    ) -> str:
        """创建流式 ASR 会话：并发校验、计费预校验，返回 sessionId

        连接引擎延迟到 WebSocket 建立时（stream-session 仅创建会话元数据），
        wsUrl 由路由层基于请求地址构建；热词在 WS 建连时随会话下发给引擎。
        """
        await self._check_concurrency(redis)
        # 统一模型路由（asr 平台默认模型），会话元数据记录 model_id/provider_id 供引擎建连与计费
        route = await resolve_voice_model(db, "asr")
        # 计费预校验（预估按 10 秒起步），不足直接拒绝
        await self.voice_billing_service.ensure_balance(
            db,
            user_id,
            await self.voice_billing_service.estimate_asr_credits(
                db, route.model_id, route.provider.id, _ESTIMATE_SECONDS
            ),
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
                "model_id": route.model_id,
                "provider_id": route.provider.id,
                "create_time": now,
            },
        )
        # 加入并发有序集合（score=创建时间戳）
        await redis.zadd(_CONCURRENT_KEY, {session_id: now})
        # 清理历史遗留的并发计数（避免上一步 ZADD 后集合膨胀）
        await redis.zremrangebyscore(_CONCURRENT_KEY, 0, now - _SESSION_TTL)

        return session_id

    async def _effective_hotwords(self, user_id: int, db: AsyncSession | None = None) -> list[str]:
        """生效热词（全局+用户级）：随请求/会话下发给引擎

        WS 场景无请求级事务（db 为空时自建会话）；获取失败仅告警，不阻断识别。
        """
        try:
            if db is not None:
                return await self.hotword_service.get_effective_words(db, user_id)
            async with get_db_session() as session:
                return await self.hotword_service.get_effective_words(session, user_id)
        except Exception as e:
            logger.warning("获取热词失败(不影响会话) user_id=%s error=%s", user_id, e)
            return []

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

    async def offline_asr(
        self, redis: Redis, db: AsyncSession, user_id: int, audio: bytes, model: str | None
    ) -> dict[str, str]:
        """离线识别完整音频（multipart 直传），处理完即弃不落盘，完成时按秒实扣。"""
        self._validate_audio(audio)
        # 统一模型路由（asr 平台默认模型）
        route = await resolve_voice_model(db, "asr")
        # 计费预校验（预估按 10 秒起步）
        await self.voice_billing_service.ensure_balance(
            db,
            user_id,
            await self.voice_billing_service.estimate_asr_credits(
                db, route.model_id, route.provider.id, _ESTIMATE_SECONDS
            ),
        )
        # 热词（全局+用户级）随请求下发给引擎，失败仅告警
        hotwords = await self._effective_hotwords(user_id, db)

        try:
            text = await get_audio_client(route.provider.protocol_type).transcribe(
                route.provider, route.model_id, audio, hotwords
            )
        except BusinessException:
            raise
        except Exception as e:
            raise BusinessException(ResultCode.BUSINESS_ERROR, f"离线识别失败: {e}") from e

        # 按音频时长（秒）实扣，音频字节数折算为秒向上取整
        audio_seconds = math.ceil(len(audio) / _BYTES_PER_SECOND)
        await self._charge(db, user_id, audio_seconds, route)

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
        """校验离线音频：仅接受 WAV（RIFF 魔数）/PCM；空文件/超限/伪装文件抛参数错误。"""
        if not audio:
            raise BusinessException(ResultCode.PARAM_ERROR, "音频文件为空")
        if len(audio) > settings.MAX_UPLOAD_SIZE:
            raise BusinessException(ResultCode.PARAM_ERROR, "音频文件大小超限")
        if audio[:4] == b"RIFF":
            return
        # PCM 为纯裸流（无头），无法校验魔数；但常见二进制格式（伪装扩展名的
        # 图片/文档/压缩包/其他容器音频）可按魔数直接拒绝，避免垃圾输入占用
        # 推理线程池并被误扣费。16bit 采样字节数必须为偶数。
        if any(audio.startswith(magic) for magic in _BINARY_MAGICS) or len(audio) % 2 != 0:
            raise BusinessException(ResultCode.PARAM_ERROR, "仅支持 WAV/PCM 音频格式")
        if len(audio) < _BYTES_PER_SECOND // 10:
            raise BusinessException(ResultCode.PARAM_ERROR, "仅支持 WAV/PCM 音频格式")

    # ==================== 流式 WebSocket 会话 ====================

    async def handle_stream_websocket(self, websocket: WebSocket, session_id: str) -> None:
        """流式 ASR WebSocket 会话编排：鉴权、双向代理、超时/时长控制、计费与状态落库。

        协议（前端 ↔ 业务后端）：
        - 握手：URL 携带 sessionId 与登录会话凭证 sid，建立时校验登录态与
          会话归属（后端实现 §6.1），失败以 4001 关闭
        - 上行：二进制 PCM（16kHz/16bit/mono）；文本 "EOS" 结束
        - 下行：JSON {"text": 增量, "isFinal": bool}
        """
        try:
            redis = await get_redis_client()
        except Exception as e:
            logger.error("获取 Redis 失败: %s", e)
            await self._reject(websocket, "服务不可用，请稍后重试")
            return

        # WebSocket 建连鉴权：校验登录态并确认与 ASR 会话归属一致
        sid_user_id = await self._resolve_sid_user_id(redis, websocket.query_params.get("sid"))
        if sid_user_id is None:
            await self._reject(websocket, "登录态无效或已过期，请重新登录后再试")
            return

        session = await self._load_session(redis, session_id)
        if not session:
            await self._reject(websocket, "ASR 会话不存在或已过期")
            return
        if int(session["user_id"]) != sid_user_id:
            await self._reject(websocket, "无权访问该 ASR 会话")
            return

        user_id = int(session["user_id"])

        # 从会话元数据恢复统一模型路由（model_id/provider_id 由创建会话时解析落 Redis），
        # 经语音协议适配器（按供应商 protocol_type 分派）建立引擎侧流式会话
        # （热词与幂等键随会话下发），失败则标记会话 failed 并关闭前端
        try:
            route = await self._restore_route(session)
            hotwords = await self._effective_hotwords(user_id)
            engine = await get_audio_client(route.provider.protocol_type).open_stream(
                route.provider, route.model_id, hotwords, session_id
            )
        except Exception as e:
            logger.error("连接 ASR 引擎失败 session=%s error=%s", session_id, e)
            await self._fail_session(redis, session_id)
            await self._reject(websocket, "识别服务不可用，请稍后重试")
            return

        try:
            await self._run_stream(websocket, redis, session_id, user_id, engine, route)
        except WebSocketDisconnect:
            logger.info("客户端断开流式 ASR 会话 session=%s", session_id)
        except asyncio.CancelledError:
            logger.info("流式 ASR 会话被取消 session=%s", session_id)
            raise
        except Exception as e:
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
        except Exception as e:
            logger.warning("拒绝 WebSocket 连接异常: %s", e)

    async def _resolve_sid_user_id(self, redis: Redis, sid: str | None) -> int | None:
        """由登录会话凭证（sid）解析用户 ID，凭证缺失/会话过期返回 None。"""
        if not sid:
            return None
        raw = await redis.get(SESSION_PREFIX + sid)
        if not raw:
            return None
        login_session = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        user_id = login_session.get("userId")
        return int(user_id) if user_id else None

    async def _run_stream(
        self,
        websocket: WebSocket,
        redis: Redis,
        session_id: str,
        user_id: int,
        engine: AudioStreamSession,
        route: VoiceRoute,
    ) -> None:
        """执行两跳双向流式代理，结束后写入会话终态。

        客户端断开/达最大时长/空闲超时/收到 EOS 均走正常结束路径：向引擎发 EOS 求得最终
        文本，按已接收音频时长计费并将会话置为 completed（断开时保留已识别的部分文本）。
        引擎侧断开同样按收尾处理（不再发 EOS），不使会话失败。

        引擎错误帧，或建连后未产出任何识别结果即断开，说明识别未起效（如模型加载失败）：
        向前端以 4001 关闭并把会话置 failed，对齐拆分前「建连即校验可用」的语义。
        """
        await websocket.accept()

        # 下行：引擎增量文本 → 前端；收到最终结果后结束
        downlink_done = asyncio.Event()
        final_holder: dict[str, str] = {}
        # 是否收到过引擎识别结果（增量或最终）：空结果帧同样算识别已起效（静音录音）
        produced_result = False

        async def pump_downlink():
            nonlocal produced_result
            try:
                async for raw in engine.recv_messages():
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("引擎返回非 JSON: %s session=%s", raw[:100], session_id)
                        continue
                    produced_result = True
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
        downlink_error: BaseException | None = None
        total_bytes = 0
        try:
            # 上行：前端音频/结束信号 → 引擎，并受空闲/时长/单块上限约束
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive(), timeout=settings.VOICE_ASR_WS_IDLE_TIMEOUT
                    )
                except TimeoutError:
                    logger.info("流式 ASR 空闲超时关闭 session=%s", session_id)
                    await engine.send_eos()
                    break

                # receive() 以 websocket.disconnect 消息返回断连（不抛 WebSocketDisconnect），
                # 必须立即结束上行循环，否则再次 receive() 抛 RuntimeError 会把正常断开误判为失败
                if data["type"] == "websocket.disconnect":
                    logger.info("客户端断开流式 ASR 会话 session=%s", session_id)
                    await engine.send_eos()
                    break

                if "bytes" in data and data["bytes"] is not None:
                    chunk = data["bytes"]
                    if len(chunk) > _MAX_BLOCK_BYTES:
                        logger.warning("音频块超限拒绝 session=%s", session_id)
                        continue
                    total_bytes += len(chunk)
                    try:
                        await engine.send_audio(chunk)
                    except EngineStreamClosed as exc:
                        logger.warning("引擎侧断开按收尾处理 session=%s error=%s", session_id, exc)
                        break
                    # 累计音频达最大时长：发送 EOS 结束，引擎输出最终结果
                    if total_bytes >= _BYTES_PER_SECOND * settings.VOICE_ASR_MAX_DURATION:
                        logger.info("流式 ASR 达最大时长自动结束 session=%s", session_id)
                        await engine.send_eos()
                        break
                elif data.get("text"):
                    if data["text"].strip() == "EOS":
                        await engine.send_eos()
                        break
        finally:
            # 等待下行把最终结果推送完（或超时取消），并取回下行任务异常作为引擎失败信号
            try:
                await asyncio.wait_for(downlink_done.wait(), timeout=10)
            except TimeoutError:
                downlink_task.cancel()
            if downlink_task.done() and not downlink_task.cancelled():
                downlink_error = downlink_task.exception()
            await engine.aclose()

        audio_seconds = math.ceil(total_bytes / _BYTES_PER_SECOND)
        if downlink_error is not None or not produced_result:
            logger.error("ASR 引擎未产出识别结果 session=%s error=%s", session_id, downlink_error)
            await self._fail_session(redis, session_id)
            try:
                await websocket.close(code=4001)
            except Exception as e:  # 前端已断开时关闭失败不影响收尾
                logger.warning("关闭前端 WebSocket 失败 session=%s: %s", session_id, e)
            # 计费不变式：仍按已收字节实扣（引擎未起效时通常为 0 秒，不产生费用）
            if audio_seconds > 0:
                await self._charge(None, user_id, audio_seconds, route)
            return
        await self._complete_session(
            redis, session_id, user_id, final_holder.get("text", ""), audio_seconds, route
        )

    async def _restore_route(self, session: dict[str, Any]) -> VoiceRoute:
        """从会话元数据恢复统一模型路由（创建会话时已解析；此处仅按 id 复查存在性，
        供应商连接信息随 provider 行读取——测试经 monkeypatch 模块级 get_db_session 拦截）"""
        model_id = session.get("model_id")
        provider_id = session.get("provider_id")
        if not model_id or not provider_id:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, "ASR 会话缺少统一模型路由信息，请重新发起会话"
            )
        async with get_db_session() as db:
            model = await ai_model_repository.get_by_model_id(db, model_id)
            provider = await ai_provider_repository.get_by_id(db, int(provider_id))
        if model is None or provider is None:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "ASR 会话模型已不存在")
        return VoiceRoute(model=model, provider=provider)

    async def _complete_session(
        self,
        redis: Redis,
        session_id: str,
        user_id: int,
        text: str,
        audio_seconds: int,
        route: VoiceRoute,
    ) -> None:
        """会话正常结束：落库状态（completed）+ 按秒实扣。"""
        session = await self._load_session(redis, session_id) or {}
        session["status"] = "completed"
        session["text"] = text
        await self._save_session(redis, session_id, session)
        if audio_seconds > 0:
            await self._charge(None, user_id, audio_seconds, route)

    async def _fail_session(self, redis: Redis, session_id: str) -> None:
        """会话失败：更新状态为 failed。"""
        session = await self._load_session(redis, session_id) or {}
        session["status"] = "failed"
        await self._save_session(redis, session_id, session)

    async def _charge(
        self, db: AsyncSession | None, user_id: int, audio_seconds: int, route: VoiceRoute
    ) -> None:
        """按音频时长实扣（WebSocket 场景 db=None 用 get_db_session 手动事务）。"""
        try:
            if db is None:
                async with get_db_session() as s:
                    await self.voice_billing_service.charge_asr(
                        s, user_id, route.model_id, route.provider.id, audio_seconds
                    )
            else:
                await self.voice_billing_service.charge_asr(
                    db, user_id, route.model_id, route.provider.id, audio_seconds
                )
        except Exception as e:
            logger.error("ASR 计费失败 user_id=%s seconds=%s error=%s", user_id, audio_seconds, e)


asr_service = AsrService()
