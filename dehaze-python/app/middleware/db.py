"""数据库事务中间件

事务边界 = 请求边界（对齐 Go Gin middleware / Spring @Transactional 语义）：
- 请求正常完成 → 响应发送前 commit
- 请求抛出异常（被 exception handler 捕获）→ rollback
- 未捕获异常 → rollback + 继续抛出

实现原理：
纯 ASGI 中间件拦截 http.response.start 消息，在响应头发送给客户端之前
完成 commit/rollback。保证客户端收到响应时数据一定已持久化。
"""

import logging

from app.database import async_session_factory
from app.service.ai.service.conversation_search_service import sync_conversation_to_es

logger = logging.getLogger(__name__)


class DBSessionMiddleware:
    """数据库会话 & 事务管理中间件（纯 ASGI，无 BaseHTTPMiddleware 限制）"""

    def __init__(self, app, sync_conversation_to_es=sync_conversation_to_es):
        self.app = app
        self._sync_conv_to_es = sync_conversation_to_es

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        session = async_session_factory()
        scope.setdefault("state", {})
        scope["state"]["db"] = session
        scope["state"]["db_should_rollback"] = False

        response_started = False

        async def send_wrapper(message):
            nonlocal response_started
            if message["type"] == "http.response.start" and not response_started:
                response_started = True
                try:
                    if scope["state"].get("db_should_rollback", False):
                        await session.rollback()
                    else:
                        await session.commit()
                        await self._sync_conversations_to_es(session)
                except Exception as e:
                    # commit 失败时回滚，避免悬挂事务
                    logger.error("事务提交失败，已回滚: %s", e)
                    await session.rollback()
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # 未捕获异常（理论上不应到达此处，exception handler 会兜底）
            if not response_started:
                await session.rollback()
            raise
        finally:
            await session.close()

    async def _sync_conversations_to_es(self, session) -> None:
        """执行事务内登记的会话 ES 同步（见 conversation_search_service.defer_conversation_sync）。

        必须在 commit 之后执行：读模型只反映已提交数据；单个同步失败仅记
        warning 不影响响应（与推理链路后台同步的错误语义一致）。
        """
        conv_ids = session.info.get("es_sync_conv_ids")
        if not conv_ids:
            return

        for conv_id in conv_ids:
            try:
                await self._sync_conv_to_es(conv_id)
            except Exception:
                logger.warning("Conversation ES sync failed, conv_id=%s", conv_id, exc_info=True)
