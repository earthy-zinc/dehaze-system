"""会话 ES 读模型服务（CQRS 读模型编排层）

职责：承载"会话全文索引"的读模型聚合与过滤策略（业务编排），底层写入/检索
原语由 es/ai_conversation_index 提供。技术适配层与读写模型策略在此分离：
- 本模块：DB 聚合（开库、with_deleted 语义决策、消息拼接格式）、用户隔离过滤
- es/ai_conversation_index：索引定义、ensure、纯文档写入（sync_conversation）

调用方错误语义约定（保持各自现状，勿混用）：
- 推理链路后台触发（reasoning_service._schedule_conversation_sync）：失败记 warning，
  不外抛，不阻塞主流程。
- 会话服务内同步调用（ai_conversation_service）：异常随请求上抛。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.infrastructure.es.ai_conversation_index import INDEX_NAME, sync_conversation
from app.infrastructure.es.es_client import es_client
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.ai_message_repository import ai_message_repository


async def sync_conversation_to_es(conv_id: int) -> None:
    """聚合会话消息内容并同步到 ES（失败由调用方处理：推理链路后台触发记 warning，
    会话服务内同步调用异常随请求上抛）"""
    async with get_db_session() as db:
        # with_deleted=True：软删/恢复/归档等状态变化也要同步到 ES（幂等更新文档）
        conv = await ai_conversation_repository.get_by_id(db, conv_id, with_deleted=True)
        if not conv:
            return
        msgs, _ = await ai_message_repository.list_by_conversation(db, conv_id, 1, 1000)
        contents = "\n".join(f"{m.role}: {m.content}" for m in msgs if m.content)
        await sync_conversation(
            conv_id, conv.user_id, conv.title, contents, conv.status, conv.deleted
        )


def defer_conversation_sync(db: AsyncSession, conv_id: int) -> None:
    """登记会话 ES 同步，延迟到请求事务提交后由 DBSessionMiddleware 执行。

    ES 读模型必须基于已提交数据：请求事务内新开 session 读不到未提交的
    标题/状态变更（新建会话整行不可见），内联同步会把旧数据写入 ES，
    导致关键字全文检索失效。
    """
    db.info.setdefault("es_sync_conv_ids", set()).add(conv_id)


async def search_conversations(
    user_id: int,
    query: str,
    *,
    status: int | None = 1,
    page: int = 1,
    size: int = 20,
) -> tuple[list[int], int]:
    """全文检索会话，返回 (会话 ID 列表, 命中总数)。

    默认仅检索活跃会话（status=1）；ES 为必选基础设施，无命中返回空列表。
    """
    filters = [
        {"term": {"user_id": user_id}},
        {"term": {"deleted": 0}},
    ]
    if status is not None:
        filters.append({"term": {"status": status}})

    hits, total = await es_client.paged_text_search(
        INDEX_NAME, query, filters, page=page, size=size
    )
    return [h["id"] for h in hits], total
