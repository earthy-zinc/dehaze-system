"""会话 ES 全文索引定义与读写原语（技术适配层）

索引名：ai_conversation
字段：id, user_id, title, message_contents, last_message_at, status, deleted
只承载索引定义（INDEX_NAME/_MAPPINGS）、ensure、纯文档写入（sync_conversation）；
聚合与过滤策略（DB 会话打开、with_deleted 语义、消息拼接格式、用户隔离过滤）
见 service 层 conversation_search_service，勿在此引入 repository / db session。
"""

import logging
from datetime import datetime

from app.infrastructure.es.es_client import es_client

logger = logging.getLogger(__name__)

INDEX_NAME = "ai_conversation"

_MAPPINGS = {
    "properties": {
        "id": {"type": "long"},
        "user_id": {"type": "long"},
        "title": {"type": "text"},
        "message_contents": {"type": "text"},
        "last_message_at": {"type": "date"},
        "status": {"type": "integer"},
        "deleted": {"type": "integer"},
    }
}


async def ensure_conversation_index() -> bool:
    """确保会话索引存在"""
    return await es_client.ensure_index(INDEX_NAME, _MAPPINGS)


async def sync_conversation(
    conv_id: int,
    user_id: int,
    title: str,
    message_contents: str,
    status: int = 1,
    deleted: int = 0,
) -> bool:
    """写入/更新会话到 ES（纯文档写入，聚合见 service 层）"""
    doc = {
        "id": conv_id,
        "user_id": user_id,
        "title": title,
        "message_contents": message_contents,
        "last_message_at": datetime.now().isoformat(),
        "status": status,
        "deleted": deleted,
    }
    return await es_client.index_doc(INDEX_NAME, str(conv_id), doc)
