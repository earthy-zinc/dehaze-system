"""会话 ES 读模型全量重建/对账入口

按 MySQL 为准重建 ai_conversation 索引，修复读写模型存量漂移
（deleted/status/title/message_contents），并清理 MySQL 中已不存在的孤儿文档。
幂等，可重复执行；仅运维手动触发，不接入定时任务。

用法（在 clarity-python 目录下执行）：
    uv run python scripts/rebuild_conversation_es.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 确保可以 import app 包（脚本位于 clarity-python/scripts/）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def _main() -> None:
    from app.infrastructure.es.ai_conversation_index import ensure_conversation_index
    from app.service.ai.service.conversation_search_service import rebuild_conversation_index

    await ensure_conversation_index()
    summary = await rebuild_conversation_index()
    print(
        f"会话 ES 读模型重建完成: 重同步={summary['resynced']}, "
        f"清理孤儿={summary['deleted_orphans']}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
