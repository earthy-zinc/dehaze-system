"""AI 对话模块 - 长期记忆服务"""

import json
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.es.ai_memory_index import delete_memory_doc
from app.models.entity.sys_ai_memory import SysAiMemory
from app.models.schema.ai_memory import MemoryCreate, MemoryResult, MemoryUpdate
from app.models.schema.common import PageResult
from app.repository.ai_memory_repository import ai_memory_repository


class AiMemoryService:
    async def list_memories(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        size: int,
        memory_type: str | None = None,
        source: str | None = None,
    ) -> PageResult[MemoryResult]:
        memories, total = await ai_memory_repository.list_by_user(
            db, user_id, memory_type, source, page, size
        )
        return PageResult(list=[MemoryResult.model_validate(m) for m in memories], total=total)

    async def list_archived(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        size: int,
        memory_type: str | None = None,
    ) -> PageResult[MemoryResult]:
        """归档记忆查看：被遗忘策略归档（archived=1）且未删除的记忆。"""
        memories, total = await ai_memory_repository.list_archived(
            db, user_id, memory_type, page, size
        )
        return PageResult(list=[MemoryResult.model_validate(m) for m in memories], total=total)

    async def create_memory(self, db: AsyncSession, user_id: int, form: MemoryCreate) -> MemoryResult:
        memory = SysAiMemory(
            user_id=user_id,
            memory_type=form.memoryType,
            content=form.content,
            metadata_=form.metadata,
            importance=form.importance,
            source=form.source,
            status=1,
            archived=0,
        )
        memory = await ai_memory_repository.create(db, memory)
        return MemoryResult.model_validate(memory)

    async def update_memory(
        self, db: AsyncSession, memory_id: int, user_id: int, form: MemoryUpdate
    ) -> MemoryResult:
        memory = await ai_memory_repository.get_by_id_and_user(db, memory_id, user_id)
        if not memory:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "记忆不存在")
        data = form.model_dump(exclude_unset=True)
        for key, value in data.items():
            setattr(memory, key, value)
        await db.flush()
        await db.refresh(memory)
        return MemoryResult.model_validate(memory)

    async def delete_memory(self, db: AsyncSession, memory_id: int, user_id: int) -> None:
        memory = await ai_memory_repository.get_by_id_and_user(db, memory_id, user_id)
        if not memory:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "记忆不存在")
        await ai_memory_repository.soft_delete_with_time(db, [memory.id])
        # 同步清除 ES 向量索引，避免残留
        await delete_memory_doc(memory.id)

    async def search_memories(
        self, db: AsyncSession, user_id: int, keyword: str, limit: int = 5
    ) -> list[MemoryResult]:
        memories = await ai_memory_repository.search_by_keyword(db, user_id, keyword, limit)
        # 先完成 ORM 实体序列化，再执行 touch（touch 的 UPDATE 会使实体属性过期，
        # 若在其后再访问属性会触发异步懒加载 MissingGreenlet）。
        results = [MemoryResult.model_validate(m) for m in memories]
        for m in memories:
            await ai_memory_repository.touch(db, m.id)
        return results

    async def get_active_memories(
        self, db: AsyncSession, user_id: int, limit: int = 10
    ) -> list[MemoryResult]:
        memories = await ai_memory_repository.get_active_by_user(db, user_id, limit)
        return [MemoryResult.model_validate(m) for m in memories]

    async def batch_clear(
        self,
        db: AsyncSession,
        user_id: int,
        confirm: bool,
        memory_type: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """批量清空记忆（软删 + 记录 delete_time，30 天内可恢复）。

        三种粒度：全部 / 指定类型 / 指定时间范围。需 confirm 二次确认。
        返回受影响条数。
        """
        if not confirm:
            raise BusinessException(ResultCode.PARAM_ERROR, "批量清空记忆为不可逆操作，需二次确认")
        return await ai_memory_repository.batch_clear(db, user_id, memory_type, start, end)

    async def restore_deleted(
        self,
        db: AsyncSession,
        user_id: int,
        confirm: bool,
        memory_type: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """恢复 30 天恢复窗口内的软删记忆（清 deleted 与 delete_time）。"""
        if not confirm:
            raise BusinessException(ResultCode.PARAM_ERROR, "恢复记忆操作需二次确认")
        memories = await ai_memory_repository.list_deleted_for_restore(
            db, user_id, memory_type, start, end
        )
        if not memories:
            return 0
        ids = [m.id for m in memories]
        return await ai_memory_repository.restore_deleted(db, ids)

    async def export_memories(
        self,
        db: AsyncSession,
        user_id: int,
        fmt: str,
    ) -> tuple[str, str]:
        """导出用户全部活跃记忆。

        fmt: json | markdown。返回 (content_type, content)。
        """
        memories = await ai_memory_repository.get_active_by_user(db, user_id, limit=10000)
        if fmt == "markdown":
            lines = ["# 长期记忆导出", ""]
            for m in memories:
                created_at = m.create_time.strftime("%Y-%m-%d %H:%M:%S") if m.create_time else "-"
                lines.extend(
                    [
                        f"## {m.memory_type}（来源：{m.source}）",
                        f"- 内容：{m.content}",
                        f"- 重要性：{m.importance}",
                        f"- 创建时间：{created_at}",
                        "",
                    ]
                )
            return "text/markdown; charset=utf-8", "\n".join(lines)
        records: list[dict[str, Any]] = []
        for m in memories:
            records.append(
                {
                    "id": m.id,
                    "memory_type": m.memory_type,
                    "content": m.content,
                    "metadata": m.metadata_,
                    "source": m.source,
                    "importance": m.importance,
                    "access_count": m.access_count,
                    "created_at": m.create_time.strftime("%Y-%m-%d %H:%M:%S")
                    if m.create_time
                    else None,
                }
            )
        return "application/json; charset=utf-8", json.dumps(
            {"user_id": user_id, "exported_at": datetime.now().isoformat(), "memories": records},
            ensure_ascii=False,
            indent=2,
        )


ai_memory_service = AiMemoryService()
