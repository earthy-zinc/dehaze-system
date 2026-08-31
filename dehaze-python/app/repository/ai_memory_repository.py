import math
from datetime import datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_memory import MEMORY_RECOVERY_WINDOW_DAYS, SysAiMemory
from app.repository.base import BaseRepository, escape_like


class AiMemoryRepository(BaseRepository[SysAiMemory]):
    model = SysAiMemory

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: int,
        memory_type: str | None = None,
        source: str | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysAiMemory], int]:
        stmt = select(SysAiMemory).where(
            SysAiMemory.user_id == user_id,
            SysAiMemory.deleted == 0,
            SysAiMemory.status == 1,
            SysAiMemory.archived == 0,
        )
        if memory_type:
            stmt = stmt.where(SysAiMemory.memory_type == memory_type)
        if source:
            stmt = stmt.where(SysAiMemory.source == source)
        stmt = stmt.order_by(
            SysAiMemory.importance.desc(),
            SysAiMemory.create_time.desc(),
        )
        return await self.paginate(db, stmt, page, size)

    async def list_active_by_type(
        self,
        db: AsyncSession,
        user_id: int,
        memory_type: str,
        limit: int = 100,
    ) -> list[SysAiMemory]:
        """查询用户某类型的活跃记忆（未删除/启用/未归档），按重要性倒序。"""
        stmt = (
            select(SysAiMemory)
            .where(
                SysAiMemory.user_id == user_id,
                SysAiMemory.memory_type == memory_type,
                SysAiMemory.deleted == 0,
                SysAiMemory.status == 1,
                SysAiMemory.archived == 0,
            )
            .order_by(
                SysAiMemory.importance.desc(),
                SysAiMemory.create_time.desc(),
            )
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_preferences(
        self,
        db: AsyncSession,
        user_id: int,
        limit: int = 20,
    ) -> list[SysAiMemory]:
        """常驻注入：语义记忆中 is_preference=1 的活跃偏好，全量注入（按重要性排序取前 limit）。"""
        memories = await self.list_active_by_type(db, user_id, "semantic", limit=limit)
        return [m for m in memories if (m.metadata_ or {}).get("is_preference")]

    async def list_by_skill(
        self,
        db: AsyncSession,
        user_id: int,
        skill: str,
        limit: int = 10,
    ) -> list[SysAiMemory]:
        """场景触发注入：程序记忆中 metadata.skill 匹配任务类型的活跃记忆。"""
        memories = await self.list_active_by_type(db, user_id, "procedural", limit=100)
        return [m for m in memories if (m.metadata_ or {}).get("skill") == skill][:limit]

    async def list_skills(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> list[str]:
        """场景触发注入兜底：返回用户程序记忆中的去重 skill 值列表。"""
        memories = await self.list_active_by_type(db, user_id, "procedural", limit=200)
        skills: list[str] = []
        seen: set[str] = set()
        for m in memories:
            skill = (m.metadata_ or {}).get("skill")
            if skill and skill not in seen:
                seen.add(skill)
                skills.append(skill)
        return skills

    async def list_archived(
        self,
        db: AsyncSession,
        user_id: int,
        memory_type: str | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysAiMemory], int]:
        """归档记忆查看：被遗忘策略归档（archived=1）且未删除的记忆。"""
        stmt = select(SysAiMemory).where(
            SysAiMemory.user_id == user_id,
            SysAiMemory.deleted == 0,
            SysAiMemory.archived == 1,
        )
        if memory_type:
            stmt = stmt.where(SysAiMemory.memory_type == memory_type)
        stmt = stmt.order_by(
            SysAiMemory.importance.desc(),
            SysAiMemory.create_time.desc(),
        )
        return await self.paginate(db, stmt, page, size)

    async def search_by_keyword(
        self,
        db: AsyncSession,
        user_id: int,
        keyword: str,
        limit: int = 5,
    ) -> list[SysAiMemory]:
        escaped = escape_like(keyword)
        stmt = (
            select(SysAiMemory)
            .where(
                SysAiMemory.user_id == user_id,
                SysAiMemory.deleted == 0,
                SysAiMemory.status == 1,
                SysAiMemory.archived == 0,
                SysAiMemory.content.like(f"%{escaped}%", escape="\\"),
            )
            .order_by(SysAiMemory.importance.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_active_by_user(
        self,
        db: AsyncSession,
        user_id: int,
        limit: int = 10,
    ) -> list[SysAiMemory]:
        stmt = (
            select(SysAiMemory)
            .where(
                SysAiMemory.user_id == user_id,
                SysAiMemory.deleted == 0,
                SysAiMemory.status == 1,
                SysAiMemory.archived == 0,
            )
            .order_by(
                SysAiMemory.importance.desc(),
                SysAiMemory.last_accessed_at.desc(),
            )
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def exists_active_content(
        self,
        db: AsyncSession,
        user_id: int,
        memory_type: str,
        content: str,
    ) -> bool:
        """是否存在内容完全一致的活跃记忆（保存前的强去重，不依赖 LLM 去重）。"""
        stmt = (
            select(SysAiMemory.id)
            .where(
                SysAiMemory.user_id == user_id,
                SysAiMemory.memory_type == memory_type,
                SysAiMemory.content == content,
                SysAiMemory.deleted == 0,
                SysAiMemory.status == 1,
                SysAiMemory.archived == 0,
            )
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def get_by_id_and_user(
        self,
        db: AsyncSession,
        memory_id: int,
        user_id: int,
    ) -> SysAiMemory | None:
        stmt = select(SysAiMemory).where(
            SysAiMemory.id == memory_id,
            SysAiMemory.user_id == user_id,
            SysAiMemory.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def touch(self, db: AsyncSession, memory_id: int) -> None:
        """检索命中后重激活：访问计数 +1、重置衰减计时器、重要性 +5（最高 100）"""
        stmt = (
            update(SysAiMemory)
            .where(SysAiMemory.id == memory_id)
            .values(
                access_count=SysAiMemory.access_count + 1,
                last_accessed_at=datetime.now(),
                importance=func.least(100, SysAiMemory.importance + 5),
            )
        )
        await db.execute(stmt)

    async def count_active(self, db: AsyncSession, user_id: int) -> int:
        """统计用户活跃（未删除、启用、未归档）记忆数量"""
        stmt = select(SysAiMemory).where(
            SysAiMemory.user_id == user_id,
            SysAiMemory.deleted == 0,
            SysAiMemory.status == 1,
            SysAiMemory.archived == 0,
        )
        return await self.count(db, stmt)

    async def archive_forgotten(
        self,
        db: AsyncSession,
        threshold: int = 10,
        half_life_days: int = 30,
    ) -> int:
        """遗忘曲线归档：priority = importance × exp(-Δt/half_life)，低于阈值归档。

        Δt 取 last_accessed_at（无则用 create_time）距当前的天数。
        返回归档数量。
        """
        stmt = select(SysAiMemory).where(
            SysAiMemory.deleted == 0,
            SysAiMemory.status == 1,
            SysAiMemory.archived == 0,
        )
        result = await db.execute(stmt)
        memories = list(result.scalars().all())

        now = datetime.now()
        to_archive = []
        for m in memories:
            last = m.last_accessed_at or m.create_time
            delta_days = (now - last).total_seconds() / 86400
            priority = m.importance * math.exp(-delta_days / half_life_days)
            if priority < threshold:
                to_archive.append(m.id)

        if to_archive:
            stmt = update(SysAiMemory).where(SysAiMemory.id.in_(to_archive)).values(archived=1)
            await db.execute(stmt)
        return len(to_archive)

    async def archive_least_important(
        self,
        db: AsyncSession,
        user_id: int,
        count: int,
    ) -> int:
        """归档用户重要性最低且最久未访问的 count 条活跃记忆，返回归档数量"""
        if count <= 0:
            return 0
        stmt = (
            select(SysAiMemory)
            .where(
                SysAiMemory.user_id == user_id,
                SysAiMemory.deleted == 0,
                SysAiMemory.status == 1,
                SysAiMemory.archived == 0,
            )
            .order_by(
                SysAiMemory.importance.asc(),
                SysAiMemory.last_accessed_at.asc(),
            )
            .limit(count)
        )
        result = await db.execute(stmt)
        ids = [m.id for m in result.scalars().all()]
        if not ids:
            return 0
        stmt = update(SysAiMemory).where(SysAiMemory.id.in_(ids)).values(archived=1)
        await db.execute(stmt)
        return len(ids)

    async def list_recent_episodic(
        self,
        db: AsyncSession,
        user_id: int,
        since: datetime,
    ) -> list[SysAiMemory]:
        """查询用户 since 之后创建的情景记忆（用于反思整合）"""
        stmt = (
            select(SysAiMemory)
            .where(
                SysAiMemory.user_id == user_id,
                SysAiMemory.memory_type == "episodic",
                SysAiMemory.deleted == 0,
                SysAiMemory.status == 1,
                SysAiMemory.archived == 0,
                SysAiMemory.create_time >= since,
            )
            .order_by(SysAiMemory.create_time.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_active_user_ids(self, db: AsyncSession) -> list[int]:
        """查询所有拥有活跃记忆的用户 ID（用于定时任务遍历）"""
        stmt = (
            select(SysAiMemory.user_id)
            .where(
                SysAiMemory.deleted == 0,
                SysAiMemory.status == 1,
                SysAiMemory.archived == 0,
            )
            .distinct()
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def soft_delete_with_time(
        self,
        db: AsyncSession,
        ids: list[int],
    ) -> int:
        """软删除并记录软删时间（30 天恢复窗口判定）。"""
        if not ids:
            return 0
        stmt = (
            update(SysAiMemory)
            .where(SysAiMemory.id.in_(ids))
            .values(deleted=1, delete_time=datetime.now())
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def batch_clear(
        self,
        db: AsyncSession,
        user_id: int,
        memory_type: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """批量清空记忆（软删 + 记录 delete_time）。

        三种粒度：
        - 全部：memory_type=None 且无时间范围
        - 指定类型：memory_type 非空
        - 指定时间范围：start/end 非空（按 create_time 过滤）
        返回受影响条数。
        """
        stmt = (
            update(SysAiMemory)
            .where(
                SysAiMemory.user_id == user_id,
                SysAiMemory.deleted == 0,
            )
            .values(deleted=1, delete_time=datetime.now())
        )
        if memory_type:
            stmt = stmt.where(SysAiMemory.memory_type == memory_type)
        if start:
            stmt = stmt.where(SysAiMemory.create_time >= start)
        if end:
            stmt = stmt.where(SysAiMemory.create_time <= end)
        result = await db.execute(stmt)
        return result.rowcount

    async def list_deleted_for_restore(
        self,
        db: AsyncSession,
        user_id: int,
        memory_type: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[SysAiMemory]:
        """查询已软删且在恢复窗口内的记忆（用于 30 天内恢复）。"""
        stmt = select(SysAiMemory).where(
            SysAiMemory.user_id == user_id,
            SysAiMemory.deleted == 1,
            SysAiMemory.delete_time >= datetime.now() - timedelta(days=MEMORY_RECOVERY_WINDOW_DAYS),
        )
        if memory_type:
            stmt = stmt.where(SysAiMemory.memory_type == memory_type)
        if start:
            stmt = stmt.where(SysAiMemory.create_time >= start)
        if end:
            stmt = stmt.where(SysAiMemory.create_time <= end)
        # 恢复路径需查已软删记录，用 include_deleted 绕过全局 deleted=0 过滤
        stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def restore_deleted(
        self,
        db: AsyncSession,
        ids: list[int],
    ) -> int:
        """恢复软删记忆（清 deleted 与 delete_time，恢复注入）。"""
        if not ids:
            return 0
        stmt = (
            update(SysAiMemory).where(SysAiMemory.id.in_(ids)).values(deleted=0, delete_time=None)
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def list_deleted_for_purge(
        self,
        db: AsyncSession,
        before_date: datetime,
    ) -> list[int]:
        """查询软删超过恢复窗口的记忆 ID（供物理清理定时任务）。"""
        stmt = select(SysAiMemory.id).where(
            SysAiMemory.deleted == 1,
            SysAiMemory.delete_time < before_date,
        )
        # 物理清理需查已软删记录，用 include_deleted 绕过全局 deleted=0 过滤
        stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return list(result.scalars().all())


ai_memory_repository = AiMemoryRepository()
