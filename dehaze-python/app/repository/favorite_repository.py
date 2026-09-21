"""
收藏管理 Repository

提供 sys_favorite 表的数据访问操作：
- 按 user_id + target_type + target_id 查询（唯一约束）
- 收藏列表分页查询（按类型/关键词/排序筛选）
- 批量软删除
- 标记失效
"""

from sqlalchemy import and_, func, select, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_audit_update_values
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_dataset import SysDataset
from app.models.entity.sys_favorite import SysFavorite
from app.models.entity.sys_log import SysPredLog
from app.repository.base import BaseRepository, escape_like


class FavoriteRepository(BaseRepository[SysFavorite]):
    model = SysFavorite

    async def get_by_user_and_target(
        self,
        db: AsyncSession,
        user_id: int,
        target_type: str,
        target_id: int,
    ) -> SysFavorite | None:
        """按唯一约束查询未删除的收藏记录"""
        stmt = select(SysFavorite).where(
            SysFavorite.user_id == user_id,
            SysFavorite.target_type == target_type,
            SysFavorite.target_id == target_id,
            SysFavorite.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_by_user_and_target(
        self,
        db: AsyncSession,
        user_id: int,
        target_type: str,
        target_id: int,
    ) -> int:
        """upsert 收藏：已存在未删除行时幂等返回原行（重置 is_invalid=0）；返回 id

        uk_user_target 含 deleted 列，软删行不再参与冲突，故"取消后重新收藏"是插入
        新的未删除行（软删历史行保留），不是复活旧行。
        """
        stmt = mysql_insert(SysFavorite).values(
            user_id=user_id,
            target_type=target_type,
            target_id=target_id,
            is_invalid=0,
        )
        stmt = stmt.on_duplicate_key_update(
            deleted=0,
            is_invalid=0,
            update_time=func.now(),
        )
        await db.execute(stmt)
        # on_duplicate_key_update 不回填 id，需重查
        result = await db.execute(
            select(SysFavorite).where(
                SysFavorite.user_id == user_id,
                SysFavorite.target_type == target_type,
                SysFavorite.target_id == target_id,
                SysFavorite.deleted == 0,
            )
        )
        row = result.scalar_one_or_none()
        return row.id if row else 0

    async def count_user_favorites(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> int:
        """统计用户未删除的收藏数量"""
        stmt = select(func.count()).where(
            SysFavorite.user_id == user_id,
            SysFavorite.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def target_exists(
        self,
        db: AsyncSession,
        target_type: str,
        target_id: int,
    ) -> bool:
        """校验收藏目标对象是否存在（合法类型白名单由 Service 层校验）

        algorithm/dataset/result 为已实现类型，必须校验；
        image/preset 为预留类型，跳过校验。
        """
        if target_type == "algorithm":
            stmt = select(SysAlgorithm.id).where(
                SysAlgorithm.id == target_id,
                SysAlgorithm.deleted == 0,
            )
        elif target_type == "dataset":
            stmt = select(SysDataset.id).where(
                SysDataset.id == target_id,
                SysDataset.deleted == 0,
            )
        elif target_type == "result":
            stmt = select(SysPredLog.id).where(
                SysPredLog.id == target_id,
            )
        else:
            # image/preset 等预留类型，跳过校验
            return True

        result = await db.execute(stmt)
        return result.scalar() is not None

    async def get_page(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        page_size: int,
        *,
        target_type: str | None = None,
        keywords: str | None = None,
        sort_order: str | None = None,
    ) -> tuple[list[dict], int]:
        """收藏列表分页查询

        algorithm 类型 LEFT JOIN sys_algorithm 取 name 作为 targetName。
        dataset 类型 LEFT JOIN sys_dataset 取 name/img 作为 targetName/targetThumbnail。
        其他类型暂不 JOIN，targetName/targetThumbnail 为空。
        排序仅支持收藏时间（sort_order=asc 正序，其余倒序）。
        """
        stmt = (
            select(
                SysFavorite,
                SysAlgorithm.name.label("target_name"),
                SysDataset.name.label("target_name_dataset"),
                SysDataset.img.label("target_thumbnail"),
            )
            .outerjoin(
                SysAlgorithm,
                and_(
                    SysFavorite.target_type == "algorithm",
                    SysFavorite.target_id == SysAlgorithm.id,
                    SysAlgorithm.deleted == 0,
                ),
            )
            .outerjoin(
                SysDataset,
                and_(
                    SysFavorite.target_type == "dataset",
                    SysFavorite.target_id == SysDataset.id,
                    SysDataset.deleted == 0,
                ),
            )
            .where(
                SysFavorite.user_id == user_id,
                SysFavorite.deleted == 0,
            )
        )

        if target_type:
            stmt = stmt.where(SysFavorite.target_type == target_type)

        if keywords:
            escaped = escape_like(keywords)
            like_pattern = f"%{escaped}%"
            stmt = stmt.where(
                SysAlgorithm.name.like(like_pattern, escape="\\")
                | SysDataset.name.like(like_pattern, escape="\\")
            )

        # 仅支持按收藏时间排序（对齐 Java：sortOrder=asc 正序，其余倒序）
        col = SysFavorite.create_time
        is_desc = sort_order != "asc"

        if is_desc:
            stmt = stmt.order_by(col.desc(), SysFavorite.id.desc())
        else:
            stmt = stmt.order_by(col.asc(), SysFavorite.id.asc())

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        rows = result.all()

        items = [
            {
                "favorite": row[0],
                "target_name": row[1] or row[2],
                "target_thumbnail": row[3],
            }
            for row in rows
        ]
        return items, total

    async def soft_delete_by_ids(
        self,
        db: AsyncSession,
        ids: list[int],
        user_id: int | None = None,
    ) -> int:
        """按 ID 列表批量软删除（仅限当前用户）

        user_id 与 BaseRepository.soft_delete_by_ids 保持签名兼容：调用方恒显式传入，
        省略时按 user_id IS NULL 匹配（无行命中）。
        """
        if not ids:
            return 0
        # uk_user_target 含 deleted 列，表设计约定 deleted 写入"删除时的行 id"：
        # 写常量 1 会让同一 (user, type, target) 第二次取消收藏撞唯一键（1062 → 500）
        values = {"deleted": SysFavorite.id}
        values.update(get_audit_update_values())

        stmt = (
            update(SysFavorite)
            .where(
                SysFavorite.id.in_(ids),
                SysFavorite.user_id == user_id,
                SysFavorite.deleted == 0,
            )
            .values(**values)
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def mark_invalid(
        self,
        db: AsyncSession,
        target_type: str,
        target_ids: list[int],
    ) -> int:
        """按目标类型 + ID 批量标记收藏失效（对象删除时由业务模块调用，对齐 Java markInvalid）"""
        if not target_ids:
            return 0
        values = {"is_invalid": 1}
        values.update(get_audit_update_values())
        stmt = (
            update(SysFavorite)
            .where(
                SysFavorite.target_type == target_type,
                SysFavorite.target_id.in_(target_ids),
            )
            .values(**values)
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def get_count_by_type(
        self,
        db: AsyncSession,
        user_id: int,
        target_type: str | None = None,
    ) -> list[dict]:
        """收藏数量统计（按类型分组）"""
        stmt = (
            select(
                SysFavorite.target_type,
                func.count().label("count"),
            )
            .where(
                SysFavorite.user_id == user_id,
                SysFavorite.deleted == 0,
            )
            .group_by(SysFavorite.target_type)
        )

        if target_type:
            stmt = stmt.where(SysFavorite.target_type == target_type)

        result = await db.execute(stmt)
        rows = result.all()

        count_map = {row[0]: row[1] for row in rows}

        counts = []
        for t in ["algorithm", "result", "dataset", "image", "preset"]:
            if target_type and target_type != t:
                continue
            counts.append({"target_type": t, "count": count_map.get(t, 0)})
        return counts


favorite_repository = FavoriteRepository()
