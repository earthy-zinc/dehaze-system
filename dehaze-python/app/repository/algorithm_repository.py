"""
算法数据访问层
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, or_, select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_algorithm import SysAlgorithm, SysAlgorithmVersion
from app.models.entity.sys_log import SysPredLog
from app.repository.base import BaseRepository


# 算法状态机常量
class AlgorithmStatus:
    DRAFT = 0       # 草稿
    TESTING = 1     # 测试中
    PENDING_AUDIT = 2  # 待审核
    PUBLISHED = 3   # 已发布
    DISABLED = 4    # 已停用
    ARCHIVED = 5    # 已归档


class AlgorithmRepository(BaseRepository[SysAlgorithm]):
    """算法数据访问层"""

    model = SysAlgorithm

    async def get_list_with_keywords(
        self,
        db: AsyncSession,
        keywords: str | None = None,
    ) -> list[SysAlgorithm]:
        """获取算法列表（支持关键词搜索）"""
        stmt = select(SysAlgorithm)
        if keywords:
            stmt = stmt.where(SysAlgorithm.name.like(f"%{keywords}%"))
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_with_children_ids(
        self,
        db: AsyncSession,
        algorithm_ids: list[int],
    ) -> list[int]:
        """获取算法及其子算法 ID（用于删除）"""
        stmt = select(SysAlgorithm.id).where(
            or_(
                SysAlgorithm.id.in_(algorithm_ids),
                SysAlgorithm.parent_id.in_(algorithm_ids),
            )
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def delete_by_ids(
        self,
        db: AsyncSession,
        ids: list[int],
    ) -> int:
        """根据 ID 列表批量删除"""
        if not ids:
            return 0
        stmt = delete(SysAlgorithm).where(SysAlgorithm.id.in_(ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def get_algorithm_options(
        self,
        db: AsyncSession,
    ) -> list[dict]:
        """获取算法下拉选项列表（树形结构）"""
        stmt = select(SysAlgorithm).where(SysAlgorithm.status == AlgorithmStatus.PUBLISHED)
        result = await db.execute(stmt)
        algorithms = result.scalars().all()

        algorithm_dict = {
            algorithm.id: {"value": algorithm.id, "label": algorithm.name, "children": []}
            for algorithm in algorithms
        }

        root_options = []
        for algorithm in algorithms:
            if algorithm.parent_id == 0:
                root_options.append(algorithm_dict[algorithm.id])
            else:
                parent = algorithm_dict.get(algorithm.parent_id)
                if parent:
                    parent["children"].append(algorithm_dict[algorithm.id])

        return root_options

    # ── 状态机 ──────────────────────────────────────

    async def update_status(
        self,
        db: AsyncSession,
        algorithm_id: int,
        status: int,
    ) -> Optional[SysAlgorithm]:
        """更新算法状态"""
        algorithm = await self.get_by_id(db, algorithm_id)
        if not algorithm:
            return None
        algorithm.status = status
        await db.flush()
        await db.refresh(algorithm)
        return algorithm

    # ── 审核 ──────────────────────────────────────

    async def audit(
        self,
        db: AsyncSession,
        algorithm_id: int,
        audit_by: int,
        passed: bool,
        remark: Optional[str] = None,
    ) -> Optional[SysAlgorithm]:
        """审核算法（通过/驳回）"""
        algorithm = await self.get_by_id(db, algorithm_id)
        if not algorithm:
            return None
        algorithm.audit_by = audit_by
        algorithm.audit_time = datetime.now(timezone.utc)
        algorithm.audit_remark = remark
        if passed:
            algorithm.status = AlgorithmStatus.PUBLISHED
        else:
            # 驳回回到测试中
            algorithm.status = AlgorithmStatus.TESTING
        await db.flush()
        await db.refresh(algorithm)
        return algorithm

    # ── 版本控制 ──────────────────────────────────────

    async def check_version_exists(
        self,
        db: AsyncSession,
        algorithm_id: int,
        version: str,
    ) -> bool:
        """检查版本号是否已在版本历史表中存在（由唯一约束保证）"""
        stmt = select(func.count()).select_from(SysAlgorithmVersion).where(
            SysAlgorithmVersion.algorithm_id == algorithm_id,
            SysAlgorithmVersion.version == version,
        )
        return ((await db.execute(stmt)).scalar() or 0) > 0

    async def create_version(
        self,
        db: AsyncSession,
        algorithm_id: int,
        version: str,
        change_log: Optional[str] = None,
        status: Optional[int] = None,
        config_json: Optional[str] = None,
        model_file_id: Optional[int] = None,
        is_active: int = 0,
    ) -> SysAlgorithmVersion:
        """新增版本记录 (对齐 Java SysAlgorithmVersion 字段)"""
        version_record = SysAlgorithmVersion(
            algorithm_id=algorithm_id,
            version=version,
            change_log=change_log,
            status=status,
            config_json=config_json,
            model_file_id=model_file_id,
            is_active=is_active,
        )
        db.add(version_record)
        await db.flush()
        await db.refresh(version_record)
        return version_record

    async def list_versions(
        self,
        db: AsyncSession,
        algorithm_id: int,
    ) -> list[SysAlgorithmVersion]:
        """查询算法版本历史"""
        stmt = (
            select(SysAlgorithmVersion)
            .where(SysAlgorithmVersion.algorithm_id == algorithm_id)
            .order_by(desc(SysAlgorithmVersion.id))
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_version_by_id(
        self,
        db: AsyncSession,
        version_id: int,
    ) -> Optional[SysAlgorithmVersion]:
        """根据ID获取版本记录"""
        stmt = select(SysAlgorithmVersion).where(SysAlgorithmVersion.id == version_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def rollback_to_version(
        self,
        db: AsyncSession,
        algorithm_id: int,
        version_id: int,
    ) -> Optional[SysAlgorithm]:
        """回滚到指定版本"""
        version = await self.get_version_by_id(db, version_id)
        if not version or version.algorithm_id != algorithm_id:
            return None
        algorithm = await self.get_by_id(db, algorithm_id)
        if not algorithm:
            return None
        # 将当前版本存入历史表（如果还没存）
        if not await self.check_version_exists(db, algorithm_id, algorithm.version):
            await self.create_version(
                db=db,
                algorithm_id=algorithm_id,
                version=algorithm.version,
                change_log=f"回滚前自动归档: {algorithm.version}",
                status=algorithm.status,
            )
        # 应用目标版本
        algorithm.version = version.version
        await db.flush()
        await db.refresh(algorithm)
        return algorithm

    # ── 监控数据 ──────────────────────────────────────

    async def get_monitor_stats(
        self,
        db: AsyncSession,
        algorithm_id: int,
    ) -> dict:
        """从预测日志表聚合监控数据"""
        stmt = select(
            func.count(SysPredLog.id).label("total_calls"),
            func.avg(SysPredLog.time).label("avg_time"),
            func.max(SysPredLog.time).label("max_time"),
            func.min(SysPredLog.time).label("min_time"),
        ).where(SysPredLog.algorithm_id == algorithm_id)
        result = await db.execute(stmt)
        row = result.one_or_none()
        if not row or row.total_calls == 0:
            return {
                "totalCalls": 0,
                "avgTime": 0.0,
                "maxTime": 0,
                "minTime": 0,
                "successRate": 0.0,
            }
        return {
            "totalCalls": row.total_calls,
            "avgTime": float(row.avg_time or 0),
            "maxTime": int(row.max_time or 0),
            "minTime": int(row.min_time or 0),
            "successRate": 1.0,  # 日志表只记录成功的调用
        }

    async def get_today_call_count(
        self,
        db: AsyncSession,
        algorithm_id: int,
    ) -> int:
        """获取算法今日调用次数（对齐 Java: create_time >= 今日零点）"""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        stmt = (
            select(func.count())
            .select_from(SysPredLog)
            .where(
                SysPredLog.algorithm_id == algorithm_id,
                SysPredLog.create_time >= today_start,
            )
        )
        return (await db.execute(stmt)).scalar() or 0



# 单例
algorithm_repository = AlgorithmRepository()
