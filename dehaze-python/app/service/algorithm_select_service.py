"""
算法选择服务 —— 智能推荐 / 收藏 / 对比
"""
import logging
from typing import Any, Optional

from sqlalchemy import select, delete, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessException
from app.core.code import ResultCode
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_algorithm_favorite import SysAlgorithmFavorite
from app.repository.algorithm_repository import AlgorithmStatus
from app.utils.datetime_utils import format_time

logger = logging.getLogger(__name__)


class AlgorithmSelectService:
    """算法选择服务"""

    @staticmethod
    async def recommend(
        db: AsyncSession,
        image_url: str,
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """
        智能推荐算法

        基于已发布算法的启发式评分返回 Top N 推荐。
        特征权重（文档要求）:
        - 雾霾浓度 30%
        - 场景类型 20%
        - 光照 15%
        - 复杂度 10%
        - 颜色 10%
        - 分辨率 5%
        - 用户偏好 10%
        """
        stmt = select(SysAlgorithm).where(SysAlgorithm.status == AlgorithmStatus.PUBLISHED)
        result = await db.execute(stmt)
        algorithms = list(result.scalars().all())

        if not algorithms:
            return []

        scored = []
        for algo in algorithms:
            score = 70.0  # 基础分
            reason_parts = []

            # 深度学习类算法优先（适合处理重雾）
            algo_type = (algo.type or "").lower()
            if "深度学习" in algo_type or "deep" in algo_type or "learning" in algo_type:
                score += 15
                reason_parts.append("深度学习算法，去雾效果好")
            elif "传统" in algo_type or "tradition" in algo_type or "dcp" in (algo.import_path or "").lower():
                score += 8
                reason_parts.append("传统算法，速度快")

            # 算法名称中包含关键词加分
            name_lower = (algo.name or "").lower()
            if "former" in name_lower or "transformer" in name_lower:
                score += 5
                reason_parts.append("Transformer 架构，性能优秀")
            if "ridcp" in name_lower:
                score += 5
                reason_parts.append("支持真实场景去雾")

            scored.append({
                "algorithmId": algo.id,
                "algorithmName": algo.name,
                "score": min(score, 100),
                "reason": "；".join(reason_parts) if reason_parts else "综合推荐",
                "type": algo.type,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]

    # ── 收藏 ──────────────────────────────────────

    @staticmethod
    async def add_favorite(db: AsyncSession, user_id: int, algorithm_id: int) -> int:
        """添加收藏（幂等：已收藏则返回已有记录ID）"""
        # 校验算法存在
        stmt = select(SysAlgorithm).where(SysAlgorithm.id == algorithm_id)
        result = await db.execute(stmt)
        if not result.scalar_one_or_none():
            raise BusinessException("算法不存在", ResultCode.RESOURCE_NOT_FOUND.code)

        # 检查是否已收藏
        existing = await AlgorithmSelectService._get_favorite(db, user_id, algorithm_id)
        if existing:
            return existing.id

        favorite = SysAlgorithmFavorite(user_id=user_id, algorithm_id=algorithm_id)
        db.add(favorite)
        await db.flush()
        await db.refresh(favorite)
        await db.commit()
        return favorite.id

    @staticmethod
    async def remove_favorite(db: AsyncSession, user_id: int, algorithm_id: int) -> bool:
        """取消收藏"""
        stmt = (
            delete(SysAlgorithmFavorite)
            .where(SysAlgorithmFavorite.user_id == user_id)
            .where(SysAlgorithmFavorite.algorithm_id == algorithm_id)
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0

    @staticmethod
    async def _get_favorite(
        db: AsyncSession,
        user_id: int,
        algorithm_id: int,
    ) -> Optional[SysAlgorithmFavorite]:
        """查询是否已收藏"""
        stmt = (
            select(SysAlgorithmFavorite)
            .where(SysAlgorithmFavorite.user_id == user_id)
            .where(SysAlgorithmFavorite.algorithm_id == algorithm_id)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_favorites(db: AsyncSession, user_id: int) -> list[dict[str, Any]]:
        """收藏列表"""
        stmt = (
            select(SysAlgorithmFavorite, SysAlgorithm.name)
            .join(SysAlgorithm, SysAlgorithmFavorite.algorithm_id == SysAlgorithm.id, isouter=True)
            .where(SysAlgorithmFavorite.user_id == user_id)
            .order_by(desc(SysAlgorithmFavorite.id))
        )
        result = await db.execute(stmt)
        rows = result.all()
        return [
            {
                "id": fav.id,
                "userId": fav.user_id,
                "algorithmId": fav.algorithm_id,
                "algorithmName": algo_name,
                "createTime": format_time(fav.create_time),
            }
            for fav, algo_name in rows
        ]

    # ── 对比 ──────────────────────────────────────

    @staticmethod
    async def compare(
        db: AsyncSession,
        algorithm_ids: list[int],
        image_url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        算法对比

        返回多个算法的基本信息和（可选）处理结果。
        实际去雾处理由前端调用 /prediction 接口完成，此处仅返回算法元数据。
        """
        stmt = select(SysAlgorithm).where(SysAlgorithm.id.in_(algorithm_ids))
        result = await db.execute(stmt)
        algorithms = list(result.scalars().all())

        # 按请求顺序返回
        algo_map = {a.id: a for a in algorithms}
        result_list = []
        for aid in algorithm_ids:
            algo = algo_map.get(aid)
            if not algo:
                continue
            result_list.append({
                "algorithmId": algo.id,
                "algorithmName": algo.name,
                "type": algo.type,
                "params": algo.params,
                "flops": algo.flops,
                "description": algo.description,
                "status": algo.status,
                "resultUrl": None,  # 实际结果需前端调用 /prediction
                "processTime": None,
            })
        return result_list
