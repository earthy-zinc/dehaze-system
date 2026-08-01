"""
算法选择服务 —— 算法树 / 详情 / 测试 / 搜索 / 对比
"""
import asyncio
import logging
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm
from app.repository.algorithm_repository import AlgorithmStatus
from app.utils.datetime_utils import format_time

logger = logging.getLogger(__name__)


class AlgorithmSelectService:
    """算法选择服务"""

    @staticmethod
    async def get_algorithm_tree(db: AsyncSession) -> list[dict[str, Any]]:
        """获取算法选择树（仅已发布状态）"""
        stmt = (
            select(SysAlgorithm)
            .where(
                SysAlgorithm.status == AlgorithmStatus.PUBLISHED,
                SysAlgorithm.deleted == 0,
            )
            .order_by(SysAlgorithm.parent_id, SysAlgorithm.id)
        )
        result = await db.execute(stmt)
        algorithms = list(result.scalars().all())

        if not algorithms:
            return []

        # 收集父节点ID（即分类节点）
        parent_ids = {a.parent_id for a in algorithms if a.parent_id and a.parent_id > 0}

        # 构建节点映射
        node_map: dict[int, dict] = {}
        for algo in algorithms:
            node_map[algo.id] = {
                "id": algo.id,
                "name": algo.name,
                "parentId": algo.parent_id or 0,
                "type": algo.type,
                "isLeaf": algo.parent_id is not None and algo.parent_id > 0,
                "children": [],
            }

        # 构建树
        tree: list[dict] = []
        for algo in algorithms:
            node = node_map[algo.id]
            parent_id = algo.parent_id or 0

            if parent_id == 0 or parent_id not in node_map:
                # 顶层节点（分类节点或直接挂在根下的算法）
                if algo.id not in parent_ids:
                    # 是叶子算法节点（不在任何节点作为父节点）
                    node["isLeaf"] = True
                else:
                    node["isLeaf"] = False
                tree.append(node)
            else:
                # 挂到父节点下
                parent_node = node_map[parent_id]
                parent_node["isLeaf"] = False
                parent_node["children"].append(node)

        # 清理空 children
        def clean_children(nodes):
            for n in nodes:
                if n["children"]:
                    clean_children(n["children"])
                else:
                    n.pop("children", None)
                # 移除临时字段
                n.pop("parentId", None)
            return nodes

        return clean_children(tree)

    @staticmethod
    async def get_algorithm_detail(db: AsyncSession, algorithm_id: int) -> dict[str, Any]:
        """获取算法详情（含评分、使用次数）"""
        stmt = (
            select(SysAlgorithm)
            .where(
                SysAlgorithm.id == algorithm_id,
                SysAlgorithm.deleted == 0,
            )
        )
        result = await db.execute(stmt)
        algo = result.scalar_one_or_none()

        if not algo:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在")

        if algo.status != AlgorithmStatus.PUBLISHED:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法未发布")

        # 计算平均评分
        avg_rating = await AlgorithmSelectService._get_avg_rating(db, algorithm_id)

        # 计算使用次数
        usage_count = await AlgorithmSelectService._get_usage_count(db, algorithm_id)

        return {
            "id": algo.id,
            "name": algo.name,
            "type": algo.type,
            "description": algo.description,
            "img": algo.img,
            "params": algo.params,
            "flops": algo.flops,
            "size": algo.size,
            "avgRating": round(avg_rating, 1),
            "usageCount": usage_count,
        }

    @staticmethod
    async def test_algorithm(
        db: AsyncSession,
        algorithm_id: int,
        image_url: str,
        user_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """上传图片测试算法效果（同步等待结果，超时返回 B0100）"""
        # 校验算法存在且已发布
        stmt = (
            select(SysAlgorithm)
            .where(
                SysAlgorithm.id == algorithm_id,
                SysAlgorithm.deleted == 0,
            )
        )
        result = await db.execute(stmt)
        algo = result.scalar_one_or_none()

        if not algo:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在")

        if algo.status != AlgorithmStatus.PUBLISHED:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法未发布")

        # 校验图片格式（仅允许常见图片格式）
        image_url_lower = image_url.lower()
        allowed_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")
        if not any(image_url_lower.endswith(ext) or f".{ext}?" in image_url_lower or f".{ext}&" in image_url_lower for ext in allowed_extensions):
            # 尝试从URL路径提取扩展名
            has_valid_ext = False
            for ext in allowed_extensions:
                if ext in image_url_lower:
                    has_valid_ext = True
                    break
            if not has_valid_ext:
                raise BusinessException(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "文件格式不支持")

        # 调用预测服务
        from app.service.prediction_service import prediction_service

        try:
            pred_result = await asyncio.wait_for(
                prediction_service.predict(
                    algorithm_id=algorithm_id,
                    image_url=image_url,
                    user_id=user_id,
                    skip_quota_check=True,  # 测试不计入配额
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            raise BusinessException(ResultCode.SYSTEM_EXECUTION_TIMEOUT, "算法测试超时，请稍后重试")

        return {
            "resultUrl": pred_result.get("resultUrl", ""),
            "processTime": pred_result.get("time", 0),
        }

    @staticmethod
    async def search_algorithms(
        db: AsyncSession,
        keyword: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """搜索算法（关键词/拼音/标签，仅已发布）"""
        stmt = (
            select(SysAlgorithm)
            .where(
                SysAlgorithm.status == AlgorithmStatus.PUBLISHED,
                SysAlgorithm.deleted == 0,
            )
        )

        if keyword:
            kw = f"%{keyword}%"
            stmt = stmt.where(
                SysAlgorithm.name.ilike(kw) | SysAlgorithm.description.ilike(kw)
            )

        stmt = stmt.order_by(SysAlgorithm.id)
        result = await db.execute(stmt)
        algorithms = list(result.scalars().all())

        search_results = []
        for algo in algorithms:
            avg_rating = await AlgorithmSelectService._get_avg_rating(db, algo.id)
            search_results.append({
                "id": algo.id,
                "name": algo.name,
                "type": algo.type,
                "description": algo.description,
                "avgRating": round(avg_rating, 1),
            })

        return search_results

    @staticmethod
    async def compare(
        db: AsyncSession,
        algorithm_ids: list[int],
    ) -> list[dict[str, Any]]:
        """算法对比（最多3个）"""
        if len(algorithm_ids) > 3:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "算法对比数量不能超过3个")

        if len(algorithm_ids) < 1:
            return []

        stmt = select(SysAlgorithm).where(
            SysAlgorithm.id.in_(algorithm_ids),
            SysAlgorithm.deleted == 0,
        )
        result = await db.execute(stmt)
        algorithms = list(result.scalars().all())

        algo_map = {a.id: a for a in algorithms}
        result_list = []
        for aid in algorithm_ids:
            algo = algo_map.get(aid)
            if not algo:
                continue

            avg_rating = await AlgorithmSelectService._get_avg_rating(db, algo.id)
            usage_count = await AlgorithmSelectService._get_usage_count(db, algo.id)

            result_list.append({
                "algorithmId": algo.id,
                "algorithmName": algo.name,
                "type": algo.type,
                "params": algo.params,
                "flops": algo.flops,
                "description": algo.description,
                "avgRating": round(avg_rating, 1),
                "usageCount": usage_count,
            })

        return result_list

    @staticmethod
    async def _get_avg_rating(db: AsyncSession, algorithm_id: int) -> float:
        """获取算法平均评分"""
        from app.models.entity.sys_rating import SysRating
        stmt = (
            select(func.avg(SysRating.rating))
            .where(
                SysRating.algorithm_id == algorithm_id,
                SysRating.deleted == 0,
            )
        )
        result = await db.execute(stmt)
        avg = result.scalar()
        return float(avg) if avg else 0.0

    @staticmethod
    async def _get_usage_count(db: AsyncSession, algorithm_id: int) -> int:
        """获取算法使用次数（预测日志记录数）"""
        from app.models.entity.sys_log import SysPredLog
        stmt = (
            select(func.count(SysPredLog.id))
            .where(SysPredLog.algorithm_id == algorithm_id)
        )
        result = await db.execute(stmt)
        count = result.scalar()
        return count or 0
