"""
算法选择服务 —— 算法树 / 详情 / 测试 / 搜索 / 对比
"""

import asyncio
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm
from app.repository.algorithm_repository import AlgorithmStatus, algorithm_repository
from app.repository.feedback_repository import rating_repository
from app.repository.pred_eval_log_repository import pred_log_repository

logger = logging.getLogger(__name__)


class AlgorithmSelectService:
    """算法选择服务"""

    async def get_algorithm_tree(self, db: AsyncSession) -> list[dict[str, Any]]:
        """获取算法选择树（仅已发布状态）"""
        algorithms = await algorithm_repository.list_published(db, order_by_tree=True)

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

    async def get_algorithm_detail(self, db: AsyncSession, algorithm_id: int) -> dict[str, Any]:
        """获取算法详情（含评分、使用次数）"""
        algo = await self._require_published(db, algorithm_id)

        # 计算平均评分
        avg_rating = await rating_repository.get_avg_rating(db, algorithm_id)

        # 计算使用次数
        usage_count = await pred_log_repository.count_by_algorithm(db, algorithm_id)

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

    async def test_algorithm(
        self,
        db: AsyncSession,
        algorithm_id: int,
        image_url: str,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """上传图片测试算法效果（同步等待结果，超时返回 B0100）"""
        # 校验算法存在且已发布
        await self._require_published(db, algorithm_id)

        # 校验图片格式（仅允许常见图片格式）
        image_url_lower = image_url.lower()
        allowed_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")
        if not any(
            image_url_lower.endswith(ext)
            or f".{ext}?" in image_url_lower
            or f".{ext}&" in image_url_lower
            for ext in allowed_extensions
        ):
            # 尝试从URL路径提取扩展名
            has_valid_ext = False
            for ext in allowed_extensions:
                if ext in image_url_lower:
                    has_valid_ext = True
                    break
            if not has_valid_ext:
                raise BusinessException(
                    ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "文件格式不支持"
                )

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
        except TimeoutError:
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_TIMEOUT, "算法测试超时，请稍后重试"
            ) from None

        return {
            "resultUrl": pred_result.get("resultUrl", ""),
            "processTime": pred_result.get("time", 0),
        }

    async def search_algorithms(
        self,
        db: AsyncSession,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        """搜索算法（关键词/拼音/标签，仅已发布）"""
        algorithms = await algorithm_repository.list_published(db, keyword=keyword)

        search_results = []
        for algo in algorithms:
            avg_rating = await rating_repository.get_avg_rating(db, algo.id)
            search_results.append(
                {
                    "id": algo.id,
                    "name": algo.name,
                    "type": algo.type,
                    "description": algo.description,
                    "avgRating": round(avg_rating, 1),
                }
            )

        return search_results

    async def compare(
        self,
        db: AsyncSession,
        algorithm_ids: list[int],
    ) -> list[dict[str, Any]]:
        """算法对比（T-AS-055：数量需在 2-3 个之间）"""
        if len(algorithm_ids) > 3 or len(algorithm_ids) < 2:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "算法对比数量需在 2-3 个之间")

        algorithms = await algorithm_repository.list_by_ids_include_unpublished(
            db, algorithm_ids
        )

        algo_map = {a.id: a for a in algorithms}
        result_list = []
        for aid in algorithm_ids:
            algo = algo_map.get(aid)
            if not algo:
                continue

            avg_rating = await rating_repository.get_avg_rating(db, algo.id)
            usage_count = await pred_log_repository.count_by_algorithm(db, algo.id)

            result_list.append(
                {
                    "algorithmId": algo.id,
                    "algorithmName": algo.name,
                    "type": algo.type,
                    "params": algo.params,
                    "flops": algo.flops,
                    "description": algo.description,
                    "avgRating": round(avg_rating, 1),
                    "usageCount": usage_count,
                }
            )

        return result_list

    async def recommend(
        self,
        db: AsyncSession,
        *,
        keyword: str | None = None,
        task_type: str | None = None,
        sample_algorithm_id: int | None = None,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        """算法推荐匹配（F-M03-007 / T-AS-060~068）。

        匹配策略（基于已发布算法）：
        - sampleAlgorithmId 存在时：与样例算法同 taskType、同分类，并按关键词/taskType 综合评分
        - keyword：匹配算法名称/描述/类型，命中越多得分越高
        - taskType：限定任务类型（为空则跨类型）
        - 按 matchScore 降序取 topN（默认 3，范围 1-10）
        - 无匹配时返回 total=0、items=[]（HTTP 200）
        """
        top_n = top_n if top_n is not None else 3
        if not (1 <= top_n <= 10):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "topN 超出 1-10 范围")

        sample_algo: SysAlgorithm | None = None
        if sample_algorithm_id is not None:
            sample_algo = await self._require_published(
                db, sample_algorithm_id
            )

        algorithms = await algorithm_repository.list_published(db)

        def _score(a: SysAlgorithm) -> int:
            score = 0
            kw = (keyword or "").strip().lower()
            if kw:
                name = (a.name or "").lower()
                desc = (a.description or "").lower()
                type_ = (a.type or "").lower()
                if kw in name:
                    score += 60
                if kw in desc:
                    score += 30
                if kw in type_:
                    score += 20
            if sample_algo is not None and a.id != sample_algo.id:
                # 与样例算法同类型/同分类加分
                if a.type and a.type == sample_algo.type:
                    score += 40
                if a.parent_id and a.parent_id == sample_algo.parent_id:
                    score += 30
            if task_type and a.type:
                score += (a.type == task_type) * 10
            return score

        candidates = []
        for a in algorithms:
            if sample_algo is not None and a.id == sample_algo.id:
                continue
            # taskType 过滤：若指定 taskType，仅保留类型匹配或为空类型的算法（T-AS-064 跨类型需不传 taskType）
            if task_type and a.type and a.type != task_type:
                continue
            score = _score(a)
            candidates.append((score, a))

        # 关键词为空、样例算法为空、taskType 为空时：不推荐空关键词（避免全量返回），返回空列表
        if not (keyword or sample_algo or task_type):
            return {"total": 0, "items": []}

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:top_n]

        items = []
        for score, a in candidates:
            if score <= 0:
                continue
            reason = (
                f"算法名称/描述与关键词「{keyword}」匹配"
                if keyword
                else "基于任务类型与分类综合匹配"
            )
            if task_type and a.type == task_type:
                reason = f"匹配任务类型「{task_type}」"
            if sample_algo is not None:
                reason = f"与样例算法「{sample_algo.name}」同类/同分类推荐"
            items.append(
                {
                    "algorithmId": a.id,
                    "algorithmName": a.name,
                    "matchScore": min(100, score),
                    "reason": reason,
                    "estimatedTime": None,
                }
            )

        return {"total": len(items), "items": items}

    async def _require_published(self, db: AsyncSession, algorithm_id: int) -> SysAlgorithm:
        """取算法并要求已发布，不存在/未发布抛 A0401（详情与测试共用）"""
        algo = await algorithm_repository.get_by_id_include_unpublished(db, algorithm_id)
        if not algo:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在")
        if algo.status != AlgorithmStatus.PUBLISHED:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法未发布")
        return algo


# 算法选择服务单例
algorithm_select_service = AlgorithmSelectService()
