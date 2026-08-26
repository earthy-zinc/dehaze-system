"""AI 知识库召回测试集 Service：测试集管理（创建/列表/执行）。

对齐《后端实现-检索引擎.md》§7.1：一组"问题 + 期望命中段落"的可重复评估基线。
单个测试集 = 一个用例；执行时按 question 复用知识库检索引擎在该库内召回，
计算 Recall@K（期望命中分块出现在 Top-K 结果中的比例）与命中率（至少命中一条的用例占比）。
知识库归属/权限校验由路由层（kb:manage）完成，本服务不重复校验。
期望 chunk 缺失（已被删除/迁移）或检索异常（检索引擎降级为空结果）按未命中处理，不阻断执行。
"""

import logging

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_knowledge_test_set import SysKnowledgeTestSet
from app.models.schema.knowledge_base import RecallTestResultVO, TestSetVO
from app.repository.knowledge_test_set_repository import knowledge_test_set_repository
from app.service.kb.search_service import search_service

logger = logging.getLogger(__name__)


class TestSetService:
    """召回测试集服务（异步）"""

    async def create_test_set(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
        question: str,
        expected_chunk_ids: list[int],
    ) -> TestSetVO:
        """创建召回测试集（一条问题 + 期望命中分块）"""
        entity = SysKnowledgeTestSet(
            knowledge_base_id=knowledge_base_id,
            question=question,
            expected_chunk_ids=expected_chunk_ids,
        )
        created = await knowledge_test_set_repository.create(db, entity)
        return TestSetVO.model_validate(created)

    async def list_test_sets(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
        page: int,
        size: int,
    ) -> dict:
        """按知识库分页查询测试集列表"""
        items, total = await knowledge_test_set_repository.paginate_by_kb(
            db, knowledge_base_id, page, size
        )
        return {
            "list": [
                TestSetVO.model_validate(i).model_dump(mode="json", by_alias=True)
                for i in items
            ],
            "total": total,
        }

    async def run_test_set(
        self,
        db: AsyncSession,
        redis: Redis,
        user_id: int,
        knowledge_base_id: int,
        test_set_id: int,
        top_k: int,
    ) -> RecallTestResultVO:
        """执行单个测试集：按 question 在该库内检索，计算 Recall@K 与命中率。

        单个测试集即一个用例，故 total_cases=1、hit_cases 为 0/1；
        recall_at_k = 命中的期望 chunk 数 / 期望 chunk 总数。
        检索异常时 search 已降级为空结果，期望 chunk 缺失则计入未命中。
        """
        test_set = await knowledge_test_set_repository.get_by_id_and_kb(
            db, test_set_id, knowledge_base_id
        )
        if not test_set:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "测试集不存在")

        result = await search_service.search(
            db,
            redis,
            user_id,
            test_set.question,
            knowledge_base_ids=[knowledge_base_id],
            top_k=top_k,
        )
        hit_ids = {r["chunkId"] for r in result.get("results", [])}
        expected = test_set.expected_chunk_ids or []
        matched = sum(1 for cid in expected if cid in hit_ids)
        total_expected = len(expected)
        recall_at_k = matched / total_expected if total_expected else 0.0
        hit_cases = 1 if matched > 0 else 0
        return RecallTestResultVO(
            test_set_id=test_set_id,
            recall_at_k=recall_at_k,
            hit_rate=hit_cases / 1,
            total_cases=1,
            hit_cases=hit_cases,
        )


test_set_service = TestSetService()
