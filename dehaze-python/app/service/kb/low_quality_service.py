"""AI 知识库低质量片段 Service。

低质量片段定义为"被用户点踩的片段"，数据源为 sys_knowledge_chunk_feedback（rating=-1）。
权限（kb:manage / 私有库可见性）由 Router 校验，本服务仅负责存在性校验与查询聚合。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.knowledge_base import LowQualityChunkVO
from app.repository.knowledge_base_repository import knowledge_base_repository
from app.repository.knowledge_chunk_feedback_repository import (
    knowledge_chunk_feedback_repository,
)


class LowQualityService:
    def __init__(
        self,
        knowledge_base_repository=knowledge_base_repository,
        knowledge_chunk_feedback_repository=knowledge_chunk_feedback_repository,
    ):
        self.knowledge_base_repository = knowledge_base_repository
        self.knowledge_chunk_feedback_repository = knowledge_chunk_feedback_repository

    async def list_low_quality_chunks(
        self, db: AsyncSession, kb_id: int, page: int, size: int
    ) -> dict:
        """按知识库查被点踩片段（thumbs_down_count 降序，分页）。

        Returns:
            {"list": [LowQualityChunkVO...], "total": int}
        """
        kb = await self.knowledge_base_repository.get_by_id(db, kb_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")

        rows, total = await self.knowledge_chunk_feedback_repository.list_low_quality_by_kb(
            db, kb_id, page, size
        )
        return {
            "list": [
                LowQualityChunkVO(**row).model_dump(mode="json", by_alias=True)
                for row in rows
            ],
            "total": total,
        }


low_quality_service = LowQualityService()
