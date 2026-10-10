import json

from sqlalchemy import func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_model import SysAiModel
from app.repository.base import BaseRepository, escape_like


class AiModelRepository(BaseRepository[SysAiModel]):
    model = SysAiModel

    async def get_by_model_id(
        self,
        db: AsyncSession,
        model_id: str,
    ) -> SysAiModel | None:
        """按业务唯一键 model_id 查询（含禁用/已删除，供更新删除与唯一性校验使用）"""
        stmt = select(SysAiModel).where(SysAiModel.model_id == model_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_model_and_provider(
        self,
        db: AsyncSession,
        model_id: str,
        provider_id: int,
    ) -> SysAiModel | None:
        """按 model_id + provider_id 联合查询（活跃行，供唯一性校验与本地模型播种判重）"""
        stmt = select(SysAiModel).where(
            SysAiModel.model_id == model_id,
            SysAiModel.provider_id == provider_id,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_enabled(
        self, db: AsyncSession, model_type: str | None = None
    ) -> list[SysAiModel]:
        stmt = select(SysAiModel).where(SysAiModel.status == 1)
        if model_type:
            stmt = stmt.where(SysAiModel.model_type == model_type)
        stmt = stmt.order_by(SysAiModel.provider_id, SysAiModel.display_name)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_enabled_by_model_id(
        self,
        db: AsyncSession,
        model_id: str,
    ) -> list[SysAiModel]:
        """查询某 model_id 的全部启用行（同模型可配多供应商，按 id 升序）"""
        stmt = (
            select(SysAiModel)
            .where(SysAiModel.model_id == model_id, SysAiModel.status == 1)
            .order_by(SysAiModel.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_enabled_by_pks(
        self,
        db: AsyncSession,
        pks: list[int],
    ) -> list[SysAiModel]:
        """按主键查询启用模型（降级链候选），无则返回空"""
        if not pks:
            return []
        stmt = (
            select(SysAiModel)
            .where(SysAiModel.id.in_(pks), SysAiModel.status == 1)
            .order_by(SysAiModel.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_fallback_targets(self, db: AsyncSession, model_pk: int) -> int:
        """统计有多少启用模型将该模型作为降级目标（用于列表降级标识）"""
        stmt = (
            select(func.count())
            .select_from(SysAiModel)
            .where(
                SysAiModel.fallback_model_id == model_pk,
                SysAiModel.status == 1,
                SysAiModel.deleted == 0,
            )
        )
        return (await db.execute(stmt)).scalar() or 0

    async def count_active_conversations(self, db: AsyncSession, model_id: str) -> int:
        """统计正在使用该模型的活跃会话数（未删除且状态为活跃）"""
        stmt = (
            select(func.count())
            .select_from(SysAiConversation)
            .where(
                SysAiConversation.model == model_id,
                SysAiConversation.deleted == 0,
                SysAiConversation.status == 1,
            )
        )
        return (await db.execute(stmt)).scalar() or 0

    async def list_active_conversation_users(
        self,
        db: AsyncSession,
        model_id: str,
    ) -> list[int]:
        """统计使用该模型的所有活跃会话用户 ID（去重，供下线/禁用通知）"""
        stmt = (
            select(SysAiConversation.user_id)
            .where(
                SysAiConversation.model == model_id,
                SysAiConversation.deleted == 0,
                SysAiConversation.status == 1,
            )
            .distinct()
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.all()]

    async def paginate_models(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
        model_type: str | None = None,
        task_tags: list[str] | None = None,
    ) -> tuple[list[SysAiModel], int]:
        stmt = select(SysAiModel)
        if keyword:
            escaped = escape_like(keyword)
            pattern = f"%{escaped}%"
            stmt = stmt.where(
                (SysAiModel.display_name.like(pattern, escape="\\"))
                | (SysAiModel.model_id.like(pattern, escape="\\"))
            )
        if model_type:
            stmt = stmt.where(SysAiModel.model_type == model_type)
        if task_tags:
            # 逗号分隔标签取 OR：task_tags JSON 数组含任一即命中；NULL 行不参与命中。
            # 参数用 literal 传 JSON 文本：cast(..., mysql.JSON) 会被 JSON 类型的
            # bind_processor 二次序列化成字符串标量，导致 JSON_OVERLAPS 永不命中
            stmt = stmt.where(
                func.json_overlaps(
                    SysAiModel.task_tags,
                    literal(json.dumps(task_tags, ensure_ascii=False)),
                )
            )
        stmt = stmt.order_by(SysAiModel.id.desc())
        return await self.paginate(db, stmt, page, size)

    async def list_by_provider_and_type(
        self,
        db: AsyncSession,
        provider_id: int,
        model_type: str,
    ) -> list[SysAiModel]:
        """按供应商 + 模型类型查询全部模型行（含停用，供价格播种等注册表维护）"""
        stmt = (
            select(SysAiModel)
            .where(
                SysAiModel.provider_id == provider_id,
                SysAiModel.model_type == model_type,
            )
            .order_by(SysAiModel.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


ai_model_repository = AiModelRepository()
