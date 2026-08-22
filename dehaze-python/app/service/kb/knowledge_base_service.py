"""AI 知识库 Service：知识库生命周期管理（创建/编辑/删除/查询）与配额/权限校验。

对齐《后端实现-文档管理.md》§2 与《后端实现-架构与公共.md》§5/§7/§8。
- 公开库仅管理员可管理；私有库仅 owner 可管理，数量受会员等级配额限制
- embedding 模型/分块策略创建后不可修改（否则已有向量维度不兼容）
- ES 索引初始化失败视为创建失败（回滚，不保留孤儿知识库）
"""

import json
import logging

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.es.kb_chunk_index import delete_kb_index, ensure_kb_index
from app.models.entity.sys_knowledge_base import SysKnowledgeBase
from app.models.schema.knowledge_base import (
    CHUNKING_STRATEGY_VALUES,
    EMBEDDING_MODEL_VALUES,
)
from app.repository.knowledge_base_repository import knowledge_base_repository
from app.repository.knowledge_document_repository import knowledge_document_repository
from app.repository.member_repository import member_repository
from app.service.kb.embedding_service import get_embedding_dim

logger = logging.getLogger(__name__)

# 可建私有知识库数上限（会员权益无对应字段，按等级映射；公共库不受限）
_PRIVATE_KB_LIMIT_BY_LEVEL = {
    "level_0": 3,
    "level_1": 10,
    "level_2": 30,
    "level_3": 100,
}

# 缓存 TTL（秒）：kb:list 10min / kb:detail 30min
_KB_LIST_TTL = 600
_KB_DETAIL_TTL = 1800

# 知识库列表默认每页数量（BasePageQuery 默认值），仅默认分页时读写缓存避免 size 污染
_DEFAULT_PAGE_SIZE = 10

# 可编辑字段白名单（分块策略与 embedding 模型创建后不可修改）
_EDITABLE_FIELDS = (
    "name",
    "description",
    "search_strategy",
    "top_k",
    "score_threshold",
    "enable_rerank",
    "rerank_model",
    "hybrid_weight",
)


def _check_visibility_and_strategies(
    visibility: str, embedding_model: str, chunking_strategy: str
) -> None:
    """校验可见性与策略枚举合法性。"""
    if visibility not in ("public", "private"):
        raise BusinessException(ResultCode.PARAM_ERROR, "可见性取值必须为 public/private")
    if embedding_model not in EMBEDDING_MODEL_VALUES:
        raise BusinessException(ResultCode.PARAM_ERROR, "不支持的 embedding 模型")
    if chunking_strategy not in CHUNKING_STRATEGY_VALUES:
        raise BusinessException(ResultCode.PARAM_ERROR, "不支持的 chunking 策略")


async def _check_manage_permission(db: AsyncSession, kb: SysKnowledgeBase, user) -> None:
    """校验知识库管理权限：私有库仅 owner，公开库仅管理员。"""
    if kb.visibility == "public":
        if not user.is_admin:
            raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "普通用户不能管理公共知识库")
        return
    if kb.create_by != user.id:
        raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权操作他人私有知识库")


async def _invalidate_cache(redis: Redis, user_id: int, kb_id: int) -> None:
    """知识库变更后失效相关缓存键。"""
    await redis.delete(f"kb:list:{user_id}", f"kb:detail:{kb_id}", f"kb:config:{kb_id}")


class KnowledgeBaseService:
    """知识库服务（异步版本）"""

    async def create(self, db: AsyncSession, redis: Redis, data: dict, user) -> int:
        """创建知识库。

        Args:
            data: 创建表单字段（已排除 None）
            user: UserContext（id/roles/is_admin）

        Returns:
            新知识库 ID
        """
        visibility = data["visibility"]
        embedding_provider = data.get("embedding_provider", "openai")
        embedding_model = data["embedding_model"]
        chunking_strategy = data["chunking_strategy"]

        _check_visibility_and_strategies(visibility, embedding_model, chunking_strategy)

        # 公开库需管理员角色；私有库校验可建数量配额
        if visibility == "public":
            if not user.is_admin:
                raise BusinessException(
                    ResultCode.ACCESS_UNAUTHORIZED, "创建公共知识库需要管理员权限"
                )
        else:
            limit = await self._resolve_private_kb_limit(db, user.id)
            current = await knowledge_base_repository.count_private_by_owner(db, user.id)
            if current >= limit:
                raise BusinessException(
                    ResultCode.BUSINESS_ERROR,
                    f"私有知识库数量已达上限({limit})，请升级会员",
                )

        # 同 create_by 下名称不得重复（未删除）
        existing = await knowledge_base_repository.get_by_name_and_owner(
            db, data["name"], user.id
        )
        if existing:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "知识库名称已存在")

        # ES 索引维度 = embedding 模型维度，初始化失败则创建失败（回滚）
        dims = get_embedding_dim(embedding_provider, embedding_model)
        if dims <= 0:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"无法获取 embedding 模型 {embedding_model} 的维度"
            )

        kb = SysKnowledgeBase(
            name=data["name"],
            description=data.get("description"),
            visibility=visibility,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
            chunk_size=data.get("chunk_size", 800),
            chunk_overlap=data.get("chunk_overlap", 80),
            search_strategy=data.get("search_strategy", "hybrid"),
            top_k=data.get("top_k", 5),
            score_threshold=data.get("score_threshold", 0.5),
            enable_rerank=1 if data.get("enable_rerank") else 0,
            rerank_model=data.get("rerank_model"),
        )
        created = await knowledge_base_repository.create(db, kb)

        if not await ensure_kb_index(created.id, dims):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "ES 索引初始化失败")

        await redis.delete(f"kb:list:{user.id}")
        return created.id

    async def _resolve_private_kb_limit(self, db: AsyncSession, user_id: int) -> int:
        """按会员等级解析可建私有库数量（会员权益无对应字段时按等级映射常量）。"""
        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            return _PRIVATE_KB_LIMIT_BY_LEVEL["level_0"]
        return _PRIVATE_KB_LIMIT_BY_LEVEL.get(
            member.level_code, _PRIVATE_KB_LIMIT_BY_LEVEL["level_0"]
        )

    async def update(self, db: AsyncSession, redis: Redis, kb_id: int, data: dict, user) -> None:
        """编辑知识库（仅可编辑项；分块策略/embedding 模型不可改）。"""
        kb = await knowledge_base_repository.get_by_id(db, kb_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")
        await _check_manage_permission(db, kb, user)

        # 请求中携带不可修改项 → 拒绝
        if "embedding_model" in data or "chunking_strategy" in data:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, "创建后不可修改 embedding 模型或分块策略"
            )

        # 名称重复校验（改名前且同用户下）
        if "name" in data and data["name"] != kb.name:
            existing = await knowledge_base_repository.get_by_name_and_owner(
                db, data["name"], user.id
            )
            if existing:
                raise BusinessException(ResultCode.BUSINESS_ERROR, "知识库名称已存在")

        update_data = {}
        for field in _EDITABLE_FIELDS:
            if field in data:
                if field == "enable_rerank":
                    update_data[field] = 1 if data[field] else 0
                else:
                    update_data[field] = data[field]
        if update_data:
            await knowledge_base_repository.update(db, kb, update_data)

        await _invalidate_cache(redis, user.id, kb_id)

    async def delete(self, db: AsyncSession, redis: Redis, kb_id: int, user) -> None:
        """删除知识库：软删库+文档、删 ES 索引、保留分块记录。"""
        kb = await knowledge_base_repository.get_by_id(db, kb_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")
        await _check_manage_permission(db, kb, user)

        # 软删知识库与关联文档（分块记录保留）
        await knowledge_base_repository.soft_delete_by_ids(db, [kb_id])
        doc_ids = await knowledge_document_repository.list_ids_by_kb(db, kb_id)
        if doc_ids:
            await knowledge_document_repository.soft_delete_by_ids(db, doc_ids)

        await delete_kb_index(kb_id)
        await _invalidate_cache(redis, user.id, kb_id)

    async def get_detail(self, db: AsyncSession, redis: Redis, kb_id: int, user_id: int) -> dict:
        """知识库详情（含统计），私有库仅 owner 可见；走 30min 缓存。"""
        cached = await redis.get(f"kb:detail:{kb_id}")
        if cached:
            return json.loads(cached)

        kb = await knowledge_base_repository.get_by_id(db, kb_id)
        if not kb:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在")
        if kb.visibility == "private" and kb.create_by != user_id:
            raise BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权查看他人私有知识库")

        from app.models.schema.knowledge_base import KnowledgeBaseVO

        vo = KnowledgeBaseVO.model_validate(kb)
        result = vo.model_dump(mode="json", by_alias=True)
        await redis.set(
            f"kb:detail:{kb_id}", json.dumps(result, ensure_ascii=False), ex=_KB_DETAIL_TTL
        )
        return result

    async def get_page(self, 
        db: AsyncSession,
        redis: Redis,
        user_id: int,
        keyword: str | None,
        page: int,
        size: int,
    ) -> dict:
        """知识库列表（paginate_visible 已按可见性过滤）；走 10min 缓存。"""
        cache_key = f"kb:list:{user_id}"
        # 仅默认分页(无关键词/第一页/默认 size)读写缓存，避免不同 size 互相污染
        cacheable = not keyword and page == 1 and size == _DEFAULT_PAGE_SIZE
        if cacheable:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)

        items, total = await knowledge_base_repository.paginate_visible(
            db, user_id, keyword, page, size
        )
        from app.models.schema.knowledge_base import KnowledgeBaseVO

        result = {
            "list": [
                KnowledgeBaseVO.model_validate(i).model_dump(mode="json", by_alias=True)
                for i in items
            ],
            "total": total,
        }
        if cacheable:
            await redis.set(cache_key, json.dumps(result, ensure_ascii=False), ex=_KB_LIST_TTL)
        return result


knowledge_base_service = KnowledgeBaseService()
