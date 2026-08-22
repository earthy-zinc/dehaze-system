"""AI 模型管理服务"""

import logging

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService
from app.models.entity.sys_ai_model import SysAiModel
from app.models.schema.ai_conversation import AiModelResult
from app.models.schema.common import PageResult
from app.repository.ai_model_repository import ai_model_repository
from app.repository.member_repository import member_repository
from app.service.ai.provider_health_service import provider_health_service
from app.service.message_service import MessageService

logger = logging.getLogger(__name__)

# 启用模型列表缓存（模型配置低频变更，缓存降低模型选择时的 DB 压力）
MODEL_LIST_CACHE_KEY = "ai:model:list"
MODEL_LIST_CACHE_TTL = CACHE_TTL_HOUR

# 会员等级编码 -> 数值等级（用于模型 VIP 过滤）
_LEVEL_MAP = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}

# 会员等级缓存（用户等级变更低频，缓存降低会员等级查询的 DB 压力）
USER_LEVEL_CACHE_KEY = "user:level:"
USER_LEVEL_CACHE_TTL = 1800  # 30 分钟

# 布尔开关字段（update_model 时统一转 int 存储）
_BOOL_FIELDS = frozenset(
    {
        "supports_multimodal",
        "supports_tool_call",
        "supports_streaming",
        "supports_prompt_cache",
        "supports_structured_output",
    }
)


async def _get_user_level(db: AsyncSession, redis: Redis, user_id: int) -> int:
    cache = CacheService(redis)
    key = f"{USER_LEVEL_CACHE_KEY}{user_id}"
    cached = await cache.get_json(key)
    if cached is not None:
        return cached
    member = await member_repository.get_by_user_id(db, user_id)
    level = _LEVEL_MAP.get(member.level_code, 0) if member else 0
    await cache.set_json(key, level, USER_LEVEL_CACHE_TTL)
    return level


async def _clear_model_cache(redis: Redis) -> None:
    await CacheService(redis).delete(MODEL_LIST_CACHE_KEY)


# 降级链最大深度：防配置环导致的无限递归
_FALLBACK_CHAIN_MAX_DEPTH = 5


def _model_route(model: SysAiModel) -> dict:
    """将模型实体序列化为候选路由项"""
    return {
        "model_pk": model.id,
        "model_id": model.model_id,
        "provider_id": model.provider_id,
        "model_config": {
            "max_output_tokens": model.max_output_tokens,
            "max_context_tokens": model.max_context_tokens,
            "supports_multimodal": model.supports_multimodal,
            "supports_tool_call": model.supports_tool_call,
            "supports_streaming": model.supports_streaming,
        },
    }


def _model_meets_caps(model: SysAiModel, required_caps: set[str]) -> bool:
    """校验模型能力是否满足全部要求（required_caps 中不存在的能力视为不要求）"""
    for cap in required_caps:
        if cap == "multimodal" and not model.supports_multimodal:
            return False
        if cap == "tool_call" and not model.supports_tool_call:
            return False
        if cap == "streaming" and not model.supports_streaming:
            return False
    return True


def _capabilities_of(model: SysAiModel) -> set[str]:
    """返回模型已声明支持的能力集合"""
    caps = set()
    if model.supports_multimodal:
        caps.add("multimodal")
    if model.supports_tool_call:
        caps.add("tool_call")
    if model.supports_streaming:
        caps.add("streaming")
    return caps


async def _provider_health_snapshot(redis, provider_id: int) -> dict:
    """读取单个供应商健康快照（P95 延迟等），契约来自 provider_health_service。

    健康聚合为按供应商逐条调用，失败时返回空快照（列表该模型档位降为 unknown）。
    """
    try:
        return await provider_health_service.get_health_snapshot(redis, provider_id)
    except Exception as exc:  # noqa: BLE001 健康快照读取失败不影响模型列表
        logger.warning("读取供应商健康快照失败: provider_id=%s err=%s", provider_id, exc)
        return {}


def _speed_tier_of(snapshot: dict) -> str:
    """由供应商健康快照的 P95 延迟推导速度档位（fast/medium/slow/unknown）"""
    p95_ms = (snapshot or {}).get("p95_latency_ms")
    if p95_ms is None:
        return "unknown"
    if p95_ms < settings.AI_SPEED_TIER_FAST_P95_MS:
        return "fast"
    if p95_ms < settings.AI_SPEED_TIER_MEDIUM_P95_MS:
        return "medium"
    return "slow"


async def _resolve_fallback(
    db: AsyncSession,
    model: SysAiModel,
) -> SysAiModel | None:
    """取 fallback_model_pk 指向的启用替换模型，无配置或不可用时返回 None"""
    if not model.fallback_model_pk:
        return None
    rows = await ai_model_repository.list_enabled_by_pks(db, [model.fallback_model_pk])
    return rows[0] if rows else None


async def _notify_model_replacement(
    db: AsyncSession,
    model: SysAiModel,
    fallback: SysAiModel | None,
) -> None:
    """向使用该模型的所有活跃会话用户推送替换模型推荐（消息中心站内信）。

    fallback 为 None 时提示管理员配置替换模型；取不到时静默跳过（无活跃用户）。
    """
    user_ids = await ai_model_repository.list_active_conversation_users(db, model.model_id)
    if not user_ids:
        return
    if fallback:
        title = f"模型 {model.display_name} 即将不可用"
        content = (
            f"您正在使用的模型「{model.display_name}」即将停用，"
            f"建议切换到替代模型「{fallback.display_name}」。"
        )
    else:
        title = f"模型 {model.display_name} 即将不可用"
        content = (
            f"您正在使用的模型「{model.display_name}」即将停用，暂未配置替代模型，"
            "请及时更换其他可用模型。"
        )
    try:
        await MessageService.send(
            db,
            {
                "type": "business",
                "title": title,
                "content": content,
                "priority": 3,
                "recipientIds": user_ids,
                "bizModule": "ai_model",
                "bizId": model.model_id,
            },
        )
    except Exception as exc:  # noqa: BLE001 通知失败不阻断模型下线/禁用
        logger.warning("模型下线通知失败: model_id=%s err=%s", model.model_id, exc)


class AiModelService:
    @staticmethod
    async def list_models(
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
    ) -> PageResult[AiModelResult]:
        models, total = await ai_model_repository.paginate_models(db, page, size, keyword)
        return PageResult(list=[AiModelResult.model_validate(m) for m in models], total=total)

    @staticmethod
    async def list_enabled_models(
        db: AsyncSession,
        redis: Redis,
        user_id: int,
    ) -> list[AiModelResult]:
        cache = CacheService(redis)
        cached = await cache.get_json(MODEL_LIST_CACHE_KEY)
        if cached is None:
            models = await ai_model_repository.list_enabled(db)
            # 降级标识：被其他启用模型作为 fallback 目标
            fallback_target_pks = {
                m.fallback_model_pk for m in models if m.fallback_model_pk is not None
            }
            cached = [
                {
                    **AiModelResult.model_validate(m).model_dump(mode="json"),
                    "speed_tier": _speed_tier_of(
                        await _provider_health_snapshot(redis, m.provider_id)
                    ),
                    "is_fallback_target": m.id in fallback_target_pks,
                }
                for m in models
            ]
            await cache.set_json(MODEL_LIST_CACHE_KEY, cached, MODEL_LIST_CACHE_TTL)
        user_level = await _get_user_level(db, redis, user_id)
        return [
            AiModelResult.model_validate(item)
            for item in cached
            if item.get("vip_level", 0) <= user_level
        ]

    @staticmethod
    async def validate_model_caps(
        model: SysAiModel,
        has_attachments: bool,
        need_tools: bool,
    ) -> None:
        """发送消息前校验模型能力，不满足抛 A0601。

        多模态：消息含图片/文件附件必须 supports_multimodal=1；
        工具调用：本次推理需工具调用必须 supports_tool_call=1。
        """
        if has_attachments and not model.supports_multimodal:
            raise BusinessException(
                ResultCode.AI_MODEL_NOT_AVAILABLE, "当前模型不支持图片/文件多模态输入"
            )
        if need_tools and not model.supports_tool_call:
            raise BusinessException(ResultCode.AI_MODEL_NOT_AVAILABLE, "当前模型不支持工具调用")

    @staticmethod
    async def get_call_routes(
        db: AsyncSession,
        model_id: str,
        required_caps: set[str],
    ) -> list[dict]:
        """降级链候选路由序列：[{"model_pk","model_id","provider_id","model_config"}...]。

        顺序：该 model_id 全部启用行（当前/备用供应商）→ 降级链各级
        （fallback_model_pk 逐级，能力匹配过滤 required_caps ⊆ 模型 supports_*，
        防环：已出现的 model_pk 跳过，链深上限 5）。
        """
        routes: list[dict] = []
        seen: set[int] = set()

        # 1. 当前 model_id 的全部启用行（同模型多供应商，保持优先级）
        current_rows = await ai_model_repository.list_enabled_by_model_id(db, model_id)
        for row in current_rows:
            seen.add(row.id)
            routes.append(_model_route(row))

        # 2. 沿降级链逐级扩展（按主键引用，能力匹配过滤）
        pending = current_rows
        depth = 0
        while pending and depth < _FALLBACK_CHAIN_MAX_DEPTH:
            next_targets = [
                m.fallback_model_pk
                for m in pending
                if m.fallback_model_pk is not None and m.fallback_model_pk not in seen
            ]
            if not next_targets:
                break
            fallback_rows = await ai_model_repository.list_enabled_by_pks(db, next_targets)
            depth += 1
            matched: list[SysAiModel] = []
            for row in fallback_rows:
                if row.id in seen:
                    continue
                seen.add(row.id)
                if _model_meets_caps(row, required_caps):
                    matched.append(row)
                    routes.append(_model_route(row))
            pending = matched

        return routes

    @staticmethod
    async def create_model(db: AsyncSession, redis: Redis, form) -> AiModelResult:
        existing = await ai_model_repository.get_by_model_and_provider(
            db, form.model_id, form.provider_id
        )
        if existing:
            if existing.deleted:
                raise BusinessException(
                    ResultCode.DATA_EXISTS, "该模型+供应商组合已被历史记录占用，不可复用"
                )
            raise BusinessException(ResultCode.DATA_EXISTS, "该模型+供应商组合已存在")
        model = SysAiModel(
            provider_id=form.provider_id,
            model_id=form.model_id,
            display_name=form.display_name,
            input_rate=form.input_rate,
            output_rate=form.output_rate,
            cached_rate=form.cached_rate,
            max_context_tokens=form.max_context_tokens,
            max_output_tokens=form.max_output_tokens,
            supports_multimodal=int(form.supports_multimodal),
            supports_tool_call=int(form.supports_tool_call),
            supports_streaming=int(form.supports_streaming),
            supports_prompt_cache=int(form.supports_prompt_cache),
            supports_structured_output=int(form.supports_structured_output),
            fallback_model_pk=form.fallback_model_pk,
            prompt_cache_prefix_len=form.prompt_cache_prefix_len,
            status=form.status,
            vip_level=form.vip_level,
        )
        model = await ai_model_repository.create(db, model)
        await _clear_model_cache(redis)
        return AiModelResult.model_validate(model)

    @staticmethod
    async def update_model(db: AsyncSession, redis: Redis, model_id: str, form) -> AiModelResult:
        model = await ai_model_repository.get_by_model_id(db, model_id)
        if not model:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模型不存在")
        data = form.model_dump(exclude_unset=True)
        old_status = model.status
        for key, value in data.items():
            if key in _BOOL_FIELDS:
                value = int(value)
            setattr(model, key, value)
        # 禁用模型（status 1→0）即"标记即将下线"，向使用中会话推送替换模型推荐
        disabling = "status" in data and old_status == 1 and int(data["status"]) == 0
        await db.flush()
        await db.refresh(model)
        if disabling and not model.deleted:
            fallback = await _resolve_fallback(db, model)
            await _notify_model_replacement(db, model, fallback)
        await _clear_model_cache(redis)
        return AiModelResult.model_validate(model)

    @staticmethod
    async def delete_model(db: AsyncSession, redis: Redis, model_id: str) -> None:
        model = await ai_model_repository.get_by_model_id(db, model_id)
        if not model:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模型不存在")
        active = await ai_model_repository.count_active_conversations(db, model_id)
        if active > 0:
            raise BusinessException(
                ResultCode.DATA_BIND_EXISTS,
                "存在活跃会话正在使用该模型，请先禁用（status=0）",
            )
        await ai_model_repository.soft_delete_by_ids(db, [model.id])
        await _clear_model_cache(redis)
