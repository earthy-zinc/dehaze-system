"""
字典服务

提供字典项和字典类型的 CRUD 功能，包含唯一性校验、删除约束检查和缓存支持

设计参照: dehaze-doc/docs/03-模块设计/基础模块/字典管理/后端实现.md
"""

import logging
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService
from app.models.entity.sys_dict import SysDict, SysDictType
from app.repository.dict_repository import dict_repository, dict_type_repository

logger = logging.getLogger(__name__)

DICT_OPTIONS_CACHE_PREFIX = "dict:options:"
DICT_OPTIONS_CACHE_TTL = CACHE_TTL_HOUR

# 系统预置字典类型编码（T-DM-025：预置类型不可删除），与 config/sql/data/sys_dict_type.sql 种子一致
SYSTEM_PRESET_DICT_TYPE_CODES = frozenset(
    {
        "gender",
        "ai_reasoning_defaults",
        "ai_guardrail_defaults",
        "ai_provider_health",
        "ai_embedding",
    }
)

# AI 相关系统预置字典默认项（对齐 config/sql/data/sys_dict.sql 种子）。
# 运行面（AgentConfigResolver / deep_agent_builder）以这些 sys_dict 默认值为
# 唯一兜底来源（代码无硬编码，缺键快速失败）；数据库初始化可能晚于表创建，
# 故启动时幂等补齐缺失项，保证内置推理链开箱即用。
_AI_DICT_TYPE_SEEDS: list[tuple[str, str]] = [
    # (type_code, 显示名)
    ("ai_reasoning_defaults", "AI 推理参数默认"),
    ("ai_guardrail_defaults", "AI 护栏默认"),
    ("ai_provider_health", "AI 供应商健康"),
    ("ai_embedding", "AI 向量化 Embedding"),
]

# (type_code, name, value, sort, defaulted, remark)
_AI_DICT_ITEM_SEEDS: list[tuple[str, str, str, int, int, str]] = [
    ("ai_reasoning_defaults", "max_steps_react", "20", 1, 1, "ReAct 最大推理步数"),
    ("ai_reasoning_defaults", "max_steps_plan", "30", 2, 1, "Plan-and-Execute 最大推理步数"),
    ("ai_reasoning_defaults", "max_steps_reflexion", "15", 3, 1, "Reflexion 单次迭代最大步数"),
    ("ai_reasoning_defaults", "max_iterations_reflexion", "3", 4, 1, "Reflexion 最大迭代次数"),
    ("ai_reasoning_defaults", "reflexion_threshold", "0.8", 5, 1, "Reflexion 质量达标阈值"),
    ("ai_reasoning_defaults", "max_parallel", "5", 6, 1, "并行子任务最大数"),
    ("ai_reasoning_defaults", "tool_timeout", "60", 7, 1, "单工具调用超时（秒）"),
    ("ai_reasoning_defaults", "token_budget", "500000", 8, 1, "单会话 Token 预算上限"),
    ("ai_reasoning_defaults", "retry_max", "2", 9, 1, "工具调用失败最大重试次数"),
    ("ai_guardrail_defaults", "prompt_injection.enabled", "true", 1, 1, "Prompt 注入防护开关"),
    ("ai_guardrail_defaults", "unauthorized_access.enabled", "true", 2, 1, "越权查询检测开关"),
    ("ai_guardrail_defaults", "sensitive_topic.enabled", "false", 3, 1, "敏感话题过滤开关"),
    ("ai_guardrail_defaults", "pii_mask.enabled", "true", 4, 1, "敏感信息脱敏开关"),
    ("ai_guardrail_defaults", "fact_check.enabled", "false", 5, 1, "事实性校验开关"),
    ("ai_guardrail_defaults", "format_check.enabled", "false", 6, 1, "格式合规校验开关"),
    ("ai_provider_health", "error_rate_warn", "0.1", 1, 1, "可疑错误率阈值(≥即为可疑)"),
    ("ai_provider_health", "error_rate_open", "0.3", 2, 1, "熔断错误率阈值(≥即为熔断)"),
    ("ai_provider_health", "min_window_calls", "20", 3, 1, "错误率判定最小调用窗口"),
    ("ai_provider_health", "consecutive_failures", "5", 4, 1, "连续失败熔断阈值"),
    ("ai_provider_health", "circuit_cooldown", "60", 5, 1, "熔断冷却时长(秒)"),
    ("ai_embedding", "provider_code", "openai", 1, 1, "Embedding 供应商编码(经 ai_provider 体系取 Key)"),
    ("ai_embedding", "model", "text-embedding-3-small", 2, 1, "Embedding 模型标识"),
    ("ai_embedding", "dims", "1536", 3, 1, "向量维度(ES dense_vector dims 联动)"),
]


async def ensure_ai_dict_defaults(db: AsyncSession, redis: Redis) -> None:
    """幂等补齐 AI 相关系统预置字典类型与默认项（缺失才补，不覆盖管理员修改）。

    运行面（AgentConfigResolver）以 ai_reasoning_defaults / ai_guardrail_defaults
    为配置唯一默认来源；数据库种子可能未同步，这里在启动时按 sys_dict.sql 契约补齐，
    保证内置推理链不因缺默认参数而快速失败。
    """
    for code, display_name in _AI_DICT_TYPE_SEEDS:
        if await dict_type_repository.get_by_code(db, code) is None:
            db.add(SysDictType(name=display_name, code=code, status=1))
    for type_code, name, value, sort, defaulted, remark in _AI_DICT_ITEM_SEEDS:
        # 仅缺项补齐；已存在的（含管理员改过值）一律保留，避免覆盖人工配置
        existing = await dict_repository.get_by_type_code_and_name(db, type_code, name)
        if existing is None:
            db.add(
                SysDict(
                    type_code=type_code,
                    name=name,
                    value=value,
                    sort=sort,
                    status=1,
                    defaulted=defaulted,
                    remark=remark,
                )
            )
    try:
        await db.flush()
    except IntegrityError:
        # 多进程并发启动时的 check-then-insert 竞态：另一进程已插入相同种子
        # （uk_type_value 唯一键冲突）。丢弃本批插入即可——库中已存在等价数据。
        await db.rollback()
    # 失效配置默认值缓存，避免补齐后仍命中旧的空缓存
    from app.service.ai.strategies.agent_config_resolver import invalidate_defaults

    await invalidate_defaults(redis)


class DictService:
    """字典服务"""

    async def get_dict_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        keywords: str | None = None,
        type_code: str | None = None,
    ) -> tuple[list, int]:
        """获取字典分页列表"""
        return await dict_repository.get_page(db, page, page_size, keywords, type_code)

    async def get_dict_form(self, db: AsyncSession, dict_id: int) -> dict[str, Any] | None:
        """获取字典表单数据"""
        return await dict_repository.get_form_by_id(db, dict_id)

    async def create_dict(self, db: AsyncSession, redis: Redis, data: dict[str, Any]) -> SysDict:
        """
        创建字典项

        业务规则:
        1. 检查类型编码是否存在
        2. 检查同一类型下键值是否唯一
        3. 创建成功后清除缓存
        """
        type_code = data.get("typeCode")
        value = data.get("value")

        if not type_code:
            raise BusinessException("字典类型编码不能为空")
        if not value:
            raise BusinessException("字典值不能为空")

        dict_type = await dict_type_repository.get_by_code(db, type_code)
        if not dict_type:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典类型不存在")

        existing = await dict_repository.get_by_type_code_and_value(db, type_code, value)
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "该类型下字典值已存在")

        result = await dict_repository.create_dict(db, data)

        await CacheService(redis).delete(f"{DICT_OPTIONS_CACHE_PREFIX}{type_code}")

        return result

    async def update_dict(
        self, db: AsyncSession, redis: Redis, dict_id: int, data: dict[str, Any]
    ) -> bool:
        """
        更新字典项

        业务规则:
        1. 检查字典是否存在
        2. typeCode 只读，不可修改
        3. 如果修改了 value，检查唯一性
        4. 更新成功后清除相关缓存
        """
        old_dict = await dict_repository.get_by_id(db, dict_id)
        if not old_dict:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典不存在")

        # typeCode 只读，移除 form 传入的 typeCode
        data.pop("typeCode", None)
        new_value = data.get("value", old_dict.value)

        if new_value != old_dict.value:
            existing = await dict_repository.get_by_type_code_and_value(
                db, old_dict.type_code, new_value
            )
            if existing and existing.id != dict_id:
                raise BusinessException(ResultCode.DATA_EXISTS, "该类型下字典值已存在")

        result = await dict_repository.update_by_id(db, dict_id, data)

        # 清除缓存（typeCode 不变，只需清除一个）
        if old_dict.type_code:
            await CacheService(redis).delete(f"{DICT_OPTIONS_CACHE_PREFIX}{old_dict.type_code}")

        return result

    async def delete_dict(self, db: AsyncSession, redis: Redis, dict_ids: list[int]) -> bool:
        """
        删除字典项

        业务规则:
        1. 校验字典数据项是否存在
        2. 删除成功后清除相关缓存
        """
        exist_count = await dict_repository.count_by_ids(db, dict_ids)
        if exist_count == 0:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND)

        type_codes = await dict_repository.get_type_codes_by_ids(db, dict_ids)

        result = await dict_repository.delete_by_ids(db, dict_ids)

        cache = CacheService(redis)
        for type_code in type_codes:
            if type_code:
                await cache.delete(f"{DICT_OPTIONS_CACHE_PREFIX}{type_code}")

        return result > 0

    async def list_dict_options(
        self, db: AsyncSession, redis: Redis, type_code: str
    ) -> list[dict[str, Any]]:
        """
        获取字典下拉列表

        使用缓存策略:
        1. 先查缓存
        2. 缓存未命中则查数据库
        3. 写入缓存
        """
        cache = CacheService(redis)
        cache_key = f"{DICT_OPTIONS_CACHE_PREFIX}{type_code}"

        cached = await cache.get_json(cache_key)
        if cached is not None:
            return cached

        options = await dict_repository.list_options_by_type(db, type_code)

        await cache.set_json(cache_key, options, DICT_OPTIONS_CACHE_TTL)

        return options


class DictTypeService:
    """字典类型服务（异步版本）"""

    async def get_dict_type_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        keywords: str | None = None,
    ) -> tuple[list, int]:
        """获取字典类型分页列表"""
        return await dict_type_repository.get_page(db, page, page_size, keywords)

    async def get_dict_type_form(self, db: AsyncSession, type_id: int) -> dict[str, Any] | None:
        """获取字典类型表单数据"""
        return await dict_type_repository.get_form_by_id(db, type_id)

    async def create_dict_type(self, db: AsyncSession, data: dict[str, Any]) -> SysDictType:
        """
        创建字典类型

        业务规则:
        1. 检查编码唯一性
        """
        code = data.get("code")

        if not code:
            raise BusinessException("字典类型编码不能为空")

        existing = await dict_type_repository.get_by_code(db, code)
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "字典类型编码已被历史记录占用")

        result = await dict_type_repository.create_type(db, data)
        return result

    async def update_dict_type(
        self, db: AsyncSession, redis: Redis, type_id: int, data: dict[str, Any]
    ) -> bool:
        """
        更新字典类型

        业务规则:
        1. 检查字典类型是否存在
        2. code 只读，不可修改（T-DM-015：编码创建后不可变）
        """
        old_type = await dict_type_repository.get_by_id(db, type_id)
        if not old_type:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典类型不存在")

        new_code = data.get("code")

        # code 只读：禁止修改编码
        if new_code is not None and new_code != old_type.code:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "字典类型编码不可修改")

        data.pop("code", None)
        result = await dict_type_repository.update_by_id(db, type_id, data)

        if result and old_type.code:
            await CacheService(redis).delete(f"{DICT_OPTIONS_CACHE_PREFIX}{old_type.code}")

        return result

    async def delete_dict_types(
        self, db: AsyncSession, redis: Redis, type_ids: list[int], force: bool = False
    ) -> bool:
        """
        删除字典类型

        业务规则:
        1. 校验字典类型是否存在
        2. force=False 时检查是否存在关联的字典数据，存在则禁止删除
        3. force=True 时级联删除关联的字典数据
        """
        exist_count = await dict_type_repository.count_by_ids(db, type_ids)
        if exist_count == 0:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND)

        # 批量查询类型编码，批量检查关联数据（避免 N+1）
        dict_types = await dict_type_repository.get_by_ids(db, type_ids)
        type_codes = [dt.code for dt in dict_types if dt.code]

        # T-DM-025：系统预置字典类型不可删除
        preset_hit = next((dt.code for dt in dict_types if dt.code in SYSTEM_PRESET_DICT_TYPE_CODES), None)
        if preset_hit:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "系统预置字典类型不可删除")

        if type_codes:
            if force:
                await dict_repository.delete_by_type_codes(db, type_codes)
                cache = CacheService(redis)
                for code in type_codes:
                    await cache.delete(f"{DICT_OPTIONS_CACHE_PREFIX}{code}")
            else:
                counts = await dict_repository.count_by_type_codes(db, type_codes)
                for dt in dict_types:
                    if counts.get(dt.code, 0) > 0:
                        raise BusinessException(
                            ResultCode.DATA_BIND_EXISTS, "存在关联的字典数据，无法删除"
                        )

        result = await dict_type_repository.delete_by_ids(db, type_ids)
        return result > 0


dict_service = DictService()
dict_type_service = DictTypeService()
