"""Agent 配置三级合并解析器

配置优先级（低→高）：系统默认（代码常量 REASONING_DEFAULTS / sys_dict: ai_guardrail_defaults）
← Agent 配置（sys_ai_agent.config JSON）← 会话级覆盖（sys_ai_conversation.config）。
高优先级配置项覆盖低优先级同名项，未覆盖的继承低优先级默认值。
"""

from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.cache.cache import CacheService
from app.repository.dict_repository import dict_repository

# 护栏系统默认字典类型：安全合规开关有运维调整场景，存 sys_dict；
# 推理参数无管理员运行时修改场景，用代码常量（REASONING_DEFAULTS）
GUARDRAIL_DEFAULTS_DICT = "ai_guardrail_defaults"
# 护栏默认配置缓存（sys_dict 变更低频，10 分钟失效）
_GUARDRAIL_DEFAULTS_CACHE_KEY = "ai:config:guardrail_defaults"
_GUARDRAIL_DEFAULTS_CACHE_TTL = 600

# 推理参数系统默认值（代码常量）：参数随算法演进由版本调整，个性化需求
# 由 Agent 级/会话级配置逐级覆盖，不设全局字典默认层
REASONING_DEFAULTS: dict[str, Any] = {
    "max_steps_react": 20,
    "max_steps_plan": 30,
    "max_steps_reflexion": 15,
    "max_iterations_reflexion": 3,
    "reflexion_threshold": 0.8,
    "max_parallel": 5,
    "tool_timeout": 60,
    "token_budget": 500000,
    "retry_max": 2,
}


async def _load_dict_values(db: AsyncSession, type_code: str) -> dict[str, Any]:
    """加载某字典类型的 {name: value} 映射，value 按可解析类型转换（int/float/bool/str）。"""
    items: dict[str, Any] = {}
    for item in await dict_repository.list_enabled_by_type_code(db, type_code):
        items[item.name] = _coerce_scalar(item.value)
    return items


def _coerce_scalar(raw: str) -> Any:
    """将字典值字符串转换为 int/float/bool，无法转换则保留字符串。"""
    if isinstance(raw, (int, float, bool)):
        return raw
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _merge_config(defaults: dict[str, Any], *overrides: dict[str, Any] | None) -> dict[str, Any]:
    """按优先级顺序合并多个配置层（后面的覆盖前面的），None 层忽略。"""
    merged: dict[str, Any] = dict(defaults)
    for layer in overrides:
        if layer:
            merged.update(layer)
    return merged


def _merge_guardrails(
    defaults: dict[str, Any], *overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """护栏逐规则合并：默认（子规则 enabled 默认 true）+ Agent/会话覆盖。

    会话覆盖可能为整个 guardrails 子对象，也可能仅覆盖个别规则；对子规则对象做深度合并。
    """
    result: dict[str, Any] = {k: dict(v) if isinstance(v, dict) else v for k, v in defaults.items()}
    for layer in overrides:
        if not isinstance(layer, dict):
            continue
        for rule_name, rule in layer.items():
            if isinstance(rule, dict) and isinstance(result.get(rule_name), dict):
                result[rule_name] = {**result[rule_name], **rule}
            else:
                result[rule_name] = rule
    return result


def _nest_dotted(flat: dict[str, Any]) -> dict[str, Any]:
    """将字典点分键（prompt_injection.enabled）组装为嵌套结构。

    sys_dict 以平铺 name 存储护栏参数，运行时消费方（GuardrailMiddleware）与
    Agent 级 config.guardrails 均为嵌套 {规则名: {参数}} 结构，加载时统一转换。
    """
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


async def load_guardrail_defaults(db: AsyncSession, redis: Redis) -> dict[str, Any]:
    """加载并缓存护栏系统默认配置（ai:config:guardrail_defaults，10 分钟）。

    返回嵌套结构 {规则名: {参数}}，与 GuardrailMiddleware 及
    Agent 级 config.guardrails 的消费口径一致。
    """
    cache = CacheService(redis)
    cached = await cache.get_json(_GUARDRAIL_DEFAULTS_CACHE_KEY)
    if cached is not None:
        return cached
    guardrails = _nest_dotted(await _load_dict_values(db, GUARDRAIL_DEFAULTS_DICT))
    await cache.set_json(_GUARDRAIL_DEFAULTS_CACHE_KEY, guardrails, _GUARDRAIL_DEFAULTS_CACHE_TTL)
    return guardrails


async def invalidate_guardrail_defaults(redis: Redis) -> None:
    """ai_guardrail_defaults 字典更新时失效护栏默认配置缓存。"""
    await CacheService(redis).delete(_GUARDRAIL_DEFAULTS_CACHE_KEY)


async def resolve(
    db: AsyncSession,
    redis: Redis,
    agent_config: dict[str, Any] | None,
    conversation_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """三级合并：系统默认 ← Agent 配置 ← 会话覆盖，产出扁平生效配置。

    Args:
        agent_config: sys_ai_agent.config JSON（推理参数 + 可选 guardrails 子对象）
        conversation_config: 会话级覆盖（同构）

    Returns:
        扁平配置 dict：推理参数平铺于顶层，护栏汇总于 guardrails 子对象。
        与消费方读取口径一致（如 config.get("max_steps")、config.get("guardrails")）。
    """
    agent_cfg = agent_config or {}
    conv_cfg = conversation_config or {}

    # 推理参数各层剥离护栏子对象，避免 guardrails 泄漏进推理参数合并
    reasoning = _merge_config(
        REASONING_DEFAULTS,
        {k: v for k, v in agent_cfg.items() if k != "guardrails"},
        {k: v for k, v in conv_cfg.items() if k != "guardrails"},
    )
    guardrails = _merge_guardrails(
        await load_guardrail_defaults(db, redis),
        (agent_cfg.get("guardrails") if isinstance(agent_cfg.get("guardrails"), dict) else None),
        (conv_cfg.get("guardrails") if isinstance(conv_cfg.get("guardrails"), dict) else None),
    )
    return {**reasoning, "guardrails": guardrails}
