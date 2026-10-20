"""模型默认值与上下文预算（跨链路共用）

- 默认对话模型解析链：请求显式指定（或会话已存）> **用户偏好**
  （`sys_ai_user_preference.default_model`，用户在设置里选的默认模型）> **Agent 配置模型**
  （会话锚定 Agent 的已发布版本快照 `model_id`，无已发布版本回退主表可编辑态）> **平台默认**
  （`sys_dict` 配置：类型 `ai_model` / 键 `default_model`，管理端字典管理可改）。
  用户偏好只作用于**用户显式发起的对话推理**（新建会话/发消息/编辑重发/重新生成/兼容入口），
  后台派生推理（标题生成/记忆抽取/建议生成/试运行）取平台默认；三者均未配置即报错，
  **不回落代码常量**（配置缺失属部署错误，静默兜底会让"用错模型"变成难查的隐性问题）。
  Agent 级排在用户偏好之后：用户显式偏好优先于系统预置的 Agent 配置。
- 输入侧上下文预算：max_context_tokens − 输出预留 − 系统余量，供消息入口的单条长度校验与
  组装/推理层的上下文裁剪共用（预算以 token 计，消息长度按 1 字符≈1 token 保守折算）。
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_agent import SysAiAgent
from app.repository.ai_agent_repository import ai_agent_repository
from app.repository.ai_agent_version_repository import ai_agent_version_repository
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_user_preference_repository import ai_user_preference_repository
from app.service.dict_service import get_dict_value

logger = logging.getLogger(__name__)

# 平台默认模型所在字典（类型/键）
DEFAULT_MODEL_DICT_TYPE = "ai_model"
DEFAULT_MODEL_DICT_NAME = "default_model"


def default_model_dict_key(model_type: str) -> str:
    """某模型类型的平台默认模型字典键：chat 用 `default_model`，其余 `default_model.{model_type}`

    与 `config/sql/data/sys_dict.sql` 的键约定一致（`default_model` / `default_model.asr` /
    `default_model.tts` / `default_model.image_edit` …）；未配置的类型返回的键不存在，
    调用方按"该类型无平台默认"处理。
    """
    return (
        DEFAULT_MODEL_DICT_NAME
        if model_type == "chat"
        else f"{DEFAULT_MODEL_DICT_NAME}.{model_type}"
    )


async def resolve_platform_default_model_id(db: AsyncSession, model_type: str) -> str | None:
    """某模型类型的平台默认模型标识（`sys_dict: ai_model.default_model[.{model_type}]`）

    与 `resolve_platform_default_model` 同源读法（启用 + 值非空，停用视同缺项）；
    仅在"读取默认模型标记"等展示场景使用，未配置返回 None 而不报错——真正需要模型的
    解析链路仍走各自入口（未配置即报错，不回落代码常量）。
    """
    return await get_dict_value(
        db, DEFAULT_MODEL_DICT_TYPE, default_model_dict_key(model_type), None
    )


# 输出侧预留缺省值与系统余量（未配 max_output_tokens 时的兜底预算口径）
_DEFAULT_OUTPUT_RESERVE_TOKENS = 1024
_OUTPUT_HEADROOM_TOKENS = 200
_MIN_INPUT_BUDGET_TOKENS = 512


async def resolve_platform_default_model(db: AsyncSession) -> str:
    """平台默认对话模型（`sys_dict: ai_model.default_model`）；未配置直接报错

    不含用户偏好/Agent 配置，供"平台默认"这一级独立使用——默认 Agent 自举时也用它，
    避免默认 Agent 的模型同步又反过来解析自身 Agent 配置造成自引用。
    """
    model_id = await resolve_platform_default_model_id(db, "chat")
    if not model_id:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR,
            "未配置平台默认模型（sys_dict: ai_model.default_model），"
            "请先在 AI 模型管理/字典管理中配置",
        )
    return model_id


async def resolve_default_chat_model(
    db: AsyncSession, user_id: int | None = None, agent_code: str | None = None
) -> str:
    """未指定模型时的默认对话模型：用户偏好 > Agent 配置模型 > 平台默认；未配置直接报错

    传入 `user_id` 时先取该用户的偏好默认模型（可选项，取不到不报错）；传入 `agent_code`
    时按会话锚定的 Agent 取模型；不传（后台派生推理，如标题生成/记忆抽取/建议生成/试运行）
    直接走平台默认。
    """
    if user_id is not None:
        preferred = await _resolve_user_preferred_model(db, user_id)
        if preferred:
            return preferred
    agent_model = await _resolve_agent_model(db, agent_code)
    if agent_model:
        return agent_model
    return await resolve_platform_default_model(db)


async def _resolve_agent_model(db: AsyncSession, agent_code: str | None) -> str | None:
    """Agent 配置模型：取已发布版本快照的 model_id（运行面口径，版本化），

    无已发布版本时回退主表可编辑态 `model_id`；Agent 不存在/无模型返回 None（跳到下一级）。
    此处不做启用性判断：Agent 显式配置的模型不可用时由消息入口按统一口径报错，
    静默回落会掩盖管理端的配置错误。
    """
    if not agent_code:
        return None
    agent: SysAiAgent | None = await ai_agent_repository.get_by_code(db, agent_code)
    if agent is None or agent.deleted:
        return None
    version = await ai_agent_version_repository.get_published_snapshot(db, agent.id)
    if version is not None:
        snapshot = ai_agent_version_repository.resolve_snapshot(version.snapshot or {})
        if snapshot.get("model_id"):
            return str(snapshot["model_id"])
    return agent.model_id or None


async def _resolve_user_preferred_model(db: AsyncSession, user_id: int) -> str | None:
    """用户偏好默认模型；模型已停用/删除/非对话型时视为未设置（回落平台默认，数据保留）"""
    preference = await ai_user_preference_repository.get_by_user_id(db, user_id)
    if not preference or not preference.default_model:
        return None
    model = await ai_model_repository.get_by_model_id(db, preference.default_model)
    if model is None or model.status != 1 or model.model_type != "chat":
        return None
    return model.model_id


def input_budget_tokens(model) -> int:
    """单轮输入侧可用 token 预算：max_context_tokens − 输出预留 − 系统余量（下限 512）"""
    output_reserve = model.max_output_tokens or _DEFAULT_OUTPUT_RESERVE_TOKENS
    return max(
        model.max_context_tokens - output_reserve - _OUTPUT_HEADROOM_TOKENS,
        _MIN_INPUT_BUDGET_TOKENS,
    )


async def assert_message_fits_context(
    db: AsyncSession, content: str, model_id: str | None, user_id: int | None = None
) -> None:
    """校验单条消息不超出所选模型的输入预算（按 1 字符≈1 token 保守折算）。

    未指定模型时按默认模型解析链取预算（`user_id` 传入则用户偏好优先）；
    模型行不存在时不在此处判定（模型有效性由后续链路按统一口径报错），避免重复报错口径。
    """
    resolved = model_id or await resolve_default_chat_model(db, user_id)
    model = await ai_model_repository.get_by_model_id(db, resolved)
    if model is None:
        return
    assert_message_fits_model(content, model, resolved)


def assert_message_fits_model(content: str, model, model_id: str) -> None:
    """按已加载的模型行校验单条消息长度（调用方已持有模型行时用，省一次查询）

    超限错误口径与 `assert_message_fits_context` 同源（同一处文案）。
    """
    limit = input_budget_tokens(model)
    if len(content) > limit:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR,
            f"单条消息超出模型上下文预算（{limit} 字符，模型 {model_id}）",
        )
