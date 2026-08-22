"""
AI 对话模块 Schema 模型
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.schema.common import BasePageQuery, OrmResult

# ── 模型管理 ──────────────────────────────────────────


class AiModelCreate(OrmResult):
    provider_id: int = Field(..., description="关联供应商ID")
    model_id: str = Field(..., min_length=1, max_length=64, description="模型标识")
    display_name: str = Field(..., min_length=1, max_length=128, description="显示名称")
    input_rate: float = Field(default=1.0, ge=0, description="输入Token计费比例")
    output_rate: float = Field(default=1.0, ge=0, description="输出Token计费比例")
    cached_rate: float = Field(default=1.0, ge=0, description="缓存命中Token计费比例")
    max_context_tokens: int = Field(default=4096, ge=1, description="最大上下文Token数")
    max_output_tokens: int = Field(default=4096, ge=1, description="最大输出Token数")
    supports_multimodal: bool = Field(default=False, description="是否支持多模态")
    supports_tool_call: bool = Field(default=False, description="是否支持工具调用")
    supports_streaming: bool = Field(default=True, description="是否支持流式输出")
    supports_prompt_cache: bool = Field(default=False, description="是否支持Prompt缓存")
    supports_structured_output: bool = Field(default=False, description="是否支持结构化输出")
    fallback_model_pk: int | None = Field(
        default=None, description="降级模型主键(关联sys_ai_model.id)"
    )
    prompt_cache_prefix_len: int = Field(default=0, ge=0, description="Prompt缓存稳定前缀长度")
    status: int = Field(default=1, description="状态(1:启用;0:禁用)")
    vip_level: int = Field(
        default=0, ge=0, le=2, description="最低可用VIP等级(0:所有用户;1:VIP1及以上;2:VIP2及以上)"
    )


class AiModelUpdate(OrmResult):
    provider_id: int | None = Field(default=None, description="关联供应商ID")
    display_name: str | None = Field(
        default=None, min_length=1, max_length=128, description="显示名称"
    )
    input_rate: float | None = Field(default=None, ge=0, description="输入Token计费比例")
    output_rate: float | None = Field(default=None, ge=0, description="输出Token计费比例")
    cached_rate: float | None = Field(default=None, ge=0, description="缓存命中Token计费比例")
    max_context_tokens: int | None = Field(default=None, ge=1, description="最大上下文Token数")
    max_output_tokens: int | None = Field(default=None, ge=1, description="最大输出Token数")
    supports_multimodal: bool | None = Field(default=None, description="是否支持多模态")
    supports_tool_call: bool | None = Field(default=None, description="是否支持工具调用")
    supports_streaming: bool | None = Field(default=None, description="是否支持流式输出")
    supports_prompt_cache: bool | None = Field(default=None, description="是否支持Prompt缓存")
    supports_structured_output: bool | None = Field(default=None, description="是否支持结构化输出")
    fallback_model_pk: int | None = Field(
        default=None, description="降级模型主键(关联sys_ai_model.id)"
    )
    prompt_cache_prefix_len: int | None = Field(
        default=None, ge=0, description="Prompt缓存稳定前缀长度"
    )
    status: int | None = Field(default=None, description="状态(1:启用;0:禁用)")
    vip_level: int | None = Field(default=None, ge=0, le=2, description="最低可用VIP等级")


class AiModelResult(OrmResult):
    id: int = Field(description="主键")
    provider_id: int = Field(description="关联供应商ID")
    model_id: str = Field(description="模型标识")
    display_name: str = Field(description="显示名称")
    input_rate: float = Field(description="输入Token计费比例")
    output_rate: float = Field(description="输出Token计费比例")
    cached_rate: float = Field(description="缓存命中Token计费比例")
    max_context_tokens: int = Field(description="最大上下文Token数")
    max_output_tokens: int = Field(description="最大输出Token数")
    supports_multimodal: int = Field(description="是否支持多模态")
    supports_tool_call: int = Field(description="是否支持工具调用")
    supports_streaming: int = Field(description="是否支持流式输出")
    supports_prompt_cache: int = Field(description="是否支持Prompt缓存")
    supports_structured_output: int = Field(description="是否支持结构化输出")
    fallback_model_pk: int | None = Field(
        default=None, description="降级模型主键(关联sys_ai_model.id)"
    )
    prompt_cache_prefix_len: int = Field(description="Prompt缓存稳定前缀长度")
    status: int = Field(description="状态(1:启用;0:禁用)")
    vip_level: int = Field(description="最低可用VIP等级(0:所有用户;1:VIP1及以上;2:VIP2及以上)")
    speed_tier: str | None = Field(
        default=None, description="速度档位(fast:快;medium:中;slow:慢;unknown:未知)"
    )
    is_fallback_target: bool | None = Field(
        default=None, description="是否作为其他启用模型的降级目标"
    )
    create_time: datetime | None = Field(default=None, description="创建时间")


class AiModelPageQuery(BasePageQuery):
    keyword: str | None = Field(default=None, description="关键字(按显示名称/模型标识模糊搜索)")


# ── 会话管理 ──────────────────────────────────────────


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255, description="会话标题")
    model: str | None = Field(default=None, max_length=64, description="会话使用的模型标识")
    systemPrompt: str | None = Field(default=None, description="系统提示词")
    modelConfig: dict[str, Any] | None = Field(default=None, description="模型参数配置")
    apiKeyId: int | None = Field(default=None, description="绑定的API Key ID")
    agentCode: str | None = Field(
        default=None, max_length=64, description="会话使用的Agent编码(为空使用默认Agent)"
    )
    scene: str | None = Field(
        default=None,
        max_length=32,
        description="会话场景(general:通用对话;image_dispatch:图像处理调度;multi_step:多步推理;algorithm_recommend:算法推荐;scheduled_task:定时任务;为空默认general)",
    )


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255, description="会话标题")
    model: str | None = Field(default=None, max_length=64, description="会话使用的模型标识")
    systemPrompt: str | None = Field(default=None, description="系统提示词")
    modelConfig: dict[str, Any] | None = Field(default=None, description="模型参数配置")
    pinned: int | None = Field(default=None, description="是否置顶(0:否;1:是)")
    status: int | None = Field(default=None, description="会话状态(1:活跃;2:已归档)")
    agentCode: str | None = Field(
        default=None, max_length=64, description="切换Agent编码(下一条消息生效)"
    )


class ConversationResult(OrmResult):
    id: int = Field(description="主键")
    title: str = Field(description="会话标题")
    model: str | None = Field(default=None, description="会话使用的模型标识")
    agent_code: str | None = Field(default=None, description="会话使用的Agent编码")
    agent_version: int | None = Field(default=None, description="会话锚定的Agent已发布版本号")
    summary: str | None = Field(default=None, description="会话摘要")
    system_prompt: str | None = Field(default=None, description="系统提示词")
    model_config_: dict[str, Any] | None = Field(
        default=None,
        validation_alias="model_config",
        serialization_alias="modelConfig",
        description="模型参数配置",
    )
    api_key_id: int | None = Field(default=None, description="绑定的API Key ID")
    message_count: int = Field(description="消息数")
    last_message_at: datetime | None = Field(default=None, description="最后消息时间")
    current_branch_message_id: int | None = Field(
        default=None, description="当前激活的分支末端消息ID"
    )
    last_read_message_id: int | None = Field(default=None, description="最后已读消息ID")
    pinned: int = Field(description="是否置顶")
    pinned_at: datetime | None = Field(default=None, description="置顶时间")
    delete_time: datetime | None = Field(default=None, description="软删时间")
    unread_count: int = Field(default=0, description="未读消息数(最后消息ID与已读ID差值)")
    title_source: str = Field(description="标题来源")
    status: int = Field(description="会话状态")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class ConversationPageQuery(BasePageQuery):
    keyword: str | None = Field(default=None, description="关键字(按标题模糊搜索)")
    status: int | None = Field(
        default=None, description="会话状态范围过滤(0:全部;1:活跃,默认;2:已归档)"
    )


class ConversationBatchAction(BaseModel):
    action: Literal["archive", "restore", "delete"] = Field(..., description="批量操作类型")
    ids: list[int] = Field(..., min_length=1, description="会话ID列表")
    confirm: bool = Field(default=False, description="批量删除二次确认标记(仅delete需要)")


class ConversationExportQuery(BaseModel):
    format: Literal["markdown", "json"] = Field(default="markdown", description="导出格式")


# ── 消息管理 ──────────────────────────────────────────


class MessageSend(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000, description="消息内容")
    model: str | None = Field(default=None, max_length=64, description="使用的模型标识")


class MessageEdit(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000, description="编辑后的消息内容")


class MessageResume(BaseModel):
    confirm: bool | None = Field(
        default=None, description="是否确认（confirm 中断必填：True 接受推荐；False 拒绝）"
    )
    params: dict[str, Any] | None = Field(
        default=None, description="确认参数（如 algorithmId 表示选择了备选算法）"
    )
    plan_edit: dict[str, Any] | None = Field(
        default=None,
        description="Plan-and-Execute 计划干预（仅计划待执行时允许）：{remove: [taskId], "
        "reorder: [taskId...], add: {description, depends_on}}",
    )


class MessageResult(OrmResult):
    id: int = Field(description="主键")
    conversation_id: int = Field(description="会话ID")
    parent_message_id: int | None = Field(default=None, description="父消息ID")
    role: str = Field(description="消息角色")
    content: str | None = Field(default=None, description="消息内容")
    tool_calls: Any | None = Field(default=None, description="工具调用列表")
    tool_call_id: str | None = Field(default=None, description="工具调用结果关联ID")
    model: str | None = Field(default=None, description="本条消息使用的模型标识")
    status: int = Field(description="消息状态")
    error: str | None = Field(default=None, description="错误信息")
    metadata_: Any | None = Field(
        default=None,
        validation_alias="metadata",
        serialization_alias="metadata",
        description="元数据",
    )
    input_tokens: int = Field(description="输入Token数")
    output_tokens: int = Field(description="输出Token数")
    cached_input_tokens: int = Field(description="缓存命中的输入Token数")
    credits: int = Field(description="消耗积分数")
    task_id: str | None = Field(default=None, description="关联异步任务ID")
    edited: int = Field(description="是否已编辑")
    original_content: str | None = Field(default=None, description="编辑前原文")
    create_time: datetime | None = Field(default=None, description="创建时间")


class MessagePageQuery(BasePageQuery):
    pass


# ── 推理步骤 ──────────────────────────────────────────


class AgentThoughtResult(OrmResult):
    id: int = Field(description="主键")
    message_id: int = Field(description="关联消息ID")
    conversation_id: int = Field(description="会话ID")
    position: int = Field(description="步骤序号")
    thought: str | None = Field(default=None, description="LLM思考内容")
    tool: str | None = Field(default=None, description="工具名称")
    tool_input: Any | None = Field(default=None, description="工具输入参数")
    observation: str | None = Field(default=None, description="工具返回摘要")
    status: int = Field(description="步骤状态")
    latency_ms: int = Field(description="工具调用耗时(毫秒)")
    error: str | None = Field(default=None, description="失败原因")
    create_time: datetime | None = Field(default=None, description="创建时间")
