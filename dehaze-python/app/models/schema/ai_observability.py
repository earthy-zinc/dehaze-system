"""AI 可观测性查询 Schema 模型（F-M08-013 后端实现 §2.6）

过程链检索/详情、异常总览、资源消耗聚合、性能趋势的请求与响应模型。
响应模型字段与 ORM 实体一致（snake_case），经 OrmResult 输出 camelCase；
GET 查询参数模型直接以 camelCase 命名（与 pageNum/pageSize 约定一致）。
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.schema.ai_conversation import AgentThoughtResult
from app.models.schema.common import BasePageQuery, OrmResult


class TracePageQuery(BasePageQuery):
    conversationId: int | None = Field(default=None, description="会话ID筛选")
    userId: int | None = Field(default=None, description="用户ID筛选(经会话归属关联)")
    status: int | None = Field(
        default=None, ge=1, le=4, description="执行状态(1:成功;2:失败;3:中断;4:超时)"
    )
    agentCode: str | None = Field(default=None, max_length=64, description="智能体编码筛选")
    model: str | None = Field(default=None, max_length=64, description="模型标识筛选")
    errorType: str | None = Field(default=None, max_length=32, description="失败类型精确筛选")
    keyword: str | None = Field(
        default=None, max_length=64, description="关键词筛选(匹配trace_id或会话标题模糊)"
    )
    capability: Literal["memory", "kb", "tools"] | None = Field(
        default=None, description="能力维度筛选(上下文构成含memory/kb/tools的构成项)"
    )
    startTime: datetime | None = Field(default=None, description="开始时间(含)")
    endTime: datetime | None = Field(default=None, description="结束时间(含)")


class TraceItem(OrmResult):
    trace_id: str = Field(description="过程链ID(复用日志链路trace_id)")
    conversation_id: int = Field(description="所属会话ID")
    message_id: int | None = Field(default=None, description="关联助手回复消息ID")
    agent_code: str | None = Field(default=None, description="执行智能体编码")
    trace_type: str = Field(
        default="conversation",
        description="过程链类型(conversation主对话;summary摘要压缩;memory_extraction记忆提取;suggestion建议推荐;step_summary步骤摘要)",
    )
    model: str | None = Field(default=None, description="实际使用模型标识")
    status: int = Field(description="执行状态(1:成功;2:失败;3:中断;4:超时)")
    error_type: str | None = Field(default=None, description="失败类型")
    duration_ms: int = Field(description="整条回复总耗时(毫秒)")
    first_token_ms: int | None = Field(default=None, description="首Token延迟(毫秒)")
    llm_call_count: int = Field(description="本次回复的LLM调用次数")
    total_tokens: int = Field(description="总Token消耗(与计费口径一致)")
    prompt_tokens: int = Field(description="输入Token消耗")
    completion_tokens: int = Field(description="输出Token消耗")
    cached_tokens: int = Field(description="缓存命中Token数")
    step_count: int = Field(description="推理步数")
    create_time: datetime | None = Field(default=None, description="记录时间")


class LlmCallItem(OrmResult):
    seq: int = Field(description="调用序号(1起递增)")
    step_position: int | None = Field(default=None, description="关联推理步骤序号")
    model: str | None = Field(default=None, description="本次调用模型")
    status: int = Field(description="调用状态(1:成功;2:失败;3:超时)")
    error_type: str | None = Field(default=None, description="失败类型")
    duration_ms: int = Field(description="本次调用总耗时(毫秒)")
    first_token_ms: int | None = Field(default=None, description="本次调用首Token延迟(毫秒)")
    prompt_tokens: int = Field(description="输入Token消耗")
    completion_tokens: int = Field(description="输出Token消耗")
    cached_tokens: int = Field(description="缓存命中Token数")
    tool_call: Any | None = Field(default=None, description="工具调用信息JSON")
    input_snapshot: Any | None = Field(default=None, description="本次调用输入构成JSON")
    output_snapshot: Any | None = Field(default=None, description="本次调用输出摘要JSON")
    create_time: datetime | None = Field(default=None, description="记录时间")


class TraceBillingItem(OrmResult):
    bill_type: str | None = Field(default=None, description="计费类型(chat;tool_llm;kb_inject;asr;tts)")
    model: str | None = Field(default=None, description="实际使用模型标识")
    actual_model: str | None = Field(default=None, description="用户原选模型标识(NULL表示未降级)")
    provider_id: int | None = Field(default=None, description="实际供应商ID")
    input_tokens: int = Field(default=0, description="输入Token数(含缓存命中部分)")
    output_tokens: int = Field(default=0, description="输出Token数")
    cached_input_tokens: int = Field(default=0, description="其中缓存命中的输入Token数")
    credits: int = Field(default=0, description="消耗积分数")
    credits_saved: int = Field(default=0, description="缓存命中节省积分数")
    error_code: str | None = Field(default=None, description="调用失败错误码(成功为NULL)")
    latency_ms: int | None = Field(default=None, description="调用耗时(毫秒)")
    request_id: str | None = Field(default=None, description="请求唯一ID(与trace_id一致)")
    create_time: datetime | None = Field(default=None, description="记录时间")


class TraceArtifactItem(OrmResult):
    id: int | None = Field(default=None, description="主键")
    type: str | None = Field(default=None, description="产物类型")
    summary: Any | None = Field(default=None, description="业务摘要元数据")
    ref_type: str | None = Field(default=None, description="引用业务表")
    ref_id: int | None = Field(default=None, description="引用业务表ID")
    create_time: datetime | None = Field(default=None, description="创建时间")


class TraceMessageItem(OrmResult):
    id: int = Field(description="主键")
    conversation_id: int = Field(description="会话ID")
    parent_message_id: int | None = Field(default=None, description="父消息ID")
    role: str = Field(description="消息角色(system;user;assistant;tool)")
    content: str | None = Field(default=None, description="消息内容")
    status: int = Field(description="消息状态(1:流式输出中;2:已完成;3:失败;4:已取消)")
    model: str | None = Field(default=None, description="本条消息使用的模型标识")
    input_tokens: int = Field(description="输入Token数")
    output_tokens: int = Field(description="输出Token数")
    create_time: datetime | None = Field(default=None, description="创建时间")


class TraceDetailResult(TraceItem):
    context_snapshot: Any | None = Field(
        default=None, description="上下文构成快照JSON(系统提示/历史/记忆/检索及压缩事件)"
    )
    llm_calls: list[LlmCallItem] = Field(
        default_factory=list, description="LLM调用明细(按seq正序回放)"
    )
    thoughts: list[AgentThoughtResult] = Field(
        default_factory=list,
        description="关联消息的推理步骤(按position正序,trace未关联消息时为空)",
    )
    messages: list[TraceMessageItem] = Field(
        default_factory=list, description="所属会话完整消息列表(按create_time/id正序)"
    )
    billing: list[TraceBillingItem] = Field(
        default_factory=list, description="关联计费记录(优先request_id=trace_id关联,回退message_id)"
    )
    artifacts: list[TraceArtifactItem] = Field(
        default_factory=list, description="关联中间产物(按message_id关联)"
    )


class SummaryResult(OrmResult):
    total: int = Field(description="过程链总数")
    success_count: int = Field(description="成功数")
    failed_count: int = Field(description="失败数")
    interrupted_count: int = Field(description="中断数")
    timeout_count: int = Field(description="超时数")
    quota_rejected: int = Field(description="配额拒绝数(按采集链路写入的拒绝类error_type统计)")
    high_risk_calls: int = Field(
        description="高风险调用数(推理步数超阈值或存在失败的工具调用)"
    )


class CostsQuery(BasePageQuery):
    dimension: Literal["model", "agent", "user"] = Field(
        default="model", description="聚合维度"
    )
    startTime: datetime | None = Field(default=None, description="开始时间(含)")
    endTime: datetime | None = Field(default=None, description="结束时间(含)")


class CostItem(OrmResult):
    model: str | None = Field(default=None, description="模型标识(model维度)")
    agent_code: str | None = Field(default=None, description="智能体编码(agent维度)")
    user_id: int | None = Field(default=None, description="用户ID(user维度)")
    trace_count: int = Field(description="过程链数")
    total_tokens: int = Field(description="总Token消耗(与计费口径一致)")
    prompt_tokens: int = Field(description="输入Token消耗")
    completion_tokens: int = Field(description="输出Token消耗")
    cached_tokens: int = Field(description="缓存命中Token数")


class CostTrendItem(OrmResult):
    date: str = Field(description="日期(YYYY-MM-DD)")
    trace_count: int = Field(description="过程链数")
    total_tokens: int = Field(description="总Token消耗")
    prompt_tokens: int = Field(description="输入Token消耗")
    completion_tokens: int = Field(description="输出Token消耗")
    cached_tokens: int = Field(description="缓存命中Token数")


class CostsResult(OrmResult):
    items: list[CostItem] = Field(description="按维度聚合结果(分页)")
    total: int = Field(description="聚合分组总数")
    trend: list[CostTrendItem] = Field(description="按日Token消耗趋势")


class TrendsQuery(BaseModel):
    dimension: Literal["model", "agent"] = Field(default="model", description="聚合维度")
    startTime: datetime | None = Field(default=None, description="开始时间(含)")
    endTime: datetime | None = Field(default=None, description="结束时间(含)")


class TrendItem(OrmResult):
    model: str | None = Field(default=None, description="模型标识(model维度)")
    agent_code: str | None = Field(default=None, description="智能体编码(agent维度)")
    date: str = Field(description="日期(YYYY-MM-DD)")
    call_count: int = Field(description="过程链数")
    success_count: int = Field(description="成功数")
    success_rate: float = Field(description="成功率(百分比0-100)")
    avg_first_token_ms: float | None = Field(
        default=None, description="平均首Token延迟(毫秒,成功调用口径)"
    )
    avg_duration_ms: float | None = Field(default=None, description="平均总耗时(毫秒)")
