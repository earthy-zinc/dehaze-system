"""智能体管理 Schema 模型

字段设计对齐后端实现文档 §10.1（Agent 级推理参数）与 §10.3（护栏配置），
config / guardrails 用 Pydantic 模型表达，避免自由 dict 导致的拼写漂移。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.schema.common import BasePageQuery, OrmResult

# ── 推理参数 / 护栏配置（sys_ai_agent.config JSON）────────────────


class GuardrailRule(BaseModel):
    """护栏规则（内置规则仅暴露开关与参数，不支持动态新增规则类型）"""

    enabled: bool = Field(default=True, description="开关")

    @field_validator("enabled", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, int):
            return bool(v)
        return v


class GuardrailConfig(BaseModel):
    """护栏配置（sys_dict: ai_guardrail_defaults 系统默认 + Agent 级覆盖）"""

    prompt_injection: GuardrailRule = Field(
        default_factory=lambda: GuardrailRule(enabled=True), description="Prompt注入防护"
    )
    unauthorized_access: GuardrailRule = Field(
        default_factory=lambda: GuardrailRule(enabled=True), description="越权查询检测"
    )
    sensitive_topic: GuardrailRule = Field(
        default_factory=lambda: GuardrailRule(enabled=False), description="敏感话题过滤"
    )
    pii_mask: GuardrailRule = Field(
        default_factory=lambda: GuardrailRule(enabled=True), description="敏感信息脱敏"
    )
    fact_check: GuardrailRule = Field(
        default_factory=lambda: GuardrailRule(enabled=False), description="事实性校验"
    )
    format_check: GuardrailRule = Field(
        default_factory=lambda: GuardrailRule(enabled=False), description="格式合规校验"
    )


class AgentConfig(BaseModel):
    """Agent 推理参数配置（空值继承 sys_dict 系统默认 ai_reasoning_defaults）"""

    max_steps: int | None = Field(
        default=None, ge=1, description="最大推理步数(覆盖按范式区分的默认值)"
    )
    token_budget: int | None = Field(default=None, ge=0, description="单会话 Token 预算上限")
    max_parallel: int | None = Field(default=None, ge=1, description="并行子任务最大数")
    tool_timeout: int | None = Field(default=None, ge=1, description="单工具调用超时(秒)")
    retry_max: int | None = Field(default=None, ge=0, description="工具调用失败最大重试次数")
    reflexion_threshold: float | None = Field(
        default=None, ge=0, le=1, description="Reflexion 质量达标阈值"
    )
    temperature: float | None = Field(default=None, description="LLM 温度参数")
    guardrails: GuardrailConfig | None = Field(default=None, description="护栏规则开关与参数")


# ── Agent 主表请求/响应 ──────────────────────────────────────────


class AgentCreate(OrmResult):
    agent_code: str = Field(..., min_length=1, max_length=64, description="Agent唯一编码")
    name: str = Field(..., min_length=1, max_length=128, description="Agent显示名称")
    description: str = Field(default="", max_length=512, description="Agent描述")
    system_prompt: str | None = Field(default=None, description="系统提示词(Markdown)")
    model_id: str = Field(..., min_length=1, max_length=64, description="关联模型标识")
    reasoning_mode: str = Field(
        default="auto",
        pattern=r"^(auto|direct|react|plan_execute|reflexion)$",
        description="推理范式",
    )
    config: AgentConfig | None = Field(default=None, description="推理参数配置")
    is_subagent: bool = Field(default=False, description="是否可作为子Agent")
    is_team: bool = Field(default=False, description="是否为Team团队")
    is_exposed: bool = Field(default=False, description="是否对外暴露为A2A子Agent")
    permissions: list[dict[str, Any]] | None = Field(default=None, description="文件系统权限规则")
    sort_order: int = Field(default=0, ge=0, description="排序序号")
    status: int = Field(default=1, ge=0, le=1, description="状态(1:启用;0:禁用)")


class AgentUpdate(OrmResult):
    name: str | None = Field(
        default=None, min_length=1, max_length=128, description="Agent显示名称"
    )
    description: str | None = Field(default=None, max_length=512, description="Agent描述")
    system_prompt: str | None = Field(default=None, description="系统提示词(Markdown)")
    model_id: str | None = Field(
        default=None, min_length=1, max_length=64, description="关联模型标识"
    )
    reasoning_mode: str | None = Field(
        default=None,
        pattern=r"^(auto|direct|react|plan_execute|reflexion)$",
        description="推理范式",
    )
    config: AgentConfig | None = Field(default=None, description="推理参数配置")
    is_subagent: bool | None = Field(default=None, description="是否可作为子Agent")
    is_team: bool | None = Field(default=None, description="是否为Team团队")
    is_exposed: bool | None = Field(default=None, description="是否对外暴露为A2A子Agent")
    permissions: list[dict[str, Any]] | None = Field(default=None, description="文件系统权限规则")
    sort_order: int | None = Field(default=None, ge=0, description="排序序号")


class AgentListItem(OrmResult):
    id: int = Field(description="主键")
    agent_code: str = Field(description="Agent唯一编码")
    name: str = Field(description="Agent显示名称")
    description: str = Field(description="Agent描述")
    model_id: str = Field(description="关联模型标识")
    reasoning_mode: str = Field(description="推理范式")
    is_subagent: int = Field(description="是否可作为子Agent")
    is_team: int = Field(description="是否为Team团队")
    is_exposed: int = Field(description="是否对外暴露")
    status: int = Field(description="状态(1:启用;0:禁用)")
    sort_order: int = Field(description="排序序号")
    create_time: datetime | None = Field(default=None, description="创建时间")


class AgentDetail(AgentListItem):
    system_prompt: str | None = Field(default=None, description="系统提示词")
    config: AgentConfig | None = Field(default=None, description="推理参数配置")
    permissions: list[dict[str, Any]] | None = Field(default=None, description="文件系统权限规则")
    skills: list[str] = Field(default_factory=list, description="关联的 Skill 名称")
    mcp_namespaces: list[str] = Field(default_factory=list, description="关联的 MCP 命名空间")
    subagents: list["SubAgentItem"] = Field(default_factory=list, description="关联的子 Agent")


class SubAgentItem(OrmResult):
    agent_id: int = Field(description="子Agent ID")
    agent_name: str = Field(description="子Agent名称")
    agent_code: str = Field(description="子Agent编码")
    description: str = Field(description="子Agent描述(触发描述)")
    endpoint_id: int | None = Field(default=None, description="外部A2A端点ID(NULL为本地)")
    priority: int = Field(description="优先级")


class AgentPageQuery(BasePageQuery):
    keyword: str | None = Field(default=None, description="关键字(按名称/编码模糊搜索)")
    status: int | None = Field(default=None, ge=0, le=1, description="状态过滤")


# ── 关联设置 ─────────────────────────────────────────────────────


class AgentSkillsForm(BaseModel):
    skills: list[str] = Field(default_factory=list, description="Skill 名称列表(覆盖式更新)")


class AgentMcpForm(BaseModel):
    mcp_namespaces: list[str] = Field(
        default_factory=list, description="MCP 命名空间列表(覆盖式更新)"
    )


class AgentSubAgentItem(BaseModel):
    agent_id: int = Field(..., description="子Agent ID(关联sys_ai_agent.id;远程A2A为本地影子记录)")
    endpoint_id: int | None = Field(default=None, description="外部A2A端点ID(NULL为本地子Agent)")
    priority: int = Field(default=0, description="优先级(数字越小越优先)")


class AgentSubAgentsForm(BaseModel):
    subagents: list[AgentSubAgentItem] = Field(
        default_factory=list, description="子Agent关联列表(覆盖式更新)"
    )


# ── 启停 / 复制 / 测试 ───────────────────────────────────────────


class AgentStatusForm(BaseModel):
    status: int = Field(..., ge=0, le=1, description="目标状态(1:启用;0:禁用)")


class AgentCopyForm(BaseModel):
    agent_code: str = Field(..., min_length=1, max_length=64, description="新Agent唯一编码")


class AgentTestForm(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="测试消息")
    conversation_config: AgentConfig | None = Field(default=None, description="会话级配置覆盖")


# ── 版本管理 ─────────────────────────────────────────────────────


class AgentPublishForm(BaseModel):
    change_note: str = Field(default="", max_length=512, description="变更说明")


class AgentVersionResult(OrmResult):
    id: int = Field(description="主键")
    agent_id: int = Field(description="关联Agent ID")
    version_no: int = Field(description="版本号")
    status: int = Field(description="版本状态(1:草稿;2:已发布)")
    change_note: str | None = Field(default=None, description="变更说明")
    operator_id: int | None = Field(default=None, description="操作人ID")
    create_time: datetime | None = Field(default=None, description="创建时间")


class AgentVersionDetail(AgentVersionResult):
    snapshot: dict[str, Any] = Field(default_factory=dict, description="配置快照")


# ── 评测集 / 样本 / 运行 ─────────────────────────────────────────


class EvalDatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="评测集名称")
    description: str = Field(default="", max_length=512, description="评测集描述")
    dataset_type: str = Field(..., pattern=r"^(dev|regression|heldout)$", description="评测集类型")


class EvalDatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128, description="评测集名称")
    description: str | None = Field(default=None, max_length=512, description="评测集描述")


class EvalDatasetResult(OrmResult):
    id: int = Field(description="主键")
    agent_id: int = Field(description="关联Agent ID")
    name: str = Field(description="评测集名称")
    description: str = Field(description="评测集描述")
    dataset_type: str = Field(description="评测集类型")
    create_time: datetime | None = Field(default=None, description="创建时间")


class EvalSampleCreate(BaseModel):
    dataset_id: int = Field(..., description="关联评测集ID")
    task_goal: str = Field(..., min_length=1, description="任务目标")
    allowed_input: str | None = Field(default=None, description="允许输入")
    tools: list[str] | None = Field(default=None, description="可用工具")
    expected_process: str | None = Field(default=None, description="期望过程")
    expected_result: str | None = Field(default=None, description="期望结果")
    forbidden_behavior: str | None = Field(default=None, description="禁止行为")
    risk_level: str = Field(default="low", pattern=r"^(low|medium|high)$", description="风险等级")


class EvalSampleUpdate(BaseModel):
    task_goal: str | None = Field(default=None, min_length=1, description="任务目标")
    allowed_input: str | None = Field(default=None, description="允许输入")
    tools: list[str] | None = Field(default=None, description="可用工具")
    expected_process: str | None = Field(default=None, description="期望过程")
    expected_result: str | None = Field(default=None, description="期望结果")
    forbidden_behavior: str | None = Field(default=None, description="禁止行为")
    risk_level: str | None = Field(
        default=None, pattern=r"^(low|medium|high)$", description="风险等级"
    )


class EvalSampleResult(OrmResult):
    id: int = Field(description="主键")
    dataset_id: int = Field(description="关联评测集ID")
    task_goal: str = Field(description="任务目标")
    allowed_input: str | None = Field(default=None, description="允许输入")
    tools: list[str] | None = Field(default=None, description="可用工具")
    expected_process: str | None = Field(default=None, description="期望过程")
    expected_result: str | None = Field(default=None, description="期望结果")
    forbidden_behavior: str | None = Field(default=None, description="禁止行为")
    risk_level: str = Field(description="风险等级")
    create_time: datetime | None = Field(default=None, description="创建时间")


class EvalRunResult(OrmResult):
    id: int = Field(description="主键")
    agent_id: int = Field(description="关联Agent ID")
    dataset_id: int = Field(description="关联评测集ID")
    trigger_type: str = Field(description="触发方式(manual/publish)")
    status: int = Field(description="执行状态(1:执行中;2:通过;3:失败)")
    score_summary: dict[str, Any] | None = Field(default=None, description="四维评分聚合")
    results: list[dict[str, Any]] | None = Field(default=None, description="样本明细")
    create_by: int | None = Field(default=None, description="创建人ID")
    create_time: datetime | None = Field(default=None, description="创建时间")


class EvalRunCreate(BaseModel):
    dataset_id: int = Field(..., description="关联评测集ID")
    trigger_type: str = Field(
        default="manual", pattern=r"^(manual|publish)$", description="触发方式"
    )


# ── 外部端点 ─────────────────────────────────────────────────────


class EndpointCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="端点名称")
    agent_card_url: str | None = Field(default=None, max_length=512, description="Agent Card地址")
    base_url: str = Field(..., min_length=1, max_length=512, description="A2A端点地址")
    auth_type: str = Field(
        default="http",
        pattern=r"^(apiKey|http|oauth2|openIdConnect|mutualTLS)$",
        description="认证方式",
    )
    credential: str | None = Field(
        default=None, max_length=512, description="凭证密文(AES加密后base64)"
    )
    status: int = Field(default=1, ge=0, le=1, description="状态(1:启用;0:禁用)")


class EndpointUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128, description="端点名称")
    agent_card_url: str | None = Field(default=None, max_length=512, description="Agent Card地址")
    auth_type: str | None = Field(
        default=None,
        pattern=r"^(apiKey|http|oauth2|openIdConnect|mutualTLS)$",
        description="认证方式",
    )
    credential: str | None = Field(default=None, max_length=512, description="凭证密文")
    status: int | None = Field(default=None, ge=0, le=1, description="状态")


class EndpointResult(OrmResult):
    id: int = Field(description="主键")
    name: str = Field(description="端点名称")
    agent_card_url: str | None = Field(default=None, description="Agent Card地址")
    base_url: str = Field(description="A2A端点地址")
    auth_type: str = Field(description="认证方式")
    agent_card: dict[str, Any] | None = Field(default=None, description="缓存的Agent Card")
    status: int = Field(description="状态")
    create_time: datetime | None = Field(default=None, description="创建时间")


class EndpointPageQuery(BasePageQuery):
    keyword: str | None = Field(default=None, description="关键字(按名称/地址模糊搜索)")
    status: int | None = Field(default=None, ge=0, le=1, description="状态过滤")
