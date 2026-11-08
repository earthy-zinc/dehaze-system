# GENERATED FROM config/schema/ai-conversation.schema.json — DO NOT EDIT.
# 由 scripts/tooling/gen-ai-contract.mjs 生成；改契约请改 Schema 后重新运行生成器。
# 字段名保持 camelCase，与 JSON 一致，不做 snake_case 转换。

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Role = Literal["system", "user", "assistant", "tool"]


MessageStatus = Literal["running", "completed", "failed", "canceled", "awaiting_input"]


BlockKind = Literal["text", "thinking", "tool_use"]


ToolStatus = Literal["running", "success", "failed", "skipped"]


PlanStatus = Literal["pending", "running", "completed", "failed"]


InterruptType = Literal["confirm", "quota", "async_wait", "plan_approve"]


ConfirmKind = Literal["algorithm_recommend", "tool_permission", "dangerous_op", "write_conflict"]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    callId: str
    name: str
    input: dict[str, Any] | None = None
    output: str | None = None
    summary: str | None = None
    status: ToolStatus
    error: str | None = None
    latencyMs: int | None = None
    agentCode: str | None = None
    subAgent: bool


class Block(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: BlockKind
    seq: int
    text: str | None = None
    durationMs: int | None = None
    tool: ToolCall | None = None


class PlanTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str | None = None
    dependsOn: list[str] | None = None
    status: PlanStatus | None = None
    paradigm: str | None = None


class PlanRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revisionNo: int
    reason: str
    actor: str | None = None
    at: str | None = None


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str | None = None
    status: PlanStatus
    tasks: list[PlanTask]
    revisions: list[PlanRevision]
    awaitingApproval: bool


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="allow")

    modelId: str
    modelName: str
    reason: str | None = None


class Alternative(BaseModel):
    model_config = ConfigDict(extra="allow")

    modelId: str
    modelName: str
    matchScore: float | None = None
    reason: str | None = None


class Interrupt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: InterruptType
    confirmKind: ConfirmKind | None = None
    action: str | None = None
    recommendation: Recommendation | None = None
    alternatives: list[Alternative] | None = None
    upgradeTip: str | None = None
    usedDaily: int | None = None
    dailyLimit: int | None = None
    usedMonthly: int | None = None
    monthlyLimit: int | None = None
    estDuration: str | None = None
    imageCount: int | None = None
    plan: list[PlanTask] | None = None
    reason: str | None = None
    detail: str | None = None
    createdAt: str
    expiresAt: str
    actionable: bool


class SubAgentUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agentCode: str
    inputTokens: int
    outputTokens: int
    cachedInputTokens: int
    credits: int


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputTokens: int
    outputTokens: int
    cachedInputTokens: int
    credits: int
    subAgents: list[SubAgentUsage] | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    conversationId: int
    parentMessageId: int | None = None
    role: Role
    status: MessageStatus
    model: str | None = None
    error: str | None = None
    blocks: list[Block]
    plan: Plan | None = None
    pendingInterrupt: Interrupt | None = None
    usage: Usage
    artifactIds: list[int]
    memoryIds: list[int]
    suggestions: list[str]
    edited: bool
    siblingIndex: int | None = None
    siblingCount: int | None = None
    createTime: str
    updateTime: str | None = None


class MessageList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    list: list[Message]
    total: int
    hasMore: bool


class MessageSend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    model: str | None = None


class MessageResume(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: bool | None = None
    params: dict[str, Any] | None = None
    planEdit: dict[str, Any] | None = None


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    conversationId: int
    messageId: int
    type: str
    refType: str | None = None
    refId: int | None = None
    summary: dict[str, Any] | None = None
    isInvalid: bool
    createTime: str | None = None


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    messageId: int
    userId: int
    conversationId: int | None = None
    rating: Literal[1, -1]
    tags: list[str] | None = None
    comment: str | None = None
    createTime: str | None = None
    updateTime: str | None = None


class MessageStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    streamSessionId: str
    userMessage: Message
    assistantMessage: Message


class BlockStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messageId: int
    block: Block


class BlockDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messageId: int
    blockId: str
    delta: str


class BlockStopEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messageId: int
    block: Block


class PlanUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messageId: int
    plan: Plan


class InterruptEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messageId: int
    interrupt: Interrupt


class UsageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messageId: int
    usage: Usage


class SuggestionsEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messageId: int
    suggestions: list[str]


class MessageEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messageId: int
    status: MessageStatus
    error: str | None = None
    usage: Usage | None = None


class ErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    messageId: int | None = None
