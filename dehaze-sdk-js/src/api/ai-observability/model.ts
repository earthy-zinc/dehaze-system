import { PageQuery } from "@/types";

// ==================== 枚举类型 ====================

/** 过程链执行状态：1-成功，2-失败，3-中断，4-超时 */
export type AiObservabilityStatus = 1 | 2 | 3 | 4;

/** LLM 调用状态：1-成功，2-失败，3-超时 */
export type AiObservabilityCallStatus = 1 | 2 | 3;

/** 资源消耗聚合维度 */
export type AiObservabilityCostDimension = "model" | "agent" | "user";

/** 性能趋势聚合维度 */
export type AiObservabilityTrendDimension = "model" | "agent";

// ==================== 过程链检索 ====================

/** 过程链检索参数（导出接口复用同一套筛选条件） */
export interface AiObservabilityTraceQuery extends PageQuery {
  conversationId?: number;
  /** 用户 ID 筛选（经会话归属关联） */
  userId?: number;
  status?: AiObservabilityStatus;
  agentCode?: string;
  model?: string;
  /** 失败类型精确筛选 */
  errorType?: string;
  /** 关键词筛选（匹配 trace_id 或会话标题模糊） */
  keyword?: string;
  /** 能力维度筛选（上下文构成含 memory/kb/tools 构成项） */
  capability?: "memory" | "kb" | "tools";
  startTime?: string;
  endTime?: string;
}

/** 过程链列表项 */
export interface AiObservabilityTraceItem {
  traceId: string;
  conversationId: number;
  /** 所属会话标题；无标题时字段缺失（后端 NonNullJSONResponse 递归剔除 null） */
  conversationTitle?: string;
  messageId?: number;
  agentCode?: string;
  model?: string;
  status: AiObservabilityStatus;
  errorType?: string;
  /** 过程链类型：conversation 主对话 / summary 摘要压缩 / memory_extraction 记忆提取 / suggestion 建议推荐 / step_summary 步骤摘要 */
  traceType?: string;
  /** 整条回复总耗时（毫秒） */
  durationMs: number;
  firstTokenMs?: number;
  llmCallCount: number;
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
  stepCount: number;
  createTime?: string;
}

// ==================== 上下文快照 ====================

/** 上下文构成项（快照内 JSON 键保持后端写入原样，不做 camelCase 转换） */
export interface AiObservabilityContextItem {
  type: "system" | "summary" | "history" | "memory" | "retrieval" | "tools" | string;
  tokens: number;
  /** 系统提示正文（仅 type=system 时非空） */
  content?: string;
  count?: number;
  counts?: { user?: number; assistant?: number; tool?: number };
  source?: "raw" | "summarized";
  /** 注入记忆原文清单（仅 type=memory 时非空，键保持后端写入原样） */
  items?: Array<{
    memory_id?: number;
    memory_type?: string;
    source?: string;
    content?: string;
  }>;
}

/** 上下文压缩/截断/推理期事件（summarize/truncate/guardrail/plan/resume） */
export interface AiObservabilityContextEvent {
  event: "summarize" | "truncate" | "guardrail" | "plan" | "resume" | string;
  tokens?: number;
  before_tokens?: number;
  after_tokens?: number;
  /** 护栏命中规则（prompt_injection/sensitive_topic/unauthorized_access/pii_mask） */
  rule?: string;
  /** 护栏命中详情 */
  detail?: string;
  /** 计划阶段 */
  phase?: string;
  /** 计划快照摘要 */
  plan_summary?: string;
  /** 中断类型（resume 事件） */
  interrupt_type?: string;
  /** 用户决策摘要（confirm 的 confirmed/algorithmId、plan_approve 的 plan_edit 等） */
  decision?: string;
  /** 原中断链路 trace_id（resume 事件关联） */
  from_trace_id?: string;
}

/** 上下文构成快照：AI 当次回复"看到了什么" */
export interface AiObservabilityContextSnapshot {
  items?: AiObservabilityContextItem[];
  events?: AiObservabilityContextEvent[];
}

// ==================== LLM 调用明细 ====================

/** 每次 LLM 调用的输入构成快照 */
export interface AiObservabilityInputSnapshot {
  messages: {
    counts: { user?: number; assistant?: number; tool?: number; system?: number };
    tokens: number;
    /** 本轮实际发给模型的每条消息原文（键保持后端写入原样 snake_case） */
    items?: Array<{ role?: string; content?: string }>;
  };
  system_tokens?: number;
  /** 该轮调用的 system 提示正文（JSON 键保持后端写入原样 snake_case） */
  system_content?: string;
  tool_count?: number;
  /** 工具定义清单（含 MCP/业务工具，键保持后端写入原样 snake_case） */
  tools?: Array<{ name?: string; description?: string }>;
  user_id?: number;
}

/** 每次 LLM 调用的输出摘要（正文截断，不含完整输出） */
export interface AiObservabilityOutputSnapshot {
  text: string;
  tool_calls?: Array<{ name: string; arguments: string }> | null;
}

/** 工具调用信息 */
export interface AiObservabilityToolCall {
  has_tool_call: boolean;
  tools?: Array<{ name: string; arguments: string }>;
}

/** LLM 调用明细（span 级，按 seq 正序回放） */
export interface AiObservabilityLlmCall {
  /** 调用序号（1 起递增） */
  seq: number;
  /** 关联推理步骤序号 */
  stepPosition?: number;
  model?: string;
  status: AiObservabilityCallStatus;
  errorType?: string;
  durationMs: number;
  firstTokenMs?: number;
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
  /** 调用发起时刻（时间线交织排序锚点） */
  startTime?: string;
  toolCall?: AiObservabilityToolCall | null;
  inputSnapshot?: AiObservabilityInputSnapshot | null;
  outputSnapshot?: AiObservabilityOutputSnapshot | null;
  /** wire 原始请求体（审计原文；未采集为 null） */
  rawRequest?: Record<string, unknown> | null;
  /** wire 原始响应（流式聚合为等价非流式结构；未采集为 null） */
  rawResponse?: Record<string, unknown> | null;
  /** 物理调用尝试明细（逐 Key/逐路由，快照 JSON 原样透传，键保持 snake_case） */
  attempts?: Array<{
    provider_id?: number;
    key_id?: number;
    model?: string;
    status: number;
    error_code?: string | null;
    latency_ms?: number;
  }> | null;
  createTime?: string;
}

/** 过程链详情内的推理步骤（结构对齐 src/api/ai-conversation 的 AiMessageThought） */
export interface AiObservabilityThought {
  id: number;
  messageId: number;
  conversationId: number;
  /** 步骤序号 */
  position: number;
  thought?: string;
  /** 工具名称 */
  tool?: string;
  toolInput?: unknown;
  /** 工具返回摘要 */
  observation?: string;
  /** 步骤状态：1-成功，2-失败，3-跳过 */
  status: number;
  /** 工具调用耗时（毫秒） */
  latencyMs: number;
  /** 失败原因（status=2 时填充） */
  error?: string;
  /** 来源 Agent 编码（空=主 Agent） */
  agentCode?: string;
  /** 是否子 Agent：0-主 Agent，1-子 Agent */
  isSubagent?: number;
  createTime?: string;
}

/** 过程链详情内的会话完整消息 */
export interface AiObservabilityTraceMessage {
  id: number;
  conversationId: number;
  parentMessageId?: number;
  role: "user" | "assistant" | "system" | "tool";
  content?: string;
  status: number;
  model?: string;
  inputTokens?: number;
  outputTokens?: number;
  createTime?: string;
}

/** 过程链详情：trace 汇总 + 上下文快照 + LLM 调用回放 + 推理步骤 + 会话消息 */
export interface AiObservabilityTraceDetail extends AiObservabilityTraceItem {
  contextSnapshot?: AiObservabilityContextSnapshot | null;
  llmCalls: AiObservabilityLlmCall[];
  /** 推理步骤（按 position 正序） */
  thoughts?: AiObservabilityThought[];
  /** 会话完整消息 */
  messages?: AiObservabilityTraceMessage[];
  /** 异常详情（失败/中断时填充） */
  errorDetail?: { message?: string; stack?: string } | null;
  /** 计费明细（关联 trace/消息） */
  billing?: AiObservabilityTraceBilling[];
  /** 中间产物 */
  artifacts?: AiObservabilityTraceArtifact[];
}

/** 过程链计费明细 */
export interface AiObservabilityTraceBilling {
  /** 计费类型 */
  billType?: string;
  model?: string;
  /** 实际路由模型（与请求模型不一致时填充） */
  actualModel?: string;
  providerId?: number;
  /** 输入 Token 数（含缓存命中；对齐后端实际行为：TraceBillingItem.input_tokens 默认 0，恒有值） */
  inputTokens: number;
  /** 输出 Token 数（同上，恒有值） */
  outputTokens: number;
  /** 其中缓存命中的输入 Token 数（同上，恒有值） */
  cachedInputTokens: number;
  credits?: number;
  creditsSaved?: number;
  errorCode?: string;
  latencyMs?: number;
  requestId?: string;
  createTime?: string;
}

/** 过程链中间产物 */
export interface AiObservabilityTraceArtifact {
  id: number;
  type?: string;
  summary?: string;
  refType?: string;
  refId?: number;
  createTime?: string;
}

// ==================== 会话审计时间线 ====================

/** 时间线查询参数（对齐后端实际行为：app/models/schema/ai_observability.py TimelineQuery 仅 include） */
export interface AiObservabilityTimelineQuery {
  /** 传 raw 返回 wire 原始报文；缺省仅返回摘要 */
  include?: "raw";
}

/** 时间线会话概要 */
export interface AiObservabilityTimelineConversation {
  id: number;
  title?: string;
  userId?: number;
  agentCode?: string;
  createTime?: string;
}

/** 时间线轮次边界消息（用户输入/助手输出） */
export interface AiObservabilityTimelineMessage {
  id: number;
  /** 消息角色（对齐后端实际行为：TimelineMessage.role 为必填 str） */
  role: string;
  content?: string;
  status?: number;
  model?: string;
  inputTokens?: number;
  outputTokens?: number;
  createTime?: string;
}

/** 时间线事件类型：用户输入/上下文组装/LLM 调用/工具执行/系统事件/计费 */
export type AiObservabilityEventKind =
  "input" | "context" | "llm_call" | "tool_exec" | "system_event" | "billing";

/** 时间线事件（轮内按 ts 交织排序；各 kind 携带各自载荷，键与后端 JSON 输出一致） */
export interface AiObservabilityTimelineEvent {
  kind: AiObservabilityEventKind;
  /** 事件时刻（llm_call 为 start_time 毫秒精度 ISO）；null 表示无原始时序，前端在轮内沉底展示 */
  ts?: string | null;
  /** input：用户消息全文 */
  message?: AiObservabilityTimelineMessage;
  /** context：上下文构成快照（构成项 + 压缩/护栏/计划/中断事件） */
  snapshot?: AiObservabilityContextSnapshot | null;
  /** llm_call：调用序号 */
  seq?: number;
  model?: string;
  status?: AiObservabilityCallStatus;
  errorType?: string;
  durationMs?: number;
  firstTokenMs?: number;
  promptTokens?: number;
  completionTokens?: number;
  cachedTokens?: number;
  toolCall?: AiObservabilityToolCall | null;
  /** llm_call：wire 原始报文（仅 include=raw 且采集存在时非空，否则 null，前端空态"无原始报文记录"） */
  rawRequest?: Record<string, unknown> | null;
  rawResponse?: Record<string, unknown> | null;
  /** llm_call：恒为纯计数摘要（单形状）。inputSnapshot 仅含 messages.counts/tokens 与可选 system_tokens/tool_count/user_id；outputSnapshot 含 text/tool_calls */
  summary?: {
    inputSnapshot?: {
      messages: {
        counts: { user?: number; assistant?: number; tool?: number; system?: number };
        tokens: number;
      };
      system_tokens?: number;
      tool_count?: number;
      user_id?: number;
    } | null;
    outputSnapshot?: AiObservabilityOutputSnapshot | null;
  } | null;
  attempts?: AiObservabilityLlmCall["attempts"];
  /** tool_exec：步骤序号 */
  position?: number;
  tool?: string;
  thought?: string;
  toolInput?: unknown;
  observation?: string;
  latencyMs?: number;
  /** 工具归属：子 Agent 编码（空=主 Agent） */
  agentCode?: string;
  /** 工具归属：0-主 Agent，1-子 Agent（对齐后端实际行为：app/models/schema/ai_observability.py TimelineEvent.is_subagent 为 int） */
  isSubagent?: number;
  /** system_event：护栏(guardrail)/计划(plan)/中断恢复(resume)等 */
  event?: string;
  detail?: unknown;
  /** billing：计费类型与用量（tokens 为短名 input/output/cached，见 service/ai_observability_service.py 事件组装） */
  billType?: string;
  credits?: number;
  tokens?: {
    input: number;
    output: number;
    cached: number;
  };
}

/** 一轮交互的旁路/主过程链（旁路 trace 挂触发轮次尾部） */
export interface AiObservabilityTimelineTrace {
  traceId: string;
  /** conversation 主对话 / summary 摘要压缩 / memory_extraction 记忆提取 / suggestion 建议推荐 / step_summary 步骤摘要 */
  traceType?: string;
  status: AiObservabilityStatus;
  /** 失败类型（对齐后端实际行为：TimelineTrace.error_type 可为 null） */
  errorType?: string | null;
  errorDetail?: { message?: string; stack?: string } | null;
  model?: string | null;
  /** 总耗时毫秒（后端 TimelineTrace.duration_ms 必填） */
  durationMs: number;
  createTime?: string | null;
  events: AiObservabilityTimelineEvent[];
}

/** 时间线轮次：一次用户输入 → 推理 → 助手回复（trace 详情合成的单轮可能缺边界消息） */
export interface AiObservabilityTimelineRound {
  userMessage?: AiObservabilityTimelineMessage | null;
  assistantMessage?: AiObservabilityTimelineMessage | null;
  traces: AiObservabilityTimelineTrace[];
}

/** 会话审计时间线：会话概要 + 按轮次组织的事件流 */
export interface AiObservabilityTimeline {
  conversation: AiObservabilityTimelineConversation;
  rounds: AiObservabilityTimelineRound[];
}

// ==================== 异常总览 ====================

/** 异常总览统计 */
export interface AiObservabilitySummary {
  total: number;
  successCount: number;
  failedCount: number;
  interruptedCount: number;
  timeoutCount: number;
  /** 配额拒绝数（按采集链路写入的拒绝类 error_type 统计） */
  quotaRejected: number;
  /** 高风险调用数（推理步数超阈值） */
  highRiskCalls: number;
}

// ==================== 资源消耗 ====================

/** 资源消耗聚合查询参数 */
export interface AiObservabilityCostsQuery extends PageQuery {
  dimension?: AiObservabilityCostDimension;
  startTime?: string;
  endTime?: string;
}

/** 按维度聚合的资源消耗项 */
export interface AiObservabilityCostItem {
  /** 模型标识（model 维度） */
  model?: string;
  /** 智能体编码（agent 维度） */
  agentCode?: string;
  /** 用户 ID（user 维度） */
  userId?: number;
  traceCount: number;
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
}

/** 按日 Token 消耗趋势 */
export interface AiObservabilityCostTrendItem {
  /** 日期（YYYY-MM-DD） */
  date: string;
  traceCount: number;
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
}

/** 资源消耗聚合结果 */
export interface AiObservabilityCostsResult {
  items: AiObservabilityCostItem[];
  /** 聚合分组总数（items 为当前分页切片） */
  total: number;
  trend: AiObservabilityCostTrendItem[];
}

// ==================== 性能趋势 ====================

/** 性能趋势查询参数 */
export interface AiObservabilityTrendsQuery {
  dimension?: AiObservabilityTrendDimension;
  startTime?: string;
  endTime?: string;
}

/** 性能趋势项 */
export interface AiObservabilityTrendItem {
  model?: string;
  agentCode?: string;
  /** 日期（YYYY-MM-DD） */
  date: string;
  callCount: number;
  successCount: number;
  /** 成功率（百分比 0-100） */
  successRate: number;
  /** 平均首 Token 延迟（毫秒，成功调用口径） */
  avgFirstTokenMs?: number;
  avgDurationMs?: number;
}
