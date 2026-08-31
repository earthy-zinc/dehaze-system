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
  startTime?: string;
  endTime?: string;
}

/** 过程链列表项 */
export interface AiObservabilityTraceItem {
  traceId: string;
  conversationId: number;
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
  toolCall?: AiObservabilityToolCall | null;
  inputSnapshot?: AiObservabilityInputSnapshot | null;
  outputSnapshot?: AiObservabilityOutputSnapshot | null;
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
  /** 父过程链 ID（子 Agent 关联主链时填充） */
  parentTraceId?: string;
}

/** 过程链计费明细 */
export interface AiObservabilityTraceBilling {
  /** 计费类型 */
  billType?: string;
  model?: string;
  /** 实际路由模型（与请求模型不一致时填充） */
  actualModel?: string;
  providerId?: number;
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
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
