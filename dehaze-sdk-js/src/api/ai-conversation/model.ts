import { PageQuery } from "@/types";

// ==================== 枚举类型 ====================

/** 会话状态：1-活跃，2-已归档 */
export type ConversationStatus = 1 | 2;

/** 消息角色 */
export type MessageRole = "user" | "assistant" | "system";

/** 消息状态：1-流式输出中，2-已完成，3-失败，4-已取消 */
export type MessageStatus = 1 | 2 | 3 | 4;

/** 推理中断类型 */
export type InterruptType = "confirm" | "quota" | "async_wait" | "plan_approve";

/**
 * 消息停止原因。
 * max_steps/token_budget_exceeded：推理护栏终止；quota_exceeded：配额预扣阻断（direct 路径终态）
 */
export type StopReason =
  | "stop"
  | "tool_calls"
  | "length"
  | "content_filter"
  | "canceled"
  | "error"
  | "max_steps"
  | "token_budget_exceeded"
  | "quota_exceeded";

/** 内容块类型 */
export type ContentBlockType = "text" | "thinking" | "tool_use";

/** 长期记忆子类型 */
export type MemoryType = "episodic" | "semantic" | "procedural";

/** 记忆来源 */
export type MemorySource = "conversation" | "feedback" | "reflection" | "manual";

/** 产物类型 */
export type ArtifactType = "image_result" | "metric_report" | "algorithm_recommend" | "file_ref";

/** 反馈评分：1-点赞，-1-点踩 */
export type FeedbackRating = 1 | -1;

// ==================== 会话管理 ====================

/** 模型配置（会话级参数） */
export interface ModelConfig {
  temperature?: number;
  maxOutputTokens?: number;
  topP?: number;
}

/** 创建会话表单 */
export interface ConversationCreateForm {
  title?: string;
  /** 模型 ID（不传则使用用户偏好默认或平台默认） */
  model?: string;
  systemPrompt?: string;
  modelConfig?: ModelConfig;
  /** 绑定的 API Key ID */
  apiKeyId?: number;
  /** 会话使用的 Agent 编码（为空使用默认 Agent） */
  agentCode?: string;
  /** 会话场景（general/image_dispatch/multi_step/algorithm_recommend/scheduled_task） */
  scene?: string;
  /** 类似问题推荐开关（关闭后回复完成不推送 suggestions 事件） */
  suggestionsEnabled?: boolean;
}

/** 会话列表查询参数 */
export interface ConversationQuery extends PageQuery {
  keyword?: string;
  status?: ConversationStatus;
  /** 管理端审计视角：admin 返回全量用户会话（需 ai:conversation:audit），省略为本人会话 */
  view?: "admin";
}

/** 会话更新表单（PATCH 部分更新；归档用 status: 2） */
export interface ConversationUpdateForm {
  title?: string;
  /** 是否置顶（0/1） */
  pinned?: number;
  /** 会话状态（1-活跃；2-已归档） */
  status?: ConversationStatus;
  model?: string;
  modelConfig?: ModelConfig;
  systemPrompt?: string;
  /** 切换 Agent 编码（下一条消息生效） */
  agentCode?: string;
  /** 类似问题推荐开关（关闭后回复完成不推送 suggestions 事件） */
  suggestionsEnabled?: boolean;
}

/** 会话视图对象 */
export interface ConversationVO {
  id: number;
  title: string;
  /** 标题来源：auto-自动生成，manual-手动修改 */
  titleSource?: string;
  model?: string;
  /** 会话使用的 Agent 编码 */
  agentCode?: string;
  /** 会话锚定的 Agent 已发布版本号 */
  agentVersion?: number;
  modelConfig?: ModelConfig;
  systemPrompt?: string;
  /** 类似问题推荐开关（0-关，1-开） */
  suggestionsEnabled?: number;
  apiKeyId?: number;
  status: ConversationStatus;
  messageCount: number;
  /** 是否置顶（0/1） */
  pinned: number;
  /** 未读消息数 */
  unreadCount?: number;
  /** 最后已读消息 ID */
  lastReadMessageId?: number;
  /** 当前激活分支末端消息 ID */
  currentBranchMessageId?: number;
  /** 会话摘要（自动压缩生成） */
  summary?: string;
  lastMessageAt?: string;
  /** 会话所属用户 ID（管理端审计视角展示） */
  userId?: number;
  /** 会话用户名（管理端审计视角展示） */
  userName?: string;
  /** 累计消耗 Token 数（input + output，管理端审计视角展示，无计费记录为 0） */
  tokenConsumed?: number;
  /** 累计消耗积分数（管理端审计视角展示，无计费记录为 0） */
  creditsConsumed?: number;
  /** 会话异常类型：failed-存在失败消息，quota-配额不足中断，canceled-存在已取消消息 */
  anomalyType?: string;
  /** 会话异常展示标签（中文，前端无需硬编码映射） */
  anomalyLabel?: string;
  /** 搜索命中消息内容时的命中消息 ID（标题命中时不返回） */
  matchedMessageId?: number;
  createTime: string;
  updateTime?: string;
}

// ==================== 消息 ====================

/** 发送消息表单（仅 content/model） */
export interface SendMessageForm {
  content: string;
  model?: string;
}

/** 恢复中断推理表单 */
export interface MessageResumeForm {
  /** confirm 中断必填：True 接受推荐；False 拒绝 */
  confirm?: boolean;
  /** 确认参数（如 algorithmId 表示选择了备选算法） */
  params?: Record<string, unknown>;
  /**
   * Plan-and-Execute 计划干预（仅计划待执行时允许）。
   *
   * `remove`/`reorder` 为任务 ID 列表、`add` 仅接受单条；外层键以 `plan_edit` 上行，
   * 内层为对外契约、一律 camelCase。
   */
  planEdit?: {
    remove?: string[];
    reorder?: string[];
    add?: { description?: string; dependsOn?: string[]; toolHint?: string; paradigm?: string };
  };
}

/**
 * 推理步骤（消息详情附带，按 position 正序）。
 *
 * 对应后端 `AgentThoughtResult`，与流式 `thought` 事件（`ThoughtEvent`）字段不同：
 * 此处为落库记录，额外带 id/messageId/conversationId/createTime。
 */
export interface AiMessageThought {
  id: number;
  messageId: number;
  conversationId: number;
  /** 步骤序号 */
  position: number;
  /** 步骤来源 Agent 编码（空=主 Agent） */
  agentCode?: string;
  /** 是否为子 Agent 的推理步骤（0-否，1-是） */
  isSubagent?: number;
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
  createTime?: string;
}

/** AI 对话消息视图对象 */
export interface AiMessageVO {
  id: number;
  conversationId: number;
  role: MessageRole;
  content?: string;
  status: MessageStatus;
  /** 父消息 ID（分支对话） */
  parentMessageId?: number;
  /** 工具调用列表 */
  toolCalls?: unknown;
  toolCallId?: string;
  /** 实际使用的模型 */
  model?: string;
  error?: string;
  /** 元数据（后端 `metadata_` 的序列化别名） */
  metadata?: unknown;
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  /** 积分消耗 */
  credits?: number;
  /** 是否已编辑 */
  edited?: number;
  /** 编辑前原文 */
  originalContent?: string;
  /** 本次回复注入的长期记忆 ID 清单（注入可见性，供展开查看引用了哪些记忆） */
  usedMemoryIds?: number[];
  /** 关联异步任务 ID */
  taskId?: string;
  createTime: string;
  /** 推理步骤/思考链（消息列表与详情返回，按 position 正序；无思考时为空数组） */
  thoughts?: AiMessageThought[];
}

/** 编辑用户消息表单 */
export interface EditMessageForm {
  content: string;
}

// ==================== SSE 事件类型 ====================

/** message.start 事件 */
export interface MessageStartEvent {
  messageId: number;
  conversationId: number;
  model: string;
  /** 流式会话 ID（断线重连：携带 Last-Event-ID 调用 stream/{streamSessionId} 端点） */
  streamSessionId: string;
}

/** 内容块类型标识 */
export interface ContentBlockStartEvent {
  index: number;
  type: ContentBlockType;
}

/** 内容块增量 */
export interface ContentBlockDeltaEvent {
  index: number;
  delta: {
    type: "text_delta" | "thinking_delta" | "input_json_delta";
    /** 文本增量 */
    text?: string;
    /** 思考增量 */
    thinking?: string;
    /** 工具参数流式增量（部分 JSON 字符串，需拼接） */
    partialJson?: string;
    /** 工具调用名（input_json_delta 首片段携带） */
    name?: string;
  };
}

/** 内容块结束 */
export interface ContentBlockStopEvent {
  index: number;
}

/** 推理步骤完成事件（status: 1 成功 / 2 失败 / 3 跳过） */
export interface ThoughtEvent {
  position: number;
  thought?: string;
  tool?: string;
  toolInput?: unknown;
  observation?: string;
  status: number;
  /** 失败原因（status=2 时透出） */
  error?: string;
  /** 本步骤耗时（毫秒） */
  latencyMs?: number;
}

/** 计划任务项（plan 事件与 plan_approve 中断共用同一形状） */
export interface PlanTask {
  id?: string;
  description?: string;
  dependsOn?: string[];
  /** 任务状态：pending|done|failed */
  status?: string;
  paradigm?: string;
  toolHint?: string;
  result?: string;
}

/** 计划修订记录（Replanner 修订后追加） */
export interface PlanRevision {
  /** 修订序号 */
  revisionNo: number;
  reason: string;
  /** 本次修订涉及的任务 ID */
  changedTaskIds: string[];
}

/**
 * Plan-and-Execute 计划（`plan` SSE 事件载荷，两条下发路径的完整形状）。
 *
 * `revisions` 两条路径恒有；`phase` 仅事件路径下发（中断路径见 `PlanPayload`）。
 */
export interface Plan {
  tasks: PlanTask[];
  /** 计划状态：pending|executing|revised|done */
  status?: string;
  revisions: PlanRevision[];
  /** 计划阶段（仅 plan 事件下发） */
  phase?: string;
}

/** `plan_approve` 中断 `data.plan` 的形状：与事件同形，仅不含 `phase` */
export type PlanPayload = Omit<Plan, "phase">;

/** suggestions 事件（回复完成后推荐追问） */
export interface SuggestionsEvent {
  questions: Array<{ question: string }>;
}

/** 中断事件 */
export interface InterruptEvent {
  type: InterruptType;
  data: InterruptData;
}

/**
 * 中断数据（全 camelCase）。按中断类型出现的键：
 * - confirm(algorithm_recommend)：{confirmKind, artifactId, recommendation, alternatives, imageFeatures}
 *   - recommendation：{recommendationId, algorithmId, algorithmName, reason, effectDescription}
 *   - alternatives：[{algorithmId, algorithmName, matchScore, reason}]
 * - confirm(dangerous_op)：{confirmKind, action, command, impact}（Shell/高风险代码执行）
 *   或 {confirmKind, action: "write_conflict", tool, resource, previousWriter, impact, reason}
 * - confirm(tool_permission)：{confirmKind, tool, reason, detail}
 * - quota：{upgradeTip, usedDaily, dailyLimit, usedMonthly, monthlyLimit, quotaDataError}
 * - async_wait：{taskId, taskType, estDuration, imageCount}
 * - plan_approve：{plan}（统一计划形状，不含 phase）
 */
export interface InterruptData {
  // confirm 子类型，决定 resume 路由
  confirmKind?: "algorithm_recommend" | "tool_permission" | "dangerous_op";
  // confirm(algorithm_recommend)：推荐主算法与备选（对应后端 summary.algorithm 结构）
  artifactId?: number;
  recommendation?: {
    recommendationId: number;
    algorithmId: number;
    algorithmName: string;
    reason: string;
    effectDescription?: string;
  };
  alternatives?: Array<{
    algorithmId: number;
    algorithmName: string;
    matchScore: number;
    reason: string;
  }>;
  imageFeatures?: { hazeLevel?: number; sceneType?: string; lighting?: string };
  // confirm(dangerous_op/tool_permission)：危险操作/写冲突/权限确认
  action?: string;
  command?: string;
  impact?: string;
  tool?: string;
  reason?: string;
  detail?: string;
  resource?: string;
  previousWriter?: string;
  // quota: 配额用量与升级引导（组装失败时 quotaDataError=true，勿伪造达标展示）
  upgradeTip?: string;
  usedDaily?: number;
  dailyLimit?: number;
  usedMonthly?: number;
  monthlyLimit?: number;
  quotaDataError?: boolean;
  // async_wait: 异步任务信息
  taskId?: string;
  taskType?: string;
  estDuration?: string;
  imageCount?: number;
  // plan_approve: 待确认的计划（与 plan 事件同形，不含 phase）
  plan?: PlanPayload;
}

/** 错误事件 */
export interface ErrorEvent {
  code: string;
  message: string;
}

/** Token 用量 */
export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  /** 积分消耗 */
  credits: number;
  /**
   * 子智能体粒度用量（仅在存在子智能体调用时下发；后端不下发时不出现该键）
   */
  subAgents?: Array<{
    /** 子智能体编码 */
    agentCode: string;
    inputTokens: number;
    outputTokens: number;
    cachedInputTokens: number;
    /** 积分消耗 */
    credits: number;
  }>;
}

/** message.end 事件 */
export interface MessageEndEvent {
  stopReason: StopReason;
  usage: TokenUsage;
}

// ==================== 上下文：产物与记忆 ====================

/** 中间产物视图对象 */
export interface ArtifactVO {
  id: number;
  conversationId: number;
  messageId?: number;
  type: ArtifactType;
  /** 引用业务表（sys_pred_log / sys_eval_log / sys_file / sys_recommendation） */
  refType?: string;
  /** 引用业务 ID */
  refId?: number;
  /** 业务摘要元数据（指标数值/算法信息等，绝不存 URL） */
  summary?: unknown;
  /** 是否失效（关联文件被删除时标记） */
  isInvalid: number;
  createTime?: string;
}

/** 长期记忆视图对象 */
export interface MemoryVO {
  id: number;
  userId: number;
  memoryType: MemoryType;
  content: string;
  metadata?: unknown;
  importance: number;
  accessCount: number;
  lastAccessedAt?: string;
  source: MemorySource;
  /** 状态（1-启用；0-禁用） */
  status: number;
  /** 是否归档 */
  archived: number;
  createTime: string;
  updateTime?: string;
}

/** 创建记忆表单 */
export interface MemoryCreateForm {
  memoryType: MemoryType;
  content: string;
  metadata?: Record<string, unknown>;
  /** 重要性评分（0-100） */
  importance?: number;
  source?: MemorySource;
}

/** 更新记忆表单 */
export interface MemoryUpdateForm {
  content?: string;
  /** 重要性评分（0-100） */
  importance?: number;
  /** 状态（1-启用；0-禁用） */
  status?: 0 | 1;
}

/** 记忆分页/清空查询参数 */
export interface MemoryQuery extends PageQuery {
  memoryType?: MemoryType;
  source?: MemorySource;
  /** 清空/恢复的时间范围起（按创建时间） */
  start?: string;
  /** 清空/恢复的时间范围止（按创建时间） */
  end?: string;
}

// ==================== 消息反馈 ====================

/** 反馈表单 */
export interface FeedbackForm {
  rating: FeedbackRating;
  /** 预设标签：点赞 accurate/detailed/concise/creative；点踩 incorrect/irrelevant/incomplete/too_long/bad_citation/harmful（点踩必选其一） */
  tags?: string[];
  /** 改进建议（可选，点踩不强制填写） */
  comment?: string;
}

/** 反馈视图对象 */
export interface FeedbackVO {
  id: number;
  messageId: number;
  userId: number;
  rating: FeedbackRating;
  tags?: string[];
  comment?: string;
  createTime: string;
  updateTime?: string;
}

// ==================== OpenAI 兼容 API ====================

/** OpenAI 兼容消息内容块 */
export type OpenAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: "auto" | "low" | "high" } };

/** OpenAI 兼容消息 */
export interface OpenAIChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | OpenAIContentPart[];
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
  name?: string;
}

/** OpenAI 兼容工具定义 */
export interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
}

/** OpenAI 兼容对话补全请求 */
export interface OpenAICompletionForm {
  model: string;
  messages: OpenAIChatMessage[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  n?: number;
  stop?: string | string[];
  max_tokens?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  tools?: OpenAITool[];
  tool_choice?: "auto" | "none" | { type: "function"; function: { name: string } };
  conversation_id?: string;
}

// ==================== Claude 兼容 API ====================

/** Claude 兼容消息内容块 */
export type ClaudeContentBlock =
  | { type: "text"; text: string }
  | {
      type: "image";
      source: {
        type: "base64";
        media_type: "image/jpeg" | "image/png" | "image/gif" | "image/webp";
        data: string;
      };
    };

/** Claude 兼容消息 */
export interface ClaudeMessage {
  role: "user" | "assistant";
  content: string | ClaudeContentBlock[];
}

/** Claude 兼容消息对话请求 */
export interface ClaudeMessageForm {
  model: string;
  messages: ClaudeMessage[];
  system?: string;
  stream?: boolean;
  /** Claude 规范要求 max_tokens 必填 */
  max_tokens: number;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  conversation_id?: string;
}

// ==================== 兼容调用审计（管理员） ====================

/** 兼容调用审计查询参数（需 ai:conversation:audit） */
export interface CompatCallQuery extends PageQuery {
  /** 按 API Key ID 筛选 */
  keyId?: number;
  /** 按模型筛选 */
  model?: string;
  /** 开始时间 */
  startTime?: string;
  /** 结束时间 */
  endTime?: string;
}
