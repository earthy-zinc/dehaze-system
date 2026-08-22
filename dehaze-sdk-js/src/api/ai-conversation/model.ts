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

/** 消息停止原因 */
export type StopReason = "stop" | "tool_calls" | "length" | "content_filter" | "canceled" | "error";

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
}

/** 会话列表查询参数 */
export interface ConversationQuery extends PageQuery {
  keyword?: string;
  status?: ConversationStatus;
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
  /** Plan-and-Execute 计划干预：{remove, reorder, add} */
  planEdit?: {
    remove?: string[];
    reorder?: string[];
    add?: { description?: string; depends_on?: string[] };
  };
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
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  /** 积分消耗 */
  credits?: number;
  /** 是否已编辑 */
  edited?: number;
  /** 编辑前原文 */
  originalContent?: string;
  /** 关联异步任务 ID */
  taskId?: string;
  createTime: string;
}

/** 编辑用户消息表单 */
export interface EditMessageForm {
  content: string;
}

// ==================== SSE 事件类型 ====================

/** message.start 事件（后端仅推送 messageId/conversationId/model，无 streamSessionId） */
export interface MessageStartEvent {
  messageId: number;
  conversationId: number;
  model: string;
  /** 流式会话 ID（断线重连用；后端 message.start 未推送，需结合其它途径获取） */
  streamSessionId?: string;
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

/** 计划任务项 */
export interface PlanTask {
  id?: string;
  description?: string;
  dependsOn?: string[];
  status?: string;
}

/** plan 事件（Plan-and-Execute 计划） */
export interface PlanEvent {
  tasks: PlanTask[];
  status?: string;
  revisions?: unknown[];
  phase?: string;
}

/** suggestions 事件（回复完成后推荐追问） */
export interface SuggestionsEvent {
  questions: Array<{ question: string }>;
}

/** 中断事件 */
export interface InterruptEvent {
  type: InterruptType;
  data: InterruptData;
}

/** 中断数据（按类型不同） */
export interface InterruptData {
  // confirm: 推荐算法/参数或危险操作详情
  recommendation?: {
    algorithm: { id: number; name: string; description?: string };
    reason: string;
    params?: Record<string, unknown>;
    matchScore: number;
    alternatives?: Array<{
      algorithm: { id: number; name: string };
      reason: string;
      matchScore: number;
    }>;
  };
  // quota: 已用配额、限额、提示
  used?: number;
  limit?: number;
  period?: "daily" | "monthly";
  message?: string;
  // async_wait: 异步任务信息
  taskId?: string;
  taskType?: string;
  estimatedDuration?: number;
  // plan_approve: 待确认的计划
  plan?: PlanEvent;
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

// ==================== 模型管理 ====================

/** 模型创建/更新表单（管理员；能力字段平铺） */
export interface AiModelForm {
  /** 关联供应商 ID（必填） */
  providerId: number;
  modelId: string;
  displayName: string;
  inputRate: number;
  outputRate: number;
  cachedRate: number;
  maxContextTokens: number;
  maxOutputTokens: number;
  supportsMultimodal: boolean;
  supportsToolCall: boolean;
  supportsStreaming: boolean;
  supportsPromptCache: boolean;
  supportsStructuredOutput: boolean;
  /** 降级模型主键（关联 sys_ai_model.id） */
  fallbackModelPk?: number;
  promptCachePrefixLen?: number;
  /** 状态（1-启用；0-禁用） */
  status?: 0 | 1;
  /** 最低可用 VIP 等级（0-所有；1-VIP1 及以上；2-VIP2 及以上） */
  vipLevel?: number;
}

/** 模型分页查询参数 */
export interface AiModelQuery extends PageQuery {
  keyword?: string;
}

/** 模型视图对象 */
export interface AiModelVO {
  id: number;
  /** 关联供应商 ID */
  providerId: number;
  /** 模型业务标识 */
  modelId: string;
  displayName: string;
  inputRate: number;
  outputRate: number;
  cachedRate: number;
  maxContextTokens: number;
  maxOutputTokens: number;
  /** 能力标识（0/1） */
  supportsMultimodal: number;
  supportsToolCall: number;
  supportsStreaming: number;
  supportsPromptCache: number;
  supportsStructuredOutput: number;
  /** 降级模型主键 */
  fallbackModelPk?: number;
  promptCachePrefixLen: number;
  /** 状态（1-启用；0-禁用） */
  status: number;
  /** 最低可用 VIP 等级 */
  vipLevel: number;
  /** 速度档位（fast/medium/slow/unknown） */
  speedTier?: string;
  /** 是否作为其他启用模型的降级目标 */
  isFallbackTarget?: boolean;
  createTime?: string;
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
