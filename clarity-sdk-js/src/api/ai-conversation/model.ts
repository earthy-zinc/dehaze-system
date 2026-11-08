import type { PageQuery } from "../../types";

// ==================== 契约 DTO（唯一来源：config/schema/ai-conversation.schema.json） ====================

/**
 * 消息、块、计划、中断、用量、产物、反馈与流式事件类型全部由契约生成。
 *
 * 此处只做重导出，不手写同名字段；改字段请改 Schema 后重跑
 * `scripts/tooling/gen-ai-contract.mjs`。
 */
export type {
  Alternative,
  Artifact,
  Block,
  BlockDeltaEvent,
  BlockKind,
  BlockStartEvent,
  BlockStopEvent,
  ConfirmKind,
  ErrorEvent,
  Feedback,
  Interrupt,
  InterruptEvent,
  InterruptType,
  Message,
  MessageEndEvent,
  MessageList,
  MessageResume,
  MessageSend,
  MessageStartEvent,
  MessageStatus,
  Plan,
  PlanRevision,
  PlanStatus,
  PlanTask,
  PlanUpdateEvent,
  Recommendation,
  Role,
  SubAgentUsage,
  SuggestionsEvent,
  ToolCall,
  ToolStatus,
  Usage,
  UsageEvent,
} from "./contract.gen";

// ==================== 会话枚举 ====================

/** 会话状态：1-活跃，2-已归档 */
export type ConversationStatus = 1 | 2;

/** 长期记忆子类型 */
export type MemoryType = "episodic" | "semantic" | "procedural";

/** 记忆来源 */
export type MemorySource = "conversation" | "feedback" | "reflection" | "manual";

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
  currentBranchMessageId?: number | null;
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

// ==================== 消息请求 ====================

/** 会话消息列表查询参数（游标分页，向上增量加载更早历史） */
export interface MessageListQuery {
  /** 游标：仅返回 id < before 的消息（>=1） */
  before?: number;
  /** 每页条数（默认 50，1..100） */
  limit?: number;
  /** 管理端审计视角：admin 读取任意用户会话的消息（需 ai:conversation:audit），省略为本人 */
  view?: "admin";
}

/** 编辑用户消息表单 */
export interface EditMessageForm {
  content: string;
}

// ==================== 上下文：记忆 ====================

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
