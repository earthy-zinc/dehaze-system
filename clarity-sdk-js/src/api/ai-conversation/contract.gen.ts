/**
 * GENERATED FROM config/schema/ai-conversation.schema.json — DO NOT EDIT.
 * 由 scripts/tooling/gen-ai-contract.mjs 生成；改契约请改 Schema 后重新运行生成器。
 */

export type Role = "system" | "user" | "assistant" | "tool";

export type MessageStatus = "running" | "completed" | "failed" | "canceled" | "awaiting_input";

export type BlockKind = "text" | "thinking" | "tool_use";

export type ToolStatus = "running" | "success" | "failed" | "skipped";

export type PlanStatus = "pending" | "running" | "completed" | "failed";

export type InterruptType = "confirm" | "quota" | "async_wait" | "plan_approve";

export type ConfirmKind =
  "algorithm_recommend" | "tool_permission" | "dangerous_op" | "write_conflict";

export interface ToolCall {
  callId: string;
  name: string;
  input?: Record<string, unknown> | null;
  output?: string | null;
  summary?: string | null;
  status: ToolStatus;
  error?: string | null;
  latencyMs?: number | null;
  agentCode?: string | null;
  subAgent: boolean;
}

export interface Block {
  id: string;
  kind: BlockKind;
  seq: number;
  text?: string | null;
  durationMs?: number | null;
  tool?: ToolCall | null;
}

export interface PlanTask {
  id: string;
  description?: string | null;
  dependsOn?: string[] | null;
  status?: PlanStatus | null;
  paradigm?: string | null;
}

export interface PlanRevision {
  revisionNo: number;
  reason: string;
  actor?: string | null;
  at?: string | null;
}

export interface Plan {
  phase?: string | null;
  status: PlanStatus;
  tasks: PlanTask[];
  revisions: PlanRevision[];
  awaitingApproval: boolean;
}

export interface Recommendation {
  modelId: string;
  modelName: string;
  reason?: string | null;
  [key: string]: unknown;
}

export interface Alternative {
  modelId: string;
  modelName: string;
  matchScore?: number | null;
  reason?: string | null;
  [key: string]: unknown;
}

export interface Interrupt {
  id: string;
  type: InterruptType;
  confirmKind?: ConfirmKind | null;
  action?: string | null;
  recommendation?: Recommendation | null;
  alternatives?: Alternative[] | null;
  upgradeTip?: string | null;
  usedDaily?: number | null;
  dailyLimit?: number | null;
  usedMonthly?: number | null;
  monthlyLimit?: number | null;
  estDuration?: string | null;
  imageCount?: number | null;
  plan?: PlanTask[] | null;
  reason?: string | null;
  detail?: string | null;
  createdAt: string;
  expiresAt: string;
  actionable: boolean;
}

export interface SubAgentUsage {
  agentCode: string;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  credits: number;
}

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  credits: number;
  subAgents?: SubAgentUsage[] | null;
}

export interface Message {
  id: number;
  conversationId: number;
  parentMessageId?: number | null;
  role: Role;
  status: MessageStatus;
  model?: string | null;
  error?: string | null;
  blocks: Block[];
  plan?: Plan | null;
  pendingInterrupt?: Interrupt | null;
  usage: Usage;
  artifactIds: number[];
  memoryIds: number[];
  suggestions: string[];
  edited: boolean;
  siblingIndex?: number | null;
  siblingCount?: number | null;
  createTime: string;
  updateTime?: string | null;
}

export interface MessageList {
  list: Message[];
  total: number;
  hasMore: boolean;
}

export interface MessageSend {
  content: string;
  model?: string | null;
}

export interface MessageResume {
  confirm?: boolean | null;
  params?: Record<string, unknown> | null;
  planEdit?: Record<string, unknown> | null;
}

export interface Artifact {
  id: number;
  conversationId: number;
  messageId: number;
  type: string;
  refType?: string | null;
  refId?: number | null;
  summary?: Record<string, unknown> | null;
  isInvalid: boolean;
  createTime?: string | null;
}

export interface Feedback {
  id?: number | null;
  messageId: number;
  userId: number;
  conversationId?: number | null;
  rating: 1 | -1;
  tags?: string[] | null;
  comment?: string | null;
  createTime?: string | null;
  updateTime?: string | null;
}

export interface MessageStartEvent {
  streamSessionId: string;
  userMessage: Message;
  assistantMessage: Message;
}

export interface BlockStartEvent {
  messageId: number;
  block: Block;
}

export interface BlockDeltaEvent {
  messageId: number;
  blockId: string;
  delta: string;
}

export interface BlockStopEvent {
  messageId: number;
  block: Block;
}

export interface PlanUpdateEvent {
  messageId: number;
  plan: Plan;
}

export interface InterruptEvent {
  messageId: number;
  interrupt: Interrupt;
}

export interface UsageEvent {
  messageId: number;
  usage: Usage;
}

export interface SuggestionsEvent {
  messageId: number;
  suggestions: string[];
}

export interface MessageEndEvent {
  messageId: number;
  status: MessageStatus;
  error?: string | null;
  usage?: Usage | null;
}

export interface ErrorEvent {
  code: string;
  message: string;
  messageId?: number | null;
}
