// AI 对话 Store 共享层：常量、跨 slice 类型与工具函数
import type {
  AiMessageVO,
  ConversationStatus,
  ConversationVO,
  FeedbackVO,
  InterruptEvent,
  MemoryVO,
  ThoughtEvent,
  TokenUsage,
} from "dehaze-sdk-js";
import type { ComputedRef, Ref } from "vue";
import { readConfirmKind } from "@/components/ai-chat/adapters/fromSdk";
import type { ChatConfirmKindVM, ChatPlanVM } from "@/components/ai-chat/types";

export type ChatScope = "self" | "admin";
export type ConversationFilterStatus = ConversationStatus | 0;
export type BatchAction = "archive" | "restore" | "delete";

export const MESSAGES_PAGE_SIZE = 50;
/** 子智能体粒度用量（message.end usage.subAgents，仅存在子智能体调用时下发） */
export type SubAgentUsage = NonNullable<TokenUsage["subAgents"]>;
/** 流式超时：120 秒无新 token 判定失败 */
export const STREAM_TIMEOUT_MS = 120_000;
export const TIMEOUT_CHECK_INTERVAL_MS = 15_000;
/** 断线重连：最大 3 次，间隔 3 秒 */
export const RECONNECT_MAX_ATTEMPTS = 3;
export const RECONNECT_INTERVAL_MS = 3_000;
/** async_wait 挂起轮询：5 秒一次，最长 10 分钟 */
export const ASYNC_POLL_INTERVAL_MS = 5_000;
export const ASYNC_POLL_MAX_TIMES = 120;

export interface ToolCallDraft {
  name?: string;
  args: string;
}

/**
 * 后端 SSE 契约：思考内容块固定独立索引（SseEventConverter._THINKING_INDEX=1，
 * 文本块固定 index=0），与块出现顺序无关（首个块即 thinking 时仍为 1）。
 * 前端据此判定思考块，并在收到 thinking 类型 start 时做契约断言（偏离即显式暴露）。
 */
export const THINKING_BLOCK_INDEX = 1;

/** 单段思考内容（后端按 content_block.start/stop 分段推送，thinking 块固定 index=1） */
export interface ThinkingSegment {
  text: string;
  /** 是否已收到 content_block.stop */
  closed: boolean;
}

/** 思考过程状态：多段思考 + 前端计时（流式期间由 SSE 事件驱动） */
export interface ThinkingState {
  segments: ThinkingSegment[];
  /** 首段开始时刻（前端时钟） */
  startAt: number;
  /** 末段结束时刻；null 表示仍在思考 */
  endAt: number | null;
  /** 是否仍在流式思考（存在未闭合段）；与 vm 层思考归约（ChatThinkingVM）保持同形，便于复用纯函数 */
  streaming: boolean;
}

/**
 * 归一 confirm 中断子类型。非 confirm 返回 undefined；未知/缺失 confirmKind 抛错——
 * 后端 resume 对未知子类型直接抛业务异常（不做静默兜底），前端路由镜像该口径，禁止静默吞掉。
 * 字段归一复用适配层 readConfirmKind（展示映射不抛错，仅路由在此拦截）。
 */
export function resolveConfirmKind(
  interrupt: InterruptEvent
): ChatConfirmKindVM | undefined {
  if (interrupt.type !== "confirm") return undefined;
  const kind = readConfirmKind(interrupt.data);
  if (kind === undefined) {
    const raw = interrupt.data?.confirmKind;
    throw new Error(`未知的确认中断子类型: ${String(raw)}`);
  }
  return kind;
}

export interface StreamSession {
  conversationId: number;
  controller: AbortController | null;
  /**
   * 流式消息对象引用（占位/当前消息）。流式增量（flush/onEnd）直接操作该引用，
   * 避免并发 fetchMessages 覆盖消息列表后 findMessage 找不到消息，
   * 导致内容不渲染、status 不清、永久"正在思考"。
   */
  message: AiMessageVO | null;
  /** 归约状态（纯数据；与响应式 ref 的同步经 effects 解释器完成） */
  state: StreamViewState;
  flushScheduled: boolean;
  lastTokenAt: number;
  streamSessionId?: string;
  /** 最近收到的 SSE 事件 ID（断点续传，per stream_session 计数，重连不可跨会话复用） */
  lastEventId?: string;
  reconnectAttempts: number;
  reconnectTimer: number | null;
  timeoutTimer: number | null;
  /** 流已到达终态（message.end / error / 用户停止），后续网络错误不再触发重连 */
  finished: boolean;
}

/**
 * 流式会话视图状态（纯数据，全部由 reduceStreamState 归约产出）。
 * 与响应式 ref 的桥接由 stream slice 的 effects 解释器完成，状态本身不含任何响应式对象，
 * 因此可脱离 transport（SDK/定时器）单测。
 */
export interface StreamViewState {
  /** assistant 消息当前 ID（新消息为占位负数，message.start 后替换） */
  messageId: number;
  /** 未刷入消息的正文缓冲（rAF 批量写回 message.content） */
  textBuffer: string;
  /** 思考态（与 ai-chat/vm 思考归约 ChatThinkingVM 同形，复用其纯函数） */
  thinking: ThinkingState | null;
  /** 未并入思考段的增量缓冲（rAF 批量刷入 segment.text，避免逐 delta 重渲染 Markdown） */
  thinkingBuffer: string;
  /** 进行中的工具块草稿（按内容块 index） */
  toolBlocks: Map<number, ToolCallDraft>;
  /** 已完成的工具调用（onEnd 挂载到消息） */
  toolCalls: unknown[];
  /** 推理步骤（按 position 正序） */
  thoughts: ThoughtEvent[];
  /** 计划（Plan-and-Execute 展示态） */
  plan?: ChatPlanVM;
  /** 待处理中断 */
  interrupts: InterruptEvent[];
  /** 推荐追问 */
  suggestions: string[];
  /** 当前流式消息 id（null 表示无进行中流式） */
  streamingMessageId: number | null;
  /** 当前中断挂起的消息 id */
  interruptedMessageId: number | null;
  /** 恢复中断的续流：首个事件（message.start）到达即视为恢复成功，此时才清理中断点 */
  clearInterruptOnStart?: boolean;
}

/**
 * 各 slice 共享的 store 状态上下文。
 * slice 工厂函数先接收该 ctx，再各自返回自身方法，由 index.ts 合并为 store。
 * session 为 stream 与 messages 共享的当前流式会话引用（fetchMessages 需据此保留流式对象）。
 */
export interface ChatCtx {
  scope: Ref<ChatScope>;
  conversations: Ref<ConversationVO[]>;
  messages: Ref<AiMessageVO[]>;
  currentConversationId: Ref<number | null>;
  streamingMessageId: Ref<number | null>;
  interruptedMessageId: Ref<number | null>;
  interrupts: Ref<InterruptEvent[]>;
  suggestions: Ref<string[]>;
  thoughtsByMessage: Ref<Record<number, ThoughtEvent[]>>;
  thinkingByMessage: Ref<Record<number, ThinkingState>>;
  toolCallsByMessage: Ref<Record<number, unknown[]>>;
  /** 计划状态：按 assistant 消息 id 累积 plan 事件（生成/更新/重规划） */
  planByMessage: Ref<Record<number, ChatPlanVM>>;
  messageMemories: Ref<Record<number, MemoryVO[]>>;
  /** 子智能体粒度用量：按 assistant 消息 id 归集 message.end 下发的 usage.subAgents */
  subAgentsByMessage: Ref<Record<number, SubAgentUsage>>;
  selectionMode: Ref<boolean>;
  scrollFollowEnabled: Ref<boolean>;
  quotedMessage: Ref<AiMessageVO | null>;
  feedbackByMessage: Ref<Record<number, FeedbackVO | null>>;
  isStreaming: ComputedRef<boolean>;
  session: StreamSession | null;
}

export function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * 本地占位消息 ID 序列（负数，与服务端正数 ID 空间隔离）。
 * 单调递减而非 -Date.now()：同毫秒内连续创建多条本地消息时（如 sendMessage 的
 * 用户消息 + assistant 占位），-Date.now()-1 与 -Date.now() 在毫秒跳变时会碰撞，
 * findMessage 按 id 误绑到用户消息，占位消息无人认领导致永久"正在思考"。
 */
let localMessageIdSeq = 0;

export function nextLocalMessageId() {
  return --localMessageIdSeq;
}

/** 将 markdown 文本转为适合语音朗读的纯文本（手动朗读与自动播报共用，避免读出 `#`/`**` 等符号） */
export function toSpeechText(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(\*|_)(.*?)\1/g, "$2")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/[|]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
