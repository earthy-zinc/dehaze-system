// AI 对话 Store 共享层：常量、跨 slice 类型与工具函数
import type {
  ConversationStatus,
  ConversationVO,
  Feedback,
  Message,
  Usage,
} from "clarity-sdk-js";
import type { Ref } from "vue";
import type { ChatMessageActionVM } from "@/components/ai-chat/types";
import type { StreamReducerState } from "./stream-reducer";

export type ChatScope = "self" | "admin";
export type ConversationFilterStatus = ConversationStatus | 0;
export type BatchAction = "archive" | "restore" | "delete";

export const MESSAGES_PAGE_SIZE = 50;
/** 流式超时：120 秒无新 token 判定失败 */
export const STREAM_TIMEOUT_MS = 120_000;
export const TIMEOUT_CHECK_INTERVAL_MS = 15_000;
/** 断线重连：最大 3 次，间隔 3 秒 */
export const RECONNECT_MAX_ATTEMPTS = 3;
export const RECONNECT_INTERVAL_MS = 3_000;
/** async_wait 挂起轮询：5 秒一次，最长 10 分钟 */
export const ASYNC_POLL_INTERVAL_MS = 5_000;
export const ASYNC_POLL_MAX_TIMES = 120;

/** 契约 Message.usage 必填，无计费时为全 0 */
export const EMPTY_USAGE: Usage = {
  inputTokens: 0,
  outputTokens: 0,
  cachedInputTokens: 0,
  credits: 0,
};

/**
 * 流会话状态（全局唯一的流式态）。
 * idle 无流；running 正在产出；interrupted 本轮通道已关、等待用户处理 pendingInterrupt。
 */
export type StreamStatus = "idle" | "running" | "interrupted";

// 中断子类型规则只有一份：实现在 utils/interruptKind，这里只做存储层的转出口，
// 避免组件与 store 各自维护合法值列表（两套列表会让 UI 可点、store 静默丢弃）。
export { resolveConfirmKind } from "@/utils/interruptKind";

/**
 * 消息动作（store 可处理的子集）。
 *
 * 唯一来源是组件层的 `ChatMessageActionVM`：copy/edit 属页面内动作（剪贴板、编辑弹窗），
 * 用 Exclude 在类型层剔除，不落 store。新增动作时组件层加变体，此处自动跟随；
 * 若新增的不是页面内动作，`dispatchMessageAction` 的穷尽检查会立即编译失败。
 */
export type MessageAction = Exclude<
  ChatMessageActionVM,
  { type: "copy" } | { type: "edit" }
>;

/**
 * 当前流会话（transport 生命周期 + 归约所需的最小上下文）。
 * 消息内容不在此复制：块/计划/中断/用量一律写在 messages[id] 实体上。
 */
export interface StreamSession extends StreamReducerState {
  conversationId: number;
  /** 建立流时的数据范围（跨页面切换 scope 后仍按原口径收尾：自动朗读/已读） */
  originScope: ChatScope;
  controller: AbortController | null;
  streamSessionId?: string;
  /** 最近收到的 SSE 事件 ID（断点续传，per stream_session 计数，重连不可跨会话复用） */
  lastEventId?: string;
  reconnectAttempts: number;
  reconnectTimer: number | null;
  timeoutTimer: number | null;
  /** 流已到达终态（message.end / 网络失败 / 用户停止），后续网络错误不再触发重连 */
  finished: boolean;
  lastTokenAt: number;
}

/**
 * 各 slice 共享的 store 状态上下文。
 * slice 工厂函数先接收该 ctx，再各自返回自身方法，由 index.ts 合并为 store。
 */
export interface ChatCtx {
  scope: Ref<ChatScope>;
  conversations: Ref<ConversationVO[]>;
  /** 消息实体（id → 契约 Message），唯一内容源 */
  messages: Ref<Record<number, Message>>;
  /** 消息顺序（id 升序；本地负数占位在末尾） */
  messageOrder: Ref<number[]>;
  currentConversationId: Ref<number | null>;
  /** 全局流式态：idle / running / interrupted */
  streamStatus: Ref<StreamStatus>;
  /** 当前流目标消息 id（interrupted 时保留，供中断卡与 resume 定位） */
  streamMessageId: Ref<number | null>;
  streamSessionId: Ref<string | null>;
  selectionMode: Ref<boolean>;
  scrollFollowEnabled: Ref<boolean>;
  quotedMessage: Ref<Message | null>;
  feedbackByMessage: Ref<Record<number, Feedback | null>>;
  session: StreamSession | null;
}

export function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * 本地占位消息 ID 序列（负数，与服务端正数 ID 空间隔离）。
 * 单调递减而非 -Date.now()：同毫秒内连续创建多条本地消息时会碰撞。
 */
let localMessageIdSeq = 0;

export function nextLocalMessageId() {
  return --localMessageIdSeq;
}

/** 将 markdown 文本转为适合语音朗读的纯文本（手动朗读与自动播报共用） */
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
