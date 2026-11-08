// 流式传输层：错误映射、连接建立、断线重连、超时监控与中止。
//
// 本层不依赖 store：状态写入与收尾通过注入的钩子完成，事件回调由调用方提供
// （buildHandlers），因此与 stream-events 保持单向依赖（events → transport）。
import { AiConversationAPI, type MessageStreamHandlers } from "clarity-sdk-js";
import {
  RECONNECT_INTERVAL_MS,
  RECONNECT_MAX_ATTEMPTS,
  STREAM_TIMEOUT_MS,
  TIMEOUT_CHECK_INTERVAL_MS,
  type ChatScope,
  type StreamSession,
} from "./shared";

/** 流式 HTTP 业务错误（SDK SSEHttpError 的形状）：按结构判定，避免跨包 instanceof 失效 */
interface StreamHttpError extends Error {
  status?: number;
  code?: string;
  msg?: string;
}

function isStreamHttpError(error: unknown): error is StreamHttpError {
  return typeof (error as StreamHttpError | null)?.status === "number";
}

/**
 * 把流式失败的 HTTP 业务错误映射为用户可读文案；返回 null 表示非 HTTP 业务错误。
 * 后端失败响应带业务码/文案时按真实原因提示，避免一律报"网络连接中断"误导重试。
 */
export function mapStreamHttpError(error: unknown): string | null {
  if (!isStreamHttpError(error)) return null;
  if (error.msg) return error.msg;
  return `请求失败（HTTP ${error.status}），请稍后重试`;
}

/** 传输层向 slice 回调的生命周期钩子 */
export interface StreamTransportHooks {
  /** 当前会话的 SSE 回调（打开/重连时复用同一套） */
  buildHandlers(session: StreamSession): MessageStreamHandlers;
  /** 失败收口：消息置 failed、流状态复位、释放会话 */
  fail(session: StreamSession, reason: string): void;
  /** 中断后连接关闭（契约内正常收束，不判失败） */
  onInterruptedClose(session: StreamSession): void;
}

/** 构造流会话（只建对象，不发起请求；ctx.session 归属由调用方持有） */
export function createStreamSession(params: {
  conversationId: number;
  messageId: number;
  localUserMessageId: number | null;
  originScope: ChatScope;
  clearInterruptOnEvent: boolean;
}): StreamSession {
  return {
    conversationId: params.conversationId,
    controller: null,
    messageId: params.messageId,
    localUserMessageId: params.localUserMessageId,
    reconnectAttempts: 0,
    reconnectTimer: null,
    timeoutTimer: null,
    originScope: params.originScope,
    finished: false,
    started: false,
    interrupted: false,
    clearInterruptOnEvent: params.clearInterruptOnEvent,
    lastTokenAt: Date.now(),
  };
}

/** 发起流：调用 opener 建立 SSE 连接，并启动超时监控 */
export function startStream(
  session: StreamSession,
  opener: (handlers: MessageStreamHandlers) => AbortController | void,
  hooks: StreamTransportHooks
): void {
  session.controller = opener(hooks.buildHandlers(session)) ?? null;
  startTimeoutWatch(session, hooks);
}

/** 中止连接（定时器由会话收尾负责清理） */
export function abortStream(session: StreamSession | null): void {
  session?.controller?.abort();
}

function startTimeoutWatch(
  session: StreamSession,
  hooks: StreamTransportHooks
): void {
  session.lastTokenAt = Date.now();
  session.timeoutTimer = window.setInterval(() => {
    if (Date.now() - session.lastTokenAt > STREAM_TIMEOUT_MS) {
      session.controller?.abort();
      hooks.fail(session, "流式输出超时（120 秒无新内容）");
    }
  }, TIMEOUT_CHECK_INTERVAL_MS);
}

/**
 * 网络断开：有流会话 ID 时自动重连（最大 3 次、间隔 3 秒），否则判定失败。
 * 已经出现过中断的流，连接关闭属契约内的正常收束（等待用户 resume），不判失败。
 */
export function handleStreamDisconnect(
  session: StreamSession,
  hooks: StreamTransportHooks
): void {
  if (session.finished) return;
  if (session.interrupted) {
    hooks.onInterruptedClose(session);
    return;
  }
  if (
    session.streamSessionId &&
    session.reconnectAttempts < RECONNECT_MAX_ATTEMPTS
  ) {
    session.reconnectAttempts++;
    session.reconnectTimer = window.setTimeout(() => {
      // 重连是一条新连接：用新 controller 替换旧引用，后续 stop/超时才能中止当前连接
      session.controller = AiConversationAPI.reconnectStream(
        session.conversationId,
        session.streamSessionId!,
        session.lastEventId ?? "",
        hooks.buildHandlers(session)
      );
    }, RECONNECT_INTERVAL_MS);
    return;
  }
  hooks.fail(session, "网络连接中断，请检查网络后重试");
}
