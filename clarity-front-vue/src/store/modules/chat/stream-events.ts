// 事件分派层：SSE 回调 → 归约事件 → 副作用调度。
//
// 依赖纯归约器 stream-reducer；不持有定时器、不发请求，网络生命周期交给传输层。
import type { MessageStreamHandlers } from "clarity-sdk-js";
import { findMessage } from "./entities";
import type { ChatCtx, StreamSession } from "./shared";
import {
  reduceStreamEvent,
  type StreamEffect,
  type StreamEvent,
} from "./stream-reducer";
import { mapStreamHttpError } from "./stream-transport";

/** 分派层向 slice 回调的钩子 */
export interface StreamEventHooks {
  /** 解释执行归约产出的副作用 */
  applyEffect(session: StreamSession, effect: StreamEffect): void;
  /** 网络断开（非 HTTP 业务错误）：交给传输层重连/判失败 */
  onDisconnect(session: StreamSession): void;
  /** HTTP 业务错误：按真实原因直接失败，不重连 */
  fail(session: StreamSession, reason: string): void;
}

export function createStreamEvents(ctx: ChatCtx, hooks: StreamEventHooks) {
  /** 归约：事件直接写实体（blocks 是唯一内容源），无中间投影 */
  function dispatch(session: StreamSession, event: StreamEvent) {
    // 终态（message.end / 超时 / 用户停止）后连接可能仍有余留帧：不得再写实体或拉回流状态。
    // 唯一例外是 suggestions：由后端在 message.end 之后异步生成，只补内容、不改终态（与 SDK 分派器一致）。
    if (session.finished && event.kind !== "suggestions") return;
    const effects = reduceStreamEvent(
      {
        messages: ctx.messages.value,
        order: ctx.messageOrder.value,
        state: session,
      },
      event
    );
    if (event.kind === "start") ctx.streamMessageId.value = session.messageId;

    // resume 续流没有 message.start：首个事件到达即视为恢复成功，清掉中断点
    if (session.clearInterruptOnEvent) {
      session.clearInterruptOnEvent = false;
      const message = findMessage(ctx, session.messageId);
      if (message?.pendingInterrupt) {
        message.pendingInterrupt = null;
      }
    }

    for (const effect of effects) hooks.applyEffect(session, effect);
  }

  /** SSE 回调 → 归约事件（network-error/close 属 transport 生命周期，直接交给重连处理） */
  function buildHandlers(session: StreamSession): MessageStreamHandlers {
    return {
      onStart(data) {
        session.lastTokenAt = Date.now();
        if (data.streamSessionId) {
          session.streamSessionId = data.streamSessionId;
          ctx.streamSessionId.value = data.streamSessionId;
        }
        dispatch(session, { kind: "start", data });
      },
      onBlockStart(data) {
        session.lastTokenAt = Date.now();
        dispatch(session, { kind: "block-start", data });
      },
      onBlockDelta(data) {
        session.lastTokenAt = Date.now();
        dispatch(session, { kind: "block-delta", data });
      },
      onBlockStop(data) {
        dispatch(session, { kind: "block-stop", data });
      },
      onPlanUpdate(data) {
        session.lastTokenAt = Date.now();
        dispatch(session, { kind: "plan", data });
      },
      onInterrupt(data) {
        session.lastTokenAt = Date.now();
        dispatch(session, { kind: "interrupt", data });
      },
      onUsage(data) {
        dispatch(session, { kind: "usage", data });
      },
      onSuggestions(data) {
        dispatch(session, { kind: "suggestions", data });
      },
      onEventId(id) {
        session.lastEventId = id;
      },
      onPing() {
        // 心跳仅保活，不重置 token 超时计时
      },
      onError(data) {
        dispatch(session, { kind: "error", data });
      },
      onEnd(data) {
        dispatch(session, { kind: "end", data });
      },
      onNetworkError(error) {
        // HTTP 业务错误（非 2xx 且响应体为业务信封）：按真实原因提示，不重连
        const reason = mapStreamHttpError(error);
        if (reason) {
          if (!session.finished) hooks.fail(session, reason);
          return;
        }
        hooks.onDisconnect(session);
      },
      onClose() {
        if (!session.finished) hooks.onDisconnect(session);
      },
    };
  }

  return { dispatch, buildHandlers };
}
