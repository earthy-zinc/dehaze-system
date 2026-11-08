// SSE 事件 → 消息实体的纯归约器（不含响应式、定时器与请求）。
//
// 契约（config/schema/ai-conversation.schema.json）里一条消息只有一个实体，blocks 是
// 唯一内容源：本归约器把每个事件直接写到 messages[messageId] 上——
//   message.start  → 用服务端 user/assistant 实体替换本地占位（rekey）
//   block.start    → 追加块
//   block.delta    → 按 blockId 原位追加（O(1)，不按 position 覆盖）
//   block.stop     → 用完整块替换
//   plan.update    → 替换 plan
//   interrupt      → 写 pendingInterrupt
//   usage          → 写 usage
//   suggestions    → 写 suggestions
//   message.end    → 写 status / error / usage
// 终态唯一：message.end 是唯一终态来源，error 事件只记录原因、不自行收尾。
import type {
  BlockDeltaEvent,
  BlockStartEvent,
  BlockStopEvent,
  ErrorEvent,
  InterruptEvent,
  Message,
  MessageEndEvent,
  MessageStartEvent,
  MessageStatus,
  PlanUpdateEvent,
  SuggestionsEvent,
  UsageEvent,
} from "clarity-sdk-js";
import { appendBlock, appendBlockDelta, replaceBlock } from "./blocks";

/** 归约器直接读写的流式上下文（StreamSession 的前半部分） */
export interface StreamReducerState {
  /** 当前流目标消息 id（start 前为本地负数占位） */
  messageId: number;
  /** 本轮新建的本地用户消息 id（send/edit），start 后置空 */
  localUserMessageId: number | null;
  /** 已收到 message.start（resume 续流没有 start） */
  started: boolean;
  /** 流内出现过中断 */
  interrupted: boolean;
  /** resume 续流：首个事件到达即清 pendingInterrupt */
  clearInterruptOnEvent: boolean;
}

export interface StreamReducerContext {
  /** 消息实体表（就地更新） */
  messages: Record<number, Message>;
  /** 消息顺序（id 数组，就地更新） */
  order: number[];
  state: StreamReducerState;
}

/** SSE 事件（按语义打标）；network-error/close 属 transport 生命周期，不入归约 */
export type StreamEvent =
  | { kind: "start"; data: MessageStartEvent }
  | { kind: "block-start"; data: BlockStartEvent }
  | { kind: "block-delta"; data: BlockDeltaEvent }
  | { kind: "block-stop"; data: BlockStopEvent }
  | { kind: "plan"; data: PlanUpdateEvent }
  | { kind: "interrupt"; data: InterruptEvent }
  | { kind: "usage"; data: UsageEvent }
  | { kind: "suggestions"; data: SuggestionsEvent }
  | { kind: "end"; data: MessageEndEvent }
  | { kind: "error"; data: ErrorEvent };

/**
 * 归约产出的副作用（由 stream slice 解释执行为状态/定时器/收尾）。
 * 消息内容本身已在归约中写入实体，这里只表达 transport 生命周期。
 */
export type StreamEffect =
  /** 流进入产出态 */
  | { type: "status"; status: "running" | "interrupted" }
  /** 本轮终态（message.end） */
  | { type: "end"; status: MessageStatus; error?: string | null }
  /** 网络/超时失败（消息已被置失败） */
  | { type: "failed"; reason: string };

/** 事件指向的消息 id（error 允许不带 id，回退到当前流目标） */
function targetId(
  ctx: StreamReducerContext,
  messageId?: number | null
): number {
  return messageId ?? ctx.state.messageId;
}

/**
 * message.start：服务端 user/assistant 实体就位。
 * 本地占位（负数 id）在顺序表中原位替换，保证用户消息与助手回复不因 rekey 跳动；
 * 已存在的服务端消息（重生成沿用的 user 消息）保持原位，不重复插入。
 *
 * 乐观链清理：重新生成/重试/编辑后本地会同时存在旧回复与新回复（同一 parent），
 * 而服务端激活链只保留新回复。这里做两件事：
 *   1) 删掉同 parent、同会话、id 不同的本地 assistant（旧回复）；
 *   2) 重新生成链中间回复时，新链在 parent 之后不再包含本地已显示的后续轮次，
 *      id 大于 parent 的其它消息（下游 user/assistant）一并裁掉，避免刷新前渲染幽灵轮次。
 * 只动本地视图（下次 fetchMessages 以服务端链为准）；正常发送的 parent 是链尾，
 * 不会命中任何既有消息。
 */
function applyStart(ctx: StreamReducerContext, data: MessageStartEvent): void {
  const { messages, order, state } = ctx;
  const user = data.userMessage;
  const assistant = data.assistantMessage;
  const oldAssistantId = state.messageId;
  const oldUserId = state.localUserMessageId;
  const parentId = assistant.parentMessageId ?? null;

  const stale = new Set<number>();
  for (const message of Object.values(messages)) {
    if (message.role !== "assistant" || message.id === assistant.id) continue;
    if (message.conversationId !== assistant.conversationId) continue;
    if ((message.parentMessageId ?? null) !== parentId) continue;
    stale.add(message.id);
  }
  // 下游轮次：id 大于 parent 的消息都是 parent 之后创建的，已不在服务端新链上。
  // parent 缺失/非服务端 id（本地占位）时不做推断，只走上面的同 parent 清理。
  const downstream = new Set<number>();
  if (parentId !== null && parentId > 0) {
    for (const message of Object.values(messages)) {
      if (message.id === user.id || message.id === assistant.id) continue;
      if (message.conversationId !== assistant.conversationId) continue;
      if (message.id <= parentId) continue;
      downstream.add(message.id);
    }
  }
  for (const id of stale) delete messages[id];
  for (const id of downstream) delete messages[id];

  if (
    oldAssistantId !== assistant.id &&
    messages[oldAssistantId]?.id === oldAssistantId
  ) {
    delete messages[oldAssistantId];
  }
  if (
    oldUserId !== null &&
    oldUserId !== user.id &&
    messages[oldUserId]?.id === oldUserId
  ) {
    delete messages[oldUserId];
  }
  messages[user.id] = user;
  messages[assistant.id] = assistant;

  const next: number[] = [];
  for (const id of order) {
    if (id === oldUserId) {
      if (!next.includes(user.id)) next.push(user.id);
      continue;
    }
    if (id === oldAssistantId) {
      if (!next.includes(assistant.id)) next.push(assistant.id);
      continue;
    }
    next.push(id);
  }
  if (!next.includes(user.id)) {
    const at = next.indexOf(assistant.id);
    next.splice(at < 0 ? next.length : at, 0, user.id);
  }
  if (!next.includes(assistant.id)) next.push(assistant.id);
  // 旧兄弟/下游轮次连同其在顺序表中的位置一起移除（新 assistant 已由上面的替换/追加就位）
  const kept = next.filter(
    (id) =>
      id === user.id ||
      id === assistant.id ||
      (!stale.has(id) && !downstream.has(id))
  );
  order.splice(0, order.length, ...kept);

  if (state.clearInterruptOnEvent) assistant.pendingInterrupt = null;
  state.messageId = assistant.id;
  state.localUserMessageId = null;
  state.started = true;
}

/** 单个 SSE 事件 → 实体写入 + 副作用（确定性；now 由调用方注入，本函数不读时钟） */
export function reduceStreamEvent(
  ctx: StreamReducerContext,
  event: StreamEvent
): StreamEffect[] {
  const { messages, state } = ctx;
  switch (event.kind) {
    case "start": {
      applyStart(ctx, event.data);
      return [{ type: "status", status: "running" }];
    }

    case "block-start": {
      const message = messages[targetId(ctx, event.data.messageId)];
      if (!message) return [];
      appendBlock(message, event.data.block);
      return [{ type: "status", status: "running" }];
    }

    case "block-delta": {
      const message = messages[targetId(ctx, event.data.messageId)];
      if (!message) return [];
      // 增量只写块内 text；定位由 blockId 下标表完成，O(1)
      appendBlockDelta(message, event.data.blockId, event.data.delta);
      return [];
    }

    case "block-stop": {
      const message = messages[targetId(ctx, event.data.messageId)];
      if (!message) return [];
      replaceBlock(message, event.data.block);
      return [];
    }

    case "plan": {
      const message = messages[targetId(ctx, event.data.messageId)];
      if (!message) return [];
      message.plan = event.data.plan;
      return [];
    }

    case "interrupt": {
      const message = messages[targetId(ctx, event.data.messageId)];
      if (!message) return [];
      message.pendingInterrupt = event.data.interrupt;
      state.interrupted = true;
      return [{ type: "status", status: "interrupted" }];
    }

    case "usage": {
      const message = messages[targetId(ctx, event.data.messageId)];
      if (!message) return [];
      message.usage = event.data.usage;
      return [];
    }

    case "suggestions": {
      const message = messages[targetId(ctx, event.data.messageId)];
      if (!message) return [];
      message.suggestions = event.data.suggestions;
      return [];
    }

    case "end": {
      const message = messages[targetId(ctx, event.data.messageId)];
      if (!message) {
        return [
          { type: "end", status: event.data.status, error: event.data.error },
        ];
      }
      message.status = event.data.status;
      if (event.data.error !== undefined) message.error = event.data.error;
      if (event.data.usage) message.usage = event.data.usage;
      if (event.data.status !== "awaiting_input")
        message.pendingInterrupt = null;
      return [
        { type: "end", status: event.data.status, error: event.data.error },
      ];
    }

    case "error": {
      const message = messages[targetId(ctx, event.data.messageId)];
      if (!message) return [];
      // error 不是终态：只记录原因，终态由随后的 message.end 表达
      message.error = event.data.message;
      return [];
    }
  }
}
