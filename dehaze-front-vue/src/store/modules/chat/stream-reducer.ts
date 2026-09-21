// 流式 SSE 事件 → 会话视图状态的纯归约器（reduce(state, event) -> { state, effects }）。
//
// 依据：把 buildHandlers 中"逐事件直接改 ref"的状态迁移收敛为确定性归约，从而可脱离 transport
// 单测。归约函数不读 Date.now / 随机 / 外部可变状态（时间由调用方以 now 注入），不触碰任何 ref、
// 定时器与请求；副作用（写 ref、rAF 批量写回、发请求、终态收尾）由外层按返回的 effects 解释执行。
//
// 落位说明：放在 store/modules/chat 而非 components/ai-chat/vm。归约对象是 store 会话视图状态
// （ThinkingState/ToolCallDraft 定义于本层 shared），并复用 ai-chat/vm 的思考归约纯函数；
// 若放到 vm/ 会造成 vm 反向依赖 store 类型，破坏"vm 为零依赖纯展示层"的分层约束。
import type {
  ContentBlockDeltaEvent,
  ContentBlockStartEvent,
  ContentBlockStopEvent,
  ErrorEvent,
  InterruptEvent,
  MessageEndEvent,
  MessageStartEvent,
  Plan,
  SuggestionsEvent,
  ThoughtEvent,
} from "dehaze-sdk-js";
import {
  toPlanVM,
  toPlanVMFromInterrupt,
} from "@/components/ai-chat/adapters/fromSdk";
import {
  appendThinkingDelta,
  closeThinkingSegment,
  openThinkingSegment,
} from "@/components/ai-chat/vm/thinking";
import { sortStepsByPosition } from "@/components/ai-chat/vm/steps";
import {
  THINKING_BLOCK_INDEX,
  type StreamViewState,
  type ThinkingState,
  type ToolCallDraft,
} from "./shared";

/** SSE 事件（按回调语义打标）；network-error/close 属 transport 生命周期，不入归约 */
export type StreamEvent =
  | { kind: "start"; data: MessageStartEvent }
  | { kind: "block-start"; data: ContentBlockStartEvent }
  | { kind: "delta"; data: ContentBlockDeltaEvent }
  | { kind: "block-stop"; data: ContentBlockStopEvent }
  | { kind: "thought"; data: ThoughtEvent }
  | { kind: "plan"; data: Plan }
  | { kind: "suggestions"; data: SuggestionsEvent }
  | { kind: "interrupt"; data: InterruptEvent }
  | { kind: "error"; data: ErrorEvent }
  | { kind: "end"; data: MessageEndEvent };

/**
 * 归约产出的副作用描述（由外层解释执行为写 ref / rAF / 终态收尾）。
 * 数组顺序即执行顺序，与既有逐事件处理顺序一致。
 */
export type StreamEffect =
  /** onStart：把服务端 messageId 绑定到占位消息（from 为变更前的本地 id，用于兜底查找） */
  | { type: "bind"; from: number }
  /** 写回当前 messageId 对应的 streamingMessageId ref */
  | { type: "streaming-id" }
  /** 恢复续流首个事件到达：清空中断点 */
  | { type: "clear-interrupts" }
  /** 立即写回思考态 ref（start/stop/闭合时，非 rAF 批量） */
  | { type: "writing" }
  | { type: "thoughts" }
  | { type: "tool-calls" }
  | { type: "plan" }
  /** push 中断并同步 interruptedMessageId/streamingMessageId */
  | { type: "interrupts" }
  | { type: "suggestions" }
  /** rAF 批量写回正文/思考缓冲 */
  | { type: "flush" }
  /** 终态收尾（清定时器、置 finished） */
  | { type: "finish" }
  | { type: "fail"; reason: string }
  | {
      type: "end";
      usage: MessageEndEvent["usage"];
      /** 消息终态：null 表示中断挂起（保持生成中，等待 resume / async 轮询） */
      status: 2 | 3 | 4 | null;
      /** 是否启动 async_wait 轮询 */
      pollAsync: boolean;
    };

export interface StreamReduction {
  state: StreamViewState;
  effects: StreamEffect[];
}

/** 取（或惰性创建）思考态；不写 ref，写回时机由 effects 驱动（保持 rAF 批量语义） */
function ensureThinking(state: StreamViewState, now: number): ThinkingState {
  return (
    state.thinking ?? {
      segments: [],
      startAt: now,
      endAt: null,
      streaming: false,
    }
  );
}

/** 把未落段的思考增量并入思考态（纯函数；空缓冲原样返回） */
function drainThinking(thinking: ThinkingState, buffer: string): ThinkingState {
  return buffer ? appendThinkingDelta(thinking, buffer) : thinking;
}

/** 契约断言：思考内容块固定 index=1（后端 SseEventConverter._THINKING_INDEX），偏离即显式暴露 */
function assertThinkingBlockIndex(index: number): void {
  if (index !== THINKING_BLOCK_INDEX) {
    throw new Error(
      `思考内容块 index 契约偏离：期望 ${THINKING_BLOCK_INDEX}，实际 ${index}`
    );
  }
}

/** 单个 SSE 事件的状态归约（确定性、无副作用；now 由调用方注入） */
export function reduceStreamState(
  state: StreamViewState,
  event: StreamEvent,
  now: number
): StreamReduction {
  switch (event.kind) {
    case "start": {
      const effects: StreamEffect[] = [
        { type: "bind", from: state.messageId },
        { type: "streaming-id" },
      ];
      let next: StreamViewState = {
        ...state,
        messageId: event.data.messageId,
        streamingMessageId: event.data.messageId,
      };
      if (state.clearInterruptOnStart) {
        next = {
          ...next,
          interrupts: [],
          interruptedMessageId: null,
          ...(next.plan
            ? { plan: { ...next.plan, awaitingApproval: false } }
            : {}),
        };
        // 顺序：先清中断点、再置计划（与既有处理一致）
        effects.push({ type: "clear-interrupts" }, { type: "plan" });
      }
      return { state: next, effects };
    }

    case "block-start": {
      if (event.data.type === "tool_use") {
        const toolBlocks = new Map(state.toolBlocks);
        toolBlocks.set(event.data.index, { args: "" });
        return { state: { ...state, toolBlocks }, effects: [] };
      }
      if (event.data.type === "thinking") {
        assertThinkingBlockIndex(event.data.index);
        const base = ensureThinking(state, now);
        return {
          state: { ...state, thinking: openThinkingSegment(base) },
          effects: [{ type: "writing" }],
        };
      }
      return { state, effects: [] };
    }

    case "delta": {
      const delta = event.data.delta;
      if (delta.type === "text_delta") {
        return {
          state: {
            ...state,
            textBuffer: state.textBuffer + (delta.text ?? ""),
          },
          effects: [{ type: "flush" }],
        };
      }
      if (delta.type === "thinking_delta") {
        // 逐 delta 仅累加缓冲（不触发响应式）；rAF 帧内一次性并入思考段
        return {
          state: {
            ...state,
            thinking: ensureThinking(state, now),
            thinkingBuffer: state.thinkingBuffer + (delta.thinking ?? ""),
          },
          effects: [{ type: "flush" }],
        };
      }
      if (delta.type === "input_json_delta") {
        const block = state.toolBlocks.get(event.data.index);
        if (block) {
          const nextBlock: ToolCallDraft = {
            args: block.args + (delta.partialJson ?? ""),
          };
          if (delta.name) nextBlock.name = delta.name;
          else if (block.name) nextBlock.name = block.name;
          const toolBlocks = new Map(state.toolBlocks);
          toolBlocks.set(event.data.index, nextBlock);
          return {
            state: { ...state, toolBlocks },
            effects: [{ type: "flush" }],
          };
        }
      }
      return { state, effects: [{ type: "flush" }] };
    }

    case "block-stop": {
      // 思考块固定 index=1：闭合末段并定格计时（多段思考由后续 start 重开）
      if (
        event.data.index === THINKING_BLOCK_INDEX &&
        state.thinking?.segments.some((seg) => !seg.closed)
      ) {
        const thinking = drainThinking(state.thinking, state.thinkingBuffer);
        return {
          state: {
            ...state,
            thinking: closeThinkingSegment(thinking, now),
            thinkingBuffer: "",
          },
          effects: [{ type: "writing" }],
        };
      }
      const block = state.toolBlocks.get(event.data.index);
      if (!block) return { state, effects: [] };
      const toolBlocks = new Map(state.toolBlocks);
      toolBlocks.delete(event.data.index);
      let parsed: unknown = block.args;
      try {
        parsed = JSON.parse(block.args);
      } catch {
        // 非法 JSON 保留原始字符串
      }
      return {
        state: {
          ...state,
          toolBlocks,
          toolCalls: [
            ...state.toolCalls,
            { name: block.name, arguments: parsed },
          ],
        },
        effects: [{ type: "tool-calls" }],
      };
    }

    case "thought": {
      const list = state.thoughts.slice();
      const index = list.findIndex(
        (item) => item.position === event.data.position
      );
      if (index >= 0) list[index] = event.data;
      else list.push(event.data);
      return {
        state: { ...state, thoughts: sortStepsByPosition(list) },
        effects: [{ type: "thoughts" }],
      };
    }

    case "plan":
      return {
        state: {
          ...state,
          plan: toPlanVM(event.data, {
            messageId: state.messageId,
            previous: state.plan,
          }),
        },
        effects: [{ type: "plan" }],
      };

    case "suggestions":
      return {
        state: {
          ...state,
          suggestions: event.data.questions.map((item) => item.question),
        },
        effects: [{ type: "suggestions" }],
      };

    case "interrupt": {
      const effects: StreamEffect[] = [{ type: "interrupts" }];
      let plan = state.plan;
      if (event.data.type === "plan_approve") {
        const interruptPlan = event.data.data?.plan;
        plan = interruptPlan
          ? toPlanVMFromInterrupt(interruptPlan, {
              messageId: state.messageId,
              previous: plan,
              awaitingApproval: true,
            })
          : plan
            ? { ...plan, awaitingApproval: true }
            : plan;
        effects.push({ type: "plan" });
      }
      // 流会随 message.end 关闭，标记收尾等待 onEnd（finish 最后执行）
      effects.push({ type: "finish" });
      return {
        state: {
          ...state,
          interrupts: [...state.interrupts, event.data],
          interruptedMessageId: state.messageId,
          streamingMessageId: null,
          plan,
        },
        effects,
      };
    }

    case "error":
      return {
        state: { ...state, streamingMessageId: null },
        effects: [
          { type: "fail", reason: event.data.message || "推理服务返回错误" },
        ],
      };

    case "end": {
      const held = state.interruptedMessageId === state.messageId;
      let status: 2 | 3 | 4 | null = null;
      if (!held) {
        const stopReason = event.data.stopReason;
        status =
          stopReason === "canceled"
            ? 4
            : stopReason === "error" || stopReason === "content_filter"
              ? 3
              : 2;
      }
      const pollAsync =
        held && state.interrupts.some((item) => item.type === "async_wait");
      return {
        state: { ...state, streamingMessageId: null },
        effects: [
          {
            type: "end",
            usage: event.data.usage,
            status,
            pollAsync,
          },
        ],
      };
    }
  }
}
