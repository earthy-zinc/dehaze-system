// 思考过程归约：纯函数、无副作用（入参不改写，返回新状态），零外部依赖。
import type { ChatThinkingVM, ChatThoughtStepVM } from "../types";

/** 是否仍在流式思考：存在未闭合段 */
function isStreaming(segments: ChatThinkingVM["segments"]): boolean {
  return segments.some((segment) => !segment.closed);
}

/**
 * 新开一段思考。
 * 末段未关闭时幂等忽略（断线全量重放会重复下发 start）；末段已关闭或无段时追加新段。
 */
export function openThinkingSegment(state: ChatThinkingVM): ChatThinkingVM {
  const last = state.segments[state.segments.length - 1];
  if (last && !last.closed) return state;
  const segments = [...state.segments, { text: "", closed: false }];
  return { ...state, segments, streaming: true };
}

/**
 * 追加思考增量。
 * 空增量无副作用；无可写段（无段或末段已闭合）时开新段，兼容无 start 的残留流。
 */
export function appendThinkingDelta(
  state: ChatThinkingVM,
  delta: string
): ChatThinkingVM {
  if (!delta) return state;
  const segments = state.segments.slice();
  const last = segments[segments.length - 1];
  if (!last || last.closed) {
    segments.push({ text: delta, closed: false });
  } else {
    segments[segments.length - 1] = { ...last, text: last.text + delta };
  }
  return { ...state, segments, streaming: true };
}

/**
 * 闭合末段并定格计时。
 * 无可闭合段时幂等忽略（不写 endAt），避免多段思考时误关闭已完成状态。
 */
export function closeThinkingSegment(
  state: ChatThinkingVM,
  now: number
): ChatThinkingVM {
  const last = state.segments[state.segments.length - 1];
  if (!last || last.closed) return state;
  const segments = state.segments.slice();
  segments[segments.length - 1] = { ...last, closed: true };
  return { ...state, segments, endAt: now, streaming: isStreaming(segments) };
}

/** 流终态收尾：闭合全部未关闭段并定格计时（已有 endAt 保持不变） */
export function finalizeThinking(
  state: ChatThinkingVM,
  now: number
): ChatThinkingVM {
  const segments = state.segments.map((segment) =>
    segment.closed ? segment : { ...segment, closed: true }
  );
  return { ...state, segments, endAt: state.endAt ?? now, streaming: false };
}

/**
 * 由推理步骤合成历史思考态（无 SSE 流式态时的回退）。
 * 仅取纯思考步骤（tool 为空且 thought 非空）为已闭合段；无内容返回 null。
 */
export function buildThinkingFromThoughts(
  thoughts: ChatThoughtStepVM[]
): ChatThinkingVM | null {
  const segments = thoughts
    .filter(
      (step): step is ChatThoughtStepVM & { thought: string } =>
        !step.tool && !!step.thought
    )
    .map((step) => ({ text: step.thought, closed: true }));
  if (segments.length === 0) return null;
  return { segments, startAt: 0, endAt: null, streaming: false };
}
