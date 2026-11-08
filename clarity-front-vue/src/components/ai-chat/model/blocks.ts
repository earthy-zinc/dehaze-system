// 消息块派生：把契约 Block[] 收敛为组件渲染所需的纯 UI 形状。
//
// 纯函数、零运行时依赖：仅 import type 契约类型，不触碰 store / SDK / Vue。
// 正文、思考分段、工具步骤、用量可见性全部按 blocks / usage 现算，组件 computed 负责缓存，
// 不产生第二份内容副本，也不做字段改名复制。
import type { Block, ToolStatus, Usage } from "clarity-sdk-js";
import type { ChatThoughtStepVM, ChatThinkingVM } from "../types";

/** 按 seq 升序（返回新数组，不修改入参） */
function sortBySeq<T extends { seq: number }>(blocks: readonly T[]): T[] {
  return blocks.slice().sort((a, b) => a.seq - b.seq);
}

/** 指定 kind 的块（按 seq 升序；blocks 缺失时按空数组处理） */
function blocksOfKind(
  blocks: readonly Block[] | null | undefined,
  kind: Block["kind"]
): Block[] {
  return sortBySeq((blocks ?? []).filter((block) => block.kind === kind));
}

/**
 * 正文：text 块按 seq 拼接。
 * 块间以空行衔接——工具调用会把正文切段，直接相连会把两段 Markdown 粘成一行。
 */
export function textOfBlocks(
  blocks: readonly Block[] | null | undefined
): string {
  return blocksOfKind(blocks, "text")
    .map((block) => block.text ?? "")
    .filter((text) => text !== "")
    .join("\n\n");
}

/**
 * 思考过程：thinking 块按 seq 分段，durationMs 决定闭合与计时。
 *
 * 计时口径：契约 Block 不带时间戳，只有收尾时的 durationMs。故以消息 createTime 作为
 * 计时起点、各思考块 durationMs 之和作为已思考时长；存在未收尾块时为"仍思考中"（endAt=null）。
 * 历史回放与流式终态用同一口径推导，保证刷新前后展示一致。
 */
export function thinkingOfBlocks(
  blocks: readonly Block[] | null | undefined,
  startAt: number
): ChatThinkingVM | null {
  const all = blocks ?? [];
  const firstThinkingIndex = all.findIndex(
    (block) => block.kind === "thinking"
  );
  if (firstThinkingIndex === -1) return null;
  // 首个思考块之前可能先跑了工具/正文：把它们的 durationMs 计入偏移，
  // 否则"思考中"计时会把工具耗时也算进去，收尾时再突然跳回真实时长。
  const leadMs = all
    .slice(0, firstThinkingIndex)
    .reduce((sum, block) => sum + (block.durationMs ?? 0), 0);
  const thinkingStart = startAt + leadMs;
  const thinking = blocksOfKind(all, "thinking");
  const segments = thinking.map((block) => ({
    text: block.text ?? "",
    closed: block.durationMs != null,
  }));
  const streaming = segments.some((segment) => !segment.closed);
  const durationMs = thinking.reduce(
    (sum, block) => sum + (block.durationMs ?? 0),
    0
  );
  return {
    segments,
    startAt: thinkingStart,
    endAt: streaming ? null : thinkingStart + durationMs,
    streaming,
  };
}

/** tool_use 块 → 工具步骤（position 取块 seq，工具输出作为 observation，状态原样透传） */
export function stepsOfBlocks(
  blocks: readonly Block[] | null | undefined
): ChatThoughtStepVM[] {
  return blocksOfKind(blocks, "tool_use").map((block) => {
    const tool = block.tool;
    const observation = tool?.output ?? tool?.summary ?? undefined;
    const status: ToolStatus = tool?.status ?? "running";
    return {
      position: block.seq,
      status,
      ...(tool?.name ? { tool: tool.name } : {}),
      ...(tool?.input != null ? { toolInput: tool.input } : {}),
      ...(observation !== undefined ? { observation } : {}),
      ...(tool?.error ? { error: tool.error } : {}),
      ...(tool?.latencyMs != null ? { latencyMs: tool.latencyMs } : {}),
    };
  });
}

/** 用量是否值得展示：全 0 且无子智能体粒度用量时为 false（消费端不渲染空壳） */
export function hasUsage(usage: Usage | null | undefined): boolean {
  if (!usage) return false;
  if ((usage.subAgents ?? []).length > 0) return true;
  return (
    usage.inputTokens !== 0 ||
    usage.outputTokens !== 0 ||
    usage.cachedInputTokens !== 0 ||
    usage.credits !== 0
  );
}
