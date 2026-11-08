// 块级写入：块是消息内容的唯一载体，增删只在实体数组上做。
//
// delta 定位依赖按数组引用缓存的 blockId 下标表（WeakMap），使逐帧增量保持 O(1)；
// 数组被整体替换（block.stop 后的实体刷新、历史重载）时按新引用惰性重建。
import type { Block, Message } from "clarity-sdk-js";

const blockIndexes = new WeakMap<Block[], Map<string, number>>();

/** 取（或惰性构建）块下标；数组被整体替换后自动按新引用重建 */
function indexOfBlock(blocks: Block[], blockId: string): number {
  let index = blockIndexes.get(blocks);
  if (!index) {
    index = new Map<string, number>();
    blocks.forEach((block, at) => index!.set(block.id, at));
    blockIndexes.set(blocks, index);
  }
  return index.get(blockId) ?? -1;
}

/** 追加块（block.start）：同 id 已存在时原位替换，避免重放重复插入 */
export function appendBlock(message: Message, block: Block): void {
  const blocks = message.blocks;
  const at = indexOfBlock(blocks, block.id);
  if (at >= 0) {
    blocks[at] = block;
    return;
  }
  blocks.push(block);
  blockIndexes.get(blocks)?.set(block.id, blocks.length - 1);
}

/**
 * 原位追加块增量（block.delta）：按 blockId 定位后直接改 text，O(1)。
 * tool_use 块的参数增量不入库——工具入参以 block.stop 下发的完整 tool.payload 为准，
 * 避免把非契约语义（半截 JSON）写进实体，保证流式终态与历史读模型同形。
 */
export function appendBlockDelta(
  message: Message,
  blockId: string,
  delta: string
): boolean {
  const blocks = message.blocks;
  const at = indexOfBlock(blocks, blockId);
  if (at < 0) return false;
  const block = blocks[at];
  if (block.kind === "tool_use") return false;
  block.text = (block.text ?? "") + delta;
  return true;
}

/** 替换块（block.stop）：服务端下发的完整块（含耗时、工具结果）整体覆盖 */
export function replaceBlock(message: Message, block: Block): void {
  const blocks = message.blocks;
  const at = indexOfBlock(blocks, block.id);
  if (at >= 0) {
    blocks[at] = block;
    return;
  }
  blocks.push(block);
  blockIndexes.get(blocks)?.set(block.id, blocks.length - 1);
}
