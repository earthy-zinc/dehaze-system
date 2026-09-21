// 推理步骤归约：纯函数、无副作用，零外部依赖。
import type { ChatThoughtStepVM } from "../types";

/** 按 position 升序稳定排序（重复 position 保持原相对顺序）；不修改入参 */
export function sortStepsByPosition<T extends { position: number }>(
  steps: readonly T[]
): T[] {
  return steps.slice().sort((a, b) => a.position - b.position);
}

/** 推理链只承载工具步骤：过滤掉纯思考（tool 为空）步骤；不修改入参 */
export function filterToolSteps(
  steps: ChatThoughtStepVM[]
): ChatThoughtStepVM[] {
  return steps.filter((step) => !!step.tool);
}
