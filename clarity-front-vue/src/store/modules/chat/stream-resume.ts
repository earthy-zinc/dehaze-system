// 中断恢复载荷归一：把调用方意图按中断子类型映射为 resume 请求。
//
// 纯函数，不读 store、不发请求；未知 confirmKind 返回 null（调用方据此不发请求）。
import type { Interrupt, MessageResume } from "clarity-sdk-js";
import { resolveConfirmKind } from "./shared";

/**
 * 按中断子类型把调用方意图归一为 resume 载荷：
 * - plan_approve：透传计划干预 planEdit
 * - confirm(algorithm_recommend)：{confirm, params:{modelId}}（modelId 取调用方选择，缺省回退推荐值）
 * - confirm(tool_permission/dangerous_op/write_conflict)：{confirm}
 * - 其余（quota/async_wait）：透传调用方载荷
 * 未知 confirmKind 返回 null（不静默兜底，也不把未知子类型发给后端）。
 */
export function buildResumeForm(
  pending: Interrupt | undefined,
  input: MessageResume
): MessageResume | null {
  if (!pending) return { ...input };
  if (pending.type === "plan_approve") {
    return input.planEdit != null ? { planEdit: input.planEdit } : {};
  }
  if (pending.type !== "confirm") return { ...input };
  const kind = resolveConfirmKind(pending);
  // 协议缺 confirmKind：UI 已禁用操作；这里再兜一层
  if (kind === undefined) return null;
  const confirm = input.confirm ?? true;
  if (kind === "algorithm_recommend") {
    const selected = input.params?.modelId;
    const modelId =
      typeof selected === "string"
        ? selected
        : (pending.recommendation?.modelId ?? undefined);
    return {
      confirm,
      ...(modelId !== undefined ? { params: { modelId } } : {}),
    };
  }
  return { confirm };
}
