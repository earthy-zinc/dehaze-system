// 中断卡的纯计算层：计划编辑列表、planEdit 构造、过期判定与标题文案。
//
// 纯函数、零运行时依赖：仅 import type 契约类型与本地 VM 类型，不碰 Vue / store / SDK 运行时。
// 组件只持有 ref、渲染与 emit，"怎么算"全部收敛在此，可脱离 DOM 单测。
import type { ConfirmKind, Interrupt, PlanTask } from "clarity-sdk-js";
import type { ChatPlanEditVM } from "../types";

/** 计划编辑行：isNew 表示本次新增、尚未有后端 id 的任务 */
export interface EditablePlanTask {
  id?: string;
  description: string;
  isNew: boolean;
}

/** 契约计划 → 可编辑行（后端任务一律视为已有，isNew=false） */
export function editablePlanOf(
  plan: readonly PlanTask[] | null | undefined
): EditablePlanTask[] {
  return (plan ?? []).map((task) => ({
    id: task.id,
    description: task.description ?? "",
    isNew: false,
  }));
}

/** 后端任务的 id 序列（无 id 的脏数据跳过，不能进 remove/reorder） */
export function planIdsOf(
  plan: readonly PlanTask[] | null | undefined
): string[] {
  return (plan ?? []).map((task) => task.id).filter((id): id is string => !!id);
}

/**
 * 计划内容签名：只有 type / expiresAt / plan 变化才重建可编辑列表。
 * VM 投影会随消息其它字段（产物、反馈）生成新对象，按引用重置会把用户正在编辑的任务清空。
 */
export function planSignatureOf(interrupt: Interrupt): string {
  return JSON.stringify({
    type: interrupt.type,
    expiresAt: interrupt.expiresAt,
    plan: interrupt.plan ?? [],
  });
}

/** 至少有一行可提交内容（有 id 或描述非空）才允许批准 */
export function canApprovePlan(tasks: readonly EditablePlanTask[]): boolean {
  return tasks.some((task) => task.description.trim() || task.id);
}

/** 是否已存在待新增行（后端 plan_edit.add 仅接受单对象，新增收敛为一条） */
export function hasNewPlanTask(tasks: readonly EditablePlanTask[]): boolean {
  return tasks.some((task) => task.isNew);
}

/** 移除第 index 行（返回新数组，不修改入参） */
export function removePlanTaskAt(
  tasks: readonly EditablePlanTask[],
  index: number
): EditablePlanTask[] {
  const next = tasks.slice();
  next.splice(index, 1);
  return next;
}

/** 上移 / 下移（delta 为 -1 / 1；越界时原样返回） */
export function movePlanTask(
  tasks: readonly EditablePlanTask[],
  index: number,
  delta: number
): EditablePlanTask[] {
  const target = index + delta;
  if (target < 0 || target >= tasks.length) return tasks.slice();
  const next = tasks.slice();
  const [task] = next.splice(index, 1);
  next.splice(target, 0, task);
  return next;
}

/** 追加一条空的新增行（已存在待新增行时原样返回） */
export function appendPlanTask(
  tasks: readonly EditablePlanTask[]
): EditablePlanTask[] {
  if (hasNewPlanTask(tasks)) return tasks.slice();
  return [...tasks, { description: "", isNew: true }];
}

/**
 * 由当前编辑行与原始 id 序列构造 planEdit：
 * - remove：原始有、当前没有的 id；
 * - reorder：任务集合未增减但顺序变化（含删除时集合变化，故只在等长时判定）；
 * - add：至多一条非空新增行。
 */
export function buildPlanEdit(
  tasks: readonly EditablePlanTask[],
  originalIds: readonly string[]
): ChatPlanEditVM {
  const currentIds = tasks
    .filter((task) => task.id && !task.isNew)
    .map((task) => task.id as string);
  const remove = originalIds.filter((id) => !currentIds.includes(id));
  const newTask = tasks.find((task) => task.isNew && task.description.trim());
  const orderChanged =
    currentIds.length === originalIds.length &&
    currentIds.some((id, index) => id !== originalIds[index]);

  const planEdit: ChatPlanEditVM = {};
  if (remove.length) planEdit.remove = remove;
  if (orderChanged) planEdit.reorder = currentIds;
  if (newTask)
    planEdit.add = { description: newTask.description.trim(), dependsOn: [] };
  return planEdit;
}

/** planEdit 是否为空（无改动时批准只需 confirm=true） */
export function isPlanEditEmpty(planEdit: ChatPlanEditVM): boolean {
  return !planEdit.remove && !planEdit.reorder && !planEdit.add;
}

/** 解析 expiresAt；缺失或非法时间返回 null */
export function parseExpiresAt(raw: string | null | undefined): number | null {
  if (!raw) return null;
  const parsed = Date.parse(raw);
  return Number.isNaN(parsed) ? null : parsed;
}

/** 是否已过期：无 expiresAt 视为不过期 */
export function isExpiredAt(expiresAtMs: number | null, now: number): boolean {
  return expiresAtMs !== null && now >= expiresAtMs;
}

/** confirm 子类型的动作文案与正文兜底（新增子类型只加一行，不复制模板） */
export interface ConfirmActionCopy {
  /** 接受按钮文案 */
  accept: string;
  acceptType: "primary" | "danger";
  /** 拒绝按钮文案 */
  reject: string;
  /** 正文兜底：后端未给 detail / reason 时使用 */
  fallback: string;
}

const CONFIRM_ACTIONS: Record<ConfirmKind, ConfirmActionCopy> = {
  algorithm_recommend: {
    accept: "采纳推荐",
    acceptType: "primary",
    reject: "拒绝",
    fallback: "需要你的确认",
  },
  tool_permission: {
    accept: "允许",
    acceptType: "primary",
    reject: "拒绝",
    fallback: "需要授权调用工具",
  },
  dangerous_op: {
    accept: "确认执行",
    acceptType: "danger",
    reject: "取消",
    fallback: "该操作存在风险",
  },
  write_conflict: {
    accept: "覆盖写入",
    acceptType: "primary",
    reject: "取消",
    fallback: "检测到写入冲突，需选择处理方式",
  },
};

/** 取 confirm 子类型的动作文案（kind 必须已通过 interruptKind 校验） */
export function confirmActionOf(kind: ConfirmKind): ConfirmActionCopy {
  return CONFIRM_ACTIONS[kind];
}

/** 卡片标题：按中断类型 + 已知子类型给出固定文案 */
export function interruptTitle(
  interrupt: Interrupt,
  kind?: ConfirmKind
): string {
  switch (interrupt.type) {
    case "plan_approve":
      return "计划待确认";
    case "quota":
      return "配额不足";
    case "async_wait":
      return "异步任务处理中";
    case "confirm":
      switch (kind) {
        case "algorithm_recommend":
          return "需要你的确认";
        case "tool_permission":
          return "工具授权确认";
        case "dangerous_op":
          return "危险操作确认";
        case "write_conflict":
          return "写入冲突";
        default:
          return "需要你的确认";
      }
    default:
      return "推理已暂停";
  }
}
