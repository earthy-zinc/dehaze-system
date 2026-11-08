// 中断子类型（confirmKind）的唯一校验与归一入口。
//
// 组件层（InterruptCard 路由渲染）与 store 层（buildResumeForm 构造载荷）共用同一份
// 合法值列表与判定规则：两处各维护一套，会出现"按钮可点但点击后被静默丢弃"的失效。
// 只依赖契约类型，不碰 Vue / store / 请求层。
import type { ConfirmKind, Interrupt } from "clarity-sdk-js";

/** 合法 confirm 子类型（与契约 ConfirmKind 对齐） */
export const KNOWN_CONFIRM_KINDS: readonly ConfirmKind[] = [
  "algorithm_recommend",
  "tool_permission",
  "dangerous_op",
  "write_conflict",
];

/** 运行时取值是否落在合法子类型内（契约外的值由后端漂移引入，不能靠类型断言兜住） */
export function isKnownConfirmKind(kind: unknown): kind is ConfirmKind {
  return (
    typeof kind === "string" &&
    (KNOWN_CONFIRM_KINDS as readonly string[]).includes(kind)
  );
}

/**
 * confirm 子类型状态：
 * - `ok`：合法子类型；
 * - `missing`：confirm 中断但契约未下发 confirmKind；
 * - `unknown`：confirm 中断下发了契约外的值；
 * - `not_confirm`：非 confirm 中断，本就没有子类型。
 *
 * `missing` 与 `unknown` 都是"协议不可用"，UI 一律按不可操作处理。
 */
export type ConfirmKindStatus = "ok" | "missing" | "unknown" | "not_confirm";

export function confirmKindStatus(interrupt: Interrupt): ConfirmKindStatus {
  if (interrupt.type !== "confirm") return "not_confirm";
  const raw: unknown = interrupt.confirmKind;
  if (raw === null || raw === undefined) return "missing";
  return isKnownConfirmKind(raw) ? "ok" : "unknown";
}

/** confirm 中断但子类型不可用（缺失或契约外取值） */
export function isConfirmKindInvalid(interrupt: Interrupt): boolean {
  const status = confirmKindStatus(interrupt);
  return status === "missing" || status === "unknown";
}

/**
 * 读取 confirm 中断子类型（直接取契约 Interrupt.confirmKind）。
 * 非 confirm、缺失、未知一律 undefined——调用方据此禁用操作，
 * 不把"协议缺字段"变成点击后的运行时异常。
 */
export function resolveConfirmKind(
  interrupt: Interrupt
): ConfirmKind | undefined {
  return confirmKindStatus(interrupt) === "ok"
    ? (interrupt.confirmKind as ConfirmKind)
    : undefined;
}
