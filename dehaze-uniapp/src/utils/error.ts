/**
 * 从未知错误中提取用户可读的消息。
 *
 * 统一处理项目内所有抛错形态：
 * - `ApiError` / `Error` 实例（含 `message`）
 * - 字符串
 * - 其他带 `message` 字段的对象（兜底）
 *
 * 注意：API 层仅抛出 `Error` / `ApiError`，历史代码中对 `msg` 字段的判断属于无效兜底，已废弃。
 */
export function getErrorMessage(error: unknown, fallback = "操作失败"): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message || fallback;
  if (error && typeof error === "object" && "message" in error) {
    const msg = (error as { message?: unknown }).message;
    if (typeof msg === "string" && msg) return msg;
  }
  return fallback;
}
