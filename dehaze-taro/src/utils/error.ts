/**
 * 共享错误处理工具
 */

/**
 * 从未知类型的错误中提取消息字符串
 * 兜底返回 fallback（默认"操作失败"）
 */
export function getErrorMessage(
  err: unknown,
  fallback = "操作失败"
): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  if (err && typeof err === "object" && "message" in err) {
    const msg = (err as { message: unknown }).message;
    if (typeof msg === "string") return msg;
  }
  return fallback;
}
