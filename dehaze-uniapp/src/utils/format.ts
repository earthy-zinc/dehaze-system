/**
 * 通用格式化工具函数
 */

/**
 * 格式化文件大小
 * @param bytes 字节数（接受 number 或数字字符串）
 * @returns 形如 "1.23 KB" / "3.45 MB" / "-（空值时）"
 */
export function formatFileSize(bytes: number | string): string {
  const n = typeof bytes === "string" ? parseInt(bytes, 10) : bytes;
  if (!n || isNaN(n)) return "-";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(2)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * 格式化为相对时间
 * - <1分钟 → "刚刚"
 * - <1小时 → "N分钟前"
 * - <1天 → "N小时前"
 * - <2天 → "昨天"
 * - 否则 → "MM-DD HH:mm"
 * @param timestamp ISO 字符串或时间戳
 */
export function formatRelativeTime(timestamp: string | number): string {
  if (!timestamp) return "-";
  const date = new Date(timestamp);
  if (isNaN(date.getTime())) return "-";
  const diff = Date.now() - date.getTime();

  if (diff < 60000) return "刚刚";
  if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
  if (diff < 172800000) return "昨天";

  return date.toLocaleDateString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * 金额分转元（接口金额单位统一为分，页面以元展示）
 * @param fen 金额（分）
 * @returns 形如 "100.00" 的元金额字符串
 */
export function fenToYuan(fen?: number): string {
  return ((fen ?? 0) / 100).toFixed(2);
}

/**
 * 金额元转分（表单以元录入，接口金额单位统一为分）
 * @param yuan 金额（元）
 * @returns 四舍五入后的整数分
 */
export function yuanToFen(yuan: number | string): number {
  return Math.round(Number(yuan) * 100);
}
