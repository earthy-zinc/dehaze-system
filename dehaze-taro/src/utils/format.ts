/**
 * 共享格式化工具函数
 */

/**
 * 格式化文件大小（字节 → B/KB/MB）
 */
export function formatFileSize(bytes?: number): string {
  if (!bytes || bytes <= 0) return "-";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/**
 * 格式化耗时（毫秒 → ms/s）
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return ms + " ms";
  return (ms / 1000).toFixed(2) + " s";
}

/**
 * 格式化雾霾程度用于展示：
 * - light/medium/heavy → 轻度/中度/重度
 * - beta=X → β=X
 * - 其他 → 原值回显
 * - 空 → 空字符串
 */
export function formatHazeLevel(level?: string): string {
  if (!level) return "";
  const preset: Record<string, string> = {
    light: "轻度",
    medium: "中度",
    heavy: "重度",
  };
  if (preset[level]) return preset[level];
  const betaMatch = level.match(/beta=([\d.]+)/i);
  if (betaMatch) return `β=${betaMatch[1]}`;
  return level;
}

/**
 * 格式化日期时间（YYYY-MM-DD HH:mm 或 YYYY-MM-DD HH:mm:ss）
 */
export function formatDateTime(
  date?: string | Date,
  withSeconds = false
): string {
  if (!date) return "-";
  const d = typeof date === "string" ? new Date(date) : date;
  if (Number.isNaN(d.getTime())) return String(date);
  const pad = (n: number) => String(n).padStart(2, "0");
  const base = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(
    d.getDate()
  )} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return withSeconds ? `${base}:${pad(d.getSeconds())}` : base;
}

/**
 * 数字格式化（千分位分隔）
 */
export function formatNumber(num: number): string {
  return num.toLocaleString("zh-CN");
}

/**
 * 数值转百分比
 * @param value 0~1 之间的小数
 * @param decimals 小数位数，默认 1
 */
export function formatPercent(value: number, decimals = 1): string {
  return (value * 100).toFixed(decimals) + "%";
}

/**
 * 截断文本并追加后缀
 * @param text 原文本
 * @param maxLength 最大长度
 * @param suffix 截断后追加的后缀，默认 "..."
 */
export function truncateText(
  text: string,
  maxLength: number,
  suffix = "..."
): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + suffix;
}
