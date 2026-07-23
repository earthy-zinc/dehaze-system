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
  if (isNaN(d.getTime())) return String(date);
  const pad = (n: number) => String(n).padStart(2, "0");
  const base = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(
    d.getDate()
  )} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return withSeconds ? `${base}:${pad(d.getSeconds())}` : base;
}
