import dayjs from "dayjs";

export function formatDateTime(
  date: string | number | Date,
  format: string = "YYYY-MM-DD HH:mm:ss"
): string {
  return dayjs(date).format(format);
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
}

export function formatNumber(num: number): string {
  return num.toLocaleString();
}

export function formatPercent(value: number, decimals: number = 1): string {
  return (value * 100).toFixed(decimals) + "%";
}

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.round(seconds % 60);

  if (h > 0) {
    return `${h}h ${m}m ${s}s`;
  }
  if (m > 0) {
    return `${m}m ${s}s`;
  }
  return `${s}s`;
}

export function truncateText(
  text: string,
  maxLength: number,
  suffix: string = "..."
): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + suffix;
}

/**
 * 金额分转元：后端金额单位统一为分，页面展示以元为单位
 */
export function fenToYuan(fen?: number): string {
  return ((fen ?? 0) / 100).toFixed(2);
}

/**
 * 金额元转分：表单以元录入，接口金额单位统一为分
 */
export function yuanToFen(yuan: number | string): number {
  return Math.round(Number(yuan) * 100);
}
