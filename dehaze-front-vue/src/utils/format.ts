/**
 * 日期时间格式化
 * @param date 日期字符串、时间戳或 Date 对象
 * @param format 格式模板，默认 "yyyy-MM-dd HH:mm:ss"，支持 yy/MM/dd/HH/mm/ss 占位符
 * @returns 格式化后的日期字符串
 */
export function formatDateTime(
  date: string | number | Date,
  format: string = "yyyy-MM-dd HH:mm:ss"
): string {
  const d = new Date(date);
  if (isNaN(d.getTime())) return "";

  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  const hours = String(d.getHours()).padStart(2, "0");
  const minutes = String(d.getMinutes()).padStart(2, "0");
  const seconds = String(d.getSeconds()).padStart(2, "0");

  return format
    .replace("yyyy", String(year))
    .replace("yy", String(year).slice(-2))
    .replace("MM", month)
    .replace("dd", day)
    .replace("HH", hours)
    .replace("mm", minutes)
    .replace("ss", seconds);
}

/**
 * 文件大小格式化
 * @param bytes 字节数
 * @returns 格式化后的大小字符串（如 "1.23 MB"）
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";

  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / Math.pow(1024, exponent);

  return `${value.toFixed(2)} ${units[exponent]}`;
}

/**
 * 数字千分位格式化
 * @param num 数字
 * @returns 千分位格式化字符串（如 "1,234,567"）
 */
export function formatNumber(num: number): string {
  return num.toLocaleString("en-US");
}

/**
 * 百分比格式化
 * @param value 值（如 0.856）
 * @param decimals 小数位数，默认 1
 * @returns 百分比字符串（如 "85.6%"）
 */
export function formatPercent(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * 秒数转为可读时间
 * @param seconds 秒数
 * @returns 可读时间字符串（如 "1h 23m 45s" 或 "45s"）
 */
export function formatDuration(seconds: number): string {
  const absSec = Math.abs(seconds);
  const hours = Math.floor(absSec / 3600);
  const minutes = Math.floor((absSec % 3600) / 60);
  const secs = absSec % 60;

  const parts: string[] = [];
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  parts.push(`${secs}s`);

  const result = parts.join(" ");
  return seconds < 0 ? `-${result}` : result;
}

/**
 * 文本截断
 * @param text 原文本
 * @param maxLength 最大长度
 * @param suffix 截断后缀，默认 "..."
 * @returns 截断后的文本
 */
export function truncateText(
  text: string,
  maxLength: number,
  suffix: string = "..."
): string {
  if (!text || text.length <= maxLength) return text;
  return text.slice(0, maxLength - suffix.length) + suffix;
}
