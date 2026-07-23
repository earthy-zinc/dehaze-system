/**
 * 时间格式化工具函数
 */

/**
 * 将毫秒数格式化为人类可读的耗时字符串
 *
 * - <1000ms：原样显示毫秒（如 `123ms`）
 * - <60s：秒（如 `8s`）
 * - >=60s：分秒（如 `1m30s`）
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  const remSec = sec % 60;
  return `${min}m${remSec}s`;
}
