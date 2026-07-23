/**
 * 文件相关工具函数
 */

/** 格式化文件大小（B/KB/MB），空值返回 '-' */
export function formatFileSize(bytes?: number): string {
  if (!bytes) return '-';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
