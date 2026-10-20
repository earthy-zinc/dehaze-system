/**
 * URL 处理工具函数
 */

/**
 * 从 URL 中提取文件名（兼容 http 与 file:// 地址）
 *
 * 例如：`https://example.com/path/image.jpg?token=xxx` → `image.jpg`
 */
export function extractFilename(url?: string, fallback = '历史图片'): string {
  if (!url) return fallback;
  const path = url.split('?')[0];
  const segments = path.split('/');
  return segments[segments.length - 1] || fallback;
}
