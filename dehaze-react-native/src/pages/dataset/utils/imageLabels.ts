/**
 * 数据集图片标签工具函数
 *
 * 抽取自 ImageViewer / ImageCard 中重复定义的函数。
 */

export { formatFileSize } from '@/utils/file';

/** 图片类型 → 中文标签 */
export function getTypeLabel(type?: string): string {
  switch (type) {
    case 'clear':
      return '清晰图';
    case 'hazy':
      return '有雾图';
    default:
      return type || '图片';
  }
}

/** Badge 变体（与 components/Badge 的 variant 对齐） */
export type BadgeVariant = 'primary' | 'foggy' | 'secondary';

export function getBadgeVariant(type?: string): BadgeVariant {
  switch (type) {
    case 'clear':
      return 'primary';
    case 'hazy':
      return 'foggy';
    default:
      return 'secondary';
  }
}

/** 雾霾程度 → 中文标签 */
export function getHazeLevelLabel(level?: string): string {
  switch (level) {
    case 'light':
      return '轻度';
    case 'medium':
      return '中度';
    case 'heavy':
      return '重度';
    default:
      return '';
  }
}
