/**
 * 数据集图片展示模块共享工具函数
 *
 * 适配新规范：
 * - 图片 type 字段：clear/hazy/trans/depth/segment
 * - haze_level 支持多种规范：light/medium/heavy、beta=X、空值
 * - Tab 切换改为"已标注/未标注"二分（已标注 = hazeLevel 非空）
 */

/** 图片类型标签映射（数据集 type 字段） */
export const IMAGE_TYPE_LABELS: Record<string, string> = {
  clear: "清晰图",
  hazy: "有雾图",
  trans: "透射图",
  depth: "深度图",
  segment: "分割图",
};

/** 图片类型样式类名映射（用于 ImageCard 角标样式） */
export const IMAGE_TYPE_BADGE_CLASS: Record<string, string> = {
  clear: "type-badge-clear",
  hazy: "type-badge-hazy",
  trans: "type-badge-trans",
  depth: "type-badge-depth",
  segment: "type-badge-segment",
};

/** 标注状态过滤类型 */
export type AnnotationFilter = "annotated" | "unannotated";

/** 标注状态过滤标签映射 */
export const ANNOTATION_FILTER_LABELS: Record<AnnotationFilter, string> = {
  annotated: "已标注",
  unannotated: "未标注",
};

/**
 * 格式化雾霾程度用于展示：
 * - light/medium/heavy → 轻度/中度/重度
 * - beta=X → β=X
 * - A=X,beta=Y → β=Y
 * - 其他 → 原值回显
 * - 空 → 空字符串（表示未标注）
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
 * 判断图片是否已标注（hazeLevel 非空视为已标注）
 */
export function isImageAnnotated(hazeLevel?: string): boolean {
  return Boolean(hazeLevel);
}

/**
 * 获取图片类型标签（兜底返回原值）
 */
export function getImageTypeLabel(type?: string): string {
  if (!type) return "";
  return IMAGE_TYPE_LABELS[type] || type;
}

/**
 * 获取图片类型角标样式类名（兜底返回空字符串）
 */
export function getImageTypeBadgeClass(type?: string): string {
  if (!type) return "";
  return IMAGE_TYPE_BADGE_CLASS[type] || "";
}
