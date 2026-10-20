/**
 * 图片类型枚举
 *
 * 数据集图片 type 字段（后端权威定义）：
 * - clear: 清晰图
 * - hazy: 有雾图
 * - trans: 透射图
 * - depth: 深度图
 * - segment: 分割图
 *
 * 注：PRED 用于去雾演示页面的预测结果展示，与数据集 type 字段独立。
 */
export const enum ImageTypeEnum {
  HAZE = "有雾图像",
  PRED = "预测图像",
  CLEAN = "清晰图像",
  TRANS = "透射图",
  DEPTH = "深度图",
  SEGMENT = "分割图",
}

/** type 字符串到枚举的映射表（数据集 type 字段） */
export const IMAGE_TYPE_LABELS: Record<string, string> = {
  clear: "清晰图",
  hazy: "有雾图",
  trans: "透射图",
  depth: "深度图",
  segment: "分割图",
};

/** type 字符串到 Tag 颜色的映射表（用于 Element Plus Tag 的 type 属性） */
export const IMAGE_TYPE_COLORS: Record<string, string> = {
  clear: "success",
  hazy: "warning",
  trans: "primary",
  depth: "info",
  segment: "info",
};

/**
 * 格式化雾霾程度用于展示：
 * - light/medium/heavy → 轻度/中度/重度
 * - beta=X → β=X
 * - A=X,beta=Y → β=Y
 * - 其他 → 原值回显
 * - 空 → 空字符串（表示未标注）
 */
export const formatHazeLevel = (level?: string): string => {
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
};
