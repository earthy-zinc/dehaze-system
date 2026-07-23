/**
 * 数据集管理模块 - 展示层数据
 *
 * SDK 类型直接从 dehaze-sdk-js 导入；视图模型 DatasetImageItem 从 @/api/dataset 导入。
 * 本文件仅保留：
 * - 展示模式等前端专用类型
 * - 格式化工具函数
 * - 标签映射常量
 */

// ==================== 类型 re-export ====================

export type { Dataset, DatasetStatistics } from "dehaze-sdk-js";
export type { DatasetImageItem } from "@/api/dataset";

// ==================== 展示层专用类型 ====================

/** 展示模式 */
export type DisplayMode = "grid" | "waterfall";

// ==================== 工具函数 ====================

/**
 * 格式化雾霾程度用于展示：
 * - light/medium/heavy → 轻度/中度/重度
 * - beta=X → β=X
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

// ==================== 标签映射常量 ====================

/** 图片类型标签映射（数据集 type 字段） */
export const IMAGE_TYPE_LABELS: Record<string, string> = {
  clear: "清晰图",
  hazy: "有雾图",
  trans: "透射图",
  depth: "深度图",
  segment: "分割图",
};
