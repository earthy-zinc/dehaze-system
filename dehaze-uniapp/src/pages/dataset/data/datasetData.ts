/**
 * 数据集管理模块 - 展示层数据
 *
 * SDK 类型直接从 dehaze-sdk-js 导入。本文件仅保留：
 * - 展示层视图模型与转换函数（DatasetImageItem / flattenDatasetItems）
 * - 展示模式等前端专用类型
 * - 格式化工具函数
 * - 标签映射常量
 */

import type { DatasetItemVO, ImageUrlVO } from "dehaze-sdk-js";

// ==================== 展示层视图模型 ====================

/** 数据集图片展示项：将 DatasetItemVO 展平为单图记录，供 ImageGrid / ImageCard 使用 */
export interface DatasetImageItem {
  /** 图片文件 ID */
  id: number;
  /** 所属数据项 ID */
  itemId: number;
  /** 文件名 */
  filename: string;
  /** 图片访问 URL */
  imageUrl: string;
  /** 缩略图 URL */
  thumbnailUrl?: string;
  /** 图片类型：clear/hazy/trans/depth/segment */
  type: string;
  /** 雾霾程度 */
  hazeLevel?: string;
  /** 图片宽度 */
  width: number;
  /** 图片高度 */
  height: number;
  /** 文件大小（字节） */
  fileSize: number;
  /** 描述 */
  description?: string;
  /** 创建时间 */
  createTime?: string;
}

/**
 * 将 DatasetItemVO 列表展平为单图列表
 * 每个 DatasetItemVO 含一张清晰图（可选）与多张有雾图（可选），展平后每张图独立成项
 */
export function flattenDatasetItems(
  items: DatasetItemVO[]
): DatasetImageItem[] {
  const result: DatasetImageItem[] = [];
  for (const item of items) {
    if (item.clearImage) {
      result.push(toDatasetImageItem(item.clearImage, item.id));
    }
    if (item.hazyImages) {
      for (const hazy of item.hazyImages) {
        result.push(toDatasetImageItem(hazy, item.id));
      }
    }
  }
  return result;
}

/** 将 ImageUrlVO 转换为 DatasetImageItem */
function toDatasetImageItem(img: ImageUrlVO, itemId: number): DatasetImageItem {
  return {
    id: img.id,
    itemId,
    filename: img.fileName || "",
    imageUrl: img.url,
    thumbnailUrl: img.thumbnailUrl,
    type: img.type,
    hazeLevel: img.hazeLevel,
    width: img.width || 0,
    height: img.height || 0,
    fileSize: img.sizeBytes || 0,
    description: img.description,
    createTime: typeof img.createTime === "string" ? img.createTime : undefined,
  };
}

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
