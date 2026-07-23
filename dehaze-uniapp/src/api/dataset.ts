/**
 * 数据集展示层视图模型
 *
 * 数据集 / 数据项 CRUD 直接使用 dehaze-sdk-js 的 DatasetAPI + DatasetItemAPI。
 * 本文件仅维护展示层视图模型：将 DatasetItemVO 展平为单图列表，
 * 供 ImageGrid / ImageCard 等组件使用。
 */

import type { DatasetItemVO, ImageUrlVO } from "dehaze-sdk-js";

/**
 * 数据集图片展示项（视图模型）
 *
 * 将 DatasetItemVO 中的 clearImage/hazyImages 展平为单条图片记录，
 * 供 ImageGrid / ImageCard 等组件使用。
 */
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
 * 将 DatasetItemVO 列表展平为 DatasetImageItem 列表
 *
 * 每个 DatasetItemVO 包含一张清晰图（可选）和多张有雾图（可选），
 * 展平后每张图片成为独立的展示项。
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
