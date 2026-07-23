/**
 * 数据集管理 API
 *
 * 直接使用 dehaze-sdk-js 的 DatasetAPI + DatasetItemAPI。
 * 数据项（DatasetItemVO）包含清晰图和有雾图配对，
 * 展示层通过 DatasetImageItem 视图模型展平为单图列表。
 */

import {
  DatasetAPI,
  DatasetItemAPI,
} from "dehaze-sdk-js";
import type {
  Dataset,
  DatasetItemVO,
  DatasetQuery,
  DatasetAddForm,
  DatasetUpdateForm,
  DatasetItemQuery,
  DatasetOption,
  ImageUrlVO,
  BatchDeleteForm,
} from "dehaze-sdk-js";

export type {
  Dataset,
  DatasetStatistics,
  DatasetItemVO,
  DatasetQuery,
  DatasetAddForm,
  DatasetUpdateForm,
  DatasetItemQuery,
  DatasetItemCreateForm,
  DatasetItemUpdateForm,
  DatasetOption,
  ImageUrlVO,
  ItemFileUpdateForm,
  BatchDeleteForm,
  BatchDeleteResultVO,
  BatchOperationResultVO,
} from "dehaze-sdk-js";

// ==================== 视图模型（展示层） ====================

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
function toDatasetImageItem(
  img: ImageUrlVO,
  itemId: number
): DatasetImageItem {
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

// ==================== 数据集 API 方法 ====================

/** 获取数据集分页列表 */
export function getDatasets(page = 1, search = "") {
  return DatasetAPI.getList({
    pageNum: page,
    pageSize: 10,
    keyword: search || undefined,
  });
}

/** 获取数据集详情 */
export function getDatasetDetail(id: number) {
  return DatasetAPI.getDatasetInfoById(id);
}

/**
 * 获取数据集数据项列表（返回原始 DatasetItemVO 分页）
 *
 * 展平为图片列表请使用 flattenDatasetItems。
 */
export function getDatasetItems(
  datasetId: number,
  query?: Partial<DatasetItemQuery>
) {
  return DatasetItemAPI.getList({
    datasetId,
    pageNum: query?.pageNum || 1,
    pageSize: query?.pageSize || 20,
    keyword: query?.keyword,
    sceneType: query?.sceneType,
    hazeLevel: query?.hazeLevel,
  });
}

/** 新增数据集 */
export function createDataset(data: DatasetAddForm) {
  return DatasetAPI.add(data);
}

/** 修改数据集 */
export function updateDataset(id: number, data: DatasetUpdateForm) {
  return DatasetAPI.update(id, data);
}

/** 删除数据集 */
export function deleteDataset(id: number) {
  return DatasetAPI.deleteById(id);
}

/** 批量删除数据集 */
export function batchDeleteDatasets(ids: number[]) {
  return DatasetAPI.batchDelete({ ids });
}

/** 获取数据集下拉选项 */
export function getDatasetOptions() {
  return DatasetAPI.getOptions();
}
