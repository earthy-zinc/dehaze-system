/**
 * 数据集管理 API
 *
 * 适配新规范：
 * - 图片 type 字段：clear/hazy/trans/depth/segment
 * - haze_level 支持多种规范：light/medium/heavy、beta=X、空值
 * - Tab 切换改为"已标注/未标注"二分（已标注 = haze_level 非空）
 *
 * API 路径：
 * - GET    /datasets          数据集列表
 * - GET    /datasets/{id}     数据集详情
 * - POST   /datasets          新增数据集
 * - PUT    /datasets/{id}     修改数据集
 * - DELETE /datasets/{id}     删除数据集
 * - GET    /datasets/options  数据集下拉选项
 */

import { get, post, put, del } from "./request";

// ==================== 类型定义 ====================

export interface Dataset {
  id: number;
  name: string;
  description: string;
  creator: string;
  thumbnail: string;
  total_images: number;
  /** 已标注图片数（haze_level 非空） */
  annotated_count: number;
  /** 未标注图片数 */
  unannotated_count: number;
  created_at: string;
  updated_at: string;
}

/** 标注状态过滤（Tab 二分） */
export type AnnotationFilter = "annotated" | "unannotated";

/** 图片类型（数据集 type 字段） */
export type ImageType = "clear" | "hazy" | "trans" | "depth" | "segment";

export interface DatasetItem {
  id: number;
  dataset_id: number;
  filename: string;
  image_url: string;
  /** 图片类型：clear/hazy/trans/depth/segment */
  type: ImageType;
  /** 雾霾程度：light/medium/heavy、beta=0.5、A=0.8,beta=0.2 等，可为空 */
  haze_level?: string;
  width: number;
  height: number;
  file_size: number;
  tags: string;
  description: string;
  created_at: string;
}

export interface DatasetItemQuery {
  page?: number;
  page_size?: number;
  /** 标注状态过滤：annotated/unannotated */
  annotation_filter?: AnnotationFilter;
  search?: string;
}

export interface PaginatedResult<T> {
  list: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// ==================== API 方法 ====================

/** 获取数据集列表 */
export async function getDatasets(page = 1, search = "") {
  return get<PaginatedResult<Dataset>>("/datasets", {
    data: { page, page_size: 10, search: search || undefined } as Record<string, unknown>,
  });
}

/** 获取数据集详情 */
export async function getDatasetDetail(id: number) {
  return get<Dataset>(`/datasets/${id}`);
}

/** 获取数据集数据项列表（图片） */
export async function getDatasetItems(
  datasetId: number,
  query: DatasetItemQuery = {}
): Promise<PaginatedResult<DatasetItem>> {
  const params: Record<string, unknown> = {
    page: query.page || 1,
    page_size: query.page_size || 20,
  };
  if (query.annotation_filter) {
    params.annotation_filter = query.annotation_filter;
  }
  if (query.search) {
    params.search = query.search;
  }
  return get<PaginatedResult<DatasetItem>>(`/datasets/${datasetId}/items`, { data: params });
}

/** 新增数据集 */
export async function createDataset(data: Partial<Dataset>) {
  return post<Dataset>("/datasets", data as unknown as Record<string, unknown>);
}

/** 修改数据集 */
export async function updateDataset(id: number, data: Partial<Dataset>) {
  return put(`/datasets/${id}`, data as unknown as Record<string, unknown>);
}

/** 删除数据集 */
export async function deleteDataset(id: number) {
  return del(`/datasets/${id}`);
}

/** 批量删除数据集 */
export async function batchDeleteDatasets(ids: number[]) {
  return del("/datasets", {
    data: { ids },
  });
}

/** 获取数据集下拉选项 */
export async function getDatasetOptions() {
  return get<{ value: number; label: string }[]>("/datasets/options");
}
