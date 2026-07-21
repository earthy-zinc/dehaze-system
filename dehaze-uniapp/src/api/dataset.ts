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

/** 数据集统计信息（对应后端 DatasetStatistics） */
export interface DatasetStatistics {
  itemCount: number;
  fileCount: number;
  totalSize: number;
  annotatedCount: number;
  unannotatedCount: number;
  sceneDistribution: Record<string, number>;
  hazeDistribution: Record<string, number>;
  formatDistribution: Record<string, number>;
}

export interface Dataset {
  id: number;
  parentId: number;
  type: string;
  name: string;
  description: string;
  path: string;
  hasChildren: boolean;
  children: Dataset[];
  status: number;
  statistics: DatasetStatistics;
  /** 图片总数（用于列表展示） */
  total: number;
  createTime: string;
  updateTime: string;
}

/** 标注状态过滤（Tab 二分） */
export type AnnotationFilter = "annotated" | "unannotated";

/** 图片类型（数据集 type 字段） */
export type ImageType = "clear" | "hazy" | "trans" | "depth" | "segment";

export interface DatasetItem {
  id: number;
  datasetId: number;
  filename: string;
  imageUrl: string;
  /** 图片类型：clear/hazy/trans/depth/segment */
  type: ImageType;
  /** 雾霾程度：light/medium/heavy、beta=0.5、A=0.8,beta=0.2 等，可为空 */
  hazeLevel?: string;
  width: number;
  height: number;
  fileSize: number;
  tags: string;
  description: string;
  createTime: string;
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
