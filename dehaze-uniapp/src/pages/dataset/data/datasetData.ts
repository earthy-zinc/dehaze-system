/**
 * 数据集管理模块 - 数据类型定义
 *
 * 适配新规范：
 * - 图片 type 字段：clear/hazy/trans/depth/segment
 * - haze_level 支持多种规范：light/medium/heavy、beta=X、空值
 * - Tab 切换改为"已标注/未标注"二分（已标注 = hazeLevel 非空）
 * - 取消"清晰图必填/有雾图必填"硬性校验
 */

/** 标注状态过滤（Tab 二分） */
export type AnnotationFilter = "annotated" | "unannotated";

/** 图片类型（数据集 type 字段，后端权威定义） */
export type ImageType = "clear" | "hazy" | "trans" | "depth" | "segment";

/** 展示模式 */
export type DisplayMode = "grid" | "waterfall";

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

/** 数据集模型（字段与后端 DatasetVO 对齐） */
export interface Dataset {
  id: number;
  parentId: number;
  name: string;
  description: string;
  type: string;
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

/** 图片模型 */
export interface DatasetImage {
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

/** 标注状态计数 */
export interface AnnotationCounts {
  /** 全部图片数 */
  all: number;
  /** 已标注（haze_level 非空） */
  annotated: number;
  /** 未标注（haze_level 为空） */
  unannotated: number;
}

/** 分页结果 */
export interface PaginatedResult<T> {
  list: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

/**
 * 格式化日期
 */
export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (days === 0) return "今天";
  if (days === 1) return "昨天";
  if (days < 7) return `${days}天前`;

  return date.toLocaleDateString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

/**
 * 格式化文件大小
 */
export function formatFileSize(bytes: number): string {
  if (!bytes) return "-";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

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
export function isImageAnnotated(image: DatasetImage): boolean {
  return Boolean(image.hazeLevel);
}

/**
 * 获取数据集列表
 */
export async function fetchDatasets(
  page = 1,
  search = ""
): Promise<{ code: number; data: PaginatedResult<Dataset> }> {
  const { getDatasets } = await import("@/api/dataset");
  const result = await getDatasets(page, search);
  return { code: 0, data: result };
}

/**
 * 获取数据集详情
 */
export async function fetchDatasetDetail(
  datasetId: number
): Promise<{ code: number; data?: Dataset; message?: string }> {
  const { getDatasetDetail } = await import("@/api/dataset");
  const detail = await getDatasetDetail(datasetId);
  return { code: 0, data: detail };
}

/**
 * 获取数据集图片
 */
export async function fetchDatasetImages(
  datasetId: number,
  page = 1,
  annotationFilter: AnnotationFilter = "annotated",
  search = ""
): Promise<{ code: number; data: PaginatedResult<DatasetImage> }> {
  const { getDatasetItems } = await import("@/api/dataset");
  const result = await getDatasetItems(datasetId, {
    page,
    page_size: 20,
    annotation_filter: annotationFilter,
    search: search || undefined,
  });

  return {
    code: 0,
    data: {
      list: result.list as DatasetImage[],
      total: result.total,
      page: result.page,
      page_size: result.page_size,
      total_pages: result.total_pages,
    },
  };
}

/** 图片类型标签映射（数据集 type 字段） */
export const IMAGE_TYPE_LABELS: Record<string, string> = {
  clear: "清晰图",
  hazy: "有雾图",
  trans: "透射图",
  depth: "深度图",
  segment: "分割图",
};

/** 图片类型颜色映射（用于卡片角标背景色） */
export const IMAGE_TYPE_COLORS: Record<string, string> = {
  clear: "#3b82f6",
  hazy: "#6b7280",
  trans: "#0ea5e9",
  depth: "#8b5cf6",
  segment: "#06b6d4",
};

/** 标注状态过滤标签映射 */
export const ANNOTATION_FILTER_LABELS: Record<AnnotationFilter, string> = {
  annotated: "已标注",
  unannotated: "未标注",
};
