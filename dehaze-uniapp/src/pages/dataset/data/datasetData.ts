/**
 * 数据集管理模块 - 数据类型定义和Mock数据
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

/** Mock数据集 */
export const MOCK_DATASETS: Dataset[] = [
  {
    id: 1,
    parentId: 0,
    name: "RESIDE数据集",
    description: "大规模真实场景图像去雾数据集，包含室内外多种场景",
    type: "图像去雾",
    path: "RESIDE",
    hasChildren: false,
    children: [],
    status: 1,
    statistics: {
      itemCount: 13990,
      fileCount: 13990,
      totalSize: 0,
      annotatedCount: 6995,
      unannotatedCount: 6995,
      sceneDistribution: {},
      hazeDistribution: {},
      formatDistribution: {},
    },
    total: 13990,
    createTime: "2024-01-15T10:30:00",
    updateTime: "2024-01-15T10:30:00",
  },
  {
    id: 2,
    parentId: 0,
    name: "O-HAZE数据集",
    description: "户外真实雾霾图像数据集，包含45对有雾/无雾图像",
    type: "图像去雾",
    path: "O-HAZE",
    hasChildren: false,
    children: [],
    status: 1,
    statistics: {
      itemCount: 90,
      fileCount: 90,
      totalSize: 0,
      annotatedCount: 45,
      unannotatedCount: 45,
      sceneDistribution: {},
      hazeDistribution: {},
      formatDistribution: {},
    },
    total: 90,
    createTime: "2024-01-10T14:20:00",
    updateTime: "2024-01-10T14:20:00",
  },
  {
    id: 3,
    parentId: 0,
    name: "I-HAZE数据集",
    description: "室内真实雾霾图像数据集，包含35对有雾/无雾图像",
    type: "图像去雾",
    path: "I-HAZE",
    hasChildren: false,
    children: [],
    status: 1,
    statistics: {
      itemCount: 70,
      fileCount: 70,
      totalSize: 0,
      annotatedCount: 35,
      unannotatedCount: 35,
      sceneDistribution: {},
      hazeDistribution: {},
      formatDistribution: {},
    },
    total: 70,
    createTime: "2024-01-08T09:15:00",
    updateTime: "2024-01-08T09:15:00",
  },
  {
    id: 4,
    parentId: 0,
    name: "Dense-Haze数据集",
    description: "密集雾霾场景数据集，专注于极端雾霾条件",
    type: "图像去雾",
    path: "Dense-Haze",
    hasChildren: false,
    children: [],
    status: 1,
    statistics: {
      itemCount: 110,
      fileCount: 110,
      totalSize: 0,
      annotatedCount: 55,
      unannotatedCount: 55,
      sceneDistribution: {},
      hazeDistribution: {},
      formatDistribution: {},
    },
    total: 110,
    createTime: "2024-01-05T16:45:00",
    updateTime: "2024-01-05T16:45:00",
  },
  {
    id: 5,
    parentId: 0,
    name: "NH-HAZE数据集",
    description: "非均匀雾霾数据集，模拟真实世界的复杂雾霾分布",
    type: "图像去雾",
    path: "NH-HAZE",
    hasChildren: false,
    children: [],
    status: 1,
    statistics: {
      itemCount: 110,
      fileCount: 110,
      totalSize: 0,
      annotatedCount: 55,
      unannotatedCount: 55,
      sceneDistribution: {},
      hazeDistribution: {},
      formatDistribution: {},
    },
    total: 110,
    createTime: "2024-01-03T11:30:00",
    updateTime: "2024-01-03T11:30:00",
  },
  {
    id: 6,
    parentId: 0,
    name: "SOTS数据集",
    description: "合成雾霾数据集，包含室内外场景",
    type: "图像去雾",
    path: "SOTS",
    hasChildren: false,
    children: [],
    status: 1,
    statistics: {
      itemCount: 1000,
      fileCount: 1000,
      totalSize: 0,
      annotatedCount: 500,
      unannotatedCount: 500,
      sceneDistribution: {},
      hazeDistribution: {},
      formatDistribution: {},
    },
    total: 1000,
    createTime: "2024-01-01T08:00:00",
    updateTime: "2024-01-01T08:00:00",
  },
];

/** 示例图片URL列表 */
const SAMPLE_IMAGES = [
  "https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1475924156734-496f6cac6ec1?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1418065460487-3e41a6c84dc5?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1511593358241-7eea1f3c84e5?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1542435503-956c469947f6?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=800&h=600&fit=crop",
  "https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=800&h=500&fit=crop",
];

/** 随机图片宽高比（用于瀑布流） */
const ASPECT_RATIOS = [
  { width: 1920, height: 1080 },
  { width: 1920, height: 1280 },
  { width: 1600, height: 1200 },
  { width: 1200, height: 1600 },
  { width: 1080, height: 1920 },
  { width: 1600, height: 900 },
];

/** 图片类型候选集合（生成 Mock 数据使用） */
const MOCK_TYPES: ImageType[] = ["clear", "hazy", "trans", "depth", "segment"];

/** 雾霾程度候选值（生成 Mock 数据使用） */
const MOCK_HAZE_LEVELS = ["light", "medium", "heavy", "beta=0.5", "beta=0.8", ""];

/**
 * 生成Mock图片数据
 */
export function generateMockImages(
  datasetId: number,
  count: number
): DatasetImage[] {
  const images: DatasetImage[] = [];
  const dataset = MOCK_DATASETS.find((d) => d.id === datasetId);
  if (!dataset) return images;

  for (let i = 0; i < count; i++) {
    const type = MOCK_TYPES[i % MOCK_TYPES.length]!;
    const hazeLevel = MOCK_HAZE_LEVELS[i % MOCK_HAZE_LEVELS.length]!;
    const aspectRatio = ASPECT_RATIOS[i % ASPECT_RATIOS.length]!;
    const sampleImage = SAMPLE_IMAGES[i % SAMPLE_IMAGES.length]!;

    images.push({
      id: datasetId * 1000 + i,
      datasetId,
      filename: `${dataset.name.replace(/\s+/g, "_")}_${type}_${String(i + 1).padStart(4, "0")}.jpg`,
      imageUrl: sampleImage,
      type,
      hazeLevel: hazeLevel || undefined,
      width: aspectRatio.width,
      height: aspectRatio.height,
      fileSize: Math.floor(Math.random() * 2000000) + 500000,
      tags: `${type},${dataset.name}`,
      description: `${dataset.name}中的${IMAGE_TYPE_LABELS[type] || type}图像`,
      createTime: new Date(
        Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000
      ).toISOString(),
    });
  }

  return images;
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
 * 获取数据集列表（优先调用后端，失败时使用 Mock 降级）
 */
export async function fetchDatasets(
  page = 1,
  search = ""
): Promise<{ code: number; data: PaginatedResult<Dataset> }> {
  try {
    const { getDatasets } = await import("@/api/dataset");
    const result = await getDatasets(page, search);
    return { code: 0, data: result };
  } catch (error) {
    console.warn("[Dataset] 后端不可用，使用 Mock 数据:", error);

    // Mock 降级
    await new Promise((resolve) => setTimeout(resolve, 300));
    let filtered = [...MOCK_DATASETS];
    if (search) {
      const kw = search.toLowerCase();
      filtered = filtered.filter((d) => d.name.toLowerCase().includes(kw) || d.description?.toLowerCase().includes(kw));
    }
    const pageSize = 10;
    const start = (page - 1) * pageSize;
    return {
      code: 0,
      data: {
        list: filtered.slice(start, start + pageSize),
        total: filtered.length,
        page,
        page_size: pageSize,
        total_pages: Math.ceil(filtered.length / pageSize),
      },
    };
  }
}

/**
 * 获取数据集详情（优先调用后端，失败时使用 Mock 降级）
 */
export async function fetchDatasetDetail(
  datasetId: number
): Promise<{ code: number; data?: Dataset; message?: string }> {
  try {
    const { getDatasetDetail } = await import("@/api/dataset");
    const detail = await getDatasetDetail(datasetId);
    return { code: 0, data: detail };
  } catch {
    // Mock 降级
    await new Promise((resolve) => setTimeout(resolve, 200));
    const dataset = MOCK_DATASETS.find((d) => d.id === datasetId);
    return dataset ? { code: 0, data: dataset } : { code: 404, message: "数据集不存在" };
  }
}

/**
 * 获取数据集图片（优先调用后端，失败时使用 Mock 降级）
 */
export async function fetchDatasetImages(
  datasetId: number,
  page = 1,
  annotationFilter: AnnotationFilter = "annotated",
  search = ""
): Promise<{ code: number; data: PaginatedResult<DatasetImage> }> {
  try {
    const { getDatasetItems } = await import("@/api/dataset");
    const result = await getDatasetItems(datasetId, {
      page,
      page_size: 20,
      annotation_filter: annotationFilter,
      search: search || undefined,
    });

    // 后端与前端统一使用 camelCase，无需转换
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
  } catch {
    // Mock 降级
    await new Promise((resolve) => setTimeout(resolve, 400));
    let allImages = generateMockImages(datasetId, 60);
    // 按标注状态过滤
    allImages = allImages.filter((img) => {
      return annotationFilter === "annotated"
        ? isImageAnnotated(img)
        : !isImageAnnotated(img);
    });
    if (search) {
      const kw = search.toLowerCase();
      allImages = allImages.filter((img) => img.filename.toLowerCase().includes(kw) || img.tags?.toLowerCase().includes(kw));
    }
    const pageSize = 20;
    const start = (page - 1) * pageSize;
    return {
      code: 0,
      data: {
        list: allImages.slice(start, start + pageSize),
        total: allImages.length,
        page,
        page_size: pageSize,
        total_pages: Math.ceil(allImages.length / pageSize),
      },
    };
  }
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
