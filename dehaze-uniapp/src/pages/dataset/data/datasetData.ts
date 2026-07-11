/**
 * 数据集管理模块 - 数据类型定义和Mock数据
 */

/** 图片类型 */
export type ImageType = "all" | "foggy" | "clear" | "annotated";

/** 展示模式 */
export type DisplayMode = "grid" | "waterfall";

/** 数据集模型 */
export interface Dataset {
  id: number;
  name: string;
  description: string;
  creator: string;
  thumbnail: string;
  total_images: number;
  foggy_count: number;
  clear_count: number;
  annotated_count: number;
  created_at: string;
  updated_at: string;
}

/** 图片模型 */
export interface DatasetImage {
  id: number;
  dataset_id: number;
  filename: string;
  image_url: string;
  image_type: "foggy" | "clear" | "annotated";
  width: number;
  height: number;
  file_size: number;
  tags: string;
  description: string;
  created_at: string;
}

/** 图片类型计数 */
export interface ImageTypeCounts {
  all: number;
  foggy: number;
  clear: number;
  annotated: number;
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
    name: "RESIDE数据集",
    description: "大规模真实场景图像去雾数据集，包含室内外多种场景",
    creator: "Li Boyi",
    thumbnail:
      "https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=400&h=400&fit=crop",
    total_images: 13990,
    foggy_count: 6995,
    clear_count: 6995,
    annotated_count: 0,
    created_at: "2024-01-15T10:30:00Z",
    updated_at: "2024-01-15T10:30:00Z",
  },
  {
    id: 2,
    name: "O-HAZE数据集",
    description: "户外真实雾霾图像数据集，包含45对有雾/无雾图像",
    creator: "Ancuti Codruta",
    thumbnail:
      "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
    total_images: 90,
    foggy_count: 45,
    clear_count: 45,
    annotated_count: 0,
    created_at: "2024-01-10T14:20:00Z",
    updated_at: "2024-01-10T14:20:00Z",
  },
  {
    id: 3,
    name: "I-HAZE数据集",
    description: "室内真实雾霾图像数据集，包含35对有雾/无雾图像",
    creator: "Ancuti Codruta",
    thumbnail:
      "https://images.unsplash.com/photo-1497366216548-37526070297c?w=400&h=400&fit=crop",
    total_images: 70,
    foggy_count: 35,
    clear_count: 35,
    annotated_count: 0,
    created_at: "2024-01-08T09:15:00Z",
    updated_at: "2024-01-08T09:15:00Z",
  },
  {
    id: 4,
    name: "Dense-Haze数据集",
    description: "密集雾霾场景数据集，专注于极端雾霾条件",
    creator: "Ancuti Codruta",
    thumbnail:
      "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=400&h=400&fit=crop",
    total_images: 110,
    foggy_count: 55,
    clear_count: 55,
    annotated_count: 0,
    created_at: "2024-01-05T16:45:00Z",
    updated_at: "2024-01-05T16:45:00Z",
  },
  {
    id: 5,
    name: "NH-HAZE数据集",
    description: "非均匀雾霾数据集，模拟真实世界的复杂雾霾分布",
    creator: "Ancuti Codruta",
    thumbnail:
      "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
    total_images: 110,
    foggy_count: 55,
    clear_count: 55,
    annotated_count: 0,
    created_at: "2024-01-03T11:30:00Z",
    updated_at: "2024-01-03T11:30:00Z",
  },
  {
    id: 6,
    name: "SOTS数据集",
    description: "合成雾霾数据集，包含室内外场景",
    creator: "Li Boyi",
    thumbnail:
      "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=400&h=400&fit=crop",
    total_images: 1000,
    foggy_count: 500,
    clear_count: 500,
    annotated_count: 0,
    created_at: "2024-01-01T08:00:00Z",
    updated_at: "2024-01-01T08:00:00Z",
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

  const imageTypes: Array<"foggy" | "clear" | "annotated"> = [
    "foggy",
    "clear",
    "annotated",
  ];

  for (let i = 0; i < count; i++) {
    const type = imageTypes[i % 3];
    const typeCount =
      type === "foggy"
        ? dataset.foggy_count
        : type === "clear"
          ? dataset.clear_count
          : dataset.annotated_count;

    if (typeCount === 0 && type === "annotated") continue;

    const aspectRatio = ASPECT_RATIOS[i % ASPECT_RATIOS.length];

    images.push({
      id: datasetId * 1000 + i,
      dataset_id: datasetId,
      filename: `${dataset.name.replace(/\s+/g, "_")}_${type}_${String(i + 1).padStart(4, "0")}.jpg`,
      image_url: SAMPLE_IMAGES[i % SAMPLE_IMAGES.length],
      image_type: type,
      width: aspectRatio.width,
      height: aspectRatio.height,
      file_size: Math.floor(Math.random() * 2000000) + 500000,
      tags: `${type},${dataset.name}`,
      description: `${dataset.name}中的${type === "foggy" ? "有雾" : type === "clear" ? "无雾" : "标注"}图像`,
      created_at: new Date(
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
  imageType: ImageType = "all",
  search = ""
): Promise<{ code: number; data: PaginatedResult<DatasetImage> }> {
  try {
    const { getDatasetItems } = await import("@/api/dataset");
    const result = await getDatasetItems(datasetId, {
      page,
      page_size: 20,
      image_type: imageType,
      search: search || undefined,
    });

    // 转换后端数据格式到前端格式
    return {
      code: 0,
      data: {
        list: result.list.map((item) => ({
          id: item.id,
          dataset_id: item.dataset_id,
          filename: item.filename,
          image_url: item.image_url,
          image_type: item.image_type,
          width: item.width,
          height: item.height,
          file_size: item.file_size,
          tags: item.tags || "",
          description: item.description || "",
          created_at: item.created_at,
        })),
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
    if (imageType !== "all") allImages = allImages.filter((img) => img.image_type === imageType);
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

/** 图片类型标签映射 */
export const IMAGE_TYPE_LABELS: Record<string, string> = {
  all: "全部",
  foggy: "有雾",
  clear: "无雾",
  annotated: "标注",
};

/** 图片类型颜色映射 */
export const IMAGE_TYPE_COLORS: Record<string, string> = {
  foggy: "#6b7280",
  clear: "#3b82f6",
  annotated: "#10b981",
};
