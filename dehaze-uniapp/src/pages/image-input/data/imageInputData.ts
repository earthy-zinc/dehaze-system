/**
 * 图像输入模块 - 数据类型定义和样例数据
 */

/** 数据集静态文件服务地址（来自 .env 的 VITE_DATASET_HOST） */
const DATASET_BASE_URL =
  import.meta.env.VITE_DATASET_HOST || "http://127.0.0.1:9000";

/** 输入方式 */
export type InputMethod = "upload" | "camera" | "sample" | "history";

/** 雾霾程度分类 */
export type FogLevel = "all" | "light" | "medium" | "heavy" | "special";

/** 难度等级 */
export type Difficulty = "简单" | "中等" | "困难";

/** 输入方式配置 */
export interface InputMethodConfig {
  method: InputMethod;
  icon: string;
  title: string;
  subtitle: string;
}

/** 样例图片 */
export interface SampleImage {
  id: number;
  name: string;
  url: string;
  /** 对应的无雾GT图片URL（用于指标评估参考） */
  cleanUrl?: string;
  category: FogLevel;
  difficulty: Difficulty;
  scene?: string;
  recommendAlgorithm?: string;
}

/** 图片数据 */
export interface ImageData {
  url: string;
  /** 后端文件 ID（上传后获得） */
  fileId?: number;
  /** 后端返回的 HTTP URL */
  remoteUrl?: string;
  /** 图片宽度（样例/上传场景有值，历史记录场景可能缺失） */
  width?: number;
  /** 图片高度 */
  height?: number;
  /** 文件大小（字节） */
  size?: number;
  name: string;
  sampleInfo?: SampleImage;
}

/** 分类Tab配置 */
export interface CategoryTab {
  key: FogLevel;
  label: string;
}

/** 输入方式配置列表 */
export const INPUT_METHODS: InputMethodConfig[] = [
  {
    method: "upload",
    icon: "photo",
    title: "上传图片",
    subtitle: "从相册选择",
  },
  {
    method: "camera",
    icon: "camera",
    title: "拍照",
    subtitle: "实时拍摄",
  },
  {
    method: "sample",
    icon: "grid",
    title: "样例图片",
    subtitle: "快速体验",
  },
  {
    method: "history",
    icon: "clock",
    title: "历史记录",
    subtitle: "最近处理",
  },
];

/** 分类Tab配置 */
export const CATEGORY_TABS: CategoryTab[] = [
  { key: "all", label: "全部" },
  { key: "light", label: "轻度雾霾" },
  { key: "medium", label: "中度雾霾" },
  { key: "heavy", label: "重度雾霾" },
  { key: "special", label: "特殊场景" },
];

/** NH-HAZE-2023 数据集基础路径 */
const NH_HAZE_HAZY = `${DATASET_BASE_URL}/datasets/NH-HAZE-2023/hazy`;
const NH_HAZE_CLEAN = `${DATASET_BASE_URL}/datasets/NH-HAZE-2023/clean`;

/** 构建样例图片数据 */
function makeSample(
  id: number,
  num: string,
  category: Exclude<FogLevel, "all">,
  difficulty: Difficulty,
  scene: string,
  recommendAlgorithm: string
): SampleImage {
  return {
    id,
    name: `${difficulty}雾霾-${scene}`,
    url: `${NH_HAZE_HAZY}/${num}.JPG`,
    cleanUrl: `${NH_HAZE_CLEAN}/${num}.JPG`,
    category,
    difficulty,
    scene,
    recommendAlgorithm,
  };
}

/** 样例图片库数据（使用 NH-HAZE-2023 真实雾图数据集） */
export const SAMPLE_IMAGES: Record<Exclude<FogLevel, "all">, SampleImage[]> = {
  light: [
    makeSample(1, "001", "light", "简单", "城市街道", "DCP"),
    makeSample(2, "002", "light", "简单", "建筑景观", "DCP"),
    makeSample(3, "003", "light", "简单", "室外场景", "AOD-Net"),
    makeSample(4, "004", "light", "简单", "街景", "DCP"),
    makeSample(5, "005", "light", "简单", "自然景观", "DCP"),
  ],
  medium: [
    makeSample(6, "006", "medium", "中等", "城市天际线", "DehazeNet"),
    makeSample(7, "007", "medium", "中等", "道路", "AOD-Net"),
    makeSample(8, "008", "medium", "中等", "远景", "DehazeNet"),
    makeSample(9, "009", "medium", "中等", "开阔地带", "FFA-Net"),
    makeSample(10, "010", "medium", "中等", "室外场景", "DehazeNet"),
  ],
  heavy: [
    makeSample(11, "011", "heavy", "困难", "城市中心", "FFA-Net"),
    makeSample(12, "012", "heavy", "困难", "街景", "GridDehazeNet"),
    makeSample(13, "013", "heavy", "困难", "远景", "FFA-Net"),
    makeSample(14, "014", "heavy", "困难", "开阔地带", "GridDehazeNet"),
    makeSample(15, "015", "heavy", "困难", "建筑群", "FFA-Net"),
  ],
  special: [
    makeSample(16, "016", "special", "困难", "特殊场景", "MSBDN"),
    makeSample(17, "017", "special", "困难", "特殊场景", "FFA-Net"),
    makeSample(18, "018", "special", "中等", "特殊场景", "DehazeNet"),
    makeSample(19, "019", "special", "简单", "特殊场景", "DCP"),
    makeSample(20, "020", "special", "中等", "特殊场景", "AOD-Net"),
  ],
};

/** 获取所有样例图片 */
export function getAllSampleImages(): SampleImage[] {
  return [
    ...SAMPLE_IMAGES.light,
    ...SAMPLE_IMAGES.medium,
    ...SAMPLE_IMAGES.heavy,
    ...SAMPLE_IMAGES.special,
  ];
}

/** 根据分类获取样例图片 */
export function getSampleImagesByCategory(category: FogLevel): SampleImage[] {
  if (category === "all") {
    return getAllSampleImages();
  }
  return SAMPLE_IMAGES[category] || [];
}

/** 获取随机样例图片 */
export function getRandomSampleImage(): SampleImage {
  const allImages = getAllSampleImages();
  const image = allImages[Math.floor(Math.random() * allImages.length)];
  if (!image) {
    throw new Error("No sample images available");
  }
  return image;
}

/** 难度颜色映射 */
export const DIFFICULTY_COLORS: Record<Difficulty, string> = {
  简单: "#10b981",
  中等: "#f59e0b",
  困难: "#ef4444",
};

/** 难度背景色映射 */
export const DIFFICULTY_BG_COLORS: Record<Difficulty, string> = {
  简单: "#d1fae5",
  中等: "#fef3c7",
  困难: "#fee2e2",
};

// ==================== 常量 ====================

/** 最大文件大小（20MB） */
export const MAX_FILE_SIZE = 20 * 1024 * 1024;

/** 压缩阈值（5MB） */
export const COMPRESS_THRESHOLD = 5 * 1024 * 1024;
