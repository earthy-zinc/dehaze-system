/**
 * 图像输入模块 - 数据类型定义和Mock数据
 */

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
  category: FogLevel;
  difficulty: Difficulty;
  scene?: string;
  recommendAlgorithm?: string;
}

/** 图片数据 */
export interface ImageData {
  url: string;
  file?: File;
  width: number;
  height: number;
  size: number;
  name: string;
  sampleInfo?: SampleImage;
}

/** 历史记录 */
export interface HistoryRecord {
  id: number;
  thumbnail: string;
  resultThumbnail?: string;
  fileName: string;
  algorithm?: string;
  timestamp: string;
  status: "success" | "failed";
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

/** 样例图片库数据 */
export const SAMPLE_IMAGES: Record<Exclude<FogLevel, "all">, SampleImage[]> = {
  light: [
    {
      id: 1,
      name: "轻度雾霾-城市街道",
      url: "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=800",
      category: "light",
      difficulty: "简单",
      scene: "城市",
      recommendAlgorithm: "DCP",
    },
    {
      id: 2,
      name: "轻度雾霾-公园景观",
      url: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
      category: "light",
      difficulty: "简单",
      scene: "公园",
      recommendAlgorithm: "DCP",
    },
    {
      id: 3,
      name: "轻度雾霾-建筑物",
      url: "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
      category: "light",
      difficulty: "简单",
      scene: "建筑",
      recommendAlgorithm: "AOD-Net",
    },
    {
      id: 4,
      name: "轻度雾霾-山景",
      url: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
      category: "light",
      difficulty: "简单",
      scene: "山景",
      recommendAlgorithm: "DCP",
    },
    {
      id: 5,
      name: "轻度雾霾-湖泊",
      url: "https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=800",
      category: "light",
      difficulty: "简单",
      scene: "湖泊",
      recommendAlgorithm: "DCP",
    },
  ],
  medium: [
    {
      id: 6,
      name: "中度雾霾-城市天际线",
      url: "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800",
      category: "medium",
      difficulty: "中等",
      scene: "城市",
      recommendAlgorithm: "DehazeNet",
    },
    {
      id: 7,
      name: "中度雾霾-道路",
      url: "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=800",
      category: "medium",
      difficulty: "中等",
      scene: "道路",
      recommendAlgorithm: "AOD-Net",
    },
    {
      id: 8,
      name: "中度雾霾-森林",
      url: "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800",
      category: "medium",
      difficulty: "中等",
      scene: "森林",
      recommendAlgorithm: "DehazeNet",
    },
    {
      id: 9,
      name: "中度雾霾-海岸",
      url: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800",
      category: "medium",
      difficulty: "中等",
      scene: "海岸",
      recommendAlgorithm: "FFA-Net",
    },
    {
      id: 10,
      name: "中度雾霾-乡村",
      url: "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800",
      category: "medium",
      difficulty: "中等",
      scene: "乡村",
      recommendAlgorithm: "DehazeNet",
    },
  ],
  heavy: [
    {
      id: 11,
      name: "重度雾霾-城市中心",
      url: "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800",
      category: "heavy",
      difficulty: "困难",
      scene: "城市",
      recommendAlgorithm: "FFA-Net",
    },
    {
      id: 12,
      name: "重度雾霾-高速公路",
      url: "https://images.unsplash.com/photo-1465447142348-e9952c393450?w=800",
      category: "heavy",
      difficulty: "困难",
      scene: "道路",
      recommendAlgorithm: "GridDehazeNet",
    },
    {
      id: 13,
      name: "重度雾霾-山区",
      url: "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800",
      category: "heavy",
      difficulty: "困难",
      scene: "山区",
      recommendAlgorithm: "FFA-Net",
    },
    {
      id: 14,
      name: "重度雾霾-港口",
      url: "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=800",
      category: "heavy",
      difficulty: "困难",
      scene: "港口",
      recommendAlgorithm: "GridDehazeNet",
    },
    {
      id: 15,
      name: "重度雾霾-工业区",
      url: "https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?w=800",
      category: "heavy",
      difficulty: "困难",
      scene: "工业",
      recommendAlgorithm: "FFA-Net",
    },
  ],
  special: [
    {
      id: 16,
      name: "特殊场景-夜景雾霾",
      url: "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=800",
      category: "special",
      difficulty: "困难",
      scene: "夜景",
      recommendAlgorithm: "MSBDN",
    },
    {
      id: 17,
      name: "特殊场景-逆光雾霾",
      url: "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800",
      category: "special",
      difficulty: "困难",
      scene: "逆光",
      recommendAlgorithm: "FFA-Net",
    },
    {
      id: 18,
      name: "特殊场景-雨雾",
      url: "https://images.unsplash.com/photo-1428908728789-d2de25dbd4e2?w=800",
      category: "special",
      difficulty: "中等",
      scene: "雨雾",
      recommendAlgorithm: "DehazeNet",
    },
    {
      id: 19,
      name: "特殊场景-晨雾",
      url: "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800",
      category: "special",
      difficulty: "简单",
      scene: "晨雾",
      recommendAlgorithm: "DCP",
    },
    {
      id: 20,
      name: "特殊场景-雪雾",
      url: "https://images.unsplash.com/photo-1491002052546-bf38f186af56?w=800",
      category: "special",
      difficulty: "中等",
      scene: "雪雾",
      recommendAlgorithm: "AOD-Net",
    },
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
  return allImages[Math.floor(Math.random() * allImages.length)];
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

// ==================== 历史记录管理 ====================

const STORAGE_KEY = "dehaze_history";
const MAX_RECORDS = 20;

/** 获取历史记录 */
export function getHistoryRecords(): HistoryRecord[] {
  try {
    const data = uni.getStorageSync(STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch (e) {
    console.error("读取历史记录失败:", e);
    return [];
  }
}

/** 保存历史记录 */
export function saveHistoryRecord(record: Omit<HistoryRecord, "id">): void {
  try {
    const history = getHistoryRecords();
    const newRecord: HistoryRecord = {
      ...record,
      id: Date.now(),
    };
    history.unshift(newRecord);

    // 限制记录数量
    if (history.length > MAX_RECORDS) {
      history.splice(MAX_RECORDS);
    }

    uni.setStorageSync(STORAGE_KEY, JSON.stringify(history));
  } catch (e) {
    console.error("保存历史记录失败:", e);
  }
}

/** 删除历史记录 */
export function deleteHistoryRecord(id: number): void {
  try {
    const history = getHistoryRecords();
    const filtered = history.filter((record) => record.id !== id);
    uni.setStorageSync(STORAGE_KEY, JSON.stringify(filtered));
  } catch (e) {
    console.error("删除历史记录失败:", e);
  }
}

/** 清空历史记录 */
export function clearHistoryRecords(): void {
  try {
    uni.removeStorageSync(STORAGE_KEY);
  } catch (e) {
    console.error("清空历史记录失败:", e);
  }
}

/** 按时间分组历史记录 */
export interface GroupedHistory {
  title: string;
  records: HistoryRecord[];
}

export function groupHistoryByTime(
  records: HistoryRecord[]
): GroupedHistory[] {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

  const groups: GroupedHistory[] = [
    { title: "今天", records: [] },
    { title: "昨天", records: [] },
    { title: "最近7天", records: [] },
    { title: "更早", records: [] },
  ];

  records.forEach((record) => {
    const recordDate = new Date(record.timestamp);
    if (recordDate >= today) {
      groups[0].records.push(record);
    } else if (recordDate >= yesterday) {
      groups[1].records.push(record);
    } else if (recordDate >= weekAgo) {
      groups[2].records.push(record);
    } else {
      groups[3].records.push(record);
    }
  });

  // 过滤空分组
  return groups.filter((group) => group.records.length > 0);
}

// ==================== 工具函数 ====================

/** 格式化文件大小 */
export function formatFileSize(bytes: number): string {
  if (!bytes) return "-";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

/** 格式化时间 */
export function formatTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) return "刚刚";
  if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
  if (diff < 172800000) return "昨天";

  return date.toLocaleDateString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** 支持的图片格式 */
export const SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "heic"];

/** 最大文件大小（20MB） */
export const MAX_FILE_SIZE = 20 * 1024 * 1024;

/** 压缩阈值（5MB） */
export const COMPRESS_THRESHOLD = 5 * 1024 * 1024;

/** 检查文件格式是否支持 */
export function isSupportedFormat(fileName: string): boolean {
  const ext = fileName.split(".").pop()?.toLowerCase();
  return ext ? SUPPORTED_FORMATS.includes(ext) : false;
}

/** 检查文件大小是否超限 */
export function isFileSizeValid(size: number): boolean {
  return size <= MAX_FILE_SIZE;
}

/** 是否需要压缩 */
export function needsCompression(size: number): boolean {
  return size > COMPRESS_THRESHOLD;
}
