/**
 * 图像输入模块 - 数据类型定义和样例数据
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
  /** 对应的无雾GT图片URL（用于指标评估参考） */
  cleanUrl?: string;
  category: FogLevel;
  difficulty: Difficulty;
  scene?: string;
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
