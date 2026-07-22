/**
 * 图像输入模块类型定义
 */

// 输入方式类型
export type InputMethod = "upload" | "camera" | "sample" | "history";

// 样例图片分类（按场景，对齐设计文档需求规格 §2.3.2）
export type SampleCategory = "all" | "city" | "nature" | "portrait" | "night";

// 难度等级
export type DifficultyLevel = "简单" | "中等" | "困难";

// 样例图片
export interface SampleImage {
  id: number;
  name: string;
  url: string; // 图片访问 URL
  thumbnailUrl?: string; // 缩略图 URL
  category: Exclude<SampleCategory, "all">;
  sceneType?: string; // 场景类型
  hazeLevel?: "light" | "medium" | "heavy"; // 雾霾程度
  recommendAlgorithm?: string; // 推荐算法
}

// 图片数据
export interface ImageData {
  url: string; // 图片 URL（临时路径或网络路径）
  path?: string; // 本地临时文件路径
  width: number; // 图片宽度
  height: number; // 图片高度
  size: number; // 文件大小（字节）
  name: string; // 文件名
  type?: string; // 文件类型 (image/jpeg, image/png 等)
  sampleInfo?: SampleImage; // 样例图片信息（如果来自样例库）
  compressed?: boolean; // 是否已压缩
  originalSize?: number; // 原始大小（压缩前）
}

// Taro 临时文件
export interface TempFile {
  path: string;
  size: number;
  type?: string;
  originalFileObj?: File;
}

// 图片信息
export interface ImageInfo {
  width: number;
  height: number;
  path: string;
  orientation?: string;
  type?: string;
}

// 上传结果
export interface UploadResult {
  url: string;
  fileId?: string;
  success: boolean;
  message?: string;
}

// 上传进度
export interface UploadProgress {
  progress: number; // 0-100
  totalBytesSent: number;
  totalBytesExpectedToSend: number;
}

// 错误类型
export class ImageInputError extends Error {
  code: string;
  details?: any;
  constructor(code: string, message: string, details?: any) {
    super(message);
    this.name = "ImageInputError";
    this.code = code;
    this.details = details;
  }
}

// 错误码
export const ErrorCodes = {
  FILE_TOO_LARGE: "FILE_TOO_LARGE",
  UNSUPPORTED_FORMAT: "UNSUPPORTED_FORMAT",
  RESOLUTION_LOW: "RESOLUTION_LOW",
  NETWORK_ERROR: "NETWORK_ERROR",
  UPLOAD_FAILED: "UPLOAD_FAILED",
  COMPRESS_FAILED: "COMPRESS_FAILED",
  PERMISSION_DENIED: "PERMISSION_DENIED",
  CAMERA_NOT_AVAILABLE: "CAMERA_NOT_AVAILABLE",
  USER_CANCEL: "USER_CANCEL",
} as const;

// 错误信息映射
export const ErrorMessages: Record<string, string> = {
  [ErrorCodes.FILE_TOO_LARGE]: "图片大小超过20MB，请选择较小的图片",
  [ErrorCodes.UNSUPPORTED_FORMAT]:
    "不支持该图片格式，请选择JPG/PNG/WEBP/HEIC格式",
  [ErrorCodes.RESOLUTION_LOW]: "图片分辨率过低，建议至少 640×480",
  [ErrorCodes.NETWORK_ERROR]: "网络连接失败，请检查网络后重试",
  [ErrorCodes.UPLOAD_FAILED]: "上传失败，请重试",
  [ErrorCodes.COMPRESS_FAILED]: "图片压缩失败，请重试",
  [ErrorCodes.PERMISSION_DENIED]: "相机/相册权限被拒绝，请在设置中开启",
  [ErrorCodes.CAMERA_NOT_AVAILABLE]: "相机不可用，请检查设备",
  [ErrorCodes.USER_CANCEL]: "用户取消操作",
};

// 文件大小限制（字节）
export const FileSizeLimit = {
  MAX_SIZE: 20 * 1024 * 1024, // 20MB
  COMPRESS_THRESHOLD: 5 * 1024 * 1024, // 5MB 以上自动压缩
  COMPRESS_QUALITY: 85, // 压缩质量 85%
} as const;

// 支持的图片格式
export const SupportedFormats = [
  "jpg",
  "jpeg",
  "png",
  "webp",
  "heic",
  "heif",
] as const;

// 最低分辨率要求
export const MinResolution = {
  WIDTH: 640,
  HEIGHT: 480,
} as const;
