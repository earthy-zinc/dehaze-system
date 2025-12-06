/**
 * 图像输入模块类型定义
 */

// 输入方式类型
export type InputMethod = 'upload' | 'camera' | 'sample' | 'history'

// 样例图片分类
export type SampleCategory = 'all' | 'light' | 'medium' | 'heavy' | 'special'

// 难度等级
export type DifficultyLevel = '简单' | '中等' | '困难'

// 样例图片
export interface SampleImage {
  id: number
  name: string
  url: string
  category: Exclude<SampleCategory, 'all'>
  difficulty: DifficultyLevel
  sceneType?: string        // 场景类型（城市/风景/建筑等）
  recommendAlgorithm?: string // 推荐算法
}

// 图片数据
export interface ImageData {
  url: string               // 图片 URL（临时路径或网络路径）
  path?: string             // 本地临时文件路径
  width: number             // 图片宽度
  height: number            // 图片高度
  size: number              // 文件大小（字节）
  name: string              // 文件名
  type?: string             // 文件类型 (image/jpeg, image/png 等)
  sampleInfo?: SampleImage  // 样例图片信息（如果来自样例库）
  compressed?: boolean      // 是否已压缩
  originalSize?: number     // 原始大小（压缩前）
}

// 历史记录状态
export type HistoryStatus = 'success' | 'failed' | 'processing'

// 历史记录
export interface HistoryRecord {
  id: number
  originalImage: string     // 原图缩略图 URL
  resultImage?: string      // 结果图缩略图 URL
  algorithm?: string        // 使用的算法名称
  algorithmId?: string      // 算法 ID
  timestamp: string         // 处理时间 ISO 格式
  status: HistoryStatus
  fileName?: string         // 原始文件名
  processingTime?: number   // 处理耗时（毫秒）
}

// 分组后的历史记录
export interface GroupedHistory {
  title: string             // 分组标题（今天/昨天/最近7天/更早）
  records: HistoryRecord[]
}

// Taro 临时文件
export interface TempFile {
  path: string
  size: number
  type?: string
  originalFileObj?: File
}

// 图片信息
export interface ImageInfo {
  width: number
  height: number
  path: string
  orientation?: string
  type?: string
}

// 上传结果
export interface UploadResult {
  url: string
  fileId?: string
  success: boolean
  message?: string
}

// 上传进度
export interface UploadProgress {
  progress: number          // 0-100
  totalBytesSent: number
  totalBytesExpectedToSend: number
}

// 历史存储接口（便于后续扩展云端同步）
export interface IHistoryStorage {
  getHistory(): Promise<HistoryRecord[]>
  addRecord(record: Omit<HistoryRecord, 'id'>): Promise<void>
  deleteRecord(id: number): Promise<void>
  clearHistory(): Promise<void>
}

// 错误类型
export interface ImageInputError {
  code: string
  message: string
  details?: any
}

// 错误码
export const ErrorCodes = {
  FILE_TOO_LARGE: 'FILE_TOO_LARGE',
  UNSUPPORTED_FORMAT: 'UNSUPPORTED_FORMAT',
  NETWORK_ERROR: 'NETWORK_ERROR',
  UPLOAD_FAILED: 'UPLOAD_FAILED',
  COMPRESS_FAILED: 'COMPRESS_FAILED',
  PERMISSION_DENIED: 'PERMISSION_DENIED',
  CAMERA_NOT_AVAILABLE: 'CAMERA_NOT_AVAILABLE',
} as const

// 错误信息映射
export const ErrorMessages: Record<string, string> = {
  [ErrorCodes.FILE_TOO_LARGE]: '图片大小超过20MB，请选择较小的图片',
  [ErrorCodes.UNSUPPORTED_FORMAT]: '不支持该图片格式，请选择JPG/PNG/WEBP/HEIC格式',
  [ErrorCodes.NETWORK_ERROR]: '网络连接失败，请检查网络后重试',
  [ErrorCodes.UPLOAD_FAILED]: '上传失败，请重试',
  [ErrorCodes.COMPRESS_FAILED]: '图片压缩失败，请重试',
  [ErrorCodes.PERMISSION_DENIED]: '相机/相册权限被拒绝，请在设置中开启',
  [ErrorCodes.CAMERA_NOT_AVAILABLE]: '相机不可用，请检查设备',
}

// 文件大小限制（字节）
export const FileSizeLimit = {
  MAX_SIZE: 20 * 1024 * 1024,           // 20MB
  COMPRESS_THRESHOLD: 5 * 1024 * 1024,  // 5MB 以上自动压缩
  COMPRESS_QUALITY: 85,                  // 压缩质量 85%
} as const

// 支持的图片格式
export const SupportedFormats = ['jpg', 'jpeg', 'png', 'webp', 'heic', 'heif'] as const
