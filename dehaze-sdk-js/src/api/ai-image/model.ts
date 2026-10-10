import { PageQuery } from "@/types";

/**
 * 图像任务状态（整数）
 * - 1: PROCESSING 处理中
 * - 2: COMPLETED 已完成
 * - 3: FAILED 失败
 * - 4: CANCELED 已取消
 */
export type ImageTaskStatus = 1 | 2 | 3 | 4;

/** 图像编辑提交请求（fileId / imageUrl 至少提供一个） */
export interface ImageEditForm {
  /** 图像模型标识（统一模型规范 slug）必填；后端仅在请求缺省 model 时回落平台默认模型 */
  model: string;
  fileId?: number;
  imageUrl?: string;
  params?: string;
  /** 推荐来源：推荐记录ID（推荐管理模块返回的 recommendationId，用于追踪推荐采纳率） */
  recommendedBy?: number;
}

/**
 * 图像任务响应
 * - POST 提交返回：taskId + status（processing，或缓存命中时 completed）、estimatedCredits
 * - GET 轮询：按 status 返回不同字段
 *   - processing: 仅 taskId + status
 *   - completed: resultUrl、resultMd5、originMd5、resultThumbnailUrl、time
 *   - failed: errorMessage、time
 *   - canceled: 仅 taskId + status
 */
export interface ImageTaskResponse {
  taskId?: number;
  status: ImageTaskStatus;
  /** 预估消耗积分（提交时返回） */
  estimatedCredits?: number;
  resultUrl?: string;
  /** 结果图内容 MD5（completed 时返回） */
  resultMd5?: string;
  /** 原图内容 MD5（completed 时返回） */
  originMd5?: string;
  resultThumbnailUrl?: string;
  /** 推理时间（秒） */
  time?: number;
  errorMessage?: string;
}

/** 图像任务历史项 */
export interface ImageTaskVO {
  id: number;
  /** 图像模型标识（统一模型规范 slug） */
  model: string;
  /** 原图内容 MD5 */
  originMd5?: string;
  originUrl?: string;
  /** 结果图内容 MD5 */
  predMd5?: string;
  predUrl?: string;
  status?: ImageTaskStatus;
  errorMessage?: string;
  /** 推理时间（秒） */
  time?: number;
  createTime?: string;
}

/** 图像任务历史查询 */
export interface ImageTaskQuery extends PageQuery {
  /** 按图像模型标识筛选 */
  model?: string;
}

/** 轮询选项 */
export interface PollOptions {
  /** 轮询间隔，默认 2000ms */
  intervalMs?: number;
  /** 最大等待时间，默认 120000ms（2分钟） */
  timeoutMs?: number;
  /** 状态更新回调：提交后与每次轮询后各触发一次；taskId 供调用方取消任务使用 */
  onPoll?: (status: ImageTaskStatus, taskId?: number) => void;
}
