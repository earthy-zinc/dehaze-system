/**
 * 去雾处理模块类型定义
 *
 * 任务状态、图像编辑参数、处理结果等。
 */

/** 处理任务状态 */
export type TaskStatus =
  | 0 // 空闲，未开始
  | 1 // 去雾处理中
  | 2 // 处理成功
  | 3 // 处理失败
  | 4; // 已取消

/** 单个任务进度信息（API 同步返回，仅记录真实状态与已用时间，不模拟百分比） */
export interface TaskProgress {
  status: TaskStatus;
  /** 已用时间（ms） */
  elapsed: number;
  /** 错误信息 */
  error?: string;
}

/** 图像编辑参数（去雾强度/饱和度/对比度/锐化） */
export interface ImageEditParams {
  /** 去雾强度 0-100 */
  strength?: number;
  /** 色彩饱和度 0-200 */
  saturation?: number;
  /** 对比度 0-200 */
  contrast?: number;
  /** 锐化程度 0-100 */
  sharpen?: number;
}

/** 单张处理结果 */
export interface ProcessingResult {
  /** 图像任务 ID */
  taskId?: number;
  resultUrl: string;
  resultThumbnailUrl?: string;
  /** 处理耗时（秒） */
  time: number;
  /** 预估消耗积分（提交时返回） */
  estimatedCredits?: number;
}
