/**
 * 去雾处理模块类型定义
 *
 * 任务状态、参数定义、批量任务等。
 */

/** 处理任务状态 */
export type TaskStatus =
  | 'idle' // 空闲，未开始
  | 'processing' // 去雾处理中
  | 'success' // 处理成功
  | 'failed' // 处理失败
  | 'canceled'; // 已取消

/** 单个任务进度信息（API 同步返回，仅记录真实状态与已用时间，不模拟百分比） */
export interface TaskProgress {
  status: TaskStatus;
  /** 已用时间（ms） */
  elapsed: number;
  /** 错误信息 */
  error?: string;
}

/** 通用算法参数 */
export interface CommonAlgorithmParams {
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
  logId?: number;
  resultUrl: string;
  resultThumbnailUrl?: string;
  /** 处理耗时（ms） */
  time: number;
  /** 是否命中缓存 */
  fromCache?: boolean;
}
