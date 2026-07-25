// ===== 预测/评估 API 类型（对应 Java PredictionController/EvaluationController） =====

/** 预测/评估任务状态：processing-处理中 / completed-已完成 / failed-失败 */
export type PredEvalTaskStatus = "processing" | "completed" | "failed";

/** 预测请求 */
export interface PredictionForm {
  algorithmId: number;
  fileId?: number;
  imageUrl?: string;
  params?: string;
}

/**
 * 预测结果
 * - POST 返回：logId + status（processing 或 completed 缓存命中）
 * - GET 轮询：根据 status 返回不同字段
 *   - processing: 仅 logId + status
 *   - completed: 包含 resultUrl、time
 *   - failed: 包含 errorMessage、time
 */
export interface PredictionResultVO {
  logId?: number;
  status: PredEvalTaskStatus;
  resultUrl?: string;
  resultThumbnailUrl?: string;
  time?: number;
  errorMessage?: string;
  /** POST 缓存命中时返回，GET 不返回 */
  fromCache?: boolean;
}

/** 预测日志 */
export interface PredLogVO {
  id: number;
  algorithmId: number;
  algorithmName?: string;
  originUrl?: string;
  predUrl?: string;
  status?: PredEvalTaskStatus;
  errorMessage?: string;
  time?: number;
  createTime?: string;
}

/** 预测日志查询 */
export interface PredLogQuery {
  pageNum?: number;
  pageSize?: number;
  algorithmId?: number;
}

/** 评估请求 */
export interface EvaluationForm {
  algorithmId: number;
  predUrl?: string;
  gtUrl?: string;
  params?: string;
}

/**
 * 评估结果
 * - POST 返回：logId + status=processing
 * - GET 轮询：根据 status 返回不同字段
 *   - processing: 仅 logId + status
 *   - completed: 包含 metrics、time
 *   - failed: 包含 errorMessage、time
 */
export interface EvaluationResultVO {
  logId?: number;
  status: PredEvalTaskStatus;
  metrics?: Record<string, number>;
  time?: number;
  errorMessage?: string;
}

/** 评估日志 */
export interface EvalLogVO {
  id: number;
  algorithmId: number;
  algorithmName?: string;
  predUrl?: string;
  gtUrl?: string;
  status?: PredEvalTaskStatus;
  errorMessage?: string;
  result?: Record<string, number> | string;
  time?: number;
  createTime?: string;
}

/** 评估日志查询 */
export interface EvalLogQuery {
  pageNum?: number;
  pageSize?: number;
  algorithmId?: number;
}

/** 轮询选项 */
export interface PollOptions {
  /** 轮询间隔，默认 2000ms */
  intervalMs?: number;
  /** 最大等待时间，默认 120000ms（2分钟） */
  timeoutMs?: number;
  /** 每次轮询回调 */
  onPoll?: (status: PredEvalTaskStatus) => void;
}
