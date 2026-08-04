import { PageQuery } from "@/types";

// ===== 预测/评估 API 类型（对应 Java PredictionController/EvaluationController） =====

/**
 * 预测/评估任务状态（与后端 LogStatusEnum 对齐，使用整数）
 * - 1: PROCESSING 处理中
 * - 2: COMPLETED 已完成
 * - 3: FAILED 失败
 */
export type PredEvalTaskStatus = 1 | 2 | 3;

/** 预测请求 */
export interface PredictionForm {
  algorithmId: number;
  fileId?: number;
  imageUrl?: string;
  params?: string;
  /** 推荐算法ID（来自推荐管理模块，用于追踪推荐采纳率） */
  recommendedBy?: number;
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
export interface PredLogQuery extends PageQuery {
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

/** 评估指标历史（对应后端 EvalMetricsVO） */
export interface EvalMetricsVO {
  /** 日志 ID */
  id: number;
  algorithmId: number;
  algorithmName?: string;
  /** 预测文件 URL */
  predUrl?: string;
  /** 参考图片 URL */
  gtUrl?: string;
  /** 评估指标结果（PSNR/SSIM 等） */
  metrics?: Record<string, number>;
  /** 处理时间（毫秒） */
  time?: number;
  /** 任务状态 */
  status?: PredEvalTaskStatus;
  /** 失败错误信息 */
  errorMessage?: string;
  createTime?: string;
}

/** 评估日志查询 */
export interface EvalLogQuery extends PageQuery {
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

// ===== 批量预测（去雾处理） =====

/** 批量预测请求 */
export interface BatchPredictionForm {
  algorithmId: number;
  items: { fileId?: number; imageUrl?: string; params?: string }[];
  /** 推荐算法ID（来自推荐管理模块） */
  recommendedBy?: number;
}

/** 批量预测结果 */
export interface BatchPredictionResultVO {
  total: number;
  results: PredictionResultVO[];
}

// ===== 参数预设（去雾处理） =====

/** 参数预设表单 */
export interface PresetForm {
  id?: number;
  name: string;
  algorithmId: number;
  params: string;
  /** 是否系统预设 */
  isSystem?: boolean;
}

/** 参数预设视图对象 */
export interface PresetVO extends PresetForm {
  id: number;
  userId?: number;
  createTime: string;
}

/** 参数预设查询 */
export interface PresetQuery extends PageQuery {
  algorithmId?: number;
  isSystem?: boolean;
}

// ===== VIP 配额（去雾处理） =====

/** VIP 配额 */
export interface PredictionQuota {
  /** 剩余次数 */
  remaining: number;
  /** 总次数 */
  total: number;
  /** 已使用次数 */
  used: number;
  /** 重置日期 */
  resetDate: string;
}

// ===== 对比报告（效果对比） =====

/** 对比报告生成请求 */
export interface CompareReportForm {
  /** 处理日志ID */
  logId: number;
  /** 报告格式 */
  format: "pdf" | "image";
  /** 是否包含指标 */
  includeMetrics?: boolean;
  /** 是否包含滤镜参数 */
  includeFilters?: boolean;
}

/** 对比报告结果（异步任务） */
export interface CompareReportResultVO {
  /** 异步任务ID */
  taskId: number;
  status: PredEvalTaskStatus;
  downloadUrl?: string;
  errorMessage?: string;
}
