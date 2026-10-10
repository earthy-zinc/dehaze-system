import { PageQuery } from "@/types";

// ===== 评估 API 类型 =====

/**
 * 评估任务状态（使用整数）
 * - 1: PROCESSING 处理中
 * - 2: COMPLETED 已完成
 * - 3: FAILED 失败
 */
export type EvalTaskStatus = 1 | 2 | 3;

/** 评估请求（Python 事实源仅接受 URL，不支持文件 ID） */
export interface EvaluationForm {
  /** 图像模型标识（统一模型规范 slug） */
  modelId: string;
  /** 预测结果图片 URL */
  predUrl: string;
  /** Ground Truth 参考图片 URL */
  gtUrl: string;
  /** 评估参数（JSON）；当前仅 Java 透传，Python 丢弃、Go 不接受该字段 */
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
  status: EvalTaskStatus;
  metrics?: Record<string, number>;
  /** 预测图内容 MD5（completed 时返回） */
  predMd5?: string;
  /** 参考图内容 MD5（completed 时返回） */
  gtMd5?: string;
  /** 评估耗时（秒） */
  time?: number;
  errorMessage?: string;
}

/** 评估日志 */
export interface EvalLogVO {
  id: number;
  /** 图像模型标识（统一模型规范 slug） */
  modelId: string;
  modelName?: string;
  predUrl?: string;
  /** 预测图内容 MD5 */
  predMd5?: string;
  gtUrl?: string;
  /** 参考图内容 MD5 */
  gtMd5?: string;
  status?: EvalTaskStatus;
  errorMessage?: string;
  result?: Record<string, number> | string;
  /** 评估耗时（秒） */
  time?: number;
  createTime?: string;
}

/** 评估指标历史 */
export interface EvalMetricsVO {
  /** 日志 ID */
  id: number;
  /** 图像模型标识（统一模型规范 slug） */
  modelId: string;
  modelName?: string;
  /** 预测文件 URL */
  predUrl?: string;
  /** 参考图片 URL */
  gtUrl?: string;
  /** 评估指标结果（PSNR/SSIM 等） */
  metrics?: Record<string, number>;
  /** 评估耗时（秒） */
  time?: number;
  /** 任务状态 */
  status?: EvalTaskStatus;
  /** 失败错误信息 */
  errorMessage?: string;
  createTime?: string;
}

/** 评估日志查询 */
export interface EvalLogQuery extends PageQuery {
  /** 按图像模型标识筛选 */
  modelId?: string;
}

// ===== 参数预设（去雾处理） =====

/** 参数预设表单 */
export interface PresetForm {
  name: string;
  /** 关联图像模型标识（统一模型规范 slug） */
  modelId: string;
  params: string;
  /** 是否默认预设(0:否;1:是)，缺省 0 */
  isDefault?: number;
}

/** 参数预设视图对象 */
export interface PresetVO {
  id: number;
  name: string;
  /** 预设类型（system-系统预设 / custom-用户自定义） */
  type: string;
  /** 关联图像模型标识（统一模型规范 slug） */
  modelId: string;
  params: string;
  userId?: number;
  /** 是否默认预设(0:否;1:是) */
  isDefault: number;
  createTime: string;
}

/** 参数预设查询 */
export interface PresetQuery extends PageQuery {
  modelId?: string;
  /** 系统预设筛选（true-仅系统预设 / false-仅用户自定义 / 缺省-两者都取） */
  isSystem?: boolean;
}

// ===== 对比报告（效果对比） =====

/** 对比报告生成请求 */
export interface CompareReportForm {
  /** 处理日志ID（报告格式固定为 HTML，无其他可选项） */
  logId: number;
}

/** 对比报告结果（异步任务） */
export interface CompareReportResultVO {
  /** 异步任务ID */
  taskId: number;
  status: EvalTaskStatus;
  downloadUrl?: string;
  errorMessage?: string;
}
