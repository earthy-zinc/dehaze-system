/** 模型预测参数 */
export interface PredParam {
  modelId: number;
  url: string;
  modelParam?: Object;
}

/** 模型预测结果 */
export interface PredResult {
  predUrl: string;
  hazeUrl: string;
}

/** 评价参数 */
export interface EvalParam {
  modelId: number;
  predUrl: string;
  gtUrl?: string;
}

/** 评价结果 */
export interface EvalResult {
  id: number;
  label: string;
  value: string;
  baseline?: string;
  better?: "higher" | "lower";
  description?: string;
}

export interface PredImageInfo {
  name: string;
  path: string;
  url: string;
}

// ===== 新预测/评估 API 类型（对应 Java PredictionController/EvaluationController） =====

/** 预测请求 */
export interface PredictionForm {
  algorithmId: number;
  fileId?: number;
  imageUrl?: string;
  params?: string;
}

/** 预测结果 */
export interface PredictionResultVO {
  logId?: number;
  resultUrl: string;
  resultThumbnailUrl?: string;
  time: number;
  fromCache?: boolean;
}

/** 预测日志 */
export interface PredLogVO {
  id: number;
  algorithmId: number;
  algorithmName?: string;
  originUrl?: string;
  predUrl?: string;
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

/** 评估结果 */
export interface EvaluationResultVO {
  logId?: number;
  metrics: Record<string, number>;
  qualified?: boolean;
  time: number;
}

/** 评估日志 */
export interface EvalLogVO {
  id: number;
  algorithmId: number;
  algorithmName?: string;
  predUrl?: string;
  gtUrl?: string;
  result?: string;
  time?: number;
  createTime?: string;
}

/** 评估日志查询 */
export interface EvalLogQuery {
  pageNum?: number;
  pageSize?: number;
  algorithmId?: number;
}
