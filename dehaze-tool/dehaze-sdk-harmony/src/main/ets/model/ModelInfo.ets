/**
 * 模型预测与效果评估数据模型
 *
 * 异步任务模式：POST 立即返回 logId + status=processing；
 * GET 轮询至 completed/failed 时返回完整字段。
 */

/** 任务状态 */
export type TaskStatus = 'processing' | 'completed' | 'failed';

/** 轮询选项 */
export class PollOptions {
  /** 轮询间隔（毫秒），默认 2000 */
  intervalMs?: number = undefined;
  /** 最大等待时间（毫秒），默认 120000 */
  timeoutMs?: number = undefined;
  /** 每次轮询回调 */
  onPoll?: (status: TaskStatus) => void = undefined;
}

/**
 * 预测请求参数
 *
 * 后端支持两种输入方式：
 * 1. 已上传文件：fileId
 * 2. 图片 URL：imageUrl
 */
export class PredParam {
  /** 算法 ID */
  algorithmId: number = 0;
  /** 已上传文件 ID（与 imageUrl 二选一） */
  fileId?: number = undefined;
  /** 图片 URL（与 fileId 二选一） */
  imageUrl?: string = undefined;
  /** 算法参数（JSON 字符串） */
  params?: string = undefined;
}

/** 预测结果（后端 vo.PredictionResultVO） */
export class PredResult {
  /** 预测日志 ID */
  logId?: number = undefined;
  /** 任务状态 */
  status: TaskStatus = 'processing';
  /** 结果图 URL（completed 时返回） */
  resultUrl?: string = undefined;
  /** 结果缩略图 URL（completed 时返回） */
  resultThumbnailUrl?: string = undefined;
  /** 处理耗时（毫秒） */
  time?: number = undefined;
  /** 失败错误信息（failed 时返回） */
  errorMessage?: string = undefined;
}

/** 预测日志（后端 model.SysPredLog，历史记录） */
export class PredLog {
  id: number = 0;
  algorithmId: number = 0;
  algorithmName?: string = undefined;
  /** 原始图 URL */
  originUrl?: string = undefined;
  /** 预测结果图 URL */
  predUrl?: string = undefined;
  /** 处理耗时 */
  time?: number = undefined;
  /** 任务状态 */
  status?: string = undefined;
  /** 失败错误信息 */
  errorMessage?: string = undefined;
  createTime?: string = undefined;
}

/** 预测日志查询参数 */
export class PredLogQuery {
  pageNum?: number = 1;
  pageSize?: number = 10;
  algorithmId?: number = undefined;
}

/**
 * 评估请求参数
 */
export class EvalParam {
  /** 算法 ID */
  algorithmId: number = 0;
  /** 预测结果图 URL */
  predUrl: string = '';
  /** 清晰参考图 URL（GT） */
  gtUrl?: string = undefined;
}

/** 评估结果（后端 vo.EvaluationResultVO） */
export class EvalResult {
  /** 评估日志 ID */
  logId?: number = undefined;
  /** 任务状态 */
  status: TaskStatus = 'processing';
  /** 评估指标（completed 时返回） */
  metrics?: Record<string, number> = undefined;
  /** 处理耗时（毫秒） */
  time?: number = undefined;
  /** 失败错误信息（failed 时返回） */
  errorMessage?: string = undefined;
}

/** 评估日志（后端 model.SysEvalLog，历史记录） */
export class EvalLog {
  id: number = 0;
  algorithmId: number = 0;
  algorithmName?: string = undefined;
  /** 预测图 URL */
  predUrl?: string = undefined;
  /** GT 图 URL */
  gtUrl?: string = undefined;
  /** 评估指标 JSON 字符串 */
  result?: string = undefined;
  /** 处理耗时 */
  time?: number = undefined;
  /** 任务状态 */
  status?: string = undefined;
  /** 失败错误信息 */
  errorMessage?: string = undefined;
  createTime?: string = undefined;
}

/** 评估日志查询参数 */
export class EvalLogQuery {
  pageNum?: number = 1;
  pageSize?: number = 10;
  algorithmId?: number = undefined;
}
