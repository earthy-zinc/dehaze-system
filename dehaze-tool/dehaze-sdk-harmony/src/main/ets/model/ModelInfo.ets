/**
 * 模型预测与效果评估数据模型
 *
 * 与后端 dehaze-go 内部模型对齐：
 * - 预测：bo.PredForm / vo.PredictionResultVO / model.SysPredLog
 * - 评估：bo.EvalForm / vo.EvaluationResultVO / model.SysEvalLog
 */

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
  /** 结果图 URL */
  resultUrl: string = '';
  /** 结果缩略图 URL */
  resultThumbnailUrl?: string = undefined;
  /** 处理耗时（毫秒） */
  time?: number = undefined;
  /** 是否命中缓存 */
  fromCache?: boolean = undefined;
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

/** 评估结果项 */
export class EvalResult {
  id: number = 0;
  /** 指标名称 */
  label: string = '';
  /** 指标值 */
  value: string = '';
  /** 基准值 */
  baseline?: string = undefined;
  /** 越高越好 / 越低越好 */
  better?: string = undefined;
  /** 指标描述 */
  description?: string = undefined;
}

/** 评估日志查询参数 */
export class EvalLogQuery {
  pageNum?: number = 1;
  pageSize?: number = 10;
  algorithmId?: number = undefined;
}
