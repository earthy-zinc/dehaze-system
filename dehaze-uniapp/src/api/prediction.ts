/**
 * 预测/去雾处理 API
 *
 * 直接使用 dehaze-sdk-js 的 ModelAPI，不再维护独立定义。
 */

import { ModelAPI } from "dehaze-sdk-js";

export type {
  PredictionForm,
  PredictionResultVO,
  PredLogVO,
  PredLogQuery,
} from "dehaze-sdk-js";

// ==================== API 方法 ====================

/** 执行预测（去雾处理） */
export function predict(data: import("dehaze-sdk-js").PredictionForm) {
  return ModelAPI.predict(data);
}

/** 查询预测任务状态 */
export function getPredictionStatus(taskId: number) {
  return ModelAPI.getPredTaskStatus(taskId);
}

/** 获取预测日志列表（历史记录） */
export function getPredictionLogs(
  query?: import("dehaze-sdk-js").PredLogQuery
) {
  return ModelAPI.getPredLogs(query);
}
