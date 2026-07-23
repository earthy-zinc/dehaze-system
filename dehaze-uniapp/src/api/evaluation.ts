/**
 * 效果评估 API
 *
 * 直接使用 dehaze-sdk-js 的 ModelAPI，不再维护独立定义。
 */

import { ModelAPI } from "dehaze-sdk-js";

export type {
  EvaluationForm,
  EvaluationResultVO,
  EvalLogVO,
  EvalLogQuery,
} from "dehaze-sdk-js";

// ==================== API 方法 ====================

/** 执行效果评估（PSNR/SSIM/LPIPS等） */
export function evaluate(data: import("dehaze-sdk-js").EvaluationForm) {
  return ModelAPI.evaluate(data);
}

/** 查询评估任务状态 */
export function getEvalTaskStatus(taskId: number) {
  return ModelAPI.getEvalTaskStatus(taskId);
}
