/**
 * 效果评估 API
 *
 * 对应后端 EvaluationController：
 * - POST /evaluation       执行效果评估
 * - GET  /evaluation/{id}  查询评估任务状态
 * - GET  /evaluation/logs  评估日志列表
 */

import { get, post } from "./request";

// ==================== 类型定义 ====================

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

// ==================== API 方法 ====================

/** 执行效果评估（PSNR/SSIM/LPIPS等） */
export async function evaluate(
  data: EvaluationForm
): Promise<EvaluationResultVO> {
  return post<EvaluationResultVO>(
    "/evaluation",
    data as unknown as Record<string, unknown>
  );
}

/** 查询评估任务状态 */
export async function getEvalTaskStatus(
  taskId: number
): Promise<EvaluationResultVO> {
  return get<EvaluationResultVO>(`/evaluation/${taskId}`);
}
