import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  EvalLogQuery,
  EvalLogVO,
  EvaluationForm,
  EvaluationResultVO,
  PollOptions,
  PredLogQuery,
  PredLogVO,
  PredictionForm,
  PredictionResultVO,
} from "./model";

const DEFAULT_INTERVAL_MS = 2000;
const DEFAULT_TIMEOUT_MS = 120000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class ModelAPI {
  /** 执行模型预测（去雾处理），返回 logId + status */
  static predict(data: PredictionForm) {
    return request<any, PredictionResultVO>({
      url: "/api/v1/prediction",
      method: "post",
      data,
    });
  }

  /** 查询预测任务状态 */
  static getPredTaskStatus(taskId: number) {
    return request<any, PredictionResultVO>({
      url: `/api/v1/prediction/${taskId}`,
      method: "get",
    });
  }

  /**
   * 提交预测并等待结果（封装 POST + 轮询 GET）
   *
   * - POST 立即返回，若 status=completed（缓存命中）直接返回
   * - status=processing 时按 intervalMs 轮询 GET，直到 completed/failed 或超时
   *
   * 默认：间隔 2s，超时 120s
   */
  static async predictAndWait(
    data: PredictionForm,
    options?: PollOptions
  ): Promise<PredictionResultVO> {
    const result = await this.predict(data);
    if (result.status !== "processing") {
      return result;
    }
    return this.pollPredTask(result.logId!, options);
  }

  /** 轮询预测任务直到终态（completed/failed）或超时 */
  private static async pollPredTask(
    logId: number,
    options?: PollOptions
  ): Promise<PredictionResultVO> {
    const interval = options?.intervalMs ?? DEFAULT_INTERVAL_MS;
    const timeout = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const deadline = Date.now() + timeout;

    while (Date.now() < deadline) {
      await sleep(interval);
      const result = await this.getPredTaskStatus(logId);
      options?.onPoll?.(result.status);
      if (result.status === "completed" || result.status === "failed") {
        return result;
      }
    }
    throw new Error(`预测任务 ${logId} 超时（${timeout}ms）`);
  }

  /** 获取预测日志分页列表 */
  static getPredLogs(query?: PredLogQuery) {
    return request<any, PageResult<PredLogVO[]>>({
      url: "/api/v1/prediction/logs",
      method: "get",
      params: query,
    });
  }

  /** 执行效果评估（PSNR/SSIM/LPIPS等），返回 logId + status */
  static evaluate(data: EvaluationForm) {
    return request<any, EvaluationResultVO>({
      url: "/api/v1/evaluation",
      method: "post",
      data,
    });
  }

  /** 查询评估任务状态 */
  static getEvalTaskStatus(taskId: number) {
    return request<any, EvaluationResultVO>({
      url: `/api/v1/evaluation/${taskId}`,
      method: "get",
    });
  }

  /**
   * 提交评估并等待结果（封装 POST + 轮询 GET）
   *
   * - POST 立即返回，若 status=completed 直接返回
   * - status=processing 时按 intervalMs 轮询 GET，直到 completed/failed 或超时
   *
   * 默认：间隔 2s，超时 120s
   */
  static async evaluateAndWait(
    data: EvaluationForm,
    options?: PollOptions
  ): Promise<EvaluationResultVO> {
    const result = await this.evaluate(data);
    if (result.status !== "processing") {
      return result;
    }
    return this.pollEvalTask(result.logId!, options);
  }

  /** 轮询评估任务直到终态（completed/failed）或超时 */
  private static async pollEvalTask(
    logId: number,
    options?: PollOptions
  ): Promise<EvaluationResultVO> {
    const interval = options?.intervalMs ?? DEFAULT_INTERVAL_MS;
    const timeout = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const deadline = Date.now() + timeout;

    while (Date.now() < deadline) {
      await sleep(interval);
      const result = await this.getEvalTaskStatus(logId);
      options?.onPoll?.(result.status);
      if (result.status === "completed" || result.status === "failed") {
        return result;
      }
    }
    throw new Error(`评估任务 ${logId} 超时（${timeout}ms）`);
  }

  /** 获取评估日志分页列表 */
  static getEvalLogs(query?: EvalLogQuery) {
    return request<any, PageResult<EvalLogVO[]>>({
      url: "/api/v1/evaluation/logs",
      method: "get",
      params: query,
    });
  }
}

export default ModelAPI;
