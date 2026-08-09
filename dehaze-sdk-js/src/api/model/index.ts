import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  BatchPredictionForm,
  BatchPredictionResultVO,
  CompareReportForm,
  CompareReportResultVO,
  EvalLogQuery,
  EvalLogVO,
  EvalMetricsVO,
  EvaluationForm,
  EvaluationResultVO,
  PollOptions,
  PredLogQuery,
  PredLogVO,
  PredictionForm,
  PredictionQuotaVO,
  PredictionResultVO,
  PresetForm,
  PresetQuery,
  PresetVO,
} from "./model";

const DEFAULT_INTERVAL_MS = 2000;
const DEFAULT_TIMEOUT_MS = 120000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** 终态状态集合（COMPLETED=2 / FAILED=3 / CANCELED=4） */
const TERMINAL_STATUSES = new Set([2, 3, 4]);

class ModelAPI {
  // ===== 预测 =====

  /** 执行模型预测（去雾处理），返回 logId + status */
  static predict(data: PredictionForm) {
    return request<PredictionResultVO>({
      url: "/api/v1/prediction",
      method: "post",
      data,
    });
  }

  /** 查询预测任务状态 */
  static getPredTaskStatus(taskId: number) {
    return request<PredictionResultVO>({
      url: `/api/v1/prediction/${taskId}`,
      method: "get",
    });
  }

  /** 取消预测任务（幂等，对终态任务直接返回当前状态） */
  static cancelPredTask(taskId: number) {
    return request<PredictionResultVO>({
      url: `/api/v1/prediction/${taskId}/cancel`,
      method: "post",
    });
  }

  /**
   * 提交预测并等待结果（封装 POST + 轮询 GET）
   *
   * - POST 立即返回，若 status=2（COMPLETED，缓存命中）直接返回
   * - status=1（PROCESSING）时按 intervalMs 轮询 GET，直到 COMPLETED/FAILED/CANCELED 或超时
   *
   * 默认：间隔 2s，超时 120s
   */
  static async predictAndWait(
    data: PredictionForm,
    options?: PollOptions
  ): Promise<PredictionResultVO> {
    const result = await this.predict(data);
    if (result.status !== 1) {
      return result;
    }
    return this.pollPredTask(result.logId!, options);
  }

  /** 轮询预测任务直到终态（COMPLETED=2 / FAILED=3 / CANCELED=4）或超时 */
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
      if (TERMINAL_STATUSES.has(result.status)) {
        return result;
      }
    }
    throw new Error(`预测任务 ${logId} 超时（${timeout}ms）`);
  }

  /** 获取预测日志分页列表 */
  static getPredLogs(query?: PredLogQuery) {
    return request<PageResult<PredLogVO[]>>({
      url: "/api/v1/prediction/logs",
      method: "get",
      params: query,
    });
  }

  // ===== 批量预测 / 配额 =====

  /** 批量预测（一次提交多张图片，上限按会员等级动态计算） */
  static batchPredict(data: BatchPredictionForm) {
    return request<BatchPredictionResultVO>({
      url: "/api/v1/prediction/batch",
      method: "post",
      data,
    });
  }

  /** 查询用户剩余处理次数 */
  static getQuota() {
    return request<PredictionQuotaVO>({
      url: "/api/v1/prediction/quota",
      method: "get",
    });
  }

  // ===== 评估 =====

  /** 执行效果评估（PSNR/SSIM/LPIPS等），返回 logId + status */
  static evaluate(data: EvaluationForm) {
    return request<EvaluationResultVO>({
      url: "/api/v1/evaluation",
      method: "post",
      data,
    });
  }

  /** 查询评估任务状态 */
  static getEvalTaskStatus(taskId: number) {
    return request<EvaluationResultVO>({
      url: `/api/v1/evaluation/${taskId}`,
      method: "get",
    });
  }

  /**
   * 提交评估并等待结果（封装 POST + 轮询 GET）
   *
   * - POST 立即返回，若 status=2（COMPLETED）直接返回
   * - status=1（PROCESSING）时按 intervalMs 轮询 GET，直到 COMPLETED/FAILED/CANCELED 或超时
   *
   * 默认：间隔 2s，超时 120s
   */
  static async evaluateAndWait(
    data: EvaluationForm,
    options?: PollOptions
  ): Promise<EvaluationResultVO> {
    const result = await this.evaluate(data);
    if (result.status !== 1) {
      return result;
    }
    return this.pollEvalTask(result.logId!, options);
  }

  /** 轮询评估任务直到终态（COMPLETED=2 / FAILED=3 / CANCELED=4）或超时 */
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
      if (TERMINAL_STATUSES.has(result.status)) {
        return result;
      }
    }
    throw new Error(`评估任务 ${logId} 超时（${timeout}ms）`);
  }

  /** 获取评估日志分页列表 */
  static getEvalLogs(query?: EvalLogQuery) {
    return request<PageResult<EvalLogVO[]>>({
      url: "/api/v1/evaluation/logs",
      method: "get",
      params: query,
    });
  }

  /** 获取评估指标历史（当前用户，仅已完成任务） */
  static getEvalMetrics(query?: EvalLogQuery) {
    return request<PageResult<EvalMetricsVO[]>>({
      url: "/api/v1/evaluation/metrics",
      method: "get",
      params: query,
    });
  }

  // ===== 参数预设 =====

  /** 参数预设列表（系统预设 + 用户自定义） */
  static getPresets(query?: PresetQuery) {
    return request<PageResult<PresetVO[]>>({
      url: "/api/v1/presets",
      method: "get",
      params: query,
    });
  }

  /** 创建自定义预设 */
  static createPreset(data: PresetForm) {
    return request<PresetVO>({
      url: "/api/v1/presets",
      method: "post",
      data,
    });
  }

  /** 更新自定义预设 */
  static updatePreset(id: number, data: PresetForm) {
    return request<PresetVO>({
      url: `/api/v1/presets/${id}`,
      method: "put",
      data,
    });
  }

  /** 删除自定义预设 */
  static deletePreset(id: number) {
    return request({
      url: `/api/v1/presets/${id}`,
      method: "delete",
    });
  }

  // ===== 对比报告（效果对比） =====

  /** 生成对比报告（异步任务） */
  static generateReport(data: CompareReportForm) {
    return request<CompareReportResultVO>({
      url: "/api/v1/compare/report",
      method: "post",
      data,
    });
  }

  /** 查询对比报告任务状态（报告生成完成后返回下载URL） */
  static getReportStatus(taskId: number) {
    return request<CompareReportResultVO>({
      url: `/api/v1/compare/report/${taskId}`,
      method: "get",
    });
  }
}

export default ModelAPI;
