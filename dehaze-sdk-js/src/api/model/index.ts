import { PageResult } from "@/types";
import request from "@/utils/request";
import { DEFAULT_POLL_INTERVAL_MS, DEFAULT_POLL_TIMEOUT_MS, sleep } from "@/utils/poll";
import { PollOptions } from "@/api/ai-image/model";
import {
  CompareReportForm,
  CompareReportResultVO,
  EvalLogQuery,
  EvalLogVO,
  EvalMetricsVO,
  EvaluationForm,
  EvaluationResultVO,
  PresetForm,
  PresetQuery,
  PresetVO,
} from "./model";

/** 终态状态集合（COMPLETED=2 / FAILED=3） */
const TERMINAL_STATUSES = new Set([2, 3]);

class ModelAPI {
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
   * - status=1（PROCESSING）时按 intervalMs 轮询 GET，直到 COMPLETED/FAILED 或超时
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

  /** 轮询评估任务直到终态（COMPLETED=2 / FAILED=3）或超时 */
  private static async pollEvalTask(
    logId: number,
    options?: PollOptions
  ): Promise<EvaluationResultVO> {
    const interval = options?.intervalMs ?? DEFAULT_POLL_INTERVAL_MS;
    const timeout = options?.timeoutMs ?? DEFAULT_POLL_TIMEOUT_MS;
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
