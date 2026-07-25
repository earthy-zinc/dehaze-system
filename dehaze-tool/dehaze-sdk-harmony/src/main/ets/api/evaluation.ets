import { HttpManager } from '../utils/HttpManager';
import { EvalParam, EvalResult, EvalLog, EvalLogQuery, PageResult, PollOptions, TaskStatus } from '../model/ModelInfo';

const DEFAULT_INTERVAL_MS: number = 2000;
const DEFAULT_TIMEOUT_MS: number = 120000;

function sleep(ms: number): Promise<void> {
  return new Promise<void>((resolve): void => {
    setTimeout(resolve, ms);
  });
}

/**
 * 效果评估 API
 *
 * 路径与后端路由对齐（dehaze-go/internal/router/evaluation.go）：
 * - POST /api/v1/evaluation       执行效果评估，立即返回 logId + status=processing
 * - GET  /api/v1/evaluation/{id}  轮询评估任务状态
 * - GET  /api/v1/evaluation/logs  评估日志列表
 */
export class EvaluationAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /** 执行效果评估，返回 logId + status */
  public static async evaluate(data: EvalParam): Promise<EvalResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<EvalResult>(`${EvaluationAPI.BASE_URL}/evaluation`, data as unknown as Record<string, Object>);
  }

  /** 查询评估任务状态 */
  public static async getStatus(taskId: number): Promise<EvalResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<EvalResult>(`${EvaluationAPI.BASE_URL}/evaluation/${taskId}`, undefined);
  }

  /**
   * 提交评估并等待结果（封装 POST + 轮询 GET）
   *
   * - POST 立即返回，若 status=completed 直接返回
   * - status=processing 时按 intervalMs 轮询 GET，直到 completed/failed 或超时
   *
   * 默认：间隔 2s，超时 120s
   */
  public static async evaluateAndWait(data: EvalParam, options?: PollOptions): Promise<EvalResult> {
    const result = await EvaluationAPI.evaluate(data);
    if (result.status !== 'processing') {
      return result;
    }
    return EvaluationAPI.pollEvalTask(result.logId!, options);
  }

  /** 轮询评估任务直到终态（completed/failed）或超时 */
  private static async pollEvalTask(logId: number, options?: PollOptions): Promise<EvalResult> {
    const interval: number = options?.intervalMs ?? DEFAULT_INTERVAL_MS;
    const timeout: number = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const deadline: number = Date.now() + timeout;

    while (Date.now() < deadline) {
      await sleep(interval);
      const result = await EvaluationAPI.getStatus(logId);
      if (options?.onPoll) {
        options.onPoll(result.status as TaskStatus);
      }
      if (result.status === 'completed' || result.status === 'failed') {
        return result;
      }
    }
    throw new Error(`评估任务 ${logId} 超时（${timeout}ms）`);
  }

  /** 获取评估日志列表 */
  public static async getLogs(query?: EvalLogQuery): Promise<PageResult<EvalLog>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<PageResult<EvalLog>>(`${EvaluationAPI.BASE_URL}/evaluation/logs`, query as unknown as Record<string, Object>);
  }
}
