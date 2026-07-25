import { HttpManager } from '../utils/HttpManager';
import { PredParam, PredResult, PredLog, PredLogQuery, PageResult, PollOptions, TaskStatus } from '../model/ModelInfo';

const DEFAULT_INTERVAL_MS: number = 2000;
const DEFAULT_TIMEOUT_MS: number = 120000;

function sleep(ms: number): Promise<void> {
  return new Promise<void>((resolve): void => {
    setTimeout(resolve, ms);
  });
}

/**
 * 模型预测 API（去雾处理）
 *
 * 路径与后端路由对齐（dehaze-go/internal/router/prediction.go）：
 * - POST /api/v1/prediction       执行去雾预测，立即返回 logId + status=processing
 * - GET  /api/v1/prediction/{id}  轮询预测任务状态
 * - GET  /api/v1/prediction/logs  预测日志列表
 */
export class PredictionAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /** 执行预测（去雾处理），返回 logId + status */
  public static async predict(data: PredParam): Promise<PredResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<PredResult>(`${PredictionAPI.BASE_URL}/prediction`, data as unknown as Record<string, Object>);
  }

  /** 查询预测任务状态 */
  public static async getStatus(taskId: number): Promise<PredResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<PredResult>(`${PredictionAPI.BASE_URL}/prediction/${taskId}`, undefined);
  }

  /**
   * 提交预测并等待结果（封装 POST + 轮询 GET）
   *
   * - POST 立即返回，若 status=completed（缓存命中）直接返回
   * - status=processing 时按 intervalMs 轮询 GET，直到 completed/failed 或超时
   *
   * 默认：间隔 2s，超时 120s
   */
  public static async predictAndWait(data: PredParam, options?: PollOptions): Promise<PredResult> {
    const result = await PredictionAPI.predict(data);
    if (result.status !== 'processing') {
      return result;
    }
    return PredictionAPI.pollPredTask(result.logId!, options);
  }

  /** 轮询预测任务直到终态（completed/failed）或超时 */
  private static async pollPredTask(logId: number, options?: PollOptions): Promise<PredResult> {
    const interval: number = options?.intervalMs ?? DEFAULT_INTERVAL_MS;
    const timeout: number = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const deadline: number = Date.now() + timeout;

    while (Date.now() < deadline) {
      await sleep(interval);
      const result = await PredictionAPI.getStatus(logId);
      if (options?.onPoll) {
        options.onPoll(result.status as TaskStatus);
      }
      if (result.status === 'completed' || result.status === 'failed') {
        return result;
      }
    }
    throw new Error(`预测任务 ${logId} 超时（${timeout}ms）`);
  }

  /** 获取预测日志列表（历史记录） */
  public static async getLogs(query?: PredLogQuery): Promise<PageResult<PredLog>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<PageResult<PredLog>>(`${PredictionAPI.BASE_URL}/prediction/logs`, query as unknown as Record<string, Object>);
  }
}
