import { HttpManager } from '../utils/HttpManager';
import { PredParam, PredResult, PredLog, PredLogQuery, PageResult } from '../model/ModelInfo';

/**
 * 模型预测 API（去雾处理）
 *
 * 路径与后端路由对齐（dehaze-go/internal/router/prediction.go）：
 * - POST /api/v1/prediction       执行去雾预测
 * - GET  /api/v1/prediction/{id}  查询预测状态
 * - GET  /api/v1/prediction/logs  预测日志列表
 */
export class PredictionAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /** 执行预测（去雾处理） */
  public static async predict(data: PredParam): Promise<PredResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<PredResult>(`${PredictionAPI.BASE_URL}/prediction`, data as unknown as Record<string, Object>);
  }

  /** 查询预测任务状态 */
  public static async getStatus(taskId: number): Promise<PredResult> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<PredResult>(`${PredictionAPI.BASE_URL}/prediction/${taskId}`, undefined);
  }

  /** 获取预测日志列表（历史记录） */
  public static async getLogs(query?: PredLogQuery): Promise<PageResult<PredLog>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<PageResult<PredLog>>(`${PredictionAPI.BASE_URL}/prediction/logs`, query as unknown as Record<string, Object>);
  }
}
