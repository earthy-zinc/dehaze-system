import { HttpManager } from '../utils/HttpManager';
import { EvalParam, EvalResult, EvalLogQuery, PageResult } from '../model/ModelInfo';

/**
 * 效果评估 API
 *
 * 路径与后端路由对齐（dehaze-go/internal/router/evaluation.go）：
 * - POST /api/v1/evaluation       执行效果评估（PSNR/SSIM/LPIPS 等）
 * - GET  /api/v1/evaluation/{id}  查询评估任务状态
 * - GET  /api/v1/evaluation/logs  评估日志列表
 */
export class EvaluationAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /** 执行效果评估 */
  public static async evaluate(data: EvalParam): Promise<EvalResult[]> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<EvalResult[]>(`${EvaluationAPI.BASE_URL}/evaluation`, data as unknown as Record<string, Object>);
  }

  /** 查询评估任务状态 */
  public static async getStatus(taskId: number): Promise<EvalResult[]> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<EvalResult[]>(`${EvaluationAPI.BASE_URL}/evaluation/${taskId}`, undefined);
  }

  /** 获取评估日志列表 */
  public static async getLogs(query?: EvalLogQuery): Promise<PageResult<EvalResult>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<PageResult<EvalResult>>(`${EvaluationAPI.BASE_URL}/evaluation/logs`, query as unknown as Record<string, Object>);
  }
}
