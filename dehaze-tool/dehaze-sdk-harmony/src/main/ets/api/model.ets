import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { EvalParam, EvalResult, PredParam, PredResult } from '../model/ModelInfo';

/**
 * 模型相关API服务接口
 */
export class ModelAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 模型预测
   *
   * @param data 预测参数
   * @returns 预测结果
   */
  public static async prediction(data: PredParam): Promise<Result<PredResult>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<PredResult>>(`${ModelAPI.BASE_URL}/model/prediction`, data);
  }

  /**
   * 模型评估
   *
   * @param data 评估参数
   * @returns 评估结果列表
   */
  public static async evaluation(data: EvalParam): Promise<Result<Array<EvalResult>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<Array<EvalResult>>>(`${ModelAPI.BASE_URL}/model/evaluation`, data);
  }
}