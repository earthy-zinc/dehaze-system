import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { Algorithm, AlgorithmQuery } from '../model/AlgorithmInfo';

/**
 * 算法相关API服务接口
 */
export class AlgorithmAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 算法树形表格
   *
   * @param queryParams 查询参数
   * @returns 算法列表
   */
  public static async getList(queryParams?: AlgorithmQuery): Promise<Result<Array<Algorithm>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<Algorithm>>>(`${AlgorithmAPI.BASE_URL}/algorithms`, queryParams);
  }

  /**
   * 获取模型下拉选项列表
   *
   * @returns 模型下拉选项列表
   */
  public static async getOption(): Promise<Result<Array<any>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<any>>>(`${AlgorithmAPI.BASE_URL}/algorithms/options`);
  }

  /**
   * 获取算法详情
   *
   * @param id 算法ID
   * @returns 算法详情
   */
  public static async getAlgorithmInfoById(id: number): Promise<Result<Algorithm>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Algorithm>>(`${AlgorithmAPI.BASE_URL}/algorithms/${id}`);
  }

  /**
   * 新增算法
   *
   * @param data 算法数据
   * @returns 操作结果
   */
  public static async add(data: Algorithm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<void>>(`${AlgorithmAPI.BASE_URL}/algorithms`, data);
  }

  /**
   * 修改算法
   *
   * @param id 算法ID
   * @param data 算法数据
   * @returns 操作结果
   */
  public static async update(id: number, data: Algorithm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${AlgorithmAPI.BASE_URL}/algorithms/${id}`, data);
  }

  /**
   * 删除算法
   *
   * @param ids 算法ID数组
   * @returns 操作结果
   */
  public static async deleteByIds(ids: Array<string>): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${AlgorithmAPI.BASE_URL}/algorithms?ids=${ids.join(',')}`);
  }
}
