import { HttpManager } from '../utils/HttpManager';
import { Algorithm, AlgorithmQuery } from '../model/AlgorithmInfo';

/**
 * 算法管理 API
 *
 * 路径与后端路由对齐（dehaze-go/internal/router/algorithm.go）：
 * - GET    /api/v1/algorithms          算法树形列表
 * - GET    /api/v1/algorithms/options   下拉选项
 * - GET    /api/v1/algorithms/{id}      算法详情
 * - POST   /api/v1/algorithms           新增算法
 * - PUT    /api/v1/algorithms/{id}      修改算法
 * - PUT    /api/v1/algorithms/{id}/status 修改状态
 * - DELETE /api/v1/algorithms           删除（批量，ids 通过 query 或 body）
 */
export class AlgorithmAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /** 获取算法树形列表 */
  public static async getList(query?: AlgorithmQuery): Promise<Algorithm[]> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Algorithm[]>(`${AlgorithmAPI.BASE_URL}/algorithms`, query as unknown as Record<string, Object>);
  }

  /** 获取算法下拉选项 */
  public static async getOptions(): Promise<Array<{ value: number; label: string }>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Array<{ value: number; label: string }>>(`${AlgorithmAPI.BASE_URL}/algorithms/options`, undefined);
  }

  /** 获取算法详情 */
  public static async getById(id: number): Promise<Algorithm> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Algorithm>(`${AlgorithmAPI.BASE_URL}/algorithms/${id}`, undefined);
  }

  /** 算法对比 */
  public static async compare(ids: number[]): Promise<Algorithm[]> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Algorithm[]>(`${AlgorithmAPI.BASE_URL}/algorithms/compare`, { ids: ids.join(',') } as unknown as Record<string, Object>);
  }

  /** 切换收藏 */
  public static async toggleFavorite(id: number): Promise<boolean> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<boolean>(`${AlgorithmAPI.BASE_URL}/algorithms/${id}/favorite`, undefined);
  }

  /** 检查是否已收藏 */
  public static async checkFavorite(id: number): Promise<boolean> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<boolean>(`${AlgorithmAPI.BASE_URL}/algorithms/favorites/check`, { id } as unknown as Record<string, Object>);
  }

  /** 获取收藏列表 */
  public static async listFavorites(): Promise<Algorithm[]> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Algorithm[]>(`${AlgorithmAPI.BASE_URL}/algorithms/favorites`, undefined);
  }

  /** 新增算法 */
  public static async add(data: Algorithm): Promise<number> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<number>(`${AlgorithmAPI.BASE_URL}/algorithms`, data as unknown as Record<string, Object>);
  }

  /** 修改算法 */
  public static async update(id: number, data: Algorithm): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.put<void>(`${AlgorithmAPI.BASE_URL}/algorithms/${id}`, data as unknown as Record<string, Object>);
  }

  /** 修改算法状态 */
  public static async updateStatus(id: number, status: number): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.put<void>(`${AlgorithmAPI.BASE_URL}/algorithms/${id}/status`, { status } as Record<string, Object>);
  }

  /** 删除算法（批量） */
  public static async deleteByIds(ids: number[]): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${AlgorithmAPI.BASE_URL}/algorithms`, { ids: ids.join(',') } as Record<string, Object>);
  }
}
