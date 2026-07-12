import { HttpManager } from '../utils/HttpManager';
import { PageResult } from '../model/PageResult';
import { InputHistory, InputHistoryQuery, CreateInputHistory } from '../model/InputHistoryInfo';

/**
 * 图像输入历史记录 API
 *
 * 路径与后端路由对齐（dehaze-go/internal/router/input_history.go）：
 * - GET    /api/v1/image-input/history         分页查询
 * - GET    /api/v1/image-input/history/{id}     获取详情
 * - POST   /api/v1/image-input/history          创建记录
 * - PUT    /api/v1/image-input/history/{id}     更新记录（如收藏）
 * - DELETE /api/v1/image-input/history/{id}     删除单条
 * - DELETE /api/v1/image-input/history/batch    批量删除
 * - DELETE /api/v1/image-input/history/clear    清空全部
 */
export class InputHistoryAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /** 分页查询历史记录 */
  public static async list(query: InputHistoryQuery): Promise<PageResult<InputHistory>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<PageResult<InputHistory>>(`${InputHistoryAPI.BASE_URL}/image-input/history`, query as unknown as Record<string, Object>);
  }

  /** 获取历史记录详情 */
  public static async getById(id: number): Promise<InputHistory> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<InputHistory>(`${InputHistoryAPI.BASE_URL}/image-input/history/${id}`, undefined);
  }

  /** 创建历史记录 */
  public static async create(data: CreateInputHistory): Promise<number> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<number>(`${InputHistoryAPI.BASE_URL}/image-input/history`, data as unknown as Record<string, Object>);
  }

  /** 更新历史记录（如添加收藏） */
  public static async update(id: number, data: Partial<InputHistory>): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.put<void>(`${InputHistoryAPI.BASE_URL}/image-input/history/${id}`, data as unknown as Record<string, Object>);
  }

  /** 删除单条历史记录 */
  public static async delete(id: number): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${InputHistoryAPI.BASE_URL}/image-input/history/${id}`, undefined);
  }

  /** 批量删除历史记录 */
  public static async batchDelete(ids: number[]): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${InputHistoryAPI.BASE_URL}/image-input/history/batch`, { ids } as Record<string, Object>);
  }

  /** 清空全部历史记录 */
  public static async clear(): Promise<void> {
    const httpManager = HttpManager.getInstance();
    await httpManager.delete<void>(`${InputHistoryAPI.BASE_URL}/image-input/history/clear`, undefined);
  }
}
