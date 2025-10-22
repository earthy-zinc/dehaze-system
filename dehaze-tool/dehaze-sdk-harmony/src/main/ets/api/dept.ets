import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { DeptForm, DeptQuery, DeptVO } from '../model/DeptInfo';

/**
 * 部门相关API服务接口
 */
export class DeptAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 部门树形表格
   *
   * @param queryParams 查询参数
   * @returns 部门列表
   */
  public static async getList(queryParams?: DeptQuery): Promise<Result<Array<DeptVO>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<DeptVO>>>(`${DeptAPI.BASE_URL}/dept`, queryParams);
  }

  /**
   * 部门下拉列表
   *
   * @returns 部门下拉选项列表
   */
  public static async getOptions(): Promise<Result<Array<any>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<any>>>(`${DeptAPI.BASE_URL}/dept/options`);
  }

  /**
   * 获取部门详情
   *
   * @param id 部门ID
   * @returns 部门表单数据
   */
  public static async getFormData(id: number): Promise<Result<DeptForm>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<DeptForm>>(`${DeptAPI.BASE_URL}/dept/${id}/form`);
  }

  /**
   * 新增部门
   *
   * @param data 部门表单数据
   * @returns 操作结果
   */
  public static async add(data: DeptForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<void>>(`${DeptAPI.BASE_URL}/dept`, data);
  }

  /**
   * 修改部门
   *
   * @param id 部门ID
   * @param data 部门表单数据
   * @returns 操作结果
   */
  public static async update(id: number, data: DeptForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${DeptAPI.BASE_URL}/dept/${id}`, data);
  }

  /**
   * 删除部门
   *
   * @param ids 部门ID字符串，多个用逗号分隔
   * @returns 操作结果
   */
  public static async deleteByIds(ids: string): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${DeptAPI.BASE_URL}/dept/${ids}`);
  }
}