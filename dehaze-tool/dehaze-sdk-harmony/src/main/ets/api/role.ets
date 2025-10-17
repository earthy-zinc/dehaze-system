import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { PageResult } from '../model/PageResult';
import { RoleForm, RoleQuery, RolePageVO } from '../model/RoleInfo';

/**
 * 角色相关API服务接口
 */
export class RoleAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 获取角色分页数据
   *
   * @param queryParams 查询参数
   * @returns 角色分页数据
   */
  public static async getPage(queryParams?: RoleQuery): Promise<Result<PageResult<Array<RolePageVO>>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<PageResult<Array<RolePageVO>>>>(`${RoleAPI.BASE_URL}/roles/page`, queryParams);
  }

  /**
   * 获取角色下拉数据源
   *
   * @param queryParams 查询参数
   * @returns 角色下拉选项列表
   */
  public static async getOptions(queryParams?: RoleQuery): Promise<Result<Array<any>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<any>>>(`${RoleAPI.BASE_URL}/roles/options`, queryParams);
  }

  /**
   * 获取角色的菜单ID集合
   *
   * @param roleId 角色ID
   * @returns 菜单ID集合
   */
  public static async getRoleMenuIds(roleId: number): Promise<Result<Array<number>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<number>>>(`${RoleAPI.BASE_URL}/roles/${roleId}/menuIds`);
  }

  /**
   * 分配菜单权限给角色
   *
   * @param roleId 角色ID
   * @param data 菜单ID数组
   * @returns 操作结果
   */
  public static async updateRoleMenus(roleId: number, data: Array<number>): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${RoleAPI.BASE_URL}/roles/${roleId}/menus`, data);
  }

  /**
   * 获取角色表单数据
   *
   * @param id 角色ID
   * @returns 角色表单数据
   */
  public static async getFormData(id: number): Promise<Result<RoleForm>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<RoleForm>>(`${RoleAPI.BASE_URL}/roles/${id}/form`);
  }

  /**
   * 添加角色
   *
   * @param data 角色表单数据
   * @returns 操作结果
   */
  public static async add(data: RoleForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<void>>(`${RoleAPI.BASE_URL}/roles`, data);
  }

  /**
   * 更新角色
   *
   * @param id 角色ID
   * @param data 角色表单数据
   * @returns 操作结果
   */
  public static async update(id: number, data: RoleForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${RoleAPI.BASE_URL}/roles/${id}`, data);
  }

  /**
   * 批量删除角色，多个以英文逗号(,)分割
   *
   * @param ids 角色ID字符串，多个用逗号分隔
   * @returns 操作结果
   */
  public static async deleteByIds(ids: string): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${RoleAPI.BASE_URL}/roles/${ids}`);
  }
}