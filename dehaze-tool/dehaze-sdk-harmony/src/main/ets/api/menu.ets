import { HttpManager } from '../utils/HttpManager';
import { Result } from '../model/Result';
import { MenuQuery, MenuVO, MenuForm, RouteVO } from '../model/MenuInfo';

/**
 * 菜单相关API服务接口
 */
export class MenuAPI {
  private static readonly BASE_URL: string = '/api/v1';

  /**
   * 获取路由列表
   *
   * @returns 路由列表
   */
  public static async getRoutes(): Promise<Result<Array<RouteVO>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<RouteVO>>>(`${MenuAPI.BASE_URL}/menus/routes`);
  }

  /**
   * 获取菜单树形列表
   *
   * @param queryParams 查询参数
   * @returns 菜单列表
   */
  public static async getList(queryParams: MenuQuery): Promise<Result<Array<MenuVO>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<MenuVO>>>(`${MenuAPI.BASE_URL}/menus`, queryParams);
  }

  /**
   * 获取菜单下拉数据源
   *
   * @returns 菜单下拉选项列表
   */
  public static async getOptions(): Promise<Result<Array<any>>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<Array<any>>>(`${MenuAPI.BASE_URL}/menus/options`);
  }

  /**
   * 获取菜单表单数据
   *
   * @param id 菜单ID
   * @returns 菜单表单数据
   */
  public static async getFormData(id: number): Promise<Result<MenuForm>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.get<Result<MenuForm>>(`${MenuAPI.BASE_URL}/menus/${id}/form`);
  }

  /**
   * 添加菜单
   *
   * @param data 菜单表单数据
   * @returns 操作结果
   */
  public static async add(data: MenuForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.post<Result<void>>(`${MenuAPI.BASE_URL}/menus`, data);
  }

  /**
   * 修改菜单
   *
   * @param id 菜单ID
   * @param data 菜单表单数据
   * @returns 操作结果
   */
  public static async update(id: string, data: MenuForm): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.put<Result<void>>(`${MenuAPI.BASE_URL}/menus/${id}`, data);
  }

  /**
   * 删除菜单
   *
   * @param id 菜单ID
   * @returns 操作结果
   */
  public static async deleteById(id: number): Promise<Result<void>> {
    const httpManager = HttpManager.getInstance();
    return await httpManager.delete<Result<void>>(`${MenuAPI.BASE_URL}/menus/${id}`);
  }
}